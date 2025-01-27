#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
# from hound_mppi import mppi
from nav_msgs.msg import Odometry, Path as navPath
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Imu, Image
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from mavros_msgs.msg import RCIn
from utils.rl_policy import RLModel
from utils.waypoints import Waypoints
from visualization_msgs.msg import Marker, MarkerArray
from ackermann_msgs.msg import AckermannDriveStamped
from tf.transformations import euler_from_quaternion
import os
from pathlib import Path
import yaml
import time
import torch
from Bezier import *
from cv_bridge import CvBridge, CvBridgeError
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from policy_factory import load_policy
from utils.generate_elevation_map import crop_heightmap


class Hound_RLHL_Control:
    def __init__(self, policy):
        with open(f"/root/catkin_ws/src/hound_core/config/policies/{policy}.yaml") as f:
            config_data = yaml.safe_load(f)

        self.throttle_to_wheelspeed = config_data["throttle_to_wheelspeed"]
        self.steering_max = config_data["steering_max"]
        self.rate = config_data["rate"]
        self.obs_type = config_data["obs_type"]
        hidden_shape = config_data["hidden_shape"]
        model_path = config_data["model_path"]

        ## state variables
        self.state_init = False
        self.imu = None
        self.odom_update = False
        self.pose = torch.zeros(6)
        self.start_action = False

        if self.obs_type == "relative":
            self.state = np.zeros(12, dtype=np.float32)
            self.model = RLModel(model_path, 
                                 acargs=(12,12,2),
                                 ackwargs={
                                   "actor_hidden_dims": hidden_shape,
                                   "critic_hidden_dims": hidden_shape
                                })
            self.include_last_action = False
        elif self.obs_type == "blind":
            self.state = np.zeros(14, dtype=np.float32)
            self.model = RLModel(model_path,
                                acargs=(14,14,2),
                                ackwargs={
                                   "actor_hidden_dims": hidden_shape,
                                   "critic_hidden_dims": hidden_shape
                                })
            self.include_last_action = True
            self.last_action_offset = 12
        elif self.obs_type == "elevation":
            self.state = np.zeros(687, dtype=np.float32)
            self.model = RLModel(model_path,
                                acargs=(687,687,2),
                                ackwargs={
                                   "actor_hidden_dims": hidden_shape,
                                   "critic_hidden_dims": hidden_shape
                                })
            self.include_last_action = True
            self.last_action_offset = 9
            self.heightmap = np.load("/root/catkin_ws/src/hound_core/config/elevation/heightmap.npy")
            self.heightmap_sub = rospy.Subscriber("/heightmap", Float32MultiArray, self.heightmap_callback)
        elif self.obs_type == "goal_based_elevation":
            self.state = np.zeros(690, dtype=np.float32)
            self.model = RLModel(model_path,
                                acargs=(690,690,2),
                                ackwargs={
                                   "actor_hidden_dims": hidden_shape,
                                   "critic_hidden_dims": hidden_shape
                                })
            self.include_last_action = True
            self.last_action_offset = 12
            self.heightmap = np.load("/root/catkin_ws/src/hound_core/config/elevation/heightmap.npy")
            self.heightmap_sub = rospy.Subscriber("/heightmap", Float32MultiArray, self.heightmap_callback)
            self.goal = np.zeros(3, dtype=np.float32)
        else:
            ValueError("must choose valid obs type")
        
        waypoints = Waypoints()
        waypoints.generate_waypoints()
        print("\n1\n")
        # initialize the odometry and imu subscribers with callbacks
        self.odom_sub = rospy.Subscriber(
            "/mavros/local_position/odom", Odometry, self.odom_callback
        )

        self.rc_sub = rospy.Subscriber('/mavros/rc/in', RCIn, self.rcin_callback)

        self.imu_sub = rospy.Subscriber("/mavros/imu/data", Imu, self.imu_callback)
        # self.grid_map_sub = rospy.Subscriber(
        #     "/grid_map_occlusion_inpainting/all_grid_map",
        #     GridMap,
        #     self.grid_map_callback,
        # )
        # self.path_sub = rospy.Subscriber(
        #     "path",
        #     navPath,
        #     self.path_callback,
        #     queue_size=10,
        # )
        self.ctrl_limits_sub = rospy.Subscriber(
            "/control_limits",
            AckermannDriveStamped,
            self.limits_callback,
            queue_size=10,
        )
        self.hard_limit = 0.3
        self.large_dt = False

        ## set up publishers:
        self.control_pub = rospy.Publisher(
            "low_level_controller/hound/control", AckermannDriveStamped, queue_size=1
        )
        self.state_pub = rospy.Publisher(
            "hl_controller/state", Float32MultiArray, queue_size=1
        )

        self.marker_pub = rospy.Publisher("marker", MarkerArray, queue_size=1)
        self.reset_pub = rospy.Publisher(
            "/simulation_reset", AckermannDriveStamped, queue_size=2
        )
        time.sleep(1)
        reset_msg = AckermannDriveStamped()
        self.reset_pub.publish(reset_msg)
        time.sleep(1)
        ## initialize controller:
        print("\n2\n")
        self.main_loop()

    def limits_callback(self, msg):
        self.hard_limit = msg.drive.speed

    def rcin_callback(self, data):
        self.start_action = data.channels[2] > 1300

    def main_loop(self):
        ## the pycuda-torch lovechild prefers it if you keep it in a single context rather than invoking
        # it in a callback which causes it to create new contexts faster than it can delete the old ones leading to rapid memory growth
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            if (self.state_init and self.odom_update):
                ctrl = self.model.inference(self.state)
                self.send_ctrl(ctrl)
                self.odom_update = False

                msg = Float32MultiArray()
                msg.data = self.state.tolist()
                self.state_pub.publish(msg)
            rate.sleep()

    def send_ctrl(self, ctrl):
        control_msg = AckermannDriveStamped()
        control_msg.header.stamp = rospy.Time.now()
        control_msg.header.frame_id = "base_link"
        control_msg.drive.steering_angle = ctrl[1] * self.steering_max
        control_msg.drive.speed = ctrl[0] * self.throttle_to_wheelspeed
        if not self.start_action:
            control_msg.drive.speed = 0
        if self.include_last_action:
            if self.start_action:
                self.state[self.last_action_offset] = ctrl[0]
                self.state[self.last_action_offset + 1] = ctrl[1]
            else:
                self.state[self.last_action_offset] = 0.0
                self.state[self.last_action_offset + 1] = 0.0
        self.control_pub.publish(control_msg)

    def obtain_state(self, odom):
        ## obtain the state from the odometry and imu messages:
        new_pose = torch.zeros(6)
        quaternion = (
            odom.pose.pose.orientation.x,
            odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z,
            odom.pose.pose.orientation.w,
        )

        rpy = euler_from_quaternion(quaternion)
        new_pose[0] = odom.pose.pose.position.x
        new_pose[1] = odom.pose.pose.position.y
        new_pose[2] = odom.pose.pose.position.z

        #make sure angles are between 0 and 2pi
        imu_quaternion = (
            self.imu.orientation.x,
            self.imu.orientation.y,
            self.imu.orientation.z,
            self.imu.orientation.w,
        )
        rpy_imu = euler_from_quaternion(imu_quaternion)

        new_pose[3] = (rpy_imu[0] + 2*np.pi) % (2*np.pi)
        new_pose[4] = (rpy_imu[1] + 2*np.pi) % (2*np.pi)
        new_pose[5] = (rpy[2] + 2*np.pi) % (2*np.pi)

        self.pose = new_pose

        if self.obs_type == "relative":
            self.obtain_relative_state(odom)
        elif self.obs_type == "blind":
            self.obtain_blind_state(odom)
        elif self.obs_type == "elevation":
            self.obtain_elevation_state(odom)
        elif self.obs_type == "goal_based_elevation":
            self.obtain_goal_based_elevation_state(odom)
        else:
            ValueError("must choose valid obs type")

    def obtain_blind_state(self, odom):
        self.state[:6] = self.pose.numpy()
        self.state[6] = odom.twist.twist.linear.x
        self.state[7] = odom.twist.twist.linear.y
        self.state[8] = odom.twist.twist.linear.z
        self.state[9] = self.imu.angular_velocity.x
        self.state[10] = self.imu.angular_velocity.y
        self.state[11] = self.imu.angular_velocity.z

    def obtain_relative_state(self, odom):
        self.state[:6] = self.pos_angle(self.pose).numpy()
        self.state[6] = odom.twist.twist.linear.x
        self.state[7] = odom.twist.twist.linear.y
        self.state[8] = odom.twist.twist.linear.z
        self.state[9] = self.imu.angular_velocity.x
        self.state[10] = self.imu.angular_velocity.y
        self.state[11] = self.imu.angular_velocity.z

    def get_local_elevation_map(self, x, y, yaw, width=20):
        elevation_map = crop_heightmap(self.heightmap, x, y, yaw, width=width)
        return elevation_map.flatten()
    
    def obtain_elevation_state(self, odom):
        #Obtain state for elevation policy
        x=self.pose[0].numpy()
        y=self.pose[1].numpy()
        yaw = self.pose[5].numpy()
        self.state[:3] = self.pose[3:6].numpy() #obtain the orientation
        self.state[3] = odom.twist.twist.linear.x
        self.state[4] = odom.twist.twist.linear.y
        self.state[5] = odom.twist.twist.linear.z 
        self.state[6] = self.imu.angular_velocity.x
        self.state[7] = self.imu.angular_velocity.y
        self.state[8] = self.imu.angular_velocity.z
        #TODO: in the observation term I also have the last action term, how do I include it here?  
        self.state[11:687] = self.get_local_elevation_map(x, y, yaw, width=26) # this should be an array of shape (N,) where N = size*size, here it will be 400(taking 20 as the size)

    def obtain_goal_based_elevation_state(self, odom):
        x=self.pose[0].numpy()
        y=self.pose[1].numpy()
        yaw = self.pose[5].numpy()
        self.state[:3] = self.goal - self.pose[:3].numpy()
        self.state[3:6] = self.pose[3:6].numpy() #obtain the orientation
        self.state[6] = odom.twist.twist.linear.x
        self.state[7] = odom.twist.twist.linear.y
        self.state[8] = odom.twist.twist.linear.z 
        self.state[9] = self.imu.angular_velocity.x
        self.state[10] = self.imu.angular_velocity.y
        self.state[11] = self.imu.angular_velocity.z
        #TODO: in the observation term I also have the last action term, how do I include it here?  
        self.state[14:690] = self.get_local_elevation_map(x, y, yaw, width=26)


    def odom_callback(self, odom):
        if self.imu is None:
            return
        self.obtain_state(odom)
        if not self.state_init:
            self.state_init = True
        self.odom_update = True  ## indicate that a new reading is available

    def imu_callback(self, imu):
        self.imu = imu

    def heightmap_callback(self, msg):
        self.heightmap = np.array(msg.data).reshape(self.heightmap.shape)

    def pos_angle(self, pos):
        waypoints = Waypoints().waypoints
        waypoints = waypoints.to(pos.device)
        distances = torch.norm(waypoints - pos[:3], dim=-1)
        closest_index = torch.argmin(distances)
        goal_index = (closest_index+1)%len(waypoints)
        goal_waypoint = waypoints[goal_index.unsqueeze(0)].squeeze(0)
        goal_angle = torch.atan2(goal_waypoint[1] - pos[1], goal_waypoint[0] - pos[0])
        goal_euler = torch.cat([torch.zeros(2, device=pos.device), goal_angle.unsqueeze(0)])
        return torch.cat([goal_waypoint, goal_euler]) - pos


if __name__ == "__main__":
    rospy.init_node("hl_controller")
    policy = rospy.get_param("~policy")
    planner = Hound_RLHL_Control(policy)
    rospy.spin()
