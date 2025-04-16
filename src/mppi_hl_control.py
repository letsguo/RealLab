#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import torch
from configs import MPPIConfig
from mppi import mppi

from nav_msgs.msg import Odometry, Path as navPath
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Imu, Image
from mavros_msgs.msg import RCIn
from visualization_msgs.msg import Marker, MarkerArray
from ackermann_msgs.msg import AckermannDriveStamped
from tf.transformations import euler_from_quaternion
import time
import math

from utils.waypoints import Waypoints

# copied imports

from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from utils.rl_policy import RLModel
import os
from pathlib import Path
import yaml
import time
from cv_bridge import CvBridge, CvBridgeError
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from utils.generate_elevation_map import crop_heightmap


class MPPI_HL_control:
    def __init__(self, mppi_config: MPPIConfig):
        self.mppi_config = mppi_config
        self.cost_config = self.mppi_config.cost_cfg
        self.dynamics_config = self.mppi_config.dynamics_cfg
        self.sampling_config = self.mppi_config.sampling_cfg
        # self.map_config = self.mppi_config.map_cfg
        self.vis_config = self.mppi_config.vis_cfg
        self.num_envs = 1

        self.goal_tolerance = 0.1  # meters
        self.velocity_tolerance = 0.05  # m/s
        self.goal = self.mppi_config.cost_cfg.goal_pos

        self.device = torch.device("cuda")
        self.mppi_controller = mppi.MPPI(self.mppi_config, self.num_envs, self.device)
        self.use_prev_opt = False

        self.state_init = False
        self.imu = None
        self.odom_update = False
        self.pose = torch.zeros(6)
        self.twists = torch.zeros(6)
        self.start_action = False
        self.pad_latch = True

        self.rate = 50
        self.steering_max = self.mppi_config.dynamics_cfg.steering_max
        self.throttle_to_wheelspeed = self.mppi_config.dynamics_cfg.throttle_to_wheelspeed

        # i don't believe i need this because the mppi controller is already using previous action so no point in storing
        # self.last_action_offset = 12
        # self.include_last_action = True

        self.value_pub = rospy.Publisher("value", Float32MultiArray, queue_size=1)
        self.obs_type = "relative"
        self.state = np.zeros(12, dtype=np.float32)

        ## map
        self.heightmap = np.load("/root/catkin_ws/src/real_lab/config/elevation/heightmap.npy")
        self.heightmap_sub = rospy.Subscriber("/heightmap", Float32MultiArray, self.heightmap_callback)

        waypoints = Waypoints()
        waypoints.generate_waypoints()
        print("\n1\n")
        # initialize the odometry and imu subscribers with callbacks
        self.odom_sub = rospy.Subscriber(
            "/mavros/local_position/odom", Odometry, self.odom_callback
        )

        self.rc_sub = rospy.Subscriber('/mavros/rc/in', RCIn, self.rcin_callback)

        self.imu_sub = rospy.Subscriber("/mavros/imu/data", Imu, self.imu_callback)

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


    def main_loop(self):
        ## the pycuda-torch lovechild prefers it if you keep it in a single context rather than invoking
        # it in a callback which causes it to create new contexts faster than it can delete the old ones leading to rapid memory growth
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            if (self.state_init and self.odom_update):
                pos_error = np.linalg.norm(self.state[:2] - self.goal[:2])
                # vel_error = np.linalg.norm(self.state[3:5])
                terminate = pos_error < self.goal_tolerance # and vel_error < self.velocity_tolerance

                ctrl = torch.tensor([0.0, 0.0], dtype=torch.float).to(self.device)
                if terminate:
                    print("terminate")
                    # self.goal_init = False
                else:
                    # TODO: verify when use_prev_opt should be set to true
                    # add num_envs to state and convert to tensor
                    expanded_state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)
                    map_tensor = torch.from_numpy(self.heightmap).float().unsqueeze(0).unsqueeze(-1)
                    map_tensor = map_tensor.repeat(1, 1, 1, 4)
                    # TODO: temporary solution i don't know if this is right
                    map_tensor = map_tensor.to(self.device)
                    self.mppi_controller.update(expanded_state, map_tensor)
                    ctrl = self.mppi_controller.optimize(expanded_state, self.use_prev_opt)
                    ctrl = ctrl.squeeze()
                    ctrl = torch.tensor([.8, 1.5])
                    # print("CONTROLS", ctrl)
                msg = Float32MultiArray()
                msg.data = self.state.tolist()
                self.state_pub.publish(msg)
                # if self.collect_data and self.start_action:
                #     msg = Float32MultiArray()
                #     data = np.zeros(15, dtype=np.float32)
                #     data[14] = self.model.get_value(self.state).tolist()[0]
                #     data[0:6] = self.pose.numpy()
                #     data[6:12] = self.twists.numpy()
                #     data[12:14] = ctrl
                #     msg.data = data.tolist()
                #     self.value_pub.publish(msg)
                #     self.pad_latch = True
                # elif self.collect_data and self.pad_latch:
                #     msg = Float32MultiArray()
                #     msg.data = np.zeros(15, dtype=np.float32).tolist()
                #     self.value_pub.publish(msg)
                #     self.pad_latch = False
                    
                self.send_ctrl(ctrl)
                self.odom_update = False
            rate.sleep()

    def send_ctrl(self, ctrl):
        control_msg = AckermannDriveStamped()
        control_msg.header.stamp = rospy.Time.now()
        control_msg.header.frame_id = "base_link"
        control_msg.drive.steering_angle = ctrl[1] * self.steering_max
        control_msg.drive.speed = ctrl[0] * self.throttle_to_wheelspeed
        if not self.start_action:
            control_msg.drive.speed = 0
        
        # i don't believe i need this
        # if self.include_last_action:
        #     if self.start_action:
        #         self.state[self.last_action_offset] = ctrl[0]
        #         self.state[self.last_action_offset + 1] = ctrl[1]
        #     else:
        #         self.state[self.last_action_offset] = 0.0
        #         self.state[self.last_action_offset + 1] = 0.0
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

        self.twists[0] = odom.twist.twist.linear.x
        self.twists[1] = odom.twist.twist.linear.y
        self.twists[2] = odom.twist.twist.linear.z
        self.twists[3] = self.imu.angular_velocity.x
        self.twists[4] = self.imu.angular_velocity.y
        self.twists[5] = self.imu.angular_velocity.z

        if self.obs_type == "relative":
            self.obtain_relative_state(odom)
        # elif self.obs_type == "blind":
        #     self.obtain_blind_state(odom)
        # elif self.obs_type == "elevation":
        #     self.obtain_elevation_state(odom)
        # elif self.obs_type == "goal_based_elevation":
        #     self.obtain_goal_based_elevation_state(odom)
        # elif self.obs_type == "rgb":
        #     self.obtain_rgb_state(odom)
        # else:
        #     ValueError("must choose valid obs type")

    # def obtain_blind_state(self, odom):
    #     self.state[:6] = self.pose.numpy()
    #     self.state[6:12] = self.twists.numpy()

    def obtain_relative_state(self, odom):
        self.state[:6] = self.pos_angle(self.pose).numpy()
        self.state[6:12] = self.twists.numpy()

    def heightmap_callback(self, msg):
        self.heightmap = np.array(msg.data).reshape(self.heightmap.shape)

    def odom_callback(self, odom):
        if self.imu is None:
            return
        self.obtain_state(odom)
        if not self.state_init:
            self.state_init = True
        self.odom_update = True  ## indicate that a new reading is available

    def imu_callback(self, imu):
        self.imu = imu

    def rcin_callback(self, data):
        try:
            self.start_action = data.channels[2] > 1300
        except Exception as e:
            pass

    def limits_callback(self, msg):
        self.hard_limit = msg.drive.speed

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
    # policy = rospy.get_param("~policy")
    # data_collection = rospy.get_param("~data_collection")
    Config = MPPIConfig()
    planner = MPPI_HL_control(Config)
    rospy.spin()