#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
# from hound_mppi import mppi
from nav_msgs.msg import Odometry, Path as navPath
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Imu
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
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from policy_factory import load_policy


class Hound_RLHL_Control:
    def __init__(self, name, throttle_to_wheelspeed= 5.0, steering_max = 0.488, alpha = [0.2,0.2,0.2,0.2,0.2,0.2], obs_type="relative"):
        ## state variables
        self.state_init = False
        self.throttle_to_wheelspeed = throttle_to_wheelspeed
        self.steering_max = steering_max
        self.imu = None
        self.alpha = torch.tensor(alpha)
        self.odom_update = False
        self.obs_type = obs_type
        self.pose = torch.zeros(6)
        self.start_action = False

        if self.obs_type == "relative":
            self.state = np.zeros(12, dtype=np.float32)
            self.model = RLModel(name, acargs=(12,12,2))
            self.include_last_action = False
        elif self.obs_type == "blind":
            self.state = np.zeros(14, dtype=np.float32)
            self.model = RLModel(name, acargs=(14,14,2))
            self.include_last_action = True
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

        self.imu_sub = rospy.Subscriber("/mavros/imu/data_raw", Imu, self.imu_callback)
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
        rate = rospy.Rate(20)
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
                self.state[-2] = ctrl[0]
                self.state[-1] = ctrl[1]
            else:
                self.state[-2] = 0.0
                self.state[-1] = 0.0
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
        new_pose[3] = (rpy[0] + 2*np.pi) % (2*np.pi)
        new_pose[4] = (rpy[1] + 2*np.pi) % (2*np.pi)
        new_pose[5] = (rpy[2] + 2*np.pi) % (2*np.pi)

        #low pass filter
        self.pose = (1.0-self.alpha) * self.pose + self.alpha * new_pose

        if self.obs_type == "relative":
            self.obtain_relative_state(odom)
        elif self.obs_type == "blind":
            self.obtain_blind_state(odom)
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

    def odom_callback(self, odom):
        if self.imu is None:
            return
        self.obtain_state(odom)
        if not self.state_init:
            self.state_init = True
        self.odom_update = True  ## indicate that a new reading is available

    def imu_callback(self, imu):
        self.imu = imu

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
    planner = Hound_RLHL_Control("ppo_drift_mpcobs2_2000.pt", obs_type="blind")
    rospy.spin()
