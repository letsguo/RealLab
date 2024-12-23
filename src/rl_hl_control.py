#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
# from hound_mppi import mppi
from nav_msgs.msg import Odometry, Path as navPath
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
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
    def __init__(self, name, throttle_to_wheelspeed= 17.0, steering_max = 0.488, speed_limit=10.0):
        ## state variables
        self.state_init = False
        self.state = np.zeros(14, dtype=np.float32)
        self.throttle_to_wheelspeed = throttle_to_wheelspeed
        self.steering_max = steering_max
        self.imu = None
        
        self.odom_update = False

        self.model = RLModel(name)

        waypoints = Waypoints()
        waypoints.generate_waypoints()

        # initialize the odometry and imu subscribers with callbacks
        self.odom_sub = rospy.Subscriber(
            "/mavros/local_position/odom", Odometry, self.odom_callback
        )
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
        self.marker_pub = rospy.Publisher("marker", MarkerArray, queue_size=1)
        self.reset_pub = rospy.Publisher(
            "/simulation_reset", AckermannDriveStamped, queue_size=2
        )
        time.sleep(1)
        reset_msg = AckermannDriveStamped()
        self.reset_pub.publish(reset_msg)
        os.system(
            "rosservice call /elevation_mapping/clear_map"
        )  ## clear the elevation map.
        time.sleep(1)
        ## initialize controller:
        self.main_loop()

    def limits_callback(self, msg):
        self.hard_limit = msg.drive.speed

    def main_loop(self):
        ## the pycuda-torch lovechild prefers it if you keep it in a single context rather than invoking
        # it in a callback which causes it to create new contexts faster than it can delete the old ones leading to rapid memory growth
        while not rospy.is_shutdown():
            if (
                self.state_init
            ):
                ctrl = self.model.inference(self.state)
                self.state[12:14] = ctrl
                self.send_ctrl(ctrl)
                self.odom_update = False

    def send_ctrl(self, ctrl):
        control_msg = AckermannDriveStamped()
        control_msg.header.stamp = rospy.Time.now()
        control_msg.header.frame_id = "base_link"
        control_msg.drive.steering_angle = ctrl[0] * self.steering_max
        control_msg.drive.speed = ctrl[1] * self.throttle_to_wheelspeed
        self.control_pub.publish(control_msg)

    def obtain_state(self, odom):
        ## obtain the state from the odometry and imu messages:
        dt = (odom.header.stamp - self.imu.header.stamp).to_sec()
        self.large_dt = False
        if dt > 0.1:
            self.large_dt = True
        quaternion = (
            odom.pose.pose.orientation.x,
            odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z,
            odom.pose.pose.orientation.w,
        )
        rpy = euler_from_quaternion(quaternion)
        pos = torch.zeros(6)
        pos[0] = odom.pose.pose.position.y
        pos[1] = odom.pose.pose.position.x
        pos[2] = odom.pose.pose.position.z
        pos[3] = rpy[0]
        pos[4] = rpy[1]
        pos[5] = rpy[2]

        self.state[:6] = self.pos_angle(pos).numpy()

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
    planner = Hound_RLHL_Control("ppo_relative.pt")
    rospy.spin()
