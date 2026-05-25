#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from nav_msgs.msg import Odometry, Path as navPath
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Imu, Image, Joy
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from utils.rl_policy import RLModel
# DEAD: waypoint/pos_angle (relative obs) path is unused -- no config selects "relative"
# from utils.waypoints import Waypoints
from visualization_msgs.msg import Marker, MarkerArray
from ackermann_msgs.msg import AckermannDriveStamped
from tf.transformations import euler_from_quaternion

import os
from pathlib import Path
import yaml
import time
import torch
from cv_bridge import CvBridge, CvBridgeError
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from utils.generate_elevation_map import crop_heightmap
# Shared mocap occupancy builders (same ones mpail_hl_control feeds its MPPI planner)
from utils.mocap_obstacle_map import (
    build_obstacle_occupancy_map,
    collect_obstacle_positions,
    occupancy_to_policy_map,
)


class GAIL_HL_Control:
    def __init__(self, policy, odom_topic, use_mocap, data_collection):
        with open(f"/root/catkin_ws/src/RealLab/config/policies/{policy}.yaml") as f:
            config_data = yaml.safe_load(f)

        self.throttle_to_wheelspeed = config_data["throttle_to_wheelspeed"]
        self.steering_max = config_data["steering_max"]
        self.rate = config_data["rate"]
        self.obs_type = config_data["obs_type"]
        hidden_shape = config_data["hidden_shape"]
        model_path = config_data["model_path"]
        model_type = config_data["model_type"]

        ## state variables
        self.state_init = False
        self.imu = None
        self.odom_update = False
        self.pose = torch.zeros(4)
        self.twists = torch.zeros(3)
        self.start_action = False
        self.pad_latch = True
        self.use_mocap = use_mocap

        # DEAD: relative/waypoint observation mode unused (no config selects "relative")
        # if self.obs_type == "relative":
        #     self.state = np.zeros(7, dtype=np.float32)
        #     self.model = RLModel(model_path,
        #                          type=model_type,
        #                          acargs=(7,7,2),
        #                          ackwargs={
        #                            "actor_hidden_dims": hidden_shape,
        #                            "critic_hidden_dims": hidden_shape
        #                         })
        #     self.include_last_action = False
        if self.obs_type == "blind":
            self.state = np.zeros(7, dtype=np.float32)
            self.model = RLModel(model_path,
                                type=model_type,
                                acargs=(7,7,2),
                                ackwargs={
                                   "actor_hidden_dims": hidden_shape,
                                   "critic_hidden_dims": hidden_shape
                                })
            self.include_last_action = False
            self.last_action_offset = 5 # TODO ask what this param do
        elif self.obs_type == "elevation":
            self.state = np.zeros(972, dtype=np.float32)
            self.model = RLModel(model_path,
                                type=model_type,
                                acargs=(972,972,2),
                                ackwargs={
                                   "actor_hidden_dims": hidden_shape,
                                   "critic_hidden_dims": hidden_shape
                                })
            self.include_last_action = True
            self.last_action_offset = 9
            if use_mocap:
                self.heightmap = np.load("/root/catkin_ws/src/hound_core/config/elevation/heightmap2.npy")
                self.heightmap_sub = rospy.Subscriber("/heightmap", Float32MultiArray, self.heightmap_callback)
            else:
                self.local_heightmap = np.zeros((31,31))
                self.local_heightmap_sub = rospy.Subscriber("/obstacles/map", Float32MultiArray, self.local_heightmap_callback)
        elif self.obs_type == "goal_based_elevation":
            self.state = np.zeros(975, dtype=np.float32)
            self.model = RLModel(model_path,
                                type=model_type,
                                acargs=(975,975,2),
                                ackwargs={
                                   "actor_hidden_dims": hidden_shape,
                                   "critic_hidden_dims": hidden_shape
                                })
            self.include_last_action = True
            self.last_action_offset = 12
            if use_mocap:
                self.heightmap = np.load("/root/catkin_ws/src/hound_core/config/elevation/heightmap2.npy")
                self.heightmap_sub = rospy.Subscriber("/heightmap", Float32MultiArray, self.heightmap_callback)
            else:
                self.local_heightmap = np.zeros((31,31))
                self.local_heightmap_sub = rospy.Subscriber("/obstacles/map", Float32MultiArray, self.local_heightmap_callback)
            self.goal = np.array(config_data["goal"], dtype=np.float32)
        elif self.obs_type == 'rgb':
            self.state = np.zeros(40 * 80 + 8, dtype=np.float32)
            self.model = RLModel(model_path, 
                                type=model_type,
                                acargs = (40 * 80 + 8, 40 * 80 + 8, 2),
                                ackwargs={
                                      "actor_hidden_dims": hidden_shape,
                                      "critic_hidden_dims": hidden_shape
                                })
            self.include_last_action = True
            self.cv_bridge = CvBridge()
            self.image_shape = (40, 80)
            self.resize_shape = (80, 60)
            self.image = np.zeros(self.image_shape[0]*self.image_shape[1])
            self.last_action_offset = 40 * 80 + 6
            self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, callback_args={"resize_shape": self.resize_shape})
            self.threshold = config_data["threshold"]
        elif self.obs_type == "mocap_occupancy":
            # Unified observation matching mpail_hl_control: 12-dim proprioception
            # (pose + twists) followed by the FLATTENED mocap occupancy policy_map,
            # built with the same utils MPAIL feeds its MPPI planner.
            self.map_obs_offset = 12
            # Map geometry as ROS params (defaults match the launch / MPAIL map_cfg).
            self.map_length_px = int(rospy.get_param("~map_length_px", 30))
            self.map_res_m_px = float(rospy.get_param("~map_res_m_px", 0.1))
            self.feature_dim = int(rospy.get_param("~feature_dim", 4))
            obstacle_side_m = float(rospy.get_param("~obstacle_side_m", 0.14))
            self.obstacle_safety_padding_m = float(
                rospy.get_param("~obstacle_safety_padding_m", 0.07)
            )
            self.obstacle_half_extent_m = (
                obstacle_side_m / 2.0 + self.obstacle_safety_padding_m
            )
            self.tracked_obstacles = rospy.get_param(
                "~tracked_obstacles",
                [
                    "obstacle_1", "obstacle_2", "obstacle_3", "obstacle_4",
                    "obstacle_5", "obstacle_6", "obstacle_7",
                ],
            )
            map_dim = self.map_length_px * self.map_length_px * self.feature_dim
            obs_dim = self.map_obs_offset + map_dim
            self.state = np.zeros(obs_dim, dtype=np.float32)
            self.model = RLModel(model_path,
                                type=model_type,
                                acargs=(obs_dim, obs_dim, 2),
                                ackwargs={
                                   "actor_hidden_dims": hidden_shape,
                                   "critic_hidden_dims": hidden_shape
                                })
            self.include_last_action = False
            # 6-DOF pose + 6-dim twists, matching MPAIL_HL_Control's state layout.
            self.pose = torch.zeros(6)
            self.twists = torch.zeros(6)
            self.obstacle_poses = {name: None for name in self.tracked_obstacles}
            self.obstacle_map_ready = False
            for obj in self.tracked_obstacles:
                rospy.Subscriber(
                    f"/mocap/{obj}/pose",
                    PoseStamped,
                    self.obstacle_pose_callback,
                    callback_args=obj,
                    queue_size=1,
                )
        else:
            ValueError("must choose valid obs type")

        self.collect_data = data_collection

        if data_collection:
            self.value_pub = rospy.Publisher("value", Float32MultiArray, queue_size=1)

        # DEAD: waypoints only feed the unused pos_angle/relative path
        # waypoints = Waypoints()
        # waypoints.generate_waypoints()
        print("\n1\n")
        # initialize the odometry and imu subscribers with callbacks
        self.odom_sub = rospy.Subscriber(
            odom_topic, Odometry, self.odom_callback
        )

        # self.rc_sub = rospy.Subscriber('/car/teleop/joy', Joy, self.rcin_callback)

        # self.imu_sub = rospy.Subscriber("/camera/gyro/sample", Imu, self.imu_callback)
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
            "/car/mux/ackermann_cmd_mux/input/navigation", AckermannDriveStamped, queue_size=1
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
        try:
            self.start_action = data.buttons[5] == 1
        except Exception as e:
            pass

    def main_loop(self):
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            if (self.state_init and self.odom_update):
                ctrl = self.model.inference(self.state)
                msg = Float32MultiArray()
                msg.data = self.state.tolist()
                self.state_pub.publish(msg)
                if self.collect_data and self.start_action:
                    msg = Float32MultiArray()
                    data = np.zeros(15, dtype=np.float32)
                    data[14] = self.model.get_value(self.state).tolist()[0]
                    data[0:6] = self.pose.numpy()
                    data[6:12] = self.twists.numpy()
                    data[12:14] = ctrl
                    msg.data = data.tolist()
                    self.value_pub.publish(msg)
                    self.pad_latch = True
                elif self.collect_data and self.pad_latch:
                    msg = Float32MultiArray()
                    msg.data = np.zeros(15, dtype=np.float32).tolist()
                    self.value_pub.publish(msg)
                    self.pad_latch = False
                    
                self.send_ctrl(ctrl)
                self.odom_update = False
            rate.sleep()

    def send_ctrl(self, ctrl):
        control_msg = AckermannDriveStamped()
        control_msg.header.stamp = rospy.Time.now()
        control_msg.header.frame_id = "base_link"
        control_msg.drive.steering_angle = -(ctrl[1] * self.steering_max)
        # control_msg.drive.speed = 0.5 #if (ctrl[0] * self.throttle_to_wheelspeed) > 0 else 0
        control_msg.drive.speed = control_msg.drive.speed = max(0.0, ctrl[0] * self.throttle_to_wheelspeed)

        # if not self.start_action:
        #     control_msg.drive.speed = 0
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
        new_pose = torch.zeros(4)
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

        new_pose[3] = (rpy[2] + 2*np.pi) % (2*np.pi)
        # new_pose[3] = (rpy[0] + 2*np.pi) % (2*np.pi)
        # new_pose[4] = (rpy[1] + 2*np.pi) % (2*np.pi)
        # new_pose[5] = (rpy[2] + 2*np.pi) % (2*np.pi)

        self.pose = new_pose

        self.twists[0] = odom.twist.twist.linear.x
        self.twists[1] = odom.twist.twist.linear.y
        self.twists[2] = odom.twist.twist.linear.z
        # lazy fix for wierd camera reference frame
        # self.twists[3] = self.imu.angular_velocity.z
        # self.twists[4] = - self.imu.angular_velocity.x
        # self.twists[5] = - self.imu.angular_velocity.y

        # DEAD: relative/waypoint observation path unused (no config selects "relative")
        # if self.obs_type == "relative":
        #     self.obtain_relative_state(odom)
        if self.obs_type == "blind":
            self.obtain_blind_state(odom)
        elif self.obs_type == "elevation":
            self.obtain_elevation_state(odom)
        elif self.obs_type == "goal_based_elevation":
            self.obtain_goal_based_elevation_state(odom)
        elif self.obs_type == "rgb":
            self.obtain_rgb_state(odom)
        elif self.obs_type == "mocap_occupancy":
            self.obtain_mocap_occupancy_state(odom)
        else:
            ValueError("must choose valid obs type")

    def obtain_blind_state(self, odom):
        self.state[:3] = self.pose.numpy()[0:3]
        self.state[3:6] = self.twists.numpy()
        self.state[-1] = self.pose.numpy()[-1]

    # DEAD: relative/waypoint observation path unused (no config selects "relative")
    # def obtain_relative_state(self, odom):
    #     self.state[:6] = self.pos_angle(self.pose).numpy()
    #     self.state[6:12] = self.twists.numpy()

    def obstacle_pose_callback(self, msg, obj_name):
        # Mirrors MPAIL_HL_Control.obstacle_pose_callback: cache latest (x, y).
        self.obstacle_poses[obj_name] = np.array(
            [msg.pose.position.x, msg.pose.position.y], dtype=np.float64
        )
        self.obstacle_map_ready = collect_obstacle_positions(
            self.obstacle_poses, self.tracked_obstacles
        ) is not None

    def obtain_mocap_occupancy_state(self, odom):
        """Unified observation matching mpail_hl_control. Mirrors MPAIL's
        obtain_state + obtain_blind_state (12-dim proprioception), then flattens
        build_policy_map() into the observation tail (the same mocap map MPAIL feeds
        its MPPI planner as a 2-D grid)."""
        # --- Mirror MPAIL_HL_Control.obtain_state: full 6-DOF pose + 6-dim twists. ---
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
        new_pose[3] = (rpy[0] + 2 * np.pi) % (2 * np.pi)   # roll
        new_pose[4] = (rpy[1] + 2 * np.pi) % (2 * np.pi)   # pitch
        new_pose[5] = (rpy[2] + 2 * np.pi) % (2 * np.pi)   # yaw
        self.pose = new_pose
        self.twists[0] = odom.twist.twist.linear.x
        self.twists[1] = odom.twist.twist.linear.y
        self.twists[2] = odom.twist.twist.linear.z
        # angular velocities left at 0, as in MPAIL (imu terms disabled)

        # --- Mirror MPAIL_HL_Control.obtain_blind_state: proprioception in dims 0:12. ---
        self.state[:6] = self.pose.numpy()
        self.state[6:12] = self.twists.numpy()

        # --- Mirror MPAIL main_loop: build the mocap policy_map, flatten it in. ---
        policy_map = self.build_policy_map()
        if policy_map is not None:
            self.state[self.map_obs_offset:] = policy_map.reshape(-1).numpy()
            self.obstacle_map_ready = True
        else:
            # Not all obstacle poses seen yet: keep the map region as zeros (free).
            self.obstacle_map_ready = False
            rospy.logwarn_throttle(
                2.0,
                "mocap_occupancy: waiting for all obstacle poses; map left empty.",
            )

    def build_policy_map(self):
        """Robot-centric occupancy from mocap. Mirrors MPAIL_HL_Control.build_policy_map.

        Differences from MPAIL: geometry comes from ROS params (self.map_length_px,
        self.map_res_m_px, self.feature_dim) instead of mpail_config.map_cfg, and the
        tensor is left on CPU because GAIL flattens it into its 1-D observation rather
        than feeding a GPU BEVMap. Returns None until every tracked obstacle pose has
        been received.
        """
        obstacle_xy = collect_obstacle_positions(
            self.obstacle_poses, self.tracked_obstacles
        )
        if obstacle_xy is None:
            return None
        occupancy = build_obstacle_occupancy_map(
            robot_xy=self.pose[:2].numpy(),
            robot_yaw=float(self.pose[5]),
            obstacle_xy=obstacle_xy,
            map_length_px=self.map_length_px,
            map_res_m_px=self.map_res_m_px,
            obstacle_half_extent_m=self.obstacle_half_extent_m,
        )
        policy_map = occupancy_to_policy_map(occupancy, self.feature_dim)
        return policy_map.unsqueeze(0)

    def get_local_elevation_map(self, x, y, yaw, width=20):
        if self.use_mocap:
            elevation_map = crop_heightmap(self.heightmap, x, y, yaw, width=width)
            return elevation_map.flatten()
        else:
            return self.local_heightmap.flatten()
    
    def obtain_elevation_state(self, odom):
        #Obtain state for elevation policy
        x=self.pose[0].numpy()
        y=self.pose[1].numpy()
        yaw = self.pose[5].numpy()
        self.state[:3] = self.pose[3:6].numpy() #obtain the orientation
        self.state[3:9] = self.twists.numpy()
        #TODO: in the observation term I also have the last action term, how do I include it here?  
        self.state[11:972] = self.get_local_elevation_map(x, y, yaw, width=31) # this should be an array of shape (N,) where N = size*size, here it will be 400(taking 20 as the size)

    def obtain_goal_based_elevation_state(self, odom):
        x=self.pose[0].numpy()
        y=self.pose[1].numpy()
        yaw = self.pose[5].numpy()
        self.state[:3] = self.goal - self.pose[:3].numpy()
        self.state[3:6] = self.pose[3:6].numpy() #obtain the orientation
        self.state[6:12] = self.twists.numpy()
        #TODO: in the observation term I also have the last action term, how do I include it here?  
        self.state[14:975] = self.get_local_elevation_map(x, y, yaw, width=31)

    def obtain_rgb_state(self, odom):
        image_offset = self.image_shape[0] * self.image_shape[1]
        self.state[:image_offset] = self.image
        self.state[image_offset:image_offset+6] = self.twists.numpy()

    def image_callback(self, msg, callback_args):
        try:
            image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            raise RuntimeError(e)
        # h x w x c
        resize_shape = callback_args["resize_shape"]
        resized_image = cv2.resize(image, resize_shape)
        H = resized_image.shape[0]

        # remove upper 1/3
        resized_image = resized_image[H//3:, ...]

        norm = np.linalg.norm(resized_image - np.array([255, 255, 255]), ord=1, axis=-1)
        gray_image = np.zeros_like(resized_image[...])
        gray_image[norm < 100] = resized_image[norm < 100]
        gray_image[norm >= 100] = np.random.uniform(0, 30, size=gray_image[norm >= 100].shape)

        # convert to grayscale
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY) / 255.

        # make it unit gaussian assuming mean std of 0.5 0.5

        flattened_image = gray_image.reshape(-1)
        if self.threshold > 0:
           flattened_image = flattened_image > self.threshold
        else:
           flattened_image = self.image = flattened_image

        self.image = (flattened_image - 0.5) / 0.5

    def odom_callback(self, odom):
        # if self.imu is None:
        #     return
        self.obtain_state(odom)
        if not self.state_init:
            self.state_init = True
        self.odom_update = True  ## indicate that a new reading is available

    def imu_callback(self, imu):
        self.imu = imu

    def heightmap_callback(self, msg):
        self.heightmap = np.array(msg.data).reshape(self.heightmap.shape)

    def local_heightmap_callback(self, msg):
        self.local_heightmap = np.array(msg.data).reshape(self.local_heightmap.shape)

    # DEAD: waypoint goal-relative transform; only called by the unused obtain_relative_state
    # def pos_angle(self, pos):
    #     waypoints = Waypoints().waypoints
    #     waypoints = waypoints.to(pos.device)
    #     distances = torch.norm(waypoints - pos[:3], dim=-1)
    #     closest_index = torch.argmin(distances)
    #     goal_index = (closest_index+1)%len(waypoints)
    #     goal_waypoint = waypoints[goal_index.unsqueeze(0)].squeeze(0)
    #     goal_angle = torch.atan2(goal_waypoint[1] - pos[1], goal_waypoint[0] - pos[0])
    #     goal_euler = torch.cat([torch.zeros(2, device=pos.device), goal_angle.unsqueeze(0)])
    #     return torch.cat([goal_waypoint, goal_euler]) - pos


if __name__ == "__main__":
    rospy.init_node("hl_controller")
    policy = rospy.get_param("~policy")
    # data_collection = rospy.get_param("~data_collection")
    # odom_topic = rospy.get_param("~odom")
    # use_mocap = rospy.get_param("~use_mocap")
    data_collection = True
    odom_topic = "/mocap/local_position/odom"
    use_mocap = False
    planner = GAIL_HL_Control(policy, odom_topic, use_mocap, data_collection)
    rospy.spin()
