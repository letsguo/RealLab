#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import torch
from configs import MPPIConfig
from mppi import mppi
from mpail import mpail_cfg
from mpail import policy
from mpail.policy import MPAILPolicy
from mpail.configs import MPAILPolicyConfig
from camera_to_bev import RGBBEVHistory
from mpail import MPAILPolicyCfg

import os
import yaml

from mpail.utils import dataclass_from_yaml_recurse

from nav_msgs.msg import Odometry, Path as navPath
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Int32
from sensor_msgs.msg import Imu, Image
# from mavros_msgs.msg import RCIn
from visualization_msgs.msg import Marker, MarkerArray
from ackermann_msgs.msg import AckermannDriveStamped
from tf.transformations import euler_from_quaternion
import time
import math

from utils.waypoints import Waypoints
from utils.mocap_obstacle_map import (
    build_obstacle_occupancy_map,
    collect_obstacle_positions,
    occupancy_to_policy_map,
)

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
import ipdb
import atexit
from io import BytesIO

import imageio

class MPAIL_HL_Control:
    def __init__(self, mpail_config: MPAILPolicyConfig, policy_model):
        print("init")
        
        self.mpail_config = mpail_config
        self.cost_config = self.mpail_config.cost_cfg
        self.dynamics_config = self.mpail_config.dynamics_cfg
        self.sampling_config = self.mpail_config.sampling_cfg
        # self.map_config = self.mppi_config.map_cfg
        self.vis_config = self.mpail_config.vis_cfg
        self.num_envs = 1

        self.device = torch.device("cuda")
        self.mpail_controller = MPAILPolicy(self.mpail_config, self.num_envs, self.device)

        # model_state_dict keys are MPAILPolicy-prefixed (costs.*, _temp_exp), not TDCost-local
        missing, unexpected = self.mpail_controller.load_state_dict(policy_model, strict=False)
        if missing:
            rospy.logwarn(f"Policy checkpoint missing keys: {missing}")
        if unexpected:
            rospy.logwarn(f"Policy checkpoint unexpected keys: {unexpected}")
        rospy.loginfo("Policy model loaded successfully.")

        self.state_init = False
        self.imu = None
        self.odom_update = False
        self.pose = torch.zeros(6)
        self.twists = torch.zeros(6)
        self.start_action = False
        self.pad_latch = True

        self.rate = 100
        self.steering_max = self.mpail_config.dynamics_cfg.steering_max
        self.throttle_to_wheelspeed = self.mpail_config.dynamics_cfg.throttle_to_wheelspeed

        # i don't believe i need this because the mppi controller is already using previous action so no point in storing
        # self.last_action_offset = 12
        # self.include_last_action = True

        self.value_pub = rospy.Publisher("value", Float32MultiArray, queue_size=1)
        self.obs_type = "blind"
        self.state = np.zeros(12, dtype=np.float32)

        # map
        self.bridge = CvBridge()
        self.camera_sub = rospy.Subscriber('/car/car/camera/color/image_raw', Image, self.camera_callback)
        self.depth_sub = rospy.Subscriber('/car/car/camera/depth/image_rect_raw', Image, self.depth_callback)

        self.camera_sub_2 = rospy.Subscriber('/camera/color/image_raw', Image, self.camera_callback)
        self.depth_sub_2 = rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_callback)
        self.latest_depth = None
        self.latest_rgb = None
        self.updated_map = None
        self.camera_update = False
        self.depth_update = False

        # Mocap obstacle poses -> robot-centric occupancy map (exteroception for policy)
        self.tracked_obstacles = rospy.get_param(
            "~tracked_obstacles",
            [
                "obstacle_1", "obstacle_2", "obstacle_3", "obstacle_4",
                "obstacle_5", "obstacle_6", "obstacle_7",
            ],
        )
        # VRPN/mocap gives pose only; set side length [m] to match Isaac spawn.size[0]
        obstacle_side_m = rospy.get_param("~obstacle_side_m", 1.6)
        self.obstacle_safety_padding_m = rospy.get_param("~obstacle_safety_padding_m", 1.0)
        self.obstacle_half_extent_m = obstacle_side_m / 2.0 + self.obstacle_safety_padding_m
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

        self.rgb_history = RGBBEVHistory(
            rgb_size=(720, 1280),
            rgb_K=torch.tensor([[635.9487, 0.0, 634.8170],
                            [0.0, 635.4133, 366.7346],
                            [0.0, 0.0, 1.0]]),
            rgb_Rt = torch.tensor([[0.0, 1.0, 0.0, 0.023],
                           [0.0, 0.0, 1.0, -0.042],
                           [1.0, 0.0, 0.0, 0.068],
                           [0.0, 0.0, 0.0, 1.0]]),
            depth_K=torch.tensor([[424.0352478027344, 0.0, 426.2065734863281],
                        [0.0, 424.0352478027344, 239.09210205078125],
                        [0.0, 0.0, 1.0]]),
            depth_Rt = torch.tensor([[0.0, 1.0, 0.0, 0.023],
                                    [0.0, 0.0, 1.0, 0.018],
                                    [1.0, 0.0, 0.0, 0.068],
                                    [0.0, 0.0, 0.0, 1.0]]),
            meter_to_pixel=94.21,
            bevmap_size=(1000, 1000)
        )

        self.odom_sub = rospy.Subscriber(
            "/mocap/local_position/odom", Odometry, self.odom_callback
        )

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

        self.publish_viz = True
        self.vis_timestep = 0
        self.record_rollout_video = rospy.get_param("~record_rollout_video", True)
        self.rollout_vis_dir = os.path.abspath(
            rospy.get_param(
                "~rollout_vis_dir",
                os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "rollout_vis",
                ),
            )
        )
        self.video_fps = int(rospy.get_param("~video_fps", 10))
        self._video_frames = []
        self._video_saved = False
        os.makedirs(self.rollout_vis_dir, exist_ok=True)
        if self.publish_viz:
            rospy.loginfo("visualization in effect")
            rospy.loginfo(
                "rollout video: record=%s dir=%s",
                self.record_rollout_video,
                self.rollout_vis_dir,
            )
            rospy.on_shutdown(self._save_rollout_video)
            atexit.register(self._save_rollout_video)

            self.current_state_pub = rospy.Publisher("/mppi/current_state", Float32MultiArray, queue_size=1)
            self.rollouts_pub      = rospy.Publisher("/mppi/rollouts", Float32MultiArray, queue_size=1)
            self.obstacle_occupancy_pub = rospy.Publisher("/mppi/obstacle_occupancy", Float32MultiArray, queue_size=1)
            self.cost_func_pub = rospy.Publisher("/mppi/cost_func", Float32MultiArray, queue_size=1)
            self.costs_pub         = rospy.Publisher("/mppi/rollout_costs", Float32MultiArray, queue_size=1)
            self.optimal_trajectory_pub = rospy.Publisher("/mppi/optimal_trajectory", Float32MultiArray, queue_size=1)
            self.obstacle_pose_pub = rospy.Publisher("/mppi/obstacle_pose", Float32MultiArray, queue_size=1)
            self.timestep = rospy.Publisher("/mppi/frame_timestep", Int32, queue_size=1)

        time.sleep(1)
        reset_msg = AckermannDriveStamped()
        self.reset_pub.publish(reset_msg)
        time.sleep(1)
        ## initialize controller:
        print("\n2\n")
        self.main_loop()


    def main_loop(self):
        rate = rospy.Rate(self.rate)
        ctrl = torch.tensor([0.0, 0.0], dtype=torch.float).to(self.device)
        while not rospy.is_shutdown():
            if (self.state_init and self.odom_update):
                expanded_state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)

                map_size = self.mpail_config.map_cfg.map_length_px   # 48
                num_channels = self.mpail_config.map_cfg.feature_dim # 4
                batch_size = 1

                policy_map = self.build_policy_map()
                if policy_map is None:
                    rospy.logwarn_throttle(
                        2.0,
                        "Waiting for all mocap obstacle poses before updating policy map.",
                    )
                    rate.sleep()
                    continue

                self.mpail_controller.update(expanded_state, policy_map)
                ctrl = self.mpail_controller.optimize(expanded_state)

                if self.publish_viz:
                    try:
                        self.publish_vis()
                    except Exception as exc:
                        rospy.logwarn_throttle(5.0, f"MPPI viz publish failed: {exc}")

                ctrl = ctrl.squeeze()
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
                self.camera_update = False
                self.depth_update = False
            rate.sleep()

    def get_visual_feature_from_obs(self) -> torch.Tensor:
        """Get the terrain visual feature from the camera.
        bevmap_resolution: the resolution of the BEV map in meters
        """
        # # we need the rgb value corresponding to the robtot position
        agent_pos_t = torch.from_numpy(self.state[:3].copy())
        agent_quat = torch.from_numpy(self.state[3:7].copy())

        agent_pose = RGBBEVHistory.make_pose(agent_pos_t, agent_quat)

        camera = self.latest_rgb.unsqueeze(0)  # (B, H, W, C)
        depth = self.latest_depth.unsqueeze(0)  # (B, H, W, 1)
        updated_map = self.rgb_history.update_bev(camera, depth, agent_pose)

        return updated_map

    
    def create_multidimensional_array_msg(self, data_array, dims_sizes, dims_names):
        """Create a Float32MultiArray message with dimension information"""
        msg = Float32MultiArray()
        
        # Create MultiArrayLayout
        msg.layout.dim = [MultiArrayDimension() for _ in range(len(dims_sizes))]
        
        # Set dimensions
        for i, (size, name) in enumerate(zip(dims_sizes, dims_names)):
            msg.layout.dim[i].size = size
            msg.layout.dim[i].stride = int(np.prod(dims_sizes[i+1:])) if i < len(dims_sizes)-1 else 1
            msg.layout.dim[i].label = name
            
        # Flatten the array and convert to list
        msg.data = data_array.flatten().tolist()
        
        return msg
    
    def _obstacle_pose_for_vis(self):
        """World-frame obstacle centers [1, N, 2] for the visualizer."""
        xy = collect_obstacle_positions(self.obstacle_poses, self.tracked_obstacles)
        if xy is None:
            return None
        return xy[np.newaxis, ...].astype(np.float32)

    def publish_vis(self):
        """Publish visualization data to ROS topics"""
        obstacle_pose = self._obstacle_pose_for_vis()
        obstacle_pose_arg = (
            torch.from_numpy(obstacle_pose[0]).to(self.device)
            if obstacle_pose is not None
            else None
        )
        x0, vis_rollouts, obstacle_occupancy, cost_func, vis_costs, opt_states, _ = (
            self.mpail_controller.publish_visualization_data(obstacle_pose=obstacle_pose_arg)
        )
        obstacle_occupancy = np.asarray(obstacle_occupancy)
        cost_func = np.asarray(cost_func)
        if obstacle_pose is None:
            obstacle_pose = np.zeros((1, 0, 2), dtype=np.float32)
        # Publish messages
        self.current_state_pub.publish(
            self.create_multidimensional_array_msg(
                x0, 
                [x0.shape[0], x0.shape[1]], 
                ['env', 'state_dim']
            )
        )
        
        self.rollouts_pub.publish(
            self.create_multidimensional_array_msg(
                vis_rollouts, 
                [vis_rollouts.shape[0], vis_rollouts.shape[1], vis_rollouts.shape[2], vis_rollouts.shape[3]], 
                ['env', 'rollout', 'horizon', 'state_dim']
            )
        )
        
        self.obstacle_occupancy_pub.publish(
            self.create_multidimensional_array_msg(
                obstacle_occupancy, 
                [obstacle_occupancy.shape[0], obstacle_occupancy.shape[1], obstacle_occupancy.shape[2]], 
                ['env', 'height', 'width']
            )
        )
        
        self.cost_func_pub.publish(
            self.create_multidimensional_array_msg(
                cost_func, 
                [cost_func.shape[0], cost_func.shape[1], cost_func.shape[2]], 
                ['env', 'height', 'width']
            )
        )
        
        self.costs_pub.publish(
            self.create_multidimensional_array_msg(
                vis_costs, 
                [vis_costs.shape[0], vis_costs.shape[1], vis_costs.shape[2]], 
                ['env', 'rollout', 'horizon']
            )
        )
        
        self.optimal_trajectory_pub.publish(
            self.create_multidimensional_array_msg(
                opt_states, 
                [opt_states.shape[0], opt_states.shape[1], opt_states.shape[2]], 
                ['env', 'horizon', 'state_dim']
            )
        )

        if obstacle_pose.size > 0:
            self.obstacle_pose_pub.publish(
                self.create_multidimensional_array_msg(
                    obstacle_pose,
                    list(obstacle_pose.shape),
                    ['env', 'obstacle', 'xy'][: len(obstacle_pose.shape)],
                )
            )

        self.vis_timestep += 1
        self.timestep.publish(self.vis_timestep)

        if self.record_rollout_video and getattr(self.mpail_controller, "vis", None):
            try:
                self.mpail_controller.create_vis(obstacle_pose=obstacle_pose_arg)
                self._capture_rollout_frame()
            except Exception as exc:
                rospy.logwarn_throttle(5.0, f"Rollout video frame capture failed: {exc}")

    def _capture_rollout_frame(self):
        vis = self.mpail_controller.vis
        vis.fig.canvas.draw()
        buf = BytesIO()
        vis.fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        self._video_frames.append(imageio.imread(buf))
        buf.close()
        if self.vis_timestep == 1 or self.vis_timestep % 50 == 0:
            rospy.loginfo(
                "Captured rollout frame %d (%d total)",
                self.vis_timestep,
                len(self._video_frames),
            )

    def _write_rollout_video_cv2(self, path):
        if not self._video_frames:
            return False
        frame0 = self._video_frames[0]
        h, w = frame0.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, self.video_fps, (w, h))
        if not writer.isOpened():
            return False
        for frame in self._video_frames:
            if frame.shape[0] != h or frame.shape[1] != w:
                frame = cv2.resize(frame, (w, h))
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame)
        writer.release()
        return os.path.isfile(path) and os.path.getsize(path) > 0

    def _save_rollout_video(self):
        if self._video_saved or not self.record_rollout_video or not self._video_frames:
            return
        self._video_saved = True
        path = os.path.join(self.rollout_vis_dir, "rollouts_video.mp4")
        n_frames = len(self._video_frames)
        if self._write_rollout_video_cv2(path):
            rospy.loginfo("Saved rollout video (%d frames): %s", n_frames, path)
            print(f"Saved rollout video ({n_frames} frames): {path}")
            return
        gif_path = os.path.join(self.rollout_vis_dir, "rollouts_video.gif")
        try:
            imageio.mimsave(gif_path, self._video_frames, fps=self.video_fps)
            rospy.logwarn("MP4 write failed; saved GIF: %s", gif_path)
            print(f"Saved rollout GIF ({n_frames} frames): {gif_path}")
        except Exception as exc:
            rospy.logwarn("Failed to save rollout video (%d frames): %s", n_frames, exc)
            print(f"Failed to save rollout video: {exc}")

    def send_ctrl(self, ctrl):
        control_msg = AckermannDriveStamped()
        control_msg.header.stamp = rospy.Time.now()
        control_msg.header.frame_id = "base_link"
        control_msg.drive.steering_angle = ctrl[1] * self.steering_max * -1
        control_msg.drive.speed = ctrl[0] * self.throttle_to_wheelspeed
        # print("Drive Speed: " + str(control_msg.drive.speed))

        # if not self.start_action:
        #     control_msg.drive.speed = 0
        
        # i don't believe i need this
        # if self.include_last_action:
        #     if self.start_action:
        #         self.state[self.last_action_offset] = ctrl[0]
        #         self.state[self.last_action_offset + 1] = ctrl[1]
        #     else:
        #         self.state[self.last_action_offset] = 0.0
        #         self.state[self.last_action_offset + 1] = 0.0
        # print('sending control')
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
        # imu_quaternion = (
        #     self.imu.orientation.x,
        #     self.imu.orientation.y,
        #     self.imu.orientation.z,
        #     self.imu.orientation.w,
        # )
        # rpy_imu = euler_from_quaternion(imu_quaternion)

        # new_pose[3] = (rpy_imu[0] + 2*np.pi) % (2*np.pi)
        # new_pose[4] = (rpy_imu[1] + 2*np.pi) % (2*np.pi)
        # new_pose[5] = (rpy[2] + 2*np.pi) % (2*np.pi)

        rpy = euler_from_quaternion(quaternion)
        new_pose[3] = (rpy[0] + 2*np.pi) % (2*np.pi)   # roll
        new_pose[4] = (rpy[1] + 2*np.pi) % (2*np.pi)   # pitch
        new_pose[5] = (rpy[2] + 2*np.pi) % (2*np.pi)   # yaw

        """ check ^ gpt above """


        self.pose = new_pose

        self.twists[0] = odom.twist.twist.linear.x
        self.twists[1] = odom.twist.twist.linear.y
        self.twists[2] = odom.twist.twist.linear.z
        # self.twists[3] = self.imu.angular_velocity.x
        # self.twists[4] = self.imu.angular_velocity.y
        # self.twists[5] = self.imu.angular_velocity.z

        if self.obs_type == "relative":
            self.obtain_relative_state(odom)
        elif self.obs_type == "blind":
            self.obtain_blind_state(odom)
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

    def obtain_blind_state(self, odom):
        self.state[:6] = self.pose.numpy()
        self.state[6:12] = self.twists.numpy()

    def obtain_relative_state(self, odom):
        print("obtaining relative pose", self.pose)
        self.state[:6] = self.pos_angle(self.pose).numpy()
        print("obtaining relative state", self.state[:6])
        self.state[6:12] = self.twists.numpy()

    def odom_callback(self, odom):
        # rospy.loginfo("Got odom, imu=%s", self.imu is not None)
        # if self.imu is None:
        #     return
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

    def camera_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_rgb = torch.from_numpy(cv_image).float() / 255.0
            self.camera_update = True
            # print("HII")
        except Exception as e:
            rospy.logerr(f"Failed to process RGB image: {e}")

    def depth_callback(self, msg):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            cv_depth = cv_depth.astype(np.float32) / 1000.0
            cv_depth = np.expand_dims(cv_depth, axis=2) 
            self.latest_depth = torch.from_numpy(cv_depth).float()
            self.depth_update = True
            # print("HII")
        except Exception as e:
            rospy.logerr(f"Failed to process Depth image: {e}")

    def limits_callback(self, msg):
        self.hard_limit = msg.drive.speed

    def obstacle_pose_callback(self, msg, obj_name):
        self.obstacle_poses[obj_name] = np.array(
            [msg.pose.position.x, msg.pose.position.y],
            dtype=np.float64,
        )
        self.obstacle_map_ready = collect_obstacle_positions(
            self.obstacle_poses, self.tracked_obstacles
        ) is not None

    def build_policy_map(self):
        """Robot-centric occupancy from mocap, shaped for BEVMap.update."""
        obstacle_xy = collect_obstacle_positions(
            self.obstacle_poses, self.tracked_obstacles
        )
        if obstacle_xy is None:
            return None

        map_cfg = self.mpail_config.map_cfg
        occupancy = build_obstacle_occupancy_map(
            robot_xy=self.pose[:2].numpy(),
            robot_yaw=float(self.pose[5]),
            obstacle_xy=obstacle_xy,
            map_length_px=map_cfg.map_length_px,
            map_res_m_px=map_cfg.map_res_m_px,
            obstacle_half_extent_m=self.obstacle_half_extent_m,
        )
        policy_map = occupancy_to_policy_map(occupancy, map_cfg.feature_dim)
        return policy_map.unsqueeze(0).to(device=self.device, dtype=torch.float32)

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


def load_yaml(filename: str) -> dict:
    """Loads an input PKL file safely.

    Args:
        filename: The path to pickled file.

    Raises:
        FileNotFoundError: When the specified file does not exist.

    Returns:
        The data read from the input file.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    with open(filename) as f:
        data = yaml.full_load(f)
    return data

if __name__ == "__main__":
    rospy.init_node("hl_controller")
    # policy = rospy.get_param("~policy")
    # data_collection = rospy.get_param("~data_collection")

    # config_dir = "/root/catkin_ws/src/WheeledLab-research/source/mpail/test/wordly-haze-1069"
    # config_dir = "/root/catkin_ws/src/WheeledLab-research/source/mpail/test/gail_disc"
    config_dir = "/root/catkin_ws/src/WheeledLab-research/source/mpail/test/test_mppi_save"

    # policy_config_path = os.path.join(config_dir, "policy_config.yaml")
    policy_config_path = "/root/catkin_ws/src/WheeledLab-research/source/mpail/test/test_mppi_save/DBF_MPAIL_2D_policy_config.yaml"
    policy_config = load_yaml(policy_config_path)

    # Find the latest model_x.pt file
    model_files = [f for f in os.listdir(config_dir) if f.startswith("model_") and f.endswith(".pt")]
    # latest_model_file = max(model_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
    latest_model_file = "model_100.pt"
    # latest_model_file = "disc_0.pt"

    model_path = os.path.join(config_dir, latest_model_file)
    policy_model = torch.load(model_path)['model_state_dict'] #uncomment this for mpail discriminator 
    # policy_model = torch.load(model_path)['cost_state_dict'] # uncommnet this for gail discriminator


    print(f"Loaded policy config from: {policy_config_path}")
    print(f"Loaded model from: {model_path}")

    cfg_class = MPAILPolicyCfg
    mppi_config = dataclass_from_yaml_recurse(MPAILPolicyCfg, policy_config)

    rollout_vis_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rollout_vis"
    )
    os.makedirs(rollout_vis_dir, exist_ok=True)
    planner = MPAIL_HL_Control(mppi_config, policy_model)
    rospy.spin()