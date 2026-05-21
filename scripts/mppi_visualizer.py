#!/usr/bin/env python3
"""Live MPPI rollout viewer subscribing to /mppi/* topics from mpail_hl_control."""
import atexit
import os
import threading
from io import BytesIO

import cv2
import imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rospy
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from std_msgs.msg import Float32MultiArray, Int32

try:
    import rospkg
except ImportError:
    rospkg = None


def _default_output_dir():
    if rospkg is not None:
        try:
            return os.path.join(rospkg.RosPack().get_path("dbf"), "rollout_vis")
        except rospkg.ResourceNotFound:
            pass
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "rollout_vis")
    )


class MPPIVisualizer:
    def __init__(self):
        rospy.init_node("mppi_visualizer", anonymous=True)

        self.vis_n_envs = rospy.get_param("~vis_n_envs", 1)
        self.vis_n_rollouts = rospy.get_param("~vis_n_rollouts", 10)
        self.map_length_px = int(rospy.get_param("~map_length_px", 20))
        self.map_res_m_px = float(rospy.get_param("~map_res_m_px", 0.15789473684210525))
        self.rollout_occupancy_idx = int(rospy.get_param("~rollout_occupancy_idx", 12))
        xlim = rospy.get_param("~xlim", None)
        ylim = rospy.get_param("~ylim", None)
        self.xlim = tuple(xlim) if xlim is not None else None
        self.ylim = tuple(ylim) if ylim is not None else None
        self.obs_side = rospy.get_param("~obstacle_side_m", 0.16)
        self.record_video = rospy.get_param("~record_video", True)
        self.save_frames = rospy.get_param("~save_frames", False)
        self.video_fps = int(rospy.get_param("~video_fps", 10))
        self.output_dir = os.path.abspath(
            rospy.get_param("~output_dir", _default_output_dir())
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self.img_frames = []
        self._video_saved = False
        self._last_captured_timestep = -1
        self.video_flush_every = int(rospy.get_param("~video_flush_every", 0))
        rospy.loginfo(
            "MPPI visualizer: record_video=%s, output_dir=%s",
            self.record_video,
            self.output_dir,
        )

        self.current_state = None
        self.rollouts = None
        self.costs = None
        self.obstacle_occupancy = None
        self.optimal_trajectory = None
        self.obstacle_pose = None
        self.frame_timestep = 0

        self.show_rollouts_scatter = True
        self.show_trajectory_trace = True
        self.show_occupancy_map = True
        self.trajectory_velocity = [[] for _ in range(self.vis_n_envs)]
        self.trajectory_traces = [[] for _ in range(self.vis_n_envs)]
        self.x = [[] for _ in range(self.vis_n_envs)]
        self.y = [[] for _ in range(self.vis_n_envs)]
        self.u = [[] for _ in range(self.vis_n_envs)]
        self.v = [[] for _ in range(self.vis_n_envs)]
        try:
            self.red_blue = mpl.colormaps["plasma"]
        except AttributeError:
            self.red_blue = mpl.cm.get_cmap("plasma")
        self.robot_marker_size = 0.35
        self._obstacle_patches = []

        self.data_lock = threading.Lock()
        self._setup_figure()
        self._setup_plot_elements()

        rospy.Subscriber("/mppi/current_state", Float32MultiArray, self.current_state_callback)
        rospy.Subscriber("/mppi/rollouts", Float32MultiArray, self.rollouts_callback)
        rospy.Subscriber("/mppi/obstacle_occupancy", Float32MultiArray, self.obstacle_occupancy_callback)
        rospy.Subscriber("/mppi/rollout_costs", Float32MultiArray, self.costs_callback)
        rospy.Subscriber("/mppi/optimal_trajectory", Float32MultiArray, self.optimal_trajectory_callback)
        rospy.Subscriber("/mppi/obstacle_pose", Float32MultiArray, self.obstacle_pose_callback)
        rospy.Subscriber("/mppi/frame_timestep", Int32, self.frame_timestep_callback)

        plt.ion()
        self.ani = FuncAnimation(self.fig, self.update, interval=100, blit=False)
        plt.show(block=False)
        rospy.on_shutdown(self._save_video_on_exit)
        atexit.register(self._save_video_on_exit)

    def reshape_array_from_msg(self, msg):
        dims = msg.layout.dim
        if not dims:
            return np.array(msg.data)
        shape = [d.size for d in dims]
        return np.array(msg.data, dtype=np.float32).reshape(shape)

    def current_state_callback(self, msg):
        with self.data_lock:
            self.current_state = self.reshape_array_from_msg(msg)

    def rollouts_callback(self, msg):
        with self.data_lock:
            self.rollouts = self.reshape_array_from_msg(msg)

    def obstacle_occupancy_callback(self, msg):
        with self.data_lock:
            self.obstacle_occupancy = self.reshape_array_from_msg(msg)

    def costs_callback(self, msg):
        with self.data_lock:
            self.costs = self.reshape_array_from_msg(msg)

    def optimal_trajectory_callback(self, msg):
        with self.data_lock:
            self.optimal_trajectory = self.reshape_array_from_msg(msg)

    def obstacle_pose_callback(self, msg):
        with self.data_lock:
            self.obstacle_pose = self.reshape_array_from_msg(msg)

    def frame_timestep_callback(self, msg):
        with self.data_lock:
            self.frame_timestep = msg.data

    def _setup_figure(self):
        self.fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        self.axes = [ax]

    def _robot_triangle_vertices(self, center_x, center_y, yaw, size):
        heading = np.array([np.cos(yaw), np.sin(yaw)])
        left = np.array([-heading[1], heading[0]])
        tip = np.array([center_x, center_y]) + heading * size
        rear_center = np.array([center_x, center_y]) - heading * (size * 0.7)
        rear_left = rear_center + left * (size * 0.45)
        rear_right = rear_center - left * (size * 0.45)
        return np.array([tip, rear_left, rear_right])

    def _setup_plot_elements(self):
        ax = self.axes[0]
        self.time_text = ax.text(
            0.05, 0.95, "", transform=ax.transAxes, fontsize=12, verticalalignment="top"
        )

        self.occupancy_images = []
        self.occupancy_rects = []
        if self.show_occupancy_map:
            map_size = (self.map_length_px - 1) * self.map_res_m_px
            half = map_size / 2
            for _ in range(self.vis_n_envs):
                self.occupancy_images.append(
                    ax.imshow(
                        np.zeros((self.map_length_px, self.map_length_px)),
                        cmap="gray_r",
                        extent=[-half, half, -half, half],
                        alpha=1.0,
                        origin="lower",
                        vmin=0,
                        vmax=1,
                    )
                )
                corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
                polygon = mpl.patches.Polygon(
                    corners, closed=True, edgecolor="black", facecolor="none", linewidth=2
                )
                ax.add_patch(polygon)
                self.occupancy_rects.append(polygon)

        if self.xlim:
            ax.set_xlim(self.xlim)
        else:
            ax.set_xlim(-1, 11)
        if self.ylim:
            ax.set_ylim(self.ylim)
        else:
            ax.set_ylim(-1, 11)

        ax.set_title("Trajectory Visualization", fontweight="bold")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        self.env_pos_scatters = {}
        self.env_control_scatters = {}
        self.env_rollout_lines = {}
        self.trajectory_start_scatters = {}
        self.env_rollout_scatters = {}

        sm_traj = mpl.cm.ScalarMappable(
            cmap=self.red_blue, norm=mpl.colors.Normalize(vmin=-1, vmax=1)
        )
        sm_traj.set_array([])
        cax = inset_axes(
            ax,
            width="30%",
            height="3%",
            loc="lower right",
            bbox_to_anchor=(-0.05, 0.08, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        self.traj_cbar = self.fig.colorbar(
            sm_traj, cax=cax, orientation="horizontal", label="Yaw Rate (deg/s)"
        )
        self.sm_traj = sm_traj

        for i in range(self.vis_n_envs):
            robot_triangle = mpl.patches.Polygon(
                self._robot_triangle_vertices(0.0, 0.0, 0.0, self.robot_marker_size),
                closed=True,
                facecolor="blue",
                edgecolor="black",
                linewidth=1.0,
                zorder=5,
            )
            ax.add_patch(robot_triangle)
            self.env_pos_scatters[i] = robot_triangle
            self.env_rollout_lines[i] = []
            self.env_rollout_scatters[i] = ax.scatter(
                [], [], alpha=0.95, s=26, edgecolors="black", linewidths=0.3, zorder=3
            )
            if self.show_rollouts_scatter:
                self.env_control_scatters[i] = ax.scatter(
                    [], [], c="green", alpha=1.0, s=20, edgecolors="black", linewidths=0.4
                )
            self.trajectory_start_scatters[i] = ax.scatter([], [], color="red", s=40, zorder=4)

        self.fig.tight_layout()

    def _capture_frame(self):
        self.fig.canvas.draw()
        buf = BytesIO()
        self.fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        self.img_frames.append(imageio.imread(buf))
        buf.close()

    def _save_frame_png(self, step):
        path = os.path.join(self.output_dir, f"step-{step}_rollouts_vis.png")
        self.fig.savefig(path, dpi=100)
        rospy.loginfo_throttle(30.0, f"Saved frame {path}")

    def _write_video_cv2(self, path):
        if not self.img_frames:
            return False
        frame0 = self.img_frames[0]
        if frame0.ndim != 3:
            return False
        h, w = frame0.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, self.video_fps, (w, h))
        if not writer.isOpened():
            return False
        for frame in self.img_frames:
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

    def _flush_video_checkpoint(self):
        """Write current frames to disk without stopping recording."""
        if not self.record_video or not self.img_frames:
            return
        path = os.path.join(self.output_dir, "rollouts_video.mp4")
        if self._write_video_cv2(path):
            rospy.loginfo_throttle(
                30.0,
                "Checkpoint video (%d frames): %s",
                len(self.img_frames),
                path,
            )

    def _save_video_on_exit(self):
        if self._video_saved or not self.record_video or not self.img_frames:
            return
        self._video_saved = True
        path = os.path.join(self.output_dir, "rollouts_video.mp4")
        n_frames = len(self.img_frames)
        if self._write_video_cv2(path):
            rospy.loginfo(
                "Saved rollout video (%d frames, %d fps): %s",
                n_frames,
                self.video_fps,
                path,
            )
            print(f"Saved rollout video ({n_frames} frames): {path}")
            return
        try:
            imageio.mimsave(path, self.img_frames, fps=self.video_fps)
            rospy.loginfo(
                "Saved rollout video via imageio (%d frames): %s", n_frames, path
            )
        except Exception as exc:
            gif_path = os.path.join(self.output_dir, "rollouts_video.gif")
            try:
                imageio.mimsave(gif_path, self.img_frames, fps=self.video_fps)
                rospy.logwarn(
                    "MP4 save failed (%s); saved GIF instead: %s", exc, gif_path
                )
            except Exception as gif_exc:
                rospy.logwarn(
                    "Failed to save rollout video (%d frames): %s; GIF also failed: %s",
                    n_frames,
                    exc,
                    gif_exc,
                )

    def _clear_obstacle_patches(self):
        for patch in self._obstacle_patches:
            try:
                patch.remove()
            except Exception:
                pass
        self._obstacle_patches = []

    def _draw_obstacles(self, env_idx=0):
        self._clear_obstacle_patches()
        if self.obstacle_pose is None:
            return
        poses = self.obstacle_pose
        if poses.ndim == 3:
            poses = poses[env_idx]
        if poses.ndim != 2 or poses.shape[1] < 2:
            return
        side = float(self.obs_side)
        for obs in poses:
            lower_left = (obs[0] - side / 2.0, obs[1] - side / 2.0)
            rect = mpl.patches.Rectangle(
                lower_left, side, side, color="gray", alpha=0.5, zorder=1
            )
            self.axes[0].add_patch(rect)
            self._obstacle_patches.append(rect)

    def update(self, _frame):
        with self.data_lock:
            if self.current_state is None or self.rollouts is None or self.costs is None:
                return []

            x0 = self.current_state
            rollouts = self.rollouts
            n_rollouts = min(self.vis_n_rollouts, rollouts.shape[1])

            for i in range(self.vis_n_envs):
                if i >= rollouts.shape[0] or i >= x0.shape[0]:
                    continue
                env_rollouts = rollouts[i]
                curr_x = float(x0[i, 0])
                curr_y = float(x0[i, 1])
                curr_yaw = float(x0[i, 5]) if x0.shape[1] > 5 else 0.0
                curr_vx = float(x0[i, 6]) if x0.shape[1] > 6 else 0.0
                curr_vy = float(x0[i, 7]) if x0.shape[1] > 7 else 0.0

                yaw = float(np.nan_to_num(curr_yaw, nan=0.0))
                self.env_pos_scatters[i].set_xy(
                    self._robot_triangle_vertices(curr_x, curr_y, yaw, self.robot_marker_size)
                )

                if (
                    self.show_rollouts_scatter
                    and self.optimal_trajectory is not None
                    and i < self.optimal_trajectory.shape[0]
                ):
                    opt = self.optimal_trajectory[i]
                    self.env_control_scatters[i].set_offsets(opt[:, :2])

                if self.show_occupancy_map and self.obstacle_occupancy is not None:
                    occ = self.obstacle_occupancy
                    if occ.ndim == 4:
                        occ = occ[i, ..., 0]
                    elif occ.ndim == 3 and occ.shape[0] == self.vis_n_envs:
                        occ = occ[i]
                    theta = yaw
                    c, s = np.cos(theta), np.sin(theta)
                    map_size = (self.map_length_px - 1) * self.map_res_m_px
                    half = map_size / 2
                    self.occupancy_images[i].set_data(np.abs(occ))
                    min_x = curr_x - map_size / 2
                    max_x = curr_x + map_size / 2
                    min_y = curr_y - map_size / 2
                    max_y = curr_y + map_size / 2
                    self.occupancy_images[i].set_extent([min_x, max_x, min_y, max_y])
                    img_transform = (
                        mpl.transforms.Affine2D().rotate_around(curr_x, curr_y, theta)
                        + self.axes[0].transData
                    )
                    self.occupancy_images[i].set_transform(img_transform)
                    corners = np.array(
                        [[-half, -half], [half, -half], [half, half], [-half, half]]
                    )
                    R = np.array([[c, -s], [s, c]])
                    self.occupancy_rects[i].set_xy(
                        (R @ corners.T).T + np.array([curr_x, curr_y])
                    )

                if self.show_rollouts_scatter:
                    all_x = np.concatenate([env_rollouts[j][:, 0] for j in range(n_rollouts)])
                    all_y = np.concatenate([env_rollouts[j][:, 1] for j in range(n_rollouts)])
                    num_points = all_x.shape[0]
                    all_colors = np.tile(
                        np.array([[0.1, 0.35, 0.9, 0.95]]), (num_points, 1)
                    )
                    occ_idx = self.rollout_occupancy_idx
                    if env_rollouts.shape[-1] > occ_idx:
                        all_occ = np.concatenate(
                            [env_rollouts[j][:, occ_idx] for j in range(n_rollouts)]
                        )
                        occupied_mask = all_occ < 0
                        if np.any(occupied_mask):
                            all_colors[occupied_mask, :3] = np.array([1.0, 0.0, 0.0])
                            all_colors[occupied_mask, 3] = 1.0
                    self.env_rollout_scatters[i].set_offsets(np.c_[all_x, all_y])
                    self.env_rollout_scatters[i].set_facecolor(all_colors)
                    self.env_rollout_scatters[i].set_edgecolors("black")

                if self.show_trajectory_trace:
                    self.trajectory_traces[i].append(x0[i].copy())
                    trace_states = np.array(self.trajectory_traces[i])
                    trace_x, trace_y = trace_states[:, 0], trace_states[:, 1]
                    if len(trace_states) > 1 and trace_states.shape[1] > 11:
                        all_yaw_rates = trace_states[:, 11] * 180.0 / np.pi
                        vmin, vmax = all_yaw_rates.min(), all_yaw_rates.max()
                        norm = mpl.colors.Normalize(
                            vmin=vmin - 0.1 if vmax <= vmin else vmin,
                            vmax=vmax + 0.1 if vmax <= vmin else vmax,
                        )
                        points = np.array([trace_x, trace_y]).T.reshape(-1, 1, 2)
                        segments = np.concatenate([points[:-1], points[1:]], axis=1)
                        segment_colors = self.red_blue(norm(all_yaw_rates[:-1]))
                        if len(self.env_rollout_lines[i]) == 0:
                            lc = LineCollection(
                                segments,
                                colors=segment_colors,
                                linewidth=2.0,
                                capstyle="round",
                                joinstyle="round",
                            )
                            self.env_rollout_lines[i].append(self.axes[0].add_collection(lc))
                        else:
                            self.env_rollout_lines[i][0].set_segments(segments)
                            self.env_rollout_lines[i][0].set_color(segment_colors)

                    if len(self.trajectory_traces[i]) > 0:
                        start_pt = np.asarray(self.trajectory_traces[i][0])[:2]
                        self.trajectory_start_scatters[i].set_offsets(np.array([start_pt]))

            self._draw_obstacles()
            self.time_text.set_text(f"Timestep: {self.frame_timestep}")

            if self.xlim is None or self.ylim is None:
                ax = self.axes[0]
                cx0, cx1 = ax.get_xlim()
                cy0, cy1 = ax.get_ylim()
                ax.set_xlim(
                    min(cx0, rollouts[..., 0].min() - 1),
                    max(cx1, rollouts[..., 0].max() + 1),
                )
                ax.set_ylim(
                    min(cy0, rollouts[..., 1].min() - 1),
                    max(cy1, rollouts[..., 1].max() + 1),
                )

        self.fig.canvas.draw_idle()

        if self.record_video and self.frame_timestep != self._last_captured_timestep:
            self._capture_frame()
            self._last_captured_timestep = self.frame_timestep
            if (
                self.video_flush_every > 0
                and len(self.img_frames) % self.video_flush_every == 0
            ):
                self._flush_video_checkpoint()
        if self.save_frames:
            self._save_frame_png(self.frame_timestep)

        return []


def main():
    MPPIVisualizer()
    rospy.spin()


if __name__ == "__main__":
    main()
