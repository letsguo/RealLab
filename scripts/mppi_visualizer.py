#!/usr/bin/env python3
"""
ROS decoupled version of rollout_viz that is used in sim. Avoids heavy computation
"""
import atexit
import os
import threading
from io import BytesIO

import cv2
import imageio
import matplotlib as mpl
# GPT fix, no gui render so don't try to initialize one
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rospy
from matplotlib.collections import LineCollection
# Agg has no event loop, func animation won't work -> pull data direct from topic
# from matplotlib.animation import FuncAnimation #old import
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from std_msgs.msg import Float32MultiArray, Int32

# path resolve trick
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

        # paramters that cna be overriden in launc file 
        self.vis_n_envs = rospy.get_param("~vis_n_envs", 1)            # how many robots/envs to draw
        self.vis_n_rollouts = rospy.get_param("~vis_n_rollouts", 10)   # cap on rollouts drawn per env
        self.map_length_px = int(rospy.get_param("~map_length_px", 20))            # occupancy grid side (px)
        self.map_res_m_px = float(rospy.get_param("~map_res_m_px", 0.15789473684210525))  # meters per pixel
        self.rollout_occupancy_idx = int(rospy.get_param("~rollout_occupancy_idx", 12))
        xlim = rospy.get_param("~xlim", None)
        ylim = rospy.get_param("~ylim", None)
        self.xlim = tuple(xlim) if xlim is not None else None
        self.ylim = tuple(ylim) if ylim is not None else None
        self.obs_side = rospy.get_param("~obstacle_side_m", 0.16)      # obstacle square side (m) for drawing
        self.record_video = rospy.get_param("~record_video", True)    # accumulate frames -> mp4 on exit
        self.save_frames = rospy.get_param("~save_frames", False)     # also dump per-step PNGs
        self.video_fps = int(rospy.get_param("~video_fps", 10))       # also the render-loop rate in main()
        self.output_dir = os.path.abspath(rospy.get_param("~output_dir", _default_output_dir()))
        self.output_filename = rospy.get_param("~output_filename", "rollouts_video.mp4")
        os.makedirs(self.output_dir, exist_ok=True)
        self._video_path = os.path.join(self.output_dir, self.output_filename)
        self._video_writer = None       # persistent cv2 writer; frames stream straight to disk
        self._video_writer_size = None
        self._use_streaming = True       # flips False if the writer can't open -> RAM fallback
        self.img_frames = []             # fallback buffer, only used when streaming unavailable
        self._video_saved = False
        self._frames_written = 0
        rospy.loginfo(
            "MPPI visualizer: record_video=%s, output_path=%s",
            self.record_video,
            self._video_path,
        )

        # read by update() state buffers
        # All start None; update() refuses to draw until the essential ones arrive.
        self.current_state = None
        self.rollouts = None
        self.costs = None
        self.obstacle_occupancy = None
        self.optimal_trajectory = None
        self.obstacle_pose = None
        self.frame_timestep = 0

        # ---- Feature toggles + per-env accumulators ----
        self.show_rollouts_scatter = True
        self.show_trajectory_trace = True
        self.show_occupancy_map = True
        # derivable from full state traces. Also unused already
        # self.trajectory_velocity = [[] for _ in range(self.vis_n_envs)]
        # trajectory_traces[i] grows over time: the full state at each rendered step.
        self.trajectory_traces = [[] for _ in range(self.vis_n_envs)]
        # self.x = [[] for _ in range(self.vis_n_envs)]
        # self.y = [[] for _ in range(self.vis_n_envs)]
        # self.u = [[] for _ in range(self.vis_n_envs)]
        # self.v = [[] for _ in range(self.vis_n_envs)]
        try:
            self.red_blue = mpl.colormaps["plasma"]
        except AttributeError:
            self.red_blue = mpl.cm.get_cmap("plasma")
        self.robot_marker_size = 0.15
        self._obstacle_patches = []

        # locks writes and upates to single thread for updating state vecs or reading in main
        self.data_lock = threading.Lock()
        self._setup_figure()         # create the matplotlib figure/axes
        self._setup_plot_elements()  # create all persistent artists once

        # ---- Subscriptions: one latest-value callback per topic ----
        rospy.Subscriber("/mppi/current_state", Float32MultiArray, self.current_state_callback)
        rospy.Subscriber("/mppi/rollouts", Float32MultiArray, self.rollouts_callback)
        rospy.Subscriber("/mppi/obstacle_occupancy", Float32MultiArray, self.obstacle_occupancy_callback)
        rospy.Subscriber("/mppi/rollout_costs", Float32MultiArray, self.costs_callback)
        rospy.Subscriber("/mppi/optimal_trajectory", Float32MultiArray, self.optimal_trajectory_callback)
        rospy.Subscriber("/mppi/obstacle_pose", Float32MultiArray, self.obstacle_pose_callback)
        rospy.Subscriber("/mppi/frame_timestep", Int32, self.frame_timestep_callback)

        # Save once when main()'s loop exits (see main()); atexit is a backstop.
        # NOT rospy.on_shutdown: that fires in the shutdown thread while the render
        # loop is still spinning, so the loop re-opens (truncates) the mp4 we just
        # finalized -> a 1-frame file. Finalizing after the loop avoids the race.
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
        """Create the single-axes figure we draw everything onto."""
        self.fig, ax = plt.subplots(1, 1, figsize=(10, 8)) # HACK: LOOK HERE FOR FIGURE SIZE
        self.axes = [ax]  # keep as a list to mirror RolloutsVisualization's API

    def _robot_triangle_vertices(self, center_x, center_y, yaw, size):
        heading = np.array([np.cos(yaw), np.sin(yaw)])
        left = np.array([-heading[1], heading[0]])  # 90deg CCW from heading
        tip = np.array([center_x, center_y]) + heading * size
        rear_center = np.array([center_x, center_y]) - heading * (size * 0.7)
        rear_left = rear_center + left * (size * 0.45)
        rear_right = rear_center - left * (size * 0.45)
        return np.array([tip, rear_left, rear_right])

    def _setup_plot_elements(self):
        ax = self.axes[0]
        # Top-left timestep readout.
        self.time_text = ax.text(
            0.05, 0.95, "", transform=ax.transAxes, fontsize=12, verticalalignment="top"
        )

        # One occupancy image + its outline rectangle per env.
        self.occupancy_images = []
        self.occupancy_rects = []
        if self.show_occupancy_map:
            # Physical size of the (square) occupancy grid in meters.
            # imshow extent is the grid's OUTER edges, which span the full
            # map_length_px*res (the builder's half_extent = map_length_px*res/2).
            # Using (px-1) here compresses the image by (px-1)/px, misaligning the
            # occupancy cells with the world-frame ground-truth obstacles.
            map_size = self.map_length_px * self.map_res_m_px
            half = map_size / 2
            for _ in range(self.vis_n_envs):
                # Grayscale image; gray_r so occupied (high) renders dark.
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
                # A square outline showing the footprint/extent of the grid.
                corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
                polygon = mpl.patches.Polygon(
                    corners, closed=True, edgecolor="black", facecolor="none", linewidth=2
                )
                ax.add_patch(polygon)
                self.occupancy_rects.append(polygon)
        if self.xlim:
            ax.set_xlim(self.xlim)
        else:
            ax.set_xlim(-3, 3)
        if self.ylim:
            ax.set_ylim(self.ylim)
        else:
            ax.set_ylim(-3, 3)

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

    def _frame_to_bgr(self, frame, size=None):
        if size is not None and (frame.shape[1], frame.shape[0]) != size:
            frame = cv2.resize(frame, size)
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def _open_video_writer(self, frame):
        """Open the persistent mp4 writer sized to `frame` (lazily, on the first
        frame). False if no mp4 codec is available -> caller uses the RAM fallback.
        """
        if frame.ndim < 2:
            return False
        h, w = frame.shape[:2]
        writer = cv2.VideoWriter(
            self._video_path, cv2.VideoWriter_fourcc(*"mp4v"), self.video_fps, (w, h)
        )
        if not writer.isOpened():
            rospy.logwarn("cv2.VideoWriter could not open %s (no mp4 codec?); buffering for exit.", self._video_path)
            return False
        self._video_writer = writer
        self._video_writer_size = (w, h)
        rospy.loginfo("Streaming rollout video to %s (%dx%d @ %d fps)", self._video_path, w, h, self.video_fps)
        return True

    def _capture_frame(self):
        """Render the figure and stream one frame to the mp4 (RAM-buffer fallback)."""
        if self._video_saved:  # never re-open the writer after the final save
            return
        self.fig.canvas.draw()
        buf = BytesIO()
        self.fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        frame = imageio.imread(buf)
        buf.close()
        # DEBUG: confirms capture fires repeatedly + whether the pose actually moves.
        if self.current_state is not None and self.current_state.ndim == 2 and self.current_state.shape[1] >= 2:
            rospy.loginfo_throttle(
                2.0,
                "capture #%d ts=%s pos=(%.3f, %.3f)",
                self._frames_written,
                self.frame_timestep,
                float(self.current_state[0, 0]),
                float(self.current_state[0, 1]),
            )
        if self._use_streaming:
            if self._video_writer is None and not self._open_video_writer(frame):
                self._use_streaming = False
            if self._video_writer is not None:
                self._video_writer.write(self._frame_to_bgr(frame, self._video_writer_size))
                self._frames_written += 1
                return
        self.img_frames.append(frame)
        self._frames_written += 1

    def _save_frame_png(self, step):
        """Optionally dump a single PNG per step (when ~save_frames is true)."""
        path = os.path.join(self.output_dir, f"step-{step}_rollouts_vis.png")
        self.fig.savefig(path, dpi=100)
        rospy.loginfo_throttle(30.0, f"Saved frame {path}")

    def _save_video_on_exit(self):
        if self._video_saved or not self.record_video:
            return
        self._video_saved = True
        path = self._video_path
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
        elif self.img_frames:
            try:
                imageio.mimsave(path, self.img_frames, fps=self.video_fps)
            except Exception as exc:
                rospy.logwarn("Failed to save rollout video: %s", exc)
                return
        else:
            return
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            rospy.loginfo("Saved rollout video (%d frames, %d fps): %s", self._frames_written, self.video_fps, path)
            print(f"Saved rollout video ({self._frames_written} frames): {path}")
        else:
            rospy.logwarn("Rollout video missing/empty (%d frames captured): %s", self._frames_written, path)

    def _clear_obstacle_patches(self):
        """Remove last frame's obstacle rectangles (they're redrawn each frame)."""
        for patch in self._obstacle_patches:
            try:
                patch.remove()
            except Exception:
                pass
        self._obstacle_patches = []

    def _draw_obstacles(self, env_idx=0):
        """Draw gray squares at each obstacle's world-frame center.
        """
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
            # Rectangle is anchored at its lower-left corner, so offset by half side.
            lower_left = (obs[0] - side / 2.0, obs[1] - side / 2.0)
            rect = mpl.patches.Rectangle(
                lower_left, side, side, color="gray", alpha=0.5, zorder=1
            )
            self.axes[0].add_patch(rect)
            self._obstacle_patches.append(rect)

    def update(self, _frame):
        """Render ONE frame from the latest buffered data.
        """
        with self.data_lock:
            # do nothign if we don't have appropriate data
            if self.current_state is None or self.rollouts is None or self.costs is None:
                return []

            x0 = self.current_state          # [env, state_dim]
            rollouts = self.rollouts         # [env, rollout, horizon, state_dim]
            n_rollouts = min(self.vis_n_rollouts, rollouts.shape[1])

            for i in range(self.vis_n_envs):
                # Skip envs the publisher didn't include this frame.
                if i >= rollouts.shape[0] or i >= x0.shape[0]:
                    continue
                env_rollouts = rollouts[i]   # [rollout, horizon, state_dim]
                # Pull pose components out of the state vector (guarded by width).
                curr_x = float(x0[i, 0])
                curr_y = float(x0[i, 1])
                curr_yaw = float(x0[i, 5]) if x0.shape[1] > 5 else 0.0
                curr_vx = float(x0[i, 6]) if x0.shape[1] > 6 else 0.0
                curr_vy = float(x0[i, 7]) if x0.shape[1] > 7 else 0.0

                # --- Robot icon ---
                yaw = float(np.nan_to_num(curr_yaw, nan=0.0))  # NaN yaw -> 0 (avoid breaking geometry)
                self.env_pos_scatters[i].set_xy(
                    self._robot_triangle_vertices(curr_x, curr_y, yaw, self.robot_marker_size)
                )

                # --- Optimal (chosen) trajectory: green dots ---
                if (
                    self.show_rollouts_scatter
                    and self.optimal_trajectory is not None
                    and i < self.optimal_trajectory.shape[0]
                ):
                    opt = self.optimal_trajectory[i]        # [horizon, state_dim]
                    self.env_control_scatters[i].set_offsets(opt[:, :2])  # x,y only

                # --- Robot-centric occupancy grid, rotated into the world ---
                if self.show_occupancy_map and self.obstacle_occupancy is not None:
                    occ = self.obstacle_occupancy
                    # Normalize to a 2-D [H, W] grid for this env.
                    if occ.ndim == 4:
                        occ = occ[i, ..., 0]
                    elif occ.ndim == 3 and occ.shape[0] == self.vis_n_envs:
                        occ = occ[i]
                    theta = yaw
                    c, s = np.cos(theta), np.sin(theta)
                    # imshow extent is the grid's OUTER edges, which span the full
                    # map_length_px*res (builder half_extent = map_length_px*res/2).
                    # Using (px-1) here compresses the image by (px-1)/px, misaligning
                    # the occupancy cells with the world-frame ground-truth obstacles.
                    map_size = self.map_length_px * self.map_res_m_px
                    half = map_size / 2
                    self.occupancy_images[i].set_data(np.abs(occ))
                    # Place the (axis-aligned) image at the robot, then rotate the
                    # whole image about the robot by yaw via an affine transform.
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
                    # Rotate the outline rectangle's corners to match the image.
                    corners = np.array(
                        [[-half, -half], [half, -half], [half, half], [-half, half]]
                    )
                    R = np.array([[c, -s], [s, c]])  # 2-D rotation matrix
                    self.occupancy_rects[i].set_xy(
                        (R @ corners.T).T + np.array([curr_x, curr_y])
                    )

                # --- The rollout point cloud (the thing we care most about) ---
                if self.show_rollouts_scatter:
                    # Flatten every rollout's (x, y) over the horizon into one cloud.
                    all_x = np.concatenate([env_rollouts[j][:, 0] for j in range(n_rollouts)])
                    all_y = np.concatenate([env_rollouts[j][:, 1] for j in range(n_rollouts)])
                    num_points = all_x.shape[0]
                    # Default color: semi-transparent blue (RGBA), one row per point.
                    all_colors = np.tile(
                        np.array([[0.1, 0.35, 0.9, 0.95]]), (num_points, 1)
                    )
                    # If the state carries the occupancy feature, recolor collisions red.
                    occ_idx = self.rollout_occupancy_idx
                    if env_rollouts.shape[-1] > occ_idx:
                        all_occ = np.concatenate(
                            [env_rollouts[j][:, occ_idx] for j in range(n_rollouts)]
                        )
                        occupied_mask = all_occ < 0  # <0 == in collision
                        if np.any(occupied_mask):
                            all_colors[occupied_mask, :3] = np.array([1.0, 0.0, 0.0])
                            all_colors[occupied_mask, 3] = 1.0
                    self.env_rollout_scatters[i].set_offsets(np.c_[all_x, all_y])
                    self.env_rollout_scatters[i].set_facecolor(all_colors)
                    self.env_rollout_scatters[i].set_edgecolors("black")

                # --- The robot's ACTUAL trajectory so far, colored by yaw rate ---
                if self.show_trajectory_trace:
                    # Append the full current state, then rebuild the polyline.
                    self.trajectory_traces[i].append(x0[i].copy())
                    trace_states = np.array(self.trajectory_traces[i])  # [steps, state_dim]
                    trace_x, trace_y = trace_states[:, 0], trace_states[:, 1]
                    # Need >=2 points to form a segment, and a yaw-rate column (idx 11).
                    if len(trace_states) > 1 and trace_states.shape[1] > 11:
                        all_yaw_rates = trace_states[:, 11] * 180.0 / np.pi  # rad/s -> deg/s
                        vmin, vmax = all_yaw_rates.min(), all_yaw_rates.max()
                        # Guard against vmin==vmax (degenerate colormap range).
                        norm = mpl.colors.Normalize(
                            vmin=vmin - 0.1 if vmax <= vmin else vmin,
                            vmax=vmax + 0.1 if vmax <= vmin else vmax,
                        )
                        # Build per-segment endpoints for a LineCollection.
                        points = np.array([trace_x, trace_y]).T.reshape(-1, 1, 2)
                        segments = np.concatenate([points[:-1], points[1:]], axis=1)
                        # Color each segment by its starting yaw rate.
                        segment_colors = self.red_blue(norm(all_yaw_rates[:-1]))
                        if len(self.env_rollout_lines[i]) == 0:
                            # First time: create the LineCollection and store it.
                            lc = LineCollection(
                                segments,
                                colors=segment_colors,
                                linewidth=2.0,
                                capstyle="round",
                                joinstyle="round",
                            )
                            self.env_rollout_lines[i].append(self.axes[0].add_collection(lc))
                        else:
                            # Subsequent frames: just update geometry + colors.
                            self.env_rollout_lines[i][0].set_segments(segments)
                            self.env_rollout_lines[i][0].set_color(segment_colors)

                    # Red dot at the very first point of the trace.
                    if len(self.trajectory_traces[i]) > 0:
                        start_pt = np.asarray(self.trajectory_traces[i][0])[:2]
                        self.trajectory_start_scatters[i].set_offsets(np.array([start_pt]))

            # Obstacles + timestep readout (shared across envs).
            self._draw_obstacles()
            self.time_text.set_text(f"Timestep: {self.frame_timestep}")

            # Auto-fit the view to the rollouts when no fixed limits were given.
            # Only ever widens (min/max with current limits), never shrinks.
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

        # Capture one frame per render tick (we only reach here once data is ready);
        if self.record_video:
            self._capture_frame()
        if self.save_frames:
            self._save_frame_png(self.frame_timestep)


def main():
    viz = MPPIVisualizer()
    rate = rospy.Rate(viz.video_fps if viz.video_fps > 0 else 10)
    try:
        while not rospy.is_shutdown():
            viz.update(None)
            rate.sleep()
    finally:
        viz._save_video_on_exit()  # finalize in this (main) thread, after the loop


if __name__ == "__main__":
    main()
