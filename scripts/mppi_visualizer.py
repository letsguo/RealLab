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
# Agg has no event loop, func animation won't work -> pull data direct from topic
# from matplotlib.animation import FuncAnimation #old import
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from std_msgs.msg import Float32MultiArray, Int32

# Paper-figure typography. Applied at module import so any figure created in
# this process inherits it -- safe because this module is only ever imported
# by the visualizer node.
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 0.8,
})

# Wong colorblind-safe palette
# (https://www.nature.com/articles/nmeth.1618 fig 2). Each role here has a
# fixed slot so palette swaps stay coherent across artists.
_PALETTE = {
    "robot": "#0072B2",       # Wong blue
    "chosen": "#F2C744",      # warm yellow -- chosen/optimal MPPI rollout polyline
    "collision": "#D55E00",   # Wong vermilion -- per-point collision recolor
    "barrier": "#000000",     # zero-contour line; black to match the viridis
                              # heatmap aesthetic (yellow=safe -> dark=unsafe,
                              # black line marking the boundary).
    "trajectory": "#000000",  # robot's actual executed path
    "start_dot": "#CC79A7",   # Wong reddish purple -- distinct from collision
    "obstacle": "#264653",    # dark teal-slate; sits naturally against the
                              # viridis heatmap's dark end without disappearing
                              # into it, no hatching needed.
}

# Per-point colormap for top-K rollout samples. Linear navy -> muted plum ->
# bright red, no white midpoint -- so the fan reads as "warm blueish with red
# on the high-cost end" matching the reference figure the user shared.
_SAMPLE_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "mppi_sample_cost",
    ["#1f3a5f", "#3a4a7c", "#7a3f6f", "#c4445f", "#d63f3f"],
)
# Cyclic warm-toned colormap for the trajectory's yaw-angle coloring
# (state[5], wrapped to [0, 2pi] by the controller). Stays inside the warm
# half of the wheel (wine -> red -> orange -> gold -> peach -> pink ->
# magenta -> wine) so opposite yaws still get distinct hues without dipping
# into greens or blues.
_YAW_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "yaw_warm_cyclic",
    [
        "#2D0A0E",  # 0.00 very dark wine
        "#5C1A22",  # 0.10 dark wine
        "#8B1E2F",  # 0.20 deep red
        "#C9303E",  # 0.30 crimson
        "#E63946",  # 0.40 red
        "#F77F00",  # 0.50 orange
        "#FCBF49",  # 0.60 gold
        "#F4A582",  # 0.70 peach
        "#E07A9B",  # 0.80 warm pink
        "#B83F73",  # 0.90 raspberry
        "#2D0A0E",  # 1.00 back to wine (cyclic)
    ],
)
# state[5] is yaw (rad), wrapped to [0, 2*pi] by mpail_hl_control.obtain_state.
_YAW_IDX = 5

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
        # Additionally render the K worst (highest-cost) rollouts so readers
        # can see what the planner rejected -- not just the top-K it chose
        # between. Set to 0 to disable. These are drawn with the same cost
        # gradient (so they naturally land at the red end of _SAMPLE_CMAP)
        # but at lower alpha to keep them visually subordinate to the top-K.
        self.vis_n_rejected = int(rospy.get_param("~vis_n_rejected", 5))
        self.map_length_px = int(rospy.get_param("~map_length_px", 20))            # occupancy grid side (px)
        self.map_res_m_px = float(rospy.get_param("~map_res_m_px", 0.15789473684210525))  # meters per pixel
        self.rollout_occupancy_idx = int(rospy.get_param("~rollout_occupancy_idx", 12))
        xlim = rospy.get_param("~xlim", None)
        ylim = rospy.get_param("~ylim", None)
        self.xlim = tuple(xlim) if xlim is not None else None
        self.ylim = tuple(ylim) if ylim is not None else None
        self.obs_side = rospy.get_param("~obstacle_side_m", 0.16)      # obstacle square side (m) for drawing
        # Fixed range for the sample-cost colorbar. Per-frame [5, 95]
        # percentile rescaling made the legend visibly resize each frame
        # (same glitch as yaw rate had before); locking the range keeps the
        # cbar still while still being adjustable per-platform via launch.
        self.sample_cost_min = float(rospy.get_param("~sample_cost_min", -2.0))
        self.sample_cost_max = float(rospy.get_param("~sample_cost_max", 5.0))
        # One-shot sidecar PNG: full h(s) heatmap + zero contour + obstacles.
        # Saved on the FIRST frame that has both barrier_Z and obstacle data
        # so we capture the configured layout. The heatmap is kept out of the
        # mp4 (it competed with the rollout fan for attention) but is useful
        # as a paper reference figure on its own. Set save_barrier_heatmap=False
        # to disable.
        self.save_barrier_heatmap = bool(rospy.get_param("~save_barrier_heatmap", True))
        self.barrier_heatmap_filename = rospy.get_param(
            "~barrier_heatmap_filename", "barrier_heatmap.png",
        )
        # Percentile of |h(s)| used to set the heatmap's symmetric colormap
        # range. Obstacle interiors are extreme outliers (h very negative)
        # that otherwise compress the free-space gradient into invisibility.
        self.barrier_clip_pct = float(rospy.get_param("~barrier_clip_pct", 90.0))
        self._heatmap_png_saved = False
        # Paper mode strips chrome (title, timestep overlay) and tightens the
        # grid; intended for stills bundled into the paper. Live demo runs
        # should leave it False so the recorded video keeps the timestep.
        self.paper_mode = bool(rospy.get_param("~paper_mode", False))
        # Crop the view to the robot-centered BEV footprint each frame instead
        # of the static xlim/ylim. Useful when the paper figure shows one
        # instance from a longer rollout -- the reader only needs the local
        # context. Note this only changes what's *displayed*: self.xlim/ylim
        # still match the controller's barrier-eval extent so the published
        # h(s) field reconstructs correctly.
        self.zoom_to_bev = bool(rospy.get_param("~zoom_to_bev", False))
        # Top-down robot silhouette dimensions (m). Defaults sized to a 1/10
        # MuSHR-like platform; override per-platform if needed.
        self._robot_length = float(rospy.get_param("~robot_length_m", 0.35))
        self._robot_width = float(rospy.get_param("~robot_width_m", 0.20))
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
        # h(s) sampled on the controller's eval grid; viz extracts the zero
        # contour itself. None means "nothing received yet".
        self.barrier_Z = None
        # Per-tick world-frame extent [x0, x1, y0, y1] of the controller's eval
        # grid. The grid is now robot-centered so a fixed bounds assumption
        # cuts the contour off whenever the robot wanders away from
        # launch-time ~xlim/~ylim; the controller tells us where the grid is
        # this tick via /mppi/barrier_grid_extent. None falls back to the
        # static ~xlim/~ylim for backwards compatibility.
        self.barrier_extent = None
        # Full-workspace h(s) + extent (separate from the per-tick BEV grid
        # above). Drives the sidecar heatmap PNG so it reads 1:1 against the
        # spatial overview no matter where the robot currently is.
        self.barrier_Z_full = None
        self.barrier_extent_full = None
        # Barrier artists (zero line + interior fill) kept as a list so we can
        # purge both each frame before redrawing.
        self._barrier_artists = []

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
        # Top-K rollout coloring uses the module-level _SAMPLE_CMAP per-point,
        # see the rollouts block in update(). No persistent self attribute
        # needed -- the cmap is stateless and accessed by name.
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
        rospy.Subscriber("/mppi/barrier_grid", Float32MultiArray, self.barrier_grid_callback)
        rospy.Subscriber("/mppi/barrier_grid_extent", Float32MultiArray, self.barrier_extent_callback)
        rospy.Subscriber("/mppi/barrier_grid_full", Float32MultiArray, self.barrier_grid_full_callback)
        rospy.Subscriber("/mppi/barrier_grid_full_extent", Float32MultiArray, self.barrier_extent_full_callback)
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

    def barrier_grid_callback(self, msg):
        """Raw h(s) on the controller's robot-centered eval grid; viz extracts
        the zero contour. Empty array clears any drawn contour next frame.
        Pairs with barrier_extent_callback for X, Y reconstruction.
        """
        arr = self.reshape_array_from_msg(msg)
        with self.data_lock:
            if arr is None or arr.size == 0 or arr.ndim != 2:
                self.barrier_Z = None
                return
            self.barrier_Z = arr

    def barrier_extent_callback(self, msg):
        """World-frame [x0, x1, y0, y1] of the per-tick eval grid. Empty
        array clears the cached extent so we fall back to the static
        ~xlim/~ylim until the next valid extent arrives.
        """
        arr = self.reshape_array_from_msg(msg)
        with self.data_lock:
            if arr is None or arr.size != 4:
                self.barrier_extent = None
                return
            flat = arr.flatten()
            self.barrier_extent = (
                float(flat[0]), float(flat[1]), float(flat[2]), float(flat[3])
            )

    def barrier_grid_full_callback(self, msg):
        """Full-workspace h(s) for the sidecar heatmap PNG (independent of
        the per-tick robot-centered grid above)."""
        arr = self.reshape_array_from_msg(msg)
        with self.data_lock:
            if arr is None or arr.size == 0 or arr.ndim != 2:
                self.barrier_Z_full = None
                return
            self.barrier_Z_full = arr

    def barrier_extent_full_callback(self, msg):
        """World-frame [x0, x1, y0, y1] of the full-workspace eval grid."""
        arr = self.reshape_array_from_msg(msg)
        with self.data_lock:
            if arr is None or arr.size != 4:
                self.barrier_extent_full = None
                return
            flat = arr.flatten()
            self.barrier_extent_full = (
                float(flat[0]), float(flat[1]), float(flat[2]), float(flat[3])
            )

    def frame_timestep_callback(self, msg):
        with self.data_lock:
            self.frame_timestep = msg.data

    def _setup_figure(self):
        """Create the single-axes figure we draw everything onto."""
        self.fig, ax = plt.subplots(1, 1, figsize=(10, 8)) # HACK: LOOK HERE FOR FIGURE SIZE
        self.axes = [ax]  # keep as a list to mirror RolloutsVisualization's API

    def _robot_polygon_vertices(self, center_x, center_y, yaw, length=None, width=None):
        """Top-down robot marker: simple triangle pointing along the heading.
        3 vertices (rear_left, rear_right, nose) -- cleaner than the previous
        pentagon, reads as a directional arrow at small scales.
        """
        length = self._robot_length if length is None else length
        width = self._robot_width if width is None else width
        heading = np.array([np.cos(yaw), np.sin(yaw)])
        left = np.array([-heading[1], heading[0]])  # 90deg CCW from heading
        center = np.array([center_x, center_y])
        half_l = length / 2.0
        half_w = width / 2.0

        rear_left = center - heading * half_l + left * half_w
        rear_right = center - heading * half_l - left * half_w
        nose = center + heading * half_l
        return np.array([rear_left, rear_right, nose])

    def _setup_plot_elements(self):
        ax = self.axes[0]
        # Top-left timestep readout. Suppressed in paper_mode -- the figure
        # caption carries that info instead.
        if not self.paper_mode:
            self.time_text = ax.text(
                0.05, 0.95, "", transform=ax.transAxes, fontsize=11,
                verticalalignment="top",
            )
        else:
            self.time_text = None

        # One occupancy image + its outline rectangle per env. Skipped in
        # paper_mode -- the rotated body-frame image confuses readers of
        # still figures; world-axis-aligned obstacle halos in _draw_obstacles
        # convey the same keep-out info more cleanly.
        self.occupancy_images = []
        self.occupancy_rects = []
        if self.show_occupancy_map and not self.paper_mode:
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

        # Title is filler for a paper figure (caption does the work); kept only
        # for the live demo mode.
        if not self.paper_mode:
            ax.set_title("Trajectory Visualization", fontweight="bold")
        ax.set_xlabel(r"$x$ [m]")
        ax.set_ylabel(r"$y$ [m]")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.tick_params(direction="in", length=4)

        # (Scale bar removed -- the visible grid already gives the reader a
        # 1 m reference and a separate scale bar was visual clutter.)

        # Two colorbars split top/bottom on the right edge -- the middle is
        # left clear so the trajectory and rollout fan don't share x-extent
        # with a colorbar. Both cbars have a 10% gap from their respective
        # edge (top for yaw, bottom for cost) so they look symmetric about
        # the midline. Scale bar sits below the cost cbar in the corner.
        #
        # Yaw angle: upper half. Fixed [0, 2*pi] norm since the controller
        # wraps yaw -- no per-frame rescaling, no legend flicker.
        self._yaw_sm = mpl.cm.ScalarMappable(
            cmap=_YAW_CMAP,
            norm=mpl.colors.Normalize(vmin=0.0, vmax=2 * np.pi),
        )
        self._yaw_sm.set_array([])
        cax_yaw = inset_axes(
            ax, width="3.5%", height="35%", loc="upper right",
            bbox_to_anchor=(0.02, -0.10, 1, 1), bbox_transform=ax.transAxes,
            borderpad=0,
        )
        self._yaw_cbar = self.fig.colorbar(
            self._yaw_sm, cax=cax_yaw, label=r"$\psi$ [rad]",
        )
        # Tick at multiples of pi/2 with LaTeX-style labels -- reads as an
        # angle in robotics-paper convention rather than [0, 6.28] decimals.
        self._yaw_cbar.set_ticks([0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0, 2.0 * np.pi])
        self._yaw_cbar.set_ticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
        self._yaw_cbar.ax.tick_params(labelsize=11)

        # MPPI sample cost: lower half. Repurposed slot from the old h(s)
        # heatmap cbar. Norm is FIXED to [sample_cost_min, sample_cost_max]
        # for the life of the run -- per-frame percentile rescaling caused
        # the cbar tick labels to resize each frame.
        self._cost_norm = mpl.colors.Normalize(
            vmin=self.sample_cost_min, vmax=self.sample_cost_max,
        )
        self._cost_sm = mpl.cm.ScalarMappable(cmap=_SAMPLE_CMAP, norm=self._cost_norm)
        self._cost_sm.set_array([])
        cax_cost = inset_axes(
            ax, width="3.5%", height="35%", loc="lower right",
            bbox_to_anchor=(0.02, 0.10, 1, 1), bbox_transform=ax.transAxes,
            borderpad=0,
        )
        self._cost_cbar = self.fig.colorbar(
            self._cost_sm, cax=cax_cost, label=r"sample cost",
        )
        self._cost_cbar.ax.tick_params(labelsize=11)

        self.env_pos_scatters = {}
        self.env_optimal_lines = {}
        self.trajectory_lcs = {}
        self.trajectory_start_scatters = {}
        self.env_rollout_scatters = {}

        for i in range(self.vis_n_envs):
            # Yellow triangle robot marker with black outline -- matches the
            # optimal-rollout polyline color, reads as a directional arrow.
            robot_body = mpl.patches.Polygon(
                self._robot_polygon_vertices(0.0, 0.0, 0.0),
                closed=True,
                facecolor=_PALETTE["chosen"],
                edgecolor="black",
                linewidth=1.2,
                zorder=5,
            )
            ax.add_patch(robot_body)
            self.env_pos_scatters[i] = robot_body
            # Robot's actual trajectory: LineCollection so each segment can be
            # colored by yaw (rad) via _YAW_CMAP. Norm matches the cbar's
            # fixed [0, 2*pi] range so segment hue and cbar tick labels stay
            # aligned without per-frame updates.
            traj_lc = LineCollection(
                [], cmap=_YAW_CMAP,
                norm=mpl.colors.Normalize(vmin=0.0, vmax=2 * np.pi),
                linewidth=2.4, zorder=3,
            )
            ax.add_collection(traj_lc)
            self.trajectory_lcs[i] = traj_lc
            self.env_rollout_scatters[i] = ax.scatter(
                [], [], alpha=0.95, s=26, edgecolors="black", linewidths=0.3, zorder=3
            )
            if self.show_rollouts_scatter:
                # Chosen rollout: dense yellow polyline that sits on top of
                # the top-K sample fan. The polyline (vs scatter) reads as a
                # continuous predicted path; yellow stays clear against the
                # blue/red sample gradient and the red barrier line.
                (line,) = ax.plot(
                    [], [], color=_PALETTE["chosen"], linewidth=3.5,
                    solid_capstyle="round", solid_joinstyle="round", zorder=5,
                )
                self.env_optimal_lines[i] = line
            self.trajectory_start_scatters[i] = ax.scatter(
                [], [], color=_PALETTE["start_dot"], s=40, zorder=4,
            )

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

    def _save_barrier_heatmap_png(self):
        """One-shot sidecar PNG of the DBF barrier field. Renders the full
        h(s) heatmap + zero contour + obstacles on a fresh figure (does NOT
        touch the main fig used for the mp4) and saves to disk. Self-gated:
        runs exactly once when the first frame with both barrier_Z and an
        obstacle pose set arrives. After that this is a no-op.
        """
        if self._heatmap_png_saved or not self.save_barrier_heatmap:
            return
        # Drives the heatmap from the FULL-WORKSPACE h(s) grid
        # (/mppi/barrier_grid_full), not the per-tick robot-centered grid the
        # mp4 contour uses. This makes the heatmap PNG line up 1:1 with the
        # spatial overview regardless of where the robot is at save time.
        if self.barrier_Z_full is None or self.barrier_Z_full.size == 0:
            return
        if self.barrier_extent_full is None:
            return  # need extent to reconstruct X, Y
        bx0, bx1, by0, by1 = self.barrier_extent_full

        Z = self.barrier_Z_full
        h, w = Z.shape
        gx = np.linspace(bx0, bx1, w)
        gy = np.linspace(by0, by1, h)
        X, Y = np.meshgrid(gx, gy)

        # Symmetric clipping at a percentile of |h| so obstacle-interior
        # outliers don't compress the free-space gradient to invisibility.
        abs_z = np.abs(Z[np.isfinite(Z)])
        if abs_z.size > 0:
            vmax = max(float(np.percentile(abs_z, self.barrier_clip_pct)), 1e-3)
        else:
            vmax = 1e-3
        norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        # viridis: yellow at vmax (positive h, safe) -> dark purple at vmin
        # (negative h, unsafe). Matches the "Positive -> Negative" reference
        # colorbar from the paper figures the user shared.
        mesh = ax.pcolormesh(
            X, Y, Z, cmap="viridis", norm=norm,
            shading="gouraud", rasterized=True,
        )
        # Black zero-level line -- reads against viridis's mid-teal/green
        # midpoint cleanly without competing chromatically.
        ax.contour(
            X, Y, Z, levels=[0],
            colors="black", linewidths=2.5,
        )

        # Obstacles for spatial context. Solid dark-teal squares, same style
        # as the mp4 figure.
        if self.obstacle_pose is not None:
            poses = self.obstacle_pose
            if poses.ndim == 3:
                poses = poses[0]
            if poses.ndim == 2 and poses.shape[1] >= 2:
                side = float(self.obs_side)
                for obs in poses:
                    rect = mpl.patches.Rectangle(
                        (obs[0] - side / 2.0, obs[1] - side / 2.0), side, side,
                        facecolor=_PALETTE["obstacle"],
                        edgecolor="black",
                        linewidth=1.0, alpha=0.95, zorder=2,
                    )
                    ax.add_patch(rect)

        ax.set_xlim(bx0, bx1)
        ax.set_ylim(by0, by1)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.tick_params(direction="in", length=4)
        ax.set_xlabel(r"$x$ [m]")
        ax.set_ylabel(r"$y$ [m]")
        fig.colorbar(mesh, ax=ax, label=r"$h(s)$", shrink=0.7)
        fig.tight_layout()

        out_path = os.path.join(self.output_dir, self.barrier_heatmap_filename)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        rospy.loginfo("Saved barrier heatmap PNG: %s", out_path)
        print(f"Saved barrier heatmap PNG: {out_path}")
        self._heatmap_png_saved = True

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
        """Solid dark-teal square per obstacle, matching the reference paper
        figure aesthetic (the obstacles read against a viridis heatmap as a
        slightly darker block, but still visible against a white background).
        No hatching, no inflation halo -- the barrier zero-contour line
        carries the keep-out info.
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
            lower_left = (obs[0] - side / 2.0, obs[1] - side / 2.0)
            rect = mpl.patches.Rectangle(
                lower_left, side, side,
                facecolor=_PALETTE["obstacle"], edgecolor="black",
                linewidth=1.0, alpha=0.95, zorder=1,
            )
            self.axes[0].add_patch(rect)
            self._obstacle_patches.append(rect)

    def update(self, _frame):
        """Render ONE frame from the latest buffered data.

        Rollout/cost-dependent blocks (the planner fan + cost coloring + auto-fit
        view) only render when those topics are populated; controllers that don't
        plan (e.g. gail_hl_control.py) can publish just /mppi/current_state +
        /mppi/obstacle_occupancy + /mppi/obstacle_pose and still get robot icon,
        map overlay, obstacle squares, and the trajectory trace.
        """
        with self.data_lock:
            # do nothign if we don't have a pose to draw
            if self.current_state is None:
                return []

            x0 = self.current_state          # [env, state_dim]
            rollouts = self.rollouts         # [env, rollout, horizon, state_dim] or None
            have_rollouts = rollouts is not None and self.costs is not None

            for i in range(self.vis_n_envs):
                # Skip envs the publisher didn't include this frame.
                if i >= x0.shape[0]:
                    continue
                if have_rollouts and i >= rollouts.shape[0]:
                    continue
                # Pull pose components out of the state vector (guarded by width).
                curr_x = float(x0[i, 0])
                curr_y = float(x0[i, 1])
                curr_yaw = float(x0[i, 5]) if x0.shape[1] > 5 else 0.0
                curr_vx = float(x0[i, 6]) if x0.shape[1] > 6 else 0.0
                curr_vy = float(x0[i, 7]) if x0.shape[1] > 7 else 0.0

                # --- Robot icon ---
                yaw = float(np.nan_to_num(curr_yaw, nan=0.0))  # NaN yaw -> 0 (avoid breaking geometry)
                self.env_pos_scatters[i].set_xy(
                    self._robot_polygon_vertices(curr_x, curr_y, yaw)
                )

                # --- Optimal (chosen) trajectory: dense yellow polyline ---
                if (
                    self.show_rollouts_scatter
                    and self.optimal_trajectory is not None
                    and i < self.optimal_trajectory.shape[0]
                ):
                    opt = self.optimal_trajectory[i]        # [horizon, state_dim]
                    self.env_optimal_lines[i].set_data(opt[:, 0], opt[:, 1])

                # --- Robot-centric occupancy grid, rotated into the world ---
                # Paper mode swaps this out for static world-aligned obstacle
                # halos in _draw_obstacles, so skip the per-frame BEV update.
                if (self.show_occupancy_map and not self.paper_mode
                        and self.obstacle_occupancy is not None):
                    occ = self.obstacle_occupancy
                    # Normalize to a 2-D [H, W] grid for this env.
                    if occ.ndim == 4:
                        occ = occ[i, ..., 0]
                    elif occ.ndim == 3 and occ.shape[0] == self.vis_n_envs:
                        occ = occ[i]
                    theta = yaw
                    c, s = np.cos(theta), np.sin(theta)

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

                if self.show_rollouts_scatter and have_rollouts:
                    env_rollouts = rollouts[i]   # [rollout, horizon, state_dim]
                    horizon = env_rollouts.shape[1]
                    n_rollouts_avail = env_rollouts.shape[0]
                    n_top = min(self.vis_n_rollouts, n_rollouts_avail)
                    # Cap rejected count so top and bot indices never overlap.
                    n_bot = min(self.vis_n_rejected, max(0, n_rollouts_avail - n_top))

                    env_costs = self.costs[i] if i < self.costs.shape[0] else None
                    if env_costs is not None and env_costs.shape[0] == n_rollouts_avail:
                        rollout_scores = np.sum(env_costs, axis=-1)           # [rollout]
                        rollout_scores = np.where(
                            np.isnan(rollout_scores), np.inf, rollout_scores
                        )
                        sorted_idx = np.argsort(rollout_scores)
                        topk_idx = sorted_idx[:n_top]
                        # Worst N (highest cost) rollouts -- the ones the
                        # planner explicitly rejected. Drawn alongside top-K
                        # so the figure shows "what was considered + thrown
                        # out", not just "what was chosen between".
                        botk_idx = sorted_idx[-n_bot:] if n_bot > 0 else np.array([], dtype=int)
                    else:
                        # Fallback when costs shape doesn't line up with
                        # rollouts -- render the first n_top so something draws.
                        topk_idx = np.arange(n_top)
                        botk_idx = np.array([], dtype=int)
                    selected_idx = np.concatenate([topk_idx, botk_idx])

                    # Flatten the selected rollouts' (x, y) over the horizon, in selection order.
                    all_x = np.concatenate([env_rollouts[k][:, 0] for k in selected_idx])
                    all_y = np.concatenate([env_rollouts[k][:, 1] for k in selected_idx])

                    # Per-point cost coloring: pull each timestep's cost from
                    # /mppi/rollout_costs (shape [env, rollout, horizon]) and
                    # map through the fixed _cost_norm. Low cost -> navy, high
                    # -> red -- so top-K naturally sit at the cool end and
                    # rejected rollouts at the warm end. The user's reference
                    # figure had this exact "wander toward red terminals" feel.
                    if env_costs is not None and env_costs.shape[0] == n_rollouts_avail:
                        selected_costs = np.stack([env_costs[k] for k in selected_idx])
                        flat_costs = selected_costs.ravel()
                    else:
                        # Costs unavailable: fall back to a flat-blue fan so we
                        # still draw something instead of erroring.
                        flat_costs = np.zeros(all_x.shape[0], dtype=np.float32)

                    # Fixed sample-cost norm (set once in setup, tunable via
                    # ~sample_cost_min/max). No per-frame norm updates -> the
                    # cbar tick labels stay stable.
                    all_colors = _SAMPLE_CMAP(self._cost_norm(flat_costs))
                    # Top-K at full alpha; rejected rollouts at lower alpha so
                    # they read as "considered but discarded" without fighting
                    # the chosen fan for attention.
                    top_count = len(topk_idx) * horizon
                    all_colors[:top_count, 3] = 0.95
                    all_colors[top_count:, 3] = 0.45

                    # Collision recolor (vermilion) takes precedence: even if
                    # cost normalization happens to map collisions to mid-range
                    # for some frames, this guarantees they pop visually.
                    occ_idx = self.rollout_occupancy_idx
                    if env_rollouts.shape[-1] > occ_idx:
                        all_occ = np.concatenate(
                            [env_rollouts[k][:, occ_idx] for k in selected_idx]
                        )
                        occupied_mask = all_occ < 0  # <0 == in collision
                        if np.any(occupied_mask):
                            collision_rgb = np.array(mpl.colors.to_rgb(_PALETTE["collision"]))
                            all_colors[occupied_mask, :3] = collision_rgb
                            all_colors[occupied_mask, 3] = 1.0
                    self.env_rollout_scatters[i].set_offsets(np.c_[all_x, all_y])
                    self.env_rollout_scatters[i].set_facecolor(all_colors)
                    self.env_rollout_scatters[i].set_edgecolors("black")

                # --- Robot's actual trajectory so far (colored by yaw angle) ---
                if self.show_trajectory_trace:
                    self.trajectory_traces[i].append(x0[i].copy())
                    trace_states = np.array(self.trajectory_traces[i])  # [steps, state_dim]
                    trace_x, trace_y = trace_states[:, 0], trace_states[:, 1]
                    # Per-segment color = yaw angle at the segment's START
                    # point. state[5] is yaw (rad), wrapped to [0, 2*pi] by
                    # mpail_hl_control.obtain_state. Norm is fixed to that
                    # same range in setup, so the cbar is stable.
                    if trace_states.shape[0] >= 2 and trace_states.shape[1] > _YAW_IDX:
                        yaw_vals = trace_states[:, _YAW_IDX]  # rad
                        seg_yaw = yaw_vals[:-1]
                        points = np.column_stack([trace_x, trace_y])[:, None, :]
                        segments = np.concatenate([points[:-1], points[1:]], axis=1)
                        self.trajectory_lcs[i].set_segments(segments)
                        self.trajectory_lcs[i].set_array(seg_yaw)
                    if len(self.trajectory_traces[i]) > 0:
                        start_pt = np.asarray(self.trajectory_traces[i][0])[:2]
                        self.trajectory_start_scatters[i].set_offsets(np.array([start_pt]))

            # Obstacles + timestep readout (shared across envs). Timestep text
            # is suppressed in paper mode.
            self._draw_obstacles()
            if self.time_text is not None:
                self.time_text.set_text(f"Timestep: {self.frame_timestep}")

            # DBF barrier: zero-level contour only. The full signed h(s)
            # heatmap was visually too busy and competed with the rollout
            # fan for attention; the boundary line alone communicates "this
            # side safe, this side unsafe" and the sample-cost cbar now
            # occupies the slot the heatmap legend used to. Purge previous
            # contour each frame (mpl <3.8 needs .collections removed
            # explicitly before .remove() on the ContourSet).
            for artist in self._barrier_artists:
                for coll in list(getattr(artist, "collections", []) or []):
                    try:
                        coll.remove()
                    except Exception:
                        pass
                try:
                    artist.remove()
                except Exception:
                    pass
            self._barrier_artists = []
            if self.barrier_Z is not None and self.barrier_Z.size > 0:
                h, w = self.barrier_Z.shape
                # Per-tick robot-centered extent wins. Static ~xlim/~ylim is a
                # backwards-compat fallback for runs against an older controller
                # that doesn't publish /mppi/barrier_grid_extent.
                if self.barrier_extent is not None:
                    bx0, bx1, by0, by1 = self.barrier_extent
                elif self.xlim is not None and self.ylim is not None:
                    bx0, bx1 = self.xlim
                    by0, by1 = self.ylim
                else:
                    bx0, bx1 = self.axes[0].get_xlim()
                    by0, by1 = self.axes[0].get_ylim()
                gx = np.linspace(bx0, bx1, w)
                gy = np.linspace(by0, by1, h)
                X, Y = np.meshgrid(gx, gy)
                self._barrier_artists.append(self.axes[0].contour(
                    X, Y, self.barrier_Z, levels=[0],
                    colors=_PALETTE["barrier"], linewidths=2.5, zorder=4,
                ))
                # One-shot sidecar PNG of the full h(s) field (out-of-band
                # from the mp4). Self-gates after the first save so calling
                # it every frame is harmless.
                self._save_barrier_heatmap_png()

            # Crop to robot-centered BEV footprint when requested. Takes
            # precedence over auto-fit -- they target opposite use cases.
            if self.zoom_to_bev and x0.shape[0] > 0:
                half_bev = self.map_length_px * self.map_res_m_px / 2.0
                cx = float(x0[0, 0])
                cy = float(x0[0, 1])
                self.axes[0].set_xlim(cx - half_bev, cx + half_bev)
                self.axes[0].set_ylim(cy - half_bev, cy + half_bev)
            # Auto-fit the view to the rollouts when no fixed limits were given.
            # Only ever widens (min/max with current limits), never shrinks.
            # (No-op for non-planner controllers — they don't publish rollouts.)
            elif (self.xlim is None or self.ylim is None) and have_rollouts:
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
