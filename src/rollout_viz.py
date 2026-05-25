import torch
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from io import BytesIO
import imageio
from typing import TYPE_CHECKING

import math

from numpy.ma import masked_array

from scipy.ndimage import rotate

if TYPE_CHECKING:
    from .vis_cfg import RolloutVisConfig
    
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
import matplotlib.patheffects as pe
import torchvision.transforms.functional as TF
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

class RolloutsVisualization:
    """
    Visualizes a hierarchical animation where new rollouts are added at each simulation timestep.

    Parameters:
        sim_timesteps (int): Total number of simulation timesteps.
        data_generator (callable): Function that provides rollout data at each simulation timestep.
                                   Expected output: [num_envs, num_rollouts, num_timestep, state_dim].
        env_ids (list): List of environment indices to visualize.
        rollout_ids (list): List of rollout indices to visualize.
        performance_metrics_fn (callable): Function to compute performance metrics.
                                           Input: [num_envs, num_rollouts].
                                           Output: [num_envs, num_rollouts].
        save_path (str): Path to save the generated video.
        trail_length (int): Length of the trajectory trail to display.
    """
    def __init__(self,
        cfg: 'RolloutVisConfig',
        map_res_m_px: float=0.1,
        map_length_px: int=256,
    ):
        self.cfg = cfg
        self.vis_n_envs = cfg.vis_n_envs
        self.vis_n_rollouts = cfg.vis_n_rollouts
        self.xlim = cfg.xlim
        self.ylim = cfg.ylim
        self.map_res_m_px = map_res_m_px
        self.map_length_px = map_length_px
        self.show_velocity = cfg.show_velocity
        self.show_elevation = cfg.show_elevation
        self.cost_range = cfg.cost_range
        # Backwards compatible quick toggle; prefer cfg.show_rollouts_scatter (default True)
        self.vis_scatter = True
        # Make rollout scatter visibility configurable via cfg.show_rollouts_scatter (default True)
        self.show_rollouts_scatter = True
        # self.show_rollouts_scatter = getattr(cfg, "show_rollouts_scatter", True)
        # self.show_trajectory_trace = cfg.show_trajectory_trace 
        self.show_trajectory_trace = True
        # self.show_occupancy_map = True if cfg.show_occupancy_map is None else cfg.show_occupancy_map
        self.show_occupancy_map = True
        self.show_train_env_zone = False
        # Debug rollout occupancy against BEV image sampling.
        # Useful for diagnosing "turns red only when on top of obstacle" behavior.
        self.show_rollout_occ_debug = getattr(cfg, "show_rollout_occ_debug", True)
        # control whether trajectory trace colors encode velocity (True) or use a constant color (False)
        self.color_trace_by_velocity = getattr(cfg, "color_trace_by_velocity", False)
        self.starting_points_scatter = [[] for _ in range(self.vis_n_envs)]  # Initialize as a list of empty lists
        self.trajectory_headings = [[] for _ in range(self.vis_n_envs)]  # Store trajectory headings for each environment
        self.trajectory_velocity = [[] for _ in range(self.vis_n_envs)] # Store magnitude of trajectory velocities for each environment
        self.trajectory_traces = [[] for _ in range(self.vis_n_envs)]  # Now stores full states
        self.x = [[] for _ in range(self.vis_n_envs)]  # Store x coordinates of trajectory for each environment
        self.y = [[] for _ in range(self.vis_n_envs)]  # Store y coordinates of trajectory for each environment
        self.u = [[] for _ in range(self.vis_n_envs)]  # Store x components of heading vectors for each environment
        self.v = [[] for _ in range(self.vis_n_envs)]  # Store y components of heading vectors for each environment
        self.red_blue = mpl.cm.get_cmap("plasma")  # or "coolwarm", "seismic", "viridis"
        self.heading_norm = mpl.colors.Normalize(vmin=-np.pi, vmax=np.pi)
        self.obstacle_positions = None
        self.obstacle_rotations = None
        self.demonstrations = None  # to be set later
        self.obs_side = None  # to be set later
        self.robot_marker_size = 0.35
        # DBF-only handles; remain None for non-DBF cost stacks.
        self._h = None
        self._q = None
        self.value_func = None
        self.reset()

        # In __init__, add storage for segment colors:
        self.trajectory_segment_colors = [[] for _ in range(self.vis_n_envs)]  # Store colors for each segment

    def update(
        self,
        x0: np.ndarray,
        rollouts: np.ndarray,
        obstacle_occupancy: np.ndarray,
        costs_fn: callable,
        rollout_costs: np.ndarray,
        elevation_map: np.ndarray,
        optimal_control: np.ndarray,
        frame_timestep: int=None,
        obstacle_pose: np.ndarray=None,
    ):
        '''
        x0: [num_envs, state_dim]
        rollouts: [num_envs, num_rollouts, num_timestep, state_dim]
        rollout_costs: [num_envs, num_rollouts, num_timesteps]
        metrics: [num_envs, num_rollouts]
        vis_rollout_ids: list of rollout indices to visualize
        frame_timestep: current simulation timestep
        '''
        self.obstacle_positions = obstacle_pose
        self.device = costs_fn.device
        # Initialize DBF-related members only when discriminator/value interfaces exist.
        self._h = None
        self._q = None
        self.value_func = None
        ss_cost = getattr(costs_fn, "ss_cost", None)
        ts_cost = getattr(costs_fn, "ts_cost", None)
        discriminator = ss_cost.get_discriminator() if (ss_cost is not None and hasattr(ss_cost, "get_discriminator")) else None
        if discriminator is not None:
            if hasattr(discriminator, "_specification_function"):
                self._h = discriminator._specification_function()
            if hasattr(discriminator, "_dynamic_constraint_lagrangian"):
                self._q = discriminator._dynamic_constraint_lagrangian
        if ts_cost is not None and hasattr(ts_cost, "get_value"):
            self.value_func = ts_cost.get_value()
        # Normalize performance metrics for color coding: [num_envs, num_rollouts]
        # normed_costs = (rollout_costs - rollout_costs.min()) \
        #                                 / (rollout_costs.max() - rollout_costs.min() + 1e-5)

        # normalize elevation map
        if self.show_elevation:
            elevation_map = elevation_map[..., 0]
            elevation_map = (elevation_map - elevation_map.min()) / (elevation_map.max() - elevation_map.min() + 1e-5)

        # Update visuals for each environment
        for i in range(self.vis_n_envs):
            env_rollouts = rollouts[i]  # [num_rollouts, num_timestep, state_dim]

            # Update scatter for the last point
            curr_x = x0[i, 0]
            curr_y = x0[i, 1]
            curr_z = x0[i, 2]
            curr_vx = x0[i, 6]
            curr_vy = x0[i, 7]
            curr_yaw = x0[i, 5]
            self.x[i].append(curr_x)
            self.y[i].append(curr_y)
            self.u[i].append(curr_vx)
            self.v[i].append(curr_vy)

            velocity_magnitude = np.sqrt(curr_vx**2 + curr_vy**2)
            # plot optimal control rollout
            if self.show_rollouts_scatter:
                optimal_control_rollout = optimal_control[i]  # [num_timesteps, control_dim]
                self.env_control_scatters[i].set_offsets(optimal_control_rollout[:, :2])  # Use scatter for trajectory

            #plot footprint
            # flx = current_footprint[i, 0, 0, 0]
            # fly = current_footprint[i, 0, 0, 1]
            # frx = current_footprint[i, 0, 1, 0]
            # fry = current_footprint[i, 0, 1, 1]
            # brx = current_footprint[i, 0, 2, 0]
            # bry = current_footprint[i, 0, 2, 1]
            # blx = current_footprint[i, 0, 3, 0]
            # bly = current_footprint[i, 0, 3, 1]
            # self.env_footprint_plots[i].set_data([flx, frx, brx, blx, flx], [fly, fry, bry, bly, fly])

            #plot velocity
            if self.show_velocity:
                velocity = np.sqrt(curr_vx**2 + curr_vy**2)
                self.velocities[i] = np.append(self.velocities[i], velocity)
                self.velocities_lines[i].set_data(range(len(self.velocities[i])), self.velocities[i])
                self.axes[1].relim()
                self.axes[1].autoscale_view()

            # Use the current state yaw for all map overlays to keep them aligned.
            yaw = np.nan_to_num(curr_yaw, nan=0.0)
            rotate_angle = -yaw * 180 / math.pi

            robot_triangle = self._robot_triangle_vertices(curr_x, curr_y, curr_yaw, self.robot_marker_size)
            self.env_pos_scatters[i].set_xy(robot_triangle)

            if self.show_elevation:
                env_elevation = elevation_map[i] # [256, 256], only visualizes 1 car
                env_elevation += 1 # shift the elevation map to be positive

                env_elevation = np.nan_to_num(env_elevation, nan=0)
                env_elevation = rotate(env_elevation, angle=rotate_angle, order=3, reshape=True, mode="constant")
                # env_elevation = masked_array(env_elevation, env_elevation < 100 if curr_z < -0.8 else env_elevation == 0) #mask out everything if car is not on track to avoid visual clutter
                self.elevation_images[i].set_data(env_elevation)
                new_min = self.min_elevation_map_x * (abs(math.cos(yaw)) + abs(math.sin(yaw)))
                new_max = new_min * -1

                self.elevation_images[i].set_extent([new_min + curr_x, new_max + curr_x, new_min + curr_y, new_max + curr_y])
                # self.elevation_images[i].autoscale(enable=False)

            if self.show_occupancy_map:
                # Match obstacle_ground_truth convention used in this codepath.
                # BEV occupancy is robot-centric; rotate back into world with +yaw.
                theta = float(yaw)
                c, s = np.cos(theta), np.sin(theta)

                map_size = (self.map_length_px - 1) * self.map_res_m_px
                half = map_size / 2
                self.occupancy_images[i].set_data(abs(obstacle_occupancy[i, ...]))
                self.occupancy_images[i].set_cmap('gray_r')  # Use gray_r for inverted grayscale
                min_x = curr_x - map_size / 2
                max_x = curr_x + map_size / 2
                min_y = curr_y - map_size / 2
                max_y = curr_y + map_size / 2
                self.occupancy_images[i].set_extent([min_x, max_x, min_y, max_y])
                img_transform = mpl.transforms.Affine2D().rotate_around(curr_x, curr_y, theta) + self.axes[0].transData
                self.occupancy_images[i].set_transform(img_transform)

                R = np.array([[c, -s], [s, c]])
                
                new_min = -half * (abs(c) + abs(s))
                new_max = new_min * -1
                # corners = np.array([
                #     [new_min, new_min],
                #     [new_max, new_min],
                #     [new_max, new_max],
                #     [new_min, new_max]
                # ])
                corners = np.array([
                    [-half, -half],
                    [ half, -half],
                    [ half,  half],
                    [-half,  half]
                ])
                rotated_corners = (R @ corners.T).T + np.array([curr_x, curr_y])
                self.occupancy_rects[i].set_xy(rotated_corners)

            # Update trajectory lines
            # Update trajectory scatter (if configured)
            if self.show_rollouts_scatter and (i in self.env_rollout_scatters) and (self.env_rollout_scatters[i] is not None):
                env_scatters: list = self.env_rollout_scatters[i]
                # Update trajectory scatter
                all_x_vals = np.concatenate([env_rollouts[j][:, 0] for j in range(self.vis_n_rollouts)])
                all_y_vals = np.concatenate([env_rollouts[j][:, 1] for j in range(self.vis_n_rollouts)])

                # Default rollout color is blue; occupied points in the propagated occupancy map are red.
                num_points = all_x_vals.shape[0]
                all_colors = np.tile(np.array([[0.1, 0.35, 0.9, 0.95]]), (num_points, 1))

                # Read occupancy directly from propagated rollout features.
                # The occupancy feature index is set by MPPI (defaults to x_dim).
                occ_idx = getattr(self, "rollout_occupancy_idx", 12)
                if env_rollouts.shape[-1] > occ_idx:
                    all_occ_vals = np.concatenate([env_rollouts[j][:, occ_idx] for j in range(self.vis_n_rollouts)])
                    occupied_mask = all_occ_vals < 0
                    if np.any(occupied_mask):
                        all_colors[occupied_mask, :3] = np.array([1.0, 0.0, 0.0])
                        all_colors[occupied_mask, 3] = 1.0

                # Optional debug: classify out-of-map points and compare rollout occupancy
                # against direct occupancy sampled from the BEV image for the same frame.
                if self.show_rollout_occ_debug:
                    map_size = (self.map_length_px - 1) * self.map_res_m_px
                    half = map_size / 2.0
                    c, s = np.cos(yaw), np.sin(yaw)

                    dx = all_x_vals - curr_x
                    dy = all_y_vals - curr_y

                    # World -> robot/body frame (row-vector convention used by BEVMap).
                    x_body = dx * c + dy * s
                    y_body = -dx * s + dy * c

                    in_bounds = (
                        (x_body >= -half) & (x_body <= half) &
                        (y_body >= -half) & (y_body <= half)
                    )
                    oob_mask = ~in_bounds
                    if np.any(oob_mask):
                        # Amber for points outside current local BEV support.
                        all_colors[oob_mask, :3] = np.array([1.0, 0.75, 0.0])
                        all_colors[oob_mask, 3] = 1.0

                    # Direct occupancy from BEV image at rollout points (where in-bounds).
                    direct_occ = np.zeros(all_x_vals.shape[0], dtype=np.float32)
                    if np.any(in_bounds):
                        # Convert robot-frame meters to map pixel indices.
                        i_idx = np.floor(y_body / self.map_res_m_px + self.map_length_px // 2).astype(np.int64)
                        j_idx = np.floor(x_body / self.map_res_m_px + self.map_length_px // 2).astype(np.int64)
                        i_idx = np.clip(i_idx, 0, self.map_length_px - 1)
                        j_idx = np.clip(j_idx, 0, self.map_length_px - 1)
                        occ_img = obstacle_occupancy[i, ...]
                        direct_occ[in_bounds] = occ_img[i_idx[in_bounds], j_idx[in_bounds]]

                    # Highlight disagreements between propagated occupancy and direct BEV sampling.
                    mismatch_pct = 0.0
                    occ_feat_pct = 0.0
                    occ_direct_pct = 0.0
                    oob_pct = float(oob_mask.mean()) if oob_mask.size > 0 else 0.0
                    if env_rollouts.shape[-1] > occ_idx:
                        occ_feat_mask = all_occ_vals < 0
                        occ_direct_mask = direct_occ < 0
                        mismatch = in_bounds & (occ_feat_mask != occ_direct_mask)
                        if np.any(mismatch):
                            # Magenta for disagreement points.
                            all_colors[mismatch, :3] = np.array([1.0, 0.0, 1.0])
                            all_colors[mismatch, 3] = 1.0
                        mismatch_pct = float(mismatch.mean()) if mismatch.size > 0 else 0.0
                        occ_feat_pct = float(occ_feat_mask.mean()) if occ_feat_mask.size > 0 else 0.0
                        occ_direct_pct = float(occ_direct_mask.mean()) if occ_direct_mask.size > 0 else 0.0

                    if i == 0 and hasattr(self, "rollout_occ_debug_text"):
                        self.rollout_occ_debug_text.set_text(
                            "rollout occ debug | "
                            f"feat_occ={occ_feat_pct*100:.1f}%  "
                            f"direct_occ={occ_direct_pct*100:.1f}%  "
                            f"oob={oob_pct*100:.1f}%  "
                            f"mismatch={mismatch_pct*100:.1f}%"
                        )

                env_scatters.set_offsets(np.c_[all_x_vals, all_y_vals])
                env_scatters.set_facecolor(all_colors)
                env_scatters.set_edgecolors('black')
            # env_scatters.set_sizes([2] * len(all_x_vals))  # Set marker size

            # Add current position to trajectory trace
            if self.show_trajectory_trace:
                # Store full state x0[i] 
                self.trajectory_traces[i].append(x0[i].copy())  # shape: [state_dim]
                self.trajectory_velocity[i].append(velocity_magnitude)
                
                # Extract x, y for visualization
                trace_states = np.array(self.trajectory_traces[i])  # shape: [num_steps, state_dim]
                trace_x, trace_y = trace_states[:, 0], trace_states[:, 1]
                
                # Use yaw rate for coloring (index 11 is w_z, convert to deg/s)
                if len(self.trajectory_traces[i]) > 1:
                    all_yaw_rates = trace_states[:, 11] * 180.0 / np.pi  # Convert rad/s to deg/s

                    # Update colormap normalization based on actual yaw rate range
                    vmin = all_yaw_rates.min()
                    vmax = all_yaw_rates.max()
                    if vmax > vmin:
                        # Update the ScalarMappable normalization dynamically
                        self.sm_traj.set_clim(vmin, vmax)
                        # Normalize values for coloring
                        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                    else:
                        # If all values are the same, use a small range around that value
                        norm = mpl.colors.Normalize(vmin=vmin - 0.1, vmax=vmax + 0.1)
                    
                    # Create LineCollection with segment colors
                    points = np.array([trace_x, trace_y]).T.reshape(-1, 1, 2)
                    if points.shape[0] > 1:
                        segments = np.concatenate([points[:-1], points[1:]], axis=1)
                        
                        # Get colors for each segment using the normalized yaw rates
                        segment_colors = self.red_blue(norm(all_yaw_rates[:-1]))
                        
                        # Create LineCollection with thick lines
                        lc = LineCollection(segments, colors=segment_colors, linewidth=0.5, 
                                          capstyle='round', joinstyle='round')
                        
                        if len(self.env_rollout_lines[i]) == 0:
                            self.env_rollout_lines[i].append(self.axes[0].add_collection(lc))
                        else:
                            self.env_rollout_lines[i][0].set_segments(segments)
                            self.env_rollout_lines[i][0].set_colors(segment_colors)
                            self.env_rollout_lines[i][0].set_linewidth(8)
                else:
                    if len(self.env_rollout_lines[i]) > 0:
                        try:
                            self.env_rollout_lines[i][0].remove()
                        except Exception:
                            pass
                        self.env_rollout_lines[i] = []
            # Update the start-point scatter to mark the first point of the trace (if available)
            try:
                if len(self.trajectory_traces[i]) > 0:
                    start_pt = self.trajectory_traces[i][0]
                    # set_offsets expects an array-like of shape (N, 2)
                    self.trajectory_start_scatters[i].set_offsets(np.array([start_pt]))
                else:
                    # clear offsets
                    self.trajectory_start_scatters[i].set_offsets(np.empty((0, 2)))
            except Exception:
                # If scatter not initialized or other error, ignore
                pass

        # Update the time text
        if frame_timestep is not None:
            self.timestep = frame_timestep
            self.time_text.set_text(f"Simulation Timestep: {self.timestep}")

        self.add_obstacles()
        if self._h is not None:
            try:
                self.zero_contour()   
            except Exception:
                pass
        # Update boundaries
        if self.xlim is None or self.ylim is None:
            current_xlim = self.axes[0].get_xlim()
            current_ylim = self.axes[0].get_ylim()
            new_xlim = (min(current_xlim[0], rollouts[..., 0].min() - 1), max(current_xlim[1], rollouts[..., 0].max() + 1))
            new_ylim = (min(current_ylim[0], rollouts[..., 1].min() - 1), max(current_ylim[1], rollouts[..., 1].max() + 1))
            self.axes[0].set_xlim(new_xlim)
            self.axes[0].set_ylim(new_ylim)

    def plot_goal(self, center=(0, 0), r_outer=1.0, r_inner=0.4, num_points=5):
        """Return the vertices of a star polygon."""
        cx, cy = center
        points = []
        angle = np.pi / num_points  # half angle between star points
        for i in range(2 * num_points):
            r = r_outer if i % 2 == 0 else r_inner
            theta = i * angle
            x = cx + r * np.cos(theta)
            y = cy + r * np.sin(theta)
            points.append((x, y))
        return points

    def _robot_triangle_vertices(self, center_x, center_y, yaw, size):
        """Return triangle vertices for a robot marker that points along yaw."""
        heading = np.array([np.cos(yaw), np.sin(yaw)])
        left = np.array([-heading[1], heading[0]])

        tip = np.array([center_x, center_y]) + heading * size
        rear_center = np.array([center_x, center_y]) - heading * (size * 0.7)
        rear_left = rear_center + left * (size * 0.45)
        rear_right = rear_center - left * (size * 0.45)

        return np.array([tip, rear_left, rear_right])

    def add_obstacles(self):
        """
            Adds obstacles to the plot as designed in the environment.
            Ground truth obstacles locations:
            [5, 7], [5, 2], [7, 5], [1.5, 2], [8, 9], [2, 5]
        """
        if self.obstacle_positions is not None:
            obstacles = self.obstacle_positions
            obstacle_rotation = None
        else:
            obstacles = np.array([[5, 7], [5, 2], [7, 5], [1.5, 2], [8, 9], [2, 5]])
        
        goal = np.array([10, 10])
        for obs in obstacles:
            # draw a square centered at (x, y) with side length 0.8 (previous circle radius was 0.4)
            side = 1.0 if self.obs_side is None else self.obs_side
            lower_left = (obs[0] - side / 2.0, obs[1] - side / 2.0)
            # create rectangle centered at obs
            obstacle = mpl.patches.Rectangle(lower_left, side, side, color='gray', alpha=0.5)

            # If rotations are available, try to compute yaw and rotate rectangle about its center
            try:
                rot = None
                if hasattr(self, 'obstacle_rotations') and self.obstacle_rotations is not None:
                    rots = self.obstacle_rotations
                    # convert torch tensor to numpy if needed
                    if hasattr(rots, 'cpu'):
                        rots = rots.cpu().numpy()
                    rots = np.asarray(rots)
                    # If shapes align, pick corresponding rotation for this obstacle index
                    if rots.ndim == 1:
                        rot = rots
                    elif rots.shape[0] == obstacles.shape[0]:
                        # Need index of current obs in obstacles
                        # We'll search for matching coordinates (best-effort)
                        try:
                            # handle float comparisons with tolerance
                            diffs = np.linalg.norm(rots * 0 + 0, axis=-1) if rots.ndim > 1 and rots.shape[1] > 1 else np.zeros(len(rots))
                        except Exception:
                            diffs = None
                        # attempt to find index by matching positions if possible
                        idx = None
                        try:
                            # If obstacles came from a numpy array in same order, we can enumerate
                            # Fallback: use loop index from Python enumerate of obstacles earlier — but we don't have it here.
                            # So we compute index by nearest neighbor in position space if possible (best effort)
                            obs_pos_arr = np.asarray(obstacles)
                            if obs_pos_arr.ndim == 2:
                                dists = np.linalg.norm(obs_pos_arr - np.array(obs), axis=1)
                                idx = int(np.argmin(dists))
                        except Exception:
                            idx = None

                        if idx is not None and idx < rots.shape[0]:
                            rot = rots[idx]
                        else:
                            # fallback: if rots is 1D or length matches, take first
                            rot = rots[0]

                yaw = 0.0
                if rot is not None:
                    rot = np.asarray(rot)
                    # if rotation is a scalar or single-element array
                    if rot.ndim == 0 or rot.size == 1:
                        yaw = float(rot)
                    elif rot.size == 3:
                        # assume euler (rx, ry, yaw)
                        yaw = float(rot[-1])
                    elif rot.size == 4:
                        # assume quaternion; try both common conventions
                        q = rot.ravel()
                        # assume (w, x, y, z)
                        w, x, y, z = q[0], q[1], q[2], q[3]
                        yaw_wxyz = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                        # assume (x, y, z, w)
                        x2, y2, z2, w2 = q[0], q[1], q[2], q[3]
                        yaw_xyzw = math.atan2(2.0 * (w2 * z2 + x2 * y2), 1.0 - 2.0 * (y2 * y2 + z2 * z2))
                        # choose one — they are equivalent up to sign depending on convention; pick yaw_wxyz
                        yaw = float(yaw_wxyz)

                # rotate about center
                cx, cy = obs[0], obs[1]
                t = mpl.transforms.Affine2D().rotate_around(cx, cy, yaw) + self.axes[0].transData
                obstacle.set_transform(t)
            except Exception:
                # If rotation handling fails, fall back to unrotated rectangle
                pass

            self.axes[0].add_artist(obstacle)
        goal_point = mpl.patches.Polygon(self.plot_goal((10, 10), r_outer=0.3, r_inner=0.1, num_points=5),
               closed=True, facecolor="gold", edgecolor="black")
        self.axes[0].add_artist(goal_point)

        # Update plot boundaries to include all obstacles
        obstacle_x = obstacles[:, 0]
        obstacle_y = obstacles[:, 1]
        current_xlim = self.axes[0].get_xlim()
        current_ylim = self.axes[0].get_ylim()
        new_xlim = (min(current_xlim[0], obstacle_x.min() - 1), max(current_xlim[1], obstacle_x.max() + 1, goal[0] + 1))
        new_ylim = (min(current_ylim[0], obstacle_y.min() - 1), max(current_ylim[1], obstacle_y.max() + 1, goal[1] + 1))
        self.axes[0].set_xlim(new_xlim)
        self.axes[0].set_ylim(new_ylim)

    def zero_contour(self):
        """
        Adds a contour line at zero for the specification function.
        """
        # Create a mesh grid based on the visualization limits
        x = torch.linspace(self.xlim[0], self.xlim[1], 100)
        y = torch.linspace(self.ylim[0], self.ylim[1], 100)
        X, Y = torch.meshgrid(x, y, indexing="xy")
        
        obstacle_positions = torch.tensor(self.obstacle_positions, dtype=torch.float32, device=self.device)
        
        # Flatten the grid for evaluation
        grid = torch.stack([X.ravel(), Y.ravel()], axis=-1).to(self.device)  # shape: [num_points, 2]
        grid_tensor = grid.reshape(x.shape[0], y.shape[0], 2)                       # shape: [num_points_x, num_points_y, 2]

        # Calculate the distace of each point in the grid to each obstacle and create a fixed occupancy grid
        grid_expanded = grid_tensor.unsqueeze(2)                                        # shape: [num_points_x, num_points_y, 1, 2]
        obs_expand = obstacle_positions.unsqueeze(0).unsqueeze(0)                       # shape: [1, 1, num_obstacles, 2]
        dists = torch.linalg.norm(grid_expanded - obs_expand, dim=-1)                   # shape: [num_points_x, num_points_y, num_obstacles]
        obstacle_size =  1.0
        half_size = obstacle_size / 2.0
        
        # Calculate if point is within square bounds: |dx| <= size/2 AND |dy| <= size/2
        dx = torch.abs(grid_expanded[..., 0] - obs_expand[..., 0])  # shape: [num_points_x, num_points_y, num_obstacles]
        dy = torch.abs(grid_expanded[..., 1] - obs_expand[..., 1])
        in_square = (dx <= half_size) & (dy <= half_size)

        # Compute occupancy for next states directly from dynamics-propagated XY points.
        occupancy_grid = -1 * in_square.any(dim=-1).int().unsqueeze(-1).reshape(-1, 1) # shape: [num_points, 1]
        grid = torch.cat((grid, occupancy_grid), dim=-1)
        
        # Evaluate the specification function on the grid
        with torch.inference_mode():
            Z = self._h(grid).cpu().numpy()

        # Reshape the output to match the grid
        Z = Z.reshape(X.shape)

        # Plot the zero contour
        cbf_contour = self.axes[0].contour(X, Y, Z, levels=[0], colors='black', linewidths=2)
        cbf_contour.set_label('barrier')

    def save_frame(self, output_dir, step: int=None):
        """
        Saves the current frame to disk.
        """
        fname = f"rollouts_vis.png"
        if step is not None:
            fname = f"step-{step}_{fname}"

        path = os.path.join(output_dir, fname)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Add legend only for the saved still image (keeps live/video rendering lighter).
        self._refresh_main_legend()
        self.fig.savefig(path)
        self._remove_main_legend()
        return path

    def _refresh_main_legend(self):
        """Create/update the main trajectory legend outside the plotting area."""
        self._remove_main_legend()
        self._main_legend = self.axes[0].legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )

    def _remove_main_legend(self):
        """Remove the main legend if it exists."""
        legend = getattr(self, "_main_legend", None)
        if legend is not None:
            try:
                legend.remove()
            except Exception:
                pass
            self._main_legend = None

    def reset(self):
        ''' Set up plot features '''

        # Create single plot or 1x2 layout if velocity is shown
        if self.show_velocity:
            self.fig, self.axes = plt.subplots(1, 2, figsize=(16, 8))
        else:
            self.fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            self.axes = [ax]  # Keep axes as a list for consistency

        self.time_text = self.axes[0].text(0.05, 0.95, '', transform=self.axes[0].transAxes,
                                      fontsize=12, verticalalignment='top')
        self.rollout_occ_debug_text = self.axes[0].text(
            0.01, 0.99, '',
            transform=self.axes[0].transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'),
            color='black',
        )

        self.min_elevation_map_x = -self.map_length_px * self.map_res_m_px/2
        self.max_elevation_map_x = -1*self.min_elevation_map_x
        # no need for y min and max since the height scan is a square
        if self.show_elevation:
            self.elevation_images = []
            for i in range(self.vis_n_envs):
                self.elevation_images.append(self.axes[0].imshow(np.zeros((256,256)), cmap='Greys',
                                                                 extent=(self.min_elevation_map_x, self.max_elevation_map_x,
                                                                         self.min_elevation_map_x, self.max_elevation_map_x),
                                                                         alpha=0.5, origin='lower', vmin=0, vmax=1))
        if self.show_occupancy_map:
            self.occupancy_images = []
            self.occupancy_rects = []
            for i in range(self.vis_n_envs):
                # Calculate the extent based on the robot's pose
                map_size = (self.map_length_px - 1) * self.map_res_m_px  # Size of the occupancy map in meters

                # Add the occupancy map image
                self.occupancy_images.append(self.axes[0].imshow(np.zeros((20, 20)), cmap='gray_r', # Use gray_r for inverted grayscale
                                                                 extent=[-map_size/2, map_size/2, -map_size/2, map_size/2],
                                                                 alpha=1.0, origin='lower', vmin=0, vmax=1))
                # Create a square centered at (0,0)
                half = map_size / 2
                corners = np.array([
                    [-half, -half],
                    [ half, -half],
                    [ half,  half],
                    [-half,  half]
                ])
                polygon = mpl.patches.Polygon(corners, closed=True, edgecolor='black', facecolor='none', linewidth=2)
                self.axes[0].add_patch(polygon)
                self.occupancy_rects.append(polygon)

        if self.demonstrations is not None:
            # accept torch.Tensor or numpy array; normalize to np.ndarray of shape [N, T, D]
            expert_trajectories = self.demonstrations
            if hasattr(expert_trajectories, "cpu"):
                expert_trajectories = expert_trajectories.cpu().numpy()
            else:
                expert_trajectories = np.asarray(expert_trajectories)

            if expert_trajectories.ndim == 2:
                expert_trajectories = expert_trajectories[np.newaxis, ...]

            for idx, trajectory in enumerate(expert_trajectories):
                if trajectory.shape[1] < 2:
                    continue
                label = "Expert Trajectory" if idx == 0 else None  # single legend entry
                self.axes[0].plot(
                    trajectory[:, 0],
                    trajectory[:, 1],
                    label=label,
                    color="orange",
                    linestyle="--",
                    alpha=0.9,
                    linewidth=1.5,
                    zorder=2,
                )

        if self.xlim:
            self.axes[0].set_xlim(self.xlim)
        else:
            # Set default limits to include start pose (around 0,0), goal (10,10), and barriers
            self.axes[0].set_xlim(-1, 11)
        if self.ylim:
            self.axes[0].set_ylim(self.ylim)
        else:
            # Set default limits to include start pose (around 0,0), goal (10,10), and barriers
            self.axes[0].set_ylim(-1, 11)

        if self.show_train_env_zone:
            # Add a dashed, unfilled square centered at (5, 4) TODO: make configurable according to train env design
            try:
                dash_center = (5.0, 4.0)
                dash_side = 2.0
                lower_left = (dash_center[0] - dash_side / 2.0, dash_center[1] - dash_side / 2.0)
                dashed_square = mpl.patches.Rectangle(
                    lower_left,
                    dash_side,
                    dash_side,
                    fill=False,
                    edgecolor='black',
                    linewidth=1.5,
                    linestyle='--',
                    alpha=1.0,
                    label='Training Environment Zone'
                )
                self.axes[0].add_artist(dashed_square)
            except Exception:
                # If anything goes wrong adding the dashed square, skip silently to avoid breaking visualization
                pass

        # self.axes[0].set_title("Rollout Visualization")
        self.axes[0].set_title("Trajectory Visualization", fontweight='bold')
        self.axes[0].set_xlabel("X Position")
        self.axes[0].set_ylabel("Y Position")
        self.axes[0].set_aspect('equal')
        self.axes[0].grid(True, alpha=0.3)  # Add grid lines

        self.timestep = 0
        self.env_pos_scatters = {} # list of scatter:pos for each env
        self.env_footprint_plots = {} # list of plots of the footprint of each env
        self.env_control_scatters = {} # list of optimal control rollouts for each env
        self.env_rollout_lines = {} # list of lines for each env
        self.trajectory_start_scatters = {}  # per-env red scatter for the start point of the trace
        self.env_prev_state_hist = []
        self.env_rollout_scatters = {}  # list of scatters for each env
        self.trajectory_traces = [[] for _ in range(self.vis_n_envs)]  # Now stores full states
        self.x = [[] for _ in range(self.vis_n_envs)]  # Reset x coordinates
        self.y = [[] for _ in range(self.vis_n_envs)]  # Reset y coordinates
        self.u = [[] for _ in range(self.vis_n_envs)]  # Reset x components of heading vectors
        self.v = [[] for _ in range(self.vis_n_envs)]  # Reset y components of heading vectors
        self.trajectory_velocity = [[] for _ in range(self.vis_n_envs)] # Reset magnitude of trajectory velocities
        # Line plot for velocity (norm(x^2 + y^2))
        if self.show_velocity:
            self.axes[1].set_title("Velocity")
            self.axes[1].set_xlabel("Timestep")
            self.axes[1].set_ylabel("Velocity")
            self.velocities = []
            self.velocities_lines = []

        # Add a (possibly empty) line handle so the trajectory trace appears in the legend.
        # The actual trace is drawn with a LineCollection; provide a matching line handle for legend.
        self._traj_legend_handle, = self.axes[0].plot(
            [], [], color="blue", linestyle="-", linewidth=2, label="Robot trajectory"
        )

        # Add colorbar for trajectory trace yaw rate coloring (will be updated dynamically)
        sm_traj = mpl.cm.ScalarMappable(cmap=self.red_blue, norm=mpl.colors.Normalize(vmin=-1, vmax=1))
        sm_traj.set_array([])
        # Create inset axes for colorbar inside the plot (horizontal, like a legend)
        # Position higher to avoid intersection with x-axis label
        cax = inset_axes(self.axes[0], width="30%", height="3%", loc='lower right', 
                        bbox_to_anchor=(-0.05, 0.08, 1, 1), bbox_transform=self.axes[0].transAxes, borderpad=0)
        self.traj_cbar = self.fig.colorbar(sm_traj, cax=cax, orientation='horizontal', 
                                            label="Yaw Rate (deg/s)")
        self.traj_cbar.ax.xaxis.set_label_position('top')
        self.traj_cbar.ax.tick_params(labelsize=8)
        self.sm_traj = sm_traj  # Store for dynamic updates

        for i in range(self.vis_n_envs):
            robot_triangle = mpl.patches.Polygon(
                self._robot_triangle_vertices(0.0, 0.0, 0.0, self.robot_marker_size),
                closed=True,
                facecolor='blue',
                edgecolor='black',
                linewidth=1.0,
                zorder=5,
            )
            self.axes[0].add_patch(robot_triangle)
            self.env_pos_scatters[i] = robot_triangle
            self.env_rollout_lines[i] = []
            self.env_rollout_scatters[i] = self.axes[0].scatter(
                [], [], alpha=0.95, s=26, edgecolors='black', linewidths=0.3, zorder=3
            )  # Scatter for rollouts with high contrast on light backgrounds

            # foot_print_plot = self.axes[0].plot([], [],'r-', alpha=0.5)[0]  # Plot for footprint
            # self.env_footprint_plots[i] = foot_print_plot

            if self.show_velocity:
                self.velocities.append(np.array([]))
                line = self.axes[1].plot([], [], label=f'Car {i}')[0]  # Line for velocity
                self.velocities_lines.append(line)

            if self.show_rollouts_scatter:
                optimal_control_scatter = self.axes[0].scatter(
                    [], [], c='green', alpha=1., s=20, edgecolors='black', lw=0.4
                )  # Scatter for optimal control
                self.env_control_scatters[i] = optimal_control_scatter

            # Create a red scatter to mark the start point of the robot trajectory trace for this env
            start_label = "Start Point" if i == 0 else None
            start_scatter = self.axes[0].scatter([], [], color='red', s=40, zorder=4, label=start_label)
            self.trajectory_start_scatters[i] = start_scatter

        # Keep legend disabled during live/video rendering; add only for saved still frames.
        self._main_legend = None
        
        # Apply tight layout to ensure everything fits in the frame
        self.fig.tight_layout()

    def close(self):
        """
        Closes the visualization.
        """
        plt.close(self.fig)

class RolloutsVideo:
    def __init__(self, env, rv: RolloutsVisualization, demonstrations: torch.Tensor=None):
        self.env = env.unwrapped
        self.rv = rv
        self.rv.demonstrations = demonstrations
        keys = self.env.cfg.scene.obstacle_collection.rigid_objects.keys()
        obstacle_positions_list = []
        obstacle_rotations_list = []

        for key in keys:
            obstacle_positions_list.append(self.env.cfg.scene.obstacle_collection.rigid_objects[key].init_state.pos[:2])

        self.obstacle_positions = torch.tensor(
            obstacle_positions_list,
            device=self.env.device,
        )
        self.rv.obs_side = self.env.cfg.scene.obstacle_collection.rigid_objects[list(keys)[0]].spawn.size[0]
        self.reset()

    def reset(self):
        self.rv.reset()
        self.img_frames = []
        self.num_resets = 0 # HACK: internal tracker

    def update(self, rollouts: np.ndarray, rollout_costs: np.ndarray,
               frame_timestep: int=None):
        self.rv.update(rollouts, rollout_costs, frame_timestep)
        self.update_video()
        self.rv.close()

    def update_video(self):
        """
        Updates the video with the latest frame.
        """
        # Redraw canvas for updates
        buf = BytesIO()
        self.rv.fig.savefig(buf, format='png')
        buf.seek(0)
        self.img_frames.append(imageio.imread(buf))
        buf.close()

    def save_video(self, output_dir, episode_num: int=None, frame_rate: int=10):
        """
        Combines frames stored in memory into a video and saves it to disk.
        """
        fname = f"rollouts_video.mp4"
        if episode_num is not None:
            fname = f"episode-{episode_num}_{fname}"

        path = os.path.join(output_dir, fname)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        imageio.mimsave(path, self.img_frames, fps=frame_rate)
        print(f"Video saved to {path}")

        return path
    
    def save_frame(self, output_dir, step: int=None):
        """
        Saves the current frame to disk.
        """
        return self.rv.save_frame(output_dir, step)


if __name__ == "__main__": # Test
    import time

    def generate_test_data(num_envs, num_rollouts, num_timesteps, state_dim):
        deltas = np.random.randn(num_envs, num_rollouts, num_timesteps, state_dim)
        for i in range(num_envs):
            # start in the same position for every rollout
            deltas[i, :, 0, :] = np.random.randn(2)
        rollouts = np.cumsum(deltas, axis=2)

        rollout_costs = np.random.rand(num_envs, num_rollouts)  # Random performance metrics
        return rollouts, rollout_costs

    def test_rollouts_visualization_in_memory():
        num_envs = 3
        num_rollouts = 5
        state_dim = 2
        trail_length = 10
        num_timesteps = 5
        sim_timesteps = 20

        vis_env_ids = [0, 1, 2]  # Visualize all environments
        visualization = RolloutsVisualization(
            vis_n_envs=num_envs,
            vis_n_rollouts=num_rollouts,
            # trail_length=trail_length,
            posx_ind=0,
            posy_ind=1
        )

        for sim_timestep in range(sim_timesteps):
            rollouts, rollout_costs = generate_test_data(num_envs, num_rollouts, num_timesteps, state_dim)
            visualization.update(rollouts, rollout_costs, sim_timestep)
            plt.pause(0.1)  # For live view during simulation

        # Save the video at the end
        visualization.save_video("rollouts_simulation_in_memory.mp4", frame_rate=10)
        plt.show()

    start = time.time()
    test_rollouts_visualization_in_memory()
    print(f"Time taken: {time.time() - start} seconds")