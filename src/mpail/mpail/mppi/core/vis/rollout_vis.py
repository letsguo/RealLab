import torch
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from io import BytesIO
import imageio

import math

from numpy.ma import masked_array

from scipy.ndimage import rotate

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
                 vis_n_envs,
                 vis_n_rollouts,
                 map_res_m_px,
                 map_length_px,
                 cost_range=None,  # Add cost_range parameter
                #  trail_length=10, # TODO: implement trail_length
                 posx_ind=0,
                 posy_ind=1,
                 xlim=None,
                 ylim=None,
                 save_dir=None,
                 show_velocity=False,
                 show_elevation=False,
                 show_trajectory_trace=False,  # Add parameter to enable trajectory trace
                 **kwargs):


        # self.trail_length = trail_length # TODO
        self.vis_n_envs = vis_n_envs
        self.vis_n_rollouts = vis_n_rollouts
        self.posx_ind = posx_ind
        self.posy_ind = posy_ind
        self.xlim = xlim
        self.ylim = ylim
        self.save_dir = save_dir
        self.map_res_m_px = map_res_m_px
        self.map_length_px = map_length_px
        self.show_velocity = show_velocity
        self.show_elevation = show_elevation
        self.cost_range = cost_range  # Store the cost range if provided
        self.show_trajectory_trace = show_trajectory_trace
        self.trajectory_traces = [[] for _ in range(self.vis_n_envs)]  # Store trajectory traces for each environment
        self.reset()

    def update(
        self,
        x0: np.ndarray,
        rollouts: np.ndarray,
        rollout_costs: np.ndarray,
        elevation_map: np.ndarray,
        optimal_control: np.ndarray,
        frame_timestep: int=None
    ):
        '''
        x0: [num_envs, state_dim]
        rollouts: [num_envs, num_rollouts, num_timestep, state_dim]
        rollout_costs: [num_envs, num_rollouts, num_timesteps]
        metrics: [num_envs, num_rollouts]
        vis_rollout_ids: list of rollout indices to visualize
        frame_timestep: current simulation timestep
        '''

        # Normalize performance metrics for color coding: [num_envs, num_rollouts]
#         # normed_costs = (rollout_costs - rollout_costs.min()) \
#                         #                 / (rollout_costs.max() - rollout_costs.min() + 1e-5)

        # Update colorbar limits only if cost_range is not provided
        if self.cost_range is None:
            self.sm.set_clim(rollout_costs.min(), rollout_costs.max())
        else:
            self.sm.set_clim(*self.cost_range)

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

            vx = env_rollouts[0, 0, 6]
            vy = env_rollouts[0, 0, 7]

            # plot optimal control rollout
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
                velocity = np.sqrt(vx**2 + vy**2)
                self.velocities[i] = np.append(self.velocities[i], velocity)
                self.velocities_lines[i].set_data(range(len(self.velocities[i])), self.velocities[i])
                self.axes[1].relim()
                self.axes[1].autoscale_view()

            yaw = env_rollouts[0, 0, 5]  # TODO: switch to query bevmap

            yaw = np.nan_to_num(yaw, nan=0)
            rotate_angle = -yaw * 180 / math.pi

            self.env_pos_scatters[i].set_offsets(np.c_[curr_x, curr_y])

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

            # Update trajectory lines
            env_scatters: list = self.env_rollout_scatters[i]
            # Update trajectory scatter
            all_x_vals = np.concatenate([env_rollouts[j][:, 0] for j in range(self.vis_n_rollouts)])
            all_y_vals = np.concatenate([env_rollouts[j][:, 1] for j in range(self.vis_n_rollouts)])

            # Update colors
            env_rollout_costs = rollout_costs[i]  # [num_rollouts]
            all_colors = np.concatenate([self.sm.to_rgba(env_rollout_costs[j]) for j in range(self.vis_n_rollouts)])

            env_scatters.set_offsets(np.c_[all_x_vals, all_y_vals])
            env_scatters.set_color(all_colors)
            # env_scatters.set_sizes([2] * len(all_x_vals))  # Set marker size

            # Add current position to trajectory trace
            if self.show_trajectory_trace:
                self.trajectory_traces[i].append((curr_x, curr_y))
                trace_x, trace_y = zip(*self.trajectory_traces[i])
                if len(self.env_rollout_lines[i]) == 0:
                    # Create a new line if it doesn't exist
                    line, = self.axes[0].plot(trace_x, trace_y, 'b-', alpha=0.5, lw=1)
                    self.env_rollout_lines[i].append(line)
                else:
                    # Update the existing line
                    self.env_rollout_lines[i][0].set_data(trace_x, trace_y)

        # Update the time text
        if frame_timestep is not None:
            self.timestep = frame_timestep
            self.time_text.set_text(f"Simulation Timestep: {self.timestep}")

        # Update boundaries
        if self.xlim is None or self.ylim is None:
            current_xlim = self.axes[0].get_xlim()
            current_ylim = self.axes[0].get_ylim()
            new_xlim = (min(current_xlim[0], rollouts[..., 0].min() - 1), max(current_xlim[1], rollouts[..., 0].max() + 1))
            new_ylim = (min(current_ylim[0], rollouts[..., 1].min() - 1), max(current_ylim[1], rollouts[..., 1].max() + 1))
            self.axes[0].set_xlim(new_xlim)
            self.axes[0].set_ylim(new_ylim)

    def save_frame(self, output_dir: str=None, step: int=None):
        """
        Saves the current frame to disk.
        """
        fname = f"rollouts_vis.png"
        if step is not None:
            fname = f"step-{step}_{fname}"

        if output_dir is None:
            path = os.path.join(self.save_dir, fname)
        else:
            path = os.path.join(output_dir, fname)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.fig.savefig(path)

    def reset(self):
        ''' Set up plot features '''

        if self.show_velocity:
            self.fig, self.axes = plt.subplots(1, 2, figsize=(16, 8))
        else:
            self.fig, self.axes = plt.subplots(1, 1, figsize=(8, 8))
            self.axes = [self.axes]  # Ensure self.axes is always a list

        # Add colorbar to visualize the normalized performance metrics
        self.cmap = get_cmap("seismic")
        if self.cost_range is None:
            norm = mpl.colors.Normalize(vmin=0, vmax=30)  # Default normalization
        else:
            norm = mpl.colors.Normalize(vmin=self.cost_range[0], vmax=self.cost_range[1])  # Use cost_range
        self.sm = mpl.cm.ScalarMappable(
            cmap=self.cmap,
            norm=norm
        )
        self.sm.set_array([])
        self.cbar = self.fig.colorbar(self.sm, ax=self.axes[0], orientation='vertical',
                          label="Cost")
        self.time_text = self.axes[0].text(0.05, 0.95, '', transform=self.axes[0].transAxes,
                                      fontsize=12, verticalalignment='top')

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

        if self.xlim:
            self.axes[0].set_xlim(self.xlim)
        if self.ylim:
            self.axes[0].set_ylim(self.ylim)

        self.axes[0].set_title("Rollout Visualization")
        self.axes[0].set_xlabel("X Position")
        self.axes[0].set_ylabel("Y Position")
        self.axes[0].set_aspect('equal')
        self.axes[0].grid(True)  # Add grid lines

        self.timestep = 0
        self.env_pos_scatters = {} # list of scatter:pos for each env
        self.env_footprint_plots = {} # list of plots of the footprint of each env
        self.env_control_scatters = {} # list of optimal control rollouts for each env
        self.env_rollout_lines = {} # list of lines for each env
        self.env_prev_state_hist = []
        self.env_rollout_scatters = {}  # list of scatters for each env
        self.trajectory_traces = [[] for _ in range(self.vis_n_envs)]  # Reset trajectory traces

        # Line plot for velocity (norm(x^2 + y^2))
        if self.show_velocity:
            self.axes[1].set_title("Velocity")
            self.axes[1].set_xlabel("Timestep")
            self.axes[1].set_ylabel("Velocity")
            self.velocities = []
            self.velocities_lines = []

        self.axes[0].legend()

        for i in range(self.vis_n_envs):
            scatter = self.axes[0].scatter([], [], s=100)  # Scatter for current states (increased size)
            self.env_pos_scatters[i] = scatter
            self.env_rollout_lines[i] = []
            self.env_rollout_scatters[i] = self.axes[0].scatter([], [], alpha=0.5, s=20)  # Scatter for trajectory (increased size)

            # foot_print_plot = self.axes[0].plot([], [],'r-', alpha=0.5)[0]  # Plot for footprint
            # self.env_footprint_plots[i] = foot_print_plot

            if self.show_velocity:
                self.velocities.append(np.array([]))
                line = self.axes[1].plot([], [], label=f'Car {i}')[0]  # Line for velocity
                self.velocities_lines.append(line)

            optimal_control_scatter = self.axes[0].scatter(
                [], [], c='green', alpha=1., s=20, edgecolors='black', lw=0.4
            )  # Scatter for optimal control
            self.env_control_scatters[i] = optimal_control_scatter

        # Adjust the size of the velocity plot
        if self.show_velocity:
            box = self.axes[1].get_position()
            self.axes[1].set_position([box.x0, box.y0, box.width * 0.5, box.height * 0.5])

    def close(self):
        """
        Closes the visualization.
        """
        plt.close(self.fig)

class RolloutsVideo:
    def __init__(self, rv: RolloutsVisualization):
        self.rv = rv
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
