import torch

from ..maps import BEVMap
import matplotlib.pyplot as plt
import numpy as np

class SimpleCarCost(torch.nn.Module):
    """
    Class for Dynamics modelling
    """
    def __init__(
        self,
        cost_cfg,
        num_envs,
        bevmap: BEVMap,
        dtype = torch.float32,
        device = torch.device("cuda"),
    ):
        super(SimpleCarCost, self).__init__()
        self.dtype = dtype
        self.d = device
        self.num_envs = num_envs
        self.bevmap = bevmap
        self.cfg = cost_cfg

        # Constants
        self.GRAVITY = 9.8
        self.car_w2 = self.cfg.car_bb_width / 2
        self.car_l2 = self.cfg.car_bb_length / 2

        # Costing parameters (e.g. thresholds)
        self.soft_speed_lim = self.cfg.soft_speed_lim  # speed limit
        self.critical_SA = self.cfg.critical_SA  # surface angle
        self.speed_target = self.cfg.speed_target  # target speed
        self.critical_RI = self.cfg.critical_RI  # Rollover Index
        self.critical_vert_acc = self.cfg.critical_vert_acc
        self.critical_vert_spd = self.cfg.critical_vert_spd
        self.wall_height = self.cfg.wall_height
        self.hard_speed_lim = self.cfg.hard_speed_lim
        self.fall_off_penalty = self.cfg.fall_off_penalty
        self.floor_elevation = self.cfg.floor_elevation

        # Cost Weights
        self.lethal_w = self.cfg.lethal_w  # lethal weight
        self.stop_w = self.cfg.stop_w
        self.speed_w = self.cfg.speed_w
        self.roll_w = self.cfg.roll_w
        self.height_diff_scale = self.cfg.height_diff_scale
        self.vel_w = self.cfg.vel_w

        # Goal Costing Parameters
        self.goal_w = self.cfg.goal_w
        self.next_goal = torch.zeros((num_envs,), device=self.d, dtype=int)
        self.lookahead = self.cfg.lookahead

        self.heading_w = self.cfg.heading_w if hasattr(self.cfg, "heading_w") else 0


    @torch.jit.export
    def set_speed_limit(self, speed_lim):
        self.speed_target = torch.tensor(speed_lim, dtype=self.dtype, device=self.d)

    def update_goal(self, pos):
        # pos: [num_envs, 2]
        # Update goal to next waypoint if within lookahead distance
        self.next_goal = torch.where(torch.linalg.norm(self.goal_waypoints[self.next_goal, :2] - pos, dim=-1) < self.lookahead, (self.next_goal + 1) % self.num_goals, self.next_goal)

    def forward(self, state, controls):

        _, K, T, _ = state.shape

        x = state[..., 0]
        y = state[..., 1]
        z = state[..., 2]
        roll = state[..., 3]
        pitch = state[..., 4]
        yaw = state[..., 5]
        vx = state[..., 6]
        vy = state[..., 7]
        vz = state[..., 8]
        wx = state[..., 9]
        wy = state[..., 10]
        wz = state[..., 11]

        # Normalizer
        normalizer = torch.tensor(1.0, device=self.d, dtype=self.dtype) / torch.tensor(float(state.shape[-2]), device=self.d, dtype=self.dtype)

        # Footprint cost (fall off + )
        # fp_costs = self.footprint_costs(x, y, yaw)

        # Velocity cost
        curr_vel = torch.sqrt(vx ** 2 + vy ** 2)
        # speeding_penalty = torch.tensor(1000000., device=self.d)
        # vel_cost = torch.maximum(self.soft_speed_lim - curr_vel, torch.tensor(0., device=self.d))
        # vel_cost = torch.where(curr_vel > self.hard_speed_lim, speeding_penalty, curr_vel)
        vel_cost = torch.abs(self.soft_speed_lim - curr_vel)
        # vel_cost = torch.where(vel_cost != speeding_penalty, self.soft_speed_lim - vel_cost, vel_cost)
        # vel_cost = vel_cost.sum(-1).unsqueeze(-1)
        vel_cost *= self.vel_w

        # Fall off cost
        ground_z = self.bevmap.get_height_xy(x, y)
        fall_off_cost = torch.where(ground_z <= self.floor_elevation,
                                    torch.tensor(self.fall_off_penalty, dtype=self.dtype, device=self.d), ground_z)

        # Roll cost
        ct = torch.cos(roll) * torch.cos(pitch)
        roll_cost = torch.clamp((torch.tensor(1.0, dtype=self.dtype, device=self.d) / ct) - self.critical_SA, 0, 10) + 5 * torch.clamp(torch.abs(vz) - self.critical_vert_spd, 0, 10.0)

        # TODO: TEST
        # wp_vec = self.goal_state.unsqueeze(dim=1).to(dtype=self.dtype, device=self.d) - state[..., -1, :2]
        # terminal_cost = torch.linalg.norm(wp_vec, dim=-1)  # [num_env, k]
        # heading_vec = torch.stack([torch.cos(yaw[..., -1]), torch.sin(yaw[..., -1])], dim=-1).to(dtype=self.dtype, device=self.d)
        # wp_vec /= terminal_cost.unsqueeze(-1)
        # heading_cost = torch.tensor(1.0, dtype=self.dtype, device=self.d) / torch.clamp(torch.sum(wp_vec * heading_vec, dim=-1), 0.1, 1)  # [1,512]

        # running_cost = normalizer * (self.roll_w * roll_cost) + self.speed_w * vel_cost
        # cost_to_go = self.goal_w * terminal_cost + self.heading_w * heading_cost

        #visualizing for debug
        # visualized_cost = (vel_cost + fall_off_cost).sum(-1).cpu().numpy()[0]
        # contains_nan = np.any(np.isnan(visualized_cost))
        # fig, ax = plt.subplots()

        # ax.bar(range(1024), visualized_cost)
        # ax.set_xlabel('Rollout #')
        # ax.set_ylabel('Cost')
        # ax.set_title('Final Costs')

        # save_path = "/home/yandabao/Desktop/IRL/wheeled_gym/logs/mppi-test/debug/debug costs.png"
        # plt.savefig(save_path, dpi=300)

        # print(f"Bar chart saved to {save_path}")

        # print(vel_cost.shape, fall_off_cost.shape, goal_cost.shape)

        # return fp_costs + goal_cost
        return fall_off_cost + fp_costs + vel_cost
        # return vel_cost + fall_off_cost + goal_cost
        # return fall_off_cost + goal_cost

    def footprint_costs(self, x, y, yaw):

        # Vehicle footprint calculations
        cy = torch.cos(yaw)
        sy = torch.sin(yaw)

        flx = x + self.car_l2 * cy - self.car_w2 * sy
        fly = y + self.car_l2 * sy + self.car_w2 * cy
        fl = torch.stack([flx, fly], dim=-1)
        frx = x + self.car_l2 * cy + self.car_w2 * sy
        fry = y + self.car_l2 * sy - self.car_w2 * cy
        fr = torch.stack([frx, fry], dim=-1)
        blx = x - self.car_l2 * cy - self.car_w2 * sy
        bly = y - self.car_l2 * sy + self.car_w2 * cy
        bl = torch.stack([blx, bly], dim=-1)
        brx = x - self.car_l2 * cy + self.car_w2 * sy
        bry = y - self.car_l2 * sy - self.car_w2 * cy
        br = torch.stack([brx, bry], dim=-1)
        
        self.rollout_footprints = torch.stack([fl, fr, bl, br], dim=-2) # save them for visualizing
        self.current_footprint = self.rollout_footprints[..., 0, :, :]

        fl_cost = torch.nan_to_num(self.bevmap.get_height_xy(fl[..., 0], fl[..., 1]), nan=self.fall_off_penalty)
        fr_cost = torch.nan_to_num(self.bevmap.get_height_xy(fr[..., 0], fr[..., 1]), nan=self.fall_off_penalty)
        bl_cost = torch.nan_to_num(self.bevmap.get_height_xy(bl[..., 0], bl[..., 1]), nan=self.fall_off_penalty)
        br_cost = torch.nan_to_num(self.bevmap.get_height_xy(br[..., 0], br[..., 1]), nan=self.fall_off_penalty)

        return fl_cost + fr_cost + bl_cost + br_cost
        
        raise NotImplementedError("Not tested")


        flx_px = self.meters_to_px(flx)
        fly_px = self.meters_to_px(fly)
        frx_px = self.meters_to_px(frx)
        fry_px = self.meters_to_px(fry)
        blx_px = self.meters_to_px(blx)
        bly_px = self.meters_to_px(bly)
        brx_px = self.meters_to_px(brx)
        bry_px = self.meters_to_px(bry)

        # State cost calculations
        state_cost = torch.zeros_like(x, dtype=self.dtype)
        state_cost = torch.max(state_cost, torch.square(self.GYMmap_path[_batch_inds, fly_px, flx_px, 0]))
        state_cost = torch.max(state_cost, torch.square(self.GYMmap_path[:, flx_px, fly_px, 0])[:, None, None])
        state_cost = torch.max(state_cost, torch.square(self.GYMmap_path[:, frx_px, fry_px, 0])[:, None, None])
        state_cost = torch.max(state_cost, torch.square(self.GYMmap_path[:, blx_px, bly_px, 0])[:, None, None])
        state_cost = torch.max(state_cost, torch.square(self.GYMmap_path[:, brx_px, bry_px, 0])[:, None, None])

        state_cost = state_cost + self.stop_w * torch.clamp(((torch.tensor(1.0, dtype=self.dtype, device=self.d) / self.GYMmap_normal[:, img_X, img_Y, 2]) - self.critical_SA), 0, 10)[:, None, None]