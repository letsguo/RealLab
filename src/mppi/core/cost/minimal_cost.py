import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cost_cfg import MinimalCostCfg

class MinimalCost(torch.nn.Module):

    def __init__(self, cost_cfg: 'MinimalCostCfg', num_envs: int,
                 device: torch.device = "cuda", dtype = torch.float,
                 **kwargs):
        super().__init__()
        self.cfg = cost_cfg
        self.to(device=device, dtype=dtype)

        self.goal_pos = torch.tensor(self.cfg.goal_pos, dtype=dtype, device=device)

        self.num_envs = num_envs

    def forward(self, state):
        pos = state[..., 0:3]
        vel = state[..., 6:9]

        # Goal cost
        goal_cost = torch.linalg.norm(pos - self.goal_pos, dim=-1)
        goal_cost = goal_cost * self.cfg.goal_w

        # Speed cost
        speed_cost = torch.abs(vel[..., 0] - self.cfg.target_speed)
        speed_cost = speed_cost * self.cfg.speed_w

        return goal_cost + speed_cost

