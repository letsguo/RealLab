import torch
import torch.distributions as td
import math
from typing import TYPE_CHECKING, Any, Dict

from mpail.mppi import MPPIPolicy

if TYPE_CHECKING:
    from .mpail_cfg import MPAILPolicyCfg

class MPAILPolicy(MPPIPolicy):
    '''Extends MPPI to add action distribution for exploration and learnable parameters'''

    dynamics: torch.nn.Module
    costs: torch.nn.Module
    sampling: torch.nn.Module
    map: torch.nn.Module

    def __init__(self,
        policy_config: 'MPAILPolicyCfg',
        num_envs: int,
        device: torch.device = "cuda",
        dtype = torch.float32,
    ):
        super().__init__(mppi_config=policy_config, num_envs=num_envs, device=device, dtype=dtype)
        self.cfg = policy_config
        self.to(device=device, dtype=dtype)
        self.num_envs = num_envs
        self.device, self.dtype = device, dtype

        # Make temperature learnable
        self._temp_exp = torch.nn.Parameter(torch.log(torch.tensor(self.cfg.temperature, device=self.device, dtype=self.dtype)))

        match self.cfg.action_dist:
            case "normal":
                self.std = self.cfg.action_dist_params.get("init_std", 1.0) \
                    * torch.ones(self.sampling.nu, device=self.device)
                self.min_std = self.cfg.action_dist_params.get("min_std", 0.1)
                self.update_action_distribution()
                self.dist_kwargs = { # For logging
                    "mean": self.mean,
                    "std": self.std,
                    "min_std": self.min_std
                }
            case "categorical":
                raise NotImplementedError("Categorical actions not implemented")
            case _:
                raise ValueError(f"Invalid action distribution: {self.cfg.action_dist}")

        print(f"[INFO] MPAILPolicy initialized. Total number of params: {sum(p.numel() for p in self.parameters())}")
        print(f"[INFO] \tDynamics: {sum(p.numel() for p in self.dynamics.parameters())}")
        print(f"[INFO] \tSampling: {sum(p.numel() for p in self.sampling.parameters())}")
        print(f"[INFO] \tCost: {sum(p.numel() for p in self.costs.parameters())}")
        print(f"[INFO] \tTemperature: {self.temperature.numel()}")
        print(self) # Prints torch summary of the model

    @property
    def temperature(self) -> torch.Tensor:
        temp = torch.exp(self._temp_exp)
        if self.cfg.min_temp is not None:
            return torch.clamp(temp, min=self.cfg.min_temp)
        return temp

    def act(self, observations: Dict[str, torch.Tensor], use_prev_opt: bool = True, vis_rollouts: bool = False) -> torch.Tensor:

        super().act(observations=observations, use_prev_opt=use_prev_opt, vis_rollouts=vis_rollouts)
        # Update the action distribution after optimization has filled relevant buffers
        self.update_action_distribution()

        return self.act_dist.sample()

    def act_inference(self, observations: Dict[str, torch.Tensor], use_prev_opt: bool = True,
                      vis_rollouts: bool = False) -> torch.Tensor:
        return super().act(observations, use_prev_opt=use_prev_opt, vis_rollouts=vis_rollouts)

    def update_action_distribution(self):
        '''
        std(X) = sqrt( E[(X - E[X])^2] ) = sqrt(sum(p(x) * (x - E[X])^2))
        '''
        self.mean, std = self.fit_distribution_params(recompute_weights=False)
        self._computed_std = std # For debugging
        self.std = torch.maximum(std, torch.ones_like(std) * self.min_std)
        self.act_dist = td.Normal(self.mean, self.std)

    def entropy(self, recompute_weights:bool = False) -> torch.Tensor:
        # Reimplements entropy to maintain compuational graph for gradients through std
        _, std = self.fit_distribution_params(recompute_weights=recompute_weights)
        return (0.5 + 0.5 * math.log(2 * math.pi) + torch.log(std)).sum(dim=-1)  # [num_envs]

    def fit_distribution_params(self, recompute_weights: bool = False):
        '''
        Fit the distribution parameters to the current state of the controller
        :param recompute_weights: If true, recomputes the weights based on the current cost values to allow
        for computational graph accumulation
        '''
        if recompute_weights:
            _cost_values = self.costs(self._rollouts)
            _not_env_dims = list(range(1, _cost_values.ndim))
            beta = torch.amin(_cost_values, dim=_not_env_dims, keepdim=True)  # [num_envs, 1, 1]
            traj_cost_non_zero = torch.sum(
                torch.exp(-(1. / self.temperature) * (_cost_values - beta)),
                dim=-1, keepdim=True
            )  # [num_envs, K, 1]
            traj_cost_non_zero = torch.clamp(traj_cost_non_zero, min=1e-9)  # Avoid unstable gradients
            eta = torch.sum(traj_cost_non_zero, dim=-2, keepdim=True)  # [num_envs, 1, 1]
            eta = torch.clamp(eta, min=1e-9)  # Avoid division by zero
            probs = (1. / eta) * traj_cost_non_zero  # [num_envs, K, 1]
        else:
            probs = self._weights[..., None]

        mean = self._opt_controls[:, 0, :]
        variance = torch.sum(
            probs * (self._next_sampled_controls - mean.unsqueeze(-2)).pow(2),  # [num_envs, K, nu]
            dim=-2
        )  # [num_envs, nu]
        variance = torch.clamp(variance, min=1e-12)  # Ensure variance is non-negative
        std = torch.sqrt(variance)  # [num_envs, nu]
        std = torch.clamp(std, min=1e-6)  # Ensure std is not too small to avoid NaNs

        return mean, std

    def get_actions_log_prob(self, actions):
        return self.act_dist.log_prob(actions).sum(dim=-1)

    def action_mean(self) -> torch.Tensor:
        return self.mean

    def action_stddev(self) -> torch.Tensor:
        return self.std

    def compute_stats(self) -> Dict[str, Any]:
        _costs = self._cost_values.detach().clone()
        mean_ss_costs = _costs[..., :-1].mean()
        min_ss_costs = _costs[..., :-1].min().detach().clone()
        max_ss_costs = _costs[..., :-1].max().detach().clone()
        mean_ts_costs = _costs[..., -1].mean()
        max_ts_costs = _costs[..., -1].max().detach().clone()
        min_ts_costs = _costs[..., -1].min().detach().clone()
        std_ss_costs = _costs[..., :-1].std()
        std_ts_costs = _costs[..., -1].std()
        _traj_costs = _costs.sum(dim=-1) # [num_envs, K]
        min_traj_cost = _traj_costs.min(dim=-1).values.mean()
        max_traj_cost = _traj_costs.max(dim=-1).values.mean()
        mean_std_traj_cost = _traj_costs.std(dim=-1).mean()
        _entropy = self.entropy(recompute_weights=False).detach().clone()
        return {
            "MPPI/Mean Action Dist Std0": self.std[...,0].mean().detach().item(),
            "MPPI/Mean Action Dist Std1": self.std[...,1].mean().detach().item(),
            "MPPI/Mean Computed Std0": self._computed_std[...,0].mean().item(),
            "MPPI/Mean Computed Std1": self._computed_std[...,1].mean().item(),
            "MPPI/Mean Entropy": _entropy.mean().item(),
            "MPPI/Min Entropy": _entropy.min().item(),
            "MPPI/Max Entropy": _entropy.max().item(),
            "MPPI/Temperature": self.temperature.detach().item(),
            "MPPI/Average Terminal Cost": mean_ts_costs,
            "MPPI/Average Single-Step Cost": mean_ss_costs,
            "MPPI/Min Single-Step Cost": min_ss_costs.item(),
            "MPPI/Max Single-Step Cost": max_ss_costs.item(),
            "MPPI/Min Terminal Cost": min_ts_costs.item(),
            "MPPI/Max Terminal Cost": max_ts_costs.item(),
            "MPPI/Std Single-Step Cost": std_ss_costs,
            "MPPI/Std Terminal Cost": std_ts_costs,
            "MPPI/Average Min Trajectory Cost": min_traj_cost,
            "MPPI/Average Max Trajectory Cost": max_traj_cost,
            "MPPI/Average Std Trajectory Cost": mean_std_traj_cost
        }