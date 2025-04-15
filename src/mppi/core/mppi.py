import torch
from typing import TYPE_CHECKING

from .maps.bev_map import BEVMap

if TYPE_CHECKING:
    from . import MPPICfg

# TODO: restore to original MPPI
class MPPI(torch.nn.Module):
    def __init__(self,
                 mppi_config: "MPPICfg",
                 num_envs: int,
                 device: torch.device = "cuda",
                 dtype = torch.float):
        super().__init__()
        self.to(device=device, dtype=dtype)
        self.device, self.dtype = device, dtype

        self.num_envs = num_envs

        self.cfg = mppi_config

        self.bevmap:BEVMap = self.cfg.map_cfg.class_type(
            self.cfg.map_cfg,
            self.num_envs,
            device=self.device,
            dtype=self.dtype
        )
        self.dynamics = self.cfg.dynamics_cfg.class_type(
            self.cfg.dynamics_cfg,
            self.num_envs,
            bevmap=self.bevmap,
            device=self.device,
        )
        self.costs = self.cfg.cost_cfg.class_type(
            self.cfg.cost_cfg,
            self.num_envs,
            device=self.device
        )
        self.sampling = self.cfg.sampling_cfg.class_type(
            self.cfg.sampling_cfg,
            self.num_envs,
            device=self.device
        )

        self.u_per_command = self.cfg.u_per_command
        self.num_envs = num_envs

        ## Initialize buffers (rollouts, costs, weights, optimal controls)
        self._x_shape = (self.num_envs, self.dynamics.x_dim) # x is agent state
        self._rollouts_shape = (self.num_envs, self.sampling.K, self.sampling.T, self.dynamics.state_dim)
        self._opt_controls_shape = (self.num_envs, self.sampling.T, self.sampling.nu)

        # For debugging, algorithmic data, and visualization
        # Updated upon calling optimize()
        self._x = torch.zeros(self._x_shape, device=self.device, dtype=self.dtype) # Current state
        self._opt_controls = torch.zeros(self._opt_controls_shape, device=self.device, dtype=self.dtype) # Optimal controls
        self._rollouts = torch.zeros(self._rollouts_shape, dtype=self.dtype, device=self.device) # Sampled rollouts
        self._cost_values = torch.zeros_like(self._rollouts[..., 0]) # [num_envs, K, T]
        self._weights = torch.zeros_like(self._cost_values[..., 0]) # [num_envs, K]
        self._opt_states = torch.zeros_like(self._rollouts[:, 0, ...])

        # For action selection wrappers
        self._next_sampled_controls = torch.zeros((self.num_envs, self.sampling.K, self.sampling.nu),
                                                  device=self.device, dtype=self.dtype) # Sampled controls

        self.reset()

        # # Initialize visualization if configured
        # self.vis = None
        # vis_cfg = getattr(self.cfg, "vis_cfg", None)
        # if vis_cfg:
        #     self.vis = vis_cfg.class_type(**vis_cfg.to_dict(),
        #                                   map_res_m_px=self.cfg.map_cfg.map_res_m_px,
        #                                   map_length_px=self.cfg.map_cfg.map_length_px,)

    @property
    def temperature(self):
        return self.cfg.temperature

    def reset(self, reset_inds: torch.Tensor=None):
        """
        Clear controller state after finishing a trial
        """
        if reset_inds is None:
            self._opt_controls[:] = 0.
        else:
            self._opt_controls[reset_inds] = 0.

    def step(self, x0, map, use_prev_opt=True) -> torch.Tensor:
        '''
        Perform update and forward pass of the MPPI controller in immediate sequence.
        Returns next best controls. Seeds optimization with previous optimal controls.
        '''
        self.update(x0, map) # Update belief
        return self.optimize(x0, use_prev_opt=use_prev_opt) # Forward pass and next best control

    def update(self, x0, map):
        '''Updates belief of the MPPI controller with new agent state (x0) and maps.'''

        # Check correct sizes
        assert x0.shape[0] == self.num_envs, "State must have the shape (num_envs, ...)"
        assert map.shape[1] == self.bevmap.map_length_px

        ## Get robot-centric BEV (not rotated into robot frame)
        self._x = x0
        _xyz = x0[..., :3]
        _yaw = x0[..., 5]
        # elev_map = elev_map.flip(1) #flip elev_map to match the perspective of the camera
        self.bevmap.update(map, _xyz, _yaw)

    def optimize(self, x0, use_prev_opt:bool=False) -> torch.Tensor:
        """
        Perform forward pass of the MPPI controller.
        :param: x0
        :use_prev_opt: If true, uses the previous optimal controls
                        for computing the next controls.

        Weight computation adapted from:
        https://github.com/UM-ARM-Lab/pytorch_mppi/blob/bfcc9150ec9066fb5a0f01b65ddb603c49c66867/src/pytorch_mppi/mppi.py#L197
        """
        assert x0.shape[-1] == self._x.shape[-1]

        # Seeds samples with previous mean if use_prev_opt is True
        if use_prev_opt:
            # Fills in the last u_per_command controls with the last control
            _last_opt_controls = self._opt_controls[:, -self.u_per_command, :].unsqueeze(-2)
            self._opt_controls[:] = torch.roll(self._opt_controls, shifts=self.u_per_command, dims=1)
            self._opt_controls[:, -self.u_per_command:, :] = _last_opt_controls
        else:
            self._opt_controls[:] = 0.

        # Sample rollouts
        with torch.no_grad(): # For nn costs
            controls = self.sampling.sample(prev_controls=self._opt_controls) # [num_envs, K, T, nu]
            self._next_sampled_controls[:] = controls[:, :, 0, :] # [num_envs, T, nu]
            self._rollouts[:] = self.dynamics(x0, controls)
            self._cost_values[:] = self.costs(self._rollouts)  # [num_envs, K, T]

        # Weight computation
        _not_env_dims = list(range(1, self._cost_values.ndim))
        beta = torch.amin(self._cost_values, dim=_not_env_dims, keepdim=True) # [num_envs, 1, 1]
        traj_cost_non_zero =  torch.sum(torch.exp(-(1. / self.temperature) * (self._cost_values - beta)),
                                         dim=-1, keepdim=True) # [num_envs, K, 1]
        eta = torch.sum(traj_cost_non_zero, dim=-2, keepdim=True) # [num_envs, 1, 1]

        # Update weights and optimal controls
        self._weights[:] = ((1. / eta) * traj_cost_non_zero).squeeze()
        self._opt_controls[:] = torch.sum(controls * self._weights[..., None, None], dim=-3)

        # Return next action
        next_controls = self._opt_controls[:, :self.u_per_command, :]
        if self._opt_controls.isnan().any():
            raise ValueError("Optimal controls contain NaNs.")
        return next_controls

    def create_vis(self):
        '''Creates a visualization using the current state of the MPPI controller.'''

        if not self.vis:
            raise ValueError("Debug visualization is not enabled. Enable visualization by providing " +
                             "a visualizer config to the MPPI configuration.")

        # TODO: visualize optimized rollout
        vis_env_ids = list(range(self.vis.vis_n_envs))
        vis_rollouts = self._rollouts[vis_env_ids] # [n_envs, n_rollouts, horizon, state_dim]
        vis_costs = self._cost_values[vis_env_ids] # [n_envs, n_rollouts, horizon]
        horizon, state_dim = vis_rollouts.shape[2:]

        # Get topk rollouts
        topk_cost_inds = torch.topk(vis_costs.sum(dim=-1), k=self.vis.vis_n_rollouts, largest=False).indices
        topk_rollout_inds = topk_cost_inds.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, horizon, state_dim)
        topk_vis_rollouts = vis_rollouts.gather(dim=1, index=topk_rollout_inds)
        topk_cost_rollout_inds = topk_cost_inds.unsqueeze(-1).expand(-1, -1, horizon)
        topk_vis_costs = vis_costs.gather(dim=1, index=topk_cost_rollout_inds)

        # Get k random rollouts
        # TODO: add this to config
        rand_inds = torch.randperm(vis_rollouts.shape[1])[:self.vis.vis_n_rollouts]
        rand_vis_rollouts = vis_rollouts[:, rand_inds]
        rand_vis_costs = vis_costs[:, rand_inds]

        vis_rollouts = torch.cat([topk_vis_rollouts, rand_vis_rollouts], dim=1)
        vis_costs = torch.cat([topk_vis_costs, rand_vis_costs], dim=1)

        # convert to numpy and send to vis
        x0 = self._x.cpu().numpy()
        vis_rollouts = vis_rollouts.cpu().numpy()
        vis_costs = vis_costs.cpu().numpy()
        elevation_map = self.bevmap.map.cpu().numpy()
        opt_states = self.dynamics(self._x, self._opt_controls).cpu().numpy() # [num_envs, T, state_dim]
        self.vis.update(
            x0,
            vis_rollouts,
            rollout_costs=vis_costs,
            elevation_map=elevation_map,
            optimal_control=opt_states
        )