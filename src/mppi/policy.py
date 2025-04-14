from typing import Any, Dict, Callable, TYPE_CHECKING

import torch

from mppi import MPPI
if TYPE_CHECKING:
    from mpail.mppi import MPPICfg


class MPPIPolicy(MPPI):
    ''' API wrapping MPPI to work with gym '''
    def __init__(self,
        mppi_config: 'MPPICfg',
        num_envs: int,
        device: torch.device = "cuda",
        dtype = torch.float,
    ):
        super().__init__(mppi_config=mppi_config, num_envs=num_envs, device=device, dtype=dtype)
        self.to(device=device, dtype=dtype)
        self.num_envs = num_envs
        self.device, self.dtype = device, dtype

    def act(
        self,
        observations: Dict[str, torch.Tensor],
        use_prev_opt: bool = True,
        vis_rollouts: bool = False,
    ) -> torch.Tensor:
        '''Updates the belief of MPPI and performs the given number of optimization iterations.

        :param observation:
        :param use_prev_opt: Whether to use the previous optimal controls for the next optimization step
            Default for episodic policy evaluation should be True.
        :param vis_rollouts: Whether to visualize rollouts

        :return: Taken action according to the policy
        expected state is:
            [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz, throttle, steering]
        '''

        assert "proprioception" in observations, "Proprioception observation group missing"
        assert "exteroception" in observations, "Exteroception observation group missing"

        pro, ext = observations["proprioception"], observations["exteroception"]

        actions = self.step(x0=pro, map=ext, use_prev_opt=use_prev_opt)

        for _ in range(self.cfg.opt_iters - 1):
            # Subsequent optimization uses previous optimal controls
            actions = self.optimize(x0=pro, use_prev_opt=True)

        if vis_rollouts:
            self.create_vis()

        return actions.squeeze(-2)  # [num_envs, nu]

    def state(self):
        '''Returns the current state of the MPPI controller with map features concatenated.
        TODO: allow passing in observation to get state without changing internal state'''
        agent_state = self._x
        map_features = self.bevmap.get_map_xy(agent_state[:, 0], agent_state[:, 1])
        return torch.cat((agent_state, map_features), dim=-1)

    @property
    def action_dim(self):
        return self.sampling.nu
