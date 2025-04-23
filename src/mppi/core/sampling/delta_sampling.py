import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sampling_cfg import DeltaSamplingCfg


class DeltaSampling(torch.nn.Module):
    """
    Class for Dynamics modelling
    """
    def __init__(
        self,
        sampling_cfg: 'DeltaSamplingCfg',
        num_envs: int,
        dtype = torch.float32,
        device = torch.device("cuda"),
    ):
        super().__init__()
        self.dtype = dtype
        self.d = device
        self.cfg = sampling_cfg

        self.nu = self.cfg.control_dim
        self.K = self.cfg.num_rollouts
        self.T = self.cfg.num_timesteps
        self.num_envs = num_envs

        self.max_temp = 0.08
        self.temperature = self.cfg.temperature
        self.scaled_dt = self.cfg.scaled_dt

        self.CTRL_NOISE = torch.zeros((self.num_envs, self.nu, self.nu), device=self.d, dtype=self.dtype)
        self.CTRL_NOISE[:,0,0] = self.cfg.noise_0
        self.CTRL_NOISE[:,1,1] = self.cfg.noise_1

        self.CTRL_NOISE_inv = torch.inverse(self.CTRL_NOISE)
        self.CTRL_NOISE_MU = torch.zeros((self.num_envs, self.nu), dtype=self.dtype, device=self.d)

        ## for torchscript we have to initialize these things to same shape and size as what we'll use later
        torch.manual_seed(0)

        random_noise = torch.randn((self.num_envs, self.K, self.T, self.nu), device=self.d, dtype=self.dtype)
        matmul = torch.einsum('nktd,ndh->nkth', random_noise, self.CTRL_NOISE)
        self.noise = matmul + self.CTRL_NOISE_MU[:, None, None, :]

        self.max_thr = self.cfg.max_thr
        self.min_thr = self.cfg.min_thr
        self.max_dv = self.cfg.max_dv
        self.cost_total = 0

    def sample(self, prev_controls: torch.Tensor=None):
        '''
        :param: state: torch.Tensor of shape (num_envs, state_dim)
        :param: U: torch.Tensor of shape (num_envs, T, nu)
        :param: (Optional) prev_controls: torch.Tensor of shape (num_envs, T, nu)

        init_controls is the initial guess
        sampling is done in the delta control space
        add this to the previous delta_controls
        integrate delta_controls and add previous controls to get controls
        find new noise after clamping
        find perturbation cost
        return controls, perturbation cost
        '''

        if prev_controls is None:
            prev_controls = torch.zeros((self.num_envs, self.T, self.nu), device=self.d, dtype=self.dtype)

        # Compute delta controls from previous control
        U_mu = torch.diff(
            prev_controls, dim=-2,
            append=torch.zeros_like(prev_controls[:, -1:, :])
        ) # [num_envs, T, nu]
        u0 = prev_controls[:, 0:1, :]

        #  Added Noise
        normal_noise = torch.randn((self.num_envs, self.K, self.T, self.nu), device = self.d, dtype = self.dtype)
        scaled_noise = torch.einsum('nktd,ndh->nkth', normal_noise, self.CTRL_NOISE)
        self.noise = scaled_noise + self.CTRL_NOISE_MU[:, None, None, :]
        # scale and add mean

        delta_U = U_mu.unsqueeze(-3) + self.noise

        # Delta control space constraints through clamping
        delta_U[..., 0] = torch.clamp(delta_U[..., 0], -self.max_dv, self.max_dv)
        delta_U[..., 1] = torch.clamp(delta_U[..., 1], -self.cfg.max_dtheta, self.cfg.max_dtheta)

        controls = prev_controls[:, 0:1, :].unsqueeze(-2) + torch.cumsum(delta_U, dim=-2) # [num_envs, K, T, nu]

        # Control space constraints through clamping
        controls[..., 0] = torch.clamp(controls[..., 0], self.min_thr, self.max_thr) ## car can't go in reverse, can't have more than 50 % speed
        controls[..., 1] = torch.clamp(controls[..., 1], -1, 1)

        return controls

    # def update_control(self, cost_total, U, state):
    #     '''
    #     find total cost such that the minimum of total cost is not 0
    #     find the weighting for all the K samples
    #     update the delta controls (weighted average)
    #     integrate delta controls and add previous controls to obtain the applied controls

    #     cost_total: torch.Tensor of shape (num_envs, K, T)
    #     U: torch.Tensor of shape (num_envs, K, T, nu)
    #     state: torch.Tensor of shape (num_envs, state_dim)

    #     return controls, delta_controls
    #     '''

    #     # min for each agent
    #     beta = torch.amin(cost_total, dim=list(range(1, cost_total.ndim)), keepdim=True)
    #     self.cost_total = cost_total.clone()
    #     cost_total_non_zero =  torch.exp((-1 / self.temperature) * (cost_total - beta)).sum(dim=-1, keepdim=True).unsqueeze(-1)
    #     U = U * cost_total_non_zero

    #     eta = torch.sum(cost_total_non_zero, dim=(-3, -2, -1), keepdim=True)
    #     U = U/eta
    #     U = torch.sum(U, dim=-3)

    #     cost_total_non_zero_unsummed =  torch.exp((-1 / self.temperature) * (cost_total - beta))

    #     eta = torch.sum(cost_total_non_zero_unsummed, dim=(-2, -1), keepdim=True)
    #     omega = (1.0 / eta) * cost_total_non_zero_unsummed

    #     U_old = (omega.unsqueeze(-1) * self.noise).sum(dim=-3)

    #     return U, U_old