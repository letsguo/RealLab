import torch
from typing import TYPE_CHECKING

from mpail.utils import resolve_obj # Accommodate both yaml and Configclass

if TYPE_CHECKING:
    from .mpail_cfg import TDCostCfg, GAIfOCostCfg, NNCostCfg

class GAIfOCost(torch.nn.Module):

    class Discriminator(torch.nn.Module):
        '''Returns classification logits. '''
        def __init__(self, discriminator):
            super().__init__()
            self.discriminator = discriminator

        def forward(self, state):
            '''Returns expert logits and gen logits (reward) for the given states
            D_theta(s) = exp(r_E) / [exp(r_E) + exp(r_G)]
            logit(D) = r_E - r_G
            '''
            return self.discriminator(state)

    def __init__(
        self,
        cfg : 'GAIfOCostCfg',
        num_envs : int,
        device: torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.cfg = cfg

        self.cfg.model_kwargs["input_dim"] = cfg.state_dim * 2 # for s and s'
        self.cfg.model_kwargs.setdefault("use_spectral_norm", False)

        self.discriminator = resolve_obj(self.cfg.model_factory)(**self.cfg.model_kwargs)

    def forward(self, state):
        '''state shape: (num_envs, state_dim). Applies negative of expert reward for cost'''
        return -self.discriminator(state).squeeze(-1)

    def get_discriminator(self):
        return self.Discriminator(discriminator=self.discriminator)

    def reward(self, state):
        '''Returns reward for the given state'''
        return -self.forward(state)


class CostToGo(torch.nn.Module):

    class Value(torch.nn.Module):
        '''Returns classification logits. '''
        def __init__(self, value):
            super().__init__()
            self.value = value

        def forward(self, state):
            '''Returns expert logits and gen logits (reward) for the given states
            D_theta(s) = exp(r_E) / [exp(r_E) + exp(r_G)]
            logit(D) = r_E - r_G
            '''
            return self.value(state)

    def __init__(
        self,
        cfg : 'NNCostCfg',
        num_envs : int,
        device: torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.cfg = cfg
        self.value = resolve_obj(self.cfg.model_factory)(**self.cfg.model_kwargs)

    def forward(self, state):
        '''state shape: (num_envs, state_dim). Applies negative of expert reward for cost'''
        return -self.value(state).squeeze(-1)

    def get_value(self):
        return self.Value(value=self.value)


class TDCost(torch.nn.Module):
    '''Temporal Difference cost function. Evaluates full trajectories by evaluating
    using a single step class and a terminal state class'''
    def __init__(
        self,
        cfg : 'TDCostCfg',
        num_envs : int,
        device: torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.cfg = cfg

        self.ss_cfg = self.cfg.single_step_class_cfg
        self.ss_cost: GAIfOCost = resolve_obj(self.ss_cfg.class_type)(
            self.ss_cfg, device=device, num_envs=num_envs
        )
        self.ss_coeff = self.cfg.ss_coeff

        self.ts_cfg = self.cfg.terminal_state_class_cfg
        self.ts_cost: CostToGo = resolve_obj(self.ts_cfg.class_type)(
            self.ts_cfg, device=device, num_envs=num_envs
        )
        self.ts_coeff = self.cfg.ts_coeff

    def forward(self, rollouts: torch.Tensor):
        '''rollouts shape: (num_envs, rollouts, horizon, state_dim)'''

        costs = torch.zeros_like(rollouts[..., 0])

        # Evaluate single step costs
        # stack s and s' for discriminator input
        _input = torch.cat([rollouts[..., :-1, :], rollouts[..., 1:, :]], dim=-1)
        ss_costs = self.ss_cost(_input)
        costs[..., :-1] += self.ss_coeff * ss_costs

        # Evaluate terminal state costs
        ts_costs = self.ts_cost(rollouts[..., -1, :])
        costs[..., -1] += self.ts_coeff * ts_costs

        if self.cfg.risk is not None:
            _risk_factors = self.cfg.risk ** torch.arange(
                costs.shape[-1], device=rollouts.device, dtype=rollouts.dtype
            )
            costs *= _risk_factors

        return costs
