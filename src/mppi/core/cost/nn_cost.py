import torch
from typing import Callable, TYPE_CHECKING
# from rsl_rl.utils import resolve_nn_activation

if TYPE_CHECKING:
    from .cost_cfg import NNCostCfg


class NNCost(torch.nn.Module):

    def __init__(self,
        cost_cfg: 'NNCostCfg',
        num_envs: int,
        device: torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.to(device=device, dtype=dtype)
        self.num_envs = num_envs
        self.cfg = cost_cfg

        if self.cfg.model is not None:
            self.model = self.cfg.model

        elif self.cfg.model_factory is not None:
            self.model = self.cfg.model_factory(**self.cfg.model_kwargs)

        else:
            raise ValueError("Either model or model_factory must be provided")

        self.input_dim = self.model[0].in_features

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        costs = self.model(input).squeeze(-1)

        if self.cfg.clip_costs:
            costs = torch.clamp(costs, min=self.cfg.clip_costs[0], max=self.cfg.clip_costs[1])

        return costs
