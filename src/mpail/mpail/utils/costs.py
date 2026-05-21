import torch
import torch.nn.functional as F
from typing import Callable, Mapping

from mpail.mppi.core.cost.utils import fcnn_factory


class LogitToCost(torch.nn.Module):
    '''This is the GAIL definition, where the cost is the log of the probability
    that the sample came from the generator (e.g. \log{D} = \log\sigma{f}).
    Lower logit -> closer to 0 probability = is expert-leaning -> low cost'''
    def forward(self, logit: torch.Tensor) -> torch.Tensor:
        return F.logsigmoid(logit)


class ValueToCost(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return -value


def _cost_factory(final_activation: Callable, **kwargs) -> torch.nn.Sequential:
    ''' Helper function to create cost functions '''
    fcnn = fcnn_factory(**kwargs)
    model = torch.nn.Sequential(*fcnn, final_activation())
    return model


def gail_cost_factory(**kwargs) -> torch.nn.Sequential:
    '''Appends LogSigmoid to output of FCNN as in GAIL'''
    return _cost_factory(final_activation=LogitToCost, **kwargs)


def value_cost_net_factory(**kwargs) -> torch.nn.Sequential:
    '''Appends -value to output of FCNN as in ValueNetwork'''
    return _cost_factory(final_activation=ValueToCost, **kwargs)

