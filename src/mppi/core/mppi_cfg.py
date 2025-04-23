from dataclasses import MISSING, dataclass
# from .isaaclab_copy.configclass import configclass
# from isaaclab import configclass
from typing import Union

from mppi.core.cost import CostBaseCfg
from mppi.core.dynamics import SimpleCarDynamicsCfg
from mppi.core.sampling import DeltaSamplingCfg
from mppi.core.maps import BEVMapCfg
from mppi.core.vis import RolloutVisConfig

@dataclass
class MPPICfg:

    seed: int = MISSING
    '''Seed for random number generator'''

    debug: bool = MISSING
    '''Debug flag'''

    temperature: float = MISSING
    '''Temperature of MPPI optimization step'''

    opt_iters: int = MISSING
    '''Number of optimization iterations per action'''

    u_per_command: int = MISSING
    '''Number of control commands per action'''

    cost_cfg : CostBaseCfg = MISSING
    '''Rollout evaluator configuration'''

    dynamics_cfg: SimpleCarDynamicsCfg = MISSING
    '''Dynamics model configuration'''

    sampling_cfg: DeltaSamplingCfg = MISSING
    '''MPPI Sampling procedure configuration'''

    map_cfg: BEVMapCfg = MISSING
    '''Map configuration'''

    vis_cfg: Union[RolloutVisConfig, None] = MISSING
    '''Visualization configuration. None disables visualization'''