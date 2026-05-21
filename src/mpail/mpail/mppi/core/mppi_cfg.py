from dataclasses import MISSING, dataclass

from mpail.mppi.core.cost import CostBaseCfg
from mpail.mppi.core.dynamics import SimpleCarDynamicsCfg
from mpail.mppi.core.sampling import DeltaSamplingCfg
from mpail.mppi.core.maps import BEVMapCfg
from mpail.mppi.core.vis import RolloutVisConfig

@dataclass(kw_only=True)
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

    vis_cfg: RolloutVisConfig | None = MISSING
    '''Visualization configuration. None disables visualization'''