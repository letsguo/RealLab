from dataclasses import dataclass, MISSING
# from ..isaaclab_copy.configclass import configclass
from typing import Type

from .delta_sampling import DeltaSampling

@dataclass
class SamplingCfg:

    control_dim: int = MISSING
    """ Control dimension """

    noise_0: float = MISSING
    """ Noise 0 """

    noise_1: float = MISSING
    """ Noise 1 """

    scaled_dt: float = MISSING
    """ Scaled dt """

    temperature: float = MISSING
    """ Temperature (keep at 0.02 or 0.03) """

    max_dv: float = MISSING
    """ Max delta throttle per step """

    max_dtheta: float = MISSING
    """ Max delta steer per step """

    max_thr: float = MISSING
    """ Max throttle (delta v) """

    min_thr: float = MISSING
    """ Min throttle """

    num_rollouts: int = MISSING
    """ Number of rollouts """

    num_timesteps: int = MISSING
    """ Number of timesteps """


@dataclass
class DeltaSamplingCfg():

    temperature: float = MISSING
    """ Temperature (keep at 0.02 or 0.03) """

    control_dim: int = MISSING
    """ Control dimension """

    noise_0: float = MISSING
    """ Noise 0 """

    noise_1: float = MISSING
    """ Noise 1 """

    scaled_dt: float = MISSING
    """ Scaled dt """

    max_dv: float = MISSING
    """ Max delta throttle per step """

    max_dtheta: float = MISSING
    """ Max delta steer per step """

    max_thr: float = MISSING
    """ Max throttle (delta v) """

    min_thr: float = MISSING
    """ Min throttle """

    num_rollouts: int = MISSING
    """ Number of rollouts """

    num_timesteps: int = MISSING
    """ Number of timesteps """
    
    class_type: Type[DeltaSampling] = DeltaSampling