from dataclasses import dataclass, MISSING

from .delta_sampling import DeltaSampling

@dataclass(kw_only=True)
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
class DeltaSamplingCfg(SamplingCfg):

    class_type: type[DeltaSampling] = DeltaSampling