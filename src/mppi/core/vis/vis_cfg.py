from dataclasses import MISSING, dataclass, field
# from ..isaaclab_copy.configclass import configclass
from typing import Type, Union


from .rollout_vis import RolloutsVisualization

@dataclass
class RolloutVisConfig:

    # vis_rollouts: bool = field(default=MISSING)
    # vis_n_envs: int = field(default=MISSING)
    # vis_n_rollouts: int = field(default=MISSING)
    # xlim: tuple = field(default=MISSING)
    # ylim: tuple = field(default=MISSING)
    # show_velocity: bool = field(default=MISSING)
    # show_elevation: bool = field(default=MISSING)
    # cost_range: Union[tuple, None] = field(default=MISSING)
    # show_trajectory_trace: bool = field(default=MISSING)
    # class_type: Type[RolloutsVisualization] = RolloutsVisualization
    vis_rollouts: bool = MISSING

    vis_n_envs: int = MISSING

    vis_n_rollouts: int = MISSING

    xlim: tuple = MISSING

    ylim: tuple = MISSING

    show_velocity: bool = MISSING

    show_elevation: bool = MISSING

    cost_range: Union[tuple, None] = MISSING

    show_trajectory_trace: bool = MISSING

    class_type: Type[RolloutsVisualization] = RolloutsVisualization