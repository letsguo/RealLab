from dataclasses import MISSING
from isaaclab.utils import configclass

from .rollout_vis import RolloutsVisualization

@configclass
class RolloutVisConfig:

    class_type: type[RolloutsVisualization] = RolloutsVisualization

    vis_rollouts: bool = MISSING

    vis_n_envs: int = MISSING

    vis_n_rollouts: int = MISSING

    xlim: tuple = MISSING

    ylim: tuple = MISSING

    show_velocity: bool = MISSING

    show_elevation: bool = MISSING

    cost_range: tuple | None = MISSING

    show_trajectory_trace: bool = MISSING