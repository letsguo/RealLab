# from isaaclab.utils import configclass
from dataclasses import dataclass, field
# from isaaclab_copy import configclass

from mppi.core.vis.rollout_vis import RolloutsVisualization

from typing import Union, Type


from mppi.core import (
    SimpleCarCostCfg, MinimalCostCfg,
    SimpleCarDynamicsCfg, SimpleCarDynamicsNoActionCfg,
    DeltaSamplingCfg, SamplingCfg,
    BEVMapCfg,
    MPPICfg,
    RolloutVisConfig
)


@dataclass ## Check. Should we use dataclass or configclass? 
class MinCostConfig(MinimalCostCfg):
    goal_w: float                   = 1.
    speed_w: float                  = 10.
    goal_pos: list                  = field(default_factory=lambda: [1.0, -2.0, 0.0])
    target_speed: float             = 2.


@dataclass
class DynamicsConfig(SimpleCarDynamicsNoActionCfg):

    feat_dim: Union[int, None] = None
    concatenate_feats: bool            = False  # concatenate features to rollout states
    wheelbase: float                = 0.33   # wheelbase
    throttle_to_wheelspeed: float   = 1.0   # throttle to wheelspeed
    steering_max: float             = 0.488   # maximum steering angle
    dt: float                       = 0.05   # time step

@dataclass
class DynamicsNoActionConfig(SimpleCarDynamicsNoActionCfg):
    '''
    Configuration class for SimpleCarDynamics
    '''
    feat_dim: Union[int, None] = None
    concatenate_feats: bool            = False  # concatenate features to rollout states
    wheelbase: float                = 0.33   # wheelbase
    throttle_to_wheelspeed: float   = 3.0   # throttle to wheelspeed
    steering_max: float             = 0.488   # maximum steering angle
    dt: float                       = 0.1   # time step

@dataclass
# class SamplingConfig():
class SamplingConfig(DeltaSamplingCfg):
    control_dim: int                = 2     # control dimension
    noise_0: float                  = 1.0   # noise 0
    noise_1: float                  = 0.5   # noise 1
    scaled_dt: float                = 0.1   # scaled dt
    max_dv: float                   = 0.20  # max delta throttle per step
    max_dtheta: float               = 0.20  # max delta steering per step
    max_thr: float                  = 1.0   # max throttle (delta v)
    min_thr: float                  = 0.0   # min throttle
    num_rollouts: int               = 1024  # number of rollouts
    num_timesteps: int              = 20    # number of timesteps
    temperature: int             = 0.02  # temperature (keep at 0.02 or 0.03)

@dataclass
class MapConfig(BEVMapCfg):
    map_length_px: int              = 120   # gym map length (pixels)
    map_res_m_px: float             = 3./19.   # gym map resolution (meters per pixel).
    map_res_hitl: float             = 0.25  # map resolution hitl
    feature_dim: int                = 4     # feature dimension
    # TODO: may need to change to match the elevation map, look at existing yaml file


@dataclass
class VisConfig():
    vis_n_envs: int                 = 4     # number of environments to visualize
    vis_n_rollouts: int             = 10     # number of rollouts to visualize
    xlim: tuple                = (-1, 1)
    ylim: tuple                = (-1, 1)
    show_velocity: bool             = False
    show_elevation: bool            = False
    cost_range: tuple               = None  # cost range for visualization
    class_type: Type[RolloutsVisualization] = RolloutsVisualization
    # show_trajectory_trace = False
    # vis_rollouts: bool = True


@dataclass
class MPPIConfig(MPPICfg):

    seed: int                       = 0
    temperature: float              = 1.
    opt_iters: int                  = 1
    u_per_command: int              = 1
    debug: bool                     = False

    cost_cfg: MinCostConfig         = MinCostConfig()
    dynamics_cfg: DynamicsConfig     = DynamicsConfig()
    sampling_cfg: DeltaSamplingCfg     = SamplingConfig()
    map_cfg: MapConfig               = MapConfig()
    vis_cfg: VisConfig               = VisConfig()


@dataclass
class CostConfig(SimpleCarCostCfg):

    goal_w: float                   = 160.0   # weight on terminal goal cost
    speed_w: float                  = 0.15   # weight with which target speed will be tracked
    roll_w: float                   = 10.0   # weight on roll index but also controls for lateral acceleration limits
    lethal_w: float                 = 1.0   # weight on lethal stuff
    stop_w: float                   = 100.0   # weight on actually lethal things like people, trees, rocks and so on
    speed_target: float             = 5.0   # target speed in m/s
    critical_SA: float              = 1.414   # critical slope angle
    critical_RI: float              = 0.8   # limiting ratio of lateral to vertical acceleration
    car_bb_width: float             = 0.3   # car bounding box width
    car_bb_length: float            = 0.4   # car bounding box length
    critical_vert_acc: float        = 4.0   # critical vertical acceleration
    critical_vert_spd: float        = 0.2   # critical vertical speed
    heading_w: float                = 57.29   # weight on heading
    soft_speed_lim: float           = 3.0   # soft speed limit // this is not a strong enough motivator for the car, uneffective
    hard_speed_lim: float           = 3.0   # a hard speed limit, if the car exceeds this speed, it will be harshly penalized
    height_diff_scale: float        = 1000.0   # height cost weight
    wp_radius: float                = 2.0
    lookahead: float                = 2.5
    wall_height: float              = 8.3   # wall height
    vel_w: float                    = 0.0   # velocity weight
    fall_off_penalty: float         = 50.0   # fall off penalty
    floor_elevation: float          = -1.0   # floor elevation