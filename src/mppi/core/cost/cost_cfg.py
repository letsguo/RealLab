import torch.nn as nn
# from isaaclab.utils import configclass
# from ..isaaclab_copy.configclass import configclass
from dataclasses import dataclass, MISSING
from typing import Callable, Any, List, Tuple, Type, Union

from .simple_car_cost import SimpleCarCost
from .minimal_cost import MinimalCost
from .nn_cost import NNCost

from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple, Type, Any

@dataclass
class CostBaseCfg:

    class_type: type = MISSING
    '''class type of cost function'''

@dataclass
class NNCostCfg:

    model: Union[nn.Sequential, None] = MISSING
    '''model to be used directly for costing'''

    model_factory: Union[Callable[[Any], nn.Sequential], None] = MISSING
    '''model factory to create model for costing'''

    model_kwargs: Union[dict, None] = MISSING
    '''model arguments if model_factory is used'''

    clip_costs: Union[Tuple[float, float], None] = MISSING
    
    class_type: Type[NNCost] = NNCost

@dataclass
class MinimalCostCfg():

    goal_w: float = MISSING
    '''weight on terminal goal cost'''

    goal_pos: list = MISSING
    '''goal position'''

    speed_w: float = MISSING
    '''weight with which target speed will be tracked'''

    target_speed: float = MISSING
    '''target speed in m/s'''

    class_type: Type[MinimalCost] = MinimalCost


@dataclass
class SimpleCarCostCfg():

    goal_w: float = None
    '''weight on terminal goal cost'''

    speed_w: float = None
    '''weight with which target speed will be tracked'''

    roll_w: float = None
    '''weight on roll index but also controls for lateral acceleration limits'''

    lethal_w: float = None
    '''weight on lethal stuff'''

    stop_w: float = None
    '''weight on actually lethal things like people, trees, rocks and so on'''

    speed_target: float = None
    '''target speed in m/s'''

    critical_SA: float = None
    '''critical slope angle'''

    critical_RI: float = None
    '''limiting ratio of lateral to vertical acceleration'''

    car_bb_width: float = None
    '''car bounding box width'''

    car_bb_length: float = None
    '''car bounding box length'''

    critical_vert_acc: float = None
    '''critical vertical acceleration'''

    critical_vert_spd: float = None
    '''critical vertical speed'''

    heading_w: float = None
    '''weight on heading'''

    soft_speed_lim: float = None
    '''soft speed limit // this is not a strong enough motivator for the car, uneffective'''

    hard_speed_lim: float = None
    '''a hard speed limit, if the car exceeds this speed, it will be harshly penalized'''

    height_diff_scale: float = None
    '''height cost weight'''

    wp_radius: float = None

    lookahead: float = None

    wall_height: float = None
    '''wall height'''

    vel_w: float = None
    '''velocity weight'''

    fall_off_penalty: float = None
    '''fall off penalty'''

    floor_elevation: float = None
    '''floor elevation'''

    class_type: Type[SimpleCarCost] = SimpleCarCost