# from isaaclab.utils import configclass
# from ..isaaclab_copy.configclass import configclass

from dataclasses import MISSING, dataclass

from .bev_map import BEVMap

@dataclass
class BEVMapCfg:

    map_length_px: int = MISSING
    ''' map length (pixels) '''

    map_res_m_px: float = MISSING
    ''' map resolution (meters per pixel). '''

    feature_dim: int = MISSING
    ''' feature dimension '''

    class_type: type = BEVMap

