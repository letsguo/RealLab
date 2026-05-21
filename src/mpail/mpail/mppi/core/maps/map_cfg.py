from dataclasses import MISSING

from .bev_map import BEVMap

@dataclass(kw_only=True)
class BEVMapCfg:

    class_type: type = BEVMap

    map_length_px: int = MISSING
    ''' map length (pixels) '''

    map_res_m_px: float = MISSING
    ''' map resolution (meters per pixel). '''

    feature_dim: int = MISSING
    ''' feature dimension '''