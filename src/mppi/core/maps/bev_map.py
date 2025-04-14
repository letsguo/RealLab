from typing import Tuple
import torch

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mpail.mppi.core.maps import BEVMapCfg


class BEVMap:
    '''
    A map of the environment that the agent uses to plan its path.
    It moves and rotates with the agent.

    Provides useful features for querying the space in physical space
    '''
    def __init__(
        self,
        map_config:'BEVMapCfg',
        num_envs: int,
        dtype=torch.float32,
        device=torch.device("cuda")
    ):
        self.dtype = dtype
        self.d = device
        self.num_envs = num_envs
        self.cfg = map_config
        self.map_length_px = self.cfg.map_length_px
        self.map_res_m_px = self.cfg.map_res_m_px
        self.feature_dim = self.cfg.feature_dim

        # Initialize map
        self.map_origin = torch.zeros((num_envs, 3), device=self.d, dtype=self.dtype)
        self.map_yaw = torch.zeros(num_envs, device=self.d, dtype=self.dtype) # radians
        self.map = torch.zeros((num_envs, self.map_length_px, self.map_length_px, self.feature_dim), device=self.d)

    def update(self, map_values, map_origin, map_yaw):
        '''
        Update the map with new height and normal maps

        Args:
            height_map (torch.Tensor):
                height map of shape (num_envs, map_length_px, map_length_px) or
                (num_envs, map_length_px * map_length_px)
            normal_map (torch.Tensor): normal map of shape (num_envs, map_length_px, map_length_px, 3)
            map_origin (torch.Tensor): origin of the map in the world frame of shape (num_envs, 3)
            map_yaw (torch.Tensor): yaw of the map in the world frame of shape (num_envs,)
        '''
        assert map_values.shape[0] == self.num_envs
        assert map_values.shape[1] == self.map_length_px
        assert map_origin.shape[1] == 3
        assert map_yaw.shape[0] == self.num_envs

        self.map[:] = map_values
        self.map_origin = map_origin
        self.map_yaw = map_yaw

    def get_map_ij(self, i:torch.Tensor, j:torch.Tensor) -> torch.Tensor:
        '''Get height at batched i,j in pixels from the map'''
        _batch_inds = torch.arange(self.num_envs, device=self.d).view(-1, *(1,)*(i.ndim-1)).expand(-1, *i.shape[1:])
        return self.map[_batch_inds, i, j]

    def xy_to_ij(self, x:torch.Tensor, y:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Convert batched x,y in meters to batched i,j in pixels'''
        # _map_origin = self.map_origin.view(self.num_envs, *(1,)*(x.ndim-1), 3)

        rots = torch.stack([
            torch.cos(self.map_yaw), -torch.sin(self.map_yaw),
            torch.sin(self.map_yaw), torch.cos(self.map_yaw),
        ], dim=-1).view(self.num_envs, 2, 2)
        xy = torch.stack([x, y], dim=-1)

        # Transform global xy to body xy. TODO: use transformation objects
        _xy = xy.view(self.num_envs, -1, 2)
        _map_origin = self.map_origin.view(self.num_envs, 1, 3)
        xy_body = torch.matmul(_xy - _map_origin[..., :2], rots).view(*xy.shape)

        # NOTE: i corresponds with y and j corresponds with x
        i = torch.clamp(
                (xy_body[..., 1] // self.map_res_m_px + self.map_length_px // 2).to(dtype=torch.long),
                0, self.map_length_px - 1
            )
        j = torch.clamp(
                (xy_body[..., 0] // self.map_res_m_px + self.map_length_px // 2).to(dtype=torch.long),
                0, self.map_length_px - 1
            )

        return i, j

    def get_map_xy(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        '''
        Get height at batched x,y in meters from the map
        TODO: could be cool do so something like:
            bevmap.xy[1.2, 3.4] = height_xy
            bevmap.ij[1, 2] = height_ij
            bevmap.xy[[1.2, 3.4], [5.6, 7.8]] = height_xy # batched
        '''
        i, j = self.xy_to_ij(x, y)
        return self.get_map_ij(i, j)
