from typing import Tuple
import torch

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mpail.mppi.core.maps import BEVMapCfg

# from wheeledlab_tasks.visual.mdp_sensors.observations import Observation
import numpy as np

class VisualBEVMap:
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

        self.coord_manager = self.build_coord_manager()

    def build_coord_manager(self):
        extent = (self.map_length_px / 2) * self.map_res_m_px
        min_bound = [-extent, -extent, -2.0]
        max_bound = [ extent,  extent,  2.0]
        grid_size = [self.map_length_px, self.map_length_px, 1]

        return Coordinate(min_bound, max_bound, grid_size)


    def get_ground_map_from_pc(self, inputs, outputs, coord_manager, key='', label_key='elevation',
                                        feature_key='bev_features', training=False, invalid_height_val=-1000):
        out = dict()
        pc = inputs['points']
        pc_batch_indices = inputs['point_batch_indices']

        # sample target points
        label_ground_height = inputs[label_key].squeeze(1)

        features = outputs[feature_key]
        H, W = self.coord_manager.grid_size[:2]
        B, H, W = label_ground_height.shape

        # x --> W, y --> H
        # convert from points to bev coordinates [0, H], [0, W]
        all_pc_hw = self.coord_manager.get_bev_coords(pc[:, :2])[:, [1, 0]]
        # b * H * W + y * W + x
        pc_unravel = H * W * pc_batch_indices + W * all_pc_hw[:, 0] + all_pc_hw[:, 1]
        dim_size = B * H * W
        bev_ground = torch.torch_scatter.scatter(
            pc[:, 2],
            pc_unravel,
            dim=0,
            dim_size=dim_size,
            reduce='min',
            out=torch.ones([dim_size]).to(pc.device) * -invalid_height_val
        ).reshape(B, H, W)

        # min_batch_samples = torch.unique(pc_batch_indices, return_counts=True)[1].min().item()
        valid_sample_mask = bev_ground != -invalid_height_val
        out['bev_valid_pc_mask'] = valid_sample_mask

        noise_map = torch.gaussian_blur(valid_sample_mask.float(), kernel_size=(5, 5), sigma=(1, 1))
        noise_map_max = noise_map.max()
        noise_map /= noise_map_max
        noise_map = 1 - noise_map
        # from PIL import Image
        # Image.fromarray((noise_map[0].cpu().numpy() * 255).astype(np.uint8)).save('noise_map.png')
        # out['pc_noise_map'] = noise_map
        n_valid_samples = valid_sample_mask.reshape(B, -1).sum(dim=-1)
        min_valid_samples = min(n_valid_samples).item()
        if training:
            min_valid_samples = min(min_valid_samples, 7000)

        context_points, context_bhw = [], [] 
        context_features = []
        for bi in range(B):
            valid_sample_indices = torch.stack(torch.where(valid_sample_mask[bi]), dim=-1)
            sample_indices = np.random.choice(len(valid_sample_indices), min_valid_samples, replace=False)
            valid_sample_indices = valid_sample_indices[sample_indices]

            g_elev = bev_ground[bi, valid_sample_indices[:, 0], valid_sample_indices[:, 1]]
            # sample bev coordinates with uniform noise
            ego_xy = coord_manager.get_ego_coords(valid_sample_indices[:, [1, 0]])

            context_points.append(torch.cat([ego_xy, g_elev[:, None]], dim=-1))
            context_features.append(features[bi, :, valid_sample_indices[:, 0], valid_sample_indices[:, 1]].transpose(1, 0))

        context_points = torch.stack(context_points, dim=0)
        context_features = torch.stack(context_features, dim=0)
        out[f"{key}_contexts"] = context_points
        out[f"{key}_context_features"] = context_features

        if training:
            valid_sample_mask = ~label_ground_height.isnan()
            n_valid_samples = valid_sample_mask.reshape(B, -1).sum(dim=-1)
            min_valid_samples = min(n_valid_samples)
            min_valid_samples = min(min_valid_samples, torch.tensor(7000)).item()
        else:
            valid_sample_mask = torch.ones((B, H, W)).to(features.device)
            min_valid_samples = H * W

        target_points, target_bhw = [], []
        noises = []
        target_features = []
        for bi in range(B):
            valid_sample_indices = torch.stack(torch.where(valid_sample_mask[bi]), dim=-1)
            sample_indices = np.random.choice(len(valid_sample_indices), min_valid_samples, replace=False)
            valid_sample_indices = valid_sample_indices[sample_indices]

            g_elev = label_ground_height[bi, valid_sample_indices[:, 0], valid_sample_indices[:, 1]]
            ego_xy = coord_manager.get_ego_coords(valid_sample_indices[:, [1, 0]])

            target_points.append(torch.cat([ego_xy, g_elev[:, None]], dim=-1))
            noises.append(noise_map[bi, valid_sample_indices[:, 0], valid_sample_indices[:, 1]])
            target_features.append(features[bi, :, valid_sample_indices[:, 0], valid_sample_indices[:, 1]].transpose(1, 0))
        target_points = torch.stack(target_points, dim=0)
        target_features = torch.stack(target_features, dim=0)

        out[f"{key}_targets"] = target_points
        out[f"{key}_target_features"] = target_features
        out[f"{key}_target_noise"] = torch.stack(noises, dim=0)
        out['coord_manager'] = coord_manager
        return out

class Coordinate(torch.nn.Module):
    def __init__(self, min_bound, max_bound, grid_size):
        super(Coordinate, self).__init__()

        min_bound = torch.tensor(min_bound, dtype=torch.float32)
        max_bound = torch.tensor(max_bound, dtype=torch.float32)
        grid_size = torch.tensor(grid_size, dtype=torch.int64)

        self.min_bound = torch.nn.Parameter(min_bound, requires_grad=False)
        self.max_bound = torch.nn.Parameter(max_bound, requires_grad=False)
        self.grid_size = torch.nn.Parameter(grid_size, requires_grad=False)

        crop_range = self.max_bound - self.min_bound
        self.intervals = torch.nn.Parameter(crop_range / self.grid_size,
                                      requires_grad=False)


    def get_center(self, point_coords):
        '''
            Returns the voxel centers for each element in point_coords.
        '''
        return (point_coords.float() + 0.5) * self.intervals + self.min_bound

    def forward(self, points, points_batch_idx):
        point_coords = (points - self.min_bound) / self.intervals
        point_coords = torch.min(torch.clip(point_coords, 0),
                                 self.grid_size[None] - 1)

        point_coords = point_coords.int()

        # concatenate batch indices to coords
        point_coords = torch.cat((points_batch_idx[:, None], point_coords), dim=-1)

        # aggregate
        voxel_coordinates, inverse_map = torch.unique(
                point_coords,
                return_inverse=True, dim=0)


        return CoordinateMap(point_coords, voxel_coordinates, inverse_map)

    def get_ego_coords(self, pixel_wh: torch.Tensor):
        '''
        This is to convert the  bev coords in the feature map into point x, y.
        Please do not use this with polar coordiate voxelizer.
        '''
        pixel_xy = pixel_wh * self.intervals[:2]
        pixel_xy += self.min_bound[:2]
        # -0.5 ~ 0.5
        noise = (torch.rand_like(pixel_xy) * 0.6 - 0.3) * self.intervals[:2]
        point_xy = pixel_xy + noise
        return point_xy

    def get_bev_coords(self, point_xy: torch.Tensor):
        '''
        This is to convert the point x, y into bev coords in the feature map.
        Please do not use this with polar coordiate voxelizer.
        output: point_coords: (N, 2), W, H
        '''
        point_coords = (point_xy - self.min_bound[:2]) / self.intervals[:2]
        point_coords = torch.min(torch.clip(point_coords, 0),
                                    self.grid_size[None][:, :2] - 1)
        point_coords = torch.round(point_coords).long()
        return point_coords