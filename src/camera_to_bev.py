#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import torch
import torch.nn.functional as F
import time
from PIL import Image as PILImage
import os


from pytorch3d import transforms as pT
# import ipdb 

class RGBBEVHistory():
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, rgb_size, rgb_K, rgb_Rt, depth_K, depth_Rt, meter_to_pixel=94.21, bevmap_size=(48, 48)):
        if not hasattr(self, 'initialized'):
            self.initialized = True

            self.rgb_H, self.rgb_W = rgb_size
            self.rgb_K = rgb_K
            self.rgb_Rt = rgb_Rt
            self.rgb_K_inv = torch.linalg.inv(rgb_K)
            self.rgb_Rt_inv = torch.linalg.inv(rgb_Rt)

            self.depth_K = depth_K
            self.depth_Rt = depth_Rt
            self.depth_K_inv = torch.linalg.inv(depth_K)
            self.depth_Rt_inv = torch.linalg.inv(depth_Rt)

            self.depth_to_rgb = torch.matmul(self.rgb_Rt_inv, self.depth_Rt)

            self.meter_to_pixel = meter_to_pixel
            self._2d_slicer = torch.tensor([0, 1, 3])

            self.bevmap_size = bevmap_size
            self.bevmap = None
            self.prev_pose = None

    def _2d(self, pose3d: torch.Tensor, slicer: torch.Tensor):
        return pose3d[:, slicer][:, :, slicer]

    def update_bev(self, rgb, depth, pose):

        if self.bevmap is None:
            shifted_prev = torch.zeros((rgb.shape[0], 3, *self.bevmap_size), device=rgb.device)
        else:
            prev_bevmap = self.bevmap
            # Update the BEV map with the new data
            transform = torch.matmul(torch.inverse(self._2d(pose, self._2d_slicer)),
                                     self._2d(self.prev_pose, self._2d_slicer))[..., :2, :]
            transform[:, 0, 2] = -transform[:, 0, 2] * self.meter_to_pixel / self.bevmap_size[0] * 2
            transform[:, 1, 2] = -transform[:, 1, 2] * self.meter_to_pixel / self.bevmap_size[1] * 2
            shifted_prev = self.warp2(prev_bevmap, transform)
            ### debug
            # save_dir = os.path.join(os.getcwd(), "bev_debug_images")
            # save_path = os.path.join(save_dir, f"rgb_no_warp_map_{time.time()}.png")
            # PILImage.fromarray((shifted_prev[0].reshape(3, *self.bevmap_size).permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8)).save(save_path)
        updated_bev = self.get_projected_rgb(rgb, depth, shifted_prev)
        self.bevmap = updated_bev
        ############### debug #################
        # from PIL import Image
        # vis = updated_bev[0].permute(1, 2, 0).cpu().numpy()
        # save_dir = os.path.join(os.getcwd(), "bev_debug_images")
        # save_path = os.path.join(save_dir, f"rgb_bevmap_{time.time()}.png")
        # PILImage.fromarray((vis * 255.).astype(np.uint8)).save(save_path)

        # vis = rgb[0].cpu().numpy()
        # save_path = os.path.join(save_dir, f"rgb_cam_{time.time()}.png")
        # PILImage.fromarray(vis.astype(np.uint8)).save(save_path)
        #########################################
        self.prev_pose = pose
        return updated_bev.permute(0, 2, 3, 1)[:, :48, :48, :]

    def get_projected_rgb(self, camera_rgb, camera_depth, shifted_bev):
        aligned_depth = self.align_depth_to_rgb(camera_depth)
        B, H, W = aligned_depth.shape
        # debug
        # from PIL import Image
        # Image.fromarray((aligned_depth[0] / aligned_depth.max() * 255).cpu().numpy().astype(np.uint8)).save(f'depths/depth_map_{time.time()}.png')
        # B x H x W x 3
        xyz_cam, valid = self.backproject(aligned_depth, self.rgb_K_inv)
        xyz_cam *= valid.unsqueeze(-1)
        # homogeneous
        xyz_cam = torch.cat([xyz_cam, xyz_cam.new_ones(xyz_cam.shape[:3] + (1,))], dim=-1)
        # Transform the coordinates from the camera frame to the map frame
        # b x h x w x 4
        # Note that h, w are image dimensions, not map dimensions.
        output_xyz = torch.einsum('bhwi,oi -> bhwo', xyz_cam, self.rgb_Rt.T.to(xyz_cam.device))
        output_xy = output_xyz[:, :, :, :2].float()
        bev_xy = self.ego_to_bevmap(output_xy)
        bev_xy_1d = bev_xy[:, :, :, 1] * self.bevmap_size[1] + bev_xy[:, :, :, 0]
        camera_rgb = camera_rgb.float()
        camera_rgb = camera_rgb.permute(0, 3, 1, 2).reshape(B, -1, H * W) / 255.
        bev_xy_1d = bev_xy_1d.reshape(B, 1, H * W)
        shifted_bev = shifted_bev.reshape(B, 3, self.bevmap_size[0] * self.bevmap_size[1]).float()

        # SANGHUN's CODE
        # projected_rgb = torch.scatter_reduce(input=torch.zeros_like(shifted_bev),
        #                                      dim=-1,
        #                                      index=bev_xy_1d.repeat(1, 3, 1),
        #                                      src=camera_rgb,
        #                                      reduce='mean')
        # projected_mask = torch.scatter_reduce(input=torch.zeros_like(shifted_bev),
        #                                       dim=-1,
        #                                       index=bev_xy_1d.repeat(1, 3, 1),
        #                                       src=torch.ones((B, 3, H*W)).to(camera_rgb.device).float(),
        #                                       reduce='sum')
        
        # CHAT REPLACEMENT - GABE
        B, C, HW = camera_rgb.shape

        # Step 1: Create the zero tensors
        output_sum = torch.zeros_like(shifted_bev)  # for summing values
        output_count = torch.zeros_like(shifted_bev)  # for counting

        # Step 2: scatter_add for the sum
        output_sum.scatter_add_(dim=-1, index=bev_xy_1d.repeat(1, C, 1), src=camera_rgb)

        # Step 3: scatter_add for the count
        ones = torch.ones((B, C, HW), device=camera_rgb.device, dtype=camera_rgb.dtype)
        output_count.scatter_add_(dim=-1, index=bev_xy_1d.repeat(1, C, 1), src=ones)

        # Step 4: compute mean manually (avoid division by zero)
        projected_rgb = output_sum / (output_count + 1e-6)  # small epsilon to avoid div by zero
        projected_mask = output_count  # sum of ones = count

        # debug
        # PILImage.fromarray((projected_rgb[0].reshape(3, *self.bevmap_size).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(f'bev_debug_images/rgb_no_warp_map_{time.time()}.png')
        # PILImage.fromarray((projected_mask[0].reshape(3, *self.bevmap_size).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(f'bev_debug_images/rgb_no_warp_mask_{time.time()}.png')
        shifted_bev[projected_mask > 0] = projected_rgb[projected_mask > 0]
        return shifted_bev.reshape(B, 3, *self.bevmap_size)

    def ego_to_bevmap(self, output_xy):
        # Convert the pixel coordinates to BEV map coordinates
        # b x h x w x 2
        output_xy = output_xy * self.meter_to_pixel
        output_xy = output_xy + torch.tensor([self.bevmap_size[0] // 2, self.bevmap_size[1] // 2], device=output_xy.device)
        # b x h x w x 2
        in_range = (output_xy[:, :, :, 0] < self.bevmap_size[1]) & \
                   (output_xy[:, :, :, 1] < self.bevmap_size[0]) & \
                   (output_xy[:, :, :, 0] >= 0) & \
                   (output_xy[:, :, :, 1] >= 0)
        output_xy = output_xy * in_range.unsqueeze(-1)
        output_xy = output_xy.long()
        return output_xy
        
    def compute_pixel_rays(self, H, W, K_inv):
        ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                                torch.arange(W, dtype=torch.float32))
        xy = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1)  # h x w x 2, pixel coordinates
        xyp = torch.matmul(xy, K_inv.T)  # h x w x 3
        return xyp

    def backproject(self, aligned_depth, rgb_K_inv):
        H, W = aligned_depth.shape[-2:]
        valid = (aligned_depth > 0.1)
        pixel_rays = self.compute_pixel_rays(H, W, rgb_K_inv)
        pixel_rays = pixel_rays.to(aligned_depth.device)

        # Coordinates of each pixel in the camera frame
        xyz_cam = aligned_depth[..., None] * pixel_rays.float()
        return xyz_cam, valid

    def warp2(self, images, affine_matrix):
        """
        Apply affine transformation to a batch of images.

        Args:
            images (Tensor): Batch of images (N, C, H, W)
            affine matrix (Tensor): Batch of affine transformation matrices (N, 2, 3)

        Returns:
            Transformed images (N, C, H, W)
        """
        # Create normalized grid
        grid = F.affine_grid(affine_matrix, size=images.size(), align_corners=False)
        # Sample image using grid
        warped = F.grid_sample(images.float(), grid, align_corners=False, padding_mode='border')
        return warped

    def align_depth_to_rgb(self, depth):
        B, H, W, C = depth.shape
        # Get the pixel rays in the camera frame
        pixel_rays = self.compute_pixel_rays(H, W, self.depth_K_inv)
        xyz_depth = depth * pixel_rays.to(depth.device)
        # make homogeneous
        xyz_depth = torch.cat([xyz_depth, torch.ones(xyz_depth.shape[:-1] + (1,)).to(depth.device)], dim=-1)
        # Transform the pixel rays to the RGB camera frame
        xyz_rgb = torch.matmul(xyz_depth, self.depth_to_rgb.T.to(depth.device))[..., :3]

        # Get the pixel coordinates in the RGB camera frame
        pixel_coords = torch.matmul(xyz_rgb, self.rgb_K.T.to(depth.device))
        # B x H x W x 3
        pixel_z = pixel_coords[..., 2]
        pixel_coords = pixel_coords[..., :2] / pixel_z.unsqueeze(-1)
        pixel_coords = pixel_coords.long()

        valid = (pixel_z > 0) & (pixel_coords[..., 0] < self.rgb_W) & (pixel_coords[..., 1] < self.rgb_H) & \
                (pixel_coords[..., 0] >= 0) & (pixel_coords[..., 1] >= 0)
        pixel_coords = pixel_coords * valid.unsqueeze(-1)
        pixel_z = pixel_z * valid

        aligned_depth = torch.zeros((B, self.rgb_H, self.rgb_W)).to(depth.device)
        # for b in range(B):
        BB = torch.arange(B)
        aligned_depth[BB, pixel_coords[BB, ..., 1], pixel_coords[BB, ..., 0]] = pixel_z[BB, ...]
        aligned_depth[aligned_depth.isnan()] = 0.0
        return aligned_depth.float()

    @staticmethod
    def make_pose(t, quat):
        # Convert quaternion to rotation matrix
        rot = pT.quaternion_to_matrix(quat)
        # Convert to homogeneous coordinates
        pose = torch.zeros((1, 4, 4), device=t.device)
        pose[:, :3, :3] = rot
        pose[:, :3, 3] = t
        pose[:, 3, 3] = 1.0
        return pose

    def reset(self):
        self.bevmap = None
        self.prev_pose = None

# rgb_mapper = RGBBEVHistory(
#     # rgb_size=(40, 60),
#     rgb_size=(720, 1280),
#     rgb_K=torch.tensor([[635.9487, 0.0, 634.8170],
#                        [0.0, 635.4133, 366.7346],
#                        [0.0, 0.0, 1.0]]),

#     # rgb_K=torch.tensor([[635.9487 * 60 / 1280, 0.0, 634.8170 * 60 / 1280],
#     #                    [0.0, 635.4133 * 40 / 720, 366.7346 * 40 / 720],
#     #                    [0.0, 0.0, 1.0]]),
#     # from real
#     # from real
#     # rgb_Rt=torch.tensor([[5.99994e-06,  1.99998e-03,  9.99998e-01, 0.023],
#     #                     [-9.99982e-01,  5.99994e-03, -5.99994e-06, -0.042],
#     #                     [-5.99994e-03, -9.99980e-01,  1.99998e-03, 0.068],
#     #                     [0.0, 0.0, 0.0, 1.0]]),
#     # rgb_Rt = torch.tensor([[1.0, 0.0, 0.0, 0.023],
#     #                     [0.0, 1.0, 0.0, -0.042],
#     #                     [0.0, 0.0, 1.0, 0.068],
#     #                     [0.0, 0.0, 0.0, 1.0]]),
#     rgb_Rt = torch.tensor([[0.0, 1.0, 0.0, 0.0],
#                            [0.0, 0.0, 1.0, 0.0],
#                            [1.0, 0.0, 0.0, 0.0],
#                            [0.0, 0.0, 0.0, 1.0]]),
#     depth_K=torch.tensor([[424.0352478027344, 0.0, 426.2065734863281],
#                         [0.0, 424.0352478027344, 239.09210205078125],
#                         [0.0, 0.0, 1.0]]),
#     # depth_K=torch.tensor([[424.0352478027344 * 60 / 828, 0.0, 426.2065734863281 * 60 / 828],
#     #                     [0.0, 424.0352478027344 * 40 / 480, 239.09210205078125 * 40 / 480],
#     #                     [0.0, 0.0, 1.0]]),
#     # from real
#     # depth_Rt=torch.tensor([[0.0, 0.0, 1.0, 0.023],
#     #                     [-1.0, 0.0, 0.0, 0.018],
#     #                     [0.0, -1.0, 0.0, 0.068],
#     #                     [0.0, 0.0, 0.0, 1.0]]),
#     # depth_Rt=torch.tensor([[1.0, 0.0, 0.0, 0.023],
#     #                     [0.0, 1.0, 0.0, 0.018],
#     #                     [0.0, 0.0, 1.0, 0.068],
#     #                     [0.0, 0.0, 0.0, 1.0]]),

#     depth_Rt = torch.tensor([[0.0, 1.0, 0.0, 0.0],
#                              [0.0, 0.0, 1.0, 0.0],
#                              [1.0, 0.0, 0.0, 0.0],
#                              [0.0, 0.0, 0.0, 1.0]]),
#     meter_to_pixel=94.21,
#     bevmap_size=(1000, 1000)
# )