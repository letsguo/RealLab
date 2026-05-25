"""Robot-centric occupancy map from mocap obstacle poses (matches Isaac obstacle_ground_truth)."""
import numpy as np
import torch
from typing import List, Optional, Sequence, Tuple


def build_obstacle_occupancy_map(
    robot_xy: np.ndarray,
    robot_yaw: float,
    obstacle_xy: np.ndarray,
    map_length_px: int,
    map_res_m_px: float,
    obstacle_half_extent_m: float,
) -> np.ndarray:
    """
    Build a robot-centric occupancy grid matching observations.obstacle_ground_truth.

    Args:
        robot_xy: (2,) robot position in world/map frame [m].
        robot_yaw: robot yaw in world frame [rad].
        obstacle_xy: (N, 2) obstacle centers in world frame [m].
        map_length_px: grid side length in cells (same as BEVMap.map_length_px).
        map_res_m_px: cell size in meters (same as BEVMap.map_res_m_px).
        obstacle_half_extent_m: half side of square obstacle footprint + padding [m].

    Returns:
        (map_length_px, map_length_px, 1) with 0 free and -1 occupied.
    """
    map_length_m = map_length_px * map_res_m_px
    half_extent = map_length_m / 2.0
    # Cell centers from -half_extent+res/2 to +half_extent-res/2
    offsets = (
        np.linspace(
            -half_extent + map_res_m_px / 2.0,
            half_extent - map_res_m_px / 2.0,
            map_length_px,
        )
    )
    grid_x, grid_y = np.meshgrid(offsets, offsets, indexing="xy")
    grid_local = np.stack([grid_x, grid_y], axis=-1)  # (H, W, 2)

    if obstacle_xy.size == 0:
        return np.zeros((map_length_px, map_length_px, 1), dtype=np.float32)

    c, s = np.cos(robot_yaw), np.sin(robot_yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    obs_local = (obstacle_xy - robot_xy) @ rot  # (N, 2)

    gx = grid_local[..., 0:1]  # (H, W, 1)
    gy = grid_local[..., 1:2]
    ox = obs_local[:, 0].reshape(1, 1, -1)
    oy = obs_local[:, 1].reshape(1, 1, -1)

    # --- CHANGED: world-aligned obstacle footprint --------------------------
    # The old test below compared body-frame deltas directly, which bakes a
    # square ALIGNED TO THE CAR'S BODY AXES into the grid. When the viz rotates
    # the robot-centric grid by +yaw into the world, that footprint comes out
    # tilted with the car and disagrees with the world-axis-aligned ground-truth
    # obstacles (the "weird looking mismatch").
    #
    # Old (body-aligned) test -- kept for reference, do not delete:
    # dx = np.abs(gx - ox)
    # dy = np.abs(gy - oy)
    # in_square = (dx <= obstacle_half_extent_m) & (dy <= obstacle_half_extent_m)
    #
    # New: rotate the cell->obstacle offset from the body frame back into the
    # world frame (R(robot_yaw)) before the axis-aligned box test, so the
    # footprint is a WORLD-aligned square regardless of car heading. (c, s are
    # cos/sin(robot_yaw) defined above.) At yaw=0 this reduces to the old test.
    rel_x = gx - ox  # body-frame offset, cell -> obstacle center
    rel_y = gy - oy
    rel_world_x = c * rel_x - s * rel_y  # R(robot_yaw) @ (rel_x, rel_y)
    rel_world_y = s * rel_x + c * rel_y
    in_square = (
        (np.abs(rel_world_x) <= obstacle_half_extent_m)
        & (np.abs(rel_world_y) <= obstacle_half_extent_m)
    )
    # ------------------------------------------------------------------------
    occupied = in_square.any(axis=-1)
    occupancy = np.where(occupied, -1.0, 0.0).astype(np.float32)
    return occupancy[..., np.newaxis]


def occupancy_to_policy_map(
    occupancy: np.ndarray,
    feature_dim: int,
) -> torch.Tensor:
    """
    Pack occupancy into the multi-channel map expected by BEVMap / car dynamics.

    Channel 0: height/occupancy (-1 occupied, 0 free).
    Channels 1-3: flat ground normal (0, 0, 1) when feature_dim >= 4.
    """
    h, w, _ = occupancy.shape
    if feature_dim == 1:
        return torch.from_numpy(occupancy.copy())
    out = np.zeros((h, w, feature_dim), dtype=np.float32)
    out[..., 0] = occupancy[..., 0]
    if feature_dim >= 4:
        out[..., 1] = 0.0
        out[..., 2] = 0.0
        out[..., 3] = 1.0
    elif feature_dim > 1:
        out[..., 1 : feature_dim] = 0.0
    return torch.from_numpy(out)


def collect_obstacle_positions(
    poses: dict,
    tracked_objects: Sequence[str],
) -> Optional[np.ndarray]:
    """Stack latest (x, y) for tracked objects; return None if any pose is missing."""
    if not tracked_objects:
        return np.zeros((0, 2), dtype=np.float64)
    xy = []
    for name in tracked_objects:
        if name not in poses or poses[name] is None:
            return None
        xy.append(poses[name])
    return np.asarray(xy, dtype=np.float64)
