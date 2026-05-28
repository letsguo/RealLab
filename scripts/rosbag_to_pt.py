#!/usr/bin/env python3

# usage:
# python3 src/DBF_RealLab/scripts/rosbag_to_pt.py src/DBF_RealLab/dbf_expert_demonstrations/medium_u.bag -o src/DBF_RealLab/dbf_pt/medium_u.pt
"""Convert expert rosbags to a [N, 2] tensor of planar (x, y) positions."""
import argparse
import os

import numpy as np
import rosbag
import torch


def read_xy_from_bag(bag_path, odom_topic="/mocap/local_position/odom"):
    """Return the planar (x, y) trajectory recorded on ``odom_topic`` as a
    ``[N, 2]`` float64 numpy array, without touching disk. Used by both the
    .pt-writing CLI here and the offline overview plotter
    (scripts/plot_overview.py) so neither has to redo bag-reading boilerplate.
    """
    xy = []
    with rosbag.Bag(bag_path, "r") as bag:
        for _topic, msg, _t in bag.read_messages(topics=[odom_topic]):
            pos = msg.pose.pose.position
            xy.append([pos.x, pos.y])
    return np.asarray(xy, dtype=np.float64)


def bag_to_pt(bag_path, output_pt, odom_topic="/mocap/local_position/odom"):
    xy = read_xy_from_bag(bag_path, odom_topic=odom_topic)
    trajectory = torch.tensor(xy, dtype=torch.float32)  # [N, 2]

    os.makedirs(os.path.dirname(os.path.abspath(output_pt)) or ".", exist_ok=True)
    torch.save(trajectory, output_pt)
    print(f"Saved trajectory {tuple(trajectory.shape)} to {output_pt}")
    return output_pt


def main():
    parser = argparse.ArgumentParser(description="Convert expert rosbag to xy .pt")
    parser.add_argument("bag_path", help="Input .bag file")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output .pt path (default: same dir/name as bag)",
    )
    parser.add_argument(
        "--odom-topic",
        default="/mocap/local_position/odom",
        help="Odometry topic in the bag",
    )
    args = parser.parse_args()
    bag_path = os.path.abspath(args.bag_path)
    if args.output:
        output_pt = os.path.abspath(args.output)
    else:
        base, _ = os.path.splitext(bag_path)
        output_pt = base + ".pt"
    bag_to_pt(bag_path, output_pt, odom_topic=args.odom_topic)


if __name__ == "__main__":
    main()
