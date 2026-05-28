#!/usr/bin/env python3
"""Offline per-run metrics for a recorded bag (produced by expert_recorder_odom).

Computes four numbers + a collision-event list and writes them to
``<bag>_metrics.json`` next to the input bag. All metrics derive from
topics that the recorder already captures (``/mocap/local_position/odom``
and ``/mocap/<obstacle>/pose``); no controller changes needed.

# usage:
#   python3 scripts/compute_run_metrics.py dbf_gail_run01.bag
#   python3 scripts/compute_run_metrics.py dbf_gail_run01.bag \\
#       --goal -1.5 -8.0 --goal-tol 0.5 \\
#       --tracked big_obstacle_1 big_obstacle_2 big_obstacle_3 \\
#       --obstacle-side 0.35 --obstacle-pad 0.1 \\
#       --variant dbf_gail

Schema of <bag>_metrics.json:
{
  "bag":           "...basename...",
  "variant":       "dbf_gail" | null,
  "n_odom":        int,
  "duration_s":    float,
  "success":       bool,
  "min_dist_to_goal_m": float,
  "collisions":    int,
  "collision_events":  [ {"t": float, "obstacle": "big_obstacle_1"}, ... ],
  "avg_velocity_m_s":  float,
  "avg_yaw_rate_rad_s": float,
  "goal":          [x, y],
  "goal_tol_m":    float,
  "obstacles":     { "big_obstacle_1": [x, y], ... },
  "obstacle_half_extent_m": float,
}
"""
import argparse
import json
import os
import sys

import numpy as np
import rosbag
from tf.transformations import euler_from_quaternion


def _odom_to_arrays(bag_path, odom_topic):
    """Return numpy arrays (t, x, y, vx, vy, yaw) from the odom topic."""
    ts, xs, ys, vxs, vys, yaws = [], [], [], [], [], []
    with rosbag.Bag(bag_path, "r") as bag:
        for _topic, msg, t in bag.read_messages(topics=[odom_topic]):
            ts.append(t.to_sec())
            xs.append(msg.pose.pose.position.x)
            ys.append(msg.pose.pose.position.y)
            vxs.append(msg.twist.twist.linear.x)
            vys.append(msg.twist.twist.linear.y)
            q = msg.pose.pose.orientation
            yaws.append(euler_from_quaternion((q.x, q.y, q.z, q.w))[2])
    if not ts:
        raise RuntimeError(f"No messages on {odom_topic!r} in {bag_path}")
    return tuple(np.asarray(a, dtype=np.float64) for a in (ts, xs, ys, vxs, vys, yaws))


def _last_obstacle_poses(bag_path, names):
    """Return {name: (x, y)} from the last /mocap/<name>/pose message per name.
    Obstacles are static — last value is as representative as first.
    """
    topics = {f"/mocap/{name}/pose": name for name in names}
    found = {}
    with rosbag.Bag(bag_path, "r") as bag:
        for topic, msg, _t in bag.read_messages(topics=list(topics.keys())):
            p = msg.pose.position
            found[topics[topic]] = (float(p.x), float(p.y))
    return found


def _compute_success(xs, ys, goal, tol):
    """Success = ever within `tol` meters of `goal`."""
    dists = np.hypot(xs - goal[0], ys - goal[1])
    return bool(dists.min() <= tol), float(dists.min())


def _compute_collisions(ts, xs, ys, obstacles, half_extent):
    """Per-obstacle rising-edge count: each entry from outside->inside counts as
    one collision event, with the entry timestamp. Re-entering after exiting
    counts as a new event. Returns (total_count, [{t, obstacle}]).
    """
    events = []
    if not obstacles:
        return 0, events
    for name, (ox, oy) in obstacles.items():
        inside = (np.abs(xs - ox) <= half_extent) & (np.abs(ys - oy) <= half_extent)
        # Rising edge = currently inside AND wasn't inside last sample.
        prev = np.concatenate([[False], inside[:-1]])
        rising = inside & (~prev)
        for idx in np.flatnonzero(rising):
            events.append({"t": float(ts[idx] - ts[0]), "obstacle": name})
    events.sort(key=lambda e: e["t"])
    return len(events), events


def _compute_avg_velocity(vxs, vys):
    return float(np.mean(np.hypot(vxs, vys)))


def _compute_avg_yaw_rate(ts, yaws):
    """Mean |Δyaw|/Δt via finite diff. Mocap leaves twist.angular at zero,
    so we re-derive here (same trick obtain_state already uses)."""
    if len(ts) < 2:
        return 0.0
    dt = np.diff(ts)
    dyaw = np.diff(yaws)
    # Wrap to (-pi, pi] so the 0/2π seam doesn't blow up the magnitude.
    dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
    valid = dt > 1e-6
    if not valid.any():
        return 0.0
    return float(np.mean(np.abs(dyaw[valid] / dt[valid])))


def compute_metrics(bag_path, goal, goal_tol, tracked, obstacle_side, obstacle_pad,
                    odom_topic="/mocap/local_position/odom"):
    """Build the full metrics dict for one bag."""
    ts, xs, ys, vxs, vys, yaws = _odom_to_arrays(bag_path, odom_topic)
    obstacles = _last_obstacle_poses(bag_path, tracked)
    half_extent = obstacle_side / 2.0 + obstacle_pad

    success, min_dist = _compute_success(xs, ys, goal, goal_tol)
    n_collisions, collision_events = _compute_collisions(
        ts, xs, ys, obstacles, half_extent
    )

    return {
        "bag": os.path.basename(bag_path),
        "n_odom": int(len(ts)),
        "duration_s": float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0,
        "success": success,
        "min_dist_to_goal_m": min_dist,
        "collisions": int(n_collisions),
        "collision_events": collision_events,
        "avg_velocity_m_s": _compute_avg_velocity(vxs, vys),
        "avg_yaw_rate_rad_s": _compute_avg_yaw_rate(ts, yaws),
        "goal": [float(goal[0]), float(goal[1])],
        "goal_tol_m": float(goal_tol),
        "obstacles": {name: list(xy) for name, xy in obstacles.items()},
        "obstacle_half_extent_m": float(half_extent),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("bag", help="Input run .bag.")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output .json path (default: <bag>_metrics.json next to the bag).",
    )
    parser.add_argument(
        "--variant", default=None,
        help="Optional variant tag (e.g., dbf_gail) recorded in the json so "
             "scripts/aggregate_metrics.py can group without filename parsing.",
    )
    parser.add_argument(
        "--goal", nargs=2, type=float, default=[-1.5, -8.0],
        metavar=("X", "Y"),
        help="Goal coordinates (m). Default: -1.5 -8.0 (lab convention).",
    )
    parser.add_argument(
        "--goal-tol", type=float, default=0.5,
        help="Success tolerance: robot must enter this radius around the goal.",
    )
    parser.add_argument(
        "--tracked", nargs="+",
        default=["big_obstacle_1", "big_obstacle_2", "big_obstacle_3"],
        help="Obstacle mocap names to read from the bag.",
    )
    parser.add_argument(
        "--obstacle-side", type=float, default=0.35,
        help="Obstacle side length (m). Default matches launch file.",
    )
    parser.add_argument(
        "--obstacle-pad", type=float, default=0.1,
        help="Safety padding added around each obstacle box for collision detection.",
    )
    parser.add_argument(
        "--odom-topic", default="/mocap/local_position/odom",
        help="Odom topic to read from.",
    )
    args = parser.parse_args()

    bag_path = os.path.abspath(args.bag)
    if not os.path.exists(bag_path):
        sys.exit(f"ERROR: bag not found: {bag_path}")

    metrics = compute_metrics(
        bag_path,
        goal=tuple(args.goal), goal_tol=args.goal_tol,
        tracked=args.tracked,
        obstacle_side=args.obstacle_side, obstacle_pad=args.obstacle_pad,
        odom_topic=args.odom_topic,
    )
    if args.variant is not None:
        metrics["variant"] = args.variant

    out_path = args.output or (os.path.splitext(bag_path)[0] + "_metrics.json")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"{metrics['bag']}: success={metrics['success']} "
          f"collisions={metrics['collisions']} "
          f"v={metrics['avg_velocity_m_s']:.2f} m/s "
          f"yaw_rate={metrics['avg_yaw_rate_rad_s']:.2f} rad/s "
          f"duration={metrics['duration_s']:.1f}s "
          f"min_dist_to_goal={metrics['min_dist_to_goal_m']:.2f} m "
          f"-> {out_path}")


if __name__ == "__main__":
    main()
