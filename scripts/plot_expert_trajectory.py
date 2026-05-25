#!/usr/bin/env python3

# usage:
# python3 src/DBF_RealLab/scripts/plot_expert_trajectory.py src/DBF_RealLab/dbf_pt/large_u.pt -o src/DBF_RealLab/dbf_expert_traj
"""Plot 2D expert trajectory from a [N, 2] xy .pt file."""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUT = os.path.join(PKG_ROOT, "dbf_expert_traj")


def load_xy_from_pt(pt_path):
    data = torch.load(pt_path, map_location="cpu")
    if isinstance(data, dict) and "states" in data:
        # legacy: list of (timestamp, tensor with x,y,...)
        states = data["states"]
        xy = np.array([[float(s[1][0]), float(s[1][1])] for s in states], dtype=np.float64)
    else:
        xy = data.numpy() if isinstance(data, torch.Tensor) else np.asarray(data)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"Expected [N, 2] xy tensor, got shape {xy.shape}")
        xy = xy.astype(np.float64)
    if len(xy) == 0:
        raise ValueError("No trajectory samples found")
    return xy


def plot_trajectory(xy, title, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    idx = np.arange(len(xy), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=idx, cmap="viridis", s=8, zorder=2)
    ax.plot(xy[:, 0], xy[:, 1], "k-", alpha=0.4, lw=1, zorder=1)
    ax.scatter(xy[0, 0], xy[0, 1], c="green", s=80, label="start", zorder=3)
    ax.scatter(xy[-1, 0], xy[-1, 1], c="red", s=80, label="end", zorder=3)
    plt.colorbar(sc, ax=ax, label="sample index")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot 2D expert trajectory from .pt")
    parser.add_argument("input", help="Path to .pt file ([N, 2] xy tensor)")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=DEFAULT_OUT,
        help=f"Directory for PNG output (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()
    inp = os.path.abspath(args.input)
    if not inp.endswith(".pt"):
        print("Input must be a .pt file", file=sys.stderr)
        sys.exit(1)

    stem = os.path.splitext(os.path.basename(inp))[0]
    out_path = os.path.join(os.path.abspath(args.output_dir), f"{stem}_trajectory.png")
    xy = load_xy_from_pt(inp)
    plot_trajectory(xy, title=stem, out_path=out_path)


if __name__ == "__main__":
    main()
