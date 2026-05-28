#!/usr/bin/env python3
"""
Convert raw observation tensors into paired (state, next_state) examples and save them.

Usage:
  # Process all .pt files in a source folder
  python scripts/convert_observations.py

  # Combine two-or-more trajectory files into one tensor
  python scripts/convert_observations.py --combine <file1.pt> <file2.pt> [<file3.pt> ...] --dst <output_dir>

  # Combine and plot (overlaid + per-file individual plots)
  python scripts/convert_observations.py --combine <f1.pt> <f2.pt> <f3.pt> ... --dst <output_dir> --plot

This script searches for .pt files in
  source/wheeledlab_tasks/wheeledlab_tasks/navigation/config/agents/mushr/processed_data
and for each tensor file it expects a torch.Tensor of shape [T, N, D]. It will convert
to a tensor of shape [T-1 * N, 2, D] where the second dimension holds (state, next_state).

Output files are written to
  source/wheeledlab_tasks/wheeledlab_tasks/navigation/config/agents/mushr/sim_data
with name `expert_obs_masked_{num_demos}.pt` where num_demos == N.

When using --combine, two or more trajectory files are loaded, each converted to
transitions, and concatenated into a single output tensor of shape [sum(M_i), 2, D].

When using --plot with --combine, generates (a) one overlaid XY plot of all the
trajectories color-coded by file, and (b) an individual XY plot per input file.

blue, red if inside object.
instaed of blue 
Terminal value is based on vlaue function.
green optimal

for i in {1..5}; do
    python3 src/DBF_RealLab/scripts/rosbag_to_pt.py \
        src/DBF_RealLab/dbf_expert_demonstrations/figure8_${i}.bag \
        -o src/DBF_RealLab/dbf_pt/figure8_${i}.pt
done

python3 src/DBF_RealLab/scripts/convert_observations.py \
    --combine src/DBF_RealLab/dbf_pt/figure8_{1..5}.pt \
    --combine-output figure8_combined.pt \
    --dst /root/catkin_ws/src/DBF_RealLab/dbf_sim_data \
    --plot
"""
import os
import glob
import argparse
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
# import ipdb

def load_and_extract_tensor(src_path: Path):
    """Load a file and extract the tensor, handling both plain tensors and dicts."""
    obj = torch.load(src_path)
    # accept plain tensor or dict containing tensor
    if isinstance(obj, dict):
        # try common keys
        for k in ("observations", "observations_", "data", "demos", "demonstrations", "demo"):
            if k in obj:
                demonstrations = obj[k]
                break
        else:
            # pick first tensor-like value
            vals = [v for v in obj.values() if isinstance(v, torch.Tensor)]
            if not vals:
                raise ValueError(f"No tensor found in dict stored at {src_path}")
            demonstrations = vals[0]
    else:
        demonstrations = obj

    if not isinstance(demonstrations, torch.Tensor):
        raise ValueError(f"Loaded object from {src_path} is not a torch.Tensor (type={type(demonstrations)})")

    # Accept 2D tensors [T, D] (N=1) or 3D tensors [T, N, D]
    if demonstrations.dim() == 2:
        # assume shape [T, D] -> add demo dimension N=1
        demonstrations = demonstrations.unsqueeze(1)
    if demonstrations.dim() < 3:
        raise ValueError(f"Expected demonstrations tensor with >=2 dims [T, D] or [T, N, D], got shape {demonstrations.shape}")

    return demonstrations


def convert_trajectory_to_transitions(demonstrations: torch.Tensor):
    """Convert a trajectory tensor [T, N, D] to (state, next_state) pairs [T-1*N, 2, D]."""
    state = demonstrations[:-1, ...]
    next_state = demonstrations[1:, ...]
    state = state.reshape(state.shape[0] * state.shape[1], state.shape[2]).unsqueeze(1)
    next_state = next_state.reshape(state.shape[0] * state.shape[1], state.shape[2]).unsqueeze(1)
    transitions = torch.cat((state, next_state), dim=1)
    return transitions


def plot_trajectories(demonstrations_list, labels, dst_dir: Path,
                      output_name: str = "combined_trajectory_plot.png"):
    """
    Overlay N trajectories on a single XY plot, color-coded by source file.

    Args:
        demonstrations_list: list of trajectory tensors, each [T, N, D] or [T, D]
        labels: list of labels (one per trajectory, e.g. file names)
        dst_dir: Directory to save the plot
        output_name: Output filename for the plot
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    n = len(demonstrations_list)
    # tab10 gives up to 10 distinct colors; fall back to a continuous map beyond that
    cmap = plt.get_cmap("tab10") if n <= 10 else plt.get_cmap("viridis")

    total_steps = 0
    for i, (demo, label) in enumerate(zip(demonstrations_list, labels)):
        # [T, N, D] -> average demos to [T, D]; [T, D] stays as-is
        traj = demo.mean(dim=1).numpy() if demo.dim() == 3 else demo.numpy()
        color = cmap(i) if n <= 10 else cmap(i / max(1, n - 1))
        ax.plot(traj[:, 0], traj[:, 1], linewidth=2, alpha=0.7, color=color,
                label=f"{label} ({len(traj)} steps)")
        # circle = start, star = end (same color as the line, black edge)
        ax.plot(traj[0, 0], traj[0, 1], 'o', color=color, markersize=9, markeredgecolor='black')
        ax.plot(traj[-1, 0], traj[-1, 1], '*', color=color, markersize=15, markeredgecolor='black')
        total_steps += len(traj)

    ax.set_xlabel('Dimension 0 (X)', fontsize=12)
    ax.set_ylabel('Dimension 1 (Y)', fontsize=12)
    ax.set_title(f'Combined Trajectories — {n} files, {total_steps} steps total '
                 f'(o = start, * = end)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    dst_dir.mkdir(parents=True, exist_ok=True)
    plot_path = dst_dir / output_name
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    print(f"Saved combined trajectory plot to {plot_path}")
    plt.close()


def plot_individual_trajectory(demonstrations: torch.Tensor, dst_dir: Path, output_name: str, title_prefix: str = ""):
    """
    Plot an individual trajectory as an XY plot.

    Args:
        demonstrations: Tensor of shape [T, N, D] with trajectory data
        dst_dir: Directory to save the plot
        output_name: Output filename for the plot
        title_prefix: Prefix for the plot title
    """
    # Handle case where demonstrations might have N=1 (already in [T, D] after unsqueeze)
    if demonstrations.dim() == 3:
        # Average across all demos if multiple: [T, N, D] -> [T, D]
        traj_2d = demonstrations.mean(dim=1).numpy()
    else:
        traj_2d = demonstrations.numpy()

    # Create XY plot (dimension 0 vs dimension 1)
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the trajectory path
    ax.plot(traj_2d[:, 0], traj_2d[:, 1], linewidth=2, alpha=0.7, color='blue', label='Trajectory')

    # Mark start and end points
    ax.plot(traj_2d[0, 0], traj_2d[0, 1], 'go', markersize=10, label='Start')
    ax.plot(traj_2d[-1, 0], traj_2d[-1, 1], 'r*', markersize=15, label='End')

    # Add arrows to show direction
    num_arrows = min(20, len(traj_2d) - 1)
    arrow_indices = np.linspace(0, len(traj_2d) - 2, num_arrows, dtype=int)
    for idx in arrow_indices:
        dx = traj_2d[idx + 1, 0] - traj_2d[idx, 0]
        dy = traj_2d[idx + 1, 1] - traj_2d[idx, 1]
        ax.arrow(traj_2d[idx, 0], traj_2d[idx, 1], dx, dy,
                head_width=0.02, head_length=0.02, fc='gray', ec='gray', alpha=0.5)

    ax.set_xlabel('Dimension 0 (X)', fontsize=12)
    ax.set_ylabel('Dimension 1 (Y)', fontsize=12)
    ax.set_title(f'{title_prefix}(XY Plot)\nTotal steps: {len(traj_2d)}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plot_path = dst_dir / output_name
    dst_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    print(f"Saved individual trajectory plot to {plot_path}")
    plt.close()


def process_file(src_path: Path, dst_dir: Path, verbose: bool = True):
    """Process a single trajectory file and save converted transitions."""
    demonstrations = load_and_extract_tensor(src_path)
    input_shape = demonstrations.shape
    num_demos = demonstrations.shape[1]

    demonstrations_out = convert_trajectory_to_transitions(demonstrations)

    dst_path = dst_dir / f"expert_obs_masked_{num_demos}_processed.pt"
    dst_dir.mkdir(parents=True, exist_ok=True)
    torch.save(demonstrations_out, dst_path)
    if verbose:
        print(f"Converted {src_path.name} -> {dst_path}  (input_shape={input_shape}, output_shape={demonstrations_out.shape})")


def process_and_combine_files(src_paths, dst_dir: Path, output_name: str = None,
                              plot: bool = False, verbose: bool = True):
    """
    Load N (>=2) trajectory files, convert each to (state, next_state) transitions,
    and combine them into a single tensor [sum(M_i), 2, D].

    Args:
        src_paths: list of Paths (or path strings) to trajectory files
        dst_dir: Destination directory for combined output
        output_name: Custom output filename (default: expert_obs_masked_combined_processed.pt)
        plot: If True, save an overlaid plot of all trajectories plus one plot per input file
        verbose: Whether to print processing info

    Returns:
        The combined transitions tensor [M, 2, D] where M = sum of per-file transitions
    """
    src_paths = [Path(p) for p in src_paths]
    if len(src_paths) < 2:
        raise ValueError(f"--combine needs at least 2 files, got {len(src_paths)}")

    # Load every trajectory
    demonstrations_list = [load_and_extract_tensor(p) for p in src_paths]

    # All files must share the same state dimension D
    dims = [d.shape[2] for d in demonstrations_list]
    if len(set(dims)) != 1:
        mismatch = {p.name: d for p, d in zip(src_paths, dims)}
        raise ValueError(f"State dimension mismatch across files: {mismatch}")

    # Convert each to transitions and concatenate along the first dim
    transitions_list = [convert_trajectory_to_transitions(d) for d in demonstrations_list]
    combined_transitions = torch.cat(transitions_list, dim=0)

    # Save combined tensor
    dst_dir.mkdir(parents=True, exist_ok=True)
    if output_name is None:
        output_name = "expert_obs_masked_combined_processed.pt"
    dst_path = dst_dir / output_name
    torch.save(combined_transitions, dst_path)

    if verbose:
        print(f"Combined transitions from {len(src_paths)} files:")
        for p, d, t in zip(src_paths, demonstrations_list, transitions_list):
            print(f"  {p.name} (shape: {tuple(d.shape)} -> {tuple(t.shape)})")
        print(f"  Saved combined tensor to {dst_path} (shape: {tuple(combined_transitions.shape)})")

    # Plot if requested: one overlaid plot of all + an individual plot per file
    if plot:
        for p, d in zip(src_paths, demonstrations_list):
            plot_individual_trajectory(d, dst_dir, f"{p.stem}_trajectory_plot.png",
                                       title_prefix=f"{p.name} ")
        plot_name = output_name.replace(".pt", "_plot.png")
        plot_trajectories(demonstrations_list, [p.name for p in src_paths], dst_dir, plot_name)

    return combined_transitions


def main():
    parser = argparse.ArgumentParser(description="Convert observation .pt files into (state,next_state) pairs.")
    parser.add_argument("--src", type=str, default="source/wheeledlab_tasks/wheeledlab_tasks/navigation/config/agents/mushr/processed_data", help="Source folder containing observations .pt files")
    parser.add_argument("--dst", type=str, default="/home/anubhav/WheeledLab-research/source/wheeledlab_tasks/wheeledlab_tasks/navigation/config/agents/mushr/real_data/processed_folder", help="Destination folder for converted files")
    parser.add_argument("--pattern", type=str, default="*.pt", help="Glob pattern to match source files")
    parser.add_argument("--combine", nargs='+', type=str, help="Combine N>=2 trajectory files: --combine <file1.pt> <file2.pt> [<file3.pt> ...]")
    parser.add_argument("--combine-output", type=str, default="expert_obs_masked_combined_processed.pt", help="Output filename when combining trajectories")
    parser.add_argument("--plot", action="store_true", help="Plot the combined trajectory (overlaid) and each input trajectory individually")
    parser.add_argument("--dry-run", action="store_true", help="Only print files that would be processed")
    args = parser.parse_args()

    # Handle combining N (>=2) files
    if args.combine:
        src_paths = [Path(p) for p in args.combine]
        if len(src_paths) < 2:
            raise SystemExit(f"--combine needs at least 2 files, got {len(src_paths)}")
        missing = [str(p) for p in src_paths if not p.exists()]
        if missing:
            raise SystemExit(f"Source file(s) do not exist: {', '.join(missing)}")

        dst_dir = Path(args.dst)
        try:
            process_and_combine_files(src_paths, dst_dir, args.combine_output, plot=args.plot, verbose=True)
        except Exception as e:
            print(f"Error combining files: {e}")
        return

    # Handle batch processing of folder
    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    if not src_dir.exists():
        raise SystemExit(f"Source directory does not exist: {src_dir}")

    files = sorted(src_dir.glob(args.pattern))
    if not files:
        print(f"No files found at {src_dir} matching {args.pattern}")
        return

    for f in files:
        if args.dry_run:
            print(f"Would process: {f}")
            continue
        try:
            process_file(f, dst_dir, verbose=True)
        except Exception as e:
            print(f"Error processing {f}: {e}")


if __name__ == "__main__":
    main()
