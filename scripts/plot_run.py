#!/usr/bin/env python3
"""
# 1. After collecting bags for one variant (e.g., dbf_gail), compute metrics per bag
for bag in dbf_expert_demonstrations/dbf_gail_*.bag; do
    python3 scripts/compute_run_metrics.py "$bag" --variant dbf_gail
done

# Repeat for the other variants (mpail, dbf_mpail, gail) ...

# 2a. Single-run figure with metrics text overlay (auto-picks up <bag>_metrics.json)
python3 scripts/plot_run.py dbf_gail_run01.bag

# 2b. DBF vs Baseline comparison with two metric lines in the box
python3 scripts/plot_run.py dbf_gail_run01.bag --baseline gail_run01.bag \
    --rollout-label DBF_GAIL --baseline-label Baseline_GAIL

# 2c. NEW: N runs of one variant, success=blue / failure=red, goal circle drawn
python3 scripts/plot_run.py --runs dbf_expert_demonstrations/dbf_gail_*.bag \
    --rollout-label DBF_GAIL

# 3. Aggregate table across all variants
python3 scripts/aggregate_metrics.py dbf_expert_demonstrations/
"""
import argparse
import glob
import os
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.lines as mpl_lines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import rosbag
import torch


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPTS_DIR, ".."))
DEFAULT_EXPERT_DIR = os.path.join(_REPO_ROOT, "dbf_pt")
# Must match src/mpail_hl_control.py's `tracked_obstacles` list -- we look up
# /mocap/<name>/pose for each entry in the run bag.
TRACKED_OBSTACLES = ["big_obstacle_1", "big_obstacle_2", "big_obstacle_3"]
ODOM_TOPIC = "/mocap/local_position/odom"
SAFETY_TOPIC = "/mppi/executed_safety_value"


# ---------------------------------------------------------------------------
# Aesthetic shared across both panels.
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 0.8,
})

# Wong colorblind-safe palette for the overview panel.
_OVERVIEW_PALETTE = {
    "robot": "#0072B2",       # Wong blue
    "chosen": "#009E73",      # Wong bluish green
    "collision": "#D55E00",   # Wong vermilion
    "barrier": "#000000",
    "trajectory": "#000000",  # rollout hero line
    "start_dot": "#CC79A7",
    "obstacle": "#264653",    # dark teal-slate
}
_EXPERT_COLOR = "#666666"

# Per-label palette for the safety panel (also reused by the overview's
# baseline rollout). DBF variants stay black (hero); Baseline variants get
# red so the DBF-vs-baseline comparison reads at a glance.
_SAFETY_PALETTE = {
    "DBF_MPAIL":      "#000000",
    "DBF_GAIL":       "#000000",
    "Baseline_MPAIL": "#D55E00",   # Wong vermilion
    "Baseline_GAIL":  "#E63946",   # crimson red
    "nominal_MPAIL":  "#D55E00",   # legacy alias
    "GAIL":           "#0072B2",
    "AIL":            "#009E73",
}
_SAFETY_FALLBACK_CYCLE = ["#000000", "#D55E00", "#0072B2", "#009E73", "#CC79A7"]

# Used by --runs mode: per-rollout outcome coloring (one variant, N runs).
# Distinct from the comparison palette above so the two modes are visually
# unambiguous: "blue/red here means success/fail, not algorithm".
_SUCCESS_COLOR = "#1f3a5f"   # deep navy
_FAILURE_COLOR = "#8B1E2F"   # deep brick red


def _safety_color_for(label, idx):
    return _SAFETY_PALETTE.get(label, _SAFETY_FALLBACK_CYCLE[idx % len(_SAFETY_FALLBACK_CYCLE)])


# ---------------------------------------------------------------------------
# Data ingestion (.bag + .pt loaders, all inline)
# ---------------------------------------------------------------------------
def read_xy_from_bag(bag_path, odom_topic=ODOM_TOPIC):
    """Return planar (x, y) trajectory recorded on ``odom_topic`` as [N, 2] float64."""
    xy = []
    with rosbag.Bag(bag_path, "r") as bag:
        for _topic, msg, _t in bag.read_messages(topics=[odom_topic]):
            pos = msg.pose.pose.position
            xy.append([pos.x, pos.y])
    return np.asarray(xy, dtype=np.float64)


def load_xy_from_pt(pt_path):
    """Load a [N, 2] xy tensor from a .pt produced by scripts/rosbag_to_pt.py."""
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


def load_xy(path, odom_topic=ODOM_TOPIC):
    """Extension-dispatch loader: .bag -> rosbag, .pt -> torch.load shim."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trajectory file does not exist: {path}")
    if path.endswith(".bag.active"):
        raise ValueError(
            f"{path} is still being recorded (.bag.active). Stop the "
            "recorder so the bag is finalized, then retry."
        )
    if path.endswith(".bag"):
        return read_xy_from_bag(path, odom_topic=odom_topic)
    if path.endswith(".pt"):
        return load_xy_from_pt(path)
    _, ext = os.path.splitext(path)
    raise ValueError(
        f"Unsupported trajectory file extension {ext!r} for {path}. "
        "Expected .bag or .pt."
    )


def default_expert_paths(expert_dir=DEFAULT_EXPERT_DIR):
    """Glob the expert directory for .pt files; sorted for stable plot order."""
    return sorted(glob.glob(os.path.join(expert_dir, "*.pt")))


def _parse_obstacles_inline(spec):
    """Parse '--obstacles "x1,y1 x2,y2 ..."' -> np.ndarray[M, 2]."""
    coords = []
    for token in spec.replace(",", " ").split():
        coords.append(float(token))
    if len(coords) % 2 != 0:
        raise ValueError(
            f"--obstacles needs an even number of floats; got {len(coords)} from {spec!r}"
        )
    return np.asarray(coords, dtype=np.float64).reshape(-1, 2)


def read_obstacles_from_bag(bag_path, names=TRACKED_OBSTACLES):
    """Pull last /mocap/<name>/pose per name from a bag, in TRACKED_OBSTACLES order.
    Returns None if no matching topics were found.
    """
    found = {}
    wanted_topics = {f"/mocap/{name}/pose": name for name in names}
    with rosbag.Bag(bag_path, "r") as bag:
        for topic, msg, _t in bag.read_messages(topics=list(wanted_topics.keys())):
            pos = msg.pose.position
            found[wanted_topics[topic]] = (pos.x, pos.y)
    if not found:
        return None
    return np.asarray(
        [found[name] for name in names if name in found],
        dtype=np.float64,
    )


def resolve_obstacles(bag_path, obstacles_override=None):
    """Return (obstacles[M,2], summary_str). Inline --obstacles wins; else
    bag /mocap/<name>/pose; else exit with an actionable error.
    """
    if obstacles_override:
        obs = _parse_obstacles_inline(obstacles_override)
        return obs, f"Obstacles from --obstacles: {len(obs)} positions"
    if not bag_path.endswith(".bag"):
        sys.exit(
            f"ERROR: {bag_path!r} is not a .bag, so obstacle positions can't "
            "be read from /mocap/<name>/pose. Pass --obstacles "
            "\"x1,y1 x2,y2 ...\" explicitly."
        )
    obs = read_obstacles_from_bag(bag_path)
    if obs is None:
        sys.exit(
            "ERROR: bag has no /mocap/<name>/pose topics for "
            f"{TRACKED_OBSTACLES}. Pass --obstacles \"x1,y1 x2,y2 ...\" instead."
        )
    return obs, (
        f"Obstacles from bag: {len(TRACKED_OBSTACLES)} tracked names "
        f"→ {len(obs)} poses"
    )


def read_safety_from_bag(bag_path, topic=SAFETY_TOPIC):
    """Return (t_sec[N], h[N]) numpy arrays from /mppi/executed_safety_value
    in a bag. Time is bag-message timestamp rebased to t=0.
    """
    ts, hs = [], []
    with rosbag.Bag(bag_path, "r") as bag:
        for _topic, msg, t in bag.read_messages(topics=[topic]):
            ts.append(t.to_sec())
            hs.append(float(msg.data))
    if not ts:
        raise RuntimeError(
            f"No messages on {topic!r} in {bag_path}. Was the controller "
            "running with the safety-value publisher active?"
        )
    ts = np.asarray(ts, dtype=np.float64)
    ts = ts - ts[0]
    hs = np.asarray(hs, dtype=np.float64)
    return ts, hs


def _load_safety_trace(bag_path):
    """Wrapper that converts missing-topic into a None+print for graceful
    degradation (overview panel still renders for old bags)."""
    try:
        return read_safety_from_bag(bag_path)
    except RuntimeError as exc:
        print(f"[plot_run] No safety topic in bag ({exc}); skipping safety panel.")
        return None


# ---------------------------------------------------------------------------
# Per-run metrics: load from sidecar JSON, fall back to inline computation.
# Used by both the metrics text overlay and the --runs success/fail coloring.
# ---------------------------------------------------------------------------
def _load_metrics_json(bag_path):
    """Return the parsed <bag>_metrics.json dict, or None if missing."""
    candidate = os.path.splitext(bag_path)[0] + "_metrics.json"
    if not os.path.exists(candidate):
        return None
    import json
    with open(candidate) as f:
        return json.load(f)


def _is_success(rollout_xy, goal, goal_tol):
    """Inline success check (so we don't require precomputed metrics.json
    for the --runs visualization). Returns True iff any point along the
    trajectory falls within goal_tol meters of `goal`."""
    if len(rollout_xy) == 0:
        return False
    dists = np.hypot(rollout_xy[:, 0] - goal[0], rollout_xy[:, 1] - goal[1])
    return bool(dists.min() <= goal_tol)


def _format_metrics_line(label, metrics):
    """Compact per-rollout summary line for the overview stats box."""
    if metrics is None:
        return f"{label}: (no metrics.json)"
    ok = "OK" if metrics.get("success") else "FAIL"
    return (
        f"{label}: {ok}  "
        f"collisions={metrics.get('collisions', '?')}  "
        f"v={metrics.get('avg_velocity_m_s', float('nan')):.2f} m/s  "
        f"|ω|={metrics.get('avg_yaw_rate_rad_s', float('nan')):.2f} rad/s"
    )


def _add_metrics_text(ax, lines):
    """Render a compact stats box in the overview's upper-right corner."""
    if not lines:
        return
    text = "\n".join(lines)
    ax.text(
        0.98, 0.98, text,
        transform=ax.transAxes,
        ha="right", va="top", fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#888888", alpha=0.85),
        zorder=8,
    )


# ---------------------------------------------------------------------------
# Drawing -- overview (spatial)
# ---------------------------------------------------------------------------
def _add_obstacle(ax, x, y, side):
    """Solid dark-teal square per obstacle."""
    rect = patches.Rectangle(
        (x - side / 2.0, y - side / 2.0), side, side,
        facecolor=_OVERVIEW_PALETTE["obstacle"], edgecolor="black",
        linewidth=1.0, alpha=0.95, zorder=1,
    )
    ax.add_patch(rect)


def _add_callout(ax, x, y, w, h, label):
    """Outline rectangle + corner label for paper-mode A/B/C zoom boxes."""
    ax.add_patch(patches.Rectangle(
        (x, y), w, h, fill=False, edgecolor="black",
        linewidth=1.0, linestyle="-", zorder=6,
    ))
    ax.text(
        x + 0.05 * w, y + h - 0.05 * h, label,
        fontsize=11, fontweight="bold", color="black",
        ha="left", va="top", zorder=7,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                  edgecolor="none", alpha=0.8),
    )


def draw_overview(ax, experts, rollouts, obstacles, callouts,
                  obstacle_side, legend_handles_extra=None):
    """Compose the spatial overview onto `ax`.

      experts:   list of (label, xy[N, 2]) — drawn as faint dotted gray.
      rollouts:  list of either (label, xy[N, 2]) OR (label, xy[N, 2], color).
                 Without an explicit color, falls back to:
                   - first entry = hero black,
                   - subsequent entries route through _safety_color_for.
                 With an explicit color (e.g., success=blue/failure=red in
                 --runs mode), that color is used verbatim per rollout.
      obstacles: [M, 2].
      callouts:  list of (x, y, w, h, label) for paper-mode zoom boxes.
      legend_handles_extra: optional list of Line2D handles appended to the
                            legend (used by --runs mode to show the
                            success/fail/expert counts instead of per-rollout
                            labels, which would clutter at N>2).
    """
    for obs_x, obs_y in obstacles:
        _add_obstacle(ax, float(obs_x), float(obs_y), side=obstacle_side)
    obs_handle = patches.Patch(
        facecolor=_OVERVIEW_PALETTE["obstacle"], edgecolor="black",
        linewidth=1.0, alpha=0.95, label="Obstacle",
    )

    for _label, xy in experts:
        ax.plot(
            xy[:, 0], xy[:, 1],
            color=_EXPERT_COLOR, linestyle=(0, (1, 2)), linewidth=2.5, alpha=0.85,
            zorder=2,
        )
    expert_legend = mpl_lines.Line2D(
        [], [],
        color=_EXPERT_COLOR, linestyle=(0, (1, 2)), linewidth=2.5,
        label=f"{len(experts)} expert trajectories",
    )

    rollout_handles = []
    for i, item in enumerate(rollouts):
        if len(item) == 3:
            label, xy, color = item
        else:
            label, xy = item
            color = _OVERVIEW_PALETTE["trajectory"] if i == 0 else _safety_color_for(label, i)
        (line,) = ax.plot(
            xy[:, 0], xy[:, 1],
            color=color, linewidth=3.5, alpha=0.95,
            zorder=3, label=label,
        )
        rollout_handles.append(line)

    for (cx, cy, cw, ch, label) in callouts:
        _add_callout(ax, cx, cy, cw, ch, label)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.tick_params(direction="in", length=4)
    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$y$ [m]")
    # --runs mode passes pre-built legend entries (e.g., "12 success", "3 fail")
    # to replace per-rollout handles, since N>2 individual labels would clutter.
    if legend_handles_extra is not None:
        legend_handles = [expert_legend, *legend_handles_extra, obs_handle]
    else:
        legend_handles = [expert_legend, *rollout_handles, obs_handle]
    ax.legend(
        handles=legend_handles,
        loc="upper left", frameon=False,
    )


# ---------------------------------------------------------------------------
# Drawing -- safety value
# ---------------------------------------------------------------------------
def draw_safety_value(ax, traces, hlim=None, tlim=None, title="Safety Value Function"):
    """traces: list of (label, t[N], h[N]) tuples. Time on x-axis, h on y-axis.
    h=0 dashed boundary running horizontally; positive h above (safe),
    negative below (unsafe); "safe ↑ / unsafe ↓" arrows hug the y-axis.
    """
    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0, zorder=1)

    for i, (label, t, h) in enumerate(traces):
        color = _safety_color_for(label, i)
        ax.plot(t, h, color=color, linewidth=2.0, alpha=0.95, zorder=3, label=label)

    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(r"safety value $h(s)$")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.tick_params(direction="in", length=4)

    if tlim is not None:
        ax.set_xlim(tlim)
    else:
        all_t = np.concatenate([t for _, t, _ in traces])
        ax.set_xlim(float(all_t.min()), float(all_t.max()))

    # Auto-fit symmetric around 0 by default so safe/unsafe halves read
    # evenly regardless of the trained model's h-scale (DBF-GAIL ~ ±1,
    # an older collapsed MPAIL ~ ±0.005). Override with --safety-hlim a b.
    if hlim is None:
        all_h = np.concatenate([h for _, _, h in traces])
        m = max(float(np.max(np.abs(all_h))), 1e-3) * 1.1
        ax.set_ylim(-m, m)
    else:
        ax.set_ylim(hlim)

    ypad = 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.text(
        0.02, ypad, r"$\uparrow$ safe",
        transform=ax.get_yaxis_transform(),
        color="#555555", fontsize=10, ha="left", va="bottom", zorder=4,
    )
    ax.text(
        0.02, -ypad, r"$\downarrow$ unsafe",
        transform=ax.get_yaxis_transform(),
        color="#555555", fontsize=10, ha="left", va="top", zorder=4,
    )

    if len(traces) > 1:
        ax.legend(loc="lower right", frameon=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_callout(values):
    """argparse nargs=5 returns [x, y, w, h, label] strings -- coerce."""
    if len(values) != 5:
        raise argparse.ArgumentTypeError(
            "--callout needs exactly 5 values: x y w h LABEL"
        )
    x, y, w, h = map(float, values[:4])
    label = str(values[4])
    return (x, y, w, h, label)


def _render_runs_mode(args):
    """N runs of one variant: each rollout colored by success/failure.
    Overview-only (no safety panel). Used to visualize what a metrics table
    summarizes.
    """
    bag_paths = [os.path.abspath(p) for p in args.runs]
    missing = [p for p in bag_paths if not os.path.exists(p)]
    if missing:
        sys.exit(f"ERROR: --runs bags not found: {missing}")

    rollouts = []      # list of (label, xy, color)
    n_success = 0
    n_failure = 0
    for p in bag_paths:
        xy = read_xy_from_bag(p, odom_topic=args.odom_topic)
        if len(xy) == 0:
            print(f"[plot_run] {os.path.basename(p)}: no odom, skipping.")
            continue
        # Prefer the precomputed metrics.json when present (cheap, consistent
        # with what aggregate_metrics.py reads); fall back to inline check.
        m = _load_metrics_json(p)
        if m is not None and "success" in m:
            success = bool(m["success"])
        else:
            success = _is_success(xy, args.goal, args.goal_tol)
        color = _SUCCESS_COLOR if success else _FAILURE_COLOR
        label = os.path.splitext(os.path.basename(p))[0]
        rollouts.append((label, xy, color))
        if success:
            n_success += 1
        else:
            n_failure += 1
        print(f"  {label}: {'SUCCESS' if success else 'FAIL'} (N={len(xy)})")

    if not rollouts:
        sys.exit("ERROR: no valid rollouts loaded from --runs.")

    expert_paths = args.expert if args.expert else default_expert_paths(args.expert_dir)
    experts = []
    for p in expert_paths:
        xy = load_xy(p, odom_topic=args.odom_topic)
        label = os.path.splitext(os.path.basename(p))[0]
        experts.append((label, xy))

    # Obstacles: resolve from the first run bag (they're static across runs
    # in this lab; if not, override with --obstacles).
    obstacles, obs_summary = resolve_obstacles(bag_paths[0], args.obstacles)
    print(obs_summary)
    print(f"Aggregate: {n_success} success, {n_failure} fail, {len(experts)} experts")

    # Custom legend entries: counts instead of per-rollout labels.
    success_handle = mpl_lines.Line2D(
        [], [], color=_SUCCESS_COLOR, linewidth=3.5,
        label=f"{n_success} successful runs",
    )
    failure_handle = mpl_lines.Line2D(
        [], [], color=_FAILURE_COLOR, linewidth=3.5,
        label=f"{n_failure} failed runs",
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    draw_overview(
        ax, experts, rollouts, obstacles, callouts=[],
        obstacle_side=args.obstacle_side,
        legend_handles_extra=[success_handle, failure_handle],
    )
    if args.xlim is not None:
        ax.set_xlim(args.xlim)
    if args.ylim is not None:
        ax.set_ylim(args.ylim)
    if args.flow_up:
        ax.invert_yaxis()

    # Mark the goal + tolerance circle so a reader can verify the success
    # criterion at a glance.
    goal_circle = patches.Circle(
        tuple(args.goal), radius=args.goal_tol,
        fill=False, edgecolor="#1f3a5f", linestyle="--",
        linewidth=1.0, zorder=5,
    )
    ax.add_patch(goal_circle)
    ax.plot(args.goal[0], args.goal[1], marker="*", color="#1f3a5f",
            markersize=12, zorder=6)

    fig.tight_layout()
    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(bag_paths[0])),
        "runs_overview.png",
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Render the paired overview + safety-value figure from a single "
            "run bag. Self-contained: no sibling .py modules.\n\n"
            "Three modes:\n"
            "  default            -> hero overview + safety panel from one bag.\n"
            "  --baseline X.bag   -> DBF (hero, black) vs Baseline (red) comparison.\n"
            "  --runs B1.bag ...  -> N rollouts of one variant, colored by\n"
            "                        success (blue) / failure (red); overview only.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("bag", nargs="?", default=None,
                        help="Run .bag produced by expert_recorder_odom. "
                             "Required for default + --baseline modes; "
                             "ignored when --runs is given.")
    parser.add_argument(
        "--expert", nargs="+", default=None,
        help=(
            "Expert trajectory paths (.bag or .pt); >= 1. "
            f"Default: glob('{DEFAULT_EXPERT_DIR}/*.pt')."
        ),
    )
    parser.add_argument(
        "--expert-dir", default=DEFAULT_EXPERT_DIR,
        help="Directory globbed for *.pt experts when --expert is omitted.",
    )
    parser.add_argument(
        "--obstacles", default=None,
        help=(
            "Inline obstacle override 'x1,y1 x2,y2 ...'. Default: read "
            "/mocap/<name>/pose from the run bag."
        ),
    )
    parser.add_argument(
        "--rollout-label", default="DBF_MPAIL",
        help="Legend label for the primary (hero) trajectory + safety trace.",
    )
    parser.add_argument(
        "--baseline", default=None,
        help=(
            "Optional second bag to overlay as a baseline (rendered in red on "
            "the overview, second trace on the safety panel). Use to compare "
            "DBF vs non-DBF runs in the same figure."
        ),
    )
    parser.add_argument(
        "--runs", nargs="+", default=None, metavar="BAG",
        help=(
            "Aggregation mode: pass N bags from runs of a SINGLE variant. Each "
            "rollout is colored by outcome (success=dark blue, failure=dark red). "
            "Goal + tolerance for success come from --goal/--goal-tol below. "
            "Safety panel is skipped in this mode."
        ),
    )
    parser.add_argument(
        "--goal", nargs=2, type=float, default=[-1.5, -8.0], metavar=("X", "Y"),
        help="Success goal point (m). Default matches lab convention.",
    )
    parser.add_argument(
        "--goal-tol", type=float, default=0.5,
        help="Success tolerance: robot must reach this radius around --goal.",
    )
    parser.add_argument(
        "--baseline-label", default="Baseline",
        help="Legend label for the baseline rollout.",
    )
    parser.add_argument(
        "--obstacle-side", type=float, default=0.35,
        help="Obstacle side length (m) for drawing.",
    )
    parser.add_argument(
        "--callout", action="append", nargs=5, default=[],
        metavar=("X", "Y", "W", "H", "LABEL"),
        help="Add a labeled rectangle to the overview marking a zoom inset. Repeatable.",
    )
    parser.add_argument(
        "--out", default=None,
        help=(
            "Output path. Default: <bag_stem>_run.png next to the bag "
            "(or <bag_stem>_overview.png + <bag_stem>_safety.png with --separate)."
        ),
    )
    parser.add_argument(
        "--separate", action="store_true",
        help="Emit two PNGs (overview + safety) instead of one combined figure.",
    )
    parser.add_argument(
        "--no-flow-up", dest="flow_up", action="store_false", default=False,
        help="Disable trajectory mirroring; trajectories flow in raw mocap-frame y.",
    )
    parser.add_argument("--xlim", nargs=2, type=float, default=None, help="Overview x-limits.")
    parser.add_argument("--ylim", nargs=2, type=float, default=None, help="Overview y-limits.")
    parser.add_argument("--safety-hlim", nargs=2, type=float, default=None,
                        help="Safety panel h-axis (y) limits (default: auto-fit symmetric around 0).")
    parser.add_argument("--safety-tlim", nargs=2, type=float, default=None,
                        help="Safety panel time-axis (x) limits (default: trace extent).")
    parser.add_argument(
        "--no-metrics", dest="show_metrics", action="store_false", default=True,
        help="Don't auto-load <bag>_metrics.json or render the stats text box.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--odom-topic", default=ODOM_TOPIC,
        help="Odometry topic to extract trajectory from.",
    )
    args = parser.parse_args()

    # --runs is a distinct mode: N bags of one variant, colored by outcome.
    # Branches off before any of the single-bag logic so the two paths stay
    # uncoupled.
    if args.runs:
        if args.baseline is not None:
            sys.exit("ERROR: --runs and --baseline are mutually exclusive.")
        _render_runs_mode(args)
        return

    if args.bag is None:
        sys.exit("ERROR: positional bag is required (or pass --runs <bags...>).")
    bag_path = os.path.abspath(args.bag)
    if not os.path.exists(bag_path):
        sys.exit(f"ERROR: bag not found: {bag_path}")

    rollout_xy = read_xy_from_bag(bag_path, odom_topic=args.odom_topic)
    if len(rollout_xy) == 0:
        sys.exit(f"ERROR: no {args.odom_topic} messages in {bag_path}")
    print(f"Loaded rollout: {os.path.basename(bag_path)} (N={len(rollout_xy)})")

    # Optional second bag for DBF-vs-baseline comparison.
    baseline_xy = None
    baseline_path = None
    if args.baseline is not None:
        baseline_path = os.path.abspath(args.baseline)
        if not os.path.exists(baseline_path):
            sys.exit(f"ERROR: --baseline bag not found: {baseline_path}")
        baseline_xy = read_xy_from_bag(baseline_path, odom_topic=args.odom_topic)
        if len(baseline_xy) == 0:
            sys.exit(f"ERROR: no {args.odom_topic} in {baseline_path}")
        print(f"Loaded baseline: {os.path.basename(baseline_path)} (N={len(baseline_xy)})")

    expert_paths = args.expert if args.expert else default_expert_paths(args.expert_dir)
    experts = []
    for p in expert_paths:
        xy = load_xy(p, odom_topic=args.odom_topic)
        label = os.path.splitext(os.path.basename(p))[0]
        experts.append((label, xy))
        print(f"Loaded expert: {os.path.basename(p)} (N={len(xy)})")
    if not experts:
        sys.exit(
            f"ERROR: no expert trajectories found. Pass --expert or drop .pt "
            f"files into {args.expert_dir}."
        )

    obstacles, obs_summary = resolve_obstacles(bag_path, args.obstacles)
    print(obs_summary)

    callouts = [_parse_callout(c) for c in args.callout]

    # Build the rollouts list (primary always; baseline if --baseline given).
    rollouts = [(args.rollout_label, rollout_xy)]
    if baseline_xy is not None:
        rollouts.append((args.baseline_label, baseline_xy))

    # Build the safety traces list. Each bag may or may not have the topic.
    safety_traces = []
    primary_safety = _load_safety_trace(bag_path)
    if primary_safety is not None:
        t, h = primary_safety
        safety_traces.append((args.rollout_label, t, h))
    if baseline_path is not None:
        baseline_safety = _load_safety_trace(baseline_path)
        if baseline_safety is not None:
            t, h = baseline_safety
            safety_traces.append((args.baseline_label, t, h))

    # Auto-load metrics.json sidecar files for the stats text box. Missing
    # files just yield None entries -- _format_metrics_line / _add_metrics_text
    # degrade gracefully.
    metrics_lines = []
    if args.show_metrics:
        primary_metrics = _load_metrics_json(bag_path)
        if primary_metrics is not None:
            metrics_lines.append(_format_metrics_line(args.rollout_label, primary_metrics))
        if baseline_path is not None:
            baseline_metrics = _load_metrics_json(baseline_path)
            if baseline_metrics is not None:
                metrics_lines.append(_format_metrics_line(args.baseline_label, baseline_metrics))

    base, _ = os.path.splitext(bag_path)

    if args.separate:
        out_overview = args.out or (base + "_overview.png")
        if args.separate and args.out:
            out_safety = os.path.splitext(args.out)[0] + "_safety.png"
        else:
            out_safety = base + "_safety.png"

        fig_o, ax_o = plt.subplots(figsize=(8, 8))
        draw_overview(
            ax_o, experts, rollouts, obstacles, callouts,
            obstacle_side=args.obstacle_side,
        )
        if args.xlim is not None:
            ax_o.set_xlim(args.xlim)
        if args.ylim is not None:
            ax_o.set_ylim(args.ylim)
        if args.flow_up:
            ax_o.invert_yaxis()
        _add_metrics_text(ax_o, metrics_lines)
        fig_o.tight_layout()
        fig_o.savefig(out_overview, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig_o)
        print(f"Saved {out_overview}")

        if safety_traces:
            fig_s, ax_s = plt.subplots(figsize=(8, 3))
            draw_safety_value(
                ax_s, safety_traces,
                hlim=args.safety_hlim, tlim=args.safety_tlim,
            )
            fig_s.tight_layout()
            fig_s.savefig(out_safety, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig_s)
            print(f"Saved {out_safety}")
        return

    # Combined vertical stack: overview 8x8 on top, wide safety 8x3 below.
    out_combined = args.out or (base + "_run.png")
    has_safety = bool(safety_traces)
    if has_safety:
        fig, (ax_o, ax_s) = plt.subplots(
            2, 1, figsize=(8, 11),
            gridspec_kw={"height_ratios": [8, 3]},
        )
    else:
        fig, ax_o = plt.subplots(figsize=(8, 8))
        ax_s = None

    draw_overview(
        ax_o, experts, rollouts, obstacles, callouts,
        obstacle_side=args.obstacle_side,
    )
    if args.xlim is not None:
        ax_o.set_xlim(args.xlim)
    if args.ylim is not None:
        ax_o.set_ylim(args.ylim)
    if args.flow_up:
        ax_o.invert_yaxis()
    _add_metrics_text(ax_o, metrics_lines)

    if has_safety:
        draw_safety_value(
            ax_s, safety_traces,
            hlim=args.safety_hlim, tlim=args.safety_tlim,
        )

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_combined)) or ".", exist_ok=True)
    fig.savefig(out_combined, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_combined}")


if __name__ == "__main__":
    main()
