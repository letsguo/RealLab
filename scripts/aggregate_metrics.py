#!/usr/bin/env python3
"""Aggregate per-run metrics.json files across many runs of multiple variants.

Walks a directory for ``*_metrics.json`` (produced by compute_run_metrics.py),
groups them by variant, and prints a summary table. Variant tag comes from:
  1. The "variant" field inside the json (preferred -- survives renames).
  2. Filename prefix matched against --groups (default: dbf_mpail, mpail,
     dbf_gail, gail) when --by-prefix is set.
  3. The raw filename stem when neither applies.

# usage:
#   python3 scripts/aggregate_metrics.py dbf_expert_demonstrations/
#   python3 scripts/aggregate_metrics.py dbf_expert_demonstrations/ --by-prefix
#   python3 scripts/aggregate_metrics.py dbf_expert_demonstrations/ \\
#       --groups dbf_mpail mpail dbf_gail gail
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from statistics import mean, stdev


DEFAULT_GROUPS = ["dbf_mpail", "mpail", "dbf_gail", "gail"]


def _classify(path, metrics, groups, by_prefix):
    """Pick a variant tag for one metrics entry."""
    if "variant" in metrics and metrics["variant"]:
        return str(metrics["variant"])
    if by_prefix:
        stem = os.path.basename(path).removesuffix("_metrics.json")
        # Pick the LONGEST matching prefix (so 'dbf_mpail' wins over 'mpail').
        for g in sorted(groups, key=len, reverse=True):
            if stem.startswith(g):
                return g
        return "unclassified"
    return os.path.basename(path).removesuffix("_metrics.json")


def _summarize(entries):
    """Compute aggregate stats for a list of per-run metric dicts."""
    n = len(entries)
    successes = sum(1 for e in entries if e.get("success"))
    collisions = [int(e.get("collisions", 0)) for e in entries]
    vels = [float(e.get("avg_velocity_m_s", 0.0)) for e in entries]
    yaws = [float(e.get("avg_yaw_rate_rad_s", 0.0)) for e in entries]
    durs = [float(e.get("duration_s", 0.0)) for e in entries]

    def _mean_std(xs):
        m = mean(xs) if xs else 0.0
        s = stdev(xs) if len(xs) > 1 else 0.0
        return m, s

    coll_m, coll_s = _mean_std(collisions)
    vel_m, vel_s   = _mean_std(vels)
    yaw_m, yaw_s   = _mean_std(yaws)
    dur_m, dur_s   = _mean_std(durs)
    return {
        "n": n,
        "success_rate": successes / n if n else 0.0,
        "n_success": successes,
        "n_failure": n - successes,
        "coll_mean": coll_m, "coll_std": coll_s,
        "vel_mean": vel_m,   "vel_std": vel_s,
        "yaw_mean": yaw_m,   "yaw_std": yaw_s,
        "dur_mean": dur_m,   "dur_std": dur_s,
    }


def _print_table(per_variant):
    """Print a fixed-width table with one row per variant."""
    cols = ["variant", "n", "success", "collisions", "vel (m/s)", "|yaw| (rad/s)", "dur (s)"]
    widths = [16, 4, 11, 14, 14, 16, 12]
    header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    print(header)
    print("-" * len(header))
    for variant in sorted(per_variant.keys()):
        s = per_variant[variant]
        row = [
            variant,
            f"{s['n']}",
            f"{s['n_success']}/{s['n']} ({100 * s['success_rate']:.0f}%)",
            f"{s['coll_mean']:.2f} ± {s['coll_std']:.2f}",
            f"{s['vel_mean']:.2f} ± {s['vel_std']:.2f}",
            f"{s['yaw_mean']:.2f} ± {s['yaw_std']:.2f}",
            f"{s['dur_mean']:.1f} ± {s['dur_std']:.1f}",
        ]
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dir", help="Directory to walk for *_metrics.json files.")
    parser.add_argument(
        "--by-prefix", action="store_true",
        help="When a metrics file has no 'variant' field, classify by filename "
             "prefix against --groups (longest match wins).",
    )
    parser.add_argument(
        "--groups", nargs="+", default=DEFAULT_GROUPS,
        help="Prefix list for --by-prefix matching. Default: dbf_mpail mpail dbf_gail gail.",
    )
    parser.add_argument(
        "--json-out", default=None,
        help="Optionally also write the per-variant summary dict as JSON to this path.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        sys.exit(f"ERROR: not a directory: {args.dir}")

    paths = sorted(glob.glob(os.path.join(args.dir, "*_metrics.json")))
    if not paths:
        sys.exit(f"ERROR: no *_metrics.json under {args.dir} -- run "
                 "scripts/compute_run_metrics.py on your bags first.")

    by_variant = defaultdict(list)
    for p in paths:
        with open(p) as f:
            m = json.load(f)
        by_variant[_classify(p, m, args.groups, args.by_prefix)].append(m)

    print(f"Found {len(paths)} metrics file(s) across {len(by_variant)} variant(s)\n")
    per_variant = {v: _summarize(entries) for v, entries in by_variant.items()}
    _print_table(per_variant)

    if args.json_out:
        out = os.path.abspath(args.json_out)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump(per_variant, f, indent=2)
        print(f"\nWrote aggregate summary to {out}")


if __name__ == "__main__":
    main()
