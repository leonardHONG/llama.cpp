#!/usr/bin/env python3
"""
Pick winning Blackwell fattn-mma configs from sweep JSON output.

For each (DKQ, DV, ncols) shape, find the config whose mean us/run is
significantly lower than the Ampere baseline (range non-overlap with baseline).
Emits a CONFIG_CASE table ready to drop into get_config_blackwell, plus a
human-readable summary.

Usage:
    python3 scripts/blackwell-fattn-pick-winners.py sweep-hsk128.json
    python3 scripts/blackwell-fattn-pick-winners.py sweep-*.json --gain-threshold 2.0
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import List


def load_results(paths: List[Path]) -> list:
    all_records = []
    for p in paths:
        all_records.extend(json.loads(p.read_text()))
    return all_records


def shape_key(rec: dict) -> tuple:
    s = rec["shape"]
    return (s["DKQ"], s["DV"], s["ncols"])


def get_target_stat(rec: dict, kv_pref: str = None):
    """Return mean us/run for the primary measured shape, picking the longest kv length
    (most stable) by default. If no shapes matched (build fail or filter mismatch), return None."""
    if rec["result"].get("build_failed"):
        return None
    shapes = rec["result"].get("shapes") or {}
    if not shapes:
        return None
    if kv_pref:
        for k, v in shapes.items():
            if kv_pref in k:
                return v
    # Default: max kv (most stable, least warmup-sensitive)
    best = max(shapes.items(), key=lambda kv: (
        int((kv[0].split("kv=")[1]).split(",")[0]) if "kv=" in kv[0] else 0
    ))
    return best[1]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", type=Path, nargs="+",
                   help="Sweep JSON files (multiple allowed)")
    p.add_argument("--gain-threshold", type=float, default=2.0,
                   help="Min %% gain to declare winner (default: 2.0)")
    p.add_argument("--require-non-overlap", action="store_true", default=True,
                   help="Require winner range not overlap baseline range")
    p.add_argument("--kv-pref", default=None,
                   help="Filter to shapes containing this kv= substring")
    args = p.parse_args()

    records = load_results(args.inputs)
    by_shape: dict = defaultdict(list)
    for rec in records:
        by_shape[shape_key(rec)].append(rec)

    print(f"# Blackwell sweep — winners (gain >= {args.gain_threshold}%)\n")
    print("# CONFIG_CASE entries (drop into get_config_blackwell):\n")

    winners_table = []
    no_winner = []

    for shape, recs in sorted(by_shape.items()):
        DKQ, DV, ncols = shape

        # Find baseline record
        baseline_recs = [r for r in recs if r.get("is_baseline")]
        baseline_stat = None
        if baseline_recs:
            baseline_stat = get_target_stat(baseline_recs[0], args.kv_pref)

        if baseline_stat is None:
            print(f"# ({DKQ},{DV},{ncols}): no valid baseline measurement", file=sys.stderr)
            continue

        # Score all candidates by mean
        scored = []
        for r in recs:
            st = get_target_stat(r, args.kv_pref)
            if st is None:
                continue
            scored.append((r, st))

        # Sort ascending mean
        scored.sort(key=lambda rs: rs[1]["mean"])

        if not scored:
            continue

        best_rec, best_stat = scored[0]
        gain = (baseline_stat["mean"] - best_stat["mean"]) / baseline_stat["mean"] * 100
        non_overlap = best_stat["max"] < baseline_stat["min"]

        is_winner = (
            gain >= args.gain_threshold
            and (not args.require_non_overlap or non_overlap)
            and not best_rec.get("is_baseline")
        )

        cfg = best_rec["config"]
        q = "true" if cfg["Q_in_reg"] else "false"

        if is_winner:
            args_str = (f"{cfg['nthreads']}, {cfg['occupancy']}, {cfg['nbatch_fa']}, "
                        f"{cfg['nbatch_K2']}, {cfg['nbatch_V2']}, "
                        f"{cfg['nbatch_combine']}, {cfg['nstages_target']}, {q}")
            line = (f"GGML_CUDA_FATTN_MMA_CONFIG_CASE("
                    f"{DKQ:3d}, {DV:3d}, {ncols:3d}, {args_str});  "
                    f"// {gain:+.1f}% vs ampere "
                    f"({best_stat['mean']:.2f}±{best_stat['stddev']:.3f} "
                    f"vs {baseline_stat['mean']:.2f}±{baseline_stat['stddev']:.3f}, "
                    f"non-overlap={non_overlap})")
            winners_table.append(line)
            print(line)
        else:
            reason = []
            if gain < args.gain_threshold:
                reason.append(f"gain {gain:+.1f}% < {args.gain_threshold}%")
            if args.require_non_overlap and not non_overlap:
                reason.append("range overlaps baseline")
            if best_rec.get("is_baseline"):
                reason.append("best is baseline")
            no_winner.append((shape, gain, ", ".join(reason)))

    print("\n# --- summary -----------------------------------------------------")
    print(f"# winners:    {len(winners_table)}")
    print(f"# no winner:  {len(no_winner)}")
    if no_winner:
        print("\n# Shapes with no winner:")
        for shape, gain, reason in no_winner:
            DKQ, DV, ncols = shape
            print(f"#   ({DKQ:3d},{DV:3d},{ncols:3d}): best gain {gain:+.2f}% — {reason}")


if __name__ == "__main__":
    main()
