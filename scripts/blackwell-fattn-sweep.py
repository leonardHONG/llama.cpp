#!/usr/bin/env python3
"""
Blackwell fattn-mma config parameter sweep.

Edits ggml/src/ggml-cuda/fattn-mma-f16.cuh's `ggml_cuda_fattn_mma_get_config_blackwell`
function body, rebuilds incrementally, runs `test-backend-ops perf -o FLASH_ATTN_EXT`
with warmup + reps, parses output, logs per-shape stats to JSON.

Backs up the source file before sweep and restores on exit (including on Ctrl-C),
so the working tree is clean when sweep finishes.

Usage:
    # Coordinate descent on one shape (single-axis variations from Ampere baseline).
    python3 scripts/blackwell-fattn-sweep.py --shape 64,64 --output sweep-hsk64.json

    # Pair interaction probing — vary two axes together to find non-coordinate optima.
    python3 scripts/blackwell-fattn-sweep.py --shape 128,128 \\
        --mode pair-interactions --output sweep-hsk128-pair.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Defaults — override via CLI flags

DEFAULT_REPO = Path.home() / "autodl-tmp" / "llama.cpp"
SRC_REL = Path("ggml/src/ggml-cuda/fattn-mma-f16.cuh")
DEFAULT_BUILD_DIR = "build-cuda"
DEFAULT_BENCH_TARGET = "test-backend-ops"

# Constraints from static_asserts in the macro (see fattn-mma-f16.cuh:26-36).
NTHREADS_OPTIONS       = [128, 256, 512]
OCCUPANCY_OPTIONS      = [1, 2, 3, 4]
NBATCH_FA_OPTIONS      = [32, 64, 128, 256]
NBATCH_K2_OPTIONS      = [32, 64, 128]
NBATCH_V2_OPTIONS      = [32, 64, 128]
NBATCH_COMBINE_OPTIONS = [32, 64, 128]
NSTAGES_OPTIONS        = [1, 2]
Q_IN_REG_OPTIONS       = [True, False]


@dataclass(frozen=True)
class FattnConfig:
    nthreads: int
    occupancy: int
    nbatch_fa: int
    nbatch_K2: int
    nbatch_V2: int
    nbatch_combine: int
    nstages_target: int
    Q_in_reg: bool

    def to_macro_args(self) -> str:
        q = "true" if self.Q_in_reg else "false"
        return (f"{self.nthreads}, {self.occupancy}, {self.nbatch_fa}, "
                f"{self.nbatch_K2}, {self.nbatch_V2}, {self.nbatch_combine}, "
                f"{self.nstages_target}, {q}")


# Per-shape Ampere baseline, copied from fattn-mma-f16.cuh:38-83.
# Each maps (DKQ, DV, ncols) -> FattnConfig.
# Used as the starting point for coordinate descent.
AMPERE_BASELINE = {
    (64, 64, 8):    FattnConfig(128, 2, 128, 32, 32, 32, 2, True),
    (64, 64, 16):   FattnConfig(128, 2,  64, 32, 32, 32, 2, True),
    (64, 64, 32):   FattnConfig(128, 2,  64, 32, 32, 32, 2, True),
    (64, 64, 64):   FattnConfig(128, 2,  64, 32, 32, 32, 2, True),
    (80, 80, 8):    FattnConfig(128, 2, 128, 40, 40, 40, 2, True),
    (80, 80, 16):   FattnConfig(128, 2,  64, 40, 40, 40, 2, True),
    (80, 80, 32):   FattnConfig(128, 2,  64, 40, 40, 40, 2, True),
    (80, 80, 64):   FattnConfig(128, 2,  64, 40, 40, 40, 2, True),
    (96, 96, 8):    FattnConfig(128, 2, 128, 48, 48, 48, 2, True),
    (96, 96, 16):   FattnConfig(128, 2,  64, 48, 48, 48, 2, True),
    (96, 96, 32):   FattnConfig(128, 2,  64, 48, 48, 48, 2, True),
    (96, 96, 64):   FattnConfig(128, 2,  64, 48, 48, 48, 2, True),
    (112, 112, 8):  FattnConfig(128, 2, 128, 56, 56, 56, 2, True),
    (112, 112, 16): FattnConfig(128, 2,  64, 56, 56, 56, 2, True),
    (112, 112, 32): FattnConfig(128, 2,  64, 56, 56, 56, 2, True),
    (112, 112, 64): FattnConfig(128, 2,  64, 56, 56, 56, 2, True),
    (128, 128, 8):  FattnConfig(128, 2, 128, 64, 64, 64, 2, True),
    (128, 128, 16): FattnConfig(128, 2,  64, 64, 64, 64, 2, True),
    (128, 128, 32): FattnConfig(128, 2,  64, 64, 64, 64, 2, True),
    (128, 128, 64): FattnConfig(128, 2,  64, 64, 64, 64, 2, True),
}

# Map FattnConfig field -> list of allowed values (for raw coord-descent fallback).
FIELD_OPTIONS = {
    "nthreads":       NTHREADS_OPTIONS,
    "occupancy":      OCCUPANCY_OPTIONS,
    "nbatch_fa":      NBATCH_FA_OPTIONS,
    "nbatch_K2":      NBATCH_K2_OPTIONS,
    "nbatch_V2":      NBATCH_V2_OPTIONS,
    "nbatch_combine": NBATCH_COMBINE_OPTIONS,
    "nstages_target": NSTAGES_OPTIONS,
    "Q_in_reg":       Q_IN_REG_OPTIONS,
}

# ---------------------------------------------------------------------------
# Motif-based search space (semantic bundles, not raw axis grid).
#
# The 8 raw params have strong couplings:
#   - (nthreads, occupancy, Q_in_reg)        : register/launch resource budget
#   - (nbatch_K2, nbatch_V2, nbatch_combine) : KV partition width
#   - (nbatch_fa, nstages_target)            : FA tile size + pipeline depth
# Independent grid-search over all 8 wastes trials on infeasible / nonsense
# combos. Instead we define a small set of meaningful motifs per cluster,
# then test hand-picked bundles that probe interactions across clusters.

@dataclass(frozen=True)
class ResourceMotif:
    name: str
    nthreads: int
    occupancy: int
    Q_in_reg: bool


RESOURCE_MOTIFS: Dict[str, ResourceMotif] = {
    m.name: m for m in [
        ResourceMotif("baseline",     128, 2, True),    # Ampere baseline
        ResourceMotif("low_reg",      128, 1, False),   # release Q regs, lower occ
        ResourceMotif("wide_block",   256, 1, False),   # more threads, lower occ
        ResourceMotif("wide_block_q", 256, 1, True),    # more threads, keep Q
        ResourceMotif("high_occ",     128, 3, True),    # squeeze more concurrent CTAs
    ]
}


@dataclass(frozen=True)
class KVPartitionMotif:
    name: str
    K2_mult: float       # multiplier vs Ampere baseline
    V2_mult: float
    combine_mult: float


KV_PARTITION_MOTIFS: Dict[str, KVPartitionMotif] = {
    m.name: m for m in [
        KVPartitionMotif("baseline",     1.0, 1.0, 1.0),
        KVPartitionMotif("wide_all",     2.0, 2.0, 2.0),  # double everything
        KVPartitionMotif("wide_load",    2.0, 2.0, 1.0),  # only K/V tiles
        KVPartitionMotif("wide_combine", 1.0, 1.0, 2.0),  # only combine
    ]
}


@dataclass(frozen=True)
class FAPipeMotif:
    name: str
    fa_mult: float                # multiplier on baseline nbatch_fa
    nstages: int                  # 1 or 2
    skip_for_ncols: Tuple[int, ...] = ()  # skip motif for these ncols values


FA_PIPE_MOTIFS: Dict[str, FAPipeMotif] = {
    m.name: m for m in [
        FAPipeMotif("baseline",   1.0, 2),
        FAPipeMotif("light_pipe", 1.0, 1),
        # fa_x2 only makes sense for ncols >= 16 — for ncols=8 baseline is
        # already nbatch_fa=128, doubling would hit the 256 cap on most shapes.
        FAPipeMotif("fa_x2",      2.0, 2, skip_for_ncols=(8,)),
    ]
}


# Hand-picked bundles to probe interaction space efficiently.
# Each bundle = (resource_motif, kv_motif, fa_pipe_motif).
# 7 bundles per (DKQ, DV, ncols) gives ~28 anchor trials for 4-bucket sweep.
RECOMMENDED_BUNDLES: List[Tuple[str, str, str]] = [
    # control
    ("baseline",     "baseline",  "baseline"),
    # one cluster at a time
    ("low_reg",      "baseline",  "baseline"),     # launch: low_reg
    ("wide_block",   "baseline",  "baseline"),     # launch: wide_block
    ("baseline",     "wide_all",  "baseline"),     # KV: wide_all
    ("baseline",     "baseline",  "light_pipe"),   # FA/pipe: nstages=1
    # cross-cluster interaction probes
    ("low_reg",      "wide_all",  "baseline"),     # launch + KV: free regs to widen tiles
    ("low_reg",      "wide_load", "light_pipe"),   # all 3 clusters: aggressive mix
]


def materialize_bundle(
    bundle: Tuple[str, str, str], baseline: FattnConfig
) -> Optional[FattnConfig]:
    """Apply a (resource, kv, fa_pipe) motif bundle to a per-shape baseline.

    Returns None if any param violates static_assert constraints.
    """
    res_name, kv_name, fa_name = bundle
    res = RESOURCE_MOTIFS[res_name]
    kv = KV_PARTITION_MOTIFS[kv_name]
    fa = FA_PIPE_MOTIFS[fa_name]

    cfg = replace(
        baseline,
        nthreads=res.nthreads,
        occupancy=res.occupancy,
        Q_in_reg=res.Q_in_reg,
        nbatch_K2=int(baseline.nbatch_K2 * kv.K2_mult),
        nbatch_V2=int(baseline.nbatch_V2 * kv.V2_mult),
        nbatch_combine=int(baseline.nbatch_combine * kv.combine_mult),
        nbatch_fa=int(baseline.nbatch_fa * fa.fa_mult),
        nstages_target=fa.nstages,
    )

    # Constraint validation against static_asserts.
    if not (cfg.nthreads % 32 == 0 and cfg.nthreads <= 512):
        return None
    if cfg.occupancy > 8:
        return None
    if not (cfg.nbatch_fa % 32 == 0 and cfg.nbatch_fa <= 256):
        return None
    if not (cfg.nbatch_K2 % 4 == 0 and cfg.nbatch_K2 <= 512):
        return None
    if not (cfg.nbatch_V2 % 4 == 0 and cfg.nbatch_V2 <= 256):
        return None
    if not (cfg.nbatch_combine % 4 == 0 and cfg.nbatch_combine <= 128):
        return None
    if cfg.nstages_target not in (1, 2):
        return None
    return cfg


def motif_candidates(
    baseline: FattnConfig, ncols: int,
    bundles: List[Tuple[str, str, str]] = None,
) -> List[Tuple[FattnConfig, str]]:
    """Generate (config, bundle_label) for each applicable motif bundle."""
    if bundles is None:
        bundles = RECOMMENDED_BUNDLES

    out: List[Tuple[FattnConfig, str]] = []
    seen = set()
    for bundle in bundles:
        res_name, kv_name, fa_name = bundle
        fa = FA_PIPE_MOTIFS[fa_name]
        if ncols in fa.skip_for_ncols:
            continue
        cfg = materialize_bundle(bundle, baseline)
        if cfg is None:
            sys.stderr.write(f"  [skip] bundle {bundle} -> constraint violation\n")
            continue
        if cfg in seen:
            continue
        seen.add(cfg)
        label = f"{res_name}+{kv_name}+{fa_name}"
        out.append((cfg, label))
    return out


# ---------------------------------------------------------------------------
# Source patching

def make_blackwell_function_body(
    entries: List[Tuple[int, int, int, FattnConfig]]
) -> str:
    lines = [
        "static constexpr __host__ __device__ fattn_mma_config "
        "ggml_cuda_fattn_mma_get_config_blackwell(const int DKQ, const int DV, const int ncols) {",
        "    // Generated by scripts/blackwell-fattn-sweep.py — DO NOT COMMIT.",
    ]
    for (DKQ, DV, ncols, cfg) in entries:
        lines.append(
            f"    GGML_CUDA_FATTN_MMA_CONFIG_CASE("
            f"{DKQ:3d}, {DV:3d}, {ncols:3d}, {cfg.to_macro_args()});"
        )
    lines.append("    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);")
    lines.append("}")
    return "\n".join(lines)


_BLACKWELL_FN_RE = re.compile(
    r"static constexpr __host__ __device__ fattn_mma_config "
    r"ggml_cuda_fattn_mma_get_config_blackwell.*?^\}",
    re.DOTALL | re.MULTILINE,
)


def patch_source(src_path: Path, body: str) -> None:
    content = src_path.read_text()
    new = _BLACKWELL_FN_RE.sub(body, content, count=1)
    if new == content:
        raise RuntimeError(
            f"Failed to find ggml_cuda_fattn_mma_get_config_blackwell in {src_path}. "
            "Pattern mismatch — has the source layout changed?"
        )
    src_path.write_text(new)


# ---------------------------------------------------------------------------
# Build / bench

def run_capture(cmd: List[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)


def rebuild(repo: Path, build_dir: str, target: str) -> Tuple[bool, str]:
    proc = run_capture(["cmake", "--build", build_dir, "-j", "--target", target], cwd=repo)
    return proc.returncode == 0, (proc.stderr or "") + (proc.stdout or "")


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def bench_once(repo: Path, build_dir: str) -> str:
    bin_path = f"./{build_dir}/bin/test-backend-ops"
    proc = run_capture([bin_path, "perf", "-b", "CUDA0", "-o", "FLASH_ATTN_EXT"], cwd=repo)
    return _ANSI_RE.sub("", proc.stdout)


def parse_us_runs(output: str) -> dict:
    """Return {full_shape_string: us_per_run} for every FLASH_ATTN_EXT result line."""
    # Lines look like:
    #   FLASH_ATTN_EXT(hsk=64,hsv=64,...,permute=[0,1,2,3]): ...  131040 runs -  8.01 us/run - ...
    pat = re.compile(
        r"FLASH_ATTN_EXT\(([^)]+)\):.*?-\s+([0-9.]+)\s+us/run", re.DOTALL,
    )
    results = {}
    for m in pat.finditer(output):
        shape = m.group(1)
        us = float(m.group(2))
        results[shape] = us
    return results


def bench_with_reps(
    repo: Path, build_dir: str, n_warmup: int, n_reps: int,
) -> dict:
    """Return {shape: [us_per_run × n_reps]}."""
    for _ in range(n_warmup):
        bench_once(repo, build_dir)

    accum: dict = {}
    for _ in range(n_reps):
        out = bench_once(repo, build_dir)
        per_shape = parse_us_runs(out)
        for shape, us in per_shape.items():
            accum.setdefault(shape, []).append(us)
    return accum


def stats(values: List[float]) -> dict:
    if not values:
        return {"n": 0}
    n = len(values)
    mean = sum(values) / n
    if n > 1:
        ss = sum((v - mean) ** 2 for v in values)
        stddev = (ss / (n - 1)) ** 0.5
    else:
        stddev = 0.0
    return {
        "n": n,
        "mean": round(mean, 4),
        "stddev": round(stddev, 4),
        "min": min(values),
        "max": max(values),
        "raw": values,
    }


# ---------------------------------------------------------------------------
# Candidate generators

def coord_descent_candidates(baseline: FattnConfig) -> List[FattnConfig]:
    """One-axis-at-a-time variations from baseline."""
    seen = {baseline}
    out = [baseline]
    for field_name, options in FIELD_OPTIONS.items():
        for v in options:
            cfg = replace(baseline, **{field_name: v})
            if cfg not in seen:
                seen.add(cfg)
                out.append(cfg)
    return out


def pair_interaction_candidates(baseline: FattnConfig) -> List[FattnConfig]:
    """All single-axis variations + pair-axis variations from baseline.

    For each (axis_i, axis_j), test every (value_i, value_j) combination
    so we can detect interactions invisible to coord-descent.
    """
    seen = {baseline}
    out = [baseline]

    # Single-axis (re-include for completeness)
    for field_name, options in FIELD_OPTIONS.items():
        for v in options:
            cfg = replace(baseline, **{field_name: v})
            if cfg not in seen:
                seen.add(cfg)
                out.append(cfg)

    # Pair-axis combinations
    field_names = list(FIELD_OPTIONS.keys())
    for f_i, f_j in combinations(field_names, 2):
        for v_i in FIELD_OPTIONS[f_i]:
            for v_j in FIELD_OPTIONS[f_j]:
                cfg = replace(baseline, **{f_i: v_i, f_j: v_j})
                if cfg not in seen:
                    seen.add(cfg)
                    out.append(cfg)
    return out


# ---------------------------------------------------------------------------
# Trial driver

def trial(
    repo: Path,
    src_path: Path,
    build_dir: str,
    target: str,
    entries: List[Tuple[int, int, int, FattnConfig]],
    n_warmup: int,
    n_reps: int,
) -> dict:
    body = make_blackwell_function_body(entries)
    patch_source(src_path, body)

    ok, build_log = rebuild(repo, build_dir, target)
    if not ok:
        # Trim build log to last 30 lines to reduce JSON bloat.
        tail = "\n".join(build_log.splitlines()[-30:])
        return {"build_failed": True, "build_log_tail": tail}

    raw = bench_with_reps(repo, build_dir, n_warmup, n_reps)
    return {"build_failed": False, "shapes": {k: stats(v) for k, v in raw.items()}}


def filter_shapes(all_shapes: dict, DKQ: int, DV: int) -> dict:
    """Keep only shapes matching given (DKQ, DV)."""
    needle = f"hsk={DKQ},hsv={DV},"
    return {k: v for k, v in all_shapes.items() if needle in k}


# ---------------------------------------------------------------------------
# Source backup / restore

class SourceGuard:
    """Backup the source file on enter, restore on exit (incl. signals)."""

    def __init__(self, src_path: Path):
        self.src = src_path
        self.backup = src_path.with_suffix(src_path.suffix + ".sweep.bak")

    def __enter__(self) -> "SourceGuard":
        shutil.copy2(self.src, self.backup)
        sys.stderr.write(f"[guard] backup: {self.backup}\n")

        def _restore_on_signal(signum, _frame):
            self.restore()
            sys.stderr.write(f"\n[guard] restored on signal {signum}\n")
            sys.exit(130)

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _restore_on_signal)
            except Exception:
                pass
        return self

    def restore(self) -> None:
        if self.backup.exists():
            shutil.copy2(self.backup, self.src)
            self.backup.unlink()

    def __exit__(self, *_a) -> None:
        self.restore()
        sys.stderr.write("[guard] restored on exit\n")


# ---------------------------------------------------------------------------
# Main

def parse_shape(s: str) -> Tuple[int, int]:
    a, b = s.split(",")
    return int(a), int(b)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", type=Path, default=DEFAULT_REPO,
                   help=f"llama.cpp repo path (default: {DEFAULT_REPO})")
    p.add_argument("--build", default=DEFAULT_BUILD_DIR,
                   help=f"build dir relative to repo (default: {DEFAULT_BUILD_DIR})")
    p.add_argument("--target", default=DEFAULT_BENCH_TARGET,
                   help=f"cmake target to build (default: {DEFAULT_BENCH_TARGET})")
    p.add_argument("--shape", required=True, type=parse_shape,
                   help="Shape (DKQ,DV), e.g. 64,64 or 128,128")
    p.add_argument("--ncols", nargs="+", type=int, default=[8, 16, 32, 64],
                   help="ncols values to sweep (default: 8 16 32 64)")
    p.add_argument("--mode", choices=["motif", "coord-descent", "pair-interactions"],
                   default="motif",
                   help="Candidate generation strategy "
                        "(motif: 7 hand-picked bundles, default; "
                        "coord-descent: single-axis variations; "
                        "pair-interactions: full pair grid)")
    p.add_argument("--n-warmup", type=int, default=5)
    p.add_argument("--n-reps", type=int, default=20)
    p.add_argument("--output", required=True, type=Path,
                   help="Output JSON file (incrementally written)")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap number of candidates per shape (debug)")
    args = p.parse_args()

    src_path = args.repo / SRC_REL
    if not src_path.exists():
        print(f"Source not found: {src_path}", file=sys.stderr)
        return 1
    bench_bin = args.repo / args.build / "bin" / "test-backend-ops"
    if not bench_bin.exists():
        print(f"Bench binary not found: {bench_bin}\n"
              f"  Run cmake --build {args.build} --target {args.target} first.",
              file=sys.stderr)
        return 1

    DKQ, DV = args.shape
    target_shapes = [(DKQ, DV, ncols) for ncols in args.ncols]

    # Collect baselines and candidate sets per shape.
    # Each entry: (shape, [(config, label), ...])
    plan: List[Tuple[Tuple[int, int, int], List[Tuple[FattnConfig, str]]]] = []
    for shape in target_shapes:
        baseline = AMPERE_BASELINE.get(shape)
        if baseline is None:
            print(f"WARN: no Ampere baseline for {shape}, skipping", file=sys.stderr)
            continue
        DKQ, DV, ncols = shape
        if args.mode == "motif":
            cands = motif_candidates(baseline, ncols)
        elif args.mode == "coord-descent":
            cands = [(c, "coord") for c in coord_descent_candidates(baseline)]
        else:  # pair-interactions
            cands = [(c, "pair") for c in pair_interaction_candidates(baseline)]
        if args.limit:
            cands = cands[: args.limit]
        plan.append((shape, cands))

    total_trials = sum(len(c) for _, c in plan)
    print(f"Plan: {len(plan)} shapes, {total_trials} total trials, "
          f"mode={args.mode}, n_warmup={args.n_warmup}, n_reps={args.n_reps}",
          file=sys.stderr)

    results: List[dict] = []
    args.output.write_text("[]")

    completed = 0
    started_at = time.time()

    with SourceGuard(src_path):
        for shape, candidates in plan:
            DKQ, DV, ncols = shape
            print(f"\n=== Sweep shape ({DKQ},{DV},{ncols}) — {len(candidates)} candidates ===",
                  file=sys.stderr)

            for i, (cfg, label) in enumerate(candidates):
                t0 = time.time()
                entries = [(DKQ, DV, ncols, cfg)]
                tr = trial(args.repo, src_path, args.build, args.target,
                           entries, args.n_warmup, args.n_reps)
                elapsed = time.time() - t0

                # Filter shapes to this DKQ/DV only (other DKQ/DV unaffected).
                if not tr.get("build_failed"):
                    tr["shapes"] = filter_shapes(tr.get("shapes", {}), DKQ, DV)

                rec = {
                    "shape": {"DKQ": DKQ, "DV": DV, "ncols": ncols},
                    "config": asdict(cfg),
                    "label": label,
                    "is_baseline": (cfg == AMPERE_BASELINE[shape]),
                    "result": tr,
                    "elapsed_sec": round(elapsed, 1),
                    "trial_idx": completed,
                }
                results.append(rec)
                args.output.write_text(json.dumps(results, indent=2))
                completed += 1

                # Progress line
                if tr.get("build_failed"):
                    status = "BUILD_FAIL"
                else:
                    means = []
                    for s, st in (tr.get("shapes") or {}).items():
                        means.append(f"{st['mean']:.2f}±{st['stddev']:.2f}")
                    status = " | ".join(means[:3]) if means else "no_match"
                eta = (time.time() - started_at) / completed * (total_trials - completed)
                print(f"  [{completed}/{total_trials}] [{label}] {cfg.to_macro_args()}  -> {status}  "
                      f"({elapsed:.0f}s, eta {eta/60:.1f}min)",
                      file=sys.stderr)

    print(f"\nDONE. Results in {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
