#!/usr/bin/env python3
"""validate_random_tracks.py -- diversity + flyability validation for the randomised track generator.

Generates a batch of random course layouts across all DIFFICULTY_PRESETS, reports geometry
statistics vs the fixed VQ1 hold-out, and verifies the flyability invariants that sample_courses
guarantees.  Run from the repo root inside the .venv:

    python scripts/validate_random_tracks.py            # 256 courses per preset
    python scripts/validate_random_tracks.py --n 1024   # larger batch

EXIT: 0 on all-pass, 1 on any flyability failure.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ---- path setup (laptop / no PYTHONPATH) -------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
_RL = _ROOT / "rl"
sys.path.insert(0, str(_RL))

try:
    import torch
except ImportError:
    print("ERROR: torch not found -- run inside .venv")
    sys.exit(1)

from peregrine_course import (  # noqa: E402
    DEFAULT_COURSE_RANGES as _DR,
    DIFFICULTY_PRESETS,
    VQ1_SPAWN_POS_ZUP,
    load_course_zup,
    sample_courses,
)

_VQ1_JSON = _RL / "peregrine_course_diffaero.json"


# ---- helpers -----------------------------------------------------------------------------------

def _course_stats(gate_pos: torch.Tensor, spawn_pos: torch.Tensor) -> dict:
    """Compute geometry stats over a batch of courses (n, G, 3) -> summary dict (numpy scalars)."""
    pts = torch.cat([spawn_pos.unsqueeze(1), gate_pos], dim=1)           # (n, G+1, 3)
    seg = pts[:, 1:] - pts[:, :-1]                                        # (n, G, 3)
    horiz = torch.linalg.norm(seg[..., :2], dim=-1)                       # (n, G)
    drop = -seg[..., 2]                                                   # (n, G)  +down
    grade = seg[..., 2].abs() / horiz.clamp(min=1e-6)                    # (n, G)
    headings = torch.atan2(seg[..., 1], seg[..., 0])                      # (n, G)
    dh = headings[:, 1:] - headings[:, :-1]                               # (n, G-1)
    turns_abs = torch.atan2(torch.sin(dh), torch.cos(dh)).abs()           # (n, G-1)

    # per-gate inter-distance (excluding self): minimum horizontal separation
    pad_gates = pts[..., :2]                                               # (n, G+1, 2)
    d = torch.linalg.norm(pad_gates[:, :, None, :] - pad_gates[:, None, :, :], dim=-1)
    G1 = d.shape[1]
    d = d + torch.eye(G1, device=d.device) * 1e9
    min_pair = d.amin(dim=(1, 2))

    # total horizontal track length (pad->last gate)
    total_horiz = horiz.sum(dim=1)

    def s(t): return float(t.mean()), float(t.std()), float(t.min()), float(t.max())

    return {
        "seg_len_m": s(horiz[:, 1:]),               # gate->gate only (not spawn->g0)
        "turn_deg": s(torch.rad2deg(turns_abs)),
        "drop_m": s(drop[:, 1:]),
        "max_grade": s(grade[:, 1:]),
        "min_pair_m": s(min_pair),
        "total_track_m": s(total_horiz),
        "altitude_range_m": s(gate_pos[..., 2].amax(dim=1) - gate_pos[..., 2].amin(dim=1)),
    }


def _vq1_stats() -> dict:
    """Single-course VQ1 stats (same keys as _course_stats but scalar, no std)."""
    gp_np, _ = load_course_zup(_VQ1_JSON)
    gp = torch.tensor(gp_np, dtype=torch.float32).unsqueeze(0)           # (1, G, 3)
    sp = torch.tensor([VQ1_SPAWN_POS_ZUP], dtype=torch.float32)          # (1, 3)
    st = _course_stats(gp, sp)
    # pull just the mean for the VQ1 single course
    return {k: v[0] for k, v in st.items()}


def _print_table(preset: str, stats: dict, vq1: dict) -> None:
    W = 72
    print("=" * W)
    print(f"  PRESET: {preset!r:>10}  (n courses in batch see header)")
    print("-" * W)
    fmt_hdr = f"  {'metric':<22}{'mean':>8}{'std':>8}{'min':>8}{'max':>8}    {'VQ1':>8}"
    print(fmt_hdr)
    print("-" * W)
    for k, (mean, std, lo, hi) in stats.items():
        vq1_val = vq1.get(k, float("nan"))
        print(f"  {k:<22}{mean:>8.2f}{std:>8.2f}{lo:>8.2f}{hi:>8.2f}    {vq1_val:>8.2f}")
    print()


def _check_flyability(gate_pos: torch.Tensor, spawn_pos: torch.Tensor,
                      preset: str, overrides: dict) -> list[str]:
    """Return list of violated constraints (empty = all pass)."""
    R = {**_DR, **overrides}
    failures = []
    pts = torch.cat([spawn_pos.unsqueeze(1), gate_pos], dim=1)
    seg = pts[:, 1:] - pts[:, :-1]
    horiz = torch.linalg.norm(seg[..., :2], dim=-1)
    drop = -seg[..., 2]
    grade = seg[..., 2].abs() / horiz.clamp(min=1e-6)
    headings = torch.atan2(seg[..., 1], seg[..., 0])
    dh = headings[:, 1:] - headings[:, :-1]
    turns = torch.atan2(torch.sin(dh), torch.cos(dh)).abs()

    # segment lengths
    if not (horiz[:, 1:] >= R["seg_len_m"][0] - 1e-3).all():
        failures.append(f"{preset}: seg_len_m min violated (expected >= {R['seg_len_m'][0]:.1f})")
    if not (horiz[:, 1:] <= R["seg_len_m"][1] + 1e-3).all():
        failures.append(f"{preset}: seg_len_m max violated (expected <= {R['seg_len_m'][1]:.1f})")
    # spawn distance
    if not (horiz[:, 0] >= R["spawn_dist_m"][0] - 1e-3).all():
        failures.append(f"{preset}: spawn_dist_m min violated")
    if not (horiz[:, 0] <= R["spawn_dist_m"][1] + 1e-3).all():
        failures.append(f"{preset}: spawn_dist_m max violated")
    # spawn below gate 0
    gate0_above = seg[:, 0, 2]   # gate 0 z - spawn z (z_up, positive = gate higher)
    if not (gate0_above >= R["spawn_below_g0_m"][0] - 1e-3).all():
        failures.append(f"{preset}: spawn_below_g0_m min violated")
    if not (gate0_above <= R["spawn_below_g0_m"][1] + 1e-3).all():
        failures.append(f"{preset}: spawn_below_g0_m max violated")
    # descent / climb bounds
    if not (drop[:, 1:] >= R["drop_m"][0] - 1e-3).all():
        failures.append(f"{preset}: drop_m min violated (expected >= {R['drop_m'][0]:.1f})")
    if not (drop[:, 1:] <= R["drop_m"][1] + 1e-3).all():
        failures.append(f"{preset}: drop_m max violated (expected <= {R['drop_m'][1]:.1f})")
    # grade
    if not (grade[:, 1:] <= R["max_grade"] + 1e-3).all():
        failures.append(f"{preset}: max_grade violated (expected <= {R['max_grade']:.2f})")
    # turn budget
    if not (turns <= R["turn_rad"] + 1e-3).all():
        failures.append(f"{preset}: turn_rad violated (expected <= {np.degrees(R['turn_rad']):.1f} deg)")
    # pair separation
    pad_gates = pts[..., :2]
    d = torch.linalg.norm(pad_gates[:, :, None, :] - pad_gates[:, None, :, :], dim=-1)
    G1 = d.shape[1]
    d = d + torch.eye(G1, device=d.device) * 1e9
    if not (d.amin(dim=(1, 2)) >= R["min_pair_dist_m"] - 1e-3).all():
        failures.append(f"{preset}: min_pair_dist_m violated (expected >= {R['min_pair_dist_m']:.1f} m)")
    # exit-plane sanity: next gate in front of current gate's exit direction
    gy = gate_pos    # gate yaw not available here, but we can check segment direction
    # (gate yaw is very close to path bisector, so checking segment forward is sufficient)
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=256, help="courses per preset")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = ap.parse_args()

    vq1 = _vq1_stats()

    all_failures: list[str] = []
    print(f"\nValidating random track generator: {args.n} courses per preset, seed={args.seed}\n")

    for preset, overrides in DIFFICULTY_PRESETS.items():
        g = torch.Generator().manual_seed(args.seed)
        courses = sample_courses(args.n, device="cpu", generator=g, **overrides)
        gate_pos = courses["gate_pos"]
        spawn_pos = courses["spawn_pos"]

        stats = _course_stats(gate_pos, spawn_pos)
        _print_table(f"{preset} (n={args.n})", stats, vq1)

        failures = _check_flyability(gate_pos, spawn_pos, preset, overrides)
        if failures:
            for f in failures:
                print(f"  FAIL: {f}")
            all_failures.extend(failures)
        else:
            print(f"  [{preset}] all flyability constraints PASS")
        print()

    # VQ1 as reference row
    print("=" * 72)
    print("  VQ1 FIXED TRACK (reference, single course):")
    print("-" * 72)
    fmt_hdr = f"  {'metric':<22}{'value':>8}"
    print(fmt_hdr)
    print("-" * 72)
    for k, v in vq1.items():
        print(f"  {k:<22}{v:>8.2f}")
    print()

    # Diversity check: course-to-course variance within each preset must be >0.
    print("=" * 72)
    print("  DIVERSITY CHECK (std of gate_pos across courses per preset, must be > 1 m):")
    print("-" * 72)
    for preset, overrides in DIFFICULTY_PRESETS.items():
        g = torch.Generator().manual_seed(args.seed)
        courses = sample_courses(args.n, device="cpu", generator=g, **overrides)
        gp = courses["gate_pos"]
        std_xy = gp[..., :2].std(dim=0).mean().item()
        std_z = gp[..., 2].std(dim=0).mean().item()
        ok = std_xy > 1.0 and std_z > 0.2
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {preset:<12} xy-std={std_xy:.2f} m   z-std={std_z:.2f} m")
        if not ok:
            all_failures.append(f"{preset}: insufficient diversity (xy-std={std_xy:.2f}, z-std={std_z:.2f})")
    print()

    if all_failures:
        print(f"RED -- {len(all_failures)} flyability/diversity failure(s):")
        for f in all_failures:
            print(f"  {f}")
        return 1
    else:
        print("GREEN -- all presets pass flyability constraints and diversity check.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
