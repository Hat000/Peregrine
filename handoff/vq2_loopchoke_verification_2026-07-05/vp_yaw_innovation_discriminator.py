"""vp_yaw_innovation_discriminator.py — is the 18 deg median vp_yaw innovation REAL ESKF yaw drift
(A) or a TIME-MISALIGNMENT ARTIFACT of the end-anchor clock mapping (B)?

Coordinator's challenge: the replay's end-anchor mapping (nav vs video on different clocks, anchored
by both ending at disarm) can be off by 100s of ms. Yaw sweeps up to ~86 deg/s in turns, so a
300 ms match error manufactures ~26 deg of FAKE innovation. Roll/pitch being slow doesn't cover yaw.

DISCRIMINATORS (this script):
 1. Split accepted innovations by |raw_gyro_yaw| at the matched nav record:
      quasi-static |g| < 0.1 rad/s   (immune to the artifact — yaw isn't moving)
      medium       0.1 <= |g| <= 0.5
      turning      |g| > 0.5 rad/s   (max artifact exposure)
    Quasi-static innovations ~1-3 deg  -> verdict B (artifact; live corrections are small trims).
    Quasi-static innovations 10 deg+   -> verdict A (real drift; vp_yaw is load-bearing).
 2. Innovation vs |yaw rate| bucket table (artifact signature = innovation scales with rate).
 3. LAG SWEEP: recompute the nearest-branch innovation for every quality-passing frame while
    shifting the video->nav time mapping by tau in [-1.5 s, +1.5 s]. If some nonzero tau collapses
    the median innovation, that tau IS the clock misalignment (direct artifact measurement, stronger
    than the +/-300 ms spot check requested). The branch selection is redone per tau exactly as the
    live gate would (nearest branch to the shifted yaw_hat, 35 deg cap).

Per-frame results (incl. the 4 branch headings) are persisted to perframe_<run>.jsonl so this and
future follow-ups do not need to re-decode video.

CPU-ONLY, no torch/YOLO. Reuses the exact gate logic + end-anchor mapping of replay_backstop_gates.py.
"""
from __future__ import annotations

import bisect
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
OUT_DIR = Path(__file__).resolve().parent

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from racer.vision.heading_vp import estimate_heading  # noqa: E402

VP_YAW_MIN_QUALITY = 0.30
VP_YAW_BRANCH_MAX_RAD = float(np.deg2rad(35.0))
VP_YAW_RANSAC_ITERS = 256
STRIDE = 3
MAX_FRAMES = 1200


def _wrap_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def load_run(run: Path):
    nav = [json.loads(l) for l in (run / "nav_estimate.jsonl").read_text().splitlines() if l.strip()]
    nav.sort(key=lambda r: int(r["sim_time_ns"]))
    nt = [int(r["sim_time_ns"]) for r in nav]
    idx = [json.loads(l) for l in (run / "video_index.jsonl").read_text().splitlines() if l.strip()]
    idx.sort(key=lambda r: int(r["sim_time_ns"]))
    return nav, nt, idx


def nearest_i(nt, t):
    i = bisect.bisect_left(nt, t)
    if i <= 0:
        return 0
    if i >= len(nt):
        return len(nt) - 1
    return i - 1 if (t - nt[i - 1]) <= (nt[i] - t) else i


def replay_vp(run: Path):
    """Decode + estimate_heading per sampled armed-window frame; persist per-frame records."""
    nav, nt, idx = load_run(run)
    blob = (run / "video.bin").read_bytes()
    nav_start, nav_end = nt[0], nt[-1]
    vend = int(idx[-1]["sim_time_ns"])

    mapped = []
    for rec in idx:
        target = nav_end - (vend - int(rec["sim_time_ns"]))
        if target < nav_start - 200_000_000:
            continue
        mapped.append((rec, max(target, nav_start)))
    sampled = mapped[::STRIDE][:MAX_FRAMES]

    out = []
    for rec, target in sampled:
        jpg = blob[rec["offset"]:rec["offset"] + rec["length"]]
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        ni = nearest_i(nt, target)
        nr = nav[ni]
        roll = float(nr.get("roll_rad") or 0.0)
        pitch = float(nr.get("pitch_rad") or 0.0)
        est = estimate_heading(img, roll, pitch, ransac_iters=VP_YAW_RANSAC_ITERS)
        out.append({
            "frame_id": rec["frame_id"],
            "target_nav_ns": int(target),
            "mean_gray": float(img.mean()),
            "nav_yaw_rad": float(nr.get("yaw_rad") or 0.0),
            "raw_gyro_yaw": float(nr.get("raw_gyro_yaw") or 0.0),
            "quality": None if est is None else float(est.quality),
            "branches_rad": None if est is None else [float(b) for b in est.branch_headings_rad],
        })
    pf = OUT_DIR / f"perframe_{run.name}.jsonl"
    with open(pf, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"[persisted] {pf}  ({len(out)} frames)")
    return out, nav, nt


def innovation_at(branches, yaw_hat):
    """Live gate's branch pick: nearest branch to yaw_hat; return signed innovation (rad) or None
    if branch-rejected (>35 deg)."""
    diffs = np.array([_wrap_pi(b - yaw_hat) for b in branches])
    k = int(np.argmin(np.abs(diffs)))
    if abs(diffs[k]) > VP_YAW_BRANCH_MAX_RAD:
        return None
    return float(diffs[k])


def analyze(run_name, frames, nav, nt):
    print(f"\n===== {run_name} =====")
    lines = [f"### {run_name}", ""]

    # --- 1+2: accepted innovations bucketed by |raw_gyro_yaw| at the matched record -------------
    buckets = {"quasi-static |g|<0.1": [], "medium 0.1-0.5": [], "turning |g|>0.5": []}
    for r in frames:
        if r["quality"] is None or r["quality"] < VP_YAW_MIN_QUALITY:
            continue
        inn = innovation_at(r["branches_rad"], r["nav_yaw_rad"])
        if inn is None:
            continue
        g = abs(r["raw_gyro_yaw"])
        b = ("quasi-static |g|<0.1" if g < 0.1 else
             "medium 0.1-0.5" if g <= 0.5 else "turning |g|>0.5")
        buckets[b].append(abs(np.rad2deg(inn)))

    hdr = f"{'bucket':24s} {'n':>5s} {'median':>8s} {'mean':>8s} {'p95':>8s} {'max':>8s}"
    print(hdr); lines += ["ACCEPTED innovation |deg| by yaw-rate bucket:", "", "```", hdr]
    for b, xs in buckets.items():
        if xs:
            a = np.asarray(xs)
            row = (f"{b:24s} {len(a):5d} {np.median(a):8.2f} {a.mean():8.2f} "
                   f"{np.percentile(a,95):8.2f} {a.max():8.2f}")
        else:
            row = f"{b:24s} {0:5d}     (none)"
        print(row); lines.append(row)
    lines += ["```", ""]

    # --- 3: LAG SWEEP — median nearest-branch |innovation| vs time-mapping shift tau -------------
    # Use ALL quality-passing frames (not only originally-accepted) to dodge the 35deg-cap
    # truncation bias. Branch re-picked per tau exactly as live would.
    qual = [r for r in frames if r["quality"] is not None and r["quality"] >= VP_YAW_MIN_QUALITY]
    taus = np.arange(-1.5, 1.501, 0.1)
    med_by_tau = []
    acc_by_tau = []
    for tau in taus:
        dt_ns = int(tau * 1e9)
        inns = []
        n_acc = 0
        for r in qual:
            ni = nearest_i(nt, r["target_nav_ns"] + dt_ns)
            yaw_hat = float(nav[ni].get("yaw_rad") or 0.0)
            inn = innovation_at(r["branches_rad"], yaw_hat)
            if inn is not None:
                inns.append(abs(np.rad2deg(inn)))
                n_acc += 1
        med_by_tau.append(np.median(inns) if inns else np.nan)
        acc_by_tau.append(n_acc)
    best_i = int(np.nanargmin(med_by_tau))
    print(f"\nLAG SWEEP (median |innovation| deg of nearest-branch pick, {len(qual)} quality-passing frames):")
    lines += [f"LAG SWEEP over time-mapping shift tau ({len(qual)} quality-passing frames):", "", "```",
              f"{'tau_s':>7s} {'median_inn_deg':>15s} {'n_accepted':>11s}"]
    for t_, m, n_ in zip(taus, med_by_tau, acc_by_tau):
        mark = "  <-- min" if int(np.where(taus == t_)[0][0]) == best_i else ""
        row = f"{t_:7.1f} {m:15.2f} {n_:11d}{mark}"
        if abs(t_ - round(t_ * 2) / 2) < 1e-9 or int(np.where(taus == t_)[0][0]) == best_i:
            print(row)  # console: every 0.5s + the min; file gets all
        lines.append(row)
    lines += ["```", ""]
    tau_best = taus[best_i]
    m0 = med_by_tau[int(np.where(np.isclose(taus, 0.0))[0][0])]
    print(f"  tau=0: median {m0:.2f} deg   BEST tau={tau_best:+.1f}s: median {med_by_tau[best_i]:.2f} deg")
    lines.append(f"tau=0 median = {m0:.2f} deg; best tau = {tau_best:+.1f}s with median {med_by_tau[best_i]:.2f} deg")
    lines.append("")

    # --- spot check: +/-300 ms sensitivity on quasi-static vs turning accepted frames ------------
    sens = {"quasi-static": [], "turning": []}
    for r in qual:
        inn0 = innovation_at(r["branches_rad"], r["nav_yaw_rad"])
        if inn0 is None:
            continue
        g = abs(r["raw_gyro_yaw"])
        key = "quasi-static" if g < 0.1 else ("turning" if g > 0.5 else None)
        if key is None:
            continue
        deltas = []
        for dt in (-0.3, 0.3):
            ni = nearest_i(nt, r["target_nav_ns"] + int(dt * 1e9))
            inn = innovation_at(r["branches_rad"], float(nav[ni].get("yaw_rad") or 0.0))
            if inn is not None:
                deltas.append(abs(np.rad2deg(inn - inn0)))
        if deltas:
            sens[key].append(max(deltas))
    print("\n+/-300 ms mapping-shift sensitivity of accepted innovations (max |change|, deg):")
    lines.append("+/-300 ms mapping-shift sensitivity (max |innovation change|, deg):")
    lines.append("")
    lines.append("```")
    for k, xs in sens.items():
        if xs:
            a = np.asarray(xs)
            row = f"{k:14s} n={len(a):4d}  median {np.median(a):6.2f}  p95 {np.percentile(a,95):6.2f}"
        else:
            row = f"{k:14s} (none)"
        print("  " + row); lines.append(row)
    lines += ["```", ""]
    return lines


def main():
    runs = [REPO / "data/runs/20260705_001321_rl_s1_f1",   # choked, lit throughout (main evidence)
            REPO / "data/runs/20260705_002939_rl_s1_f1"]   # clean (mostly dark; lit head)
    all_lines = ["# vp_yaw innovation discriminator — real drift (A) vs time-misalignment artifact (B)",
                 "", f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}. See module docstring for design.", ""]
    for run in runs:
        pf = OUT_DIR / f"perframe_{run.name}.jsonl"
        if pf.exists():
            frames = [json.loads(l) for l in pf.read_text().splitlines() if l.strip()]
            nav, nt, _ = load_run(run)
            print(f"[reuse] {pf} ({len(frames)} frames)")
        else:
            t0 = time.time()
            frames, nav, nt = replay_vp(run)
            print(f"[replayed] {run.name} in {time.time()-t0:.1f}s")
        all_lines += analyze(run.name, frames, nav, nt)
    out = OUT_DIR / "INNOVATION_DISCRIMINATOR.md"
    out.write_text("\n".join(all_lines), encoding="utf-8")
    print(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
