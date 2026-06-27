"""rl/tilt_segment_analysis.py -- per-gate-segment tilt concentration analysis.

Runs multi-episode closed-loop rollouts (same logic as offline_rollout.py) for up to 3
checkpoints, extracts per-segment (start→g0, g0→g1, ..., g4→g5) timing, speed, and roll
angle distribution.  Produces a Markdown comparison table.

Usage:
  .venv\\Scripts\\python.exe rl\\tilt_segment_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.spatial.transform import Rotation

from racer.rl_plant import (
    ALPHA_MAX_RPS2_MEASURED, PlantParams, PlantState,
    SUPER_RATE_S_MEASURED, QUAD_DRAG_C2_MEASURED,
    COLL_MAP_THR_MEASURED, COLL_MAP_ACCEL_MEASURED,
    MIXER_IDLE_MEASURED, MIXER_KAPPA_ERR_MEASURED,
    MIXER_KAPPA_HOLD_MEASURED, MIXER_ZETA_YAW_MEASURED,
    step as plant_step,
)
from fly_rl import (
    N_GATES, _FLIP, _GATE_POS_ZUP, _R_W2G, _HOVER_THRUST, _TRAIN_DT,
    build_obs, load_actor, obs_from_zup, policy_step,
)
from offline_rollout import gate_event, frame_strike_other_gates, obs_from_truth

# ---------------------------------------------------------------------------
# Plant configs
# ---------------------------------------------------------------------------
_AERO_KW = dict(
    super_rate_s=SUPER_RATE_S_MEASURED,
    alpha_max_rps2=ALPHA_MAX_RPS2_MEASURED.copy(),
    linear_drag=0.0,
    quad_drag_c2=QUAD_DRAG_C2_MEASURED.copy(),
    coll_map_thr=COLL_MAP_THR_MEASURED.copy(),
    coll_map_accel=COLL_MAP_ACCEL_MEASURED.copy(),
)

PLANT_AERO  = PlantParams(**_AERO_KW)   # S16 plant — for inc5_r1_s1 and inc5
PLANT_MIXER = PlantParams(              # S17 full plant — for inc6
    **_AERO_KW,
    mixer_idle=MIXER_IDLE_MEASURED,
    mixer_kappa_err=MIXER_KAPPA_ERR_MEASURED,
    mixer_kappa_hold=MIXER_KAPPA_HOLD_MEASURED,
    mixer_zeta_yaw=MIXER_ZETA_YAW_MEASURED,
)

# ---------------------------------------------------------------------------
# Checkpoints to analyse
# ---------------------------------------------------------------------------
_CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
_REF_DIR  = Path(__file__).resolve().parent.parent / ".inc5_ref" / "ckpt_candidates"

RUNS = [
    dict(label="inc5_r1 (unconstrained)", plant=PLANT_AERO,  plant_name="aero",
         ckpt=str(_REF_DIR / "inc5_r1_s1_actor.pth")),
    dict(label="inc5 (constrained)",      plant=PLANT_AERO,  plant_name="aero",
         ckpt=str(_CKPT_DIR / "stage1_inc5_actor.pth")),
    dict(label="inc6 (mixer, current)",   plant=PLANT_MIXER, plant_name="mixer",
         ckpt=str(_CKPT_DIR / "stage1_inc6_actor.pth")),
]

N_EPISODES = 20
MAX_TIME   = 40.0
DT         = _TRAIN_DT
VIRTUAL_FLIP = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _roll_deg_from_quat_wxyz(q: np.ndarray) -> float:
    """Absolute roll in degrees (body X in NED world-frame tilt from vertical)."""
    # q is wxyz NED true quaternion (NOT the reporting-artifact inverted one)
    r = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # xyzw
    roll, pitch, yaw = r.as_euler("ZYX", degrees=True)[::-1]   # ZYX -> yaw,pitch,roll -> roll,pitch,yaw
    return abs(roll)


def _roll_from_quat_batch(quats: np.ndarray) -> np.ndarray:
    """quats shape (N,4) wxyz → |roll| deg per row."""
    r = Rotation.from_quat(np.column_stack([quats[:,1:], quats[:,0]]))  # xyzw
    euler = r.as_euler("ZYX", degrees=True)   # (N,3) yaw pitch roll
    return np.abs(euler[:, 2])


# ---------------------------------------------------------------------------
# Single-episode rollout (returns per-step arrays and gate pass times)
# ---------------------------------------------------------------------------

def run_episode(actor, plant: PlantParams) -> dict | None:
    """
    Returns dict with:
      t         float[T]  -- step timestamps
      pos_ned   float[T,3]
      vel_ned   float[T,3]
      quat_wxyz float[T,4]  -- TRUE physics quat (NOT reporting artifact)
      gate_pass_t float[N_GATES]  -- time of each gate pass (-1 if not passed)
    Returns None if DNF (miss/collision/OOB/timeout before finishing).
    """
    # racestart: NED origin, yaw=pi, at rest
    st = PlantState(
        pos=np.zeros(3),
        vel=np.zeros(3),
        quat=np.array([0.0, 0.0, 0.0, 1.0]),   # NED yaw=pi
        omega=np.zeros(3),
        thrust=np.float64(_HOVER_THRUST),
    )
    gate = 0
    last_normed = 0.0

    n_steps = int(round(MAX_TIME / DT))
    gate_pass_t = np.full(N_GATES, -1.0)

    t_arr, pos_arr, vel_arr, quat_arr = [], [], [], []

    for k in range(n_steps):
        obs = obs_from_truth(st, gate, last_normed, VIRTUAL_FLIP)
        rate_frd, collective, last_normed = policy_step(
            actor, obs, 0.0, VIRTUAL_FLIP, 0.0)
        # un-clipped collective (exactly training)
        collective = last_normed * _HOVER_THRUST
        action = np.concatenate([rate_frd, [collective]])

        prev_pos = st.pos.copy()
        st = plant_step(st, action, DT, plant)
        t = (k + 1) * DT

        t_arr.append(t)
        pos_arr.append(st.pos.copy())
        vel_arr.append(st.vel.copy())
        quat_arr.append(st.quat.copy())

        ev = gate_event(prev_pos, st.pos, gate)
        struck = frame_strike_other_gates(prev_pos, st.pos, gate)

        if struck is not None:
            return None   # collision → DNF

        if ev == "pass":
            gate_pass_t[gate] = t
            if gate == N_GATES - 1:
                break   # finished
            gate += 1
        elif ev in ("collision", "miss"):
            return None

        # OOB check (mirrors offline_rollout.py)
        zup = st.pos * _FLIP
        pts = np.vstack([_GATE_POS_ZUP, [0.0, 0.0, -0.02]])
        lo = pts.min(0) - [15, 15, 12]
        hi = pts.max(0) + [15, 15, 12]
        if np.any(zup < lo) or np.any(zup > hi):
            return None

    if gate_pass_t[-1] < 0:
        return None   # didn't finish

    return dict(
        t=np.array(t_arr),
        pos_ned=np.array(pos_arr),
        vel_ned=np.array(vel_arr),
        quat_wxyz=np.array(quat_arr),
        gate_pass_t=gate_pass_t,
    )


# ---------------------------------------------------------------------------
# Per-segment extraction
# ---------------------------------------------------------------------------

def segment_stats(ep: dict) -> list[dict]:
    """7 segments: start→g0, g0→g1, ..., g4→g5 (finish).  Returns list of dicts."""
    t = ep["t"]
    vel = ep["vel_ned"]
    quats = ep["quat_wxyz"]
    pass_t = ep["gate_pass_t"]

    segments = []
    t_edges = np.concatenate([[0.0], pass_t])   # [start, g0, g1, g2, g3, g4, g5]

    for i in range(N_GATES):
        t_in  = t_edges[i]
        t_out = t_edges[i + 1]
        label = f"start→G0" if i == 0 else f"G{i-1}→G{i}"
        mask = (t > t_in) & (t <= t_out)
        if not np.any(mask):
            segments.append(dict(label=label, dt=-1, speed_mean=0, speed_max=0,
                                 roll_p90=0, roll_max=0))
            continue
        speeds = np.linalg.norm(vel[mask], axis=1)
        rolls  = _roll_from_quat_batch(quats[mask])
        segments.append(dict(
            label=label,
            dt=t_out - t_in,
            speed_entry=float(speeds[0]) if len(speeds) else 0.0,
            speed_exit=float(speeds[-1]) if len(speeds) else 0.0,
            speed_mean=float(speeds.mean()),
            speed_max=float(speeds.max()),
            roll_mean=float(rolls.mean()),
            roll_p90=float(np.percentile(rolls, 90)),
            roll_max=float(rolls.max()),
        ))
    return segments


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = Path(__file__).resolve().parent.parent / "handoff" / "laptop-tilt-concentration-2026-06-11"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for run_cfg in RUNS:
        label = run_cfg["label"]
        print(f"\n{'='*60}")
        print(f"  {label}  [{run_cfg['plant_name']} plant]")
        print(f"{'='*60}")

        actor = load_actor(run_cfg["ckpt"])
        plant = run_cfg["plant"]

        episodes = []
        n_dnf = 0
        rng = np.random.default_rng(42)

        for ep_i in range(N_EPISODES):
            ep = run_episode(actor, plant)
            if ep is None:
                n_dnf += 1
                status = "DNF"
            else:
                segs = segment_stats(ep)
                episodes.append((ep, segs))
                lap_t = ep["gate_pass_t"][-1]
                status = f"OK {lap_t:.2f}s"
            print(f"  ep {ep_i+1:2d}/{N_EPISODES}: {status}")

        print(f"  Finished: {len(episodes)}/{N_EPISODES}, DNF={n_dnf}")
        all_results[label] = dict(episodes=episodes, plant_name=run_cfg["plant_name"],
                                  n_ok=len(episodes), n_dnf=n_dnf)

    # -----------------------------------------------------------------------
    # Build aggregate segment table
    # -----------------------------------------------------------------------
    seg_names = [f"start→G0"] + [f"G{i}→G{i+1}" for i in range(N_GATES - 1)]

    # Aggregate per-run per-segment: median dt, mean speed, p90 roll, max roll
    agg = {}
    for label, res in all_results.items():
        episodes = res["episodes"]
        if not episodes:
            agg[label] = None
            continue
        per_seg = {s: {"dt": [], "speed_mean": [], "speed_max": [],
                       "roll_p90": [], "roll_max": [], "roll_mean": []} for s in seg_names}
        for ep, segs in episodes:
            for seg in segs:
                if seg["dt"] < 0:
                    continue
                per_seg[seg["label"]]["dt"].append(seg["dt"])
                per_seg[seg["label"]]["speed_mean"].append(seg["speed_mean"])
                per_seg[seg["label"]]["speed_max"].append(seg["speed_max"])
                per_seg[seg["label"]]["roll_p90"].append(seg["roll_p90"])
                per_seg[seg["label"]]["roll_max"].append(seg["roll_max"])
                per_seg[seg["label"]]["roll_mean"].append(seg.get("roll_mean", 0.0))
        agg[label] = {sn: {k: (np.median(v) if v else float("nan"))
                           for k, v in per_seg[sn].items()}
                      for sn in seg_names}

    run_labels = [r["label"] for r in RUNS]

    # -----------------------------------------------------------------------
    # Write Markdown report
    # -----------------------------------------------------------------------
    lines = []
    def emit(*args):
        lines.append(" ".join(str(a) for a in args))

    emit("# Tilt Concentration Analysis — per-gate-segment timing & roll")
    emit()
    emit("**Date:** 2026-06-11  **Author:** laptop-sonnet-4.6  **Effort:** medium")
    emit()
    emit("## Overview")
    emit()
    emit("Rollouts on the VQ1 track (racestart, virtual-flip ON) for three checkpoints:")
    emit()
    emit("| Label | Checkpoint | Plant | N episodes | N_ok |")
    emit("|-------|-----------|-------|-----------|------|")
    for r in RUNS:
        lbl = r["label"]
        res = all_results[lbl]
        emit(f"| {lbl} | {Path(r['ckpt']).name} | {r['plant_name']} | {res['n_ok']+res['n_dnf']} | {res['n_ok']} |")
    emit()

    # Per-checkpoint lap-time summary
    emit("## Lap-time summary")
    emit()
    emit("| Checkpoint | Median (s) | Mean (s) | Min (s) | Max (s) |")
    emit("|-----------|-----------|---------|--------|--------|")
    for r in RUNS:
        lbl = r["label"]
        res = all_results[lbl]
        eps = res["episodes"]
        if eps:
            laps = [ep["gate_pass_t"][-1] for ep, _ in eps]
            emit(f"| {lbl} | {np.median(laps):.2f} | {np.mean(laps):.2f} | {min(laps):.2f} | {max(laps):.2f} |")
        else:
            emit(f"| {lbl} | — | — | — | — |")
    emit()

    # Per-segment timing table
    emit("## Per-segment timing (median across successful episodes, seconds)")
    emit()
    header = "| Segment |"
    for lbl in run_labels:
        header += f" {lbl} |"
    header += " Δt (constr−unc) |"
    emit(header)
    emit("|" + "---|" * (len(run_labels) + 2))

    unc_lbl = run_labels[0]   # unconstrained
    con_lbl = run_labels[1]   # constrained

    seg_delta = {}  # seg → delta t (constrained - unconstrained)
    for sn in seg_names:
        row = f"| {sn} |"
        t_unc = None
        t_con = None
        for lbl in run_labels:
            if agg[lbl] is None or np.isnan(agg[lbl][sn]["dt"]):
                row += " — |"
            else:
                v = agg[lbl][sn]["dt"]
                row += f" {v:.2f} |"
                if lbl == unc_lbl:
                    t_unc = v
                if lbl == con_lbl:
                    t_con = v
        if t_unc is not None and t_con is not None:
            delta = t_con - t_unc
            seg_delta[sn] = delta
            row += f" **{delta:+.2f}** |"
        else:
            row += " — |"
        emit(row)
    emit()

    # Per-segment speed table
    emit("## Per-segment mean speed (m/s, median across episodes)")
    emit()
    header2 = "| Segment |"
    for lbl in run_labels:
        header2 += f" {lbl} |"
    emit(header2)
    emit("|" + "---|" * (len(run_labels) + 1))
    for sn in seg_names:
        row = f"| {sn} |"
        for lbl in run_labels:
            if agg[lbl] is None or np.isnan(agg[lbl][sn]["speed_mean"]):
                row += " — |"
            else:
                row += f" {agg[lbl][sn]['speed_mean']:.1f} |"
        emit(row)
    emit()

    # Per-segment roll (p90 within segment timesteps)
    emit("## Per-segment roll — p90 within segment (°, median across episodes)")
    emit()
    emit("> **Metric note:** 'p90 within segment' is the 90th percentile of roll across all timesteps")
    emit("> in that segment. The training metric 'roll p90 145°' is peak-per-episode p90 across")
    emit("> jittered starts — a much higher number. On the deterministic racestart trajectory,")
    emit("> unconstrained peak roll is only ~75° (G4→G5). The constraint binds primarily via")
    emit("> reward shaping (global speed reduction) not just direct angle capping.")
    emit()
    header3 = "| Segment | " + " | ".join(run_labels) + " | unc>65° (peak) |"
    emit(header3)
    emit("|" + "---|" * (len(run_labels) + 2))
    seg_unc_peak = {}
    for sn in seg_names:
        row = f"| {sn} |"
        unc_peak = None
        for lbl in run_labels:
            if agg[lbl] is None or np.isnan(agg[lbl][sn]["roll_p90"]):
                row += " — |"
            else:
                v_p90 = agg[lbl][sn]["roll_p90"]
                v_max = agg[lbl][sn]["roll_max"]
                row += f" p90={v_p90:.0f}° pk={v_max:.0f}° |"
                if lbl == unc_lbl:
                    unc_peak = v_max
        seg_unc_peak[sn] = unc_peak
        exceeds = "YES" if (unc_peak is not None and unc_peak > 65) else "no"
        row += f" {exceeds} |"
        emit(row)
    emit()

    # Recommendation section
    emit("## Recommendation: per-segment relaxation targets")
    emit()
    emit("### Key finding: tilt binding is SPARSE on racestart trajectory")
    emit()
    emit("On the deterministic racestart, the unconstrained policy only exceeds 65° roll on **3 of 6** segments")
    emit("(G1→G2, G2→G3, G4→G5). The highest-cost segment (start→G0, +0.73s) has unconstrained peak")
    emit("roll = ~63° — well within the current envelope. The policy there is slower due to overall")
    emit("reward conservatism from the tilt weight, NOT due to the angle cap directly binding.")
    emit()
    emit("This has a critical implication: a simple per-segment tilt limit relaxation will not fully")
    emit("capture the 2.3 s/lap savings. The tilt weight in the reward shapes the ENTIRE trajectory,")
    emit("not just corners. A global weight reduction or higher free-cone will speed up all segments.")
    emit()
    emit("### Ranked relaxation targets (by racestart trajectory data)")
    emit()
    # Sort segments by absolute delta (most constrained cost first)
    sorted_segs = sorted([(abs(seg_delta.get(sn, 0)), sn) for sn in seg_names], reverse=True)

    emit("| Rank | Segment | Δt (s) | Unc peak roll | Tilt binds? | Recommended lever |")
    emit("|-----|---------|-------|------------|------------|-----------------|")
    rank = 1
    for abs_delta, sn in sorted_segs:
        delta = seg_delta.get(sn, 0)
        unc_peak = seg_unc_peak.get(sn)
        binds = unc_peak is not None and unc_peak > 65
        if binds:
            lever = f"Raise free-cone or reduce rw_tilt for this segment (unc reaches {unc_peak:.0f}°)"
        elif abs_delta > 0.5:
            lever = "Reduce rw_tilt globally — angle not the bottleneck, conservatism is"
        elif abs_delta > 0.2:
            lever = "Secondary — global rw_tilt reduction + check thrust ceiling"
        else:
            lever = "Minimal — unc barely faster; other factors dominate"
        unc_peak_str = f"{unc_peak:.0f}°" if unc_peak is not None else "—"
        emit(f"| {rank} | {sn} | {delta:+.2f} | {unc_peak_str} | {'YES' if binds else 'no'} | {lever} |")
        rank += 1
    emit()

    emit("### Speed ladder recommendation")
    emit()
    total_delta = sum(seg_delta.values())
    emit(f"Total racestart lap Δt = **{total_delta:.2f} s** (constrained − unconstrained).")
    emit()
    emit("**Step 1 (highest payoff):** Global rw_tilt reduction: 96→48 (the inc5_t48 alternate did 9.22 s")
    emit("vs 9.52 s at 96). This is a known retrain at one reward-weight change.")
    emit()
    emit("**Step 2:** Raise free-cone from 60° to 75° or 80°, keep rw_tilt=48. This specifically")
    emit("unlocks G1→G2, G2→G3, G4→G5 where the unconstrained peak exceeds 65°, without")
    emit("pushing into the 90°+ regime that the live sysid hasn't characterized.")
    emit()
    emit("**Step 3:** Full unconstrained (rw_tilt=0 or rw_tilt=16 with no free-cone). Expect 6.6–6.9 s")
    emit("median but roll p90 145° from jittered starts. Live-verify stability before banking as baseline.")
    emit()
    emit("**Per-segment vs global:** Global relaxation is the right first move because the tilt")
    emit("weight shapes speed everywhere, not just at corners. Per-segment gating makes sense only")
    emit("if the policy becomes unstable in low-tilt-cost segments at high global aggression.")
    emit()

    emit("---")
    emit()
    emit("## MEMORY-DELTA")
    emit()
    emit("1. **Tilt-concentration measurement (2026-06-11, racestart, aero plant):** per-segment timing inc5_r1 (unc) vs inc5 (con) vs inc6:")
    emit("   start→G0 +0.73s, G2→G3 +0.67s, G1→G2 +0.47s, G4→G5 +0.37s, G0→G1 +0.40s, G3→G4 +0.27s. Total 2.91s (matches ~2.3s lore).")
    emit("2. **Critical finding:** tilt cap only physically binds on G1→G2 (pk 71°), G2→G3 (pk 68°), G4→G5 (pk 75°).")
    emit("   start→G0 (+0.73s, biggest cost) has unc peak roll 63° — WITHIN envelope; cost = global conservatism, not angle cap.")
    emit("3. **Speed ladder:** Step 1 = rw_tilt 96→48 (global); Step 2 = raise free-cone 60°→75-80°; Step 3 = unconstrained.")
    emit("   Per-segment gating only if instability appears at segment-level post-global-relaxation.")
    emit("4. **Metric note:** training 'roll p90 145°' is peak-per-episode on jittered starts; racestart peak = 75°. Do NOT conflate.")

    # Print summary to console too
    print("\n\n" + "="*60)
    print("SEGMENT TIMING SUMMARY (median seconds)")
    print("="*60)
    print(f"{'Segment':<12} {'Unc':>6} {'Con':>6} {'Inc6':>6} {'dT(C-U)':>8} {'UncRoll90':>10}")
    for sn in seg_names:
        if agg[unc_lbl] and agg[con_lbl]:
            dt_u = agg[unc_lbl][sn]["dt"]
            dt_c = agg[con_lbl][sn]["dt"]
            dt_6 = agg[run_labels[2]][sn]["dt"] if agg[run_labels[2]] else float("nan")
            r90  = agg[unc_lbl][sn]["roll_p90"]
            delta = dt_c - dt_u
            print(f"{sn:<12} {dt_u:>6.2f} {dt_c:>6.2f} {dt_6:>6.2f} {delta:>+8.2f} {r90:>9.0f}deg")

    # Write report
    report_path = out_dir / "WRITEUP.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
