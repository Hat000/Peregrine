"""doctrine_probes.py -- LAPTOP-TRAINING-DOCTRINE computational probes (2026-06-12).

Offline-twin computations behind the Part-1 question ledger. All probes roll the
inc6 actor against the fully-measured S17 mixer plant (racer.rl_plant) through
fly_rl's exact obs/action pipeline -- the same machinery as rl/offline_rollout.py,
extended with full-state traces, in-bin residual-force injection, global force-DR
scaling, and per-gate pass-offset capture.

Subcommands:
  q1     crab aerodynamics: observed vs drag-optimal orientation about the thrust
         axis at cruise; yaw-budget for coordinated flight; corner-tax exposure.
  q2q6   regime-binned residual injection (the measured +2-2.8 m/s^2 climb-bin
         gap, sign chosen to reproduce live std_f1): gate-3 outcome, terminal
         command-saturation forensics (Q2 analog), standing-vs-handoff bin
         exposure (Q6).
  q3     pass-offset distributions under start jitter x latency (margin audit).
  q4     global force-model scaling sweep (the dr_aero DR axes, applied as
         deterministic offsets): does global-scale robustness exist, and does it
         cover a regime-LOCAL error?

Usage:
  .venv\\Scripts\\python.exe handoff\\laptop-training-doctrine-2026-06-12\\scripts\\doctrine_probes.py q1
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "rl"))

import numpy as np

from racer.rl_plant import (ALPHA_MAX_RPS2_MEASURED, PlantParams, PlantState,
                            SUPER_RATE_S_MEASURED, QUAD_DRAG_C2_MEASURED,
                            COLL_MAP_THR_MEASURED, COLL_MAP_ACCEL_MEASURED,
                            MIXER_IDLE_MEASURED, MIXER_KAPPA_ERR_MEASURED,
                            MIXER_KAPPA_HOLD_MEASURED, MIXER_ZETA_YAW_MEASURED,
                            quat_rotate, quat_rotate_inverse, step as plant_step)
from fly_rl import N_GATES, _FLIP, _GATE_POS_ZUP, _R_W2G, _HOVER_THRUST, _TRAIN_DT, load_actor, policy_step
from offline_rollout import obs_from_truth, make_start, _RATE_SIGN_LIVE, _HALF_OPEN, _HALF_OUTER

CKPT = str(_ROOT / "rl" / "checkpoints" / "stage1_inc6_actor.pth")
DT = _TRAIN_DT
_GATE_POS_NED = _GATE_POS_ZUP * _FLIP


def mixer_params(latency: int = 0, coll_k: float = 1.0, c2_scale: float = 1.0) -> PlantParams:
    """The fully-measured S17 mixer plant (offline_rollout --plant mixer), with optional
    GLOBAL force scaling shaped exactly like the dr_aero resample: collective-table
    deltas-from-hover scale by coll_k (hover point pinned), quad-drag c2 by c2_scale."""
    kvals = COLL_MAP_ACCEL_MEASURED.copy()
    if coll_k != 1.0:
        k_hov = np.interp(0.2656, COLL_MAP_THR_MEASURED, COLL_MAP_ACCEL_MEASURED)
        kvals = k_hov + coll_k * (kvals - k_hov)
    return PlantParams(
        transport_delay_steps=latency,
        rate_sign=_RATE_SIGN_LIVE.copy(),
        super_rate_s=SUPER_RATE_S_MEASURED,
        alpha_max_rps2=ALPHA_MAX_RPS2_MEASURED.copy(),
        linear_drag=0.0,
        quad_drag_c2=QUAD_DRAG_C2_MEASURED.copy() * c2_scale,
        coll_map_thr=COLL_MAP_THR_MEASURED.copy(),
        coll_map_accel=kvals,
        mixer_idle=MIXER_IDLE_MEASURED,
        mixer_kappa_err=MIXER_KAPPA_ERR_MEASURED,
        mixer_kappa_hold=MIXER_KAPPA_HOLD_MEASURED,
        mixer_zeta_yaw=MIXER_ZETA_YAW_MEASURED)


def gate_cross(prev_ned, cur_ned, gate):
    """(event, y, z) at the interpolated crossing of the TARGET gate plane, else (None,0,0)."""
    prev_rel = _R_W2G @ (prev_ned * _FLIP - _GATE_POS_ZUP[gate])
    cur_rel = _R_W2G @ (cur_ned * _FLIP - _GATE_POS_ZUP[gate])
    fwd = prev_rel[0] < 0.0 and cur_rel[0] >= 0.0
    bwd = prev_rel[0] > 0.0 and cur_rel[0] <= 0.0
    if not (fwd or bwd):
        return None, 0.0, 0.0
    f = -prev_rel[0] / ((cur_rel[0] - prev_rel[0]) or 1e-9)
    y = prev_rel[1] + f * (cur_rel[1] - prev_rel[1])
    z = prev_rel[2] + f * (cur_rel[2] - prev_rel[2])
    linf = max(abs(y), abs(z))
    if fwd and linf < _HALF_OPEN:
        return "pass", y, z
    if _HALF_OPEN <= linf <= _HALF_OUTER:
        return "collision", y, z
    return ("miss", y, z) if fwd else (None, 0.0, 0.0)


def frame_strike_other(prev_ned, cur_ned, target):
    for g in range(N_GATES):
        if g == target:
            continue
        ev, _, _ = gate_cross(prev_ned, cur_ned, g)
        if ev == "collision":
            return g
    return None


def tilt_of(quat):
    b3 = quat_rotate(quat, np.array([0.0, 0.0, -1.0]))  # body up, world NED
    return float(np.degrees(np.arccos(np.clip(-b3[2], -1.0, 1.0))))


def in_bin(st, v_lo=12.0, v_hi=18.0, tilt_lo=35.0, tilt_hi=90.0):
    v = float(np.linalg.norm(st.vel))
    return v_lo <= v <= v_hi and tilt_lo <= tilt_of(st.quat) <= tilt_hi


def rollout(actor, start_kind="simstart", latency=0, coll_k=1.0, c2_scale=1.0,
            residual_ned=None, residual_bin=None, jitter_rng=None, max_time=40.0,
            handoff_speed=10.0, handoff_dist=3.0, coll_mult_inbin=1.0):
    """Closed-loop rollout, full-state trace. residual_ned: (3,) extra world accel
    applied while in residual_bin (dict of in_bin kwargs). jitter_rng: pose jitter
    on the start (matches the deploy-matrix scale: pos +-0.3 m, vel +-0.2 m/s)."""
    args = argparse.Namespace(gate=0, handoff_dist=handoff_dist,
                              handoff_speed=handoff_speed, thrust0=-1.0)
    st, gate = make_start(start_kind, args)
    if latency > 0:  # seed the transport buffer with hover (PlantState.hover pattern)
        hov = np.zeros((latency, 4)); hov[:, 3] = _HOVER_THRUST
        st = replace(st, act_buf=hov)
    if jitter_rng is not None:
        st = replace(st, pos=st.pos + jitter_rng.uniform(-0.3, 0.3, 3),
                     vel=st.vel + jitter_rng.uniform(-0.2, 0.2, 3))
    params = mixer_params(latency, coll_k, c2_scale)
    # in-bin MULTIPLICATIVE collective deficit: model error that also weakens corrections
    params_bin = None
    if coll_mult_inbin != 1.0:
        params_bin = mixer_params(latency, coll_k, c2_scale)
        params_bin = replace(params_bin,
                             coll_map_accel=params_bin.coll_map_accel * coll_mult_inbin)
    last_normed = 0.0
    n_steps = int(round(max_time / DT))
    trace = dict(t=[], pos=[], vel=[], quat=[], omega=[], act=[], normed=[], gate=[], inbin=[])
    passes = []  # (gate, y, z, t)
    outcome = "TIMEOUT"
    res = np.zeros(3) if residual_ned is None else np.asarray(residual_ned, float)
    bin_kw = residual_bin or {}
    for k in range(n_steps):
        obs = obs_from_truth(st, gate, last_normed, virtual_flip=True)
        rate_frd, collective, last_normed = policy_step(actor, obs, 0.0, True, 0.0)
        collective = last_normed * _HOVER_THRUST
        action = np.concatenate([rate_frd, [collective]])
        prev_pos = st.pos.copy()
        hit_bin = bool((residual_ned is not None or params_bin is not None)
                       and in_bin(st, **bin_kw))
        st = plant_step(st, action, DT,
                        params_bin if (params_bin is not None and hit_bin) else params)
        if hit_bin:  # extra world accel integrated exactly like the plant's semi-implicit step
            st = replace(st, vel=st.vel + res * DT, pos=st.pos + res * DT * DT)
        t = (k + 1) * DT
        trace["t"].append(t); trace["pos"].append(st.pos.copy()); trace["vel"].append(st.vel.copy())
        trace["quat"].append(st.quat.copy()); trace["omega"].append(st.omega.copy())
        trace["act"].append(action); trace["normed"].append(last_normed)
        trace["gate"].append(gate); trace["inbin"].append(hit_bin)
        struck = frame_strike_other(prev_pos, st.pos, gate)
        if struck is not None:
            outcome = f"COLLISION(g{struck})"
            break
        ev, y, z = gate_cross(prev_pos, st.pos, gate)
        if ev == "pass":
            passes.append((gate, y, z, t))
            if gate == N_GATES - 1:
                outcome = "FINISHED"
                break
            gate += 1
        elif ev == "collision":
            passes.append((gate, y, z, t))
            outcome = f"COLLISION(g{gate} off=[{y:+.2f},{z:+.2f}])"
            break
        elif ev == "miss":
            outcome = f"MISS(g{gate})"
            break
        zup = st.pos * _FLIP
        pts = np.vstack([_GATE_POS_ZUP, [0.0, 0.0, -0.02]])
        if np.any(zup < pts.min(0) - [15, 15, 12]) or np.any(zup > pts.max(0) + [15, 15, 12]):
            outcome = "OOB"
            break
    for k in trace:
        trace[k] = np.asarray(trace[k])
    return outcome, passes, trace


# --------------------------------------------------------------------------------------- Q1
def drag_body(v_b, c2):
    c = np.where(v_b >= 0.0, c2[:, 0], c2[:, 1])
    return -(c * np.abs(v_b) * v_b)            # body-frame drag specific force


def q1(actor):
    print("=" * 88)
    print("Q1 -- crab aerodynamics on the measured plant (twin racestart, mixer plant)")
    outcome, passes, tr = rollout(actor, "racestart")
    print(f"[rollout] {outcome}, gates {[p[0] for p in passes]}, t_end={tr['t'][-1]:.2f}s")
    c2 = QUAD_DRAG_C2_MEASURED
    rows = []
    for i in range(len(tr["t"])):
        v = tr["vel"][i]; q = tr["quat"][i]
        sp = float(np.linalg.norm(v))
        if sp < 12.0:
            continue
        v_b = quat_rotate_inverse(q, v)
        ss = float(np.degrees(np.arctan2(v_b[1], v_b[0])))
        d_obs = drag_body(v_b, c2)
        vhat_b = v_b / sp
        a_track_obs = float(d_obs @ vhat_b)     # along-track drag decel (negative)
        # sweep rotation about the thrust axis (body z): v_b' = Rz(phi)^T v_b
        best = (0.0, a_track_obs)
        for phi in np.linspace(-np.pi, np.pi, 181):
            cph, sph = np.cos(phi), np.sin(phi)
            vb2 = np.array([cph * v_b[0] + sph * v_b[1],
                            -sph * v_b[0] + cph * v_b[1], v_b[2]])
            a_tr = float(drag_body(vb2, c2) @ (vb2 / sp))
            if a_tr > best[1]:                  # least decelerating
                best = (float(np.degrees(phi)), a_tr)
        rows.append((tr["t"][i], sp, ss, tilt_of(q), -a_track_obs, -best[1], best[0],
                     float(np.linalg.norm(d_obs))))
    rows = np.array(rows)
    m = rows[:, 1] >= 15.0
    print(f"\ncruise ticks |v|>=15: n={m.sum()}  (12-15 m/s: n={(~m).sum()})")
    for lab, sel in (("|v|>=15", m), ("12<=|v|<15", ~m)):
        if sel.sum() == 0:
            continue
        r = rows[sel]
        print(f"  [{lab}] sideslip mean {r[:,2].mean():+.1f} deg  tilt mean {r[:,3].mean():.1f} deg")
        print(f"          along-track drag decel: observed {r[:,4].mean():.2f} m/s^2 "
              f"vs thrust-axis-rotation optimum {r[:,5].mean():.2f} m/s^2 "
              f"(excess {(r[:,4]-r[:,5]).mean():.2f} m/s^2, "
              f"{100*((r[:,4]-r[:,5])/np.maximum(r[:,4],1e-9)).mean():.0f}%)")
        print(f"          optimal extra body-z rotation: mean {r[:,6].mean():+.0f} deg "
              f"(0 = current orientation already optimal)")
    # equilibrium-speed estimate: at drag-bound cruise dT/T ~ dD/(2D)
    r = rows[m]
    dD = (r[:, 4] - r[:, 5]).mean(); D = r[:, 4].mean()
    print(f"\n  drag-equilibrium lap-time sensitivity: dT/T ~ dD/(2D) = {dD/(2*D)*100:.1f}% "
          f"(on the ~9 s twin RL segment: ~{dD/(2*D)*9:.2f} s)")
    # yaw budget for coordinated flight: heading must track course direction
    vel = tr["vel"]; spd = np.linalg.norm(vel, axis=1)
    sel = spd >= 12.0
    crs = np.unwrap(np.arctan2(vel[:, 1], vel[:, 0]))
    crs_rate = np.abs(np.gradient(crs, DT))
    print(f"  coordinated-flight yaw budget: course-rate (|v|>=12) "
          f"mean {crs_rate[sel].mean():.2f} rad/s, p95 {np.percentile(crs_rate[sel],95):.2f} rad/s "
          f"(yaw cmd cap 3.14 -> realized ~x2.23 map; alpha_max yaw 80 rad/s^2)")
    # reward exposure of the observed style: rw_rate cost + corner tax along the lap
    w = tr["omega"]; wn = np.linalg.norm(w, axis=1)
    a_thr_span = tr["normed"] / 3.765
    a_rate_span = (tr["act"][:, 0:3] + 3.14) / 6.28
    tax = np.abs(a_thr_span - 0.5) * np.linalg.norm(2 * (a_rate_span - 0.5), axis=1)
    print(f"  style reward exposure (whole lap): rw_rate 0.05*||w|| mean/step = {0.05*wn.mean():.3f}; "
          f"corner-tax factor mean = {tax.mean():.3f} (weight c16)")
    # what would coordinated flight cost in rw_rate? course rate carried as sustained yaw rate:
    print(f"  coordinated alternative adds sustained |yaw rate| ~= course-rate -> extra rw_rate "
          f"~= {0.05*crs_rate[sel].mean():.3f}/step vs observed {0.05*wn.mean():.3f}/step total")
    np.savez(str(Path(__file__).parent / "q1_trace.npz"), **tr)


# ----------------------------------------------------------------------------------- Q2/Q6
def q2q6(actor):
    print("=" * 88)
    print("Q2/Q6 -- regime-binned residual injection (climb-bin gap emulation, mixer plant)")
    # Sign to reproduce live std_f1 vs twin: live ends HIGH (-D) and SHORT (+N).
    # Magnitudes from frame_residual_report std_f1: N+2.79 D+2.36 in the 12-18 x tilt-35-90 bin.
    for tag, res in [("nominal", None),
                     ("res_1.0x", np.array([+2.79, 0.0, -2.36]) * 0.5),
                     ("res_full", np.array([+2.79, 0.0, -2.36])),
                     ("res_1.2x", np.array([+2.79, 0.0, -2.36]) * 1.2)]:
        for start in ("simstart", "handoff"):
            kw = dict(handoff_speed=10.0, handoff_dist=3.0) if start == "handoff" else {}
            outcome, passes, tr = rollout(actor, start, residual_ned=res,
                                          residual_bin={}, **kw)
            nbin = int(tr["inbin"].sum())
            g3 = [p for p in passes if p[0] == 3]
            g3s = f"g3 off=[{g3[0][1]:+.2f},{g3[0][2]:+.2f}]" if g3 else "g3 not reached/--"
            print(f"  [{tag:9s} {start:8s}] {outcome:34s} bin-ticks={nbin:3d} "
                  f"({nbin*DT:.2f}s) {g3s}")
            if res is not None and start == "simstart" and not outcome.startswith("FINISHED"):
                # Q2 forensics: final 1.5 s -- saturated fight or passive drift?
                n = len(tr["t"]); w0 = max(0, n - 45)
                rates = tr["act"][w0:, 0:3]; thr = tr["normed"][w0:]
                sat_rate = (np.abs(rates) > 0.95 * 3.14).mean()
                print(f"      [Q2 forensics last 1.5s] |rate cmd| p50/p95 = "
                      f"{np.percentile(np.abs(rates),50):.2f}/{np.percentile(np.abs(rates),95):.2f} rad/s "
                      f"(cap 3.14, frac>95%cap {sat_rate:.2f}); normed thrust p50/p95 = "
                      f"{np.percentile(thr,50):.2f}/{np.percentile(thr,95):.2f} (cap 3.765)")
    # multiplicative in-bin collective deficit (corrections weakened too), live latency 2
    for tag, mult, lat in [("mult0.92_lat2", 0.92, 2), ("mult0.88_lat2", 0.88, 2),
                           ("mult0.92_lat0", 0.92, 0), ("nominal_lat2", 1.0, 2)]:
        outcome, passes, tr = rollout(actor, "simstart", latency=lat, coll_mult_inbin=mult)
        nbin = int(tr["inbin"].sum())
        g3 = [p for p in passes if p[0] == 3]
        g3s = f"g3 off=[{g3[0][1]:+.2f},{g3[0][2]:+.2f}]" if g3 else "--"
        print(f"  [{tag:14s} simstart] {outcome:34s} bin-ticks={nbin:3d} {g3s}")
        if not outcome.startswith("FINISHED"):
            n = len(tr["t"]); w0 = max(0, n - 45)
            rates = tr["act"][w0:, 0:3]; thr = tr["normed"][w0:]
            print(f"      [Q2 forensics last 1.5s] |rate cmd| p50/p95 = "
                  f"{np.percentile(np.abs(rates),50):.2f}/{np.percentile(np.abs(rates),95):.2f} "
                  f"(cap 3.14); normed thrust p50/p95 = "
                  f"{np.percentile(thr,50):.2f}/{np.percentile(thr,95):.2f} (cap 3.765)")
    # Q6 exposure: nominal traces, time-in-bin before the gate-3 plane
    for start in ("simstart", "handoff"):
        kw = dict(handoff_speed=10.0, handoff_dist=3.0) if start == "handoff" else {}
        outcome, passes, tr = rollout(actor, start, **kw)
        t_g3 = next((p[3] for p in passes if p[0] == 3), None)
        binmask = np.array([in_bin(SimpleState(v, q)) for v, q in zip(tr["vel"], tr["quat"])])
        pre = binmask[tr["t"] <= (t_g3 or tr["t"][-1])]
        print(f"  [Q6 {start:8s}] {outcome}: bin-exposure before gate-3 plane = "
              f"{pre.sum()*DT:.2f} s of {t_g3 or float('nan'):.2f} s")


class SimpleState:
    def __init__(self, vel, quat):
        self.vel = vel; self.quat = quat


# --------------------------------------------------------------------------------------- Q5
def q5(actor):
    """Recovery probe: restart the policy from the nominal trajectory state ~1.5 s before
    the gate-3 plane, displaced 0.75/1.5 m along each axis. Does it re-thread, give up, or
    hit the frame? (The state class the reset distribution never visits.)"""
    print("=" * 88)
    print("Q5 -- off-trajectory recovery at gate-3 approach (displaced restarts, mixer plant)")
    outcome, passes, tr = rollout(actor, "simstart")
    t_g3 = next(p[3] for p in passes if p[0] == 3)
    k0 = int(round((t_g3 - 1.5) / DT)) - 1
    base = PlantState(pos=tr["pos"][k0], vel=tr["vel"][k0], quat=tr["quat"][k0],
                      omega=tr["omega"][k0], thrust=np.float64(tr["act"][k0][3]))
    gate0 = int(tr["gate"][k0])
    print(f"  base state t={tr['t'][k0]:.2f}s (gate-3 plane at {t_g3:.2f}s), "
          f"|v|={np.linalg.norm(base.vel):.1f} m/s, target gate {gate0}")
    params = mixer_params(0)
    for axis, name in ((0, "N"), (1, "E"), (2, "D")):
        for mag in (0.75, 1.5, -0.75, -1.5):
            st = replace(base, pos=base.pos + np.eye(3)[axis] * mag)
            gate = gate0
            last_normed = float(tr["normed"][k0])
            outcome2 = "TIMEOUT"
            g3off = None
            for k in range(int(10.0 / DT)):
                obs = obs_from_truth(st, gate, last_normed, virtual_flip=True)
                rate_frd, _, last_normed = policy_step(actor, obs, 0.0, True, 0.0)
                action = np.concatenate([rate_frd, [last_normed * _HOVER_THRUST]])
                prev = st.pos.copy()
                st = plant_step(st, action, DT, params)
                struck = frame_strike_other(prev, st.pos, gate)
                if struck is not None:
                    outcome2 = f"COLLISION(g{struck})"
                    break
                ev, y, z = gate_cross(prev, st.pos, gate)
                if ev == "pass":
                    if gate == 3:
                        g3off = (y, z)
                    if gate == N_GATES - 1:
                        outcome2 = "FINISHED"
                        break
                    gate += 1
                elif ev in ("collision", "miss"):
                    outcome2 = f"{ev.upper()}(g{gate} off=[{y:+.2f},{z:+.2f}])"
                    break
            g3s = f"g3 off=[{g3off[0]:+.2f},{g3off[1]:+.2f}]" if g3off else ""
            print(f"  [{name}{mag:+.2f} m] {outcome2:38s} {g3s}")


# ---------------------------------------------------------------------------------- corridor
def corridor(actor):
    """Approach-corridor geometry: in-plane offset and slope over the last metres before
    each gate plane. A plane-band collision model only scores the crossing point; the sim's
    gate frame is volumetric -- a steep approach converts upstream offset into frame strikes
    (live std_f1: 0.5 m high AND 0.5 m short = hit the upper frame before the plane)."""
    print("=" * 88)
    print("CORRIDOR -- near-gate approach geometry (nominal twin, mixer plant, simstart)")
    outcome, passes, tr = rollout(actor, "simstart")
    print(f"[rollout] {outcome}")
    pos = tr["pos"]
    print(f"  {'gate':>4} {'x=-3m Linf':>11} {'x=-1m Linf':>11} {'cross Linf':>11} "
          f"{'|dz/dx|':>8} {'|dy/dx|':>8}  (slope near plane; Linf vs 0.75 aperture)")
    for g, _, _, _t in passes:
        rel = np.array([_R_W2G @ (p * _FLIP - _GATE_POS_ZUP[g]) for p in pos])
        # approach samples: gate-frame x in [-3, 0]
        m = (rel[:, 0] >= -3.0) & (rel[:, 0] <= 0.2)
        if m.sum() < 3:
            continue
        r = rel[m]
        order = np.argsort(r[:, 0])
        r = r[order]
        linf = np.max(np.abs(r[:, 1:3]), axis=1)
        def at(x):
            return float(np.interp(x, r[:, 0], linf))
        dy = np.gradient(r[:, 1], r[:, 0]); dz = np.gradient(r[:, 2], r[:, 0])
        print(f"  {g:>4} {at(-3.0):>11.2f} {at(-1.0):>11.2f} {at(0.0):>11.2f} "
              f"{np.abs(dz[-3:]).mean():>8.2f} {np.abs(dy[-3:]).mean():>8.2f}")


# --------------------------------------------------------------------------------------- Q3
def q3(actor, n=48):
    print("=" * 88)
    print(f"Q3 -- pass-offset distribution under start jitter x latency (n={n}, mixer plant)")
    rng = np.random.default_rng(7)
    offs = {g: [] for g in range(N_GATES)}
    outcomes = []
    for i in range(n):
        lat = int(rng.integers(0, 4))
        outcome, passes, _ = rollout(actor, "simstart", latency=lat, jitter_rng=rng)
        outcomes.append(outcome)
        for g, y, z, _t in passes:
            if not outcome.startswith("COLLISION") or g != passes[-1][0]:
                offs[g].append((y, z))
    fin = sum(1 for o in outcomes if o == "FINISHED")
    print(f"  outcomes: {fin}/{n} FINISHED; others: "
          f"{[o for o in outcomes if o != 'FINISHED'][:6]}")
    print(f"  {'gate':>4} {'n':>4} {'|y| p50/p95':>14} {'|z| p50/p95':>14} {'Linf p95':>9} "
          f"{'margin p5 (0.75-Linf)':>22}")
    for g in range(N_GATES):
        a = np.array(offs[g]) if offs[g] else np.zeros((0, 2))
        if len(a) == 0:
            print(f"  {g:>4}    0")
            continue
        linf = np.max(np.abs(a), axis=1)
        print(f"  {g:>4} {len(a):>4} "
              f"{np.percentile(np.abs(a[:,0]),50):>6.2f}/{np.percentile(np.abs(a[:,0]),95):.2f} "
              f"{np.percentile(np.abs(a[:,1]),50):>6.2f}/{np.percentile(np.abs(a[:,1]),95):.2f} "
              f"{np.percentile(linf,95):>9.2f} {np.percentile(0.75-linf,5):>22.2f}")


# --------------------------------------------------------------------------------------- Q4
def q4(actor):
    print("=" * 88)
    print("Q4 -- global force-model scaling (dr_aero axes as deterministic offsets, mixer plant)")
    print("  contrast with q2q6 res_full: global robustness vs regime-local fragility")
    for coll_k in (0.88, 0.94, 1.0, 1.06, 1.12):
        for c2s in (1.0,) if coll_k != 1.0 else (0.77, 1.25):
            outcome, passes, _ = rollout(actor, "simstart", coll_k=coll_k, c2_scale=c2s)
            g3 = [p for p in passes if p[0] == 3]
            g3s = f"g3 off=[{g3[0][1]:+.2f},{g3[0][2]:+.2f}]" if g3 else "--"
            print(f"  coll_k={coll_k:.2f} c2x{c2s:.2f}: {outcome:34s} {g3s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probe", choices=["q1", "q2q6", "q3", "q4", "q5", "corridor", "all"])
    ap.add_argument("--n", type=int, default=48)
    args = ap.parse_args()
    actor = load_actor(CKPT)
    if args.probe in ("q1", "all"):
        q1(actor)
    if args.probe in ("q2q6", "all"):
        q2q6(actor)
    if args.probe in ("q5", "all"):
        q5(actor)
    if args.probe in ("corridor", "all"):
        corridor(actor)
    if args.probe in ("q3", "all"):
        q3(actor, args.n)
    if args.probe in ("q4", "all"):
        q4(actor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
