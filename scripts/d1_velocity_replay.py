#!/usr/bin/env python3
"""D1 OFFLINE VALIDATION: is the vision-referenced lateral velocity actually better?

Replays the recorded flight corpus (``ego_obs.jsonl``) through
``racer.ego_velocity.LateralVelocityFuser`` and reports, for the OLD (raw dead-reckoned KF)
and NEW (fused) obs velocity:

  * slope / correlation of the VISION-MEASURED body velocity regressed on the obs velocity
    (an honest channel would give slope +1.0),
  * the fraction of samples whose SIGN disagrees with the vision measurement,
  * RMS error against the vision measurement,
  * and (``--actor``) whether the roll-correction ONSET moves earlier on an approach when the
    corrected obs is pushed through the real trained actor.

METHOD NOTES -- read these before quoting any number
----------------------------------------------------
1. ROTATION COMPENSATION IS MANDATORY. The gate's apparent motion in the body frame is
   ``dr = -w x r dt - v dt``; at 8 m range a 0.3 rad/s body rate moves the lever at 2.4 m/s,
   which swamps the ~1.7 m/s std of real lateral velocity. Differencing the lever WITHOUT
   de-rotating it therefore measures the drone's ROTATION, not its translation -- and because
   the drone rotates to correct lateral drift, the artefact is ANTI-correlated with the true
   velocity. ``--no-rotcomp`` reproduces that flawed measurement for comparison.
2. The reference velocity is built from RAW ACCEPTED FIXES ONLY (``pose_seen``), never from
   the builder's held/propagated lever -- a held lever has the dead-reckoned velocity
   integrated into it and would make the test circular.
3. HOLD-OUT: the fused estimate scored over a window is the one FROZEN one fix BEFORE the
   window opens, so the estimate and the reference share no measurement. ``--in-sample``
   drops that and lets the filter see the window it is scored on (it flatters the fix).
4. Ticks with a non-zero ``aim_off`` (manual pilot offset) are dropped.
5. GATE-SEAM TELEPORT. The seeker can re-lock onto a DIFFERENT gate while RACE_STATUS still
   reports the old index, so guarding only on ``gate_index`` is not enough: the lever jumps
   mid-window and the reference reads a fabricated velocity. Measured: 1.57% of same-gate
   consecutive fix pairs move >3 m more than any plausible velocity explains (p99 4.4 m, max
   29 m), and leaving them in DEPRESSES the measured quality of the raw channel a long way
   (LEFT slope 0.904 -> 0.649, corr 0.833 -> 0.427). ``--max-apparent-speed`` (default 20 m/s)
   rejects such windows. It is judged on VISION+GYRO alone, never on the KF velocity under
   test -- gating on ``|D_vis - D_dr|`` instead would select for windows where dead reckoning
   already agrees and flatter the OLD arm. The A/B verdict is stable across the whole
   threshold sweep 12 -> infinity (NEW beats OLD on corr/sign/rms at every value, placebo
   loses at every value). Credit: the v21-release-dive session, which hit the same seam in an
   unrelated analysis (a phantom AUC 0.723 that collapsed to 0.565 once seam-free).

Usage:
    python scripts/d1_velocity_replay.py --roots DIR [DIR ...] [--gain 0.15]
    python scripts/d1_velocity_replay.py --roots DIR --actor path/to/actor.pth \
        --case 20260726_181345_v19pick_Ws0_f1 --case-ticks 128 157
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _p in (str(_REPO / "src"), str(_REPO / "rl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from racer.ego_velocity import LateralVelocityFuser, VelocityFusionConfig, _rot_body  # noqa: E402

# obs layout (racer.ego_obs): [0:3] velocity, [5:8] body rates -- both VIRTUAL-FLIPPED
# (_RZ_PI_BODY = diag(-1,-1,1)), so the TRUE body-FLU value is [-x, -y, +z].
_UNFLIP = np.array([-1.0, -1.0, 1.0])


# ------------------------------------------------------------------------------------------
# corpus
# ------------------------------------------------------------------------------------------
def load_rows(session: Path) -> list[dict] | None:
    p = session / "ego_obs.jsonl"
    if not p.exists():
        return None
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows or None


def find_sessions(roots) -> list[tuple[str, Path]]:
    out = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        if (root / "ego_obs.jsonl").exists():
            out.append((root.name, root))
            continue
        for d in sorted(root.iterdir()):
            if d.is_dir() and (d / "ego_obs.jsonl").exists():
                out.append((d.name, d))
    return out


def _aim_off_zero(v) -> bool:
    if v is None:
        return True
    if isinstance(v, (int, float)):
        return float(v) == 0.0
    try:
        return all(float(x) == 0.0 for x in v)
    except (TypeError, ValueError):
        return False


def series(rows: list[dict]) -> list[dict]:
    """Per-tick record: TRUE body-FLU dead-reckoned velocity / rates, the RAW fix or None."""
    out, prev_t = [], None
    for r in rows:
        if not _aim_off_zero(r.get("aim_off")):
            continue
        obs = np.asarray(r.get("obs", []), dtype=np.float64)
        if obs.size < 21 or not np.isfinite(obs).all():
            continue
        t = float(r["sim_time_ns"]) / 1e9
        dt = 0.0 if prev_t is None else t - prev_t
        prev_t = t
        fix = None
        rel = r.get("rel_flu")
        if r.get("pose_seen") and rel is not None:
            rel = np.asarray(rel, dtype=np.float64)
            if rel.shape == (3,) and np.isfinite(rel).all():
                fix = rel
        out.append(dict(t=t, dt=dt, gi=r.get("gate_index"), k=r.get("k"), obs=obs,
                        v_dr=_UNFLIP * obs[0:3], w=_UNFLIP * obs[5:8], fix=fix,
                        act=r.get("act_raw") or []))
    return out


# ------------------------------------------------------------------------------------------
# vision reference + causal fusion replay
# ------------------------------------------------------------------------------------------
def replay(S, cfg: VelocityFusionConfig, rotcomp=True, max_dt=0.30):
    """Causal pass. Returns (per-tick fused bias, per-fix hold-out bias snapshots)."""
    fuser = LateralVelocityFuser(cfg)
    bias_tick, fix_events = [], []
    gi_prev = None
    for i, s in enumerate(S):
        if gi_prev is not None and s["gi"] != gi_prev:
            fuser.on_gate_change()
        gi_prev = s["gi"]
        fuser.propagate(s["w"], s["v_dr"], s["dt"])
        if s["fix"] is not None:
            b_before = fuser.bias.copy()          # frozen BEFORE this fix is consumed
            u_before = None if fuser._u_last is None else fuser._u_last.copy()
            fuser.on_fix(s["fix"])
            fix_events.append(dict(i=i, t=s["t"], gi=s["gi"], r=s["fix"],
                                   b_holdout=b_before, u_holdout=u_before,
                                   b_insample=fuser.bias.copy(),
                                   u_insample=fuser._u_last.copy()))
        bias_tick.append(fuser.correction())
    return bias_tick, fix_events


def _applied(b, u):
    """The correction actually emitted: b projected off the line of sight."""
    if b is None or u is None:
        return np.zeros(3)
    n = float(np.linalg.norm(u))
    if n < 1e-9:
        return np.zeros(3)
    u = u / n
    return b - u * float(u @ b)


def windows(S, fix_events, T=0.5, rotcomp=True, max_dt=0.30, in_sample=False, placebo=None,
            max_apparent_speed=20.0):
    """Non-overlapping scoring windows between accepted fixes of the SAME gate."""
    out = []
    a = 0
    while a < len(fix_events) - 1:
        fa = fix_events[a]
        pick = None
        for b in range(a + 1, len(fix_events)):
            fb = fix_events[b]
            if fb["gi"] != fa["gi"]:
                break
            Tw = fb["t"] - fa["t"]
            if Tw < T * 0.8:
                continue
            if Tw > T * 1.6:
                break
            pick = (b, fb, Tw)
            break
        if pick is None:
            a += 1
            continue
        b, fb, Tw = pick
        A = fa["r"].copy()
        D = np.zeros(3)
        P = fa["r"].copy()          # previous fix, rotated forward -- track-continuity gate
        P_dt = 0.0
        ok = True
        for i in range(fa["i"] + 1, fb["i"] + 1):
            s = S[i]
            if not (0.0 < s["dt"] <= max_dt):
                ok = False
                break
            dR = _rot_body(s["w"], s["dt"]) if rotcomp else np.eye(3)
            A = dR @ A
            D = dR @ D + s["v_dr"] * s["dt"]
            P = dR @ P
            P_dt += s["dt"]
            # GATE-SEAM TELEPORT: the seeker can re-lock onto a DIFFERENT gate while RACE_STATUS
            # still reports the old index, which jumps the lever and manufactures a huge apparent
            # velocity. Reject the window on VISION+GYRO evidence only -- gating on |D_vis - D_dr|
            # instead would select for windows where dead reckoning already agrees and silently
            # flatter the OLD arm. (v21-release-dive 2026-07-27: the same seam produced an
            # AUC 0.723 phantom in an unrelated analysis.)
            if s["fix"] is not None and P_dt > 1e-6:
                if float(np.linalg.norm(P - s["fix"])) / P_dt > max_apparent_speed:
                    ok = False
                    break
                P = s["fix"].copy()
                P_dt = 0.0
        if ok:
            v_vis = (A - fb["r"]) / Tw                       # vision-referenced mean body velocity
            v_old = D / Tw                                   # dead-reckoned mean over the same window
            key_b, key_u = ("b_insample", "u_insample") if in_sample else ("b_holdout", "u_holdout")
            corr_v = _applied(fa[key_b], fa[key_u])
            if placebo is not None:
                # INTEGRITY CHECK: keep the correction's MAGNITUDE, randomise its direction inside
                # the same LOS-perpendicular plane. A metric that improves here too is measuring
                # smoothing, not information -- do not ship on it.
                g = placebo.standard_normal(3)
                g = _applied(g, fa[key_u])
                ng = float(np.linalg.norm(g))
                corr_v = g * (float(np.linalg.norm(corr_v)) / ng) if ng > 1e-9 else corr_v * 0.0
            v_new = v_old + corr_v
            u = fb["r"] / max(np.linalg.norm(fb["r"]), 1e-9)
            out.append(dict(v_vis=v_vis, v_old=v_old, v_new=v_new, u=u, T=Tw,
                            rng=float(np.linalg.norm(fb["r"])), gi=fa["gi"]))
        a = b
    return out


# ------------------------------------------------------------------------------------------
# metrics
# ------------------------------------------------------------------------------------------
def _fit(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 10:
        return dict(n=len(x), slope=float("nan"), corr=float("nan"),
                    sign=float("nan"), rms=float("nan"))
    return dict(n=len(x), slope=float(np.polyfit(x, y, 1)[0]),
                corr=float(np.corrcoef(x, y)[0, 1]),
                sign=float(np.mean(np.sign(x) != np.sign(y))),
                rms=float(np.sqrt(np.mean((y - x) ** 2))))


def report(W, title, axis=1, perp=False):
    """axis 1 == LEFT. perp=True scores the LOS-perpendicular projection of each vector."""
    if not W:
        print(f"{title}: no windows")
        return None
    def comp(key):
        if not perp:
            return np.array([w[key][axis] for w in W])
        v = []
        for w in W:
            u = w["u"]
            d = w[key] - u * float(u @ w[key])
            v.append(d[axis])
        return np.array(v)
    y = comp("v_vis")
    o = _fit(comp("v_old"), y)
    n = _fit(comp("v_new"), y)
    print(f"  {title}  (n={o['n']})")
    print(f"      OLD  slope {o['slope']:+.3f}  corr {o['corr']:+.3f}  "
          f"sign-disagree {o['sign']*100:5.1f}%  rms {o['rms']:.3f} m/s")
    print(f"      NEW  slope {n['slope']:+.3f}  corr {n['corr']:+.3f}  "
          f"sign-disagree {n['sign']*100:5.1f}%  rms {n['rms']:.3f} m/s")
    return o, n


# ------------------------------------------------------------------------------------------
# actor onset test
# ------------------------------------------------------------------------------------------
def actor_onset(actor_path, S, bias_tick, k_lo=None, k_hi=None):
    """Push OLD and NEW obs through the real trained actor; report the roll channel and when
    a sustained roll correction first appears."""
    import torch
    from fly_rl import load_ego_actor
    actor = load_ego_actor(str(actor_path))
    rows = []
    with torch.no_grad():
        for s, b in zip(S, bias_tick):
            if k_lo is not None and not (k_lo <= (s["k"] or -1) <= k_hi):
                continue
            o_old = s["obs"].astype(np.float32).copy()
            o_new = o_old.copy()
            # obs velocity is VIRTUAL-FLIPPED; the correction is in TRUE body FLU
            o_new[0:3] = (o_old[0:3].astype(np.float64) + _UNFLIP * b).astype(np.float32)
            a_old = actor(torch.from_numpy(o_old).unsqueeze(0)).squeeze(0).numpy()
            a_new = actor(torch.from_numpy(o_new).unsqueeze(0)).squeeze(0).numpy()
            rows.append(dict(k=s["k"], t=s["t"], gi=s["gi"], b=b.copy(),
                             roll_old=float(a_old[1]), roll_new=float(a_new[1]),
                             lat_err=float(s["fix"][1]) if s["fix"] is not None else None,
                             rng=float(np.linalg.norm(s["fix"])) if s["fix"] is not None else None))
    return rows


def onset_index(vals, ts, thresh, sign, min_hold_s=0.10):
    """First time |roll| crosses ``thresh`` in direction ``sign`` and stays for min_hold_s."""
    n = len(vals)
    for i in range(n):
        if sign * vals[i] < thresh:
            continue
        j, held = i, False
        while j < n and ts[j] - ts[i] <= min_hold_s:
            if sign * vals[j] < thresh:
                break
            j += 1
        else:
            held = True                      # ran off the end still above threshold
        if j < n and ts[j] - ts[i] > min_hold_s:
            held = True                      # every sample inside the hold window was above
        if held:
            return i
    return None


# ------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", required=True,
                    help="directories of session dirs (or a session dir itself)")
    ap.add_argument("--gain", type=float, nargs="+", default=[0.15])
    ap.add_argument("--window", type=float, default=0.5, help="scoring window length (s)")
    ap.add_argument("--no-rotcomp", action="store_true",
                    help="reproduce the FLAWED uncompensated measurement (see method note 1)")
    ap.add_argument("--in-sample", action="store_true", help="drop the hold-out (flatters the fix)")
    ap.add_argument("--max-apparent-speed", type=float, default=20.0,
                    help="reject a scoring window containing a fix-to-fix lever jump faster than "
                         "this (m/s) -- the gate-seam teleport. 1e9 disables the gate.")
    ap.add_argument("--placebo", action="store_true",
                    help="integrity check: keep the correction MAGNITUDE, randomise its direction. "
                         "Any metric that still improves is measuring smoothing, not information.")
    ap.add_argument("--actor", default=None, help="actor .pth for the roll-onset test")
    ap.add_argument("--case", default=None, help="session name for the onset test")
    ap.add_argument("--case-ticks", type=int, nargs=2, default=None)
    ap.add_argument("--roll-thresh", type=float, default=0.30, help="rad/s onset threshold")
    args = ap.parse_args()

    sessions = find_sessions(args.roots)
    print(f"sessions: {len(sessions)}   rotcomp={not args.no_rotcomp}   "
          f"scoring={'IN-SAMPLE' if args.in_sample else 'HOLD-OUT (1 fix delayed)'}   "
          f"window={args.window:g}s")
    cache = {}
    for name, d in sessions:
        rows = load_rows(d)
        if rows and len(rows) > 20:
            S = series(rows)
            if len(S) > 20:
                cache[name] = S
    print(f"usable: {len(cache)}")

    for g in args.gain:
        cfg = VelocityFusionConfig(gain=float(g))
        rng_pl = np.random.default_rng(20260727) if args.placebo else None
        W = []
        for name, S in cache.items():
            bias_tick, fx = replay(S, cfg, rotcomp=not args.no_rotcomp)
            W.extend(windows(S, fx, T=args.window, rotcomp=not args.no_rotcomp,
                             in_sample=args.in_sample, placebo=rng_pl,
                             max_apparent_speed=args.max_apparent_speed))
        print(f"\n================ gain = {g:g}"
              f"{'  [PLACEBO -- direction randomised]' if args.placebo else ''} ================")
        report(W, "LEFT  (body-FLU axis 1)", axis=1)
        report(W, "LEFT  (LOS-perpendicular part)", axis=1, perp=True)
        report(W, "UP    (LOS-perpendicular part)", axis=2, perp=True)
        report(W, "FWD   (body-FLU axis 0)", axis=0)
        for lo, hi in ((0, 4), (4, 8), (8, 30)):
            sub = [w for w in W if lo <= w["rng"] < hi]
            if len(sub) > 40:
                report(sub, f"LEFT range[{lo},{hi}) m", axis=1)

    if args.actor and args.case:
        S = cache.get(args.case)
        if S is None:
            print(f"\n[onset] case session {args.case!r} not found in --roots", file=sys.stderr)
            return 2
        cfg = VelocityFusionConfig(gain=float(args.gain[0]))
        bias_tick, _ = replay(S, cfg, rotcomp=not args.no_rotcomp)
        klo, khi = (args.case_ticks if args.case_ticks else (None, None))
        R = actor_onset(args.actor, S, bias_tick, klo, khi)
        print(f"\n================ ACTOR ROLL ONSET -- {args.case} "
              f"(ticks {klo}..{khi}, gain {args.gain[0]:g}) ================")
        ts = [r["t"] for r in R]
        # which way does the drone need to roll? sign of the mean lateral gate offset over the leg
        lat = [r["lat_err"] for r in R if r["lat_err"] is not None]
        need = float(np.sign(np.mean(lat))) if lat else 1.0
        print(f"  mean gate LEFT-offset over the leg: {np.mean(lat) if lat else float('nan'):+.3f} m "
              f"-> corrective roll sign {need:+.0f}")
        io = onset_index([r["roll_old"] for r in R], ts, args.roll_thresh, need)
        inw = onset_index([r["roll_new"] for r in R], ts, args.roll_thresh, need)
        def fmt(i):
            return "never" if i is None else f"k={R[i]['k']} t={R[i]['t']:.3f}s rng={R[i]['rng']}"
        print(f"  OLD onset: {fmt(io)}")
        print(f"  NEW onset: {fmt(inw)}")
        if io is not None and inw is not None:
            print(f"  SHIFT: {ts[io] - ts[inw]:+.3f} s (positive == NEW corrects EARLIER)")
        print(f"  roll |delta| mean {np.mean([abs(r['roll_new']-r['roll_old']) for r in R]):.4f} "
              f"max {max(abs(r['roll_new']-r['roll_old']) for r in R):.4f} rad/s")
        print(f"  applied |correction| mean "
              f"{np.mean([np.linalg.norm(r['b']) for r in R]):.3f} m/s")
        print("\n  k     t(s)   rng   b_left  roll_old  roll_new   d")
        for r in R:
            print(f"  {r['k']:>4} {r['t']:7.3f} {('%5.2f' % r['rng']) if r['rng'] else '    -'} "
                  f"{r['b'][1]:+7.3f} {r['roll_old']:+9.4f} {r['roll_new']:+9.4f} "
                  f"{r['roll_new']-r['roll_old']:+7.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
