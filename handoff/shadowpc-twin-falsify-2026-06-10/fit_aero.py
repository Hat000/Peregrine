"""P1 aero/drag fit: linear vs quadratic, per body axis, world vs body frame; P3 vertical.

Method (per run, per sample): residual aero accel = a_meas - a_thrust(model tilt+collective)
- g.  Horizontal drag is fit on COAST samples (collective ~hover, near-level: the thrust-model
error is second-order there), pooled across runs.  Candidate forms, fit by least squares on
the residual-vs-velocity cloud:

    linear     a = -d1 * v                      (the twin: d1 = 0.2111, isotropic, world frame)
    quadratic  a = -c2 * |v| * v
    mixed      a = -d1 * v - c2 * |v| * v

Axis split: velocity is projected into the BODY-horizontal frame (yaw only) -> body-x (fwd)
vs body-y (lateral) components fit separately; back/fwd/lat run families compared. Vertical
(P3): per collective-step segments of vert_*/coll_*, regress vertical aero residual vs vz ->
slope = vertical drag, intercept = collective-map error at that stick.

Usage (repo root): .venv\\Scripts\\python.exe handoff\\shadowpc-twin-falsify-2026-06-10\\fit_aero.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runs as R

DRAG_RUNS = ["recon_lat12", "drag_back08", "drag_back17", "drag_back25", "drag_back32",
             "drag_fwd17", "drag_fwd25", "drag_lat17p", "drag_lat17n", "drag_lat25p"]


def coast_cloud(labels, phases=("coast",), min_v=0.3, tilt_max_deg=6.0):
    """Pooled (v_world(M,3), a_aero(M,3), yaw(M), label(M)) over the given phase windows,
    restricted to near-level samples (thrust-model error second-order) above min_v."""
    V, A, Y, L = [], [], [], []
    for lb in labels:
        run = R.load(lb)
        res = R.aero_resid(run)
        m = np.zeros(len(run.t), dtype=bool)
        for p in phases:
            m |= run.seg(p)
        tilt = np.degrees(np.maximum(np.abs(run.rpy[:, 0]), np.abs(run.rpy[:, 1])))
        m &= (tilt < tilt_max_deg) & (np.hypot(run.vel[:, 0], run.vel[:, 1]) > min_v)
        m[:2] = m[-2:] = False                       # finite-diff edge rows
        V.append(run.vel[m]); A.append(res[m]); Y.append(run.rpy[m, 2])
        L += [lb] * int(m.sum())
    return np.vstack(V), np.vstack(A), np.concatenate(Y), np.array(L)


def fit_forms(v, a):
    """LS fits of a = -d1*v, a = -c2|v|v, a = -d1*v - c2|v|v on stacked horizontal comps.
    Returns dict with coefficients + RMS residual per form."""
    sp = np.hypot(v[:, 0], v[:, 1])
    x = np.concatenate([v[:, 0], v[:, 1]])           # both horizontal axes stacked
    xq = np.concatenate([sp * v[:, 0], sp * v[:, 1]])
    y = np.concatenate([a[:, 0], a[:, 1]])
    out = {}
    d1 = -float(np.dot(x, y) / np.dot(x, x))
    out["linear"] = {"d1": d1, "rms": float(np.std(y + d1 * x))}
    c2 = -float(np.dot(xq, y) / np.dot(xq, xq))
    out["quad"] = {"c2": c2, "rms": float(np.std(y + c2 * xq))}
    M = np.column_stack([x, xq])
    coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    out["mixed"] = {"d1": -float(coef[0]), "c2": -float(coef[1]),
                    "rms": float(np.std(y - M @ coef))}
    out["n"] = len(y)
    return out


def body_axis_fit(v, a, yaw):
    """Project into body-horizontal (yaw rotation) and fit the QUAD form per body axis."""
    c, s = np.cos(yaw), np.sin(yaw)
    vbx = c * v[:, 0] + s * v[:, 1]
    vby = -s * v[:, 0] + c * v[:, 1]
    abx = c * a[:, 0] + s * a[:, 1]
    aby = -s * a[:, 0] + c * a[:, 1]
    sp = np.hypot(vbx, vby)
    res = {}
    for nm, vv, aa in (("body_x", vbx, abx), ("body_y", vby, aby)):
        xq = sp * vv
        keep = np.abs(vv) > 0.5
        c2 = -float(np.dot(xq[keep], aa[keep]) / np.dot(xq[keep], xq[keep]))
        d1 = -float(np.dot(vv[keep], aa[keep]) / np.dot(vv[keep], vv[keep]))
        res[nm] = {"c2": c2, "d1_lin": d1, "n": int(keep.sum())}
    return res


def speed_profile_table():
    """Per run: peak speed + late-coast effective decay vs twin prediction."""
    print("\n== per-run speed summary (accel end / coast decay) ==")
    print(f"{'run':14s} {'tilt':>5s} {'v_acc_end':>9s} {'v_coast0':>8s} {'v_coast_end':>11s} "
          f"{'coast_dur':>9s} {'eff_d_lin':>9s}")
    for lb in DRAG_RUNS:
        run = R.load(lb)
        m = run.seg("coast")
        if not m.any():
            continue
        i0, i1 = np.argmax(m), len(m) - np.argmax(m[::-1]) - 1
        sp = np.hypot(run.vel[:, 0], run.vel[:, 1])
        macc = run.seg("accel")
        v_acc = sp[macc][-1] if macc.any() else np.nan
        # effective linear-d over the DECAYING part (from peak speed inside coast)
        ipk = i0 + int(np.argmax(sp[i0:i1 + 1]))
        dur = run.t[i1] - run.t[ipk]
        d_eff = np.log(sp[ipk] / sp[i1]) / dur if dur > 0.5 else np.nan
        tilt = "".join(ch for ch in lb if ch.isdigit())
        print(f"{lb:14s} {tilt:>5s} {v_acc:9.2f} {sp[ipk]:8.2f} {sp[i1]:11.2f} "
              f"{dur:9.2f} {d_eff:9.3f}")


def vertical_fits():
    """P3: per constant-collective segment, regress vertical aero residual vs vz."""
    print("\n== vertical: per collective-step a_z residual vs vz ==")
    print(f"{'run/phase':22s} {'thr':>6s} {'n':>4s} {'K_meas (m/s2)':>13s} {'twin K':>7s} "
          f"{'dz_lin (1/s)':>12s} {'dz_quad c2':>10s} {'vz range':>14s}")
    for lb in ("vert_mid", "vert_high", "coll_hover", "coll_speed"):
        run = R.load(lb)
        acc = R.accel(run)
        for p in sorted(set(run.phase)):
            if not (p.startswith(("c0", "c1", "climb4", "climb5", "desc")) or p == "coast_v"):
                continue
            m = run.seg(p)
            m[:2] = m[-2:] = False
            if m.sum() < 8:
                continue
            thr = float(np.median(run.thr[m]))
            tilt = np.cos(run.rpy[m, 0]) * np.cos(run.rpy[m, 1])
            vz = run.vel[m, 2]
            # measured vertical specific force from the rotor = a_z - g (NED), per sample;
            # normalize by tilt -> body-axis magnitude; regress vs vz (up = -vz)
            az_rotor = (acc[m, 2] - R.G) / np.maximum(tilt, 0.5)     # = -K(thr) - drag_z/tilt
            keep = np.isfinite(az_rotor)
            if keep.sum() < 8 or np.ptp(vz[keep]) < 0.8:
                # not enough vz spread for a slope: report the mean K only
                K = -float(np.mean(az_rotor[keep])) if keep.any() else np.nan
                print(f"{lb+'/'+p:22s} {thr:6.3f} {int(keep.sum()):4d} {K:13.2f} "
                      f"{R.G*thr/0.2656:7.2f} {'--':>12s} {'--':>10s} "
                      f"[{vz.min():+5.1f},{vz.max():+5.1f}]")
                continue
            A = np.column_stack([np.ones(keep.sum()), vz[keep]])
            coef, *_ = np.linalg.lstsq(A, az_rotor[keep], rcond=None)
            K = -float(coef[0]); dz = float(coef[1])                  # az_rotor = -K + dz*vz
            Aq = np.column_stack([np.ones(keep.sum()), np.abs(vz[keep]) * vz[keep]])
            cq, *_ = np.linalg.lstsq(Aq, az_rotor[keep], rcond=None)
            print(f"{lb+'/'+p:22s} {thr:6.3f} {int(keep.sum()):4d} {K:13.2f} "
                  f"{R.G*thr/0.2656:7.2f} {dz:12.3f} {float(cq[1]):10.4f} "
                  f"[{vz.min():+5.1f},{vz.max():+5.1f}]")


def main():
    speed_profile_table()

    v, a, yaw, lbl = coast_cloud(DRAG_RUNS)
    print(f"\n== pooled horizontal coast cloud: {len(v)} samples ==")
    forms = fit_forms(v, a)
    print(f" linear : d1={forms['linear']['d1']:.4f} /s          rms={forms['linear']['rms']:.3f} m/s2"
          f"   (twin: 0.2111)")
    print(f" quad   : c2={forms['quad']['c2']:.4f} /m           rms={forms['quad']['rms']:.3f} m/s2")
    print(f" mixed  : d1={forms['mixed']['d1']:.4f} c2={forms['mixed']['c2']:.4f}  "
          f"rms={forms['mixed']['rms']:.3f} m/s2")

    print("\n== body-axis split (quad form), coast samples ==")
    ba = body_axis_fit(v, a, yaw)
    for k, d in ba.items():
        print(f" {k}: c2={d['c2']:.4f} /m  (linear-form d1={d['d1_lin']:.3f})  n={d['n']}")

    print("\n== per-family quad fits (frame test: same c2 fwd/back/lat = body-symmetric) ==")
    for fam, labels in (("back", ["drag_back08", "drag_back17", "drag_back25", "drag_back32"]),
                        ("fwd", ["drag_fwd17", "drag_fwd25"]),
                        ("lat", ["recon_lat12", "drag_lat17p", "drag_lat17n", "drag_lat25p"])):
        vv, aa, yy, _ = coast_cloud(labels)
        f = fit_forms(vv, aa)
        print(f" {fam:4s}: quad c2={f['quad']['c2']:.4f} (rms {f['quad']['rms']:.3f})  "
              f"linear d1={f['linear']['d1']:.4f} (rms {f['linear']['rms']:.3f})  n={f['n']}")

    # binned mean a vs v -- the SHAPE diagnostic (linear = flat a/v, quad = rising |a|/v)
    print("\n== binned |a_aero| vs |v| (pooled coast; a/v constant = linear, rising = quad) ==")
    sp = np.hypot(v[:, 0], v[:, 1])
    ah = -(a[:, 0] * v[:, 0] + a[:, 1] * v[:, 1]) / np.maximum(sp, 1e-6)   # along-track decel
    for lo in np.arange(0.5, 8.0, 0.75):
        m = (sp >= lo) & (sp < lo + 0.75)
        if m.sum() < 25:
            continue
        print(f"  v {lo:4.2f}-{lo+0.75:4.2f}: n={int(m.sum()):5d}  a_along={np.mean(ah[m]):+6.3f} "
              f"m/s2  a/v={np.mean(ah[m])/np.mean(sp[m]):+6.4f}  a/v2={np.mean(ah[m])/np.mean(sp[m])**2:+6.4f}")

    vertical_fits()


if __name__ == "__main__":
    main()
