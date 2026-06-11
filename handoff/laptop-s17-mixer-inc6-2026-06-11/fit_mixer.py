"""Fit the motor-mixer coupling constants from the SHADOWPC-LIVE-DEPLOY-DIAG probe points.

Input data = the measured table in handoff/shadowpc-live-deploy-diag-2026-06-11/WRITEUP.md
Section 2 (recording 20260611_194826_mixer_probe, profile mixer_probe.json at 44c8e54; the raw
recording is ShadowPC-local, so this consumes the WRITEUP's published aggregates -- each line
below cites its row).  Run from repo root:

    .venv\\Scripts\\python.exe handoff\\laptop-s17-mixer-inc6-2026-06-11\\fit_mixer.py

MODEL (the diag Section 8 recommendation, in commanded-collective units):

    u_i   = clip(c + S_i . d, idle, 1.0)          per-motor command, X-quad sign matrix S (4x3)
    d_ax  = kappa_err * (target_ax - omega_ax) + kappa_hold * omega_ax     [signed, per axis]
    d_yaw *= zeta / (zeta + c)                    yaw differential effectiveness falls with
                                                  collective (rotor-speed-dependent yaw torque)
    c_eff = mean(u)                -> the S16 knot table -> a_up   (the parasitic-lift channel)
    delta_ax = (S^T u)_ax / 4      -> realized differential; authority Q_ax = (delta/d) / r_fit_ax
               scales the S14 slew limit alpha_max (r_fit = delta/d at the S14 fit condition --
               hover collective, single-axis pi command -- because alpha_max was measured WITH
               the mixer already throttling there; Q keeps the fit point exact).

Identity (motor units == commanded-collective units) is adopted because the c100 probe point
(motors mean 0.874 / max 1.0 / a_up 58.9 at commanded 1.0) is reproduced by identity + top-clip
asymmetry (re-level differential ~0.25: u=[1,1,.75,.75] -> mean 0.875, K(0.875)=65 ~= 58.9 after
the observed re-level tilt wobble), while an affine motor map predicts K(1.0)=78.3 -- refuted.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from racer.rl_plant import (COLL_MAP_ACCEL_MEASURED, COLL_MAP_THR_MEASURED)

# ---- the validated rate-loop constants (S14; racer.rl_plant) --------------------------------
RATE_GAIN = np.array([2.501, 2.504, 2.231])
S_MAP = 0.30

def target(axis: int, cmd: float) -> float:
    """Super-rate steady target |rad/s| for a sustained |cmd| on one axis (S14 static map)."""
    g = RATE_GAIN[axis] / (1.0 - S_MAP * min(abs(cmd), np.pi) / np.pi)
    return g * abs(cmd)

def K(c: float) -> float:
    """S16 measured convex collective->accel knot table (commanded-collective units)."""
    return float(np.interp(c, COLL_MAP_THR_MEASURED, COLL_MAP_ACCEL_MEASURED))

# ---- WRITEUP Section 2 measured aggregates ---------------------------------------------------
# z00_y31 (thr 0, yaw 3.14): motors [0.08, 0.73, 0.73, 0.08] early; mean 0.258 late
#   (0.40 early); a_up 9.36 m/s^2 phase-avg; sustained spin settles toward the mapped target.
# z00_y12 (thr 0, yaw 1.2): mean 0.126, a_up 1.73 (WRITEUP itself: twin a_up(0.126)~1.1, noisy).
# zhov_y31 (thr 0.2656, yaw 3.14): mean 0.287, max 0.76, a_up 12.1 -- "mean preserved".
# z00_rp15 (thr 0, roll+pitch 1.5): mean 0.243, max 0.64 (tumbles; max is the clean early read).
# z00_r0  (thr 0, rates 0): mean 0.064, a_up 0.64 -> free fall; idle ~0.05-0.065.
# c100    (thr 1.0, rates ~0, twin-falsify): mean 0.874, max 1.0, a_up 58.9 -> top headroom 0.126.

T_Y31 = target(2, 3.14)            # 10.006 rad/s
T_Y12 = target(2, 1.2)             #  3.024 rad/s
T_R15 = target(0, 1.5)             #  4.379 rad/s
T_P15 = target(1, 1.5)             #  4.384 rad/s

print("=== super-rate steady targets at the probe commands ===")
print(f"  yaw 3.14 -> {T_Y31:.4f} rad/s   yaw 1.2 -> {T_Y12:.4f}   roll 1.5 -> {T_R15:.4f}   "
      f"pitch 1.5 -> {T_P15:.4f}")

# ---- kappa_err: the error-driven differential (transient; omega ~ 0, e = target) -------------
# yaw rail, early: high pair 0.73 (c = 0 -> d = u_high) over e = 10.006
k_err_yaw = 0.73 / T_Y31
# roll+pitch rail, early: max motor 0.64 = d_r + d_p (the +/+ motor; c = 0)
k_err_rp = 0.64 / (T_R15 + T_P15)
print("\n=== kappa_err (collective units per rad/s of rate error) ===")
print(f"  yaw-rail fit  : 0.73 / {T_Y31:.4f}            = {k_err_yaw:.5f}")
print(f"  r+p-rail fit  : 0.64 / {T_R15 + T_P15:.4f}            = {k_err_rp:.5f}")
KAPPA_ERR = 0.073
print(f"  -> nominal {KAPPA_ERR}  (two independent probes agree to 0.2%)")

# ---- kappa_hold: the sustained differential at settled rate (e ~ 0, omega = target) -----------
# yaw rail, late: high pair 0.46 over omega = 10.006
KAPPA_HOLD_FIT = 0.46 / T_Y31
KAPPA_HOLD = 0.046
print("\n=== kappa_hold (collective units per rad/s of held rate) ===")
print(f"  yaw-rail settled: 0.46 / {T_Y31:.4f}          = {KAPPA_HOLD_FIT:.5f} -> nominal {KAPPA_HOLD}")
print("  (single-point fit; roll/pitch unmeasured (probe tumbles) -> symmetric + DR band)")

# ---- zeta_yaw: yaw differential effectiveness vs collective ----------------------------------
# zhov_y31 settled mean 0.287 with c = 0.2656, idle 0.05: solve the clipped-pair mean for d:
#   mean = (2*(c+d) + 2*idle)/4 = 0.287  (low pair clipped at idle when d > c - idle)
HOVER = 0.2656
IDLE = 0.05
d_hov = 2.0 * 0.287 - HOVER - IDLE          # = (c + d + idle)/2 = 0.287 -> d
eta_hov = d_hov / (KAPPA_HOLD * T_Y31)      # vs the c=0 settled differential
ZETA = eta_hov * HOVER / (1.0 - eta_hov)    # eta = zeta/(zeta + c)
print("\n=== zeta_yaw (yaw effectiveness: d_yaw *= zeta/(zeta + c)) ===")
print(f"  zhov_y31 settled mean 0.287 -> d(c=hover) = {d_hov:.4f}")
print(f"  eta(hover) = {d_hov:.4f} / {KAPPA_HOLD * T_Y31:.4f} = {eta_hov:.4f}  -> zeta = {ZETA:.4f}")
ZETA_NOM = 0.34
print(f"  -> nominal {ZETA_NOM}")
print(f"  TENSION (documented): zhov max-motor predicts {HOVER + KAPPA_ERR * T_Y31 * ZETA_NOM / (ZETA_NOM + HOVER):.3f} "
      f"early vs measured 0.76 (phase max); the mean constraint wins -> wide DR band [0.20, 0.55]")

# ---- idle --------------------------------------------------------------------------------------
print("\n=== idle ===")
print("  z00_r0 motors mean 0.064; yaw-rail low pair 0.08 (carries small r/p stabilisation);")
print("  diag's own reading ~0.05 -> nominal 0.05, DR [0.04, 0.08]")

# ---- model reproduction of every probe row -----------------------------------------------------
def motors(c: float, d: np.ndarray, idle: float = IDLE) -> np.ndarray:
    u0 = np.clip(c + (d[0] + d[1] + d[2]), idle, 1.0)
    u1 = np.clip(c + (-d[0] + d[1] - d[2]), idle, 1.0)
    u2 = np.clip(c + (d[0] - d[1] - d[2]), idle, 1.0)
    u3 = np.clip(c + (-d[0] - d[1] + d[2]), idle, 1.0)
    return np.array([u0, u1, u2, u3])

def eta(c: float, zeta: float = ZETA_NOM) -> float:
    return zeta / (zeta + max(c, 0.0))

print("\n=== model vs measured (per probe row) ===")
rows = []
# z00_y31 early (e = target, omega = 0)
d = np.array([0.0, 0.0, KAPPA_ERR * T_Y31 * eta(0.0)])
u = motors(0.0, d)
rows.append(("z00_y31 early motors", f"[{', '.join(f'{x:.2f}' for x in sorted(u))}]",
             "[0.08, 0.08, 0.73, 0.73] (sorted)"))
# z00_y31 settled (e = 0, omega = target)
d = np.array([0.0, 0.0, KAPPA_HOLD * T_Y31 * eta(0.0)])
u = motors(0.0, d)
rows.append(("z00_y31 settled mean -> a_up", f"mean {u.mean():.3f} -> K = {K(u.mean()):.2f} m/s^2",
             "mean 0.258 -> a_up 9.36"))
# z00_y12 settled
d = np.array([0.0, 0.0, KAPPA_HOLD * T_Y12 * eta(0.0)])
u = motors(0.0, d)
rows.append(("z00_y12 settled mean", f"{u.mean():.3f}", "0.126 (phase avg incl. transient)"))
# zhov_y31 settled
d = np.array([0.0, 0.0, KAPPA_HOLD * T_Y31 * eta(HOVER)])
u = motors(HOVER, d)
rows.append(("zhov_y31 settled mean", f"{u.mean():.3f} -> K = {K(u.mean()):.1f}",
             "0.287 -> a_up 12.1"))
# z00_rp15 early max motor
d = np.array([KAPPA_ERR * T_R15, KAPPA_ERR * T_P15, 0.0])
u = motors(0.0, d)
rows.append(("z00_rp15 early max motor", f"{u.max():.3f}", "0.64"))
# z00_r0
u = motors(0.0, np.zeros(3))
rows.append(("z00_r0 mean -> a_up", f"{u.mean():.3f} -> K = {K(u.mean()):.2f}",
             "0.064 -> 0.64 (free fall)"))
# c100 with a modest re-level differential ~0.25 (roll axis)
d = np.array([0.25, 0.0, 0.0])
u = motors(1.0, d)
rows.append(("c100 (re-level d~0.25) mean", f"{u.mean():.3f} (max {u.max():.2f})",
             "0.874 (max 1.0)"))
for name, got, want in rows:
    print(f"  {name:34s} model {got:42s} measured {want}")

# ---- r_fit: authority normalisation at the S14 slew-fit condition ------------------------------
print("\n=== r_fit (authority normalisation; S14 alpha_max fit point: c=hover, single-axis pi) ===")
for ax, name in enumerate(("roll", "pitch", "yaw")):
    t = target(ax, np.pi)
    dax = KAPPA_ERR * t * (eta(HOVER) if ax == 2 else 1.0)
    dvec = np.zeros(3); dvec[ax] = dax
    u = motors(HOVER, dvec)
    s = np.array([[1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], dtype=float)
    delta = float(s[ax] @ u) / 4.0
    print(f"  {name:5s}: target {t:.3f}  d {dax:.4f}  delta {delta:.4f}  r_fit = {delta / dax:.4f}")

print("\n=== shipped nominals ===")
print(f"  MIXER_IDLE_MEASURED       = {IDLE}      DR [0.04, 0.08]")
print(f"  MIXER_KAPPA_ERR_MEASURED  = {KAPPA_ERR}     DR [0.060, 0.085]")
print(f"  MIXER_KAPPA_HOLD_MEASURED = {KAPPA_HOLD}     DR [0.030, 0.060]")
print(f"  MIXER_ZETA_YAW_MEASURED   = {ZETA_NOM}      DR [0.20, 0.55]")
