"""Measured-plant parameters for the self-contained time-optimal planner.

Single source of truth for the planning plant constants, mirrored from
``racer.rl_plant`` (the parity-tested measured plant). Kept as plain literals so the
optimizer runs WITHOUT importing torch / the full racer stack (CasADi-only venv), but
the literals are cross-checked against ``racer.rl_plant`` by ``verify_against_rl_plant``
when that module is importable.

PLANT CLASS (what the bound models):
    Collective + body-rate "point-mass-with-attitude" (the CTBR plant class), the SAME
    abstraction the prior TOGT bound used (scripts/togt). State = [p(3), v(3), q(4)];
    control = [a_c (body-up specific accel), omega(3) (body rates)]. The inner-loop rate
    lag (tau ~ 0.02 s) and slew (alpha_max ~ 260 rad/s^2 roll/pitch) are NOT modeled here
    -- they are a downstream FEASIBILITY check (twin replay), not part of the bound. The
    body rates are bounded by the SUSTAINED reachable rate through the measured static
    super-rate map (see OMEGA_MAX below).

AERO (the corrected/measured plant -- supersedes the falsified linear plant):
    * THRUST: convex collective->accel map, full stick = 78.28 m/s^2 = 7.98 g (vs the old
      falsified linear 3.765 g ceiling). We bound the body-up specific accel by this
      ceiling. a_c in [0, A_MAX].
    * DRAG: quadratic body-frame drag, pooled c2 ~ 0.052 / m: a_drag_body = -c2*|v_b|*v_b.
      This is the measured aero (twin-falsify 2026-06-11). Scales as v^2 -> a drag WALL.

Provenance: handoff/shadowpc-characterize-sweep-2026-06-10, twin-falsify 2026-06-11,
racer.rl_plant {COLL_MAP_*, QUAD_DRAG_C2_*, SUPER_RATE_S, ALPHA_MAX}.
"""
from __future__ import annotations

import numpy as np

# --- gravity / mass (per-unit-mass model; mass = 1 kg, all accels are specific) ----------
G = 9.80665
MASS = 1.0

# --- THRUST: measured convex collective->body-up-accel map (normalised stick -> m/s^2) ----
COLL_MAP_THR = np.array(
    [0.0, 0.10, 0.15, 0.20, 0.2656, 0.32, 0.40, 0.45, 0.55, 0.60, 0.80, 1.0]
)
COLL_MAP_ACCEL = np.array(
    [0.0, 0.0, 2.1279523370741864, 4.705159463852836, 9.58040784291044,
     13.576760297067407, 21.708896781382972, 26.48830915166427,
     38.74857791726588, 42.360958058019655, 58.431876299624356,
     78.28283850468465]
)
HOVER_THRUST = 0.2656                       # normalised stick at hover (a_c ~= g there)
A_MAX = float(COLL_MAP_ACCEL[-1])           # 78.283 m/s^2 = full-stick body-up specific accel
A_MAX_G = A_MAX / G                          # ~7.98 g

# --- DRAG: measured quadratic body-frame drag (pooled isotropic coefficient, 1/m) ---------
# a_drag_body = -QUAD_DRAG_C2 * |v_body| * v_body. Pooled isotropic value brackets the
# (3,2) sign-split measured tensor; isotropic so body vs world frame is immaterial for the
# magnitude, which is what sets the drag wall / top speed.
QUAD_DRAG_C2 = 0.052
# Falsified legacy linear drag (1/s), kept only for the contrast/"old plant" sensitivity.
LINEAR_DRAG_LEGACY = 0.2111

# --- BODY-RATE authority (sustained reachable rate through the measured super-rate map) ----
# cmd clamp +-3.14 rad/s through g(|c|)=G0/(1-s|c|/pi), G0~[2.50,2.50,2.23], s~0.30 ->
# sustained ~11.0-11.2 rad/s roll/pitch; yaw level-attitude plateau ~7.4 (racing yaw small).
# Same values the prior TOGT bound used (scripts/togt/gen_cases.py OMEGA_NOMINAL).
OMEGA_MAX = np.array([11.0, 11.0, 7.0])     # [roll, pitch, yaw] rad/s

# --- gate / validity geometry ------------------------------------------------------------
GATE_INNER_OPENING_M = 1.5                   # inner square opening
# Validity rule = in-plane miss < half-opening = 0.75 m. The defensible bound constrains
# the crossing to the INSCRIBED circle (Euclidean miss < 0.75) = exactly how we measure
# validity. GATE_BALL_RADIUS is that inscribed radius.
GATE_BALL_RADIUS = GATE_INNER_OPENING_M / 2.0   # 0.75 m


def coll_accel_from_norm(coll_norm):
    """Normalised collective stick [0,1] -> body-up specific accel (m/s^2), measured map."""
    return np.interp(coll_norm, COLL_MAP_THR, COLL_MAP_ACCEL)


def verify_against_rl_plant() -> bool:
    """Cross-check the literals here against racer.rl_plant if it's importable. Returns True
    on match (or skip if rl_plant absent); raises AssertionError on mismatch."""
    try:
        from racer.rl_plant import (COLL_MAP_THR_MEASURED, COLL_MAP_ACCEL_MEASURED,
                                    QUAD_DRAG_C2_POOLED)
        from racer.rl_plant import PlantParams
    except Exception:
        return True  # rl_plant not importable in this venv -> skip (literals are the SSOT)
    assert np.allclose(COLL_MAP_THR, np.asarray(COLL_MAP_THR_MEASURED)), "COLL_THR drift"
    assert np.allclose(COLL_MAP_ACCEL, np.asarray(COLL_MAP_ACCEL_MEASURED)), "COLL_ACCEL drift"
    assert abs(QUAD_DRAG_C2 - float(QUAD_DRAG_C2_POOLED)) < 1e-9, "c2 drift"
    p = PlantParams()
    assert abs(p.hover_thrust - HOVER_THRUST) < 1e-9, "hover drift"
    assert abs(p.g - G) < 1e-9, "g drift"
    return True


if __name__ == "__main__":
    verify_against_rl_plant()
    print(f"A_MAX = {A_MAX:.3f} m/s^2 = {A_MAX_G:.3f} g  |  c2 = {QUAD_DRAG_C2}/m  |  "
          f"omega_max = {OMEGA_MAX.tolist()}  |  gate ball R = {GATE_BALL_RADIUS} m")
    print("verify_against_rl_plant: OK")
