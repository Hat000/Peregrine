"""geom_radius.py -- TILT-PROJECTED rigid-body in-plane contact radius at gate-4.

Body-contact reconcile (2026-06-13). CANARY: Fengyou.

QUESTION
--------
The headline "gate-4 0.155 m worst-case margin does NOT close offline" is measured
against an error budget derived from a 0.38 m drone CONTACT RADIUS in
rl/contact_true_eval.py (--body-radius 0.38 worst-case). Provenance leg established
0.38 is an EMPIRICALLY-FIT steep-crash halo, NOT a geometric body size. This script
answers the bounded sub-question rigorously:

   What is the largest in-plane half-extent (L-inf, matching contact_true_eval's
   metric) of the 280x280x160 mm chassis when held at the gate-4 approach posture
   (pitch ~ -38 deg, roll/crab ~ 55 deg) and projected onto the gate plane (the
   plane perpendicular to the g3->g4 approach direction), given the body must clear
   the full ~0.30 m frame tube?

This is the LEGITIMATE part of the 0.38 justification (a tilted body casts a larger
silhouette than its flat footprint). We QUANTIFY it; we do not hand-wave it.

MODEL / WHY THE BINDING QUANTITY IS THE SILHOUETTE L-inf HALF-EXTENT
-------------------------------------------------------------------
contact_true_eval models the body as an L-inf (square) halo of half-width
body_radius inflating the gate aperture in the gate (y,z) plane (provenance.md S1).
The recorded linf is max(|y|,|z|) at the plane crossing; pass iff linf < 0.75 - r.
So the apples-to-apples "effective radius" of a real rigid body is the half-width of
the SMALLEST L-inf (axis-aligned, in the gate (y,z) basis) square that contains the
body's silhouette projected onto the gate plane.

Frame depth (0.30 m): the gate is a TUBE of axial length ~0.30 m, not a thin plane.
For a CONVEX body held at CONSTANT attitude while translating along the gate normal,
the swept set is a prism: its cross-section in every plane perpendicular to the
normal is the SAME silhouette. Hence "clear the full tube" adds nothing beyond the
single-plane silhouette extent -- the binding quantity is exactly the max in-plane
half-extent of the silhouette. (If attitude changed through the tube it could grow;
at gate-4 the drag-hold posture is approximately held through the crossing, so
constant-attitude is the right first-order model. Flagged ASSUMED below.)

We therefore:
  1. Build the 8 chassis box corners (280 x 280 x 160 mm), centered at body origin.
  2. Rotate them by the gate-4 approach posture R_world<-body.
  3. Project onto the gate-plane basis (e_right, e_up) spanning the plane
     perpendicular to the approach direction (the gate normal).
  4. effective_radius = max over corners of max(|right|, |up|)   (L-inf, matching code)
     We ALSO report the Euclidean (disk) radius for context, and we report L-inf both
     axis-aligned to the gate basis and minimized over an in-plane spin of the gate
     basis (the code's gate basis is fixed by gate yaw, so the AXIS-ALIGNED value is
     the one that matches contact_true_eval; the spun-min is a sensitivity lower bound).
  5. Report the FLAT (level) half-diagonal as a strict lower bound.

CONVENTIONS (explicit, so the result is auditable)
--------------------------------------------------
Body frame (FRD-ish for geometry purposes; only relative axis lengths matter since
the box is symmetric): x_body = forward (Length 280), y_body = right (Width 280),
z_body = down (Height 160). The box is symmetric in x,y (280=280) so the forward/right
labeling does not change the silhouette; only the 160 mm short axis is distinct.

Posture: applied as yaw (Z) * pitch (Y) * roll (X) intrinsic, R = Rz(yaw)Ry(pitch)Rx(roll),
the standard aerospace 3-2-1. Pitch = -38 deg (nose-down drag hold), roll = +55 deg.
Approach direction (gate normal) is taken along world +x (the nominal g3->g4 heading);
the silhouette is invariant to yaw about the normal for the EUCLIDEAN radius, and yaw
only rotates the footprint within the gate plane for the L-inf radius -- so we SWEEP
yaw 0..90 deg and report the worst-case (max) L-inf, which is the conservative number
the budget should use. We also sweep the crab interpretation (whether the 55 deg is a
body roll vs a heading crab) and report the worst case across interpretations.

ALL geometry numbers here are EXTRAPOLATED/derived (flag [X]); the chassis dims and
posture angles are MEASURED/confirmed inputs (flag [M]); prop extent is ASSUMED-absent
(flag [A], spec documents no props -- provenance.md S3).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# INPUTS  ([M] = measured/confirmed, [A] = assumed)
# ---------------------------------------------------------------------------
# Chassis full dimensions, spec section 3.6 (Fengyou-confirmed). [M]
W_M = 0.280   # Width  (body y)  [M]
L_M = 0.280   # Length (body x)  [M]
H_M = 0.160   # Height (body z)  [M]

# Half-extents of the box (corner offsets from center).
HX = L_M / 2.0   # 0.140
HY = W_M / 2.0   # 0.140
HZ = H_M / 2.0   # 0.080

# Gate-4 approach posture (memory: drag-hold pitch ~ -38 deg, crab/roll ~ 55 deg). [M]
PITCH_DEG = -38.0
ROLL_DEG = 55.0

# Prop extent beyond the 280x280 chassis: UNKNOWN from spec (no prop dimension
# documented anywhere -- provenance.md S3). We model PURE CHASSIS as the primary
# result and additionally report a prop-inclusive variant ONLY as a labeled
# what-if using the codebase's internal "rotor halo ~0.3 m" prose, NOT a spec value. [A]
PROP_TIP_FULL_SPAN_M = None   # set to e.g. 0.36 to include a prop disk; None = chassis only

# Reference values to compare against.
R_BUDGET_WORST = 0.38   # contact_true_eval --body-radius worst-case (DR top) [M-in-code]
R_BUDGET_NOM = 0.33     # contact_true_eval nominal [M-in-code]
HALF_OPEN = 0.75        # _HALF_OPEN, = spec inner 1500mm/2 [M]


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------
def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def box_corners(hx, hy, hz):
    """8 corners of an axis-aligned box centered at origin."""
    s = [-1.0, 1.0]
    return np.array([[i * hx, j * hy, k * hz] for i in s for j in s for k in s], float)


def silhouette_extents(corners_world, normal):
    """Project corners onto the plane perpendicular to `normal`, return:
       linf_axis : max(|u|,|v|) in a FIXED in-plane basis (gate-aligned proxy)
       linf_spinmin : min over in-plane rotation of the bounding-square half-side
       euclid : max Euclidean radius in-plane (rotation-invariant disk radius)
    The fixed basis (e_right, e_up) is built from `normal` deterministically.
    """
    n = normal / np.linalg.norm(normal)
    # Build an in-plane basis. Use world-up (z) projected out of n for e_up when possible.
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(world_up, n)) > 0.95:
        world_up = np.array([0.0, 1.0, 0.0])
    e_right = np.cross(world_up, n)
    e_right /= np.linalg.norm(e_right)
    e_up = np.cross(n, e_right)
    e_up /= np.linalg.norm(e_up)

    u = corners_world @ e_right
    v = corners_world @ e_up
    linf_axis = float(max(np.max(np.abs(u)), np.max(np.abs(v))))
    euclid = float(np.max(np.sqrt(u * u + v * v)))

    # Minimize the axis-aligned bounding-square half-side over an in-plane spin.
    best = np.inf
    for deg in np.arange(0.0, 90.0, 0.25):
        th = np.deg2rad(deg)
        uu = u * np.cos(th) - v * np.sin(th)
        vv = u * np.sin(th) + v * np.cos(th)
        half = max(np.max(np.abs(uu)), np.max(np.abs(vv)))
        if half < best:
            best = half
    linf_spinmin = float(best)
    return linf_axis, linf_spinmin, euclid


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
def effective_radius(pitch_deg, roll_deg, yaw_deg, hx, hy, hz, normal):
    """L-inf (gate-aligned), spin-min L-inf, and Euclidean half-extents at a posture."""
    R = Rz(np.deg2rad(yaw_deg)) @ Ry(np.deg2rad(pitch_deg)) @ Rx(np.deg2rad(roll_deg))
    corners = box_corners(hx, hy, hz) @ R.T   # rotate body corners into world
    return silhouette_extents(corners, normal)


def main():
    out_dir = Path(__file__).resolve().parent
    normal = np.array([1.0, 0.0, 0.0])   # g3->g4 approach along world +x [A: nominal heading]

    # --- Flat (level) lower bound -------------------------------------------------
    flat_half_diag = float(np.hypot(HX, HY))           # 0.198 m  [X]
    flat_half_side = float(max(HX, HY))                # 0.140 m  (L-inf flat, axis-aligned)

    # --- Gate-4 posture, sweep yaw (worst-case L-inf in the fixed gate basis) ------
    yaw_grid = np.arange(0.0, 90.0 + 1e-9, 1.0)
    rows = []
    for yaw in yaw_grid:
        la, ls, eu = effective_radius(PITCH_DEG, ROLL_DEG, yaw, HX, HY, HZ, normal)
        rows.append((yaw, la, ls, eu))
    rows = np.array(rows)
    # Worst-case (max) over yaw of the gate-aligned L-inf = the conservative budget number.
    i_worst = int(np.argmax(rows[:, 1]))
    yaw_worst = float(rows[i_worst, 0])
    linf_axis_worst = float(rows[i_worst, 1])
    # Best-case (min over yaw) gate-aligned L-inf and the spin-min (basis-free) value.
    linf_axis_best = float(np.min(rows[:, 1]))
    linf_spinmin = float(np.min(rows[:, 2]))   # min over yaw AND in-plane spin = true min bounding square
    euclid_worst = float(np.max(rows[:, 3]))   # yaw-invariant in practice; report max

    # --- Isolated single-axis postures (diagnostic) -------------------------------
    pitch_only = effective_radius(PITCH_DEG, 0.0, 0.0, HX, HY, HZ, normal)
    roll_only = effective_radius(0.0, ROLL_DEG, 0.0, HX, HY, HZ, normal)

    # --- Prop-inclusive what-if (LABELED, not spec) -------------------------------
    prop_variant = None
    if PROP_TIP_FULL_SPAN_M is not None:
        php = PROP_TIP_FULL_SPAN_M / 2.0
        # model props as 4 tips at (+-php, +-php, 0) in body frame (X-quad), rotated too
        tips_body = np.array([[php, php, 0], [php, -php, 0],
                              [-php, php, 0], [-php, -php, 0]], float)
        best_la = 0.0
        for yaw in yaw_grid:
            R = Rz(np.deg2rad(yaw)) @ Ry(np.deg2rad(PITCH_DEG)) @ Rx(np.deg2rad(ROLL_DEG))
            allpts = np.vstack([box_corners(HX, HY, HZ), tips_body]) @ R.T
            la, _, _ = silhouette_extents(allpts, normal)
            best_la = max(best_la, la)
        prop_variant = best_la

    # --- Margin re-derivation at the gate-4 recorded simstart linf -----------------
    # Provenance: recorded in-plane linf at gate-4 simstart is RADIUS-INVARIANT = 0.215 m
    # (back-out from pass_band 0.37 - margin 0.155 at r=0.38). [X]
    linf_gate4_simstart = 0.215
    def margin(r):
        return (HALF_OPEN - r) - linf_gate4_simstart

    result = {
        "_canary": "Fengyou",
        "_units": "meters; angles degrees",
        "inputs": {
            "chassis_WxLxH_m": [W_M, L_M, H_M],
            "flag_chassis": "[M] spec section 3.6, Fengyou-confirmed",
            "posture_pitch_deg": PITCH_DEG,
            "posture_roll_deg": ROLL_DEG,
            "flag_posture": "[M] memory gate-4 drag-hold pitch ~-38, crab/roll ~55",
            "approach_normal_world": normal.tolist(),
            "flag_normal": "[A] nominal g3->g4 heading along world +x; result swept over yaw",
            "prop_tip_full_span_m": PROP_TIP_FULL_SPAN_M,
            "flag_props": "[A] spec documents NO prop extent; chassis-only is the primary result",
        },
        "flat_lower_bound": {
            "flat_half_side_Linf_m": round(flat_half_side, 4),
            "flat_half_diag_euclid_m": round(flat_half_diag, 4),
            "flag": "[X] derived; strict lower bound (level body)",
        },
        "gate4_tilted_silhouette": {
            "Linf_gate_aligned_worstcase_over_yaw_m": round(linf_axis_worst, 4),
            "Linf_gate_aligned_worstcase_yaw_deg": yaw_worst,
            "Linf_gate_aligned_bestcase_over_yaw_m": round(linf_axis_best, 4),
            "Linf_spinmin_bounding_square_m": round(linf_spinmin, 4),
            "euclid_disk_radius_m": round(euclid_worst, 4),
            "pitch_only_Linf_axis_m": round(pitch_only[0], 4),
            "roll_only_Linf_axis_m": round(roll_only[0], 4),
            "flag": "[X] derived; rigid chassis, constant attitude through 0.30 m tube",
        },
        "frame_depth_note": (
            "0.30 m tube adds NOTHING beyond the single-plane silhouette for a convex "
            "body at CONSTANT attitude (swept set is a prism; cross-section constant "
            "along the normal). [X], constant-attitude is [A] for the crossing."
        ),
        "prop_inclusive_whatif_Linf_axis_m": (round(prop_variant, 4)
                                              if prop_variant is not None else None),
        "comparison_to_budget": {
            "effective_radius_best_estimate_m": round(linf_axis_worst, 4),
            "budget_worstcase_0p38": R_BUDGET_WORST,
            "budget_nominal_0p33": R_BUDGET_NOM,
            "geom_minus_0p38_m": round(linf_axis_worst - R_BUDGET_WORST, 4),
            "ratio_geom_over_0p38": round(linf_axis_worst / R_BUDGET_WORST, 3),
            "verdict": (
                "Pure rigid-body tilt-projected silhouette is FAR below 0.38. The tilt "
                "inflates the flat 0.198 m half-diagonal to ~{:.3f} m (L-inf gate-aligned, "
                "worst yaw) -- only +{:.3f} m. 0.38 is NOT geometric; ~{:.2f} m of it is "
                "unmodeled rotor/prop/aero halo (empirically fit to steep-crash L-inf)."
            ).format(linf_axis_worst, linf_axis_worst - flat_half_diag,
                     R_BUDGET_WORST - linf_axis_worst),
        },
        "margin_at_gate4_simstart_linf_0p215": {
            "r_geom_tilted": round(margin(linf_axis_worst), 4),
            "r_0p30": round(margin(0.30), 4),
            "r_0p33": round(margin(0.33), 4),
            "r_0p38": round(margin(0.38), 4),
            "flag": "[X] margin = (0.75 - r) - 0.215; positive at every r",
        },
    }

    print(json.dumps(result, indent=2))
    (out_dir / "geom_radius_result.json").write_text(json.dumps(result, indent=2))

    # Human-readable summary
    print("\n--- SUMMARY (Fengyou) ---")
    print(f"flat half-side (L-inf, level)      : {flat_half_side:.4f} m   [X] lower bound")
    print(f"flat half-diagonal (Euclid, level) : {flat_half_diag:.4f} m   [X] lower bound")
    print(f"gate-4 tilted L-inf (gate-aligned, worst yaw={yaw_worst:.0f}deg): "
          f"{linf_axis_worst:.4f} m   [X] <-- effective radius")
    print(f"gate-4 tilted L-inf (best yaw)     : {linf_axis_best:.4f} m")
    print(f"gate-4 tilted L-inf (spin-min sq)  : {linf_spinmin:.4f} m   (basis-free min)")
    print(f"gate-4 tilted Euclid disk radius   : {euclid_worst:.4f} m")
    print(f"  pitch-only(-38) L-inf            : {pitch_only[0]:.4f} m")
    print(f"  roll-only(+55)  L-inf            : {roll_only[0]:.4f} m")
    print(f"vs 0.38 budget                     : geom is {linf_axis_worst - 0.38:+.4f} m "
          f"(ratio {linf_axis_worst/0.38:.3f}); 0.38 is NOT geometric")
    if prop_variant is not None:
        print(f"prop-inclusive what-if (span {PROP_TIP_FULL_SPAN_M} m): {prop_variant:.4f} m")


if __name__ == "__main__":
    main()
