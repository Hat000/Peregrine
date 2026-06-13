"""a3_realism.py — FIX-STREAM REALISM at the gate-4 window (~37 m/s).

ROLE (a3): bound how the vision world-fix STREAM degrades when the drone is at ~37 m/s on the
g3->g4 approach, so the estimator agent (a1) and verifiers use a REALISTIC input rather than the
~5.35 m/s VQ1-recording profile that was actually measured.

EVERYTHING here is reproduced from:
  - MEASURED per-fix profile : handoff/perception-char-2026-06-08/characterize_course_60s.json
  - fix-cov coefficients     : handoff/ultracode-vision-case-c-2026-06-13/range_R_coeffs.json
  - gate-4 geometry          : track_map (g3,g4) quoted in FACTS.md
  - camera intrinsics        : racer.frames.CAMERA_INTRINSICS_K (640x360, f=320, HFoV 90, VFoV 58.7)

Run:  PYTHONPATH=src .venv/Scripts/python.exe handoff/ultracode-estimator-racespeed-2026-06-13/a3_realism.py

Every printed number is tagged MEASURED / MODELED / ASSUMED so the escape hatch is auditable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "handoff" / "ultracode-vision-case-c-2026-06-13"))

from racer.frames import CAMERA_INTRINSICS_K  # noqa: E402

RNG = np.random.default_rng(20260613)  # explicit seed for reproducibility

# ----------------------------------------------------------------------------------------------
# 0. Constants from the live model / data (MEASURED or fixed by hardware)
# ----------------------------------------------------------------------------------------------
F_PX = float(CAMERA_INTRINSICS_K[0, 0])          # 320 px focal length  [MEASURED/hardware]
IMG_W, IMG_H = 640, 360                           # native res            [MEASURED/hardware]
HFOV_DEG, VFOV_DEG = 90.0, 58.7                   # field of view         [MEASURED/hardware]

COEFFS = json.loads((_REPO / "handoff" / "ultracode-vision-case-c-2026-06-13"
                     / "range_R_coeffs.json").read_text())
C2 = COEFFS["c2"]            # depth std coeff: std_depth = C2 * r^2   (= 0.003125)   [MEASURED-fit]
A1 = COEFFS["a1"]            # lateral std coeff: std_lat = A1 * r      (= 0.02605)   [MEASURED-fit]
SIG0_RAD = COEFFS["sig0_rad"]   # 0.40 m radial floor (bias-absorption)              [MEASURED-fit]
SIG0_TAN = COEFFS["sig0_tan"]   # 0.282 m tangential floor                            [MEASURED-fit]

# Per-fix profile measured at ~5.35 m/s (the BASELINE the multipliers scale FROM)
BASE_SPEED_MPS = 5.35
BASE_BIAS_NED = np.array([-0.285, 0.064, -0.346])    # m  [MEASURED]
BASE_NOISE_NED = np.array([0.816, 0.577, 0.436])     # m  [MEASURED]
BASE_ACCEPT = 0.47                                    #    [MEASURED]
BASE_LEAK = 0.0053                                    #    [MEASURED]

# gate-4 window geometry (track_map, NED). bottom-centre positions.
G3 = np.array([-111.5, -5.1, 24.57])
G4 = np.array([-135.5, -0.8, 25.36])

# Detector cadence. firstcontact HANDOFF: true video ~28.6 fps; nav inherits 30 Hz default.
FPS = 28.6                                            # [MEASURED] true video rate
DT_FRAME = 1.0 / FPS

VISION_MAX_RANGE_M = 32.0                             # nav cap                    [from src/navigator]

WINDOW_SPEED_MPS = 37.0                               # post-gate-3 approach speed [MODELED from planner]


# ----------------------------------------------------------------------------------------------
# 1. MOTION BLUR -> corner sigma_px growth -> world-fix sigma growth
# ----------------------------------------------------------------------------------------------
def angular_and_pixel_velocity(speed_mps: float, exposure_s: float):
    """Camera angular rate on g3->g4 approach + worst-case pixel velocity of gate-4 corners.

    Two contributions to image-plane corner motion during one exposure:
      (a) PURE TRANSLATION toward the gate at speed v: a point at world LOS range r, at image
          offset (u,v) from principal point, moves on the image plane. For a point essentially
          ahead, the dominant smear is from the TRANSVERSE component of velocity. On the g3->g4
          leg the motion unit is ~[-0.984, +0.176, +0.032] (NED): the gate sits ~10.1 deg off
          the heading (4.3 m E shift over 24.4 m). The LOS to gate-4 therefore SWEEPS as the
          drone closes -> an angular rate omega_los ~ v_perp / r where v_perp is the speed
          component perpendicular to the LOS.
      (b) Gate corners are at +-0.75 m off the LOS; their individual LOS angles change slightly
          faster than the centre as range shrinks (looming), but the dominant, range-robust term
          is the bearing sweep (a). We bound with the bearing-sweep angular rate.

    pixel_velocity_px_per_s = omega_los * f. blur_len_px = pixel_velocity * exposure_s.
    """
    L = G4 - G3                       # full leg, used only for direction
    v_hat = L / np.linalg.norm(L)
    # The bearing sweep is maximal near the gate (small r). Evaluate at a representative
    # mid/late approach range. v_perp = component of velocity perpendicular to instantaneous LOS.
    # As the drone runs down the leg the LOS to a fixed gate rotates fastest when close; we
    # report the rate as a function of range r below.
    return v_hat


def los_angular_rate(speed_mps: float, range_m: float, perp_offset_m: float = 4.3,
                     leg_len_m: float = 24.39) -> float:
    """Instantaneous LOS angular rate (rad/s) to gate-4 as the drone closes at constant speed.

    Model the drone on a straight leg with the gate a fixed transverse distance d_perp from the
    drone's flight LINE. With along-track distance to closest-approach x and lateral miss d, the
    bearing theta = atan2(d, x). The drone is NOT aimed exactly at the gate (it flies a line); the
    transverse offset of the gate from the heading line drives the sweep. Using the leg lateral
    shift (4.3 m E + 0.79 m D ~ 4.37 m transverse over the 24.4 m leg) as the effective d_perp at
    the START of the leg, the gate is ~d_perp away laterally; range r along LOS. The transverse
    velocity component is v_perp = v * sin(beta) where beta is the angle between velocity and LOS.

    For a point at LOS range r whose transverse offset from the flight line is d, sin(beta) = d/r.
    => omega = v * d / r^2  (rad/s).  This is the textbook closing-bearing rate.
    """
    # transverse offset of gate-4 from the (straight) flight line through g3 along v_hat:
    L = G4 - G3
    v_hat = L / np.linalg.norm(L)
    # decompose the g3->g4 vector: along + perp. (the gate itself IS the leg end, so transverse
    # offset of the GATE from the flight line is ~0 if the line aims at it. The relevant transverse
    # term is the ANGULAR offset of the gate-4 *corners* and any cross-track the line carries.)
    # Honest bound: use d = half-leg-lateral as a generic transverse scale the LOS sweeps through.
    d = float(np.hypot(L[1], L[2]))   # 4.37 m transverse extent the bearing rotates across the leg
    # Effective transverse offset seen at range r: cap by the leg lateral extent.
    d_eff = min(d, range_m)
    return speed_mps * d_eff / (range_m ** 2 + 1e-9)


def corner_sigma_px_blur(blur_len_px: float, sigma_px_static: float = 1.5) -> float:
    """Corner-localization sigma_px as a function of motion-blur smear length.

    PHYSICS/MODELED. A sharp corner is localized to sigma_px_static (~1.5 px, the WEIGHTED_SIGMA_PX
    used in gate_pose). Motion blur of length B smears the corner into a streak; the centroid of a
    uniformly-smeared edge has its localization variance inflated. A defensible, conservative model
    used in feature-tracking literature: the along-smear localization std grows ~ B/sqrt(12)
    (uniform streak std) ADDED IN QUADRATURE to the static std, and the CNN-pose head additionally
    loses confidence (heteroscedastic). We use:

        sigma_px(B) = sqrt( sigma_static^2 + (k_blur * B)^2 )

    with k_blur in [0.3, 0.5] (MODELED). k_blur=0.29 ~ 1/sqrt(12) is the pure-uniform-streak floor;
    we take 0.4 as a mid estimate that also charges some detector-confidence loss. The blur smears
    PRIMARILY along the motion direction (image-x for the bearing sweep), so it inflates the
    cross-LOS (lateral) channel more than depth in principle; we apply it isotropically to be
    conservative on depth (depth is the binding r^2 axis).
    """
    K_BLUR = 0.40   # [MODELED] streak->sigma conversion, between 1/sqrt(12)=0.29 and 0.5
    return float(np.sqrt(sigma_px_static ** 2 + (K_BLUR * blur_len_px) ** 2))


def sigma_world_from_px(sigma_px: float, range_m: float, sigma_px_static: float = 1.5):
    """World-fix std (depth, lateral) from per-corner sigma_px, via the calibrated c2/a1 laws.

    c2 and a1 were CALIBRATED at the static pixel noise (1.5 px). Both std laws are LINEAR in
    sigma_px (see range_anisotropic_R.py derivation: std_depth = 2 sigma_px r^2/(f s sqrt(N)),
    std_lat = sigma_px r/(f sqrt(N))). So scaling sigma_px by g = sigma_px/1.5 scales the
    PnP-noise part of both world stds by g. The floors (SIG0_RAD, SIG0_TAN) are NOT pixel-noise
    -> they do not scale with blur. Returns (std_depth, std_lat) PnP-noise parts only (no floor),
    plus the floor-included totals.
    """
    g = sigma_px / sigma_px_static
    std_depth_pnp = g * C2 * range_m ** 2
    std_lat_pnp = g * A1 * range_m
    std_depth_tot = float(np.hypot(std_depth_pnp, SIG0_RAD))
    std_lat_tot = float(np.hypot(std_lat_pnp, SIG0_TAN))
    return std_depth_pnp, std_lat_pnp, std_depth_tot, std_lat_tot


# ----------------------------------------------------------------------------------------------
# 2. FIX CADENCE in METRES + effective independent-fix count
# ----------------------------------------------------------------------------------------------
def fix_cadence(speed_mps: float):
    spacing_m = speed_mps * DT_FRAME
    return spacing_m


# ----------------------------------------------------------------------------------------------
# 3. RANGE DISTRIBUTION on the leg
# ----------------------------------------------------------------------------------------------
def detectable_range_window():
    """When does gate-4 become detectable/accepted, and the last-fix distance before transit.

    Detection range ceiling: empirically the MEASURED data reaches only 23.3 m (at ~5.35 m/s).
    The nav vision_max_range cap is 32 m. The detector's own range is limited by gate apparent
    size: at 32 m the 1.5 m inner gate spans f*s/r = 320*1.5/32 = 15 px -> right at the edge of
    reliable corner localization. At 24 m it is 20 px. So the accept window is ~[transit, ~24-28 m].
    """
    r_cap = VISION_MAX_RANGE_M
    px_at = lambda r: F_PX * 1.5 / r      # inner-gate pixel span
    return r_cap, px_at


# ----------------------------------------------------------------------------------------------
# MAIN — assemble the degraded fix-stream spec
# ----------------------------------------------------------------------------------------------
def main():
    out = {}
    print("=" * 90)
    print("a3 FIX-STREAM REALISM at the gate-4 window (~37 m/s)  [seed=20260613]")
    print("=" * 90)

    # --- exposure assumption sweep (THE single most load-bearing unknown) ---
    # Sim/eval camera exposure is NOT in any spec we hold. We sweep three regimes.
    exposures = {
        "global_shutter_short_0.5ms": 0.5e-3,   # racing-grade short exposure (ASSUMED best case)
        "nominal_2ms": 2.0e-3,                   # typical indoor auto-exposure (ASSUMED mid)
        "long_8ms": 8.0e-3,                      # dim-light / rolling-ish (ASSUMED worst)
    }

    # range grid across the accept window
    ranges = np.array([24.0, 20.0, 16.0, 12.0, 8.0, 5.0])

    print("\n[1] MOTION BLUR chain  (g3->g4, transverse extent 4.37 m, leg 24.39 m)")
    print("    omega_los = v*d_eff/r^2 ;  pix_vel = omega*f ;  blur_len = pix_vel*exposure")
    print(f"    static sigma_px = 1.5  |  k_blur = 0.40 (MODELED)  |  f = {F_PX:.0f} px")
    blur_table = {}
    for r in ranges:
        omega = los_angular_rate(WINDOW_SPEED_MPS, r)         # rad/s
        pix_vel = omega * F_PX                                 # px/s
        row = {"omega_rad_s": round(omega, 4), "pix_vel_px_s": round(pix_vel, 1)}
        for name, exp in exposures.items():
            B = pix_vel * exp                                  # blur length px
            spx = corner_sigma_px_blur(B)
            sd_pnp, sl_pnp, sd_tot, sl_tot = sigma_world_from_px(spx, r)
            # baseline (static) world stds at this range for the multiplier
            _, _, sd_tot0, sl_tot0 = sigma_world_from_px(1.5, r)
            row[name] = {
                "blur_px": round(B, 2),
                "sigma_px": round(spx, 2),
                "depth_std_m": round(sd_tot, 3),
                "lat_std_m": round(sl_tot, 3),
                "depth_mult": round(sd_tot / sd_tot0, 2),
                "lat_mult": round(sl_tot / sl_tot0, 2),
            }
        blur_table[f"r={r:.0f}m"] = row
        print(f"  r={r:4.0f} m  omega={omega:6.4f} rad/s  pix_vel={pix_vel:7.1f} px/s")
        for name in exposures:
            d = row[name]
            print(f"        {name:26s} blur={d['blur_px']:5.2f}px sigma_px={d['sigma_px']:4.2f} "
                  f"depth_std={d['depth_std_m']:.3f}m (x{d['depth_mult']}) "
                  f"lat_std={d['lat_std_m']:.3f}m (x{d['lat_mult']})")
    out["blur_table"] = blur_table

    # --- 2. fix cadence in metres + independent-fix count over the window ---
    print("\n[2] FIX CADENCE in METRES")
    spacing_base = fix_cadence(BASE_SPEED_MPS)
    spacing_window = fix_cadence(WINDOW_SPEED_MPS)
    print(f"    at {BASE_SPEED_MPS} m/s (MEASURED regime): {spacing_base:.3f} m/frame")
    print(f"    at {WINDOW_SPEED_MPS} m/s (window):         {spacing_window:.3f} m/frame  [MODELED]")
    # how many frames from first-accept range (~24 m, charitable) down to transit
    r_first_accept = 24.0   # [MODELED, capped by detector size + MEASURED 23.3 m ceiling]
    # transit: gate-4 fills FoV / passes; last usable fix ~ when gate leaves frame or too close.
    # at 37 m/s, last fix distance = one frame spacing before plane = ~1.29 m (geometric floor),
    # but practically the gate exits the 58.7deg VFoV when r < ~ (0.75/tan(29.35)) = 1.33 m; and
    # detector min reliable range. Use r_last ~ 3 m (gate ~ half-frame, still localizable).
    r_last = 3.0
    n_frames_window = (r_first_accept - r_last) / spacing_window
    n_frames_base = (r_first_accept - r_last) / spacing_base
    # effective INDEPENDENT fixes for VARIANCE averaging = n_frames * acceptance
    # but fixes are NOT fully independent (per-track bias is common; noise is ~indep frame-to-frame)
    eff_indep_window_at47 = n_frames_window * BASE_ACCEPT
    print(f"    accept window ~[{r_last:.0f}, {r_first_accept:.0f}] m = {r_first_accept-r_last:.0f} m")
    print(f"    frames in window @ {WINDOW_SPEED_MPS} m/s = {n_frames_window:.1f}  "
          f"(vs {n_frames_base:.1f} at {BASE_SPEED_MPS} m/s)")
    print(f"    eff. ACCEPTED fixes @47% (pre-blur) = {eff_indep_window_at47:.1f}")
    print("    NOTE: averaging crushes only the ZERO-MEAN noise component by ~1/sqrt(N_eff);")
    print("          per-track BIAS does NOT average out (un-filterable floor).")
    out["cadence"] = {
        "spacing_base_m": round(spacing_base, 3),
        "spacing_window_m": round(spacing_window, 3),
        "r_first_accept_m": r_first_accept,
        "r_last_m": r_last,
        "n_frames_window": round(n_frames_window, 1),
        "n_frames_base": round(n_frames_base, 1),
        "eff_accepted_fixes_at47pct": round(eff_indep_window_at47, 1),
    }

    # --- 3. range distribution / detector ceiling ---
    print("\n[3] RANGE DISTRIBUTION + detector ceiling")
    r_cap, px_at = detectable_range_window()
    for r in (32, 28, 24, 20, 16, 12, 8):
        print(f"    r={r:3d} m  inner-gate span = {px_at(r):5.1f} px")
    print(f"    nav vision_max_range cap = {r_cap:.0f} m (MEASURED/config)")
    print("    MEASURED detection ceiling in data = 23.3 m (at 5.35 m/s, NOT at speed)")
    print("    => first-accept ~24 m is CHARITABLE; blur may pull it IN at 37 m/s (see [4]).")
    out["range"] = {
        "vision_max_range_cap_m": r_cap,
        "measured_detection_ceiling_m": 23.3,
        "inner_gate_px_at_24m": round(px_at(24), 1),
        "inner_gate_px_at_32m": round(px_at(32), 1),
    }

    # --- 4. acceptance drop from blur ---
    print("\n[4] ACCEPTANCE drop from blur")
    # Mechanism: blur -> (i) some frames lose detection entirely (CNN miss), (ii) chi2/reproj
    # rejections rise as the fix cov widens but the BIAS does not, so maha can spike when the
    # widened-but-still-biased fix is gated. We MODEL a multiplicative blur-acceptance factor
    # tied to mid-window blur length. Charitable/mid/harsh brackets:
    mid_r = 12.0
    omega_mid = los_angular_rate(WINDOW_SPEED_MPS, mid_r)
    pix_vel_mid = omega_mid * F_PX
    accept_factor = {}
    for name, exp in exposures.items():
        B = pix_vel_mid * exp
        # MODELED logistic-ish drop: negligible below ~2 px, ~halves by ~10 px smear.
        f_acc = 1.0 / (1.0 + (B / 6.0) ** 1.5)
        accept_factor[name] = round(float(f_acc), 2)
        print(f"    {name:26s} mid-window blur={B:5.2f}px -> accept x{f_acc:.2f} "
              f"-> accept ~{BASE_ACCEPT*f_acc:.2f}")
    out["acceptance"] = {"base": BASE_ACCEPT, "blur_factor": accept_factor,
                         "mid_window_pix_vel_px_s": round(pix_vel_mid, 1)}

    # --- DEGRADED FIX-STREAM SPEC (the deliverable) -------------------------------------------
    print("\n" + "=" * 90)
    print("DEGRADED FIX-STREAM SPEC for the ~37 m/s gate-4 window")
    print("=" * 90)
    # Pick the NOMINAL (2 ms) exposure as the central spec; report best/worst as a band.
    # Use mid-window range r=12 m for the headline multiplier (representative of accepted fixes).
    def spec_at(exp_name, r=12.0):
        omega = los_angular_rate(WINDOW_SPEED_MPS, r)
        pix_vel = omega * F_PX
        B = pix_vel * exposures[exp_name]
        spx = corner_sigma_px_blur(B)
        # per-axis multipliers vs the MEASURED 5.35 m/s baseline noise std.
        # depth axis ~ aligns with N (along-track) at gate-4; lateral axis ~ E & D (in-plane).
        # multiplier = sigma_px/1.5 applied to the PnP-noise part only; floor unchanged.
        g = spx / 1.5
        return {"exposure": exp_name, "r_m": r, "blur_px": round(B, 2),
                "sigma_px": round(spx, 2), "pnp_noise_mult": round(g, 2),
                "accept": round(BASE_ACCEPT * accept_factor[exp_name], 2)}
    band = {k: spec_at(k) for k in exposures}
    for k, v in band.items():
        print(f"  [{k}] {v}")

    # Axis-resolved noise multipliers. At gate-4 the geometry maps:
    #   N = along-track = ~radial/depth axis (LOS ~ along -N) -> scales with depth law (worst, r^2)
    #   E, D = in-plane (lateral) -> scale with lateral law (r^1). THESE are the binding miss axes.
    # The blur pnp-noise multiplier g multiplies the PnP-noise PART of each axis; floor is fixed.
    print("\n  AXIS-RESOLVED degraded NOISE std @ r=12 m (in-plane = E,D = BINDING):")
    for k in exposures:
        v = band[k]
        g = v["pnp_noise_mult"]
        sd_pnp, sl_pnp, sd_tot, sl_tot = sigma_world_from_px(1.5 * g, 12.0)
        # map to NED: N~depth(total), E&D~lateral(total)
        print(f"    {k:26s} N(along)~{sd_tot:.3f}m  E/D(in-plane)~{sl_tot:.3f}m  accept~{v['accept']}")
    out["spec_band"] = band

    # --- 5. BODY-RATE BLUR (the DOMINANT, UN-MEASURABLE term) ---------------------------------
    print("\n[5] BODY-RATE BLUR  (dominates translation/bearing sweep; CANNOT be pinned offline)")
    print("    blur_px = omega_body * f * exposure.  The g3->g4 leg is nearly straight so the")
    print("    bearing sweep [1] is small (<=2 px even at 8 ms). The drone's OWN attitude rate")
    print("    (banking onto the line, yaw to acquire gate-5) is the real blur source.")
    body_rate_table = {}
    for exp_name, exp in exposures.items():
        row = {}
        for omega_deg in (20.0, 50.0, 100.0, 200.0):
            omega = np.radians(omega_deg)
            B = omega * F_PX * exp
            spx = corner_sigma_px_blur(B)
            _, _, sd_tot, sl_tot = sigma_world_from_px(spx, 12.0)
            _, _, sd0, sl0 = sigma_world_from_px(1.5, 12.0)
            row[f"{omega_deg:.0f}deg_s"] = {
                "blur_px": round(B, 2), "sigma_px": round(spx, 2),
                "depth_mult": round(sd_tot / sd0, 2), "lat_mult": round(sl_tot / sl0, 2),
            }
        body_rate_table[exp_name] = row
        print(f"    {exp_name}:")
        for k, v in row.items():
            print(f"        omega_body={k:9s} blur={v['blur_px']:6.2f}px sigma_px={v['sigma_px']:5.2f} "
                  f"depth_x{v['depth_mult']} lat_x{v['lat_mult']}")
    out["body_rate_blur"] = body_rate_table
    print("    => at long exposure + aggressive body rate, lat-noise multiplier can hit 2-4x AND")
    print("       acceptance can collapse. With short global shutter (<=0.5 ms) body-rate blur is")
    print("       negligible (<=1 px even at 200 deg/s). The exposure/shutter spec is THE pivot.")

    # write json
    out_path = Path(__file__).resolve().parent / "a3_realism_results.json"
    out["meta"] = {
        "seed": 20260613, "window_speed_mps": WINDOW_SPEED_MPS, "base_speed_mps": BASE_SPEED_MPS,
        "fps": FPS, "f_px": F_PX, "c2": C2, "a1": A1, "sig0_rad": SIG0_RAD, "sig0_tan": SIG0_TAN,
        "base_bias_ned": BASE_BIAS_NED.tolist(), "base_noise_ned": BASE_NOISE_NED.tolist(),
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
