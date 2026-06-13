"""b2_verify.py -- ADVERSARIAL verification of A2 (bias-is-binding).

ROLE (estimator-racespeed agent b2): charged to REFUTE A2's claim that the gate-4
in-plane bias floor after global de-bias = 0.517 m (clears 0.05 m = false), i.e. that
"the un-filterable in-plane bias is the binding term."

Four independent attacks, recomputed from raw course_bundle/frames.json + pg/course_g*/
+ characterize_g*.json (NOT from A2's intermediates):

  (1) GENERALIZATION / circularity of VISION-CAL global de-bias. The de-bias offset is
      fit on the SAME 6-gate recording it is evaluated on. Leave-one-gate-out (LOGO):
      estimate the global offset from the OTHER 5 gates, apply to held-out gate-4, and
      see whether the "de-biased" residual is the same -> circular, or collapses ->
      overfit. Also test: is a single global constant even the right model?

  (2) GATE-4 IN-PLANE PROJECTION & TRANSIT. Independently recompute the gate-4 normal,
      opening-centre, and the in-plane (E,D) error AS A FUNCTION OF RANGE from the dense
      pg/course_g4 window. The binding moment is the TRANSIT (range -> 0), not a 8-27 m
      range-window mean. A2's 0.45-0.52 m is a far-range-dominated average.

  (3) PER-TRACK-CONSTANT vs PER-FIX-RANDOM. Fit each in-plane axis vs range:
      off = intercept + slope*range. A range-SLOPE term is NOT a per-track-constant
      registration bias -- it is a geometry-correlated systematic that (a) shrinks to ~0
      at transit and (b) is the SAME function every lap, so a 2nd cal lap that fits the
      slope removes it deterministically. Distinguish the genuinely constant part
      (intercept at transit range) from the range-correlated part.

  (4) DEPTH-SCALE ARTIFACT. Is the apparent in-plane bias actually a PnP depth-scale
      (range_err) artifact that the chi2 gate / range-sanity already removes, or that
      collapses at short range? Decompose along the sighting ray (depth) vs perpendicular.

Run:  PYTHONPATH=src .venv/Scripts/python.exe \
        handoff/ultracode-estimator-racespeed-2026-06-13/b2_verify.py
(bash). Add case-C dir to sys.path internally. Offline ONLY. Seed fixed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
CASE_C = REPO / "handoff/ultracode-vision-case-c-2026-06-13"
CHAR_DIR = REPO / "handoff/perception-char-2026-06-08"
MAP_PATH = REPO / "handoff/shadowpc-firstcontact-2026-06-02/track_map.json"
sys.path.insert(0, str(CASE_C))

import vision_cal  # noqa: E402  (case-C prototype; measure, don't re-derive)
from racer.gate_mapper import MEASURED_FIX_BIAS_NED  # noqa: E402

RNG = np.random.default_rng(20260613)
N_BOOT = 20000
CHI2_GATE = 16.27
GATE4 = 4
VARIANCE_BAR = 0.05
INPLANE_IDX = (1, 2)   # E (lateral), D (vertical) -- VERIFIED below
ALONGTRACK_IDX = 0     # N


# ---------------------------------------------------------------------------
# Independent geometry: gate-4 normal + opening centre, straight from the map quat
# ---------------------------------------------------------------------------
def gate_frame(map_path=MAP_PATH):
    mp = json.loads(Path(map_path).read_text())
    rec = next(r for r in mp["gates"] if int(r["gate_id"]) == GATE4)
    q = rec["orientation_ned_wxyz"]
    R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    h = float(rec.get("height_m") or 2.72)
    col2 = R[:, 2] if R[:, 2][2] >= 0 else -R[:, 2]   # height/down axis
    bottom = np.asarray(rec["position_ned"], float)
    opening = bottom - 0.5 * h * col2
    # gate axes in world NED
    normal = R[:, 1]        # the flight-through axis (col1) -- verify it's ~ +/-N below
    lateral = R[:, 0]       # in-gate horizontal (col0)
    height = col2           # in-gate vertical (down-positive)
    # g3 position to confirm flight direction
    g3 = np.asarray(next(r for r in mp["gates"] if int(r["gate_id"]) == 3)["position_ned"], float)
    flight_dir = bottom - g3
    flight_dir = flight_dir / np.linalg.norm(flight_dir)
    return dict(R=R, opening=opening, bottom=bottom, normal=normal, lateral=lateral,
                height=height, flight_dir=flight_dir, g3=g3, h=h)


# ---------------------------------------------------------------------------
# Raw fix loaders (independent of A2's helpers)
# ---------------------------------------------------------------------------
def load_accepted_g4_fixes(pool_course60s=True):
    """KF-accepted gate-4 fixes: associated to gate-4, 4-corner, maha<=chi2 gate.
    Returns list of dicts with frame_id, true_range, off_ned, maha, range_err, source."""
    files = ["characterize_g4.json"]
    if pool_course60s:
        files.append("characterize_course_60s.json")
    by_fid = {}
    for fname in files:
        d = json.loads((CHAR_DIR / fname).read_text())
        for r in d["rows"]:
            if not (r.get("associated") and r.get("gate_id") == GATE4):
                continue
            if r.get("n_corners") != 4 or "off_ned" not in r:
                continue
            m = r.get("maha", np.nan)
            if not (np.isfinite(m) and m <= CHI2_GATE):
                continue
            by_fid[int(r["frame_id"])] = dict(
                frame_id=int(r["frame_id"]),
                true_range=float(r["true_range_m"]),
                off_ned=np.asarray(r["off_ned"], float),
                maha=float(m),
                range_err=float(r.get("range_err_m", np.nan)),
                source=fname,
            )
    rows = sorted(by_fid.values(), key=lambda x: x["true_range"])
    return rows


def boot_mean_ci(s, n_boot=N_BOOT, alpha=0.05):
    s = np.asarray(s, float)
    n = len(s)
    idx = RNG.integers(0, n, size=(n_boot, n))
    means = s[idx].mean(1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(s.mean()), float(lo), float(hi)


# ===========================================================================
def main():
    out = {"seed": 20260613, "n_boot": N_BOOT, "role": "b2 ADVERSARIAL verifier of A2"}

    # ---- ATTACK 2 (first, it grounds everything): geometry independent recompute ----
    gf = gate_frame()
    out["attack2_geometry"] = {
        "gate4_opening_centre_ned": gf["opening"].round(4).tolist(),
        "gate_normal_col1_ned": gf["normal"].round(4).tolist(),
        "gate_lateral_col0_ned": gf["lateral"].round(4).tolist(),
        "gate_height_col2_ned": gf["height"].round(4).tolist(),
        "flight_dir_g3_to_g4_ned": gf["flight_dir"].round(4).tolist(),
        "normal_dot_flight": round(float(abs(gf["normal"] @ gf["flight_dir"])), 4),
        "inplane_axes_confirmed": "E (col0=+E) and D (col2=+down); along-track=N (col1=-N)",
        "A2_inplane_assumption_correct": bool(abs(gf["normal"][0]) > 0.99
                                              and abs(gf["lateral"][1]) > 0.99
                                              and abs(gf["height"][2]) > 0.99),
    }

    # ---- The full accepted gate-4 fix stream (range-sorted) ----
    rows = load_accepted_g4_fixes(pool_course60s=True)
    rng = np.array([r["true_range"] for r in rows])
    off = np.array([r["off_ned"] for r in rows])      # (n,3) N,E,D
    n = len(rows)
    out["g4_accepted_stream"] = {
        "n": n,
        "range_span_m": [round(float(rng.min()), 2), round(float(rng.max()), 2)],
        "mean_off_ned": off.mean(0).round(4).tolist(),
        "std_off_ned": off.std(0, ddof=1).round(4).tolist(),
    }

    # ---- ATTACK 3: per-track-CONSTANT vs RANGE-SLOPE, per in-plane axis ----
    # off_axis = intercept + slope*range. intercept = constant part; transit value at
    # range->small is what actually bites the gate-4 plane crossing.
    slope_fits = {}
    for ax, name in zip(range(3), ("N", "E", "D")):
        sl, ic = np.polyfit(rng, off[:, ax], 1)
        # transit estimate: extrapolate to the closest realistic transit range.
        # The drone crosses the plane ~ range 0, but last usable fix is bounded; report
        # the fit at range 0, at 3 m, and the measured nearest-fix value.
        slope_fits[name] = dict(
            slope_m_per_m=round(float(sl), 4),
            intercept_m=round(float(ic), 4),
            at_range_0=round(float(ic), 4),
            at_range_3m=round(float(ic + 3 * sl), 4),
            at_range_5m=round(float(ic + 5 * sl), 4),
            mean_over_window=round(float(off[:, ax].mean()), 4),
        )
    out["attack3_range_slope_fit"] = slope_fits

    # in-plane transit estimate from the slope fit (E,D at near-transit ranges)
    e_fit = slope_fits["E"]; d_fit = slope_fits["D"]
    inplane_transit = {
        "at_range_0": round(float(np.hypot(e_fit["at_range_0"], d_fit["at_range_0"])), 4),
        "at_range_3m": round(float(np.hypot(e_fit["at_range_3m"], d_fit["at_range_3m"])), 4),
        "at_range_5m": round(float(np.hypot(e_fit["at_range_5m"], d_fit["at_range_5m"])), 4),
        "A2_window_mean": round(float(np.hypot(off[:, 1].mean(), off[:, 2].mean())), 4),
    }
    out["attack3_inplane_at_transit"] = inplane_transit

    # ---- ATTACK 2 cont.: directly measure in-plane error at the closest fixes ----
    # The 5 closest-range accepted fixes (the transit-relevant set)
    k = 5
    near = off[:k]   # already range-sorted ascending
    near_inplane_each = np.hypot(near[:, 1], near[:, 2])
    near_mean_ed = near[:, INPLANE_IDX].mean(0)
    out["attack2_nearest_fixes"] = {
        "k": k,
        "ranges_m": [round(float(rng[i]), 2) for i in range(k)],
        "inplane_E_D_each": near[:, INPLANE_IDX].round(3).tolist(),
        "inplane_norm_each_m": near_inplane_each.round(4).tolist(),
        "mean_inplane_E_D": near_mean_ed.round(4).tolist(),
        "mean_inplane_norm_m": round(float(np.hypot(*near_mean_ed)), 4),
        "median_inplane_norm_m": round(float(np.median(near_inplane_each)), 4),
    }

    # ---- ATTACK 1: leave-one-gate-out generalization of the global de-bias ----
    # Recover per-gate offsets via the REAL resurvey (so we attack A2's own instrument).
    regs, diag = vision_cal.resurvey()
    reg = {r.gate_id: np.asarray(r.offset_ned, float) for r in regs}
    gids = sorted(reg)
    offsets = np.array([reg[g] for g in gids])   # (6,3) gate sense (est-map)

    # full-data global (what A2 uses) vs LOGO global (fit WITHOUT gate-4)
    global_all = offsets.mean(0)
    others = np.array([reg[g] for g in gids if g != GATE4])
    global_logo = others.mean(0)                 # de-bias estimated from OTHER gates only
    g4_off = reg[GATE4]

    # In-fix sense the de-bias = -global. The de-biased gate-4 residual:
    #   resid = g4_off - global   (gate sense). Compare full-data vs LOGO.
    resid_full = g4_off - global_all
    resid_logo = g4_off - global_logo
    out["attack1_logo_generalization"] = {
        "global_debias_full_data_gate_sense": global_all.round(4).tolist(),
        "global_debias_LOGO(without_g4)_gate_sense": global_logo.round(4).tolist(),
        "g4_offset_gate_sense": g4_off.round(4).tolist(),
        "g4_residual_after_full_data_debias": resid_full.round(4).tolist(),
        "g4_residual_after_LOGO_debias": resid_logo.round(4).tolist(),
        "g4_inplane_resid_full_data_m": round(float(np.hypot(resid_full[1], resid_full[2])), 4),
        "g4_inplane_resid_LOGO_m": round(float(np.hypot(resid_logo[1], resid_logo[2])), 4),
        "interpretation": (
            "If LOGO residual ~ full-data residual, the de-bias is NOT circular but the "
            "per-gate E offset is genuinely un-removable by a global constant (supports A2). "
            "If LOGO residual differs a lot, the global model is unstable / the offset is "
            "dominated by something other than a shared constant."
        ),
    }

    # Is a single global CONSTANT the right model at all? per-gate E offsets:
    e_offsets = offsets[:, 1]
    out["attack1_global_constant_validity"] = {
        "per_gate_E_offset_gate_sense": {int(g): round(float(reg[g][1]), 4) for g in gids},
        "E_offset_std_across_gates_m": round(float(e_offsets.std(ddof=1)), 4),
        "note": ("Per-gate E offsets swing +0.46 (g4) to -0.55 (g3): a single global "
                 "constant cannot fit them. BUT these resurvey offsets are robust-medians "
                 "over each gate's FULL range window (mostly far fixes) -- attack 3 shows "
                 "the E offset is range-driven, so the across-gate E scatter is largely a "
                 "by-product of each gate being seen at a DIFFERENT range mix, not 6 "
                 "independent constant registrations."),
    }

    # ---- ATTACK 4: depth-scale artifact -- ray (depth) vs perpendicular decomposition ----
    # Use vision_cal.depth_decomposition for gate-4 (GT pose from frames.json),
    # AND an independent short-range vs long-range range_err correlation.
    decomp = vision_cal.depth_decomposition()
    g4d = decomp.get(GATE4, {})
    out["attack4_depth_decomp"] = {
        k: (v.round(4).tolist() if isinstance(v, np.ndarray) else v)
        for k, v in g4d.items()
    }
    # correlation of in-plane error magnitude with |range_err| and with range
    rerr = np.array([r["range_err"] for r in rows])
    inplane_each = np.hypot(off[:, 1], off[:, 2])
    out["attack4_correlations"] = {
        "corr_inplane_vs_range": round(float(np.corrcoef(rng, inplane_each)[0, 1]), 4),
        "corr_inplane_vs_abs_range_err": round(float(np.corrcoef(np.abs(rerr), inplane_each)[0, 1]), 4),
        "corr_E_vs_range": round(float(np.corrcoef(rng, off[:, 1])[0, 1]), 4),
        "corr_D_vs_range": round(float(np.corrcoef(rng, off[:, 2])[0, 1]), 4),
        "note": ("Strong positive corr(in-plane, range) => the 'bias' is a far-range "
                 "geometry/depth effect that collapses at transit, NOT a transit-time "
                 "constant. corr(in-plane,|range_err|) tests the depth-scale hypothesis."),
    }

    # ---- ATTACK 2b: in-plane bias by RANGE CUTOFF (the decisive table) ----
    # A2 quotes the <=27 m window mean (0.52) and stops at <=12 m (0.34). The actual
    # last-usable accepted gate-4 fix is at ~8.25 m; the near-transit band is far smaller.
    def boot_inplane_band(ed, nb=N_BOOT):
        nn = len(ed)
        idx = RNG.integers(0, nn, size=(nb, nn))
        b = ed[idx].mean(1)
        norms = np.sqrt((b ** 2).sum(1))
        pt = float(np.hypot(*ed.mean(0)))
        lo, hi = np.percentile(norms, [2.5, 97.5])
        return round(pt, 4), round(float(lo), 4), round(float(hi), 4)

    band_table = {}
    for lab, cut in [("le_9m", 9.0), ("le_10m", 10.0), ("le_12m", 12.0),
                     ("le_16m", 16.0), ("all_8_27m", 99.0)]:
        mask = rng <= cut
        ed = off[mask][:, INPLANE_IDX]
        pt, lo, hi = boot_inplane_band(ed)
        band_table[lab] = {"n": int(mask.sum()), "inplane_bias_m": pt, "ci95": [lo, hi]}
    out["attack2b_inplane_by_range_cutoff"] = band_table

    # nearest-3 (the last-usable-before-transit set, range 8.25-9.14 m)
    near3 = off[:3, INPLANE_IDX]
    out["attack2b_last_usable_fix"] = {
        "min_associated_range_m": round(float(rng.min()), 2),
        "note": "closer than ~8 m the detector locks gate-5 / loses 4-corner -> fix rejected",
        "nearest3_ranges_m": [round(float(rng[i]), 2) for i in range(3)],
        "nearest3_mean_inplane_E_D": near3.mean(0).round(4).tolist(),
        "nearest3_mean_inplane_norm_m": round(float(np.hypot(*near3.mean(0))), 4),
        "nearest1_inplane_norm_m": round(float(np.hypot(off[0, 1], off[0, 2])), 4),
    }

    # monotone-regime slope (exclude the 3 depth-flip outliers range>21 where E flips +)
    mono = rng <= 21.0
    mono_fit = {}
    for ax, name in zip(range(3), ("N", "E", "D")):
        sl, ic = np.polyfit(rng[mono], off[mono, ax], 1)
        mono_fit[name] = {"slope_m_per_m": round(float(sl), 4), "intercept_m": round(float(ic), 4)}
    out["attack2b_monotone_regime"] = {
        "n_mono": int(mono.sum()),
        "n_depthflip_outliers_excluded": int((~mono).sum()),
        "fit": mono_fit,
        "note": ("E and D both shrink monotonically with decreasing range until ~21 m, where 3 "
                 "depth-flip fixes flip E positive. A linear fit on the FULL set is corrupted by "
                 "these; the near-range collapse is the real signal. Linear extrapolation to "
                 "range 0 is NOT valid (relationship is non-linear / collapses)."),
    }

    # ---- ATTACK 3b: does a 2nd cal-lap SLOPE fit remove it deterministically? ----
    # Fit the slope on the data, predict residual at the transit range AFTER removing
    # BOTH the global constant AND the fitted range-slope. This is what a determinism-
    # per-track cal lap (which sees the SAME slope every lap) achieves.
    transit_range = 3.0   # representative near-transit usable-fix range; report sensitivity
    resid_after_slope = {}
    for ax, name in zip(range(3), ("N", "E", "D")):
        sl, ic = np.polyfit(rng, off[:, ax], 1)
        pred = ic + sl * rng
        residuals = off[:, ax] - pred         # scatter around the deterministic curve
        resid_after_slope[name] = dict(
            transit_value_m=round(float(ic + sl * transit_range), 4),
            scatter_std_about_curve_m=round(float(residuals.std(ddof=1)), 4),
        )
    out["attack3b_callap_slope_removal"] = {
        "transit_range_m": transit_range,
        "per_axis": resid_after_slope,
        "inplane_transit_after_slope_removal_m": round(float(np.hypot(
            resid_after_slope["E"]["transit_value_m"],
            resid_after_slope["D"]["transit_value_m"])), 4),
        "inplane_scatter_about_curve_m": round(float(np.hypot(
            resid_after_slope["E"]["scatter_std_about_curve_m"],
            resid_after_slope["D"]["scatter_std_about_curve_m"])), 4),
    }

    # ---- VERDICT SYNTHESIS ----
    # A2 headline: in-plane bias floor after de-bias = 0.517 m, does NOT clear 0.05.
    # b2 counter-numbers:
    inplane_transit_0 = inplane_transit["at_range_0"]
    inplane_transit_3 = inplane_transit["at_range_3m"]
    near_mean = out["attack2_nearest_fixes"]["mean_inplane_norm_m"]
    out["VERDICT"] = {
        "verdict": "WEAKENED",
        "A2_claim": "gate-4 in-plane bias floor after global de-bias = 0.517 m; clears 0.05 = false",
        "b2_geometry_check": "A2 in-plane axes (E,D) and N-normal CONFIRMED correct (attack 2 fails)",
        "b2_inplane_bias_le9m_m": band_table["le_9m"]["inplane_bias_m"],
        "b2_inplane_bias_le9m_ci95": band_table["le_9m"]["ci95"],
        "b2_nearest3_inplane_m": out["attack2b_last_usable_fix"]["nearest3_mean_inplane_norm_m"],
        "b2_nearest1_inplane_m": out["attack2b_last_usable_fix"]["nearest1_inplane_norm_m"],
        "A2_window_mean_inplane_m": band_table["all_8_27m"]["inplane_bias_m"],
        "b2_logo_inplane_resid_m": out["attack1_logo_generalization"]["g4_inplane_resid_LOGO_m"],
        "HOLDS": (
            "DIRECTION of A2 is correct: absolute world-fixing does NOT reach 0.05 m at gate-4 "
            "(even the closest-fix band is 0.11-0.19 m, > both the 0.05 m bar and the 0.155 m "
            "margin); the gate-relative observation is the right fix. Geometry is correct. "
            "Per-gate E offsets genuinely cannot be fit by one global constant."
        ),
        "BREAKS": (
            "MAGNITUDE is overstated. A2's 0.45-0.52 m is a RANGE-WINDOW MEAN (8-27 m) dominated "
            "by far fixes (in-plane reaches 0.69 m at 13-20 m). The transit-relevant last-usable "
            "fix is at ~8.25 m where in-plane = 0.11 m (nearest-3 mean 0.23 m). FRAMING as an "
            "un-filterable CONSTANT is wrong: it is RANGE-COLLAPSING, so a recency/range-weighted "
            "R (downweight far fixes -- exactly what range_anisotropic_R does) and a cal-lap "
            "slope fit materially reduce it. A2's own table stopped at <=12 m (0.34) and never "
            "showed the <=9 m band (0.19)."
        ),
        "corrected_floor_transit_m": "0.11-0.34 (range-band dependent), NOT 0.52",
    }

    print(json.dumps(out, indent=2, default=str))
    (Path(__file__).with_name("b2_verify_results.json")).write_text(
        json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    main()
