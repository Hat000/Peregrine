"""a2_bias.py -- BIAS decomposition of the case-C vision world-fix at gate-4.

ROLE (estimator-racespeed agent a2): the KF crushes ZERO-MEAN noise by averaging
fixes; it does NOT remove a consistent BIAS (it tracks a constant offset as real
drift / signal). VISION-CAL global de-bias removes the GLOBAL (constant-across-gates)
offset; the PER-GATE / PER-TRACK residual is a constant within one track for one gate
-> does NOT average out within the gate-4 approach and is identical every lap.

This script decomposes each per-fix world-fix error into:
  (1) GLOBAL bias      -- constant across ALL gates (removable by global de-bias)
  (2) PER-GATE residual bias -- constant within a track for a given gate
                                (the un-filterable, per-track floor)
  (3) zero-mean per-fix NOISE  -- crushed by the KF

per NED axis, then PROJECTS onto the gate-4 frame:
  in-plane = E (lateral) + D (vertical); along-track = N (gate-4 normal ~ -N).
The binding in-plane miss = sqrt(E_resid^2 + D_resid^2).

It uses the REAL vision_cal.py re-survey (handoff/ultracode-vision-case-c) to recover
the global offset (= -MEASURED_FIX_BIAS) and the per-gate residuals via the SHIPPED
robust median/MAD averager (racer.gate_mapper.estimate_map_pose_aided). It then
computes the gate-4-specific residual in-plane bias, AFTER global de-bias, with a CI,
two independent ways (re-survey path + raw KF-accepted gate-4 fix path), and answers:
is sqrt(E_resid^2 + D_resid^2) < 0.05 m at gate-4 after de-bias?

Run:  PYTHONPATH="src;handoff/ultracode-vision-case-c-2026-06-13" .venv/Scripts/python.exe \
        handoff/ultracode-estimator-racespeed-2026-06-13/a2_bias.py
(or bash:  PYTHONPATH=src:handoff/ultracode-vision-case-c-2026-06-13 .venv/Scripts/python.exe ...)
Offline analysis ONLY. No live sim, no src edits, no commits.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CASE_C = REPO / "handoff/ultracode-vision-case-c-2026-06-13"
CHAR_DIR = REPO / "handoff/perception-char-2026-06-08"
sys.path.insert(0, str(CASE_C))

import vision_cal  # noqa: E402  (case-C prototype; built upon, not re-derived)
from racer.gate_mapper import MEASURED_FIX_BIAS_NED  # noqa: E402

RNG = np.random.default_rng(20260613)   # explicit seed for reproducible bootstrap CIs
N_BOOT = 20000
GATE4 = 4
AXES = ("N", "E", "D")
# Gate-4 frame: in-plane axes are E (lateral) and D (vertical); along-track is N.
INPLANE_IDX = (1, 2)   # E, D
ALONGTRACK_IDX = 0     # N
VARIANCE_BAR = 0.05    # the <0.05 m 1-sigma / un-filterable-bias bar (margin/3)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_good_course_rows():
    """course_60s 'good' fixes: associated, |off_ned|<3 m (the in-loop-accepted-like
    cut that reproduces the FACTS per-fix bias/noise). Returns (rows, offs[N,3])."""
    d = json.loads((CHAR_DIR / "characterize_course_60s.json").read_text())
    rows = []
    for r in d["rows"]:
        if not r.get("associated") or "off_ned" not in r:
            continue
        off = np.asarray(r["off_ned"], float)
        if np.linalg.norm(off) < 3.0:
            rows.append(r)
    offs = np.array([r["off_ned"] for r in rows], float)
    return rows, offs


def kf_accepted_gate4_fixes():
    """The fixes the case-C KF actually fuses on the gate-4 approach: associated to
    gate-4, 4-corner, innovation-gate-passing (maha <= 16.27 = navigator chi2 gate).
    Pull from BOTH the per-gate dense bundle (characterize_g4) and the course_60s run,
    dedup by frame_id. off_ned = p_fix - true_drone (world NED, m)."""
    by_fid = {}
    for fname in ("characterize_g4.json", "characterize_course_60s.json"):
        d = json.loads((CHAR_DIR / fname).read_text())
        for r in d["rows"]:
            if not (r.get("associated") and r.get("gate_id") == GATE4):
                continue
            if r.get("n_corners") != 4 or "off_ned" not in r:
                continue
            m = r.get("maha", np.nan)
            if not (np.isfinite(m) and m <= vision_cal.CHI2_GATE):
                continue
            by_fid[int(r["frame_id"])] = r
    rows = list(by_fid.values())
    offs = np.array([r["off_ned"] for r in rows], float)
    return rows, offs


# ---------------------------------------------------------------------------
# Bootstrap CI on a per-axis mean (bias) and on the in-plane bias norm
# ---------------------------------------------------------------------------
def boot_mean_ci(samples_1d, n_boot=N_BOOT, alpha=0.05):
    """Percentile bootstrap CI for the MEAN of a 1-D sample (the per-axis bias)."""
    s = np.asarray(samples_1d, float)
    n = len(s)
    idx = RNG.integers(0, n, size=(n_boot, n))
    means = s[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(s.mean()), float(lo), float(hi)


def boot_inplane_norm_ci(offs_ed, debias_ed=None, n_boot=N_BOOT, alpha=0.05):
    """Bootstrap CI for sqrt(E_bias^2 + D_bias^2). offs_ed: (N,2) per-fix [E,D] errors.
    If debias_ed given (2,), subtract it from every fix first (global de-bias)."""
    x = np.asarray(offs_ed, float).copy()
    if debias_ed is not None:
        x = x - np.asarray(debias_ed, float)[None, :]
    n = len(x)
    idx = RNG.integers(0, n, size=(n_boot, n))
    boot = x[idx].mean(axis=1)                      # (n_boot, 2) bootstrap mean [E,D]
    norms = np.sqrt((boot ** 2).sum(axis=1))
    point = float(np.sqrt((x.mean(axis=0) ** 2).sum()))
    lo, hi = np.percentile(norms, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


# ===========================================================================
def main():
    out = {"seed": 20260613, "n_boot": N_BOOT}

    # -- 0. reproduce the FACTS per-fix bias / noise (sanity) ----------------
    rows_good, offs_good = load_good_course_rows()
    out["course_good"] = {
        "n": len(rows_good),
        "per_fix_bias_ned": offs_good.mean(0).round(4).tolist(),
        "per_fix_noise_std_ned": offs_good.std(0, ddof=1).round(4).tolist(),
    }

    # -- 1. RE-SURVEY: per-gate offsets (est_centre - map) = -mean(off_ned)|gate --
    #    These are the per-gate registration BIASES. The mean across gates is the
    #    GLOBAL bias (component 1); the residual per gate is the PER-GATE bias (2).
    regs, diag = vision_cal.resurvey()
    reg_by_id = {r.gate_id: r for r in regs}
    offsets = np.array([reg_by_id[g].offset_ned for g in sorted(reg_by_id)], float)  # (6,3)
    offset_sigma = np.array([reg_by_id[g].offset_std_ned for g in sorted(reg_by_id)], float)
    gate_ids = sorted(reg_by_id)

    # GLOBAL bias in the GATE-MEASUREMENT sense = mean of per-gate offsets across gates.
    # (offset = est-map = -mean(off_ned); so this equals -MEASURED_FIX_BIAS in fix sense.)
    global_bias_gate = offsets.mean(0)              # mean across 6 gates (unweighted)
    per_gate_resid = offsets - global_bias_gate     # (6,3) genuinely per-gate, non-cancelling
    across_gate_sigma = per_gate_resid.std(0, ddof=1)

    out["resurvey"] = {
        "diag": {k: v for k, v in diag.items() if k != "mapper_diag"},
        "per_gate_offset_ned": {int(g): offsets[i].round(4).tolist() for i, g in enumerate(gate_ids)},
        "per_gate_offset_sigma_ned": {int(g): offset_sigma[i].round(4).tolist() for i, g in enumerate(gate_ids)},
        "global_bias_gate_sense_ned": global_bias_gate.round(4).tolist(),
        "global_bias_fix_sense_ned(=-gate)": (-global_bias_gate).round(4).tolist(),
        "MEASURED_FIX_BIAS_NED(ref)": list(map(float, MEASURED_FIX_BIAS_NED)),
        "per_gate_residual_ned": {int(g): per_gate_resid[i].round(4).tolist() for i, g in enumerate(gate_ids)},
        "across_gate_residual_sigma_ned": across_gate_sigma.round(4).tolist(),
    }

    # -- 2. GATE-4 specific, from the re-survey (per-gate registration) ------
    g4_i = gate_ids.index(GATE4)
    g4_offset = offsets[g4_i]                       # est-map registration bias (gate sense)
    g4_resid_after_global = per_gate_resid[g4_i]    # AFTER removing global bias
    g4_est_sigma = offset_sigma[g4_i]               # estimation 1-sigma of the offset

    # in-plane (E,D) before vs after global de-bias (re-survey path)
    g4_inplane_raw = float(np.hypot(g4_offset[1], g4_offset[2]))
    g4_inplane_debiased = float(np.hypot(g4_resid_after_global[1], g4_resid_after_global[2]))
    # estimation-uncertainty band on the de-biased in-plane residual (propagate offset sigma;
    # global-bias estimation uncertainty ~ sigma/sqrt(6) is small, fold it in quadrature)
    glob_sigma_ed = np.sqrt((offset_sigma[:, INPLANE_IDX] ** 2).sum(0)) / len(gate_ids)
    g4_inplane_sigma_components = np.sqrt(g4_est_sigma[list(INPLANE_IDX)] ** 2 + glob_sigma_ed ** 2)

    out["gate4_resurvey_path"] = {
        "offset_ned (raw, est-map)": g4_offset.round(4).tolist(),
        "residual_ned (after global de-bias)": g4_resid_after_global.round(4).tolist(),
        "offset_est_sigma_ned": g4_est_sigma.round(4).tolist(),
        "inplane_E_D_raw": [round(float(g4_offset[1]), 4), round(float(g4_offset[2]), 4)],
        "inplane_E_D_after_debias": [round(float(g4_resid_after_global[1]), 4),
                                     round(float(g4_resid_after_global[2]), 4)],
        "inplane_bias_norm_raw_m": round(g4_inplane_raw, 4),
        "inplane_bias_norm_after_global_debias_m": round(g4_inplane_debiased, 4),
        "inplane_residual_est_sigma_E_D": g4_inplane_sigma_components.round(4).tolist(),
        "alongtrack_N_resid_after_debias_m": round(float(g4_resid_after_global[ALONGTRACK_IDX]), 4),
    }

    # -- 3. GATE-4 specific, from the RAW KF-accepted gate-4 fix stream ------
    #    This is the most direct estimate of the bias the KF sees as it approaches g4:
    #    the mean of the actual accepted fixes. off_ned = fix - true, so the bias the KF
    #    tracks is +mean(off_ned). Global de-bias subtracts MEASURED_FIX_BIAS from each fix.
    g4_rows, g4_offs = kf_accepted_gate4_fixes()
    n4 = len(g4_offs)
    g4_fix_bias = g4_offs.mean(0)                   # what the KF tracks as truth (fix sense)
    g4_fix_noise = g4_offs.std(0, ddof=1)
    # global de-bias = subtract MEASURED_FIX_BIAS_NED from each fix
    debias = np.asarray(MEASURED_FIX_BIAS_NED, float)
    g4_offs_db = g4_offs - debias[None, :]
    g4_fix_bias_db = g4_offs_db.mean(0)

    # bootstrap CIs on the in-plane (E,D) bias norm, raw and de-biased
    raw_pt, raw_lo, raw_hi = boot_inplane_norm_ci(g4_offs[:, INPLANE_IDX])
    db_pt, db_lo, db_hi = boot_inplane_norm_ci(g4_offs[:, INPLANE_IDX], debias_ed=debias[list(INPLANE_IDX)])
    # per-axis CIs (de-biased)
    e_b, e_lo, e_hi = boot_mean_ci(g4_offs_db[:, 1])
    dd_b, d_lo, d_hi = boot_mean_ci(g4_offs_db[:, 2])
    n_b, n_lo, n_hi = boot_mean_ci(g4_offs_db[:, 0])

    out["gate4_raw_fix_path"] = {
        "n_kf_accepted_fixes": int(n4),
        "true_range_span_m": [round(min(r["true_range_m"] for r in g4_rows), 2),
                              round(max(r["true_range_m"] for r in g4_rows), 2)],
        "fix_bias_ned (raw, KF tracks this)": g4_fix_bias.round(4).tolist(),
        "fix_noise_std_ned": g4_fix_noise.round(4).tolist(),
        "fix_bias_ned (after global de-bias)": g4_fix_bias_db.round(4).tolist(),
        "inplane_bias_norm_raw_m": round(raw_pt, 4),
        "inplane_bias_norm_raw_ci95": [round(raw_lo, 4), round(raw_hi, 4)],
        "inplane_bias_norm_after_global_debias_m": round(db_pt, 4),
        "inplane_bias_norm_after_debias_ci95": [round(db_lo, 4), round(db_hi, 4)],
        "E_resid_after_debias_ci95": [round(e_b, 4), [round(e_lo, 4), round(e_hi, 4)]],
        "D_resid_after_debias_ci95": [round(dd_b, 4), [round(d_lo, 4), round(d_hi, 4)]],
        "N_alongtrack_resid_after_debias_ci95": [round(n_b, 4), [round(n_lo, 4), round(n_hi, 4)]],
    }

    # -- 4. VERDICT ----------------------------------------------------------
    # Report BOTH paths. The headline un-filterable floor = the de-biased in-plane
    # residual. Use the more conservative (larger) of the two estimates as the floor,
    # and the upper CI bound for the clears-0.05 test.
    floor_raw = max(g4_inplane_raw, raw_pt)
    floor_debiased = max(g4_inplane_debiased, db_pt)
    # clears 0.05 only if even the UPPER 95% CI of the de-biased norm is < 0.05
    clears = (db_hi < VARIANCE_BAR) and (g4_inplane_debiased < VARIANCE_BAR)

    out["VERDICT"] = {
        "inplane_bias_floor_raw_m": round(floor_raw, 4),
        "inplane_bias_floor_after_global_debias_m": round(floor_debiased, 4),
        "bar_m": VARIANCE_BAR,
        "clears_0.05_after_debias": bool(clears),
        "debias_ci95_upper_m": round(db_hi, 4),
        "headline": (
            f"Gate-4 un-filterable in-plane bias floor: RAW ~{floor_raw:.2f} m, "
            f"after GLOBAL de-bias ~{floor_debiased:.2f} m "
            f"(95% CI [{db_lo:.2f},{db_hi:.2f}]). "
            f"{'CLEARS' if clears else 'DOES NOT CLEAR'} the 0.05 m bar."
        ),
    }

    print(json.dumps(out, indent=2, default=str))
    # also drop a machine-readable copy beside this script
    (Path(__file__).with_name("a2_bias_results.json")).write_text(
        json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    main()
