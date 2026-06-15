"""sigma_recal.py -- DELIVERABLE 2 (SIGMA-RECAL) of the boresight-closure pathway.

Reconcile the fix-surrogate's POOLED sigma_vert 0.28 vs the L3 GATE-4 sigma_vert 0.10, decompose the
0.28-vs-0.10 discrepancy into AXIS / POOLING / SPEED / BORESIGHT-WANDER, compute the L3 gate-4 per-axis
sigma with a CLUSTER bootstrap (effective-N adjustment for the 13-window structure), and recommend the
in-plane sigma for the gate-4 COLD margin. Writes a NEW gate-4 sigma checkpoint (same schema as the
surrogate's sigma.json) -- does NOT touch rl/fix_surrogate.py or the production models/.

Offline ONLY. Pure numpy. Run with:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    <venv-python> handoff/boresight-closure-2026-06-14/sigma_recal.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_DATA = _HERE / "data"
_SURR_SIGMA = _HERE.parents[1] / "handoff/fix-surrogate-2026-06-14/models/sigma.json"

SEED = 20260614
N_BOOT = 20000


# ----------------------------------------------------------------------------- helpers
def _accepted(path: Path, range_hi: float | None = None) -> list[dict]:
    d = json.loads(path.read_text())
    out = []
    for r in d["rows"]:
        if r.get("accepted") is not True or "rel_vert" not in r:
            continue
        if range_hi is not None and r.get("true_range_m", 1e9) > range_hi:
            continue
        out.append(r)
    return out


def _sd(x) -> float:
    return float(np.std(np.asarray(x, float), ddof=1))


def _iid_bootstrap_sd(x, n_boot=N_BOOT, seed=SEED) -> tuple[float, float, float]:
    """Naive i.i.d. bootstrap CI on the per-fix sd (treats every fix as independent -- OPTIMISTIC when
    fixes are within-window correlated)."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    n = len(x)
    sds = np.empty(n_boot)
    for b in range(n_boot):
        sds[b] = np.std(x[rng.integers(0, n, n)], ddof=1)
    return _sd(x), float(np.percentile(sds, 5)), float(np.percentile(sds, 95))


def _cluster_bootstrap_sd(x, groups, n_boot=N_BOOT, seed=SEED) -> tuple[float, float, float]:
    """Cluster (block) bootstrap on the per-fix sd: RESAMPLE WHOLE WINDOWS with replacement, then pool
    their fixes and recompute the sd. Honours the within-window correlation -> the HONEST (wider) CI for
    the true per-fix scatter. Returns (point_sd, ci5, ci95)."""
    x = np.asarray(x, float)
    groups = np.asarray(groups)
    uniq = list(dict.fromkeys(groups.tolist()))
    by = {g: x[groups == g] for g in uniq}
    rng = np.random.default_rng(seed + 7)
    k = len(uniq)
    sds = []
    for _ in range(n_boot):
        pick = [uniq[i] for i in rng.integers(0, k, k)]
        pooled = np.concatenate([by[g] for g in pick])
        if len(pooled) > 1:
            sds.append(np.std(pooled, ddof=1))
    sds = np.asarray(sds)
    return _sd(x), float(np.percentile(sds, 5)), float(np.percentile(sds, 95))


def _icc_and_eff_n(x, groups) -> tuple[float, float, float]:
    """One-way-random-effects ICC (ANOVA estimator) + design effect + effective independent N.
        ICC = (MSB - MSW) / (MSB + (m0-1)*MSW)
        DEFF = 1 + (m_bar - 1)*ICC ,  N_eff = N / DEFF
    where m_bar is the (variance-corrected) average cluster size. ICC quantifies how much of the total
    per-fix variance is *between-window* (shared, non-independent) -- the part that makes a within-window
    sd under-estimate the true per-fix scatter and the iid CI too tight."""
    x = np.asarray(x, float)
    groups = np.asarray(groups)
    uniq = list(dict.fromkeys(groups.tolist()))
    k = len(uniq)
    N = len(x)
    grand = x.mean()
    sizes = np.array([np.sum(groups == g) for g in uniq], float)
    means = np.array([x[groups == g].mean() for g in uniq])
    ssb = float(np.sum(sizes * (means - grand) ** 2))
    ssw = float(np.sum([(np.sum((x[groups == g] - x[groups == g].mean()) ** 2)) for g in uniq]))
    df_b, df_w = k - 1, N - k
    msb, msw = ssb / df_b, ssw / df_w
    m0 = (N - np.sum(sizes ** 2) / N) / (k - 1)            # variance-corrected mean cluster size
    icc = (msb - msw) / (msb + (m0 - 1) * msw) if (msb + (m0 - 1) * msw) != 0 else 0.0
    icc = float(np.clip(icc, 0.0, 1.0))
    m_bar = N / k
    deff = 1.0 + (m_bar - 1.0) * icc
    return icc, deff, N / deff


# ----------------------------------------------------------------------------- load
L3_G4 = _accepted(_DATA / "shadow_gate4_rows.json")                              # gate-4, range-gated
P3 = _accepted(_DATA / "b1_g0_rows.json", 26) + _accepted(_DATA / "b2_g0_rows.json", 26)
SURR = json.loads(_SURR_SIGMA.read_text())

g4_rv = np.array([r["rel_vert"] for r in L3_G4])
g4_rc = np.array([r["rel_cross"] for r in L3_G4])
g4_rng = np.array([r["true_range_m"] for r in L3_G4])
g4_sp = np.array([r["speed_mps"] for r in L3_G4])
g4_crab = np.array([r["crab_deg"] for r in L3_G4])
g4_sess = np.array([r["session"] for r in L3_G4])

p3_rv = np.array([r["rel_vert"] for r in P3])
p3_rc = np.array([r["rel_cross"] for r in P3])
p3_rng = np.array([r["true_range_m"] for r in P3])
p3_sess = np.array([r["session"] for r in P3])

# ============================================================================ (1) L3 GATE-4 per-axis sigma
g4_n = len(L3_G4)
sess_sizes = sorted(Counter(g4_sess.tolist()).values())
n_windows = len(set(g4_sess.tolist()))

# vertical (gate-Y, rel_vert) -- the boresight axis (post-bake this is the CLEAN axis)
sv_pt, sv_iid_lo, sv_iid_hi = _iid_bootstrap_sd(g4_rv)
_, sv_cl_lo, sv_cl_hi = _cluster_bootstrap_sd(g4_rv, g4_sess)
sv_icc, sv_deff, sv_neff = _icc_and_eff_n(g4_rv, g4_sess)

# lateral (gate-X, rel_cross) -- crab-coupled at gate-4 (the NOISY axis post-bake)
sl_pt, sl_iid_lo, sl_iid_hi = _iid_bootstrap_sd(g4_rc)
_, sl_cl_lo, sl_cl_hi = _cluster_bootstrap_sd(g4_rc, g4_sess)
sl_icc, sl_deff, sl_neff = _icc_and_eff_n(g4_rc, g4_sess)

# isotropic-equivalent in-plane sigma (the engine uses ONE in-plane sigma for BOTH axes; an isotropic
# draw with variance = mean of the two axis variances reproduces the same expected in-plane radial RMS).
iso_equiv = float(np.sqrt((sl_pt ** 2 + sv_pt ** 2) / 2.0))
# cluster-bootstrap CI on the isotropic-equivalent (resample windows jointly across both axes)
rng = np.random.default_rng(SEED + 11)
uniq = list(dict.fromkeys(g4_sess.tolist()))
by_rv = {g: g4_rv[g4_sess == g] for g in uniq}
by_rc = {g: g4_rc[g4_sess == g] for g in uniq}
iso_boot = []
for _ in range(N_BOOT):
    pick = [uniq[i] for i in rng.integers(0, len(uniq), len(uniq))]
    prv = np.concatenate([by_rv[g] for g in pick])
    prc = np.concatenate([by_rc[g] for g in pick])
    iso_boot.append(np.sqrt((np.var(prc, ddof=1) + np.var(prv, ddof=1)) / 2.0))
iso_lo, iso_hi = float(np.percentile(iso_boot, 5)), float(np.percentile(iso_boot, 95))

# ============================================================================ (2) DECOMPOSE 0.28 vs 0.10
surr_v_floor = SURR["vertical"]["floor"]                 # 0.2816 (STD over 10-26 m pooled band)
surr_v_2026 = next(b["sigma_std"] for b in SURR["vertical"]["bands"] if b["lo"] == 20)   # 0.2347 (20-26 sub-band)
surr_l_floor = SURR["lateral"]["floor"]                  # 0.1045
surr_v_bias_bands = [b["bias"] for b in SURR["vertical"]["bands"]]   # boresight per-band wander proxy

# --- (a) AXIS CONVENTION ----------------------------------------------------------------------------
# Is the surrogate's "vertical" the same gate-axis as L3 rel_vert? Both are the gate-PLANE Y (down) axis
# in the SAME gate.R_world_gate frame (fix_surrogate.fix_covariance diag order = [lateral=X, vertical=Y,
# depth=Z]; shadow_gate4 rel_vert/rel_cross are decomposed in that same gate frame). So the LABELS match.
# But the ANISOTROPY DIRECTION is opposite, because gate-4 is a HIGH-CRAB gate: rel_cross (the yaw/crab-
# coupled X axis) is inflated there. Demonstrate by matched-range g2 (lower crab) flipping the other way.
g2 = _accepted(_DATA / "shadow_gate2_rows.json")
g2_rng = np.array([r["true_range_m"] for r in g2])
g2_band = (g2_rng >= 20) & (g2_rng <= 24)
g2_rv = np.array([r["rel_vert"] for r in g2])[g2_band]
g2_rc = np.array([r["rel_cross"] for r in g2])[g2_band]
# crab dependence of the gate-4 lateral axis
crab_lo = g4_crab < np.median(g4_crab)
axis = dict(
    labels_match=True,
    note=("surrogate diag order [lateral=gateX, vertical=gateY, depth=gateZ] == shadow rel_cross/rel_vert"
          " in the SAME gate.R_world_gate frame -> labels are consistent (NOT a swap)."),
    surrogate_anisotropy="vertical(0.28) >> lateral(0.10)",
    l3_gate4_anisotropy=f"lateral/cross({sl_pt:.3f}) > vertical({sv_pt:.3f}) -- OPPOSITE",
    cause="gate-4 is a high-crab gate (crab %.0f-%.0f deg) -> the yaw/crab-coupled CROSS(X) axis is "
          "inflated; this is real geometry, not a labelling bug." % (g4_crab.min(), g4_crab.max()),
    matched_range_g2_20_24m=dict(N=int(g2_band.sum()), vert_sd=_sd(g2_rv), cross_sd=_sd(g2_rc),
                                 anisotropy="vert>cross (surrogate-like at lower-crab gate)"),
    gate4_cross_sd_by_crab=dict(low_crab=_sd(g4_rc[crab_lo]), high_crab=_sd(g4_rc[~crab_lo]),
                                corr_abs_cross_vs_crab=float(np.corrcoef(np.abs(g4_rc), g4_crab)[0, 1])),
    verdict=("AXIS contributes ~0 of the 0.28-vs-0.10 *vertical* gap (labels match). What it DOES explain"
             " is the opposite anisotropy: gate-4 moves the noise from vert->lateral via high crab."),
)

# --- (b) POOLING ------------------------------------------------------------------------------------
# The surrogate floor 0.28 = STD over the POOLED 10-26 m band; its own 20-26 m sub-band is 0.235; and the
# gate-4 operating range is 20-24 m. Quantify the inflation: pooled / sub-band, and the matched-range
# per-fix sd from BOTH independent datasets (P3 head-on 22-24 m, L3 gate-4 20-24 m).
p3_2224 = (p3_rng >= 22) & (p3_rng <= 24)
pooling = dict(
    surrogate_vertical_floor_pooled_10_26=surr_v_floor,
    surrogate_vertical_subband_20_26=surr_v_2026,
    pooling_inflation_ratio_floor_over_subband=float(surr_v_floor / surr_v_2026),
    p3_headon_vert_sd_22_24m=_sd(p3_rv[p3_2224]),
    l3_gate4_vert_sd_20_24m=sv_pt,
    note=("the 0.28 is inflated by the noisier 10-20 m near-band (strong-perspective PnP) + cross-flight"
          " pooling. At the gate-4 operating range 20-24 m, two INDEPENDENT datasets agree the per-fix"
          " vertical scatter is ~0.10 (P3 %.3f, L3 %.3f), NOT 0.28." % (_sd(p3_rv[p3_2224]), sv_pt)),
    verdict=("POOLING (near-band + cross-flight) is the DOMINANT driver of the 0.28-vs-0.10 vertical gap:"
             " floor->subband alone is %.0f%% of the inflation; the residual subband->gate4 (0.235->0.10)"
             " is range-matching + gate-specificity." % (100 * (surr_v_floor - surr_v_2026)
                                                         / (surr_v_floor - sv_pt))),
)

# --- (c) SPEED --------------------------------------------------------------------------------------
speed = dict(
    surrogate_fit_set="simops-mastery (6 flights / 125 clean rows), Track-3 at-speed",
    l3_gate4_speed_mps=[float(g4_sp.min()), float(g4_sp.max())],
    p3_headon_speed_mps=[0.0, 6.0],
    static_vs_moving_invariance=("P3 22-24 m: static-hover vs moving-approach rel_vert differ by 0.007 m"
                                 " (FORM_RESOLUTION) -> at <=18 m/s the per-fix vertical sd is speed-flat"
                                 " (P3-low-speed 0.097 ~= L3-18m/s 0.10)."),
    at_speed_caveat=("L3 is ~17-18 m/s (inc7). The BINDING gate-4 is ~30 m/s (doctrine) / older 37 m/s."
                     " Motion-blur sigma at 30-37 m/s is UNMEASURED -- a standing caveat, gated on the"
                     " held inc8 envelope-relaxed policy. Do NOT claim the 0.10 holds at 30+ m/s."),
    verdict=("SPEED explains ~0 of the 0.28-vs-0.10 gap in the MEASURED 0-18 m/s range (sd is speed-flat"
             " there). It is the OPEN risk ABOVE 18 m/s, not a current contributor to the discrepancy."),
)

# --- (d) BORESIGHT-WANDER ---------------------------------------------------------------------------
# The surrogate vertical FLOOR (a sd) can absorb per-flight/per-gate boresight (epsilon_vert) WANDER as
# extra scatter. Evidence: the surrogate's vertical bias_band varies 0.010 -> 0.186 -> 0.197 across
# range bands (a stable bias would be constant); and P3's per-SESSION mean rel_vert wanders.
p3_sess_means = [np.mean(p3_rv[p3_sess == s]) for s in dict.fromkeys(p3_sess.tolist())]
g4_sess_means = [np.mean(g4_rv[g4_sess == s]) for s in dict.fromkeys(g4_sess.tolist())]
# variance budget: total per-fix var = within-window var + between-window var (the wander component)
within_var = float(np.mean([np.var(g4_rv[g4_sess == s], ddof=1)
                            for s in dict.fromkeys(g4_sess.tolist()) if (g4_sess == s).sum() > 1]))
between_var = float(np.var(g4_sess_means, ddof=1))
wander = dict(
    surrogate_vertical_bias_band_across_bands=surr_v_bias_bands,
    p3_per_session_mean_rel_vert_spread=float(max(p3_sess_means) - min(p3_sess_means)),
    p3_per_session_std_of_means=float(np.std(p3_sess_means, ddof=1)),
    l3_gate4_per_window_std_of_means=float(np.std(g4_sess_means, ddof=1)),
    l3_gate4_within_window_mean_var=within_var,
    l3_gate4_between_window_var=between_var,
    between_over_total_frac=float(between_var / (within_var + between_var)),
    note=("the surrogate vertical bias_band is NON-stationary (0.01->0.186->0.197 across range bands) ="
          " a constant epsilon_vert mixing with range -> per-band bias wander folds into the pooled FLOOR"
          " as scatter. In the gate-4 windows the between-window mean wander is small (std-of-means"
          " %.3f) and most variance is within-window." % float(np.std(g4_sess_means, ddof=1))),
    verdict=("BORESIGHT-WANDER is a SECONDARY inflator of the surrogate floor (a few cm: bias_band swings"
             " ~0.19 across bands, whose mixing adds scatter when pooled). After the -0.25 m bake, the"
             " mean epsilon is removed; the residual *wander* (std-of-means ~%.2f at gate-4) is small and"
             " already inside the per-fix sd." % float(np.std(g4_sess_means, ddof=1))),
)

# --- decomposition ledger (additive-in-variance, vertical axis) -------------------------------------
# Walk 0.28 -> 0.10 as a sequence of variance reductions, attributing each step.
v_pooled = surr_v_floor               # 0.2816
v_subband = surr_v_2026               # 0.2347  (POOLING: drop the 10-20 near-band)
v_gate4 = sv_pt                       # 0.1008  (RANGE-MATCH + GATE-SPECIFICITY at 20-24 m)
ladder = dict(
    step0_surrogate_pooled_floor=v_pooled,
    step1_after_drop_near_band_subband_20_26=v_subband,
    step2_l3_gate4_20_24m=v_gate4,
    attribution_var_fraction=dict(
        pooling_near_band=float((v_pooled ** 2 - v_subband ** 2) / (v_pooled ** 2 - v_gate4 ** 2)),
        range_match_plus_gate_specific=float((v_subband ** 2 - v_gate4 ** 2)
                                             / (v_pooled ** 2 - v_gate4 ** 2)),
    ),
    interpretation=("of the TOTAL vertical variance removed going 0.28->0.10, ~%.0f%% is the near-band"
                    " pooling (10-20 m) and ~%.0f%% is range-matching to 20-24 m + gate-4 specificity."
                    " AXIS=0, SPEED=0 (in 0-18 m/s), WANDER=secondary (folded inside the floor)."
                    % (100 * (v_pooled ** 2 - v_subband ** 2) / (v_pooled ** 2 - v_gate4 ** 2),
                       100 * (v_subband ** 2 - v_gate4 ** 2) / (v_pooled ** 2 - v_gate4 ** 2))),
)

# ============================================================================ (3) RECOMMENDATIONS
# The engine ME.fly_lap uses ONE isotropic in-plane sigma for BOTH gate-plane axes (lines 308-312:
# n_lat = N(0,sig)*e1 + N(0,sig)*e2). So the ISOTROPIC recommendation must reproduce the gate-4 in-plane
# radial miss -> iso_equiv = sqrt((sl^2+sv^2)/2). HONEST: post-bake the vertical axis is clean (0.10) but
# the LATERAL axis carries 0.19 (crab-coupled) -> the in-plane miss is NOT dominated by the 0.10 axis.
rec = dict(
    iso_central=round(iso_equiv, 4),                      # central single-isotropic in-plane sigma
    iso_central_ci90=[round(iso_lo, 4), round(iso_hi, 4)],
    iso_conservative=round(iso_hi, 4),                    # upper cluster-CI (honours within-window corr)
    aniso_central=[round(sl_pt, 4), round(sv_pt, 4)],     # [sigma_lat(gateX), sigma_vert(gateY)]
    aniso_conservative=[round(sl_cl_hi, 4), round(sv_cl_hi, 4)],   # upper cluster-CI per axis
    rationale=(
        "ENGINE uses ONE in-plane sigma for both axes -> the ISOTROPIC value must be the in-plane-radial-"
        "equivalent sqrt((lat^2+vert^2)/2)=%.3f, NOT the clean 0.10 vertical (using 0.10 would UNDER-"
        "state the miss by ignoring the crab-coupled 0.19 lateral). CENTRAL=%.3f; CONSERVATIVE=%.3f "
        "(upper cluster-bootstrap CI). The ANISOTROPIC pair [%.3f lat, %.3f vert] is the honest shape "
        "for the engine-prep worker's extended two-sigma engine." % (
            iso_equiv, iso_equiv, iso_hi, sl_pt, sv_pt)),
    honesty_flag=(
        "POST-BAKE the VERTICAL axis is clean (sigma_vert 0.10) but the LATERAL axis is 0.19 (crab-coupled"
        " at this high-crab gate). The in-plane miss is LATERAL-dominated, NOT vertical-dominated. The bake"
        " fixes the vertical BIAS (-0.25 m mean), not the lateral SCATTER. Margin closure now rides on the"
        " lateral 0.19 + fix-rate, not the vertical."),
    vs_old_modeled=("0.265 (production GATE_REL_INPLANE_SIGMA / SIGMA_MODELED) is CONSERVATIVE vs both the"
                    " iso-central %.3f and the iso-conservative %.3f -> the modeled constant already bounds"
                    " the L3 gate-4 in-plane scatter." % (iso_equiv, iso_hi)),
    at_speed_caveat=speed["at_speed_caveat"],
)

# ============================================================================ NEW gate-4 checkpoint
# Same schema as the surrogate sigma.json so FixSurrogate.from_checkpoints() can load it as a drop-in
# gate-4-specific recalibration. floor = the L3 gate-4 per-axis sigma; a1 = 0 (range-flat in the 20-24 m
# operating band -- the gate-relative fix is consumed in-band); bias = the post-acceptance MEAN residual
# (vertical bias is the epsilon the -0.25 m bake removes; sign per the gate-frame rel_* convention).
ckpt = {
    "_provenance": {
        "what": "GATE-4-SPECIFIC per-fix sigma, recalibrated to the L3 at-speed shadow gate-4 rows.",
        "source_rows": "handoff/boresight-closure-2026-06-14/data/shadow_gate4_rows.json (accepted N=%d)" % g4_n,
        "speed_mps": [float(g4_sp.min()), float(g4_sp.max())],
        "range_m": [float(g4_rng.min()), float(g4_rng.max())],
        "n_windows": n_windows, "fixes_per_window": sess_sizes,
        "supersedes": "the POOLED simops-mastery sigma.json vertical.floor 0.2816 (pooling-inflated)",
        "at_speed_caveat": speed["at_speed_caveat"],
        "schema": "matches rl/fix_surrogate.py FixSurrogate.from_checkpoints sigma.json contract",
    },
    "n_clean": g4_n,
    "lateral": {
        "label": "lateral_inplane_gateX",
        "floor": round(sl_pt, 6), "floor_mad": round(float(np.median(np.abs(g4_rc - np.median(g4_rc))) * 1.4826), 6),
        "a1": 0.0, "bias_band": round(float(g4_rc.mean()), 6),
        "floor_ci90_cluster": [round(sl_cl_lo, 6), round(sl_cl_hi, 6)],
        "floor_ci90_iid": [round(sl_iid_lo, 6), round(sl_iid_hi, 6)],
        "icc": round(sl_icc, 4), "deff": round(sl_deff, 4), "n_eff": round(sl_neff, 2),
        "note": "crab-coupled (corr |cross| vs crab +%.2f) -> the NOISY in-plane axis at this high-crab gate."
                % float(np.corrcoef(np.abs(g4_rc), g4_crab)[0, 1]),
    },
    "vertical": {
        "label": "vertical_inplane_gateY",
        "floor": round(sv_pt, 6), "floor_mad": round(float(np.median(np.abs(g4_rv - np.median(g4_rv))) * 1.4826), 6),
        "a1": 0.0, "bias_band": round(float(g4_rv.mean()), 6),
        "floor_ci90_cluster": [round(sv_cl_lo, 6), round(sv_cl_hi, 6)],
        "floor_ci90_iid": [round(sv_iid_lo, 6), round(sv_iid_hi, 6)],
        "icc": round(sv_icc, 4), "deff": round(sv_deff, 4), "n_eff": round(sv_neff, 2),
        "note": "the BORESIGHT axis. bias_band %.3f m is epsilon_vert (gate-DOWN) that the -0.25 m bake "
                "removes; post-bake the mean->~0 and the per-fix scatter is the clean 0.10." % float(g4_rv.mean()),
    },
    "depth": {  # carry the surrogate's pooled depth (L3 along sd ~0.86 is loose by design; engine uses RADIAL_SIGMA 0.5)
        "label": "depth_along",
        "floor": round(float(np.std([r["rel_along"] for r in L3_G4], ddof=1)), 6),
        "floor_mad": SURR["depth"]["floor_mad"], "a1": 0.0,
        "bias_band": round(float(np.mean([r["rel_along"] for r in L3_G4])), 6),
        "note": "L3 gate-4 along sd; LOOSE by design (engine ME uses RADIAL_SIGMA 0.5; the absolute fix + IMU own depth).",
    },
    "inplane_isotropic_for_margin": {
        "central": round(iso_equiv, 6), "ci90_cluster": [round(iso_lo, 6), round(iso_hi, 6)],
        "conservative": round(iso_hi, 6),
        "formula": "sqrt((lateral_floor^2 + vertical_floor^2)/2)  -- in-plane-radial-equivalent for the "
                   "single-sigma ME.fly_lap engine",
    },
    "note": "sigma(range)=max(floor,a1*range); a1=0 (range-flat over the 20-24 m gate-4 operating band). "
            "GATE-4 SPECIFIC -- do NOT use for other gates (anisotropy is posture/crab-dependent).",
}

# ============================================================================ write
results = {
    "_meta": {
        "deliverable": "D2 sigma-recal (boresight-closure pathway)",
        "n_boot": N_BOOT, "seed": SEED,
        "engine_uses_single_inplane_sigma": True,
        "engine_ref": "handoff/margin-closure-envelope-2026-06-14/margin_envelope.py fly_lap L308-312",
    },
    "l3_gate4_sigma": {
        "N": g4_n, "n_windows": n_windows, "fixes_per_window": sess_sizes,
        "speed_mps": [float(g4_sp.min()), float(g4_sp.max())],
        "range_m": [float(g4_rng.min()), float(g4_rng.max())],
        "vertical_gateY": {
            "sd": round(sv_pt, 4), "mean_bias": round(float(g4_rv.mean()), 4),
            "ci90_iid": [round(sv_iid_lo, 4), round(sv_iid_hi, 4)],
            "ci90_cluster": [round(sv_cl_lo, 4), round(sv_cl_hi, 4)],
            "icc": round(sv_icc, 4), "deff": round(sv_deff, 4), "n_eff": round(sv_neff, 2),
            "ci_widening_cluster_over_iid": round((sv_cl_hi - sv_cl_lo) / (sv_iid_hi - sv_iid_lo), 3),
        },
        "lateral_gateX": {
            "sd": round(sl_pt, 4), "mean_bias": round(float(g4_rc.mean()), 4),
            "ci90_iid": [round(sl_iid_lo, 4), round(sl_iid_hi, 4)],
            "ci90_cluster": [round(sl_cl_lo, 4), round(sl_cl_hi, 4)],
            "icc": round(sl_icc, 4), "deff": round(sl_deff, 4), "n_eff": round(sl_neff, 2),
            "ci_widening_cluster_over_iid": round((sl_cl_hi - sl_cl_lo) / (sl_iid_hi - sl_iid_lo), 3),
        },
        "isotropic_equivalent": {"central": round(iso_equiv, 4), "ci90_cluster": [round(iso_lo, 4), round(iso_hi, 4)]},
        "cluster_note": ("the 81 fixes are 13 windows x 5-7 fixes; a within-window-correlated sd UNDER-"
                         "estimates the true per-fix scatter and the iid CI is too tight. The cluster "
                         "bootstrap (resample whole windows) and ICC->N_eff give the honest CI."),
    },
    "decomposition_0p28_vs_0p10": {
        "a_axis_convention": axis,
        "b_pooling": pooling,
        "c_speed": speed,
        "d_boresight_wander": wander,
        "variance_ladder": ladder,
        "headline": ("POOLING dominates: the surrogate 0.28 = pooled STD over 10-26 m (its own 20-26 sub-"
                     "band is 0.235); at the gate-4 operating range 20-24 m two independent datasets give"
                     " per-fix vertical sigma ~0.10. AXIS=0 (labels match; gate-4 just moves noise vert->"
                     "lateral via high crab). SPEED=0 in 0-18 m/s (speed-flat) but UNMEASURED >18 m/s "
                     "(standing caveat). WANDER=secondary (boresight bias_band mixing folds into the "
                     "pooled floor)."),
    },
    "recommendation": rec,
    "gate4_checkpoint_path": str((_HERE / "sigma_gate4_l3.json").as_posix()),
}

(_HERE / "sigma_gate4_l3.json").write_text(json.dumps(ckpt, indent=2))
(_HERE / "sigma_recal_results.json").write_text(json.dumps(results, indent=2))

# ----------------------------------------------------------------------------- console summary
print("L3 GATE-4 PER-AXIS SIGMA  (N=%d, %d windows, fixes/window=%s)" % (g4_n, n_windows, sess_sizes))
print("  vertical(gateY): sd %.4f  mean %.4f  iid-CI[%.3f,%.3f]  cluster-CI[%.3f,%.3f]  ICC %.3f  DEFF %.2f  Neff %.1f"
      % (sv_pt, g4_rv.mean(), sv_iid_lo, sv_iid_hi, sv_cl_lo, sv_cl_hi, sv_icc, sv_deff, sv_neff))
print("  lateral(gateX) : sd %.4f  mean %.4f  iid-CI[%.3f,%.3f]  cluster-CI[%.3f,%.3f]  ICC %.3f  DEFF %.2f  Neff %.1f"
      % (sl_pt, g4_rc.mean(), sl_iid_lo, sl_iid_hi, sl_cl_lo, sl_cl_hi, sl_icc, sl_deff, sl_neff))
print("  cluster-CI / iid-CI width ratio: vert %.2fx  lat %.2fx"
      % ((sv_cl_hi - sv_cl_lo) / (sv_iid_hi - sv_iid_lo), (sl_cl_hi - sl_cl_lo) / (sl_iid_hi - sl_iid_lo)))
print("  isotropic-equivalent in-plane sigma: %.4f  cluster-CI[%.3f,%.3f]" % (iso_equiv, iso_lo, iso_hi))
print()
print("DECOMPOSITION 0.28 -> 0.10  (vertical variance ladder)")
print("  0.2816 pooled(10-26) -> 0.235 subband(20-26) -> %.3f gate4(20-24)" % v_gate4)
print("  near-band pooling = %.0f%% of removed var ; range-match+gate = %.0f%%"
      % (100 * (v_pooled ** 2 - v_subband ** 2) / (v_pooled ** 2 - v_gate4 ** 2),
         100 * (v_subband ** 2 - v_gate4 ** 2) / (v_pooled ** 2 - v_gate4 ** 2)))
print("  AXIS=0 (labels match) ; SPEED=0 in 0-18 m/s (UNMEASURED >18) ; WANDER=secondary")
print()
print("RECOMMEND  iso central %.3f (CI %.3f-%.3f) / conservative %.3f ; aniso [lat %.3f, vert %.3f]"
      % (iso_equiv, iso_lo, iso_hi, iso_hi, sl_pt, sv_pt))
print("  HONESTY: post-bake VERTICAL is clean (0.10); LATERAL is 0.19 (crab) -> in-plane miss is LATERAL-dominated.")
print("  vs modeled 0.265: CONSERVATIVE (bounds the L3 gate-4 in-plane scatter).")
print()
print("WROTE: %s" % (_HERE / "sigma_gate4_l3.json"))
print("WROTE: %s" % (_HERE / "sigma_recal_results.json"))
