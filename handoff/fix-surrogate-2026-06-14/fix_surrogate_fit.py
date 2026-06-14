"""fix_surrogate_fit.py -- calibrate the inc8 analytic FIX-SURROGATE sub-models from the
cached Track-3 dataset (handoff/fix-surrogate-2026-06-14/data/, built by build_dataset.py).

Four independent sub-models, each a small swappable checkpoint under models/:
  A. accept   -- P(accept | a gate is in image at this geometry): logistic in {range, |bearing|,
                 view_angle} fit on the per-(frame,gate) in-image candidates. Factorizes the funnel
                 accept-rate as  P(>=1 gate in image) x P(accept | in image).
  B. sigma    -- per-fix lateral(in-plane cross-track) / vertical(in-plane) / depth(along-track)
                 1-sigma vs range, fit on the clean offered pooled fix-error rows. Reproduces the
                 binding-band MEASURED lateral sigma ~0.10 and depth ~0.8 (Track-3, vision-at-speed).
  C. crab     -- camera-pointing -> gate-in-FoV fraction -> accept-rate map: rotate every candidate's
                 camera-frame bearing by a yaw correction delta (= crab removed) and re-project; the
                 any-gate-in-image fraction vs crab, x P(accept|in-image-frame). Anchored at the
                 measured inc7 crab so crab=inc7 reproduces (in-FoV 0.33, accept 0.07) by construction.

Plus leave_one_flight_out_cv(): an adversarial calibration check (fit on 5 flights, predict the held-out
one) for A and B, so the report can state the surrogate is calibrated, not overfit.

Pure numpy/scipy (no sklearn). Deterministic. Run as __main__ to (re)write all checkpoints + print the
reproduction-vs-Track-3 table. Import the fit_* functions for the adversarial verification fan-out.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MODELS = HERE / "models"
CHI2 = 16.27
HFOV_HALF_DEG = 45.0          # horizontal half-FoV (fx=320, W=640 -> 90 deg full)
VFOV_HALF_DEG = 29.36         # vertical half-FoV (fy=320, H=360 -> 58.7 deg full)

# Camera intrinsics / principal point (racer.frames.CAMERA_INTRINSICS_K) -- duplicated as plain
# constants so the crab sweep needs no racer import (the fit library stays dependency-light).
FX = FY = 320.0
CX, CY = 320.0, 180.0
IMG_W, IMG_H = 640, 360


# ----------------------------------------------------------------------------- helpers
def _load_geom() -> dict:
    return dict(np.load(DATA / "geom_frames.npz", allow_pickle=True))


def _load_cand() -> dict:
    return dict(np.load(DATA / "cand_frames.npz", allow_pickle=True))


def _load_fix_errors() -> list[dict]:
    return json.loads((DATA / "fix_errors.json").read_text())


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
    """Ridge-regularised logistic regression via L-BFGS. X is (n, d) WITHOUT intercept; an
    intercept column is prepended. Features are z-scored internally for conditioning, then the
    weights are mapped back to RAW-feature space so the returned beta applies to [1, *raw]."""
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd < 1e-9] = 1.0
    Xs = (X - mu) / sd
    Xs1 = np.hstack([np.ones((len(Xs), 1)), Xs])
    d = Xs1.shape[1]

    def nll(b):
        p = _sigmoid(Xs1 @ b)
        eps = 1e-9
        ll = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)).sum()
        reg = 0.5 * l2 * np.sum(b[1:] ** 2)        # don't penalise intercept
        g = Xs1.T @ (p - y)
        g[1:] += l2 * b[1:]
        return ll + reg, g

    res = minimize(nll, np.zeros(d), jac=True, method="L-BFGS-B")
    bs = res.x
    # map standardised weights -> raw-feature weights:  b_raw_j = bs_j / sd_j ;
    # b0_raw = bs_0 - sum_j bs_j * mu_j / sd_j
    b_raw = np.zeros(d)
    b_raw[1:] = bs[1:] / sd
    b_raw[0] = bs[0] - np.sum(bs[1:] * mu / sd)
    return b_raw


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney AUC; 0.5 = chance."""
    pos = p[y == 1]
    neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# range AND range^2 -> the logit is QUADRATIC in range, i.e. a unimodal band-pass: accept is
# suppressed both at very short range (frontal-PnP depth flips / partial gates -> the <10 m junk)
# and beyond the navigator's range cap (~26 m), peaking in the ~18-24 m offered window. A linear-in-
# range logit cannot represent this (the measured accept rate is 0.03 -> 0.84 -> 0.00 across range).
ACCEPT_FEATURES = ["range", "range_sq", "abs_bearing", "view_angle"]


def _accept_design(cand: dict, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rng = cand["range"][mask]
    abear = np.abs(cand["bearing"][mask])
    va = cand["view_angle"][mask]
    X = np.column_stack([rng, rng ** 2, abear, va])
    y = cand["accepted_to_k"][mask].astype(float)
    return X, y


# ----------------------------------------------------------------------------- A. accept
def _bandpass(r, pmax, rlo, wlo, rhi, whi):
    r = np.asarray(r, float)
    return pmax * _sigmoid((r - rlo) / wlo) * _sigmoid((rhi - r) / whi)


def _fit_bandpass(centers, rates, counts):
    """Least-squares fit of the range band-pass to binned accept rates (weighted by sqrt count)."""
    from scipy.optimize import least_squares
    w = np.sqrt(np.maximum(counts, 1.0))

    def resid(p):
        pmax, rlo, wlo, rhi, whi = p
        return w * (_bandpass(centers, pmax, rlo, wlo, rhi, whi) - rates)

    p0 = [0.85, 13.0, 2.0, 27.0, 2.0]
    lb = [0.1, 4.0, 1.0, 18.0, 1.0]   # edge widths >= 1 m: honest sharp range cap, still a usable gradient
    ub = [1.0, 18.0, 8.0, 40.0, 8.0]
    res = least_squares(resid, p0, bounds=(lb, ub), max_nfev=5000)
    return [float(x) for x in res.x]


def fit_accept_model(cand: dict | None = None) -> dict:
    """P(accept | gate in image, geometry).

    The measured accept rate vs range is a sharp BAND-PASS (0.03 -> 0.84 -> 0.00): the navigator
    rejects very-close fixes (frontal-PnP depth flips, partial gates) and caps range at ~26 m, peaking
    in the ~18-24 m offered window. A logistic (even quadratic) cannot represent this against the huge
    far-range reject mass, so the PRIMARY model is a smooth two-sided band-pass in range:
        p_accept(geom) = in_image(geom) ? pmax*sig((r-rlo)/wlo)*sig((rhi-r)/whi) : p_out
    A quadratic-logit is kept alongside ONLY as a cross-flight rank-ordering (AUC) generalisation
    diagnostic -- it is NOT used for the probability level."""
    cand = cand if cand is not None else _load_cand()
    in_img = cand["in_image"]
    X, y = _accept_design(cand, in_img)
    rng_in = X[:, 0]
    # --- primary: range band-pass fit to fine-binned in-image accept rates ---
    edges = [0, 8, 12, 16, 20, 24, 28, 32, 40, 60, 200]
    centers, rates, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (rng_in >= lo) & (rng_in < hi)
        if m.sum() >= 1:
            centers.append(0.5 * (lo + min(hi, 60)))
            rates.append(float(y[m].mean()))
            counts.append(int(m.sum()))
    bp = _fit_bandpass(np.array(centers), np.array(rates), np.array(counts))
    p_bp = _bandpass(rng_in, *bp)
    # --- diagnostic: quadratic logistic (AUC only) ---
    beta = _fit_logistic(X, y, l2=2.0)
    p_log = _sigmoid(np.hstack([np.ones((len(X), 1)), X]) @ beta)
    auc = _auc(y, p_log)
    # reproduction table (coarse) -- emp vs band-pass
    coarse = [(0, 10), (10, 18), (18, 24), (24, 32), (32, 999)]
    table = []
    for lo, hi in coarse:
        m = (rng_in >= lo) & (rng_in < hi)
        if m.sum() >= 3:
            table.append(dict(lo=lo, hi=hi, n=int(m.sum()), p_emp=float(y[m].mean()),
                              p_bandpass=float(p_bp[m].mean()), p_logit=float(p_log[m].mean())))
    p_accept_out = float(cand["accepted_to_k"][~in_img].mean())
    return dict(
        kind="range_bandpass", bandpass_params=bp,
        bandpass_param_names=["pmax", "rlo", "wlo", "rhi", "whi"],
        n_in_image=int(in_img.sum()), n_accept_in_image=int(y.sum()),
        base_rate_in_image=float(y.mean()),
        bandpass_marginal=float(p_bp.mean()),
        logit_features=ACCEPT_FEATURES, logit_beta=[float(b) for b in beta], logit_auc=float(auc),
        p_accept_out_of_image=p_accept_out, range_table=table,
    )


def predict_accept_bandpass(bandpass_params, range_m):
    return _bandpass(range_m, *bandpass_params)


def predict_accept_logit(beta: list[float], range_m, abs_bearing_deg, view_angle_deg):
    b = np.asarray(beta, float)
    rng = np.atleast_1d(range_m).astype(float)
    X = np.column_stack([rng, rng ** 2,
                         np.atleast_1d(abs_bearing_deg).astype(float),
                         np.atleast_1d(view_angle_deg).astype(float)])
    return _sigmoid(np.hstack([np.ones((len(X), 1)), X]) @ b)


# ----------------------------------------------------------------------------- B. sigma
def _robust_sigma(a: np.ndarray) -> float:
    """1.4826*MAD -- robust to the heavy frontal-flip tail."""
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return float("nan")
    return float(1.4826 * np.median(np.abs(a - np.median(a))))


def fit_sigma_model(fix_rows: list[dict] | None = None) -> dict:
    """Per-fix in-plane (lateral, vertical) + depth (along) 1-sigma vs range, from the clean
    offered fix-error rows. Returns a flat floor + linear growth law per axis and the binding-band
    reproduction. Convention (gates face +-X, so the gate plane ~= world E-D, gate normal ~= N):
      lateral(in-plane cross-track) ~ pooled 'lateral';  vertical(in-plane) ~ 'vert'(offD);
      depth(along-track / gate-normal) ~ pooled 'along'."""
    rows = fix_rows if fix_rows is not None else _load_fix_errors()
    clean = [r for r in rows if r["offered"] and abs(r["absfix"]) < 3.0]
    rng = np.array([r["range"] for r in clean], float)
    lat = np.array([r["lateral"] for r in clean], float)
    vert = np.array([r["vert"] for r in clean], float)
    along = np.array([r["along"] for r in clean], float)
    tag = np.array([r["tag"] for r in clean])
    # per-flight n in the binding band (the small-n the adversarial overfit audit flagged)
    bb_mask = (rng >= 10) & (rng < 26)
    per_flight_n = {t: int(((tag == t) & bb_mask).sum()) for t in sorted(set(tag.tolist()))}

    def axis_law(vals, label):
        # Per-band sigma reported BOTH ways: std (the analyze_shadow_vision headline measure, which
        # the memory's "sigma~0.10" target is stated in) and robust 1.4826*MAD (tail-insensitive
        # cross-check). The clean subset (offered & |off|<3) is already de-tailed, so within the
        # binding band the two agree to ~0.03; the FLOOR is the std measure to match the stated target.
        bands = [(0, 10), (10, 20), (20, 26), (26, 999)]
        band_tab = []
        for lo, hi in bands:
            m = (rng >= lo) & (rng < hi)
            if m.sum() >= 2:
                band_tab.append(dict(lo=lo, hi=hi, n=int(m.sum()),
                                     sigma_std=float(np.std(vals[m])), sigma_mad=_robust_sigma(vals[m]),
                                     bias=float(np.median(vals[m]))))
        # binding band = 10..26 m (the transit window the offered fixes live in)
        bb = (rng >= 10) & (rng < 26)
        sig_band = float(np.std(vals[bb])) if bb.any() else float("nan")
        sig_band_mad = _robust_sigma(vals[bb])
        bias_band = float(np.median(vals[bb])) if bb.any() else float("nan")
        # bootstrap CI on the std floor (the adversarial overfit audit: only ~16-24 rows/flight ->
        # +-15-18% sampling uncertainty; report it rather than treating the floor as exact).
        vb = vals[bb]
        ci68 = [float("nan"), float("nan")]
        boot_std = float("nan")
        if vb.size >= 5:
            rg = np.random.default_rng(0)
            boots = np.array([np.std(vb[rg.integers(0, vb.size, vb.size)]) for _ in range(2000)])
            ci68 = [float(np.percentile(boots, 16)), float(np.percentile(boots, 84))]
            boot_std = float(np.std(boots))
        # gentle linear growth slope a1 (robust theil-sen on |val-median| vs range), floored at 0
        bbv = np.abs(vals[bb] - np.median(vals[bb]))
        bbr = rng[bb]
        a1 = 0.0
        if bbr.size >= 3 and np.ptp(bbr) > 1.0:
            sl = []
            for i in range(len(bbr)):
                for j in range(i + 1, len(bbr)):
                    if abs(bbr[j] - bbr[i]) > 1e-6:
                        sl.append((bbv[j] - bbv[i]) / (bbr[j] - bbr[i]))
            a1 = float(max(0.0, np.median(sl))) if sl else 0.0
        return dict(label=label, floor=float(sig_band), floor_mad=float(sig_band_mad), a1=a1,
                    bias_band=bias_band, floor_ci68=ci68, floor_boot_std=boot_std,
                    n_binding_band=int(bb.sum()), bands=band_tab)

    return dict(
        n_clean=len(clean), per_flight_n_binding_band=per_flight_n,
        lateral=axis_law(lat, "lateral_inplane"),
        vertical=axis_law(vert, "vertical_inplane"),
        depth=axis_law(along, "depth_along"),
        note=("sigma(range) = max(floor, a1*range); floor = STD over the 10-26 m binding band "
              "(matches the analyze_shadow_vision / memory ~0.10 target). floor_ci68 = bootstrap "
              "16-84 pct sampling CI -- the floors are NOT exact (only ~16-24 clean rows/flight)."),
    )


def predict_sigma(model: dict, axis: str, range_m: float) -> float:
    a = model[axis]
    return float(max(a["floor"], a["a1"] * float(range_m)))


# ----------------------------------------------------------------------------- C. crab map
def fit_crab_map(geom: dict | None = None, p_range_peak: float | None = None) -> dict:
    """Camera-pointing -> ACTIVE-gate-in-FoV fraction -> accept-rate, the lever inc8 controls.

    Per-frame counterfactual: hold the trajectory fixed and ask "if the policy flew at target crab C
    instead of its measured crab, where would the gate it is approaching sit in frame?" The body is
    yaw-crabbed off velocity; the gate it approaches is ~along velocity, so its camera AZIMUTH tracks
    -crab. With eps = (gate-vs-velocity azimuth, ~const per frame), azimuth(C) = eps -/+ C, recovered
    from the measured (azimuth_g, crab). Elevation is a pitch quantity -> unchanged by yaw crab. The
    active gate is in FoV iff |azimuth(C)| < HFoV/2 AND |elevation| < VFoV/2 AND in front (tz>0).
      accept_rate_to_active(C) = active_in_fov(C) * P(accept | active gate in image).

    Anchored so that C = the measured crab reproduces the measured active-gate in-FoV (separable-gate
    vs the pixel projection agree to ~1%). The any-gate coverage 0.33 / accept 0.069 are reported
    separately as the calibration provenance (they are NOT the reward target -- adjacent gates inflate
    'any-gate', which is why a naive any-gate sweep points the WRONG way)."""
    geom = geom if geom is not None else _load_geom()
    crab = geom["crab_deg"]
    az = geom["azimuth_g"]
    el = geom["elevation_g"]
    tz = geom["tz_g"]
    fin = np.isfinite(crab) & np.isfinite(az)
    crab, az, el, tz = crab[fin], az[fin], el[fin], tz[fin]
    in_fov_px = geom["in_fov_g"][fin]
    acc_g = geom["accepted_to_g"][fin]
    crab_inc7 = float(np.median(np.abs(crab)))
    # auto-detect the azimuth<->crab sign: eps = gate-vs-velocity azimuth should be the SMALL residual
    eps_plus = az + crab          # if azimuth ~ -crab
    eps_minus = az - crab         # if azimuth ~ +crab
    if abs(np.median(eps_plus)) <= abs(np.median(eps_minus)):
        eps, sign = eps_plus, +1.0   # azimuth(C) = eps - C  (azimuth ~ -crab)
    else:
        eps, sign = eps_minus, -1.0  # azimuth(C) = eps + C  (azimuth ~ +crab)

    def az_at(C):
        return eps - sign * C

    in_vfov = (np.abs(el) < VFOV_HALF_DEG) & (tz > 0)

    def active_in_fov(C):
        return float(((np.abs(az_at(C)) < HFOV_HALF_DEG) & in_vfov).mean())

    # P(accept | active gate in image) -- the per-frame conditional the reward multiplies in
    p_accept_active = float(acc_g[in_fov_px].mean()) if in_fov_px.any() else float("nan")

    # anchor self-check: separable az/el gate at the measured per-frame crab vs the pixel in_fov_g
    sep_at_meas = float(((np.abs(az) < HFOV_HALF_DEG) & in_vfov).mean())

    crabs = list(range(0, 75, 5))
    table = []
    for C in crabs:
        f = active_in_fov(float(C))
        table.append(dict(crab_deg=float(C), active_in_fov=f, accept_rate=f * p_accept_active))
    fracs = np.array([t["active_in_fov"] for t in table])
    best = int(np.argmax(fracs))

    # --- ELEVATION CO-BINDING DIAGNOSTIC (the headline finding) ---
    # At the fixable (offered) range the active gate is in HORIZONTAL FoV after de-crabbing but
    # mostly OUT of the VERTICAL FoV (the +20-deg up-tilted camera misses the gate in the steep
    # transit posture). Yaw/crab is therefore an elevation-limited lever. We quantify the azimuth-only
    # gain, the elevation block, and the 2-axis pointing CEILING (az AND el centred) at offered range.
    offered = (rng_g := geom["range_g"][fin])  # noqa: F841 -- keep rng for the band below
    rng = geom["range_g"][fin]
    band = (rng >= 18) & (rng < 28) & (tz > 0)   # the accept band-pass support (offered window)
    nb = int(band.sum())
    acc_inwin = float(acc_g[band].mean()) if nb else float("nan")

    def az_ok(C):
        return np.abs(az_at(C)) < HFOV_HALF_DEG
    el_ok = np.abs(el) < VFOV_HALF_DEG
    # In-window CEILING: if the policy held the active gate centred (az AND el) through its 18-28 m
    # approach, every in-window frame is in image, so per-frame accept -> the band-pass PEAK p_range
    # (~0.83). That is the maximum fix density the camera-pointing reward can buy in the window, vs the
    # measured in-window accept (gate mostly off-frame today). gain = ceiling / current.
    ceiling = float(p_range_peak) if p_range_peak is not None else float("nan")
    elev = dict(
        n_offered_band=nb,
        az_ok_at_crab0=float(az_ok(0.0)[band].mean()) if nb else float("nan"),
        el_ok=float(el_ok[band].mean()) if nb else float("nan"),
        both_ok_at_crab0=float((az_ok(0.0) & el_ok)[band].mean()) if nb else float("nan"),
        abs_el_p50=float(np.percentile(np.abs(el[band]), 50)) if nb else float("nan"),
        abs_el_p90=float(np.percentile(np.abs(el[band]), 90)) if nb else float("nan"),
        vfov_half_deg=VFOV_HALF_DEG,
        current_accept_in_window=acc_inwin,
        pointing_ceiling_accept_in_window=ceiling,
        pointing_gain=(ceiling / acc_inwin) if (nb and acc_inwin > 1e-9 and ceiling == ceiling) else float("nan"),
    )
    return dict(
        crab_inc7_deg=crab_inc7, sign=sign, p_accept_active_in_image=p_accept_active,
        sep_gate_at_measured_crab=sep_at_meas, pixel_in_fov_active=float(in_fov_px.mean()),
        any_gate_coverage=float(geom["any_gate_in_image"].mean()),
        any_gate_accept_rate=float(geom["accepted"].mean()),
        crab_grid=crabs, table=table,
        best_crab_deg=float(table[best]["crab_deg"]),
        best_active_in_fov=float(table[best]["active_in_fov"]),
        best_accept_rate=float(table[best]["accept_rate"]),
        inc7_active_in_fov=active_in_fov(crab_inc7),
        inc7_accept_to_active=active_in_fov(crab_inc7) * p_accept_active,
        elevation_diag=elev,
    )


def crab_to_fix_rate(crab_model: dict, crab_deg: float) -> tuple[float, float]:
    """Interpolate (active_in_fov_frac, accept_rate_to_active) at an arbitrary crab from the table."""
    tab = sorted(crab_model["table"], key=lambda t: t["crab_deg"])
    cs = np.array([t["crab_deg"] for t in tab])
    ff = np.array([t["active_in_fov"] for t in tab])
    ar = np.array([t["accept_rate"] for t in tab])
    return float(np.interp(crab_deg, cs, ff)), float(np.interp(crab_deg, cs, ar))


# ----------------------------------------------------------------------------- D. adversarial CV
def leave_one_flight_out_cv() -> dict:
    """Fit on 5 flights, predict the held-out one -- for the accept logistic and the sigma floors.
    Returns per-flight held-out accept AUC + |predicted - actual| accept-rate, and held-out sigma
    floor vs the full-fit floor (overfit if the held-out spread is large vs the point estimate)."""
    cand = _load_cand()
    fix = _load_fix_errors()
    tags = sorted(set(cand["tag"].tolist()))
    accept_cv, sigma_cv = [], []
    for held in tags:
        tr = cand["tag"] != held
        te = (cand["tag"] == held) & cand["in_image"]
        trm = tr & cand["in_image"]
        if trm.sum() < 20 or te.sum() < 3:
            continue
        Xtr, ytr = _accept_design(cand, trm)
        Xte, yte = _accept_design(cand, te)
        beta = _fit_logistic(Xtr, ytr, l2=2.0)
        pte = _sigmoid(np.hstack([np.ones((len(Xte), 1)), Xte]) @ beta)
        # band-pass held-out: fit on train flights, score the held-out flight's peak-window accept
        edges = [0, 8, 12, 16, 20, 24, 28, 32, 40, 200]
        rtr = Xtr[:, 0]
        cc, rr, nn = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (rtr >= lo) & (rtr < hi)
            if m.sum() >= 1:
                cc.append(0.5 * (lo + min(hi, 40))); rr.append(float(ytr[m].mean())); nn.append(int(m.sum()))
        bp_tr = _fit_bandpass(np.array(cc), np.array(rr), np.array(nn))
        peakwin = (Xte[:, 0] >= 18) & (Xte[:, 0] < 28)
        bp_peak_pred = float(_bandpass(Xte[peakwin, 0], *bp_tr).mean()) if peakwin.any() else float("nan")
        bp_peak_emp = float(yte[peakwin].mean()) if peakwin.any() else float("nan")
        accept_cv.append(dict(held=held, n_te=int(te.sum()),
                              auc=_auc(yte, pte),
                              rate_actual=float(yte.mean()), rate_pred=float(pte.mean()),
                              bandpass_peak_emp=bp_peak_emp, bandpass_peak_pred=bp_peak_pred))
        # sigma held-out floor (lateral, binding band)
        clean_tr = [r for r in fix if r["tag"] != held and r["offered"] and abs(r["absfix"]) < 3.0
                    and 10 <= r["range"] < 26]
        clean_te = [r for r in fix if r["tag"] == held and r["offered"] and abs(r["absfix"]) < 3.0
                    and 10 <= r["range"] < 26]
        if len(clean_tr) >= 5 and len(clean_te) >= 3:
            lat_te = np.array([r["lateral"] for r in clean_te])
            dep_te = np.array([r["along"] for r in clean_te])
            sigma_cv.append(dict(
                held=held, n_te=len(clean_te),
                # STD held-out (comparable to the std calibration target ~0.104)
                lat_floor_std_te=float(np.std(lat_te)), depth_floor_std_te=float(np.std(dep_te)),
                lat_floor_std_tr=float(np.std([r["lateral"] for r in clean_tr])),
                depth_floor_std_tr=float(np.std([r["along"] for r in clean_tr])),
                # MAD held-out (robust cross-check; reads LOWER than the std target by construction)
                lat_floor_mad_te=_robust_sigma(lat_te), depth_floor_mad_te=_robust_sigma(dep_te)))
    aucs = [c["auc"] for c in accept_cv if np.isfinite(c["auc"])]
    rate_err = [abs(c["rate_pred"] - c["rate_actual"]) for c in accept_cv]
    bp_err = [abs(c["bandpass_peak_pred"] - c["bandpass_peak_emp"]) for c in accept_cv
              if np.isfinite(c["bandpass_peak_pred"]) and np.isfinite(c["bandpass_peak_emp"])]
    lat_te = [s["lat_floor_std_te"] for s in sigma_cv if np.isfinite(s["lat_floor_std_te"])]
    dep_te = [s["depth_floor_std_te"] for s in sigma_cv if np.isfinite(s["depth_floor_std_te"])]
    lat_mad = [s["lat_floor_mad_te"] for s in sigma_cv if np.isfinite(s["lat_floor_mad_te"])]
    return dict(
        accept_cv=accept_cv, sigma_cv=sigma_cv,
        measure_note=("sigma_*_heldout_* are STD (comparable to the std calibration floor ~0.104); "
                      "lat_floor_mad_te in sigma_cv is the robust MAD measure (~0.075, lower by "
                      "construction -- NOT a held-out failure)."),
        accept_mean_auc=float(np.mean(aucs)) if aucs else float("nan"),
        accept_max_rate_err=float(np.max(rate_err)) if rate_err else float("nan"),
        bandpass_peak_heldout_mae=float(np.mean(bp_err)) if bp_err else float("nan"),
        sigma_lat_floor_heldout_std_mean=float(np.mean(lat_te)) if lat_te else float("nan"),
        sigma_lat_floor_heldout_std_spread=float(np.std(lat_te)) if lat_te else float("nan"),
        sigma_lat_floor_heldout_mad_mean=float(np.mean(lat_mad)) if lat_mad else float("nan"),
        sigma_depth_floor_heldout_std_mean=float(np.mean(dep_te)) if dep_te else float("nan"),
        sigma_depth_floor_heldout_std_spread=float(np.std(dep_te)) if dep_te else float("nan"),
    )


# ----------------------------------------------------------------------------- main
def main() -> int:
    MODELS.mkdir(parents=True, exist_ok=True)
    cand = _load_cand()
    geom = _load_geom()
    fix = _load_fix_errors()

    accept = fit_accept_model(cand)
    sigma = fit_sigma_model(fix)
    crab = fit_crab_map(geom, p_range_peak=accept["bandpass_params"][0])
    cv = leave_one_flight_out_cv()

    (MODELS / "accept.json").write_text(json.dumps(accept, indent=2, default=float))
    (MODELS / "sigma.json").write_text(json.dumps(sigma, indent=2, default=float))
    (MODELS / "crab.json").write_text(json.dumps(crab, indent=2, default=float))
    (MODELS / "cv.json").write_text(json.dumps(cv, indent=2, default=float))

    print("=== A. ACCEPT  P(accept | gate in image) = range BAND-PASS ===")
    bp = accept["bandpass_params"]
    print(f"  bandpass [pmax,rlo,wlo,rhi,whi] = {[round(x,3) for x in bp]}")
    print(f"  base-rate in-image {accept['base_rate_in_image']:.3f}  bandpass-marginal {accept['bandpass_marginal']:.3f}"
          f"  logit-AUC {accept['logit_auc']:.3f}  P(accept|out-of-image) {accept['p_accept_out_of_image']:.4f}")
    for t in accept["range_table"]:
        print(f"    range [{t['lo']:>2},{t['hi']:>3}) n={t['n']:>4}  emp {t['p_emp']:.3f}  "
              f"bandpass {t['p_bandpass']:.3f}  (logit {t['p_logit']:.3f})")
    print("=== B. SIGMA (m) vs range  [floor = std on de-tailed 10-26 m band; target lat~0.10 depth~0.8] ===")
    print(f"  per-flight n (binding band): {sigma['per_flight_n_binding_band']}")
    for ax in ("lateral", "vertical", "depth"):
        a = sigma[ax]
        print(f"  {a['label']:>17}: floor(std) {a['floor']:.3f}  CI68 [{a['floor_ci68'][0]:.3f},{a['floor_ci68'][1]:.3f}]"
              f"  floor(mad) {a['floor_mad']:.3f}  a1 {a['a1']:.4f}  bias_band {a['bias_band']:+.3f}  n {a['n_binding_band']}")
        for b in a["bands"]:
            print(f"      [{b['lo']:>2},{b['hi']:>3}) n={b['n']:>3}  sig_std {b['sigma_std']:.3f}  "
                  f"sig_mad {b['sigma_mad']:.3f}  bias {b['bias']:+.3f}")
    print("=== C. CRAB MAP  (ACTIVE-gate-in-FoV vs target crab) ===")
    print(f"  crab_inc7 {crab['crab_inc7_deg']:.1f} deg  sign {crab['sign']:+.0f}  "
          f"P(accept|active-in-image) {crab['p_accept_active_in_image']:.3f}")
    print(f"  ANCHOR: separable-gate@measured-crab {crab['sep_gate_at_measured_crab']:.3f} vs "
          f"pixel in_fov_g {crab['pixel_in_fov_active']:.3f}  (agree?)")
    print(f"  PROVENANCE any-gate coverage {crab['any_gate_coverage']:.3f} / accept {crab['any_gate_accept_rate']:.3f} "
          f"(target 0.33 / 0.07)")
    print(f"  inc7 crab {crab['crab_inc7_deg']:.0f}: active_in_fov {crab['inc7_active_in_fov']:.3f}  "
          f"accept_to_active {crab['inc7_accept_to_active']:.3f}")
    print(f"  OPTIMUM crab {crab['best_crab_deg']:.0f}: active_in_fov {crab['best_active_in_fov']:.3f}  "
          f"accept_to_active {crab['best_accept_rate']:.3f}")
    for t in crab["table"]:
        if int(t["crab_deg"]) % 10 == 0:
            print(f"    crab {t['crab_deg']:>5.0f}  active_in_fov {t['active_in_fov']:.3f}  accept {t['accept_rate']:.3f}")
    e = crab["elevation_diag"]
    print(f"  >> ELEVATION CO-BIND (window 18-28 m, n={e['n_offered_band']}): "
          f"az_ok@crab0 {e['az_ok_at_crab0']:.2f}  el_ok {e['el_ok']:.2f}  both {e['both_ok_at_crab0']:.2f}")
    print(f"     |el| p50 {e['abs_el_p50']:.1f} p90 {e['abs_el_p90']:.1f} deg vs VFoV half {e['vfov_half_deg']:.1f}")
    print(f"     in-window accept: current {e['current_accept_in_window']:.3f} -> 2-axis-pointing CEILING "
          f"{e['pointing_ceiling_accept_in_window']:.3f}  (gain x{e['pointing_gain']:.1f})")
    print("=== D. ADVERSARIAL leave-one-flight-out CV ===")
    print(f"  accept held-out mean AUC {cv['accept_mean_auc']:.3f}  max |rate_pred-actual| {cv['accept_max_rate_err']:.4f}")
    print(f"  band-pass peak (18-28 m) held-out MAE {cv['bandpass_peak_heldout_mae']:.3f}")
    print(f"  sigma lateral floor held-out STD: mean {cv['sigma_lat_floor_heldout_std_mean']:.3f}  "
          f"spread {cv['sigma_lat_floor_heldout_std_spread']:.3f}  (vs std target 0.104; MAD held-out "
          f"{cv['sigma_lat_floor_heldout_mad_mean']:.3f})")
    print(f"  sigma depth   floor held-out STD: mean {cv['sigma_depth_floor_heldout_std_mean']:.3f}  "
          f"spread {cv['sigma_depth_floor_heldout_std_spread']:.3f}  (vs std target 0.852)")
    print(f"wrote checkpoints -> {MODELS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
