"""VG (Vision-Gyro) kinematic instrument -- the definitive drone-motion vs estimate-motion separator.

PHYSICS
-------
The gate is a FIXED world object. Let r(t) = drone->gate in the TRUE body FLU frame (the logged
`rel_flu`, unflipped). For a fixed world point,

    r_dot = -omega x r - v_body                                                (1)

so the body-frame motion of the gate mixes DRONE ROTATION (omega, measured by the gyro -- trusted)
with DRONE TRANSLATION (v_body -- the thing obs[0:3] lies about).  De-rotating removes the first:

    q_i := R_wb(t_i) @ r_i        (world/quasi-inertial frame, gyro-propagated attitude)
    q_i  = G - p_i                (G = fixed gate position, p_i = drone position)
 => -q_i = p_i - G = THE DRONE'S OWN TRAJECTORY RELATIVE TO THE GATE, measured by vision+gyro ONLY.

No KF, no obs[0:3].  Writing r = rho * bhat, the transverse part of q depends on the BEARING
(pixel-clean) and uses rho only as a SCALE; the radial part is the range itself (the flappy
channel).  That asymmetry is the whole instrument:

    transverse error  ~  rho * bearing_noise      (few cm)
    radial error      ~  range flap              (metres)

SEPARATION
----------
A real drone trajectory is acceleration-bounded (|a| <= ~40 m/s^2 = 3.765 g thrust rail + g), so over
a <=0.6 s window a QUADRATIC in t fits real motion to ~cm.  Anything the quadratic cannot explain is
ESTIMATE motion.  We fit p(t) with an anisotropic (transverse-tight / radial-loose) Huber-weighted
least squares and report the residual split.  The fitted trajectory = DRONE MOTION; the residuals =
ESTIMATE MOTION.

FRAME CONVENTIONS (load-bearing)
--------------------------------
  logged rel_flu    : TRUE body FLU [fwd, left, up], UNflipped.  (-lat = gate to the drone's RIGHT)
  obs[0:3] v        : VIRTUAL-FLIPPED diag(-1,-1,1) -> true v_flu = [-o0, -o1, +o2]   (UNTRUSTED)
  obs[3:5] roll,pit : VIRTUAL-FLIPPED -> true roll = -o3, true pitch = -o4
  obs[5:8] rates    : VIRTUAL-FLIPPED -> true w_flu = [-o5, -o6, +o7]                 (TRUSTED)
  obs[9:11]         : gravity-leveled coarse sector (NOT used here)
  rel_flu is in the TILTED body frame -- it MUST be gravity-leveled before any vertical geometry
  (at the -17.8 deg resting pitch a level 10 m gate reads rel_flu[2] ~ -3 m).

z_bias: the seeker LOWERS every emitted gate by `ego_gate_z_bias` along camera +Y (optical down,
~20 deg off world vertical, gate_seeker.py:898).  TRUE gate height = emitted + z_bias*cos(20deg).
"""
import json
import os

import numpy as np

GATE_HALF = 0.75          # inner opening 1.5 m square (gate_pose.GATE_INNER_SIZE_M)
ZBIAS_COS = float(np.cos(np.radians(20.0)))   # camera +Y is ~20 deg off world-vertical


# ----------------------------------------------------------------------------------- loading ----
def load_session(rd):
    recs = [json.loads(l) for l in open(os.path.join(rd, "ego_obs.jsonl"), encoding="utf-8")]
    seek = {}
    p = os.path.join(rd, "seeker.jsonl")
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            try:
                j = json.loads(l)
                seek[j["k"]] = j
            except Exception:
                pass
    timing = {}
    p = os.path.join(rd, "ego_timing.jsonl")
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            try:
                j = json.loads(l)
                timing[j["k"]] = j
            except Exception:
                pass
    meta = json.load(open(os.path.join(rd, "meta.json"), encoding="utf-8"))
    return recs, seek, timing, meta


def build(recs, seek, meta):
    """Per-tick arrays in TRUE (unflipped) conventions."""
    n = len(recs)
    d = {}
    d["k"] = np.array([r["k"] for r in recs], dtype=int)
    t = np.array([r["sim_time_ns"] for r in recs], dtype=np.float64) / 1e9
    d["t"] = t - t[0]
    d["gi"] = np.array([r.get("gate_index", 0) for r in recs], dtype=int)
    obs = np.array([(r["obs"] if r.get("obs") and len(r["obs"]) == 21 else [np.nan] * 21)
                    for r in recs], dtype=float)
    d["obs"] = obs
    d["v_obs_flu"] = np.column_stack([-obs[:, 0], -obs[:, 1], obs[:, 2]])   # UNTRUSTED
    d["w_flu"] = np.column_stack([-obs[:, 5], -obs[:, 6], obs[:, 7]])       # TRUSTED gyro
    d["roll"] = -obs[:, 3]
    d["pitch"] = -obs[:, 4]
    # rel_flu: TRUE body FLU drone->gate.  Older log schemas (v1/vpef/vtrackA/v15 and part of v16)
    # never emitted the `rel_flu` key -- but obs[11:14] IS the same vector, VIRTUAL-FLIPPED by
    # diag(-1,-1,1) (ego_obs.py:529) and MASKED to zeros once confidence hits 0 (obs[14]).  The flip
    # is an involution, so rel_flu = [-obs11, -obs12, +obs13] recovers it exactly (verified against
    # the flights that log BOTH: agreement to the 1e-5 log-rounding floor).
    rel = np.full((n, 3), np.nan)
    for i, r in enumerate(recs):
        v = r.get("rel_flu")
        if v is not None:
            rel[i] = v
        else:
            o = obs[i]
            if np.isfinite(o[11:15]).all() and o[14] > 0.0:
                rel[i] = (-o[11], -o[12], o[13])
    d["rel"] = rel
    d["rel_src"] = "log" if any(r.get("rel_flu") is not None for r in recs) else "obs"
    d["seen"] = np.array([bool(r.get("pose_seen", False)) for r in recs])
    d["age"] = np.array([float(r.get("age_s") or 0.0) for r in recs])
    d["conf"] = np.array([float(r.get("conf") or 0.0) for r in recs])
    d["rate_frd"] = np.array([(r.get("rate_frd") or [np.nan] * 3) for r in recs], dtype=float)
    d["thrust"] = np.array([float(r.get("normed_thrust") or np.nan) for r in recs])
    d["collective"] = np.array([float(r.get("collective") or np.nan) for r in recs])
    d["kf"] = np.array([(r.get("kf_pos_ned") or [np.nan] * 3) for r in recs], dtype=float)
    d["p_zup_kf"] = np.column_stack([d["kf"][:, 0], -d["kf"][:, 1], -d["kf"][:, 2]])
    ao = np.zeros((n, 3))
    for i, r in enumerate(recs):
        a = r.get("aim_off", None)
        if a is not None:
            try:
                ao[i] = (float(a[0]), float(a[1]), 1.0)
            except Exception:
                ao[i] = (0.0, 0.0, 1.0)
    d["aim_off"] = ao
    # ---- seeker per-tick ----
    reason = []
    ncorn = np.full(n, np.nan)
    rsrc = []
    fresh_frame = np.zeros(n, dtype=bool)
    ncand = np.full(n, np.nan)
    for i, kk in enumerate(d["k"]):
        j = seek.get(int(kk)) or {}
        s0 = j.get("slot0") or {}
        reason.append(s0.get("reason") or "")
        v = s0.get("n_corners")
        ncorn[i] = np.nan if v is None else float(v)
        rsrc.append(s0.get("range_src") or "")
        fresh_frame[i] = bool(j.get("fresh_frame", False))
        v = s0.get("n_cand")
        ncand[i] = np.nan if v is None else float(v)
    d["reason"] = np.array(reason, dtype=object)
    d["ncorn"] = ncorn
    d["rsrc"] = np.array(rsrc, dtype=object)
    d["fresh_frame"] = fresh_frame
    d["ncand"] = ncand
    # A FRESH FIX tick: the estimator snapped to a brand-new measurement this tick
    # (fix_gain default 1.0 => rel_flu IS the raw measurement, no KF contamination).
    d["fresh"] = d["seen"] & (d["age"] <= 1e-9) & np.isfinite(d["rel"]).all(axis=1)
    d["rho"] = np.linalg.norm(d["rel"], axis=1)
    d["zbias"] = float(meta.get("ego_gate_z_bias") or 0.0)
    return d


# --------------------------------------------------------------------------------- attitude ----
def _skew(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def _expm_rotvec(v):
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3) + _skew(v)
    K = _skew(v / th)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def gyro_attitude(d, i0, i1, anchor):
    """Pure-gyro body->world attitude over ticks [i0, i1], anchored to the AHRS roll/pitch at
    tick `anchor` (yaw origin arbitrary -> the frame is gravity-leveled, heading-free).

    R_wb evolves as R_dot = R [w]x  =>  R_{i+1} = R_i expm([w] dt).
    Returns R[i] for i in [i0, i1] (indexed from i0)."""
    m = i1 - i0 + 1
    anchor = int(np.clip(anchor, i0, i1))
    R = np.zeros((m, 3, 3))
    cr, sr = np.cos(d["roll"][anchor]), np.sin(d["roll"][anchor])
    cp, sp = np.cos(d["pitch"][anchor]), np.sin(d["pitch"][anchor])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    R_anchor = Ry @ Rx                        # body(anchor) -> gravity-leveled heading frame
    ja = anchor - i0
    R[ja] = R_anchor
    for j in range(ja + 1, m):
        i = i0 + j
        dt = d["t"][i] - d["t"][i - 1]
        w = 0.5 * (d["w_flu"][i] + d["w_flu"][i - 1])
        if not np.isfinite(w).all() or not np.isfinite(dt) or dt <= 0:
            R[j] = R[j - 1]
            continue
        R[j] = R[j - 1] @ _expm_rotvec(w * dt)
    for j in range(ja - 1, -1, -1):
        i = i0 + j
        dt = d["t"][i + 1] - d["t"][i]
        w = 0.5 * (d["w_flu"][i] + d["w_flu"][i + 1])
        if not np.isfinite(w).all() or not np.isfinite(dt) or dt <= 0:
            R[j] = R[j + 1]
            continue
        # R[j+1] = R[j] @ expm(w dt)  =>  R[j] = R[j+1] @ expm(-w dt)
        R[j] = R[j + 1] @ _expm_rotvec(-w * dt)
    return R


def ahrs_attitude(d, i0, i1, anchor):
    """AHRS roll/pitch + GYRO-INTEGRATED yaw -> body->(gravity-leveled, arbitrary-heading) frame.
    Independent cross-check of gyro_attitude: the two disagree exactly where the accel-leveled
    AHRS lies (the measured bank-reversal lean error)."""
    m = i1 - i0 + 1
    anchor = int(np.clip(anchor, i0, i1))
    yaw = np.zeros(m)
    for j in range(1, m):
        i = i0 + j
        dt = d["t"][i] - d["t"][i - 1]
        out = 0.0
        for ii in (i - 1, i):
            w, r_, p_ = d["w_flu"][ii], d["roll"][ii], d["pitch"][ii]
            cp = np.cos(p_)
            if abs(cp) < 1e-6 or not np.isfinite(w).all():
                out += 0.0
            else:
                out += 0.5 * (w[1] * np.sin(r_) + w[2] * np.cos(r_)) / cp
        yaw[j] = yaw[j - 1] + out * (dt if np.isfinite(dt) and dt > 0 else 0.0)
    yaw = yaw - yaw[anchor - i0]
    R = np.zeros((m, 3, 3))
    for j in range(m):
        i = i0 + j
        cy, sy = np.cos(yaw[j]), np.sin(yaw[j])
        cr, sr = np.cos(d["roll"][i]), np.sin(d["roll"][i])
        cp, sp = np.cos(d["pitch"][i]), np.sin(d["pitch"][i])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        R[j] = Rz @ Ry @ Rx
    return R


# ------------------------------------------------------------------------------------- fit -----
def vg_fit(d, idx, anchor=None, sig_t_floor=0.04, sig_t_rel=0.008,
           sig_r_floor=0.25, sig_r_rel=0.12, huber=2.5, iters=6, use_ahrs=False, att=None):
    """Anisotropic Huber-weighted quadratic fit of the drone trajectory RELATIVE TO THE GATE,
    in a gravity-leveled quasi-inertial frame, from vision bearings + gyro ONLY.

    idx : tick indices of FRESH vision fixes to fit (ascending, >= 4).
    Returns dict with p0/v0/a0 at the reference time, residual split, and diagnostics.
    """
    idx = np.asarray(idx, dtype=int)
    if len(idx) < 4:
        return None
    if att is not None:
        R, i0 = att
        i1 = i0 + len(R) - 1
        if int(idx[0]) < i0 or int(idx[-1]) > i1:
            return None
    else:
        i0, i1 = int(idx[0]), int(idx[-1])
        if anchor is None:
            anchor = int(idx[len(idx) // 2])
        anchor = int(np.clip(anchor, i0, i1))
        R = (ahrs_attitude if use_ahrs else gyro_attitude)(d, i0, i1, anchor)
    t = d["t"][idx]
    tref = float(t[-1])                       # reference = LAST fresh fix (terminal geometry)
    rel = d["rel"][idx]
    rho = d["rho"][idx]
    # z_bias: the emitted gate sits LOW by zbias*cos(20 deg) -> restore TRUE gate
    reln = rel.copy()
    Rl = np.array([R[i - i0] for i in idx])
    q = np.einsum("nij,nj->ni", Rl, reln)     # gate - drone, leveled inertial frame
    q[:, 2] += d["zbias"] * ZBIAS_COS         # TRUE gate height (leveled frame Z is world up)
    p = -q                                    # drone - gate  == THE DRONE TRAJECTORY
    bhat = np.einsum("nij,nj->ni", Rl, reln / np.maximum(rho, 1e-9)[:, None])
    dt = t - tref
    # design: p(t) = c0 + c1 dt + c2 dt^2
    A = np.stack([np.ones_like(dt), dt, dt * dt], axis=1)       # (n,3)
    sig_t = np.maximum(sig_t_floor, sig_t_rel * rho)
    sig_r = np.maximum(sig_r_floor, sig_r_rel * rho)
    # PERCEPTION-QUALITY WEIGHTING. The deploy code itself documents the corner-free bbox range as
    # "a floor, not a measurement" (centre_emit.py:135-145), and close-in fragment duplicates
    # disagree 0.2-0.4 m laterally (gate_seeker.py:1067-1072). Price both.
    nc = d["ncorn"][idx]
    src = d["rsrc"][idx]
    for n_ in range(len(idx)):
        c = nc[n_]
        if str(src[n_]) == "bbox":
            sig_r[n_] *= 4.0
            sig_t[n_] *= 2.0
        elif np.isfinite(c) and c <= 2:
            sig_r[n_] *= 2.5
            sig_t[n_] *= 2.0
        elif np.isfinite(c) and c == 3:
            sig_r[n_] *= 1.5
            sig_t[n_] *= 1.3
    w = np.ones(len(idx))
    coef = None
    for _ in range(iters):
        # build normal equations in the LOCAL (radial, transverse) metric
        M = np.zeros((9, 9))
        b = np.zeros(9)
        for n_, i in enumerate(idx):
            B = bhat[n_]
            P = np.eye(3) - np.outer(B, B)
            W = (1.0 / sig_t[n_] ** 2) * P + (1.0 / sig_r[n_] ** 2) * np.outer(B, B)
            W = W * w[n_]
            a = A[n_]
            # J = kron(a, I3): p_pred = sum_m a_m * c_m
            J = np.kron(a[None, :], np.eye(3)).reshape(3, 9)
            M += J.T @ W @ J
            b += J.T @ W @ p[n_]
        try:
            coef = np.linalg.solve(M + 1e-9 * np.eye(9), b)
        except np.linalg.LinAlgError:
            return None
        C = coef.reshape(3, 3)                      # rows: c0, c1, c2 (each 3-vector)
        pred = A @ C
        res = p - pred
        rr = np.einsum("ni,ni->n", res, bhat)
        rt = np.linalg.norm(res - rr[:, None] * bhat, axis=1)
        z = np.sqrt((rr / sig_r) ** 2 + (rt / sig_t) ** 2)
        w = np.where(z <= huber, 1.0, huber / np.maximum(z, 1e-9))
    C = coef.reshape(3, 3)
    pred = A @ C
    res = p - pred
    rr = np.einsum("ni,ni->n", res, bhat)
    rt = np.linalg.norm(res - rr[:, None] * bhat, axis=1)
    p_end, v_end, a_end = C[0], C[1], 2.0 * C[2]
    return dict(idx=idx, t=t, tref=tref, p_raw=p, p_fit=pred, bhat=bhat,
                p_end=p_end, v_end=v_end, a_end=a_end, w=w,
                res_radial=rr, res_trans=rt,
                rms_radial=float(np.sqrt(np.mean(rr ** 2))),
                rms_trans=float(np.sqrt(np.mean(rt ** 2))),
                rho=rho, R=R, i0=i0, anchor=anchor)


def vg_smooth(d, idx, half_win_s=0.30, anchor=None, **kw):
    """Local robust (LOESS-style) VG trajectory over a whole approach.

    For every fresh tick, refit the anisotropic Huber quadratic over its +-half_win_s neighbours and
    take the fitted value AT that tick.  The result follows real (acceleration-bounded) drone motion
    and rejects range flap.  Per-tick residual = ESTIMATE motion.

    Returns arrays over `idx`: p_raw, p_smooth, res_radial, res_trans, plus the shared attitude.
    """
    idx = np.asarray(idx, dtype=int)
    if len(idx) < 6:
        return None
    i0, i1 = int(idx[0]), int(idx[-1])
    if anchor is None:
        anchor = int(idx[len(idx) // 2])
    R = gyro_attitude(d, i0, i1, anchor)
    t = d["t"][idx]
    rel = d["rel"][idx]
    rho = d["rho"][idx]
    Rl = np.array([R[i - i0] for i in idx])
    q = np.einsum("nij,nj->ni", Rl, rel)
    q[:, 2] += d["zbias"] * ZBIAS_COS
    p_raw = -q
    p_sm = np.full_like(p_raw, np.nan)
    v_sm = np.full_like(p_raw, np.nan)
    for n_ in range(len(idx)):
        m = np.abs(t - t[n_]) <= half_win_s
        if m.sum() < 5:
            order = np.argsort(np.abs(t - t[n_]))[:5]
            m = np.zeros(len(t), dtype=bool)
            m[order] = True
        f = vg_fit(d, idx[m], att=(R, i0), **kw)
        if f is None:
            continue
        dt = t[n_] - f["tref"]
        C = np.stack([f["p_end"], f["v_end"], 0.5 * f["a_end"]])
        p_sm[n_] = C[0] + C[1] * dt + C[2] * dt * dt
        v_sm[n_] = C[1] + 2.0 * C[2] * dt
    res = p_raw - p_sm
    bhat = np.einsum("nij,nj->ni", Rl, rel / np.maximum(rho, 1e-9)[:, None])
    rr = np.einsum("ni,ni->n", res, bhat)
    rt = np.linalg.norm(res - rr[:, None] * bhat, axis=1)
    return dict(idx=idx, t=t, rho=rho, p_raw=p_raw, p_sm=p_sm, v_sm=v_sm,
                res_radial=rr, res_trans=rt, R=R, i0=i0)


def shape_stats(p, v_axis, r_hi=12.0, r_lo=1.0):
    """Centre-then-diverge shape on a trajectory `p` (drone rel gate, leveled frame).

    Transverse offset is measured against the TERMINAL approach axis `v_axis` (unit), so the
    statistic is the drone's real lateral displacement from the line it was actually flying.
    Window: |p| in [r_lo, r_hi], matching the legacy detector's range gate.
    Returns lmin, reopen (end-minus-min), reopen_max, r_min, l_end, n.
    """
    a = v_axis / max(float(np.linalg.norm(v_axis)), 1e-9)
    rng = np.linalg.norm(p, axis=1)
    s = p @ a
    y = p - s[:, None] * a
    lat = np.linalg.norm(y, axis=1)
    ok = np.isfinite(rng) & np.isfinite(lat) & (rng <= r_hi) & (rng >= r_lo)
    jj = np.where(ok)[0]
    out = dict(n=int(len(jj)), lmin=np.nan, r_min=np.nan, l_end=np.nan,
               reopen=np.nan, reopen_max=np.nan)
    if len(jj) < 5:
        return out
    A = lat[jj]
    im = int(np.argmin(A))
    out.update(lmin=float(A[im]), r_min=float(rng[jj[im]]), l_end=float(A[-1]),
               reopen=float(A[-1] - A[im]), reopen_max=float(np.max(A[im:]) - A[im]))
    return out


def closest_approach(p_end, v_end, a_end=None, tmax=0.60):
    """Ballistic continuation of the FITTED trajectory from the last fresh fix.
    Returns (t*, miss_vec, |miss|) where t* minimises |p(t)| forward in time.
    Uses the linear (constant-velocity) term only by default -- the quadratic term is
    fit-window curvature and extrapolates badly."""
    sp = float(np.linalg.norm(v_end))
    if sp < 0.5:
        return np.nan, np.full(3, np.nan), np.nan
    tstar = float(-(p_end @ v_end) / (sp * sp))
    tstar = float(np.clip(tstar, 0.0, tmax))
    miss = p_end + v_end * tstar
    return tstar, miss, float(np.linalg.norm(miss))
