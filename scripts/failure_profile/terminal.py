"""Terminal gate geometry from the VG instrument -- v2, validated on confirmed passes.

WHY v2.  v1 read the miss off a ballistic continuation whose velocity came from a QUADRATIC fit.
On the 1137 confirmed passes (true miss < 0.75 m by construction) that estimator put only 50% of
passes inside the opening horizontally, with p75 = 4.5 m -- the fitted velocity DIRECTION is
poisoned by the radial (range) channel, and the error is then multiplied by the extrapolation arm.

v2 changes three things:
  1. LINEAR (not quadratic) robust fit over a ~1.0 s window -> a far stabler direction of travel.
  2. The approach axis is taken from that long-window direction, and the miss is carried as a
     TRANSVERSE OFFSET y(t) that is extrapolated over the *known* blind interval only.
  3. The closing speed comes from a robust linear fit of RANGE, not from the position fit.

Everything is still vision+gyro only.  Nothing here reads obs[0:3].
"""
import numpy as np

import vg


def _huber_linfit(t, p, bhat, sig_t, sig_r, iters=6, huber=2.5):
    """Anisotropic Huber LINEAR fit p(t) = c0 + c1*(t - t[-1]).  Returns (c0, c1, w, res)."""
    dt = t - t[-1]
    A = np.stack([np.ones_like(dt), dt], axis=1)
    w = np.ones(len(t))
    coef = None
    for _ in range(iters):
        M = np.zeros((6, 6))
        b = np.zeros(6)
        for n_ in range(len(t)):
            B = bhat[n_]
            W = ((1.0 / sig_t[n_] ** 2) * (np.eye(3) - np.outer(B, B))
                 + (1.0 / sig_r[n_] ** 2) * np.outer(B, B)) * w[n_]
            J = np.kron(A[n_][None, :], np.eye(3)).reshape(3, 6)
            M += J.T @ W @ J
            b += J.T @ W @ p[n_]
        try:
            coef = np.linalg.solve(M + 1e-9 * np.eye(6), b)
        except np.linalg.LinAlgError:
            return None
        C = coef.reshape(2, 3)
        res = p - A @ C
        rr = np.einsum("ni,ni->n", res, bhat)
        rt = np.linalg.norm(res - rr[:, None] * bhat, axis=1)
        z = np.sqrt((rr / sig_r) ** 2 + (rt / sig_t) ** 2)
        w = np.where(z <= huber, 1.0, huber / np.maximum(z, 1e-9))
    C = coef.reshape(2, 3)
    res = p - A @ C
    rr = np.einsum("ni,ni->n", res, bhat)
    rt = np.linalg.norm(res - rr[:, None] * bhat, axis=1)
    return C[0], C[1], w, rr, rt


def terminal_geometry(d, sel, i_end, win_axis=1.0, win_near=0.55, t_extrap_cap=0.40):
    """VG terminal geometry for one gate event.

    sel   : ascending fresh-fix tick indices belonging to this approach (aim_off already removed)
    i_end : the tick at which the event happened (crash = last tick; pass = the advance tick)

    Returns None, or a dict with the transverse miss (y_h, y_v) carried to the gate plane, plus
    everything needed to judge how much of it is extrapolated.
    """
    t = d["t"]
    sel = np.asarray(sel, int)
    if len(sel) < 5:
        return None
    lf = int(sel[-1])
    selA = sel[t[sel] >= t[lf] - win_axis]
    if len(selA) < 5:
        selA = sel[-min(len(sel), 8):]
    i0, i1 = int(selA[0]), int(selA[-1])
    R = vg.gyro_attitude(d, i0, i1, lf)
    Rl = np.array([R[i - i0] for i in selA])
    rel = d["rel"][selA]
    rho = d["rho"][selA]
    q = np.einsum("nij,nj->ni", Rl, rel)
    q[:, 2] += d["zbias"] * vg.ZBIAS_COS
    p = -q                                     # drone position relative to gate, leveled frame
    bhat = np.einsum("nij,nj->ni", Rl, rel / np.maximum(rho, 1e-9)[:, None])
    sig_t = np.maximum(0.04, 0.008 * rho)
    sig_r = np.maximum(0.25, 0.12 * rho)
    nc, src = d["ncorn"][selA], d["rsrc"][selA]
    for n_ in range(len(selA)):
        if str(src[n_]) == "bbox":
            sig_r[n_] *= 4.0
            sig_t[n_] *= 2.0
        elif np.isfinite(nc[n_]) and nc[n_] <= 2:
            sig_r[n_] *= 2.5
            sig_t[n_] *= 2.0
        elif np.isfinite(nc[n_]) and nc[n_] == 3:
            sig_r[n_] *= 1.5
            sig_t[n_] *= 1.3
    fitA = _huber_linfit(t[selA], p, bhat, sig_t, sig_r)
    if fitA is None:
        return None
    pA, vA, wA, rrA, rtA = fitA
    spA = float(np.linalg.norm(vA))
    if spA < 0.5:
        return None
    a = vA / spA                                # robust direction of travel (approach axis)

    # ---- near-window LINEAR refit for the terminal state (position + transverse velocity) ----
    selN = sel[t[sel] >= t[lf] - win_near]
    if len(selN) < 4:
        selN = sel[-min(len(sel), 5):]
    kN = np.searchsorted(selA, selN)
    kN = kN[(kN >= 0) & (kN < len(selA))]
    if len(kN) < 4:
        return None
    fitN = _huber_linfit(t[selA][kN], p[kN], bhat[kN], sig_t[kN], sig_r[kN])
    if fitN is None:
        return None
    pN, vN, wN, rrN, rtN = fitN

    # ---- robust CLOSURE speed from RANGE (not from the position fit) ----
    tr = t[selA]
    m = tr >= tr[-1] - win_near
    if m.sum() >= 4:
        A2 = np.stack([np.ones(m.sum()), tr[m] - tr[-1]], axis=1)
        wq = 1.0 / np.maximum(sig_r[m], 1e-6) ** 2
        for _ in range(4):
            W = np.diag(wq)
            cf = np.linalg.lstsq(A2.T @ W @ A2, A2.T @ W @ rho[m], rcond=None)[0]
            rs = rho[m] - A2 @ cf
            s = 1.4826 * np.median(np.abs(rs - np.median(rs))) + 1e-6
            wq = np.where(np.abs(rs) <= 2.5 * s, 1.0, 2.5 * s / np.abs(rs)) / np.maximum(sig_r[m], 1e-6) ** 2
        closure = float(-cf[1])
    else:
        closure = float(-(vN @ (p[-1] / max(np.linalg.norm(p[-1]), 1e-9))))

    # ---- transverse decomposition against the robust axis ----
    s_end = float(-(pN @ a))                        # >0 = still short of the plane
    y_end = pN - (pN @ a) * a
    ydot = vN - (vN @ a) * a
    along = float(vN @ a)
    dt_blind = float(t[i_end] - t[lf])
    # time from the last fresh fix to the gate plane
    if along > 0.5:
        t_plane = s_end / along
    elif closure > 0.5:
        t_plane = s_end / closure
    else:
        t_plane = np.nan
    t_use = float(np.clip(t_plane, 0.0, t_extrap_cap)) if np.isfinite(t_plane) else 0.0
    y_plane = y_end + ydot * t_use
    return dict(
        p_end=pN, v_end=vN, axis=a, speed=spA, closure=closure,
        s_end=s_end, d_end=float(np.linalg.norm(pN)),
        y_h_now=float(np.hypot(y_end[0], y_end[1])), y_v_now=float(y_end[2]),
        y_h=float(np.hypot(y_plane[0], y_plane[1])), y_v=float(y_plane[2]),
        ydot_h=float(np.hypot(ydot[0], ydot[1])), ydot_v=float(ydot[2]),
        t_plane=float(t_plane) if np.isfinite(t_plane) else np.nan,
        t_used=t_use, capped=int(np.isfinite(t_plane) and t_plane > t_extrap_cap),
        dt_blind=dt_blind, n_axis=len(selA), n_near=len(kN),
        rms_r=float(np.sqrt(np.mean(rrN ** 2))), rms_t=float(np.sqrt(np.mean(rtN ** 2))),
        rms_r_axis=float(np.sqrt(np.mean(rrA ** 2))), rms_t_axis=float(np.sqrt(np.mean(rtA ** 2))),
        last_rho=float(rho[-1]), lf=lf,
        # extrapolated distance -- how much of the answer is measured vs projected
        extrap_m=float(np.linalg.norm(ydot) * t_use),
    )


def approach_sel(d, gate, i_end):
    """Fresh-fix indices for the approach to `gate` ending at tick i_end (aim_off excluded)."""
    ao = d["aim_off"]
    aim = (np.abs(ao[:, 0]) > 1e-6) | (np.abs(ao[:, 1]) > 1e-6)
    ok = d["fresh"] & ~aim & (d["gi"] == gate)
    idx = np.where(ok[:i_end + 1])[0]
    return idx
