"""Independent honest tilt-capped TOPP — adjudicate the 60-deg dispute.

Two readings exist in the panel:
  (A) C1 / probe_robustness: tilt cap binds ONLY cornering (a_lat = g*tan(tilt)); tangential
      (brake/forward) is free to use the full thrust ball.  -> 60deg ~ 5.3-5.5 s.
  (B) envelope-realizability verifier: tilt is the angle of the TOTAL specific force f = a_des - grav;
      braking/forward tilt the body too.  Cap |f_horiz| <= g*tan(tilt) AND |f| <= A jointly.
      -> 60deg ~ 9.8 s.

This script reproduces BOTH on the same centre line + same live constants, so the ~4.3 s gap is
attributable, not asserted. PURE OFFLINE, reads constants live from rl_plant.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline

_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_ROOT / "src"))
import racer.rl_plant as rp  # noqa: E402

G = 9.80665
A = float(rp.COLL_MAP_ACCEL_MEASURED[-1])   # 78.283 full-stick body-up accel
C2 = 0.052                                   # pooled quad drag

GATES = np.array([
    [-23.30, -0.40, -0.03], [-46.89, -2.50, 5.07], [-74.59, 1.20, 13.67],
    [-111.49, -5.10, 24.57], [-135.49, -0.80, 25.36], [-159.19, -4.40, 25.97],
])
START = np.array([0.0, 0.0, -0.02])


def build_line(n=4000):
    wp = np.vstack([START, GATES])
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    u = np.concatenate([[0.0], np.cumsum(seg)])
    cs = CubicSpline(u, wp, bc_type="natural")
    s = np.linspace(0, u[-1], n)
    r = cs(s); d1 = cs(s, 1); d2 = cs(s, 2)
    ds = np.maximum(np.linalg.norm(d1, axis=1), 1e-9)
    tang = d1 / ds[:, None]
    kappa = np.linalg.norm(np.cross(d1, d2), axis=1) / ds ** 3
    # principal normal (curvature direction), for centripetal force orientation
    dT = np.gradient(tang, axis=0)
    nrm = np.linalg.norm(dT, axis=1, keepdims=True)
    normal = np.where(nrm > 1e-9, dT / np.maximum(nrm, 1e-12), 0.0)
    arc = np.concatenate([[0.0], np.cumsum((ds[:-1] + ds[1:]) / 2 * np.diff(s))])
    return r, tang, normal, kappa, arc


def topp_reading_A(tang, normal, kappa, arc, tilt_deg):
    """Reading A: cone caps cornering only; tangential uses full thrust ball."""
    N = len(arc); seglen = np.diff(arc); tz = tang[:, 2]
    tilt = np.radians(tilt_deg)
    a_h = A * np.sin(tilt)
    lat_cap = min(a_h, G * np.tan(tilt)) if tilt_deg < 89.9 else a_h
    v_wall = np.sqrt(a_h / C2)
    v_corner = np.where(kappa > 1e-6, np.sqrt(lat_cap / np.maximum(kappa, 1e-12)), np.inf)
    v = np.minimum(v_corner, v_wall); v[0] = 0.0
    for i in range(N - 1):
        lat = kappa[i] * v[i] ** 2
        ap = np.sqrt(max(0.0, a_h * a_h - lat * lat)) + G * tz[i]
        ad = C2 * v[i] ** 2
        v[i + 1] = min(v[i + 1], np.sqrt(v[i] ** 2 + 2 * max(0.05, ap - ad) * seglen[i]))
    for i in range(N - 2, -1, -1):
        lat = kappa[i + 1] * v[i + 1] ** 2
        ap = np.sqrt(max(0.0, a_h * a_h - lat * lat)) + G * tz[i + 1]
        ad = C2 * v[i + 1] ** 2
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * max(0.05, ap + ad) * seglen[i]))
    t = np.sum(seglen / np.maximum((v[:-1] + v[1:]) / 2, 1e-6))
    return t, float(v.max())


def _max_atan(i, vi, tang, normal, kappa, tilt_deg, brake):
    """Max along-track accel (signed: brake=True means decel magnitude) achievable while keeping
    BOTH |f| <= A and the TOTAL specific-force tilt <= cap.
    f = a_des - grav (NED, grav = +g e_z because z down). a_des = a_tan*tang + a_cent*normal.
    tilt = angle of f from world-vertical (e_z)."""
    cap = np.radians(tilt_deg)
    tanv = tang[i]; nrm = normal[i]
    a_cent = kappa[i] * vi * vi
    grav = np.array([0.0, 0.0, G])  # specific force to hold = -grav contribution; f = a_des - grav
    best = 0.0
    # scan along-track accel; sign convention: forward = +tang, brake = -tang
    sgn = -1.0 if brake else +1.0
    for a_t in np.linspace(0.0, A + G, 220):
        a_des = sgn * a_t * tanv + a_cent * nrm
        f = a_des - grav            # thrust must supply f (NED). |f| is collective accel.
        fmag = np.linalg.norm(f)
        if fmag > A + 1e-6:
            break
        f_horiz = np.hypot(f[0], f[1]); f_vert = abs(f[2])
        tilt = np.arctan2(f_horiz, max(f_vert, 1e-9))
        if cap < np.radians(89.9) and tilt > cap + 1e-6:
            break
        best = a_t
    return best


def topp_reading_B(tang, normal, kappa, arc, tilt_deg):
    """Reading B (honest): cone caps the TOTAL specific-force tilt for cornering AND brake AND fwd."""
    N = len(arc); seglen = np.diff(arc)
    # corner-limited ceiling: max v s.t. holding gravity + centripetal stays within cone & ball
    cap = np.radians(tilt_deg)
    grav = np.array([0.0, 0.0, G])
    v_corner = np.full(N, np.inf)
    for i in range(N):
        if kappa[i] < 1e-6:
            continue
        # find max v where f=a_cent*normal - grav has tilt<=cap and |f|<=A
        lo, hi = 0.0, 80.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            a_cent = kappa[i] * mid * mid
            f = a_cent * normal[i] - grav
            fh = np.hypot(f[0], f[1]); fv = abs(f[2]); tilt = np.arctan2(fh, max(fv, 1e-9))
            ok = (np.linalg.norm(f) <= A + 1e-9) and (tilt <= cap + 1e-9 or tilt_deg >= 89.9)
            if ok:
                lo = mid
            else:
                hi = mid
        v_corner[i] = lo
    v = v_corner.copy(); v[0] = 0.0
    for i in range(N - 1):
        ap = _max_atan(i, v[i], tang, normal, kappa, tilt_deg, brake=False)
        ad = C2 * v[i] ** 2
        v[i + 1] = min(v[i + 1], np.sqrt(v[i] ** 2 + 2 * max(0.02, ap - ad) * seglen[i]))
    for i in range(N - 2, -1, -1):
        ab = _max_atan(i + 1, v[i + 1], tang, normal, kappa, tilt_deg, brake=True)
        ad = C2 * v[i + 1] ** 2
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * max(0.02, ab + ad) * seglen[i]))
    t = np.sum(seglen / np.maximum((v[:-1] + v[1:]) / 2, 1e-6))
    return t, float(v.max())


if __name__ == "__main__":
    r, tang, normal, kappa, arc = build_line()
    print(f"line len {arc[-1]:.2f} m, kappa_max {kappa.max():.4f} (r_min {1/kappa.max():.1f} m)")
    print(f"A_UP_MAX {A:.2f} m/s2 ({A/G:.2f} g), drag wall ~{np.sqrt(np.sqrt(A**2-G**2)/C2):.1f} m/s")
    print(f"{'cap':>4} {'A:cornering-only':>18} {'B:honest-total-tilt':>20}")
    for cap in (60, 65, 75, 80, 90):
        tA, vA = topp_reading_A(tang, normal, kappa, arc, cap)
        tB, vB = topp_reading_B(tang, normal, kappa, arc, cap)
        print(f"{cap:>4}  {tA:>7.2f}s v{vA:>5.1f}     {tB:>7.2f}s v{vB:>5.1f}")
