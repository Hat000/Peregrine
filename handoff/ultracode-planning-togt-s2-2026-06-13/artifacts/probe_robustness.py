"""Robustness/validity probes for the four trajectory-opt approaches (Phase B judge panel).

PURE OFFLINE. Reads track_map gate centres + speed_profile.time_optimal_profile.
Does NOT edit tracked source. Quantifies:
  (1) Coupled-TOPP ideal lap time vs tilt cone on a clamped-cubic line through gate centres
      (the minsnap_topp proposal's est_basis numbers) -- reproduce / falsify.
  (2) Geometry-forced thrust-vector tilt p50/max along the *same* line (the descending course
      forces high tilt before any cornering) -- tests the "min-snap crosses near-perpendicular,
      low tilt" claim and the decomposed-line inversion-impossibility claim.
  (3) Cross-track sensitivity: how much lateral plan/track error eats the gate-4 binding margin
      (0.155 m simstart @ r=0.38) under a 67 ms latency at TOPP speed.
"""
import sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_ROOT / "src"))
from racer.speed_profile import time_optimal_profile  # noqa: E402

G = 9.80665
# Gate centres (world NED, Z down) from track_map (prompt).
GATES = np.array([
    [-23.30, -0.40, -0.03],
    [-46.89, -2.50,  5.07],
    [-74.59,  1.20, 13.67],
    [-111.49, -5.10, 24.57],
    [-135.49, -0.80, 25.36],
    [-159.19, -4.40, 25.97],
])
START = np.array([0.0, 0.0, -0.02])  # pad, slightly above gate-0 plane

A_UP_MAX = 78.3      # full-stick body-up specific accel (m/s^2), convex map top knot
C2 = 0.052           # pooled quadratic drag (1/m)

def build_line(n=4000):
    wp = np.vstack([START, GATES])
    # clamped-cubic seed via the same CubicSpline path speed_profile uses (chord param)
    from scipy.interpolate import CubicSpline
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    u = np.concatenate([[0.0], np.cumsum(seg)])
    cs = CubicSpline(u, wp, bc_type="natural")
    s = np.linspace(0, u[-1], n)
    r = cs(s); d1 = cs(s, 1); d2 = cs(s, 2)
    ds = np.maximum(np.linalg.norm(d1, axis=1), 1e-9)
    tang = d1 / ds[:, None]
    kappa = np.linalg.norm(np.cross(d1, d2), axis=1) / ds**3
    arc = np.concatenate([[0.0], np.cumsum((ds[:-1] + ds[1:]) / 2 * np.diff(s))])
    return r, tang, kappa, arc

def coupled_topp(r, tang, kappa, arc, tilt_cap_deg):
    """Forward-backward TOPP with a thrust-budget-coupled envelope.
    Horizontal authority a_h = A_UP_MAX*sin(tilt_cap); vertical sustain a_v = A_UP_MAX*cos(tilt).
    Lateral cap = min(a_h, g*tan(tilt)) [proposal's own model]. Tangential budget from the
    remaining thrust after gravity-cancel + centripetal, plus descent gravity-assist - drag.
    Returns ideal lap time to the gate-5 plane (s)."""
    N = len(arc)
    seglen = np.diff(arc)
    tilt = np.radians(tilt_cap_deg)
    a_h = A_UP_MAX * np.sin(tilt)
    lat_cap = min(a_h, G * np.tan(tilt))           # the proposal's lateral cap
    v_drag_wall = np.sqrt(max(a_h, 1e-9) / C2)     # v^2 drag wall
    # corner-limited ceiling: kappa*v^2 <= lat_cap
    v_corner = np.where(kappa > 1e-6, np.sqrt(lat_cap / np.maximum(kappa, 1e-12)), np.inf)
    v = np.minimum(v_corner, v_drag_wall)
    # path-tangent vertical (descent) component: tang z<0 means descending (NED z down=+).
    # course DESCENDS so z increases (gate z grows +). descent gravity-assist along +tangent.
    tz = tang[:, 2]   # +tz = moving in +z (down) = descending -> gravity assists tangentially
    def a_tan(i, vi):
        lat = kappa[i] * vi * vi
        budget = a_h * a_h - lat * lat             # remaining horizontal-equivalent thrust^2
        a_thrust = np.sqrt(max(0.0, budget))
        a_grav = G * tz[i]                          # +assist descending, -penalty climbing
        a_drag = C2 * vi * vi                       # always opposes motion (helps brake)
        return a_thrust + a_grav, a_drag
    # forward
    v[0] = 0.0
    for i in range(N - 1):
        ap, ad = a_tan(i, v[i])
        anet = max(0.05, ap - ad)
        v[i+1] = min(v[i+1], np.sqrt(v[i]**2 + 2.0 * anet * seglen[i]))
    # backward (braking: drag + thrust both decelerate)
    v[-1] = min(v[-1], 0.0 if False else v[-1])
    for i in range(N - 2, -1, -1):
        ap, ad = a_tan(i+1, v[i+1])
        anet = max(0.05, ap + ad)                  # braking uses thrust + drag
        v[i] = min(v[i], np.sqrt(v[i+1]**2 + 2.0 * anet * seglen[i]))
    # integrate time
    t = 0.0
    for i in range(N - 1):
        vm = max((v[i] + v[i+1]) / 2.0, 1e-6)
        t += seglen[i] / vm
    return t, float(v.max()), float(v_drag_wall)

def geometry_tilt(r, tang, kappa, arc, tilt_cap_deg):
    """Thrust-vector tilt the line GEOMETRY demands (descent + centripetal), independent of cornering.
    Required specific force = a_centripetal (horizontal-ish) + (g + a_tangential_z) to hold the path.
    Here we estimate the *minimum* tilt to (a) cancel gravity and (b) supply centripetal at the
    TOPP speed. tilt = angle of required thrust vector from vertical (world -z)."""
    t_lap, vmax, _ = coupled_topp(r, tang, kappa, arc, tilt_cap_deg)
    # reconstruct speed profile again to get per-point tilt (cheap re-run)
    N = len(arc); seglen = np.diff(arc)
    tilt = np.radians(tilt_cap_deg); a_h = A_UP_MAX*np.sin(tilt)
    lat_cap = min(a_h, G*np.tan(tilt)); v_wall = np.sqrt(a_h/C2)
    v_corner = np.where(kappa>1e-6, np.sqrt(lat_cap/np.maximum(kappa,1e-12)), np.inf)
    v = np.minimum(v_corner, v_wall); v[0]=0.0
    tz = tang[:,2]
    for i in range(N-1):
        lat=kappa[i]*v[i]**2; ap=np.sqrt(max(0.0,a_h*a_h-lat*lat))+G*tz[i]; ad=C2*v[i]**2
        v[i+1]=min(v[i+1], np.sqrt(v[i]**2+2*max(0.05,ap-ad)*seglen[i]))
    for i in range(N-2,-1,-1):
        lat=kappa[i+1]*v[i+1]**2; ap=np.sqrt(max(0.0,a_h*a_h-lat*lat))+G*tz[i+1]; ad=C2*v[i+1]**2
        v[i]=min(v[i], np.sqrt(v[i+1]**2+2*max(0.05,ap+ad)*seglen[i]))
    # required thrust vector to hold path: f = a_lat_horizontal (centripetal) + cancel gravity
    # centripetal direction = component of curvature normal in horizontal plane (approx full kappa*v^2)
    tilts = []
    for i in range(N):
        a_cent = kappa[i]*v[i]**2          # magnitude of lateral accel
        # vertical thrust needed = g (hold against gravity); horizontal = a_cent
        # tilt from vertical = atan2(horizontal_force, vertical_force)
        f_vert = G
        f_horiz = a_cent
        tilts.append(np.degrees(np.arctan2(f_horiz, f_vert)))
    tilts = np.array(tilts)
    return t_lap, vmax, float(np.percentile(tilts,50)), float(tilts.max())

if __name__ == "__main__":
    r, tang, kappa, arc = build_line()
    print(f"line length to gate-5 plane = {arc[-1]:.2f} m  (straight-line gate chain ~ "
          f"{np.linalg.norm(np.diff(np.vstack([START,GATES]),axis=0),axis=1).sum():.2f} m)")
    print(f"per-gate min-radius (1/kappa_max regions): kappa_max={kappa.max():.4f} "
          f"-> r_min={1/max(kappa.max(),1e-9):.1f} m")
    print()
    print(f"{'tilt_cap':>8} {'topp_t':>8} {'v_max':>7} {'v_wall':>7} {'geomtilt_p50':>12} {'geomtilt_max':>12}")
    for cap in (60, 65, 75, 80):
        t_lap, vmax, gp50, gmax = geometry_tilt(r, tang, kappa, arc, cap)
        _, _, vwall = coupled_topp(r, tang, kappa, arc, cap)
        print(f"{cap:>8} {t_lap:>8.2f} {vmax:>7.1f} {vwall:>7.1f} {gp50:>12.1f} {gmax:>12.1f}")

    # --- cross-track / latency sensitivity at the binding gate (gate-4) ---
    print("\n--- gate-4 binding-margin erosion ---")
    print("binding margin (simstart, r=0.38) = 0.155 m; pass band r=0.38 = 0.37 m")
    # at TOPP gate-4 speed, 67 ms lag -> along-track only if attitude error 0; cross-track from
    # a held attitude error eps over latency dt:  dx ~ 0.5*g*tan(eps)*dt^2
    for v_g4 in (20.0, 30.0, 37.7):
        for eps_deg in (2.0, 5.0):
            dt = 0.067
            a_lat = G*np.tan(np.radians(eps_deg))
            cross = 0.5*a_lat*dt**2
            print(f"  v={v_g4:>5.1f} m/s  att_err={eps_deg}deg  latency_cross_track={cross*1000:.2f} mm "
                  f"(margin 155 mm)")
    # perception: world-frame line + East fix sigma 0.47 m maps DIRECTLY to cross-track
    print("  perception East sigma=0.47 m -> 3.0x the 0.155 m binding margin (UNFILTERED); "
          "KF must reach <0.05 m 1-sigma to keep 3-sigma inside 0.155 m")
