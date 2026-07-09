"""Offline hit-map + trajectory diagnostic from an ego_render_rollout .npz.

Answers Fengyou's question: where do the ~80% non-threading crossings land on the gate?
Computes, per episode, the gate-frame (lateral y, vertical z) point where the drone crosses the
target-gate plane (x_gate=0), classifies thread / frame-clip / wide-miss / no-cross, and renders:
  (1) the GATE-PLANE HIT MAP  -- scatter of crossing points + aperture(0.75) + effective-pass(~0.4,
      body-radius-inflated) + outer-frame(1.36) boxes + marginal histograms.
  (2) TOP-DOWN trajectories (gate-frame x=approach, y=lateral) + optimal head-on racing lines.
  (3) SIDE trajectories (x=approach, z=vertical) -- exposes any vertical bias.
Emits a PNG (embedded in the artifact) + a text breakdown.
"""
import sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

NPZ = sys.argv[1] if len(sys.argv) > 1 else "ego_hitmap_vglpan.npz"
OUT = sys.argv[2] if len(sys.argv) > 2 else "ego_hitmap_vglpan.png"
LABEL = sys.argv[3] if len(sys.argv) > 3 else "deterministic · DR-off (deployed)"
LIM = 4.5
HALF_IN = 0.75      # aperture half (1.5 m inner opening)
HALF_OUT = 1.36     # outer physical frame half (2.72 m)
BODY_R = 0.33       # mean body radius (0.28-0.38) -> effective clean-pass half = HALF_IN - r
EFF = HALF_IN - BODY_R  # ~0.42 m: crossing must land inside this for the body to clear cleanly


def w2g(d, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    x = c * d[..., 0] + s * d[..., 1]
    y = -s * d[..., 0] + c * d[..., 1]
    return np.stack([x, y, d[..., 2]], axis=-1)


def hermite(P0, P1, T0, T1, n=40):
    t = np.linspace(0, 1, n)[:, None]
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return h00 * P0 + h10 * T0 + h01 * P1 + h11 * T1


def main():
    d = np.load(NPZ)
    pos, reset_step = d["pos"], d["reset_step"]            # (T,N,3), (N,)
    gate_pos, gate_yaw, spawn_pos = d["gate_pos"], d["gate_yaw"], d["spawn_pos"]
    out_succ, out_coll = d["out_succ"], d["out_coll"]
    out_miss, out_oob = d["out_miss"], d["out_oob"]
    T, N, _ = pos.shape
    print(f"loaded {NPZ}: T={T} N={N}  w_g_half={float(d['w_g_half']):.3f}")

    hits = []          # (y,z,cls) at first forward crossing;  cls: 0 thread,1 frame,2 wide,3 no-cross
    trajs_g = []       # gate-frame trajectory per env (for plotting)
    spawn_g = []       # gate-frame spawn per env
    for e in range(N):
        te = int(reset_step[e]); te = min(max(te, 2), T)
        gate = gate_pos[e, 0]; yaw = gate_yaw[e, 0]
        rel = w2g(pos[:te, e, :] - gate, yaw)             # (te,3) gate frame
        trajs_g.append(rel)
        spawn_g.append(w2g(spawn_pos[e] - gate, yaw))
        x = rel[:, 0]
        k = np.where((x[:-1] < 0) & (x[1:] >= 0))[0]      # explicit forward crossings of x=0
        if len(k):
            i = k[0]
            f = -x[i] / (x[i + 1] - x[i] + 1e-12)
            y = rel[i, 1] + f * (rel[i + 1, 1] - rel[i, 1])
            z = rel[i, 2] + f * (rel[i + 1, 2] - rel[i, 2])
        else:
            # reset overwrites the actual crossing step -> EXTRAPOLATE from the last approach segment
            # (the drone ends ~0.1-0.8 m short of the plane, still moving forward). Project to x=0.
            if te < 2:
                hits.append((np.nan, np.nan, 3)); continue
            p1, p0 = rel[-1], rel[-2]
            vx = p1[0] - p0[0]
            if vx <= 1e-3 or p1[0] < -3.0:   # not approaching forward / stalled far short = never reached
                hits.append((np.nan, np.nan, 3)); continue
            t = -p1[0] / vx                  # steps to reach x=0 from the last point
            y = p1[1] + t * (p1[1] - p0[1])
            z = p1[2] + t * (p1[2] - p0[2])
        linf = max(abs(y), abs(z))
        cls = 0 if linf < HALF_IN else (1 if linf <= HALF_OUT else 2)
        hits.append((y, z, cls))
    hits = np.array(hits)
    cls = hits[:, 2].astype(int)
    ncross = cls < 3
    n_thread = int((cls == 0).sum()); n_frame = int((cls == 1).sum())
    n_wide = int((cls == 2).sum()); n_nocross = int((cls == 3).sum())
    yv, zv = hits[ncross, 0], hits[ncross, 1]
    linf_all = np.maximum(np.abs(yv), np.abs(zv))
    eff_thread = int((linf_all < EFF).sum())
    print(f"THREAD(geom<0.75)={n_thread} FRAME[0.75,1.36]={n_frame} WIDE(>1.36)={n_wide} "
          f"NO-CROSS(floor/oob/timeout)={n_nocross}  | eff-clean(<{EFF:.2f})={eff_thread}")
    if ncross.any():
        print(f"crossings: |lateral y| mean={np.abs(yv).mean():.2f} median={np.median(np.abs(yv)):.2f} | "
              f"|vertical z| mean={np.abs(zv).mean():.2f} median={np.median(np.abs(zv)):.2f} | "
              f"z bias(mean signed)={zv.mean():+.2f}")

    # ---------- figure ----------
    plt.rcParams.update({"font.size": 10, "figure.facecolor": "white", "axes.facecolor": "white"})
    fig = plt.figure(figsize=(15, 6.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1, 1], height_ratios=[4, 1],
                          hspace=0.05, wspace=0.28)
    COL = {0: "#1a9850", 1: "#f5a300", 2: "#d73027"}
    LAB = {0: f"thread (<0.75) n={n_thread}", 1: f"frame-clip [0.75,1.36] n={n_frame}",
           2: f"wide miss (>1.36) n={n_wide}"}

    # (1) HIT MAP
    axm = fig.add_subplot(gs[0, 0])
    for box, c, lw, lab in [(HALF_OUT, "#d73027", 1.5, "outer frame 1.36 m"),
                            (HALF_IN, "#333", 2.0, "aperture 0.75 m"),
                            (EFF, "#1a9850", 1.5, f"clean-pass ~{EFF:.2f} m (body r)")]:
        axm.add_patch(Rectangle((-box, -box), 2 * box, 2 * box, fill=False,
                                edgecolor=c, lw=lw, ls="--" if box != HALF_IN else "-", label=lab))
    for cc in (2, 1, 0):
        m = cls == cc
        axm.scatter(hits[m, 0], hits[m, 1], s=14, c=COL[cc], alpha=0.7,
                    edgecolors="none", label=LAB[cc])
    axm.axhline(0, color="#bbb", lw=0.6); axm.axvline(0, color="#bbb", lw=0.6)
    axm.set_xlim(-LIM, LIM); axm.set_ylim(-LIM, LIM); axm.set_aspect("equal")
    axm.set_xlabel("lateral offset  y  (m)   [+ = drone's left]")
    axm.set_ylabel("vertical offset  z  (m)   [+ = up]")
    axm.set_title(f"GATE-PLANE HIT MAP  (N={N})", fontweight="bold")
    axm.legend(loc="upper right", fontsize=7, framealpha=0.9)

    # marginal: overlay lateral(y) + vertical(z) distributions -> exposes the vertical high-bias
    axhx = fig.add_subplot(gs[1, 0])
    if ncross.any():
        axhx.hist(yv, bins=45, range=(-LIM, LIM), color="#4575b4", alpha=0.6, label="lateral y")
        axhx.hist(zv, bins=45, range=(-LIM, LIM), color="#f5820a", alpha=0.6, label="vertical z")
        axhx.axvline(np.median(zv), color="#f5820a", lw=1.5, ls="--")
    axhx.axvline(0, color="#888", lw=0.8); axhx.set_yticks([])
    axhx.set_xlim(-LIM, LIM); axhx.set_xlabel("crossing offset (m)")
    axhx.legend(fontsize=7, loc="upper left")

    # (2) TOP-DOWN
    axt = fig.add_subplot(gs[:, 1])
    rng = np.random.default_rng(0)
    samp = rng.choice(N, size=min(40, N), replace=False)
    for e in samp:
        r = trajs_g[e]; cc = cls[e]
        axt.plot(r[:, 0], r[:, 1], color=COL.get(cc, "#999"), alpha=0.35, lw=0.8)
        sg = spawn_g[e]
        line = hermite(np.array([sg[0], sg[1]]), np.array([0.0, 0.0]),
                       np.array([-sg[0], -sg[1]]), np.array([abs(sg[0]) * 0.6, 0.0]))
        axt.plot(line[:, 0], line[:, 1], color="#666", alpha=0.25, lw=0.7, ls=":")
    axt.axvline(0, color="k", lw=1)
    axt.plot([0, 0], [-HALF_IN, HALF_IN], color="#1a9850", lw=4, solid_capstyle="butt")
    axt.scatter([spawn_g[e][0] for e in samp], [spawn_g[e][1] for e in samp],
                s=18, c="#222", marker="x", label="spawn")
    axt.set_xlabel("approach axis  x  (m)  [gate at 0]"); axt.set_ylabel("lateral  y  (m)")
    axt.set_title("TOP-DOWN paths + optimal line (dotted)", fontweight="bold")
    axt.legend(loc="upper left", fontsize=7)

    # (3) SIDE
    axs = fig.add_subplot(gs[:, 2])
    for e in samp:
        r = trajs_g[e]; cc = cls[e]
        axs.plot(r[:, 0], r[:, 2], color=COL.get(cc, "#999"), alpha=0.35, lw=0.8)
    axs.axvline(0, color="k", lw=1)
    axs.plot([0, 0], [-HALF_IN, HALF_IN], color="#1a9850", lw=4, solid_capstyle="butt")
    axs.scatter([spawn_g[e][0] for e in samp], [spawn_g[e][2] for e in samp],
                s=18, c="#222", marker="x")
    axs.set_xlabel("approach axis  x  (m)"); axs.set_ylabel("vertical  z  (m)")
    axs.set_title("SIDE paths (vertical)", fontweight="bold")

    fig.suptitle(
        f"vglpan  [{LABEL}]  |  thread {n_thread}/{N}={100*n_thread/N:.0f}%  "
        f"frame-clip {n_frame} ({100*n_frame/N:.0f}%)  wide {n_wide} ({100*n_wide/N:.0f}%)  "
        f"never-reached {n_nocross} ({100*n_nocross/N:.0f}%)  |  "
        f"median vert-bias {np.median(zv) if ncross.any() else float('nan'):+.1f} m",
        fontsize=12, fontweight="bold", y=0.99)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
