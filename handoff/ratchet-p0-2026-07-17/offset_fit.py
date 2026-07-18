"""REPLAY-RATCHET P0.2 -- systematic-offset fit (analysis only, no shipping-code changes).

Answers OFFSET-FIT.md Q1-Q4 from the three settled replays
(20260718_02*_ratchet_p02_r{1,2,3}_f1/sysid_vq2_log.csv) + the champion source
(20260714_202317_panel_run_f1/ego_obs.jsonl) + the tape (tape_full[.raw].csv).

Signals available are ASYMMETRIC:
  * champion : kf_pos_ned (estimator NED position track) + obs attitude, NO IMU.
  * replays  : IMU (gyro_* raw wire, accel_* body specific force) + cmd_w*, NO position.
So the replay line is recovered by STRAPDOWN integration of its IMU and compared to the
champion's kf_pos_ned line.  All conventions are honored as-is (the raw-gyro<->command sign is
the known self-consistent "tail-first" alias); the gyro sign used for physical integration is
VALIDATED empirically by reproducing the champion's forward+lateral arrival (see GYRO_SIGN).

Run:  <venv>/Scripts/python.exe handoff/ratchet-p0-2026-07-17/offset_fit.py
Stdlib + numpy only (matplotlib optional for the PNG).
"""  # noqa: W605
from __future__ import annotations
import csv, json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPLAYS = ["20260718_021613_ratchet_p02_r1_f1",
           "20260718_021824_ratchet_p02_r2_f1",
           "20260718_022415_ratchet_p02_r3_f1"]
CHAMP = "20260714_202317_panel_run_f1"
GYRO_SIGN = -1.0          # physical FRD body rate = GYRO_SIGN * gyro_raw  (validated below)
EGO_RATE_SCALE = 1.2      # wire rate = cmd_w* * 1.2 (client cmd_rate_scale forced in ego mode)
RATE_HALF = 3.14          # action->rate rescale span
G = np.array([0.0, 0.0, 9.81])   # NED gravity (down +)

# champion gate passes (active_gate_index transitions), tape-relative times & kf_pos_ned
GATES = {  # name: (t_rel_s, champ kf_pos_ned)
    "g0": (2.945, np.array([11.05, 0.47, -4.69])),
    "g1": (5.168, np.array([27.43, 9.93, -5.67])),
    "g2": (6.189, np.array([35.64, 12.90, -4.77])),
}


# ---------------------------------------------------------------- loaders
def load_replay(name):
    rows = list(csv.DictReader(open(ROOT / "data" / "runs" / name / "sysid_vq2_log.csv")))
    def f(rr, k):
        return np.array([float(r[k]) if r[k] not in ("", "None") else np.nan for r in rr])
    boot = [r for r in rows if r["phase"] == "boot"]
    prog = [r for r in rows if r["phase"] == "prog"]
    t = f(prog, "sim_time_ns") / 1e9
    return dict(
        t=t - t[0],
        gyro=np.stack([f(prog, "gyro_x"), f(prog, "gyro_y"), f(prog, "gyro_z")], 1),
        accel=np.stack([f(prog, "accel_x"), f(prog, "accel_y"), f(prog, "accel_z")], 1),
        cmd=np.stack([f(prog, "cmd_wx"), f(prog, "cmd_wy"), f(prog, "cmd_wz")], 1),
        boot_accel=np.array([float(boot[-1]["accel_x"]), float(boot[-1]["accel_y"]),
                             float(boot[-1]["accel_z"])]),
    )


def load_champ():
    recs = [json.loads(l) for l in open(ROOT / "data" / "runs" / CHAMP / "ego_obs.jsonl")]
    P = np.array([r["kf_pos_ned"] for r in recs])
    T = np.array([r["sim_time_ns"] for r in recs]) / 1e9
    return T - T[0], P, recs


# ---------------------------------------------------------------- strapdown
def R_align(f_body):
    """Minimal rotation R_wb (body->world NED, yaw0=0) taking rest specific force to [0,0,-g]."""
    a = f_body / np.linalg.norm(f_body)
    b = np.array([0.0, 0.0, -1.0])
    v = np.cross(a, b); s = np.linalg.norm(v); c = float(np.dot(a, b))
    if s < 1e-9:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)


def strapdown(rep, gyro_sign=GYRO_SIGN, gyro_bias=None):
    t = rep["t"]; gyro = rep["gyro"].copy(); accel = rep["accel"]
    if gyro_bias is not None:
        gyro = gyro - gyro_bias
    R = R_align(rep["boot_accel"])
    v = np.zeros(3); p = np.zeros(3)
    P = [p.copy()]; V = [v.copy()]; H = [0.0]
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        w = gyro_sign * gyro[i - 1]
        th = w * dt; ang = np.linalg.norm(th)
        if ang > 1e-12:
            k = th / ang
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            dR = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K
        else:
            dR = np.eye(3)
        R = R @ dR
        aw = R @ accel[i - 1] + G
        v = v + aw * dt; p = p + v * dt
        P.append(p.copy()); V.append(v.copy())
        H.append(math.degrees(math.atan2(R[1, 0], R[0, 0])))
    return dict(t=t, P=np.array(P), V=np.array(V), H=np.array(H))


def at(t, arr, tt):
    return arr[int(np.argmin(np.abs(t - tt)))]


# ---------------------------------------------------------------- Q0 validation
def q0_validate(reps, champ_T, champ_P):
    print("== Q0  gyro-sign validation (strapdown must reproduce champion forward+lateral) ==")
    for sgn in (+1.0, -1.0):
        s = strapdown(reps[0], gyro_sign=sgn)
        g0 = at(s["t"], s["P"], 2.945)
        print(f"   gyro_sign={sgn:+.0f}: r1 @g0 NED={np.round(g0,2)}  champ=[11.05,0.47,-4.69]")
    print(f"   -> GYRO_SIGN={GYRO_SIGN:+.0f} matches champion East sign (physical FRD = sign*raw).\n")


# ---------------------------------------------------------------- Q1
def cross_track(champ_T, champ_P, s):
    """Frame-invariant: signed cross-track (right +) of the replay vs the champion path,
    decomposed on the champion's instantaneous course, at matched time."""
    out_t, out_ct, out_al = [], [], []
    for i, tt in enumerate(s["t"]):
        if tt > champ_T[-1]:
            break
        kc = int(np.argmin(np.abs(champ_T - tt)))
        k0 = max(0, kc - 3); k1 = min(len(champ_T) - 1, kc + 3)
        course = champ_P[k1, :2] - champ_P[k0, :2]
        nrm = np.linalg.norm(course)
        if nrm < 1e-6:
            continue
        fwd = course / nrm
        right = np.array([-fwd[1], fwd[0]])   # NED right = rotate course +90deg about Down (N->E)
        d = s["P"][i, :2] - champ_P[kc, :2]
        out_t.append(tt); out_ct.append(float(d @ right)); out_al.append(float(d @ fwd))
    return np.array(out_t), np.array(out_ct), np.array(out_al)


def q1(reps, champ_T, champ_P):
    print("== Q1  characterize the systematic offset ==")
    S = [strapdown(r) for r in reps]
    # forward-match quality (validates the frame + integration)
    fm = [at(s["t"], s["P"], 2.945)[0] for s in S]
    print(f"   forward(N) @g0: replays {np.round(fm,2)} vs champ 11.05  "
          f"(match to {np.mean(fm)-11.05:+.2f} m -> strapdown/frame trustworthy)")

    # East-offset onset (champion-NED frame)
    print("   East offset dE(t) = replay_mean_E - champ_E   (champion NED frame):")
    ts = np.array([0.2, 0.5, 0.8, 1.2, 1.6, 2.0, 2.4, 2.945, 3.5, 4.2, 5.168])
    dE_series = []
    for tt in ts:
        es = [at(s["t"], s["P"], tt)[1] for s in S]
        ce = at(champ_T, champ_P, tt)[1]
        dE_series.append(np.mean(es) - ce)
        print(f"     t={tt:5.2f}s  dE={np.mean(es)-ce:+5.2f} m  (spread {np.std(es):.2f})")
    dE_series = np.array(dE_series)

    # power law over pre-chaos window
    m = (ts >= 0.6) & (ts <= 2.945) & (dE_series > 0.02)
    n_exp = np.polyfit(np.log(ts[m]), np.log(dE_series[m]), 1)[0]
    a_lat = np.polyfit(0.5 * ts[m]**2, dE_series[m], 1)[0]
    print(f"   onset shape:  dE ~ t^{n_exp:.2f}   (n~1 translation, ~2 const-heading, ~3 rate-bias)")
    print(f"   best constant differential lateral accel  Da_E = {a_lat:.3f} m/s^2")

    # frame-invariant cross-track + heading-vs-translation
    print("   cross-track (frame-invariant, on champion course) & course offset:")
    print(f"     {'t':>5} {'crossT':>7} {'repCourse-champCourse':>22}")
    for s in S[:1]:
        t_ct, ct, al = cross_track(champ_T, champ_P, s)
    # replay-mean course offset vs champion course
    for tt in [1.2, 1.6, 2.0, 2.4, 2.945]:
        cts = []
        for s in S:
            t_ct, ctv, _ = cross_track(champ_T, champ_P, s)
            cts.append(at(t_ct, ctv, tt))
        # champion course & replay course
        kc = int(np.argmin(np.abs(champ_T - tt)))
        vch = (champ_P[min(kc+3,len(champ_T)-1)] - champ_P[max(kc-3,0)])[:2]
        chc = math.degrees(math.atan2(vch[1], vch[0]))
        rcs = []
        for s in S:
            vj = at(s["t"], s["V"], tt)[:2]
            rcs.append(math.degrees(math.atan2(vj[1], vj[0])))
        print(f"     {tt:5.2f} {np.mean(cts):+7.2f}      repCourse {np.mean(rcs):+6.1f} - champ {chc:+6.1f} "
              f"= {np.mean(rcs)-chc:+5.1f} deg")

    # equivalent heading bias (course-difference plateau), and gate misses
    print("   lateral miss at gate planes (replay_mean_E - champ_E):")
    for nm, (tt, pc) in GATES.items():
        es = [at(s["t"], s["P"], tt)[1] for s in S]
        print(f"     {nm} t={tt:.3f}: dE={np.mean(es)-pc[1]:+.2f} m  (rep {np.mean(es):+.2f} "
              f"+/- {np.std(es):.2f}, champ {pc[1]:+.2f})")
    return S


# ---------------------------------------------------------------- Q2
def q2_chaos(reps):
    print("\n== Q2  chaotic divergence (pairwise replay residuals) ==")
    S = [strapdown(r) for r in reps]
    tmax = min(s["t"][-1] for s in S)
    grid = np.arange(0.0, tmax, 0.025)
    def interp(s, arr):
        return np.stack([np.interp(grid, s["t"], arr[:, j]) for j in range(arr.shape[1])], 1)
    Hs = [np.interp(grid, s["t"], s["H"]) for s in S]
    Vs = [interp(s, s["V"]) for s in S]
    Ps = [interp(s, s["P"]) for s in S]
    Gs = [interp(s, r["gyro"]) for s, r in zip(S, reps)]
    pairs = [(0, 1), (0, 2), (1, 2)]
    dH = np.mean([np.abs(Hs[i] - Hs[j]) for i, j in pairs], 0)
    dV = np.mean([np.linalg.norm(Vs[i] - Vs[j], axis=1) for i, j in pairs], 0)
    dP = np.mean([np.linalg.norm(Ps[i] - Ps[j], axis=1) for i, j in pairs], 0)
    dG = np.mean([np.linalg.norm(Gs[i] - Gs[j], axis=1) for i, j in pairs], 0)
    print(f"   {'t':>5} {'d|gyro|':>8} {'dHead':>7} {'d|V|':>7} {'d|P|':>7}   (pairwise mean over 3 pairs)")
    for tt in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
        if tt > grid[-1]:
            break
        k = int(np.argmin(np.abs(grid - tt)))
        print(f"   {tt:5.2f} {dG[k]:8.3f} {dH[k]:7.2f} {dV[k]:7.3f} {dP[k]:7.3f}")
    # e-folding of the position residual over the growth band
    band = (grid >= 1.0) & (grid <= 4.0) & (dP > 1e-3)
    lam = np.polyfit(grid[band], np.log(dP[band]), 1)[0]
    print(f"   position-residual growth 1-4 s: lambda={lam:.3f}/s  ->  e-fold {1/lam:.2f} s, "
          f"double {math.log(2)/lam:.2f} s")
    # horizon at 0.5 m aperture precision
    cross = grid[dP >= 0.5]
    hz = cross[0] if len(cross) else float("nan")
    print(f"   open-loop horizon (pairwise d|P| reaches 0.5 m): t={hz:.2f} s  (cf. reported 3-5 s)")


# ---------------------------------------------------------------- Q3
def q3_resample():
    print("\n== Q3  resample-bias check (ZOH 40 Hz vs raw timeline command integrals) ==")
    def load_tape(fn):
        rows = list(csv.DictReader(open(ROOT / fn)))
        t = np.array([float(r["t"]) for r in rows])
        a = {k: np.array([float(r[k]) for r in rows]) for k in ("a_roll", "a_pitch", "a_yaw")}
        sk = np.array([int(r["src_k"]) for r in rows])
        return t, a, sk
    tz, az, skz = load_tape("tape_full.csv")
    tr, ar, skr = load_tape("tape_full.raw.csv")
    # wire rate (deg/s) = 3.14 * a * ego_rate_scale ; integrate to net attitude change (deg)
    def net_deg(t, a, axis, tend=None):
        m = np.ones(len(t), bool) if tend is None else (t <= tend)
        # ZOH: each sample holds until next; integral = sum(a_i * dt_i)
        tt = t[m]; aa = a[axis][m]
        dt = np.diff(tt, append=tt[-1] + (tt[-1]-tt[-2] if len(tt) > 1 else 0.025))
        return math.degrees(RATE_HALF * EGO_RATE_SCALE * float(np.sum(aa * dt)))
    # g0 leg end: champion passed gate0 at src_k=76
    t_g0_z = tz[skz <= 76][-1]; t_g0_r = tr[skr <= 76][-1]
    print(f"   {'axis':>7} {'leg':>5} {'raw(deg)':>9} {'ZOH(deg)':>9} {'ZOH-raw':>8}")
    for axis, nm in [("a_yaw", "yaw"), ("a_roll", "roll"), ("a_pitch", "pitch")]:
        for leg, (tez, ter) in [("g0", (t_g0_z, t_g0_r)), ("full", (None, None))]:
            r = net_deg(tr, ar, axis, ter); z = net_deg(tz, az, axis, tez)
            print(f"   {nm:>7} {leg:>5} {r:9.3f} {z:9.3f} {z-r:8.3f}")
    print("   (expected ~0: ZOH of a piecewise-constant command preserves the integral;"
          " residual = edge quantization.)")


def plot(reps, champ_T, champ_P):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"\n(plot skipped: {e})"); return
    S = [strapdown(r) for r in reps]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    m = champ_T <= 6.3
    ax[0].plot(champ_P[m, 0], champ_P[m, 1], "k-", lw=2, label="champion kf_pos_ned")
    for i, s in enumerate(S):
        mm = s["t"] <= 6.3
        ax[0].plot(s["P"][mm, 0], s["P"][mm, 1], "-", alpha=.7, label=f"replay r{i+1} strapdown")
    for nm, (tt, pc) in GATES.items():
        ax[0].plot(pc[0], pc[1], "r*", ms=13); ax[0].annotate(nm, (pc[0], pc[1]))
    ax[0].set_xlabel("North (m)"); ax[0].set_ylabel("East (m, +=right)")
    ax[0].set_title("Ground track: replays fly RIGHT (+E) of champion"); ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3); ax[0].invert_yaxis()
    ts = np.linspace(0.1, 5.0, 60)
    dE = [np.mean([at(s["t"], s["P"], tt)[1] for s in S]) - at(champ_T, champ_P, tt)[1] for tt in ts]
    sp = [np.std([at(s["t"], s["P"], tt)[1] for s in S]) for tt in ts]
    ax[1].plot(ts, dE, "b-", label="dE = replay_mean - champ (right+)")
    ax[1].fill_between(ts, np.array(dE)-np.array(sp), np.array(dE)+np.array(sp), alpha=.2)
    for nm, (tt, _) in GATES.items():
        ax[1].axvline(tt, color="r", ls=":", alpha=.5); ax[1].annotate(nm, (tt, 0.2))
    ax[1].set_xlabel("t since first cmd (s)"); ax[1].set_ylabel("East offset (m)")
    ax[1].set_title("Systematic right-offset: ~0 at t0, ~t^2.8, +2m @ g0"); ax[1].grid(alpha=.3)
    ax[1].legend(fontsize=8)
    out = Path(__file__).parent / "offset_fit.png"
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print(f"\n(wrote {out.name})")


def main():
    reps = [load_replay(r) for r in REPLAYS]
    champ_T, champ_P, _ = load_champ()
    q0_validate(reps, champ_T, champ_P)
    q1(reps, champ_T, champ_P)
    q2_chaos(reps)
    q3_resample()
    plot(reps, champ_T, champ_P)


if __name__ == "__main__":
    main()
