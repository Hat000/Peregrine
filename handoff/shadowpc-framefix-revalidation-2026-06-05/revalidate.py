"""Frame-fix re-validation (OFFLINE replay) — Phases 1-3.

Hypothesis under test: the ODOMETRY body->world velocity frame bug (fixed in c3b5a8e) is the
root cause of the gate-0 saga's "KF velocity lags 4x" (project_ctbr_control_sysid.md §5:
truth -0.92 m/s, KF reported -0.22) and the lateral oscillation.

Everything is computed from the recorded RAW MAVLink messages (no pre-fix checkout needed):
  * ODOMETRY  -> body twist (vx/vy/vz in child_frame_id=BODY) + attitude quaternion + pose.
  * LOCAL_POSITION_NED -> WORLD velocity (truth) + world position.
  * HIGHRES_IMU -> sim_time_ns master clock + accel_body (KF predict).

Phase 1 (Check 1): raw body vs world disagree (sign-flip at -180); rotated body->world agrees
                   with LPN truth (incl. OFF-LEVEL: ADD #2); LPN truth == d(pos)/dt.
Phase 2 (Check 3): replay the Navigator KF twice (OLD mixed velocity vs NEW rotated) vs truth.
Phase 3 (Check 0): which velocity field the controller/KF actually consumed.

Usage:  python revalidate.py <phase> <run_dir_name> [run_dir_name ...]
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# worktree root = .../stupefied-booth-c1ad07 (parents[2]); its src has the FIX (c3b5a8e).
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from racer.frames import euler_from_quat_wxyz, world_vec_from_body_quat  # noqa: E402
from racer.recording import RecordingReader  # noqa: E402

MAIN_RUNS = Path("C:/Users/Shadow/Peregrine/data/runs")


def _frame_name(val: int) -> str:
    from pymavlink import mavutil
    try:
        return mavutil.mavlink.enums["MAV_FRAME"][int(val)].name
    except Exception:  # noqa: BLE001
        return f"?{val}"


@dataclass
class Streams:
    # ODOMETRY
    o_t: np.ndarray          # recv epoch seconds
    o_q: np.ndarray          # (N,4) wxyz
    o_vbody: np.ndarray      # (N,3) body twist
    o_pos: np.ndarray        # (N,3) world pos
    o_rpy: np.ndarray        # (N,3) roll,pitch,yaw deg
    o_rates: np.ndarray      # (N,3) body angular rate
    frame_id: int
    child_frame_id: int
    frame_id_const: bool
    child_const: bool
    # LOCAL_POSITION_NED (truth, world)
    l_t: np.ndarray
    l_vworld: np.ndarray     # (M,3)
    l_pos: np.ndarray        # (M,3)
    # HIGHRES_IMU
    i_t: np.ndarray
    i_simns: np.ndarray
    i_accel: np.ndarray      # (K,3)


def load(run: str) -> Streams:
    rd = RecordingReader(MAIN_RUNS / run)
    o_t, o_q, o_vb, o_pos, o_rpy, o_rt = [], [], [], [], [], []
    l_t, l_vw, l_pos = [], [], []
    i_t, i_ns, i_ac = [], [], []
    fids, cids = set(), set()
    for m in rd.iter_mavlink():
        t = m.get_type()
        ts = float(getattr(m, "_timestamp", 0.0))
        if t == "ODOMETRY":
            q = np.array([float(v) for v in m.q], dtype=np.float64)
            r, p, y = euler_from_quat_wxyz(q)
            fids.add(int(getattr(m, "frame_id", -1)))
            cids.add(int(getattr(m, "child_frame_id", -1)))
            o_t.append(ts); o_q.append(q)
            o_vb.append([float(m.vx), float(m.vy), float(m.vz)])
            o_pos.append([float(m.x), float(m.y), float(m.z)])
            o_rpy.append([np.degrees(r), np.degrees(p), np.degrees(y)])
            o_rt.append([float(m.rollspeed), float(m.pitchspeed), float(m.yawspeed)])
        elif t == "LOCAL_POSITION_NED":
            l_t.append(ts)
            l_vw.append([float(m.vx), float(m.vy), float(m.vz)])
            l_pos.append([float(m.x), float(m.y), float(m.z)])
        elif t == "HIGHRES_IMU":
            i_t.append(ts); i_ns.append(int(m.time_usec) * 1000)
            i_ac.append([float(m.xacc), float(m.yacc), float(m.zacc)])
    fid = sorted(fids)[0] if fids else -1
    cid = sorted(cids)[0] if cids else -1
    return Streams(
        o_t=np.array(o_t), o_q=np.array(o_q), o_vbody=np.array(o_vb),
        o_pos=np.array(o_pos), o_rpy=np.array(o_rpy), o_rates=np.array(o_rt),
        frame_id=fid, child_frame_id=cid, frame_id_const=(len(fids) == 1),
        child_const=(len(cids) == 1),
        l_t=np.array(l_t), l_vworld=np.array(l_vw), l_pos=np.array(l_pos),
        i_t=np.array(i_t), i_simns=np.array(i_ns), i_accel=np.array(i_ac),
    )


def interp_lpn(s: Streams, t: np.ndarray) -> np.ndarray:
    """Linear-interpolate the LPN world velocity onto times t (3 cols)."""
    return np.column_stack([np.interp(t, s.l_t, s.l_vworld[:, k]) for k in range(3)])


def rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(a))))


def phase1(run: str) -> None:
    s = load(run)
    print(f"\n############## PHASE 1 / Check 1 — {run} ##############")

    # --- ADD #1: frame enums (definitive) ---
    print(f"[ADD#1] ODOMETRY.frame_id       = {s.frame_id} ({_frame_name(s.frame_id)})  "
          f"constant={s.frame_id_const}")
    print(f"[ADD#1] ODOMETRY.child_frame_id = {s.child_frame_id} ({_frame_name(s.child_frame_id)})  "
          f"constant={s.child_const}")
    print(f"        ODOMETRY n={len(s.o_t)}  LPN n={len(s.l_t)}  IMU n={len(s.i_t)}")

    # --- truth self-consistency: LPN world velocity vs central-difference d(pos)/dt ---
    # The recv-time clock jitters and the sim DUPLICATES LPN messages (sub-ms bursts; given1 also
    # has an epoch reset -> 61 m teleport). Naive central-diff over recv-time blows up on those.
    # Robust version: keep a clean subsequence (recv-time advanced >=2 ms since last kept), then
    # central-difference; reject finite-diff points implying >50 m/s (teleport/reset discontinuity).
    keep = [0]
    for i in range(1, len(s.l_t)):
        if s.l_t[i] - s.l_t[keep[-1]] >= 2e-3:
            keep.append(i)
    keep = np.array(keep)
    lt, lp, lv = s.l_t[keep], s.l_pos[keep], s.l_vworld[keep]
    fd = np.full_like(lv, np.nan)
    fd[1:-1] = (lp[2:] - lp[:-2]) / (lt[2:] - lt[:-2])[:, None]
    ok = ~np.isnan(fd[:, 0]) & (np.linalg.norm(fd, axis=1) < 50.0)   # drop teleport spikes
    mv = ok & (np.linalg.norm(lv, axis=1) > 0.3)                     # compare only while moving
    err = np.linalg.norm(lv[mv] - fd[mv], axis=1)
    print(f"[truth x-check] LPN v_world vs robust d(pos)/dt over {int(mv.sum())} moving samples "
          f"(deduped {len(s.l_t)}->{len(keep)}, dropped {int((~ok).sum())} teleport/edge): "
          f"median={np.median(err):.4f}  vec-RMS={rms(err):.4f}  p90={np.percentile(err,90):.4f} m/s")

    # --- the -180 lateral/forward segment: rotate ODO body->world, compare to LPN ---
    yaw = s.o_rpy[:, 2]
    speed = np.linalg.norm(s.o_vbody, axis=1)
    seg = (np.abs(np.abs(yaw) - 180.0) < 10.0) & (speed > 0.3)
    print(f"\n[-180 segment] {int(seg.sum())} ODOMETRY samples at |yaw|>170 deg AND moving (|v_body|>0.3)")
    if seg.sum() == 0:
        print("  !! no -180 moving segment")
    else:
        idx = np.where(seg)[0]
        ot = s.o_t[idx]
        vbody = s.o_vbody[idx]
        # rotate each body twist to world via its own ODOMETRY quaternion (the fix)
        vrot = np.array([world_vec_from_body_quat(vbody[i], s.o_q[idx[i]]) for i in range(len(idx))])
        vtruth = interp_lpn(s, ot)   # LPN world velocity at the ODOMETRY times

        err_raw = vbody - vtruth     # RAW body stored as world (the bug)
        err_rot = vrot - vtruth      # ROTATED (the fix)
        print(f"  RAW  body-as-world vs LPN truth:  per-axis RMS = "
              f"{np.round([rms(err_raw[:,k]) for k in range(3)],4)}  vec-RMS = {rms(np.linalg.norm(err_raw,axis=1)):.4f} m/s")
        print(f"  ROT  rotated->world vs LPN truth: per-axis RMS = "
              f"{np.round([rms(err_rot[:,k]) for k in range(3)],4)}  vec-RMS = {rms(np.linalg.norm(err_rot,axis=1)):.4f} m/s")
        # sign-flip evidence on x and y
        for ax, nm in ((0, "vx"), (1, "vy")):
            bigt = np.abs(vtruth[:, ax]) > 0.3
            flip = bigt & (np.sign(vbody[:, ax]) != np.sign(vtruth[:, ax]))
            print(f"    {nm}: of {int(bigt.sum())} samples with |truth|>0.3, "
                  f"RAW body sign-flipped vs truth = {int(flip.sum())} ({100*flip.sum()/max(bigt.sum(),1):.0f}%)")
        # a few concrete rows
        print("    sample rows (t, yaw, body->truth->rotated) per axis:")
        for j in np.linspace(0, len(idx) - 1, 4).astype(int):
            print(f"      yaw={s.o_rpy[idx[j],2]:+.0f}  "
                  f"vx: body{vbody[j,0]:+.2f} truth{vtruth[j,0]:+.2f} rot{vrot[j,0]:+.2f} | "
                  f"vy: body{vbody[j,1]:+.2f} truth{vtruth[j,1]:+.2f} rot{vrot[j,1]:+.2f}")

    # --- ADD #2: rotation validation OFF-LEVEL (pitched AND rolled), on REAL motion only ---
    # CRITICAL: filter to REAL motion (LPN velocity corroborated by d(pos)/dt). Some runs (given1)
    # are FROZEN telemetry -- a stuck pose+velocity that would "agree" trivially without exercising
    # the rotation. Real-motion mask = |v_fd - v_lpn| < 0.6 m/s & |v_lpn| > 0.5 m/s.
    roll = s.o_rpy[:, 0]; pitch = s.o_rpy[:, 1]
    vtruth_all = interp_lpn(s, s.o_t)
    pos_i = np.column_stack([np.interp(s.o_t, s.l_t, s.l_pos[:, k]) for k in range(3)])
    vfd = np.full_like(pos_i, np.nan)
    vfd[2:-2] = (pos_i[4:] - pos_i[:-4]) / (s.o_t[4:] - s.o_t[:-4])[:, None]   # ~+-2 ODO samples
    real = (~np.isnan(vfd[:, 0])) & (np.linalg.norm(vfd - vtruth_all, axis=1) < 0.6) \
        & (np.linalg.norm(vtruth_all, axis=1) > 0.5)
    vrot_all = np.array([world_vec_from_body_quat(s.o_vbody[i], s.o_q[i]) for i in range(len(s.o_t))])
    err_all = np.linalg.norm(vrot_all - vtruth_all, axis=1)
    nreal = int(real.sum())
    print(f"\n[ADD#2] REAL-motion samples (vel corroborated by d(pos)/dt): {nreal}")
    if nreal >= 20:
        print(f"  real-motion roll range=[{roll[real].min():.0f},{roll[real].max():.0f}] deg, "
              f"pitch range=[{pitch[real].min():.0f},{pitch[real].max():.0f}] deg")
        for tag, m in (("|roll|>8",      real & (np.abs(roll) > 8)),
                       ("|pitch|>8",     real & (np.abs(pitch) > 8)),
                       ("|roll|>8&|pitch|>5", real & (np.abs(roll) > 8) & (np.abs(pitch) > 5))):
            n = int(m.sum())
            if n:
                print(f"    real & {tag:18s}: n={n:5d}  rotated-vs-truth RMS={rms(err_all[m]):.4f}  "
                      f"max={err_all[m].max():.3f}  RAW-vs-truth RMS="
                      f"{rms(np.linalg.norm(s.o_vbody[m]-vtruth_all[m],axis=1)):.3f} m/s")
            else:
                print(f"    real & {tag:18s}: n=0 (not exercised in this run)")
    else:
        print("  (insufficient real-motion samples -- likely a frozen/garbage recording)")


def replay_kf(run: str, old: bool):
    """Replay the PRODUCTION MavlinkClient._handle + Navigator over the recorded stream.
    old=True neutralizes the rotation (identity) so _handle stores the RAW body twist as
    velocity_ned -> reproduces the pre-fix interleaved world/body mix. old=False = the fix.
    Returns per-IMU-tick arrays: t (recv s), kf_vel (3), yaw deg, ingested meas velocity_ned (3)."""
    import racer.mavlink_client as mc
    from racer.navigator import Navigator, NavigatorConfig

    saved = mc.world_vec_from_body_quat
    if old:
        mc.world_vec_from_body_quat = lambda v, q: np.asarray(v, dtype=np.float64)
    try:
        client = mc.MavlinkClient()
        nav = Navigator(gates=[], detector=None, config=NavigatorConfig(use_vision=False))
        t, kfv, yaw, meas = [], [], [], []
        for m in RecordingReader(MAIN_RUNS / run).iter_mavlink():
            client._handle(m)
            ns = nav.update(client.state)
            if m.get_type() == "HIGHRES_IMU":   # one record per KF step (sim_time advanced)
                t.append(float(getattr(m, "_timestamp", 0.0)))
                v = ns.velocity_ned
                kfv.append(np.asarray(v, dtype=np.float64) if v is not None else [np.nan]*3)
                yaw.append(float(client.state.yaw))
                mv = client.state.velocity_ned
                meas.append(np.asarray(mv, dtype=np.float64) if mv is not None else [np.nan]*3)
    finally:
        mc.world_vec_from_body_quat = saved
    return (np.array(t), np.array(kfv), np.degrees(np.array(yaw)), np.array(meas))


def phase2(run: str) -> None:
    s = load(run)
    print(f"\n############## PHASE 2 / Check 3 (decisive replay) — {run} ##############")
    to, ko, yo, mo = replay_kf(run, old=True)    # OLD = buggy mix
    tn, kn, yn, mn = replay_kf(run, old=False)   # NEW = fixed rotation
    # truth = LPN world velocity interpolated onto the KF-output (IMU) times
    truth_o = np.column_stack([np.interp(to, s.l_t, s.l_vworld[:, k]) for k in range(3)])
    truth_n = np.column_stack([np.interp(tn, s.l_t, s.l_vworld[:, k]) for k in range(3)])
    # segment: at the -180 course heading AND genuinely moving
    def seg(yaw, truth):
        return (np.abs(np.abs(yaw) - 180.0) < 10.0) & (np.linalg.norm(truth, axis=1) > 0.5)
    so, sn = seg(yo, truth_o), seg(yn, truth_n)
    print(f"  -180 moving segment: OLD {int(so.sum())} ticks, NEW {int(sn.sum())} ticks")

    for tag, k, tr, m, sg in (("OLD (buggy mix)", ko, truth_o, mo, so),
                              ("NEW (rotated fix)", kn, truth_n, mn, sn)):
        e = k[sg] - tr[sg]
        # what the KF INGESTED (measurement) vs what it OUTPUT, vs truth, per axis means
        print(f"\n  [{tag}]  over {int(sg.sum())} ticks:")
        print(f"    KF-vel ERROR vs truth: per-axis RMS = {np.round([rms(e[:,j]) for j in range(3)],3)}  "
              f"vec-RMS = {rms(np.linalg.norm(e,axis=1)):.3f} m/s")
        for ax, nm in ((0, "vx(fwd)"), (1, "vy(lat)")):
            big = np.abs(tr[sg][:, ax]) > 0.5
            if big.sum() >= 5:
                tmean = tr[sg][big, ax].mean(); kmean = k[sg][big, ax].mean(); mmean = m[sg][big, ax].mean()
                ratio = kmean / tmean if abs(tmean) > 1e-6 else float("nan")
                print(f"    {nm}: truth_mean={tmean:+.3f}  KF_mean={kmean:+.3f}  (KF/truth={ratio:.2f})  "
                      f"ingested_meas_mean={mmean:+.3f}   [n={int(big.sum())}]")


def phase3(run: str) -> None:
    s = load(run)
    print(f"\n############## PHASE 3 / Check 0 (data flow) — {run} ##############")
    print("  CODE FACTS (what consumed the velocity):")
    print("   - navigator.py:281-285  KF.update_velocity(ds.velocity_ned, given_vel_std=0.10)")
    print("   - fly_vq1.py:355-358    gs = client.state ; rawv = gs.velocity_ned ;")
    print("                           vel = [rawv[0], rawv[1], KF_vz]   (default, not --use-kf-state)")
    print("   => BOTH the KF and the 'raw given horizontal velocity' workaround read ds.velocity_ned,")
    print("      which IS the interleaved ODOMETRY(body)+LPN(world) mix.")
    # quantify what the workaround actually fed the lateral loop at -180: the OLD mixed field
    to, ko, yo, mo = replay_kf(run, old=True)
    truth = np.column_stack([np.interp(to, s.l_t, s.l_vworld[:, k]) for k in range(3)])
    sg = (np.abs(np.abs(yo) - 180.0) < 10.0) & (np.linalg.norm(truth, axis=1) > 0.5)
    print(f"\n  Over the -180 moving segment ({int(sg.sum())} ticks) — the field fly_vq1 fed the controller:")
    for ax, nm in ((0, "vx(fwd/along-track)"), (1, "vy(lat/cross-track)")):
        big = np.abs(truth[sg][:, ax]) > 0.3
        if big.sum() >= 5:
            tmean = truth[sg][big, ax].mean()
            fed = mo[sg][big, ax].mean()           # what gs.velocity_ned gave the controller
            flip = np.mean(np.sign(mo[sg][big, ax]) != np.sign(truth[sg][big, ax]))
            print(f"    {nm:22s}: truth_mean={tmean:+.3f}  workaround-fed_mean={fed:+.3f}  "
                  f"(fed/truth={fed/tmean if abs(tmean)>1e-6 else float('nan'):.2f})  "
                  f"sign-flipped {100*flip:.0f}% of samples  [n={int(big.sum())}]")
    print("  => the 'pristine raw given velocity' the lateral fix switched TO was the SAME corrupted")
    print("     mix (not pristine); at -180 it was even more cancelled than the KF output.")


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "1"
    runs = sys.argv[2:]
    for r in runs:
        if phase == "1":
            phase1(r)
        elif phase == "2":
            phase2(r)
        elif phase == "3":
            phase3(r)
