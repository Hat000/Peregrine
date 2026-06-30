"""VQ2 FORWARD-pitch-sign disambiguation probe — 2026-06-30.

Context (A10): the gyro fix is validated and the live attitude estimate is trustworthy, BUT when the
seeker commands FORWARD the drone pitches NOSE-UP and drifts BACKWARD, losing the gate. OFFLINE the
controller is PROVABLY correct (a pure-forward accel demand -> NEGATIVE / nose-down pitch-rate cmd, ~-0.49
through the deployed signs). So the inversion is a LIVE-WIRE convention the offline twin can't see. Two
candidates, exactly ONE is true:
  (1) gyro/attitude PITCH polarity  — est_pitch (or raw gyro y) disagrees with the physical nose.
  (2) command-axis body_rate_sign[1] — the controller DID command nose-down, but the live sim realizes
      the pitch COMMAND inverted, so the nose physically goes UP.

This probe drives the FULL deployed vq2_case_c stack with NO vision/detector in the loop:
  * warmup: run the real seeker's settle/anchor HOLD (frame=None => level-hover hold) to reach a stable,
    roughly LEVEL attitude — the same hold the deployed seeker uses post-arm.
  * inject (~INJECT_S): feed the REAL seeker controller (make_seeker_controller signs) a Setpoint whose
    ONLY term is accel_ned = ACCEL * [cos(yaw), sin(yaw), 0] — a PURE FORWARD horizontal acceleration
    demand along the drone's current heading (no position/velocity, no vision). This is exactly the
    egress/pursuit forward feedforward, isolated.
  * cooldown: hold level again, then disarm.

Per tick we log the four synchronized quantities the decision tree needs:
  - cmd_pitch  : the commanded body_rate[1] ACTUALLY sent to the wire (post body_rate_sign AND the
                 cmd_rate_scale=0.4 uplink) — i.e. the final pitch-rate the sim receives.
  - raw_gyro_y : the RAW HIGHRES_IMU ygyro, BEFORE the gyro_sign correction (= state.gyro_body[1] /
                 gyro_sign[1]). The estimator-independent realized rotation.
  - est_pitch  : the estimator's nav.pitch (deg) — the same value nav_estimate.jsonl logs.
  - accel_demand: the accel_ned injected (confirm it is pure forward; ~0 during warmup/cooldown).
And Fengyou eyeballs: physical NOSE (up/down) + TRANSLATION (forward/back) during the INJECT window.

DECISION TREE (the script prints the inject-window summary; the BRANCH needs Fengyou's eyeball):
  A) Nose DOWN + moves FORWARD  -> forward chain CORRECT; the A10 blocker is REGIME/ACQUISITION, NOT a
     sign. (cmd_pitch should be < 0 and the physical nose follows it down.)
  B) Nose UP + moves BACKWARD   -> it's a sign:
     - cmd_pitch < 0 (controller DID command nose-down) but nose physically UP  => COMMAND-AXIS inversion
       => fix = flip body_rate_sign[1], vq2_case_c ONLY.
     - est_pitch disagrees with the physical nose (e.g. est nose-DOWN while physically nose-UP), or
       raw_gyro_y polarity is wrong vs the visible rotation                      => GYRO/ATTITUDE inversion
       => fix at gyro_sign / AHRS; do NOT also flip the command.

Bounded + sim => no risk. Do NOT trust position_ned (VQ2 pos=NO = estimator fiction); use it only for
the qualitative drift DIRECTION, which Fengyou confirms by eye.

Usage:
  python handoff/vq2-forward-pitch-sign-2026-06-30/forward_pitch_probe.py \
      [--deploy-profile vq2_case_c] [--accel 3.0] [--warmup-s 3.0] [--inject-s 2.0] [--rate 30]
"""
from __future__ import annotations
import sys, time, json, argparse, threading
from datetime import datetime
from pathlib import Path
import numpy as np

sys.path.insert(0, "src")
from racer.mavlink_client import MavlinkClient
from racer.contracts import Setpoint, Gate
from racer.deploy_profile import get_profile
from racer.navigator import Navigator
from racer.gate_seeker import GateSeeker, GateSeekerConfig
from racer.recording import Recorder
from racer.vision.jpeg_receiver import VIDEO_PORT, JpegUdpReceiver


def _deg(x: float) -> float:
    return float(np.degrees(x)) if x is not None else float("nan")


def build_seeker(profile, settle_s: float):
    """The deployed case-C nav + seeker, NO detector (this probe never feeds a frame, so the
    seeker's settle/anchor regime holds a level hover the whole warmup). The seeker's controller
    IS make_seeker_controller() with the flight-proven SEEKER_SIGNS (body_rate_sign=[1,1,-1], etc.)."""
    # A single dummy gate satisfies Navigator construction; with frame=None the vision/gate chain
    # never fires, so it is pure AHRS dead-reckoning (the attitude estimate we want).
    dummy = Gate(gate_id=0, position_ned=np.array([100.0, 0.0, -2.0]), R_world_gate=np.eye(3))
    nav = Navigator(gates=[dummy], detector=None, config=profile.nav_config)
    seeker = GateSeeker(config=GateSeekerConfig(settle_s=settle_s, anchor_release_detections=10**9),
                        detector=None)
    return nav, seeker


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="udp:127.0.0.1:14550")
    ap.add_argument("--deploy-profile", default="vq2_case_c")
    ap.add_argument("--accel", type=float, default=8.0,
                    help="forward horizontal accel demand magnitude (m/s^2). 8 (=max_accel cap) -> ~39deg "
                         "nose-down target, well clear of the ~-17deg resting attitude so the pitch CMD "
                         "is unambiguously nose-down (a smaller 3 m/s^2 demand NULLS against the rest tilt).")
    ap.add_argument("--warmup-s", type=float, default=3.0, help="seeker settle/anchor HOLD before inject")
    ap.add_argument("--inject-s", type=float, default=3.0, help="constant forward-accel inject duration")
    ap.add_argument("--cooldown-s", type=float, default=1.0, help="level hold after inject before disarm")
    ap.add_argument("--rate", type=float, default=30.0, help="command loop rate (Hz)")
    ap.add_argument("--out", default="handoff/vq2-forward-pitch-sign-2026-06-30")
    ap.add_argument("--video-port", type=int, default=VIDEO_PORT)
    ap.add_argument("--no-record", action="store_true", help="skip onboard-video recording")
    ap.add_argument("--label", default="vq2_fwd_pitch_probe")
    a = ap.parse_args()

    # -- onboard-video recorder (so the run is reviewable as an mp4 via render_vision_video.py) --
    rec = None
    session_dir = None
    stop_video = threading.Event()
    vthread = None
    if not a.no_record:
        session_dir = Path("data/runs") / f"{datetime.now():%Y%m%d_%H%M%S}_{a.label}"
        rec = Recorder(session_dir); rec.start()

        def _video():
            while not stop_video.is_set():
                try:
                    with JpegUdpReceiver(port=a.video_port) as rx:
                        for fr in rx.frames(max_wait_s=5.0):
                            rec.record_frame(fr)
                            if stop_video.is_set():
                                break
                except Exception:
                    pass
                if not stop_video.is_set():
                    time.sleep(0.3)
        vthread = threading.Thread(target=_video, name="video", daemon=True)
        vthread.start()
        print(f"recording onboard video -> {session_dir}")

    profile = get_profile(a.deploy_profile)
    gyro_sign = np.asarray(profile.gyro_sign, dtype=np.float64)
    client = MavlinkClient(a.endpoint, cmd_rate_scale=profile.cmd_rate_scale, gyro_sign=tuple(gyro_sign))
    print(f"profile={profile.name}  cmd_rate_scale={client.cmd_rate_scale:g}  gyro_sign={tuple(gyro_sign)}")
    client.connect(wait_heartbeat=False, timeout_s=15.0)

    print("waiting for a live race GO (RACE_STATUS started + sim clock advancing) ...")
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        client.pump()
        rs = client.race_status
        if rs and rs.get("started") and client.state.sim_time_ns > 0:
            break
        time.sleep(0.02)
    else:
        print("FAIL: no live race within 120 s"); return 1
    print(f"race live (sim_t={client.state.sim_time_ns/1e9:.2f}s) -> ARM")
    client.arm()
    t0 = time.monotonic()
    while time.monotonic() - t0 < 0.5:
        client.pump(); time.sleep(0.01)
    print(f"  arm ack: {client.last_command_ack}")

    nav, seeker = build_seeker(profile, settle_s=a.warmup_s)

    log: list[dict] = []
    dt = 1.0 / a.rate
    tstart = time.monotonic()
    t_warm_end = a.warmup_s
    t_inj_end = a.warmup_s + a.inject_s
    t_end = t_inj_end + a.cooldown_s

    while True:
        now = time.monotonic() - tstart
        if now >= t_end:
            break
        client.pump()
        s = client.state
        nav_state = nav.update(s, None)            # frame=None -> pure AHRS attitude, no vision

        if now < t_warm_end:
            phase = "warmup"
            cmd = seeker.command_visual(nav_state, None, 0, is_final_gate=False)
            accel_demand = np.zeros(3)
        elif now < t_inj_end:
            phase = "inject"
            los = np.array([np.cos(nav_state.yaw), np.sin(nav_state.yaw), 0.0])  # forward, world NED
            accel_demand = a.accel * los
            sp = Setpoint(sim_time_ns=int(s.sim_time_ns), accel_ned=accel_demand)
            cmd = seeker.controller.command(nav_state, sp)
        else:
            phase = "cooldown"
            cmd = seeker.command_visual(nav_state, None, 0, is_final_gate=False)
            accel_demand = np.zeros(3)

        client.send_command(cmd)

        br = cmd.body_rate if cmd.body_rate is not None else np.full(3, np.nan)
        cmd_pitch_wire = float(br[1]) * client.cmd_rate_scale          # post sign + uplink scale
        gb = s.gyro_body
        gy_corr = float(gb[1]) if gb is not None else float("nan")     # what the AHRS consumes
        raw_gyro_y = (gy_corr / gyro_sign[1]) if (gb is not None and gyro_sign[1] != 0) else float("nan")
        pos = s.position_ned
        log.append({
            "t": round(now, 4), "sim_ns": int(s.sim_time_ns), "phase": phase,
            "cmd_pitch_wire": round(cmd_pitch_wire, 4),
            "cmd_body_rate": [round(float(x), 4) for x in br],
            "thrust": round(float(cmd.thrust), 4) if cmd.thrust is not None else None,
            "raw_gyro_y": round(raw_gyro_y, 4),
            "gyro_y_corrected": round(gy_corr, 4),
            "est_roll_deg": round(_deg(nav_state.roll), 2),
            "est_pitch_deg": round(_deg(nav_state.pitch), 2),
            "est_yaw_deg": round(_deg(nav_state.yaw), 2),
            "accel_demand": [round(float(x), 3) for x in accel_demand],
            "dr_pos": ([round(float(x), 2) for x in pos] if pos is not None else None),
        })
        time.sleep(dt)

    try:
        client.disarm()
    except Exception:
        pass

    # stop + flush the video recorder
    if rec is not None:
        stop_video.set()
        if vthread is not None:
            vthread.join(timeout=2.0)
        try:
            rec.close()
        except Exception:
            pass
        print(f"recorded {rec.n_frames} frames -> {session_dir}")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "probe_log.json").write_text(json.dumps(log, indent=1))

    inj = [r for r in log if r["phase"] == "inject"]
    print("\n=== FORWARD-PITCH-SIGN PROBE ===")
    print(f"injected forward accel = +{a.accel:.2f} m/s^2 along heading for {a.inject_s:.1f}s "
          f"(pure forward: accel_demand[2]==0)")
    # downsampled ~6 Hz table
    print(f"\n{'t':>5} {'phase':>8} {'cmd_pitch':>9} {'raw_gyro_y':>10} {'gyro_y_cor':>10} "
          f"{'est_pitch':>9} {'est_yaw':>8} {'dr_x':>8} {'thr':>5}")
    step = max(1, int(round(a.rate / 6.0)))
    for r in log[::step]:
        drx = (r["dr_pos"][0] if r["dr_pos"] else float('nan'))
        print(f"{r['t']:>5.1f} {r['phase']:>8} {r['cmd_pitch_wire']:>+9.3f} {r['raw_gyro_y']:>+10.3f} "
              f"{r['gyro_y_corrected']:>+10.3f} {r['est_pitch_deg']:>+9.1f} {r['est_yaw_deg']:>+8.1f} "
              f"{drx:>+8.1f} {(r['thrust'] if r['thrust'] is not None else float('nan')):>5.2f}")

    if inj:
        cp = np.array([r["cmd_pitch_wire"] for r in inj])
        ep = np.array([r["est_pitch_deg"] for r in inj])
        rg = np.array([r["raw_gyro_y"] for r in inj])
        d_est = ep[-1] - ep[0]
        print("\n--- INJECT-WINDOW SUMMARY (combine with Fengyou's eyeball) ---")
        print(f"  cmd_pitch (wire) : mean {cp.mean():+.3f}  -> controller commands "
              f"{'NOSE-DOWN (neg)' if cp.mean() < 0 else 'NOSE-UP (pos)'}")
        print(f"  est_pitch (deg)  : {ep[0]:+.1f} -> {ep[-1]:+.1f}  (delta {d_est:+.1f}deg) -> estimator "
              f"believes nose went {'UP' if d_est > 0 else 'DOWN'}")
        print(f"  raw_gyro_y       : mean {np.nanmean(rg):+.3f} rad/s (estimator-independent realized "
              f"pitch rate, pre-sign)")
        print("\n  BRANCH = function of (cmd_pitch sign, est_pitch trend, PHYSICAL nose Fengyou sees):")
        print("   * physical nose DOWN + moves FWD                       -> A  (chain correct; regime bug)")
        print("   * physical nose UP  + cmd_pitch < 0 (cmd was nose-down) -> B-command (flip body_rate_sign[1])")
        print("   * physical nose UP  + est_pitch disagrees w/ physical   -> B-gyro    (fix gyro_sign/AHRS)")
    print(f"\nwrote {out/'probe_log.json'}  ({len(log)} ticks)")
    if session_dir is not None:
        print(f"onboard video session: {session_dir}")
        print(f"  render mp4: python scripts/render_vision_video.py {session_dir} "
              f"{session_dir}/onboard.mp4 --slowmo 2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
