"""VQ2 gyro-frame SIGN probe (estimator-independent) — 2026-06-30.

Goal: determine, for EACH body axis, the sign of the live VQ2 HIGHRES_IMU gyro vs the COMMANDED
body-rate, to know if the A9 pitch inversion is y-only or a full FRD<->FLU handedness flip (which
would also invert z/yaw).

Method (no estimator, no stable flight needed):
  1. Reuse the proven flying path (racer.mavlink_client.MavlinkClient, the same ARM + SET_ATTITUDE_TARGET
     body-rate uplink that flew A9). cmd_rate_scale=1.0 so the COMMANDED sign is raw/unscaled.
  2. Wait for a live race (RACE_STATUS started + sim clock advancing), then ARM (MAV_CMD 400 p1=1).
  3. Stream BODY_RATE (type_mask 0b10000000) single-axis pulses, ~RATE rad/s for PULSE_S, returning to
     zero between: +pitch, +roll, +yaw. Keep thrust constant. We only need the FIRST ~0.2 s of each
     pulse's gyro response, before any tumble/coupling dominates.
  4. Log raw HIGHRES_IMU x/y/zgyro (state.gyro_body) at full rate throughout.
  5. Per pulse: commanded axis+sign vs the SIGN of the realized raw gyro on that axis (mean over the
     pulse's first WINDOW_S). Pitch should reproduce A9 (cmd +, gyro - => INVERTED) as a sanity check;
     roll & yaw are the new info.

The code ASSUMES HIGHRES_IMU gyro is "TRUE FRD, no sign change" (mavlink_client.py:249). This probe tests
that assumption directly. Do NOT trust position_ned (VQ2 returns pos=NO = estimator fiction).

Usage: python gyro_frame_probe.py [--rate 1.0] [--thrust 0.30] [--out <dir>]
"""
from __future__ import annotations
import sys, time, json, argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, "src")
from racer.mavlink_client import MavlinkClient
from racer.contracts import ControlCommand, ControlMode

ap = argparse.ArgumentParser()
ap.add_argument("--endpoint", default="udp:127.0.0.1:14550")
ap.add_argument("--rate", type=float, default=1.0, help="commanded single-axis rate magnitude (rad/s, raw)")
ap.add_argument("--thrust", type=float, default=0.30, help="constant collective during the probe")
ap.add_argument("--pulse-s", type=float, default=0.40)
ap.add_argument("--settle-s", type=float, default=0.40)
ap.add_argument("--window-s", type=float, default=0.20, help="averaging window from each pulse start")
ap.add_argument("--hz", type=float, default=100.0)
ap.add_argument("--out", default="handoff/vq2-gyro-sign-probe-2026-06-30")
a = ap.parse_args()

# axis index in the FRD body_rate vector and the matching gyro component: 0=roll(x),1=pitch(y),2=yaw(z)
AXES = {"roll": 0, "pitch": 1, "yaw": 2}
PULSES = [("pitch", +1), ("roll", +1), ("yaw", +1)]   # +pitch first = the A9 sanity check

client = MavlinkClient(a.endpoint, cmd_rate_scale=1.0)   # raw command sign (no 0.4/2.5x scaling)
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
    print("FAIL: no live race within 120 s"); sys.exit(1)
print(f"race live (sim_t={client.state.sim_time_ns/1e9:.2f}s) -> ARM")
client.arm()
t0 = time.monotonic()
while time.monotonic() - t0 < 0.5:   # let the ARM ack land
    client.pump(); time.sleep(0.01)
print(f"  arm ack: {client.last_command_ack}")

log = []   # [t_rel, sim_ns, xgyro, ygyro, zgyro, cmd_roll, cmd_pitch, cmd_yaw, label]
tstart = time.monotonic()
dt = 1.0 / a.hz

def phase(roll, pitch, yaw, label, dur):
    end = time.monotonic() + dur
    while time.monotonic() < end:
        client.pump()
        client.send_command(ControlCommand(mode=ControlMode.BODY_RATE,
            body_rate=np.array([roll, pitch, yaw], float), thrust=a.thrust))
        g = client.state.gyro_body
        log.append([round(time.monotonic() - tstart, 4), int(client.state.sim_time_ns),
                    float(g[0]) if g is not None else np.nan,
                    float(g[1]) if g is not None else np.nan,
                    float(g[2]) if g is not None else np.nan,
                    roll, pitch, yaw, label])
        time.sleep(dt)

phase(0, 0, 0, "settle", 0.5)
for name, sign in PULSES:
    r = [0.0, 0.0, 0.0]; r[AXES[name]] = sign * a.rate
    phase(r[0], r[1], r[2], f"+{name}", a.pulse_s)
    phase(0, 0, 0, f"zero_after_{name}", a.settle_s)

# best-effort disarm (sim resets on relaunch anyway)
try:
    import pymavlink.mavutil as mavutil
    client.conn.mav.command_long_send(client.conn.target_system, client.conn.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 0, 0, 0, 0, 0, 0, 0)
except Exception:
    pass

L = np.array([row[:8] for row in log], float)   # numeric cols: 0=t 1=sim_ns 2=xg 3=yg 4=zg 5=cr 6=cp 7=cy
labels = [row[8] for row in log]
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
(out / "probe_log.json").write_text(json.dumps(
    [{"t": r[0], "sim_ns": r[1], "xgyro": r[2], "ygyro": r[3], "zgyro": r[4],
      "cmd": [r[5], r[6], r[7]], "label": r[8]} for r in log], indent=1))

print("\n=== GYRO-FRAME SIGN PROBE RESULT ===")
print(f"commanded rate magnitude = {a.rate:+.2f} rad/s (raw, cmd_rate_scale=1.0); thrust={a.thrust}")
print(f"{'axis':>6} | {'cmd sign':>8} | {'realized raw-gyro sign':>22} | {'mean gyro (rad/s)':>17} | INVERTED?")
print("-" * 78)
rows_out = []
for name, sign in PULSES:
    ax = AXES[name]
    # rows in this pulse's window [pulse_start, pulse_start+window_s]
    idx = [i for i, lb in enumerate(labels) if lb == f"+{name}"]
    if not idx:
        print(f"{name:>6} | (no samples)"); continue
    t_start = L[idx[0], 0]
    win = [i for i in idx if L[i, 0] <= t_start + a.window_s]
    gax = L[win, 2 + ax]                       # realized gyro on the commanded axis
    gax = gax[np.isfinite(gax)]
    mean_g = float(np.mean(gax)) if len(gax) else float("nan")
    cmd_s = "+" if sign > 0 else "-"
    real_s = "+" if mean_g > 0 else "-"
    inv = "YES (inverted)" if (sign > 0) != (mean_g > 0) else "no"
    print(f"{name:>6} | {cmd_s:>8} | {real_s:>22} | {mean_g:>+17.2f} | {inv}")
    rows_out.append({"axis": name, "cmd_sign": cmd_s, "realized_gyro_sign": real_s,
                     "mean_gyro_rad_s": round(mean_g, 3), "inverted": inv, "n": len(gax)})
(out / "result_table.json").write_text(json.dumps(rows_out, indent=2))
print(f"\nwrote {out/'result_table.json'} and {out/'probe_log.json'}")
