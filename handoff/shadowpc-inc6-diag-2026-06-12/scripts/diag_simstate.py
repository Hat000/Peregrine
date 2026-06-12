"""Diagnostic: what state is the sim in, does 31000 do anything, do commands take effect?"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import numpy as np
from racer.mavlink_client import MavlinkClient
from racer.contracts import ControlCommand, ControlMode
from fly_vq1 import _arm_cmd

client = MavlinkClient("udp:127.0.0.1:14550")
client.connect(wait_heartbeat=False, timeout_s=15.0)

def pump(dur):
    end = time.monotonic() + dur
    while time.monotonic() < end:
        client.pump(); time.sleep(0.004)

pump(2.0)
s = client.state
rs = client.race_status
print("race_status:", rs)
print("sim_t:", s.sim_time_ns / 1e9, "pos:", None if s.position_ned is None else np.round(s.position_ned, 2))
t0 = s.sim_time_ns
pump(1.0)
print("sim_t advanced by:", (client.state.sim_time_ns - t0) / 1e9, "s in 1.0 wall s")
print("collisions:", client.collisions[-3:] if client.collisions else [])
print("gates:", len(client.track_gates or []))

print("\n-- sending 31000 and watching RACE_STATUS for 6 s --")
client.send_sim_reset()
for i in range(12):
    pump(0.5)
    rs = client.race_status
    to_go = (rs["race_start_boot_time_ms"] - rs["sim_boot_time_ms"]) / 1000.0 if rs else None
    print(f"  t+{(i+1)*0.5:.1f}s started={rs and rs['started']} to_go={to_go} "
          f"sim_t={client.state.sim_time_ns/1e9:.2f} ack={client.last_command_ack}")

print("\n-- arming + thrust 0.32 for 2 s, watching motors/alt --")
client.last_command_ack = None
client.arm()
client.wait_command_ack(_arm_cmd(), timeout_s=3.0)
armed = client.wait_armed(True, timeout_s=5.0)
print("armed:", armed)
for i in range(40):
    s = client.state
    cmd = ControlCommand(mode=ControlMode.BODY_RATE, sim_time_ns=int(s.sim_time_ns),
                         body_rate=np.zeros(3), thrust=0.32)
    client.send_command(cmd)
    pump(0.05)
    if i % 8 == 7:
        act = client.actuator_outputs
        print(f"  motors={None if act is None else [round(float(x),3) for x in act['motors']]} "
              f"z={client.state.position_ned[2]:+.2f} vz={client.state.velocity_ned[2]:+.2f}")
print("disarming")
client.disarm(force=True)
client.wait_armed(False, timeout_s=3.0)
