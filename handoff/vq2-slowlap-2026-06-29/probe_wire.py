"""Quick live-wire probe for VQ2: confirm position-denied + race-live, sample over ~4 s."""
import sys, time
sys.path.insert(0, "src")
from racer.mavlink_client import MavlinkClient

c = MavlinkClient()
c.connect(wait_heartbeat=True, timeout_s=15.0)
print("connected.")
t0 = time.monotonic()
last_st = -1
rs_updates = 0
samples = 0
last_rs_repr = None
while time.monotonic() - t0 < 4.0:
    c.pump()
    s = c.state
    rs = c.race_status
    st = int(s.sim_time_ns)
    if st != last_st:
        last_st = st
        samples += 1
    rrepr = repr(rs)
    if rrepr != last_rs_repr:
        rs_updates += 1
        last_rs_repr = rrepr
    time.sleep(0.01)

s = c.state
rs = c.race_status
print("=== WIRE PROBE ===")
print("position_ned :", s.position_ned)
print("attitude_wxyz:", getattr(s, "attitude_wxyz", None))
print("sim_time_ns  :", int(s.sim_time_ns))
print("sim_time_s   :", int(s.sim_time_ns) / 1e9)
print("distinct sim_time samples in 4s:", samples)
print("distinct RACE_STATUS updates in 4s:", rs_updates)
print("RACE_STATUS  :", rs)
if rs:
    tg = rs["race_start_boot_time_ms"] - rs["sim_boot_time_ms"]
    print("to_go_s      :", tg / 1000.0)
    print("started      :", rs.get("started"))
    print("finished     :", rs.get("finished"))
    print("active_gate  :", rs.get("active_gate_index"))
print("track_gates (map on wire):", len(c.track_gates or []))
