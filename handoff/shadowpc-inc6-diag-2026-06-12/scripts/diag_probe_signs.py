"""Open-loop command->response sign check from the mixer_probe2 recording.

commands.jsonl carries every SET_ATTITUDE_TARGET we sent (rate_sysid profile mode).
The tlog carries the ODOMETRY responses. For each probe phase window, print the
commanded body rate vs the realized TRUE rates (raw-quat convention, w_raw*[1,-1,1])
and the raw-quat euler attitude evolution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from racer.frames import euler_from_quat_wxyz

RUN = ROOT / "data" / "runs" / "20260612_034154_mixer_probe2"

# --- commands ---
cmds = []
with open(RUN / "commands.jsonl", encoding="utf-8") as f:
    for line in f:
        c = json.loads(line)
        cmds.append(c)
print("command record keys:", sorted(cmds[0].keys()))
print("first 2:", json.dumps(cmds[0])[:300]); print(json.dumps(cmds[1])[:300])

# group phases by 'phase' field if present
phases = {}
for c in cmds:
    ph = c.get("phase", "?")
    phases.setdefault(ph, []).append(c)
print("\nphases:", {k: len(v) for k, v in phases.items()})

# --- odometry from tlog ---
from pymavlink import mavutil
odo = []
conn = mavutil.mavlink_connection(str(RUN / "mavlink.tlog"))
while True:
    m = conn.recv_match(type="ODOMETRY", blocking=False)
    if m is None:
        break
    odo.append({"t": float(m._timestamp),
                "q": np.array([float(v) for v in m.q]),
                "w": np.array([m.rollspeed, m.pitchspeed, m.yawspeed])})
conn.close()
print(f"odo: {len(odo)} samples, t {odo[0]['t']:.1f}..{odo[-1]['t']:.1f}")

ot = np.array([o["t"] for o in odo])

for ph, cl in phases.items():
    ts = [c.get("t", c.get("t_mono", 0)) for c in cl]
    t0, t1 = min(ts), max(ts)
    # commanded rates
    br = np.array([c.get("body_rate", c.get("rate", [0, 0, 0])) for c in cl], dtype=np.float64)
    thr = np.array([c.get("thrust", c.get("collective", 0.0)) for c in cl], dtype=np.float64)
    sel = (ot >= t0) & (ot <= t1)
    if sel.sum() < 5:
        continue
    sub = [o for o, s in zip(odo, sel) if s]
    w_true = np.array([o["w"] * np.array([1.0, -1.0, 1.0]) for o in sub])
    rpy = np.degrees([euler_from_quat_wxyz(o["q"]) for o in sub])
    print(f"\n-- phase {ph}: {len(cl)} cmds over {t1-t0:.2f}s, cmd rate mean "
          f"{np.round(br.mean(0),2).tolist()} thr {thr.mean():.2f}")
    print(f"   realized TRUE rates  mean {np.round(w_true.mean(0),2).tolist()}  "
          f"last {np.round(w_true[-1],2).tolist()}")
    print(f"   raw-quat rpy deg: first {np.round(rpy[0],1).tolist()} "
          f"mid {np.round(rpy[len(rpy)//2],1).tolist()} last {np.round(rpy[-1],1).tolist()}")
