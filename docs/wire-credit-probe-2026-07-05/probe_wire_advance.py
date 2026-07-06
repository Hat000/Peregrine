"""Measure WHERE RACE_STATUS.active_gate_index advances relative to the physical gate plane.

For every recording with (a) an active_gate_index transition and (b) a position stream,
interpolate position (recv-time base) at each transition and express it in the target
gate's frame: depth<0 = BEFORE the plane (early credit). RACE_STATUS staleness only biases
depth POSITIVE (late detection), so negative depth is a hard 'early' verdict.
"""
import glob, os, struct, sys
import numpy as np
from pymavlink import mavutil

sys.path.insert(0, "C:/Users/Shadow/Peregrine/rl")
from fly_rl import _FLIP, _GATE_POS_ZUP, _R_W2G, N_GATES  # noqa: E402

def probe(tlog):
    m = mavutil.mavlink_connection(tlog)
    rs, pos = [], []
    while True:
        msg = m.recv_match(type=["ENCAPSULATED_DATA", "LOCAL_POSITION_NED", "ODOMETRY"], blocking=False)
        if msg is None: break
        t = float(getattr(msg, "_timestamp", 0.0))
        k = msg.get_type()
        if k == "ENCAPSULATED_DATA":
            raw = bytes(msg.data)
            if raw and raw[0] == 1 and len(raw) >= struct.calcsize("<BQqqIq"):
                _, sim_boot, race_start, _, agi, _ = struct.unpack_from("<BQqqIq", raw)
                rs.append((t, int(agi), int(race_start)))
        elif k == "LOCAL_POSITION_NED":
            pos.append((t, msg.x, msg.y, msg.z))
        else:  # ODOMETRY
            pos.append((t, msg.x, msg.y, msg.z))
    if not rs or len(pos) < 10:
        return None
    pos = np.array(pos); out = []
    for i in range(1, len(rs)):
        t1, g1, s1 = rs[i]; t0, g0, s0 = rs[i - 1]
        if s1 != s0 or g1 <= g0 or g1 > N_GATES:  # same race, forward transition
            continue
        gate = g0  # the gate just credited
        j = np.searchsorted(pos[:, 0], t1)
        if j == 0 or j >= len(pos):
            continue
        ta, tb = pos[j - 1, 0], pos[j, 0]
        w = 0.0 if tb == ta else (t1 - ta) / (tb - ta)
        p = pos[j - 1, 1:4] * (1 - w) + pos[j, 1:4] * w
        rel = _R_W2G @ (np.asarray(p) * _FLIP - _GATE_POS_ZUP[gate])
        # depth along exit axis (gate frame x); linf in-plane
        out.append((gate, float(rel[0]), float(max(abs(rel[1]), abs(rel[2]))),
                    float(t1 - rs[i - 1][0]), float(tb - ta)))
    return out, len(rs), len(pos)

roots = ["C:/Users/Shadow/Peregrine/data/runs/2026*",
         "C:/Users/Shadow/Peregrine/.claude/worktrees/*/data/runs/2026*"]
for pat in roots:
    for run in sorted(glob.glob(pat)):
        tl = os.path.join(run, "mavlink.tlog")
        if not os.path.exists(tl): continue
        try:
            r = probe(tl)
        except Exception as e:
            print(f"{os.path.basename(run)}: ERROR {type(e).__name__}: {e}"); continue
        if r is None: continue
        trans, n_rs, n_pos = r
        if not trans: continue
        print(f"{os.path.basename(run)} (rs={n_rs}, pos={n_pos}):")
        for gate, depth, linf, rs_gap, pos_gap in trans:
            print(f"   gate {gate} credited at depth={depth:+7.2f} m  linf={linf:5.2f} m"
                  f"  (rs_gap={rs_gap*1e3:4.0f} ms, pos_gap={pos_gap*1e3:4.0f} ms)")
