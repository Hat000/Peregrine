"""Peregrine flight-forensics core library (pure Python, no numpy).
Loads a session (obs+timing+meta), computes per-tick derived arrays in the
leveled frame, detects gate advances + storm starts, and exposes helpers.
Frames/conventions per the task schema block (trusted).
"""
import json, math, os

V15_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v15flights", "data", "runs")
HIST_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hist", "data", "runs")


def _norm(v):
    return math.sqrt(sum(x * x for x in v))


def leveled_z(vec, roll, pitch):
    """World-vertical component of a body-FLU vector given leveled roll/pitch.
    +z = up (world). vec=(x,y,z) body-FLU."""
    x, y, z = vec
    return -math.sin(pitch) * x + math.sin(roll) * math.cos(pitch) * y + math.cos(roll) * math.cos(pitch) * z


def load_session(sess, root=V15_ROOT):
    d = os.path.join(root, sess)
    meta = {}
    mp = os.path.join(d, "meta.json")
    if os.path.isfile(mp):
        try:
            meta = json.load(open(mp))
        except Exception:
            meta = {}
    rows = []
    op = os.path.join(d, "ego_obs.jsonl")
    if not os.path.isfile(op):
        return None
    with open(op) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    if not rows:
        return None
    # timing joined by k
    timing = {}
    tp = os.path.join(d, "ego_timing.jsonl")
    if os.path.isfile(tp):
        with open(tp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    t = json.loads(line)
                    timing[t.get("k")] = t
                except Exception:
                    pass

    t0 = rows[0]["sim_time_ns"]
    n = len(rows)
    S = {
        "sess": sess, "meta": meta, "n": n,
        "t": [], "gate_index": [], "obs": [], "rate_frd": [], "actor_mean": [],
        "speed": [], "vx": [], "vy": [], "vz_body": [], "vz_lvl": [],
        "roll": [], "pitch": [], "gx": [], "gy": [], "gz": [],
        "prev_thrust": [], "sector": [],
        "rel": [], "rng": [], "hrng": [], "dh": [],  # slot0
        "conf0": [], "area0": [], "age0": [], "seen0": [],
        "rel1": [], "conf1": [], "area1": [], "seen1": [],
        "rel_flu1": [],
        "normed_thrust": [], "collective": [], "assist": [], "kf": [],
        "fresh_detect": [], "detect_ms": [], "work_ms": [], "loop_dt": [],
    }
    prev_t = None
    for r in rows:
        tt = (r["sim_time_ns"] - t0) / 1e9
        S["t"].append(tt)
        S["gate_index"].append(r.get("gate_index", 0))
        o = r["obs"]
        S["obs"].append(o)
        S["rate_frd"].append(r.get("rate_frd", [0, 0, 0]))
        S["actor_mean"].append(r.get("actor_mean", [0, 0, 0, 0]))
        vel = o[0:3]
        roll, pitch = o[3], o[4]
        S["vx"].append(o[0]); S["vy"].append(o[1]); S["vz_body"].append(o[2])
        S["speed"].append(_norm(vel))
        S["vz_lvl"].append(leveled_z(vel, roll, pitch))
        S["roll"].append(roll); S["pitch"].append(pitch)
        S["gx"].append(o[5]); S["gy"].append(o[6]); S["gz"].append(o[7])
        S["prev_thrust"].append(o[8])
        S["sector"].append(tuple(r.get("sector") or o[9:11]))
        rel = o[11:14]
        S["rel"].append(rel)
        S["rng"].append(_norm(rel))
        S["hrng"].append(math.hypot(rel[0], rel[1]))
        S["dh"].append(leveled_z(rel, roll, pitch))
        S["conf0"].append(o[14]); S["area0"].append(o[15])
        S["age0"].append(r.get("age_s"))
        S["seen0"].append(1 if r.get("conf", 0.0) and r.get("conf", 0.0) > 1e-6 else 0)
        rel1 = o[16:19]
        S["rel1"].append(rel1)
        S["conf1"].append(o[19]); S["area1"].append(o[20])
        S["seen1"].append(1 if any(abs(x) > 1e-9 for x in o[16:21]) else 0)
        S["rel_flu1"].append(r.get("rel_flu1"))
        S["normed_thrust"].append(r.get("normed_thrust"))
        S["collective"].append(r.get("collective"))
        S["assist"].append(bool(r.get("assist")))
        S["kf"].append(r.get("kf_pos_ned") or [0, 0, 0])
        k = r.get("k")
        tm = timing.get(k, {})
        S["fresh_detect"].append(bool(tm.get("fresh_detect")) if tm else None)
        S["detect_ms"].append(tm.get("detect_ms"))
        S["work_ms"].append(tm.get("work_ms"))
        S["loop_dt"].append((tt - prev_t) if prev_t is not None else None)
        prev_t = tt
    S["dur"] = S["t"][-1]
    S["maxv"] = max(S["speed"])
    # gate advances: index where gate_index increments
    adv = []
    for i in range(1, n):
        if S["gate_index"][i] != S["gate_index"][i - 1]:
            adv.append((i, S["gate_index"][i - 1], S["gate_index"][i], S["t"][i]))
    S["adv"] = adv
    S["gates_passed"] = meta.get("gate_index", S["gate_index"][-1])
    # config tags
    S["rs"] = meta.get("ego_rate_scale")
    S["pclmp"] = meta.get("ego_pitch_clamp")
    S["map"] = os.path.basename(str(meta.get("ego_coarse_map", "")))
    S["det_hold"] = meta.get("ego_det_hold")
    S["stale_h"] = meta.get("ego_stale_horizon")
    S["coll"] = meta.get("collisions")
    S["state"] = meta.get("final_state")
    ck = os.path.basename(str(meta.get("ego_ckpt", "")))
    S["ckpt"] = ck
    if "Qs1" in ck:
        S["fam"] = "Qs1"
    elif "Ws0" in ck:
        S["fam"] = "Ws0"
    elif "vpeffs0" in ck:
        S["fam"] = "champion"
    elif "v1A" in ck:
        S["fam"] = "v1A"
    elif "v1R" in ck:
        S["fam"] = "v1R"
    elif "trackA" in ck:
        S["fam"] = "trackA"
    elif "vn16" in ck:
        S["fam"] = "vn16"
    elif "vpef8nc" in ck:
        S["fam"] = "vpef8nc"
    else:
        S["fam"] = ck.split("_")[0]
    # storm detection
    S["storm"] = detect_storm(S)
    return S


def detect_storm(S):
    """Storm/unsettled start = procedural spawn thrash, not policy behaviour.
    Signature: within the first ~2.5 s the drone shows a burst of high body-rate /
    high-speed thrash without net forward progress, OR garbage tick-0 geometry.
    Returns dict with flag + evidence."""
    t = S["t"]; n = S["n"]
    # ticks within first 2.5 s
    early = [i for i in range(n) if t[i] <= 2.5]
    if not early:
        early = list(range(min(20, n)))
    # gyro magnitude burst
    gyro_mag = [math.sqrt(S["gx"][i] ** 2 + S["gy"][i] ** 2 + S["gz"][i] ** 2) for i in early]
    max_gyro = max(gyro_mag) if gyro_mag else 0.0
    max_spd_early = max(S["speed"][i] for i in early) if early else 0.0
    # net forward progress over first 3 s from kf position
    late3 = [i for i in range(n) if t[i] <= 3.0]
    if late3:
        j = late3[-1]
        kf0 = S["kf"][0]; kfj = S["kf"][j]
        net = math.hypot(kfj[0] - kf0[0], kfj[1] - kf0[1])
    else:
        net = 0.0
    # tick-0 geometry sanity
    o0 = S["obs"][0]
    garbage0 = (_norm(o0[11:14]) > 60) or (_norm(o0[0:3]) > 3.0) or any(math.isnan(x) if isinstance(x, float) else False for x in o0)
    coll = S["coll"] or 0
    gates = S["gates_passed"] or 0
    # heuristic flag. Storm = spawn contact-thrash (procedural), matches prompt "coll 57-200".
    flag = False
    reason = []
    if garbage0:
        flag = True; reason.append("garbage_tick0")
    if net < 3.0 and coll >= 15:
        flag = True; reason.append("stuck_lowprogress")
    if coll >= 50:
        flag = True; reason.append("coll>=50")
    return {"flag": flag, "reason": reason, "max_gyro_early": round(max_gyro, 2),
            "max_spd_early": round(max_spd_early, 2), "net_prog3s": round(net, 2),
            "coll": coll, "gates": gates, "garbage0": garbage0}


# ---------- statistics helpers (pure python) ----------
def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    m = len(xs)
    return xs[m // 2] if m % 2 else 0.5 * (xs[m // 2 - 1] + xs[m // 2])


def quantile(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def all_sessions(root=V15_ROOT):
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


if __name__ == "__main__":
    # smoke test
    S = load_session("20260719_074221_v15pick_Ws0_f1")
    print("sess", S["sess"], "fam", S["fam"], "gates", S["gates_passed"], "dur", round(S["dur"], 2),
          "maxv", round(S["maxv"], 2), "storm", S["storm"]["flag"])
    print("advances:", [(a[2], round(a[3], 2)) for a in S["adv"]])
    print("last tick: hrng", round(S["hrng"][-1], 2), "dh", round(S["dh"][-1], 2),
          "speed", round(S["speed"][-1], 2), "vz_lvl", round(S["vz_lvl"][-1], 2))
