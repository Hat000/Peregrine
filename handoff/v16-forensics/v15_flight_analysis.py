"""Per-session obs-level analysis of the v15 pick flights: slot1 duty, sector trace,
post-pass blackouts, pitch-cmd oscillation near gates, end-state post-mortems."""
import json
import math
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v15flights", "data", "runs")


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def analyze(sess):
    d = os.path.join(ROOT, sess)
    try:
        meta = json.load(open(os.path.join(d, "meta.json")))
    except Exception:
        meta = {}
    rows = []
    with open(os.path.join(d, "ego_obs.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    if not rows:
        return None
    t0 = rows[0]["sim_time_ns"]
    T = [(r["sim_time_ns"] - t0) / 1e9 for r in rows]
    dur = T[-1] if T else 0.0
    gi = [r.get("gate_index", 0) for r in rows]
    obs = [r["obs"] for r in rows]
    pitch = [r["rate_frd"][1] for r in rows]
    yaw = [r["rate_frd"][2] for r in rows]
    conf0 = [r.get("conf", 0.0) for r in rows]
    seen1 = [1 if any(abs(x) > 1e-9 for x in r["obs"][16:21]) else 0 for r in rows]
    maxv = max(norm(o[0:3]) for o in obs)

    def flips_per_s(sig, dead=0.05):
        s_prev, n = 0, 0
        for x in sig:
            s = 1 if x > dead else (-1 if x < -dead else 0)
            if s != 0 and s_prev != 0 and s != s_prev:
                n += 1
            if s != 0:
                s_prev = s
        return n / max(dur, 1e-9)

    # gate advances
    adv = [(i, gi[i]) for i in range(1, len(gi)) if gi[i] != gi[i - 1]]
    # sector per gate index
    sec = {}
    for r, g in zip(rows, gi):
        s = tuple(r.get("sector") or r["obs"][9:11])
        sec.setdefault(g, set()).add(s)
    # post-advance slot0 blackout + slot1 coverage during it
    blk = []
    for i, g in adv:
        j = i
        while j < len(rows) and conf0[j] <= 1e-6:
            j += 1
        dt_blind = (T[min(j, len(T) - 1)] - T[i]) if j > i else 0.0
        s1_duty = (sum(seen1[i:j]) / max(j - i, 1)) if j > i else 1.0
        blk.append((g, round(T[i], 2), round(dt_blind, 2), round(s1_duty, 2)))
    # near-gate pitch windows: 1 s before each advance
    near_flips, near_p2p, cruise_sig = [], [], []
    near_idx = set()
    for i, _ in adv:
        i0 = max(0, i - 40)
        seg = pitch[i0:i]
        near_idx.update(range(i0, i))
        if len(seg) > 5:
            f = 0
            sp = 0
            for x in seg:
                s = 1 if x > 0.05 else (-1 if x < -0.05 else 0)
                if s != 0 and sp != 0 and s != sp:
                    f += 1
                if s != 0:
                    sp = s
            near_flips.append(f)  # flips within the 1 s window
            near_p2p.append(round(max(seg) - min(seg), 2))
    cruise_sig = [p for i, p in enumerate(pitch) if i not in near_idx]
    def sflip_count(seg):
        f, sp = 0, 0
        for x in seg:
            s = 1 if x > 0.05 else (-1 if x < -0.05 else 0)
            if s != 0 and sp != 0 and s != sp:
                f += 1
            if s != 0:
                sp = s
        return f
    cruise_fps = sflip_count(cruise_sig) / max(len(cruise_sig) / 40.0, 1e-9)
    # end state
    tail = rows[-1]
    end = {
        "gate": gi[-1], "t": round(dur, 2), "conf0": tail.get("conf"),
        "rel0": [round(x, 1) for x in tail["obs"][11:14]],
        "slot1": round(sum(seen1[-40:]) / max(min(40, len(seen1)), 1), 2),
        "vz_body": round(tail["obs"][2], 2), "z_ned": round((tail.get("kf_pos_ned") or [0, 0, 0])[2], 2),
    }
    # slot1 duty overall + while slot0 blind
    blind_idx = [i for i in range(len(rows)) if conf0[i] <= 1e-6]
    s1_blind = round(sum(seen1[i] for i in blind_idx) / max(len(blind_idx), 1), 2)
    return {
        "sess": sess[16:], "state": meta.get("final_state"), "gates": meta.get("gate_index"),
        "coll": meta.get("collisions"), "dur": round(dur, 1), "maxv": round(maxv, 1),
        "yaw_absmean": round(sum(abs(y) for y in yaw) / len(yaw), 3),
        "yaw_fps": round(flips_per_s(yaw), 2),
        "pitch_absmean": round(sum(abs(p) for p in pitch) / len(pitch), 3),
        "pitch_fps_cruise": round(cruise_fps, 2),
        "near_gate_pitch": {"flips_per_window": near_flips, "p2p": near_p2p},
        "slot1_duty": round(sum(seen1) / len(seen1), 2), "slot1_when_blind": s1_blind,
        "slot0_blind_duty": round(len(blind_idx) / len(rows), 2),
        "advances": blk, "sector_by_gate": {g: sorted(v) for g, v in sorted(sec.items())},
        "end": end,
    }


def main():
    sessions = sorted(os.listdir(ROOT))
    for s in sessions:
        r = analyze(s)
        if r is None:
            print(f"{s}: EMPTY")
            continue
        print(json.dumps(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
