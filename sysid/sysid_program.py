"""Deterministic open-loop command PROGRAM for plant-dynamics sysID (40 Hz, dt=0.025 s).

Writes ``sysid_program.csv`` with columns ``t, a_thrust, a_roll, a_pitch, a_yaw`` -- RAW policy
actions in [-1, 1] (the tanh-squashed network output, i.e. the exact input to DiffAero's
``rescale_action``). BOTH the DiffAero side (``sysid_diffaero_replay.py``) and the VQ2 side consume
THIS file. Also writes ``sysid_program_segments.json`` (segment windows for the response summary).

ACTION SEMANTICS (why these amplitudes are physically reasonable):
    rescale_action maps a in [-1,1] -> physical with bounds ACT_MIN=[0,-3.14,-3.14,-3.14],
    ACT_MAX=[5,3.14,3.14,3.14] (cfg/dynamics/quad.yaml controller block):
        normed_thrust = 2.5*(a_thrust+1)      in [0, 5]   (1 == hover)
        rate_flu      = 3.14 * a_rate         in [-3.14, 3.14] rad/s
    Hover (normed_thrust == 1) => a_thrust = -0.6.
    A rate action of 0.4 -> 1.256 rad/s; 0.6 -> 1.884 rad/s; chirp 0.4 -> +-1.256 rad/s (all << 3.14).
    Every excitation is amplitude-SYMMETRIC (equal +/- steps, doublets, zero-mean chirp) so the NET
    commanded rotation over each axis's battery ~= 0 -- the open-loop (rate-only, no attitude hold)
    drone returns near level between axes, keeping the later thrust battery ~level.

BATTERY per rate axis (roll, then pitch, then yaw -- yaw is the hunt axis, ordered last so its
    excitation is the freshest before analysis): step +0.4 (0.5 s) / 0 (1.0 s) / step -0.4 (0.5 s) /
    0 (1.0 s) / doublet +0.6,-0.6 (0.3 s each) / linear chirp 0.5->6 Hz over 3.0 s at amp 0.4 /
    1.0 s trim.
THRUST battery: +0.4 above hover (0.5 s) / hover (0.5 s) / -0.4 below hover (0.5 s) / hover (1.0 s).
"""
import csv
import json
import os

import numpy as np

DT = 0.025                     # 40 Hz
HOVER_A = -0.6                 # a_thrust that rescales to normed_thrust == 1 (hover)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(OUT_DIR, "sysid_program.csv")
SEG_PATH = os.path.join(OUT_DIR, "sysid_program_segments.json")

CHAN = {"thrust": 0, "roll": 1, "pitch": 2, "yaw": 3}


def _n(dur):
    return int(round(dur / DT))


def build():
    rows = []          # each row: [a_thrust, a_roll, a_pitch, a_yaw]
    segs = []          # metadata windows for the summary
    t = 0.0

    def append_const(dur, thrust=HOVER_A, roll=0.0, pitch=0.0, yaw=0.0, seg=None):
        nonlocal t
        n = _n(dur)
        if seg is not None:
            seg["t0"] = round(t, 6)
        for _ in range(n):
            rows.append([thrust, roll, pitch, yaw])
            t_local = round(len(rows) * DT, 6)  # time of the NEXT sample boundary
        if seg is not None:
            seg["t1"] = round(t + n * DT, 6)
            segs.append(seg)
        t = round(t + n * DT, 6)

    def append_chirp(dur, chan, amp, f0, f1, seg=None):
        nonlocal t
        n = _n(dur)
        if seg is not None:
            seg["t0"] = round(t, 6)
        idx = CHAN[chan]
        for k in range(n):
            tl = k * DT
            phase = 2.0 * np.pi * (f0 * tl + 0.5 * (f1 - f0) / dur * tl * tl)
            row = [HOVER_A, 0.0, 0.0, 0.0]
            row[idx] = amp * float(np.sin(phase))
            rows.append(row)
        if seg is not None:
            seg["t1"] = round(t + n * DT, 6)
            segs.append(seg)
        t = round(t + n * DT, 6)

    # 1. initial hover trim ------------------------------------------------------
    append_const(2.0, seg={"name": "hover_trim_init", "chan": "none", "kind": "trim"})

    # 2. rate-axis batteries -----------------------------------------------------
    for axis in ("roll", "pitch", "yaw"):
        kw_pos = {axis: 0.4}
        kw_neg = {axis: -0.4}
        kw_dp = {axis: 0.6}
        kw_dn = {axis: -0.6}
        append_const(0.5, seg={"name": f"{axis}_step_pos", "chan": axis, "action": 0.4,
                               "kind": "step"}, **kw_pos)
        append_const(1.0, seg={"name": f"{axis}_trim_a", "chan": axis, "kind": "trim"})
        append_const(0.5, seg={"name": f"{axis}_step_neg", "chan": axis, "action": -0.4,
                               "kind": "step"}, **kw_neg)
        append_const(1.0, seg={"name": f"{axis}_trim_b", "chan": axis, "kind": "trim"})
        append_const(0.3, seg={"name": f"{axis}_doublet_pos", "chan": axis, "action": 0.6,
                               "kind": "doublet"}, **kw_dp)
        append_const(0.3, seg={"name": f"{axis}_doublet_neg", "chan": axis, "action": -0.6,
                               "kind": "doublet"}, **kw_dn)
        append_chirp(3.0, axis, 0.4, 0.5, 6.0,
                     seg={"name": f"{axis}_chirp", "chan": axis, "amp": 0.4,
                          "f0": 0.5, "f1": 6.0, "kind": "chirp"})
        append_const(1.0, seg={"name": f"{axis}_trim_c", "chan": axis, "kind": "trim"})

    # 3. thrust battery ----------------------------------------------------------
    append_const(0.5, thrust=HOVER_A + 0.4,
                 seg={"name": "thrust_step_pos", "chan": "thrust", "action": HOVER_A + 0.4,
                      "kind": "thrust_step"})
    append_const(0.5, seg={"name": "thrust_hover_a", "chan": "thrust", "kind": "trim"})
    append_const(0.5, thrust=HOVER_A - 0.4,
                 seg={"name": "thrust_step_neg", "chan": "thrust", "action": HOVER_A - 0.4,
                      "kind": "thrust_step"})
    append_const(1.0, seg={"name": "thrust_hover_b", "chan": "thrust", "kind": "trim"})

    return rows, segs


def main():
    rows, segs = build()
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "a_thrust", "a_roll", "a_pitch", "a_yaw"])
        for k, r in enumerate(rows):
            w.writerow(["%.5f" % (k * DT)] + ["%.6f" % v for v in r])
    meta = {"dt": DT, "hz": round(1.0 / DT, 3), "n_steps": len(rows),
            "hover_a_thrust": HOVER_A,
            "act_min": [0.0, -3.14, -3.14, -3.14], "act_max": [5.0, 3.14, 3.14, 3.14],
            "segments": segs}
    with open(SEG_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print("wrote %s  (%d rows, %.3f s @ %.0f Hz)" % (CSV_PATH, len(rows), len(rows) * DT, 1 / DT))
    print("wrote %s  (%d segments)" % (SEG_PATH, len(segs)))


if __name__ == "__main__":
    main()
