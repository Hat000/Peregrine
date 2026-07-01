"""VQ2 Vision Step-1: red_glow vs YOLO-ensemble RECALL on real live-sim frames — 2026-06-30.

The live YOLO-in-the-loop run is BLOCKED by an arg-wiring bug (fly_rl --checkpoint feeds BOTH the RL
actor load AND the gate-seeker's GateDetector.load, with no separate flag; a "modelA++modelB" spec fails
load_actor). So this measures the CORE question offline instead: on the SAME recorded live-sim frames the
seeker actually saw, does the trained YOLO drop valid_empty vs red_glow's live 74%?

valid_empty = fraction of frames where the detector -> estimate_gate_pose yields NO finite-range pose
(exactly what the seeker's [seeker-diag] counts). Running both detectors on identical frames isolates
DETECTOR RECALL from flight dynamics.

Usage: python detector_recall_compare.py [<run_dir>]   (default: the A13 red_glow baseline run)
"""
import sys, json, time
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, "src")
from racer.contracts import Frame
from racer.vision.red_glow_detector import RedGlowGateDetector
from racer.vision.detector import GateDetector
from racer.vision.gate_pose import estimate_gate_pose

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/runs/20260630_225052_vq2_slow_seeker_a13_f1")
ENS = "models/gate_clean_ens_course_L110.pt++models/gate_clean_ens_precision_L107.pt"
SINGLE = "models/gate_clean_ens_course_L110.pt"

idx = [json.loads(l) for l in (RUN / "video_index.jsonl").read_text().splitlines() if l.strip()]
blob = (RUN / "video.bin").read_bytes()


def valid_pose(obs_list) -> bool:
    for o in obs_list:
        try:
            ps = estimate_gate_pose(o, compute_covariance=False)
            if ps and np.isfinite(ps.range_m):
                return True
        except Exception:
            pass
    return False


def frame(i):
    r = idx[i]
    img = cv2.imdecode(np.frombuffer(blob[r["offset"]:r["offset"] + r["length"]], np.uint8), cv2.IMREAD_COLOR)
    return img, Frame(r["frame_id"], r["sim_time_ns"], img, 0)


def run(det, sel):
    v = 0; lat = []
    for i in sel:
        img, fr = frame(i)
        if img is None:
            continue
        t = time.time(); ok = valid_pose(det.detect(fr)); lat.append((time.time() - t) * 1000); v += ok
    return v, lat


sel = list(range(0, len(idx), 2))
n = len(sel)
print(f"{RUN.name}: {len(idx)} frames, sampling {n}")
rg = RedGlowGateDetector()
rv, _ = run(rg, sel)
print(f"red_glow      valid {rv}/{n} = {100*rv/n:.0f}%  -> valid_empty {100*(1-rv/n):.0f}%  (live seeker-diag: 74%)")
s = GateDetector.load(SINGLE); sv, sl = run(s, sel)
print(f"yolo single   valid {sv}/{n} = {100*sv/n:.0f}%  -> valid_empty {100*(1-sv/n):.0f}%  ; lat median {np.median(sl):.0f} ms")
e = GateDetector.load(ENS); ev, el = run(e, sel)
print(f"yolo ensemble valid {ev}/{n} = {100*ev/n:.0f}%  -> valid_empty {100*(1-ev/n):.0f}%  ; lat median {np.median(el):.0f} ms")
