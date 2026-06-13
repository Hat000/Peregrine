"""IN-LOOP vision latency harness (Task D, vision case-C readiness, 2026-06-13).

Measures, OFFLINE, the wall-time of the per-tick vision compute chain that turns a camera
frame into a KF position fix, and decomposes it into stages:

    detector forward pass  ->  PnP (estimate_gate_pose)  ->  associate  ->
    gate_pose_to_world_position  ->  kf.update_position

This is the IN-LOOP / COMPUTE latency L_total. It is DISTINCT from:
  * the 67 ms command-side ACTUATION latency (command-vs-realized cross-correlation), and
  * the content-vs-telemetry timestamp offset Delta that latency_fit.py (vision-pkg2) measured
    from off_ned = -Delta*v + c  (a data-ASSOCIATION / capture-stamp lag, not compute time).
L_total is the time the estimator's fix is STALE by purely from doing the work, on top of
frame-age. It is the dt that the predict-forward / rewind compensation (latency_design.md) must
cover.

HONESTY CONTRACT (adversarial self-check, enforced in the report header):
  * Every number is labelled with the EXACT hardware it ran on.
  * This laptop has torch 2.x +cpu, CUDA UNAVAILABLE -> the detector forward pass is timed on
    CPU and reported as an UPPER BOUND. A separate FLOP-based ESTIMATE bounds the eval-HW
    (~100 TOPS onboard edge) detector time and is labelled ESTIMATE, never measured.
  * A CPU detector time is NEVER allowed to masquerade as the eval-HW latency: the two are
    reported in separate, explicitly-labelled blocks and the L_total budget is composed once
    per HW assumption (cpu-upper-bound vs edge-estimate).

PnP/assoc/KF are pure-numpy/OpenCV and HW-portable to first order (the edge CPU is comparable
to or faster than this laptop for tiny dense linear algebra), so their measured laptop times
transfer with far less uncertainty than the GPU/NPU-bound detector; we still flag them MEASURED-
ON-LAPTOP and treat them as an upper bound for the edge.

Run:  .venv/Scripts/python.exe handoff/ultracode-vision-case-c-2026-06-13/latency_harness.py
Writes nothing under src/. Prints a structured report to stdout; also writes
latency_results.json next to this file.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
HERE = Path(__file__).resolve().parent

from racer.contracts import Gate, GateObservation, GatePose  # noqa: E402
from racer.frames import CAMERA_INTRINSICS_K, R_camera_from_body, R_world_from_body  # noqa: E402
from racer.localization import gate_pose_to_world_position  # noqa: E402
from racer.state_estimator import LinearKF  # noqa: E402
from racer.vision.association import associate, predict_gates_in_camera  # noqa: E402
from racer.vision.gate_pose import (  # noqa: E402
    GATE_INNER_SIZE_M,
    estimate_gate_pose,
    gate_object_points,
    project_gate_corners,
)

# --------------------------------------------------------------------------- #
#  Course / sensor constants (from the live stack + track map)                #
# --------------------------------------------------------------------------- #
NATIVE_WH = (640, 360)                 # camera native res (frames.json native_resolution_wh)
STRIDE_PADDED_WH = (640, 384)          # 360 padded UP to the YOLO stride-32 multiple at inference
K = CAMERA_INTRINSICS_K
MODEL_PATH = ROOT / "models" / "gate_yolo11s_curriculum_v2.pt"
TRACK_MAP = ROOT / "handoff" / "shadowpc-firstcontact-2026-06-02" / "track_map.json"

# Edge-budget assumption for the FLOP-based detector estimate. The onboard compute is documented
# as "~100 TOPS edge" (memory). TOPS is INT8; the model runs fp16/int8 on an NPU/GPU. We bound
# with an explicit, stated sustained-utilisation band rather than a single fabricated number.
EDGE_PEAK_TOPS = 100.0                 # documented onboard edge budget (INT8 peak)
# Realised throughput is a fraction of peak (memory-bound layers, kernel launch, NMS, pre/post).
# Small dense conv nets on edge NPUs realise ~10-35% of INT8 peak in practice; we report the band.
EDGE_UTIL_BAND = (0.10, 0.35)
# The bare FLOP/TOPS quotient is a COMPUTE FLOOR ONLY; it omits preprocess (letterbox+normalise),
# NMS/pose-decode, kernel-launch, and memory-bound layers, which dominate at this tiny resolution.
# Empirically a 9.7M-param yolo11s at 640x384 on a ~100-TOPS-class edge accelerator (Jetson Orin
# NX / similar) runs end-to-end ~5-15 ms. We use THIS empirical bracket for the L_total budget so
# the budget is not built on the over-optimistic sub-ms compute floor. Labelled ESTIMATE.
EDGE_EMPIRICAL_FWD_MS = (5.0, 15.0)    # (optimistic p50, conservative p90) end-to-end on edge HW


def _quantiles(samples_s: list[float]) -> dict:
    a = np.asarray(samples_s, dtype=np.float64) * 1e3   # -> ms
    return {
        "n": int(a.size),
        "p50_ms": float(np.percentile(a, 50)),
        "p90_ms": float(np.percentile(a, 90)),
        "p99_ms": float(np.percentile(a, 99)),
        "mean_ms": float(a.mean()),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
    }


def _time_calls(fn, n, warmup=5):
    """Time `fn` n times (perf_counter), return list of seconds. Warmup excluded."""
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


# --------------------------------------------------------------------------- #
#  Synthetic geometry: realistic gate sightings via project_gate_corners      #
# --------------------------------------------------------------------------- #
def synth_observation(range_m, az_deg=0.0, el_deg=0.0, yaw_deg=0.0, noise_px=0.7,
                      frame_id=0, rng=None):
    """A GateObservation for a gate at the given range, off-axis bearing, and gate yaw.

    Builds the camera-frame gate pose, projects the 4 inner corners with project_gate_corners
    (the exact inverse of the PnP), adds sub-pixel detector noise + per-corner confidences, and
    returns a GateObservation in the canonical IPPE corner order. This reproduces the detector's
    *output* (corner pixels) without the detector, so PnP/assoc/KF can be timed on geometry that
    matches the real course (ranges drawn from the empirical 4-corner-fix distribution).
    """
    rng = np.random.default_rng(frame_id) if rng is None else rng
    az, el, yaw = np.radians([az_deg, el_deg, yaw_deg])
    # gate centre in camera frame: range along boresight, offset by bearing (camera optical Z fwd)
    t = np.array([range_m * np.sin(az),
                  range_m * np.sin(el),
                  range_m * np.cos(az) * np.cos(el)], dtype=np.float64)
    # gate rotation in camera frame: a yaw about the gate's vertical (so corners aren't degenerate)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R_cam_gate = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    try:
        corners = project_gate_corners(R_cam_gate, t)
    except ValueError:
        return None
    if np.any(corners[:, 0] < -200) or np.any(corners[:, 0] > NATIVE_WH[0] + 200):
        return None
    corners = corners + rng.normal(0.0, noise_px, size=corners.shape)
    conf = np.clip(rng.normal(0.85, 0.08, size=4), 0.3, 1.0)
    return GateObservation(
        frame_id=int(frame_id),
        sim_time_ns=int(frame_id) * 13_000_000,    # ~77 Hz frame spacing (placeholder stamps)
        corners_px=corners.astype(np.float64),
        corner_ids=np.array([0, 1, 2, 3]),
        corner_confidence=conf.astype(np.float64),
        score=0.9,
        bbox_xywh=None,
        gate_id=None,
    )


def load_gates():
    data = json.loads(TRACK_MAP.read_text())
    gates = []
    n = len(data["gates"])
    pos = [np.asarray(g["position_ned"], float) for g in data["gates"]]
    for i, g in enumerate(data["gates"]):
        j = i + 1 if i + 1 < n else i
        seg = pos[j] - pos[i - 1 if j == i else i]
        z = seg / (np.linalg.norm(seg) + 1e-12)
        x = np.cross([0, 0, 1.0], z); x = x / (np.linalg.norm(x) + 1e-12)
        y = np.cross(z, x)
        R = np.column_stack([x, y, z])
        gates.append(Gate(gate_id=int(g["gate_id"]), position_ned=pos[i],
                          R_world_gate=R, inner_size_m=GATE_INNER_SIZE_M))
    return gates


def empirical_ranges():
    """4-corner associated true_range_m samples from the perception-char dataset (real course)."""
    rs = []
    base = ROOT / "handoff" / "perception-char-2026-06-08"
    for g in range(6):
        d = json.loads((base / f"characterize_g{g}.json").read_text())
        for r in d["rows"]:
            if r.get("detected") and r.get("associated") and r.get("n_corners") == 4:
                tr = float(r["true_range_m"])
                if 1.0 <= tr <= 32.0:        # in-loop fixes are accepted only inside vision_max_range
                    rs.append(tr)
    return np.asarray(rs)


# --------------------------------------------------------------------------- #
#  STAGE (i): PnP / assoc / world-fix / KF update                             #
# --------------------------------------------------------------------------- #
def measure_pnp_chain(n_geom=120, reps_per_geom=12):
    """Time estimate_gate_pose + associate + gate_pose_to_world_position + kf.update_position,
    over many realistic geometries (range sampled from the empirical course distribution)."""
    gates = load_gates()
    ranges = empirical_ranges()
    rng = np.random.default_rng(20260613)

    # Pre-build a population of (obs, gate, R_wb, predicted) so timing excludes setup.
    cases = []
    g3 = next(g for g in gates if g.gate_id == 3)            # mid-course gate as the world frame
    drone_pos = g3.position_ned + np.array([12.0, 0.0, 0.0])  # ~12 m short of gate 3, on course
    # attitude: roughly level, camera tilts up; small roll/pitch to be representative
    R_wb = R_world_from_body(np.radians(8.0), np.radians(-12.0), np.radians(2.0))
    predicted = predict_gates_in_camera(gates, drone_pos, R_wb)
    attempts = 0
    while len(cases) < n_geom and attempts < n_geom * 20:
        attempts += 1
        rmag = float(rng.choice(ranges)) if ranges.size else float(rng.uniform(4, 28))
        az = float(rng.uniform(-12, 12)); el = float(rng.uniform(-8, 8))
        yaw = float(rng.uniform(-25, 25))
        obs = synth_observation(rmag, az, el, yaw, noise_px=0.7,
                                frame_id=len(cases) + 1, rng=rng)
        if obs is None:
            continue
        cases.append((obs, g3, R_wb, predicted))

    # The PnP-chain wall-time is independent of WHICH map gate is the world anchor; we only need
    # a genuinely-visible gate so associate()/world-fix run on a real PredictedGate. Pick the
    # nearest predicted gate from this pose.
    if not predicted:
        raise RuntimeError("no gate predicted in camera from the chosen pose")
    anchor_id = min(predicted, key=lambda gid: predicted[gid].range_m)
    gate = next(g for g in gates if g.gate_id == anchor_id)

    timings = {k: [] for k in
               ["pnp", "assoc", "worldfix", "kf_update", "stage_i_total"]}
    for (obs, _g, R_wb_c, pred) in cases:
        kf = LinearKF.initialize(drone_pos, np.zeros(3), pos_std=1.0, vel_std=1.0)
        # priors mimic the navigator (fresh map+attitude prediction as PnP prior)
        pg = pred[anchor_id]
        prior = GatePose(obs.frame_id, obs.sim_time_ns, pg.R_cam_gate, pg.t_cam_gate, 0.0,
                         gate_id=anchor_id)

        def do_pnp():
            return estimate_gate_pose(obs, prior=prior, compute_covariance=True)

        def do_assoc():
            return associate(obs, pred, 1.6, 2.5)

        t_pnp = _time_calls(do_pnp, reps_per_geom, warmup=2)
        pose = do_pnp()
        if pose is None:
            continue
        t_assoc = _time_calls(do_assoc, reps_per_geom, warmup=2)

        def do_worldfix():
            return gate_pose_to_world_position(pose, gate, R_wb_c)

        t_wf = _time_calls(do_worldfix, reps_per_geom, warmup=2)
        zc = do_worldfix()

        def do_kf():
            kf2 = LinearKF.initialize(drone_pos, np.zeros(3), pos_std=1.0, vel_std=1.0)
            kf2.update_position(zc[0], zc[1])

        t_kf = _time_calls(do_kf, reps_per_geom, warmup=2)

        # take the median per-geom (reduces scheduler jitter) then aggregate across geoms
        timings["pnp"].append(float(np.median(t_pnp)))
        timings["assoc"].append(float(np.median(t_assoc)))
        timings["worldfix"].append(float(np.median(t_wf)))
        timings["kf_update"].append(float(np.median(t_kf)))
        timings["stage_i_total"].append(
            float(np.median(t_pnp) + np.median(t_assoc) + np.median(t_wf) + np.median(t_kf))
        )

    return {k: _quantiles(v) for k, v in timings.items() if v}


# --------------------------------------------------------------------------- #
#  STAGE (ii): YOLO detector forward pass (CPU MEASURED = UPPER BOUND)         #
# --------------------------------------------------------------------------- #
def measure_detector_cpu(n=30, warmup=5):
    """Time the YOLO forward pass on CPU at the padded inference resolution.

    CPU number = strict UPPER BOUND for the eval HW (which is a ~100 TOPS NPU/GPU). Returned
    with an explicit 'measured_on' tag so it can never be mislabelled as eval-HW latency.
    """
    try:
        import warnings
        warnings.filterwarnings("ignore")
        import torch
        from ultralytics import YOLO
    except Exception as e:  # pragma: no cover
        return {"available": False, "reason": f"{type(e).__name__}: {e}"}

    cuda = bool(torch.cuda.is_available())
    model = YOLO(str(MODEL_PATH))
    model.model.eval()
    img = np.random.randint(0, 255, (NATIVE_WH[1], NATIVE_WH[0], 3), dtype=np.uint8)

    def fwd():
        # imgsz fixed to the padded inference size; verbose off; full predict path
        # (pre-process + forward + NMS/decode) = the real in-loop cost.
        model.predict(img, imgsz=STRIDE_PADDED_WH[::-1], device="cpu", verbose=False)

    torch.set_grad_enabled(False)
    samples = _time_calls(fwd, n, warmup=warmup)
    res = _quantiles(samples)
    res.update({
        "available": True,
        "device": "cpu",
        "cuda_available": cuda,
        "is_upper_bound": True,
        "imgsz_hw": list(STRIDE_PADDED_WH[::-1]),
        "torch_version": torch.__version__,
        "cpu": platform.processor() or platform.machine(),
    })
    return res


def estimate_detector_edge():
    """FLOP-based ESTIMATE of detector forward time on the ~100 TOPS edge budget.

    NOT measured. GFLOPs from ultralytics get_flops at the padded inference size; INT8 TOPS peak
    de-rated by a stated sustained-utilisation band. Returned with is_estimate=True.
    """
    try:
        import warnings
        warnings.filterwarnings("ignore")
        from ultralytics import YOLO
        from ultralytics.utils.torch_utils import get_flops
        m = YOLO(str(MODEL_PATH))
        # get_flops counts MACs*2 (= FLOPs) at the given square imgsz; scale to the padded HxW area.
        gflops_640 = float(get_flops(m.model, imgsz=640))
        area_scale = (STRIDE_PADDED_WH[0] * STRIDE_PADDED_WH[1]) / (640 * 640)
        gflops_pad = gflops_640 * area_scale
        n_params = sum(p.numel() for p in m.model.parameters())
    except Exception as e:  # pragma: no cover
        return {"available": False, "reason": f"{type(e).__name__}: {e}"}

    # GFLOPs -> "GOPs" 1:1 for the op-count; INT8 TOPS ~ ops/s. Time = ops / (peak * util).
    # 1 GFLOP @ X effective-TOPS = 1e9 / (X*1e12) s. effective = peak * util.
    band = []
    for util in EDGE_UTIL_BAND:
        eff_tops = EDGE_PEAK_TOPS * util
        t_ms = (gflops_pad * 1e9) / (eff_tops * 1e12) * 1e3
        band.append((util, t_ms))
    return {
        "available": True,
        "is_estimate": True,
        "gflops_640sq": round(gflops_640, 3),
        "gflops_at_inference_size": round(gflops_pad, 3),
        "params_millions": round(n_params / 1e6, 3),
        "edge_peak_tops_int8": EDGE_PEAK_TOPS,
        "util_band": EDGE_UTIL_BAND,
        "compute_floor_ms_low_util": round(band[0][1], 3),   # 10% util -> slow end of FLOP floor
        "compute_floor_ms_high_util": round(band[1][1], 3),  # 35% util -> fast end of FLOP floor
        # Headline numbers used by the L_total budget: empirical end-to-end edge bracket, which
        # sits ABOVE the compute floor (preprocess + NMS/decode + memory-bound layers).
        "edge_fwd_p50_ms": EDGE_EMPIRICAL_FWD_MS[0],
        "edge_fwd_p90_ms": EDGE_EMPIRICAL_FWD_MS[1],
        "note": ("FLOP/TOPS quotient (compute_floor_*) is a floor only; the budget uses the "
                 "empirical end-to-end edge bracket edge_fwd_p50/p90 (5-15 ms), which the FLOP "
                 "floor confirms is compute-feasible (compute floor << 5 ms => detector is NOT "
                 "compute-bound; preprocess/NMS/memory dominate). All ESTIMATE, not measured here."),
    }


# --------------------------------------------------------------------------- #
#  Budget composition + the 30 Hz / >=30 m/s rule interaction                 #
# --------------------------------------------------------------------------- #
def compose_and_report(stage_i, det_cpu, det_edge):
    speeds = {"VQ1_recording_p50": 5.35, "VQ2_target": 20.0, "VQ2_high": 30.0}

    def budget(det_p50, det_p90, label, kind):
        i50 = stage_i["stage_i_total"]["p50_ms"]
        i90 = stage_i["stage_i_total"]["p90_ms"]
        L50 = det_p50 + i50
        L90 = det_p90 + i90
        out = {
            "hw_assumption": label, "kind": kind,
            "detector_p50_ms": round(det_p50, 3), "detector_p90_ms": round(det_p90, 3),
            "pnp_chain_p50_ms": round(i50, 3), "pnp_chain_p90_ms": round(i90, 3),
            "L_total_p50_ms": round(L50, 3), "L_total_p90_ms": round(L90, 3),
            "uncompensated_pos_err_m": {},
            "extra_distance_at_30Hz_window_m": {},
        }
        for sname, v in speeds.items():
            out["uncompensated_pos_err_m"][sname] = {
                "p50": round(v * L50 / 1e3, 3), "p90": round(v * L90 / 1e3, 3),
            }
            # how much the L_total alone shifts the effective last-accepted-fix distance
            out["extra_distance_at_30Hz_window_m"][sname] = {
                "p50": round(v * L50 / 1e3, 3), "p90": round(v * L90 / 1e3, 3),
            }
        return out

    budgets = []
    if det_cpu.get("available"):
        budgets.append(budget(det_cpu["p50_ms"], det_cpu["p90_ms"],
                              "laptop CPU (torch+cpu) — UPPER BOUND", "cpu_upper_bound"))
    if det_edge.get("available"):
        # edge budget uses the EMPIRICAL end-to-end bracket (5-15 ms), not the bare FLOP floor.
        budgets.append(budget(det_edge["edge_fwd_p50_ms"], det_edge["edge_fwd_p90_ms"],
                              "~100 TOPS edge (empirical 5-15 ms estimate)", "edge_estimate"))
    return {"speeds_mps": speeds, "budgets": budgets}


def main():
    print("=" * 78)
    print("IN-LOOP VISION LATENCY HARNESS — Task D (case-C readiness)  2026-06-13")
    print("=" * 78)
    print(f"host        : {platform.platform()}")
    print(f"cpu         : {platform.processor() or platform.machine()}")
    print(f"python      : {sys.version.split()[0]}")
    print(f"native res  : {NATIVE_WH}  | inference (stride-padded): {STRIDE_PADDED_WH}")
    print("NOTE: detector timed on CPU = UPPER BOUND (no CUDA on this laptop); a FLOP-based")
    print("      edge ESTIMATE is reported separately and never mixed into the CPU number.")
    print()

    print("[stage i] timing PnP / associate / world-fix / KF update over realistic geometries...")
    stage_i = measure_pnp_chain()
    for k in ["pnp", "assoc", "worldfix", "kf_update", "stage_i_total"]:
        if k in stage_i:
            q = stage_i[k]
            print(f"   {k:14s} p50 {q['p50_ms']:7.3f} ms   p90 {q['p90_ms']:7.3f} ms"
                  f"   (n={q['n']}, mean {q['mean_ms']:.3f})")
    print()

    print("[stage ii-a] timing YOLO detector forward pass on CPU (UPPER BOUND)...")
    det_cpu = measure_detector_cpu()
    if det_cpu.get("available"):
        print(f"   detector(CPU)  p50 {det_cpu['p50_ms']:7.1f} ms   p90 {det_cpu['p90_ms']:7.1f} ms"
              f"   [device={det_cpu['device']}, UPPER BOUND, cuda={det_cpu['cuda_available']}]")
    else:
        print(f"   detector CPU timing unavailable: {det_cpu.get('reason')}")
    print()

    print("[stage ii-b] FLOP-based detector ESTIMATE on ~100 TOPS edge (NOT measured)...")
    det_edge = estimate_detector_edge()
    if det_edge.get("available"):
        print(f"   GFLOPs@640^2 {det_edge['gflops_640sq']}  -> @{STRIDE_PADDED_WH} "
              f"{det_edge['gflops_at_inference_size']} GFLOPs  ({det_edge['params_millions']} M params)")
        print(f"   FLOP compute floor: {det_edge['compute_floor_ms_high_util']:.2f} ms (35% util) "
              f"-> {det_edge['compute_floor_ms_low_util']:.2f} ms (10% util)  [INT8 {EDGE_PEAK_TOPS} TOPS peak]")
        print(f"   edge end-to-end ESTIMATE used in budget: {det_edge['edge_fwd_p50_ms']:.0f}-"
              f"{det_edge['edge_fwd_p90_ms']:.0f} ms (empirical bracket)")
        print(f"   {det_edge['note']}")
    else:
        print(f"   edge estimate unavailable: {det_edge.get('reason')}")
    print()

    report = compose_and_report(stage_i, det_cpu, det_edge)
    print("[L_total budget + uncompensated error v*L_total]")
    for b in report["budgets"]:
        print(f"  --- {b['hw_assumption']}  ({b['kind']}) ---")
        print(f"      detector p50/p90 {b['detector_p50_ms']:.2f}/{b['detector_p90_ms']:.2f} ms"
              f"  + pnp-chain {b['pnp_chain_p50_ms']:.3f}/{b['pnp_chain_p90_ms']:.3f} ms")
        print(f"      L_total  p50 {b['L_total_p50_ms']:.2f} ms   p90 {b['L_total_p90_ms']:.2f} ms")
        for sname, e in b["uncompensated_pos_err_m"].items():
            print(f"        v={report['speeds_mps'][sname]:5.1f} m/s ({sname:18s}): "
                  f"uncompensated err p50 {e['p50']:.3f} m / p90 {e['p90']:.3f} m")
    print()

    out = {
        "host": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "native_res": NATIVE_WH, "inference_res": STRIDE_PADDED_WH,
        "stage_i_pnp_chain": stage_i,
        "stage_ii_detector_cpu_upper_bound": det_cpu,
        "stage_ii_detector_edge_estimate": det_edge,
        "budget": report,
    }
    (HERE / "latency_results.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {HERE / 'latency_results.json'}")


if __name__ == "__main__":
    main()
