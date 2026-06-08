"""scripts/extract_run.py -- compact cmd+telemetry extract for a live recording (the live-vs-twin
overlay artifact, Task-1-sysid-extract style). Reads a ``data/runs/<stamp>_<label>/`` session and emits
a single JSON with: the recorded controller config / probe meta (meta.json), the COMMANDED stream
(commands.jsonl), and the TRUE telemetry parsed from ``mavlink.tlog`` -- LOCAL_POSITION_NED (given
world pos/vel), ODOMETRY (true attitude quat), and ACTUATOR_OUTPUT_STATUS (the parser-independent
motor witness, Task-3 style). Plus a summary block for the rung report.

Handles BOTH cmd-log schemas (auto-detected by the ``phase`` key):
  * fly_vq1  (faithful/hover-hold): pos, vel(KF vz), thrust, roll, pitch, body_rate, sim_t.
  * rate_sysid (open-loop probe):   pos, vel(world), thrust(collective), actuators(motors), rpy,
    phase, kind -- per-phase collective steps.

Times: commands.jsonl ``sim_t``/``sim_time_ns`` (HIGHRES_IMU sim epoch) for the commanded stream;
LOCAL_POSITION_NED / ATTITUDE use a boot epoch -- each stream is reported on its OWN t-rel-to-first
sample (both start ~the GO), noted per stream. ``--hz`` downsamples the saved series; pass a large
value (e.g. 200) to keep ALL samples -- REQUIRED to resolve the ~6 Hz hover limit cycle (a 10 Hz
downsample aliases it). Default 50.

Usage:  python scripts/extract_run.py data/runs/<stamp>_<label> [--out PATH] [--hz 200]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from scipy.spatial.transform import Rotation

from racer.recording import RecordingReader


def _downsample(t: np.ndarray, hz: float) -> np.ndarray:
    if len(t) == 0:
        return np.array([], dtype=int)
    keep, last, dt = [], -1e9, 1.0 / hz
    for i, ti in enumerate(t):
        if ti - last >= dt:
            keep.append(i)
            last = ti
    if keep[-1] != len(t) - 1:
        keep.append(len(t) - 1)
    return np.asarray(keep, dtype=int)


def load_commands(d: Path) -> dict:
    rows = [json.loads(l) for l in open(d / "commands.jsonl") if l.strip()]
    is_probe = bool(rows) and ("phase" in rows[0])
    pos = np.array([r["pos"] for r in rows], dtype=np.float64)
    vel = np.array([r["vel"] for r in rows], dtype=np.float64)
    if is_probe:                                            # rate_sysid open-loop probe
        t = np.array([r["sim_time_ns"] for r in rows], dtype=np.float64) * 1e-9
        rpy = np.array([r["rpy"] for r in rows], dtype=np.float64)
        motors = np.array([float(np.mean(r["actuators"])) if r.get("actuators") else np.nan for r in rows])
        return {"schema": "probe", "t": t - t[0], "alt": -pos[:, 2], "z": pos[:, 2],
                "world_vz": vel[:, 2], "collective": np.array([r["thrust"] for r in rows]),
                "motors": motors, "pitch_deg": np.degrees(rpy[:, 1]), "roll_deg": np.degrees(rpy[:, 0]),
                "x": pos[:, 0], "y": pos[:, 1], "phase": [r["phase"] for r in rows]}
    t = np.array([r["sim_t"] for r in rows], dtype=np.float64) * 1e-9     # fly_vq1 faithful/hover-hold
    return {"schema": "faithful", "t": t - t[0], "alt": -pos[:, 2], "x": pos[:, 0], "y": pos[:, 1],
            "kf_vz": vel[:, 2], "thrust": np.array([r["thrust"] for r in rows]),
            "roll_deg": np.degrees([r["roll"] for r in rows]),
            "pitch_deg": np.degrees([r["pitch"] for r in rows])}


def load_tlog(d: Path) -> dict:
    lpn_t, lpn_z, lpn_vz = [], [], []
    act_t, act_mean = [], []
    for m in RecordingReader(d).iter_mavlink():
        ty = m.get_type()
        if ty == "LOCAL_POSITION_NED":
            lpn_t.append(m.time_boot_ms * 1e-3); lpn_z.append(m.z); lpn_vz.append(m.vz)
        elif ty == "ACTUATOR_OUTPUT_STATUS":
            a = np.array(m.actuator, dtype=np.float64)
            a = a[np.isfinite(a) & (a > 1e-4)]
            if a.size:
                act_t.append(m.time_usec * 1e-6); act_mean.append(float(a.mean()))

    def rel(x):
        x = np.asarray(x, dtype=np.float64)
        return x - x[0] if x.size else x
    return {"lpn_t": rel(lpn_t), "lpn_alt": -np.asarray(lpn_z), "lpn_vz": np.asarray(lpn_vz),
            "act_t": rel(act_t), "act_mean": np.asarray(act_mean)}


def summarize_faithful(c, tl, meta) -> dict:
    m = c["t"] >= (c["t"][-1] - 5.0)
    cfg = meta.get("controller_config", {})
    s = {"settle_window_s": 5.0,
         "alt_mean_m": float(np.nanmean(c["alt"][m])),
         "alt_range_m": [float(np.nanmin(c["alt"][m])), float(np.nanmax(c["alt"][m]))],
         "alt_std_m": float(np.nanstd(c["alt"][m])),
         "alt_slope_mps": float(np.polyfit(c["t"][m], c["alt"][m], 1)[0]),
         "kf_vz_mean": float(np.nanmean(c["kf_vz"][m])), "kf_vz_absmax": float(np.nanmax(np.abs(c["kf_vz"][m]))),
         "thrust_mean": float(np.nanmean(c["thrust"][m])),
         "thrust_range": [float(np.nanmin(c["thrust"][m])), float(np.nanmax(c["thrust"][m]))],
         "pitch_mean_deg": float(np.nanmean(c["pitch_deg"][m])), "roll_mean_deg": float(np.nanmean(c["roll_deg"][m])),
         "horiz_drift_max_m": float(np.nanmax(np.hypot(c["x"], c["y"])))}
    if cfg:
        s["thrust_frac_at_hi_clip"] = float(np.mean(c["thrust"][m] >= cfg["alt_thrust_hi"] - 5e-4))
        s["thrust_frac_at_lo_clip"] = float(np.mean(c["thrust"][m] <= cfg["alt_thrust_lo"] + 5e-4))
    if len(tl["lpn_t"]):
        ml = tl["lpn_t"] >= (tl["lpn_t"][-1] - 5.0)
        s["true_vz_mean"] = float(np.nanmean(tl["lpn_vz"][ml]))
        s["true_vz_absmax"] = float(np.nanmax(np.abs(tl["lpn_vz"][ml])))
        s["true_vz_std"] = float(np.nanstd(tl["lpn_vz"][ml]))
    if len(tl["act_t"]):
        ma = tl["act_t"] >= (tl["act_t"][-1] - 5.0)
        s["motor_mean"] = float(np.nanmean(tl["act_mean"][ma]))
    return s


def summarize_probe(c) -> dict:
    """Per-phase DESCRIPTIVE stats only (collective, vz start/end, motor mean) -- NOT a hover/slope
    fit (that is the offline Commander's job). vz is NED world (+ = descending)."""
    phases = []
    cur = None
    for i, ph in enumerate(c["phase"]):
        if cur is None or ph != cur["name"]:
            if cur is not None:
                phases.append(cur)
            cur = {"name": ph, "i0": i}
        cur["i1"] = i
    if cur is not None:
        phases.append(cur)
    out = []
    for p in phases:
        sl = slice(p["i0"], p["i1"] + 1)
        t, vz, col, mot = c["t"][sl], c["world_vz"][sl], c["collective"][sl], c["motors"][sl]
        dur = float(t[-1] - t[0]) if len(t) > 1 else 0.0
        out.append({"phase": p["name"], "collective_cmd": float(np.nanmedian(col)),
                    "motor_mean": float(np.nanmean(mot)), "dur_s": round(dur, 3), "n": int(len(t)),
                    "vz_start": round(float(vz[0]), 3), "vz_end": round(float(vz[-1]), 3),
                    "pitch_mean_deg": round(float(np.nanmean(c["pitch_deg"][sl])), 2)})
    return {"note": "NED world vz (+=down); per-phase descriptive only, no fit", "phases": out}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--out", default=None)
    ap.add_argument("--hz", type=float, default=50.0, help="downsample rate for the saved series (use 200 to keep all)")
    args = ap.parse_args()
    d = Path(args.session)
    meta = json.loads((d / "meta.json").read_text())
    c = load_commands(d)
    tl = load_tlog(d)

    def ds(t, *arrs):
        k = _downsample(t, args.hz)
        return [t[k].round(3).tolist()] + [np.asarray(a)[k].round(4).tolist() for a in arrs]

    lt = ds(tl["lpn_t"], tl["lpn_alt"], tl["lpn_vz"]) if len(tl["lpn_t"]) else [[]]
    at = ds(tl["act_t"], tl["act_mean"]) if len(tl["act_t"]) else [[]]
    out = {"session": str(d), "label": meta.get("label"), "schema": c["schema"],
           "faithful": meta.get("faithful"), "hover_hold": meta.get("hover_hold"),
           "final_state": meta.get("final_state"), "aborted": meta.get("aborted"),
           "controller_config": meta.get("controller_config"), "bounds": meta.get("bounds"),
           "true_lpn": {"note": "LOCAL_POSITION_NED, boot epoch; GIVEN/true world vz (+=down)",
                        "fields": ["t_s", "alt_m", "true_vz_neddown"], "series": lt},
           "motors": {"note": "ACTUATOR_OUTPUT_STATUS active-motor mean (parser-independent witness)",
                      "fields": ["t_s", "mean"], "series": at}}
    if c["schema"] == "probe":
        out["probe_meta"] = {"mode": meta.get("mode"), "thrust_base": meta.get("thrust")}
        out["summary"] = summarize_probe(c)
        out["commanded"] = {"note": "sim epoch; collective steps, world vz (+=down), motor witness",
                            "fields": ["t_s", "collective", "world_vz_neddown", "motor_mean", "pitch_deg", "alt_m", "phase"],
                            "series": ds(c["t"], c["collective"], c["world_vz"], c["motors"], c["pitch_deg"], c["alt"])
                            + [[c["phase"][i] for i in _downsample(c["t"], args.hz)]]}
    else:
        out["summary"] = summarize_faithful(c, tl, meta)
        out["commanded"] = {"note": "sim epoch; kf_vz = KF vertical vel the controller used (+=down)",
                            "fields": ["t_s", "alt_m", "kf_vz_neddown", "thrust", "pitch_deg", "roll_deg", "x", "y"],
                            "series": ds(c["t"], c["alt"], c["kf_vz"], c["thrust"], c["pitch_deg"], c["roll_deg"], c["x"], c["y"])}
    out_path = Path(args.out) if args.out else (d / "extract.json")
    out_path.write_text(json.dumps(out, indent=2))

    # -- printed summary --
    print(f"=== extract: {d.name}  schema={c['schema']}  -> {out_path}")
    if c["schema"] == "probe":
        print(f"  mode={meta.get('mode')} aborted={meta.get('aborted')}  (per-phase, NED vz +=down; NO fit here)")
        for p in out["summary"]["phases"]:
            print(f"  {p['phase']:11s} col_cmd={p['collective_cmd']:.3f} motor={p['motor_mean']:.3f} "
                  f"dur={p['dur_s']:.1f}s vz {p['vz_start']:+.2f}->{p['vz_end']:+.2f}  pitch {p['pitch_mean_deg']:+.1f}deg")
    else:
        s = out["summary"]
        print(f"  final={meta.get('final_state')}  ALT hold mean {s['alt_mean_m']:.3f} m std {s['alt_std_m']:.3f} "
              f"slope {s['alt_slope_mps']:+.3f}")
        if "true_vz_absmax" in s:
            print(f"  TRUE vz |max| {s['true_vz_absmax']:.3f} std {s['true_vz_std']:.3f}  thrust mean {s['thrust_mean']:.3f} "
                  f"[{s['thrust_range'][0]:.2f},{s['thrust_range'][1]:.2f}]  motors {s.get('motor_mean', float('nan')):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
