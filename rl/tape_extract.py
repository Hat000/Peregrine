"""REPLAY-RATCHET tape extractor: ego_obs.jsonl -> ``--sysid-replay`` command tape.

PREMISE (ratchet strategy, 2026-07-17): the VQ2 eval sim is deterministic with fixed gates,
so re-sending a recorded champion flight's per-tick wire commands reproduces its trajectory.
This tool converts a recorded ego flight log (``data/runs/<session>/ego_obs.jsonl``, written
by ``rl/fly_rl.py`` ``_fly_ego``) into the CSV action program consumed by the ALREADY-FLOWN
command player: ``rl/fly_rl.py --sysid-replay`` (fly_rl.py ~2052-2197, sysid battery lineage).

WHAT THE LOG RECORDS (load-bearing, verified against fly_rl.py @ dac05a5 == 28404fa semantics
-- the ego command path is unchanged between the champion commit and branch head):

  * ``rate_frd`` (3) + ``collective`` -- EXACTLY the values passed to client.send_command
    (fly_rl.py:2312-2317, logged at :2352-2353): post yaw-clamp / pitch-fence / roll-fence /
    virtual-flip / assist / floor / governor.  This is the PRE-CLIENT-SCALE wire level: inside
    MavlinkClient.send_command (src/racer/mavlink_client.py:486-495) the body rates are further
    multiplied by ``cmd_rate_scale`` (ego mode FORCES it to ``--ego-rate-scale``, fly_rl.py:3168)
    and the thrust is sent unscaled.  The champion panel runs flew ego_rate_scale=1.2, so the
    TRUE wire saw rate_frd*1.2.  REPLAY FIDELITY RULE: fly the tape with the SAME
    ``--ego-rate-scale`` as the recording (echoed in the printed replay command); the scale can
    NOT be baked into the tape because rescale_action's rate span is +-3.14 rad/s and a baked
    1.2x would clip.
  * ``normed_thrust`` -- the EMITTED g-units (post assist/floor/governor); collective ==
    clip(normed_thrust * 0.2656, 0, 1) holds to <6e-6 on the champion log, so a_thrust alone
    carries the full thrust channel.
  * ``sim_time_ns`` -- sim clock at command time.  The recording loop UNDER-RAN its 40 Hz
    target (champion: 26.4 Hz effective, dt 14..104 ms), while the player consumes ONE ROW PER
    LOOP TICK at ``--rate`` and IGNORES the t column (fly_rl.py:2061).  A raw row-per-source-tick
    tape would therefore replay ~1.4x TOO FAST.  This tool's PRIMARY output is a zero-order-hold
    resample of the command timeline onto the player's fixed tick grid; the raw-timestamp
    variant (``*.raw.csv``) is emitted for analysis only.

PLAYER INPUT FORMAT (fly_rl.py:2057-2063 + sysid/sysid_program.py):
  header row (skipped unconditionally), then rows ``t, a_thrust, a_roll, a_pitch, a_yaw[, seg,
  ...extra]`` -- RAW policy actions in [-1,1]; the loader reads r[1:5] (+ r[5] as a passthrough
  segment label); t and any extra columns are ignored.  Each row is one control tick.  The row
  is mapped to the wire by sysid_wire_from_action (fly_rl.py:745-759): rescale onto
  [0,5]g / +-3.14 rad/s, optional virtual flip (Rz(pi) = diag(-1,-1,1)), FLU->FRD via [1,-1,1],
  collective = clip(normed*0.2656, 0, 1) -- with ALL deploy clamps OFF.  This tool emits the
  EXACT algebraic inverse of that map, so the player reproduces the logged wire values to float
  precision (the tape inherits only the log's own rounding: 1e-4 rad/s / ~3e-6 collective).

FLIP CONVENTION: default ``--flip off`` matches the established sysid-replay recipe
(handoff/sysid-plant-response-2026-07-16.md: ``--no-virtual-flip`` mandatory) and makes the
a_roll/a_pitch/a_yaw columns readable as REAL-body FLU actions.  The inversion is exact for
either setting; extraction flip MUST equal the replay's ``--virtual-flip/--no-virtual-flip``
flag (pinned in the printed command).

USAGE
  python rl/tape_extract.py data/runs/20260714_202317_panel_run_f1 --validate
  python rl/tape_extract.py <run_dir> -o tape.csv --truncate-at-gate 2 --margin-ticks 12

Stdlib-only on purpose (runs on ShadowPC without the training venv).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

# --- constants PINNED to rl/fly_rl.py (the player side). If fly_rl changes these, the
# round-trip cross-check test (tests/test_tape_extract.py) fails loudly. ---
_HOVER_THRUST = 0.2656        # fly_rl.py:123  _HOVER_THRUST
_THRUST_HALF = 2.5            # rescale: normed = 2.5*(a+1), bounds [0,5] g  (fly_rl.py:742-743)
_RATE_HALF = 3.14             # rescale: rate_flu = 3.14*a, bounds +-3.14    (fly_rl.py:742-743)

# sysid_wire_from_action (fly_rl.py:745-759) composed per-axis sign maps:
#   flip OFF: rate_frd = 3.14*a[1:4] * [1,-1,1]          (FLU->FRD only)
#   flip ON : rate_frd = 3.14*(diag(-1,-1,1)@a[1:4]) * [1,-1,1] = 3.14*a[1:4] * [-1,1,1]
_FRD_SIGN = {False: (1.0, -1.0, 1.0), True: (-1.0, 1.0, 1.0)}


def forward_wire(a, flip: bool):
    """Reimplementation of fly_rl.sysid_wire_from_action (pure python). Returns
    (rate_frd[3], collective, normed_thrust)."""
    s = _FRD_SIGN[bool(flip)]
    normed = _THRUST_HALF * (a[0] + 1.0)
    rate_frd = [_RATE_HALF * a[1] * s[0], _RATE_HALF * a[2] * s[1], _RATE_HALF * a[3] * s[2]]
    collective = min(max(normed * _HOVER_THRUST, 0.0), 1.0)
    return rate_frd, collective, normed


def invert_wire(rate_frd, normed_thrust: float, flip: bool):
    """Exact inverse of forward_wire: logged wire values -> raw action [a_thrust,a_roll,a_pitch,
    a_yaw].  Raises ValueError if the values fall outside the representable action box."""
    s = _FRD_SIGN[bool(flip)]
    a = [
        normed_thrust / _THRUST_HALF - 1.0,
        rate_frd[0] / (_RATE_HALF * s[0]),
        rate_frd[1] / (_RATE_HALF * s[1]),
        rate_frd[2] / (_RATE_HALF * s[2]),
    ]
    eps = 1e-6
    if not (-1.0 - eps <= a[0] <= 1.0 + eps):
        raise ValueError(f"normed_thrust {normed_thrust} outside [0,5] g -> not representable")
    for i in (1, 2, 3):
        if abs(a[i]) > 1.0 + eps:
            raise ValueError(f"rate_frd axis {i-1} = {rate_frd[i-1]} outside +-{_RATE_HALF} rad/s")
    return a


# ---------------------------------------------------------------------------------------------


def load_run(path: Path):
    """Accept a session run dir (preferred: picks up meta.json) or an ego_obs.jsonl path.
    Returns (records, meta_or_None, obs_path)."""
    path = Path(path)
    if path.is_dir():
        obs_path = path / "ego_obs.jsonl"
        meta_path = path / "meta.json"
    else:
        obs_path = path
        meta_path = path.parent / "meta.json"
    if not obs_path.is_file():
        raise FileNotFoundError(f"no ego_obs.jsonl at {obs_path}")
    records = []
    with open(obs_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{obs_path}:{ln+1}: bad JSON ({exc})") from None
    meta = None
    if meta_path.is_file():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return records, meta, obs_path


def analyze(records):
    """Timing / channel / gate analysis shared by --validate and the sidecar. Returns a dict;
    hard problems are collected under 'errors', soft ones under 'warnings'."""
    errors, warnings = [], []
    n = len(records)
    if n < 2:
        return {"errors": [f"only {n} records -- nothing to extract"], "warnings": [],
                "n_records": n}

    for i, r in enumerate(records):
        for key in ("sim_time_ns", "rate_frd", "collective", "normed_thrust", "gate_index"):
            if key not in r:
                errors.append(f"record k={r.get('k', i)}: missing '{key}'")
                return {"errors": errors, "warnings": warnings, "n_records": n}
        vals = list(r["rate_frd"]) + [r["collective"], r["normed_thrust"]]
        if any((not isinstance(v, (int, float))) or v != v or math.isinf(v) for v in vals):
            errors.append(f"record k={r.get('k', i)}: NaN/inf/non-numeric command channel")

    ts = [r["sim_time_ns"] for r in records]
    dts = [(b - a) / 1e6 for a, b in zip(ts, ts[1:])]
    n_nonmono = sum(1 for d in dts if d <= 0)
    if n_nonmono:
        errors.append(f"{n_nonmono} non-monotonic sim_time steps")
    dts_sorted = sorted(dts)
    med = dts_sorted[len(dts_sorted) // 2]
    big_gaps = [(i + 1, d) for i, d in enumerate(dts) if d > max(3.0 * med, 100.0)]
    for k, d in big_gaps[:10]:
        warnings.append(f"tick gap {d:.0f} ms into k={k} (>3x median {med:.0f} ms) -- "
                        f"ZOH will hold the prior command across it, matching the recording")

    # collective <-> normed_thrust consistency: a_thrust carries thrust IFF this holds.
    coll_err = max(abs(r["collective"] - min(max(r["normed_thrust"] * _HOVER_THRUST, 0.0), 1.0))
                   for r in records)
    if coll_err > 1e-4:
        errors.append(f"collective vs clip(normed*{_HOVER_THRUST}) mismatch up to {coll_err:.2e} "
                      f"-- thrust channel NOT representable by a_thrust alone; investigate the log")

    g = [int(r["gate_index"]) for r in records]
    transitions = [
        {"k": i, "gate_from": g[i - 1], "gate_to": g[i],
         "t_rel_s": round((ts[i] - ts[0]) / 1e9, 4)}
        for i in range(1, n) if g[i] != g[i - 1]
    ]
    for tr in transitions:
        if tr["gate_to"] < tr["gate_from"]:
            errors.append(f"gate_index DROPPED {tr['gate_from']}->{tr['gate_to']} at k={tr['k']} "
                          f"(sim reset mid-log?)")

    # KF horizontal speed (finite-difference of kf_pos_ned) -- the ARREST-design signal.
    kf_speed = [0.0]
    for a, b in zip(records, records[1:]):
        dt = (b["sim_time_ns"] - a["sim_time_ns"]) / 1e9
        pa, pb = a.get("kf_pos_ned"), b.get("kf_pos_ned")
        if dt > 0 and pa and pb:
            kf_speed.append(math.hypot(pb[0] - pa[0], pb[1] - pa[1]) / dt)
        else:
            kf_speed.append(kf_speed[-1])

    ranges = {}
    for ax in range(3):
        v = [r["rate_frd"][ax] for r in records]
        ranges[f"rate_frd[{ax}]"] = (min(v), max(v))
    ranges["collective"] = (min(r["collective"] for r in records),
                            max(r["collective"] for r in records))
    ranges["normed_thrust"] = (min(r["normed_thrust"] for r in records),
                               max(r["normed_thrust"] for r in records))

    return {
        "errors": errors, "warnings": warnings,
        "n_records": n,
        "duration_s": (ts[-1] - ts[0]) / 1e9,
        "dt_ms": {"min": min(dts), "median": med, "mean": sum(dts) / len(dts),
                  "p95": dts_sorted[int(0.95 * len(dts_sorted))], "max": max(dts)},
        "effective_hz": 1000.0 * len(dts) / sum(dts),
        "gate_transitions": transitions,
        "kf_speed": kf_speed,
        "kf_speed_max": max(kf_speed),
        "kf_speed_final": kf_speed[-1],
        "channel_ranges": ranges,
        "assist_ticks": sum(1 for r in records if r.get("assist")),
        "floor_ticks": sum(1 for r in records if r.get("floor")),
        "gov_ticks": sum(1 for r in records if r.get("gov_engaged")),
        "collective_consistency_err": coll_err,
    }


def truncate_at_gate(records, gate_k: int, margin_ticks: int, analysis):
    """Keep the tape through gate ``gate_k``'s pass (first record whose gate_index > gate_k)
    plus ``margin_ticks`` further source ticks.  Raises if that gate was never passed."""
    cut = None
    for i, r in enumerate(records):
        if int(r["gate_index"]) > gate_k:
            cut = i
            break
    if cut is None:
        reached = max(int(r["gate_index"]) for r in records)
        raise ValueError(
            f"--truncate-at-gate {gate_k}: the log never advances past gate {gate_k} "
            f"(max active_gate_index reached: {reached}; passes: "
            f"{[(t['gate_from'], t['gate_to']) for t in analysis['gate_transitions']]})")
    end = min(len(records), cut + 1 + margin_ticks)
    return records[:end], cut


def build_source_rows(records, flip: bool):
    """Per source tick: (t_rel_s, a[4], seg, gate, src_k) + round-trip errors vs the logged
    wire values.  rate_err pins the MAPPING LOGIC (must be ~float eps: inversion of the logged
    rates re-emits them exactly).  coll_err is bounded below by the log's own rounding
    (normed_thrust and collective are logged at 5 dp independently, so recomputing collective
    from the 5-dp normed differs from the 5-dp logged collective by up to ~7e-6 -- champion
    measured 5.7e-6)."""
    t0 = records[0]["sim_time_ns"]
    rows, rate_err, coll_err = [], 0.0, 0.0
    for i, r in enumerate(records):
        a = invert_wire(r["rate_frd"], r["normed_thrust"], flip)
        rf, coll, _ = forward_wire(a, flip)
        rate_err = max(rate_err, max(abs(x - y) for x, y in zip(rf, r["rate_frd"])))
        coll_err = max(coll_err, abs(coll - r["collective"]))
        seg = f"g{int(r['gate_index'])}" + ("+a" if r.get("assist") else "")
        rows.append({
            "t": (r["sim_time_ns"] - t0) / 1e9,
            "a": a, "seg": seg, "gate": int(r["gate_index"]),
            "src_k": int(r.get("k", i)),
        })
    return rows, rate_err, coll_err


def resample_zoh(rows, rate_hz: float):
    """Zero-order-hold resample onto the player's fixed tick grid t_i = i/rate (grid t=0 ==
    first recorded command tick).  Row i carries the LAST source command with t <= t_i -- i.e.
    the command that was ACTIVE at that sim time in the recording."""
    if rate_hz <= 0:
        raise ValueError("rate must be > 0")
    dt = 1.0 / rate_hz
    t_end = rows[-1]["t"]
    n_out = int(math.floor(t_end / dt + 1e-9)) + 1
    out, j = [], 0
    for i in range(n_out):
        t_i = i * dt
        while j + 1 < len(rows) and rows[j + 1]["t"] <= t_i + 1e-12:
            j += 1
        src = rows[j]
        out.append({**src, "t": t_i})
    return out


def write_tape(path: Path, rows, kf_speed=None):
    """CSV the player parses (fly_rl.py:2057-2063: header skipped, r[1:5]=actions, r[5]=seg
    label; further columns ignored).  Extra cols: gate, src_k, kf_speed (analysis only)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "a_thrust", "a_roll", "a_pitch", "a_yaw", "seg",
                    "gate", "src_k", "kf_speed"])
        for r in rows:
            sp = "" if kf_speed is None else f"{kf_speed[r['src_k']]:.2f}" \
                if r["src_k"] < len(kf_speed) else ""
            w.writerow([f"{r['t']:.4f}"] + [f"{v:.8f}" for v in r["a"]]
                       + [r["seg"], r["gate"], r["src_k"], sp])


def replay_command(tape_path, meta, flip: bool, rate_hz: float, duration_s: float):
    """The EXACT fly_rl invocation that reproduces the recorded wire (modulo log rounding).
    Every pinned flag is load-bearing -- see the module docstring."""
    ego_ckpt = (meta or {}).get("ego_ckpt", "<EGO_CKPT.pth>")
    scale = (meta or {}).get("ego_rate_scale", 1.0)
    endpoint = (meta or {}).get("endpoint", "udp:127.0.0.1:14550")
    flip_flag = "--virtual-flip" if flip else "--no-virtual-flip"
    return (f"python rl/fly_rl.py --endpoint {endpoint} --ego-ckpt {ego_ckpt} "
            f"--sysid-replay {tape_path} --rate {rate_hz:g} --ego-rate-scale {scale:g} "
            f"{flip_flag} --sysid-climb-s 0 --sysid-settle-s 0 "
            f"--seeker-detector red_glow --max-seconds {math.ceil(duration_s + 15)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", help="session run dir (with meta.json) or an ego_obs.jsonl path")
    ap.add_argument("-o", "--out", default=None,
                    help="output tape CSV (default <run_dir>/tape.csv; also writes "
                         "<stem>.raw.csv + <stem>.tapemeta.json)")
    ap.add_argument("--rate", type=float, default=None,
                    help="resample rate Hz == the replay loop's --rate "
                         "(default: meta.json rate_hz, else 40)")
    ap.add_argument("--truncate-at-gate", type=int, default=None, metavar="K",
                    help="keep the tape through gate K's pass (first tick with "
                         "active_gate_index > K) + --margin-ticks source ticks")
    ap.add_argument("--margin-ticks", type=int, default=12,
                    help="source ticks kept past the gate-K pass (default 12, ~0.45 s at the "
                         "champion's 26.4 Hz effective)")
    ap.add_argument("--flip", choices=["off", "on"], default="off",
                    help="virtual-flip convention of the REPLAY invocation (default off == the "
                         "sysid-replay recipe's --no-virtual-flip; must match the flag)")
    ap.add_argument("--validate", action="store_true",
                    help="print the analysis report; without -o, dry-run (no files written)")
    ap.add_argument("--no-raw", action="store_true", help="skip the *.raw.csv variant")
    args = ap.parse_args(argv)

    flip = args.flip == "on"
    records, meta, obs_path = load_run(Path(args.run))
    rate_hz = args.rate if args.rate else float((meta or {}).get("rate_hz", 40.0))

    analysis = analyze(records)
    if analysis["errors"]:
        for e in analysis["errors"]:
            print(f"[tape] ERROR: {e}")
        return 2

    cut_k = None
    if args.truncate_at_gate is not None:
        records, cut_k = truncate_at_gate(records, args.truncate_at_gate,
                                          args.margin_ticks, analysis)

    rows, rate_err, coll_err = build_source_rows(records, flip)
    if rate_err > 1e-6 or coll_err > 1e-4:
        print(f"[tape] ERROR: inversion round-trip error (rates {rate_err:.2e} / collective "
              f"{coll_err:.2e}) exceeds bounds (1e-6 / 1e-4) -- the action->wire map in this "
              f"tool no longer matches the log; do NOT fly this tape.")
        return 2
    tape = resample_zoh(rows, rate_hz)

    if args.validate:
        a = analysis
        print(f"[tape] source: {obs_path}")
        print(f"[tape] {a['n_records']} source ticks, {a['duration_s']:.3f} s, "
              f"effective {a['effective_hz']:.2f} Hz "
              f"(dt ms min/med/p95/max = {a['dt_ms']['min']:.1f}/{a['dt_ms']['median']:.1f}/"
              f"{a['dt_ms']['p95']:.1f}/{a['dt_ms']['max']:.1f})")
        if meta:
            print(f"[tape] meta: rate_hz={meta.get('rate_hz')} "
                  f"ego_rate_scale={meta.get('ego_rate_scale')} "
                  f"virtual_flip={meta.get('virtual_flip')} final={meta.get('final_state')} "
                  f"gate_index={meta.get('gate_index')} collisions={meta.get('collisions')}")
        print(f"[tape] gate passes ({len(a['gate_transitions'])}):")
        for tr in a["gate_transitions"]:
            sp = a["kf_speed"][tr["k"]] if tr["k"] < len(a["kf_speed"]) else float("nan")
            print(f"        k={tr['k']:4d} t={tr['t_rel_s']:7.3f}s  "
                  f"gate {tr['gate_from']}->{tr['gate_to']}  (KF speed {sp:.2f} m/s)")
        for name, (lo, hi) in a["channel_ranges"].items():
            print(f"[tape] {name}: [{lo:+.4f}, {hi:+.4f}]")
        print(f"[tape] kf_speed max {a['kf_speed_max']:.2f} m/s, at tape end "
              f"{a['kf_speed'][len(records)-1]:.2f} m/s"
              + (f" (TRUNCATED at gate {args.truncate_at_gate}, cut k={cut_k}"
                 f"+{args.margin_ticks})" if cut_k is not None else ""))
        print(f"[tape] assist={a['assist_ticks']} floor={a['floor_ticks']} "
              f"gov={a['gov_ticks']} ticks; collective<->normed err "
              f"{a['collective_consistency_err']:.2e}; inversion round-trip "
              f"rates {rate_err:.2e} / collective {coll_err:.2e}")
        print(f"[tape] resample: {len(rows)} source -> {len(tape)} rows @ {rate_hz:g} Hz "
              f"(ZOH; the player ignores t and steps one row per tick)")
        for w in analysis["warnings"]:
            print(f"[tape] WARNING: {w}")

    write = args.out is not None or not args.validate
    if write:
        if args.out:
            out = Path(args.out)
        else:
            base = obs_path.parent if obs_path.parent.name else Path(".")
            suffix = f"_g{args.truncate_at_gate}" if args.truncate_at_gate is not None else ""
            out = base / f"tape{suffix}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        write_tape(out, tape, analysis["kf_speed"])
        raw_out = out.with_suffix(".raw.csv")
        if not args.no_raw:
            write_tape(raw_out, rows, analysis["kf_speed"])
        cmd = replay_command(out, meta, flip, rate_hz, tape[-1]["t"])
        sidecar = {
            "source": str(obs_path),
            "flip": flip,
            "rate_hz": rate_hz,
            "n_source_ticks": len(rows),
            "n_tape_rows": len(tape),
            "duration_s": rows[-1]["t"],
            "truncate_at_gate": args.truncate_at_gate,
            "margin_ticks": args.margin_ticks if args.truncate_at_gate is not None else None,
            "cut_src_index": cut_k,
            "gate_transitions": analysis["gate_transitions"],
            "kf_speed_at_tape_end": analysis["kf_speed"][len(records) - 1],
            "recorded_ego_rate_scale": (meta or {}).get("ego_rate_scale"),
            "recorded_virtual_flip": (meta or {}).get("virtual_flip"),
            "inversion_roundtrip_err": {"rates": rate_err, "collective": coll_err},
            "replay_cmd": cmd,
        }
        with open(out.with_suffix(".tapemeta.json"), "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2)
        print(f"[tape] wrote {out} ({len(tape)} rows @ {rate_hz:g} Hz)"
              + ("" if args.no_raw else f" + {raw_out.name} ({len(rows)} raw ticks)")
              + f" + {out.with_suffix('.tapemeta.json').name}")
        print(f"[tape] REPLAY (every flag is load-bearing -- see module docstring):")
        print(f"        {cmd}")
        if (meta or {}).get("ego_rate_scale") not in (None, 1.0):
            print(f"[tape] NOTE: recording flew ego_rate_scale={meta['ego_rate_scale']:g}; the "
                  f"tape is PRE-scale, so the replay MUST keep --ego-rate-scale "
                  f"{meta['ego_rate_scale']:g} or the wire rates will be wrong.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
