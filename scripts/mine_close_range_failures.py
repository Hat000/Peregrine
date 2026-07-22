"""Mine the EXACT frames where perception died close to a gate, for hand labelling.

Not "failure-mined" in general -- this targets the one failure we are chasing. ``seeker.jsonl``
records, per control tick, why slot0 emitted nothing and how far the tracked gate was; the ticks
where ``reason == "valid_poses_empty"`` inside a few metres are frames where the detector produced
NO usable pose at all while the gate filled the view. Measured over 14 flights that is 11.4% of
ticks inside 2 m against 0.4% at 8-15 m.

Those frames are worth more per label than anything else available: they are the current model, on
the current course, failing in the regime the line/segmentation path exists to fix.

Join is on ``frame_id``, which both seeker.jsonl and video_index.jsonl carry, so no clock
reconciliation is needed (the ego_obs/video clocks do NOT agree -- do not join on time).

Usage:
  python scripts/mine_close_range_failures.py --out C:/Users/Shadow/vq2_close_inbox_2026-07-22
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "data" / "runs"


def mine_session(sess: Path, max_range_m: float, reasons: set[str], stride: int):
    """(frame_id, reason, range_m) for ticks that failed close in, thinned by ``stride`` frames."""
    sk, vi = sess / "seeker.jsonl", sess / "video_index.jsonl"
    if not sk.exists() or not vi.exists() or not (sess / "video.bin").exists():
        return []
    index = {}
    for line in vi.open():
        if line.strip():
            d = json.loads(line)
            index[d["frame_id"]] = (d["offset"], d["length"])
    hits, last = [], -10 ** 9
    for line in sk.open():
        if not line.strip():
            continue
        d = json.loads(line)
        s0 = d.get("slot0") or {}
        r, rng = s0.get("reason"), s0.get("track_range_m")
        if r not in reasons or rng is None or rng > max_range_m:
            continue
        fid = d.get("frame_id")
        if fid is None or fid not in index:
            continue
        # consecutive failing ticks show nearly the same picture; thin them or the batch is 200
        # near-duplicates of three moments.
        if fid - last < stride:
            continue
        last = fid
        hits.append((fid, r, float(rng), index[fid]))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-range", type=float, default=4.0)
    ap.add_argument("--stride", type=int, default=8, help="min frame_id gap within a session")
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--reasons", default="valid_poses_empty,continuity_reject")
    args = ap.parse_args()

    reasons = set(args.reasons.split(","))
    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    sessions = sorted(p for p in RUNS.glob("*") if p.is_dir())
    total, per_reason = 0, {}
    for sess in sessions:
        hits = mine_session(sess, args.max_range, reasons, args.stride)
        if not hits:
            continue
        blob = (sess / "video.bin").read_bytes()
        for fid, reason, rng, (off, length) in hits:
            if total >= args.limit:
                break
            jpg = blob[off:off + length]
            if len(jpg) < 1000 or jpg[:2] != b"\xff\xd8":       # not a JPEG -> skip, don't guess
                continue
            name = f"{reason}_{rng:04.1f}m_{sess.name}_f{fid}.jpg"
            (out / "frames" / name).write_bytes(jpg)
            per_reason[reason] = per_reason.get(reason, 0) + 1
            total += 1
        if total >= args.limit:
            break

    print(f"mined {total} frames from {len(sessions)} sessions -> {out/'frames'}")
    for k, v in sorted(per_reason.items()):
        print(f"  {k:22s} {v}")
    print("\nNOTE: these are the frames the CURRENT model fails on, so a seeder will mostly come up\n"
          "empty on them -- that is the point, not a bug. They need human eyes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
