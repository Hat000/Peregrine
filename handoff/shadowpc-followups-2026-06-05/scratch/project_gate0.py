"""Throwaway: decode representative frames + project map gate-0 to quantify apparent gate size vs
range. Compares firstcontact (static, 23.3 m) to gate0_course1 (full approach). Saves PNGs to view.

Projection uses ONLY the pristine given pose (LPN pos) + ODOMETRY attitude + the camera model
(frames). Gate-0 inner square under the CENTER hypothesis (map + half-outer-height lift). This is a
SANITY view of where/how big the gate is -- the laptop runs the rigorous detector->PnP verdict.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

SRC = Path("C:/Users/Shadow/Peregrine/.claude/worktrees/blissful-sutherland-c8ebd9/src")
sys.path.insert(0, str(SRC))
from racer.frames import project_camera_point, world_point_in_camera  # noqa: E402
from racer.recording import RecordingReader  # noqa: E402

OUT = Path("C:/Users/Shadow/Peregrine/.claude/worktrees/blissful-sutherland-c8ebd9/"
           "handoff/shadowpc-followups-2026-06-05/scratch")
RUNS = Path("C:/Users/Shadow/Peregrine/data/runs")
GATE0_MAP = np.array([-23.30, -0.40, -0.03])
HALF_OUTER_H = 1.36          # memo: bottom-edge -> center lift (col2 of true quat), outer half-height
INNER = 1.5


def R_wb_from_quat(q):
    q = np.asarray(q, float)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def gate_inner_corners_center_hypo():
    """4 inner-square corners (1.5 m) about the CENTER hypothesis. Gate faces -X; width=+Y, height=+Z(down)."""
    c = GATE0_MAP + np.array([0.0, 0.0, -HALF_OUTER_H])      # lift to opening center
    h = INNER / 2.0
    return [c + np.array([0, sy * h, sz * h]) for sy in (-1, 1) for sz in (-1, 1)], c


def project(run_dir, pick_x, label):
    reader = RecordingReader(run_dir)
    meta = reader.meta
    t0u, t0m = int(meta["t0_unix_ns"]), int(meta["t0_monotonic_ns"])
    lpn, odo = [], []
    for msg in reader.iter_mavlink():
        t = msg.get_type()
        if t == "LOCAL_POSITION_NED":
            lpn.append((float(msg._timestamp), np.array([msg.x, msg.y, msg.z], float)))
        elif t == "ODOMETRY":
            odo.append((float(msg._timestamp), [float(v) for v in msg.q]))
    lpn.sort(key=lambda r: r[0]); odo.sort(key=lambda r: r[0])
    odo_t = np.array([o[0] for o in odo])
    # choose the LPN sample whose x is closest to pick_x
    xs = np.array([p[1][0] for p in lpn])
    i = int(np.argmin(np.abs(xs - pick_x)))
    recv, pos = lpn[i]
    j = int(np.argmin(np.abs(odo_t - recv)))
    q = odo[j][1]
    R_wb = R_wb_from_quat(q)
    rng = float(np.linalg.norm(GATE0_MAP - pos))
    corners, center = gate_inner_corners_center_hypo()
    uvs = []
    for w in corners + [center, GATE0_MAP]:
        pc = world_point_in_camera(w, R_wb, pos)
        uv = project_camera_point(pc)
        uvs.append(uv)
    cu = [u for u in uvs[:4] if u is not None]
    msg_sz = "OFF-IMAGE/behind" if len(cu) < 4 else (
        f"inner_px_w~{max(u[0] for u in cu)-min(u[0] for u in cu):.1f} "
        f"inner_px_h~{max(u[1] for u in cu)-min(u[1] for u in cu):.1f}")
    print(f"\n[{label}] pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) range_to_gate={rng:.2f}m")
    print(f"   corners(center-hypo) uv: {[None if u is None else (round(u[0]),round(u[1])) for u in uvs[:4]]}")
    print(f"   gate-CENTER uv: {None if uvs[4] is None else (round(uvs[4][0]),round(uvs[4][1]))}  "
          f"map-POINT uv: {None if uvs[5] is None else (round(uvs[5][0]),round(uvs[5][1]))}")
    print(f"   apparent size: {msg_sz}   (image 640x360)")

    # decode + save the frame nearest this recv (dedup by frame_id)
    best = None
    for e in reader.iter_video_index():
        fr_unix = (t0u + (e["recv_monotonic_ns"] - t0m)) / 1e9
        if best is None or abs(fr_unix - recv) < best[0]:
            best = (abs(fr_unix - recv), e)
    e = best[1]
    with open(run_dir / "video.bin", "rb") as f:
        f.seek(e["offset"]); jpeg = f.read(e["length"])
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    # draw projected gate (center hypo) for the eyeball check
    vis = img.copy()
    if all(u is not None for u in uvs[:4]):
        pts = np.array([uvs[k] for k in (0, 1, 3, 2)], np.int32)  # order to a quad
        cv2.polylines(vis, [pts], True, (0, 255, 0), 1)
    if uvs[4] is not None:
        cv2.drawMarker(vis, (int(uvs[4][0]), int(uvs[4][1])), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
    name_raw = OUT / f"proj_{label}_raw.png"
    name_vis = OUT / f"proj_{label}_overlay.png"
    cv2.imwrite(str(name_raw), img)
    cv2.imwrite(str(name_vis), vis)
    print(f"   frame {e['frame_id']} {w}x{h}  saved {name_raw.name} + {name_vis.name}")


if __name__ == "__main__":
    project(RUNS / "20260602_000538_firstcontact_race1", 0.0, "firstcontact_origin")
    project(RUNS / "20260604_025347_gate0_course1", -18.0, "course1_x-18")
    project(RUNS / "20260604_025347_gate0_course1", -10.0, "course1_x-10")
    project(RUNS / "20260604_025347_gate0_course1", -4.0, "course1_x-4")
