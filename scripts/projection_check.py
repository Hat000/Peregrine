"""First-contact probe: validate frames.py + confirm the camera geometry.

Mostly OFFLINE: it validates our own projection math (the documented NED-body -> OpenCV-camera
+ 20-deg-up-tilt sign-error trap) and prints the camera geometry, including the corrected
VFoV (~58.7 deg -- the spec mislabels '90 deg VFoV', which is actually the HORIZONTAL FoV).

With --frame (a captured JPEG) or --from-recording (a session dir), it overlays the computed
horizon line + principal point on a real frame so you can EYEBALL that frames.py matches what
the sim actually shows -- the against-the-sim check available before the detector is trained.
(Full sim-truth validation = reproject a DETECTED gate's corners; that comes with the detector.)

This targets the SPEC camera model (VADR-TS-002 sec 3.8); it is independent of Elodin.

Usage:
  python scripts/projection_check.py
  python scripts/projection_check.py --frame data/runs/<...>/frame.jpg --out annotated.png
  python scripts/projection_check.py --from-recording data/runs/<stamp>_label
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from racer import frames

K = frames.CAMERA_INTRINSICS_K
CX, CY = K[0, 2], K[1, 2]
_LEVEL = frames.R_world_from_body(0.0, 0.0, 0.0)   # drone level, facing north
_ORIGIN = np.zeros(3)


def _project_world(p_world: np.ndarray):
    return frames.project_camera_point(frames.world_point_in_camera(p_world, _LEVEL, _ORIGIN))


def _geometry_report() -> None:
    hfov, vfov = frames.horizontal_fov_deg(), frames.vertical_fov_deg()
    lo, hi = frames.camera_elevation_band_deg()
    print("== camera model (spec VADR-TS-002 sec 3.8) ==")
    print(f"  image: {frames.IMAGE_WIDTH}x{frames.IMAGE_HEIGHT}  principal point: ({CX:.0f},{CY:.0f})")
    print(f"  fx=fy={K[0,0]:.0f}")
    print(f"  HFoV = {hfov:.2f} deg   VFoV = {vfov:.2f} deg")
    print(f"  --> the spec's '90 deg VFoV' is really the HORIZONTAL FoV; VFoV is ~{vfov:.1f} deg.")
    print(f"  camera tilt: +{np.rad2deg(frames.CAMERA_PITCH_RAD):.0f} deg up; "
          f"elevation band it sees: [{lo:+.1f}, {hi:+.1f}] deg (it 'looks UP').")


def _consistency_checks() -> bool:
    """Sign/convention checks on frames.py. Returns True iff all pass."""
    checks: list[tuple[str, bool, str]] = []

    # 1) level point straight ahead must land BELOW image centre (camera tilts up).
    uv = _project_world(np.array([10.0, 0.0, 0.0]))
    checks.append(("level-ahead point projects below centre (tilt-up sign)",
                   uv is not None and uv[1] > CY, f"v={uv[1]:.1f} vs cy={CY:.0f}" if uv else "None"))

    # 2) a point at +20 deg elevation ahead must land near centre (it's on the optical axis).
    up = 10.0 * np.tan(frames.CAMERA_PITCH_RAD)
    uv = _project_world(np.array([10.0, 0.0, -up]))   # NED: up is -z
    checks.append(("+20 deg-elevation point projects near centre",
                   uv is not None and abs(uv[1] - CY) < 5.0, f"v={uv[1]:.1f}" if uv else "None"))

    # 3) a point ahead and to the RIGHT must land right of centre.
    uv = _project_world(np.array([10.0, 5.0, 0.0]))
    checks.append(("right-of-vehicle point projects right of centre",
                   uv is not None and uv[0] > CX, f"u={uv[0]:.1f} vs cx={CX:.0f}" if uv else "None"))

    # 4) a point BEHIND the camera must not project.
    uv = _project_world(np.array([-10.0, 0.0, 0.0]))
    checks.append(("point behind the camera does not project", uv is None, str(uv)))

    # 5) pixel -> ray -> pixel round-trips.
    u0, v0 = 500.0, 120.0
    ray = np.linalg.inv(K) @ np.array([u0, v0, 1.0])
    uv = frames.project_camera_point(ray)
    checks.append(("pixel->ray->pixel round-trips",
                   uv is not None and abs(uv[0] - u0) < 1e-6 and abs(uv[1] - v0) < 1e-6, str(uv)))

    print("\n== frames.py consistency checks ==")
    ok = True
    for name, passed, detail in checks:
        ok = ok and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  ({detail})")
    return ok


def _horizon_row() -> float:
    """Image row where the 0-deg-elevation (level, straight-ahead) ray projects."""
    uv = _project_world(np.array([1000.0, 0.0, 0.0]))
    return uv[1] if uv else float("nan")


def _annotate(img, out_path: Path) -> None:
    import cv2

    vis = img.copy()
    h, w = vis.shape[:2]
    v_h = _horizon_row()
    if v_h == v_h and 0 <= v_h < h:   # not NaN, in-frame
        cv2.line(vis, (0, int(v_h)), (w, int(v_h)), (255, 255, 0), 1)
        cv2.putText(vis, f"horizon (0 deg elev) v={v_h:.0f}", (5, int(v_h) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    cv2.drawMarker(vis, (int(CX), int(CY)), (0, 0, 255), cv2.MARKER_CROSS, 16, 1)
    cv2.putText(vis, "principal pt / optical axis (+20 deg up)", (5, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.imwrite(str(out_path), vis)
    print(f"\nannotated frame -> {out_path}  (eyeball: does the cyan line sit on the real horizon?)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", help="captured JPEG/PNG to overlay the horizon on")
    ap.add_argument("--from-recording", help="session dir; overlays on its first frame")
    ap.add_argument("--out", default="projection_check_annotated.png")
    args = ap.parse_args()

    _geometry_report()
    ok = _consistency_checks()
    print(f"\nframes.py self-consistency: {'ALL PASS' if ok else 'FAILURES -- fix before trusting projection'}")

    img = None
    if args.frame:
        import cv2

        img = cv2.imread(args.frame)
        if img is None:
            print(f"could not read {args.frame}", file=sys.stderr)
    elif args.from_recording:
        from racer.recording import RecordingReader

        img = next((f.image_bgr for f in RecordingReader(args.from_recording).frames()), None)
        if img is None:
            print(f"no frames in {args.from_recording}", file=sys.stderr)
    if img is not None:
        _annotate(img, Path(args.out))

    print("\nNOTE: validates the SPEC camera model + our frames.py; full sim-truth check = reproject")
    print("a DETECTED gate's corners once the detector is trained. (Independent of Elodin.)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
