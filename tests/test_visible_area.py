"""GateObservation visible-area byproduct (RL egocentric obs contract, 2026-07-06):
range-free foreshortening ratio = inner-quad area / max-edge^2 (~|cos approach angle|)."""
import numpy as np

from racer.contracts import GateObservation


def _obs(corners, ids=None):
    return GateObservation(
        frame_id=0, sim_time_ns=0, corners_px=np.asarray(corners, dtype=float),
        corner_ids=(None if ids is None else np.asarray(ids)))


def test_headon_square_ratio_is_one():
    L = 100.0
    o = _obs([[0, 0], [L, 0], [L, L], [0, L]])
    assert abs(o.inner_area_px - L * L) < 1e-6
    assert abs(o.visible_area_ratio - 1.0) < 1e-9


def test_inplane_rotation_invariant():
    # a rotated (but un-foreshortened) square: area and max-edge scale together -> ratio stays 1
    L, th = 80.0, np.radians(30.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    sq = (np.array([[0, 0], [L, 0], [L, L], [0, L]], float) - L / 2) @ R.T + 50.0
    assert abs(_obs(sq).visible_area_ratio - 1.0) < 1e-9


def test_foreshortened_ratio_tracks_cos():
    # compress the vertical axis by cos(45deg): area = L*(L*c), max-edge = L -> ratio = c
    L, c = 100.0, float(np.cos(np.radians(45.0)))
    o = _obs([[0, 0], [L, 0], [L, L * c], [0, L * c]])
    assert abs(o.visible_area_ratio - c) < 1e-3


def test_three_corners_is_none():
    o = _obs([[0, 0], [100, 0], [100, 100]], ids=[0, 1, 2])
    assert o.inner_area_px is None and o.visible_area_ratio is None
