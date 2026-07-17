"""Pins the v1smoke-3312029 DET_EVAL crash: _accum_yaw must reach the RAW env's ``_w`` /
``_yaw_cmd_clamp`` through DiffAero's RecordEpisodeStatistics guard wrapper, which raises on ANY
underscore-prefixed attribute from the top (see tests/test_unwrap_env.py for the captured wrapper).
Before the fix: ``env._w`` raised AttributeError (DET_EVAL: FAILED) and the clamp getattr silently
read 0.0. After: _unwrap_env_with drills to the holder; clamp and achieved-yaw come from the raw env.

Run from repo ROOT: .venv\\Scripts\\python.exe -m pytest tests/test_accum_yaw_unwrap.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rl"))
torch = pytest.importorskip("torch")


class _RawEnv:
    def __init__(self, n):
        self._w = torch.zeros((n, 3))
        self._w[:, 2] = 1.5
        self._yaw_cmd_clamp = 0.7


class _Guard:
    """Verbatim semantics of DiffAero's RecordEpisodeStatistics underscore guard."""

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        return getattr(self.env, name)


def _import_train_ego():
    import importlib
    return importlib.import_module("peregrine_train_ego")


def test_guard_raises_bare_access():
    g = _Guard(_RawEnv(4))
    with pytest.raises(AttributeError):
        _ = g._w  # the exact pre-fix failure


def test_accum_yaw_through_guard_no_raise_and_correct_clamp():
    te = _import_train_ego()
    n = 4
    wrapped = _Guard(_Guard(_RawEnv(n)))  # two levels, like TrainRunner chains
    # phys action [thrust, roll, pitch, yaw]; yaw = 2.0 rad/s, above the 0.7 clamp
    phys = torch.zeros((n, 4))
    phys[:, 3] = 2.0
    ys = te._accum_yaw(wrapped, phys, None, None)
    assert ys is not None and ys["cmd_n"] == n
    # clamp must have come from the RAW env (0.7), not the guard-swallowed 0.0:
    # post-clamp |cmd| sum = n * 0.7 (if clamp had read 0.0, clamp_yaw_command no-ops -> 2.0 each)
    assert abs(float(ys["cmd_abs_sum"]) - n * 0.7) < 1e-6, (
        f"clamp not applied from raw env: cmd_abs_sum={float(ys['cmd_abs_sum'])}")
    # achieved yaw must be the raw env's _w[..., 2]
    assert abs(float(ys["ach_abs_sum"]) - n * 1.5) < 1e-6


def test_accum_yaw_flip_counting_still_works_through_guard():
    te = _import_train_ego()
    n = 2
    wrapped = _Guard(_RawEnv(n))
    ys = None
    for k in range(6):
        phys = torch.zeros((n, 4))
        phys[:, 3] = 0.6 if k % 2 == 0 else -0.6  # alternate above deadband
        ys = te._accum_yaw(wrapped, phys, None, ys)
    # 6 steps alternating -> 5 flips per env
    assert float(ys["flips"].sum()) == 5.0 * n
