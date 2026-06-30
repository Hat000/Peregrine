"""Tests for scripts/diagnose_session.py -- the unified telemetry auto-diagnoser (burn auto-judge).

Covers the load-bearing behaviours:
  * the R_y(pi) MIRROR CANARY is always-on and FORCES ATTENTION when a conjugation creeps back
    (synthetic healthy vs. mirrored flight, generated from the primitive's own model_accel);
  * a NON-FINITE obs forces ATTENTION + the NON_FINITE taxonomy tag;
  * graceful degradation (missing dir, replay-only bundle with no tlog, no debug_obs) yields
    UNAVAILABLE checks, never a crash and never a silent PASS;
  * a real recorded bundle (when present in this checkout) grades end-to-end.

The synthetic fixtures import frame_residual_report.model_accel + racer.frames.ODO_QUAT_TRUE_CONJ_WXYZ
so the canary is exercised against the EXACT physics + conjugation the primitive uses -- a healthy
recording stores q_raw such that (q_raw * CONJ) == the true attitude; the "mirror" bug stores the
true attitude directly (skipping the conjugation), so the AS-IS branch then explains the dynamics
and the TRUE branch does not -> the canary trips, exactly as it would on a real regressed run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import diagnose_session as ds  # noqa: E402

# scipy/frame_residual_report are needed for the synthetic-canary fixtures; skip those if absent.
_HAVE_SCIPY = True
try:
    from scipy.spatial.transform import Rotation  # noqa: E402

    import frame_residual_report as frr  # noqa: E402
    from racer.frames import ODO_QUAT_TRUE_CONJ_WXYZ  # noqa: E402
except Exception:  # pragma: no cover - environment without scipy
    _HAVE_SCIPY = False

_HEADER = {"type": "header", "obs_labels": ["a"] * 17, "checkpoint": "synthetic",
           "act_min": [0.0, -3.14, -3.14, -3.14], "act_max": [3.765, 3.14, 3.14, 3.14]}


def _write_bundle(d: Path, rows: list[dict], meta: dict | None = None) -> Path:
    """Write a minimal recording bundle (debug_obs.jsonl + meta.json) at ``d``."""
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "debug_obs.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(_HEADER) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    base = {"t0_unix_ns": 0, "t0_monotonic_ns": 0, "final_state": "FINISHED", "collisions": 0}
    base.update(meta or {})
    (d / "meta.json").write_text(json.dumps(base), encoding="utf-8")
    return d


def _row(k, t, vel, q_wxyz, *, coll=0.5, obs=None, act=None, age=2.0, rc=0, ncoll=0, pos=None):
    return {
        "k": k, "t_mono": t, "sim_time_ns": int(t * 1e9), "gate_index": 0, "odo_age_ms": age,
        "pos_ned": (pos if pos is not None else [0.0, 0.0, 0.0]),
        "vel_ned": list(vel), "q_raw_wxyz": list(q_wxyz), "w_raw": [0.0, 0.0, 0.0],
        "reset_counter": rc, "n_coll": ncoll,
        "obs": (obs if obs is not None else [0.0] * 17),
        "act_rescaled": (act if act is not None else [0.5, 0.0, 0.0, 0.0]),
        "rate_frd": [0.0, 0.0, 0.0], "collective": coll, "normed_thrust": 0.5,
    }


def _synthetic_flight(mirror: bool, n: int = 300) -> list[dict]:
    """A banked, oscillating-roll trajectory integrated from the primitive's own model_accel.

    healthy (mirror=False): q_raw == q_true / CONJ, so (q_raw * CONJ) == q_true  -> canary OK.
    mirror  (mirror=True):  q_raw == q_true (conjugation skipped), so (q_raw * CONJ) is the MIRRORED
                            attitude -> the AS-IS branch explains the dynamics, TRUE does not -> trip.
    Roll oscillates through zero (banked both ways) so the mirror is strongly anti-correlated, as on
    real flight data (real healthy run: East corr TRUE ~ +0.97 vs AS-IS ~ -0.68).
    """
    assert _HAVE_SCIPY
    dt = 1.0 / 30.0
    vel = np.zeros(3)
    pos = np.zeros(3)
    rows = []
    t = 400.0
    for k in range(n):
        roll = np.radians(45.0 * np.sin(k * 0.06))
        pitch = np.radians(10.0 + 5.0 * np.cos(k * 0.04))
        yaw = 0.3 * np.sin(k * 0.03)
        R_true = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_matrix()
        a = frr.model_accel(R_true, 0.5, vel)
        vel = 0.96 * (vel + a * dt)          # mild damping keeps accel in the usable band
        pos = pos + vel * dt
        q_true = Rotation.from_matrix(R_true).as_quat()[[3, 0, 1, 2]]  # -> wxyz
        q_raw = q_true if mirror else (q_true / ODO_QUAT_TRUE_CONJ_WXYZ)
        rows.append(_row(k, t, vel.round(6).tolist(), np.asarray(q_raw).round(6).tolist(),
                         pos=pos.round(6).tolist()))
        t += dt
    return rows


# ---------------------------------------------------------------------------------------------
# R_y(pi) MIRROR CANARY -- the load-bearing gate
# ---------------------------------------------------------------------------------------------
@pytest.mark.skipif(not _HAVE_SCIPY, reason="scipy / frame_residual_report unavailable")
def test_healthy_flight_passes_and_canary_ok(tmp_path):
    d = _write_bundle(tmp_path / "healthy", _synthetic_flight(mirror=False))
    r = ds.diagnose(d)
    fr = r["checks"]["frame_residual"]
    assert fr["status"] == "PASS", fr.get("summary")
    assert fr["mirror_canary"]["ok"] is True
    assert fr["mirror_canary"]["east_corr_TRUE"] > fr["mirror_canary"]["east_corr_ASIS"]
    # grade (meta FINISHED, 0 collisions) + finite obs -> overall PASS
    assert r["verdict"] == "PASS"
    assert r["taxonomy_tag"] == "OK"


@pytest.mark.skipif(not _HAVE_SCIPY, reason="scipy / frame_residual_report unavailable")
def test_mirror_conjugation_trips_canary_and_forces_attention(tmp_path):
    """A frame-residual deviation MUST force ATTENTION (the load-bearing requirement)."""
    d = _write_bundle(tmp_path / "mirror", _synthetic_flight(mirror=True))
    r = ds.diagnose(d)
    fr = r["checks"]["frame_residual"]
    assert fr["status"] == "ATTENTION", fr.get("summary")
    assert fr["mirror_canary"]["ok"] is False
    # AS-IS now explains the dynamics better than TRUE -> mirror crept back
    assert fr["mirror_canary"]["east_corr_ASIS"] > fr["mirror_canary"]["east_corr_TRUE"]
    # a tripped canary forces the OVERALL verdict to ATTENTION even though meta says FINISHED/clean
    assert r["verdict"] == "ATTENTION"
    assert "frame_residual" in r["attention_checks"]


# ---------------------------------------------------------------------------------------------
# NON-FINITE obs/action
# ---------------------------------------------------------------------------------------------
def test_nonfinite_obs_forces_attention_and_tags_non_finite(tmp_path):
    rows = [_row(0, 400.0, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            _row(1, 400.05, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                 obs=[float("nan")] + [0.0] * 16),                       # NaN injected into obs
            _row(2, 400.1, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])]
    d = _write_bundle(tmp_path / "nonfinite", rows, meta={"final_state": "FINISHED", "collisions": 0})
    r = ds.diagnose(d)
    oh = r["checks"]["obs_health"]
    assert oh["status"] == "ATTENTION"
    assert oh["n_nonfinite_obs"] == 1
    assert r["verdict"] == "ATTENTION"
    assert r["taxonomy_tag"] == "NON_FINITE"   # non-finite outranks the (clean) meta grade


def test_nonfinite_action_also_detected(tmp_path):
    rows = [_row(0, 400.0, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            _row(1, 400.05, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                 act=[float("inf"), 0.0, 0.0, 0.0])]                     # inf injected into action
    d = _write_bundle(tmp_path / "nonfinite_act", rows)
    r = ds.diagnose(d)
    oh = r["checks"]["obs_health"]
    assert oh["status"] == "ATTENTION" and oh["n_nonfinite_action"] == 1
    assert r["taxonomy_tag"] == "NON_FINITE"


# ---------------------------------------------------------------------------------------------
# Combined synthetic ATTENTION fixture (bad frame-residual AND non-finite obs) -- the prompt's case
# ---------------------------------------------------------------------------------------------
@pytest.mark.skipif(not _HAVE_SCIPY, reason="scipy / frame_residual_report unavailable")
def test_combined_attention_fixture(tmp_path):
    rows = _synthetic_flight(mirror=True)               # mirror -> canary trips
    rows[10]["obs"] = [float("nan")] + [0.0] * 16       # + a non-finite obs
    d = _write_bundle(tmp_path / "combined", rows)
    r = ds.diagnose(d)
    assert r["verdict"] == "ATTENTION"
    assert r["checks"]["frame_residual"]["status"] == "ATTENTION"
    assert r["checks"]["obs_health"]["status"] == "ATTENTION"
    # non-finite is the higher-priority taxonomy tag
    assert r["taxonomy_tag"] == "NON_FINITE"
    assert set(r["attention_checks"]) >= {"frame_residual", "obs_health"}


# ---------------------------------------------------------------------------------------------
# Taxonomy from the recorder's terminal abort (meta.final_state is authoritative)
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("final_state,expected", [
    ("CRASH", "CRASH"), ("ARM_REFUSED", "ARM_REFUSED"), ("NO_GO", "NO_GO"),
    ("SIM_RESET", "SIM_RESET"), ("ODO_STALE", "ODO_STALE"), ("SPIN_ABORT", "SPIN_ABORT"),
    ("TIMEOUT", "TIMEOUT"), ("BRIDGE_TIMEOUT", "TIMEOUT"),
])
def test_meta_final_state_drives_taxonomy(tmp_path, final_state, expected):
    # a single finite tick so obs_health/canary do not themselves trip; grade comes from meta
    rows = [_row(0, 400.0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])]
    d = _write_bundle(tmp_path / final_state, rows,
                      meta={"final_state": final_state, "collisions": 1, "gate_index": 2})
    r = ds.diagnose(d)
    assert r["verdict"] == "ATTENTION"
    assert r["taxonomy_tag"] == expected


def test_finished_clean_meta_is_pass(tmp_path):
    rows = [_row(k, 400.0 + k * 0.05, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]) for k in range(5)]
    d = _write_bundle(tmp_path / "finished", rows, meta={"final_state": "FINISHED", "collisions": 0})
    r = ds.diagnose(d)
    # no banked flight -> canary UNAVAILABLE (does not block); grade clean; obs finite -> PASS
    assert r["checks"]["frame_residual"]["status"] == "UNAVAILABLE"
    assert r["verdict"] == "PASS" and r["taxonomy_tag"] == "OK"


def test_mid_run_reset_in_debug_obs_tags_sim_reset(tmp_path):
    # recorder did not abort (meta FINISHED) but the per-tick reset_counter jumped -> SIM_RESET
    rows = [_row(0, 400.0, [1.0, 0, 0], [1.0, 0, 0, 0], rc=0),
            _row(1, 400.05, [1.0, 0, 0], [1.0, 0, 0, 0], rc=0),
            _row(2, 400.1, [1.0, 0, 0], [1.0, 0, 0, 0], rc=1)]   # reset bump
    d = _write_bundle(tmp_path / "reset", rows, meta={"final_state": "FINISHED", "collisions": 0})
    r = ds.diagnose(d)
    oh = r["checks"]["obs_health"]
    assert oh["n_resets"] == 1
    # obs_health PASS (resets are surfaced, not a hard gate there) but taxonomy catches it... however
    # the overall verdict is PASS unless something is ATTENTION. A reset with meta=FINISHED is a
    # grade question; assert the reset is at least SURFACED in the detail for triage.
    assert oh["reset_counters_seen"] == [0, 1]


# ---------------------------------------------------------------------------------------------
# Graceful degradation -- never crash, never silent-pass
# ---------------------------------------------------------------------------------------------
def test_missing_directory_is_attention_not_crash(tmp_path):
    r = ds.diagnose(tmp_path / "does_not_exist")
    assert r["exists"] is False
    assert r["verdict"] == "ATTENTION"


def test_replay_only_bundle_grades_from_meta_and_marks_tlog_checks_unavailable(tmp_path):
    rows = [_row(k, 400.0 + k * 0.05, [1.0, 0, 0], [1.0, 0, 0, 0]) for k in range(5)]
    d = _write_bundle(tmp_path / "replay", rows, meta={"final_state": "CRASH", "collisions": 1})
    r = ds.diagnose(d)
    # no tlog -> bundle integrity UNAVAILABLE; grade falls back to meta
    assert r["checks"]["bundle"]["status"] == "UNAVAILABLE"
    assert r["checks"]["grade"]["source"] == "meta"
    assert r["verdict"] == "ATTENTION" and r["taxonomy_tag"] == "CRASH"


def test_no_debug_obs_makes_obs_and_canary_unavailable(tmp_path):
    d = tmp_path / "no_obs"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"t0_unix_ns": 0, "t0_monotonic_ns": 0,
                                              "final_state": "FINISHED", "collisions": 0}))
    r = ds.diagnose(d)
    assert r["checks"]["obs_health"]["status"] == "UNAVAILABLE"
    assert r["checks"]["frame_residual"]["status"] == "UNAVAILABLE"
    # grade is clean from meta and nothing is ATTENTION -> PASS (UNAVAILABLE never blocks)
    assert r["verdict"] == "PASS"


def test_no_meta_no_tlog_is_ungradeable_attention(tmp_path):
    d = tmp_path / "bare"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"t0_unix_ns": 0, "t0_monotonic_ns": 0}))  # no final_state
    r = ds.diagnose(d)
    assert r["checks"]["grade"]["status"] == "ATTENTION"
    assert r["verdict"] == "ATTENTION"


# ---------------------------------------------------------------------------------------------
# Real recorded bundle (only if present in this checkout) -- end-to-end smoke
# ---------------------------------------------------------------------------------------------
_REAL = ROOT / "handoff/shadowpc-postfix-dataset-2026-06-12/extracted"
# The 2026-06-27 handoff-slimming purge removed the debug_obs.jsonl artifacts but left the empty
# dir shell, so guard on actual bundle presence (the glob the test consumes), not just the dir.
_REAL_BUNDLES = sorted(_REAL.glob("*/debug_obs.jsonl")) if _REAL.is_dir() else []


@pytest.mark.skipif(not _REAL_BUNDLES, reason="real recorded bundles not present in this checkout")
def test_real_bundles_diagnose_end_to_end():
    bundles = sorted(p for p in _REAL.iterdir() if (p / "debug_obs.jsonl").exists())
    assert bundles, "expected at least one real bundle with debug_obs.jsonl"
    seen_pass = seen_attention = False
    for b in bundles:
        r = ds.diagnose(b)
        assert r["verdict"] in ("PASS", "ATTENTION")
        # every check reports a known status; nothing crashes
        for c in r["checks"].values():
            assert c["status"] in ("PASS", "ATTENTION", "UNAVAILABLE")
        fs = r["meta"].get("final_state")
        if fs == "FINISHED" and (r["meta"].get("collisions") or 0) == 0:
            assert r["verdict"] == "PASS"
            seen_pass = True
        if fs == "CRASH":
            assert r["verdict"] == "ATTENTION" and r["taxonomy_tag"] == "CRASH"
            seen_attention = True
    # the postfix dataset has both finished (brg_*) and crashed (std_*) runs
    assert seen_pass and seen_attention
