"""Offline gate mapper: robust primitives, cases A/B/C, schema round-trips, failure modes.

Acceptance thresholds come from the measured-noise calibration sweep (see
handoff/laptop-gate-mapper-2026-06-11/WRITEUP.md): case A pose-aided on the real VQ1
geometry lands ~0.5-0.7 m total (bias-dominated; ~0.35 m in-plane) uncorrected and
~0.1-0.25 m bias-corrected; case C (no pose) aligned ~1.3-4.9 m. Thresholds sit ~30-50%
above the observed multi-seed maxima so they catch regressions, not seed luck.
"""
import json

import numpy as np
import pytest

from racer.gate_mapper import (
    GateMapEstimate,
    GateSighting,
    MEASURED_FIX_BIAS_NED,
    MappedGate,
    MapperConfig,
    RelativeGateSighting,
    cluster_sightings,
    dump_sightings_json,
    estimate_map_no_pose,
    estimate_map_pose_aided,
    load_sightings_json,
    robust_position_estimate,
    robust_yaw_estimate,
    wrap_pi,
)
from racer.gate_mapper_synth import (
    MEASURED_NOISE,
    VQ1_TRACK_RECORDS,
    evaluate_map,
    generate_pose_aided_sightings,
    generate_relative_sightings,
    simulate_exploration_path,
    true_gates,
    visible_gate_indices,
)

CENTRES, YAWS = true_gates()


def _path(**kw):
    return simulate_exploration_path(CENTRES, **kw)


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
def test_wrap_pi_branch_cut():
    assert wrap_pi(np.pi) == pytest.approx(np.pi)
    assert wrap_pi(-np.pi) == pytest.approx(np.pi)        # (-pi, pi]
    assert wrap_pi(3 * np.pi / 2) == pytest.approx(-np.pi / 2)
    np.testing.assert_allclose(wrap_pi([0.1, -0.1]), [0.1, -0.1])


def test_robust_position_clean_gaussian():
    rng = np.random.default_rng(0)
    sigma = np.array([0.7, 0.5, 0.3])
    pts = rng.normal([10.0, -5.0, 2.0], sigma, size=(200, 3))
    mean, cov, mask, _ = robust_position_estimate(pts, sigma)
    assert np.linalg.norm(mean - [10, -5, 2]) < 0.2
    assert mask.sum() >= 190                              # barely rejects clean data
    assert np.all(np.diag(cov) < sigma**2 / 50)           # ~sigma^2/n scaling


def test_robust_position_survives_structured_outliers():
    # 20% wrong-gate-style outliers 24 m away (worse than the measured 5-15%)
    rng = np.random.default_rng(1)
    sigma = np.array([0.73, 0.47, 0.29])
    good = rng.normal(0.0, 1.0, size=(80, 3)) * sigma
    bad = np.tile([24.0, 0, 0], (20, 1)) + rng.normal(0, 1, (20, 3)) * sigma
    mean, _cov, mask, _ = robust_position_estimate(np.vstack([good, bad]), sigma)
    assert np.linalg.norm(mean) < 0.35
    assert not mask[80:].any()                            # every outlier rejected


def test_robust_position_leak_tail():
    # the measured leak: 3-8 m random-direction outliers at a few percent
    rng = np.random.default_rng(2)
    sigma = np.array([0.73, 0.47, 0.29])
    pts = rng.normal(0.0, 1.0, size=(100, 3)) * sigma
    for i in range(3):
        d = rng.normal(size=3)
        pts[i] = d / np.linalg.norm(d) * rng.uniform(3, 8)
    mean, _, mask, _ = robust_position_estimate(pts, sigma)
    assert np.linalg.norm(mean) < 0.3
    assert mask[3:].mean() > 0.95


def test_robust_position_few_points_and_empty():
    sigma = np.ones(3)
    mean, cov, mask, _ = robust_position_estimate(np.array([[1.0, 2, 3], [1.2, 2, 3]]), sigma)
    np.testing.assert_allclose(mean, [1.1, 2, 3])
    assert mask.all() and np.allclose(np.diag(cov), 1.0)
    with pytest.raises(ValueError):
        robust_position_estimate(np.zeros((0, 3)), sigma)


def test_robust_yaw_wrap_and_flips():
    rng = np.random.default_rng(3)
    sig = np.deg2rad(3.0)
    a = wrap_pi(np.pi + rng.normal(0, sig, 60))           # cluster straddling the +-pi cut
    a[:4] = wrap_pi(a[:4] + np.pi / 2)                    # catastrophic 90-deg flips
    yaw, std, mask = robust_yaw_estimate(a, sig)
    assert abs(wrap_pi(yaw - np.pi)) < np.deg2rad(1.5)
    assert std < np.deg2rad(2.0)
    assert not mask[:4].any()
    y_none, s_none, m = robust_yaw_estimate(np.zeros(0), sig)
    assert y_none is None and s_none is None and m.size == 0


def test_cluster_sightings_separation_and_order():
    rng = np.random.default_rng(4)
    pts, t = [], []
    for k, c in enumerate(CENTRES):
        pts.append(c + rng.normal(0, 0.5, (30, 3)))
        t.append(np.full(30, 10.0 * k) + np.arange(30) * 0.1)
    pts, t = np.vstack(pts), np.concatenate(t)
    labels = cluster_sightings(pts, np.argsort(t), radius_m=5.0)
    assert len(np.unique(labels)) == 6
    for k in range(6):                                    # ids in first-seen (course) order
        assert np.all(labels[30 * k: 30 * (k + 1)] == k)


# ---------------------------------------------------------------------------
# case A — pose-aided
# ---------------------------------------------------------------------------
def test_case_a_labeled_acceptance_measured_noise():
    """~1 exploration pass, measured noise+leak+assoc errors: all 6 gates, total error
    bias-dominated but bounded, IN-PLANE (what the 0.75 m half-opening budgets) well under."""
    for seed in range(3):
        s = generate_pose_aided_sightings(_path(), rng=np.random.default_rng(seed))
        ev = evaluate_map(estimate_map_pose_aided(s))
        assert ev["n_matched"] == 6 and ev["n_extra_est"] == 0
        assert ev["pos_err_max_m"] < 0.85
        assert ev["inplane_err_max_m"] < 0.55
        assert ev["yaw_err_max_deg"] < 5.0


def test_case_a_bias_correction_sign_and_gain():
    """The measured FIX bias enters gate measurements NEGATED; correction must remove it.
    Pin the sign end-to-end with a noise-free-but-biased generator."""
    noise = MEASURED_NOISE.scaled(leak_rate=0.0, assoc_error_rate=0.0)
    noise = noise.scaled(sigma_ned=np.full(3, 1e-3))
    s = generate_pose_aided_sightings(_path(), noise=noise, rng=np.random.default_rng(0))
    est = estimate_map_pose_aided(s)
    for g in est.gates:                                   # error == -bias exactly
        np.testing.assert_allclose(g.position_ned - CENTRES[g.gate_id],
                                   -MEASURED_FIX_BIAS_NED, atol=0.02)
    cfg = MapperConfig(bias_correction_ned=MEASURED_FIX_BIAS_NED)
    ev = evaluate_map(estimate_map_pose_aided(s, config=cfg))
    assert ev["pos_err_max_m"] < 0.02


def test_case_a_bias_corrected_acceptance():
    cfg = MapperConfig(bias_correction_ned=MEASURED_FIX_BIAS_NED)
    for seed in range(3):
        s = generate_pose_aided_sightings(_path(), rng=np.random.default_rng(seed))
        ev = evaluate_map(estimate_map_pose_aided(s, config=cfg))
        assert ev["pos_err_max_m"] < 0.45


def test_case_a_cluster_mode_ignores_labels():
    """Unlabeled clustering: association errors cannot exist (geometry is true to the gate
    actually seen), so the map matches the labeled run on clean ids and beats it when the
    upstream labels are garbage."""
    s = generate_pose_aided_sightings(_path(), rng=np.random.default_rng(7), labeled=False)
    est = estimate_map_pose_aided(s)
    assert est.diagnostics["association"] == "cluster"
    ev = evaluate_map(est)
    assert ev["n_matched"] == 6 and ev["pos_err_max_m"] < 0.9
    assert [g.gate_id for g in est.gates] == [0, 1, 2, 3, 4, 5]   # course order


def test_case_a_empty_input():
    est = estimate_map_pose_aided([])
    assert est.gates == [] and est.diagnostics["n_sightings"] == 0


def test_case_a_partial_coverage_phantom_label_group_merged():
    """Partial coverage hazard: an unseen gate's label can be 100% neighbour mislabels —
    a tight phantom at the WRONG gate that no internal outlier screen can catch. The
    geometry cross-check must fold it into the real gate instead of emitting a 24 m-off
    'gate 1' (measured 24-40 m map errors on quarter-lap sweeps before the fix)."""
    poses = _path()
    cut = poses[len(poses) // 4][0]                       # quarter lap: only gate 0 seen
    for seed in range(3):
        s = [x for x in generate_pose_aided_sightings(poses, rng=np.random.default_rng(seed))
             if x.t < cut]
        labs = {x.gate_id for x in s}
        est = estimate_map_pose_aided(s)
        ev = evaluate_map(est)
        assert ev["pos_err_max_m"] < 1.0                  # no phantom parked at a wrong gate
        if len(labs) > 1:                                 # mislabels existed -> were merged
            assert est.diagnostics["n_label_groups_merged"] >= 1
            assert len(est.gates) < len(labs)


# ---------------------------------------------------------------------------
# case B — prior fusion
# ---------------------------------------------------------------------------
def _rough_prior(rng, offset_m=2.0):
    recs = []
    for r in VQ1_TRACK_RECORDS:
        d = rng.normal(size=3)
        d = d / np.linalg.norm(d) * offset_m
        recs.append({**r, "position_ned": list(np.asarray(r["position_ned"]) + d)})
    return recs


def test_case_b_fusion_beats_rough_prior():
    rng = np.random.default_rng(11)
    prior = _rough_prior(rng, offset_m=2.0)
    prior_err = max(np.linalg.norm(np.asarray(p["position_ned"])
                                   - np.asarray(t["position_ned"]))
                    for p, t in zip(prior, VQ1_TRACK_RECORDS))
    assert prior_err > 1.5                                # the prior really is rough
    s = generate_pose_aided_sightings(_path(), rng=rng)
    ev = evaluate_map(estimate_map_pose_aided(s, prior_records=prior))
    assert ev["n_matched"] == 6
    assert ev["pos_err_max_m"] < 1.0                      # fused ~= measurement (n large)
    assert ev["pos_err_max_m"] < prior_err


def test_case_b_prior_trust_scales_fusion():
    """Tight prior + few sightings -> stay near prior; loose prior -> follow measurement."""
    rng = np.random.default_rng(12)
    prior = _rough_prior(np.random.default_rng(99), offset_m=2.0)
    # starve the mapper: only the first fifth of the lap (mostly gate-0 sightings)
    poses = _path()
    s = [x for x in generate_pose_aided_sightings(poses, rng=rng)
         if x.t < poses[len(poses) // 5][0]]
    est_tight = estimate_map_pose_aided(s, prior_records=prior,
                                        config=MapperConfig(prior_sigma_pos=0.05))
    est_loose = estimate_map_pose_aided(s, prior_records=prior,
                                        config=MapperConfig(prior_sigma_pos=50.0))
    prior_pos = {r["gate_id"]: np.asarray(r["position_ned"]) - [0, 0, 1.36]
                 for r in prior}
    g0_tight = next(g for g in est_tight.gates if g.gate_id == 0)
    g0_loose = next(g for g in est_loose.gates if g.gate_id == 0)
    d_tight = np.linalg.norm(g0_tight.position_ned - prior_pos[0])
    d_loose = np.linalg.norm(g0_loose.position_ned - prior_pos[0])
    assert d_tight < 0.15                                 # pinned to the trusted prior
    assert d_loose > 5 * d_tight                          # free to follow the data


def test_case_b_unseen_gates_pass_through_flagged():
    rng = np.random.default_rng(13)
    poses = _path()
    cut = poses[len(poses) // 2][0]
    s = [x for x in generate_pose_aided_sightings(poses, rng=rng) if x.t < cut]
    est = estimate_map_pose_aided(s, prior_records=VQ1_TRACK_RECORDS)
    assert len(est.gates) == 6
    tail_flags = [g.flags for g in est.gates if g.gate_id >= 5]
    assert any("prior_only" in f for f in tail_flags)
    ev = evaluate_map(est)
    assert ev["pos_err_max_m"] < 1.0                      # prior-only gates are exact here


def test_case_b_no_sightings_prior_passthrough():
    est = estimate_map_pose_aided([], prior_records=VQ1_TRACK_RECORDS)
    assert len(est.gates) == 6
    assert all("prior_only" in g.flags for g in est.gates)
    ev = evaluate_map(est)
    assert ev["pos_err_max_m"] < 1e-6


# ---------------------------------------------------------------------------
# case C — no pose
# ---------------------------------------------------------------------------
STRAIGHT_RECORDS = [
    {"gate_id": i, "position_ned": [-20.0 - 24.0 * i, 0.0, 3.0],
     "orientation_ned_wxyz": [0.7071067811865476, 0.0, 0.0, 0.7071067811865476],
     "width_m": 2.72, "height_m": 2.72} for i in range(3)
]


def test_case_c_clean_noise_converges_tight():
    """Straight 3-gate course (no path corner hidden inside a blind gap): with near-zero
    noise the adjustment must be near-exact — a residual systematic here means an
    assembly/sign bug. On the REAL course the smoothness bridge has a noise-INDEPENDENT
    floor (~1.5 m: the const-accel bridge cuts the corner the path turns at the unseen
    transit); that physics lives in the acceptance test + the writeup, not here."""
    centres, _yaws = true_gates(STRAIGHT_RECORDS)
    noise = MEASURED_NOISE.scaled(0.1, leak_rate=0.0, assoc_error_rate=0.0)
    s = generate_relative_sightings(simulate_exploration_path(centres),
                                    records=STRAIGHT_RECORDS, noise=noise,
                                    rng=np.random.default_rng(0))
    # whiten with the TRUE (scaled) sigma — offline we calibrate sigma to the data; a
    # 10x-overstated sigma would over-weight the smoothness prior against clean obs
    cfg = MapperConfig(sigma_ned=noise.sigma_ned)
    est = estimate_map_no_pose(s, config=cfg, n_gates_hint=3)
    ev = evaluate_map(est, records=STRAIGHT_RECORDS, align_translation=True)
    assert ev["n_matched"] == 3 and ev["n_extra_est"] == 0
    assert ev["pos_err_max_m"] < 0.25
    # raw error is GAUGE: the anchor pose's true world position (start z ~1.6 m here) is
    # unknowable without pose — the map carries it as one rigid offset, nothing more
    ev_raw = evaluate_map(est, records=STRAIGHT_RECORDS, align_translation=False)
    assert ev_raw["pos_err_max_m"] < ev["pos_err_max_m"] + 2.0
    assert ev_raw["translation_offset_removed_m"] == 0.0


def test_case_c_acceptance_measured_noise():
    """Full measured noise incl. leak: degraded but BOUNDED (calibrated 1.3-4.9 m aligned)."""
    for seed in range(2):
        s = generate_relative_sightings(_path(scan_pitch_down_deg=25.0),
                                        rng=np.random.default_rng(100 + seed))
        est = estimate_map_no_pose(s, n_gates_hint=6)
        ev = evaluate_map(est, align_translation=True)
        ev_raw = evaluate_map(est, align_translation=False)
        assert ev["n_matched"] == 6
        assert ev["pos_err_max_m"] < 6.0
        assert ev_raw["pos_err_max_m"] < 10.0
        assert ev["yaw_err_max_deg"] < 6.0
        assert est.diagnostics["n_virtual_poses"] > 0     # the gap bridge engaged


def test_case_c_labels_used_for_naming_only():
    s = generate_relative_sightings(_path(scan_pitch_down_deg=25.0),
                                    rng=np.random.default_rng(5), labeled=True)
    est = estimate_map_no_pose(s, n_gates_hint=6)
    assert est.diagnostics["labels"] == "given(naming-only)"
    assert sorted(g.gate_id for g in est.gates) == [0, 1, 2, 3, 4, 5]
    ev = evaluate_map(est, align_translation=True)
    # naming must agree with geometry (majority vote over true-geometry association)
    for pg in ev["per_gate"]:
        assert pg["est_gate_id"] == pg["true_gate"]


def test_case_c_disconnected_without_bridge_is_flagged():
    """No smoothness prior + no co-visibility => the honest failure mode: multiple
    components, 'disconnected' flags, a warning — and the numbers are garbage, which is
    exactly why the flags must exist."""
    cfg = MapperConfig(accel_prior_sigma=None)
    s = generate_relative_sightings(_path(), rng=np.random.default_rng(6))
    est = estimate_map_no_pose(s, config=cfg, n_gates_hint=6)
    d = est.diagnostics
    assert d["covis_components"] > 1
    assert "warning_connectivity" in d
    assert any("disconnected" in g.flags for g in est.gates)


def test_case_c_empty_and_tiny_inputs():
    assert estimate_map_no_pose([]).gates == []
    q = np.array([1.0, 0, 0, 0])
    s = [RelativeGateSighting(frame_id=i, rel_position_frd=np.array([10.0, 0, 0]),
                              quat_wxyz=q, t=0.07 * i) for i in range(5)]
    est = estimate_map_no_pose(s)
    assert len(est.gates) == 1
    np.testing.assert_allclose(est.gates[0].position_ned, [10, 0, 0], atol=1e-6)


def test_case_c_lever_world_rotates_through_attitude():
    q_yaw90 = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])  # yaw +90 deg
    s = RelativeGateSighting(frame_id=0, rel_position_frd=np.array([5.0, 0.0, 0.0]),
                             quat_wxyz=q_yaw90)
    np.testing.assert_allclose(s.lever_world(), [0.0, 5.0, 0.0], atol=1e-12)


# ---------------------------------------------------------------------------
# schema seams
# ---------------------------------------------------------------------------
def test_truth_parse_matches_live_map_loader():
    """The synth truth / prior parse mirrors navigator.gates_from_track_records
    (corner_to_center=True) — centres AND through-direction must agree exactly."""
    from racer.navigator import gates_from_track_records

    live = gates_from_track_records(VQ1_TRACK_RECORDS, corner_to_center=True)
    for g, c, y in zip(live, CENTRES, YAWS):
        np.testing.assert_allclose(g.position_ned, c, atol=1e-9)
        thru = g.R_world_gate[:, 2]
        assert abs(wrap_pi(np.arctan2(thru[1], thru[0]) - y)) < 1e-9


def test_track_records_roundtrip_through_live_loader(tmp_path):
    """Mapper output -> capture-schema JSON -> navigator.load_track_map must reproduce the
    estimated opening centres + through-yaws (the consume-interchangeably contract)."""
    from racer.navigator import load_track_map

    gates = [MappedGate(gate_id=i, position_ned=CENTRES[i], yaw=float(YAWS[i]),
                        pos_cov=np.eye(3) * 0.01, yaw_std=0.01,
                        n_sightings=50, n_used=48) for i in range(6)]
    est = GateMapEstimate(gates=gates, diagnostics={"mode": "test"})
    p = est.save_track_map_json(tmp_path / "map.json", note="roundtrip test")
    live = load_track_map(p, corner_to_center=True)
    assert [g.gate_id for g in live] == [0, 1, 2, 3, 4, 5]
    for lg, mg in zip(live, gates):
        np.testing.assert_allclose(lg.position_ned, mg.position_ned, atol=1e-6)
        thru = lg.R_world_gate[:, 2]
        assert abs(wrap_pi(np.arctan2(thru[1], thru[0]) - mg.yaw)) < 1e-6
    payload = json.loads(p.read_text())
    assert payload["num_gates"] == 6 and len(payload["mapper"]["per_gate"]) == 6


def test_estimated_map_roundtrip_close_loop():
    """End-to-end: noisy sightings -> mapper -> records -> mapper's own prior parse;
    the reloaded centres must match the estimate (no convention drift)."""
    s = generate_pose_aided_sightings(_path(), rng=np.random.default_rng(21))
    est = estimate_map_pose_aided(s)
    from racer.gate_mapper import _parse_prior_records

    re = _parse_prior_records(est.to_track_records())
    for g, c in zip(est.gates, re["pos"]):
        np.testing.assert_allclose(g.position_ned, c, atol=1e-9)


def test_sightings_json_roundtrip(tmp_path):
    sa = generate_pose_aided_sightings(_path(), rng=np.random.default_rng(31))[:25]
    pa = dump_sightings_json(tmp_path / "a.json", sa, note="t")
    mode, back = load_sightings_json(pa)
    assert mode == "pose_aided" and len(back) == 25
    np.testing.assert_allclose(back[3].gate_world_ned, sa[3].gate_world_ned)
    assert back[3].gate_id == sa[3].gate_id and back[3].t == pytest.approx(sa[3].t)

    sc = generate_relative_sightings(_path(), rng=np.random.default_rng(32))[:25]
    pc = dump_sightings_json(tmp_path / "c.json", sc)
    mode, back = load_sightings_json(pc)
    assert mode == "relative" and len(back) == 25
    np.testing.assert_allclose(back[7].rel_position_frd, sc[7].rel_position_frd)
    np.testing.assert_allclose(back[7].quat_wxyz, sc[7].quat_wxyz)
    with pytest.raises(ValueError):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"schema": "nope", "sightings": []}))
        load_sightings_json(bad)


# ---------------------------------------------------------------------------
# generator sanity (the validation harness is only as honest as its generator)
# ---------------------------------------------------------------------------
def test_generator_determinism_and_noise_stats():
    a1 = generate_pose_aided_sightings(_path(), rng=np.random.default_rng(42))
    a2 = generate_pose_aided_sightings(_path(), rng=np.random.default_rng(42))
    assert len(a1) == len(a2)
    np.testing.assert_allclose(a1[10].gate_world_ned, a2[10].gate_world_ned)
    # empirical fix-error stats over many sightings ~ the measured model (sign: -e)
    noise = MEASURED_NOISE.scaled(leak_rate=0.0, assoc_error_rate=0.0)
    errs = []
    for seed in range(4):
        for s in generate_pose_aided_sightings(_path(), noise=noise,
                                               rng=np.random.default_rng(seed)):
            errs.append(s.gate_world_ned - CENTRES[s.gate_id])
    errs = np.asarray(errs)
    np.testing.assert_allclose(errs.mean(axis=0), -MEASURED_FIX_BIAS_NED, atol=0.12)
    np.testing.assert_allclose(errs.std(axis=0), MEASURED_NOISE.sigma_ned, rtol=0.25)


def test_generator_visibility_uses_camera_model():
    # a gate dead ahead at 10 m is visible; behind is not; beyond max range is not
    R_level = np.eye(3)
    c = np.array([[10.0, 0.0, -3.0]])                      # slightly above (camera tilts up)
    assert visible_gate_indices(np.zeros(3), R_level, c, MEASURED_NOISE) == [0]
    assert visible_gate_indices(np.array([20.0, 0, -3.0]), R_level, c, MEASURED_NOISE) == []
    far = np.array([[MEASURED_NOISE.max_range_m + 5.0, 0.0, -3.0]])
    assert visible_gate_indices(np.zeros(3), R_level, far, MEASURED_NOISE) == []


def test_run_mapper_offline_cli(tmp_path):
    """End-to-end through the CLI seam: sightings file in -> live-loadable map out."""
    import subprocess
    import sys as _sys

    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    s = generate_pose_aided_sightings(_path(), rng=np.random.default_rng(50))
    sfile = dump_sightings_json(tmp_path / "s.json", s)
    out = tmp_path / "map.json"
    r = subprocess.run(
        [_sys.executable, str(root / "scripts" / "run_mapper_offline.py"), str(sfile),
         "--out", str(out), "--n-gates", "6", "--report", "--bias-correct"],
        capture_output=True, text=True, cwd=root, timeout=300)
    assert r.returncode == 0, r.stderr
    from racer.navigator import load_track_map

    gates = load_track_map(out, corner_to_center=True)
    assert len(gates) == 6
    for g in gates:
        assert min(np.linalg.norm(g.position_ned - c) for c in CENTRES) < 0.6

    # relative mode (case C) on a short two-gate slice, prior ignored with a notice
    poses = _path()
    sc = [x for x in generate_relative_sightings(poses, rng=np.random.default_rng(51))
          if x.t < poses[len(poses) // 3][0]]
    cfile = dump_sightings_json(tmp_path / "c.json", sc)
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"gates": VQ1_TRACK_RECORDS}))
    out2 = tmp_path / "map_c.json"
    r2 = subprocess.run(
        [_sys.executable, str(root / "scripts" / "run_mapper_offline.py"), str(cfile),
         "--out", str(out2), "--prior", str(prior)],
        capture_output=True, text=True, cwd=root, timeout=300)
    assert r2.returncode == 0, r2.stderr
    assert "ignored in relative" in r2.stderr
    assert len(load_track_map(out2, corner_to_center=True)) >= 2


def test_case_c_matches_case_a_information():
    """The same noise core feeds both generators: with pose RESTORED externally, case-C
    levers reproduce case-A implied gate measurements (frame/sign consistency)."""
    poses = _path()
    noise = MEASURED_NOISE.scaled(leak_rate=0.0, assoc_error_rate=0.0)
    sa = generate_pose_aided_sightings(poses, noise=noise, rng=np.random.default_rng(9))
    sc = generate_relative_sightings(poses, noise=noise, rng=np.random.default_rng(9),
                                     labeled=True)        # identical rng draw order
    assert len(sa) == len(sc)
    for a, c in zip(sa[:50], sc[:50]):
        pos = np.asarray(poses[c.frame_id][1])
        np.testing.assert_allclose(pos + c.lever_world(), a.gate_world_ned, atol=1e-9)
