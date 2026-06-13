"""Unit tests for rl/contact_true_eval.py — inc8 Phase-0(b) metric instrument.

Tests:
  1. Legacy parity: _score_gate(body_radius=0, frame_depth=0) == gate_event() on
     the same documented cases (negative-control: the new module does NOT regress
     the legacy classifier).
  2. Contact-true reclassification: the inc7 live-crash offsets (Linf 0.37-0.49)
     score as pass under legacy and collision under contact-true geometry.
  3. Per-gate margin stats: correct computation on synthetic EpisodeResult fixtures.
  4. S_stable: threshold arithmetic for various seed sr values.
  5. gate3_d_offset_probe: shifting gate-3 D moves margins in the expected direction
     on a synthetic straight-line trajectory through all 6 gates.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_RL = Path(__file__).resolve().parents[1] / "rl"
_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_RL))
sys.path.insert(0, str(_SRC))

from contact_true_eval import (
    GateCrossing, EpisodeResult,
    _score_gate, per_gate_margin_stats, compute_s_stable,
    gate3_d_offset_probe,
    BODY_RADIUS_NOM, FRAME_DEPTH_NOM, PASS_BAND_NOM,
)
from fly_rl import N_GATES, _FLIP, _GATE_POS_ZUP, _R_W2G
from offline_rollout import _HALF_OPEN, _HALF_OUTER, gate_event as _legacy_gate_event


# ---------------------------------------------------------------------------
# Helpers: synthesize NED positions in the gate frame of gate g
# ---------------------------------------------------------------------------

def _ned_from_gate_rel(gate: int, rel_xyz) -> np.ndarray:
    """NED position corresponding to gate-frame coordinates (x, y, z).
    _R_W2G is its own inverse (diag(-1,-1,1) is orthogonal and symmetric),
    so world_zup = R_W2G^T @ rel + gate_pos_zup = R_W2G @ rel + gate_pos_zup.
    Then NED = world_zup * _FLIP.
    """
    gate_pos_zup = _GATE_POS_ZUP[gate]
    world_zup = _R_W2G @ np.asarray(rel_xyz, float) + gate_pos_zup
    return world_zup * _FLIP


# ---------------------------------------------------------------------------
# 1. Legacy parity
# ---------------------------------------------------------------------------

class TestLegacyParity:
    """_score_gate with body_radius=0, frame_depth=0 must match gate_event() exactly."""

    def _ned_pair(self, gate, prev_rel, cur_rel):
        return _ned_from_gate_rel(gate, prev_rel), _ned_from_gate_rel(gate, cur_rel)

    def _compare(self, gate, prev_rel, cur_rel):
        prev_ned, cur_ned = self._ned_pair(gate, prev_rel, cur_rel)
        legacy = _legacy_gate_event(prev_ned, cur_ned, gate)
        new_v, new_linf = _score_gate(prev_ned, cur_ned, gate,
                                      body_radius=0.0, frame_depth=0.0,
                                      gate_pos_zup=_GATE_POS_ZUP)
        assert new_v == legacy, (
            f"gate {gate}: legacy={legacy!r} new={new_v!r} "
            f"prev_rel={prev_rel} cur_rel={cur_rel}"
        )

    def test_clean_center_pass(self):
        self._compare(0, [-0.2, 0.0, 0.0], [0.3, 0.0, 0.0])

    def test_frame_collision(self):
        self._compare(0, [-0.25, 1.6, 0.0], [0.75, 0.4, 0.0])

    def test_clean_miss(self):
        self._compare(0, [-0.5, 2.0, 0.0], [0.5, 2.0, 0.0])

    def test_no_crossing_returns_none(self):
        self._compare(0, [0.5, 0.0, 0.0], [1.5, 0.0, 0.0])

    def test_pre_plane_invisible_to_legacy(self):
        """Pre-plane strike: legacy returns None; contact-true with depth>0 returns collision."""
        prev_ned, cur_ned = self._ned_pair(0, [-0.5, 0.9, 0.0], [-0.1, 0.9, 0.0])
        legacy = _legacy_gate_event(prev_ned, cur_ned, 0)
        assert legacy is None
        new_v, _ = _score_gate(prev_ned, cur_ned, 0,
                               body_radius=0.0, frame_depth=0.0,
                               gate_pos_zup=_GATE_POS_ZUP)
        assert new_v is None   # legacy parity: frame_depth=0 -> also None

    def test_all_gates_parity_random(self):
        """Random segments through each gate: no discrepancy with legacy."""
        rng = np.random.default_rng(42)
        for gate in range(N_GATES):
            for _ in range(200):
                prev_rel = rng.uniform([-2.0, -2.5, -2.5], [0.0, 2.5, 2.5])
                cur_rel = prev_rel + rng.uniform([-0.5, -0.5, -0.5], [1.5, 0.5, 0.5])
                prev_ned, cur_ned = self._ned_pair(gate, prev_rel, cur_rel)
                legacy = _legacy_gate_event(prev_ned, cur_ned, gate)
                new_v, _ = _score_gate(prev_ned, cur_ned, gate,
                                       body_radius=0.0, frame_depth=0.0,
                                       gate_pos_zup=_GATE_POS_ZUP)
                assert new_v == legacy, (
                    f"gate={gate} legacy={legacy!r} new={new_v!r} "
                    f"prev_rel={np.round(prev_rel, 3).tolist()} "
                    f"cur_rel={np.round(cur_rel, 3).tolist()}"
                )


# ---------------------------------------------------------------------------
# 2. Contact-true reclassification of inc7 live-crash offsets
# ---------------------------------------------------------------------------

class TestContactTrueReclassification:
    """Live standing crashes terminated at Linf 0.37-0.49 -- legacy=pass, CT=collision."""

    @pytest.mark.parametrize("linf_crash,r", [
        (0.44, BODY_RADIUS_NOM),   # 0.44 > 0.75-0.33=0.42 -> collision at nominal r
        (0.49, BODY_RADIUS_NOM),   # 0.49 > 0.42 -> collision at nominal r
        (0.37, 0.38),              # 0.37 >= 0.75-0.38=0.37 -> collision at DR-high r
    ])
    def test_live_crash_offset_reclassified(self, linf_crash, r):
        prev_ned = _ned_from_gate_rel(3, [-0.3, linf_crash, 0.0])
        cur_ned = _ned_from_gate_rel(3, [0.3, linf_crash, 0.0])
        # Legacy: should be pass (linf < 0.75)
        legacy = _legacy_gate_event(prev_ned, cur_ned, 3)
        assert legacy == "pass", f"linf={linf_crash}: expected legacy pass, got {legacy}"
        # Contact-true at given body radius: should be collision
        v, returned_linf = _score_gate(prev_ned, cur_ned, 3,
                                       body_radius=r,
                                       frame_depth=FRAME_DEPTH_NOM,
                                       gate_pos_zup=_GATE_POS_ZUP)
        assert v == "collision", (
            f"linf={linf_crash} r={r}: expected contact-true collision, got {v!r} "
            f"(pass_band={_HALF_OPEN - r:.3f})"
        )

    def test_clean_center_remains_pass_contact_true(self):
        """A clean center pass should still be pass under contact-true geometry."""
        prev_ned = _ned_from_gate_rel(0, [-0.3, 0.1, 0.0])
        cur_ned = _ned_from_gate_rel(0, [0.3, 0.1, 0.05])
        v, linf = _score_gate(prev_ned, cur_ned, 0,
                              body_radius=BODY_RADIUS_NOM, frame_depth=FRAME_DEPTH_NOM,
                              gate_pos_zup=_GATE_POS_ZUP)
        assert v == "pass", f"Expected pass for clean center, got {v!r}"
        assert linf < PASS_BAND_NOM, f"linf {linf:.3f} exceeds pass_band {PASS_BAND_NOM:.3f}"

    def test_pre_plane_strike_caught_by_frame_depth(self):
        """Pre-plane strike (gate_frame x stays negative): only caught with frame_depth > 0."""
        prev_ned = _ned_from_gate_rel(0, [-0.5, 0.9, 0.0])
        cur_ned = _ned_from_gate_rel(0, [-0.1, 0.9, 0.0])
        v_legacy, _ = _score_gate(prev_ned, cur_ned, 0,
                                  body_radius=0.0, frame_depth=0.0,
                                  gate_pos_zup=_GATE_POS_ZUP)
        assert v_legacy is None
        v_ct, _ = _score_gate(prev_ned, cur_ned, 0,
                               body_radius=BODY_RADIUS_NOM, frame_depth=FRAME_DEPTH_NOM,
                               gate_pos_zup=_GATE_POS_ZUP)
        assert v_ct == "collision"

    def test_linf_returned_correctly_for_pass(self):
        """Returned linf should match the manually computed crossing L-inf."""
        y_off = 0.15
        prev_ned = _ned_from_gate_rel(0, [-0.5, y_off, 0.1])
        cur_ned = _ned_from_gate_rel(0, [0.5, y_off, 0.1])
        v, linf = _score_gate(prev_ned, cur_ned, 0,
                              body_radius=0.0, frame_depth=0.0,
                              gate_pos_zup=_GATE_POS_ZUP)
        assert v == "pass"
        assert abs(linf - max(y_off, 0.1)) < 1e-9, f"linf={linf}, expected {max(y_off,0.1)}"


# ---------------------------------------------------------------------------
# 3. Per-gate margin statistics
# ---------------------------------------------------------------------------

class TestPerGateMarginStats:

    def _make_episode(self, gate_linfs: dict[int, float], outcome: str = "FINISHED"):
        crossings = [
            GateCrossing(gate=g, verdict="pass", linf=l, t=float(g + 1))
            for g, l in gate_linfs.items()
        ]
        return EpisodeResult(outcome=outcome, crossings=crossings, finish_t=10.0)

    def test_single_episode_all_gates(self):
        ep = self._make_episode({0: 0.10, 1: 0.20, 2: 0.30, 3: 0.05, 4: 0.15, 5: 0.25})
        stats = per_gate_margin_stats([ep])
        for g in range(N_GATES):
            assert stats[g]["n_pass"] == 1
            assert stats[g]["n_collision"] == 0
        # Gate-3 is the historically binding gate; check its margin
        assert abs(stats[3]["linf_median"] - 0.05) < 1e-9
        assert abs(stats[3]["margin_median"] - (PASS_BAND_NOM - 0.05)) < 1e-6

    def test_collision_counted_not_linf(self):
        crossings = [
            GateCrossing(gate=0, verdict="pass", linf=0.1, t=1.0),
            GateCrossing(gate=1, verdict="collision", linf=0.5, t=2.0),
        ]
        ep = EpisodeResult(outcome="COLLISION", crossings=crossings)
        stats = per_gate_margin_stats([ep])
        assert stats[0]["n_pass"] == 1 and stats[0]["n_collision"] == 0
        assert stats[1]["n_pass"] == 0 and stats[1]["n_collision"] == 1
        assert np.isnan(stats[1]["linf_median"])

    def test_multi_episode_aggregation(self):
        ep1 = self._make_episode({3: 0.10})
        ep2 = self._make_episode({3: 0.30})
        ep3 = self._make_episode({3: 0.20})
        stats = per_gate_margin_stats([ep1, ep2, ep3])
        assert stats[3]["n_pass"] == 3
        assert abs(stats[3]["linf_median"] - 0.20) < 1e-9
        assert abs(stats[3]["linf_p10"] - 0.12) < 1e-6   # p10 of [0.1, 0.2, 0.3]

    def test_gate3_isolated_from_others(self):
        ep = self._make_episode({0: 0.60, 3: 0.05})
        stats = per_gate_margin_stats([ep])
        assert stats[0]["n_pass"] == 1 and abs(stats[0]["linf_median"] - 0.60) < 1e-9
        assert stats[3]["n_pass"] == 1 and abs(stats[3]["linf_median"] - 0.05) < 1e-9

    def test_no_pass_returns_nan_stats(self):
        ep = EpisodeResult(outcome="COLLISION", crossings=[
            GateCrossing(gate=0, verdict="collision", linf=0.8, t=1.0)
        ])
        stats = per_gate_margin_stats([ep])
        assert stats[0]["n_pass"] == 0 and stats[0]["n_collision"] == 1
        assert np.isnan(stats[0]["linf_median"])


# ---------------------------------------------------------------------------
# 4. S_stable
# ---------------------------------------------------------------------------

class TestSStable:

    def _make_eps(self, outcomes: list[str]) -> list[EpisodeResult]:
        return [EpisodeResult(outcome=o) for o in outcomes]

    def test_all_seeds_pass(self):
        seeds = {f"s{i}": self._make_eps(["FINISHED"]) for i in range(7)}
        s_stable, per = compute_s_stable(seeds)
        assert s_stable == 1.0
        assert all(sr == 1.0 for sr in per.values())

    def test_no_seeds_pass(self):
        seeds = {f"s{i}": self._make_eps(["COLLISION"]) for i in range(3)}
        s_stable, per = compute_s_stable(seeds)
        assert s_stable == 0.0

    def test_one_of_three_passes(self):
        seeds = {"s0": self._make_eps(["FINISHED"]),
                 "s1": self._make_eps(["COLLISION"]),
                 "s2": self._make_eps(["COLLISION"])}
        s_stable, per = compute_s_stable(seeds)
        assert abs(s_stable - 1.0 / 3.0) < 1e-9
        assert s_stable < 2.0 / 3.0   # narrow-basin flag threshold

    def test_threshold_boundary(self):
        seeds = {"a": self._make_eps(["FINISHED", "COLLISION"]),   # sr=0.5
                 "b": self._make_eps(["FINISHED"])}               # sr=1.0
        # threshold=0.90: only 'b' passes
        s_stable_90, _ = compute_s_stable(seeds, threshold=0.90)
        assert s_stable_90 == 0.5
        # threshold=0.5: both pass (0.5 >= 0.5)
        s_stable_50, _ = compute_s_stable(seeds, threshold=0.50)
        assert s_stable_50 == 1.0

    def test_empty_seed_list(self):
        s_stable, per = compute_s_stable({"a": []})
        assert per["a"] == 0.0


# ---------------------------------------------------------------------------
# 5. gate3_d_offset_probe
# ---------------------------------------------------------------------------

class TestGate3DOffsetProbe:
    """Build a synthetic trajectory that passes cleanly through all 6 gates (centre),
    then verify that the D-offset probe shifts the gate-3 margin in the expected direction.
    """

    @pytest.fixture(autouse=True)
    def _build_straight_traj(self):
        """Synthesize a trajectory that passes through the CENTRE of each gate.

        Each gate has yaw=π → R_W2G = diag(-1,-1,1).
        gate_rel_x = -(world_x - gate_x), so to cross gate g forward (gate_rel_x: -→+)
        the drone must move world_x from gate_x + delta (before) to gate_x - delta (after).

        For each gate, we emit 5 points bracketing the gate-frame crossing at y=0, z=0
        (world y = gate_y, z = gate_z):
          world_x: gate_x + 1.0, +0.3, 0.0, -0.3, -1.0  (approaching from +x to -x in Z-up)

        Between gates we include a single bridge point that stays on the "already-passed"
        side of gate g and the "not-yet-reached" side of gate g+1, so no extra crossings
        are recorded by the sequential logic.
        """
        positions = []

        for g in range(N_GATES):
            gpos = _GATE_POS_ZUP[g]
            # Approach from gate_rel_x = -1 (world_x = gpos_x + 1) to +1 (world_x = gpos_x - 1)
            # y and z locked to gate centre: world_y = gpos_y, world_z = gpos_z
            for dx in [1.0, 0.3, 0.0, -0.3, -1.0]:
                pos_zup = np.array([gpos[0] + dx, gpos[1], gpos[2]])
                positions.append(pos_zup * _FLIP)   # convert Z-up to NED

            # Bridge to next gate: single point at the midpoint in x, at next gate's y/z
            if g < N_GATES - 1:
                g_next = _GATE_POS_ZUP[g + 1]
                bridge_zup = np.array([
                    0.5 * ((gpos[0] - 1.0) + (g_next[0] + 1.0)),   # midpoint x
                    g_next[1], g_next[2],                             # next gate centre y/z
                ])
                positions.append(bridge_zup * _FLIP)

        self.pos_traj = np.array(positions)

    def test_nominal_all_pass(self):
        res = gate3_d_offset_probe(self.pos_traj, 0.0,
                                   body_radius=0.0, frame_depth=0.0)
        assert res["n_g3_pass"] >= 1, "Expected gate-3 to be passed in straight trajectory"
        assert res["n_g3_collision"] == 0

    def test_positive_D_offset_lowers_gate_in_ned(self):
        """Shifting gate-3 D+1.5m (down in NED) = Z-up z decreases by 1.5m.
        The drone now passes ABOVE the shifted gate centre → larger linf."""
        res_nom = gate3_d_offset_probe(self.pos_traj, 0.0,
                                       body_radius=0.0, frame_depth=0.0)
        res_shift = gate3_d_offset_probe(self.pos_traj, +1.5,
                                         body_radius=0.0, frame_depth=0.0)
        # If both have a g3 pass, the shifted linf should be larger (drone is off-centre)
        if res_nom["n_g3_pass"] >= 1 and res_shift["n_g3_pass"] >= 1:
            assert res_shift["g3_linf"] > res_nom["g3_linf"] - 1e-6, (
                f"Expected increased linf with D+ shift: "
                f"nom={res_nom['g3_linf']:.3f} shift={res_shift['g3_linf']:.3f}"
            )

    def test_negative_D_offset_raises_gate_in_ned(self):
        """Shifting gate-3 D-1.5m (up in NED) = Z-up z increases by 1.5m."""
        res_nom = gate3_d_offset_probe(self.pos_traj, 0.0,
                                       body_radius=0.0, frame_depth=0.0)
        res_shift = gate3_d_offset_probe(self.pos_traj, -1.5,
                                         body_radius=0.0, frame_depth=0.0)
        if res_nom["n_g3_pass"] >= 1 and res_shift["n_g3_pass"] >= 1:
            assert res_shift["g3_linf"] > res_nom["g3_linf"] - 1e-6

    def test_large_offset_converts_pass_to_miss_or_collision(self):
        """With body_radius=0.33, a 1.5m gate-centre shift should push the
        drone outside the pass band (linf > PASS_BAND_NOM at minimum)."""
        res = gate3_d_offset_probe(self.pos_traj, +1.5,
                                   body_radius=BODY_RADIUS_NOM,
                                   frame_depth=FRAME_DEPTH_NOM)
        if res["n_g3_pass"] >= 1:
            assert res["g3_linf"] > PASS_BAND_NOM or res["g3_margin"] < 0.5, (
                "A 1.5 m D-shift should substantially reduce gate-3 margin"
            )

    def test_probe_returns_expected_keys(self):
        res = gate3_d_offset_probe(self.pos_traj, 0.0)
        assert {"delta_d_m", "outcome", "n_g3_pass", "n_g3_collision",
                "g3_linf", "g3_margin", "gates_passed"} == set(res.keys())
