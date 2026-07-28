"""Tests for the v1.5 SPEED-DISCIPLINE + YAW-QUIETNESS package (2026-07-18). Five mechanisms, all
config-gated + DEFAULT-OFF so a config without the new keys trains BYTE-IDENTICAL to v1:

  A OVERSPEED EPISODE-ABORT   rl/peregrine_racing_ego.overspeed_abort_mask (pure trigger) + the env fold
                              into ``oob`` (OOB-class terminal: pays terminal_oob, KEEPS banked).
  B PROGRESS-CREDIT SATURATION rl/ego_reward.progress_credit_saturate + compute_ego_reward wiring.
  C YAW AMPLITUDE / DUTY       rl/ego_reward.yaw_duty_penalty (free band; champion 0.232 pays 0).
  D YAW JERK                   rl/ego_reward.yaw_jerk_penalty (L1 |delta yaw_cmd|).
  E EVAL OBSERVABILITY         YAW_EVAL satur_duty= / DET_EVAL max_speed= (append-only; source-format test).

Every reward piece is a PURE torch function (no diffaero), so B/C/D unit-test on the laptop directly and
wired through compute_ego_reward. The PeregrineRacingEgo env needs diffaero (cluster-only), so A is covered
by (a) the pure overspeed_abort_mask trigger and (b) the OOB-CLASS terminal SEMANTICS via terminal_penalty
(the env-level fold ``oob_full |= overspeed`` is a readable one-liner validated on the cluster smoke). E is
covered by a source-level assertion that the new fields are APPENDED at the END (old fields unmoved -- the
downstream-grep contract).

Run from repo ROOT:
    .venv\\Scripts\\python.exe -m pytest tests/test_v15_terms.py -q
"""
import re
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rl"))

import ego_reward as R                                                    # noqa: E402
import peregrine_racing_ego as C                                         # noqa: E402 (module import is offline-safe)

DT = torch.float64
DT_S = 1.0 / 30.0            # control dt (30 Hz)


def _t(x):
    return torch.tensor(x, dtype=DT)


# ================================================================================================
# B -- PROGRESS-CREDIT SATURATION (pure progress_credit_saturate).
# ================================================================================================
def test_saturation_below_vstar_unchanged():
    """At/below v* the positive credit is UNCHANGED (no reward lost in the useful speed band)."""
    # cap = progress(2.0) * vcap(7.5) * dt = 0.5. Credits below the cap are untouched.
    r = _t([0.0, 0.1, 0.4, 0.4999, 0.5])
    out = R.progress_credit_saturate(r, rw_progress=2.0, progress_vcap_mps=7.5, dt=DT_S)
    assert torch.equal(out, r)                    # EXACTLY unchanged at/below the cap


def test_saturation_above_vstar_capped():
    """Above v* the credit is CAPPED at rw_progress*v*·dt -- marginal credit for going faster is ZERO."""
    cap = 2.0 * 7.5 * DT_S                          # 0.5
    r = _t([0.5001, 0.667, 0.867, 1.30])           # ~10, 13, 19.5 m/s along-track (all > 7.5)
    out = R.progress_credit_saturate(r, rw_progress=2.0, progress_vcap_mps=7.5, dt=DT_S)
    assert torch.allclose(out, torch.full_like(r, cap))
    # marginal credit above v* is zero: two different fast speeds get the SAME credit
    assert out[0].item() == pytest.approx(out[-1].item())


def test_saturation_never_negative_leaves_backward_as_is():
    """SATURATION not penalty: it never turns a positive credit negative, and NEGATIVE progress (backing
    up the segment) is priced AS-IS (only the positive side is capped)."""
    r = _t([-1.0, -0.3, -0.001, 0.0, 0.2, 0.9])
    out = R.progress_credit_saturate(r, rw_progress=2.0, progress_vcap_mps=7.5, dt=DT_S)
    # negatives + sub-cap positives unchanged; the one above-cap positive (0.9) -> 0.5 (never negative)
    assert torch.equal(out[:5], r[:5])
    assert out[5].item() == pytest.approx(0.5)
    assert (out >= torch.minimum(r, torch.zeros_like(r))).all()    # never pushed below 0 from a positive


def test_saturation_off_is_v1_identical():
    """progress_vcap_mps<=0 -> the input tensor is returned UNCHANGED (byte-identical v1 math), even for
    credits that WOULD be capped when armed, and for negative caps."""
    r = _t([-0.3, 0.2, 0.667, 5.0])
    for vcap in (0.0, -1.0):
        out = R.progress_credit_saturate(r, rw_progress=2.0, progress_vcap_mps=vcap, dt=DT_S)
        assert torch.equal(out, r), vcap


# ------------------------------------------------------------------------------------------------
# B wired: below v* byte-identical to v1; above v* the credited progress is capped; component exposed.
# ------------------------------------------------------------------------------------------------
def _reward_kw(n, *, s_curr=None, s_prev=None, vel_world=None, yaw_cmd=None, yaw_clamp=0.0,
               yaw_cmd_delta=None):
    """Minimal compute_ego_reward kwargs; progress driven by s_curr-s_prev, all other terms inert. NO
    area_true/dist_to_gate -> the area coupling is OFF so r_prog == the (saturated) segment credit."""
    return dict(
        s_curr=(torch.zeros(n, dtype=DT) if s_curr is None else s_curr),
        s_prev=(torch.zeros(n, dtype=DT) if s_prev is None else s_prev),
        gate_passed=torch.zeros(n, dtype=torch.bool), pass_linf=torch.zeros(n, dtype=DT), w_g_half=0.375,
        gate_collision=torch.zeros(n, dtype=torch.bool), gate_miss=torch.zeros(n, dtype=torch.bool),
        oob=torch.zeros(n, dtype=torch.bool), banked_progress_return=torch.zeros(n, dtype=DT),
        newly_finished=torch.zeros(n, dtype=torch.bool), time_left_s=torch.zeros(n, dtype=DT),
        tilt_cos_r33=torch.ones(n, dtype=DT), omega=torch.zeros(n, 3, dtype=DT),
        action_norm=torch.full((n, 4), 0.5, dtype=DT), last_action_norm=torch.full((n, 4), 0.5, dtype=DT),
        vel_world=(torch.zeros(n, 3, dtype=DT) if vel_world is None else vel_world),
        curr_center=torch.zeros(n, 3, dtype=DT), next_center=torch.zeros(n, 3, dtype=DT), dt=DT_S,
        yaw_cmd=yaw_cmd, yaw_clamp=yaw_clamp, yaw_cmd_delta=yaw_cmd_delta,
    )


def test_saturation_wired_caps_fast_keeps_slow():
    """Wired into compute_ego_reward: a FAST along-track step is capped, a SLOW one is byte-identical to
    the OFF (v1) reward; the returned r_prog (banked) reflects the cap; component prog_sat_forfeit exposed."""
    # s_curr-s_prev = along-track advance this step: 0.1 (3 m/s), 0.333 (10 m/s > 7.5), 0.05 (1.5 m/s)
    sc, sp = _t([0.1, 0.333, 0.05]), torch.zeros(3, dtype=DT)
    w_on = R.EgoRewardWeights(progress=2.0, progress_vcap_mps=7.5)
    w_off = R.EgoRewardWeights(progress=2.0, progress_vcap_mps=0.0)
    r_on, c_on, rp_on = R.compute_ego_reward(w_on, **_reward_kw(3, s_curr=sc, s_prev=sp))
    r_off, c_off, rp_off = R.compute_ego_reward(w_off, **_reward_kw(3, s_curr=sc, s_prev=sp))
    # slow steps (idx 0, 2) identical; fast step (idx 1) capped 0.666 -> 0.5
    assert rp_off.tolist() == pytest.approx([0.2, 0.666, 0.1], abs=1e-9)
    assert rp_on.tolist() == pytest.approx([0.2, 0.5, 0.1], abs=1e-9)
    assert r_on[0].item() == pytest.approx(r_off[0].item())        # slow -> byte-parity
    assert r_on[1].item() < r_off[1].item()                        # fast -> capped (lower)
    assert c_on["prog_sat_forfeit"] == pytest.approx((0.666 - 0.5) / 3.0, abs=1e-6)
    assert c_off["prog_sat_forfeit"] == pytest.approx(0.0)         # OFF -> no forfeit


# ================================================================================================
# C -- YAW AMPLITUDE / DUTY (pure yaw_duty_penalty). The champion 0.232 absmean must pay EXACTLY 0.
# ================================================================================================
def test_duty_zero_inside_free_band_including_champion_0232():
    """|yaw_cmd| <= free_band pays EXACTLY 0 -- CRUCIALLY at the champion's 0.232 absmean (band 0.25), so
    the yaw->gate-in-view->altitude coupling is preserved (the hard rule)."""
    yc = _t([0.0, 0.1, 0.232, 0.25])               # all inside/at the 0.25 band
    r = R.yaw_duty_penalty(yc, rw_yaw_duty=0.15, free_band=0.25, clamp=0.7)
    assert torch.equal(r, torch.zeros_like(r)), r  # EXACTLY zero (relu -> 0 inside the band)


def test_duty_linear_above_band_to_rail():
    """Linear from 0 at the band to -rw_yaw_duty at the +-clamp rail: pen = rw*relu(|cmd|-band)/(clamp-band)."""
    rw, band, clamp = 0.15, 0.25, 0.7
    span = clamp - band                            # 0.45
    yc = _t([0.25, 0.475, 0.7, -0.7])              # band, midpoint, +rail, -rail (symmetric in |cmd|)
    r = R.yaw_duty_penalty(yc, rw, band, clamp)
    assert (-r[0]).item() == pytest.approx(0.0)
    assert (-r[1]).item() == pytest.approx(rw * (0.475 - band) / span)   # midpoint -> ~0.075
    assert (-r[2]).item() == pytest.approx(rw)                           # +rail -> rw
    assert (-r[3]).item() == pytest.approx(rw)                           # -rail -> rw (uses |cmd|)


def test_duty_off_is_inert():
    """rw_yaw_duty==0 -> zeros byte-identical, even at the rail."""
    yc = _t([0.7, -0.7, 0.5])
    r = R.yaw_duty_penalty(yc, rw_yaw_duty=0.0, free_band=0.25, clamp=0.7)
    assert torch.equal(r, torch.zeros_like(r))
    w = R.EgoRewardWeights()
    assert w.yaw_duty == 0.0 and w.yaw_duty_free_band == 0.25       # dataclass defaults


def test_duty_wired_component_and_champion_free():
    """Wired into compute_ego_reward: yaw_duty_pen component present; the champion's 0.232 pays 0 while the
    rail is priced; the reward is lowered by exactly the duty penalty."""
    yc = _t([0.232, 0.7])
    kw = _reward_kw(2, yaw_cmd=yc, yaw_clamp=0.7)
    w_on = R.EgoRewardWeights(yaw_duty=0.15, yaw_duty_free_band=0.25)
    w_off = R.EgoRewardWeights(yaw_duty=0.0)
    r_on, c_on, _ = R.compute_ego_reward(w_on, **kw)
    r_off, c_off, _ = R.compute_ego_reward(w_off, **_reward_kw(2, yaw_cmd=None))
    assert r_on[0].item() == pytest.approx(r_off[0].item())         # champion 0.232 -> byte-parity (free)
    assert (r_off[1] - r_on[1]).item() == pytest.approx(0.15)       # rail -> lowered by rw_yaw_duty
    assert c_on["yaw_duty_pen"] == pytest.approx((0.0 + 0.15) / 2.0)
    assert c_off["yaw_duty_pen"] == pytest.approx(0.0)


# ================================================================================================
# D -- YAW JERK (pure yaw_jerk_penalty). L1 |delta yaw_cmd|: bang-bang pays; steady pays 0.
# ================================================================================================
def test_jerk_bangbang_pays_per_flip_linear():
    """L1 penalty scale-linear in the flip amplitude: a +-rail flip (|delta|=1.4) pays 2x a half flip."""
    dd = _t([0.0, 0.7, 1.4, -1.4])                 # steady, half, full +rail flip, full -rail flip
    r = R.yaw_jerk_penalty(dd, rw_yaw_jerk=0.05)
    assert (-r[0]).item() == pytest.approx(0.0)                     # steady command -> 0
    assert (-r[1]).item() == pytest.approx(0.05 * 0.7)             # 0.035
    assert (-r[2]).item() == pytest.approx(0.05 * 1.4)             # 0.070 == 2x the half flip
    assert (-r[3]).item() == pytest.approx(0.05 * 1.4)             # |delta| -> sign-agnostic


def test_jerk_constant_cmd_pays_zero():
    """A CONSTANT (sustained) yaw command has delta==0 -> pays EXACTLY 0 (a needed turn is never taxed)."""
    dd = torch.zeros(4, dtype=DT)
    r = R.yaw_jerk_penalty(dd, rw_yaw_jerk=0.05)
    assert torch.equal(r, torch.zeros_like(r))


def test_jerk_off_is_inert():
    """rw_yaw_jerk==0 -> zeros byte-identical."""
    dd = _t([1.4, -1.4, 0.7])
    r = R.yaw_jerk_penalty(dd, rw_yaw_jerk=0.0)
    assert torch.equal(r, torch.zeros_like(r))
    assert R.EgoRewardWeights().yaw_jerk == 0.0


def test_jerk_wired_reuses_yaw_cmd_delta():
    """Wired: yaw_jerk reuses yaw_cmd_delta (the SAME post-clamp delta yaw_dither reads); component exposed;
    a rail flip lowers the reward by exactly rw_yaw_jerk*|delta|."""
    dd = _t([0.0, 1.4])
    kw = _reward_kw(2, yaw_cmd_delta=dd)
    w_on = R.EgoRewardWeights(yaw_jerk=0.05)
    w_off = R.EgoRewardWeights(yaw_jerk=0.0)
    r_on, c_on, _ = R.compute_ego_reward(w_on, **kw)
    r_off, _, _ = R.compute_ego_reward(w_off, **_reward_kw(2, yaw_cmd_delta=None))
    assert r_on[0].item() == pytest.approx(r_off[0].item())         # steady -> parity
    assert (r_off[1] - r_on[1]).item() == pytest.approx(0.05 * 1.4) # flip lowers reward by the L1 penalty
    assert c_on["yaw_jerk_pen"] == pytest.approx((0.0 + 0.05 * 1.4) / 2.0)


# ================================================================================================
# A -- OVERSPEED EPISODE-ABORT: pure trigger + OOB-class terminal SEMANTICS.
# ================================================================================================
def test_overspeed_trigger_fires_above_inert_at_zero():
    """overspeed_abort_mask: True strictly ABOVE the cap; the cap value itself is NOT over (uses >), and
    <=0 (OFF) is permanently False (byte-identical) for arbitrary speeds."""
    speed = _t([0.0, 3.0, 11.9, 12.0, 12.001, 30.0])
    fired = C.overspeed_abort_mask(speed, overspeed_abort_mps=12.0)
    assert fired.tolist() == [False, False, False, False, True, True]
    for off in (0.0, -1.0):
        assert torch.equal(C.overspeed_abort_mask(speed, off),
                           torch.zeros_like(speed, dtype=torch.bool)), off


def test_overspeed_termination_is_oob_class_keeps_banked():
    """termination class = OOB-LIKE: an overspeed folded into ``oob`` pays terminal_oob and KEEPS banked
    gate progress (death pricing UNCHANGED), UNLIKE a crash-class contact which forfeits banked. Proven via
    the pure terminal_penalty under the default progress-scaled terminal, in BOTH forfeit-mask regimes."""
    w = R.EgoRewardWeights()                        # terminal_progress_scaled=True; bases 200
    n = 3
    banked = _t([50.0, 50.0, 50.0])
    no = torch.zeros(n, dtype=torch.bool)
    yes = torch.ones(n, dtype=torch.bool)
    floor_only = torch.zeros(n, dtype=DT)           # nothing lethal this step (parabola regime mask)
    # overspeed routed as oob -> terminal_oob (200), NO banked forfeit (parabola-regime floor-only mask)
    pen_para = R.terminal_penalty(no, no, yes, banked, w, forfeit_mask=floor_only)
    assert torch.allclose(pen_para, _t([200.0, 200.0, 200.0]))
    # legacy-regime mask (None -> forfeit defaults to CONTACT-only) -> oob STILL keeps banked
    pen_legacy = R.terminal_penalty(no, no, yes, banked, w, forfeit_mask=None)
    assert torch.allclose(pen_legacy, _t([200.0, 200.0, 200.0]))
    # CONTRAST: a crash-class contact (in the forfeit mask) forfeits banked + base (250) -- overspeed does NOT
    pen_contact = R.terminal_penalty(yes, no, no, banked, w, forfeit_mask=yes.to(DT))
    assert torch.allclose(pen_contact, _t([250.0, 250.0, 250.0]))


# ================================================================================================
# BYTE-IDENTITY REGRESSION: a config with the new keys at defaults trains bit-identical to v1.
# ================================================================================================
def test_v15_defaults_byte_identical_to_v1_reward():
    """The whole package is DEFAULT-OFF: EgoRewardWeights() (v1.5 code) with ALL new terms at their
    defaults produces a BYTE-IDENTICAL reward tensor to an explicit-all-off config, across a mix of
    progress / speed / yaw inputs that WOULD trigger every new term if armed."""
    n = 4
    sc = _t([0.1, 0.5, -0.2, 0.9]); sp = torch.zeros(n, dtype=DT)
    vel = _t([[3.0, 0, 0], [13.0, 0, 0], [0, 20.0, 0], [8.0, 0, 0]])   # incl. >12 m/s
    yc = _t([0.232, 0.7, -0.7, 0.5])
    dd = _t([0.0, 1.4, -1.4, 0.7])
    kw = _reward_kw(n, s_curr=sc, s_prev=sp, vel_world=vel, yaw_cmd=yc, yaw_clamp=0.7, yaw_cmd_delta=dd)
    r_default, c_default, _ = R.compute_ego_reward(R.EgoRewardWeights(), **kw)
    r_off, _, _ = R.compute_ego_reward(
        R.EgoRewardWeights(progress_vcap_mps=0.0, yaw_duty=0.0, yaw_jerk=0.0), **kw)
    assert torch.equal(r_default, r_off)            # BYTE-IDENTICAL to v1
    # the new components are PRESENT and ZERO when off (observability, no training effect)
    for k in ("prog_sat_forfeit", "yaw_duty_pen", "yaw_jerk_pen"):
        assert c_default[k] == pytest.approx(0.0), k


# ================================================================================================
# E -- EVAL PRINT FORMAT (append-only). Source-level: new fields APPENDED at END, old fields UNMOVED
# (the downstream-grep contract). peregrine_train_ego imports diffaero, so we assert on the SOURCE.
# ================================================================================================
_TRAIN_SRC = (ROOT / "rl" / "peregrine_train_ego.py").read_text(encoding="utf-8")


def test_yaw_eval_appends_satur_duty_after_old_fields():
    """YAW_EVAL: satur_duty= is appended AFTER n_passed= (the last v1 field) -- old fields keep their order."""
    src = _TRAIN_SRC
    # the composed line ends with "... n_passed={n_passed:.4f} satur_duty={satur_duty:.4f}"
    assert "satur_duty={satur_duty:.4f}" in src
    # n_passed={..} is UNIQUE to YAW_EVAL; the satur_duty token is now SHARED by PITCH_EVAL / ROLL_EVAL
    # (which precede _emit_yaw_eval in file order), so scope the satur_duty search to AFTER n_passed to pin
    # the YAW line's OWN satur_duty (a bare src.index would grab the pitch line's earlier token).
    i_np = src.index("n_passed={n_passed:.4f}")
    i_sd = src.index("satur_duty={satur_duty:.4f}", i_np)
    assert i_np < i_sd, "satur_duty must come AFTER n_passed (append-only)"
    # old fields precede the new one, in their original order
    for tok in ("signflips_per_s=", "cmd_absmean=", "ach_absmean=", "roll_swing=", "n_passed="):
        assert src.index(tok) < i_sd, tok
    # the nan/no-data branch also carries satur_duty=nan (parseable in every branch)
    assert "n_passed=nan satur_duty=nan" in src


def test_det_eval_appends_max_speed_after_old_fields():
    """DET_EVAL: max_speed= is appended AFTER n_passed_gates= (the last v1 field), on both the episodes and
    no-episodes branches -- old fields unmoved."""
    src = _TRAIN_SRC
    assert "max_speed={max_speed:.2f}" in src
    i_npg = src.index("n_passed_gates={r['n_passed_gates']:.4f}")
    i_ms = src.index("max_speed={max_speed:.2f}")
    assert i_npg < i_ms, "max_speed must come AFTER n_passed_gates (append-only)"
    for tok in ("thread={r['success_rate']", "collision={r['collision_rate']",
                "miss={r['miss_rate']", "oob={r['oob_rate']"):
        assert src.index(tok) < i_ms, tok
    # no-episodes branch also carries max_speed
    assert "no episodes completed in {steps} steps max_speed={max_speed:.2f}" in src


def test_roll_eval_line_emitted_with_three_fields():
    """ROLL_EVAL (close-in roll limit-cycle fix, 2026-07-22): the greppable line carries the SAME three
    fields as PITCH_EVAL (signflips_per_s / cmd_absmean / satur_duty) in BOTH the data and the nan/no-data
    branch, and is WIRED into _run_det_eval alongside _emit_pitch_eval so checkpoint selection finally gates
    the roll limit cycle that currently passes invisibly."""
    src = _TRAIN_SRC
    # the composed DATA line (split across two f-string fragments, exactly as PITCH_EVAL is)
    assert "ROLL_EVAL[{label}] signflips_per_s={signflips_per_s:.3f} " in src
    assert "cmd_absmean={cmd_absmean:.4f} satur_duty={satur_duty:.4f}" in src
    # the nan/no-data branch is also parseable with the same three fields
    assert "ROLL_EVAL[{label}] signflips_per_s=nan cmd_absmean=nan satur_duty=nan" in src
    # WIRED: accumulated per step (channel 1) and emitted alongside the pitch line
    assert "rs = _accum_roll(env, phys, m, rs)" in src
    assert "_emit_roll_eval(env, label, steps, rs)" in src
