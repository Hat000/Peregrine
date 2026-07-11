"""EGO keypoint-persistence debounce — deploy side of the matched pair (Fengyou 2026-07-11).

Pins the fly_ego --ego-kp-persist contract [ego-deploy 2026-07-11]:

  * DIRECTIVE: gate info may transmit only after the keypoint-detectability condition has
    held for N CONSECUTIVE frames; a single miss resets the streak. Deploy "frame" = one
    fresh camera frame (frame_id changed) where detect_gate_lever returned an accepted
    pose; training twin = +env.ego_kp_persist_frames (33 ms estimator ticks).

  * The debounce (_kp_persist_step) is unit-tested as a PURE function: n<=1 == OFF ==
    bit-identical passthrough (pose AND None, streak untouched); n=2 suppresses exactly
    the first hit and transmits on the 2nd; ANY fresh-frame miss (None for any reason:
    valid_poses_empty / continuity_reject / coast) resets the streak to 0; n=3 needs
    three; the streak saturates at n (never grows unbounded); the suppressed flag is
    True exactly when an ACCEPTED pose was withheld.

  * Reset hook: the function is pure -- the gate-advance reset is the caller zeroing the
    streak (kp_streak = 0 beside seeker.reset()/last_lever_fid = None), mirroring
    training's self._kp_persist_count[advance] = 0. Pinned at the source level plus a
    unit test that a zeroed streak requires a full fresh run of N hits.

  * Source-structure pins (test_ego_takeoff_assist.py read_text precedent): the debounce
    is called exactly ONCE inside _fly_ego, strictly AFTER detect_gate_lever( and BEFORE
    builder.update( (so the builder's existing pose=None path absorbs suppressed frames
    with zero builder edits); the gate-advance block contains the streak reset; argparse
    default is 0 (OFF); meta.json records ego_kp_persist on the ego-only path.

  * OFF-path regression: tests/test_ego_deploy_actor.py / test_ego_deploy_obs.py /
    test_ego_takeoff_assist.py run unmodified -- their staying green IS the deploy
    OFF-path check (default 0 == today's behavior).
"""
import re
import sys
from pathlib import Path

_RL = Path(__file__).resolve().parents[1] / "rl"
if str(_RL) not in sys.path:
    sys.path.insert(0, str(_RL))

import fly_rl  # noqa: E402

_kp = fly_rl._kp_persist_step

# A sentinel standing in for an accepted GatePose (the function never introspects it).
_POSE = object()


def _run(seq, n, streak=0):
    """Drive the debounce over a hit/miss sequence; return (out, streaks, sups, streak)."""
    out, streaks, sups = [], [], []
    for p in seq:
        streak, pose_out, sup = _kp(streak, p, n)
        out.append(pose_out)
        streaks.append(streak)
        sups.append(sup)
    return out, streaks, sups, streak


# --------------------------------------------------------------------------- OFF == passthrough
def test_n0_and_n1_are_pure_passthrough():
    """n<=1 == OFF: pose and None pass through bit-identically and the streak is UNTOUCHED
    (whatever value it holds), so default 0 is byte-identical to today's behavior."""
    for n in (0, 1):
        for streak_in in (0, 3, 7):
            s, p, sup = _kp(streak_in, _POSE, n)
            assert (s, p, sup) == (streak_in, _POSE, False)
            s, p, sup = _kp(streak_in, None, n)
            assert (s, p, sup) == (streak_in, None, False)


def test_negative_n_is_off_too():
    """Defensive: any n <= 1 (incl. a bogus negative) is OFF, never an always-suppress."""
    s, p, sup = _kp(0, _POSE, -3)
    assert (s, p, sup) == (0, _POSE, False)


# --------------------------------------------------------------------------- consecutive hits
def test_n2_first_hit_suppressed_second_transmits():
    """[P,P,P] at n=2 -> (suppressed, pass, pass): transmission begins ON the 2nd
    consecutive hit and continues while the streak holds."""
    out, streaks, sups, _ = _run([_POSE, _POSE, _POSE], n=2)
    assert out == [None, _POSE, _POSE]
    assert sups == [True, False, False]
    assert streaks == [1, 2, 2]                      # saturates at n


def test_n3_needs_three_consecutive_hits():
    out, _, sups, _ = _run([_POSE, _POSE, _POSE, _POSE], n=3)
    assert out == [None, None, _POSE, _POSE]
    assert sups == [True, True, False, False]


# --------------------------------------------------------------------------- miss-reset
def test_miss_resets_streak_to_zero():
    """[P,None,P,P] at n=2 -> (suppressed, reset-to-0, suppressed, pass): ANY fresh-frame
    miss (detector None for any reason) restarts the debounce from scratch."""
    out, streaks, sups, _ = _run([_POSE, None, _POSE, _POSE], n=2)
    assert out == [None, None, None, _POSE]
    assert streaks == [1, 0, 1, 2]
    assert sups == [True, False, True, False]        # a miss is NOT 'suppressed' (nothing withheld)


def test_miss_after_transmitting_requires_full_restreak():
    """Once transmitting, a single miss drops back to a full N-hit re-streak (no grace)."""
    out, _, _, _ = _run([_POSE, _POSE, None, _POSE, _POSE], n=2)
    assert out == [None, _POSE, None, None, _POSE]


# --------------------------------------------------------------------------- saturation
def test_streak_saturates_at_n():
    """The counter clamps at n: a long hit run never grows the streak past n, and recovery
    after a miss still takes exactly n fresh hits."""
    _, streaks, _, streak = _run([_POSE] * 10, n=2)
    assert max(streaks) == 2 and streak == 2
    out, streaks2, _, _ = _run([None, _POSE, _POSE], n=2, streak=streak)
    assert streaks2[0] == 0                          # miss zeroes even a saturated streak
    assert out == [None, None, _POSE]                # full re-streak required


def test_suppressed_flag_only_when_pose_withheld():
    """suppressed is True exactly when an ACCEPTED pose was converted to None -- never on
    a miss, never on a transmitted pose, never when OFF."""
    for n, seq, expect in [
        (2, [_POSE, _POSE, None, _POSE], [True, False, False, True]),
        (1, [_POSE, None], [False, False]),
        (3, [None, _POSE, _POSE, _POSE], [False, True, True, False]),
    ]:
        _, _, sups, _ = _run(seq, n=n)
        assert sups == expect


# --------------------------------------------------------------------------- reset hook (unit)
def test_external_streak_reset_models_gate_advance():
    """The gate-advance hook zeroes the caller-held streak (pure function -> the reset IS
    the caller passing 0): a rebuilt streak then needs the full N hits for the NEW gate."""
    _, _, _, streak = _run([_POSE, _POSE], n=2)
    assert streak == 2                               # transmitting on the old gate
    streak = 0                                       # <- the _fly_ego gate-advance reset
    out, _, _, _ = _run([_POSE, _POSE], n=2, streak=streak)
    assert out == [None, _POSE]                      # new gate: first hit suppressed again


# --------------------------------------------------------------------------- source-structure pins
def _ego_slice(src):
    return src[src.index("def _fly_ego("):src.index("def _fly_armed(")]


def test_debounce_wired_once_between_detect_and_builder_update():
    """Within _fly_ego the debounce is called exactly ONCE, strictly AFTER the
    detect_gate_lever( call and BEFORE builder.update( -- so the builder receives None on
    suppressed frames through its existing pose=None path (zero builder edits)."""
    src = (_RL / "fly_rl.py").read_text(encoding="utf-8")
    ego = _ego_slice(src)
    assert ego.count("_kp_persist_step(") == 1
    assert (ego.index("detect_gate_lever(")
            < ego.index("_kp_persist_step(")
            < ego.index("builder.update("))


def test_gate_advance_block_resets_the_streak():
    """kp_streak = 0 appears inside the gate-advance block (between seeker.reset() and the
    next _fresh_frame occurrence) -- the deploy mirror of training's
    self._kp_persist_count[advance] = 0 (matched-pair rule)."""
    src = (_RL / "fly_rl.py").read_text(encoding="utf-8")
    ego = _ego_slice(src)
    adv_start = ego.index("seeker.reset()")
    adv_end = ego.index("_fresh_frame", adv_start)
    assert "kp_streak = 0" in ego[adv_start:adv_end]


def test_argparse_default_is_off():
    """--ego-kp-persist exists with type=int, default=0 (0 or 1 == OFF == today)."""
    src = (_RL / "fly_rl.py").read_text(encoding="utf-8")
    assert re.search(
        r'ap\.add_argument\(\s*"--ego-kp-persist",\s*type=int,\s*default=0\b', src), \
        "--ego-kp-persist must exist with default=0 (OFF)"


def test_meta_records_kp_persist_on_ego_path_only():
    """meta.json gets "ego_kp_persist": args.ego_kp_persist inside the ego-only dict (the
    getattr(args, "ego_ckpt", None) guard keeps VQ1/gate-seeker meta byte-identical)."""
    src = (_RL / "fly_rl.py").read_text(encoding="utf-8")
    key = '"ego_kp_persist": args.ego_kp_persist'
    guard = 'if getattr(args, "ego_ckpt", None) else {}'
    assert key in src
    assert src.index('"ego_ckpt": str(args.ego_ckpt)') < src.index(key) < src.index(guard)
