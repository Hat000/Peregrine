"""Unit tests for _unwrap_env_with / _require_anneal_holder (rl/peregrine_train_ego.py).

THE BUG THIS PINS (footgun L16, 2026-07-09): DiffAero's TrainRunner wraps the raw env in
RecordEpisodeStatistics (diffaero/utils/runner.py -- captured verbatim, read-only, from
C:/Users/Fengy/Downloads/Projects/Adroit/adroit-connector/command_history.log:37094-37110):

    class RecordEpisodeStatistics:
        def __init__(self, env, max_window_length=None):
            self.env = env
            ...
        def __getattr__(self, name):
            \"\"\"Returns an attribute with ``name``, unless ``name`` starts with an underscore.\"\"\"
            if name.startswith("_"):
                raise AttributeError(f"accessing private attribute '{name}' is prohibited")
            return getattr(self.env, name)

TrainRunner installs this AROUND the raw env (``self.env = RecordEpisodeStatistics(env)``,
command_history.log:37137), so ``hasattr(trainrunner.env, "_egorw")`` is ALWAYS False -- the wrapper's own
``__getattr__`` refuses ANY underscore-prefixed name outright; it never even reaches the forwarding
``getattr(self.env, name)`` line for those names. The 2026-07-08 cross-zero-anneal campaign used exactly
that hasattr-through-the-wrapper check and trained UNANNEALED the entire time with nobody noticing
(MEMORY.md footgun L16, exposed by a bit-identical dose-response pair). ``_unwrap_env_with``
(rl/peregrine_train_ego.py) works around it by walking the wrapper chain's OWN ``__dict__``
(``vars(e)['env']``, never ``getattr``) to find the object that actually owns the attribute, however many
levels down. ``_require_anneal_holder`` builds on it: a requested-but-unfindable anneal now RAISES
RuntimeError by default instead of silently disabling (opt out via ``+anneal_allow_skip=true``).

Run from repo ROOT: .venv\\Scripts\\python.exe -m pytest tests/test_unwrap_env.py -q
"""
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rl"))

torch = pytest.importorskip("torch")                          # peregrine_train_ego.py imports torch


def _ensure_submodule(dotted, parent, child_name, **attrs):
    """Idempotently register ``dotted`` in sys.modules (creating it if absent) and fill in any of
    ``attrs`` that aren't already set on it -- ADDITIVE, so this never clobbers a partial diffaero stub
    another test module (e.g. tests/test_measured_aero.py::_import_adapter) may already have installed
    earlier in this same pytest session; it only fills in whatever's still missing."""
    mod = sys.modules.get(dotted)
    if mod is None:
        mod = types.ModuleType(dotted)
        sys.modules[dotted] = mod
    for k, v in attrs.items():
        if not hasattr(mod, k):
            setattr(mod, k, v)
    if not hasattr(parent, child_name):
        setattr(parent, child_name, mod)
    return mod


def _install_diffaero_stub():
    """rl/peregrine_train_ego.py has UNGUARDED top-level ``import diffaero.*`` (it is the Adroit-only
    training launcher -- unlike e.g. inc8_warmstart.py / diffaero_dynamics.py, which guard theirs so they
    stay laptop-importable, see tests/test_inc8_warmstart.py). Stub just enough of the diffaero package
    tree (DYNAMICS_ALIAS / ENV_ALIAS dicts, a TrainRunner class with a ``.run`` method, a ``main``
    callable) for the module to import on a diffaero-less laptop -- mirrors the stub pattern in
    tests/test_measured_aero.py::_import_adapter. We only need the pure-Python
    _unwrap_env_with / _require_anneal_holder helpers; nothing here needs to actually train.

    IMPORTANT cross-file ordering note: importing peregrine_train_ego.py transitively imports
    diffaero_dynamics.py (``from diffaero_dynamics import PeregrinePlantDynamics``), which -- since
    diffaero_dynamics.py's OWN diffaero import is try/except-guarded -- silently falls back to
    ``BaseDynamics = object`` unless ``diffaero.dynamics.base_dynamics.BaseDynamics`` is ALREADY
    registered before that import runs. diffaero_dynamics.py is a module-level singleton (imported once,
    cached in sys.modules for the rest of the pytest session), so if THIS test file's collection is what
    first triggers that import, an incomplete stub here would permanently poison
    tests/test_measured_aero.py's PeregrinePlantDynamics instantiation tests for the whole session
    (object.__init__ rejects the extra (cfg, device) args) regardless of what
    tests/test_measured_aero.py::_import_adapter installs afterwards -- the binding is already baked in.
    We therefore install the SAME faithful BaseDynamics stand-in test_measured_aero.py uses (verbatim),
    so whichever test file imports diffaero_dynamics first, the result is identical and functional."""
    diffaero = sys.modules.get("diffaero")
    if diffaero is None:
        diffaero = types.ModuleType("diffaero")
        sys.modules["diffaero"] = diffaero

    _ensure_submodule("diffaero.algo", diffaero, "algo")
    dyn = _ensure_submodule("diffaero.dynamics", diffaero, "dynamics", DYNAMICS_ALIAS={})

    class _StubBaseDynamics:
        """Verbatim mirror of tests/test_measured_aero.py::_import_adapter's BaseDynamics stand-in
        (value-faithful: real grad_decay only scales gradients)."""
        def __init__(self, cfg, device):
            self.n_agents = int(getattr(cfg, "n_agents", 1))
            self.n_envs = int(getattr(cfg, "n_envs", 1))
            self.dt = float(cfg.dt)
            self.alpha = float(getattr(cfg, "alpha", 1.0))
            self.device = device

        def grad_decay(self, x):
            return x

        def detach(self):
            self._state = self._state.detach()

    base_mod = _ensure_submodule("diffaero.dynamics.base_dynamics", dyn, "base_dynamics",
                                  BaseDynamics=_StubBaseDynamics)
    if not hasattr(dyn, "base_dynamics"):
        dyn.base_dynamics = base_mod

    _ensure_submodule("diffaero.env", diffaero, "env", ENV_ALIAS={})
    utils = _ensure_submodule("diffaero.utils", diffaero, "utils")

    class _StubTrainRunner:
        def run(self):
            raise NotImplementedError("stub TrainRunner.run -- not exercised by this test")

    _ensure_submodule("diffaero.utils.runner", utils, "runner", TrainRunner=_StubTrainRunner)
    script = _ensure_submodule("diffaero.script", diffaero, "script")
    _ensure_submodule("diffaero.script.train", script, "train", main=lambda *a, **k: None)


_install_diffaero_stub()
import peregrine_train_ego as ego_launcher                                # noqa: E402


# ========================================================================================================
# Mocks: RecordEpisodeStatistics's __getattr__ semantics (verbatim, see module docstring), a second
# generic passthrough layer (some OTHER wrapper DiffAero might insert -- makes the chain 2 levels deep,
# not 1), and a raw env that owns the attribute directly.
# ========================================================================================================
class _MockRecordEpisodeStatistics:
    """Mirrors diffaero.utils.runner.RecordEpisodeStatistics.__getattr__ verbatim (the private-attribute
    guard is the whole point here; __init__ is trimmed to just what matters for this test)."""
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        if name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        return getattr(self.env, name)


class _PassthroughWrapper:
    """A second, generic wrapper layer with NO special __getattr__ -- stands in for whatever else DiffAero
    might insert between RecordEpisodeStatistics and the raw env, so the chain _unwrap_env_with has to
    drill through is genuinely 2 levels deep (RecordEpisodeStatistics -> this -> raw env), not just 1."""
    def __init__(self, env):
        self.env = env


class _RawEnv:
    """Stands in for the raw peregrine_racing_ego env instance, which owns ``_egorw`` / ``_estimator``
    directly in its OWN __dict__ (no wrapping)."""
    pass


class _Cfg:
    """Minimal attribute bag standing in for the resolved hydra cfg (getattr access only)."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _build_chain(attr_name=None, attr_value=None):
    """Builds RecordEpisodeStatistics(Passthrough(RawEnv)) -- the 2-level chain -- and, if ``attr_name``
    is given, sets it directly on the RawEnv. Returns (outer_wrapper, raw_env)."""
    raw = _RawEnv()
    if attr_name is not None:
        setattr(raw, attr_name, attr_value)
    middle = _PassthroughWrapper(raw)
    outer = _MockRecordEpisodeStatistics(middle)
    return outer, raw


# ================================================================================================
# (a) _unwrap_env_with finds the raw holder through a 2-level chain.
# ================================================================================================
def test_unwrap_env_with_finds_holder_through_two_level_chain():
    sentinel = object()
    outer, raw = _build_chain("_egorw", sentinel)
    holder = ego_launcher._unwrap_env_with(outer, "_egorw")
    assert holder is raw
    assert holder._egorw is sentinel


def test_unwrap_env_with_finds_estimator_holder_too():
    sentinel = object()
    outer, raw = _build_chain("_estimator", sentinel)
    holder = ego_launcher._unwrap_env_with(outer, "_estimator")
    assert holder is raw


# ================================================================================================
# (b) returns None (or the strict path raises) when absent.
# ================================================================================================
def test_unwrap_env_with_returns_none_when_absent():
    outer, _raw = _build_chain()                       # no _egorw anywhere in the chain
    assert ego_launcher._unwrap_env_with(outer, "_egorw") is None


def test_unwrap_env_with_returns_none_on_bare_none_env():
    assert ego_launcher._unwrap_env_with(None, "_egorw") is None


def test_unwrap_env_with_gives_up_past_max_depth():
    # a chain deeper than max_depth (default 12) never reaches the holder.
    raw = _RawEnv()
    raw._egorw = object()
    node = raw
    for _ in range(15):                                # > default max_depth=12
        node = _PassthroughWrapper(node)
    assert ego_launcher._unwrap_env_with(node, "_egorw") is None
    # sanity: a SHALLOWER version of the same chain DOES find it.
    node2 = raw
    for _ in range(5):
        node2 = _PassthroughWrapper(node2)
    assert ego_launcher._unwrap_env_with(node2, "_egorw") is raw


def test_require_anneal_holder_raises_by_default_when_requested_but_absent():
    """footgun L16: a requested-but-unfindable anneal must kill the run loudly, not silently skip."""
    outer, _raw = _build_chain()                       # no _egorw anywhere -> holder not found
    sched = {"start": 0.0, "end": 1.0, "hold_frac": 0.1, "n_updates": 100}
    with pytest.raises(RuntimeError, match="_egorw.*holder not found"):
        ego_launcher._require_anneal_holder(outer, "_egorw", sched, "cross-zero-anneal", _Cfg())


def test_require_anneal_holder_allow_skip_opts_back_into_silent_skip():
    outer, _raw = _build_chain()
    sched = {"start": 0.0, "end": 1.0, "hold_frac": 0.1, "n_updates": 100}
    result = ego_launcher._require_anneal_holder(
        outer, "_egorw", sched, "cross-zero-anneal", _Cfg(anneal_allow_skip=True))
    assert result is None


def test_require_anneal_holder_is_a_noop_when_anneal_not_requested():
    """sched is None (the anneal wasn't requested at all) -> no lookup, no raise -- the byte-identical OFF
    path. Uses a chain with NO holder at all to prove the None-sched short-circuit never even looks."""
    outer, _raw = _build_chain()
    assert ego_launcher._require_anneal_holder(outer, "_egorw", None, "cross-zero-anneal", _Cfg()) is None


def test_require_anneal_holder_returns_the_real_holder_when_present():
    sentinel = object()
    outer, raw = _build_chain("_egorw", sentinel)
    sched = {"start": 0.0, "end": 1.0, "hold_frac": 0.1, "n_updates": 100}
    result = ego_launcher._require_anneal_holder(outer, "_egorw", sched, "cross-zero-anneal", _Cfg())
    assert result is raw


# ================================================================================================
# (c) hasattr on the wrapper is False for underscore attrs (documents the original bug).
# ================================================================================================
def test_hasattr_is_false_for_underscore_attrs_on_the_wrapper():
    """The ROOT CAUSE _unwrap_env_with works around: a naive ``hasattr(trainrunner.env, "_egorw")`` check
    is ALWAYS False for a RecordEpisodeStatistics-wrapped env, even though the raw env genuinely owns
    ``_egorw`` -- the wrapper's __getattr__ refuses any underscore-prefixed name outright, it never even
    reaches the ``getattr(self.env, name)`` forwarding line. This is exactly what silently disabled the
    cross-zero anneal for the whole 2026-07-08 campaign (MEMORY.md footgun L16)."""
    sentinel = object()
    outer, raw = _build_chain("_egorw", sentinel)
    assert hasattr(raw, "_egorw") is True                     # the raw env genuinely has it
    assert hasattr(outer, "_egorw") is False                  # but the wrapper hides it entirely

    # non-underscore attributes DO forward correctly through a single __getattr__ hop (the wrapper isn't
    # simply broken/empty -- it specifically singles out underscore-prefixed names).
    raw.n_envs = 4096
    direct_wrapper = _MockRecordEpisodeStatistics(raw)
    assert direct_wrapper.n_envs == 4096
    assert hasattr(direct_wrapper, "_egorw") is False


def test_mock_record_episode_statistics_getattr_raises_attributeerror_directly():
    """Pins the underlying mechanism hasattr() swallows: direct attribute access on the wrapper for an
    underscore name raises AttributeError, not e.g. returns None."""
    outer, _raw = _build_chain()
    with pytest.raises(AttributeError, match="private attribute"):
        outer._egorw


# ================================================================================================
# (d) SPIN-ABORT THRESHOLD anneal (2026-07-11, the boot-learnability rung): resolver gating,
#     END-HOLD schedule math, and the lifeline-source wiring pins (both triggers + the armed-bases
#     L16 guard). Pure-Python; nothing here trains.
# ================================================================================================
def test_resolve_spin_abort_anneal_off_by_default_and_parses_knobs():
    """Unset / falsy gate -> None (byte-identical OFF); armed gate -> defaults start_scale=2.6,
    hold_frac=0.25; explicit knobs win."""
    assert ego_launcher._resolve_spin_abort_anneal(_Cfg(env=None)) is None
    assert ego_launcher._resolve_spin_abort_anneal(_Cfg(env=_Cfg())) is None
    assert ego_launcher._resolve_spin_abort_anneal(
        _Cfg(env=_Cfg(spin_abort_anneal=False))) is None

    s = ego_launcher._resolve_spin_abort_anneal(
        _Cfg(env=_Cfg(spin_abort_anneal=True), n_updates=4000))
    assert s == {"start_scale": 2.6, "hold_frac": 0.25, "n_updates": 4000}

    s = ego_launcher._resolve_spin_abort_anneal(
        _Cfg(env=_Cfg(spin_abort_anneal=True, spin_abort_scale_start=3.0,
                      spin_abort_hold_frac=0.5), n_updates=100))
    assert s == {"start_scale": 3.0, "hold_frac": 0.5, "n_updates": 100}


def test_spin_abort_schedule_is_end_hold():
    """END-HOLD shape (the guarantee direction): scale starts at start_scale, decays LINEARLY to 1.0
    by update (1-hold_frac)*N, then HOLDS 1.0 through the end -- the final hold_frac of training runs
    at the EXACT configured fence. This is deliberately the reverse of _cross_zero_schedule /
    _noise_scale_schedule (which hold START first); a start-hold shape here would reach the real fence
    only at the very last update and the saved ckpt would barely train under it."""
    f = ego_launcher._spin_abort_schedule
    N, start, hold = 4000, 2.6, 0.25
    ramp_end = int((1.0 - hold) * N)                          # update 3000
    assert f(0, N, start, hold) == pytest.approx(start)
    mid = f(ramp_end // 2, N, start, hold)
    assert 1.0 < mid < start                                  # strictly inside the ramp
    assert f(ramp_end, N, start, hold) == pytest.approx(1.0)
    # END-HOLD: every update after the ramp trains at the exact fence (scale 1.0).
    for i in (ramp_end + 1, ramp_end + 500, N - 1, N):
        assert f(i, N, start, hold) == pytest.approx(1.0)
    # monotone non-increasing over the whole run (never re-loosens).
    vals = [f(i, N, start, hold) for i in range(0, N + 1, 100)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))
    # degenerate budgets never divide by zero.
    assert f(0, 0, start, hold) in (pytest.approx(start), pytest.approx(1.0))


def test_spin_abort_lifeline_wiring_source_pins():
    """The lifeline (cluster-only execution) is pinned at the SOURCE, the repo convention for
    laptop-unexecutable wiring: the hook must (a) resolve via _require_anneal_holder on the
    '_spin_rate_abort' holder, (b) capture BOTH armed bases and RAISE on a disabled trigger (the L16
    silent-no-op-under-an-annealed-name guard), and (c) mutate BOTH _spin_rate_abort AND
    _spin_rev_abort per update -- vhov4's wobble measured rot_accum ~20.7 >> the 9.42 rev threshold,
    so a rate-only anneal would leave the rev trigger executing the early phase alone."""
    import inspect
    src = inspect.getsource(ego_launcher._run_with_ego_lifelines)
    assert '_require_anneal_holder(env, "_spin_rate_abort", sa_sched' in src
    assert 'sa_sched["base_rate"] = float(sa_env._spin_rate_abort)' in src
    assert 'sa_sched["base_rev"] = float(sa_env._spin_rev_abort)' in src
    assert 'raise RuntimeError' in src.split('sa_sched["base_rev"]', 1)[1].split("agent.step", 1)[0]
    assert 'sa_env._spin_rate_abort = sa_sched["base_rate"] * sav' in src
    assert 'sa_env._spin_rev_abort = sa_sched["base_rev"] * sav' in src


def test_resolve_yaw_clamp_anneal_off_by_default_and_parses_knobs():
    """Unset / falsy gate -> None (byte-identical OFF); armed -> defaults start_scale=4.57 (base 0.7
    -> ~3.2 rad/s ~ the rail == armed-but-free), hold_frac=0.4 (tightens FASTER than the fence's
    0.25 so de-spinning stays ahead of the executioner)."""
    assert ego_launcher._resolve_yaw_clamp_anneal(_Cfg(env=None)) is None
    assert ego_launcher._resolve_yaw_clamp_anneal(_Cfg(env=_Cfg())) is None
    s = ego_launcher._resolve_yaw_clamp_anneal(
        _Cfg(env=_Cfg(yaw_clamp_anneal=True), n_updates=4000))
    assert s == {"start_scale": 4.57, "hold_frac": 0.4, "n_updates": 4000}


def test_yaw_clamp_anneal_lifeline_wiring_source_pins():
    """Source pins (cluster-only wiring): holder = '_yaw_cmd_clamp'; base captured pre-mutation and
    a clamp-OFF base RAISES (anneal of scale*0 is stuck at OFF forever == an L16 silent no-op --
    and the sign convention means a small positive clamp is nearly-FROZEN yaw, so starting from 0
    is doubly wrong); per-update mutation via the shared END-HOLD schedule."""
    import inspect
    src = inspect.getsource(ego_launcher._run_with_ego_lifelines)
    assert '_require_anneal_holder(env, "_yaw_cmd_clamp", yc_sched' in src
    assert 'yc_sched["base"] = float(yc_env._yaw_cmd_clamp)' in src
    assert 'raise RuntimeError' in src.split('yc_sched["base"] = ', 1)[1].split("agent.step", 1)[0]
    assert 'yc_env._yaw_cmd_clamp = yc_sched["base"] * ycv' in src


def test_resolve_perception_anneal_off_by_default_and_parses_knobs():
    """Unset / falsy gate -> None (byte-identical OFF); armed -> defaults start_scale=25.0 (base
    0.02 -> 0.5/tick at birth: the farm-then-wean carrot), hold_frac=0.3 (farm dissolved to the
    farm-neutral base by 70% of budget; racing rewards own the last 30%)."""
    assert ego_launcher._resolve_perception_anneal(_Cfg(env=None)) is None
    assert ego_launcher._resolve_perception_anneal(_Cfg(env=_Cfg())) is None
    s = ego_launcher._resolve_perception_anneal(
        _Cfg(env=_Cfg(perception_anneal=True), n_updates=4000))
    assert s == {"start_scale": 25.0, "hold_frac": 0.3, "n_updates": 4000}


def test_perception_anneal_lifeline_wiring_source_pins():
    """Source pins: holder = '_egorw' (the exact object ego_reward reads); base captured
    pre-mutation and r_perc-OFF base RAISES (scale*0 stuck at OFF == L16 silent no-op); per-update
    mutation of _egorw.perception via the shared END-HOLD schedule."""
    import inspect
    src = inspect.getsource(ego_launcher._run_with_ego_lifelines)
    assert '_require_anneal_holder(env, "_egorw", pc_sched' in src
    assert 'pc_sched["base"] = float(getattr(pc_env._egorw, "perception", 0.0))' in src
    assert 'raise RuntimeError' in src.split('pc_sched["base"] = ', 1)[1].split("agent.step", 1)[0]
    assert 'pc_env._egorw.perception = pc_sched["base"] * pcv' in src


def test_resolve_progress_ramp_off_by_default_and_parses_knobs():
    """Unset / falsy gate -> None (byte-identical OFF); armed -> defaults start_scale=0.0 (ramp IN
    from zero forward-pull) hold_frac=0.3 (full progress reward for the last 30% so the graduate is
    not reward-shifted)."""
    assert ego_launcher._resolve_progress_ramp(_Cfg(env=None)) is None
    assert ego_launcher._resolve_progress_ramp(_Cfg(env=_Cfg())) is None
    s = ego_launcher._resolve_progress_ramp(
        _Cfg(env=_Cfg(progress_ramp=True), n_updates=4000))
    assert s == {"start_scale": 0.0, "hold_frac": 0.3, "n_updates": 4000}


def test_progress_ramp_is_a_ramp_in_via_spin_abort_schedule():
    """start_scale=0 through _spin_abort_schedule ramps 0->1 over the front (1-hold_frac) then
    END-HOLDs at 1.0 -- so _egorw.progress goes 0 -> base and stays base for the last hold_frac.
    (The re-dive fix: forward pull enters gently so the warmed altitude skill isn't overpowered.)"""
    f = ego_launcher._spin_abort_schedule
    N, hold = 4000, 0.3
    ramp_end = int((1.0 - hold) * N)
    assert f(0, N, 0.0, hold) == pytest.approx(0.0)              # no forward pull at birth
    assert 0.0 < f(ramp_end // 2, N, 0.0, hold) < 1.0            # ramping
    assert f(ramp_end, N, 0.0, hold) == pytest.approx(1.0)       # full by (1-hold)*N
    for i in (ramp_end + 1, N - 1, N):
        assert f(i, N, 0.0, hold) == pytest.approx(1.0)          # END-HOLD at full
    vals = [f(i, N, 0.0, hold) for i in range(0, N + 1, 100)]
    assert all(a <= b for a, b in zip(vals, vals[1:]))           # monotone non-decreasing (ramp IN)


def test_progress_ramp_lifeline_wiring_source_pins():
    """Source pins: holder = '_egorw'; base captured pre-mutation and a zero-progress base RAISES
    (ramp 0->0 == L16 silent no-op); per-update mutation of _egorw.progress via the shared schedule
    with start_scale=0 (ramp in, NOT the clamp/perception ramp-down)."""
    import inspect
    src = inspect.getsource(ego_launcher._run_with_ego_lifelines)
    assert '_require_anneal_holder(env, "_egorw", pr_sched' in src
    assert 'pr_sched["base"] = float(getattr(pr_env._egorw, "progress", 0.0))' in src
    assert 'raise RuntimeError' in src.split('pr_sched["base"] = ', 1)[1].split("agent.step", 1)[0]
    assert 'pr_env._egorw.progress = pr_sched["base"] * prv' in src


# ================================================================================================
# (e) ATTITUDE-CAP penalty-WEIGHT RAMP-IN (Track B, 2026-07-12): the annealed soft attitude caps.
#     A gentle reshape of the WORKING ego champion so it flies natively inside a pitch/roll band --
#     the caps ramp IN from 0 (vpeffs0 untouched at birth) to the target weights, END-HOLD at full.
#     Resolver gating + the ramp-in schedule reuse + the DUAL-weight lifeline wiring pins.
# ================================================================================================
def test_resolve_att_cap_anneal_off_by_default_and_parses_knobs():
    """Unset / falsy gate -> None (byte-identical OFF); armed -> defaults start_scale=0.0 (INERT at
    birth = the working flyer untouched) hold_frac=0.3 (full caps for the last 30% so the graduate's
    converged regime IS the flown band); explicit knobs win."""
    assert ego_launcher._resolve_att_cap_anneal(_Cfg(env=None)) is None
    assert ego_launcher._resolve_att_cap_anneal(_Cfg(env=_Cfg())) is None
    assert ego_launcher._resolve_att_cap_anneal(
        _Cfg(env=_Cfg(att_cap_anneal=False))) is None
    s = ego_launcher._resolve_att_cap_anneal(
        _Cfg(env=_Cfg(att_cap_anneal=True), n_updates=3000))
    assert s == {"start_scale": 0.0, "hold_frac": 0.3, "n_updates": 3000}
    s = ego_launcher._resolve_att_cap_anneal(
        _Cfg(env=_Cfg(att_cap_anneal=True, att_cap_start=0.1, att_cap_hold_frac=0.5), n_updates=100))
    assert s == {"start_scale": 0.1, "hold_frac": 0.5, "n_updates": 100}


def test_att_cap_anneal_is_a_ramp_in_via_spin_abort_schedule():
    """start_scale=0 through _spin_abort_schedule ramps 0->1 over the front (1-hold_frac) then END-HOLDs
    at 1.0 -- so both att_pitch/att_roll go 0 -> target and stay target for the last hold_frac. The caps
    are INERT at birth (scale 0) so the warm-started champion is untouched, then reshape gradually."""
    f = ego_launcher._spin_abort_schedule
    N, hold = 3000, 0.3
    ramp_end = int((1.0 - hold) * N)
    assert f(0, N, 0.0, hold) == pytest.approx(0.0)              # caps INERT at birth (vpeffs0 untouched)
    assert 0.0 < f(ramp_end // 2, N, 0.0, hold) < 1.0            # ramping in
    assert f(ramp_end, N, 0.0, hold) == pytest.approx(1.0)       # full caps by (1-hold)*N
    for i in (ramp_end + 1, N - 1, N):
        assert f(i, N, 0.0, hold) == pytest.approx(1.0)          # END-HOLD at the full cap
    vals = [f(i, N, 0.0, hold) for i in range(0, N + 1, 100)]
    assert all(a <= b for a, b in zip(vals, vals[1:]))           # monotone non-decreasing (ramp IN)


def test_att_cap_anneal_lifeline_wiring_source_pins():
    """Source pins (cluster-only wiring): holder = '_egorw' (the exact object ego_reward reads); BOTH
    cap-weight bases captured pre-mutation and a BOTH-OFF base RAISES (ramping scale*0 on both is an L16
    silent no-op under an annealed run name); per-update mutation of BOTH _egorw.att_pitch AND
    _egorw.att_roll via the shared ramp-in (start_scale=0) END-HOLD schedule."""
    import inspect
    src = inspect.getsource(ego_launcher._run_with_ego_lifelines)
    assert '_require_anneal_holder(env, "_egorw", ac_sched' in src
    assert 'ac_sched["base_pitch"] = float(getattr(ac_env._egorw, "att_pitch", 0.0))' in src
    assert 'ac_sched["base_roll"] = float(getattr(ac_env._egorw, "att_roll", 0.0))' in src
    assert 'raise RuntimeError' in src.split('ac_sched["base_roll"] = ', 1)[1].split("agent.step", 1)[0]
    assert 'ac_env._egorw.att_pitch = ac_sched["base_pitch"] * acv' in src
    assert 'ac_env._egorw.att_roll = ac_sched["base_roll"] * acv' in src


# ================================================================================================
# ANTI-DITHER yaw-jerk penalty-WEIGHT RAMP-IN (Track A, 2026-07-13): the yaw_dither term annealed 0->target
# so a hot full-strength penalty never detonates the champion (the nodither collapse). Same RAMP-IN shape
# as the att-cap anneal (start_scale=0 through _spin_abort_schedule -> 0->1 END-HOLD).
# ================================================================================================
def test_resolve_yaw_dither_anneal_off_by_default_and_parses_knobs():
    """Unset / falsy gate -> None (byte-identical OFF); armed -> defaults start_scale=0.0 (INERT at birth =
    the working flyer untouched) hold_frac=0.3 (full penalty for the last 30% so the graduate's converged
    regime IS the flown anti-dither); explicit knobs win."""
    assert ego_launcher._resolve_yaw_dither_anneal(_Cfg(env=None)) is None
    assert ego_launcher._resolve_yaw_dither_anneal(_Cfg(env=_Cfg())) is None
    assert ego_launcher._resolve_yaw_dither_anneal(
        _Cfg(env=_Cfg(yaw_dither_anneal=False))) is None
    s = ego_launcher._resolve_yaw_dither_anneal(
        _Cfg(env=_Cfg(yaw_dither_anneal=True), n_updates=12000))
    assert s == {"start_scale": 0.0, "hold_frac": 0.3, "n_updates": 12000}
    s = ego_launcher._resolve_yaw_dither_anneal(
        _Cfg(env=_Cfg(yaw_dither_anneal=True, yaw_dither_start=0.1, yaw_dither_hold_frac=0.5), n_updates=100))
    assert s == {"start_scale": 0.1, "hold_frac": 0.5, "n_updates": 100}


def test_yaw_dither_anneal_is_a_ramp_in_via_spin_abort_schedule():
    """start_scale=0 through _spin_abort_schedule ramps 0->1 over the front (1-hold_frac) then END-HOLDs at
    1.0 -- so rw_yaw_dither goes 0 -> target and stays target for the last hold_frac. The penalty is INERT
    at birth (scale 0) so the warm-started champion is untouched, then grows the still-yaw skill gradually."""
    f = ego_launcher._spin_abort_schedule
    N, hold = 12000, 0.3
    ramp_end = int((1.0 - hold) * N)
    assert f(0, N, 0.0, hold) == pytest.approx(0.0)              # penalty INERT at birth (champion untouched)
    assert 0.0 < f(ramp_end // 2, N, 0.0, hold) < 1.0            # ramping in
    assert f(ramp_end, N, 0.0, hold) == pytest.approx(1.0)       # full penalty by (1-hold)*N
    for i in (ramp_end + 1, N - 1, N):
        assert f(i, N, 0.0, hold) == pytest.approx(1.0)          # END-HOLD at the full penalty
    vals = [f(i, N, 0.0, hold) for i in range(0, N + 1, 100)]
    assert all(a <= b for a, b in zip(vals, vals[1:]))           # monotone non-decreasing (ramp IN)


def test_yaw_dither_anneal_lifeline_wiring_source_pins():
    """Source pins (cluster-only wiring): holder = '_egorw' (the exact object ego_reward reads); the base
    captured pre-mutation and a <=0 base RAISES (ramping scale*0 is an L16 silent no-op under an annealed
    run name); per-update mutation of _egorw.yaw_dither via the shared ramp-in (start_scale=0) END-HOLD
    schedule."""
    import inspect
    src = inspect.getsource(ego_launcher._run_with_ego_lifelines)
    assert '_require_anneal_holder(env, "_egorw", yd_sched' in src
    assert 'yd_sched["base"] = float(getattr(yd_env._egorw, "yaw_dither", 0.0))' in src
    assert 'raise RuntimeError' in src.split('yd_sched["base"] = ', 1)[1].split("agent.step", 1)[0]
    assert 'yd_env._egorw.yaw_dither = yd_sched["base"] * ydv' in src


# ================================================================================================
# STAGE-1 (vtrackAr5) helpers: n_passed_gates harvest, DET_EVAL metric, critic-only grad clip.
# ================================================================================================
def test_harvest_npg_reads_completed_episode_gates_into_buffer():
    """_harvest_npg is a PURE read of stats_raw['n_passed_gates'] (already reset-masked by the env) into
    the recent-window deque -- the passive rolling-best harvest. Empty tensor / missing key -> no-op."""
    from collections import deque
    buf = deque()
    info = {"stats_raw": {"n_passed_gates": torch.tensor([2.0, 4.0, 1.0])}, "reset": torch.tensor([True])}
    assert ego_launcher._harvest_npg(info, buf) == 3
    assert list(buf) == [2.0, 4.0, 1.0]
    # empty (no episode terminated this step) -> no-op
    assert ego_launcher._harvest_npg({"stats_raw": {"n_passed_gates": torch.tensor([])}}, buf) == 0
    # missing key / non-dict info -> no-op, buffer unchanged
    assert ego_launcher._harvest_npg({"stats_raw": {}}, buf) == 0
    assert ego_launcher._harvest_npg(None, buf) == 0
    assert list(buf) == [2.0, 4.0, 1.0]


class _FakeDetEnv:
    """Minimal env for _run_det_eval: every step terminates 2 episodes (reset both True) with a known
    n_passed_gates so the aggregation + per-episode mean is checkable."""
    def reset(self):
        return torch.zeros(2)

    def rescale_action(self, a):
        return a

    def step(self, a):
        sr = {"success_rate": torch.tensor([1.0, 0.0]),
              "collision_rate": torch.tensor([0.0, 1.0]),
              "miss_rate": torch.tensor([0.0, 0.0]),
              "oob_rate": torch.tensor([0.0, 0.0]),
              "n_passed_gates": torch.tensor([2.0, 4.0])}
        return torch.zeros(2), None, None, {"stats_raw": sr, "reset": torch.tensor([True, True])}


class _FakeDetAgent:
    def act(self, obs, test=False):
        return obs, None


def test_run_det_eval_aggregates_n_passed_gates():
    """_run_det_eval returns the per-completed-episode mean metrics INCLUDING n_passed_gates (the
    BANK-FIRST discriminator): with 2 episodes/step of [2,4] gates, mean = 3.0; success = 0.5."""
    cfg = _Cfg(eval_det_steps=1, runname="detmetric")
    r = ego_launcher._run_det_eval(None, _FakeDetEnv(), _FakeDetAgent(), cfg)
    assert r is not None
    assert r["n_passed_gates"] == pytest.approx(3.0)
    assert r["success_rate"] == pytest.approx(0.5)
    # eval_det_steps=0 -> skip -> None
    assert ego_launcher._run_det_eval(None, _FakeDetEnv(), _FakeDetAgent(),
                                      _Cfg(eval_det_steps=0, runname="x")) is None


def test_select_critic_params_picks_only_critic_and_leaves_actor_untouched():
    """The Stage-1 critic-only grad clip must select ONLY critic-named params (actor_grad_norm is healthy;
    the critic is the one that spikes). Clipping the selection must not touch actor grads."""
    import torch.nn as nn

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.critic = nn.Linear(3, 1)
            self.actor = nn.Linear(3, 2)

    class _Agent:
        def __init__(self, m):
            self.agent = m

    m = _M()
    (m.critic(torch.ones(1, 3)).sum() + m.actor(torch.ones(1, 3)).sum()).backward()
    sel_ids = {id(p) for p in ego_launcher._select_critic_params(_Agent(m))}
    assert sel_ids == {id(p) for p in m.critic.parameters()}
    assert sel_ids.isdisjoint({id(p) for p in m.actor.parameters()})
    # clipping the critic selection hard leaves the actor grad bit-identical
    actor_grad_before = m.actor.weight.grad.clone()
    torch.nn.utils.clip_grad_norm_(ego_launcher._select_critic_params(_Agent(m)), max_norm=1e-6)
    assert torch.equal(m.actor.weight.grad, actor_grad_before)
    assert float(m.critic.weight.grad.norm()) <= 1e-6 + 1e-9


def test_find_runner_checkpoints_prefers_sibling_then_child():
    """The runner's checkpoints/ lands as a SIBLING of the decorated logdir (${RUNDIR}/checkpoints per the
    sbatch), but a child layout is also tolerated; None when neither exists."""
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        logdir = os.path.join(d, "run__stamp")
        os.makedirs(logdir)
        assert ego_launcher._find_runner_checkpoints(_Cfg(logdir=logdir)) is None
        sib = os.path.join(d, "checkpoints")
        os.makedirs(sib)
        assert ego_launcher._find_runner_checkpoints(_Cfg(logdir=logdir)) == sib
