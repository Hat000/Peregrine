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
