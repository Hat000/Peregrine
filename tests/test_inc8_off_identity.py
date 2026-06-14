"""OFF-FALLBACK byte-identity guard (static) -- PeregrineRacingInc8 with inc8=false must be inc7.

PeregrineRacingInc8 needs diffaero (GPU/cluster), so the NUMERICAL byte-identity check runs in the
Adroit smoke. This laptop test STATICALLY pins the structural guarantee that makes byte-identity hold:
every behaviour-bearing override (step / get_observations / get_state) short-circuits to ``super()``
on its FIRST statement when ``self._inc8_on`` is false, and __init__ returns before constructing any
inc8 state. A regression that buries the guard (so the inc8 path runs even when OFF) fails here.
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "rl" / "peregrine_racing_inc8.py"


def _cls():
    tree = ast.parse(SRC.read_text())
    return next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "PeregrineRacingInc8")


def _method(cls, name):
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _first_is_off_guard_returning_super(fn, name):
    """First stmt: ``if not self._inc8_on: return super().<name>(...)``."""
    stmt = fn.body[0]
    assert isinstance(stmt, ast.If), f"{name}: first statement is not an if-guard"
    t = stmt.test
    assert isinstance(t, ast.UnaryOp) and isinstance(t.op, ast.Not), f"{name}: guard is not 'not ...'"
    assert isinstance(t.operand, ast.Attribute) and t.operand.attr == "_inc8_on", f"{name}: not _inc8_on"
    ret = stmt.body[0]
    assert isinstance(ret, ast.Return), f"{name}: guard body is not a return"
    call = ret.value
    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute), f"{name}: not a call"
    assert call.func.attr == name, f"{name}: returns super().{call.func.attr}, expected {name}"
    assert isinstance(call.func.value, ast.Call) and getattr(call.func.value.func, "id", "") == "super", \
        f"{name}: not delegating to super()"


def test_step_obs_state_short_circuit_to_super_when_off():
    cls = _cls()
    for name in ("step", "get_observations", "get_state"):
        _first_is_off_guard_returning_super(_method(cls, name), name)


def test_init_returns_before_building_inc8_state_when_off():
    """__init__ has an early ``if not self._inc8_on: return`` BEFORE any ``self._emu`` / obs_dim=20
    assignment, so OFF constructs no inc8 state and leaves obs_dim at the parent's 17."""
    init = _method(_cls(), "__init__")
    early_return_idx = None
    for i, stmt in enumerate(init.body):
        if (isinstance(stmt, ast.If) and isinstance(stmt.test, ast.UnaryOp)
                and isinstance(stmt.test.op, ast.Not)
                and isinstance(stmt.test.operand, ast.Attribute)
                and stmt.test.operand.attr == "_inc8_on"
                and any(isinstance(s, ast.Return) for s in stmt.body)):
            early_return_idx = i
            break
    assert early_return_idx is not None, "__init__ lacks an 'if not self._inc8_on: return' guard"
    # nothing after the guard may assign obs_dim or _emu BEFORE the guard (they must be guarded)
    src = ast.get_source_segment(SRC.read_text(), init)
    pre = src.split("if not self._inc8_on")[0]
    assert "self.obs_dim = 20" not in pre
    assert "self._emu" not in pre


def test_reset_idx_calls_super_first():
    """reset_idx spawns via super().reset_idx FIRST, then (only if inc8 on) re-inits the emulator."""
    fn = _method(_cls(), "reset_idx")
    first = fn.body[0]
    assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
    assert isinstance(first.value.func, ast.Attribute) and first.value.func.attr == "reset_idx"
    assert getattr(first.value.func.value.func, "id", "") == "super"
