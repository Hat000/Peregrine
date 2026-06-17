"""Laptop unit tests for the inc8 WARM-START knob (rl/inc8_warmstart.py + the trainer wiring).

The worker brief's three guarantees:
  (a) +init_from UNSET => maybe_warmstart is a TRUE no-op (returns None, never calls agent.load, never
      touches the env warmup) == the BYTE-IDENTICAL fresh-init trainer. Plus: the module imports with
      NO torch/diffaero at module level (off-path stays clean on the dev laptop).
  (b) +init_from SET => actor+critic are loaded so a 0-update policy REPRODUCES the source policy's
      actions within tol. Exercised against a diffaero-FAITHFUL stand-in (actor_mean MLP + actor_logstd
      Parameter + critic MLP) whose save/load mirror diffaero's StochasticActor / CriticV /
      StochasticActorCriticV / PPO EXACTLY (verified from the diffaero source:
      actor.pth={"actor_mean": state_dict, "actor_logstd": Parameter}; critic.pth=critic state_dict;
      ActorCriticBase.load delegates to actor.load + critic.load; PPO.load -> self.agent.load). The
      real-agent numerical check runs in the Adroit precheck (the laptop has no diffaero).
  (c) warmup=0 => the look-at gain factor == 1.0 from update 0 (inc8_reward.lookat_warmup_factor), AND
      maybe_warmstart FORCES a positive env look-at warmup to 0 on a warm-start.

Plus a static (AST) guard that the trainer wires maybe_warmstart, calls it once before _orig_run, and
has NO other checkpoint-load path -- so the OFF (+init_from unset) branch can never run a load. The
trainer needs diffaero to import, so we pin its source structurally (exactly like
tests/test_inc8_off_identity.py reads + AST-parses the env source).

Run from repo ROOT:  .venv\\Scripts\\python.exe -m pytest tests/test_inc8_warmstart.py -q
"""
import ast
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rl"))

import inc8_warmstart as WS                                            # noqa: E402  (imports w/o torch)

torch = pytest.importorskip("torch")                                  # (b)/(c) need torch; laptop has it
import torch.nn as nn                                                 # noqa: E402
import inc8_reward as R8                                              # noqa: E402  (needs torch)


# ============================================ (a) OFF == byte-identical no-op ========================
class _ExplodingAgent:
    """agent.load must NEVER be called on the off-path; if it is, fail loudly (not byte-identical)."""
    def load(self, path):                                             # noqa: D401
        raise AssertionError(f"agent.load({path!r}) called on the OFF path -- NOT byte-identical!")


def test_resolve_init_from_off_sentinels():
    for off in (None, "", "   ", "none", "NONE", "null", "Null"):
        assert WS.resolve_init_from(off) is None
    assert WS.resolve_init_from("/scratch/x/periodic") == "/scratch/x/periodic"
    assert WS.resolve_init_from("  /scratch/x/periodic  ") == "/scratch/x/periodic"   # trimmed


def test_off_path_is_total_noop():
    """No init_from => returns None, never calls agent.load, never mutates the env warmup. Mirrors
    test_inc8_stability_knobs.test_maybe_widen_critic_off_is_noop (the byte-identical guarantee)."""
    env = SimpleNamespace(obs_dim=20, _lookat_warmup_updates=200)
    assert WS.maybe_warmstart(_ExplodingAgent(), env, SimpleNamespace()) is None     # no attr at all
    assert env._lookat_warmup_updates == 200                          # untouched -- changed NOTHING
    assert WS.maybe_warmstart(_ExplodingAgent(), env, SimpleNamespace(init_from=None)) is None
    assert WS.maybe_warmstart(_ExplodingAgent(), env, SimpleNamespace(init_from="none")) is None
    assert WS.maybe_warmstart(_ExplodingAgent(), env, SimpleNamespace(init_from="")) is None
    assert env._lookat_warmup_updates == 200


def test_module_imports_clean_without_torch():
    """inc8_warmstart must have NO top-level torch/diffaero import (off-path import-clean on the laptop;
    heavy work deferred into the on-path). AST over MODULE-LEVEL statements only."""
    tree = ast.parse((ROOT / "rl" / "inc8_warmstart.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                assert not a.name.startswith(("torch", "diffaero")), f"top-level import {a.name}"
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith(("torch", "diffaero")), node.module


# ===================================== diffaero-faithful stand-in agent (for b/c) ====================
# Mirrors the diffaero classes EXACTLY (read from github.com/flyingbitac/diffaero source):
#   StochasticActor: actor_mean = build_network(...); actor_logstd = nn.Parameter(zeros(1, act_dim))
#       save -> {"actor_mean": actor_mean.state_dict(), "actor_logstd": actor_logstd} @ actor.pth
#       load -> actor_mean.load_state_dict(d["actor_mean"]); actor_logstd.data.copy_(d["actor_logstd"])
#   CriticV: critic = build_network(...,1); save -> critic.state_dict() @ critic.pth; load -> inverse
#   StochasticActorCriticV(ActorCriticBase): .actor + .critic; save/load delegate to both
#   PPO: .save(path)=makedirs+self.agent.save(path); .load(path)=self.agent.load(path)
class _StandinActor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor_mean = nn.Sequential(nn.Linear(obs_dim, 16), nn.ELU(), nn.Linear(16, act_dim))
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def save(self, path):
        torch.save({"actor_mean": self.actor_mean.state_dict(), "actor_logstd": self.actor_logstd},
                   os.path.join(path, "actor.pth"))

    def load(self, path):
        d = torch.load(os.path.join(path, "actor.pth"), weights_only=True)
        self.actor_mean.load_state_dict(d["actor_mean"])
        self.actor_logstd.data.copy_(d["actor_logstd"].to(self.actor_logstd.device))


class _StandinCritic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.critic = nn.Sequential(nn.Linear(obs_dim, 16), nn.ELU(), nn.Linear(16, 1))

    def save(self, path):
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pth"))

    def load(self, path):
        self.critic.load_state_dict(torch.load(os.path.join(path, "critic.pth"), weights_only=True))


class _StandinActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = _StandinActor(obs_dim, act_dim)
        self.critic = _StandinCritic(obs_dim)

    def save(self, path):
        self.actor.save(path)
        self.critic.save(path)

    def load(self, path):
        self.actor.load(path)
        self.critic.load(path)


class _StandinPPO:
    """The contract maybe_warmstart drives: agent.load(path) loads actor+critic. The real GuardedPPO
    inherits exactly this PPO.load."""
    def __init__(self, obs_dim, act_dim):
        self.agent = _StandinActorCritic(obs_dim, act_dim)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.agent.save(path)

    def load(self, path):
        self.agent.load(path)


def _write_sidecar(path, obs_dim):
    with open(os.path.join(path, "actor.json"), "w") as f:
        json.dump({"act_max_thrust": 3.765, "act_max_rate": 3.14,
                   "obs_dim": obs_dim, "inc8": True, "r5_arm": "A"}, f)


# ============================================ (b) loaded reproduces source actions ===================
def test_warmstart_loads_and_reproduces_actions(tmp_path):
    """Save a source agent; warm-start a DIFFERENTLY-initialised agent from it; with 0 updates its
    actor mean + logstd + deployed tanh(mean) action + critic value reproduce the source within tol."""
    obs_dim, act_dim = 20, 4
    torch.manual_seed(0)
    src = _StandinPPO(obs_dim, act_dim)
    ckpt = tmp_path / "periodic"
    src.save(str(ckpt))
    _write_sidecar(str(ckpt), obs_dim)

    torch.manual_seed(999)                              # DIFFERENT init -> must be overwritten by load
    dst = _StandinPPO(obs_dim, act_dim)
    obs = torch.randn(8, obs_dim)
    # the test only proves something if the two policies DISAGREE before the load
    assert not torch.allclose(src.agent.actor.actor_mean(obs), dst.agent.actor.actor_mean(obs),
                              atol=1e-3)

    env = SimpleNamespace(obs_dim=obs_dim, _lookat_warmup_updates=0)
    out = WS.maybe_warmstart(dst, env, SimpleNamespace(init_from=str(ckpt)))
    assert out == str(ckpt)

    with torch.no_grad():
        assert torch.allclose(src.agent.actor.actor_mean(obs), dst.agent.actor.actor_mean(obs),
                              atol=1e-6)                                   # actor mean reproduced
        assert torch.allclose(src.agent.actor.actor_logstd, dst.agent.actor.actor_logstd, atol=1e-6)
        assert torch.allclose(torch.tanh(src.agent.actor.actor_mean(obs)),
                              torch.tanh(dst.agent.actor.actor_mean(obs)), atol=1e-6)   # deployed action
        assert torch.allclose(src.agent.critic.critic(obs), dst.agent.critic.critic(obs),
                              atol=1e-6)                                   # critic resumed too


def test_warmstart_obs_dim_mismatch_raises(tmp_path):
    """A 17-dim inc7 checkpoint cannot warm-start a 20-dim inc8 env -- caught on the sidecar BEFORE a
    cryptic state_dict size error."""
    ckpt = tmp_path / "periodic17"
    ckpt.mkdir()
    _StandinPPO(17, 4).save(str(ckpt))
    _write_sidecar(str(ckpt), 17)
    env = SimpleNamespace(obs_dim=20, _lookat_warmup_updates=0)
    with pytest.raises(ValueError, match="obs_dim"):
        WS.maybe_warmstart(_StandinPPO(20, 4), env, SimpleNamespace(init_from=str(ckpt)))


def test_missing_dir_and_files_raise(tmp_path):
    env = SimpleNamespace(obs_dim=20, _lookat_warmup_updates=0)
    with pytest.raises(FileNotFoundError):                       # dir does not exist
        WS.maybe_warmstart(_StandinPPO(20, 4), env,
                           SimpleNamespace(init_from=str(tmp_path / "nope")))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="missing"):      # dir exists, no actor.pth/critic.pth
        WS.maybe_warmstart(_StandinPPO(20, 4), env, SimpleNamespace(init_from=str(empty)))


def test_warmstart_missing_sidecar_still_loads(tmp_path):
    """The sidecar obs-dim gate is advisory: a checkpoint with no actor.json still loads (agent.load
    shape-checks the tensors itself)."""
    obs_dim = 20
    ckpt = tmp_path / "periodic_nosidecar"
    _StandinPPO(obs_dim, 4).save(str(ckpt))                      # save WITHOUT a sidecar
    env = SimpleNamespace(obs_dim=obs_dim, _lookat_warmup_updates=0)
    assert WS.maybe_warmstart(_StandinPPO(obs_dim, 4), env,
                              SimpleNamespace(init_from=str(ckpt))) == str(ckpt)


# ============================================ (c) warmup=0 => gain factor == 1 =======================
def test_warmup_factor_is_one_when_warmup_zero():
    """warmup_updates=0 => lookat_warmup_factor == 1.0 at EVERY update (gain held full from update 0 --
    no ramp to disrupt the loaded pointing). For contrast a positive warmup DOES ramp from 0."""
    for idx in (0, 1, 5, 50, 200, 4000):
        assert R8.lookat_warmup_factor(idx, 0) == 1.0
    assert R8.lookat_warmup_factor(0, 200) == 0.0
    assert R8.lookat_warmup_factor(100, 200) == pytest.approx(0.5)
    assert R8.lookat_warmup_factor(200, 200) == 1.0


def test_warmstart_forces_env_warmup_to_zero(tmp_path):
    """On a warm-start, a positive env look-at warmup is forced to 0 so the loaded pointing is held at
    full gain (not re-ramped from 0)."""
    obs_dim = 20
    ckpt = tmp_path / "periodic"
    _StandinPPO(obs_dim, 4).save(str(ckpt))
    _write_sidecar(str(ckpt), obs_dim)
    env = SimpleNamespace(obs_dim=obs_dim, _lookat_warmup_updates=200)   # a (mistaken) positive warmup
    WS.maybe_warmstart(_StandinPPO(obs_dim, 4), env, SimpleNamespace(init_from=str(ckpt)))
    assert env._lookat_warmup_updates == 0                              # forced off


def test_warmstart_no_warmup_attr_is_safe(tmp_path):
    """An env without a _lookat_warmup_updates attr (e.g. a non-inc8 env) is left alone -- no crash."""
    obs_dim = 20
    ckpt = tmp_path / "periodic"
    _StandinPPO(obs_dim, 4).save(str(ckpt))
    _write_sidecar(str(ckpt), obs_dim)
    env = SimpleNamespace(obs_dim=obs_dim)                               # no warmup attr
    assert WS.maybe_warmstart(_StandinPPO(obs_dim, 4), env,
                              SimpleNamespace(init_from=str(ckpt))) == str(ckpt)


# ============================================ trainer wiring (static / AST) ==========================
TRAIN_SRC = (ROOT / "rl" / "peregrine_train_inc8.py").read_text()
TRAIN_AST = ast.parse(TRAIN_SRC)


def _func(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def test_trainer_imports_maybe_warmstart():
    assert "from inc8_warmstart import maybe_warmstart" in TRAIN_SRC


def test_trainer_calls_maybe_warmstart_once_before_training():
    """_run_with_lifelines calls maybe_warmstart exactly once, and the _orig_run(self) training entry
    comes AFTER it (load happens before the training loop starts)."""
    fn = _func(TRAIN_AST, "_run_with_lifelines")
    ws_calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == "maybe_warmstart"]
    assert len(ws_calls) == 1, "maybe_warmstart must be called exactly once in _run_with_lifelines"
    orig_run_lines = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Name) and n.func.id == "_orig_run"]
    assert orig_run_lines and ws_calls[0].lineno < min(orig_run_lines)


def test_trainer_has_no_other_checkpoint_load_path():
    """The ONLY checkpoint-load path in the trainer is maybe_warmstart (whose off-path is the proven
    no-op). No bare agent.load / torch.load directly in the trainer that could run when +init_from is
    unset and break byte-identity."""
    assert "torch.load" not in TRAIN_SRC
    assert "agent.load" not in TRAIN_SRC
