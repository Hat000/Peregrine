"""VQ2 egocentric launcher: register + train the ego env (component C) via the ENV_ALIAS / monkeypatch
pattern, exactly like rl/peregrine_train_inc8.py. ZERO edits to the pristine diffaero clone.

  * dynamics: DYNAMICS_ALIAS["peregrine_plant"] -> PeregrinePlantDynamics  (system-ID'd plant + DR)
  * env:      ENV_ALIAS["peregrine_racing_ego"] -> PeregrineRacingEgo       (VQ2 egocentric env)
  * algo:     AGENT_ALIAS["appo"]               -> the privileged-critic APPO (asymmetric)

Typical invocation (default OFF == byte-identical inc7; turn ON with +env.ego=true):

  python peregrine_train_ego.py \
      env=racing env.name=peregrine_racing_ego +env.ego=true \
      +env.standing_start_frac=1.0 \
      dynamics=quad dynamics.name=peregrine_plant dynamics.g=9.80665 \
      algo=appo n_envs=2048 n_updates=6000 headless=True device=0

The ladder is a separate module (rl/vq2_ego_curriculum.py) driven by the sbatch (DESIGN.md §E).

======================================================================================================
FOLDED-FORWARD FIX #1 -- the LOAD-BEARING appo / privileged-critic assert, wired LIVE here.
======================================================================================================
The assert lives in ego_reward.assert_privileged_critic_connected (exposed on the env as
env.assert_appo_critic) but is INERT unless something CALLS it at runtime. This launcher wraps
TrainRunner.run (the inc8 lifeline pattern) and, right AFTER the agent is built (self.agent exists),
reads the BUILT critic's input_dim and calls env.assert_appo_critic(algo_name, critic_input_dim). A
run launched with algo=ppo (GuardedPPO's SYMMETRIC obs-only critic -- the inc8 seed-collapse root
cause) or with an accidentally-symmetric critic therefore raises AssertionError at startup, BEFORE any
compute is burned, instead of silently training a damped seed-collapsed policy.

======================================================================================================
FOLDED-FORWARD FIX #2 -- the WARM-START LIFELINE, wired LIVE here (the ego ladder's stage->stage chain).
======================================================================================================
The ego curriculum (rl/vq2_ego_curriculum.py) is a stage->stage WARM-STARTED ladder: each stage
CONTINUES from the previous stage's checkpoint via ``+init_from=<prev>/checkpoints`` (the runner's
end-of-run save dir, which holds actor.pth + critic.pth + actor.json -- the exact set maybe_warmstart
needs). WITHOUT wiring the lifeline this launcher would leave +init_from a DEAD hydra key (hydra's '+'
adds it in struct mode, then NOTHING reads it), so every ladder stage would train FRESH -- silently
wasting the whole warm-start chain. This wrapper mirrors the PROVEN inc8 lifeline
(rl/peregrine_train_inc8.py::_run_with_lifelines) for the ego generation:

  * PERIODIC SAVE (every save_freq updates) -> ``<logdir>/periodic`` (+ periodic_prev) + EMERGENCY save
    on any exception: the crash-resilience half. (The runner ALSO writes its own ``<rundir>/checkpoints``
    end-of-run save -- that is the dir the NEXT stage's +init_from points at.)
  * WARM-START LOAD: maybe_warmstart(agent, env, cfg) loads actor+critic from ``cfg.init_from`` into the
    freshly-built agent (weights-only; optimizer + rollout buffer stay fresh). UNSET => byte-identical
    fresh init (the off-path touches nothing). The ego env has no look-at gain-warmup, so the inc8
    _force_lookat_warmup_off inside maybe_warmstart is a safe no-op here.
  * LOGSTD RESET (``+warmstart_reset_logstd`` [+ _std]): after the load, reset the exploration logstd to
    an intermediate level so the transferred MEANS carry over but the ground-down exploitation schedule
    does NOT (the audit-A3 collapse fix). Applied here in the launcher so it fires regardless of which
    inc8_warmstart.py version is on PYTHONPATH (the newer maybe_warmstart also applies it -- an
    idempotent double-set of the same raw value; harmless).
  * CRITIC WARMUP (``+critic_warmup_updates=N``): zero every actor-named gradient for the first N PPO
    updates so ONLY the critic adapts to the new stage's return scale before the policy moves (the
    audit-A1 boundary-divergence fix). Unset/0 => byte-identical.

These are exactly the keys rl/peregrine_vq2_ego.sbatch's BOUNDARY_OV passes
(+warmstart_reset_logstd=true +warmstart_reset_logstd_std=0.18 +critic_warmup_updates=100).
"""
import os
import json
import shutil
import tempfile
import traceback

import torch

import diffaero.algo as _algo          # noqa: E402
import diffaero.dynamics as _dyn        # noqa: E402
import diffaero.env as _env             # noqa: E402
from diffaero.utils.runner import TrainRunner
from diffaero_dynamics import PeregrinePlantDynamics
from peregrine_racing import PeregrineRacing
from peregrine_racing_ego import (PeregrineRacingEgo, clamp_yaw_command,
                                  EGO_OBS_DIM, EGO_OBS_DIM_V2, EGO_OBS_V2_EXTRA,
                                  pad_actor_input_cols_zero)
from inc8_warmstart import maybe_warmstart, resolve_init_from
# NOISE-CEILING lever (RC2 fix): resolve_noise_anneal is a pure getattr (None when +algo.noise_anneal is
# unset -> byte-identical off-path, no torch); apply_noise_schedule clamps actor_logstd to the scheduled
# std ceiling BEFORE each rollout. Without this the constant entropy bonus drives actor_logstd to the
# LOG_STD_MAX ceiling (std=exp(2)=7.39, bang-bang) with nothing capping it -- the ego single_gate collapse.
from inc8_noise_anneal import resolve_noise_anneal, apply_noise_schedule

try:                                    # newer inc8_warmstart.py exports the logstd-reset helper
    from inc8_warmstart import reset_actor_logstd
except ImportError:                     # older inc8_warmstart.py predates it -> self-contained fallback
    import math as _math

    # tanh-squash bounds -- PINNED to diffaero network/agents.py (LOG_STD_MAX=2, LOG_STD_MIN=-5); the
    # squash is logstd = MIN + 0.5*(MAX-MIN)*(tanh(raw)+1). Verbatim from the newer inc8_warmstart.py so
    # +warmstart_reset_logstd fires the same whether the synced inc8_warmstart.py is old or new.
    _LOG_STD_MIN, _LOG_STD_MAX = -5.0, 2.0

    def _raw_logstd_for_std(target_std: float) -> float:
        x = (_math.log(target_std) - _LOG_STD_MIN) / (0.5 * (_LOG_STD_MAX - _LOG_STD_MIN)) - 1.0
        x = max(min(x, 1.0 - 1e-9), -1.0 + 1e-9)
        return _math.atanh(x)

    def reset_actor_logstd(agent, target_std: float = 0.18) -> int:
        """Set every ``actor_logstd`` RAW parameter so the squashed per-dim std equals ``target_std``.
        Returns the count reset (0 -> loud warning: layout changed, reset silently missed)."""
        raw = _raw_logstd_for_std(target_std)
        module = getattr(agent, "agent", agent)
        n_reset = 0
        for name, p in module.named_parameters():
            if "actor_logstd" in name:
                p.data.fill_(raw)
                n_reset += 1
                print(f"[warmstart] RESET exploration: {name} -> raw {raw:.4f} (squashed std "
                      f"~{target_std:.3f}/dim -- intermediate exploration on transferred means).")
        if n_reset == 0:
            print("[warmstart] WARNING: warmstart_reset_logstd requested but NO actor_logstd parameter "
                  "found -- network layout changed? Exploration was NOT reset.")
        return n_reset

_dyn.DYNAMICS_ALIAS["peregrine_plant"] = lambda cfg, device: PeregrinePlantDynamics(
    cfg, device, backend="torch")
_env.ENV_ALIAS["peregrine_racing"] = PeregrineRacing            # keep inc7 available
_env.ENV_ALIAS["peregrine_racing_ego"] = PeregrineRacingEgo     # the VQ2 egocentric env


# ================================================================================================
# FOLDED-FORWARD FIX #1: fire the appo / privileged-critic assert right after the agent is built.
# ================================================================================================
def _algo_name(cfg) -> str:
    """Best-effort extraction of the selected algo name from the resolved cfg. The hydra group choice
    (algo=appo) populates cfg.algo.name; fall back to the group node's _target_ / str."""
    algo_cfg = getattr(cfg, "algo", None)
    name = getattr(algo_cfg, "name", None)
    if name:
        return str(name)
    tgt = getattr(algo_cfg, "_target_", None)
    return str(tgt) if tgt else str(algo_cfg)


def _built_critic_input_dim(agent):
    """Read the input_dim of the BUILT critic value-net (the same backbone path inc8_critic_width uses:
    agent.agent.critic.critic.input_dim). Returns None if the diffaero network API has moved (the assert
    then falls back to the env's own state_dim vs obs_dim check on a None-safe path -> loud, not silent)."""
    try:
        critic_backbone = agent.agent.critic.critic
        return getattr(critic_backbone, "input_dim", None)
    except Exception:                               # pragma: no cover - API drift
        return None


def _weights_finite(agent) -> bool:
    return all(torch.isfinite(p).all() for p in agent.agent.parameters())


# ================================================================================================
# OBS-V2 WARM-PAD driver: bridge a 21-dim champion actor into the 23-dim obs_v2 net (zero-pad).
# ================================================================================================
def maybe_pad_ckpt_for_obs_v2(cfg, env):
    """When +env.ego_obs_v2 is armed (env obs is the 23-dim APPEND-ONLY v2 layout) but +init_from points
    at a 21-dim champion actor (vpeffs0), rewrite the checkpoint into a TEMP dir with the actor's
    first-layer input matrix zero-padded by EGO_OBS_V2_EXTRA columns (the 2 appended next-gate coarse dims
    contribute NOTHING at step 0 -> bit-for-bit champion behaviour), copy the critic verbatim (UNAFFECTED
    -- the privileged critic state layout is unchanged), write an actor.json sidecar (obs_dim=23 so
    maybe_warmstart's obs-dim gate passes 23==23 instead of tripping on 21!=23), and REPOINT cfg.init_from
    at the padded copy so the subsequent maybe_warmstart(agent, env, cfg) loads the matching 23-dim actor.

    Returns the padded dir, or None on the byte-identical no-op paths: knob OFF (a pure getattr BEFORE any
    torch / I/O), no +init_from (a fresh v2 start with no warm base), env not v2, missing actor.pth (let
    maybe_warmstart raise its own actionable message), or the ckpt already matches the env dim (a same-
    layout 23-dim resume needs no pad). Called right BEFORE maybe_warmstart in the run wrapper."""
    if not bool(getattr(cfg, "ego_obs_v2", False)):
        return None                                          # OFF: byte-identical -- touch nothing.
    init_from = resolve_init_from(getattr(cfg, "init_from", None))
    if init_from is None:
        return None                                          # fresh v2 start (no warm base) -> no pad.
    env_obs_dim = int(getattr(env, "obs_dim", 0) or 0)
    if env_obs_dim != EGO_OBS_DIM_V2:
        return None                                          # defensive: only pad INTO the 23-dim v2 net.
    actor_path = os.path.join(init_from, "actor.pth")
    if not os.path.isfile(actor_path):
        return None                                          # let maybe_warmstart raise the actionable msg.
    ckpt = torch.load(actor_path, map_location="cpu")
    if not (isinstance(ckpt, dict) and "actor_mean" in ckpt):
        raise ValueError(
            f"[obs-v2 warm-pad] {actor_path!r} is not the diffaero PPO actor.pth format "
            f"({{'actor_mean': state_dict, 'actor_logstd': ...}}) -- cannot warm-pad.")
    w0 = ckpt["actor_mean"].get("head.0.linear.weight")
    ckpt_in = int(w0.shape[1]) if w0 is not None else 0
    if ckpt_in == env_obs_dim:
        return None                                          # already 23-dim -> same-layout resume, no pad.
    if ckpt_in != EGO_OBS_DIM:
        raise ValueError(
            f"[obs-v2 warm-pad] +init_from actor is {ckpt_in}-dim; the obs_v2 warm-pad only bridges the "
            f"{EGO_OBS_DIM}-dim champion -> {EGO_OBS_DIM_V2}-dim v2 net (append-only +{EGO_OBS_V2_EXTRA}-col "
            f"zero pad). A {ckpt_in}-dim base is an unexpected layout -- reconcile before warm-starting.")

    # pad the actor first layer with EGO_OBS_V2_EXTRA ZERO input columns; carry logstd + all deeper layers
    # verbatim. The critic (critic.pth) is copied UNCHANGED -- obs_v2 is an ACTOR-obs-only change.
    out_ckpt = dict(ckpt)
    out_ckpt["actor_mean"] = pad_actor_input_cols_zero(ckpt["actor_mean"], EGO_OBS_V2_EXTRA)
    pad_dir = tempfile.mkdtemp(prefix="ego_obs_v2_warmpad_")
    torch.save(out_ckpt, os.path.join(pad_dir, "actor.pth"))
    critic_src = os.path.join(init_from, "critic.pth")
    if os.path.isfile(critic_src):
        shutil.copyfile(critic_src, os.path.join(pad_dir, "critic.pth"))
    # actor.json sidecar: obs_dim=23 so maybe_warmstart's obs-dim gate passes; carry forward any other
    # source-sidecar fields (action bounds etc.) unchanged so nothing downstream regresses.
    sidecar = {}
    src_json = os.path.join(init_from, "actor.json")
    if os.path.isfile(src_json):
        try:
            with open(src_json) as f:
                sidecar = json.load(f)
        except Exception:                                    # pragma: no cover - corrupt sidecar
            sidecar = {}
    sidecar["obs_dim"] = EGO_OBS_DIM_V2
    with open(os.path.join(pad_dir, "actor.json"), "w") as f:
        json.dump(sidecar, f)
    # REPOINT init_from at the padded copy (existing root key -> assignable even under omegaconf struct
    # mode; fall back to open_dict if a stricter container rejects the direct set).
    try:
        cfg.init_from = pad_dir
    except Exception:                                        # pragma: no cover - stricter omegaconf struct
        from omegaconf import open_dict
        with open_dict(cfg):
            cfg.init_from = pad_dir
    print(f"[obs-v2 warm-pad] {init_from} (actor {ckpt_in}-dim) -> padded {pad_dir} "
          f"({EGO_OBS_DIM_V2}-dim: +{EGO_OBS_V2_EXTRA} ZERO first-layer cols); critic + logstd verbatim. "
          f"maybe_warmstart will load the padded 23-dim actor next.")
    return pad_dir


_orig_run = TrainRunner.run


def _cross_zero_schedule(update_idx: int, n_updates: int, start: float, end: float,
                         hold_frac: float) -> float:
    """GEOMETRIC (log-linear) decay of the parabola ZERO RADIUS (rw_cross_zero_m) from ``start`` to
    ``end`` over the back ``1-hold_frac`` of training; hold ``start`` for the first ``hold_frac``. Mirrors
    inc8_noise_anneal's std schedule. Shrinking the zero WITHIN a run sharpens the near-centre crossing
    gradient smoothly as the policy centres -- avoiding the discrete warm-start-into-tighter-zero collapse
    (vglp3/vglp3b: a 4->3 STEP detonated the warm-start; a gradual anneal has no discontinuity).

    Falls back to a LINEAR ramp whenever either endpoint is <=0 (geometric ``start*(end/start)**p`` is
    undefined/degenerate there -- division by zero or a zero base). The cross-zero caller never hits this
    branch (start/end are always >0 radii), but the clip-terminal-anneal caller does by design
    (``clip_pen_start`` defaults to 0.0 -- a penalty weight ramping UP from off). The linear form still
    correctly HOLDS ``start`` through ``hold_frac`` then ramps to ``end``; the old fallback here was
    ``return end`` unconditionally, which ignored ``hold_frac`` and ``p`` entirely and snapped straight to
    ``end`` from update 0 -- silently wrong for any start<=0 caller (dead code for cross-zero, but would
    have detonated the clip-terminal anneal at update 0 instead of ramping it)."""
    N = max(int(n_updates), 1)
    i0 = hold_frac * N
    span = max(N - i0, 1.0)
    p = min(max((update_idx - i0) / span, 0.0), 1.0)
    if start > 0.0 and end > 0.0:
        return start * (end / start) ** p
    return start + (end - start) * p


def _resolve_cross_zero_anneal(cfg):
    """Parse the parabola-zero anneal from cfg.env, or None when OFF (byte-identical default). Gated by
    ``+env.cross_zero_anneal`` (truthy); ``+env.cross_zero_start/end/hold_frac`` optional. PURE getattr."""
    env = getattr(cfg, "env", None)
    if env is None or not bool(getattr(env, "cross_zero_anneal", False)):
        return None
    return dict(
        start=float(getattr(env, "cross_zero_start", 5.0)),
        end=float(getattr(env, "cross_zero_end", 0.75)),
        hold_frac=float(getattr(env, "cross_zero_hold_frac", 0.1)),
        n_updates=int(getattr(cfg, "n_updates", 0) or 0),
    )


def _unwrap_env_with(env, attr, max_depth=12):
    """Drill the ``.env`` wrapper chain to the object whose OWN __dict__ holds ``attr`` (e.g. the raw
    peregrine_racing_ego holding ``_egorw``/``_estimator``). DiffAero's TrainRunner wraps the env in
    RecordEpisodeStatistics (utils/runner.py), and the wrapper chain's ``__getattr__`` forwarding does
    NOT reliably reach these attributes from the top -- so ``hasattr(self.env, "_egorw")`` returns False
    and the in-run anneals silently SKIP (the cross-zero anneal was inert the whole 2026-07-08 campaign).
    This walks ``vars(e)['env']`` (instance dict, NOT getattr -> no __getattr__ recursion) to find the
    real holder. Returns the holder object, or None if not found within max_depth."""
    e = env
    for _ in range(max_depth):
        if e is None:
            return None
        d = getattr(e, "__dict__", {})
        if attr in d:
            return e
        e = d.get("env", None)
    return None


def _require_anneal_holder(env, attr, sched, hook_name, cfg):
    """Drill the env chain (_unwrap_env_with) for the holder carrying ``attr`` on behalf of a requested
    in-run anneal. Returns None with NO side effect when the anneal itself was never requested (``sched``
    is None -- the byte-identical OFF path, no print). When the anneal WAS requested (``sched`` is not
    None) but the holder can't be found in the chain, the default is to RAISE RuntimeError -- footgun L16
    (2026-07-09): the pre-fix behaviour here was to print '... SKIPPED' and silently fall back to an
    UNANNEALED run, which let the entire 2026-07-08 cross-zero campaign train at a fixed value under an
    annealed run name with nobody noticing until a bit-identical dose-response pair exposed it. A
    requested-but-unfindable hook MUST kill the run loudly, not train a silently-degraded baseline. Set
    the top-level ``+anneal_allow_skip=true`` to opt back into the old silent-skip-and-continue behaviour
    (e.g. for a deliberately-degraded smoke run)."""
    if sched is None:
        return None
    holder = _unwrap_env_with(env, attr)
    if holder is not None:
        return holder
    allow_skip = bool(getattr(cfg, "anneal_allow_skip", False))
    msg = f"[{hook_name}] requested but {attr!r} holder not found in env chain"
    if allow_skip:
        print(f"{msg} -- SKIPPED (anneal_allow_skip=true)")
        return None
    raise RuntimeError(
        f"{msg} -- refusing to silently disable a requested in-run anneal (footgun L16). Set "
        f"+anneal_allow_skip=true if a silent skip is genuinely intended.")


def _noise_scale_schedule(update_idx: int, n_updates: int, start: float, end: float,
                          hold_frac: float) -> float:
    """LINEAR ramp of the estimator global noise multiplier (ego_noise_scale) from ``start`` to ``end``
    over the back ``1-hold_frac`` of training; hold ``start`` for the first ``hold_frac``. A NOISE
    CURRICULUM: learn to centre on a clean (or low-noise) signal first, then ramp the measured vision
    noise in so the policy adapts that centring skill to be noise-ROBUST -- the data-motivated response to
    the vglpns0 finding (perfect estimator -> 64% thread / 0.32 m; the ~0.88 m floor is perception-noise-
    limited). LINEAR (not geometric) because start can be 0 (geometric can't cross 0). Continuous -> no
    discontinuity -> avoids the discrete-warm-start collapse (a FIXED intermediate noise detonated: vglpns5
    0.5 -> collapsed basin; the continuous ramp is the L6-safe cure)."""
    N = max(int(n_updates), 1)
    i0 = hold_frac * N
    span = max(N - i0, 1.0)
    p = min(max((update_idx - i0) / span, 0.0), 1.0)
    return start + (end - start) * p


def _resolve_noise_scale_anneal(cfg):
    """Parse the estimator noise-scale anneal from cfg.env, or None when OFF (byte-identical default).
    Gated by ``+env.noise_scale_anneal`` (truthy); ``+env.noise_scale_start/end/hold_frac`` optional."""
    env = getattr(cfg, "env", None)
    if env is None or not bool(getattr(env, "noise_scale_anneal", False)):
        return None
    return dict(
        start=float(getattr(env, "noise_scale_start", 0.0)),
        end=float(getattr(env, "noise_scale_end", 1.0)),
        hold_frac=float(getattr(env, "noise_scale_hold_frac", 0.1)),
        n_updates=int(getattr(cfg, "n_updates", 0) or 0),
    )


def _resolve_clip_terminal_anneal(cfg):
    """Parse the clip-terminal penalty-WEIGHT anneal from cfg.env, or None when OFF (byte-identical
    default). Gated by ``+env.clip_pen_anneal`` (truthy); ``+env.clip_pen_start/end/hold_frac`` optional.
    Mutates ``env._egorw.clip_terminal_w`` per update -- mirrors the cross-zero/noise-scale anneal pattern
    exactly (resolve -> _unwrap_env_with(env, "_egorw") -> per-update mutate + print). Uses
    _cross_zero_schedule, whose LINEAR fallback handles the default ``clip_pen_start=0.0`` correctly
    (geometric is undefined at start=0; see _cross_zero_schedule's docstring). ``clip_terminal_w`` is a
    lever a parallel agent is adding to ego_reward.py -- this anneal setattr's it on ``_egorw`` regardless
    of whether the attribute exists yet (the ``from_cfg`` wiring that reads it back into the reward
    computation lands separately; setattr on a plain object simply creates the attribute)."""
    env = getattr(cfg, "env", None)
    if env is None or not bool(getattr(env, "clip_pen_anneal", False)):
        return None
    return dict(
        start=float(getattr(env, "clip_pen_start", 0.0)),
        end=float(getattr(env, "clip_pen_end", 1.0)),
        hold_frac=float(getattr(env, "clip_pen_hold_frac", 0.1)),
        n_updates=int(getattr(cfg, "n_updates", 0) or 0),
    )


def _spin_abort_schedule(update_idx: int, n_updates: int, start_scale: float,
                         hold_frac: float) -> float:
    """LINEAR decay of the fatal-spin-abort threshold SCALE from ``start_scale`` -> 1.0 over the FRONT
    ``1-hold_frac`` of training, then HOLD 1.0 for the LAST ``hold_frac`` -- END-HOLD, deliberately the
    REVERSE of the start-hold anneals above, because the no-spin GUARANTEE lives at the END: the final
    ``hold_frac`` of training runs at the EXACT configured fence, so the saved checkpoint's converged
    regime IS the flown fence (a start-hold shape would reach scale 1.0 only at the last update and the
    ckpt would barely train under the real fence). The END-state fence is NEVER loosened (no-spin
    directive, Fengyou 2026-07-10); only the PATH to it is -- the boot-learnability fix: the full fence
    executes the incompetent EARLY phase (vperc0/2, vdclamp0, vhov0-4 all dead at the 0.4 s clock;
    dgbf0f = the same boot with no fence was flying by 15% of budget)."""
    N = max(int(n_updates), 1)
    span = max((1.0 - hold_frac) * N, 1.0)
    p = min(max(update_idx / span, 0.0), 1.0)
    return float(start_scale) + (1.0 - float(start_scale)) * p


def _resolve_spin_abort_anneal(cfg):
    """Parse the fatal-spin-abort threshold anneal from cfg.env, or None when OFF (byte-identical
    default). Gated by ``+env.spin_abort_anneal`` (truthy); ``+env.spin_abort_scale_start/hold_frac``
    optional. ONE scale factor multiplies BOTH triggers (rate clock ``_spin_rate_abort`` AND leaky rev
    threshold ``_spin_rev_abort``) -- vhov4's wobble-hover measured rot_accum ~20.7 rad (>> the 9.42
    rev threshold), so annealing only the rate clock would leave the rev trigger executing the early
    phase alone. start default 2.6: rate 3.5->9.1 / rev 1.5->3.9 at birth, above the measured wobble
    ~5.2 rad/s mean so the early learnable regime is genuinely fence-free. PURE getattr."""
    env = getattr(cfg, "env", None)
    if env is None or not bool(getattr(env, "spin_abort_anneal", False)):
        return None
    return dict(
        start_scale=float(getattr(env, "spin_abort_scale_start", 2.6)),
        hold_frac=float(getattr(env, "spin_abort_hold_frac", 0.25)),
        n_updates=int(getattr(cfg, "n_updates", 0) or 0),
    )


def _resolve_yaw_clamp_anneal(cfg):
    """Parse the yaw-COMMAND-clamp anneal from cfg.env, or None when OFF (byte-identical default).
    Gated by ``+env.yaw_clamp_anneal`` (truthy); ``+env.yaw_clamp_scale_start/hold_frac`` optional.
    Scales the applied clamp (env._yaw_cmd_clamp) from start_scale*base DOWN to base (END-HOLD via
    _spin_abort_schedule -- the guarantee lives at the END, same argument as the spin-abort anneal).
    THE JOINT-ANNEAL RATIONALE (2026-07-11 matrix): the lineage's only discoverable flying gait is
    the SPIN gait -- clamp 0.7 from birth blocks its discovery (vpaa0: no liftoff ever), while a
    free-yaw policy that DOES lift off refuses to de-spin when only the fence tightens (vpaa1:
    flew mid-run, then collapsed into 87% spin-deaths). This anneal squeezes the gait's yaw
    AMPLITUDE gradually while it still has room to fly; run it FASTER than the fence anneal
    (hold_frac default 0.4 > the fence's 0.25 means the clamp lands at base by 60% of budget,
    the fence at 75%) so de-spinning stays ahead of the executioner. start default 4.57: base 0.7
    -> ~3.2 rad/s at birth (>= the +/-3.14 rail == armed-but-effectively-free; NEVER start from
    scale 0: clamp<=0 means OFF and small positive values mean nearly-FROZEN yaw -- bigger clamp
    == looser is the sign convention). PURE getattr."""
    env = getattr(cfg, "env", None)
    if env is None or not bool(getattr(env, "yaw_clamp_anneal", False)):
        return None
    return dict(
        start_scale=float(getattr(env, "yaw_clamp_scale_start", 4.57)),
        hold_frac=float(getattr(env, "yaw_clamp_hold_frac", 0.4)),
        n_updates=int(getattr(cfg, "n_updates", 0) or 0),
    )


def _resolve_perception_anneal(cfg):
    """Parse the PERCEPTION-reward (r_perc pointing carrot) anneal from cfg.env, or None when OFF
    (byte-identical default). Gated by ``+env.perception_anneal`` (truthy);
    ``+env.perception_scale_start/hold_frac`` optional. FARM-THEN-WEAN (Fengyou 2026-07-11): the
    boot crisis showed plain exploration's only discoverable flying gait is the SPIN gait, while
    r_perc's known 'failure' at meaningful weight -- the airborne gate-STARING farm -- is precisely
    a fence-legal, clamp-compatible, non-spinning attractor exploration CAN find (pointing pays for
    attitude control from tick 0, long before gate passage pays anything). So: start the carrot BIG
    (scale_start*base, default 25*0.02 = 0.5/tick), let the farm teach liftoff + pointing, then
    anneal DOWN to the configured farm-neutral base (END-HOLD via _spin_abort_schedule) so racing
    rewards take over and the farm dissolves before graduation. Mutates env._egorw.perception (the
    exact object ego_reward reads). Base MUST be armed (>0): annealing scale*0 is stuck OFF (L16).
    PURE getattr."""
    env = getattr(cfg, "env", None)
    if env is None or not bool(getattr(env, "perception_anneal", False)):
        return None
    return dict(
        start_scale=float(getattr(env, "perception_scale_start", 25.0)),
        hold_frac=float(getattr(env, "perception_hold_frac", 0.3)),
        n_updates=int(getattr(cfg, "n_updates", 0) or 0),
    )


def _resolve_progress_ramp(cfg):
    """Parse the forward-PROGRESS-reward RAMP-IN from cfg.env, or None when OFF (byte-identical
    default). Gated by ``+env.progress_ramp`` (truthy); ``+env.progress_ramp_start/hold_frac``
    optional. THE RE-DIVE FIX (2026-07-11, warm-boot vwhb0): warming the boot from an airborne
    hover ckpt buys a ~10x-longer opening (l_episode 0.33 -> 3.2) but the drone STILL slow-sinks to
    the floor, because rw_progress (dense per-metre homing toward a gate spawned 0.5-6 m ABOVE) pays
    for pitch-forward closure, which sheds vertical thrust -> the warmed altitude skill is gradually
    overpowered. This ramps the progress WEIGHT in from progress_ramp_start*base (default 0) UP to
    the configured base over the front, END-HOLD at base for the last hold_frac -- so the altitude
    skill stays dominant while forward flight is introduced gently, and the last hold_frac trains at
    the real (full) progress reward so the graduate is not reward-shifted. REUSES _spin_abort_schedule
    (start_scale=0 -> ramps 0->1 over the front, holds 1.0). Mutates env._egorw.progress. Base MUST be
    armed (>0): a zero-progress stage would ramp 0->0 (silent no-op, L16). start=0 is INTENDED here
    (unlike the clamp/perception hooks whose START is the loose extreme), so ONLY the base is guarded.
    PURE getattr."""
    env = getattr(cfg, "env", None)
    if env is None or not bool(getattr(env, "progress_ramp", False)):
        return None
    return dict(
        start_scale=float(getattr(env, "progress_ramp_start", 0.0)),
        hold_frac=float(getattr(env, "progress_ramp_hold_frac", 0.3)),
        n_updates=int(getattr(cfg, "n_updates", 0) or 0),
    )


def _resolve_att_cap_anneal(cfg):
    """Parse the ATTITUDE-CAP penalty-WEIGHT RAMP-IN from cfg.env, or None when OFF (byte-identical
    default). Gated by ``+env.att_cap_anneal`` (truthy); ``+env.att_cap_start/hold_frac`` optional.
    THE GENTLE-CAP LEVER (Track B, 2026-07-12): a SHORT warm-started fine-tune whose ONLY job is to reshape
    the WORKING ego champion (vpeffs0) so it flies natively inside a pitch/roll band. Hot-applying a
    full-strength attitude penalty to a competent policy DETONATES it (the nodither lesson: a strong
    penalty slammed onto a working flyer destroys it), so this ramps the SOFT attitude-cap WEIGHTS
    (att_pitch AND att_roll) in from att_cap_start*base (default 0 -> the caps are INERT at birth = the
    vpeffs0 behaviour untouched) UP to the configured target weights over the FRONT (1-hold_frac), then
    END-HOLDs at the full caps for the last hold_frac -- so the drone reshapes GRADUALLY into the band and
    the saved checkpoint's converged regime IS the full cap (END-HOLD, the same guarantee-at-the-end
    argument as progress_ramp / spin-abort / yaw-clamp). The LIMITS (att_pitch_limit_rad /
    att_roll_limit_rad) are FIXED from update 0; ONLY the penalty WEIGHTS ramp. REUSES _spin_abort_schedule
    (start_scale=0 -> ramps 0->1 over the front, holds 1.0). Mutates env._egorw.att_pitch AND
    env._egorw.att_roll (the exact object ego_reward reads). At least ONE base MUST be armed (>0): ramping
    scale*0 on BOTH is a silent no-op under an annealed run name (footgun L16). start_scale=0 is INTENDED
    (ramp in from no cap, unlike the clamp/perception hooks whose START is the loose extreme), so ONLY the
    bases are guarded. PURE getattr."""
    env = getattr(cfg, "env", None)
    if env is None or not bool(getattr(env, "att_cap_anneal", False)):
        return None
    return dict(
        start_scale=float(getattr(env, "att_cap_start", 0.0)),
        hold_frac=float(getattr(env, "att_cap_hold_frac", 0.3)),
        n_updates=int(getattr(cfg, "n_updates", 0) or 0),
    )


def _resolve_yaw_dither_anneal(cfg):
    """Parse the ANTI-DITHER yaw-jerk penalty-WEIGHT RAMP-IN from cfg.env, or None when OFF (byte-identical
    default). Gated by ``+env.yaw_dither_anneal`` (truthy); ``+env.yaw_dither_start/hold_frac`` optional.
    THE ANTI-DITHER LEVER (Track A, 2026-07-13): the anti-dither yaw-jerk penalty (ego_reward.yaw_dither_
    penalty, R_yawdith = -rw_yaw_dither*(yaw_cmd_t - yaw_cmd_{t-1})^2) removes the yaw command's +-clamp rail-
    flip oscillation the position-free obs leaves unpriced. HOT-applying it at full strength to a competent
    champion DETONATES the policy (the 'nodither' fine-tune collapsed the 96.9% flyer to 0.16% -- a strong
    penalty slammed onto a working flyer destroys it, the SAME failure mode the att-cap anneal exists for).
    This ramps the yaw-dither WEIGHT in from yaw_dither_start*base (default 0 -> the term is INERT at birth =
    the champion behaviour untouched) UP to the configured target weight over the FRONT (1-hold_frac), then
    END-HOLDs at the full weight for the last hold_frac -- so the policy grows the still-yaw skill GRADUALLY
    and the saved checkpoint's converged regime IS the full anti-dither penalty (END-HOLD, the same guarantee-
    at-the-end argument as att_cap_anneal / progress_ramp / spin-abort / yaw-clamp). REUSES _spin_abort_schedule
    (start_scale=0 -> ramps 0->1 over the front, holds 1.0). Mutates env._egorw.yaw_dither (the exact object
    ego_reward reads). Base MUST be armed (>0): ramping scale*0 is a silent no-op under an annealed run name
    (footgun L16). start_scale=0 is INTENDED (ramp in from no penalty, unlike the clamp/perception hooks whose
    START is the loose extreme), so ONLY the base is guarded. PURE getattr."""
    env = getattr(cfg, "env", None)
    if env is None or not bool(getattr(env, "yaw_dither_anneal", False)):
        return None
    return dict(
        start_scale=float(getattr(env, "yaw_dither_start", 0.0)),
        hold_frac=float(getattr(env, "yaw_dither_hold_frac", 0.3)),
        n_updates=int(getattr(cfg, "n_updates", 0) or 0),
    )


def _resolve_recovery_anneal(cfg):
    """Parse the RECOVERY / DAMPING penalty-WEIGHT RAMP-IN from cfg.env, or None when OFF (byte-identical
    default). Gated by ``+env.recovery_anneal`` (truthy); ``+env.recovery_start/hold_frac`` optional.
    THE RECOVERY LEVER (2026-07-14): the recovery reward (ego_reward.recovery_roll_penalty (A) and/or
    cross_level_penalty (B)) teaches bank-hard-then-LEVEL -- it prices a SUSTAINED bank while the gate is
    lined up ahead (A) and/or a BANKED gate-crossing (B) so the gate-over-gate roll ACCUMULATION (22deg ->
    70deg) damps, WITHOUT penalizing the ~61deg turn-bank the course needs (form A frees an off-axis gate;
    form B is silent between gates). HOT-applying a roll penalty at full strength to a competent champion
    risks the same DETONATION the att-cap / yaw-dither anneals exist for (a strong penalty slammed onto a
    working flyer destroys it), so this ramps BOTH recovery WEIGHTS (roll_recover AND cross_level) in from
    recovery_start*base (default 0 -> the terms are INERT at birth = the base behaviour untouched) UP to the
    configured target weights over the FRONT (1-hold_frac), then END-HOLDs at the full weights for the last
    hold_frac -- so the policy grows the level-out / level-cross skill GRADUALLY and the saved checkpoint's
    converged regime IS the full recovery penalty (END-HOLD, the same guarantee-at-the-end argument as
    att_cap_anneal / yaw_dither_anneal / progress_ramp / spin-abort / yaw-clamp). The bearing half-width
    (roll_recover_theta0_rad) is FIXED from update 0; ONLY the penalty WEIGHTS ramp. REUSES
    _spin_abort_schedule (start_scale=0 -> ramps 0->1 over the front, holds 1.0). Mutates env._egorw.
    roll_recover AND env._egorw.cross_level (the exact object ego_reward reads). At least ONE base MUST be
    armed (>0): ramping scale*0 on BOTH is a silent no-op under an annealed run name (footgun L16).
    start_scale=0 is INTENDED (ramp in from no penalty, unlike the clamp/perception hooks whose START is the
    loose extreme), so ONLY the bases are guarded. PURE getattr."""
    env = getattr(cfg, "env", None)
    if env is None or not bool(getattr(env, "recovery_anneal", False)):
        return None
    return dict(
        start_scale=float(getattr(env, "recovery_start", 0.0)),
        hold_frac=float(getattr(env, "recovery_hold_frac", 0.3)),
        n_updates=int(getattr(cfg, "n_updates", 0) or 0),
    )


_DET_EVAL_KEYS = ("success_rate", "collision_rate", "miss_rate", "oob_rate", "n_passed_gates")


def _run_det_eval(self, env, agent, cfg, tag="", yaw_log=False, delay=None):
    """FAITHFUL deterministic box-exit on the LIVE training env, AFTER training completes.

    Motivation: the training box-exit is STOCHASTIC, and the standalone offline harness
    (ego_render_rollout.py) diverges badly from training (reports ~46% oob where training logs 0.05%),
    so we had NO trustworthy measurement of the DEPLOYED (deterministic-mean) policy. This runs the
    policy with test=True on ``self.env`` -- the exact env instance that produced the trusted training
    metrics -- so it is faithful by construction. It runs post-training (checkpoint already saved), so
    it cannot affect the result. Prints a greppable ``DET_EVAL[...]`` line. ``+eval_det_steps=0`` skips.

    Aggregates ``n_passed_gates`` (mean gates threaded per completed episode -- the env exposes it in
    stats_raw at peregrine_racing_ego.py) alongside success/collision/miss/oob: at 8 gates episode
    success collapses toward 0, so n_passed_gates is the DISCRIMINATING BANK-FIRST metric that drives
    the rolling-best checkpoint selection + the seed-selection tiebreak. ``tag`` decorates the runname in
    the print (e.g. ``/best``) so the best-vs-final promotion evals stay greppable apart. RETURNS the
    per-episode-mean metrics dict (or None), consumed by _promote_best_on_npg."""
    steps = int(getattr(cfg, "eval_det_steps", 300) or 0)
    if steps <= 0 or env is None or agent is None:
        return None
    import torch
    label = f"{getattr(cfg, 'runname', '?')}{tag}"
    try:
        agg, n_ep = {}, 0
        ys = None                                          # yaw accumulators (lazy init; only when yaw_log)
        roll_sum = 0.0                                      # sum of per-episode peak |roll| (deg); yaw_log
        obs = env.reset()
        with torch.no_grad():
            for _ in range(steps):
                action, _ = agent.act(obs, test=True)          # DETERMINISTIC mean (deployed policy)
                phys = env.rescale_action(action)              # PHYSICAL cmd; env re-clamps yaw internally
                obs, _l, _t, info = env.step(phys)
                sr = info.get("stats_raw", {})
                m = info.get("reset")
                if yaw_log:                                    # per-STEP yaw-hunting accumulation
                    ys = _accum_yaw(env, phys, m, ys)
                if m is None:
                    continue
                n = int(m.sum().item())
                if not n:
                    continue
                n_ep += n
                for k in _DET_EVAL_KEYS:
                    if k in sr:
                        agg[k] = agg.get(k, 0.0) + float(sr[k].sum().item())
                if yaw_log and "peak_roll_deg" in sr:
                    roll_sum += float(sr["peak_roll_deg"].sum().item())
        r = None
        if n_ep:
            r = {k: agg.get(k, 0.0) / n_ep for k in _DET_EVAL_KEYS}
            print(f"DET_EVAL[{label}] n_ep={n_ep} "
                  f"thread={r['success_rate']:.4f} collision={r['collision_rate']:.4f} "
                  f"miss={r['miss_rate']:.4f} oob={r['oob_rate']:.4f} "
                  f"n_passed_gates={r['n_passed_gates']:.4f}  (test=True, LIVE env, {steps} steps)")
        else:
            print(f"DET_EVAL[{label}] no episodes completed in {steps} steps")
        if yaw_log:                                        # single greppable YAW_EVAL[...] line
            _emit_yaw_eval(env, label, delay, steps, ys, roll_sum, n_ep, r)
        return r
    except Exception as e:  # never let the post-hoc eval fail a completed run
        print(f"DET_EVAL: FAILED ({type(e).__name__}: {e})")
        return None


def _accum_yaw(env, phys_action, reset_mask, ys):
    """Per-STEP YAW accumulation for the latency-vs-hunting diagnostic (only under _run_det_eval's
    ``yaw_log``). ``phys_action`` is the PHYSICAL action passed to env.step ([thrust, roll, pitch, yaw]
    rad/s); the env clamps channel 3 internally, so we re-apply the SAME clamp_yaw_command(env._yaw_cmd_
    clamp) to recover the APPLIED (post-clamp) yaw command. Achieved yaw = env._w[...,2] (realized FLU
    body omega_z) read AFTER the step. Sign-flips use a 0.05 rad/s DEADBAND, per-env, matching deploy's
    method: the committed sign only updates when |cmd|>deadband, and a flip is counted when that committed
    sign reverses. ``reset_mask`` zeroes the committed sign at episode boundaries (no cross-episode flip)
    and excludes just-reset envs from the ACHIEVED mean (env._w is re-seeded by reset_idx that step).
    Returns the (lazily-initialised) accumulator dict."""
    import torch
    clamp = float(getattr(env, "_yaw_cmd_clamp", 0.0) or 0.0)
    cmd_yaw = clamp_yaw_command(phys_action, clamp)[..., 3].reshape(-1)
    ach_yaw = env._w[..., 2].reshape(-1)
    if ys is None:
        z = torch.zeros_like(cmd_yaw)
        ys = {"last_sign": z.clone(), "flips": z.clone(),
              "cmd_abs_sum": cmd_yaw.new_zeros(()), "cmd_n": 0,
              "ach_abs_sum": cmd_yaw.new_zeros(()), "ach_n": cmd_yaw.new_zeros(())}
    active = cmd_yaw.abs() > 0.05                           # DEADBAND (rad/s)
    s = torch.sign(cmd_yaw)
    prev = ys["last_sign"]
    flip = active & (prev != 0) & (s != prev)
    ys["flips"] = ys["flips"] + flip.to(cmd_yaw.dtype)
    ys["last_sign"] = torch.where(active, s, prev)
    ys["cmd_abs_sum"] = ys["cmd_abs_sum"] + cmd_yaw.abs().sum()
    ys["cmd_n"] += int(cmd_yaw.numel())
    if reset_mask is not None:
        rm = reset_mask.reshape(-1).to(torch.bool)
        valid = (~rm).to(cmd_yaw.dtype)
        ys["last_sign"] = torch.where(rm, torch.zeros_like(ys["last_sign"]), ys["last_sign"])
    else:
        valid = torch.ones_like(cmd_yaw)
    ys["ach_abs_sum"] = ys["ach_abs_sum"] + (ach_yaw.abs() * valid).sum()
    ys["ach_n"] = ys["ach_n"] + valid.sum()
    return ys


def _emit_yaw_eval(env, label, delay, steps, ys, roll_sum, n_ep, r):
    """Compute + print the single greppable ``YAW_EVAL[...]`` line from the per-step accumulators.
    dt = env.dt (control-step seconds). ``signflips_per_s`` is the MEAN per-env commanded-yaw sign-flip
    rate = flip_total / (n_envs * steps * dt) -- directly comparable to deploy's ~8/s single-drone rate.
    ``cmd_absmean`` / ``ach_absmean`` = mean |commanded| / |achieved| yaw rate (rad/s). ``roll_swing`` =
    mean per-episode peak |roll| (deg) over completed episodes. ``n_passed`` from the DET episode
    aggregation (nan if no episode completed)."""
    dt = float(getattr(env, "dt", 0.0) or 0.0)
    dstr = str(delay) if delay is not None else "?"
    if ys is None or dt <= 0.0:
        print(f"YAW_EVAL[{label}] delay={dstr} signflips_per_s=nan cmd_absmean=nan ach_absmean=nan "
              f"roll_swing=nan n_passed=nan  (no steps accumulated or dt unavailable)")
        return
    n_envs = int(ys["last_sign"].numel())
    flip_total = float(ys["flips"].sum().item())
    total_env_s = n_envs * steps * dt
    signflips_per_s = flip_total / total_env_s if total_env_s > 0 else float("nan")
    cmd_absmean = float(ys["cmd_abs_sum"].item()) / max(int(ys["cmd_n"]), 1)
    ach_n = float(ys["ach_n"].item())
    ach_absmean = (float(ys["ach_abs_sum"].item()) / ach_n) if ach_n > 0 else float("nan")
    roll_swing = (roll_sum / n_ep) if n_ep else float("nan")
    n_passed = float(r.get("n_passed_gates", float("nan"))) if r else float("nan")
    print(f"YAW_EVAL[{label}] delay={dstr} signflips_per_s={signflips_per_s:.3f} "
          f"cmd_absmean={cmd_absmean:.4f} ach_absmean={ach_absmean:.4f} "
          f"roll_swing={roll_swing:.2f} n_passed={n_passed:.4f}")


def _run_rollout_only(self, env, agent, cfg):
    """EVAL-ONLY (``++rollout_only=true``): load an actor+critic checkpoint into the freshly-built agent
    and run ONE faithful deterministic det-eval WITH yaw logging on the live env, then EXIT -- no
    training, no periodic/best saves, no promotion. Fully opt-in: the whole training path is untouched
    when ``rollout_only`` is unset. The checkpoint (``++rollout_ckpt=<dir>``, or ``+init_from=<dir>`` as a
    fallback) MUST be a full agent.save dir (actor.pth + critic.pth) -- the runner's ``checkpoints/`` dir;
    ``agent.load`` is diffaero's PPO.load, the same call _promote_best_on_npg uses. The injected latency
    (``++dynamics.transport_delay_steps=K``) is read back from the LIVE plant purely to decorate the
    YAW_EVAL line. Returns None."""
    import os
    raw = getattr(cfg, "rollout_ckpt", None)
    if raw in (None, "", "none", "null"):
        raw = getattr(cfg, "init_from", None)
    ckpt = None if raw in (None, "", "none", "null") else str(raw)
    if ckpt is None or not os.path.isdir(ckpt):
        raise FileNotFoundError(
            f"[rollout-only] ++rollout_ckpt (or +init_from) must be a checkpoint DIR holding actor.pth + "
            f"critic.pth (a runner checkpoints/ dir); got {ckpt!r}.")
    agent.load(ckpt)
    # applied latency = the LIVE plant's ring-buffer depth (source of truth), decorates YAW_EVAL.
    delay = None
    prm = getattr(getattr(env, "dynamics", None), "params", None)
    if prm is not None:
        delay = int(getattr(prm, "transport_delay_steps", 0) or 0)
    tag = str(getattr(cfg, "rollout_tag", "") or "")
    tag = f"/{tag}" if tag else ""
    print(f"[rollout-only] LOADED {ckpt}; transport_delay_steps={delay}; running yaw det-eval "
          f"(eval_det_steps={int(getattr(cfg, 'eval_det_steps', 300) or 0)}, "
          f"n_envs={getattr(cfg, 'n_envs', '?')}, yaw_cmd_clamp={getattr(env, '_yaw_cmd_clamp', '?')}).")
    _run_det_eval(self, env, agent, cfg, tag=tag, yaw_log=True, delay=delay)
    return None


# ================================================================================================
# STAGE-1 (vtrackAr5) checkpoint-selection + critic-grad-clip helpers. All gated OFF by default so the
# non-select path is byte-identical.
# ================================================================================================
def _harvest_npg(info, buf):
    """PASSIVE read of the per-completed-episode ``n_passed_gates`` tensor from a step ``info`` dict into
    the recent-window ``buf`` (a deque). The env already materializes ``stats_raw['n_passed_gates']`` (a
    reset-masked, already-synced tensor of the episodes that terminated THIS step -- the same tensor
    DET_EVAL reads) on every step, so this is a PURE read: no extra env.step/reset, no new device sync, no
    desync. Returns the count harvested. No-op (0) when ``buf`` is None, ``info`` is not a dict, or the key
    is absent/empty."""
    if buf is None or not isinstance(info, dict):
        return 0
    sr = info.get("stats_raw")
    if not isinstance(sr, dict):
        return 0
    npg = sr.get("n_passed_gates")
    if npg is None:
        return 0
    try:
        if hasattr(npg, "numel"):
            if int(npg.numel()) == 0:
                return 0
            vals = npg.detach().flatten().cpu().tolist()
        else:
            vals = list(npg)
    except Exception:                                   # pragma: no cover - never fail a training step
        return 0
    buf.extend(vals)
    return len(vals)


def _select_critic_params(agent):
    """The CRITIC-ONLY parameter set for the Stage-1 critic-grad-clip: every named parameter whose name
    contains ``'critic'`` and currently carries a grad. Selecting by NAME keeps the actor untouched --
    actor_grad_norm is a healthy ~3, while the 8-gate critic_grad spikes to ~1519 (m8)/918 (m8b); we clip
    ONLY the critic so a value-net spike cannot smear the healthy policy gradient."""
    return [p for n, p in agent.agent.named_parameters() if "critic" in n and p.grad is not None]


def _find_runner_checkpoints(logger, best_dir=None):
    """Locate the runner's end-of-run ``checkpoints/`` dir (the dir the deploy pull + the next stage's
    ``+init_from`` consume). The runner writes ``best_npg/``, ``best/``, ``checkpoints/``, ``periodic/``
    side by side under the SAME logdir, so ``checkpoints/`` is a SIBLING of ``best_npg/``.

    PRIMARY resolution (Bug 2 fix): when the ACTUAL ``best_npg/`` path is known (``best_dir``, already
    verified to exist by the caller), resolve ``checkpoints/`` as its SIBLING -- ``dirname(best_dir)/
    checkpoints``. This pins the exact filesystem root the runner really saved into, and is immune to
    ``logger.logdir`` re-resolving to a DIFFERENT root at promotion time (the observed failure: best_npg/
    lived under ``.../diffaero_repo/outputs/train/<date>/<time>/`` while the logdir-based guess pointed at
    ``.../diffaero/outputs/train/<runname>/`` -- a path-root mismatch, so the ``near RUNDIR`` heuristic
    could not see the checkpoints/ dir that DID exist next to best_npg/).

    FALLBACK: if the sibling-of-best_dir is absent (or best_dir was not supplied), fall back to the old
    ``logger.logdir`` candidates (in-logdir and parent-of-logdir). Return the first that exists, else None
    (promotion then FLAGS + skips rather than guessing)."""
    cands = []
    if best_dir:
        # SIBLING of the real best_npg/ -- same root the runner actually saved to. Tried FIRST.
        cands.append(os.path.join(os.path.dirname(os.path.normpath(best_dir)), "checkpoints"))
    logdir = getattr(logger, "logdir", None)
    if logdir:
        cands.append(os.path.join(logdir, "checkpoints"))
        cands.append(os.path.join(os.path.dirname(os.path.normpath(logdir)), "checkpoints"))
    for c in cands:
        if os.path.isdir(c):
            return c
    return None


def _promote_best_on_npg(self, env, agent, cfg, npg_state, final_metrics, logger):
    """After training: DET-eval the rolling ``best_npg/`` snapshot vs the runner's ``checkpoints/`` (the
    FINAL == the in-memory agent, already DET-eval'd into ``final_metrics``), and if best's DETERMINISTIC
    n_passed_gates >= final's, copy ``best_npg/`` over ``checkpoints/`` so ``+init_from`` and the deploy
    pull consume the PEAK (every 8-gate run peaks then regresses 24-42% and otherwise ships its degraded
    final). Loads best_npg/ into the agent to eval it (post-run: harmless). No-op if best_npg/ was never
    saved (never crossed a save boundary). NEVER selects on value/reward -- only deterministic gates."""
    best_dir = npg_state.get("best_dir")
    if not best_dir or not os.path.isdir(best_dir):
        print("[ckpt-select] no best_npg/ snapshot to promote (never crossed a save boundary) -- "
              "checkpoints/ (the final) ships unchanged.")
        return
    ckpt_dir = _find_runner_checkpoints(logger, best_dir=best_dir)
    if ckpt_dir is None:
        print(f"[ckpt-select] FLAG: runner checkpoints/ dir not found near {getattr(logger,'logdir','?')} "
              f"-- cannot promote. best_npg/ is at {best_dir}; point +init_from / the deploy pull there "
              f"manually if it out-scored the final.")
        return
    final_npg = float(final_metrics.get("n_passed_gates", float("-inf"))) if final_metrics else float("-inf")
    agent.load(best_dir)                                # load the peak snapshot to DET-eval it
    best_metrics = _run_det_eval(self, env, agent, cfg, tag="/best",
                                 yaw_log=bool(getattr(cfg, "eval_yaw_log", False)))
    best_npg = float(best_metrics.get("n_passed_gates", float("-inf"))) if best_metrics else float("-inf")
    print(f"[ckpt-select] DET n_passed_gates: best_npg={best_npg:.4f} vs final={final_npg:.4f}")
    if best_npg > float("-inf") and best_npg >= final_npg:
        import shutil
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        shutil.copytree(best_dir, ckpt_dir)
        print(f"BEST_CKPT_PROMOTED[{getattr(cfg,'runname','?')}] best_npg/ -> checkpoints/ "
              f"(best {best_npg:.4f} >= final {final_npg:.4f}) -- deploy + init_from now consume the PEAK.")
    else:
        print(f"[ckpt-select] KEEP final (final {final_npg:.4f} > best {best_npg:.4f}); checkpoints/ "
              f"left as the runner saved it.")


def _run_with_ego_lifelines(self):
    """Wrap TrainRunner.run for the ego generation. AFTER the agent is built and BEFORE the training
    loop it (in order):
      (fix #1) fires env.assert_appo_critic so a ppo / symmetric-critic run raises AssertionError
               instead of silently seed-collapsing;
      (fix #2) installs the WARM-START LIFELINE -- periodic/emergency saves, +init_from load,
               +warmstart_reset_logstd, +critic_warmup_updates -- so the ego ladder's stage->stage
               warm-start chain is LIVE (mirrors rl/peregrine_train_inc8.py::_run_with_lifelines).
    Non-ego runs (``+env.ego`` unset/false) take the byte-identical fresh-init path (no lifeline)."""
    env, agent, cfg = getattr(self, "env", None), getattr(self, "agent", None), self.cfg
    ego_on = bool(getattr(getattr(cfg, "env", object()), "ego", False))

    # ---- EVAL-ONLY yaw rollout (++rollout_only=true): load a checkpoint, run ONE yaw-logging det-eval
    # on the live env, EXIT. Opt-in; unset -> this branch is skipped and the trainer stays byte-identical.
    # Placed FIRST so eval-only needs neither the appo assert nor the warm-start lifeline machinery. ----
    if env is not None and agent is not None and bool(getattr(cfg, "rollout_only", False)):
        return _run_rollout_only(self, env, agent, cfg)

    # ---- FIX #1: appo / privileged-critic assert (only meaningful on the ego path) ----
    if env is not None and agent is not None and hasattr(env, "assert_appo_critic") and ego_on:
        critic_input_dim = _built_critic_input_dim(agent)
        if critic_input_dim is None:
            raise RuntimeError(
                "[ego-launcher] could not read the built critic input_dim (agent.agent.critic.critic."
                "input_dim) -- the diffaero network API may have moved; reconcile peregrine_train_ego.py "
                "against the Adroit clone before training. The appo/privileged-critic assert MUST run.")
        env.assert_appo_critic(_algo_name(cfg), int(critic_input_dim))
        print(f"[ego-launcher] appo/privileged-critic assert PASSED: algo={_algo_name(cfg)!r}, "
              f"critic_input_dim={int(critic_input_dim)} == state_dim={env.state_dim} "
              f"(!= obs_dim={env.obs_dim}).")

    # ---- non-ego (OFF) path: byte-identical fresh-init trainer, no lifeline ----
    if not ego_on or agent is None:
        return _orig_run(self)

    # ================================ FIX #2: the warm-start lifeline ================================
    logger = self.logger

    # NOISE CEILING (RC2): resolve the std-ceiling schedule (None when +algo.noise_anneal unset ->
    # byte-identical). Applied per-update BELOW, BEFORE each rollout, so actor_logstd cannot run away to
    # the LOG_STD_MAX bang-bang ceiling. Mirror of rl/peregrine_train_inc8.py's proven wiring.
    noise_sched = resolve_noise_anneal(cfg)
    if noise_sched is not None:
        print(f"[noise-anneal] ON: {noise_sched}")

    # PARABOLA-ZERO anneal (Fengyou 2026-07-08): shrink env._egorw.cross_zero_m gradually WITHIN the run
    # (the in-run cure for the discrete-shrink warm-start collapse). Requires the live racing env + parabola
    # on; None (OFF) is byte-identical. env._egorw IS the exact object the reward reads (self._egorw), so
    # mutating cross_zero_m before each rollout takes effect the next step.
    # STRICT (footgun L16, 2026-07-09): a requested-but-unfindable holder RAISES by default via
    # _require_anneal_holder (opt into the old silent-skip with +anneal_allow_skip=true) -- applies to
    # all three anneal hooks below (cross-zero, noise-scale, clip-terminal).
    cz_sched = _resolve_cross_zero_anneal(cfg)
    cz_env = _require_anneal_holder(env, "_egorw", cz_sched, "cross-zero-anneal", cfg)
    if cz_sched is not None:
        if cz_env is not None:
            print(f"[cross-zero-anneal] ON: {cz_sched} (from {getattr(cz_env._egorw, 'cross_zero_m', '?')})")
        else:
            cz_sched = None          # allow_skip path -- _require_anneal_holder already printed SKIPPED

    # NOISE CURRICULUM (Fengyou 2026-07-09): anneal the estimator global noise multiplier ego_noise_scale
    # from start->end WITHIN the run (the data-motivated response to vglpns0 = perception-noise-limited).
    # _estimator.set_noise_scale mutates the frozen EgoEstimatorConfig live; None (OFF) is byte-identical.
    ns_sched = _resolve_noise_scale_anneal(cfg)
    ns_env = _require_anneal_holder(env, "_estimator", ns_sched, "noise-scale-anneal", cfg)
    if ns_sched is not None:
        if ns_env is not None:
            print(f"[noise-scale-anneal] ON: {ns_sched} (from {getattr(ns_env._estimator.cfg, 'noise_scale', '?')})")
        else:
            ns_sched = None          # allow_skip path -- _require_anneal_holder already printed SKIPPED

    # CLIP-TERMINAL PENALTY anneal: ramp env._egorw.clip_terminal_w from start->end WITHIN the run (a
    # lever a parallel agent is adding to ego_reward.py). Mirrors the cross-zero/noise-scale pattern
    # exactly; None (OFF) is byte-identical.
    ct_sched = _resolve_clip_terminal_anneal(cfg)
    ct_env = _require_anneal_holder(env, "_egorw", ct_sched, "clip-terminal-anneal", cfg)
    if ct_sched is not None:
        if ct_env is not None:
            print(f"[clip-terminal-anneal] ON: {ct_sched} (from {getattr(ct_env._egorw, 'clip_terminal_w', '?')})")
        else:
            ct_sched = None          # allow_skip path -- _require_anneal_holder already printed SKIPPED

    # SPIN-ABORT THRESHOLD anneal (boot-learnability crisis, 2026-07-11): scale BOTH fence triggers
    # loose at birth and anneal to the EXACT configured fence (END-HOLD -- see _spin_abort_schedule's
    # docstring for why the guarantee direction is reversed vs the other anneals). Mutates the raw
    # env's _spin_rate_abort / _spin_rev_abort, which step() reads per tick. Bases are captured HERE
    # (before the first mutation) and MUST be armed (>0): a requested anneal on a fence-off stage
    # would be a silent no-op running under an annealed name -- exactly footgun L16 -- so it RAISES.
    sa_sched = _resolve_spin_abort_anneal(cfg)
    sa_env = _require_anneal_holder(env, "_spin_rate_abort", sa_sched, "spin-abort-anneal", cfg)
    if sa_sched is not None:
        if sa_env is not None:
            sa_sched["base_rate"] = float(sa_env._spin_rate_abort)
            sa_sched["base_rev"] = float(sa_env._spin_rev_abort)
            if sa_sched["base_rate"] <= 0.0 or sa_sched["base_rev"] <= 0.0:
                raise RuntimeError(
                    "[spin-abort-anneal] requested but the fence is (partly) OFF (base rate_abort="
                    f"{sa_sched['base_rate']:.3f}, rev_abort={sa_sched['base_rev']:.3f}) -- annealing a "
                    "disabled trigger is a silent no-op under an annealed run name (footgun L16); arm "
                    "ego_spin_rate_abort AND ego_spin_rev_abort, or drop +env.spin_abort_anneal.")
            print(f"[spin-abort-anneal] ON: {sa_sched} (END-HOLD: exact fence for the last "
                  f"{sa_sched['hold_frac']:.0%} of updates)")
        else:
            sa_sched = None          # allow_skip path -- _require_anneal_holder already printed SKIPPED

    # YAW-CLAMP anneal (the joint-anneal cell; see _resolve_yaw_clamp_anneal's rationale). Mutates
    # env._yaw_cmd_clamp (read per step at the clamp_yaw_command application site). Base captured
    # pre-mutation and MUST be armed (>0): clamp<=0 means OFF, and an anneal toward OFF would pass
    # through nearly-frozen yaw (the sign convention is bigger == looser) -- so it RAISES (L16).
    yc_sched = _resolve_yaw_clamp_anneal(cfg)
    yc_env = _require_anneal_holder(env, "_yaw_cmd_clamp", yc_sched, "yaw-clamp-anneal", cfg)
    if yc_sched is not None:
        if yc_env is not None:
            yc_sched["base"] = float(yc_env._yaw_cmd_clamp)
            if yc_sched["base"] <= 0.0:
                raise RuntimeError(
                    "[yaw-clamp-anneal] requested but the clamp is OFF (base ego_yaw_cmd_clamp_rad_s="
                    f"{yc_sched['base']:.3f}) -- annealing scale*0 is stuck at OFF forever (silent no-op "
                    "under an annealed run name, footgun L16); arm ego_yaw_cmd_clamp_rad_s (the END value, "
                    "e.g. 0.7) or drop +env.yaw_clamp_anneal.")
            print(f"[yaw-clamp-anneal] ON: {yc_sched} (END-HOLD: exact clamp for the last "
                  f"{yc_sched['hold_frac']:.0%} of updates)")
        else:
            yc_sched = None          # allow_skip path -- _require_anneal_holder already printed SKIPPED

    # PERCEPTION-CARROT anneal (farm-then-wean; see _resolve_perception_anneal's rationale).
    # Mutates env._egorw.perception -- the same holder as the cross-zero/clip-terminal hooks.
    # Base captured pre-mutation and MUST be armed (>0): scale*0 is stuck OFF forever (L16).
    pc_sched = _resolve_perception_anneal(cfg)
    pc_env = _require_anneal_holder(env, "_egorw", pc_sched, "perception-anneal", cfg)
    if pc_sched is not None:
        if pc_env is not None:
            pc_sched["base"] = float(getattr(pc_env._egorw, "perception", 0.0))
            if pc_sched["base"] <= 0.0:
                raise RuntimeError(
                    "[perception-anneal] requested but r_perc is OFF (base rw_perception="
                    f"{pc_sched['base']:.4f}) -- annealing scale*0 is stuck at OFF forever (silent "
                    "no-op under an annealed run name, footgun L16); arm rw_perception (the END "
                    "farm-neutral value, e.g. 0.02) or drop +env.perception_anneal.")
            print(f"[perception-anneal] ON: {pc_sched} (farm-then-wean: END-HOLD at the "
                  f"farm-neutral base for the last {pc_sched['hold_frac']:.0%} of updates)")
        else:
            pc_sched = None          # allow_skip path -- _require_anneal_holder already printed SKIPPED

    # PROGRESS RAMP-IN (the re-dive fix; see _resolve_progress_ramp's rationale). Mutates
    # env._egorw.progress from start_scale*base UP to base (END-HOLD). Base captured pre-mutation and
    # MUST be armed (>0): a zero-progress stage would ramp 0->0 (L16 silent no-op). start_scale=0 is
    # INTENDED (ramp in from no forward pull), so only the base is guarded.
    pr_sched = _resolve_progress_ramp(cfg)
    pr_env = _require_anneal_holder(env, "_egorw", pr_sched, "progress-ramp", cfg)
    if pr_sched is not None:
        if pr_env is not None:
            pr_sched["base"] = float(getattr(pr_env._egorw, "progress", 0.0))
            if pr_sched["base"] <= 0.0:
                raise RuntimeError(
                    "[progress-ramp] requested but rw_progress is OFF (base progress="
                    f"{pr_sched['base']:.4f}) -- ramping 0->0 is a silent no-op under an annealed run "
                    "name (footgun L16); arm rw_progress (the full forward-pull value) or drop "
                    "+env.progress_ramp.")
            print(f"[progress-ramp] ON: {pr_sched} (RAMP-IN {pr_sched['start_scale']:.2f}*base -> "
                  f"base, END-HOLD at full for the last {pr_sched['hold_frac']:.0%} of updates)")
        else:
            pr_sched = None          # allow_skip path -- _require_anneal_holder already printed SKIPPED

    # ATTITUDE-CAP RAMP-IN (Track B, 2026-07-12; see _resolve_att_cap_anneal's rationale). Mutates BOTH
    # env._egorw.att_pitch AND env._egorw.att_roll from att_cap_start*base UP to base (END-HOLD) so the
    # working flyer reshapes gradually into the pitch/roll band. Bases captured pre-mutation; at least ONE
    # MUST be armed (>0): ramping 0->0 on both is a silent no-op under an annealed run name (L16).
    # start_scale=0 is INTENDED (ramp in from no cap), so only the bases are guarded.
    ac_sched = _resolve_att_cap_anneal(cfg)
    ac_env = _require_anneal_holder(env, "_egorw", ac_sched, "att-cap-anneal", cfg)
    if ac_sched is not None:
        if ac_env is not None:
            ac_sched["base_pitch"] = float(getattr(ac_env._egorw, "att_pitch", 0.0))
            ac_sched["base_roll"] = float(getattr(ac_env._egorw, "att_roll", 0.0))
            if ac_sched["base_pitch"] <= 0.0 and ac_sched["base_roll"] <= 0.0:
                raise RuntimeError(
                    "[att-cap-anneal] requested but BOTH attitude caps are OFF (base rw_att_pitch="
                    f"{ac_sched['base_pitch']:.4f}, rw_att_roll={ac_sched['base_roll']:.4f}) -- ramping "
                    "0->0 is a silent no-op under an annealed run name (footgun L16); arm rw_att_pitch "
                    "and/or rw_att_roll (the full cap weights) or drop +env.att_cap_anneal.")
            print(f"[att-cap-anneal] ON: {ac_sched} (RAMP-IN {ac_sched['start_scale']:.2f}*base -> "
                  f"base, END-HOLD at full for the last {ac_sched['hold_frac']:.0%} of updates)")
        else:
            ac_sched = None          # allow_skip path -- _require_anneal_holder already printed SKIPPED

    # ANTI-DITHER YAW-JERK RAMP-IN (Track A, 2026-07-13; see _resolve_yaw_dither_anneal's rationale). Mutates
    # env._egorw.yaw_dither from yaw_dither_start*base UP to base (END-HOLD) so the working flyer grows the
    # still-yaw skill gradually rather than being detonated by a hot full-strength penalty (the nodither
    # collapse). Base captured pre-mutation and MUST be armed (>0): ramping 0->0 is a silent no-op under an
    # annealed run name (L16). start_scale=0 is INTENDED (ramp in from no penalty), so only the base is guarded.
    yd_sched = _resolve_yaw_dither_anneal(cfg)
    yd_env = _require_anneal_holder(env, "_egorw", yd_sched, "yaw-dither-anneal", cfg)
    if yd_sched is not None:
        if yd_env is not None:
            yd_sched["base"] = float(getattr(yd_env._egorw, "yaw_dither", 0.0))
            if yd_sched["base"] <= 0.0:
                raise RuntimeError(
                    "[yaw-dither-anneal] requested but the anti-dither penalty is OFF (base rw_yaw_dither="
                    f"{yd_sched['base']:.4f}) -- ramping 0->0 is a silent no-op under an annealed run name "
                    "(footgun L16); arm rw_yaw_dither (the full anti-dither weight, e.g. 0.5) or drop "
                    "+env.yaw_dither_anneal.")
            print(f"[yaw-dither-anneal] ON: {yd_sched} (RAMP-IN {yd_sched['start_scale']:.2f}*base -> "
                  f"base, END-HOLD at full for the last {yd_sched['hold_frac']:.0%} of updates)")
        else:
            yd_sched = None          # allow_skip path -- _require_anneal_holder already printed SKIPPED

    # RECOVERY / DAMPING RAMP-IN (2026-07-14; see _resolve_recovery_anneal's rationale). Mutates BOTH
    # env._egorw.roll_recover AND env._egorw.cross_level from recovery_start*base UP to base (END-HOLD) so
    # the flyer grows the level-out / level-cross skill gradually rather than being detonated by a hot
    # full-strength roll penalty. Bases captured pre-mutation; at least ONE MUST be armed (>0): ramping
    # 0->0 on both is a silent no-op under an annealed run name (L16). start_scale=0 is INTENDED (ramp in
    # from no penalty), so only the bases are guarded.
    rc_sched = _resolve_recovery_anneal(cfg)
    rc_env = _require_anneal_holder(env, "_egorw", rc_sched, "recovery-anneal", cfg)
    if rc_sched is not None:
        if rc_env is not None:
            rc_sched["base_roll_recover"] = float(getattr(rc_env._egorw, "roll_recover", 0.0))
            rc_sched["base_cross_level"] = float(getattr(rc_env._egorw, "cross_level", 0.0))
            if rc_sched["base_roll_recover"] <= 0.0 and rc_sched["base_cross_level"] <= 0.0:
                raise RuntimeError(
                    "[recovery-anneal] requested but BOTH recovery weights are OFF (base rw_roll_recover="
                    f"{rc_sched['base_roll_recover']:.4f}, rw_cross_level={rc_sched['base_cross_level']:.4f}) "
                    "-- ramping 0->0 is a silent no-op under an annealed run name (footgun L16); arm "
                    "rw_roll_recover and/or rw_cross_level (the full recovery weights) or drop "
                    "+env.recovery_anneal.")
            print(f"[recovery-anneal] ON: {rc_sched} (RAMP-IN {rc_sched['start_scale']:.2f}*base -> "
                  f"base, END-HOLD at full for the last {rc_sched['hold_frac']:.0%} of updates)")
        else:
            rc_sched = None          # allow_skip path -- _require_anneal_holder already printed SKIPPED

    # CHECKPOINT SELECTION ON n_passed_gates (Stage-1 vtrackAr5). Gated on ++ckpt_select_metric=
    # n_passed_gates (unset -> byte-identical: no harvest wrapper, no best_npg/, no promotion). BANK-FIRST:
    # every 8-gate run PEAKS then regresses 24-42% and otherwise ships its degraded FINAL; selecting on the
    # deterministic n_passed_gates ships the PEAK down the warm chain. A PASSIVE env.step wrapper (installed
    # just after agent.step below) reads stats_raw['n_passed_gates'] into a recent-window deque; at each
    # save boundary we save best_npg/ on rolling-mean improvement (BEST_CKPT line), and after training the
    # promotion DET-evals best_npg/ vs the final and copies the winner over checkpoints/.
    select_on_npg = str(getattr(cfg, "ckpt_select_metric", "") or "").strip() == "n_passed_gates"
    npg_state = {"buf": None, "best": float("-inf"),
                 "best_dir": os.path.join(logger.logdir, "best_npg")}
    if select_on_npg:
        from collections import deque
        _nenvs = int(getattr(cfg, "n_envs", 0) or 0)
        _sf = int(getattr(cfg, "save_freq", 0) or 0)
        _maxlen = max(_nenvs * _sf * 2, 20000)          # recent-window ~ episodes over ~save_freq updates
        npg_state["buf"] = deque(maxlen=_maxlen)
        print(f"[ckpt-select] ON (++ckpt_select_metric=n_passed_gates): rolling-best -> "
              f"{npg_state['best_dir']} (recent-window deque maxlen={_maxlen}).")

    # PERIODIC SAVE every save_freq updates -> <logdir>/periodic (+ periodic_prev). This is the
    # crash-resilience save; the runner ALSO writes its own <rundir>/checkpoints end-of-run save (the
    # dir the NEXT stage's +init_from points at). Wrap agent.step (the same hook inc8 uses).
    orig_step = agent.step
    counter = {"i": 0}

    def step_with_periodic_save(*a, **k):
        # NOISE CEILING (RC2): clamp actor_logstd to the scheduled std ceiling BEFORE this rollout so the
        # sampled actions obey it (OFF -> noise_sched None -> skipped, byte-identical). Mirrors inc8.
        if noise_sched is not None:
            nv = apply_noise_schedule(agent, counter["i"], noise_sched)
            if counter["i"] % max(int(cfg.log_freq), 1) == 0:
                print(f"[noise-anneal] update {counter['i']}: std_ceil={nv['std_ceil']:.4f} "
                      f"entropy_weight={nv['entropy_weight']:.5f} (progress={nv['progress']:.2f})")
        if cz_sched is not None:
            czv = _cross_zero_schedule(counter["i"], cz_sched["n_updates"],
                                       cz_sched["start"], cz_sched["end"], cz_sched["hold_frac"])
            cz_env._egorw.cross_zero_m = czv
            if counter["i"] % max(int(cfg.log_freq), 1) == 0:
                print(f"[cross-zero-anneal] update {counter['i']}: cross_zero_m={czv:.3f}")
        if ns_sched is not None:
            nsv = _noise_scale_schedule(counter["i"], ns_sched["n_updates"],
                                        ns_sched["start"], ns_sched["end"], ns_sched["hold_frac"])
            ns_env._estimator.set_noise_scale(nsv)
            if counter["i"] % max(int(cfg.log_freq), 1) == 0:
                print(f"[noise-scale-anneal] update {counter['i']}: ego_noise_scale={nsv:.3f}")
        if ct_sched is not None:
            ctv = _cross_zero_schedule(counter["i"], ct_sched["n_updates"],
                                       ct_sched["start"], ct_sched["end"], ct_sched["hold_frac"])
            ct_env._egorw.clip_terminal_w = ctv
            if counter["i"] % max(int(cfg.log_freq), 1) == 0:
                print(f"[clip-terminal-anneal] update {counter['i']}: clip_terminal_w={ctv:.3f}")
        if sa_sched is not None:
            sav = _spin_abort_schedule(counter["i"], sa_sched["n_updates"],
                                       sa_sched["start_scale"], sa_sched["hold_frac"])
            sa_env._spin_rate_abort = sa_sched["base_rate"] * sav
            sa_env._spin_rev_abort = sa_sched["base_rev"] * sav
            if counter["i"] % max(int(cfg.log_freq), 1) == 0:
                print(f"[spin-abort-anneal] update {counter['i']}: scale={sav:.3f} "
                      f"rate_abort={sa_env._spin_rate_abort:.2f} rev_abort={sa_env._spin_rev_abort:.2f}")
        if yc_sched is not None:
            ycv = _spin_abort_schedule(counter["i"], yc_sched["n_updates"],
                                       yc_sched["start_scale"], yc_sched["hold_frac"])
            yc_env._yaw_cmd_clamp = yc_sched["base"] * ycv
            if counter["i"] % max(int(cfg.log_freq), 1) == 0:
                print(f"[yaw-clamp-anneal] update {counter['i']}: scale={ycv:.3f} "
                      f"yaw_cmd_clamp={yc_env._yaw_cmd_clamp:.2f}")
        if pc_sched is not None:
            pcv = _spin_abort_schedule(counter["i"], pc_sched["n_updates"],
                                       pc_sched["start_scale"], pc_sched["hold_frac"])
            pc_env._egorw.perception = pc_sched["base"] * pcv
            if counter["i"] % max(int(cfg.log_freq), 1) == 0:
                print(f"[perception-anneal] update {counter['i']}: scale={pcv:.3f} "
                      f"rw_perception={pc_env._egorw.perception:.4f}")
        if pr_sched is not None:
            prv = _spin_abort_schedule(counter["i"], pr_sched["n_updates"],
                                       pr_sched["start_scale"], pr_sched["hold_frac"])
            pr_env._egorw.progress = pr_sched["base"] * prv
            if counter["i"] % max(int(cfg.log_freq), 1) == 0:
                print(f"[progress-ramp] update {counter['i']}: scale={prv:.3f} "
                      f"rw_progress={pr_env._egorw.progress:.3f}")
        if ac_sched is not None:
            acv = _spin_abort_schedule(counter["i"], ac_sched["n_updates"],
                                       ac_sched["start_scale"], ac_sched["hold_frac"])
            ac_env._egorw.att_pitch = ac_sched["base_pitch"] * acv
            ac_env._egorw.att_roll = ac_sched["base_roll"] * acv
            if counter["i"] % max(int(cfg.log_freq), 1) == 0:
                print(f"[att-cap-anneal] update {counter['i']}: scale={acv:.3f} "
                      f"att_pitch={ac_env._egorw.att_pitch:.3f} att_roll={ac_env._egorw.att_roll:.3f}")
        if yd_sched is not None:
            ydv = _spin_abort_schedule(counter["i"], yd_sched["n_updates"],
                                       yd_sched["start_scale"], yd_sched["hold_frac"])
            yd_env._egorw.yaw_dither = yd_sched["base"] * ydv
            if counter["i"] % max(int(cfg.log_freq), 1) == 0:
                print(f"[yaw-dither-anneal] update {counter['i']}: scale={ydv:.3f} "
                      f"rw_yaw_dither={yd_env._egorw.yaw_dither:.3f}")
        if rc_sched is not None:
            rcv = _spin_abort_schedule(counter["i"], rc_sched["n_updates"],
                                       rc_sched["start_scale"], rc_sched["hold_frac"])
            rc_env._egorw.roll_recover = rc_sched["base_roll_recover"] * rcv
            rc_env._egorw.cross_level = rc_sched["base_cross_level"] * rcv
            if counter["i"] % max(int(cfg.log_freq), 1) == 0:
                print(f"[recovery-anneal] update {counter['i']}: scale={rcv:.3f} "
                      f"rw_roll_recover={rc_env._egorw.roll_recover:.3f} "
                      f"rw_cross_level={rc_env._egorw.cross_level:.3f}")
        out = orig_step(*a, **k)
        counter["i"] += 1
        if counter["i"] % max(int(cfg.save_freq), 1) == 0:
            if _weights_finite(agent):
                pdir = os.path.join(logger.logdir, "periodic")
                prev = os.path.join(logger.logdir, "periodic_prev")
                if os.path.isdir(pdir):
                    import shutil
                    shutil.rmtree(prev, ignore_errors=True)
                    os.replace(pdir, prev)
                agent.save(pdir)
            else:
                print("[lifeline] weights non-finite at update %d -- periodic save SKIPPED"
                      % counter["i"])
            # ROLLING-BEST on deterministic n_passed_gates (Stage-1): at the SAME save boundary, if the
            # recent-window rolling mean improved, snapshot best_npg/. Passive read of the harvest buffer
            # (no env.step/reset). Guarded on _weights_finite (already checked in the branch above) and a
            # non-empty buffer. NEVER selects on value/reward.
            if select_on_npg and npg_state["buf"] is not None and _weights_finite(agent) \
                    and len(npg_state["buf"]) > 0:
                roll = sum(npg_state["buf"]) / len(npg_state["buf"])
                if roll > npg_state["best"]:
                    npg_state["best"] = roll
                    agent.save(npg_state["best_dir"])
                    print(f"BEST_CKPT[{getattr(cfg, 'runname', '?')}] upd={counter['i']} "
                          f"n_passed_gates_roll={roll:.4f}")
        return out

    agent.step = step_with_periodic_save

    # PASSIVE env.step HARVEST (Stage-1 ckpt-select): wrap env.step to read stats_raw['n_passed_gates']
    # into the recent-window deque -- installed right after agent.step per the recipe, OFF-path untouched.
    # The runner calls self.env.step each rollout tick; env IS self.env, so this instance-attribute shadow
    # intercepts it (same mechanism as the agent.step wrap). PRECHECK must confirm the buffer POPULATES
    # (the harvest is seen by the real runner loop).
    if select_on_npg and npg_state["buf"] is not None:
        _orig_env_step = env.step

        def _env_step_harvest(*a, **k):
            out = _orig_env_step(*a, **k)
            try:
                info = out[3] if isinstance(out, (tuple, list)) and len(out) > 3 else None
                _harvest_npg(info, npg_state["buf"])
            except Exception:                           # pragma: no cover - never fail a training step
                pass
            return out

        env.step = _env_step_harvest

    # OBS-V2 WARM-PAD: when +env.ego_obs_v2 is armed and +init_from is a 21-dim champion actor, rewrite
    # the checkpoint into a temp dir with 2 ZERO first-layer input columns appended (the appended next-gate
    # coarse dims contribute nothing at step 0) and REPOINT cfg.init_from at it so the load below matches
    # the 23-dim v2 net. No-op (byte-identical, no I/O) when the knob is off / no init_from / dims match.
    maybe_pad_ckpt_for_obs_v2(cfg, env)
    # WARM-START: load actor+critic from cfg.init_from into the freshly-built agent (weights-only;
    # optimizer + rollout buffer stay fresh). UNSET => no-op, BYTE-IDENTICAL fresh init. The ego env has
    # no look-at gain-warmup, so the inc8 _force_lookat_warmup_off inside maybe_warmstart is a safe
    # no-op. Called AFTER the agent + env are built, BEFORE the first training step (inc8 contract).
    warmstart_from = maybe_warmstart(agent, env, cfg)
    if warmstart_from is not None:
        print(f"[lifeline] CONTINUING ego training from warm-started weights: {warmstart_from}")
        # LOGSTD RESET after the load (+warmstart_reset_logstd): transfer the MEANS, reset the
        # ground-down exploitation schedule to an intermediate level so the new stage keeps escape
        # velocity. Applied here so it fires regardless of the synced inc8_warmstart.py version (the
        # newer maybe_warmstart also applies it -> an idempotent double-set of the same raw value).
        if bool(getattr(cfg, "warmstart_reset_logstd", False)):
            target = float(getattr(cfg, "warmstart_reset_logstd_std", 0.18))
            reset_actor_logstd(agent, target_std=target)

    # CRITIC-ONLY WARMUP (+critic_warmup_updates=N): zero every actor-named gradient for the first N PPO
    # updates so ONLY the critic adapts to the new stage's return scale before the policy moves (the
    # audit-A1 boundary-divergence fix). Unset/0 => byte-identical. Applies to fresh starts too (a
    # random critic's advantages are equally garbage). counter["i"] is the same 0-based PPO-update index.
    #
    # CRITIC-ONLY GRAD CLIP (Stage-1 vtrackAr5, ++critic_grad_clip=C, unset/<=0 -> no-op): at this SAME
    # agent.optim.step monkeypatch, BEFORE the wrapped step, clip_grad_norm over ONLY the critic-named
    # params. The 8-gate critic_grad spikes to ~1519 (m8)/918 (m8b) (~7x the healthy 2-gate regime); a
    # 10-norm cut removes the spikes WITHOUT touching the actor (actor_grad_norm is a healthy ~3). Training
    # -only, zero deploy footprint. Wrap when EITHER warmup or clip is armed so the clip works even without
    # a critic_warmup (both unset -> the wrapper is not installed -> byte-identical).
    critic_warmup = int(getattr(cfg, "critic_warmup_updates", 0) or 0)
    critic_grad_clip = float(getattr(cfg, "critic_grad_clip", 0.0) or 0.0)
    if critic_warmup > 0 or critic_grad_clip > 0.0:
        _cw_state = {"released": False}
        _pre_cw_step = agent.optim.step

        def _critic_warmup_step(*a, **k):
            # (1) critic-only grad clip -- BEFORE the optim step, actor grads untouched.
            if critic_grad_clip > 0.0:
                crit = _select_critic_params(agent)
                if crit:
                    torch.nn.utils.clip_grad_norm_(crit, max_norm=critic_grad_clip)
            # (2) critic warmup -- zero actor grads for the first N updates.
            if critic_warmup > 0:
                if counter["i"] < critic_warmup:
                    for _n, _p in agent.agent.named_parameters():
                        if "actor" in _n and _p.grad is not None:
                            _p.grad.zero_()
                elif not _cw_state["released"]:
                    _cw_state["released"] = True
                    print(f"[critic-warmup] RELEASED at update {counter['i']}: actor learning resumes "
                          f"(critic adapted alone for the first {critic_warmup} updates).")
            return _pre_cw_step(*a, **k)

        agent.optim.step = _critic_warmup_step
        if critic_warmup > 0:
            print(f"[critic-warmup] ON: actor gradients ZEROED for the first {critic_warmup} PPO updates "
                  f"(critic-only adaptation to the new return scale).")
        if critic_grad_clip > 0.0:
            print(f"[critic-grad-clip] ON: critic-only grad-norm clip at {critic_grad_clip} "
                  f"(actor grads untouched).")

    try:
        result = _orig_run(self)
        # FINAL == the in-memory agent == the runner's checkpoints/ save; DET-eval it first.
        # ++eval_yaw_log=true (default OFF -> byte-identical): also emit the greppable YAW_EVAL[...] line
        # (signflips_per_s + roll_swing from peak_roll_deg) on THIS post-training eval, so a run's final +
        # promoted-best checkpoints report their yaw-oscillation without a separate rollout_only pass. Pure
        # read inside the SAME det-eval loop (no extra env steps / resets) -> training behaviour unchanged.
        _yl = bool(getattr(cfg, "eval_yaw_log", False))
        final_metrics = _run_det_eval(self, env, agent, cfg, yaw_log=_yl)   # faithful deterministic box-exit, live env
        # CHECKPOINT SELECTION (Stage-1): DET-eval best_npg/ and promote it over checkpoints/ if it >= final.
        if select_on_npg:
            _promote_best_on_npg(self, env, agent, cfg, npg_state, final_metrics, logger)
        return result
    except Exception:
        traceback.print_exc()
        try:
            if _weights_finite(agent):
                agent.save(os.path.join(logger.logdir, "emergency"))
                print("[lifeline] EMERGENCY checkpoint saved to %s"
                      % os.path.join(logger.logdir, "emergency"))
            else:
                print("[lifeline] weights non-finite -- emergency save skipped")
        except Exception:
            traceback.print_exc()
        raise


TrainRunner.run = _run_with_ego_lifelines

from diffaero.script.train import main    # noqa: E402  (must follow the registrations above)

if __name__ == "__main__":
    main()
