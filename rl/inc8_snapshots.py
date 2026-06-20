"""inc8 SNAPSHOT-RETENTION knob -- keep DENSE numbered checkpoints during training so a DETERMINISTIC
checkpoint-selection sweep (rl/inc8_select_ckpt.py) can pick the best-flying-on-the-mean snapshot AFTER
the run, instead of trusting the runner's STOCHASTIC high-water 'best' (which did NOT transfer to the
deterministic mean -- the rc1 failure, sigma_p0 job 3278151).

WHY (not the existing saves). diffaero's TrainRunner keeps only ``best/`` (overwritten on each new
STOCHASTIC success_rate high-water) and our launcher's rotating ``periodic/`` + ``periodic_prev/`` (last
two). None of those is selectable on a DETERMINISTIC criterion: ``best`` is the very stochastic-success
artifact we are trying to escape, and the last-two periodics are an arbitrary tail. With noise-annealing
on (rl/inc8_noise_anneal.py) the deterministic-flyable mean emerges somewhere in the anneal window, and
we cannot know a priori which update it is -- so we RETAIN a dense ladder of snapshots across the window
and let the post-hoc instrument (rl/inc8_sigmap0_torch_eval.py, deterministic test=True) score each and
pick the winner. Snapshots are cheap (an MLP actor+critic is ~0.3 MB; 25-50 of them << the run's other
artifacts) and the selection is a separate, re-runnable SLURM job, so the 8 h training run stays pristine
(instrument-before-subject: a selection bug costs a 30 min re-eval, not a re-train).

WHAT. ``cfg.snapshot_every=K`` writes ``<logdir>/snapshots/upd<NNNNN>/`` every K completed updates via
the SAME sidecar-wrapped ``agent.save`` the periodic save uses (-> actor.pth + critic.pth + actor.json
with obs_dim/inc8/r5_arm). ``cfg.snapshot_from_frac`` (default 0.0) skips the early stochastic era (e.g.
0.4 -> only retain the back 60% where the annealed mean is meaningful).

OFF == BYTE-IDENTICAL. With ``snapshot_every`` unset/<=0, ``resolve_snapshots`` returns None on a pure
getattr -- no save, no I/O, no state touched. Heavy work is trivial (os.path + agent.save) and stays on
the on-path. Imports clean on the dev laptop (no torch/diffaero).

OVERRIDE STRINGS (root-level, like n_updates/save_freq/seed; ``+`` because not in the pristine cfg):
  +snapshot_every=100  [+snapshot_from_frac=0.4]
"""
from __future__ import annotations

import os
from typing import Optional


def resolve_snapshots(cfg) -> Optional[dict]:
    """Parse the snapshot config from the ROOT cfg, or None when OFF (byte-identical default). Gated by
    ``cfg.snapshot_every`` (>0). PURE -- no torch -- so the OFF path is import-clean + side-effect-free."""
    every = int(getattr(cfg, "snapshot_every", 0) or 0)
    if every <= 0:
        return None
    return dict(
        every=every,
        from_frac=float(getattr(cfg, "snapshot_from_frac", 0.0) or 0.0),
        n_updates=int(getattr(cfg, "n_updates", 0) or 0),
    )


def should_snapshot(update_idx: int, snap: dict) -> bool:
    """PURE cadence predicate (laptop-testable): True iff update_idx>0, is a multiple of ``every``, and is
    at/after ``from_frac * n_updates`` (the from_frac gate is skipped when n_updates is unknown/<=0)."""
    if update_idx <= 0 or update_idx % snap["every"] != 0:
        return False
    if snap["n_updates"] > 0 and update_idx < snap["from_frac"] * snap["n_updates"]:
        return False
    return True


def maybe_write_snapshot(agent, logger, update_idx: int, snap: dict) -> Optional[str]:
    """Save a numbered checkpoint when ``update_idx`` is a cadence hit (past from_frac), via ``agent.save``
    (the sidecar-wrapped save -> actor.pth + critic.pth + actor.json). Returns the dir, or None when not a
    hit. Call AFTER the training step with the post-increment update counter (matches the periodic save)."""
    if not should_snapshot(update_idx, snap):
        return None
    snap_dir = os.path.join(logger.logdir, "snapshots", f"upd{update_idx:05d}")
    agent.save(snap_dir)
    print(f"[snapshot] saved checkpoint at update {update_idx} -> {snap_dir}")
    return snap_dir
