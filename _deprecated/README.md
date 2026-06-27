# `_deprecated/` — stashed dead weight (2026-06-27)

Files here are **not** in any active import / test / sbatch / deploy path. They were verified
orphan (or tied to retired increments) during the dormancy-exit cleanup and moved here to declutter
the live tree while preserving forensic/reference value. **Nothing in the codebase imports from this
directory.** Everything also remains in git history. To revive one, `git mv` it back.

## Forensic / analysis scripts (standalone — run manually, nothing imports them)
- **`tilt_segment_analysis.py`** (was `rl/`) — tilt/acceleration analysis on recorded flights.
  References retired `stage1_inc5/inc6` checkpoints; pass `--checkpoint` explicitly if re-run.
- **`replay_obs.py`** (was `rl/`) — replays recorded telemetry and reconstructs the obs training saw.
  Forensic only; takes an explicit `--checkpoint` arg (default path no longer resolves from here).
- **`tb_parse.py`** (was `rl/`) — generic TensorBoard scalar dumper. Superseded for inc8 monitoring by
  `rl/inc8_tb_trace.py` (which stays live).
- **`fit_vertical.py`** (was `scripts/`) — one-off vertical-dynamics fit. Superseded by `twin_tune.py`
  (coordinate-descent gain tuning, all axes). Cited only as provenance in `scripts/twin_hover.py`.

## Retired-increment SLURM launchers (inc5/inc6 are dead increments; inc7=baseline, inc8=current)
- **`peregrine_racing_inc5.sbatch`**, **`peregrine_racing_inc6.sbatch`** (were `rl/`) — superseded by the
  parameterized `peregrine_racing_s13/s14.sbatch` (inc7) and the `peregrine_inc8_*.sbatch` family.

> Kept live (do NOT move): `peregrine_train.py` (smoke sbatch consumer), `spike_vertical_slice.py`
> (imported by tests), `rl/checkpoints/stage1_inc{1,3,4,5,6}_*` (loaded by fly_rl/offline_rollout/tests —
> small, 784 KB), `cluster/yolo_train.sbatch` (shipped by `push_dir.py`), `peregrine_racing_inc7.sbatch`.
