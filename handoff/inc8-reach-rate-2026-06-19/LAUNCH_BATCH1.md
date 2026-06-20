# inc8-reach-rate — Batch 1 launch runbook (exact commands)

Campaign: lift DETERMINISTIC reach/pass-rate of gate-4 (corrected binding number) from ~0.47 → ~0.85.
Levers this batch (single-var discipline): **g_pitch DOWN** (lever #2) + **gentler noise floor** (lever #1
refinement). Reward/speed weights HELD at rc1's working values. NO file edits — pure submit knobs (the
det-stab lever files are already synced+sha-verified on Adroit).

## Why these cells (from the det-stab seed0 bracket, all on /scratch)
| g_pitch | floor | det reach (best snap) | note |
|---|---|---|---|
| 1.0 | 0.03 | **0.467** (upd1700, in std=0.6 hold) | anneal→0.03 then KILLED it (upd4000 reach 0.044) |
| 1.5 | 0.03 | 0.129 | |
| 2.0 | 0.03 | 0.000 | |
| 3.0 | 0.03 | 0.000 | |
| **0.0** | — | **NEVER RUN** (det) | rc1-config + det-stab; the true reach floor, untested |
| **0.5** | — | **NEVER RUN** | |

Trend: det reach rises monotonically as g_pitch FALLS. gp0.0/gp0.5 are the untested high-reach cells.
The peak was reached at std=0.6 (the HOLD); the aggressive anneal→0.03 collapses it ⇒ gentler floor.

## STEP 0 — confirm daemon + cluster clean
```
python adroit.py x "echo OK; hostname; squeue -u fl3689 -o '%.12i %.26j %.8T %.7M %R'; checkquota | head -8"
```

## STEP 1 — submit Batch 1 (seed0, parallel)
```
python adroit.py x "S=/scratch/network/fl3689/peregrine_repo/rl/peregrine_inc8_detstab.sbatch; \
sbatch --export=ALL,GPITCH=0.0,NOISE_STD_HOLD=0.6,NOISE_STD_FLOOR=0.3,NOISE_HOLD_FRAC=0.5,SNAP_FROM=0.3,RUNTAG=rr_gp0p0_f03 \$S; \
sbatch --export=ALL,GPITCH=0.5,NOISE_STD_HOLD=0.6,NOISE_STD_FLOOR=0.3,NOISE_HOLD_FRAC=0.5,SNAP_FROM=0.3,RUNTAG=rr_gp0p5_f03 \$S; \
sbatch --export=ALL,GPITCH=1.0,NOISE_STD_HOLD=0.6,NOISE_STD_FLOOR=0.3,NOISE_HOLD_FRAC=0.5,SNAP_FROM=0.3,RUNTAG=rr_gp1p0_f03 \$S; \
sbatch --export=ALL,GPITCH=0.0,NOISE_STD_HOLD=0.6,NOISE_STD_FLOOR=0.6,NOISE_HOLD_FRAC=0.5,SNAP_FROM=0.3,RUNTAG=rr_gp0p0_flat \$S; \
echo ---QUEUE---; squeue -u fl3689 -o '%.12i %.26j %.8T %.7M %R'"
```
Cells: gp0.0/f0.3 · gp0.5/f0.3 · gp1.0/f0.3 (floor refinement vs existing gp1.0/f0.03=0.467) ·
gp0.0/flat0.6 (no-collapse ceiling + entropy→0; lets PPO self-determine the std collapse).

## STEP 2 — monitor (each run ~50 min on A100; precheck + stochastic flightcheck inside)
```
python adroit.py x "squeue -u fl3689 -o '%.12i %.26j %.8T %.7M %R'; for t in rr_gp0p0_f03 rr_gp0p5_f03 rr_gp1p0_f03 rr_gp0p0_flat; do echo \"=== \$t ===\"; O=/scratch/network/fl3689/peregrine_inc8_detstab_\$t.out; grep -E 'PRECHECK PASSED|PRECHECK FAILED|FLIGHTCHECK_VERDICT|DETSTAB_SEED0_RC|INC8_DETSTAB_DONE' \$O 2>/dev/null | tail -4; RD=/scratch/network/fl3689/diffaero/outputs/train/inc8_detstab_seed0_\$t; echo -n '  snapshots: '; ls -d \$RD/snapshots/upd* 2>/dev/null | wc -l; done; date"
```

## STEP 3 — deterministic selection per finished run (each select ~ up to 1 h, n_envs=256)
GPITCH **MUST match** the trained gain (else false NO-GO, like rc1-at-3). REACH_THRESH lowered to 0.10 so
STAGE-2 σ_p0 runs on the best even when reach is modest.
```
python adroit.py x "SEL=/scratch/network/fl3689/peregrine_repo/rl/inc8_select_ckpt.sbatch; B=/scratch/network/fl3689/diffaero/outputs/train; \
sbatch --export=ALL,RUNDIR=\$B/inc8_detstab_seed0_rr_gp0p0_f03,GPITCH=0.0,REACH_THRESH=0.10,TOPK=4,RUNTAG=sel_rr_gp0p0_f03 \$SEL; \
sbatch --export=ALL,RUNDIR=\$B/inc8_detstab_seed0_rr_gp0p5_f03,GPITCH=0.5,REACH_THRESH=0.10,TOPK=4,RUNTAG=sel_rr_gp0p5_f03 \$SEL; \
sbatch --export=ALL,RUNDIR=\$B/inc8_detstab_seed0_rr_gp1p0_f03,GPITCH=1.0,REACH_THRESH=0.10,TOPK=4,RUNTAG=sel_rr_gp1p0_f03 \$SEL; \
sbatch --export=ALL,RUNDIR=\$B/inc8_detstab_seed0_rr_gp0p0_flat,GPITCH=0.0,REACH_THRESH=0.10,TOPK=4,RUNTAG=sel_rr_gp0p0_flat \$SEL; \
echo ---QUEUE---; squeue -u fl3689 -o '%.12i %.26j %.8T %.7M %R'"
```

## STEP 4 — read selection results
```
python adroit.py x "for t in sel_rr_gp0p0_f03 sel_rr_gp0p5_f03 sel_rr_gp1p0_f03 sel_rr_gp0p0_flat; do echo \"========== \$t ==========\"; O=/scratch/network/fl3689/inc8_select_ckpt_\$t.out; echo '-- STAGE1 top-5 det reach --'; grep '^SELECT ' \$O 2>/dev/null | sort -k2 -gr | uniq | head -5; echo '-- STAGE2 full sigma_p0 --'; grep 'SIGMAP0_TORCH_SUMMARY' \$O 2>/dev/null | grep 'horizons' ; grep -oE 'reach_rate=[0-9.]+ success_rate=[0-9.]+ sigma_p0_lat=[0-9.nan]+ bias_lat=[+-0-9.nan]+ lat_p90=[0-9.nan]+ lat_p99=[0-9.nan]+ .*fix_rate=[0-9.]+ .*verdict=[A-Z-]+' \$O 2>/dev/null | head -8; done"
```

## DECISION
- Best det reach ≥ ~0.85 AND a snapshot with success ≥ ~0.80, p99 < 0.45 → Batch 2: that config + seeds 1,2
  (≥3-seed GO) and freeze. Pull the winning checkpoint via the artifact pipe (gitignored), NOT git.
- Reach lifts but plateaus < 0.85 → Batch 2 = lever #3 (reward/speed weight tuning) on the best g_pitch/floor,
  AND characterize the residual failure mode (which gate/segment/speed the misses occur at).
- Reach does not lift over the existing 0.467 → falsified; report the supported plateau + failure-mode
  analysis + recommended next lever.
