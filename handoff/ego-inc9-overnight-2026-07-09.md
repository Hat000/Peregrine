# Ego inc9 — overnight handoff (2026-07-09, RL commander)

Fengyou — a productive night with one hard blocker at the end (Adroit daemon down). Full detail is in
`docs/vq2-ego-oracle-dossier/` (committed); this is the fast morning briefing.

## Headline findings (all data-derived)

1. **The single-gate centring floor is PERCEPTION-NOISE-limited, not control-limited** (overturns the
   earlier read). The `ego_noise_scale` ablation: a **perfect estimator → 63% thread / 0.32 m** (det
   0.641) vs champion 20% / 0.88 m. ~0.28 m of measurement noise is amplified by the closed loop into
   ~0.5 m of crossing offset. Underneath perception sits a ~36% control clip-residual. **This validates
   your idea (a)** ("reduce the noise arriving to the policy").
2. **Speed lever REFUTED** (`rw_vmax` cap made centring worse — flattens the far-field gradient).
3. **`r_perc` (Swift/Geles perception reward) REJECTED** — a positive dense point-at-gate term is
   farmable (fly-away to keep the gate in view) → hard collapse. Does not transfer to our stack.
4. **Your idea (b) is literature-backed** — Song IROS N=1→N=2 gate obs cuts crashes 23%→2.5%; CPC/TOGT:
   the crossing point is defined by the gate *sequence*, so single-gate 90% may be the wrong target.
5. **LATENT BUG found + fixed:** the in-run anneal hooks (`env._egorw`/`env._estimator`) never reached
   the raw env through DiffAero's `RecordEpisodeStatistics` wrapper → **the cross-zero anneal was
   silently inert the whole 2026-07-08 campaign** (`vglpan` = fixed `cross_zero=4.0`, not the 4→0.75
   anneal). Fix = `_unwrap_env_with` drill (`4f91bd8`). Footgun banked as dossier L16 (verify every hook
   prints `ON`; bit-identical dose-response = tripwire). The perception ablation is unaffected (it set
   noise via env-construction config, not the hook).

## The blocker (needs you first thing)

- **Adroit serve-daemon is DOWN** (repeated `ConnectionResetError`). Restart needs a Duo push — I did not
  trigger it overnight.
- **`vcza2` (job 3298993) and `vnsb` (job 3298994) died early (~step 570)** within the same 5-min window
  the daemon died → almost certainly an **Adroit-side event** (node/login failure), not my fix (the two
  jobs exercise different hooks yet died together; the fix is py-compiled + guarded). **No usable result
  from these two.** They are: `vcza2` = the cross-zero anneal 4→0.75 *for real* (never-run centring
  lever); `vnsb` = the noise curriculum 0→1 (warm from vglp4, cross-zero fixed 4).

## Morning actions

1. **Re-establish the Adroit daemon** (Duo), then `squeue`/`sacct` + tail
   `/scratch/network/fl3689/peregrine_vq2_ego_{vcza2,vnsb}.out` to confirm the death cause. If Adroit-side
   → **relaunch both** (same sbatch lines below). If it turns out to be my fix, the crash trace will say so.
2. **Decide the fork** (the real question the night surfaced): centring is perception-bound, so —
   (1) **vision-commander relay** for a better estimator/detector [perception is the critical path], and/or
   (2) **multi-gate regime** (your idea b — the real VQ2, where N=2 is field-standard and the crossing
   point is defined), and/or (3) **noise-robustness** (the noise-curriculum result, once it runs clean).
   My recommendation: relaunch `vcza2`/`vnsb` to get those two data points, AND open the vision relay,
   since the estimator is the highest-leverage lever for centring.

## Relaunch commands (once daemon is up)

```
sbatch --export=ALL,SEED=0,RUNTAG=vcza2,STAGES=single_gate_varied_gvf_lpara_anneal,UPD_single_gate_varied_gvf_lpara_anneal=4000 /scratch/network/fl3689/peregrine_repo/rl/peregrine_vq2_ego.sbatch
sbatch --export=ALL,SEED=0,RUNTAG=vnsb,STAGES=single_gate_varied_gvf_lpara_nscale,UPD_single_gate_varied_gvf_lpara_nscale=2000 /scratch/network/fl3689/peregrine_repo/rl/peregrine_vq2_ego.sbatch
```
(Deploy is current — hash-verified `4f91bd8`. Confirm `[cross-zero-anneal] ON` / `[noise-scale-anneal] ON`
in each log after warm-start; if a dose-response pair returns bit-identical, the hook silently skipped.)

## State

- Dossier committed on `claude/vq2-ego-single-gate-rl-0b78fb` (docs 00–10 + this handoff).
- Code committed on `claude/optimistic-chaum-6c893b` (`5e47bfe` noise_scale · `7a26d4a` r_perc ·
  `983faf6` noise-curriculum · `4f91bd8` anneal-inert fix). **Nothing pushed.**
- MEMORY.md (home) updated + compacted under limit. **Repo `memory/MEMORY.md` mirror is stale (pre-ego)
  — needs reconciliation.**
