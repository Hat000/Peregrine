# MANIFEST — shadowpc-postfix-dataset-2026-06-12

**Purpose:** Post-fix flight data for laptop dynamics refit session.
**Source:** `handoff/shadowpc-live-confirm-2026-06-12/WRITEUP.md` (8 indexed flights).
**Checkpoint:** `rl/checkpoints/stage1_inc6_actor.pth`
**Zip:** `debug_obs_8runs.zip` (0.36 MB) — `debug_obs.jsonl` + `meta.json` per run.
**Columns NOT dropped:** full debug_obs retained (size well under 80 MB threshold).
**Excluded per-run:** `mavlink.tlog`, `video.bin`, `video_index.jsonl` (not needed for dynamics refit).

---

## Run table

| # | Run dir | Start mode | Outcome | Zero-contact? |
|---|---------|-----------|---------|---------------|
| 1 | `20260612_183920_rl_inc6_frameaudit_std_f1` | Standing | CRASH @ gate 3 (3 gates passed) | Yes |
| 2 | `20260612_184029_rl_inc6_frameaudit_std_f2` | Standing | CRASH @ t=0 — spawn artefact (0 ticks) | N/A |
| 3 | `20260612_184209_rl_inc6_frameaudit_brg_f1` | Bridge | **FINISHED** 16.24 s (5 gates) | **Yes** |
| 4 | `20260612_184246_rl_inc6_frameaudit_brg_f2` | Bridge | **FINISHED** 17.74 s (5 gates) | **Yes** |
| 5 | `20260612_184406_rl_inc6_frameaudit_std_ext_f1` | Standing | CRASH @ gate 3 (3 gates passed) | Yes |
| 6 | `20260612_184433_rl_inc6_frameaudit_std_ext_f2` | Standing | CRASH @ t=0 — spawn artefact (0 ticks) | N/A |
| 7 | `20260612_184443_rl_inc6_frameaudit_std_ext_f3` | Standing | CRASH @ gate 3 (3 gates passed) | Yes |
| 8 | `20260612_184501_rl_inc6_frameaudit_std_ext_f4` | Standing | CRASH @ gate 3 (3 gates passed) | Yes |

**Zero-contact definition:** no collision event before the terminal gate-3 crash (runs 2, 6 are
spawn artefacts — collision on tick 0, not policy-driven, marked N/A).

---

## Notes for refit

- Runs 2 & 6 have `debug_obs.jsonl` with header only (470 B each); exclude from dynamics fit.
- All 6 valid runs (1, 3, 4, 5, 7, 8) have 208–265 usable ticks at 30 Hz.
- `debug_obs` fields: `pos_ned`, `vel_ned`, `q_raw_wxyz`, `w_raw`, `act_rescaled` (thrust + rates),
  plus full actor internals (`actor_mean`, `tanh`, `act_rescaled`, `k`, `sim_time_ns`, gate telemetry).
- Frame convention: `virtual_flip=true`, `yaw_scale=1.0`. Mirror canary confirmed TRUE +0.97 on all sessions.
- Rate gains: standing-start ~0.94–1.00, bridge/high-speed ~0.85 (regime-dependent attenuation, not a convention bug).
