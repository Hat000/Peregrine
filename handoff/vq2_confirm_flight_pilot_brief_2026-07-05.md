# VQ2 confirm-flight pilot brief — loop-choke cut, 2026-07-05

Self-contained checklist for a Sonnet pilot agent. Validates commit `f1c5db9`
(`use_floor_height=False` + `vp_yaw_decimate=15` in the `vq2_case_c` deploy profile) against
the success gate: **`[loop-rate]` reliably >=25 Hz + operator eyes**.

Every command below was checked against the CURRENT code in this worktree
(`rl/fly_rl.py`, `scripts/vq2ctl.py`) at HEAD `f1c5db9`. `docs/vq2ctl.md` does NOT exist in
this worktree despite being referenced from `scripts/vq2ctl.py`'s own `guide` text and from
memory — treat `scripts/vq2ctl.py guide` (run it, or read the `GUIDE` string in the script) as
the canonical doc instead. If it reappears before you fly, it is not a contradiction, just a
worktree gap.

The GPU is SHARED with another track (recon-map reconstruction). **Do not fly if it's busy.**

---

## 0. Preflight gate: GPU ownership check (MANDATORY, abort if it fails)

Run:

```powershell
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

- **Abort (report, do not fly)** if `utilization.gpu` is pinned high (e.g. persistently >50%
  when nothing of ours is running) or `memory.used` is most of `memory.total`, with no
  known-ours process explaining it.
- Note: on this box `used_memory` for compute-apps frequently reports `[N/A]` (a Windows
  WDDM/driver permissions quirk, confirmed live 2026-07-05 — this is NOT a tool failure).
  Don't rely on the per-process memory column; use `utilization.gpu` + the aggregate
  `memory.used/memory.total` from the first query as the real signal, and use the process
  list only to spot an unexpected `python.exe` / training job by name.
- This is a **shared GPU with the recon-map track** — a busy GPU most likely means someone
  else's job is running. Abort and report; do not contend for it.

## 1. Reset to the WAITING ROOM — do NOT GO yet (do NOT use computer-use/screenshots)

**ORDERING IS LOAD-BEARING (learned the hard way, run 20260705_211007): the race must NOT be
GO'd until fly_rl is attached and passively waiting.** On that run the old brief used
`vq2ctl.py fly` (which GOes the race itself) BEFORE launching fly_rl — fly_rl late-joined
`to_go=-22.7s`, the drone sat idle through ~23s of race clock, and the flight was cut short
by the race timer (`STALLED` mid-pursuit). Attach FIRST, GO LAST.

```powershell
C:\Users\Shadow\Peregrine\.venv\Scripts\python.exe scripts\vq2ctl.py to-menu
C:\Users\Shadow\Peregrine\.venv\Scripts\python.exe scripts\vq2ctl.py waiting
C:\Users\Shadow\Peregrine\.venv\Scripts\python.exe scripts\vq2ctl.py status --probe
```

- Loop the three commands (max 3 passes) until `status --probe` shows the WAITING room with
  `started=False` — that is the ONE trustworthy fresh-race checkpoint (a post-race main menu
  keeps streaming the dead race's `started=True`; the waiting room resets it).
- Do NOT run `vq2ctl.py fly` in this protocol — `fly` GOes the race itself, which recreates
  the late-join failure above. The GO happens in step 4, AFTER fly_rl is attached.
- **No computer-use / no screenshots for flight ops** — `vq2ctl.py` drives the sim via
  verified-foreground key-sends + a passive MAVLink probe; that IS the flight tool.
- **Never probe while fly_rl is attached.** `vq2ctl.py probe`/`status --probe` bind UDP 14550
  and will split fly_rl's MAVLink stream once fly_rl has connected. Probing is allowed ONLY
  here in step 1, before fly_rl launches. From step 2 on, sim nudges are pure key-send
  commands only (`focus`, `go`, `restart`, `to-menu` — these do not bind the socket).

## 2. Launch fly_rl for the confirm flight

Exact invocation (all flags verified against `build_parser()` in `rl/fly_rl.py`, current HEAD):

```powershell
C:\Users\Shadow\Peregrine\.venv\Scripts\python.exe rl\fly_rl.py `
  --gate-seeker `
  --deploy-profile vq2_case_c `
  --seeker-detector yolo `
  --seeker-weights models\vq2_darkred_negv1_2026-07-02_fp16_384x640.engine `
  --ignore-collisions `
  --max-seconds 120 `
  --label loopchoke_confirm 2>&1 | Tee-Object -FilePath data\runs\loopchoke_confirm_console.log
```

Flag-by-flag, verified against the CURRENT argparse (`rl/fly_rl.py` `build_parser()`):

- `--gate-seeker` (`action="store_true"`, default off): flies the transparent gate-seeker on
  the case-C stack instead of the RL policy. Required for this confirm flight.
- `--deploy-profile vq2_case_c` (default IS `vq2_case_c` already, but pass it explicitly for
  self-documentation). This profile is exactly what commit `f1c5db9` touched
  (`use_floor_height=False`, `vp_yaw_decimate=15` — see `src/racer/deploy_profile.py`).
- `--seeker-detector yolo` (default IS `yolo` already; pass explicitly). The `yolo` path
  REQUIRES `--seeker-weights` to look like real detector weights or `fly_rl` exits loud at
  startup (`_validate_seeker_detector`) — do not omit `--seeker-weights`.
- `--seeker-weights models\vq2_darkred_negv1_2026-07-02_fp16_384x640.engine` — the canonical
  TRT engine (per `handoff/gen9_commander_briefing_2026-07-04.md`: "flight default", parity-
  exact with the `.pt`, ~2x faster, 82% good-fix). Confirmed present on disk at
  `models/vq2_darkred_negv1_2026-07-02_fp16_384x640.engine`. **Requires the `.venv` python**
  (`C:\Users\Shadow\Peregrine\.venv\Scripts\python.exe` — it has `tensorrt`; the separate
  `vq2yolo-venv` does NOT). Never fly the raw `.pt` for this — `.engine` is the flight path.
- `--async-detect` is NOT passed: default is `"auto"`, which resolves to `ON` for
  `vq2_case_c` (`DeployProfile.async_detect=True` for this profile, confirmed in
  `src/racer/deploy_profile.py`). This is exactly the mechanism the loop-choke cut depends on
  (decoupled detect worker + `vp_yaw_decimate` cutting the remaining hot-loop CV cost) — leave
  it on `auto` rather than forcing, so a profile change can't silently diverge from what's
  flown.
- `--ignore-collisions` (`action="store_true"`, default off): OBSERVE mode — do not abort on a
  hard collision, keep flying + recording. Appropriate for a tuning/confirm flight (see full
  trajectory even if it clips something), NOT for a real submission run.
- `--max-seconds 120` — matches the default already (`default=120.0`), pass explicitly.
- `--label loopchoke_confirm` — cosmetic; makes `data/runs/<stamp>_loopchoke_confirm_f1`
  easy to find. Any label works; `session` dir naming is `session_stamp()_{label}_f{flight}`
  (`rl/fly_rl.py` `main()`).
- `--flights` is NOT passed: default 1 (`"number of back-to-back attempts... DEFAULT 1 = the
  submission shape"`). One flight is right for this confirm.
- Everything else (arm attempts, spin-abort, odo-stale, etc.) is left at its documented
  default — no reason to deviate for a confirm flight.

**Do NOT invent flags.** Every flag above exists verbatim in `build_parser()` today; if you
add anything else, check it against `rl/fly_rl.py --help` first (system python is fine for
`--help` since it doesn't import torch/tensorrt at parse time... actually building the parser
does not touch torch, but ACTUALLY RUNNING the flight does — use the `.venv` python for the
real invocation regardless).

## 3. Capture stdout to a file (belt-and-braces with `perf_summary.json`)

The command in step 2 already pipes through `Tee-Object` to
`data\runs\loopchoke_confirm_console.log` — this is the PowerShell 5.1-safe form. Notes:

- PowerShell 5.1: do **not** redirect a native exe's stderr with `2>&1` inside a pipeline in
  ways that wrap it as a `NativeCommandError` (it can flip `$?` to `false` even on a clean
  exit) — the `2>&1 | Tee-Object` form above is fine because it's merging streams for display,
  not treating a non-zero-looking wrapped error as fatal; just don't add
  `-ErrorAction Stop`/`try/catch` around the whole pipeline expecting clean exception
  semantics from it.
- This is **belt-and-braces**, not a replacement for `perf_summary.json` (this dive's task 1
  addition): if the console log is somehow lost (truncated scrollback, dropped SSH/RDP
  session), `data/runs/<stamp>_loopchoke_confirm_f1/perf_summary.json` is the durable copy of
  the `[loop-rate]` / `[vision-timing]` / `[async-detect]` / `[seeker-diag]` numbers. The
  whole exit epilogue (`perf_summary.json` AND `nav_estimate.jsonl`) runs in a `finally`
  around the tick loop, so it is written on EVERY exit — clean `TIMEOUT`/`FINISHED`/`CRASH`,
  an uncaught mid-tick exception, or a Ctrl-C (unit-tested against the real loop in
  `tests/test_perf_summary_json.py`). On the exception/Ctrl-C paths `perf_summary.json`'s
  `final_state` reads `EXCEPTION`/`INTERRUPTED` (the same vocabulary `fly_rl`'s own summary
  uses), and the recorder's background-thread files (`mavlink.tlog`, `video.bin`,
  `commands.jsonl`) are closed by `main()`'s per-flight `finally` as before. Grab BOTH the
  console log and `perf_summary.json` after the flight; if the console log is intact you
  don't strictly need `perf_summary.json`, but check for it regardless as your first line of
  evidence per the belt-and-braces intent.

## 4. GO protocol (attach-first, GO-last)

- Launch fly_rl (step 2) against the `started=False` waiting room from step 1 and let it
  reach its passive wait: fly_rl's default posture is
  `">>> Waiting PASSIVELY for the race GO (no sim-control command will be sent...)"` — the
  submission-safe default (`args.dev_auto_reset` off; this brief does not pass it).
- Only when fly_rl is visibly in its passive wait, send the GO with pure key-sends (safe
  while attached — no socket bind):

```powershell
C:\Users\Shadow\Peregrine\.venv\Scripts\python.exe scripts\vq2ctl.py focus
C:\Users\Shadow\Peregrine\.venv\Scripts\python.exe scripts\vq2ctl.py go
```

  `focus` must report `foreground: OK` BEFORE `go` (a `go` without focus misses the sim —
  known gotcha: a NO_GO timeout after 180s means the Enter never landed, not a stack bug).
- Verify the GO took from fly_rl's OWN console (it recognizes the race start within a few
  seconds, `to_go` near 0 — NOT a late-join at -20s). If fly_rl hasn't recognized GO within
  ~15s, re-run `focus` + `go` (max 2 retries), never probe.
- Report **ready** (fly_rl attached, connected, armed-and-waiting-for-race-recognition or
  already flying) to the operator (Fengyou). Do not narrate the whole startup log — one line.
- The operator gives the actual go-ahead to let it fly and WATCHES LIVE.
- **After the flight**: ask the operator what THEY saw FIRST, before you dump telemetry or
  your own read of the numbers. If the operator says they watched live, SKIP rendering an mp4
  (`handoff/tools/render_cmd_indicator.py` or the stock renderer) — it's wasted GPU/CPU work
  duplicating what a human already watched happen in real time.

## 5. Success gate for THIS flight

Check, in order:

1. **`[loop-rate]` line** in console output (and mirrored in `perf_summary.json`'s
   `achieved_hz` / `loop_over_budget_pct`): `achieved_hz >= 25.0` "reliably" — i.e. don't
   accept a flight that limped to exactly 25.0 with `CHOKED` printed; the line itself prints
   `OK`/`CHOKED` (`rate_ok = achieved_hz >= 0.9 * args.rate and over_pct < 5.0` against
   `--rate` default 30, so the code's own bar is ~27 Hz + <5% over-budget ticks — tighter than
   the 25 Hz gate this brief states; report BOTH numbers, they should usually agree).
2. **`perf_summary.json` present** at `data/runs/<stamp>_loopchoke_confirm_f1/perf_summary.json`
   with non-empty `achieved_hz`/`vision_step_ms`/`async_detect` keys — confirms task 1's
   write path actually fired on a real flight (not just the offline unit test).
3. **No new garbage-lock/teleport regressions** — cross-check against the known failure
   signatures from prior runs (A33/A34 spec docs in `handoff/`): a track_range/offset_z_world
   that FREEZES for seconds then JUMPS tens of metres, or a `pass_wire`/`seeker_regime` that
   never fires despite a clean gate lock. `nav_estimate.jsonl` (written on every exit path,
   same `finally` as `perf_summary.json`) has the per-tick fields to check this
   (`track_range_m`, `offset_z_world`, `pass_wire`, `seeker_regime`).
4. Confirm the profile actually in effect matches commit `f1c5db9`: the `[gate-seeker]`
   startup print states `use_floor_height`/decimate values only indirectly — instead check
   `meta.json`'s `checkpoint`/`label` fields are sane and, if in doubt, `git show
   f1c5db9:src/racer/deploy_profile.py` to diff against what's checked out.
5. Operator eyes: did the drone visibly hold a steady ~30 Hz-smooth trajectory (no
   stepped/jerky ZOH artifacts) through at least gate 0? This is the qualitative half of the
   gate and only the operator can call it.

## 6. If the GPU gate fails, or fly_rl exits with `NO_GO`/`ARM_REFUSED`/`NO_MAP`

Report to the operator immediately; do not retry blindly. `NO_MAP`/`ARM_REFUSED` are printed
loudly with a reason (`rl/fly_rl.py` `_fly_gate_seeker`/`fly_once`) — quote the reason
verbatim rather than re-summarizing it.
