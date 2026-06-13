# LAPTOP-FLYRL-AUTONOMY-HARDENING — WRITEUP

**Date:** 2026-06-13 · **Branch/worktree:** `flyrl-autonomy-hardening` (`Anduril-wt-flyrl`) · **Model:** opus-4.8.
**Source audit:** `handoff/ultracode-autonomy-readiness-2026-06-13/REPORT.md`.
**Scope:** make the SUBMITTED RL entry (`rl/fly_rl.py` + launch path) submission-safe. **No retrain, no
policy/plant/reward change.** The 10 audit failure modes collapse onto 4 fixes (F-A..F-D) + one doc-only
label fix. Behavior-changing parts (F-C) were done VERIFY-FIRST.

> **DO NOT MERGE TO MAIN without Fengyou's go** — the launch path's arm / GO-accept / abort / disarm
> behavior changed and MUST be live-verified on ShadowPC first (checklist §6).

---

## 1. What each fix changed

### F-A — committed wrapper + safe defaults (closes R1 DQ, R2, inc4-trap, debug-I/O)
- **`rl/submit_rl.py` (NEW):** the committed, submission-safe entrypoint. Pins
  `--checkpoint <abs>/stage1_inc7_actor.pth --no-bridge --no-auto-reset --no-debug-obs --flights 1`.
  Safety pins are appended **last** so they are authoritative over any forwarded flag (argparse: last
  occurrence wins); refuses to launch if the inc7 `.pth` is missing. Non-safety flags (`--endpoint`,
  `--label`, …) forward verbatim.
- **Safe DEFAULTS flipped in `fly_rl.py`** so even a bare `python rl/fly_rl.py` is safe:
  - `--checkpoint` default `stage1_inc4` → **`stage1_inc7`** (the retired inc4 default shipped an
    aero-blind actor).
  - `--bridge` default `True` → **`False`** (standing start is the deployment target AND map-free; the
    bridge loaded a gitignored `data/runs` map → `FileNotFoundError` after arm).
  - `--debug-obs` default `True` → **`False`** (removes the only per-tick blocking file I/O in the loop).
  - **Auto-reset is now opt-in:** new `--dev-auto-reset` (default off); `auto_reset = dev_auto_reset and
    not no_auto_reset`. The submitted path therefore **never emits a sim-control command** (`MAV_CMD
    31000`) — it waits passively for the organizer's GO. `--no-auto-reset` is a hard override. This is the
    rank-1 §7 DQ fix (R1).

### F-B — finally-disarm + broadened handlers (closes the "dies ARMED" half of R2/R3/R4/n3)
- Factored the armed-flight body of `fly_once` into a sibling `_fly_armed()`; `fly_once` now calls it
  inside a **`try/finally` that force-disarms + `wait_armed(False)` on EVERY exit** (normal return,
  exception, or break). Mirrors `fly_vq1.py`'s finally-disarm. Removed the now-redundant inline
  bridge-fail disarm and the post-loop `[safety]` disarm (the finally owns it).
- `main()` per-flight handler: added `except Exception` → **backstop disarm + loud log + re-raise** (not
  silently swallowed). Outer handler: added `except Exception` → `traceback.print_exc()` + backstop
  disarm + **non-zero exit (`rc=1`)**; kept `except KeyboardInterrupt: pass` for clean Ctrl-C.

### F-C — per-field ODOMETRY freshness + finite gate (closes D1, D2, R4, R5) — VERIFY-FIRST
- **`DroneState.odo_recv_ns`** (new field, default 0), set **only** by the ODOMETRY handler in
  `mavlink_client.py`. The shared `recv_monotonic_ns` is also bumped by LPN/HIGHRES_IMU, so it cannot
  detect a selective ODOMETRY drop; `odo_recv_ns` can.
- **`telemetry_health(state, now_ns, stale_s)`** — pure, unit-testable helper returning
  `no_fix | stale | non_finite | ok` (None-check first; freshness via `odo_recv_ns`; finiteness over
  pos/vel/quat/rate + zero-norm-quat reject `q·q>1e-12`).
- **Loop gate placed ABOVE the spin guard and `build_obs`:** on `stale`/`non_finite` the loop sends a
  **SAFE HOVER** (zero body-rate + hover collective — never the stale RL command, never a NaN into
  `build_obs`) and holds for a bounded `--odo-recovery-s` window, then aborts cleanly with a new
  `ODO_STALE`/`NON_FINITE` state (the `fly_once` finally disarms). Because the gate sits above the spin
  guard, the spin guard only ever reads a **fresh** rate → the D2 false-`SPIN_ABORT`-on-frozen-`|w|` is
  closed structurally.
- Fixed the `--debug-obs` `odo_age_ms` field to use `odo_recv_ns` (it previously used the shared stamp →
  read ~0 ms even while the attitude was seconds stale — the exact blind spot D1 exploited).
- New args: `--odo-stale-s` (0.15), `--odo-recovery-s` (0.5).

### F-D — late-join GO gate + bounded arm-retry (closes N1, AR1; MUST ship with F-A)
- **`wait_fresh_go`** gained a passive **late-join** branch: when **not** `auto_reset` and the countdown
  elapsed > 2 s ago (`to_go ≤ −2000`), accept the first-seen **STARTED** race that is still at the start
  line (**gate 0, not finished, drone ≤ 5 m from origin**). This is exactly the config F-A's
  `--no-auto-reset` mandates; without it a late-joined race was a silent NO_GO. Added a NO_GO diagnostic
  log (`to_go`, `started`, `finished`, `gi`, `auto_reset`) at the deadline.
- **Arm** is now a bounded re-send loop (`--arm-attempts` 3, `--arm-backoff-s` 0.5, pumping during
  backoff); the final attempt escalates to `arm(force=True)` (the 21196 pre-arm bypass). Recovers a
  transient `TEMPORARILY_REJECTED` / slow `SAFETY_ARMED` bit instead of declaring `ARM_REFUSED` on the
  first try. Budget stays well inside the 8-min cap.

### Label fix (doc-only) + small refactor
- `obs[12]` was labeled "collective"/implied "[0,1]"; it actually holds the previous **rescaled
  `normed_thrust` in `[0, act_max]` g-units**. Renamed the `OBS_LABELS` entry `collective_prev` →
  `prev_normed_thrust` (only consumer is the line itself + historical `debug_obs.jsonl` data — no code
  matches it) and corrected the module / `obs_from_zup` / `policy_step` docstrings. No behavior change.
- Extracted `build_parser()` from `main()` so the autonomy-critical CLI defaults are unit-testable
  without opening a socket.

**Files touched (worktree):** `rl/fly_rl.py`, `rl/submit_rl.py` (new), `src/racer/contracts.py`,
`src/racer/mavlink_client.py`. **Out of scope, untouched:** P4-C05 gate-yaw hardcode, CR1-01 navigator
KF IMU-predict frame, all policy/plant/reward code.

---

## 2. F-C verify-first evidence

Both scripts exercise the **real shipped functions** offline (no socket).
`handoff/laptop-flyrl-autonomy-hardening-2026-06-13/verify/`.

### Does the failure propagate? (`fc_verify_first.py`, run against the UNEDITED code)
Yes — **all five propagate; the escape hatch does NOT apply** (`build_obs` does not sanitize):

| Case | Mechanism (observed) | Reaches wire? |
|------|----------------------|---------------|
| **D1** frozen ODOMETRY (real `_handle`: 1 ODOMETRY then 30 LPN) | shared `recv_monotonic_ns` advanced **1.79 ms** while quat+rate stayed frozen verbatim | **yes** — stale attitude → finite wire command, invisible to every guard |
| **R4a** zero-norm quat | `scipy from_quat` → `ValueError` inside `build_obs` | escapes loop → dies armed |
| **R4b** NaN-component quat | `ValueError: Found zero norm quaternions` | escapes loop → dies armed |
| **R5a** NaN position | NaN obs → `policy_step` | **yes** — `rate=[nan,nan,nan]`, `thrust=nan` |
| **R5b** inf-component quat | `from_quat` normalizes inf/inf→nan | **yes** — `[nan,nan,nan]`, `nan` |

`np.clip(nan, ±3.14) = nan` (clamp insufficient — the `isfinite` reject is load-bearing).

### Does the fix catch it, and stay silent on a healthy run? (`fc_verify_after.py`)
All checks PASS:
- **Parser fix isolates freshness:** during the ODOMETRY outage `odo_recv_ns` is FROZEN while the shared
  stamp advances; `telemetry_health` flags `stale` at a 200 ms age (was invisible before).
- **Gate logic:** healthy→`ok`; zero-norm/NaN quat, NaN/inf pos/vel/rate → `non_finite` (so `build_obs`
  is never reached — the R4 `ValueError` path is unreachable); all-None → `no_fix` (does NOT arm the
  recovery timer).
- **No false-trip:** benign 75 Hz jitter of 0–10 dropped frames (≤146.7 ms) stays `ok`; 11 dropped
  (160 ms) → `stale`. Boundary 149 ms `ok` / 151 ms `stale`.

### Thresholds and why they won't false-trip
- **`--odo-stale-s = 0.15`.** ODOMETRY nominal period = 1/75 Hz = **13.3 ms**, so 0.15 s = **~11 missed
  frames ≈ 10× the benign inter-arrival jitter**. At the 30 Hz control rate the freshest ODOMETRY is at
  most ~1 control tick (33 ms) + jitter old on a healthy run — far under 150 ms. 17+ live runs showed no
  malformed quat or attitude glitch. So the gate cannot trip a healthy flight but catches a genuine
  selective ODOMETRY stall. (Gated on organizer Q2 — per-msg-type reliability — for whether D1/D2 are
  live vs latent; the gate is cheap insurance either way.)
- **`--odo-recovery-s = 0.5`.** 0.5 s < the existing **1.5 s** sim-stall guard < the **2.0 s** spin
  window, so on a selective drop the odo gate acts FIRST while bounding open-loop/hover time. The
  recovery action is a neutral **SAFE HOVER** (zero rate + hover collective), never re-latching the
  stale RL command and never feeding `build_obs` a NaN/zero-norm quat. If the stream recovers in
  < 150 ms the gate never trips at all — a guard that aborts a valid run is worse than the risk it
  guards, and 150 ms gives a 10× margin before any hover is injected.
- **Spin-guard interplay (D2):** moving the freshness/finite gate above the spin guard means the spin
  guard only ever reads a fresh rate; a stale-window tick `continue`s before the spin guard, and on
  recovery the commanded-zero rate drops `|w|` below `spin_rate_abort`, so the spin timer resets. No
  false `SPIN_ABORT` from a frozen `|w|`, and a genuine post-recovery tumble still fires correctly.

---

## 3. Offline verification run (this worktree)

| Script | Result |
|--------|--------|
| `verify/fc_verify_first.py` | all 5 degraded-input failures propagate on unedited code (escape hatch N/A) |
| `verify/fc_verify_after.py` | parser-fix + gate-logic + no-false-trip — **ALL PASS** |
| `verify/fa_fd_verify.py` | safe defaults, `--dev-auto-reset`/`--no-auto-reset` override, wrapper pins authoritative, late-join accept/reject, fresh-GO no-regression, dev auto-reset still resets, arm-retry recovery — **ALL PASS** |
| full `pytest` suite | **657 passed in 276.99 s** (exit 0; see §5) |

---

## 4. LIVE-VERIFICATION CHECKLIST (ShadowPC — REQUIRED before submission)

The hardened launch path changes arm / GO-accept / abort / disarm behavior, so it **must** be confirmed
live before the submission is trusted. **Dev-rig note:** the ShadowPC batch tooling now needs
`--dev-auto-reset` to get the old auto-restart-between-flights behavior (default is OFF); multi-flight
`--flights N` also needs it.

1. **Clean standing-start run:** `python rl/submit_rl.py` (and separately bare `python rl/fly_rl.py`) →
   auto-arms → flies → **FINISHES clean** on inc7 standing start. Confirm sr, gate-3 clears, lap time in
   the ~11.45 s deployment band (fresh sim).
2. **No sim-control on the judged wire:** capture the MAVLink wire log for a submitted run and confirm
   **zero `MAV_CMD 31000`** (passive-GO posture). The startup banner should print the passive-wait line,
   not the `[DEV] auto-reset ON` line.
3. **Late-join GO (F-D/N1):** start the race BEFORE the process attaches (or attach into a running race
   at the start line) → confirm **`LATE-JOIN GO!`** fires and it arms+flies, instead of a silent NO_GO.
   Also confirm a race already past gate 0 / finished / off-origin is correctly NOT joined.
4. **Arm-retry (F-D/AR1):** if a transient arm rejection is reproducible, confirm recovery within 3
   attempts and that the force-arm last attempt works; otherwise just confirm the normal first-attempt
   arm path is unaffected.
5. **F-C gates stay SILENT on a healthy run:** confirm no `[odo-guard]` / `ODO_STALE` / `NON_FINITE` in a
   clean flight; with `--debug-obs` confirm `odo_age_ms` reads small (≪ 150 ms) throughout. (If a healthy
   run ever trips the gate, raise `--odo-stale-s` — but investigate the stall first.)
6. **F-B disarm-on-crash:** confirm the normal end-of-run disarm still happens; optionally force a crash
   (e.g. `--bridge --map <nonexistent>`), confirm the drone **disarms via the finally** rather than
   staying armed, and the process exits non-zero with a traceback.
7. **Post-session:** re-run `scripts/frame_residual_report.py` (standing conjugation-audit rule).

---

## 5. Test suite

```
PYTHONPATH=<worktree>/src .venv/Scripts/python.exe -m pytest -q
=> 657 passed in 276.99s (0:04:36)   # exit 0, full suite, worktree src confirmed
```
(The DroneState `odo_recv_ns` field addition and the `build_parser`/`_fly_armed`/`telemetry_health`
refactors are covered green; no test regressions.)

---

## 6. Residual / handoff notes
- D1/D2 (and N1) reachability hinge on organizer answers (Q1 launch ordering, Q2 per-msg-type telemetry
  reliability) — the fixes are cheap insurance regardless. The organizer email (NEXT-queue ⑨) should go
  out.
- P1 audit item **n3** (recorder pre-arm `OSError` on a non-writable CWD) is now caught by F-B's broadened
  outer handler (logged + disarmed + non-zero exit) but is **not** made non-fatal/CWD-independent here —
  flagged as a follow-up if the eval CWD writability is in doubt (audit §8 P1).
- P2 polish (NaN-safe `frames` guards, wire-level finite reject before `send_command`, confirmation-driven
  final disarm) not done — F-C's pre-`build_obs` gate already makes the R4/R5 paths unreachable on the RL
  loop; the P2 items are defense-in-depth for other call sites.
