# P1-CALIB-APPLY — APPLY INFRA REPORT

**Session:** P1-CALIB-APPLY (reports to P1 VISION-ACCURACY commander via Fengyou)
**Model:** claude-opus-4.8 · **Effort:** HIGH · **Date:** 2026-06-14
**Branch:** `calib-v2-apply` (local, created at `origin/p1-calib-v2` tip `a980823`)

---

## VERDICT: 🛑 STOP — patch NOT applied; clean unified-diff re-issue required.

**Decision (Fengyou / P1 commander, this session): strict escape-hatch — "Strict STOP, re-issue as diff."**
No hand-application was performed (neither `Edit` nor hand-authored diff). `src/` is **pristine**.

---

## 1. Baseline (recorded BEFORE any change) — GREEN ✅
```
703 passed, 35 skipped in 290.09s   (venv: .venv/Scripts/python.exe, pytest 9.0.3, from repo ROOT)
```
Matches the expected ~703 / 35. Tree is healthy. `git diff -- src/` is **empty** (no edits made).

## 2. Why the patch could not be applied — FORMAT, not drift
`handoff/p1-vision-accuracy-2026-06-14/scratch-calib-v2/proposed_calib_dualform.patch` is a
**human-readable DESIGN ARTIFACT** (its own line 1: *"PROPOSED, NOT APPLIED"*), authored that way
because the calib-v2 authoring session was forbidden from touching `src/`. It uses
`# --- insertion point …` comment markers and has **no** unified-diff structure
(no `--- a/`, no `+++ b/`, no `@@` hunk headers, no `***` markers).

Evidence (exit 128 both):
```
git apply --check --whitespace=nowarn  …/proposed_calib_dualform.patch
  → error: No valid patches in input (allow with "--allow-empty")
git apply --check -3                    …/proposed_calib_dualform.patch
  → error: No valid patches in input (allow with "--allow-empty")
```
`"No valid patches in input"` = git found **zero** parseable hunks (a FORMAT rejection). A *drift*
rejection would instead read `"patch does not apply"` / `"while searching for…"`. `patch -p1` would
fail identically — there are no markers for any parser to anchor on.

## 3. ZERO src drift — current `src/` matches the artifact's context BYTE-FOR-BYTE
Verified by direct read. Every insertion-point context line in the artifact matches the live source:

| Artifact insertion point | File | Current src lines | Match |
|---|---|---|---|
| #1a imports (`from __future__`…`from scipy…`) | `src/racer/frames.py` | 13–16 | ✅ exact |
| #1b `CAMERA_PITCH_RAD = np.deg2rad(20.0)` | `src/racer/frames.py` | 18 | ✅ exact |
| #1c `def R_camera_from_body()` (3 body lines) | `src/racer/frames.py` | 168–171 | ✅ exact |
| #2a `from racer.frames import ATTITUDE_NOISE_STD_RAD, R_camera_from_body` | `src/racer/localization.py` | 29 | ✅ exact |
| #2c `gate_pose_to_world_position` (`position_ned = gate.position_ned - lever`) | `src/racer/localization.py` | 85–87 | ✅ exact |
| #2d `gate_relative_inplane_fix` (`z_ned = gate.position_ned - lever`) | `src/racer/localization.py` | 161–163 | ✅ exact |

**Conclusion:** the edits are fully + unambiguously specified and there is no drift — only the FORMAT
of the artifact blocks a machine apply. The patch's own all-zero-default byte-identity claim is sound;
it simply cannot be `git apply`-ed in its current form.

## 4. What a clean re-issue must contain (so the next apply is a one-shot `git apply`)
Re-issue the SAME edits (verbatim from §2 of `CALIB_V2_DESIGN.md` + the artifact) re-expressed as a
proper **unified diff** against the current tree (the artifact is the source of truth — only its
FORMAT changes, content is unchanged). The cleanest mechanical route, to avoid any hand-typed diff:
apply the artifact's intent to a scratch checkout and `git diff > proposed_calib_dualform.diff`, or
have the authoring session re-export it as a real diff. Line numbers above are current and stable.

The re-issued diff must, at the all-zero default, remain byte-identical:
- `frames.BoresightCorrection(pitch_rad=0.0, roll_rad=0.0, vert_offset_m=0.0)` (frozen dataclass) +
  module-level `BORESIGHT = BoresightCorrection()`.
- `R_camera_from_body()` composes `BORESIGHT` such that all-zero ⇒ `R_roll == I` and the pitch arg
  `== CAMERA_PITCH_RAD` exactly ⇒ `np.array_equal` vs today's 20°-only mount.
- `localization._apply_camera_vert_offset(...)` guarded on `vert_offset_m != 0.0` ⇒ the metric line
  is **skipped** at default ⇒ both `+L` lever sites (`gate_pose_to_world_position`,
  `gate_relative_inplane_fix`) are `np.array_equal` to today; `+L` sign preserved.

**Land WITH the test** `handoff/…/scratch-calib-v2/proposed_test_calib_dualform.py`
→ `tests/test_calib_dualform.py` (imports are already standard absolute `racer.*`; no path fix needed).
It is **19 pass / 3 skip** standalone today; once `frames.BORESIGHT` exists the 3 skips activate, so the
suite should land at **726 passed / 35 skipped** (703 + 22 new + 1 net), zero regressions. The 3 activated
tests ARE the byte-identity pins (`np.array_equal` on mount; lever-shift `+L`-preserved).

## 5. State left behind
- `src/` **PRISTINE** — `git diff -- src/` empty. **No** hand-application (escape hatch honored).
- Branch `p1-calib-v2` content otherwise unchanged at `a980823`; this report is the only addition.
- `e` is still **unknown** — defaults stay all-zero; no nonzero correction was introduced anywhere.

---

## MEMORY-DELTA (≤10 lines)
- 🛑 P1-CALIB-APPLY **STOPPED at escape hatch** (Fengyou/commander: "strict STOP, re-issue as diff"). NO src change; `src/` pristine; nothing merged/pushed to src.
- ROOT CAUSE: `proposed_calib_dualform.patch` is a human-readable DESIGN ARTIFACT (marked "PROPOSED, NOT APPLIED"), NOT a unified diff → `git apply` fails on FORMAT ("No valid patches in input"), exit 128. Same for `-3` / `patch -p1`.
- ZERO src drift — verified context matches BYTE-FOR-BYTE: frames.py 13–16/18/168–171; localization.py 29/85–87/161–163.
- Baseline GREEN: **703 passed / 35 skipped** (venv pytest 9.0.3, from ROOT) — tree healthy, untouched.
- ACTION FOR COMMANDER: re-issue the SAME edits as a real unified diff (`--- a/ +++ b/ @@`; content unchanged, only format) → then a one-shot `git apply` lands default-zero infra. Land WITH the test → `tests/test_calib_dualform.py` (expect **726 passed / 35 skipped**, 0 regressions).
- `e` STILL unknown → defaults all-zero; `e` drops in later as a one-line `{form,value}` on `frames.BORESIGHT`.
