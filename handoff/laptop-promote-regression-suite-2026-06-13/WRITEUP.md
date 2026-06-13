# LAPTOP-PROMOTE-REGRESSION-SUITE — WRITEUP

**Date:** 2026-06-13. **For:** Fengyou. **Machine:** laptop, branch `main` (no concurrent
editing session on main — the autonomy-hardening work is on a separate worktree, so main was free).

## Mission
Promote the 8 external-invariant regression scripts produced by the substrate audit
(`handoff/ultracode-substrate-audit-2026-06-13/regression_suite/`) into the permanent `tests/`
suite as discovered pytest tests, locking the externally-cleared conventions in as regression
guards. Test-infrastructure only — **no non-test source was modified** (verified: `git diff --stat`
shows only `tests/` + this handoff).

## What was promoted
- **`tests/_audit_io.py`** — the shared loader + Test-A force model, promoted from the audit's
  `scratch/_audit_io.py`. **Only change vs the audit copy:** repo-root discovery is now
  depth-independent (`_find_root` walks up to the ancestor holding both `src/` and `handoff/`),
  so it no longer relies on `parents[3]`. Everything else (loader, `force_model`, `test_a_force_vs_fd`,
  `Rmats`, `tilt_deg`, candidate signs) is byte-for-byte the audit substrate. Not named `test_*`,
  so pytest does not collect it as a test. Added one helper: `dataset_present(name)` for the skip-guards.
- **8 test modules** (30 test functions total), each with its path-setup block rewritten to import
  `_audit_io` as a sibling in `tests/` and to derive `rl/`, `src/`, `scripts/` paths from `A.ROOT`.

| Promoted file | Tests | Invariant guarded | Negative control (what trips it) | Confirmed trips? |
|---|---:|---|---|---|
| `test_frame_force_vs_fd_mirror_canary.py` | 1 | R_y(π) ODOMETRY-quat conjugation: FD(pristine `vel_ned`) vs attitude-derived force, **East axis**, tilt>35° | AS-IS `[1,1,1,1]` quat must anti-correlate on East (`asis_e <= -0.40`) | ✅ asserted on real refit+postfix; **literal mutation:** AS-IS East corr **−0.838** vs TRUE **+0.989** → AS-IS fails the `>=0.90` positive gate |
| `test_train_deploy_obs_elementwise.py` | 5 | Train↔deploy 17-dim obs seam: gate-frame pos/vel (NED↔Z-up FLIP), next-gate lookup, `obs[12]` collective-memory; + full bit-exact seam on postfix | `test_negative_control_structural_axes`: AS-IS R_w2g makes `pos_g` diverge `>STRUCT_TOL*100`; AS-IS East `<=-0.40`; refit-supersession control (stale yaw must NOT match current build, but its roll must) | ✅ 3 dedicated negative-control assertions pass on real data |
| `test_twin_diffaero_extreme_parity.py` | 5 | DiffAero↔rl_plant frame/sign-alias at extreme states; external East invariant through the deployed plant + torch/numpy backend parity | AS-IS quat East corr `< EAST_CORR_MAX_BUG` **and** `< TRUE − 0.5` (refit+postfix); parity guards bite the `_QFLIP`/`_BODY_UP` sign flip | ✅ `test_external_invariant_negative_control_asis_breaks_east` passes |
| `test_mavlink_velocity_single_rotation.py` | 1 | Body-vs-world velocity-frame MIX (c3b5a8e): FD(`pos_ned`) == `vel_ned`, tilt>35° | Zero-rotation (raw body twist) **and** double-rotation candidates must de-correlate (`min(corr) < BUG_CORR_MAX`) | ✅ both frame-mix bugs asserted to fail the same gate |
| `test_confirmed_p4_c05.py` | 5 | `fly_rl.obs_from_zup`/`build_obs` hardcoded yaw=π gate frame (VQ2 hazard): deploy obs must equal yaw-aware train obs on any course | `test_p4_c05_lens2b_external_invariant_nonpi`: on a synthesized **non-π** course the buggy hardcoded-π seam diverges `>0.30 m` on consumed obs dims (fires at ~4.22 m); AS-IS attitude East `<0.0` | ✅ negative control fires at >0.30 m on the buggy tree (bug-confirming, per audit §2.1) |
| `test_confirmed_cr2_01.py` | 3 | Yaw-about-vertical sign in ODOMETRY rate read / FLU→FRD wire: L3 heading-rate + Q1 quat-FD body-z (Test-A is yaw-blind) | Flipped yaw sign: L3 `corr_flip < -0.45`; Q1 `resid_flip > 8× resid_true`; refit stale-yaw recompute `corr < -0.99` & `maxabs > 1.0` | ✅ 4 flip/stale-data negative-control assertions pass |
| `test_confirmed_cr4_01.py` | 5 | Yaw-channel R_y(π) conjugation + provenance that refit stale yaw is diagnostic-only | Pure-yaw-flip `[1,-1,1,1]` anchor must **collapse** on N&E (`corr < 0.60`, refit+postfix); reverting to bcc `[1,1,1,1]` fails East gate | ✅ yaw-flip collapse asserted on both datasets |
| `test_confirmed_cr4_03.py` | 5 | refit recorded wire validates OLD bcc93f9 map, not current `[+1,-1,+1]`; provenance assertion on recorded `rate_frd[2]` | LENS1 bcc map mismatch `>_MISMATCH_MIN` on postfix; LENS2 yaw-anchor corr **sign-flips** between datasets (postfix `>0`, refit `<0`) | ✅ counterfactual bcc flip breaks LENS1; sign-flip asserted |

**Total: 30 new test functions across 8 files + 1 shared util.**

## Negative-control philosophy (why these aren't trivial passes)
Each test does NOT merely assert the shipped convention holds — it **also** runs the *wrong*
convention (AS-IS quat, flipped yaw, zero/double rotation, bcc wire map, hardcoded-π seam on a
non-π course) through the **identical** gate and asserts that it **fails**. Because those
negative-control assertions pass on the real ShadowPC recordings, we have direct positive
evidence that the discriminator separates correct from incorrect on live data — the "guard the
guard" pattern. A literal mutation on the flagship force-anchor gate confirms teeth quantitatively:

```
TRUE  [1,-1,1,-1]            East corr = +0.989   PASSES gate (>=0.90)
AS-IS [1, 1,1, 1] (regress)  East corr = -0.838   FAILS gate  -> regression CAUGHT
```

This East-axis force-vs-FD discriminator is the shared positive control behind the canary,
twin-parity, p4_c05, cr4_01, and cr4_03 tests — the only lens that catches the R_y(π) bug class
that has bitten this project 4×.

## Slug-collision footgun — resolved
The audit (§2.2) warned that **inside its own scratch dir** `test_confirmed_cr4_03.py` had been
overwritten (old COLL_MAP → CR4-03 wire-map). In `tests/` there was **no** pre-existing
`test_confirmed_*` file (`git ls-files` + `ls` confirmed before writing), so all 8 promotions
clobbered nothing and use fresh, non-colliding slugs. **COLL_MAP is re-adjudicated not-a-bug**
(MEMORY READ-FIRST + audit P2-C04/CR5-01), so it was deliberately **not** re-homed — there is no
COLL_MAP test in this promotion.

## Portability guard (important for the commit)
- `postfix/extracted/*/debug_obs.jsonl` **is git-tracked** (8 runs).
- `refit/extracted/` **is gitignored** — only `debug_obs_17runs.zip` is tracked; the 17 extracted
  runs exist locally only because they were unzipped.

All 8 modules carry a module-level `pytestmark = pytest.mark.skipif(...)` requiring **both**
datasets' `extracted/` dirs. On this laptop both are present → **nothing skips, all 30 run and
pass** (the audited behavior). On a fresh clone that has not unzipped refit, the modules **skip
cleanly instead of ERRORing**. No test depends on a recording that is absent here, so the
escape-hatch (`tests/manual/`) was **not** needed.

## Suite result
- **Baseline (main, before):** `657 passed` (full run, 347 s).
- **After promotion:** **`687 passed`** (657 + 30; full run, 233 s). Zero skips on this machine, zero failures.

## Files added
```
tests/_audit_io.py                              (shared util; depth-independent ROOT)
tests/test_frame_force_vs_fd_mirror_canary.py   (guardian)
tests/test_train_deploy_obs_elementwise.py      (guardian)
tests/test_twin_diffaero_extreme_parity.py      (guardian)
tests/test_mavlink_velocity_single_rotation.py  (guardian)
tests/test_confirmed_p4_c05.py                  (finding: VQ2 yaw=π hardcode hazard)
tests/test_confirmed_cr2_01.py                  (finding: yaw-about-vertical sign)
tests/test_confirmed_cr4_01.py                  (finding: yaw-channel conjugation + provenance)
tests/test_confirmed_cr4_03.py                  (finding: refit validates OLD bcc93f9 wire map)
```

## Notes / residue
- The audit's other 3 scripts (`cr1_01`, `cr3_02`, `p3_c06`) were **out of this mission's scope**
  (the 8 = 4 guardians + 4 finding-specific). `cr1_01` is NEEDS-LIVE for the *absolute* yaw wire
  sign and `p3_c06` is cosmetic-class — both can be promoted later if wanted; they were not touched.
- `tests/_audit_io.py` does not import `rl/contact_true_eval.py` (out of scope, edited elsewhere) —
  verified, consistent with the audit's invariant.
