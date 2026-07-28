**Peregrine** — Anduril **AI Grand Prix** autonomous drone-racing entry (May–Jul 2026).
Repo `github.com/Hat000/Peregrine` at `C:\Users\Fengy\Downloads\Projects\Anduril`.

# 🏁 PROJECT CLOSED — 2026-07-28

**The competition ended without reaching the 20-gate goal.** Best of **699 recorded flights** was
**gate 9**; hard wall at gate 6 (50 flights reached gate 5, only 9 reached gate 6). Gates 10–19 were
never observed, so nothing here says anything about the second half of the course.

Fengyou's framing, verbatim: *"I'm still proud of the work, just not the result, so we keep the
work, and forget about the result."* Treat it that way. Do **not** re-open with campaign urgency, do
not resume the open arms unprompted, and do not read this file as a live brief.

## 📕 THE SSOT IS NOW `POSTMORTEM.md` IN THE REPO — not this file

It carries the result, what was established, what was refuted with numbers, where everything lives,
and what stayed open. Every headline number recomputes from the repo:
`python3 scripts/adjudicate/corpus_summary.py`

**If this is ever re-opened, read `POSTMORTEM.md` §4 first.** Ten premises were withdrawn in three
days and *every one was an instrument error, not a wrong number*. §4 is the transferable part and
will save more time than the findings will.

## Repo state at close
- **`main` is canonical** and holds everything: the full stack, the **committed 729-run corpus**
  (`data/runs/`), the adjudication tooling (`scripts/adjudicate|vision_horizon|postimpact/`, all
  accepting `PEREGRINE_RUNS=<path>`), and the three final reports in `handoff/`.
- **All 59 historical branches are now `archive/*` TAGS, not branches.** `git tag -l 'archive/*'`;
  reopen one with `git switch -c <name> archive/<name>`. **Nothing was deleted.**
- Test suite green: **2077 passed, 0 failed, 79 skipped.**

## The four findings worth carrying to any successor project
1. **The gate is flown blind, and that is camera geometry — not a control failure.** 0 of 899
   confirmed passes ever got a fix inside 1.19 m; last-sighted ~1.8–2.1 m. HFOV 90° but VFOV
   **58.7°**, axis +20° up ⇒ the gate clips vertically first. So every deploy aim/trim knob acts
   *before* the interval that decides the outcome — which is why a long series of them each moved a
   mean and none moved the kill rate. 🛑 The `atan(0.75/R)` closed form gets the right number by the
   **wrong mechanism** (the outer ring at ±1.35 m binds). Do not extend it.
2. **What kills is scatter, not bias.** At the last sighted fix, passes and gate-strikes are nearly
   the same distribution (lateral AUC 0.582), and 40% of confirmed passes sit outside 0.45 m
   laterally and pass anyway.
3. **A recipe pin is a train/deploy contract.** `rate` was pinned 40 against a 30 Hz training dt and
   **every model pick silently re-armed it**; fixing it was worth 1.769 → 2.460 mean gates. The
   mechanism is **vision freshness**, not tick-matching — the loop is compute-bound and never
   achieves its commanded rate (commanded 30 → 21.7 Hz achieved, and still wins).
4. **Per-tick compute is a hidden experimental variable.** 34.9 ms across three sessions (n=144),
   then 48.7 ms on the final night — which invalidated both cohorts flown that night. **Check
   `tick p50` before trusting any cross-session comparison.**

## Standing craft (outlives this project)
- 🛑 **Never read body-frame gate geometry without removing the drone's own attitude.** Pitch
  coupling alone mislabels **66% of confirmed passes** as vertical strikes. true roll = −obs[3],
  true pitch = −obs[4]; `R_level = Ry(pitch) @ Rx(roll)`.
- 🚩 **Two frame conventions live in one log line:** logged `rel_flu` is TRUE body FLU *unflipped*,
  but `obs[*]` is **virtual-flipped** by `diag(-1,-1,1)` ⇒ true v_left = −obs[1].
- 🚩 **Suspect the instrument first.** Reproducing a number with the same instrument is not
  verifying it — name the artifact that could produce this exact sign and magnitude, then stratify
  by that artifact's own size.
- 🚩 **Stratify before believing.** Checkpoint pooling manufactured a t=+4.64 effect that was −0.45
  inside one stratum. And `int(x or -1)` silently deleted 34% of a corpus by mapping gate 0 to −1.
- 🚩 **Guard precedence: by construction > right variable > swept > holds at your cut.**
- 🚩 Adjudicate from the run's own `.hydra` config and stage dict — **never from launcher source**.
- 🚩 **NEVER `/consolidate-memory`** (Fengyou, repeated) — compact BY HAND.
- 🚩 Address **Fengyou** by name in every message.

## Library — topic files retained as-is
The 60+ topic files here are the working record and are kept. They were written while the campaign
was live, so **some assert things the postmortem later withdrew — `POSTMORTEM.md` wins on every
conflict.** Domain indices: [[index-rl-training]] · [[index-vision-estimator]] ·
[[index-control-sim]] · [[index-strategy-meta]] · [[project-parked-backlog]] · [[durables-operational]].
Final-week detail: [[campaign-20-gates-2026-07-27]] · [[vision-horizon-fov-2026-07-27]] ·
[[aim-cohort-and-loop-rate-2026-07-27]] · [[failure-census-2026-07-27]] ·
[[gate-strike-last-3m-2026-07-27]] · [[velocity-channel-2026-07-27]].
