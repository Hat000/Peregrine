# Cleanup Audit Appendix — COMMANDER.md (bucket: commander)

**Area state.** `COMMANDER.md` (the OVERALL COMMANDER boot prompt / self-improving craft doctrine, located in the sibling clone `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/COMMANDER.md`) is in good health: §3/§4 (growth loop + lineage) are intact and coherent, the lineage is gap-free Gen 1→5 with monotonic dates, and the prohibited stale "σ_p0 NO-GO / near-field-estimator pivot / RL-is-wrong-tool" OLD-view is ABSENT from this file. The only real defect is ONE point-in-time state fact (`:49`) that went stale after the deterministic-stability retrain landed; everything else is low-severity re-tense / readability / intentional-duplication notes. **No high-severity items. No deletions recommended — the prime directive (lose nothing) is fully honored; every action is re-tense, clarify, or no-op.**

Findings below are grouped by `issue_type`, then by severity (high first). Total findings after dedup: **7** (no cross-agent duplicates — all from the single `cmd:COMMANDER.md` agent; each is a distinct issue and is preserved individually).

---

## STALE (2)

### S-1 — Gen 5 prerequisite-lesson anecdote carries a now-stale point-in-time fact
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/COMMANDER.md:49`
- **Issue type:** stale
- **Severity:** med
- **Evidence:** `COMMANDER.md:49` (written 2026-06-19, before the det-stability retrain): *"I authorized a VQ2 closed-loop FLIGHT validation — detector → nav → flight — without checking that a flight controller for that loop exists; it doesn't yet — inc7 is case-A and brittle on self-loc, **inc8 isn't deterministically flyable**."* This is CONTRADICTED by the current corrected truth in `memory/MEMORY.md:8`: *"the deterministic-stability RETRAIN (the noise-anneal/std-cap lever, MERGED 42ddb0b, green_gate GREEN) gave the FIRST det-flyable 2-axis inc8 (gp1.0 det reach 0.467; rc1 0.000)."* The retrain (commit `42ddb0b`, parent of this clone's HEAD `c5d60a0`) landed AFTER COMMANDER.md was last edited (`ae4ccf5`/`4c25b55`).
- **Recommended action:** This is a CRAFT anecdote, and the LESSON ("verify a validation's prerequisites exist before authorizing it") is still valid and must stay. Only the supporting parenthetical fact has moved. Soften the dated fact to past tense so it stops reading as current state, e.g. change *"inc8 isn't deterministically flyable"* to *"inc8 wasn't yet deterministically flyable AT THAT TIME (it later got a det-flyable 2-axis result via the std-cap retrain — but live state belongs in `memory/`, see §3.4)."* **Do NOT delete the entry.**
- **Info-loss risk:** none — the incident/lesson is preserved; only the stale clause is re-tensed. The current det-flyable status is already captured in `memory/MEMORY.md:8`, so no information is lost by demoting it here.

### S-2 — Gate-4 σ_p0 doctrine references checked for the prohibited stale OLD-view — NONE found (clearing finding)
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/COMMANDER.md:47`, `:64`, `:65`, `:66`
- **Issue type:** stale
- **Severity:** low
- **Evidence:** Grepped COMMANDER.md for `0.08|no-go|near-field|pivot|fix-seat|wrong tool|fix-rate`. The only hits are `:51` (budget, unrelated) and `:62` ("inc8 architecture pivot" = the legitimate 2026-06-16 reward-architecture pivot, NOT the retracted RL-pivot). The σ_p0 references that DO exist are all CRAFT lessons, not verdicts: `:47` *"The gate-4 σ_p0 number was blocked FOUR times by harness gaps... and ZERO times by the physics"* — CONSISTENT WITH the correction (`memory/MEMORY.md:8` confirms gate-4 is a reach/pass-rate problem, not a physics wall); `:64`/`:65`/`:66` reference σ_p0 only as the subject of harness-debugging / git-safety / effort-tier anecdotes. NO statement asserts the 0.08 bar, a NO-GO verdict, a fix-seating wall, a near-field-estimator pivot, or "RL is wrong tool" as current truth.
- **Recommended action:** No action — recorded to document that the prohibited stale OLD-view is ABSENT from COMMANDER.md (the retracted-pivot risk lives only in `memory/` topic files, which are out of this file's scope). Confirms COMMANDER.md did not need the `511e85c` correction.
- **Info-loss risk:** none — informational / clearing finding.

---

## CONTRADICTION (1)

### C-1 — §3 rule 4 ("keep STATE out") vs Gen 4/Gen 5 lineage entries carrying point-in-time state
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/COMMANDER.md:49` and `:65` (against the rule at `:58`)
- **Issue type:** contradiction
- **Severity:** low
- **Evidence:** The rule, `COMMANDER.md:58`: *"Keep STATE OUT of this file — state lives in `memory/` and goes stale here. This file is durable craft only."* Violated by `COMMANDER.md:49` *"inc8 isn't deterministically flyable"* (a state claim that already went stale — see S-1) and `COMMANDER.md:65` (Gen 4): *"retired mid-burn with TWO live goals (laptop system-id + ShadowPC vision) and `main` local-only ahead of origin by 50"* (a point-in-time retirement-state snapshot inside the durable-craft file).
- **Recommended action:** Acknowledge that LINEAGE entries (§4) legitimately record what-happened-that-generation, so a date-stamped state mention is acceptable THERE as history. The live tension is only the §2 craft line (`:49`) leaking current-tense state — fix that per S-1. For `:65`, leave as-is (it is explicitly historical, date-stamped Gen 4); flag only so a future commander does NOT mistake the Gen-4 "`main` local-only ahead by 50" as a STANDING git directive (the standing rule is the de-numberized craft at `:45` rule 3, which is correct).
- **Info-loss risk:** none — recommending re-tense / clarify, not removal.

---

## DUPLICATION (1)

### D-1 — Operating-directive overlap between COMMANDER.md §2 and MEMORY.md "Operating directives" (intentional)
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/COMMANDER.md:25-51` vs `memory/MEMORY.md:33-37`
- **Issue type:** duplication
- **Severity:** low
- **Evidence:** Overlapping content pairs:
  - **canary** — `COMMANDER.md:21` *"Address **Fengyou by name in every message** (the canary; a missing name = context degradation → tell him to rotate the session)."* vs `MEMORY.md:34` *"the **OVERALL COMMANDER persistent session** addresses **Fengyou** by name in EVERY message (missing name = context degradation → rotate session)."*
  - **sole-writer / banking** — `COMMANDER.md:29` *"You are the SOLE memory writer... Maintain BY HAND — NEVER `/consolidate-memory`. Mirror `memory/` ↔ `~/.claude`..."* vs `MEMORY.md:36` *"NEVER use the /consolidate-memory skill... Mirror `memory/` ↔ `~/.claude`."*
  - **prompt-emission** — `COMMANDER.md:41` *"re-emit COMPLETE (never splices), one copy-paste box per relay. Each carries SESSION + MODEL(version) + EFFORT..."* vs `MEMORY.md:37` *"re-emit COMPLETE worker prompts, never splices. Every prompt carries SESSION + MODEL(version) + EFFORT + canary..."*
  - The split IS explicitly designed — `MEMORY.md:33` *"This MEMORY.md = project STATE; COMMANDER.md = durable CRAFT"* and `COMMANDER.md:51` *"(Full project-specific directives + footguns live in `MEMORY.md` — this section is the generalizable craft.)"*.
- **Recommended action:** Do NOT collapse — this overlap is intentional (a boot prompt that is self-sufficient on craft + a state index that restates the live directives). Leave as-is; optionally tighten by having `MEMORY.md:33`'s pointer be the single source for the canary/banking WORDING and have COMMANDER.md cite it — but that risks the boot prompt no longer being self-contained. Recommend: keep both, accept the redundancy as deliberate robustness.
- **Info-loss risk:** none — recommending no change (or only a cross-reference); both copies retained.

---

## PRESENTABILITY (2)

### P-1 — §2 doctrine paragraph density / readability
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/COMMANDER.md:43` (1708 chars), `:45` (1242), `:41` (1110); also lineage `:65` (1148), `:66` (1004)
- **Issue type:** presentability
- **Severity:** low
- **Evidence:** §3 rule 1 mandates new doctrine be *"concise, actionable, with the one-line 'why'"* (`COMMANDER.md:55`), but several §2 entries are single run-on paragraphs >1 KB. Line 43 (cold-start prompt doctrine) is 1708 chars in one block — an 11-item enumerated list (1)…(11) plus a parenthetical case study, all inline with no line breaks. Line 41 (worker-prompt + EFFORT-tier rule) packs the AUP checklist AND the full tier enumeration AND the Gen-5 self-mistake into one paragraph.
- **Recommended action:** When a future commander next edits §2, break line 43's inline (1)…(11) into a short bulleted sub-list and split the Gen-5 EFFORT-tier self-mistake parenthetical out of line 41's body (the mistake is already recorded in the Gen 5 lineage at `:66`, so the inline copy at `:41` can be shortened to a pointer). Cosmetic only — no rush.
- **Info-loss risk:** none if reformatting only. NOTE before shortening line 41's EFFORT-mistake parenthetical: verify it is fully captured at `:66` (it is — *"labeled an `extra`-effort σ_p0 worker prompt 'EXTRA / maximal (ultracode — self-orchestrate)'..."*), so the inline copy can be reduced to a pointer without loss.

### P-2 — Lineage integrity (§4) coherence check (clearing finding)
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/COMMANDER.md:61-66`
- **Issue type:** presentability
- **Severity:** low
- **Evidence:** Lineage is coherent and complete: Gen 1 (2026-06-16) → Gen 2 (2026-06-16) → Gen 3 (2026-06-18) → Gen 4 (2026-06-18) → Gen 5 (2026-06-19), all opus-4.8, each with model + date + "the one key thing added" per the §4 format spec (`:61`). Sequential numbering, no gaps, dates monotonic. Each entry's "key thing added" is traceable to actual §2 content (Gen 3→cold-start doctrine at `:43`; Gen 4→git-safety `:45` + instrument-before-subject `:47`; Gen 5→EFFORT-tier `:41` + mistake-capture `:55`). `git log` confirms one commit per generation. The §3 protection clause (`:53` *"never delete §3 or §4"*) is intact.
- **Recommended action:** No action — lineage is sound. The next commander (Gen 6) should append a new dated line per §3.3 and may re-tense the stale `:49` fact noted in S-1.
- **Info-loss risk:** none — informational / clearing finding.

---

## GAP (0)

None found for this area.

---

## PRUNE (0)

None found for this area. (No content is recommended for removal — every actionable item is re-tense, clarify, or no-op.)

---

## Summary of recommended edits (lose nothing)
1. **The only substantive edit:** re-tense the stale clause at `COMMANDER.md:49` from *"inc8 isn't deterministically flyable"* to past tense, with a pointer to live state in `memory/` (S-1, also resolves C-1's live tension). med severity.
2. **Optional cosmetic, no rush:** break up the dense §2 paragraphs at `:43` and `:41` next time §2 is edited (P-1).
3. **No-op / leave as-is:** the intentional MEMORY.md↔COMMANDER.md directive overlap (D-1); the Gen-4 historical state snapshot at `:65` (C-1, history is acceptable in §4); the absent stale OLD-view (S-2) and sound lineage (P-2) are clearing findings.
