# VADR-TS-003 extract: race format, timing, scoring, RACE_STATUS payload

Source: `260624_Technical_Spec_0003.pdf` (repo root, git-tracked — see README for commit).
Document ID **VADR-TS-003**, Issue **00.03**, Date **2026-06-24**. 12 pages total.
Extracted verbatim with `pypdf` (6.14.2); page numbers below are PDF page numbers
(1-indexed, matching the document's own page footer/TOC numbering).

**IMPORTANT — read this before the extracts below.** The task brief expected a
RACE_STATUS / ENCAPSULATED_DATA payload definition "likely near §9.x". This document's
§9 is "Round Two – Qualification Phase" (VQ2 rules: flight modes, telemetry
restrictions, scoring) — it contains **no binary payload / struct definition at all**.
The entire 12-page document was searched (full text extracted page-by-page, plus the
two embedded images inspected visually) for `RACE_STATUS`, `ENCAPSULATED_DATA`,
`struct`, `active_gate`, `gate_index`, and `appendix` — **zero matches**. The
ENCAPSULATED_DATA/RACE_STATUS wire format is apparently **not part of the published
spec**; it is only known from reverse-engineering the sim (see
`src/racer/mavlink_client.py` docstring: "Message coverage was confirmed against the
official `PyAIPilotExample` client shipped inside the sim zip" — i.e. the struct layout
came from that reference client's source, not from VADR-TS-003). Flagging this
explicitly rather than guessing or fabricating a §9.x citation.

Similarly: the doc gives **no explicit numeric gate count** and **no lap count** —
see below, verbatim.

---

## §3.1 General Environment (p.5) — race format / course elements

> The race takes place within a high-fidelity real-time physics simulator.
> • start gate
> • sequential race gates
> • finish gate
> • vertical and horizontal obstacles
> • boundary elements
> • terrain and environmental structures
>
> Gates will be visually distinctive to the environment, but consistent throughout the
> Virtual Qualifier 1 track.

No numeric gate count is given ("sequential race gates", plural, unquantified). No
mention of laps — the course is described as a single point-to-point sequence
(start gate → intermediate/sequential gates → finish gate), not a lapped circuit.

## §8.2 Course Structure (p.11, Round One — Qualification Phase)

> 8.2 Course Structure
> • start gate
> • intermediate gates
> • finish gate

Same pattern: no explicit count of intermediate gates, no lap count, no mention of
multiple laps anywhere in the document (searched full text for "lap" — zero hits).

## §8.3 Maximum Run Duration (p.11) — time limit / timeout

> 8.3 Maximum Run Duration
> Maximum run duration: 8 minutes.

This is the only duration/timeout figure in the document. It appears under **Round
One** specifically (§8, "Round One — Qualification Phase"); the doc does not restate
or override this figure under §9 (Round Two). No explicit statement of what happens at
the 8-minute boundary (e.g. forced disarm, DNF, truncated recording) is given anywhere
in the text — flagging this as an ambiguity, not filling it in.

The word "timeout" itself does not appear in the document; "Maximum run duration" is
the closest concept.

## §9.4 Leaderboard, Scoring & Ranking Logic (p.12) — scoring (finish time vs partial credit)

> 9.4. Leaderboard, Scoring & Ranking Logic
>
> Ranking Criteria: Leaderboard (Qualification Event) ranking is strictly determined
> based on the timing results of valid, completed runs.
> • Faster times rank higher.
> • Attempts: Contestants are permitted an unlimited number of attempts to set their
>   fastest time.
> • Team Scoring: Leaderboard scoring is team-centric. The best single timing result
>   from a team dictates that team's overall rank, regardless of which individual team
>   member achieved the time.

Key phrase: **"valid, completed runs"** — scoring is explicitly finish-time-based, and
explicitly restricted to *completed* runs. There is no partial-credit provision
described anywhere in the document (no mention of "DNF", "partial", "gates passed" as
a scoring criterion, or any ranking mechanism for a run that does not finish). This
directly matters for both recorded sessions in this package: neither run reaches
`finished=True` (see README), so under this spec's literal wording neither run would
produce a ranked/scored time.

## §9.1–9.3 (p.11–12) — surrounding VQ2 context (verbatim, for completeness)

> 9. Round Two – Qualification Phase
> 9.1 Objective
> Objective: The core objective of Phase 2 is to qualify for the Physical Qualifier
> Event in September.
>
> 9.2 User Flow & Simulator Interface
> To help you adapt to these changes, teams can now choose between two distinct flight
> modes:
> • Training Flights: These flights allow you to test your algorithms and state
>   estimation pipelines freely. They are not counted toward the official leaderboard.
> • Competitive Flights: These are official, time-tracked runs that count directly
>   toward your qualification standing.
> • Event Selection: The simulator interface will present two distinct, manually
>   selectable event blocks to the user:
>   o Training
>   o Qualification (VQ2) - runs count for qualification for Physical Qualifier
> • Execution: The contestant must select the desired event block to launch the
>   respective simulator mode.
>
> ⚠ Important Note on Code Integrity: Once you submit a time-tracked race for
> qualification, DCL reserves the right to review your codebase. If the Race Director
> suspects any form of cheating or manipulation of the simulator constraints, a formal
> code audit will be triggered.
>
> 9.3 Telemetry API Restrictions
> To ensure competitive integrity during the qualification phase, certain direct data
> streams from the simulator API are restricted, and strict track validation is
> enforced:
> • Blocked Telemetry Messages: Starting in Phase 2, the following interfaces/messages
>   will no longer be available directly from the simulator API:
>   o ATTITUDE
>   o LOCAL_POSITION_NED
>   o ODOMETRY
>   o GATE_INFO

This corroborates the existing memory note ("VQ2 wire 3379 — pose BLOCKED": ATTITUDE /
LOCAL_POSITION_NED / ODOMETRY blocked in both modes) — it is spec-documented, not just
empirically observed. Note the doc calls the fourth blocked message `GATE_INFO`; the
codebase's TRACK_INFO (ENCAPSULATED_DATA data_type==2) is presumably the same concept
under a different name adopted by the implementation — another naming point not
resolvable from this document alone.

## §4.3 Supported MAVLink Messages (p.8) — for cross-reference

> Message | Direction | Purpose
> HEARTBEAT | Simulator → Client | Connection status
> ATTITUDE | Simulator → Client | Vehicle attitude
> HIGHRES_IMU | Simulator → Client | Vehicle status
> SET_POSITION_TARGET_LOCAL_NED | Client → Simulator | Control interface
> SET_ATTITUDE_TARGET | Client → Simulator | Control interface
> TIMESYNC | Simulator → Client | Timing
> HIGHRES_IMU | Simulator → Client | Measurements

Note: this table does **not** list ENCAPSULATED_DATA, COLLISION,
DATA_TRANSMISSION_HANDSHAKE, ODOMETRY, LOCAL_POSITION_NED, COMMAND_ACK, STATUSTEXT, or
ACTUATOR_OUTPUT_STATUS — all of which the codebase actively parses (per
`mavlink_client.py`'s own docstring, confirmed against the reference client, not this
spec table). The spec's message table is a partial/illustrative list, not an
exhaustive wire contract.
