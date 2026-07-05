# Race-wire evidence package — 2026-07-05

Raw MAVLink RACE_STATUS + COLLISION wire dumps from two existing recorded RL sessions,
plus a verbatim extract of the relevant spec sections from VADR-TS-003. Produced for
the Peregrine RL team as an evidence package (not an analysis/verdict — see
`race_outcome.py` for that layer). Generated read-only against the two source session
dirs; nothing in the other worktree was modified.

## Struct layout (source of truth: `src/racer/mavlink_client.py`)

`ENCAPSULATED_DATA.data[0]` is a discriminator byte the sim repurposes to multiplex two
payload kinds (mavlink_client.py:87-93):

```
_ENCAP_RACE_STATUS = 1     # mavlink_client.py:87
_ENCAP_TRACK_INFO  = 2     # mavlink_client.py:88
```

**RACE_STATUS** (`data_type == 1`), struct format `<BQqqIq` (mavlink_client.py:91,
decoded by `parse_race_status` at mavlink_client.py:96-110):

| field | struct code | type | meaning |
|---|---|---|---|
| `data_type` | `B` | uint8 | discriminator, always `1` here |
| `sim_boot_ms` | `Q` | uint64 | sim boot time, ms |
| `race_start_boot_ms` | `q` | int64 | boot-ms the race started; **negative = not started** |
| `race_finish_ns` | `q` | int64 | race finish time, ns; **negative = not finished / ongoing** |
| `active_gate_index` | `I` | uint32 | the gate that is **next** (not yet passed) |
| `last_gate_race_time` | `q` | int64 | race-clock time of the last gate passed |

`parse_race_status` (mavlink_client.py:96-110) derives `started = race_start_boot_ms >= 0`
and `finished = race_finish_ns >= 0` — i.e. negative sentinel, not a separate bool field.
Per `race_outcome.py:10`: "An `active_gate_index` increment i->i+1 means gate i was PASSED."

`TRACK_INFO` (`data_type == 2`) is a chunked gate-map transfer (announced via
DATA_TRANSMISSION_HANDSHAKE, reassembled in `_ingest_track_chunk`,
mavlink_client.py:350-374); **neither of the two dumped sessions contains any
`data_type == 2` records** (see per-run breakdown below) — `handoff/tools/dump_race_wire.py`
still tags them with a note if one ever appears.

**COLLISION** is a separate, standard MAVLink message type (not multiplexed through
ENCAPSULATED_DATA). Fields used (per `mavlink_client.py:317-324` /
`race_outcome.py:21-23,128-133`):

| field | meaning |
|---|---|
| `id` | `1001` = gate collision, `1002` = environment collision |
| `threat_level` | `1` or `2` (2 = harder hit) |
| `horizontal_minimum_delta` | impulse, exposed by the dump/tools as `impulse` |

## Tool

`handoff/tools/dump_race_wire.py` — read-only, one JSONL line per ENCAPSULATED_DATA or
COLLISION message, via `racer.recording.RecordingReader(session_dir).iter_mavlink()`
(same reader `race_outcome.py` uses). Each line:

```json
{"recv_timestamp": <float, seconds>, "msg_type": "ENCAPSULATED_DATA"|"COLLISION",
 "data_type": <int|null>, "raw_hex": "<full payload bytes as hex>",
 "decoded": {...} | null, ...}
```

`recv_timestamp` is the `_timestamp` attribute pymavlink's `mavutil.mavlink_connection`
attaches when reading a `.tlog`: the 8-byte big-endian UNIX-epoch-microsecond prefix
that `racer.recording.Recorder._writer_loop` stamps at **message receive time**
(`time.monotonic_ns()`, bridged to a synthesized UNIX epoch via the `meta.json`
`t0_unix_ns`/`t0_monotonic_ns` pair — see `recording.py:18-19,179-180`). It is a
receive-time stamp, not a send-time or write-time stamp. Usage:

```
python handoff/tools/dump_race_wire.py <session_dir> -o out.jsonl
```

## Dump files

| file | source session dir (read-only, different worktree) |
|---|---|
| `20260705_012253_rl_s1_f1.race_wire.jsonl` | `.../agent-a08b24f288de69e40/data/runs/20260705_012253_rl_s1_f1` |
| `20260704_231155_rl_s1_f1.race_wire.jsonl` | `.../agent-a08b24f288de69e40/data/runs/20260704_231155_rl_s1_f1` |

Both sessions' `meta.json` show `label: "rl_s1"`, `flight: 1`, `max_seconds: 120.0`,
`final_state: "TIMEOUT"`, `checkpoint: stage1_inc7_actor.pth` — these are RL policy
flights that ran to the flight-stack's own 120s wall-clock timeout, not the sim's
8-minute spec limit (VADR-TS-003 §8.3; see `ts003_race_format_extract.md`).

### `20260705_012253_rl_s1_f1.race_wire.jsonl`

- 1923 lines total: 599 `ENCAPSULATED_DATA`(data_type=1) + 1324 `COLLISION`. Zero
  `data_type=2` (TRACK_INFO) records.
- first `recv_timestamp`: `1783214573.9480` — last: `1783214718.8988` (span 144.95 s)

| event | from | to | recv_timestamp |
|---|---|---|---|
| first RACE_STATUS seen | — | `active_gate_index=0`, `started=False`, `finished=False` | 1783214573.9480 |
| `started` | False | True | 1783214595.4764 |
| `active_gate_index` | 0 | 1 | 1783214601.9914 |
| (end of recording) | — | `active_gate_index=1`, `started=True`, `finished=False` | 1783214718.8988 |

No further `active_gate_index` change and `finished` never goes True — the run passes
gate 0 only, then times out (consistent with `meta.json` `final_state: "TIMEOUT"`,
`gate_index: 1`).

Collisions: **all 1324 are `id=1002` (environment)** — 1316 at `threat_level=1`, 8 at
`threat_level=2`. **Zero `id=1001` (gate) collisions.** First collision
`recv_timestamp=1783214605.321` (3.3 s after `started=True`), last
`1783214702.806`. Inter-arrival times cluster at 0.0001–0.0008 s — i.e. collision
messages are arriving in dense bursts (sustained-contact spam), not as discrete
one-off impact events. Impulse (`horizontal_minimum_delta`) range: 0.0001–3.5754.

### `20260704_231155_rl_s1_f1.race_wire.jsonl`

- 10614 lines total: 569 `ENCAPSULATED_DATA`(data_type=1) + 10045 `COLLISION`. Zero
  `data_type=2` (TRACK_INFO) records.
- first `recv_timestamp`: `1783206715.0608` — last: `1783206851.6022` (span 136.54 s)

| event | from | to | recv_timestamp |
|---|---|---|---|
| first RACE_STATUS seen | — | `active_gate_index=0`, `started=False`, `finished=False` | 1783206715.0608 |
| `started` | False | True | 1783206727.8211 |
| `active_gate_index` | 0 | 1 | 1783206734.5938 |
| (end of recording) | — | `active_gate_index=1`, `started=True`, `finished=False` | 1783206851.6022 |

Same pattern: only gate 0 passed, never finishes, times out
(`final_state: "TIMEOUT"`, `gate_index: 1`).

Collisions: **all 10045 are `id=1002` (environment)** — 10042 at `threat_level=1`, 3 at
`threat_level=2`. **Zero `id=1001` (gate) collisions.** First collision
`recv_timestamp=1783206737.467` (2.6 s after `started=True`), last
`1783206851.602` (i.e. collisions continue right up to the recording's last sample —
essentially the whole post-launch flight is in continuous contact with something).
Impulse range: 0.0002–8.1833 (higher peak than the other run).

## Things that don't match a naive reading of the struct/spec — flagged, not resolved

1. **`meta.json` `collisions: 200` vs raw wire counts of 1324 / 10045.** The flight
   stack's own recorded metadata undercounts raw COLLISION messages by 6.6x and 50x
   respectively. 200 looks like a cap (both files report exactly `200`, suspicious for
   two different flights of different length/severity) or a debounced/deduped counter
   computed by different logic than "count every raw COLLISION message". This dump
   intentionally reports the raw wire count; do not conflate the two without checking
   whatever code writes `meta.json`'s `collisions` field (not inspected in this task —
   out of scope for a read-only capture, flagging for the RL team instead).
2. **Every collision in both runs is environment (`id=1002`), never gate (`id=1001`),**
   despite `active_gate_index` advancing 0→1 in both runs. This is plausible (env
   collision, e.g. wall/floor contact, is independent of gate-passing) but worth the RL
   team's attention given the density/duration of the environment-contact bursts
   (near-zero inter-arrival gaps for the entire post-launch flight in the second run) —
   this reads like the drone is dragging along a surface, not a single sharp impact.
3. **RACE_STATUS payload is 253 raw bytes; the decoded struct only needs 37
   (`struct.calcsize("<BQqqIq") == 37`).** This is expected/benign — ENCAPSULATED_DATA
   is a fixed-width 253-byte MAVLink field, `parse_race_status` only reads the first 37
   bytes it needs — but noting it since a naive byte-length check might otherwise look
   like a mismatch.
4. **No `data_type == 2` (TRACK_INFO) in either recording.** Consistent with existing
   memory (`vq2-wire-3379-pose-blocked.md`: gate-map blocked on the wire) and with
   VADR-TS-003 §9.3's blocked-message list (which names `GATE_INFO`, not `TRACK_INFO`
   — the doc's name and the code's name may or may not refer to the same thing; not
   resolvable from the spec text alone, see `ts003_race_format_extract.md`).
5. **VADR-TS-003 has no RACE_STATUS/ENCAPSULATED_DATA payload definition anywhere**
   (task brief expected one "near §9.x"; actual §9 is UI/telemetry-restriction/scoring
   text, no binary layout). See `ts003_race_format_extract.md` for the full search
   trail. The struct in `mavlink_client.py` is derived from the reference
   `PyAIPilotExample` client, not from this spec document.
6. **No explicit gate count or lap count anywhere in VADR-TS-003.** Course is described
   qualitatively ("start gate", "sequential race gates" / "intermediate gates", "finish
   gate") with no number given, and the word "lap" never appears — treat the course as
   an unquantified single-pass sequence per the spec text, not a fixed N-gate lapped
   circuit, unless the RL team has a number from another source (e.g. the actual
   TRACK_INFO gate count observed on the wire, which neither of these two recordings
   captured).

## Spec extract

See `ts003_race_format_extract.md` for verbatim section text (race format §3.1/§8.2,
time limit §8.3, scoring §9.4, plus supporting §9.1-9.3 and §4.3 context) and the
explicit note on the missing RACE_STATUS payload definition.
