# Nocturne

A **screenless, listen-and-type shell for Claude Code.** Lie down, headphones
on, keyboard in your lap. Claude's answers are read aloud; you type replies.
Music plays while Claude is working (an ambient "still thinking" indicator) and
gets out of the way while Claude is speaking. No screen required.

> **Design principle — audio is the only output channel.** Every piece of
> state (thinking, speaking, done, error, needs-your-input, context-low) is
> conveyed by sound: speech or a short, distinct *earcon*.

> **How it works — it *is* your Claude Code, spoken.** Nocturne is an audio
> skin over the Claude Code you already use. By default it spawns the
> subscription-authenticated `claude` CLI in headless stream-json mode and
> drives it over stdin/stdout — so it runs on **your Pro/Max subscription, with
> no API key and no per-token API charges.** Same session, same tools, same
> `CLAUDE.md`; the only difference is you *hear* the answers and *type* replies.
> (An optional `sdk` driver uses the API key / credits instead — off by default.)

---

## The interaction states

The whole app is a loop over five states, each owning an audio behaviour:

| State | Trigger | Audio |
|-------|---------|-------|
| **IDLE** | waiting for you to type | silence + soft *your-turn* earcon |
| **WORKING** | you submit, until the first sentence is ready | music fades in; optional terse breadcrumbs ("running the tests…") |
| **SPEAKING** | assistant prose is ready | music ducks out; TTS reads the prose |
| **NEEDS_CONFIRM** | Claude wants a risky/irreversible action | music pauses, urgent earcon, spoken yes/no gate |
| **ERROR** | SDK/tool error | error earcon + one-sentence summary |

Because music plays for exactly as long as real work takes, its duration is a
genuine "still thinking" signal during long agentic runs.

---

## Install

Requires **Python 3.10+** and **Claude Code** installed and **logged in**
(Pro/Max subscription). **No `ANTHROPIC_API_KEY` and no API credits** in the
default `cli` mode.

```bash
cd nocturne
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -e ".[all]"     # or pick extras: .[audio], .[spotify], .[sapi]
```

### Log in to Claude Code (once)

Nocturne uses your Claude Code login — the same subscription you already pay
for. If a spawned `claude` reports *"not logged in"*, sign in once:

```bash
claude              # opens the OAuth login in your browser, then exit
# or, for a long-lived headless token:
claude setup-token
```

Credentials persist to `~/.claude/.credentials.json`; Nocturne's spawned CLI
reads them automatically. **Do not** set `ANTHROPIC_API_KEY` unless you want the
`sdk` driver (API billing) — if it's set, the CLI would use it and charge
credits, so the default `cli` driver explicitly ignores it.

Extras:

- `sapi` → `pyttsx3` (the Windows/OS voice)
- `audio` → `sounddevice` + `numpy` (Piper/ElevenLabs playback, ambient pad, non-Windows earcons)
- `spotify` → `spotipy`
- `all` → everything

The **built-in Claude Code voice** (`claude`) and Windows **SAPI** backends need
no extra downloads. Piper/ElevenLabs are opt-in (see below).

## Run

```bash
nocturne                                  # or: python -m nocturne  (uses your subscription)
nocturne --voice claude --mode bedtime    # pick a voice + start in bedtime mode
nocturne --list-voices                    # which TTS backends are usable here
nocturne --cwd /path/to/project           # run the Claude Code session in that project
nocturne --continue                       # resume the most recent Claude Code session
nocturne --driver sdk                     # opt into the API-key/credits path instead
```

No key to export in the default mode — just make sure `claude` is logged in.

---

## Voices (TTS)

Pluggable; choose per run with `--voice` or in `nocturne.toml` under `[tts]`:

| Backend | What it is | Setup |
|---------|-----------|-------|
| `claude` | **Built-in system voice** (PowerShell `System.Speech` on Windows, `say` on macOS, `spd-say`/`espeak` on Linux) | none — zero dependency |
| `sapi` | OS voice via `pyttsx3` | `pip install pyttsx3` |
| `piper` | Local neural, great for bedtime | install the `piper` binary + a `.onnx` voice; set `piper_model` |
| `elevenlabs` | Cloud, best quality | set `ELEVENLABS_API_KEY`, `elevenlabs_voice_id` |
| `none` | Silent / prints text | — |
| `auto` | first that works (piper → elevenlabs → **claude** → sapi) | — |

> **Windows note:** `auto` prefers the built-in `claude` (System.Speech) voice
> over `sapi`, because pyttsx3's SAPI5 driver only speaks the *first* utterance
> per process and then goes silent — bad for a stream of sentences.

Rate lives in `[tts] rate` (you'll likely want `1.6`–`2.0`).

---

## Spotify (the thinking indicator)

Needs **Premium** and an **active device** (Spotify open somewhere). Set:

```bash
export SPOTIFY_CLIENT_ID=...
export SPOTIFY_CLIENT_SECRET=...
```

Create an app at <https://developer.spotify.com/dashboard> with redirect URI
`http://127.0.0.1:8888/callback`. Scopes used: `user-modify-playback-state`,
`user-read-playback-state`.

On WORKING it resumes playback (fade in); on SPEAKING it pauses or ducks (fade
out). Your current playback is snapshotted on start and **restored on exit**.
Point `thinking_context_uri` at a playlist to force specific music.

**No Premium / no device / disabled?** It degrades gracefully to a generated
local **ambient pad** so you still get a thinking indicator. (Spotify's API
terms forbid commercial/broadcast/sync-to-visual use; a personal single-user
tool is fine.)

---

## Controls

Typed `/commands` (press Enter):

`/again` replay last · `/faster` · `/slower` · `/skip` · `/stop` · `/context`
speak % remaining · `/recap` summarise session · `/mode [bedtime|focused]` ·
`/voice <backend>` · `/help` · `/quit`

Control chords (instant, no Enter):

`Ctrl-K` skip/stop speech · `Ctrl-R` replay · `Ctrl-N` read slower ·
`Ctrl-T` toggle mode · `Ctrl-G` speak context

**Barge-in:** start typing while Claude is speaking and speech cuts immediately
(the keystroke still goes into your next message). Submit a new message mid-turn
and it interrupts the agent and starts the next turn.

---

## Screenless safety

You can't watch what Claude Code does, so every tool call is announced aloud as
a breadcrumb ("editing config.py", "running a command") and results are
summarised, never dumped.

- **On the `cli` driver (default):** permission behaviour follows Claude Code's
  own `permission_mode` and your `~/.claude/settings.json` rules — the same
  gating you already have interactively. Set `[agent] permission_mode` to
  `plan` (read-only), `default`, `acceptEdits`, or `bypassPermissions`; bedtime
  mode switches to `plan`. *(A spoken yes/no gate over the CLI's
  `--permission-prompt-tool` is implemented for the `sdk` driver and is the next
  step for `cli` — see Status.)*
- **On the `sdk` driver:** dangerous/irreversible calls (`rm -rf`,
  `git push --force`, `sudo`, writing to `.env`/ssh keys, external
  `send`/`delete` MCP actions, …) hit an **audible confirmation gate** — music
  pauses, an urgent earcon plays, the action is described aloud, and it waits
  for a typed **yes/no** — via the SDK's `PreToolUse` hook.

- **Bedtime mode** → confirm-before-write (`plan` permission mode), softer/quiet
  voice. **Focused mode** → more autonomous. Spoken on startup; toggle any time.
- **Idle auto-sleep** → after a few minutes of silence it asks "still there?",
  then goes quiet so nothing blasts if you fall asleep.
- **Away ping** (optional) → if a confirmation goes unanswered, send a phone
  push (ntfy/Pushover) instead of talking into an empty room.

---

## Context & rate-limit warnings

Token usage from each turn's `ResultMessage` is compared against the model's
window (`[context] window_tokens`; 200k standard, 1M on newer models). Nocturne
speaks a warning at `warn_thresholds` (default 75%, 90%) and a distinct heads-up
just before auto-compaction so you can choose to wrap up. `/context` reports the
current figure on demand; rate-limit heads-ups are spoken too.

---

## Morning transcript

Every session is logged to a timestamped markdown file under `[session] log_dir`
(prompts, prose, tools run, files touched). `/recap` speaks a summary so far.

---

## Configuration

One TOML file. Resolution order: `--config` → `$NOCTURNE_CONFIG` →
`./nocturne.toml` → `~/.config/nocturne/nocturne.toml` → the shipped default.
See [`nocturne.toml`](nocturne.toml) for every field with comments.

## Layout

```
nocturne/
  __main__.py        CLI entrypoint (python -m nocturne)
  app.py             orchestrator / state machine
  claude_cli.py      DEFAULT driver: spawn the subscription Claude Code CLI (stream-json)
  agent_loop.py      optional sdk driver: ClaudeSDKClient wrapper (API key)
  events.py          internal event vocabulary (both drivers emit these)
  speech_filter.py   prose⇄tool⇄result classification, sentence streaming, code suppression
  speaker.py         the speaking pump (queue, ducking callbacks, replay, barge-in)
  context_tracker.py context-window tracking + warnings
  safety.py          danger heuristics + PreToolUse confirmation hook
  spotify.py         thinking-indicator control + ambient fallback
  notify.py          away notifications (ntfy / Pushover)
  logsession.py      morning transcript
  input.py           prompt_toolkit prompt, barge-in, chords
  config.py          TOML → dataclasses
  states.py          the five states
  tts/               base + sapi, piper, elevenlabs, claude_voice, none
  audio/             earcons (generated) + mixer (fades/duck/ambient)
```

## Tests

```bash
pip install pytest pytest-asyncio
pytest
```

44 tests cover the speech filter, context tracker, config, safety heuristics,
SDK + CLI stream-json message translation, a real CLI-driver subprocess
round-trip (against a fake `claude`), and end-to-end turn orchestration (fake
agent + recording TTS) — no audio hardware, API key, or login needed to run them.

---

## Status

All build-order phases implemented and tested against fakes: the listen-and-type
loop, TTS, streaming + speech filtering, Spotify + ambient fallback,
earcons/hotkeys/barge-in, context/rate-limit warnings, bedtime mode, idle-sleep,
transcript/resume, and away notifications. The default **`cli` driver runs on
your subscription** and its stream-json protocol is verified end-to-end against a
fake `claude`.

**Needs on-device verification** (can't run headless): (1) a real logged-in
`claude` turn — confirms the stream-json *input* schema against the live CLI;
(2) audio playback + the prompt_toolkit loop (needs a TTY + speakers); (3) the
spoken yes/no permission gate for the `cli` driver via `--permission-prompt-tool`
(the MCP contract is under-documented and must be pinned against the live CLI).
Drive `nocturne` in a real terminal for final acceptance.
```
