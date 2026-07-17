# Episode-report engine — hardening spec (2026-07-14)

Owner: god (design + acceptance) · Builder: Jim · Status: DISPATCHED

## What this is

`scratch/episode_report.py` is the game-agnostic perception substrate for the
copilot: it decodes a RAM-capture `.npz` + the game's `training.json` into
variable ROLES → an EVENT STREAM → a compact POST-MORTEM, so the model reasons
over labeled facts instead of squinting at pixels. god built and proved it on
3 real captures; this spec is the punch list to make it production-grade
before Pam's usability pass and studio integration.

## Fixtures (real captures, in `scratch/fixtures/`)

| Fixture | Game | Steps | Ground truth (verified by god + Schuyler) |
|---|---|---|---|
| `lm.npz` + `lm_training.json` | Little Mermaid (NES) | 1400 | lives 2→1(@477)→0(@988); health cycles 3→0 then RESETS to 3 on respawn; playerPage loops 0→6 ×3, never exceeds 6; stage=0 throughout; pearls/score=0. Both deaths at playerPage=6, roomPos=1. |
| `mario.npz` + `mario_training.json` | Super Mario Bros (NES) | 560 (real game-over) | done = `lives == 0`. `time` is a COUNTDOWN timer. Real deaths ≈2 (lives losses), then terminal. Scrolling/playerX flicker creates false loop signals. Reached playerPage 6, died there. score +20. |
| `fzero.npz` + `fzero_training.json` | F-Zero (SNES) | 1400 | health=1643-scale energy bar (continuous, ticks constantly); done = `health < 100` OR `race_on == 0`. Survived the window. Racing story (pos/speed/laps) is what matters; current output is 348 events of health-tick spam. |

Capture substrate note: `sheeprl/_retro_ram_capture.py` DISABLES wrapper
done-conditions during capture, so traces run PAST in-game death — deaths must
be detected from RAM vars (lives/health via `_cond_true`), never from a
`terminated` flag. Already true in the engine; do not regress it.

## Defects to fix (all game-agnostic — no per-game code, no game-name checks)

1. **Significance-threshold continuous resources.** A resource whose range is
   large (e.g. span > ~64 distinct values or range >> typical delta) should not
   emit a `loss`/`regain` event per tick. Emit only significant changes —
   e.g. cumulative-delta buckets (report when total change since last event
   exceeds N% of span) or hysteresis. F-Zero health must go from 155 noise
   events to a handful of meaningful "took sustained damage" / "recovered"
   events. LM health (span 3, discrete) must KEEP its per-point loss events.

2. **Detect countdown timers.** A var that decreases near-monotonically at a
   near-constant rate for most of the episode is a CLOCK, not a resource being
   damaged. Role it `timer` (context-like); exclude from damage counts and
   from `loss` spam. Mario `time` is the fixture case. Detection must be
   shape-based (monotonic-down fraction + step-regularity), not name-based.

3. **Oscillator/jitter downgrade for undeclared counters.** The unrewarded
   counter-climbs rule (`1 < n_distinct <= 64 and rises >= 1`) promotes jittery
   sub-counters (Mario `scrolling`) to `progress`, producing ×29 false loops.
   Proposed rule: if `rises > span + 1` (it resets more times than it has
   distinct headroom) it's an oscillator → `context`, not `progress`. Tune on
   the fixtures; LM `playerPage` (loops ×3, span 6) must SURVIVE as progress.

4. **Death dedup.** A TERMINAL event coincident (within a few steps) with a
   life-counter `loss` is ONE death, not two. Also ignore life underflow
   (e.g. lives 0→-1 or 0→255 wrap after the real game-over). Mario must read
   ~2 deaths + terminal, not 4 death/fail.

5. **Racing progress semantics.** F-Zero surfaces almost nothing useful.
   Generic gap: rewarded vars whose meaning is rate/position (`speed`, `pos`)
   need a summary treatment — min/max/mean + trend for rate-like vars
   (high-cardinality, non-monotonic, rewarded or context), and lap/checkpoint
   progression if a small cyclic counter exists. Keep it generic: "for
   continuous rewarded/context vars, report distributional summary + trend in
   the post-mortem" rather than anything F-Zero-specific.

## Acceptance criteria (god will run these — verbatim gates)

- **LM (regression — must not change in substance):** OUTCOME still 2
  death/fail; FAILURE LOCATION still playerPage=6 (2/2) + roomPos=1 (2/2);
  loop ×3 story intact; "never advanced stage, collected nothing" intact.
- **Mario:** reads "died ~2×, reached playerPage 6, died there"; `time` is a
  timer, NOT damage; no false looping-×29; score +20 kept; event stream ≤ ~30.
- **F-Zero:** event stream ≤ ~30 (from 348); no per-tick health spam; verdict
  is a racing-appropriate story (survived, speed/pos summary); done-condition
  semantics (`health<100`, `race_on==0`) still threshold-correct (no false
  deaths from a big-but-above-threshold health drop).
- **All three:** zero per-game code — no game names, no var-name special
  cases (matching on SHAPE and CONFIG is fine); output stays compact enough
  to hand a 27B model (~15-30 events + post-mortem ≤ ~25 lines).
- Runs under system python3 (numpy only) exactly as today:
  `python3 scratch/episode_report.py scratch/fixtures/<g>.npz scratch/fixtures/<g>_training.json`

## Non-goals (do NOT do these)

- No studio/backend integration, no API endpoints, no copilot primer changes
  — that's a later phase owned by god.
- No new dependencies, no rewrite — evolve the existing 253-line file, keep
  its structure and comment density.
- No changes to `sheeprl/_retro_ram_capture.py`.
