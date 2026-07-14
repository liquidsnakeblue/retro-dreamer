# Bug Sweep — 2026-07-14 (Dwight)

Schuyler's standing order: bugs found along the way get investigated and FIXED.
Five known studio bugs worked. Each is **fixed+verified+committed** or
**escalated with evidence** (per god's rule: reward-shaping judgment calls are
not guessed). Live server: `http://localhost:8091` (cwd
`/home/liquidsnakeblue/retro-dreamer`, served via the `fzero-dreamer` venv
interpreter). One backend restart was performed (training was **idle** + last
copilot activity ~65h prior — both within god's restart preconditions) to load
the route/tool/trainer changes and verify E2E.

Commits (`git log --oneline`):
```
236f457 fix(training): /training/stop stamps user-intent; watchdog honors it
1b38ccb fix(tools): record_episode honors game_id (scopes latest, validates game)
e56bd51 fix(metrics): populate avg_length by printing+parsing episode length
0f6f5f4 fix(primer): tool job status is done|failed, not done|error
```

---

## 1. `/api/metrics/history` `avg_length` always `0.0` — FIXED

**Root cause.** `/api/metrics/history` and `/training/status` return
`avg_length` from `_MetricsTracker` (`backend/training/trainer.py:802`), which
parses SheepRL subprocess stdout. SheepRL's dreamer_v3 loop **computes**
`ep_len` (`sheeprl/.../dreamer_v3/dreamer_v3.py:635`, `:646`) and even aggregates
it to TensorBoard as `Game/ep_len_avg` — but its `fabric.print` lines emitted
**only** `policy_step=…, reward_env_0=…` and never the length. So `avg_length`
was initialized to `0.0` (`trainer.py:810`) and never updated. (The richer
`MetricsCollector` in `callbacks.py` is dead code — nothing calls its
`log_episode`; the live tracker is `_MetricsTracker`.)

**Fix.** (a) Add `length_env_{i}={ep_len}` to both `fabric.print` sites in
`dreamer_v3.py` (`:640` gym≥1.0 path, `:650` gym 0.29 path — `ep_len[-1]`
there). (b) Rewrite `_MetricsTracker.parse_log_line` to parse both
`reward_env_\d+=…` and `length_env_\d+=…` with regex and maintain a rolling
100-episode `avg_length` (matching `avg_return`'s window). The regex also
hardens the reward parse: adding length introduced a **second `=`** on the line,
which broke the old `line.split("reward_env_")[1].split("=")[1]` logic.

**Verification.** Cannot run a training pass this sweep (boundaries: no training
start/stop). Instead unit-tested `_MetricsTracker` against the **exact** new
print format (3 samples incl. a `track=go` suffix and a negative reward):
`avg_length=1124.0`, `avg_return=59.067`, `max_return=132.5` — all correct, field
now populates. Live E2E confirms on the next training run (the field is already
present in `/training/status`; it simply stays 0.0 until episodes complete).

---

## 2. `record_episode` ignores its `game_id` — FIXED

**Root cause.** `POST /api/tools/record_episode` (`backend/tools.py:211`) built
the recorder command from `req.checkpoint`, `req.seconds`, `req.state` — but
**never passed `req.game_id`**. Worse, `checkpoint="latest"` resolved
**globally**: the recorder's `find_latest_checkpoint()` uses
`catalog.get_watch_head()` (any running session's head) or falls back to a
**cross-game mtime scan**. So a caller asking to record game A could silently
record game B's brain. (Confirmed by Jim's friction list + as-built notes §8.)

**Fix.** Honor `game_id`: (a) validate it via `_game_dir(req.game_id)` → **404**
if the game has no integration dir; (b) when `checkpoint=="latest"`, resolve
**that game's** resumable head via `catalog.get_resumable_head(con, game_id)`
and pass a concrete checkpoint path to the recorder (bypassing its global
resolution entirely); (c) if the game has no checkpoint, reject explicitly with
**409** instead of recording the wrong game.

**Verification (E2E, post-restart).**
- Bad game_id → `POST /api/tools/record_episode {"game_id":"DoesNotExist",…}`
  returned **HTTP 404** `game 'DoesNotExist' has no custom integration dir`
  (previously ignored → would have spawned a global-latest recording).
- Scoped resolution confirmed via `catalog.get_resumable_head`:
  `FZero-Snes` → `…/dreamer_v3/FZero-Snes/2026…/ckpt`, `FZero-Test` →
  `…/dreamer_v3/FZero-Test/2026…/ckpt` (distinct, per-game paths — no longer
  global).

---

## 3. `/training/stop` doesn't mark user-intent (watchdog can't tell stop vs crash) — FIXED

**Root cause.** `POST /api/training/stop` set the trainer to `IDLE` but wrote no
intent. `scripts/overnight_watchdog.py:94` restarts training whenever
`state == "error" or (seen_training and state == "idle")` — so a **user** Stop
was indistinguishable from a **crash** and got auto-resumed. Confirmed in the
wild 2026-07-10 (as-built §12): a UI Stop button click was resumed ~2min later.

**Fix.** (a) `/training/stop` now writes `training-state/stopped_by_user.json`
`{"ts": <unix>, "reason": "manual_stop"}`. (b) The watchdog now tracks
`last_training_seen_t` and, when it detects a bad idle/error, calls
`user_stopped_after(last_training_seen_t)` — if a marker exists **newer than**
the last `training` sighting, it logs, **consumes the marker**, and does **not**
restart (resumes only on the next explicit `/training/start`). A crash leaves no
fresh marker, so the existing restart path is unchanged; a stale marker from an
earlier stop is correctly ignored (its `ts` predates the new training sighting).

**Verification (E2E + unit).**
- `POST /training/stop` (idle) → `training-state/stopped_by_user.json` written
  with fresh `ts` + `reason: manual_stop`. ✅
- `user_stopped_after()` unit-tested for three cases: fresh marker (suppress ✅),
  stale marker (no-suppress ✅), no marker / crash (no-suppress ✅).

---

## 4. Primer says `done|error`, code emits `done|failed` — FIXED

**Root cause.** `backend/copilot_primer.md:250-251` documented the tool-job
status contract as `queued -> running -> done|error`. The code
(`backend/tools.py:61,67,77`) emits `running` → `done` (exit 0) or `failed`
(nonzero / exception) — **never `error`**. So the copilot was told to poll for a
status string that never appears.

**Which side is true?** The **code**. `tools.py:61` `job["status"] = "done" if
code == 0 else "failed"`; `:67` `"failed"` on exception. Aligned the primer to
reality.

**Fix.** Primer now reads: `running -> done|failed (code reality in
backend/tools.py: 'done' on exit 0, 'failed' otherwise)`.

**Verification.** `grep` confirms `tools.py` emits only `done|failed/running`;
primer text read-back shows the corrected line.

---

## 5. F-Zero speed-reward units mismatch — **ESCALATED** (judgment call, not guessed)

**The defect (verified by math, not a guess).** `games/FZero-Snes/training.json`
configures speed as `mode: quadratic, max_speed: 500, base_reward: 0.1,
scaling_coefficient: 12.0, power: 2.0`. The live reward code
(`sheeprl/.../envs/retro_dreamer.py:337-346` — the `RetroDreamerWrapper`, not the
legacy `fzero_fixed.py`) computes:
```
norm = min(speed / max_speed, 1.0)          # min(speed/500, 1.0)
reward += 12.0 * 0.1 * (norm ** 2)          # 1.2 * norm²
```
The actual RAM speed variable is a 16-bit unsigned value
(`games/FZero-Snes/data.json`: `speed = {"address": 2, "type": "<u2"}`) observed
at **66–4129** (avg 3139.6, from a real capture — `docs/pam-engine-test.md:83`).
So for **any speed ≥ 500**, `norm` is clamped to `1.0` and the reward is a flat
`1.2` — i.e. the "quadratic" shape is **dead across the entire realistic racing
band**. Driving 500 vs 4129 earns identical reward; the agent gets **zero
"go-faster" gradient** above 500. That is a real, twice-confirmable defect.

**Why escalated, not fixed.** God's rule: apply the config fix only if the
correct calibration is **determinable from evidence** (e.g. a RAM→km/h scale
factor); otherwise report options. There is **no documented scale factor** in
the repo (`data.json` has only address+type, no units; nothing maps the RAM
value to km/h). Picking the replacement `max_speed` is therefore a
**reward-shaping judgment call**, not an evidence-driven fix. Options for god:

1. **`max_speed ≈ 4129` (observed max).** Spans the full observed range; the
   quadratic then shapes continuously from ~66 to top speed. Risk: 4129 may be a
   collision/glitch peak, not a meaningful "top speed," so the ceiling could be
   too high (reward concentrated at the low end).
2. **`max_speed ≈ 3500–4000` (a sensible racing ceiling).** Keeps useful
   shaping; needs a judgment on where "fast enough" sits.
3. **Switch speed to `linear` mode** with a chosen ceiling — flatter gradient,
   less sensitive to the exact max.
4. **Capture the km/h mapping first** (correlate the RAM value against F-Zero's
   in-game km/h display), then set `max_speed` from real physics.
5. Side note (separate defect, already flagged for Jim's engine round-2):
   `assign_roles` misses `mode:`-style reward configs, so `speed` is reported as
   `context` not `rewarded` — orthogonal to this calibration.

**No code/config changed for item 5** — awaiting god's call on which option
(or whether to capture the km/h scale first).

## Item 5 resolution (god, 2026-07-14)

**Decision: keep quadratic mode, recalibrate `max_speed` 500 → 4500** (applied to
`games/FZero-Snes/training.json`). Rationale: observed racing max is 4129 in a real
1400-step capture (avg 3139.6); 4500 gives ~9% boost headroom. This is the smallest
semantic change that restores gradient — option (a) with headroom; the quadratic
shape (superlinear go-faster incentive) was a design choice and stays.

Before/after at real speeds (norm=min(speed/max_speed,1), reward=0.1+12·norm²):

| speed | old (max=500) | new (max=4500) |
|---|---|---|
| 66 | 0.31 | 0.10 |
| 500 | 12.10 ← saturated from here up | 0.25 |
| 1500 | 12.10 | 1.43 |
| 3140 (avg) | 12.10 | 5.94 |
| 4129 (obs max) | 12.10 | 10.20 |

Known side-effect (accepted): average reward magnitude at typical speeds drops vs the
saturated flat value, which rebalances speed vs pos (10.0) and health penalty — that is
inherent to un-saturating. E2E confirmation = next F-Zero training run (same
pending-verify bucket as the avg_length fix). F-Zero is parked; no run in flight.
