# Copilot Usability Test: Episode Report Engine

Date: 2026-07-14
Tool: `/home/liquidsnakeblue/retro-dreamer/scratch/episode_report.py`
Scope: 3 games — Little Mermaid (NES), Super Mario Bros (NES), F-Zero (SNES)

---

## 1. The Little Mermaid (NES) — `lm.npz`

### (a) Command

```bash
python3 /home/liquidsnakeblue/retro-dreamer/scratch/episode_report.py \
  /home/liquidsnakeblue/retro-dreamer/scratch/fixtures/lm.npz \
  /home/liquidsnakeblue/retro-dreamer/scratch/fixtures/lm_training.json
```

### (b) Diagnosis

**What happened:** The brain learned a repeatable route through 6 pages (rooms) of the level. It reached a maximum scroll depth of 493 and room position 5. It died twice — at step 477 (lives 2→1) and step 988 (lives 1→0) — both times after losing all 3 health points one at a time. Between deaths, health was restored to 3 (step 477 and 988), indicating level restarts or checkpoints.

**Where and why it fails:** The brain is stuck in a death loop. Every run follows the same pattern: advance to playerPage 6, lose health at steps ~38 steps apart (385→414→442, then 882→919→952), die, restart, repeat. The report says the death location is consistently playerPage=6 (2/2 deaths), and after step 347, no progress axis ever set a new high — 1053 of 1400 steps stalled. The `stage` variable never advanced (stayed 0), and neither `green_pearls_found` nor `red_pearls_found` gained anything. The brain learned to move through rooms but never learned what the game actually wants it to do (collect pearls, advance stages).

**What to try next:** The reward config gives 0.05 per score point and 1.0 per pearl, but the brain collected zero pearls and earned zero score — meaning the reward signal from objectives was never triggered. Either the brain hasn't discovered pearls yet, or the RAM variables for pearl collection aren't capturing the game state correctly. I'd verify that `green_pearls_found` and `red_pearls_found` actually change during manual gameplay. If they do, the training likely needs more steps with shaped rewards (e.g., bigger reward for scrolling depth, or a proximity reward for being near pearls) so the brain is incentivized to explore beyond its learned death-loop corridor.

### (c) Self-Assessment

**What was easy:** The event stream told a clear story — two death cycles, regular health decrements, looped resets. The post-mortem "STALL" and "NEVER ADVANCED" lines made the core problem obvious at a glance. The "FAILURE LOCATION" section precisely pinned where deaths happened.

**What was hard/ambiguous:**
- The report says `lives` is `resource,terminal` but the terminal condition is `lives < 0` (from training.json). The brain's lives go 2→1→0 at death, and the done condition (`less-than 0`) was never triggered — yet the brain clearly "died" (health drained, then reset). This means deaths here are life-loss restarts, not episode termination. The report lists "2 death/fail" from `_dedup_deaths` counting life losses, but the episode never actually terminated. This distinction matters: the brain survived 1400 steps but wasted them looping.
- `scrollY` reached 493 then shows "last 400" — the report doesn't explain why the last value is lower than the max beyond the generic "looped back" event.
- The report says `score` is an `objective` (rewarded, only-increases) but scored 0. It's unclear whether the score RAM variable was never touched, or whether the brain's actions never triggered scoring. The report can't tell me *why* score stayed at 0, only that it did.

**What was missing:** I wanted to know how long each life lasted in steps (life 1: steps 0–477 = 477 steps; life 2: steps 477–988 = 511 steps; life 3: steps 988–1400 = 412 steps). The engine doesn't compute per-life duration, which would help judge whether the brain is getting better or worse across lives. I also wanted the `playerPage` value at the start of each life to confirm it restarts from page 0.

---

## 2. Super Mario Bros (NES) — `mario.npz`

### (a) Command

```bash
python3 /home/liquidsnakeblue/retro-dreamer/scratch/episode_report.py \
  /home/liquidsnakeblue/retro-dreamer/scratch/fixtures/mario.npz \
  /home/liquidsnakeblue/retro-dreamer/scratch/fixtures/mario_training.json
```

### (b) Diagnosis

**What happened:** This was a short 560-step episode where the brain died twice and triggered the terminal condition (lives == 0) at step 436. The brain only managed +20 total score. It reached playerPage 6 (max) at step 382, but 178 of 560 steps stalled after that. The level timer started at 400, went as low as 361, and reset twice — suggesting the brain triggered at least two time-based resets or level transitions.

**Where and why it fails:** The brain dies very early (first life lost at step 95, only 75 steps after gaining score at step 35), then somehow reaches playerPage 6 on the second life (step 382) before dying again at step 436. The report only records one death at a specific location (playerPage=6, 1/2 deaths) — the first death at step 95 had no progress-axis location because playerPage hadn't advanced yet. The brain barely scored (+20 total) and the level variables (`levelHi`, `levelLo`) stayed constant — meaning it never left the first level. The time resets ×2 is suspicious: the timer only counted down 39 ticks total (400→361), which suggests the brain was moving slowly enough that time wasn't the threat, and the resets came from dying and respawning.

**What to try next:** The reward config gives 0.1 per playerX movement (wrapped) and 0.001 per score point — meaning positional movement is rewarded 100× more than score. This heavily incentivizes the brain to move right at all costs, without caring about jumping, collecting, or surviving. The brain's behavior matches this: it reaches page 6 (far right) but dies repeatedly and scores almost nothing. I'd increase the `lives` penalty from 50 or add a health/invincibility variable to the reward, and reduce the playerX reward weight so the brain isn't penalized enough for dying to learn avoidance, but rewarded enough for moving to just run into walls.

### (c) Self-Assessment

**What was easy:** The event stream was compact (6 events) and easy to follow. The post-mortem clearly showed: died twice, stalled after page 6, scored 20. The timer summary (start/low/end/resets) gave good temporal context.

**What was hard/ambiguous:**
- The report lists "2 death/fail" and "done-condition fired ×1". The first death (step 95, lives 2→1) was recorded as a loss event, but the second death (step 436, lives 1→0) triggered the terminal condition. The "FAILURE LOCATION" only shows playerPage=6 (1/2) — meaning the first death's location on the progress axis was not at page 6, but the report doesn't say what page it was at. The playerPage variable wasn't classified as constant, but the first death happened before page 6 was reached, so its location would have been lower. The report doesn't tell me what playerPage was at step 95.
- `levelHi` and `levelLo` are both `context [constant this episode]` — this confirms the brain never left World 1-1, but the report doesn't say what level values they held.
- `xscrollHi` and `xscrollLo` vary but have no declared role. These would be very useful for understanding Mario's true horizontal position, but without knowing their encoding (they're high/low bytes of a 16-bit value), the engine can't interpret them.

**What was missing:** The progress-axis value at each death event would be extremely useful — right now "FAILURE LOCATION" only shows the modal value across deaths, not the value at each specific death. I also wanted to see `playerX` cyclic trend (it's classified as `cyclic,progress` but the post-mortem only shows relative stats for cyclic signals with ≤64 distinct values — `playerX` likely has >64 distinct values here, so it falls through without a post-mortem summary).

---

## 3. F-Zero (SNES) — `fzero.npz`

### (a) Command

```bash
python3 /home/liquidsnakeblue/retro-dreamer/scratch/episode_report.py \
  /home/liquidsnakeblue/retro-dreamer/scratch/fixtures/fzero.npz \
  /home/liquidsnakeblue/retro-dreamer/scratch/fixtures/fzero_training.json
```

### (b) Diagnosis

**What happened:** Over 1400 steps, the brain survived without dying. Health started at 2048, dropped to a low of 468, and recovered back to 2018 by the end. There were 9 significant damage events (steps 20 through 819) followed by 10 health regain events (steps 866 through 1099). The brain maintained forward position throughout — the `pos` signal shows a net movement of +202 over the episode with a consistent trend of 196.9→200.4. Speed averaged 3139.6, trending upward from 3160.5 to 3443.9.

**Where and why it fails:** The brain learned a two-phase pattern: it crashes a lot in the first half (9 damage events, health falling from 2048 down to 468 between steps 20–819), then something changes around step 866 and it enters a recovery phase (10 regain events, health climbing back to 2018). The damage events are very regular — each hit is roughly 160–242 health, suggesting the brain is hitting the same obstacles repeatedly. The recovery phase is equally regular at +160 per tick, suggesting the brain found a safe zone or a health pickup location. The done conditions (health < 100 or race_on == 0) were never triggered, so the brain stayed alive and the race stayed on.

However, there are no declared objectives — the reward config only has `pos` (cyclic position with wrap 65536), `health` (penalty), and `speed` (quadratic reward). The brain's net position gain of +202 over 1400 steps is small relative to the 65536 wrap, meaning it's barely gaining ground around the track. The brain is surviving but not racing effectively.

**What to try next:** The brain clearly learned to recover health after step 866, but its forward progress is minimal. The speed reward uses a quadratic mode with `max_speed: 500` and `scaling_coefficient: 12.0` — but the actual speed values average 3139.6 with a max of 4129, which is far above the configured `max_speed` of 500. This mismatch means the speed reward is likely saturated or producing unexpected gradients. I'd verify what the `speed` RAM variable actually represents (it may be in different units than expected), fix the `max_speed` parameter, and consider adding a lap counter or race-position variable as a declared objective so the brain has a target beyond just "don't crash."

### (c) Self-Assessment

**What was easy:** The continuous-resource tracking for `health` was excellent — it correctly identified significant movement events at a proper threshold rather than noise-level changes. The `pos` cyclic signal summary (relative min/max/net/trend) gave a compact picture of track progress. The verdict "survived with active pos/speed signals" was accurate.

**What was hard/ambiguous:**
- The `speed` variable is classified as `context [varies, no declared role]` despite having a complex reward config entry (`mode: quadratic`). The `assign_roles` function only looks at variables declared in the done condition, or with simple `reward`/`penalty` keys — it doesn't recognize `mode: quadratic` reward configs. So speed's role is under-reported. The post-mortem does show it as a "signal · speed" with trend stats, but it's labeled as `context` rather than a rewarded variable.
- Similarly, `race_on` is `terminal` (in done-condition) but its value was never printed. I don't know if it stayed at 1 (race active) or something else.
- `x` and `y` are both `context [varies, no declared role]` — these are likely screen-space positions. Without knowing their meaning, the report can't use them, but they might contain useful information about where on the track the brain is.
- The health regain events are very regular (+160 exactly), which looks artificial. The report flags them as events but doesn't comment on the regularity — a human reader would wonder if this is a fixed-rate healing mechanic or a data artifact.

**What was missing:** I wanted to know whether the damage and recovery phases correlate with track position. The health drops happen at steps 20–819 and recovery at steps 866–1099. If I could see `pos` at each of those timestamps, I'd know if the brain is hitting a dangerous track section then finding a safe one. The engine doesn't cross-reference health events with position, so I can't tell if the brain is learning to avoid hazards or just waiting out a dangerous zone. I also wanted a summary of how many steps the brain spent above/below the done threshold (health < 100) to understand how close it came to actually dying — the minimum was 468, which is well above 100, so it was never in danger.

---

## Summary

| Game | Steps | Deaths | Key Finding | Tool Helped With |
|------|-------|--------|-------------|-----------------|
| Little Mermaid | 1400 | 2 (lives lost, no terminal) | Death loop at page 6, never collects pearls, stalled after step 347 | Event stream showed the loop clearly; stall detection was spot-on |
| Super Mario Bros | 560 | 2 (terminal triggered) | Barely scores, reaches page 6 but dies, never leaves level 1-1 | Timer summary gave good context; compact event list was easy to parse |
| F-Zero | 1400 | 0 | Survives, learns recovery, but minimal forward progress. Speed units mismatch. | Continuous-resource tracking for health was excellent; cyclic pos summary was useful |
