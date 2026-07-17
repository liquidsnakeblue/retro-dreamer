# Episode-report hardening results (2026-07-14)

All three acceptance fixtures ran with system `python3` after a successful
`python3 -m py_compile scratch/episode_report.py` check.

## Defect fixes

1. Continuous resources now use a shape-derived cumulative significance threshold, retaining discrete per-point damage while collapsing gauge ticks.
2. Countdown timers are detected from downward-transition dominance, decrement consistency, and cadence regularity, then excluded from resource damage.
3. Undeclared counters with more completed climbs than `span + 1` are downgraded to contextual oscillators instead of progress axes.
4. Small terminal resource counters are identified from their pre-terminal shape; post-floor underflow is ignored and coincident terminal firings are deduplicated from death totals.
5. Configured continuous context signals and small wrapped signals now receive generic distribution, trend, and relative-progress summaries.

## LM regression fixture

Command: `python3 scratch/episode_report.py scratch/fixtures/lm.npz scratch/fixtures/lm_training.json`

```text
====================================================================
EPISODE REPORT · lm.npz · 1400 steps
====================================================================

ROLES (derived from config + trajectory, no per-game code):
  forks_found          context                [constant this episode]
  playerY              context                [varies, no declared role]
  playerX              cyclic,progress        [rewarded, wrap=256 (cyclic position)]
  stage                milestone              [reward=100 (major)]
  green_pearls_found   objective              [rewarded, only-increases (accumulator)]
  red_pearls_found     objective              [rewarded, only-increases (accumulator)]
  score                objective              [rewarded, only-increases (accumulator)]
  roomPos              progress               [counter climbs×3 (unrewarded)]
  playerPage           progress               [counter climbs×2 (unrewarded)]
  scrollY              progress               [rewarded positional]
  health               resource               [penalty=5]
  lives                resource,terminal      [in done-condition, penalty=50]

EVENT STREAM (21 events):
  step   286  milestone      roomPos reached 5 (max)
  step   319  milestone      playerPage reached 6 (max)
  step   327  reset/loop     roomPos 5->0 (looped back)
  step   347  milestone      scrollY reached 493 (max)
  step   385  loss           health 3->2 (-1)
  step   414  loss           health 2->1 (-1)
  step   442  loss           health 1->0 (-1)
  step   477  regain         health 0->3
  step   477  loss           lives 2->1 (-1)
  step   482  reset/loop     playerPage 6->0 (looped back)
  step   482  reset/loop     scrollY 422->0 (looped back)
  step   825  reset/loop     roomPos 5->0 (looped back)
  step   882  loss           health 3->2 (-1)
  step   919  loss           health 2->1 (-1)
  step   952  loss           health 1->0 (-1)
  step   988  regain         health 0->3
  step   988  loss           lives 1->0 (-1)
  step   993  reset/loop     playerPage 6->0 (looped back)
  step   993  reset/loop     scrollY 418->0 (looped back)
  step  1336  reset/loop     roomPos 5->0 (looped back)
  step  1394  loss           health 3->2 (-1)

POST-MORTEM:
OUTCOME: 2 death/fail · 7 damage events over 1400 steps.
  reach · playerPage: max 6 first hit @step 319 (last 6)
  reach · roomPos: max 5 first hit @step 286 (last 1)
  reach · scrollY: max 493 first hit @step 347 (last 400)
  reach · stage: NEVER ADVANCED (stayed 0)
  objective · green_pearls_found: 0 gained
  objective · red_pearls_found: 0 gained
  objective · score: 0 gained
  FAILURE LOCATION (at deaths): playerPage=6 (2/2); roomPos=1 (2/2); scrollY=422 (1/2)
  STALL: no progress axis set a new high after step 347 (1053 of 1400 steps stalled).
VERDICT (skeleton): died 2×, looping (×3), stalled after early progress, never advanced stage, collected nothing.
```

## Mario fixture

Command: `python3 scratch/episode_report.py scratch/fixtures/mario.npz scratch/fixtures/mario_training.json`

```text
====================================================================
EPISODE REPORT · mario.npz · 560 steps
====================================================================

ROLES (derived from config + trajectory, no per-game code):
  coins                context                [varies, no declared role]
  levelHi              context                [constant this episode]
  levelLo              context                [constant this episode]
  scrolling            context                [oscillator climbs×29 within span 5]
  xscrollHi            context                [varies, no declared role]
  xscrollLo            context                [varies, no declared role]
  playerX              cyclic,progress        [rewarded, wrap=256 (cyclic position)]
  score                objective              [rewarded, only-increases (accumulator)]
  playerPage           progress               [counter climbs×1 (unrewarded)]
  lives                resource,terminal      [in done-condition, penalty=50]
  time                 timer                  [regular near-monotonic countdown]

EVENT STREAM (6 events):
  step    35  objective+     score +20 (=20)
  step    95  loss           lives 2->1 (-1)
  step   382  milestone      playerPage reached 6 (max)
  step   436  loss           lives 1->0 (-1)
  step   436  TERMINAL       done: lives equal 0
  step   437  reset/loop     playerPage 6->0 (looped back)

POST-MORTEM:
OUTCOME: 2 death/fail · done-condition fired ×1 over 560 steps.
  reach · playerPage: max 6 first hit @step 382 (last 6)
  objective · score: 20 gained
  timer · time: start 400, low 361, end 397; resets ×2
  FAILURE LOCATION (at deaths): playerPage=6 (1/2)
  STALL: no progress axis set a new high after step 382 (178 of 560 steps stalled).
VERDICT (skeleton): died 2×, stalled after early progress.
```

## F-Zero fixture

Command: `python3 scratch/episode_report.py scratch/fixtures/fzero.npz scratch/fixtures/fzero_training.json`

```text
====================================================================
EPISODE REPORT · fzero.npz · 1400 steps
====================================================================

ROLES (derived from config + trajectory, no per-game code):
  reverse              context                [oscillator climbs×10 within span 1]
  speed                context                [varies, no declared role]
  y                    context                [varies, no declared role]
  x                    context                [varies, no declared role]
  pos                  cyclic,progress        [rewarded, wrap=65536 (cyclic position)]
  health               resource,terminal      [in done-condition, penalty=1.0]
  race_on              terminal               [in done-condition]

EVENT STREAM (19 events):
  step    20  loss           health 2048->1887 (-161, significant)
  step    78  regain         health 1887->2048 (+161, significant)
  step   350  loss           health 2048->1887 (-161, significant)
  step   421  loss           health 1887->1670 (-217, significant)
  step   484  loss           health 1670->1504 (-166, significant)
  step   581  loss           health 1504->1338 (-166, significant)
  step   611  loss           health 1338->1106 (-232, significant)
  step   695  loss           health 1106->864 (-242, significant)
  step   765  loss           health 864->693 (-171, significant)
  step   819  loss           health 693->528 (-165, significant)
  step   866  regain         health 528->696 (+168, significant)
  step   876  regain         health 696->856 (+160, significant)
  step   886  regain         health 856->1016 (+160, significant)
  step   965  regain         health 1016->1176 (+160, significant)
  step   975  regain         health 1176->1336 (+160, significant)
  step   985  regain         health 1336->1496 (+160, significant)
  step  1079  regain         health 1496->1668 (+172, significant)
  step  1089  regain         health 1668->1828 (+160, significant)
  step  1099  regain         health 1828->1988 (+160, significant)

POST-MORTEM:
OUTCOME: survived the window (no death/terminal) · 9 significant damage events over 1400 steps.
  objective · none declared
  resource · health: min 468, mean 1566.5, max 2048, final 2018; trend 1972.5→1971.0
  signal · speed: min 66, mean 3139.6, max 4129; trend 3160.5→3443.9
  cyclic signal · pos: relative min 0, mean 201.7, max 208; trend 196.9→200.4, net +202
VERDICT (skeleton): survived with active pos/speed signals.
```

## Round 2

1. Mode-style or future reward-config entries that lack a scalar delta-reward key now receive a generic `rewarded` role, with continuous signals retaining their distribution/trend summary.
2. Every deduplicated death now has a raw non-cyclic progress-axis snapshot, followed by the existing modal failure-location summary.
3. Deduplicated death boundaries now produce half-open per-life rows with duration, raw start coordinates, and within-life maxima; any captured post-terminal tail is labeled separately.
4. Every configured done input now reports its final observation, operator-directed closest approach, threshold, and whether/when the condition was met.
5. Significant continuous-resource loss events now carry a compact position snapshot, preferring non-cyclic progress axes and otherwise using configured unwrapped cyclic progress.

### LM round-2 fixture

Command: `python3 scratch/episode_report.py scratch/fixtures/lm.npz scratch/fixtures/lm_training.json`

```text
====================================================================
EPISODE REPORT · lm.npz · 1400 steps
====================================================================

ROLES (derived from config + trajectory, no per-game code):
  forks_found          context                [constant this episode]
  playerY              context                [varies, no declared role]
  playerX              cyclic,progress        [rewarded, wrap=256 (cyclic position)]
  stage                milestone              [reward=100 (major)]
  green_pearls_found   objective              [rewarded, only-increases (accumulator)]
  red_pearls_found     objective              [rewarded, only-increases (accumulator)]
  score                objective              [rewarded, only-increases (accumulator)]
  roomPos              progress               [counter climbs×3 (unrewarded)]
  playerPage           progress               [counter climbs×2 (unrewarded)]
  scrollY              progress               [rewarded positional]
  health               resource               [penalty=5]
  lives                resource,terminal      [in done-condition, penalty=50]

EVENT STREAM (21 events):
  step   286  milestone      roomPos reached 5 (max)
  step   319  milestone      playerPage reached 6 (max)
  step   327  reset/loop     roomPos 5->0 (looped back)
  step   347  milestone      scrollY reached 493 (max)
  step   385  loss           health 3->2 (-1)
  step   414  loss           health 2->1 (-1)
  step   442  loss           health 1->0 (-1)
  step   477  regain         health 0->3
  step   477  loss           lives 2->1 (-1)
  step   482  reset/loop     playerPage 6->0 (looped back)
  step   482  reset/loop     scrollY 422->0 (looped back)
  step   825  reset/loop     roomPos 5->0 (looped back)
  step   882  loss           health 3->2 (-1)
  step   919  loss           health 2->1 (-1)
  step   952  loss           health 1->0 (-1)
  step   988  regain         health 0->3
  step   988  loss           lives 1->0 (-1)
  step   993  reset/loop     playerPage 6->0 (looped back)
  step   993  reset/loop     scrollY 418->0 (looped back)
  step  1336  reset/loop     roomPos 5->0 (looped back)
  step  1394  loss           health 3->2 (-1)

POST-MORTEM:
OUTCOME: 2 death/fail · 7 damage events over 1400 steps.
  reach · playerPage: max 6 first hit @step 319 (last 6)
  reach · roomPos: max 5 first hit @step 286 (last 1)
  reach · scrollY: max 493 first hit @step 347 (last 400)
  reach · stage: NEVER ADVANCED (stayed 0)
  objective · green_pearls_found: 0 gained
  objective · red_pearls_found: 0 gained
  objective · score: 0 gained
  done margin · lives: final 0; low 0 vs < 0 — not met (strict boundary; margin 0)
  death @477: playerPage=6, roomPos=1, scrollY=422
  death @988: playerPage=6, roomPos=1, scrollY=418
  LIFE SEGMENTS:
    life 1 · steps 0–477 (477) · start playerPage=0, roomPos=0, scrollY=0; max playerPage=6, roomPos=5, scrollY=493
    life 2 · steps 477–988 (511) · start playerPage=6, roomPos=1, scrollY=422; max playerPage=6, roomPos=5, scrollY=493
    life 3 · steps 988–1400 (412) · start playerPage=6, roomPos=1, scrollY=418; max playerPage=6, roomPos=5, scrollY=493
  FAILURE LOCATION (at deaths): playerPage=6 (2/2); roomPos=1 (2/2); scrollY=422 (1/2)
  STALL: no progress axis set a new high after step 347 (1053 of 1400 steps stalled).
VERDICT (skeleton): died 2×, looping (×3), stalled after early progress, never advanced stage, collected nothing.
```

### Mario round-2 fixture

Command: `python3 scratch/episode_report.py scratch/fixtures/mario.npz scratch/fixtures/mario_training.json`

```text
====================================================================
EPISODE REPORT · mario.npz · 560 steps
====================================================================

ROLES (derived from config + trajectory, no per-game code):
  coins                context                [varies, no declared role]
  levelHi              context                [constant this episode]
  levelLo              context                [constant this episode]
  scrolling            context                [oscillator climbs×29 within span 5]
  xscrollHi            context                [varies, no declared role]
  xscrollLo            context                [varies, no declared role]
  playerX              cyclic,progress        [rewarded, wrap=256 (cyclic position)]
  score                objective              [rewarded, only-increases (accumulator)]
  playerPage           progress               [counter climbs×1 (unrewarded)]
  lives                resource,terminal      [in done-condition, penalty=50]
  time                 timer                  [regular near-monotonic countdown]

EVENT STREAM (6 events):
  step    35  objective+     score +20 (=20)
  step    95  loss           lives 2->1 (-1)
  step   382  milestone      playerPage reached 6 (max)
  step   436  loss           lives 1->0 (-1)
  step   436  TERMINAL       done: lives equal 0
  step   437  reset/loop     playerPage 6->0 (looped back)

POST-MORTEM:
OUTCOME: 2 death/fail · done-condition fired ×1 over 560 steps.
  reach · playerPage: max 6 first hit @step 382 (last 6)
  objective · score: 20 gained
  timer · time: start 400, low 361, end 397; resets ×2
  done margin · lives: final -1; nearest 0 vs == 0 — met @step 436
  death @95: playerPage=0
  death @436: playerPage=6
  LIFE SEGMENTS:
    life 1 · steps 0–95 (95) · start playerPage=0; max playerPage=1
    life 2 · steps 95–436 (341) · start playerPage=0; max playerPage=6
    post-terminal tail · steps 436–560 (124) captured after done
  FAILURE LOCATION (at deaths): playerPage=6 (1/2)
  STALL: no progress axis set a new high after step 382 (178 of 560 steps stalled).
VERDICT (skeleton): died 2×, stalled after early progress.
```

### F-Zero round-2 fixture

Command: `python3 scratch/episode_report.py scratch/fixtures/fzero.npz scratch/fixtures/fzero_training.json`

```text
====================================================================
EPISODE REPORT · fzero.npz · 1400 steps
====================================================================

ROLES (derived from config + trajectory, no per-game code):
  reverse              context                [oscillator climbs×10 within span 1]
  y                    context                [varies, no declared role]
  x                    context                [varies, no declared role]
  pos                  cyclic,progress        [rewarded, wrap=65536 (cyclic position)]
  health               resource,terminal      [in done-condition, penalty=1.0]
  speed                rewarded               [mode=quadratic, continuous signal]
  race_on              terminal               [in done-condition]

EVENT STREAM (19 events):
  step    20  loss           health 2048->1887 (-161, significant) @ pos=+198 rel
  step    78  regain         health 1887->2048 (+161, significant)
  step   350  loss           health 2048->1887 (-161, significant) @ pos=+206 rel
  step   421  loss           health 1887->1670 (-217, significant) @ pos=+204 rel
  step   484  loss           health 1670->1504 (-166, significant) @ pos=+204 rel
  step   581  loss           health 1504->1338 (-166, significant) @ pos=+204 rel
  step   611  loss           health 1338->1106 (-232, significant) @ pos=+207 rel
  step   695  loss           health 1106->864 (-242, significant) @ pos=+208 rel
  step   765  loss           health 864->693 (-171, significant) @ pos=+205 rel
  step   819  loss           health 693->528 (-165, significant) @ pos=+204 rel
  step   866  regain         health 528->696 (+168, significant)
  step   876  regain         health 696->856 (+160, significant)
  step   886  regain         health 856->1016 (+160, significant)
  step   965  regain         health 1016->1176 (+160, significant)
  step   975  regain         health 1176->1336 (+160, significant)
  step   985  regain         health 1336->1496 (+160, significant)
  step  1079  regain         health 1496->1668 (+172, significant)
  step  1089  regain         health 1668->1828 (+160, significant)
  step  1099  regain         health 1828->1988 (+160, significant)

POST-MORTEM:
OUTCOME: survived the window (no death/terminal) · 9 significant damage events over 1400 steps.
  objective · none declared
  resource · health: min 468, mean 1566.5, max 2048, final 2018; trend 1972.5→1971.0
  signal · speed: min 66, mean 3139.6, max 4129; trend 3160.5→3443.9
  cyclic signal · pos: relative min 0, mean 201.7, max 208; trend 196.9→200.4, net +202
  done margin · health: final 2018; low 468 vs < 100 — not met (margin 368)
  done margin · race_on: final 1; nearest 1 vs == 0 — not met (distance 1)
VERDICT (skeleton): survived with active pos/speed signals.
```
