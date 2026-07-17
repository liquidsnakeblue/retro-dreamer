# LM Done-Condition Verification

Date: 2026-07-14
Origin: Capability-test finding — `lives` is typed `|i1` (signed byte @176), done is `lives < 0`
Question: Does the done condition fire at game over, or does lives stay at 0 and never trigger?

---

## (a) Method

1. **Identified the lives variable**: Address 176, type `|i1` (signed byte) per `data.json`.
2. **Ran two custom probe scripts** against the live emulator (stable-retro, FCEUX core) with `sg render` group access:
   - `scratch/lm_lives_probe_v2.py` — randomized actions across 8 states, 10000 steps each, tracking `lookup_all()['lives']` (stable-retro's signed interpretation) and `memory.blocks[0][176]` (raw unsigned byte).
   - `scratch/lm_gameover_probe.py` — after lives hit 0, continued for 2000 additional post-death steps sampling every 100 steps, watching for the byte to go negative.
3. **States tested**: Level1-gauntlet (1 life, died at step 9395), Level1-seam-edge (1 life, died at step 7284), plus Level1-corridor-deep, Level1-descent-bottom, Level1-summit, Level1-descent-top, Level1-corridor-entry, Level1 (title screen).
4. **Read-only**: No config edits, no training actions. Only raw emulator steps.

## (b) Raw Evidence

### Configuration

**data.json lives entry:**
```json
"lives": { "address": 176, "type": "|i1" }
```

**training.json done condition:**
```json
"done": { "variables": { "lives": { "op": "less-than", "reference": 0 } } }
```

**stable-retro scenario.json done condition:**
```json
"done": { "variables": { "lives": { "op": "negative" } } }
```

### Observed Byte Values at and After Final Death

**Level1-gauntlet (1 life start):**

| Step | Event | lookup (signed) | raw byte | raw hex |
|------|-------|-----------------|----------|---------|
| 0 | Start | 1 | 1 | 0x01 |
| 9395 | Death | **0** | **0** | **0x00** |
| 9895 | +500 post-death | 0 | 0 | 0x00 |
| 10395 | +1000 post-death | 0 | 0 | 0x00 |
| 10895 | +1500 post-death | 0 | 0 | 0x00 |
| 11395 | +2000 post-death | 0 | 0 | 0x00 |

**Level1-seam-edge (1 life start):**

| Step | Event | lookup (signed) | raw byte | raw hex |
|------|-------|-----------------|----------|---------|
| 0 | Start | 1 | 1 | 0x01 |
| 7284 | Death | **0** | **0** | **0x00** |
| 7784 | +500 post-death | 0 | 0 | 0x00 |
| 8284 | +1000 post-death | 0 | 0 | 0x00 |
| 8784 | +1500 post-death | 0 | 0 | 0x00 |
| 9284 | +2000 post-death | 0 | 0 | 0x00 |

**Cross-state summary (all 8 states):**
- Level1-gauntlet: 1 death at step 9395, done_flag never fired
- Level1-seam-edge: 1 death at step 7284, done_flag never fired
- Level1-summit: 1 life lost (2→1), no game over, done_flag never fired
- All other states: 0 deaths in 10000 steps, done_flag never fired

**Key observation:** After the final death, the raw byte at address 176 is `0x00` and stays `0x00` for at least 2000 subsequent emulator frames (≈33 seconds of gameplay). It never transitions to `0xFF` (-1 in signed interpretation).

### Wrapper Behavior Confirmation

The RetroDreamerWrapper `_check_done()` evaluates:
```python
if op == "less-than" and val < ref:  # ref = 0
    return True
```
Since `val` (lives) is 0 and `ref` is 0, the condition `0 < 0` is **False**. The episode does NOT terminate.

Similarly, stable-retro's scenario `op: "negative"` checks for `val < 0`, which is also **False** when lives = 0.

## (c) VERDICT: **NEVER-FIRES**

The done condition `lives < 0` **does not fire at game over**. The lives byte transitions from 1 to 0 at the moment of final death and stays at 0. The NES game (Little Mermaid) does not write a negative value to address 176 — it treats the byte as an unsigned counter that stops at zero.

**Evidence basis:** Direct byte observation at address 176, confirmed across 2 independent death events, verified for 2000 post-death steps (≈33 seconds). Neither stable-retro's native done flag nor the wrapper's `_check_done()` fired.

## (d) Proposed Fix (config-level, NOT applied)

**Problem:** The lives variable at address 176 is typed `|i1` (signed byte) but the game treats it as an unsigned counter (0–255). The `lives < 0` condition requires the byte to go to -1 (0xFF), which this game never does.

**Proposed fix — Change the done condition from `lives < 0` to `lives == 0`:**

```json
"done": {
  "variables": {
    "lives": {
      "op": "equal",
      "reference": 0
    }
  }
}
```

**Rationale:**
- The lives byte cleanly counts down: 2 → 1 → 0, with 0 meaning "game over, no lives remaining."
- This matches how Super Mario Bros handles lives in its training.json (`"op": "equal", "reference": 0`), which works correctly.
- No ambiguity: `lives == 0` fires exactly when the last life is lost. The byte is observed to be stable at 0 (not bouncing between 0 and 1 on death animation frames).

**Alternative (if keeping signed type is desired):** Change the data.json type from `|i1` to `|u1` (unsigned byte) for consistency, though this doesn't affect the done condition fix — it just removes the misleading type declaration. The current `|i1` type suggests the game might go negative, which it doesn't.

**Risk assessment:** Low risk. The `lives == 0` pattern is well-tested on Mario and fires correctly. The LM lives byte is observed to be stable at 0 post-death (no flicker).

**What this fixes:**
- Episodes terminate at game over, preventing replay-buffer pollution with post-death transitions where the game is effectively stuck in a game-over loop
- The wrapper's terminal signal flows correctly to DreamerV3, enabling proper episode boundary handling
- Training reward attribution is clean (no reward leakage from post-death steps)

---

## Post-fix validation

Date: 2026-07-14
Fix applied: `op: "less-than"` → `op: "equal"`, `reference: 0`
Harness: `scratch/lm_fix_validation_v3.py` (RetroDreamerWrapper + raw retro, `sg render` access)

### (a) Done-flag now fires at final death

**Level1-gauntlet (1 life start):**

| Step | Event | lives | raw byte | raw hex | `_check_done` |
|------|-------|-------|----------|---------|---------------|
| 0 | Start | 1 | 1 | 0x01 | False |
| 9395 | Death | **0** | **0** | **0x00** | **True** ✓ |

**Level1-seam-edge (1 life start):**

| Step | Event | lives | raw byte | raw hex | `_check_done` |
|------|-------|-------|----------|---------|---------------|
| 0 | Start | 1 | 1 | 0x01 | False |
| 7284 | Death | **0** | **0** | **0x00** | **True** ✓ |

Both states: done config reloaded fresh (`op=equal, ref=0`). `_check_done()` returns `True` at lives=0, matching the new condition exactly.

### (b) No insta-done at episode start (all 8 states)

| State | lives at step 1 | raw byte | terminated within 5 steps |
|-------|-----------------|----------|---------------------------|
| Level1-corridor-deep | 1 | 0x01 | No |
| Level1-gauntlet | 1 | 0x01 | No |
| Level1-descent-bottom | 1 | 0x01 | No |
| Level1-seam-edge | 1 | 0x01 | No |
| Level1-summit | 2 | 0x02 | No |
| Level1-descent-top | 1 | 0x01 | No |
| Level1-corridor-entry | 1 | 0x01 | No |
| Level1 | 2 | 0x02 | No |

All 8 states start with lives ≥ 1. No state insta-terminates.

### (c) Wrapper propagation — direct `_check_done()` unit test

| lives value | `_check_done` returns | Expected | Result |
|-------------|----------------------|----------|--------|
| 3 | False | False | ✓ |
| 2 | False | False | ✓ |
| 1 | False | False | ✓ |
| 0 | **True** | **True** | ✓ |
| -1 | False | False | ✓ |

Wrapper's `terminated` flag propagates cleanly: `lives == 0` triggers termination; all other values do not. The `op: equal` semantics are correct — only exact match fires.

### OVERALL: **PASS**

All three validation gates passed:
1. ✅ No insta-done on any of 8 states (all start with lives ≥ 1)
2. ✅ Done-flag fires at lives=0 on both death-causing states (Level1-gauntlet step 9395, Level1-seam-edge step 7284)
3. ✅ Wrapper propagation correct across all 5 test values (lives 0–3 plus -1)
