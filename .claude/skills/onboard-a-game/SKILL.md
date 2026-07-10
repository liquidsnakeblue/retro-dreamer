---
name: onboard-a-game
description: Use when onboarding a new retro game into Retro-Dreamer Studio — from imported ROM to first training run (RAM variables, reward config, save states)
---

# Onboard a Game

Take a freshly imported ROM to training-ready. Every step's output is verified
by a tool before moving on. Tools are HTTP endpoints on :8091 (see system
prompt); long jobs: poll /api/tools/jobs/<id>.

## Checklist (in order)

1. **Confirm the workspace**: GET /api/workspaces + GET /api/games/{id} —
   ROM present, configs scaffolded, button layout matches the console.
2. **Actions first**: edit actions.json to 4-8 USEFUL combos (movement +
   primary buttons). Every action a human would hold for seconds is a
   candidate; menu-only buttons (START/SELECT) stay OUT of the action space.
3. **A first save state**: build_state with a menu-walk plan from power-on.
   Verify each plan step's screenshot by Reading it. The state must boot into
   LIVE GAMEPLAY (not a menu). Name it descriptively (e.g. `level1`).
4. **RAM variables** (3-6 to start): the goal is a progress counter, a
   health/lives value, and an "in-play" flag.
   - ram_capture while something plays (random actions are fine pre-brain)
   - mark event steps (death, level end) from the capture's var log or by
     recording and watching
   - ram_diff (boundary mode, 2-3 captures per event) -> candidates
   - adopt candidates into data.json; verify each by a fresh capture where
     the value moves EXACTLY as the event predicts. Never guess addresses.
5. **Reward draft** in training.json — the proven recipe:
   - progress delta reward (signed, `wrap` for fixed-width counters,
     `max_delta` cap against teleports/glitches)
   - damage penalty ~1/unit; optional heal_reward strictly below the penalty
   - done on the in-play flag + destruction threshold; NO facing/direction
     rules (an F-Zero reverse-done rule strangled training for a day)
   - `warmup_steps` ~10 if the progress counter snaps at spawn
6. **Probe until green**: reward_probe over EVERY state with actions:"all".
   Green = deviation < 0.001, no fountains, dones fire. Then imagine the
   degenerate strategies (circling, idling, farming) and reason through each:
   would it pay? If unsure, probe a plan that plays the exploit.
7. **Hand off**: summarize what was wired and verified; the human starts
   training from the dashboard. Recommend `small` model for the first run
   (fast signal), scaling later.
