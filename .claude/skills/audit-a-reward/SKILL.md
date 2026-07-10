---
name: audit-a-reward
description: Use when reviewing or changing a game's training.json reward/done config in Retro-Dreamer Studio — hunt exploits before they poison a replay buffer
---

# Audit a Reward

A wrong reward trains confidently toward garbage. This audit found (in one
project): a 1000:1 silently-added base reward, a +653k/lap wrap fountain, an
unearned +1980 spawn lump, and a done rule that killed 4 of 5 actions in 1.5s.
Assume this config has one too.

## The audit

1. **Read the actual config** (GET /api/games/{id}/config/training.json) and
   data.json. Map every reward term to the RAM variable it reads.
2. **Counter mechanics**: for every delta-rewarded variable ask — is it
   fixed-width (needs `wrap`)? does it snap/park at spawn or transitions
   (needs `warmup_steps` / is the snap load-bearing signal)? can it teleport
   (needs `max_delta`)?
3. **Asymmetry check**: is backward/negative movement priced (delta:"signed")?
   is healing paid at or above the damage penalty (farmable — heal_reward must
   stay strictly below penalty)?
4. **Done rules**: does each rule fire on states the agent MUST visit while
   learning (e.g. facing backward)? A done that punishes exploration strangles
   training. Prefer game-truth flags (in-play byte) + destruction thresholds.
5. **Probe**: reward_probe, all states, actions:"all", 400+ steps. Green =
   deviation <0.001 everywhere, no |step reward| spikes, dones fire with the
   expected reason.
6. **Degenerate-strategy table**: for circling, reversing over a
   wrap point, idling at max speed bonus, damage-heal loops, and parking on
   pay zones — write down the per-step net. Every one must be NEGATIVE
   against honest play. If any is positive or near-zero, redesign before it
   trains.
7. **Change protocol**: reward changes require an env restart to take effect
   (configs read at env creation) and KEEP the replay buffer (the reward head
   recalibrates; expect a brief reward_loss blip). Never pair a reward change
   with a buffer reset.
