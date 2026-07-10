---
name: diagnose-a-run
description: Use when a Retro-Dreamer training run looks wrong — returns flat/falling, weird behavior on video, suspected regression or reward exploit
---

# Diagnose a Run

Graphs lie by omission and small samples wobble. Diagnose with data, decide
with video, and never call regression on fewer than ~10 episodes.

## Triage order

1. **Status + trend**: GET /api/training/status; recent episode returns from
   /api/training/logs (lines carry `track=<state>` — build a per-state
   scoreboard). Bimodal returns (finishes vs deaths) are HEALTHY mid-learning;
   a shrinking death-mode is progress even when the average is flat.
2. **Tight clusters are clues**: deaths at nearly identical returns = a
   reproducible same-spot death, not noise. Find it on video.
3. **Watch before theorizing**: record_episode on the suspect state; extract
   frames; Read them. The one time this project's graphs screamed
   "regression," video showed the agent learning pit-stops — the dip was a
   new skill installing, and "fixing" it would have destroyed it.
4. **Reward-side checks**: reward_loss spiking = the world model is surprised
   by payouts (config changed? exploit found?). Run audit-a-reward if the
   config is at all suspect.
5. **Known non-bugs**: post-reward-change reward_loss blip (recalibration);
   average dip right after adding new states/tasks (diversity tuition);
   entropy wiggle early; episode counters resetting to 0 after a resume.
6. **Escalate honestly**: two rounds of tools without a supported hypothesis
   -> report what was ruled out and recommend the frontier model. A wrong
   confident diagnosis costs a day; "unresolved, here's the evidence" costs
   nothing.

## Hard rule

Never propose stopping, restarting, or rolling back a live run — diagnose and
report; the human decides interventions.
