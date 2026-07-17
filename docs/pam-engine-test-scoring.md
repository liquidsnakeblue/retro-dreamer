# Phase 2 scoring — Pam's episode-report usability test (2026-07-14)

Scored by god against verified ground truth (fixtures ground truth in
`episode-report-hardening.md`; Pam was contractually barred from reading it and
her text shows no leakage). Her run: `docs/pam-engine-test.md`.

## Verdict: PASS — first attempt

The anti-confabulation contract held. Every factual claim traces to a report
line or the allowed `training.json`; where the report was silent she wrote the
gap explicitly ("the report doesn't tell me what playerPage was at step 95")
instead of inferring. This is exactly the behavior the copilot needs.

| Game | Diagnosis vs ground truth | Notes |
|---|---|---|
| LM | ✅ Exact | Death loop at page 6, stall after 347, never advanced stage, zero objectives — matches observed truth. Bonus: correctly distinguished life-loss restarts from episode termination (lives<0 never fired at lives=0). |
| Mario | ✅ Correct | 2 deaths, terminal @436, page 6 reach, +20 score, never left the level. Reward-imbalance read (playerX 0.1 vs score 0.001 → move-right-at-all-costs) is grounded and plausible-labeled. |
| F-Zero | ✅ Correct + found a real defect | Survived, damage-then-recovery phases, minimal net track progress. Caught that `assign_roles` misses `mode: quadratic` reward configs (VERIFIED: speed has no simple `reward` key → roled `context`). Also flagged max_speed=500 vs actual RAM speed 66–4129 units mismatch — a real studio config question, separate from the engine. |

## Her improvement asks → round-2 engine work (dispatched to Jim)

1. **Mode-based reward recognition** (defect, verified): `assign_roles` only
   checks `reward`/`penalty` keys; any var rewarded via `mode:`-style config
   (F-Zero speed quadratic) silently loses its role.
2. **Per-death axis snapshot**: each death event should carry the progress-axis
   coordinates at that step (not just the modal FAILURE LOCATION).
3. **Per-life segments**: duration + start position + max reach per life —
   answers "is it getting better across lives" (LM lives: 477/511/412 steps).
4. **Done-margin line**: for each done-condition var, final value + closest
   approach to the threshold (covers unprinted race_on AND "how close did it
   come to dying": health min 468 vs ref 100).
5. **Damage-location cross-ref**: annotate significant resource losses with the
   position value at that step (F-Zero: is damage clustered on a track section?).

Deferred to Phase 4 (decode-layer, not report engine): hi/lo byte-pair fusion
(xscrollHi/Lo → one 16-bit position).

## What this means for the build order

Phase 2 goal ("Pam tests until she can use it correctly") is met — one clean
pass, correct usage on two games she'd never diagnosed. The iteration value now
flows the other way: her feedback hardens the engine. After Jim's round 2
passes god's gates, proceed to Phase 4 (studio integration: capture→report as a
copilot tool, user-message injection hidden from UI).
