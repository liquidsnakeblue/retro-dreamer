# Grounding-E Red Team (Devil's Advocate) — Dwight, 2026-07-14

Card `grounding-e-redteam`. Critiquing `docs/copilot-grounding-e-proposal.md`
(Pam-2) before Jim implements. Read: the proposal, `backend/copilot.py`
(`_reader` L103-139, `_emit` L53-61), `backend/copilot_primer.md` (Hard rules
7-9, watch_brain section), `backend/watch_brain_job.py` (RESULT shape),
`docs/pam-engine-test-scoring.md` (the cited precedent).

**Honesty note:** findings below are real and each tied to specific code/text —
not manufactured. The proposal's *architecture* (two layers: shape generation +
observe escapes) is sound and matches the copilot-patterns literature. The flaws
are in the **post-check's design** and an **implementation gap**, not the
placement philosophy.

---

## Verdict (up top): **SHIP-WITH-CHANGES** (leaning rethink on layer 2)

- Ship **layer 1** (citation directive) as a cheap experiment — but measure its
  effect with a **human-scored** retest, NOT via `_check_grounding` (it's a rubber
  ruler — see W7).
- Do **not** ship `_check_grounding` as the gate. It both misses real
  confabulation (W2/W3) and fires on noise (W5), and its own acceptance gate
  ("zero unsupported claims") is measured by itself — so "passing" proves
  nothing. Worse, shipping it creates *false confidence* that confabulation is
  caught, which is more dangerous than an honest "we don't check yet."
- If a post-check is wanted, move toward **structured-output validation**
  (Alternative A) or **scoped/held-text** (Alternative D) — see below.

---

## Ranked weaknesses (each with a concrete failure scenario)

### W1 — The citation exemption is sentence-level and semantic-blind → decorative quotes pass (STRONGEST)
`_check_grounding` L103-106: if a sentence contains **any** verbatim quote that
appears anywhere in `report_text`, the **whole sentence** is exempted —
regardless of whether the quote *supports* the causal claim.
**Scenario:** report line `MILESTONE: first death at step 45`. Copilot writes:
> `"MILESTONE: first death at step 45"` — so the brain **couldn't dodge the enemy projectiles**.
The MILESTONE quote is real and present → check passes. But it establishes
nothing about "enemy projectiles." The citation becomes theater: the model
optimizes for *including a quote*, not for *the quote supporting the claim*. This
is the classic proxy-vs-target failure. A local model under a citation directive
will reliably produce **confabulation-with-decorative-quotes**.

### W2 — Hook A assumes a code path that does not exist today (IMPLEMENTATION GAP)
The proposal's hook A says "In `_reader`, when we see a Bash tool_result…". But
`_reader` (copilot.py:114-138) only branches on `type=="assistant"` (text +
tool_use), `type=="result"` (turn-done meta), and `type=="system"` (init).
**There is no `user`/`tool_result` branch** — tool_results are not parsed today,
so `_last_report_text` can never be populated by the sketched hook. Worse, it's
unverified whether Claude Code's stream-json even echoes tool_results to stdout
for this configuration (the copilot reads `report_text` via its *own* Bash curl
to `/api/tools/jobs/<id>`; that result lives in the model's context, not
necessarily in the backend's stream). **Before any code is written, confirm
tool_results are actually observable in `_reader` — else the entire post-check is
unimplementable as specified.**

### W3 — The "defensive pass" reintroduces the blunt-matching the proposal argues against
L108-110: `unsupported = [t for t in terms if t.lower() not in report_text.lower()]`.
A substring match anywhere in the report exempts the term.
**Scenario:** report contains `LOOP` events and a level label `Wall Zone`.
- "it **looped the track** lap after lap" → "loop" is in the report → not
  unsupported → no warning. (The proposal itself, L34-41, calls exactly this out
  as the failure mode of blunt matching — then re-implements it.)
- "it hit a **wall**" → "wall" substring in "Wall Zone" → not unsupported →
  passes. The occurrence-vs-relevance conflation the proposal warns against is
  baked into its own safety net.

### W4 — The causal regex is a leaky bucket: synonym confabulation sails through
`_CAUSAL_PATTERNS` is a fixed English lexicon. Recall is **unknown and unbounded-low**:
- "the agent **collided** with the barricade" → no match ("collided"/"barricade"
  absent; list has crash/hit-a-wall/bumped-into).
- "it **rammed** the obstacle" → no match.
- "couldn't **steer away** in time" → `steered?\s+(away\s+)?from` requires
  "from"; this phrase lacks it → no match.
Since the gate is "zero unsupported claims" and the detector can't see these,
**the gate can be passed by confabulation that simply avoids the listed words.**
A confabulating model isn't adversarial, but it is varied — synonyms are the
default, not the exception.

### W5 — The check runs on ALL assistant text → alarm fatigue (wide blast radius)
Causal vocabulary appears in legitimate non-diagnosis text: tool explanations
("reward_probe checks if the agent **avoids** the **hazard**"), conceptual
answers ("DreamerV3 learns a world model to **plan** ahead"), even meta
commentary. All match patterns, likely carry no quote → false warnings. If
warnings are dashboard-visible (open question 1), the human rapidly learns to
ignore them, and the real signal — when it comes — is lost. A safety net that
cries wolf is no safety net.

### W6 — `_last_report_text` is a single stale global with no scoping
Module-global, single-slot, no invalidation. **Scenarios:**
- **Cross-game bleed:** watch_brain on LM, then user asks about F-Zero. The
  F-Zero diagnosis is checked against LM's report → nonsense (an LM "page" term
  "grounds" an F-Zero claim, or vice versa).
- **Race:** two watch_brain jobs in flight → last-returner wins; claims about
  the other are checked against the wrong report.
- This is **the same association ambiguity that just bit Gate F** (vision):
  assuming "the most recent X" = "the X this claim is about." Don't repeat it.

### W7 — The gate is measured by the tool under test (rubber ruler)
Test plan §3 (acceptance gate): "zero unsupported causal claims" — but
"unsupported" is defined by `_check_grounding`, which has the W2/W3/W4 holes.
**A model that confabulates with synonyms + a decorative quote passes the gate
while being ungrounded.** The measurement can't bear the weight placed on it.
Pam's clean precedent (pam-engine-test-scoring) passed because a **human (god)**
scored it against ground truth — not because a regex did. Keep a human in the
acceptance loop.

### W8 — Non-blocking + emit-after = no prevention (harm already done)
The post-check runs AFTER the assistant text is emitted (L65-69). By the time
the `[⚠ ungrounded claim]` marker appears, the user has already read "it sped
into a wall" and may have acted on it (tuned a reward, mis-concluded). This is
fine as **telemetry** but not as a **safety net** (the proposal's word, L32). If
the goal is prevention on diagnosis turns, the text must be **held** until
checked for that turn class only (doesn't break chitchat). Don't call a
post-hoc footnote a safety net.

---

## Alternative mechanisms worth grafting

**A. Structured-output validation (my top pick — replaces `_check_grounding`).**
Instead of regex-scanning freeform prose, constrain the diagnosis to a schema:
each causal claim is a slot `{claim, evidence_quote, anchor}`. The post-check
then verifies **deterministically** that `evidence_quote` is a real
*char-span substring* of `report_text` AND that the `anchor` (step# + event
label, e.g. `DAMAGE@120`) exists in the report's structured events. This turns
"did the model ground its claim?" from a regex guess into a substring +
structural-existence check — far harder to game, zero synonym-blindness, and it
matches the "reason in code, model decides on structured evidence" pattern
(Theme 4 of my copilot-patterns doc). Cost: a schema/envelope the model must
fill; Qwen handles structured output tolerably with a firm template.

**B. Hold-and-check for diagnosis turns only (reduces W5/W8 blast radius).**
If keeping a freeform post-check, gate it to turns within N of a watch_brain
result AND only when the user asked a diagnostic question — and **block-then-
emit** for that class only. Prevention where it matters; flow preserved
elsewhere.

**C. Self-epistemics field.** Add `{grounded | inferred | speculative}` per
claim; post-check flags `inferred/speculative` claims lacking an explicit hedge.
Leverages the model's own uncertainty tracking rather than an external lexicon.

---

## What the proposal gets right (don't cut these)
- Two-layer instinct (shape generation + observe escapes) — correct and
  literature-backed.
- Engine (`episode_report.py`) stays read-only; guard lives in copilot.py —
  blast radius contained.
- Deterministic (code) check, not a model-judge — right call for cost/latency.
- Non-blocking as a *first pass* is defensible **if** it's labeled telemetry,
  not "safety net."

## Minimum changes before ship (if shipping the post-check at all)
1. Resolve W2 first — confirm tool_results reach `_reader`; if not, the hook is
   infeasible and the whole layer-2 plan changes.
2. Fix W1 — citation must be checked for **relevance to the claim**, not
   sentence-level presence. (Structured output A solves this cleanly.)
3. Fix W6 — scope `_last_report_text` per (game, job); invalidate on topic
   change.
4. Fix W5 — scope the check to diagnosis turns; or accept alarm fatigue as a
   known cost and log-only (not dashboard-visible).
5. Fix W7 — keep a **human-scored** acceptance gate; do not let
   `_check_grounding`'s output be the pass criterion.

**Bottom line:** layer 1 is a cheap, safe experiment (ship + measure by human).
Layer 2 as written is a rubber ruler that creates false confidence; either move
to structured-output validation (A) or defer it until W2 is resolved and W1/W6
are fixed. Shipping `_check_grounding` as the gate is the single biggest risk in
the proposal.
