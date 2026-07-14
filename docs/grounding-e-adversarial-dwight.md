# Grounding-E Adversarial Verification (Dwight, 2026-07-14)

Card `grounding-e-adversarial-verify`. RUNTIME probes against Jim's shipped
checker (commits 62a797a + b6987f9 + 277c0cd) — not a code review. Two
validators exist:
- **`_validate_grounding_claims`** (structured `{job_id, claims:[{claim, evidence_quote, anchor:{step,event}}]}`) — the deterministic GATE (my red-team graft, adopted).
- **`_check_grounding`** (freeform text) — secondary LEXICAL TELEMETRY only (`del report_text`; my W4 demotion, applied).

Reports registered via `cache_served_watch_report(job_id, report)` (the real
serving path) so `_served_report(job_id)` resolves. Probes run against BOTH the
engine-generated fixture reports AND a real completed watch_brain report
(`training-state/tools/watch_brain-191fc147/report.txt`).

> **Methodology note:** my phase-1 structured harness was malformed (treated
> no-exception as PASS, didn't register the report) — phase-2 white-box caught
> and corrected it. All verdicts below are from the corrected harness inspecting
> returned warning lists against a properly-served report. Disclosed.

---

## Verdict: **SOUND** (no undisclosed holes)

The structured gate is tight on every deterministic guarantee it claims. The
only "slips" are **(1) the provenance≠entailment residual — which Jim
disclosed and which is explicitly the layer-3 human-acceptance gate's job**, and
**(2) the deliberately-incomplete freeform synonym lexicon — telemetry, never a
gate**. Both are where the design says they are; neither is an undiscovered
hole.

## Structured validator (`_validate_grounding_claims`) — all deterministic checks CAUGHT

| # | probe (class) | payload (verbatim, abridged) | expected | actual |
|---|---|---|---|---|
| 1 | (a) fabricated quote absent | quote `"step 999 loss health 3->2 (-1, torpedo hit)"` (not in report) | CAUGHT | **CAUGHT** ✓ `evidence_quote is not verbatim` |
| 2 | (b) near-miss paraphrase | `"step 20 loss health 2048->1887..."` (single-space, real has multi-space padding) | CAUGHT | **CAUGHT** ✓ `not verbatim` |
| 3 | (e) claim not in diagnosis | claim `"grabbed the trident power-up"` absent from diag text | CAUGHT | **CAUGHT** ✓ `not an exact diagnosis sentence` |
| 4 | seam: anchor event mismatch | real step 20, wrong event `milestone` (real=`loss`) | CAUGHT | **CAUGHT** ✓ `anchor does not exist` |
| 5 | seam: quote misattributed to wrong anchor | real quote, anchor `regain@477` (quote belongs to `loss@20`) | CAUGHT | **CAUGHT** ✓ `evidence_quote does not belong to its anchor` |
| 6 | tail: TWO claims blocks | `<GROUNDING_CLAIMS>...<GROUNDING_CLAIMS>...` | CAUGHT | **CAUGHT** ✓ `must be the single final block` |
| 7 | tail: text after claims block | `...claims...More text after.` | CAUGHT | **CAUGHT** ✓ `must be the single final block` |
| 8 | tail: opening tag, no close | `<GROUNDING_CLAIMS>{bad` | CAUGHT | **CAUGHT** ✓ `opening tag has no closing tag` |
| 9 | norm: smart-quote in quote | `(` (U+FF08) instead of `(` | CAUGHT | **CAUGHT** ✓ (not a verbatim substring) |

**9/9 deterministic guarantees hold.** The claim-must-be-a-diagnosis-sentence,
verbatim-substring, anchor-exists, quote-belongs-to-anchor, and tail-parse
checks all fire correctly.

## Disclosed residual — provenance ≠ entailment (CONFIRMED, slips trivially)

Tested **correctly** (claim present in the diagnosis + real verbatim quote +
real anchor, but the claim is semantically unsupported by the quote). Passes
the structured gate. Confirmed against BOTH fixture and **real watch_brain-191fc147 report**:

| probe | claim | quote/anchor | result |
|---|---|---|---|
| (c1) | "could not dodge the enemy projectiles" | real `loss@20` health line | **PASS (slips)** |
| (c2) | "panicked and froze, causing the crash" (pure confab) | same real quote/anchor | **PASS (slips)** |
| (c3) real report | "froze in fear and could not decide" | real `loss@151` from watch_brain-191fc147 | **PASS (slips)** |
| (c4) real report | "confused by the color palette" (absurd) | same real quote/anchor | **PASS (slips)** |
| (d1) | "completed stage 3" (false; stage never advanced) | real `loss@20` anchor | **PASS (slips)** |

**How easily / naturally:** trivially. Even **absurd** confabulation ("confused
by the color palette") passes the moment it carries any real quote + real
anchor. This is exactly the residual Jim disclosed (`_validate_grounding_claims`
docstring L253: *"whether the evidence semantically entails the claim remains
part of the human-scored acceptance pass"*). **Implication for the layer-3
decision: the human-scored acceptance gate is fully load-bearing for
entailment — the structured gate provides zero entailment protection.** This is
not a hole in the gate; it's the boundary the design drew, and it holds as
specified.

## Freeform telemetry (`_check_grounding`) — synonym gaps confirmed (expected; not a gate)

Controls (regex hits) all FLAG ✓: `hit a wall`, `could not avoid obstacles`,
`sped into a wall`. Synonym gaps (deliberately incomplete lexicon, docstring
L62-65) all CLEAR (slip):

| probe | why it slips |
|---|---|
| "collided with an **enemy** submarine" | regex needs `enemies?` (plural); singular `enemy` misses |
| "**rammed** the barricade" (no "into") | regex requires `rammed into`; bare `rammed the` misses |
| "**smashed** into a wall" | `smashed` not in the verb list (`hit/collided/bumped/crashed/rammed`) |

These are **telemetry only** (`del report_text`; never a gate). Severity low.
Natural-language confabulation will routinely use synonyms the lexicon doesn't
list, so freeform telemetry's recall is — as documented — not a coverage
guarantee. The structured gate is the mechanism that matters.

## False-positive sweep — CLEAN ✓ (6/6 honest paths unflagged)

| honest / legit text | result |
|---|---|
| "The brain died twice at playerPage 6." | CLEAR ✓ |
| "It stalled near the end." | CLEAR ✓ |
| "The brain lost health three times before resetting." | CLEAR ✓ |
| "The training plan keeps replay_ratio at 0.125." (planner) | CLEAR ✓ |
| "The report does not establish why it crashed." (honest-unknown) | CLEAR ✓ |
| "DreamerV3 learns a world model to plan ahead." (conceptual) | CLEAR ✓ |

Zero false positives. Honest diagnoses, legit summaries, planner talk,
honest-unknowns, and conceptual statements all sail through unflagged. The
`_PLANNER_CONTEXT` / `_NONASSERTION` exemptions are doing their job.

---

## Ranked severity (surviving "exploits" — all disclosed/by-design)

1. **Provenance ≠ entailment (disclosed).** Trivially exploitable: any
   confabulation — even absurd — passes the structured gate with a real
   quote+anchor. **Not a hole**; it's the layer-3 human gate's entire job.
   Implication: layer-3 must actually inspect entailment, not just presence;
   if layer-3 is weak, confabulation ships. Severity HIGH *if* layer-3 is
   weak, but KNOWN/DESIGNED.
2. **Freeform synonym recall (disclosed, telemetry-only).** 3+ natural
   synonyms slip ("enemy" singular, "rammed the" no-into, "smashed"). Low
   severity — telemetry, not a gate; the structured path is authoritative.

## Bottom line for god
Jim's checker is **SOUND**. Every deterministic guarantee the structured gate
claims (verbatim quote, anchor existence, quote-anchor attribution,
diagnosis-sentence membership, tail-parse rules) fires correctly at runtime.
Zero false positives on honest paths. The two "slips" are precisely the two
residuals Jim disclosed, and both sit exactly where the design places them
(entailment → layer-3 human gate; synonym recall → non-gate telemetry). The
layer-3 (human-scored acceptance) decision is the one that matters now: it is
**fully load-bearing for semantic entailment** — the structured gate provides
none, by design. No patches required for soundness; the synonym lexicon could
be widened opportunistically but it's correctly non-load-bearing.

## Repro
Harnesses in `scratch/adv_phase1.py` (blind, recorded) + `scratch/adv_phase2.py`
(corrected, comprehensive). Run: `python scratch/adv_phase2.py` (uses
fzero-dreamer venv). Real-report probes via inline heredoc against
`watch_brain-191fc147/report.txt`. No backend source edited; no backend
restart; no training actions; CPU-only. Per boundaries, this doc is the only
commit.
