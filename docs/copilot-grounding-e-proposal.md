# Copilot Grounding E — Proposal

**Author:** Pam (worker-pam2) | **Date:** 2026-07-14 | **Status:** Awaiting god sign-off

## Problem

On the Phase 4 Gate E retest (job `watch_brain-224f9314`), the copilot correctly grounded its
numerical claims but still invented causal/game-semantic statements absent from the report:

- "could not avoid obstacles" (no obstacle vocabulary in report)
- "sped into a wall" (no collision cause in report)
- "lap after lap" (loop/oscillator ≠ lap — report never says this)

One primer tightening already shipped (9fd27e7: Hard rule 9, sample-scoping). Further
prompt-wording iteration is judged low-yield.

## Chosen Mechanism: Claim-citation template + deterministic post-check

Two layers, each necessary:

1. **Claim-citation directive** (primer): Forces the copilot to carry a brief verbatim
   quote from the report when asserting a causal conclusion. The contract style that passed
   god's scoring in `docs/pam-engine-test-scoring.md` — Pam's usability test was clean
   because every factual claim traced to a report line. This shapes the model's reasoning
   during token generation, not after.

2. **Grounding post-check** (code, `_reader`): A deterministic scan of assistant text
   that flags causal vocabulary unsupported by the report. Catches confabulation the
   primer fails to prevent.

The citation contract handles the *reasoning* (think-before-you-assert). The post-check
handles the *safety net* (catch what slips through).

**Why not pure vocabulary post-check alone?** A vocabulary check that flags "causal words
not in the report" is too blunt — it would flag "it keeps dying" on a report full of
DEATH events, or "it stopped progressing" on a report with STALL events. The report's
vocabulary is *structured* (DEATH, DAMAGE, LOOP, STALL) while natural-language summaries
use different terms (dying, stuck, not advancing). Distinguishing legitimate summary
rephrasing from true confabulation requires understanding *what* the claim is about, not
just *which words* it uses. The citation contract sidesteps this by requiring evidence,
not matching vocabulary.

**Why not pure citation directive alone?** Prompt instructions alone failed once already
(9fd27e7). The post-check provides an observable signal — the dashboard can surface
warnings, and we can measure whether the directive actually works.

## Exact hook points

### A. `backend/copilot.py` — `_reader` thread

Store the last watch_brain report text, then run grounding check on assistant text.

```python
# Module-level (near _events, _seq):
_last_report_text: Optional[str] = None

# In _reader, when we see a Bash tool_result:
# Detect watch_brain poll results by the RESULT prefix with report_text key
if t == "result":
    # The tool_result JSON contains the job result
    # Parse report_text from the Bash output if present
    # (watch_brain_job.py prints: RESULT {"report_text": "...", ...})
    pass  # extraction logic below

# In _reader, on each assistant text block, AFTER emitting:
if _last_report_text:
    warnings = _check_grounding(text, _last_report_text)
    for w in warnings:
        _emit("grounding-warning", w["message"], detail=w["detail"])
```

The report text is extracted from the tool_result when the copilot's Bash poll command
returns the watch_brain job result. The RESULT line contains `report_text` — we parse
this and cache it for the next assistant turn.

### B. `backend/copilot_primer.md` — Hard rules section

Add one sentence after Hard rule 8:

> When asserting a causal conclusion about gameplay (e.g. "it hit a wall", "it couldn't
> avoid obstacles", "it completed a lap"), include a brief verbatim quote from the
> report that supports the inference — e.g. "DAMAGE at step 120 (health -8, at
> playerPage +42)" — so the claim is anchored to evidence.

### C. New function: `_check_grounding(text, report_text) -> list[dict]`

```python
_CAUSAL_PATTERNS = [
    r'\b(obstacle|wall|enemy|boss|trap|hazard|hit\s+(a\s+)?wall|crash|bumped?\s+into)\b',
    r'\b(lap|looped?\s+(the\s+)?track|circuit|round|race\s+(through|complete))\b',
    r'\b(avoid|dodged?\b|evaded?\b|steered?\s+(away\s+)?from|missed?\s*(an?\s+)?opportunity)\b',
    r'\b(collect|pickup|gather|grabs?\s+(coins?|items?|power-?ups?))\b',
    r'\b(strategy|tactic|plan|decision|chose?\s+(to|not\s+))\b',
]

def _check_grounding(text: str, report_text: str) -> list:
    """Flag causal claims not supported by the report."""
    warnings = []
    for sentence in _split_sentences(text):
        matches = [m for pat in _CAUSAL_PATTERNS for m in re.finditer(pat, sentence, re.I)]
        if not matches:
            continue
        # Check if the sentence carries a verbatim quote from the report
        quotes = re.findall(r'"([^"]{8,})"', sentence)
        if any(q in report_text for q in quotes):
            continue  # Causal claim is citation-anchored
        terms = [m.group(0) for m in matches]
        # Verify these terms are actually in the report (defensive pass)
        unsupported = [t for t in terms if t.lower() not in report_text.lower()]
        if unsupported:
            warnings.append({
                "message": f"Unsupported causal claim: {sentence.strip()}",
                "detail": f"Terms not in report: {', '.join(unsupported)}",
            })
    return warnings
```

### D. `backend/copilot_primer.md` — Diagnosing workflow

Update the diagnosing workflow to reference the citation expectation:

> watch_brain on the suspect state → read its report → ground every causal
> conclusion in a report quote → then use GET /api/training/status + modal/typical
> episode returns.

## Report vocabulary (for reference)

The episode report uses a fixed, structured vocabulary:

- **Events:** DEATH, DAMAGE, TERMINAL, RESTART, OSCILLATE, LOOP, STALL, RECOVERY, MILESTONE, SPAN
- **Labels:** terminal, resource, progress, objective, milestone, timer, rewarded, cyclic, oscillating, looping, stalled
- **Structure:** Step ranges, variable values, deltas, position coordinates, life segments

Natural-language summaries that rephrase these ("it keeps dying" for DEATH events) are
fine. True confabulation ("it hit a wall") invents game-semantic detail not present in
any report line.

## Test plan

1. **Unit test:** Write 5 test cases — 3 confabulation examples from the real retest
   ("could not avoid obstacles", "sped into a wall", "lap after lap") + 2 legitimate
   summaries ("the brain died twice at playerPage 6" [should pass], "it stalled near
   the end" [should pass]). Verify the checker flags the first 3 and passes the last 2.

2. **Fixture replay:** Run the checker against the existing fixture reports (LM, Mario,
   F-Zero) with synthetic confabulation diagnoses. Verify zero false positives on
   report-accurate text.

3. **Live retest:** Run watch_brain on LM with the copilot, verify the diagnosis contains
   zero unsupported causal claims. This is the acceptance gate.

## Scope and boundaries

- **NO code edits** until god signs off.
- Engine (`backend/episode_report.py`) is READ-ONLY — the guard lives in copilot.py.
- No training actions. No primer edits before sign-off.
- Definition of done: god-verified retest of a watch_brain diagnosis with zero
  unsupported causal claims.

## Open questions

1. Should grounding warnings be visible in the dashboard chat (as a system-style
   annotation) or only in server logs? Recommendation: visible, as a subtle
   `[⚠ ungrounded claim]` inline marker so the human sees it.

2. Should the post-check block the assistant text or emit alongside it?
   Recommendation: emit alongside — blocking breaks the copilot flow; warnings
   are the right signal for a first pass.
