# Vision-Pipe Frame-Identity Reversal — Diagnosis (Dwight, 2026-07-14)

Card `product-vision-association`. Gate F (Phase 4) failed with a suspicious
signature: images flowed through the product path (claude-local → :8082
image-fix proxy → llama.cpp :6789) and the model produced an *accurate*
description of one frame, but **assigned it to the wrong filename**. Two
hypotheses to separate:
- **(a)** the :8082 proxy's image re-injection loses name→image association, so
  any multi-image turn is a coin-flip; vs
- **(b)** the model was actually blind and confabulated from a ground-truth key
  god mistakenly published in `docs/phase4-studio-integration-spec.md`.

**Verdict (TL;DR):** **Neither pure hypothesis holds.** The proxy *is*
structurally lossy (it strips the `tool_use_id`↔image binding and appends images
**unlabeled** at message end) — but it **preserves order**, and the model
reliably recovers association by position: **0 reversals in 6 single + 3 double
real-path runs + 1 raw 2-images-in-one-message API call** (10/10 correct color
binding). The model is demonstrably **not blind** (background colors correct
10/10; reads "MAPLE" cleanly). So (a)'s "coin-flip" is disproven, and (b)'s
"blind confabulation" is disproven — but the **test-contamination flaw god
already flagged is real** and remains the most likely contributor to Gate F's
specific reversal, compounded by maximally-confusable stimuli (busy gameplay
frame vs a pure-blank frame) and the proxy's loss of an explicit binding.

---

## 1. Method

- **Own test frames, programmatic ground truth, published nowhere.** `ffmpeg`
  solid-color 320×320 + `drawtext`. Verified pixel RGB with PIL. Lived in
  `~/vision-diag-scratch/` (outside retro-dreamer/docs).
  - First batch (`frame_alpha`=BLUE+"ALPHA", `frame_november`=GREEN+"NOVEMBER")
    was **confounded** — the words matched the filenames, so a blind model could
    answer by name alone (same leak class god flagged). Discarded for verdict;
    kept in `results/` for the record.
  - **Unconfounded batch (used for verdict):** `frame_1`=ORANGE+"MAPLE",
    `frame_2`=PURPLE+"CIRCLE". Filenames give zero content hint.
- **Model-side tests via the real product path:** `claude-local -p` (sets
  `ANTHROPIC_BASE_URL=http://localhost:8082`; verified in `~/.bashrc:122-148`;
  Claude Code 2.1.208; model `qwen3.6-27b`). Confirmed :8082 routes to the proxy
  (PID 395338 = `~/lmstudio-proxy/proxy.py`).
  - **Test 1 — one image per turn** (baseline vision), 3× each frame.
  - **Test 2 — two images in one turn** (reproduces Gate F), 3×.
- **Direct code + synthetic transform:** read `proxy.py:fix_messages`; called
  it in isolation on a crafted 2-image tool_result payload to see *exactly* what
  reaches the model.
- **Raw `/v1/messages` through :8082** with two images in **one** user message
  (bypasses Claude Code tool orchestration — removes any doubt about whether the
  two reads were parallelized).

## 2. Results (raw answers)

**Test 1 — one image per turn (unconfounded frames):**

| run | frame_1 (truth ORANGE/MAPLE) | frame_2 (truth PURPLE/CIRCLE) |
|---|---|---|
| 1 | `COLOR=orange WORD=MAPLE` ✅ | `COLOR=purple WORD=BLUE` (color ✅, word OCR ✗) |
| 2 | `COLOR=orange WORD=MAPLE` ✅ | `COLOR=purple WORD=BLUE` (color ✅, word OCR ✗) |
| 3 | `COLOR=orange WORD=MAPLE` ✅ | `COLOR=purple WORD=BLUE` (color ✅, word OCR ✗) |

Background color **6/6 correct**. Word OCR perfect on frame_1 (3/3), consistently
wrong on frame_2 ("CIRCLE"→"BLUE") — a genuine perception/OCR error, not
confabulation (the word "CIRCLE" exists in no repo doc).

**Test 2 — two images in one turn (unconfounded), 3×:**

| run | frame_1.png | frame_2.png | reversal? |
|---|---|---|---|
| 1 | orange / MAPLE ✅ | purple / CIRCLE ✅ | none |
| 2 | orange / MAPLE ✅ | purple / CIRCLE ✅ | none |
| 3 | orange / MAPLE ✅ | purple / CIRCLE ✅ | none |

**3/3 correct association, no identity reversal.** (Curiously frame_2's word was
read correctly here but not in the single test — context/priming affecting OCR;
orthogonal to association.)

**Raw `/v1/messages`, two images in ONE user message, straight through :8082:**
```
frame_1.png: COLOR=Orange WORD=MAPLE
frame_2.png: COLOR=Purple WORD=RULE
```
Colors both correct, **association correct (no reversal)**, frame_2 word OCR
again off ("CIRCLE"→"RULE").

## 3. Proxy code findings (`~/lmstudio-proxy/proxy.py`, `fix_messages` L38-72)

The proxy exists because the backend (llama.cpp via LM Studio) does not accept
images *inside* `tool_result` blocks, so it must hoist them to message level.
The transform:

1. Walks each message's content blocks. For each `tool_result` block, finds
   `image` parts in its inner content, **removes them**, and substitutes the
   text `"[image moved to message level]"`.
2. After all blocks are processed, `new_content.extend(extracted_images)` —
   appends **every** extracted image as bare image blocks at the **end** of the
   message.

**Synthetic transform of a 2-image tool_result message** (what the model
receives):
```
block[0] tool_result  tool_use_id=tool_A   inner_text='[image moved to message level]'
block[1] tool_result  tool_use_id=tool_B   inner_text='[image moved to message level]'
block[2] IMAGE        (frame_1 pixels)     carries tool_use_id link? NO
block[3] IMAGE        (frame_2 pixels)     carries tool_use_id link? NO
```
So the **explicit binding is destroyed** (no `tool_use_id` on the image, no
filename label), and the in-place text gives no positional hint. **Order is
preserved** (tool_A's image before tool_B's). The model must infer the
image→filename mapping from **position alone**.

## 4. Verdict per hypothesis

**(a) Proxy loses association → coin-flip multi-image turns: NOT SUPPORTED.**
The proxy is genuinely lossy (§3 — it drops the only explicit binding), which is
a real latent defect. But it preserves order, and in every multi-image test the
model recovered the correct mapping (3/3 real-path double + 1 raw 2-image API
call, 0 reversals). It is **not** a coin-flip; it is order-correlated and, in
practice with this model, reliable.

**(b) Model blind, confabulated from the leaked spec key: NOT SUPPORTED as
"blind."** The model sees the images — background colors 10/10 across all
methods, and it reads rendered words ("MAPLE" 6/6 when content is unconfounded).
It is not confabulating from a key in my tests (my frames are unpublished).
**However**, god's flagged test-design flaw is real: the Gate F ground-truth key
(`frame_c` = Ariel/open-water/3-hearts; `frame_e` = pure black) was published
inline in `phase4-studio-integration-spec.md:107-109`, a doc the copilot session
(cwd = retro-dreamer) can read. So for **Gate F specifically**, contamination
cannot be excluded — but my clean tests show the model does not *need* the key
to see/associate correctly, so contamination is a bias, not the mechanism.

**Synthesis — most likely cause of Gate F's reversal (a confluence, not either
hypothesis alone):**
1. **Test contamination (primary, god-flagged):** the answer key was in-context,
   priming the model's expectation.
2. **Maximally confusable stimuli:** a busy gameplay frame vs a **pure-blank**
   frame — "black" is an attribute that could attach to either, so one uncertain
   perception + a primed expectation yields a clean swap.
3. **Proxy's loss of explicit binding (secondary enabler):** the one signal that
   would have disambiguated by construction is stripped, leaving only positional
   inference, which is fragile precisely when (1) and (2) align.

My clean test — two visually distinct, equally-salient frames, no leaked key —
did not reproduce a reversal in **10/10** image bindings.

## 5. Recommended fix (propose; apply only on god ack — read-only on proxy per boundaries)

1. **Eliminate test contamination (highest value, trivial, do first).** Never
   publish the ground-truth key in any repo doc the copilot can read.
   Pre-register keys externally (e.g., a file under `~/` outside the repo, or a
   hashed key). This alone removes the (b) confound and reopens Gate F to a fair
   retest.
2. **Harden `fix_messages` to preserve an explicit binding (real defect, ~10-line
   safe reversible patch).** When hoisting each tool_result's image(s) to message
   level, (i) append them **immediately after** their owning `tool_result` block
   (co-located, not all-at-end), and (ii) prefix with a text label carrying the
   `tool_use_id` (and the tool_result's surviving text, which often contains the
   filename for Read). e.g. insert
   `{"type":"text","text":"[image for tool_use_id=tool_A]"}` before the image.
   Gives the model an order-independent binding; trivially reversible.
3. **Model-limitation note (not a bug):** rendered-text OCR is imperfect on this
   27B local model ("CIRCLE"→"BLUE"/"RULE"). For real gameplay frames, prefer
   structural/color signals over fine text; don't gate on reading in-frame text.

**Artifacts:** test frames + harness + raw outputs in `~/vision-diag-scratch/`
(`results/` = confounded v1 for record; `results2/` = unconfounded v2 used for
verdict). No retro-dreamer repo files were modified.
