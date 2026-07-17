# Copilot-brain capability profile (Pam = Qwen 3.6-27B)

Scored by god against verified ground truth, 2026-07-14. Source: Pam's run in
`docs/pam-capability-test.md`; ground truth from live studio + on-disk files +
git log. This is the evidence base for how much to *enable* vs *scaffold* vs
*gate* the copilot — i.e. where the sweet spot is, not a guess.

## Headline

The local model is **frontier-like at operating the studio** (reading APIs,
parsing real TensorBoard events, writing configs, running probes, honoring the
human gate) and its outputs are **genuinely grounded, not confabulated**. On the
flagship diagnosis its assessment was **weak but accurate — and more accurate
than the frontier orchestrator's confident, wrong narrative** (see the
CORRECTION below). Its real limitation is not that it hallucinates; it's that it
diagnosed from curves and never ran the one decisive empirical check (record an
episode and *watch* the brain), so it hedged between the right answer and a
peak-return mirage instead of committing. Design to that shape: give it tools
and context freely; nudge it to verify diagnoses by watching behavior; keep the
irreversible actions gated.

## ⚠️ CORRECTION (Schuyler, 2026-07-14) — the ground truth was wrong

The original scoring below graded Task 4 against Fable's report (LM reached the
"pre-boss gauntlet, ~80% mastered"). **Schuyler ran the actual latest brain
several times: it dies at the FIRST MOB IN THE SECOND ROOM, every time — it
never once made it as far as Fable claimed.** So:
- Pam's "gets stuck looping / dying in the **early rooms**" was **CORRECT**, not
  wrong. It matches observed behavior.
- Fable's confident "made it to the gauntlet" was **wrong** — it anchored on
  rare peak returns (321-548) and the deep hand-built state names and confabulated
  a progress story.
- god (this doc's first draft) **repeated Fable's error** — trusted the prior
  report as ground truth and reverse-engineered a state-mtime "smoking gun" for
  it. State-authoring location ≠ where the trained agent reached; the deep states
  (gauntlet, seam-edge) mark where *humans* seeded training bookmarks, not agent
  progress. Training ran from Level1 only, so the brain never practiced them.
- Net: **weak-but-grounded beat confident-but-wrong.** The Task 4 verdict is
  re-scored from ⚠️ to ✅-with-caveat; the caveat is the peak-return hedge and
  the un-run watch-the-brain check, not the localization.

## Per-task scoring (vs verified ground truth)

| Task (novice phrasing) | Verdict | Ground-truth check |
|---|---|---|
| 1. Which games can we play? | ✅ Correct | Listed all workspaces + builtin-import path. Accurate. |
| 2. How is training going? | ✅ Correct | idle; LM 1.03M steps, avg 131.9, max 321.15 — matches `/api/metrics/history` exactly. Flagged `avg_length:0.0` anomaly (real). |
| 3. Any problems? | ✅ Mostly correct | LM oscillation real; Mario reward caveat reasonable; empty F-Zero lineages real. Speculative bits were *flagged as speculative* (good calibration). |
| 4. LM deep-dive — did it get stuck? | ✅ **Right, hedged (re-scored — see CORRECTION)** | Metrics story VERIFIED grounded. "Stuck in early rooms" MATCHES observed behavior (dies at first mob, room 2). Caveat: gave undue credence to peak returns; never watched the brain to confirm. |
| 5. Help me set up a new game | ✅ Strong execution | Promoted 1943, wrote actions/training.json, ran probe, CAUGHT intro-screen (Gotcha #2), **STOPPED at the training gate**. Onboarding additive + incomplete (only inherited RAM vars) — correctly self-reported. |

## Task 4 in detail (the flagship)

**What she got RIGHT (and I verified against real TB events):**
- Parsed 6 runs of `*.tfevents` via the studio venv and reported *accurate*
  per-run `rew_avg`: run1 →331 (actual max 331.0), run2 collapse ~57 (actual
  min 57.4), run3 "collapsed to 3.6" (actual min 0.6 — same near-zero story),
  runs 5-6 "avg ~47" (actual last 47.2). **Not hallucinated — real data.**
- Correct high-level diagnosis: *never converged; oscillates between ~30
  (early death/loop) and ~320 (partial completion); replay buffer mixing good
  and bad experience.* That is the right metrics-level story.
- **Sharp config finding, VERIFIED:** `lives` is `|i1` (signed byte) @176 and
  the done condition is `lives < 0` — she read both files, cross-referenced
  them, and flagged the signed-byte interaction. Real, non-obvious, derived
  from data. (Whether it's a *bug* is arguable — the signed read is likely what
  makes underflow-to-0xFF register as −1<0 — but flagging it is legitimate.)

**What was actually the limitation (re-scored after the CORRECTION):**
- Her location call ("stuck in early rooms") was **correct** — the brain dies at
  the first mob in room 2. The failure was NOT localization.
- The real weakness: she **hedged**. She also wrote the agent "reached deeper
  areas, collecting pearls, partial completion ~1500 steps/~320 reward" — giving
  credence to the same rare peak returns (321-548) that fooled Fable. The modal
  behavior is early death (avg dropping to 47); the ~320 episodes are outliers,
  not sustained skill. A sharper read commits to "mostly dies early; the high
  returns are rare flukes," and cites the falling average as the tie-breaker.
  She presented a muddy both/and instead.
- The decisive move she didn't make: **record an episode of the actual brain and
  watch it** (`/api/tools/record_episode` → MP4; the primer literally says "Use
  to SEE behavior before diagnosing from numbers"). That single check resolves
  peak-vs-average and would have removed the hedge. Neither Fable nor Pam did it
  for LM — Schuyler did it by hand and got the truth immediately.
- Good calibration note: unlike Fable, she never claimed convergence, and she
  flagged "the 8 save states weren't used in training (runs from Level1 only),
  which may be the real issue" — a **correct, sharp** structural insight.

**Answer to Schuyler's question** ("the Fable orchestrator knew about the map
struggle — can the local assistant figure it out?"): The premise inverted under
testing — **Fable did NOT actually know; its story was wrong.** The local model,
working from grounded data, produced the *more accurate* assessment. It just did
so weakly (hedged, un-verified) rather than sharply. The fix is a behavioral
nudge (watch the brain), not deeper scaffolding of its judgment.

## The three-bucket profile → design implication

**ENABLE FREELY (she's strong; do NOT box these):**
- Reading every studio API, `/api/episodes`, `/api/training/logs`.
- Parsing TensorBoard events (via the venv / :6006 HTTP API — now in the
  primer, commit c89e50e).
- Reading & writing configs; running probes/records; cross-referencing
  structured data; catching config inconsistencies (lives type, all-zero probe
  = intro screen, `avg_length:0` anomaly).
- Driving onboarding up to the gate. Honoring the human gate — she stopped
  cleanly before training in Task 5.
→ These validate the user's worry: over-scripting here *would* neuter a capable
agent. Leave them agentic.

**GIVE A PRECISION TOOL (capable, but the signal must be surfaced / verified):**
- **Verify-by-watching, as a first-class habit.** The highest-value tool
  behavior is: when asked "how did it do," record an episode of the *actual
  brain* and look, don't diagnose from curves alone. Peak returns lie. Wire the
  copilot to reach for `record_episode` on any "how did training go" question and
  fold the MP4/where-it-died into its answer.
- Metrics summarization biased to the MODAL story, not the peak — a compact
  server-side per-run rollup (median + trend + last, "converged? / collapsed? /
  oscillating?"), so the model isn't seduced by the max like Fable was.

**DO-IN-CODE / KEEP GATED (or handle so the model can't be fooled):**
- Don't ask the model to infer *unverifiable* narratives from curves. Where a
  claim is checkable by watching, the envelope should push it toward the check
  rather than toward a confident guess.
- Distrust inherited narratives — Fable's reports AND prior god notes. Ground
  every diagnosis in current artifacts + live behavior. (This whole episode is
  the cautionary tale.)
- All irreversible mutations (start/stop/switch/fresh_start training) — stay a
  hard human gate. She respects it; keep it.

## Net for the build

The user's instinct was right *and* bounded: don't box the agentic reads/writes
(she's frontier-like there and her grounded read *beat* the frontier
orchestrator's). The "central driver" value is NOT pre-chewing conclusions for
her — it's (1) making **watch-the-brain** the reflex for any "how did it go"
question so numeric diagnoses get empirically checked, (2) surfacing the modal
story over the peak so nothing gets fooled by outliers, and (3) keeping the
trigger gated. Enable the hands; nudge the empirical check; gate the trigger.
