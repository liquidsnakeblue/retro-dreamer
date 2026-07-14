# Copilot-as-Control-Plane: External Patterns → retro-dreamer

**Author:** Dwight (external research lead) · **Grounded:** 2026-07-14
**Scope:** Survey how leading agentic "copilot as control plane" products work,
distill what fits OUR case (brain = local **Qwen 3.6-27B**: strong at
tool-use/step-by-step, weak at deep reasoning + world knowledge).
**Companion docs:** `copilot-analysis-jim.md` (our as-built architecture),
`copilot-inventory-pam.md` (real usage/inventory).

> **Fetch note:** deep page-fetch (`web_crawl`) is broken on my MCP host
> (`AGENT_BRIDGE_TOKEN` gap). Every named claim below is grounded in
> `web_search` result titles/snippets (Google/Serper), **not** fabricated and
> not deep-read. Treat URLs as "found here"; verify before citing externally.

---

## Theme 1 — Full Ambient Context (keep the assistant continuously state-aware)

The winning pattern across every serious agent is: **the model never holds
system state in its head — code builds a compact context envelope and re-injects
it every turn.** Anthropic frames this as "context engineering" (context is a
**finite, curated** resource, not free) [Anthropic – *Effective context
engineering for AI agents*, Sep 2025]. Sourcegraph and Haystack echo the same
four-pillar model.

- **Cursor** is the reference for *codebase* ambient context: a RAG pipeline
  (chunk → embed → metadata → privacy → store → semantic search) keeps the agent
  grounded in the whole repo without dumping it raw [*How Cursor Actually Indexes
  Your Codebase*, TowardsDataScience]; plus `@`-mentions, `.cursorrules`/Rules,
  and MCP servers that "pull in everything in your project"
  [forum.cursor.com]. Mechanisms are configurable per-turn, not all-or-nothing
  [*Context Management Strategies for Cursor*, 2026].
- **Devin (Cognition)** argues for a **single-threaded, continuous-context
  agent** — all reasoning flows sequentially under unified context, explicitly
  to "manage finite context windows over long sessions"
  [LinkedIn/ricky-ho; ZenML LLMops database]. No multi-agent state loss.
- **Warp AI** grounds every answer **in terminal context** — command history,
  environment, and prior command **output** — so suggestions are situated, not
  generic [warp.dev/ai; amplifilabs].

**Three mechanisms recur** — pick deliberately:
| Mechanism | Example | Pros | Cons / fit for weak model |
|---|---|---|---|
| Per-turn injected **state envelope** | Cursor index, Devin session | Deterministic, cheap, no extra round-trip | Must be **compact**; weak model drowns in noise |
| **Queryable state tools** | Warp output-context, MCP `read_context` | Pulls only what's relevant | Model must know **what** to ask (hard when weak) |
| **Event subscriptions** | Devin long-session awareness | Proactive, not just reactive | Needs a durable event bus (we lack this today) |

**Verdict for us:** a weak-knowledge model should rely on **injected envelopes
+ queryable tools**, NOT free-form retrieval (it won't know what to retrieve).
Keep the envelope tiny and CODE-built.

---

## Theme 2 — Guided Assisted Actions (wizards w/ defaults + plain explanations)

Mature copilots replace blank forms with **proposed, explained defaults** the
human tweaks, not empty fields:

- **Vercel v0** turns a plain-English intent into a working app (React/Tailwind/
  shadcn) inside a **Vercel Sandbox** VM, iterating via follow-ups
  [vercel.com/blog/announcing-v0-generative-ui; v0.app; mindstudio]. Intent →
  artifact, not intent → questionnaire.
- **Warp** "proposes executable commands with your toolchain context, **then**
  runs" [digitalapplied] — propose-first, execute-second.
- **Form-fill copilots** (Microsoft Power Apps *AI form fill assistance*
  [learn.microsoft.com]; assistant-ui *Form-Filling AI Copilot*; NavaPBC
  caseworker tool with human **oversight**) all do the same move: suggest
  field values, human reviews/edits. The pattern is "AI fills, human
  blesses."

The shared shape: **model proposes a populated, annotated spec → rendered as an
editable card → one approve/run.** Crucially the proposal carries *why each
value*, in one line — because the human (and a weak model) both benefit from the
rationale being externalized.

---

## Theme 3 — Central Driver + Human Gate (propose → approve for mutations)

The consensus for "assistant operates a dangerous tool safely": **read freely,
gate every write.**

- **GitHub Copilot Workspace** is the canonical flow: **task → spec → plan →
  implement → PR**, auto-versioning context+history so it's reviewable from
  anywhere [githubnext.com; github.community discussion #142971].
- **n8n** has first-class *Human-in-the-loop for tools*: mark a tool as
  requiring review → "the workflow **pauses and waits**" for approval before
  that tool executes [docs.n8n.io; community.n8n.io].
- **kagent (Kubernetes)** makes it explicit — an agent that "reads your cluster
  freely but **pauses and asks for approval before creating, updating, or
  deleting**" [kagent.dev; maniak.io *Building AI Agents You Can Actually
  Trust on Kubernetes*]. Medium summarizes the rule bluntly: "every **write**
  operation the agent proposes must pass through an **approval gate**."
- **Devin** lists "preventing destructive operations" as a core problem
  [ZenML].

The universal primitive = **propose a mutation as a structured card; the UI,
not the model, executes on human click.** The model never holds the trigger.

---

## Theme 4 — Weak-Model Compensation (reason in code, decide on structured evidence, escalate)

This is the theme that decides whether a 27B-local brain can *drive* reliably.

- **Small/local models are unreliable at free-form reasoning & raw structured
  output** [towardsai *Structured Output With Local…*, 2025]; reliability comes
  from **schema enforcement** tooling (BAML gets stable JSON "even with models
  as small as Gemma 4B") [r/LocalLLaMA].
- **Do the reasoning in code, let the model decide/narrate on compact
  structured evidence.** Cursor's own agent harness is "file editing, codebase
  search, terminal execution" — the **model orchestrates tools; the tools do
  the math** [cursor.com/blog/agent-best-practices]. Never ask a weak model to
  mentally parse a 200-line log.
- **Escalate the single hard turn to a stronger model.** Model routing is now
  standard: cheap-local-for-majority, cloud-only-when-needed
  [r/openclaw *Making the most out of local models (task escalation)*;
  mindstudio *AI Model Routing — cut costs 60%*; arxiv xRouter 2510.08439;
  digitalapplied *LLM Model Routing 2026*]. The local model stays the
  **driver/orchestrator**; a stronger model is a **tool** it calls for one
  step.

---

## Concrete Recommendations for OUR Copilot (7)

Grounded in Jim's architecture (headless Claude Code + Qwen 3.6-27B, Bash+curl
to `/api/*`, 1s poll, in-memory 500-event ring) and Pam's inventory (no
persisted transcripts, `training/stop` can't signal user-intent so the watchdog
re-arms). Numbers cite the theme/source above.

1. **Per-turn STATE ENVELOPE, built by code.** Add `/api/copilot/context`
   returning compact JSON — active game+lineage+step, run/suspend/stopped
   flag, last metrics, pending tool-job ids, VRAM headroom — and have the
   primer instruct the model it's authoritative. This is the Cursor-index /
   Anthropic-context-engineering pattern (Theme 1) and the single biggest win
   for a weak-knowledge brain: it stops guessing state.
2. **Read-tools free; write-tools propose→approve.** Classify routes:
   `catalog`/`status`/`reward_probe` (read) run autonomously;
   `training/{start,stop,suspend,switch}`, `games/import` emit an **approval
   card** the human clicks — exactly kagent/n8n (Theme 3). Model proposes, UI
   executes. Side-effect fix: `training/stop` should stamp a `stopped-by-user`
   intent so the **watchdog stops confusing a user-stop with a crash** (Pam's
   open wart).
3. **Guided start/resume wizards with explained defaults.** "Train F-Zero" →
   copilot returns one card: suggested arch (w/ VRAM-advisor hint), practice
   rotation, heal-reward on, buffer config — each with a one-line *why*. Human
   approves/edits. Replaces raw `GameConfigEditor` (Theme 2, v0/Power-Apps
   form-fill).
4. **Reason in CODE; feed the model compact structured evidence.** Forbid the
   model from parsing logs/computing deltas itself. Stand up analyzer helpers
   that return `{verdict, delta, recommended_action}` JSON; the model
   narrates/decides on that. Direct compensation for Qwen's weak deep-reasoning
   (Theme 4).
5. **Strong-model escalation as a tool.** Add a `consult_strong_model` tool
   routed to CLIProxyAPI / a beefier LM-Studio endpoint, called for flagged
   tasks (reward design, debugging a diverging run, low-confidence turns).
   Qwen stays driver; the cloud model is one callable step (Theme 4 routing).
6. **Persist the transcript + `/resume`.** The 500-event in-memory ring dies on
   restart (Pam/Jim). Write events to disk and support session resume so the
   "central driver" actually remembers across bounces — mirrors Devin's
   continuous-context thesis and Copilot Workspace's versioned history.
7. **Event subscriptions → proactive control plane.** Let the copilot
   *subscribe* to meaningful state changes (crash, reward divergence,
   checkpoint, watchdog auto-resume) and surface them unprompted. Turns it
   from a reactive chat tab into the studio's actual nervous system
   (Warp-style ambient grounding + Devin long-session awareness).

*(Optional 8th: enforce a **schema** on the copilot's action proposals (BAML
-style) so the UI can render approval cards reliably even from Qwen.)*
