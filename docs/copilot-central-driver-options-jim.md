# Copilot as Central Driver: Architecture Options

Design-options date: 2026-07-13. This builds on
[the as-built copilot analysis](copilot-analysis-jim.md) and deliberately stops
short of a final implementation spec. Endpoint names proposed below are
illustrative. The governing constraint is that code should collect facts,
derive defaults, validate plans, and enforce approvals; Qwen 3.6 27B should
explain those results and elicit the few choices that remain.

## 1. FULL AMBIENT CONTEXT

### What “fully context-aware” should mean

The copilot should not merely have permission to discover state. At the start
of every user turn it should receive a coherent, timestamped view of the facts
most likely to change the answer, and it should have one typed query surface
for expanding that view. “Always know” should cover the following exact state:

| State family | Facts the copilot should always have | Existing source and present gap |
|---|---|---|
| **Per-client focus** | Selected game, active tab, current training draft, and any pending proposal. The selected game is client context, not global studio state. | App owns selectedGame and activeTab locally and passes the game to TrainingControls, but CopilotPanel receives no props ([frontend/src/App.tsx:12-17](../frontend/src/App.tsx#L12), [frontend/src/App.tsx:52-56](../frontend/src/App.tsx#L52), [frontend/src/App.tsx:122-131](../frontend/src/App.tsx#L122)). |
| **Set-up custom games** | Every custom workspace’s id, display name, system, ROM presence, default state, state count, config presence/validity, trainability, readiness blockers, and whether it has a trained brain. | GET /api/games is the discovery source. GameManager marks custom entries with source=custom, has_custom_config=true, and rom_ready ([backend/games/manager.py:117-139](../backend/games/manager.py#L117)). GET /api/workspaces is not workspace discovery: it only iterates catalog games, so an untrained custom workspace may be absent ([backend/api/routes.py:179-209](../backend/api/routes.py#L179)). Trainability and blockers are not computed today. |
| **ROM-ready but not onboarded** | Complete filterable inventory of built-ins with an installed ROM but no custom workspace, plus display name, system, shipped state count, and pre-seeded RAM-map availability. | GET /api/games already provides source=builtin, has_custom_config=false, and rom_ready, so the class is derivable in code ([backend/games/manager.py:143-168](../backend/games/manager.py#L143)). Promotion already exists ([backend/api/routes.py:393-406](../backend/api/routes.py#L393)). Individual built-in detail omits rom_ready, so the projection must join list and detail data ([backend/games/manager.py:206-236](../backend/games/manager.py#L206)). |
| **Live training identity and health** | idle/training/stopping/error, process health, game, model size, initial state/rotation, launch mode, lineage/session, start time, current step/episode, steps/sec, latest and recent returns, average/max return, error, and a last-metric timestamp. | GET /api/training/status exposes game, state, step, episode, elapsed time, speed, average/max return, GPU fields, and error ([backend/api/routes.py:224-242](../backend/api/routes.py#L224)). It omits the effective launch config, lineage/head, launch mode, latest return, process identity, and freshness. Metrics are parsed from stdout and average the last 100 returns ([backend/training/trainer.py:815-850](../backend/training/trainer.py#L815)). |
| **Logs, metrics, and TensorBoard** | Whether each source is available and fresh; a bounded code-derived trend summary; last significant log/error; TensorBoard URL, process health, game scope, and actual log directory. Raw histories remain queryable rather than injected. | GET /api/training/logs returns a tail of the trainer’s 200-line memory buffer, GET /api/metrics/history returns only the current aggregate despite its name, and GET /api/tensorboard/logdir returns the active trainer game’s path ([backend/api/routes.py:306-328](../backend/api/routes.py#L306), [backend/training/trainer.py:145-146](../backend/training/trainer.py#L145), [backend/training/trainer.py:853-863](../backend/training/trainer.py#L853)). TensorBoard is started once at server startup, so process scope/health needs explicit reporting ([backend/server.py:104-106](../backend/server.py#L104)). |
| **Brains, lineages, and heads** | For each trained game: active lineage, all lineage names/statuses, running session, resumable head id/step/time, checkpoint/replay availability, originating model/config, validation state, and compatibility warnings. | The catalog schema already has games, lineages, sessions, and snapshots with many of these fields ([backend/catalog.py:20-64](../backend/catalog.py#L20)); head selection is authoritative ([backend/catalog.py:96-150](../backend/catalog.py#L96)). GET /api/workspaces exposes only lineage name/status/running and head step/path ([backend/api/routes.py:179-209](../backend/api/routes.py#L179)). GET /api/checkpoints is scoped to the trainer’s current run rather than the full catalog ([backend/api/routes.py:295-299](../backend/api/routes.py#L295), [backend/training/trainer.py:764-787](../backend/training/trainer.py#L764)). |
| **Per-game states** | Default state and, for every state, id/file, human label, group, description/objective, screenshot reference, live-gameplay verification, last reward-probe result/config hash, difficulty/recommended use, and whether it may join a training rotation. | GET /api/games/{id} already returns filenames and custom annotated_states with label/group; missing annotations are synthesized ([backend/games/manager.py:174-204](../backend/games/manager.py#L174)). GET /api/games/{id}/states returns names only ([backend/api/routes.py:354-363](../backend/api/routes.py#L354)), which is all TrainingControls uses ([frontend/src/hooks/useGameConfig.ts:44-76](../frontend/src/hooks/useGameConfig.ts#L44)). Verification/probe results are not persisted per state. |
| **Per-game configuration** | Config presence, validity, revision/hash, and compact semantic summaries: RAM variables/types; action names/buttons/count; reward variables/formulas; done rules; default state and metadata. Exact data.json, actions.json, training.json, scenario.json, and metadata.json remain expandable. | Existing routes read or write one config at a time ([backend/api/routes.py:366-390](../backend/api/routes.py#L366)). GameManager validates training/actions on write and provides built-in fallbacks ([backend/games/manager.py:265-326](../backend/games/manager.py#L265)). There is no bulk summary, revision, or readiness result. |
| **Hardware/training recommendation** | GPU, VRAM, model sizes that fit, recommended size, and code-derived dependent defaults for batch, replay ratio, and environment count. | GET /api/advisor/model_size already returns GPU, VRAM, recommended size, fits, and a note ([backend/api/routes.py:156-176](../backend/api/routes.py#L156)). It does not return the dependent preset values from TrainingConfig ([backend/training/config.py:74-88](../backend/training/config.py#L74)). |
| **Background studio work** | Running/recent probes, captures, state builders, walkers, and recordings, including status, target game, progress/result/error, and freshness, so the copilot does not duplicate work. | GET /api/tools/jobs lists the current in-memory jobs and GET /api/tools/jobs/{id} returns a log tail ([backend/tools.py:222-238](../backend/tools.py#L222)). Jobs are not durable and do not have a normalized progress/error contract. |
| **Derived capabilities** | Code-computed actions currently safe/possible: can_start, can_resume, can_switch, can_promote, needs_onboarding, config blockers, approval level, and the exact reason for every false result. | No aggregate endpoint computes these today. Qwen currently has to infer them from several routes and a long primer. This is precisely the reasoning that should move into code. |

The authoritative projection should contain complete lists and exact
machine-readable facts. The small prompt envelope does not need to repeat every
raw config, log line, checkpoint path, or all 1,000-plus built-ins. For example,
it can include the full custom-workspace list, the count and revision of
promotable built-ins, the focused game’s full summary, live training, active
jobs, and capability flags. A filtered inventory query can expand built-ins
when the user names or asks to browse games.

### Delivery options

| Option | Shape | Advantages | Costs / failure modes |
|---|---|---|---|
| **A. Full envelope injected on every turn** | Extend the text-only send contract ([backend/copilot.py:158-176](../backend/copilot.py#L158)) and prepend one complete studio snapshot before each user message. | No preliminary model tool call; the answer always begins with fresh facts; simplest mental model. | Repeats mostly static data into the long-lived conversation, rapidly consuming context; a full built-in list/config set is noisy; projection and API callers can drift if they use separate code; hidden prompt bulk is hard to inspect. |
| **B. Queryable context tool only** | Add one typed tool or GET /api/studio/state with scopes/filters; Qwen calls it whenever needed. | Small prompts; detail is paid for only when relevant; easy for humans and tests to inspect the same JSON. | Does not satisfy “always aware”: Qwen may skip or under-scope the call, use remembered state, or spend another slow tool round trip. Current local settings allow one concurrent tool use, making mandatory discovery visible latency ([copilot-analysis-jim.md:256-268](copilot-analysis-jim.md#L256)). |
| **C. One projection, hybrid delivery — RECOMMENDED** | A server-side StudioStateBuilder produces a coherent revisioned projection. A compact slice is injected automatically at each turn, while GET /api/studio/state (or an equivalent typed tool) exposes the same projection for filtered/full expansion. | Always supplies identity, focus, health, and blockers; avoids a mandatory Qwen call; full facts remain inspectable/queryable; one code path prevents two truths; supports later UI consumers and tests. | Moderate backend work; needs caching/invalidation for expensive game discovery and per-client focus in the send request; still requires strict size budgets and action-time revalidation. |

### Recommended hybrid rules

1. **Build once in code.** A single builder joins GameManager discovery, catalog,
   trainer/status, tool jobs, advisor data, and per-game summaries. Both prompt
   injection and GET /api/studio/state serialize that builder; Qwen never
   performs the joins.
2. **Separate global state from client focus.** CopilotPanel sends focus_game_id,
   active_tab, and pending proposal/draft revision with each message. Do not put
   a browser’s selected game into a global server singleton.
3. **Keep the injected slice bounded.** Target roughly 2–4 KB / 1–2K tokens:
   generated_at and revision; focus; full live-run summary; all custom games in
   compact form; promotable-built-in count/revision; focused game state/config/
   head summary; advisor result; active jobs; capability flags. The current
   claude-local context cap is large, but the primer and conversation already
   occupy it and repeated envelopes accumulate
   ([copilot-analysis-jim.md:67-72](copilot-analysis-jim.md#L67)).
4. **Expand, do not dump.** Full configs, raw logs, TensorBoard series,
   checkpoint paths, and the built-in catalog are fetched by scope/filter and
   summarized by deterministic code before Qwen sees them.
5. **Expose freshness.** Every dynamic subsection carries observed_at or
   last_updated_at. Cache expensive game inventory/config summaries and
   invalidate on import, promote, config write, state creation, or catalog
   change; live status is read at message receipt.
6. **Never authorize from remembered context.** Every mutation proposal binds
   to studio_revision plus its effective payload. The server refreshes
   preconditions on confirmation and rejects a stale plan. Ambient context
   improves the conversation; action-time validation protects the system.

## 2. GUIDED TRAINING-START ASSIST

### Existing building blocks

The backend already accepts model_size, batch_size, replay_ratio, num_envs,
fresh_start, game_id, initial_state, and resume_prefill
([backend/api/routes.py:12-20](../backend/api/routes.py#L12)). It validates the
model-size name, resolves an omitted initial state through game metadata,
applies numeric overrides, and starts training immediately
([backend/api/routes.py:44-117](../backend/api/routes.py#L44)). POST
/api/training/switch already performs graceful suspend then start
([backend/api/routes.py:120-134](../backend/api/routes.py#L120)).

TrainingControls already exposes the relevant choices and constructs the exact
start/switch body ([frontend/src/components/TrainingControls.tsx:13-59](../frontend/src/components/TrainingControls.tsx#L13)).
However, it owns that draft privately and its buttons execute immediately,
without a review or confirmation step
([frontend/src/components/TrainingControls.tsx:77-109](../frontend/src/components/TrainingControls.tsx#L77)).
The advisor is fetched but only displayed; the form remains hardcoded to large
([frontend/src/components/TrainingControls.tsx:13-18](../frontend/src/components/TrainingControls.tsx#L13),
[frontend/src/components/TrainingControls.tsx:114-131](../frontend/src/components/TrainingControls.tsx#L114)).

Resume is not a separate operation. POST /api/training/resume is unsupported
([backend/api/routes.py:212-221](../backend/api/routes.py#L212)); calling start
with fresh_start=false resolves the selected game’s catalog head, falls back to
a game-scoped checkpoint scan, or starts new if no head exists
([backend/training/trainer.py:332-354](../backend/training/trainer.py#L332)).
With fresh_start=true the trainer skips the head and wipes the non-restored
lineage replay directory
([backend/training/trainer.py:327-381](../backend/training/trainer.py#L327)).
That consequence is currently represented only by a checkbox labeled “Start
new model” ([frontend/src/components/TrainingControls.tsx:176-185](../frontend/src/components/TrainingControls.tsx#L176)).

### Proposed conversational flow

The flow below is a design option for a code-backed assistant, not a demand
that Qwen improvise the sequence.

1. **Resolve focus and readiness in code.** Start from focus_game_id. If it is a
   ROM-ready built-in without a workspace, explain that it can be promoted and
   onboarded but cannot train yet. If it is custom, return exact readiness
   blockers: ROM, configs, usable actions/reward, verified state, probe status,
   or brain compatibility. Do not ask parameter questions until trainable.
2. **Choose operation before parameters.**
   - No resumable head: mode is new-no-head; fresh_start=false naturally starts
     a new brain.
   - Head exists: recommend resume and show lineage/head step, checkpoint age,
     replay availability, and locked/compatible settings.
   - User explicitly wants a new brain: show exactly what will not be resumed
     and what replay will be removed, then require high-friction approval.
   - Another game is running: propose switch, name the game that will be
     suspended, and use the existing atomic switch route only after approval.
3. **Model size.** For a genuinely new brain, call the existing advisor and
   present its recommended and fitting sizes in plain language. Code chooses
   matching preset defaults. For resume, derive the head’s architecture from
   its resolved session config and treat it as locked unless a validated fork
   operation exists; Qwen must not guess it.
4. **Initial state or rotation.** Read GET /api/games/{id}, prefer its default,
   and present states by human label/group with code-supplied descriptions,
   readiness, and intended use. Existing custom metadata already supports
   label/group, and the environment supports plus/comma-separated per-episode
   rotations ([sheeprl/sheeprl/envs/retro_dreamer.py:64-74](../sheeprl/sheeprl/envs/retro_dreamer.py#L64),
   [sheeprl/sheeprl/envs/retro_dreamer.py:203-218](../sheeprl/sheeprl/envs/retro_dreamer.py#L203)).
   Recommend one verified default for a first run; offer a verified rotation
   only when its rationale is known.
5. **Replay ratio.** Default to the code’s 0.125 setting and explain it as
   gradient updates per environment step; offer the existing supported choices
   only if the user has a speed/sample-efficiency goal. The base default and
   unit conversion already live in TrainingConfig
   ([backend/training/config.py:33-41](../backend/training/config.py#L33)).
   The server, not Qwen, validates bounds and returns the recommendation.
6. **Number of environments.** Derive the initial value from the selected model
   preset: debug=4, small=6, medium=8, large=6, xl=4
   ([backend/training/config.py:74-88](../backend/training/config.py#L74)).
   On resume, preserve the replay-compatible value unless code proves a safe
   migration; the trainer rejects restored replay when num_envs/action count
   differ ([backend/training/trainer.py:356-379](../backend/training/trainer.py#L356)).
7. **Keep advanced defaults out of the novice dialogue.** Effective batch size,
   prefill, and other required fields still appear in the final plan, but the
   assistant asks about them only when there is a warning or an expert requests
   control.
8. **Review under a real human gate.** Render an immutable proposal card with
   game, operation (new/resume/switch), lineage/head, model size, state(s),
   replay ratio, num_envs, effective batch, replay consequence, warnings, and
   exact request body. Confirm and Cancel are UI controls; conversational text
   alone does not execute.
9. **Revalidate, execute, and verify.** Confirmation refreshes the studio
   revision, consumes a one-time plan, invokes the existing start or switch
   implementation, and then reports the actual returned mode/config plus fresh
   training status. If the live run or head changed, require a new plan.

### What exists versus what is new

| Concern | Already exists | New surface/data needed |
|---|---|---|
| Game and readiness | Game list/detail, ROM-ready classification, promotion, per-file configs, states ([backend/api/routes.py:335-406](../backend/api/routes.py#L335)). | Code-derived trainable/readiness blockers, config validity/revisions, persisted state/probe readiness, and a joined game summary. |
| Model recommendation | GET /api/advisor/model_size supplies recommended/fits ([backend/api/routes.py:156-176](../backend/api/routes.py#L156)). | Recommendation response joined with dependent batch/replay/num_env defaults and resume-architecture constraints. |
| State selection | Game detail has default, raw states, and label/group for annotated custom states ([backend/games/manager.py:174-204](../backend/games/manager.py#L174)). | Extend state descriptors or add a states-with-descriptions/training-options view: description, screenshot, verified-live, probe result/hash/time, recommended use/rotation. Validate every submitted state server-side. |
| Resume decision | Catalog heads and GET /api/workspaces; start with fresh_start=false resumes ([backend/catalog.py:96-126](../backend/catalog.py#L96), [backend/api/routes.py:179-209](../backend/api/routes.py#L179)). | Expose head/session resolved config, model architecture, replay compatibility, active lineage, and a truthful mode of resume/new-no-head/fresh. Clarify lineage semantics: fresh start currently attaches a new session to main rather than creating a first-class branch ([backend/training/trainer.py:495-520](../backend/training/trainer.py#L495)). |
| Parameter defaults | TrainingConfig and frontend choices already encode defaults/options ([backend/training/config.py:21-88](../backend/training/config.py#L21), [frontend/src/components/TrainingControls.tsx:141-174](../frontend/src/components/TrainingControls.tsx#L141)). | A read-only planner that resolves effective values and validates positive/ranged numbers and cross-field constraints. Today the UI always posts num_envs=6, overriding medium/xl/debug presets ([frontend/src/components/TrainingControls.tsx:19-24](../frontend/src/components/TrainingControls.tsx#L19), [frontend/src/components/TrainingControls.tsx:49-58](../frontend/src/components/TrainingControls.tsx#L49)). |
| Human approval | Primer policy requires an explicit yes, and start/switch routes exist. | An enforced two-stage proposal/confirm contract, plan revision/hash/expiry/idempotency, and an approval credential/capability held by the browser rather than exposed to Qwen. Current start executes immediately and the copilot has permission-bypassed Bash ([backend/api/routes.py:47-95](../backend/api/routes.py#L47), [backend/copilot.py:133-144](../backend/copilot.py#L133)). |
| Result and status | Start returns status/model/game/state/fresh; status returns live aggregate metrics ([backend/api/routes.py:111-117](../backend/api/routes.py#L111), [backend/api/routes.py:224-242](../backend/api/routes.py#L224)). | Return effective config, actual mode/head/lineage, plan id, and confirmation audit data; include these in live status with timestamps. |

One strong API option is a read-only POST /api/training/plan followed by a
UI-bound POST /api/training/plans/{id}/confirm. The plan service performs all
joins and decisions and returns effective_request, choices still needed,
warnings, readiness blockers, and a state revision. A lighter option is GET
/api/training/options?game_id=... plus a frontend-only approval card that calls
the existing start/switch route. The former is more enforceable and auditable;
the latter is a smaller first experiment but is not a complete human gate while
the unrestricted copilot can call mutation routes directly.

## 3. CENTRAL-DRIVER ROLE + FIRST SLICE

### Role options

| Option | Relationship to existing tabs | Tradeoff |
|---|---|---|
| **1. Contextual advisor with approval cards** | Copilot explains ambient state and proposes exact actions; the user still navigates Metrics, Live Play, Game Config, and Training Controls manually. | Smallest and safest change; easy to test. It remains a peer tab rather than the central driver, and its proposal form can duplicate TrainingControls. |
| **2. Typed studio conductor — RECOMMENDED** | Copilot becomes a persistent rail/provider. It may issue allowlisted reversible UI intents such as select_game, open_tab, focus_metric, and prefill_training_draft. State changes become immutable approval cards. Tabs remain the authoritative evidence/edit surfaces. | Meets “central driver” without brittle automation; visible, typed, testable, and auditable. Requires shared frontend state, a structured intent/event schema, a proposal broker, and conflict handling when the user edits after a proposal. |
| **3. Model-driven DOM/browser operator** | Qwen clicks selectors, types fields, and changes tabs as if it were the user. | Fastest wow-demo and broadest apparent reach, but brittle to markup, hidden state, races, and accessibility; hard to audit; especially poor for a slower 27B model. It can also bypass the intended gate. Reject for product architecture. |

The recommended contract is: **the copilot is the intent orchestrator; tabs are
inspection and edit surfaces. It may navigate and prefill, but it never silently
mutates.** Metrics and Live Play provide evidence it can open/highlight; Game
Config shows proposed diffs; Training Controls becomes a shared draft/manual
fallback and the home for approval details. The model should emit only typed
intent/proposal data validated by code, not DOM instructions.

Today that bridge does not exist. Copilot events are limited to
user/assistant/tool/meta/raw, sends contain only text, and TrainingControls owns
its draft privately
([frontend/src/components/CopilotPanel.tsx:7-13](../frontend/src/components/CopilotPanel.tsx#L7),
[frontend/src/components/CopilotPanel.tsx:68-77](../frontend/src/components/CopilotPanel.tsx#L68),
[frontend/src/components/TrainingControls.tsx:13-28](../frontend/src/components/TrainingControls.tsx#L13)).
Conditional tab rendering also unmounts Copilot when another tab opens
([frontend/src/App.tsx:85-132](../frontend/src/App.tsx#L85)). A conductor
therefore needs persistent/shared state rather than simulated clicks.

### Minimal first slice that proves the architecture

Keep the current tabs and existing start/switch implementations. Build one
vertical path: **ambient state -> deterministic training plan -> Qwen narration
-> typed approval -> existing mutation -> fresh status**.

1. Add a read-only StudioStateBuilder and compact GET /api/studio/state with
   focus_game_id. Compose existing game discovery, catalog/workspaces, live
   status/config, advisor, selected-game states/config summaries, and current
   tool jobs. Include revision and timestamps.
2. Pass selectedGame and status into CopilotPanel and include focus_game_id in
   send. Automatically inject the compact state slice on every turn. Display
   the observed time/revision so the user can see what “current” means.
3. Add one deterministic training-plan path for already-onboarded custom games.
   It decides new/resume/switch, applies advisor and TrainingConfig defaults,
   enumerates state labels, validates compatibility, and returns a typed
   training_start_proposal. Qwen only narrates the facts and asks unresolved
   preference questions.
4. Render one proposal card in Copilot: exact game, mode/head, model, initial
   state/rotation, replay ratio, num_envs, batch, consequences, and request.
   Confirm/Cancel is the only mutation point. Confirmation calls the existing
   POST /api/training/start or /api/training/switch through a UI-bound broker;
   Cancel changes nothing.
5. On success, refresh ambient status and emit the reversible UI intent
   open_tab(metrics). That proves the copilot can guide the interface without
   generic DOM control.

An acceptance demonstration is intentionally narrow: with an onboarded game
selected, “start training this” or “continue this game” must identify the right
game and head, recommend code-derived settings, explain available states, show
an exact review card, do nothing before Confirm, invoke the correct existing
route after Confirm, and show the resulting live run.

Defer full onboarding, arbitrary config edits, stop/promote approval flows,
durable chat/session transport, and generic UI automation. This first slice
delivers immediate value and tests the two architectural bets that matter:
ambient context is a code-owned projection, and the copilot drives through
typed, human-approved intents rather than model-authored shell/UI behavior.

