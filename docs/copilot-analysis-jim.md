# Built-in Copilot: Current Architecture, Capabilities, and Limits

Grounding date: 2026-07-13. This is an as-built reading of the repository and
the current claude-local user settings. It describes what exists today, not the
intended end state.

## 1. ARCHITECTURE & FLOW

### End-to-end request path

~~~text
CopilotPanel
  POST /api/copilot/start
    -> FastAPI starts one headless claude process
    -> claude-local sends Qwen requests through localhost:8082
    -> backend/copilot_primer.md is appended to the system prompt

  POST /api/copilot/send {"text": "..."}
    -> one stream-JSON user message is written to the same process stdin
    -> Qwen may use Claude Code tools, normally Bash + curl
    -> curl calls the studio's /api/* and /api/tools/* routes on :8091
    -> long tool jobs run as CPU-only subprocesses and are polled by job id

  claude stream-JSON stdout
    -> backend reader converts selected records to UI events
    -> 500-entry in-memory event ring
    -> CopilotPanel polls GET /api/copilot/events?since=N every second
~~~

The normal API router, tool router, and copilot router are mounted into the same
FastAPI application ([backend/server.py:157-160](../backend/server.py#L157)).
The copilot router owns the /api/copilot prefix
([backend/copilot.py:32](../backend/copilot.py#L32)).

### Process launch

POST /api/copilot/start checks the single module-global process, clears the
backend display history, ensures the image-fix proxy, then launches claude in
the repository root ([backend/copilot.py:116-155](../backend/copilot.py#L116)).
The environment points Claude Code at ~/.claude-local and the local proxy,
removes token auth, disables the attribution header, and sets a 900,000 ms API
timeout ([backend/copilot.py:126-132](../backend/copilot.py#L126)).

The exact command assembled by the code is:

~~~text
claude
  --setting-sources user
  --model qwen3.6-27b
  --disallowedTools WebSearch,WebFetch
  --append-system-prompt-file backend/copilot_primer.md
  --dangerously-skip-permissions
  -p
  --input-format stream-json
  --output-format stream-json
  --verbose
~~~

See [backend/copilot.py:133-150](../backend/copilot.py#L133). The requested
model is therefore Qwen 3.6 27B; the :8082 proxy determines what ultimately
serves that name. The primer is explicitly appended, not used as a replacement
system prompt. Project skills exist under .claude/skills and the as-built notes
call them copilot skills ([docs/studio-v2-as-built.md:29-31](studio-v2-as-built.md#L29)),
but the launcher does not explicitly append or invoke those files; it relies on
Claude Code's project-skill discovery.

One source comment says there are no output-token caps
([backend/copilot.py:6-12](../backend/copilot.py#L6)), but the user settings
selected by the launch currently cap output at 16,384 tokens, cap thinking at
12,288, allow one concurrent tool use, disable auto-memory, and cap context at
186,607 tokens
([~/.claude-local/settings.json:10-24](/home/liquidsnakeblue/.claude-local/settings.json#L10)).

### Copilot management endpoints and event flow

| Endpoint | Current behavior |
|---|---|
| POST /api/copilot/start | Optional body has resume=false, but resume is a future stub. A live child returns already_running; otherwise a new child is started and its PID returned ([backend/copilot.py:112-155](../backend/copilot.py#L112)). |
| POST /api/copilot/send | Requires {"text": string}; emits a user event, writes one stream-JSON message to the live child's stdin, flushes, and returns sent. A dead process gives HTTP 409 ([backend/copilot.py:158-176](../backend/copilot.py#L158)). |
| GET /api/copilot/events?since=N | Returns running, retained events with seq greater than N, and last_seq ([backend/copilot.py:179-184](../backend/copilot.py#L179)). |
| POST /api/copilot/stop | Terminates, waits up to ten seconds, kills if needed, clears the tracked process, and emits stopped ([backend/copilot.py:187-199](../backend/copilot.py#L187)). |

The reader handles malformed JSON as raw output, extracts assistant text and
tool-use blocks, and turns init/result/end records into meta events
([backend/copilot.py:73-109](../backend/copilot.py#L73)). It does not surface
tool-result bodies or structured result errors; the result event becomes only
"turn done." Stderr goes to /tmp/retro-copilot.err.log rather than the chat
([backend/copilot.py:145-149](../backend/copilot.py#L145)). No
--include-partial-messages flag is passed, so this is completed-message/event
streaming rather than token streaming.

Events are process-memory objects shaped as seq, ts, kind, text, and optional
detail. The ring retains 500 entries
([backend/copilot.py:34-48](../backend/copilot.py#L34)). It is not a transcript
database.

### Dashboard behavior

CopilotPanel owns local events, running, draft, and start-busy state
([frontend/src/components/CopilotPanel.tsx:19-25](../frontend/src/components/CopilotPanel.tsx#L19)).
It polls events every second, detects a sequence rewind after a server restart,
and retains only the newest 400 UI events
([frontend/src/components/CopilotPanel.tsx:27-49](../frontend/src/components/CopilotPanel.tsx#L27)).
Start sends an empty object; send transmits only the user's text; stop has no
body ([frontend/src/components/CopilotPanel.tsx:55-77](../frontend/src/components/CopilotPanel.tsx#L55)).
Assistant output is rendered as Markdown, while tool calls are compact,
expandable rows ([frontend/src/components/CopilotPanel.tsx:139-200](../frontend/src/components/CopilotPanel.tsx#L139)).

The dashboard has a selectedGame state and passes it to training controls and
the config editor, but CopilotPanel receives no selected-game prop
([frontend/src/App.tsx:52-56](../frontend/src/App.tsx#L52),
[frontend/src/App.tsx:122-131](../frontend/src/App.tsx#L122)). Switching away
from the Copilot tab unmounts the panel, so its draft and local state disappear;
on remount it reconstructs what it can from the backend ring.

## 2. WHAT IT CAN PERCEIVE & DO

### Static knowledge and workflow guidance

The 397-line primer tells the model it is a game-onboarding and run-diagnosis
assistant and makes the document authoritative over prior knowledge
([backend/copilot_primer.md:1-12](../backend/copilot_primer.md#L1)). It includes:

- novice intent translation, game-name resolution, and explicit confirmation
  before destructive or costly operations
  ([backend/copilot_primer.md:47-90](../backend/copilot_primer.md#L47));
- custom workspace layout and exact data.json, actions.json, training.json, and
  metadata.json schemas
  ([backend/copilot_primer.md:92-242](../backend/copilot_primer.md#L92));
- tool descriptions and expected result interpretation
  ([backend/copilot_primer.md:244-303](../backend/copilot_primer.md#L244));
- training controls with a natural-language human gate
  ([backend/copilot_primer.md:304-319](../backend/copilot_primer.md#L304));
- known failure patterns, hard rules, onboarding order, and diagnosis order
  ([backend/copilot_primer.md:321-397](../backend/copilot_primer.md#L321)).

The three project skills reinforce narrower workflows: onboarding sequences
actions, states, RAM discovery, reward design, and probing
([.claude/skills/onboard-a-game/SKILL.md:12-49](../.claude/skills/onboard-a-game/SKILL.md#L12));
reward audit looks for wrap, reset, asymmetry, done-rule, and farming exploits
([.claude/skills/audit-a-reward/SKILL.md:13-38](../.claude/skills/audit-a-reward/SKILL.md#L13));
run diagnosis starts with status/log trends, then recorded behavior, then reward
checks ([.claude/skills/diagnose-a-run/SKILL.md:11-37](../.claude/skills/diagnose-a-run/SKILL.md#L11)).

### Effective Claude Code tools

The repository does not enforce an allowlist. It explicitly disallows only
WebSearch and WebFetch and skips all permission checks
([backend/copilot.py:133-144](../backend/copilot.py#L133)). Consequently the
effective tool set is broader than the advertised HTTP layer: it includes
whatever non-denied tools the installed Claude Code/user configuration exposes.
The event formatter explicitly anticipates at least Bash and Read
([backend/copilot.py:89-103](../backend/copilot.py#L89)), and the as-built notes
acknowledge full Bash access as unacceptable for a product
([docs/studio-v2-as-built.md:95-97](studio-v2-as-built.md#L95)).

The primer directs the model to use Bash + curl against localhost:8091
([backend/copilot_primer.md:244-253](../backend/copilot_primer.md#L244)).
Nothing in code prevents direct filesystem reads/writes or arbitrary shell
commands; the instruction to use canonical config APIs and obtain approval is
prompt policy, not an enforcement boundary.

### Exact studio tool endpoints

All tool POSTs return a job_id. The job manager starts a CPU-only subprocess in
a daemon thread, writes combined output to a log, and parses a final
RESULT JSON line ([backend/tools.py:36-83](../backend/tools.py#L36)).

| Endpoint | Accepted request / result purpose |
|---|---|
| POST /api/tools/reward_probe | game_id, states, steps=400, actions="all"; runs each constant action and reports reward/done consistency ([backend/tools.py:97-110](../backend/tools.py#L97)). |
| POST /api/tools/ram_capture | game_id, state, steps=2000, checkpoint="head"; resolves the game's catalog head and records full RAM ([backend/tools.py:113-138](../backend/tools.py#L113)). |
| POST /api/tools/ram_diff | window=60 and captures containing npz/event_step; intersects boundary changes ([backend/tools.py:141-151](../backend/tools.py#L141)). |
| POST /api/tools/build_state | game_id, plan, out_state_name, optional start_state; scripts menu inputs and writes a state/screenshots ([backend/tools.py:154-175](../backend/tools.py#L154)). |
| POST /api/tools/run_walker | game_id, start_state, n_captures, checkpoint, flag/live value, tap button, prefix; lets a trained policy earn progression states ([backend/tools.py:178-201](../backend/tools.py#L178)). |
| POST /api/tools/record_episode | game_id, state, seconds=60, checkpoint="latest"; records an MP4 ([backend/tools.py:204-219](../backend/tools.py#L204)). |
| GET /api/tools/jobs/{job_id} | Returns in-memory job state, parsed result, and a log tail ([backend/tools.py:222-233](../backend/tools.py#L222)). |
| GET /api/tools/jobs | Lists current in-memory jobs ([backend/tools.py:236-238](../backend/tools.py#L236)). |

The primer also tells it to call:

- GET /api/workspaces, GET /api/training/status, GET /api/games, GET
  /api/games/{id}, POST /api/games/promote, and GET/PUT game config
  ([backend/copilot_primer.md:290-303](../backend/copilot_primer.md#L290));
- GET /api/advisor/model_size and POST /api/training/start, /stop, /suspend,
  and /switch ([backend/copilot_primer.md:304-319](../backend/copilot_primer.md#L304));
- POST /api/games/import during onboarding
  ([backend/copilot_primer.md:372-392](../backend/copilot_primer.md#L372));
- GET /api/training/logs through the diagnosis skill
  ([.claude/skills/diagnose-a-run/SKILL.md:13-16](../.claude/skills/diagnose-a-run/SKILL.md#L13)).

Those routes exist in
[backend/api/routes.py:47-242](../backend/api/routes.py#L47) and
[backend/api/routes.py:306-448](../backend/api/routes.py#L306). Because the
model has Bash, this list is guidance rather than a technical ceiling; it can
discover or call other locally reachable routes.

### Perception, vision, context, and memory

- **Game/training state:** no live state is injected at start or before a turn.
  Start receives {}, send receives only text, and the panel supplies no selected
  game. The copilot knows the current run only after it chooses to curl status,
  workspaces, logs, game detail, and configs. It is not subscribed to the
  metrics WebSocket and has no automatic awareness when the run changes.
- **Vision:** the launch path is designed to preserve image input: comments say
  vision requires the :8082 image-fix proxy, and start ensures that proxy
  ([backend/copilot.py:6-8](../backend/copilot.py#L6),
  [backend/copilot.py:51-70](../backend/copilot.py#L51)). The workflows tell it
  to Read saved screenshots and to extract/read frames from recorded episodes
  ([.claude/skills/onboard-a-game/SKILL.md:25-27](../.claude/skills/onboard-a-game/SKILL.md#L25),
  [.claude/skills/diagnose-a-run/SKILL.md:19-22](../.claude/skills/diagnose-a-run/SKILL.md#L19)).
  This is on-demand file vision, not a live game/video feed.
- **Session continuity:** successive sends share the same long-lived claude
  process and therefore its in-process conversation. Continuity ends when that
  child or the server ends. StartReq.resume is explicitly unimplemented
  ([backend/copilot.py:112-114](../backend/copilot.py#L112)); there is no session
  ID in events or API responses, and current user settings disable Claude Code
  auto-memory
  ([~/.claude-local/settings.json:19-24](/home/liquidsnakeblue/.claude-local/settings.json#L19)).
- **Persistent memory:** none at the studio layer. The event ring is volatile
  display history, not model memory; it is cleared for a new child and lost on
  server restart. Claude Code may leave its own history files, but this code
  neither resumes nor retrieves them. The as-built notes confirm no restart
  survival or resume support
  ([docs/studio-v2-as-built.md:50-57](studio-v2-as-built.md#L50)).

## 3. FRICTION / FAILURE POINTS VISIBLE IN THE CODE

1. **It is context-blind at the moment the user asks.** "What is wrong with
   this run?" carries neither the dashboard's selected game nor a fresh run
   snapshot. The model must infer which game the user means and remember to
   fetch all relevant sources. That adds latency, wastes Qwen's limited context
   and tool budget, and makes stale conversational context easy to mistake for
   current state
   ([frontend/src/App.tsx:52-56](../frontend/src/App.tsx#L52),
   [frontend/src/App.tsx:122-131](../frontend/src/App.tsx#L122),
   [frontend/src/components/CopilotPanel.tsx:55-77](../frontend/src/components/CopilotPanel.tsx#L55)).

2. **The primer and actual tool contracts have already drifted.**
   The primer says jobs transition queued -> running -> done|error
   ([backend/copilot_primer.md:249-253](../backend/copilot_primer.md#L249));
   code starts at running and terminates as done|failed
   ([backend/tools.py:60-80](../backend/tools.py#L60)). The primer advertises
   random RAM capture before a brain exists
   ([backend/copilot_primer.md:270-272](../backend/copilot_primer.md#L270)),
   but RamCaptureReq has no random field and the script always loads a
   checkpoint-backed player
   ([backend/tools.py:113-138](../backend/tools.py#L113),
   [sheeprl/_retro_ram_capture.py:50-79](../sheeprl/_retro_ram_capture.py#L50)).
   That breaks the advertised new-game RAM-discovery path precisely when no
   trained brain exists. Also, record_episode accepts game_id but never uses it;
   "latest" resolves a global watch head and can record the wrong game
   ([backend/tools.py:204-219](../backend/tools.py#L204),
   [sheeprl/_retro_record.py:55-83](../sheeprl/_retro_record.py#L55)).

3. **Long, opaque latency makes correct work look hung.** The implementation
   explicitly treats minutes-long reasoning as normal and allows a 15-minute
   API timeout ([backend/copilot.py:6-12](../backend/copilot.py#L6)); tool jobs
   themselves take minutes
   ([backend/copilot_primer.md:249-253](../backend/copilot_primer.md#L249)).
   The current user config permits only one tool use at a time
   ([~/.claude-local/settings.json:16-23](/home/liquidsnakeblue/.claude-local/settings.json#L16)).
   The UI exposes process-level running/offline only, not queued, thinking,
   current tool, elapsed time, or cancel-this-turn
   ([frontend/src/components/CopilotPanel.tsx:20-24](../frontend/src/components/CopilotPanel.tsx#L20),
   [frontend/src/components/CopilotPanel.tsx:80-100](../frontend/src/components/CopilotPanel.tsx#L80)).
   Even the as-built smoke test for a simple workspace curl took about 25
   seconds ([docs/studio-v2-as-built.md:47-48](studio-v2-as-built.md#L47)).

4. **The 27B model is being asked to compensate for missing product
   structure.** A long, monolithic primer must teach schemas, safety, game
   semantics, workflow selection, tool syntax, and conversational style at once.
   The local model then has to plan arbitrary Bash/curl sequences and interpret
   mostly unstructured logs. The primer itself admits a ceiling: after two
   failed tool-driven rounds it must recommend a frontier model rather than
   continue ([backend/copilot_primer.md:355-370](../backend/copilot_primer.md#L355)).
   There is no automatic model routing, stronger-model fallback, deterministic
   preflight, or task state machine. This does not prove that every Qwen answer
   is poor; it shows the architecture puts high-reliability orchestration work
   on the model it already expects may need escalation.

5. **The event transport can lose or duplicate information.** The events route
   snapshots events under the lock but reads _seq after releasing it; an event
   emitted between those operations may be absent from the response but
   included in last_seq. The UI advances to last_seq and will never request that
   event ([backend/copilot.py:179-184](../backend/copilot.py#L179),
   [frontend/src/components/CopilotPanel.tsx:40-42](../frontend/src/components/CopilotPanel.tsx#L40)).
   Conversely, an async setInterval can overlap requests based on the same
   cursor, append duplicates/out of order, and collide on seq-only React keys
   ([frontend/src/components/CopilotPanel.tsx:27-48](../frontend/src/components/CopilotPanel.tsx#L27),
   [frontend/src/components/CopilotPanel.tsx:104-112](../frontend/src/components/CopilotPanel.tsx#L104)).

6. **Failures are silent and prompts can be lost.** Poll exceptions are ignored.
   Start, stop, and send do not check response.ok or render server errors.
   Send clears the user's draft before the request succeeds
   ([frontend/src/components/CopilotPanel.tsx:27-77](../frontend/src/components/CopilotPanel.tsx#L27)).
   Backend tool-result details are hidden, and CLI stderr lives only in a temp
   log. A user sees little distinction among "thinking," "tool failed," "proxy
   failed," and "process died."

7. **Displayed history implies continuity that the model does not have.**
   Starting a new child clears the backend ring but preserves sequence numbers
   ([backend/copilot.py:119-125](../backend/copilot.py#L119)). The mounted UI
   clears only on sequence rewind, so a stop/start leaves old messages directly
   above the new memory-less session. There is no session boundary or ID.
   Backend and frontend also silently truncate at different sizes (500 versus
   400), and the resume flag is a stub
   ([frontend/src/components/CopilotPanel.tsx:32-43](../frontend/src/components/CopilotPanel.tsx#L32)).

8. **Tool jobs and process lifecycle are fragile.** Jobs have no cancellation
   or timeout and live only in the _jobs dictionary; the docstring's jobs.json
   persistence was never implemented
   ([backend/tools.py:1-9](../backend/tools.py#L1),
   [backend/tools.py:32-83](../backend/tools.py#L32),
   [docs/studio-v2-as-built.md:50-54](studio-v2-as-built.md#L50)).
   Concurrent starts can both pass the liveness check before either stores its
   process, send races with stop, and server shutdown does not explicitly stop
   the copilot
   ([backend/copilot.py:119-153](../backend/copilot.py#L119),
   [backend/copilot.py:162-175](../backend/copilot.py#L162),
   [backend/server.py:113-118](../backend/server.py#L113)).

9. **Safety is advisory, not enforced.** The primer's human gate is good policy,
   but the backend does not require an approval token for config writes,
   training start/stop/switch, or arbitrary Bash. The server binds to 0.0.0.0,
   allows wildcard CORS, and has no route authentication
   ([backend/server.py:141-155](../backend/server.py#L141),
   [backend/server.py:200-205](../backend/server.py#L200)). Combined with
   --dangerously-skip-permissions, a mistaken or adversarial prompt has a much
   larger blast radius than "the same studio tools as the UI."

10. **The primer is deployment-coupled.** It hardcodes the studio at :8091
    ([backend/copilot_primer.md:244-247](../backend/copilot_primer.md#L244)),
    while server.py defaults to :8080 unless RETRO_DREAMER_PORT overrides it
    ([backend/server.py:200-205](../backend/server.py#L200)). The present
    deployment may be correct, but another launch configuration silently breaks
    every instructed curl call.

## 4. CONCRETE IMPROVEMENT DIRECTIONS

| Direction | Effort | Concrete shape and expected payoff |
|---|---:|---|
| **Inject an authoritative context envelope on every turn** | M | Pass selectedGame into CopilotPanel and have the backend attach a timestamped, compact snapshot: selected game, live training status, workspace/head, states, current config hashes, and last episode summary. Add one-click intents such as "Diagnose this run" and "Audit this reward." The model starts from the user's actual object instead of spending its first several calls resolving "this." |
| **Repair and type the studio tool contract** | M | Fix random pre-brain RAM capture, make record_episode resolve the requested game's head, standardize terminal states, add job cancel/timeouts/persistence, and expose structured progress/result/error objects. Generate model-facing schemas from the same Pydantic/OpenAPI definitions and add contract tests so the primer cannot drift from code. |
| **Replace unrestricted Bash with a gated studio action adapter** | L | Give the copilot explicit read tools and mutation tools instead of arbitrary curl/shell. Read operations can run directly; config diffs and training control return a proposed action that the UI must approve. This improves Qwen reliability through typed arguments while turning the primer's human gate into an enforcement boundary. |
| **Build a real session and turn transport** | M | Add session_id and turn_id, persist transcripts/summaries, implement resume, and use SSE/WebSocket (or an atomic long-poll cursor) for ordered events. Surface queued/thinking/tool/failed/done state, timestamps, elapsed time, tool outputs, retry, and per-turn cancel. Check HTTP status before clearing drafts and prevent overlapping polls/sends. |
| **Route common workflows deterministically and tier the model** | L | Implement the three existing skills as server-side evidence pipelines: code gathers status/config/log/video/probe evidence in a known order, then Qwen interprets the compact result. Keep Qwen for grounded summaries and routine operations; automatically offer a stronger local/frontier fallback for ambiguous planning or after the existing two-round escalation threshold. This makes model quality a tiered product choice rather than the sole control plane. |

