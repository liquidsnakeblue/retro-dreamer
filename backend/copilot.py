"""Resident copilot: a headless claude-local (Claude Code CLI against the
local Qwen proxy) session managed by the studio, streamed to the dashboard's
chat panel. The copilot drives the SAME HTTP tool layer as the UI and Claude —
the harness is a body, the tools are the studio's.

Qwen realities baked in (operational notes, 2026-07):
- vision requires the :8082 image-fix proxy (direct :6789 silently drops images)
- reasoning model: minutes-long thinking is normal — 900s API timeout, no
  output-token caps
- CLAUDE_CODE_ATTRIBUTION_HEADER=0 or the per-request header kills KV cache
- lean context: tools return distilled results; the 3090 box holds ~2 heavy
  conversations
"""

import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRIMER_PATH = PROJECT_ROOT / "backend" / "copilot_primer.md"
CONFIG_DIR = Path.home() / ".claude-local"
PROXY_SCRIPT = Path.home() / "lmstudio-proxy" / "proxy.py"
PROXY_ENV = Path.home() / "lmstudio-proxy" / ".env"
STUDIO_STATE_START = "<STUDIO_STATE>"
STUDIO_STATE_END = "</STUDIO_STATE>"

router = APIRouter(prefix="/api/copilot")

_lock = threading.Lock()
_proc: Optional[subprocess.Popen] = None
_events: list = []  # display events plus typed proposal cards
_seq = 0
_studio_state_builder = None
_served_reports: dict[str, str] = {}
_served_report_meta: dict[str, dict] = {}
_served_report_seq = 0

GROUNDING_CLAIMS_START = "<GROUNDING_CLAIMS>"
GROUNDING_CLAIMS_END = "</GROUNDING_CLAIMS>"
_SERVED_REPORT_LIMIT = 64
_WATCH_JOB_ID = re.compile(r"\bwatch_brain-[0-9a-f]{8}\b", re.IGNORECASE)
_GROUNDING_BLOCK = re.compile(
    re.escape(GROUNDING_CLAIMS_START)
    + r"(.*?)"
    + re.escape(GROUNDING_CLAIMS_END),
    re.DOTALL,
)
_REPORT_EVENT_LINE = re.compile(
    r"^\s*step\s+(\d+)\s+([A-Za-z][A-Za-z0-9_+/-]*)(?=\s).*$",
    re.MULTILINE,
)


# Secondary vocabulary telemetry only. Structured claim validation below is
# the deterministic mechanism; this deliberately incomplete lexicon is never
# treated as a gate or a coverage guarantee. Generic strategy/plan/decision/chose
# vocabulary is omitted because it describes legitimate planner flow.
_CAUSAL_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r"\b(?:hit|hits|hitting|collid(?:e[sd]?|ing)\s+with|bump(?:ed|ing)?\s+into|"
    r"crash(?:ed|ing)?\s+into|ramm?(?:ed|ing)?\s+into)\s+(?:an?\s+|the\s+)?"
    r"(?:obstacles?|walls?|enemies?|boss(?:es)?|traps?|hazards?|barricades?)\b",
    r"\b(?:sped|drove|ran|raced|slammed)\s+into\s+(?:an?\s+|the\s+)?"
    r"(?:obstacles?|walls?|enemies?|traps?|hazards?|barricades?)\b",
    r"\b(?:could(?:\s+not|n't)|can(?:\s+not|'t)|failed\s+to|fails?\s+to)?\s*"
    r"(?:avoid|dodge|evade)(?:d|s|ing)?\s+(?:an?\s+|the\s+)?"
    r"(?:obstacles?|walls?|enemies?|boss(?:es)?|traps?|hazards?)\b",
    r"\b(?:laps?\s+after\s+laps?|(?:completed?|finished?)\s+(?:\d+\s+|an?\s+)?laps?|"
    r"(?:no\s+)?laps?\s+(?:completed|finished)|\d+\s+(?:full\s+)?laps?|"
    r"loop(?:ed|ing)?\s+(?:the\s+)?track)\b",
    r"\b(?:health|regains?|recovery)\s+(?:means?|came\s+from|was|is|were|are|=)\s+"
    r"(?:an?\s+|the\s+)?pick-?ups?\b",
    r"\b(?:collect(?:ed|ing|s)?|pick(?:ed|ing)?\s+up|gather(?:ed|ing|s)?|"
    r"grab(?:bed|bing|s)?)\s+(?:coins?|items?|power-?ups?)\b",
))
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])(?:[\"'”’*_`]*)\s+|\n+")
_ADVERSATIVE_BREAK = re.compile(
    r"\s*(?:[,;]\s*)?\b(?:but|however|yet)\b\s*[:,]?\s*",
    re.IGNORECASE,
)
_NONASSERTION = re.compile(
    r"\b(?:report|capture|evidence|data)\s+(?:does\s+not|doesn't|cannot|can't)\s+"
    r"(?:say|show|support|establish|identify|prove)\b|"
    r"\b(?:i|we)\s+(?:cannot|can't|could(?:\s+not|n't))\s+"
    r"(?:infer|conclude|determine|tell|verify)\b|"
    r"\b(?:do\s+not|don't|never)\s+(?:claim|infer|assume|say)\b|"
    r"\b(?:no|insufficient)\s+evidence\b|"
    r"\b(?:unknown|unverified|unsupported|not\s+established)\b|"
    r"\b(?:check|see|find\s+out|test|determine)\s+(?:if|whether)\b",
    re.IGNORECASE,
)
_PLANNER_CONTEXT = re.compile(
    r"\b(?:training\s+plan|resume\s+strategy|approval\s+decision|proposal|"
    r"reward(?:s|ed|ing)?|penalt(?:y|ies)|replay\s+buffer|batch\s+size|"
    r"config(?:uration)?|prompt|wording|saying|claiming|planner|world\s+model|"
    r"recommend(?:ed|ing|ation)?|suggest(?:ed|ing|ion)?|curriculum|"
    r"(?:our|training)\s+goal)\b|"
    r"\b(?:should|could|may|might|can)\s+(?:learn|teach|train|"
    r"be\s+(?:taught|trained))\b|"
    r"\b(?:needs?\s+to|ought\s+to)\s+(?:learn|dodge|avoid|teach|train|"
    r"be\s+(?:taught|trained))\b|"
    r"^\s*(?:add|consider|propose|configure|test|try|train|teach|use)\b",
    re.IGNORECASE,
)
_MARKDOWN_LEAD = re.compile(r"^\s*(?:(?:[-+*]|\d+[.)]|>)\s+)+")
_MARKDOWN_DECORATION = re.compile(r"[*_`~]")


def set_studio_state_builder(builder):
    global _studio_state_builder
    _studio_state_builder = builder


def compose_user_message(text: str, studio_state: dict) -> str:
    envelope = json.dumps(studio_state, separators=(",", ":"), ensure_ascii=False)
    return f"{STUDIO_STATE_START}\n{envelope}\n{STUDIO_STATE_END}\n{text}"


def _looks_like_episode_report(value: object) -> bool:
    return (
        isinstance(value, str)
        and "EPISODE REPORT" in value
        and "EVENT STREAM" in value
        and "POST-MORTEM:" in value
    )


def cache_served_watch_report(job_id: str, report_text: str) -> None:
    """Cache the exact completed report when the jobs endpoint serves it."""
    global _served_report_seq
    if not _WATCH_JOB_ID.fullmatch(job_id) or not _looks_like_episode_report(report_text):
        return
    with _lock:
        _served_report_seq += 1
        _served_reports[job_id] = report_text
        _served_report_meta[job_id] = {
            "served_at": time.time(),
            "serve_seq": _served_report_seq,
        }
        if len(_served_reports) > _SERVED_REPORT_LIMIT:
            oldest = min(
                _served_reports,
                key=lambda item: _served_report_meta[item]["served_at"],
            )
            del _served_reports[oldest]
            del _served_report_meta[oldest]


def _served_report(job_id: str) -> Optional[str]:
    with _lock:
        return _served_reports.get(job_id)


def _served_report_receipt(job_id: str) -> tuple[Optional[str], int]:
    with _lock:
        report = _served_reports.get(job_id)
        meta = _served_report_meta.get(job_id)
        return report, meta["serve_seq"] if meta else 0


def _serve_sequence() -> int:
    with _lock:
        return _served_report_seq


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_BREAK.split(text) if part.strip()]


def _normalize_claim_sentence(text: str) -> str:
    normalized = _MARKDOWN_LEAD.sub("", text)
    normalized = _MARKDOWN_DECORATION.sub("", normalized)
    return " ".join(normalized.split()).strip()


def _check_grounding(text: str, report_text: str) -> list[dict]:
    """Return secondary lexical telemetry for concrete gameplay-causal prose."""
    del report_text  # Structured validation, not substring matching, is primary.
    warnings = []
    for sentence in _split_sentences(text):
        # A disclaimer only exempts its own clause. This preserves a warning
        # for "not established, but the brain hit a wall" while leaving
        # honest unknowns and planner talk quiet.
        for clause in _ADVERSATIVE_BREAK.split(sentence):
            clause = clause.strip()
            if not clause or _NONASSERTION.search(clause) or _PLANNER_CONTEXT.search(clause):
                continue
            terms = []
            for pattern in _CAUSAL_PATTERNS:
                terms.extend(match.group(0) for match in pattern.finditer(clause))
            if not terms:
                continue
            unique_terms = list(dict.fromkeys(term.casefold() for term in terms))
            warnings.append({
                "message": f"Grounding vocabulary telemetry: {clause}",
                "detail": f"Matched causal phrases: {', '.join(unique_terms)}",
            })
    return warnings


def _split_grounding_claims(text: str) -> tuple[str, Optional[dict], Optional[str]]:
    """Separate one machine-readable claims tail from user-facing prose."""
    first_start = text.find(GROUNDING_CLAIMS_START)
    if first_start < 0:
        return text, None, None
    matches = list(_GROUNDING_BLOCK.finditer(text))
    if not matches:
        return text[:first_start].rstrip(), None, "opening tag has no closing tag"
    visible = _GROUNDING_BLOCK.sub("", text).strip()
    if (
        len(matches) != 1
        or text.count(GROUNDING_CLAIMS_START) != 1
        or text.count(GROUNDING_CLAIMS_END) != 1
    ):
        return text[:first_start].rstrip(), None, "claims block must be the single final block"
    match = matches[0]
    if text[match.end():].strip():
        return visible, None, "claims block must be the single final block"
    raw = match.group(1).strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return visible, None, f"claims JSON is invalid: {exc.msg}"
    return visible, payload, None


def _report_event_lines(report_text: str) -> dict[tuple[int, str], list[str]]:
    events = {}
    for match in _REPORT_EVENT_LINE.finditer(report_text):
        key = (int(match.group(1)), match.group(2).casefold())
        events.setdefault(key, []).append(match.group(0).strip())
    return events


def _grounding_warning(message: str, detail: str) -> dict:
    return {"message": f"Grounding telemetry: {message}", "detail": detail}


def _validate_grounding_claims(
    payload: object,
    diagnosis_text: str,
    expected_job_ids: set[str],
) -> list[dict]:
    """Validate deterministic provenance against the job-keyed served report.

    Quote/anchor membership is code-checkable; whether the evidence semantically
    entails the claim remains part of the human-scored acceptance pass.
    """
    if not isinstance(payload, dict):
        return [_grounding_warning("claims payload is not an object", "Expected JSON object")]
    job_id = payload.get("job_id")
    claims = payload.get("claims")
    if not isinstance(job_id, str):
        return [_grounding_warning("claims payload has no job_id", "job_id must be a string")]
    if job_id not in expected_job_ids:
        return [_grounding_warning(
            f"claims reference {job_id} outside this diagnosis turn",
            f"Expected one of: {', '.join(sorted(expected_job_ids)) or 'none'}",
        )]
    report_text = _served_report(job_id)
    if report_text is None:
        return [_grounding_warning(
            f"report for {job_id} was not served",
            "The completed /api/tools/jobs result must be served before validation",
        )]
    if not isinstance(claims, list):
        return [_grounding_warning("claims is not a list", f"job_id={job_id}")]

    event_lines = _report_event_lines(report_text)
    diagnosis_sentences = {
        _normalize_claim_sentence(sentence)
        for sentence in _split_sentences(diagnosis_text)
    }
    warnings = []
    for index, item in enumerate(claims):
        label = f"claim[{index}]"
        if not isinstance(item, dict):
            warnings.append(_grounding_warning(f"{label} is not an object", job_id))
            continue
        claim = item.get("claim")
        quote = item.get("evidence_quote")
        anchor = item.get("anchor")
        if not isinstance(claim, str) or not claim.strip():
            warnings.append(_grounding_warning(f"{label} has no claim text", job_id))
            continue
        if _normalize_claim_sentence(claim) not in diagnosis_sentences:
            warnings.append(_grounding_warning(
                f"{label} is not an exact diagnosis sentence", claim
            ))
        if not isinstance(quote, str) or len(quote.strip()) < 8:
            warnings.append(_grounding_warning(
                f"{label} has no usable evidence_quote", claim
            ))
            continue
        if quote not in report_text:
            warnings.append(_grounding_warning(
                f"{label} evidence_quote is not verbatim", quote
            ))
        if not isinstance(anchor, dict):
            warnings.append(_grounding_warning(f"{label} has no anchor object", claim))
            continue
        step = anchor.get("step")
        event = anchor.get("event")
        if isinstance(step, bool) or not isinstance(step, int) or not isinstance(event, str):
            warnings.append(_grounding_warning(
                f"{label} anchor must contain integer step and string event",
                json.dumps(anchor, default=str),
            ))
            continue
        lines = event_lines.get((step, event.casefold()))
        if not lines:
            warnings.append(_grounding_warning(
                f"{label} anchor does not exist in the report",
                f"{event}@{step}",
            ))
        elif quote in report_text and not any(quote in line for line in lines):
            warnings.append(_grounding_warning(
                f"{label} evidence_quote does not belong to its anchor",
                f"{event}@{step}",
            ))
    return warnings


def _turn_grounding_warnings(
    texts: list[str],
    job_ids: set[str],
    payloads: list[object],
    parse_errors: list[str],
    payload_positions: Optional[list[int]] = None,
    last_assistant_position: Optional[int] = None,
) -> list[dict]:
    expected = {job_id for job_id in job_ids if _served_report(job_id) is not None}
    if not expected:
        return []
    warnings = [
        _grounding_warning("claims block could not be parsed", error)
        for error in parse_errors
    ]
    if len(expected) != 1:
        warnings.append(_grounding_warning(
            f"diagnosis turn served {len(expected)} watch reports",
            "Exactly one served report is supported per diagnosis turn",
        ))
    if len(payloads) != 1:
        warnings.append(_grounding_warning(
            f"expected exactly one structured claims block, received {len(payloads)}",
            "Append one final claims block for the served report",
        ))
    elif payload_positions is not None and (
        not payload_positions
        or payload_positions[0] != (
            len(texts) - 1
            if last_assistant_position is None
            else last_assistant_position
        )
    ):
        warnings.append(_grounding_warning(
            "structured claims block was not the final assistant text block",
            "No assistant prose may follow the claims tail",
        ))
    diagnosis_text = "\n".join(text for text in texts if text)
    referenced = set()
    for payload in payloads:
        if isinstance(payload, dict) and isinstance(payload.get("job_id"), str):
            referenced.add(payload["job_id"])
        warnings.extend(
            _validate_grounding_claims(payload, diagnosis_text, expected)
        )
    for missing in sorted(expected - referenced):
        warnings.append(_grounding_warning(
            f"diagnosis for {missing} omitted structured claims",
            f"Append {GROUNDING_CLAIMS_START} JSON {GROUNDING_CLAIMS_END}",
        ))

    # Vocabulary matching is deliberately secondary telemetry. Scan the whole
    # diagnosis, not only its final text block, so an earlier claim cannot be
    # concealed by a harmless sign-off.
    if diagnosis_text:
        report_text = _served_report(sorted(expected)[0]) or ""
        warnings.extend(_check_grounding(diagnosis_text, report_text))
    return warnings


def _tool_result_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    return ""


def _tool_result_contains_report(content: str, report_text: str) -> bool:
    # Bash may print the report directly or nested inside a JSON object, where
    # newlines are escaped. Either way the receipt must contain the exact text.
    escaped_ascii = json.dumps(report_text)[1:-1]
    escaped_unicode = json.dumps(report_text, ensure_ascii=False)[1:-1]
    return (
        report_text in content
        or escaped_ascii in content
        or escaped_unicode in content
    )


def _emit(kind: str, text: str, detail: str = None):
    global _seq
    with _lock:
        _seq += 1
        ev = {"seq": _seq, "ts": time.time(), "kind": kind, "text": text}
        if detail:
            ev["detail"] = detail
        _events.append(ev)
        del _events[:-500]


def emit_proposal(proposal: dict):
    """Publish code-authored proposal data without routing it through Qwen."""
    global _seq
    # Round-trip through JSON so neither the caller nor a dashboard consumer
    # can mutate the planner's copy by retaining a shared nested object.
    safe_proposal = json.loads(json.dumps(proposal))
    with _lock:
        _seq += 1
        _events.append({
            "seq": _seq,
            "ts": time.time(),
            "kind": "proposal",
            "proposal": safe_proposal,
        })
        del _events[:-500]


def _ensure_proxy():
    import urllib.request

    try:
        urllib.request.urlopen("http://localhost:8082/", timeout=2)
        return
    except Exception:
        pass
    env = os.environ.copy()
    if PROXY_ENV.exists():
        for line in PROXY_ENV.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    subprocess.Popen(
        ["python3", str(PROXY_SCRIPT)], env=env,
        stdout=open("/tmp/lmstudio-proxy.log", "ab"),
        stderr=subprocess.STDOUT,
    )
    time.sleep(1.5)


def _reader(proc: subprocess.Popen):
    """Parse the CLI's stream-json output into display events."""
    turn_texts = []
    turn_job_ids = set()
    turn_payloads = []
    turn_payload_positions = []
    turn_parse_errors = []
    turn_bash_uses = {}
    turn_assistant_position = -1
    turn_start_serve_seq = _serve_sequence()
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            _emit("raw", line[:500])
            continue
        t = ev.get("type")
        if t == "assistant":
            for block in (ev.get("message") or {}).get("content", []):
                if block.get("type") == "text" and block.get("text"):
                    turn_assistant_position += 1
                    text = block["text"]
                    visible, payload, parse_error = _split_grounding_claims(text)
                    turn_texts.append(visible)
                    if payload is not None:
                        turn_payloads.append(payload)
                        turn_payload_positions.append(turn_assistant_position)
                    if parse_error:
                        turn_parse_errors.append(parse_error)
                    if visible:
                        # Preserve the exact model block in raw event detail
                        # while keeping its machine tail out of the chat bubble.
                        _emit(
                            "assistant",
                            visible,
                            detail=text if payload is not None or parse_error else None,
                        )
                elif block.get("type") == "tool_use":
                    turn_assistant_position += 1
                    name = block.get("name", "?")
                    inp = block.get("input") or {}
                    command = inp.get("command", "")
                    if name == "Bash" and isinstance(command, str) and block.get("id"):
                        turn_bash_uses[block["id"]] = set(_WATCH_JOB_ID.findall(command))
                    # One readable line up front; full input as expandable detail
                    if name == "Bash":
                        label = inp.get("description") or inp.get("command", "").split("\n")[0][:120]
                        detail = inp.get("command", "")
                    elif "file_path" in inp:
                        label = inp["file_path"]
                        detail = "" if name == "Read" else json.dumps(inp, indent=2)[:4000]
                    else:
                        blob = json.dumps(inp)
                        label = blob[:120]
                        detail = json.dumps(inp, indent=2)[:4000] if len(blob) > 120 else ""
                    _emit("tool", f"{name} — {label}", detail=detail)
        elif t == "user":
            for block in (ev.get("message") or {}).get("content", []):
                if block.get("type") != "tool_result" or block.get("is_error"):
                    continue
                command_job_ids = turn_bash_uses.get(block.get("tool_use_id"))
                if command_job_ids is None:
                    continue
                content = _tool_result_text(block.get("content"))
                if not _looks_like_episode_report(content):
                    continue
                candidates = command_job_ids | set(_WATCH_JOB_ID.findall(content))
                for job_id in candidates:
                    report_text, serve_seq = _served_report_receipt(job_id)
                    if (
                        report_text is not None
                        and serve_seq > turn_start_serve_seq
                        and _tool_result_contains_report(content, report_text)
                    ):
                        turn_job_ids.add(job_id)
        elif t == "result":
            for warning in _turn_grounding_warnings(
                turn_texts,
                turn_job_ids,
                turn_payloads,
                turn_parse_errors,
                turn_payload_positions,
                turn_assistant_position,
            ):
                _emit(
                    "grounding-warning",
                    warning["message"],
                    detail=warning["detail"],
                )
            _emit("meta", f"turn done ({ev.get('num_turns', '?')} turns, "
                          f"{ev.get('duration_ms', 0) / 1000:.0f}s)")
            turn_texts.clear()
            turn_job_ids.clear()
            turn_payloads.clear()
            turn_payload_positions.clear()
            turn_parse_errors.clear()
            turn_bash_uses.clear()
            turn_assistant_position = -1
            turn_start_serve_seq = _serve_sequence()
        elif t == "system" and ev.get("subtype") == "init":
            _emit("meta", f"session ready (model {ev.get('model', '?')})")
    _emit("meta", "copilot session ended")


class StartReq(BaseModel):
    resume: bool = False  # future: resume a prior session id


@router.post("/start")
def start(req: StartReq = None):
    global _proc, _events
    with _lock:
        if _proc is not None and _proc.poll() is None:
            return {"status": "already_running"}
        # Clear history but do NOT reset _seq: pollers track last-seen seq,
        # and a rewind makes them silently ignore the new session.
        _events = []
    _ensure_proxy()
    env = os.environ.copy()
    env["CLAUDE_CONFIG_DIR"] = str(CONFIG_DIR)
    env["ANTHROPIC_BASE_URL"] = "http://localhost:8082"
    env["ANTHROPIC_API_KEY"] = "sk-no-key-required"
    env.pop("ANTHROPIC_AUTH_TOKEN", None)
    env["CLAUDE_CODE_ATTRIBUTION_HEADER"] = "0"
    env["API_TIMEOUT_MS"] = "900000"  # reasoning model thinks in minutes
    cmd = [
        "claude",
        "--setting-sources", "user",
        "--model", "qwen3.6-27b",
        "--disallowedTools", "WebSearch,WebFetch",
        "--append-system-prompt-file", str(PRIMER_PATH),
        "--dangerously-skip-permissions",
        "-p",
        "--input-format", "stream-json",
        "--output-format", "stream-json",
        "--verbose",
    ]
    proc = subprocess.Popen(
        cmd, cwd=str(PROJECT_ROOT), env=env,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=open("/tmp/retro-copilot.err.log", "ab"),
        text=True, bufsize=1,
    )
    with _lock:
        _proc = proc
    threading.Thread(target=_reader, args=(proc,), daemon=True).start()
    _emit("meta", "copilot starting (Qwen 3.6 27B via claude-local, headless)")
    return {"status": "started", "pid": proc.pid}


class SendReq(BaseModel):
    text: str
    focus_game_id: Optional[str] = None
    active_tab: Optional[str] = None


@router.post("/send")
def send(req: SendReq):
    if _proc is None or _proc.poll() is not None:
        raise HTTPException(409, "copilot not running — POST /api/copilot/start first")
    if _studio_state_builder is None:
        raise HTTPException(500, "Studio state builder not initialized")
    studio_state = _studio_state_builder.build(
        req.focus_game_id, active_tab=req.active_tab, projection="compact"
    )
    enveloped_text = compose_user_message(req.text, studio_state)
    msg = {
        "type": "user",
        "message": {"role": "user", "content": [{"type": "text", "text": enveloped_text}]},
    }
    # The API event is the auditable raw user payload; the dashboard strips
    # only the anchored studio-state envelope before rendering the bubble.
    _emit("user", enveloped_text)
    try:
        _proc.stdin.write(json.dumps(msg) + "\n")
        _proc.stdin.flush()
    except (BrokenPipeError, OSError) as exc:
        raise HTTPException(500, f"copilot stdin write failed: {exc}")
    return {
        "status": "sent",
        "studio_revision": studio_state["revision"],
        "observed_at": studio_state["generated_at"],
    }


@router.get("/events")
def events(since: int = 0):
    with _lock:
        evs = [e for e in _events if e["seq"] > since]
        running = _proc is not None and _proc.poll() is None
        last_seq = _seq
    return {"running": running, "events": evs, "last_seq": last_seq}


@router.post("/stop")
def stop():
    global _proc
    with _lock:
        if _proc is not None and _proc.poll() is None:
            _proc.terminate()
            try:
                _proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _proc.kill()
        _proc = None
    _emit("meta", "copilot stopped")
    return {"status": "stopped"}
