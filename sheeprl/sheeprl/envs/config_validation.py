"""Validation for the studio's per-game config files (training.json,
actions.json). Deliberately dependency-free (stdlib only): the training
wrapper imports it as a package module, and the backend server loads it by
FILE PATH (importlib) to validate configs at API write time — so a broken
config bounces back to its author (human or copilot) with a fix-it message
instead of silently training a dead agent.
"""
from typing import Dict

OP_ALIASES = {
    "less-than": "less-than", "<": "less-than", "lt": "less-than",
    "greater-than": "greater-than", ">": "greater-than", "gt": "greater-than",
    "equal": "equal", "==": "equal", "=": "equal", "eq": "equal",
}

_REWARD_MODES = {"binary", "quadratic", "linear", "exponential"}
_REWARD_VAR_KEYS = {
    "reward", "penalty", "heal_reward", "delta", "wrap", "max_delta",
    "mode", "op", "reference", "max_speed", "max_value", "base_reward",
    "scaling_coefficient", "power", "min_threshold",
}
_DONE_VAR_KEYS = {"op", "reference", "success"}
_MILESTONE_KEYS = {"var", "op", "reference", "reward"}
_NOVELTY_KEYS = {"keys", "reward"}
_COUNTER_KEYS = {"var", "context", "reward", "decay", "max_per_context", "max_event_delta"}


def validate_training_config(game_id: str, cfg: dict, data_vars=None) -> None:
    """Reject training.json configs the reward/done engine would silently
    ignore. Every error message states the fix — these are read by humans
    and by the copilot.

    data_vars: optional set of the game's data.json variable names. When
    provided, milestone 'var' and novelty 'keys' entries are checked against
    it — a typo'd name would otherwise validate and then NEVER pay (a dead
    quest breadcrumb looks identical to "agent hasn't gotten there yet")."""
    problems = []

    for name, var in (cfg.get("reward", {}).get("variables", {}) or {}).items():
        where = f"reward.variables.{name}"
        unknown = set(var) - _REWARD_VAR_KEYS
        if unknown:
            problems.append(
                f"{where}: unknown key(s) {sorted(unknown)} — recognized keys: "
                f"{sorted(_REWARD_VAR_KEYS)}. For 'pay per unit gained' use "
                f'{{"reward": <coeff>}} (+ optional "delta":"signed", "wrap", '
                f'"max_delta"); for damage use {{"penalty": <coeff>}}.'
            )
        mode = var.get("mode")
        if mode is not None and mode not in _REWARD_MODES:
            problems.append(
                f"{where}: unknown mode '{mode}' — valid modes: "
                f"{sorted(_REWARD_MODES)}. Delta rewards use NO mode key."
            )
        if "op" in var and var["op"] not in OP_ALIASES:
            problems.append(f"{where}: unknown op '{var['op']}' — use one of "
                            f"{sorted(set(OP_ALIASES.values()))} (or <, >, ==)")
        if not ({"reward", "penalty"} & set(var)) and mode not in _REWARD_MODES - {"binary"}:
            problems.append(
                f"{where}: config has neither 'reward' nor 'penalty' — this "
                f"variable would never pay anything."
            )

    for name, var in (cfg.get("reward", {}).get("milestones", {}) or {}).items():
        where = f"reward.milestones.{name}"
        unknown = set(var) - _MILESTONE_KEYS
        if unknown:
            problems.append(
                f"{where}: unknown key(s) {sorted(unknown)} — a milestone is a "
                f"ONE-SHOT payout the first time <var> <op> <reference> becomes "
                f'true in an episode; it takes exactly {{"var": <data.json '
                f'variable>, "op": <op>, "reference": <number>, "reward": <amount>}}.'
            )
        missing = {"var", "op", "reference", "reward"} - set(var)
        if missing:
            problems.append(
                f"{where}: missing required key(s) {sorted(missing)} — all "
                f"four of var/op/reference/reward are required (an implicit "
                f"reference is a footgun)."
            )
        if var.get("op") not in OP_ALIASES:
            problems.append(
                f"{where}: op '{var.get('op')}' not recognized — use one of "
                f"{sorted(set(OP_ALIASES.values()))} (or <, >, ==)"
            )
        if data_vars is not None and "var" in var and var["var"] not in data_vars:
            problems.append(
                f"{where}: var '{var['var']}' is not a data.json variable of "
                f"{game_id} — it would NEVER fire. Available: {sorted(data_vars)}"
            )

    for name, var in (cfg.get("reward", {}).get("novelty", {}) or {}).items():
        where = f"reward.novelty.{name}"
        unknown = set(var) - _NOVELTY_KEYS
        if unknown:
            problems.append(
                f"{where}: unknown key(s) {sorted(unknown)} — a novelty rule "
                f"pays once per episode for each NEW combination of its keys' "
                f'values; it takes exactly {{"keys": [<data.json variables>], '
                f'"reward": <amount per new combination>}}.'
            )
        keys = var.get("keys")
        if not isinstance(keys, list) or not keys or not all(isinstance(k, str) for k in keys):
            problems.append(
                f"{where}: 'keys' must be a non-empty list of data.json "
                f"variable names (e.g. [\"level\", \"screen_id\"]) — their "
                f"combined values define what counts as a new place."
            )
        if "reward" not in var:
            problems.append(
                f"{where}: 'reward' is required (amount paid per newly seen "
                f"combination)."
            )
        if data_vars is not None and isinstance(keys, list):
            bad = [k for k in keys if isinstance(k, str) and k not in data_vars]
            if bad:
                problems.append(
                    f"{where}: key(s) {bad} are not data.json variables of "
                    f"{game_id} — the rule would NEVER pay. Available: "
                    f"{sorted(data_vars)}"
                )

    for name, var in (cfg.get("reward", {}).get("counters", {}) or {}).items():
        where = f"reward.counters.{name}"
        unknown = set(var) - _COUNTER_KEYS
        if unknown:
            problems.append(
                f"{where}: unknown key(s) {sorted(unknown)} — a counter rule "
                f"pays for INCREMENTS of <var> attributed to the current "
                f"<context> tuple, at reward*decay^n, capped at "
                f"max_per_context events per context per episode; it takes "
                f'{{"var", "context": [vars], "reward", "max_per_context", '
                f'optional "decay" (default 1.0), optional "max_event_delta" '
                f"(default 1)}}."
            )
        missing = {"var", "context", "reward", "max_per_context"} - set(var)
        if missing:
            problems.append(
                f"{where}: missing required key(s) {sorted(missing)} — "
                f"max_per_context is REQUIRED (an uncapped counter is a "
                f"farmable reward fountain)."
            )
        ctx = var.get("context")
        if ctx is not None and (
            not isinstance(ctx, list) or not ctx
            or not all(isinstance(k, str) for k in ctx)
        ):
            problems.append(
                f"{where}: 'context' must be a non-empty list of data.json "
                f"variable names — events are attributed to (and capped per) "
                f"this tuple."
            )
        decay = var.get("decay", 1.0)
        if not isinstance(decay, (int, float)) or not (0 < decay <= 1):
            problems.append(
                f"{where}: 'decay' must be in (0, 1] — each successive event "
                f"in a context pays reward*decay^n."
            )
        v = var.get("var")
        if "var" in var and (not isinstance(v, str) or not v):
            problems.append(f"{where}: 'var' must be a non-empty variable name.")
        rw = var.get("reward")
        if "reward" in var and (
            isinstance(rw, bool) or not isinstance(rw, (int, float))
        ):
            problems.append(f"{where}: 'reward' must be a number.")
        mpc = var.get("max_per_context")
        if mpc is not None and (
            isinstance(mpc, bool) or not isinstance(mpc, int) or mpc <= 0
        ):
            problems.append(
                f"{where}: 'max_per_context' must be a positive integer."
            )
        med = var.get("max_event_delta")
        if med is not None and (
            isinstance(med, bool) or not isinstance(med, int) or med <= 0
        ):
            problems.append(
                f"{where}: 'max_event_delta' must be a positive integer "
                f"(events larger than this are rejected as garbage)."
            )
        if data_vars is not None:
            bad = [
                k for k in ([var.get("var")] + (ctx if isinstance(ctx, list) else []))
                if isinstance(k, str) and k not in data_vars
            ]
            if bad:
                problems.append(
                    f"{where}: name(s) {bad} are not data.json variables of "
                    f"{game_id} — the rule would NEVER pay. Available: "
                    f"{sorted(data_vars)}"
                )

    for name, var in (cfg.get("done", {}).get("variables", {}) or {}).items():
        where = f"done.variables.{name}"
        unknown = set(var) - _DONE_VAR_KEYS
        if unknown:
            problems.append(
                f"{where}: unknown key(s) {sorted(unknown)} — done conditions "
                f'take exactly {{"op": <op>, "reference": <number>}} '
                f"('value' is not a key; use 'reference')."
            )
        op = var.get("op")
        if op not in OP_ALIASES:
            problems.append(
                f"{where}: op '{op}' not recognized — use one of "
                f"{sorted(set(OP_ALIASES.values()))} (or <, >, ==)"
            )
        if "success" in var and not isinstance(var["success"], bool):
            problems.append(
                f"{where}: 'success' must be a boolean — it marks this done "
                f"condition as a SUCCESSFUL episode end (emitted as truncated, "
                f"value bootstraps) rather than a failure (terminated)."
            )

    if problems:
        raise ValueError(
            f"training.json for {game_id} has schema errors (the engine would "
            f"silently ignore these — fix them before training):\n  - "
            + "\n  - ".join(problems)
        )


def resolve_action_mappings(action_defs: list, env_buttons: list, game_id: str):
    """Compile actions.json entries into emulator button rows, validated
    against the REAL button list of the loaded core (env.buttons).

    Authoring format is button NAMES: "buttons": ["RIGHT", "A"] (empty list
    = no-op). Legacy 0/1 index arrays are still accepted but validated hard.

    Every way an action map has ever gone wrong is a load-time error here:
    unknown button names, rows pressing layout holes (indices where no
    button exists), rows longer than the real layout, and two actions that
    press the exact same buttons. Every error message states the fix —
    these are read by humans and by the copilot.

    Returns (rows, labels): 0/1 rows for the emulator and a human label per
    action derived from the buttons actually pressed (never trust the
    author's name field — it is display-only).
    """
    n = len(env_buttons)
    valid_names = [b for b in env_buttons if b]
    problems = []
    rows, names = [], []

    for idx, a in enumerate(action_defs):
        name = a.get("name", f"action {idx}")
        spec = a.get("buttons", [])
        names.append(name)
        row = [0] * n
        if all(isinstance(b, str) for b in spec):
            for b in spec:
                hit = next(
                    (i for i, eb in enumerate(env_buttons)
                     if eb and eb.upper() == b.upper()), None,
                )
                if hit is None:
                    problems.append(
                        f"'{name}': unknown button '{b}' — {game_id} has "
                        f"exactly these buttons: {valid_names}"
                    )
                else:
                    row[hit] = 1
        elif all(isinstance(b, int) and not isinstance(b, bool) for b in spec):
            if len(spec) > n:
                problems.append(
                    f"'{name}': {len(spec)} entries but {game_id} exposes "
                    f"{n} slots ({env_buttons}) — rewrite with button NAMES, "
                    f'e.g. {{"buttons": ["RIGHT", "A"]}}'
                )
                continue
            for i, v in enumerate(spec):
                if not v:
                    continue
                if not env_buttons[i]:
                    problems.append(
                        f"'{name}': presses index {i}, which is a HOLE in the "
                        f"{game_id} layout — no button exists there, the press "
                        f"does nothing. Layout: {env_buttons}. Rewrite with "
                        f"button NAMES instead of index arrays."
                    )
                else:
                    row[i] = 1
        else:
            problems.append(
                f"'{name}': buttons must be ALL names ([\"RIGHT\", \"A\"]) or "
                f"ALL 0/1 ints — got {spec!r}"
            )
        rows.append(row)

    labels = [
        "+".join(env_buttons[i] for i, v in enumerate(row) if v) or "NoOp"
        for row in rows
    ]
    seen: Dict[tuple, int] = {}
    for i, row in enumerate(rows):
        key = tuple(row)
        if key in seen:
            j = seen[key]
            problems.append(
                f"'{names[i]}' and '{names[j]}' press exactly the same "
                f"buttons ({labels[i]}) — duplicate actions waste the "
                f"agent's action space; remove one (this is usually a "
                f"mislabeled row, check what each row REALLY presses)."
            )
        else:
            seen[key] = i

    if problems:
        raise ValueError(
            f"actions.json for {game_id} is broken (the agent would train "
            f"on wrong or dead inputs — fix before training):\n  - "
            + "\n  - ".join(problems)
        )
    return rows, labels


def check_done(done_config, info):
    """Evaluate done conditions; return the FIRST matching variable name, or
    None. Order matters and is the config's dict order (JSON order preserved):
    list failure conditions (health, reverse) BEFORE success conditions
    (race_on) so a crash that coincides with a success flag on the same step
    is still classified as a death.

    The caller maps the matched condition to the gymnasium 5-tuple via the
    condition's optional "success" flag: success=True ends the episode as
    TRUNCATED (the value function bootstraps — finishing a race is not death,
    and the continue predictor must not learn 'finish line = terminal' and
    smear that fear across visually identical non-final laps), anything else
    is TERMINATED (true failure, value zero).

    Pure function. Kept module-level so tests can drive it directly.
    """
    for var_name, var_cfg in (done_config or {}).items():
        if var_name not in info:
            continue
        op = OP_ALIASES.get(var_cfg.get("op"), var_cfg.get("op"))
        ref = var_cfg.get("reference", 0)
        val = info[var_name]
        if (
            (op == "less-than" and val < ref)
            or (op == "greater-than" and val > ref)
            or (op == "equal" and val == ref)
        ):
            return var_name
    return None


def episode_end_flags(done_config, info, terminated, truncated):
    """Map done-condition evaluation onto the gymnasium 5-tuple flags.

    Takes the flags as already set by the inner env (frame-skip loop) and
    returns the final (terminated, truncated) pair:
    - no condition matched: flags pass through unchanged
    - failure condition matched: terminated=True
    - success condition matched ("success": true): truncated=True AND
      terminated is forced False — this deliberately downgrades any
      scenario.json-era terminal the inner env raised for the same frame,
      because a successful episode end must bootstrap (the continue
      predictor must not learn 'success screen = death').
    check_done's first-match config ordering means failure conditions listed
    before success ones win same-step ties (crash on the finish frame is
    still a death).

    Pure function — the wrapper delegates here so tests can drive the exact
    production routing without an emulator.
    """
    matched = check_done(done_config, info)
    if matched is None:
        return terminated, truncated
    if (done_config or {}).get(matched, {}).get("success"):
        return False, True
    return True, truncated


def score_milestones(milestones_cfg, info, fired, baseline=False):
    """First-time-only milestone payouts: one-shot reward the first time an
    op condition becomes true this episode (e.g. Zelda sword flag > 0).

    baseline=True is the SPAWN-ARMED suppression pass, run on the episode's
    first reward step: any milestone already true is consumed (added to
    `fired`) WITHOUT paying. Rationale: a milestone rewards a TRANSITION the
    agent caused; a save state that spawns with the condition true (e.g. a
    rotation state saved after the sword grab) would otherwise pay an
    unearned lump every episode — reward for existing, not for acting. The
    first step is the earliest possible look because stable-retro's reset()
    info does not carry data.json variables; the cost is that a milestone
    genuinely earned within the very first frame_skip window (one agent
    action) is silently consumed — acceptable, no real milestone flips that
    fast from a legitimate spawn.

    Milestones observed FALSE during the baseline pass stay armed and pay
    normally when they later become true. Negative milestones (e.g. died)
    follow the same rule: a state saved mid-death-cutscene should not be
    charged for a death the agent didn't cause.

    Pure function: mutates only `fired` (a per-episode set the caller owns).
    Kept module-level so tests can drive it with recorded RAM traces.
    """
    total = 0.0
    for name, cfg in milestones_cfg.items():
        if name in fired:
            continue
        val = info.get(cfg.get("var"))
        if val is None:
            continue
        ref = cfg.get("reference", 0)
        op = OP_ALIASES.get(cfg.get("op"), cfg.get("op"))
        if (
            (op == "greater-than" and val > ref)
            or (op == "less-than" and val < ref)
            or (op == "equal" and val == ref)
        ):
            fired.add(name)
            if not baseline:
                total += cfg.get("reward", 0.0)
    return total


def score_counters(counters_cfg, prev_info, info, state):
    """Counted-event rewards attributed to a place, with diminishing returns.

    A rule {"var", "context": [vars], "reward", "decay", "max_per_context",
    "max_event_delta"} pays for INCREMENTS of var (each unit = one event),
    attributed to the current context tuple, at reward * decay^n where n is
    the number of events already paid for that context this episode, hard-
    capped at max_per_context events per context.

    Deliberate semantics, matched to real counter bytes (Zelda kill streak):
    - An event only counts when the context tuple is IDENTICAL in the
      previous and current step: scroll-flicker frames and a stale counter
      value carried onto a newly entered screen can never pay (arrival is
      not an event; only an increment WHILE ON the screen is).
    - Decreases are ignored (damage-reset of a streak counter is not an
      event and cannot re-arm extra payments: paid ordinals per context
      only ever grow within an episode).
    - Jumps larger than max_event_delta are rejected as garbage.

    Pure function: mutates only `state` ({rule: {context_tuple: paid}}).
    Kept module-level so tests can drive it with recorded RAM traces.
    """
    total = 0.0
    for name, cfg in counters_cfg.items():
        ctx_keys = cfg.get("context", [])
        prev_ctx = tuple(prev_info.get(k) for k in ctx_keys)
        cur_ctx = tuple(info.get(k) for k in ctx_keys)
        if any(v is None for v in cur_ctx) or prev_ctx != cur_ctx:
            continue
        pv, cv = prev_info.get(cfg.get("var")), info.get(cfg.get("var"))
        if pv is None or cv is None:
            continue
        delta = int(cv) - int(pv)
        if delta <= 0 or delta > cfg.get("max_event_delta", 1):
            continue
        rule_state = state.setdefault(name, {})
        paid = rule_state.get(cur_ctx, 0)
        cap = cfg.get("max_per_context", 0)
        decay = cfg.get("decay", 1.0)
        for _ in range(delta):
            if paid >= cap:
                break
            total += cfg.get("reward", 0.0) * (decay ** paid)
            paid += 1
        rule_state[cur_ctx] = paid
    return total
