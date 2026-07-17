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
_DONE_VAR_KEYS = {"op", "reference"}
_MILESTONE_KEYS = {"var", "op", "reference", "reward"}
_NOVELTY_KEYS = {"keys", "reward"}


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
