"""Game-agnostic reward/done probe — the studio's pre-flight for ANY game.

Builds RetroDreamerWrapper directly (no training run or checkpoint needed),
holds each action constant for N steps per requested state, and reports:
  - reward-vs-formula deviation (independent reimplementation of the
    training.json semantics — catches wrapper bugs, the probe's whole point)
  - done behavior (when, and which variable plausibly fired)
  - per-step reward magnitude extremes (fountain detection)
  - info-variable ranges (sanity: wired addresses actually move)

Last stdout line is "RESULT <json>" for the studio job manager.

Usage:
  python _retro_tool_probe.py <game_id> <game_dir> <states_csv> <steps> [actions_csv|all]
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PYGLET_HEADLESS", "1")
import pyglet

pyglet.options["shadow_window"] = False

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from sheeprl.envs.retro_dreamer import OP_ALIASES, RetroDreamerWrapper

game_id, game_dir, states_csv, steps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
actions_arg = sys.argv[5] if len(sys.argv) > 5 else "all"

training = json.loads((Path(game_dir) / "training.json").read_text()) if (Path(game_dir) / "training.json").exists() else {}
reward_cfg = training.get("reward", {}).get("variables", {})
milestone_cfg = training.get("reward", {}).get("milestones", {})
novelty_cfg = training.get("reward", {}).get("novelty", {})
counter_cfg = training.get("reward", {}).get("counters", {})
warmup = training.get("reward", {}).get("warmup_steps", 0)


def expected_reward(prev, cur, step, visited, fired, counters):
    """Independent reimplementation of RetroDreamerWrapper reward semantics.

    visited/fired are per-episode sets the caller owns (novelty + milestone
    state). They must update on EVERY step — including warmup steps, whose
    returned reward is zeroed — mirroring the wrapper's compute-then-zero
    order, so spawn-true milestones/screens are consumed without paying.
    """
    r = 0.0
    for var, cfg in reward_cfg.items():
        if var not in cur or var not in prev:
            continue
        if "penalty" in cfg:
            loss = max(0, prev[var] - cur[var])
            pcap = cfg.get("max_delta")
            if pcap:
                loss = min(loss, pcap)
            r -= loss * cfg["penalty"]
            heal = cfg.get("heal_reward")
            if heal:
                r += max(0, cur[var] - prev[var]) * heal
        if "reward" in cfg and "op" not in cfg and "mode" not in cfg:
            d = cur[var] - prev[var]
            wrap = cfg.get("wrap")
            if wrap:
                d = (d + wrap // 2) % wrap - wrap // 2
            cap = cfg.get("max_delta")
            if cap:
                d = max(-cap, min(cap, d))
            gain = d if cfg.get("delta") == "signed" else max(0, d)
            r += gain * cfg["reward"]
        mode = cfg.get("mode")
        if mode == "quadratic":
            mx = cfg.get("max_speed", cfg.get("max_value", 500.0))
            if cur[var] >= cfg.get("min_threshold", 0.0) and mx > 0:
                norm = min(cur[var] / mx, 1.0)
                r += cfg.get("scaling_coefficient", 1.0) * cfg.get("base_reward", 0.1) * norm ** cfg.get("power", 2.0)
        elif mode == "linear":
            mx = cfg.get("max_speed", cfg.get("max_value", 500.0))
            if cur[var] >= cfg.get("min_threshold", 0.0) and mx > 0:
                r += cfg.get("base_reward", 0.1) * min(cur[var] / mx, 1.0)

    for name, cfg in milestone_cfg.items():
        if name in fired:
            continue
        v = cur.get(cfg.get("var"))
        if v is None:
            continue
        ref = cfg.get("reference", 0)
        op = OP_ALIASES.get(cfg.get("op"), cfg.get("op"))
        if (op == "greater-than" and v > ref) or (op == "less-than" and v < ref) or (op == "equal" and v == ref):
            fired.add(name)
            r += cfg.get("reward", 0.0)

    for name, cfg in novelty_cfg.items():
        key = tuple(cur.get(k) for k in cfg.get("keys", []))
        if any(v is None for v in key):
            continue
        seen = visited.setdefault(name, set())
        if key not in seen:
            seen.add(key)
            r += cfg.get("reward", 0.0)

    # Counted events per place, diminishing (independent reimplementation of
    # the wrapper's score_counters): pays for var INCREMENTS only when the
    # context tuple is identical across the two steps.
    for name, cfg in counter_cfg.items():
        ctx_keys = cfg.get("context", [])
        pctx = tuple(prev.get(k) for k in ctx_keys)
        cctx = tuple(cur.get(k) for k in ctx_keys)
        if any(v is None for v in cctx) or pctx != cctx:
            continue
        pv, cv = prev.get(cfg.get("var")), cur.get(cfg.get("var"))
        if pv is None or cv is None:
            continue
        d = int(cv) - int(pv)
        if d <= 0 or d > cfg.get("max_event_delta", 1):
            continue
        rs = counters.setdefault(name, {})
        paid = rs.get(cctx, 0)
        for _ in range(d):
            if paid >= cfg.get("max_per_context", 0):
                break
            r += cfg.get("reward", 0.0) * (cfg.get("decay", 1.0) ** paid)
            paid += 1
        rs[cctx] = paid

    return 0.0 if step <= warmup else r


results = []
printed_actions = False
for state in states_csv.split(","):
    env = RetroDreamerWrapper(
        game_id=game_id,
        game_dir=game_dir,
        initial_state=state,
        frame_skip=4,
        # This is an authoring pre-flight: it intentionally probes the current
        # workspace draft rather than a training/checkpoint manifest.
        allow_mutable_actions=True,
    )
    if not printed_actions:
        # What each action REALLY presses (resolved against the live core) —
        # a mislabeled action map should be visible in every probe output.
        print("ACTIONS: " + " | ".join(
            f"{i}:{lbl}" for i, lbl in enumerate(env.action_labels)), flush=True)
        printed_actions = True
    n_actions = env.action_space.n
    action_list = list(range(n_actions)) if actions_arg == "all" else [int(a) for a in actions_arg.split(",")]
    for a in action_list:
        obs, info = env.reset(seed=42)
        # stable-retro's reset() info does NOT carry the data.json variables —
        # they first appear in step() info. Step once (unchecked for formula
        # deviation) to establish the tracked set, exactly like the wrapper's
        # own first _calculate_reward call.
        obs, reward, term, trunc, info = env.step(a)
        tracked = [k for k in info if isinstance(info.get(k), (int, float))]
        prev = {k: info[k] for k in tracked}
        var_min = dict(prev)
        var_max = dict(prev)
        # Per-episode stateful-reward simulation (novelty visited-sets +
        # fired milestones) — fresh per probe episode, like reset(). The
        # wrapper already consumed step-1's keys/milestones, so warm the
        # simulated sets from step-1 info (discard the score). This mirrors
        # the wrapper's first-step BASELINE pass exactly: spawn-true
        # milestones are consumed unpaid on both sides (score_milestones
        # baseline=True), so rotation states saved past an objective can
        # never pay an unearned lump at step 1.
        visited, fired, counters = {}, set(), {}
        expected_reward(prev, prev, 1, visited, fired, counters)
        # Step 1 is formula-unchecked but still counts toward fountain
        # detection — a huge unearned spawn payout must not hide there.
        total, max_dev, max_abs, end, end_reason = float(reward), 0.0, abs(float(reward)), None, "survived"
        for step in range(2, steps + 1):
            obs, reward, term, trunc, info = env.step(a)
            cur = {k: info[k] for k in tracked}
            dev = abs(float(reward) - expected_reward(prev, cur, step, visited, fired, counters))
            max_dev = max(max_dev, dev)
            max_abs = max(max_abs, abs(float(reward)))
            total += float(reward)
            for k in tracked:
                var_min[k] = min(var_min[k], cur[k])
                var_max[k] = max(var_max[k], cur[k])
            prev = cur
            if term or trunc:
                end = step
                done_vars = training.get("done", {}).get("variables", {})
                fired = []
                for dv, dcfg in done_vars.items():
                    from sheeprl.envs.retro_dreamer import OP_ALIASES
                    v, ref = cur.get(dv), dcfg.get("reference", 0)
                    op = OP_ALIASES.get(dcfg.get("op"), dcfg.get("op"))
                    if (op == "less-than" and v is not None and v < ref) or (
                        op == "equal" and v == ref) or (op == "greater-than" and v is not None and v > ref):
                        fired.append(dv)
                end_reason = f"done({'+'.join(fired) or 'TimeLimit/other'})"
                break
        results.append({
            "state": state, "action": a, "end_step": end, "end_reason": end_reason,
            "return": round(total, 1), "reward_formula_max_deviation": round(max_dev, 6),
            "max_abs_step_reward": round(max_abs, 1),
            "frozen_vars": [k for k in tracked if var_min[k] == var_max[k] and k != "track_state"],
        })
        print(f"{state} action={a}: end={end or 'survived'} ({end_reason}) "
              f"ret={total:.1f} maxdev={max_dev:.6f}", flush=True)
    env.close()

verdict = {
    "ok": all(r["reward_formula_max_deviation"] < 0.001 for r in results),
    "fountains": [r for r in results if r["max_abs_step_reward"] > 5000],
    "never_done": [f"{r['state']}/a{r['action']}" for r in results if r["end_step"] is None],
    "probes": results,
}
print("RESULT " + json.dumps(verdict), flush=True)
