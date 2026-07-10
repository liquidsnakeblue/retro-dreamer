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
from sheeprl.envs.retro_dreamer import RetroDreamerWrapper

game_id, game_dir, states_csv, steps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
actions_arg = sys.argv[5] if len(sys.argv) > 5 else "all"

training = json.loads((Path(game_dir) / "training.json").read_text()) if (Path(game_dir) / "training.json").exists() else {}
reward_cfg = training.get("reward", {}).get("variables", {})
warmup = training.get("reward", {}).get("warmup_steps", 0)


def expected_reward(prev, cur, step):
    """Independent reimplementation of RetroDreamerWrapper reward semantics."""
    if step <= warmup:
        return 0.0
    r = 0.0
    for var, cfg in reward_cfg.items():
        if var not in cur or var not in prev:
            continue
        if "penalty" in cfg:
            loss = max(0, prev[var] - cur[var])
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
    return r


results = []
for state in states_csv.split(","):
    env = RetroDreamerWrapper(
        game_id=game_id, game_dir=game_dir, initial_state=state, frame_skip=4
    )
    n_actions = env.action_space.n
    action_list = list(range(n_actions)) if actions_arg == "all" else [int(a) for a in actions_arg.split(",")]
    for a in action_list:
        obs, info = env.reset(seed=42)
        obs, reward, term, trunc, info = env.step(a)
        tracked = [k for k in info if isinstance(info.get(k), (int, float))]
        prev = {k: info[k] for k in tracked}
        var_min = dict(prev)
        var_max = dict(prev)
        total, max_dev, max_abs, end, end_reason = float(reward), abs(float(reward)), 0.0, None, "survived"
        for step in range(2, steps + 1):
            obs, reward, term, trunc, info = env.step(a)
            cur = {k: info[k] for k in tracked}
            dev = abs(float(reward) - expected_reward(prev, cur, step))
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
