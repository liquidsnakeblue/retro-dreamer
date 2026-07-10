"""Build a save state by walking a game with a scripted button plan —
from power-on or from an existing state. The productized gp_state_builder.

plan JSON: list of [wait_frames, "BUTTON+BUTTON"|""] steps; buttons held for
2 frames then released for the rest of the wait. A frame PNG is saved after
every plan step so a human (or the copilot's vision) can verify the walk.

Last stdout line: RESULT {state_path, screenshots_dir, final_vars}

Usage:
  python _retro_build_state.py <game_id> <game_dir> <plan.json> <out_state_name> <shots_dir> [start_state]
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import retro

game_id, game_dir, plan_path, out_name, shots_dir = sys.argv[1:6]
start_state = sys.argv[6] if len(sys.argv) > 6 else None

game_dir = Path(game_dir)
shots = Path(shots_dir)
shots.mkdir(parents=True, exist_ok=True)
retro.data.Integrations.add_custom_path(str(game_dir.parent))

state_arg = retro.State.NONE
if start_state:
    f = game_dir / "states" / f"{start_state}.state"
    state_arg = str(f) if f.exists() else start_state

env = retro.make(
    game=game_id,
    state=state_arg,
    inttype=retro.data.Integrations.CUSTOM_ONLY,
    use_restricted_actions=retro.Actions.ALL,
    render_mode="rgb_array",
)
BUTTONS = env.buttons


def act(s):
    a = np.zeros(len(BUTTONS), dtype=np.uint8)
    for b in s.split("+"):
        if b:
            a[BUTTONS.index(b)] = 1
    return a


plan = json.loads(Path(plan_path).read_text())
env.reset()
frame = None
for i, (wait, buttons) in enumerate(plan):
    hold, noop = act(buttons), act("")
    for f in range(int(wait)):
        frame, *_ = env.step(hold if f < 2 else noop)
    Image.fromarray(frame).save(shots / f"step_{i:02d}_{buttons or 'wait'}.png")
    print(f"step {i:02d}: wait={wait} buttons={buttons or '(none)'}", flush=True)

out = game_dir / "states" / f"{out_name}.state"
out.parent.mkdir(exist_ok=True)
with gzip.open(out, "wb") as fh:
    fh.write(env.em.get_state())
Image.fromarray(env.em.get_screen()).save(shots / "final.png")
env.data.update_ram()
final_vars = {k: v for k, v in env.data.lookup_all().items()}
env.close()
print("RESULT " + json.dumps({
    "state_path": str(out), "screenshots_dir": str(shots), "final_vars": final_vars,
}), flush=True)
