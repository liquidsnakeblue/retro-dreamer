"""DreamerV3 training configuration — generalized for any gym-retro game.

Every field here actually reaches SheepRL (via the CLI overrides built in
backend/training/trainer.py::_launch_sheeprl) or is consumed by the dashboard
itself. SheepRL owns everything else — architecture, optimizers, buffer, env
details — through its own configs (sheeprl/sheeprl/configs/**), selected by
the ``model_size`` preset name.

A large legacy surface (WorldModelConfig/ActorCriticConfig dataclasses, a
caller-less to_sheeprl_config(), and dead optimizer/env/rendering fields) was
removed 2026-07-09 after an audit found it accepted values that silently
never reached training.
"""

from dataclasses import dataclass, asdict
from pathlib import Path

import yaml


@dataclass
class TrainingConfig:
    """Training launch config — flows to the SheepRL subprocess CLI."""

    # Game selection
    game_id: str = "FZero-Snes"
    initial_state: str = "go"
    env_state: str = "go"  # legacy fallback still read by the trainer

    # Model size preset -> SheepRL algo config (dreamer_v3_XS/S/M/L/XL)
    model_size: str = "small"  # debug/small/medium/large/xl

    # Training
    batch_size: int = 16
    batch_length: int = 64
    # SheepRL units: gradient updates per policy step. Paper "train ratio"
    # = this x batch_size x batch_length (x1024 at 16x64). 0.125 = paper 128,
    # the Atari100k setting; the old default 4 was paper 4096 — 128x the
    # paper's flagship game runs.
    replay_ratio: float = 0.125
    num_envs: int = 6

    # Buffer prefill
    prefill_steps: int = 10000
    # On resume: >0 means "do not restore the replay buffer from the checkpoint,
    # re-collect this many steps with the current policy instead" — the recovery
    # path when the buffer's memmap files are lost or corrupt.
    resume_prefill: int = 0

    # Cadence
    checkpoint_every: int = 1000  # save every N steps
    log_every: int = 100          # log metrics every N steps

    # Paths (dashboard-side: TensorBoard callback + episode renderer)
    logdir: str = "./logs"
    episode_dir: str = "./episodes"

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        # Tolerate files saved by the old fat config: ignore removed keys
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_preset(cls, size: str = "small") -> "TrainingConfig":
        """Per-size launch defaults. Architecture comes from SheepRL's
        dreamer_v3_<size> config, not from here; paper batch is 16x64 for
        ALL model sizes."""
        presets = {
            "debug": dict(batch_size=4, batch_length=16, num_envs=4,
                          checkpoint_every=500, log_every=10),
            "small": dict(num_envs=6),    # SheepRL S, ~18M params
            "medium": dict(num_envs=8),   # SheepRL M, ~37M params
            "large": dict(num_envs=6),    # SheepRL L, ~77M params
            "xl": dict(num_envs=4),       # SheepRL XL, ~200M — the paper's game config
        }
        preset = presets.get(size, presets["small"])
        return cls(model_size=size, **preset)
