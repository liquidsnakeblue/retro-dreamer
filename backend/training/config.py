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

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class TrainingConfig:
    """Training launch config — flows to the SheepRL subprocess CLI."""

    # Game selection
    game_id: str = "FZero-Snes"
    initial_state: str = "go"
    env_state: str = "go"  # legacy fallback still read by the trainer
    # Optional optimistic-concurrency guard supplied by the planner. The
    # trainer always recomputes the workspace manifest immediately before
    # launch and rejects a stale planned hash.
    action_manifest_hash: str | None = None

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

    # Checkpoint cadence + bounded retention. At XL size a checkpoint is
    # currently ~2.3 GiB, so the old 1k/keep-10 policy spent ~23 GiB per run
    # and wrote a multi-gigabyte file about once a minute. Keep a short recent
    # recovery window plus a bounded set of coarse rollback milestones.
    checkpoint_every: int = 10_000
    checkpoint_keep_last: int = 3
    checkpoint_milestone_every: int = 50_000
    checkpoint_keep_milestones: int = 5
    log_every: int = 100          # log metrics every N steps

    # Paths (dashboard-side: TensorBoard callback + episode renderer)
    logdir: str = "./logs"
    episode_dir: str = "./episodes"

    def validate(self) -> "TrainingConfig":
        """Reject unsafe checkpoint policies before a training child starts."""
        if (
            self.action_manifest_hash is not None
            and not re.fullmatch(r"[0-9a-f]{64}", self.action_manifest_hash)
        ):
            raise ValueError("action_manifest_hash must be a lowercase SHA-256 hex digest")
        integer_fields = (
            "checkpoint_every",
            "checkpoint_keep_last",
            "checkpoint_milestone_every",
            "checkpoint_keep_milestones",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer, got {value!r}")

        if not 1 <= self.checkpoint_every <= 10_000_000:
            raise ValueError("checkpoint_every must be between 1 and 10,000,000")
        if not 1 <= self.checkpoint_keep_last <= 1_000:
            raise ValueError("checkpoint_keep_last must be between 1 and 1,000")
        if not 0 <= self.checkpoint_milestone_every <= 100_000_000:
            raise ValueError(
                "checkpoint_milestone_every must be between 0 and 100,000,000"
            )
        if not 0 <= self.checkpoint_keep_milestones <= 1_000:
            raise ValueError(
                "checkpoint_keep_milestones must be between 0 and 1,000"
            )
        if (self.checkpoint_milestone_every == 0) != (
            self.checkpoint_keep_milestones == 0
        ):
            raise ValueError(
                "checkpoint_milestone_every and checkpoint_keep_milestones "
                "must both be zero to disable milestones"
            )
        if (
            self.checkpoint_milestone_every
            and self.checkpoint_milestone_every < self.checkpoint_every
        ):
            raise ValueError(
                "checkpoint_milestone_every must be at least checkpoint_every"
            )
        return self

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
        return cls(**{k: v for k, v in data.items() if k in known}).validate()

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
        return cls(model_size=size, **preset).validate()
