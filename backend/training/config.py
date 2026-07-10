"""DreamerV3 training configuration — generalized for any gym-retro game."""

from dataclasses import dataclass, field, asdict
from typing import Optional
import yaml
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


@dataclass
class WorldModelConfig:
    """World model architecture config."""
    deter_size: int = 4096
    stoch_size: int = 32
    stoch_classes: int = 32
    hidden_size: int = 1024
    cnn_depth: int = 48
    cnn_keys: str = "image"
    mlp_keys: str = ""
    encoder_layers: int = 0
    decoder_layers: int = 0
    reward_layers: int = 5
    cont_layers: int = 5
    units: int = 1024


@dataclass
class ActorCriticConfig:
    """Actor-critic config."""
    actor_layers: int = 5
    critic_layers: int = 5
    actor_units: int = 1024
    critic_units: int = 1024
    actor_dist: str = "onehot"  # discrete actions
    actor_entropy: float = 3e-4
    actor_min_std: float = 0.1
    critic_slow: bool = True
    critic_slow_fraction: float = 0.02
    discount: float = 0.997
    lambda_: float = 0.95
    imag_horizon: int = 15


@dataclass
class TrainingConfig:
    """Training loop config — generalized for any gym-retro game."""
    # Game selection
    game_id: str = "FZero-Snes"
    initial_state: str = "go"

    # Model size preset
    model_size: str = "small"  # debug/small/medium/large/xl

    # Training
    batch_size: int = 16
    batch_length: int = 64
    # SheepRL units: gradient updates per policy step. Paper "train ratio"
    # = this x batch_size x batch_length (x1024 at 16x64). 0.125 = paper 128,
    # the Atari100k setting; the old default 4 was paper 4096 — 128x the
    # paper's flagship game runs.
    replay_ratio: float = 0.125
    train_ratio: int = 512  # legacy, unused
    learning_rate: float = 1e-4
    adam_eps: float = 1e-8
    grad_clip: float = 1000.0
    weight_decay: float = 0.0

    # Replay buffer
    replay_capacity: int = 1_000_000
    prefill_steps: int = 10000
    # On resume: >0 means "do not restore the replay buffer from the checkpoint,
    # re-collect this many steps with the current policy instead" — the recovery
    # path when the buffer's memmap files are lost or corrupt.
    resume_prefill: int = 0

    # Environment
    env_state: str = "go"
    env_scenario: str = "training"
    env_action_mode: str = "full"
    obs_size: tuple[int, int] = (64, 64)
    max_episode_steps: int = 4500
    num_envs: int = 8  # parallel environment instances

    # Rendering
    render_every: int = 50  # render one episode every N episodes
    eval_every: int = 500   # run evaluation every N episodes

    # Checkpointing
    checkpoint_every: int = 1000  # save every N steps
    log_every: int = 100          # log metrics every N steps

    # Paths
    logdir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    episode_dir: str = "./episodes"

    # World model
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)

    # Actor-critic
    actor_critic: ActorCriticConfig = field(default_factory=ActorCriticConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        wm = WorldModelConfig(**data.pop("world_model", {}))
        ac = ActorCriticConfig(**data.pop("actor_critic", {}))
        return cls(world_model=wm, actor_critic=ac, **data)

    def to_sheeprl_config(self) -> DictConfig:
        """Convert to an OmegaConf DictConfig compatible with SheepRL's DreamerV3.

        This produces the nested config structure that SheepRL's build_agent()
        and train() functions expect, mapped from our flat preset system.
        """
        wm = self.world_model
        ac = self.actor_critic

        # Map our presets to SheepRL model dimensions
        # SheepRL sizes: XS(256/24/1), S(512/32/2), M(640/48/3), L(768/64/4), XL(1024/96/5)
        sheeprl_sizes = {
            "debug": {"dense_units": 256, "mlp_layers": 1, "cnn_mult": 16,
                       "recurrent": 256, "hidden": 256},
            "small": {"dense_units": 512, "mlp_layers": 2, "cnn_mult": 32,
                       "recurrent": 512, "hidden": 512},
            "medium": {"dense_units": 640, "mlp_layers": 3, "cnn_mult": 48,
                        "recurrent": 1024, "hidden": 640},
            "large": {"dense_units": 768, "mlp_layers": 4, "cnn_mult": 64,
                       "recurrent": 2048, "hidden": 768},
            "xl": {"dense_units": 1024, "mlp_layers": 5, "cnn_mult": 96,
                    "recurrent": 4096, "hidden": 1024},
        }
        sz = sheeprl_sizes.get(self.model_size, sheeprl_sizes["small"])

        cnn_ln = {"cls": "sheeprl.models.models.LayerNormChannelLast", "kw": {"eps": 1e-3}}
        mlp_ln = {"cls": "sheeprl.models.models.LayerNorm", "kw": {"eps": 1e-3}}

        cfg = OmegaConf.create({
            "seed": 42,
            "dry_run": False,
            "root_dir": self.logdir,
            "run_name": f"{self.game_id}_dreamerv3",

            "env": {
                "screen_size": self.obs_size[0],  # 64
                "frame_stack": -1,  # disabled — SheepRL handles this
                "num_envs": self.num_envs,
                "clip_rewards": False,
                "action_repeat": 1,
                "sync_env": True,
                "max_episode_steps": self.max_episode_steps,
            },

            "algo": {
                "name": "dreamer_v3",
                "gamma": ac.discount,
                "lmbda": ac.lambda_,
                "horizon": ac.imag_horizon,
                "replay_ratio": 1,
                "learning_starts": self.prefill_steps,
                "per_rank_pretrain_steps": 0,
                "per_rank_batch_size": self.batch_size,
                "per_rank_sequence_length": self.batch_length,
                "total_steps": 100_000_000,  # effectively infinite — we control stop
                "run_test": False,
                "unimix": 0.01,
                "hafner_initialization": True,

                "cnn_keys": {
                    "encoder": ["rgb"],
                    "decoder": ["rgb"],
                },
                "mlp_keys": {
                    "encoder": [],
                    "decoder": [],
                },

                "world_model": {
                    "stochastic_size": wm.stoch_size,
                    "discrete_size": wm.stoch_classes,
                    "kl_dynamic": 0.5,
                    "kl_representation": 0.1,
                    "kl_free_nats": 1.0,
                    "kl_regularizer": 1.0,
                    "continue_scale_factor": 1.0,
                    "clip_gradients": self.grad_clip,
                    "decoupled_rssm": False,
                    "learnable_initial_recurrent_state": True,

                    "encoder": {
                        "cnn_channels_multiplier": sz["cnn_mult"],
                        "cnn_act": "torch.nn.SiLU",
                        "dense_act": "torch.nn.SiLU",
                        "mlp_layers": sz["mlp_layers"],
                        "cnn_layer_norm": cnn_ln,
                        "mlp_layer_norm": mlp_ln,
                        "dense_units": sz["dense_units"],
                    },
                    "recurrent_model": {
                        "recurrent_state_size": sz["recurrent"],
                        "layer_norm": mlp_ln,
                        "dense_units": sz["dense_units"],
                    },
                    "transition_model": {
                        "hidden_size": sz["hidden"],
                        "dense_act": "torch.nn.SiLU",
                        "layer_norm": mlp_ln,
                    },
                    "representation_model": {
                        "hidden_size": sz["hidden"],
                        "dense_act": "torch.nn.SiLU",
                        "layer_norm": mlp_ln,
                    },
                    "observation_model": {
                        "cnn_channels_multiplier": sz["cnn_mult"],
                        "cnn_act": "torch.nn.SiLU",
                        "dense_act": "torch.nn.SiLU",
                        "mlp_layers": sz["mlp_layers"],
                        "cnn_layer_norm": cnn_ln,
                        "mlp_layer_norm": mlp_ln,
                        "dense_units": sz["dense_units"],
                    },
                    "reward_model": {
                        "dense_act": "torch.nn.SiLU",
                        "mlp_layers": sz["mlp_layers"],
                        "layer_norm": mlp_ln,
                        "dense_units": sz["dense_units"],
                        "bins": 255,
                    },
                    "discount_model": {
                        "learnable": True,
                        "dense_act": "torch.nn.SiLU",
                        "mlp_layers": sz["mlp_layers"],
                        "layer_norm": mlp_ln,
                        "dense_units": sz["dense_units"],
                    },
                    "optimizer": {
                        "lr": self.learning_rate,
                        "eps": self.adam_eps,
                        "weight_decay": self.weight_decay,
                    },
                },

                "actor": {
                    "cls": "sheeprl.algos.dreamer_v3.agent.Actor",
                    "ent_coef": ac.actor_entropy,
                    "min_std": ac.actor_min_std,
                    "max_std": 1.0,
                    "init_std": 2.0,
                    "dense_act": "torch.nn.SiLU",
                    "mlp_layers": sz["mlp_layers"],
                    "layer_norm": mlp_ln,
                    "dense_units": sz["dense_units"],
                    "clip_gradients": 100.0,
                    "unimix": 0.01,
                    "action_clip": 1.0,
                    "moments": {
                        "decay": 0.99,
                        "max": 1.0,
                        "percentile": {
                            "low": 0.05,
                            "high": 0.95,
                        },
                    },
                    "optimizer": {
                        "lr": 8e-5,
                        "eps": 1e-5,
                        "weight_decay": 0,
                    },
                },

                "critic": {
                    "dense_act": "torch.nn.SiLU",
                    "mlp_layers": sz["mlp_layers"],
                    "layer_norm": mlp_ln,
                    "dense_units": sz["dense_units"],
                    "per_rank_target_network_update_freq": 1,
                    "tau": ac.critic_slow_fraction,
                    "bins": 255,
                    "clip_gradients": 100.0,
                    "optimizer": {
                        "lr": 8e-5,
                        "eps": 1e-5,
                        "weight_decay": 0,
                    },
                },
            },

            "buffer": {
                "size": self.replay_capacity,
                "memmap": True,
                "checkpoint": False,
                "from_numpy": True,
                "validate_args": False,
            },

            "checkpoint": {
                "resume_from": None,
                "every": self.checkpoint_every,
                "save_last": True,
            },

            "metric": {
                "log_level": 1,
                "log_every": self.log_every,
                "sync_on_compute": False,
                "aggregator": {
                    "_target_": "sheeprl.utils.metric.MetricAggregator",
                    "metrics": {
                        "Rewards/rew_avg": {"_target_": "torchmetrics.MeanMetric", "sync_on_compute": False},
                        "Game/ep_len_avg": {"_target_": "torchmetrics.MeanMetric", "sync_on_compute": False},
                        "Loss/world_model_loss": {"_target_": "torchmetrics.MeanMetric", "sync_on_compute": False},
                        "Loss/observation_loss": {"_target_": "torchmetrics.MeanMetric", "sync_on_compute": False},
                        "Loss/reward_loss": {"_target_": "torchmetrics.MeanMetric", "sync_on_compute": False},
                        "Loss/state_loss": {"_target_": "torchmetrics.MeanMetric", "sync_on_compute": False},
                        "Loss/continue_loss": {"_target_": "torchmetrics.MeanMetric", "sync_on_compute": False},
                        "State/kl": {"_target_": "torchmetrics.MeanMetric", "sync_on_compute": False},
                        "State/post_entropy": {"_target_": "torchmetrics.MeanMetric", "sync_on_compute": False},
                        "State/prior_entropy": {"_target_": "torchmetrics.MeanMetric", "sync_on_compute": False},
                        "Loss/policy_loss": {"_target_": "torchmetrics.MeanMetric", "sync_on_compute": False},
                        "Loss/value_loss": {"_target_": "torchmetrics.MeanMetric", "sync_on_compute": False},
                        "Grads/world_model": {"_target_": "torchmetrics.MeanMetric", "sync_on_compute": False},
                        "Grads/actor": {"_target_": "torchmetrics.MeanMetric", "sync_on_compute": False},
                        "Grads/critic": {"_target_": "torchmetrics.MeanMetric", "sync_on_compute": False},
                    },
                },
            },

            "distribution": {
                "type": "auto",
            },
        })
        return cfg

    @classmethod
    def from_preset(cls, size: str = "small") -> "TrainingConfig":
        """Create config from a model size preset."""
        presets = {
            "debug": dict(
                batch_size=4, batch_length=16, train_ratio=32,
                render_every=10, eval_every=50,
                checkpoint_every=500, log_every=10, num_envs=4,
                world_model=WorldModelConfig(
                    deter_size=512, stoch_size=16, stoch_classes=16,
                    hidden_size=256, cnn_depth=24, units=256,
                    reward_layers=2, cont_layers=2,
                ),
                actor_critic=ActorCriticConfig(
                    actor_layers=2, critic_layers=2,
                    actor_units=256, critic_units=256,
                    imag_horizon=5,
                ),
            ),
            "small": dict(num_envs=6,
                world_model=WorldModelConfig(
                    deter_size=1024, stoch_size=32, stoch_classes=32,
                    hidden_size=512, cnn_depth=32, units=512,
                    reward_layers=3, cont_layers=3,
                ),
                actor_critic=ActorCriticConfig(
                    actor_layers=3, critic_layers=3,
                    actor_units=512, critic_units=512,
                ),
            ),
            "medium": dict(num_envs=8,
                world_model=WorldModelConfig(
                    deter_size=2048, stoch_size=32, stoch_classes=32,
                    hidden_size=768, cnn_depth=48, units=768,
                ),
                actor_critic=ActorCriticConfig(
                    actor_units=768, critic_units=768,
                ),
            ),
            "large": dict(  # SheepRL L, ~77M params (paper batch is 16x64 for ALL sizes)
                batch_size=16, batch_length=64, train_ratio=512, num_envs=6,
                world_model=WorldModelConfig(
                    deter_size=4096, stoch_size=32, stoch_classes=32,
                    hidden_size=1024, cnn_depth=48, units=1024,
                    reward_layers=5, cont_layers=5,
                ),
                actor_critic=ActorCriticConfig(
                    actor_layers=5, critic_layers=5,
                    actor_units=1024, critic_units=1024,
                    imag_horizon=15,
                ),
            ),
            "xl": dict(  # SheepRL XL, ~200M params — the paper's game config
                batch_size=16, batch_length=64, train_ratio=512, num_envs=4,
                world_model=WorldModelConfig(
                    deter_size=8192, stoch_size=32, stoch_classes=32,
                    hidden_size=1536, cnn_depth=96, units=1536,
                    reward_layers=5, cont_layers=5,
                ),
                actor_critic=ActorCriticConfig(
                    actor_layers=5, critic_layers=5,
                    actor_units=1536, critic_units=1536,
                    imag_horizon=15,
                ),
            ),
        }
        preset = presets.get(size, presets["small"])
        wm = preset.pop("world_model", WorldModelConfig())
        ac = preset.pop("actor_critic", ActorCriticConfig())
        return cls(model_size=size, world_model=wm, actor_critic=ac, **preset)
