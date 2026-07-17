"""Generalized retro environment wrapper for DreamerV3 via SheepRL.

Replaces the F-Zero-specific wrappers with a single game-agnostic class.
Training loads actions from a launch-bound immutable manifest; explicit
authoring tools may opt into the mutable actions.json workspace file.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import gymnasium as gym
import numpy as np
import retro
from gymnasium import spaces

from sheeprl.utils.imports import _IS_STABLE_RETRO_AVAILABLE
from sheeprl.action_manifest import load_action_manifest

if not _IS_STABLE_RETRO_AVAILABLE:
    raise ModuleNotFoundError("stable-retro is required for retro-dreamer")


# Config validation lives in config_validation.py (dependency-free so the
# backend server can also load it by file path at API write time). Re-export
# here — the probe and older callers import these names from this module.
from sheeprl.envs.config_validation import (  # noqa: F401
    OP_ALIASES,
    resolve_action_mappings,
    score_counters,
    validate_training_config,
)


class RetroDreamerWrapper(gym.Wrapper):
    """Game-agnostic retro wrapper for DreamerV3 / SheepRL.

    - Registers a custom integration directory so retro can find the game
    - Loads training.json for reward shaping and done conditions
    - Loads an immutable action manifest for the discrete action mapping
    - Produces Dict observations with key "rgb" in CHW uint8 format
    - Discrete action space (len = number of action combos)
    """

    def __init__(
        self,
        id: str = "retro-dreamer",
        game_id: str = "FZero-Snes",
        game_dir: str = "",
        initial_state: str = "go",
        render_mode: str = "rgb_array",
        screen_size: int = 64,
        frame_skip: int = 4,
        grayscale: bool = False,
        seed: Optional[int] = None,
        action_manifest: str = "",
        action_manifest_hash: str = "",
        allow_mutable_actions: bool = False,
        **kwargs,
    ):
        if kwargs:
            # Hydra passes only declared wrapper keys, so anything here is a
            # typo or an unsupported option — surface it instead of silently
            # composing a config that does nothing.
            print(f"[RetroDreamerWrapper] WARNING: ignoring unknown kwargs: {sorted(kwargs)}")
        self.game_id = game_id
        self.game_dir = Path(game_dir)
        # Multi-track rotation: '+'-separated state names ("go+BBP1+SOP1")
        # rotate randomly per reset. ',' also accepted, but '+' is the
        # canonical delimiter — a bare comma in a Hydra CLI override value
        # parses as a choice-sweep and errors out.
        import re as _re
        self.initial_states = [
            s.replace(".state", "")
            for s in _re.split(r"[+,]", initial_state)
            if s.strip()
        ]
        self.initial_state = self.initial_states[0]
        self.frame_skip = frame_skip
        self.screen_size = screen_size
        self.grayscale = grayscale

        # Determine integration type: custom dir takes priority, fall back to built-in
        self._use_custom = self.game_dir.exists() and (self.game_dir / "data.json").exists()

        if self._use_custom:
            # Register custom integration so retro finds our game configs
            parent_dir = str(self.game_dir.parent)
            retro.data.Integrations.add_custom_path(parent_dir)
            inttype = retro.data.Integrations.CUSTOM_ONLY
        else:
            # Use built-in stable-retro data
            inttype = retro.data.Integrations.STABLE

        # Load training config (reward shaping + done conditions) — our extension
        training_path = self.game_dir / "training.json"
        if training_path.exists():
            with open(training_path) as f:
                self.training_config = json.load(f)
        else:
            self.training_config = {}
        # Unknown keys silently pay zero reward / never fire done — a config
        # written against an imagined schema must die HERE, not after hours
        # of zero-signal training. data.json vars (when available) let the
        # validator also catch typo'd milestone/novelty variable names.
        data_vars = None
        data_path = self.game_dir / "data.json"
        if data_path.exists():
            try:
                with open(data_path) as f:
                    data_vars = set((json.load(f).get("info") or {}).keys())
            except Exception:
                data_vars = None
        validate_training_config(game_id, self.training_config, data_vars=data_vars)

        # Training/evaluation binds to the exact ordered actions captured at
        # launch.  Reading actions.json directly is reserved for explicit
        # authoring probes: otherwise a file edit (or worker reconstruction)
        # could silently change the meaning of an existing policy's outputs.
        if action_manifest:
            if not action_manifest_hash:
                raise ValueError(
                    "action_manifest_hash is required with action_manifest; "
                    "an unanchored manifest path can be rebound"
                )
            manifest = load_action_manifest(
                action_manifest,
                expected_game_id=game_id,
                expected_hash=action_manifest_hash,
            )
            self.action_manifest = str(Path(action_manifest))
            self.action_manifest_hash = manifest["sha256"]
            self._action_defs = manifest["actions"]
        elif allow_mutable_actions:
            if action_manifest_hash:
                raise ValueError(
                    "action_manifest_hash was supplied without an action_manifest path"
                )
            actions_path = self.game_dir / "actions.json"
            if actions_path.exists():
                with open(actions_path) as f:
                    actions_data = json.load(f)
                self._action_defs = actions_data["actions"]
            else:
                # Authoring-only fallback: let retro expose FILTERED actions.
                self._action_defs = None  # handled below after env creation
            self.action_manifest = ""
            self.action_manifest_hash = ""
        else:
            raise ValueError(
                "immutable action_manifest is required for training and evaluation; "
                "this run/checkpoint is not bound to exact ordered actions. "
                "Migrate the legacy checkpoint or set allow_mutable_actions=true "
                "only in an explicit authoring tool."
            )

        # Determine state; preload raw bytes for every rotation entry so
        # reset() can swap tracks without touching retro's path resolution
        import gzip as _gzip
        states_dir = self.game_dir / "states"
        self._state_bytes: Dict[str, bytes] = {}
        for name in self.initial_states:
            state_file = states_dir / f"{name}.state"
            if state_file.exists():
                with _gzip.open(state_file, "rb") as fh:
                    self._state_bytes[name] = fh.read()
            elif len(self.initial_states) > 1:
                raise FileNotFoundError(
                    f"rotation state '{name}' not found in {states_dir}"
                )
        state = self.initial_state
        if states_dir.exists():
            state_file = states_dir / f"{self.initial_state}.state"
            if state_file.exists():
                state = str(state_file)

        # Create retro environment
        use_restricted = retro.Actions.ALL if self._action_defs else retro.Actions.FILTERED
        self._env = retro.make(
            game=game_id,
            state=state,
            inttype=inttype,
            use_restricted_actions=use_restricted,
            render_mode=render_mode,
        )

        # Ground truth for validation: the loaded core's own button list,
        # padded to the emulator's action row width.
        n_buttons = self._env.action_space.shape[0]
        env_buttons = list(self._env.buttons)
        env_buttons += [None] * (n_buttons - len(env_buttons))

        if self._action_defs is None:
            # No actions.json: one action per REAL button + no-op (holes skipped)
            self.action_mappings = [[0] * n_buttons]
            self.action_labels = ["NoOp"]
            for i, b in enumerate(env_buttons):
                if not b:
                    continue
                row = [0] * n_buttons
                row[i] = 1
                self.action_mappings.append(row)
                self.action_labels.append(b)
        else:
            self.action_mappings, self.action_labels = resolve_action_mappings(
                self._action_defs, env_buttons, game_id
            )

        super().__init__(self._env)

        if seed is not None:
            self.action_space.seed(seed)

        # Override action space to Discrete
        self.action_space = spaces.Discrete(len(self.action_mappings))

        # Override observation space: Dict with "rgb" key, CHW format
        if grayscale:
            obs_shape = (1, screen_size, screen_size)
        else:
            obs_shape = (3, screen_size, screen_size)

        self.observation_space = spaces.Dict({
            "rgb": spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        })

        # Episode tracking
        self.episode_step = 0
        self.episode_reward = 0.0
        self.prev_info: Dict[str, Any] = {}

        # Stateful exploration rewards, PER-EPISODE scope (cleared on reset):
        # visited-set novelty + first-time-only milestones. Scope is
        # deliberate — a run-persistent set would make the same state pay
        # differently as training progresses (non-stationary reward, poisons
        # value learning). Per-episode, the agent learns a ROUTE: sweep new
        # territory efficiently every life.
        self._visited: Dict[str, set] = {}
        self._milestones_fired: set = set()
        self._counter_state: Dict[str, Dict[tuple, int]] = {}

        # Optional raw A/V tap: called (frame_rgb, audio_int16) for EVERY
        # emulator frame inside the frame_skip loop — full 60fps + sound for
        # live streaming, independent of the 64x64 agent observation.
        self.frame_callback = None

    def _process_observation(self, obs: np.ndarray) -> np.ndarray:
        obs = cv2.resize(obs, (self.screen_size, self.screen_size), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            obs = np.expand_dims(obs, axis=-1)
        # HWC -> CHW
        obs = np.transpose(obs, (2, 0, 1))
        return obs

    def reset(self, **kwargs):
        if "seed" in kwargs:
            np.random.seed(kwargs["seed"])

        # Multi-track rotation: swap the emulator's restore-state before the
        # underlying reset applies it (RetroEnv.reset -> em.set_state)
        if len(self.initial_states) > 1:
            name = self.initial_states[np.random.randint(len(self.initial_states))]
            self._env.initial_state = self._state_bytes[name]
            self._env.statename = f"{name}.state"
            self._current_track = name
        else:
            self._current_track = self.initial_state

        obs, info = self._env.reset(**kwargs)
        info["track_state"] = self._current_track
        self.episode_step = 0
        self.episode_reward = 0.0
        self.prev_info = info
        self._visited = {}
        self._milestones_fired = set()
        self._counter_state = {}
        return {"rgb": self._process_observation(obs)}, info

    def step(self, action):
        # Convert discrete action index to MultiBinary array
        if isinstance(action, np.ndarray):
            action = int(action.flat[0]) if action.size > 0 else 0
        elif isinstance(action, list):
            action = int(action[0]) if action else 0
        else:
            action = int(action)
        action = max(0, min(action, len(self.action_mappings) - 1))
        multibinary_action = np.array(self.action_mappings[action], dtype=np.int8)

        terminated = False
        truncated = False

        for _ in range(self.frame_skip):
            obs, _, done, trunc, info = self._env.step(multibinary_action)
            if self.frame_callback is not None:
                self.frame_callback(obs, self._env.em.get_audio())
            if done or trunc:
                terminated = done
                truncated = trunc
                break

        # Reward is computed ONCE per agent step, from training.json only —
        # the single source of truth. stable-retro's scenario.json reward is
        # discarded: it used to be silently ADDED to this shaping, drowning
        # training.json's coefficients ~1000:1. Per-subframe evaluation
        # against a prev_info that only advances below would also overcount
        # deltas (d+2d+3d+4d = 10d over a 4-frame skip instead of 4d).
        total_reward = self._calculate_reward(info)
        # Spawn artifact suppression: F-Zero's track-position counter parks
        # ~198 units before the line and SNAPS forward when first crossed
        # (~step 6), paying a large unearned lump for merely existing.
        # Zeroing shaping for a short warmup kills exactly that; identical
        # snaps at LAP crossings later in the episode are real completion
        # signal and stay.
        if self.episode_step < self.training_config.get("reward", {}).get("warmup_steps", 0):
            total_reward = 0.0

        processed_obs = self._process_observation(obs)

        if self._check_done(info):
            terminated = True

        self.episode_step += 1
        self.episode_reward += total_reward
        self.prev_info = info
        # Surface which rotation track this episode runs on — reset() sets it,
        # but episode-end consumers read the TERMINAL STEP's info, not reset's
        info["track_state"] = getattr(self, "_current_track", self.initial_state)

        return {"rgb": processed_obs}, total_reward, terminated, truncated, info

    def _calculate_reward(self, info: Dict[str, Any]) -> float:
        reward = 0.0
        reward_config = self.training_config.get("reward", {}).get("variables", {})

        for var_name, var_cfg in reward_config.items():
            if var_name not in info:
                continue

            # Penalty (health loss, etc.). max_delta (when set) caps the
            # charged loss: packed bytes (e.g. Zelda 0x66F, containers in
            # the high nibble) can swing by 16 on transients — a capped
            # penalty pays the real event, never the encoding artifact.
            if "penalty" in var_cfg and var_name in self.prev_info:
                loss = max(0, self.prev_info[var_name] - info[var_name])
                pcap = var_cfg.get("max_delta")
                if pcap:
                    loss = min(loss, pcap)
                reward -= loss * var_cfg["penalty"]
                # Optional smaller payment for REGAINING the variable (pit
                # strips). MUST stay < penalty: then any deliberate
                # damage->heal cycle is strictly net-negative (no farming),
                # while a damaged agent finally gets paid at the moment of
                # healing instead of only via distant survival value.
                heal = var_cfg.get("heal_reward")
                if heal:
                    gained = max(0, info[var_name] - self.prev_info[var_name])
                    reward += gained * heal

            # Direct reward on the variable's delta
            if "reward" in var_cfg and "op" not in var_cfg and "mode" not in var_cfg:
                if var_name in self.prev_info:
                    delta = info[var_name] - self.prev_info[var_name]
                    # Fixed-width counters (e.g. F-Zero's u2 track position,
                    # which sits near 65535 before the start line) wrap; take
                    # the shortest modular distance so a wrap forward is a
                    # small +, a wrap backward a small -. Without this, a
                    # positive-only clip pays +wrap for driving backward
                    # across the wrap point — a reward fountain.
                    wrap = var_cfg.get("wrap")
                    if wrap:
                        delta = (delta + wrap // 2) % wrap - wrap // 2
                    # Bound single-step deltas: legit values are tiny (0-1
                    # per step racing, ~198 lap-line snaps) — anything huge
                    # is a teleport/glitch, not progress.
                    cap = var_cfg.get("max_delta")
                    if cap:
                        delta = max(-cap, min(cap, delta))
                    if var_cfg.get("delta") == "signed":
                        gain = delta
                    else:
                        gain = max(0, delta)
                    reward += gain * var_cfg["reward"]

            # Binary mode (op-based)
            mode = var_cfg.get("mode", "binary" if "op" in var_cfg else None)
            if mode == "binary" and "op" in var_cfg and "reward" in var_cfg:
                ref = var_cfg.get("reference", 0)
                val = info[var_name]
                op = OP_ALIASES.get(var_cfg["op"], var_cfg["op"])
                if op == "greater-than" and val > ref:
                    reward += var_cfg["reward"]
                elif op == "less-than" and val < ref:
                    reward += var_cfg["reward"]
                elif op == "equal" and val == ref:
                    reward += var_cfg["reward"]

            elif mode == "quadratic":
                max_val = var_cfg.get("max_speed", var_cfg.get("max_value", 500.0))
                base_r = var_cfg.get("base_reward", 0.1)
                coeff = var_cfg.get("scaling_coefficient", 1.0)
                power = var_cfg.get("power", 2.0)
                threshold = var_cfg.get("min_threshold", 0.0)
                val = info[var_name]
                if val >= threshold and max_val > 0:
                    norm = min(val / max_val, 1.0)
                    reward += coeff * base_r * (norm ** power)

            elif mode == "linear":
                max_val = var_cfg.get("max_speed", var_cfg.get("max_value", 500.0))
                base_r = var_cfg.get("base_reward", 0.1)
                threshold = var_cfg.get("min_threshold", 0.0)
                val = info[var_name]
                if val >= threshold and max_val > 0:
                    norm = min(val / max_val, 1.0)
                    reward += base_r * norm

            elif mode == "exponential":
                max_val = var_cfg.get("max_speed", var_cfg.get("max_value", 500.0))
                base_r = var_cfg.get("base_reward", 0.1)
                coeff = var_cfg.get("scaling_coefficient", 1.0)
                threshold = var_cfg.get("min_threshold", 0.0)
                val = info[var_name]
                if val >= threshold and max_val > 0:
                    norm = min(val / max_val, 1.0)
                    reward += coeff * base_r * (np.exp(norm) - 1) / (np.e - 1)

        # ---- Stateful exploration rewards (breadcrumb framework) ----
        # Both blocks run every step, INCLUDING reward-warmup steps: step()
        # zeroes the returned total during warmup but the sets still update,
        # so a milestone/screen already true at spawn is consumed silently
        # and can never pay an unearned lump (same rule as warmup_steps).

        # First-time-only milestones: one-shot payout the first time an op
        # condition becomes true this episode (e.g. Zelda sword flag > 0).
        for name, ms_cfg in self.training_config.get("reward", {}).get("milestones", {}).items():
            if name in self._milestones_fired:
                continue
            val = info.get(ms_cfg.get("var"))
            if val is None:
                continue
            ref = ms_cfg.get("reference", 0)
            op = OP_ALIASES.get(ms_cfg.get("op"), ms_cfg.get("op"))
            if (
                (op == "greater-than" and val > ref)
                or (op == "less-than" and val < ref)
                or (op == "equal" and val == ref)
            ):
                self._milestones_fired.add(name)
                reward += ms_cfg.get("reward", 0.0)

        # Visited-set novelty: pay once per NEW combination of the listed
        # RAM variables this episode (e.g. keys ["level","screen_id"] pays
        # per new screen). Inherently farm-proof: revisits pay nothing.
        for name, nv_cfg in self.training_config.get("reward", {}).get("novelty", {}).items():
            key = tuple(info.get(k) for k in nv_cfg.get("keys", []))
            if any(v is None for v in key):
                continue
            seen = self._visited.setdefault(name, set())
            if key not in seen:
                seen.add(key)
                reward += nv_cfg.get("reward", 0.0)

        # Counted events per place with diminishing returns (see
        # score_counters docstring): e.g. kills on each screen pay
        # geometrically less and cap out — refarming a screen tends to zero.
        counters_cfg = self.training_config.get("reward", {}).get("counters", {})
        if counters_cfg:
            reward += score_counters(
                counters_cfg, self.prev_info, info, self._counter_state
            )

        return reward

    def _check_done(self, info: Dict[str, Any]) -> bool:
        done_config = self.training_config.get("done", {}).get("variables", {})

        for var_name, var_cfg in done_config.items():
            if var_name not in info:
                continue
            op = OP_ALIASES.get(var_cfg.get("op"), var_cfg.get("op"))
            ref = var_cfg.get("reference", 0)
            val = info[var_name]
            if op == "less-than" and val < ref:
                return True
            elif op == "greater-than" and val > ref:
                return True
            elif op == "equal" and val == ref:
                return True
        return False
