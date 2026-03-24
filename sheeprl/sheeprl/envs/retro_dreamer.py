"""Generalized retro environment wrapper for DreamerV3 via SheepRL.

Replaces the F-Zero-specific wrappers with a single game-agnostic class.
Loads per-game config from the games/ directory: actions.json, training.json, data.json.
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

if not _IS_STABLE_RETRO_AVAILABLE:
    raise ModuleNotFoundError("stable-retro is required for retro-dreamer")


class RetroDreamerWrapper(gym.Wrapper):
    """Game-agnostic retro wrapper for DreamerV3 / SheepRL.

    - Registers a custom integration directory so retro can find the game
    - Loads training.json for reward shaping and done conditions
    - Loads actions.json for the discrete action mapping
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
        **kwargs,
    ):
        self.game_id = game_id
        self.game_dir = Path(game_dir)
        self.initial_state = initial_state.replace(".state", "")
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

        # Load actions config — our extension
        actions_path = self.game_dir / "actions.json"
        if actions_path.exists():
            with open(actions_path) as f:
                actions_data = json.load(f)
                self.action_mappings = [a["buttons"] for a in actions_data["actions"]]
        else:
            # Fallback: use retro's FILTERED actions (let retro decide)
            self.action_mappings = None  # handled below after env creation

        # Determine state
        state = self.initial_state
        states_dir = self.game_dir / "states"
        if states_dir.exists():
            state_file = states_dir / f"{self.initial_state}.state"
            if state_file.exists():
                state = str(state_file)

        # Create retro environment
        use_restricted = retro.Actions.ALL if self.action_mappings else retro.Actions.FILTERED
        self._env = retro.make(
            game=game_id,
            state=state,
            inttype=inttype,
            use_restricted_actions=use_restricted,
            render_mode=render_mode,
        )

        # If no actions.json, derive action mappings from env's MultiBinary space
        if self.action_mappings is None:
            n_buttons = self._env.action_space.shape[0] if hasattr(self._env.action_space, 'shape') else 12
            # Default: individual buttons + no-op
            self.action_mappings = [[0] * n_buttons]  # no-op
            for i in range(n_buttons):
                row = [0] * n_buttons
                row[i] = 1
                self.action_mappings.append(row)

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

        obs, info = self._env.reset(**kwargs)
        self.episode_step = 0
        self.episode_reward = 0.0
        self.prev_info = info
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

        total_reward = 0.0
        terminated = False
        truncated = False

        for _ in range(self.frame_skip):
            obs, reward, done, trunc, info = self._env.step(multibinary_action)
            shaped_reward = self._calculate_reward(info, reward)
            total_reward += shaped_reward
            if done or trunc:
                terminated = done
                truncated = trunc
                break

        processed_obs = self._process_observation(obs)

        if self._check_done(info):
            terminated = True

        self.episode_step += 1
        self.episode_reward += total_reward
        self.prev_info = info

        return {"rgb": processed_obs}, total_reward, terminated, truncated, info

    def _calculate_reward(self, info: Dict[str, Any], base_reward: float) -> float:
        reward = base_reward
        reward_config = self.training_config.get("reward", {}).get("variables", {})

        for var_name, var_cfg in reward_config.items():
            if var_name not in info:
                continue

            # Penalty (health loss, etc.)
            if "penalty" in var_cfg and var_name in self.prev_info:
                loss = max(0, self.prev_info[var_name] - info[var_name])
                reward -= loss * var_cfg["penalty"]

            # Direct reward with comparison operator
            if "reward" in var_cfg and "op" not in var_cfg and "mode" not in var_cfg:
                if var_name in self.prev_info:
                    gain = max(0, info[var_name] - self.prev_info[var_name])
                    reward += gain * var_cfg["reward"]

            # Binary mode (op-based)
            mode = var_cfg.get("mode", "binary" if "op" in var_cfg else None)
            if mode == "binary" and "op" in var_cfg and "reward" in var_cfg:
                ref = var_cfg.get("reference", 0)
                val = info[var_name]
                if var_cfg["op"] == "greater-than" and val > ref:
                    reward += var_cfg["reward"]
                elif var_cfg["op"] == "less-than" and val < ref:
                    reward += var_cfg["reward"]
                elif var_cfg["op"] == "equal" and val == ref:
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

        return reward

    def _check_done(self, info: Dict[str, Any]) -> bool:
        done_config = self.training_config.get("done", {}).get("variables", {})

        for var_name, var_cfg in done_config.items():
            if var_name not in info:
                continue
            op = var_cfg.get("op")
            ref = var_cfg.get("reference", 0)
            val = info[var_name]
            if op == "less-than" and val < ref:
                return True
            elif op == "greater-than" and val > ref:
                return True
            elif op == "equal" and val == ref:
                return True
        return False
