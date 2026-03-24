"""Battletoads NES environment wrapper for SheepRL using stable-retro."""

import os
from typing import Any, Dict, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
import retro
from gymnasium import spaces

from sheeprl.utils.imports import _IS_STABLE_RETRO_AVAILABLE

if not _IS_STABLE_RETRO_AVAILABLE:
    raise ModuleNotFoundError("stable-retro is required for Battletoads environment")


class BattletoadsSheepRLWrapper(gym.Wrapper):
    """Battletoads wrapper designed specifically for SheepRL compatibility.

    This wrapper properly handles:
    - Single environment instances (retro limitation)
    - Dict observation spaces (SheepRL requirement)
    - Proper shape handling for vectorized environments
    - Custom reward shaping for beat 'em up gameplay
    """

    def __init__(
        self,
        id: str = "Battletoads-Level1",
        render_mode: str = "rgb_array",
        screen_size: int = 64,
        frame_stack: int = 1,
        frame_skip: int = 4,
        seed: Optional[int] = None,
        grayscale: bool = False,
        game_path: str = "/root/Battletoads Game Files",
        initial_state: str = "Level1",
        **kwargs
    ):
        """Initialize Battletoads environment with SheepRL-compatible interface."""
        self.game_path = game_path
        self.initial_state = initial_state.replace(".state", "")
        self.frame_skip = frame_skip
        self.screen_size = screen_size
        self.grayscale = grayscale

        # Load configuration files
        import json
        training_path = os.path.join(game_path, "training.json")
        if os.path.exists(training_path):
            with open(training_path, "r") as f:
                self.training_config = json.load(f)
        else:
            self.training_config = {}

        data_path = os.path.join(game_path, "data.json")
        if os.path.exists(data_path):
            with open(data_path, "r") as f:
                self.data_config = json.load(f)
        else:
            self.data_config = {}

        # Close any existing retro environment
        try:
            import gc
            gc.collect()
        except:
            pass

        # Create base retro environment
        self._env = retro.make(
            game="Battletoads-Nes",
            state=self.initial_state,
            use_restricted_actions=retro.Actions.FILTERED,
            render_mode=render_mode,
        )

        super().__init__(self._env)

        if seed is not None:
            self.seed(seed)

        # Create observation space compatible with SheepRL
        if grayscale:
            obs_shape = (1, screen_size, screen_size)
        else:
            obs_shape = (3, screen_size, screen_size)

        self.observation_space = spaces.Dict({
            "rgb": spaces.Box(
                low=0,
                high=255,
                shape=obs_shape,
                dtype=np.uint8
            )
        })

        # Track episode info
        self.episode_step = 0
        self.episode_reward = 0.0
        self.prev_info = {}
        self.max_x_reached = 0

    def seed(self, seed: int):
        """Set random seed."""
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def _process_observation(self, obs: np.ndarray) -> np.ndarray:
        """Process observation to match SheepRL expectations."""
        obs = cv2.resize(obs, (self.screen_size, self.screen_size), interpolation=cv2.INTER_AREA)

        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            obs = np.expand_dims(obs, axis=-1)

        obs = np.transpose(obs, (2, 0, 1))
        return obs

    def reset(self, **kwargs):
        """Reset environment and return SheepRL-compatible observation."""
        if 'seed' in kwargs:
            np.random.seed(kwargs['seed'])

        obs, info = self._env.reset(**kwargs)

        self.episode_step = 0
        self.episode_reward = 0.0
        self.prev_info = info
        self.max_x_reached = 0

        processed_obs = self._process_observation(obs)
        return {"rgb": processed_obs}, info

    def step(self, action):
        """Execute action with frame skipping and custom rewards."""
        total_reward = 0.0
        terminated = False
        truncated = False

        for _ in range(self.frame_skip):
            obs, reward, done, trunc, info = self._env.step(action)

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
        """Calculate shaped reward for Battletoads."""
        reward = base_reward

        reward_config = self.training_config.get("reward", {}).get("variables", {})

        # Score reward - reward for defeating enemies
        if "score" in reward_config and "score" in info:
            score_config = reward_config["score"]
            if "reward" in score_config and "score" in self.prev_info:
                score_gain = max(0, info["score"] - self.prev_info["score"])
                reward += score_gain * score_config["reward"]

        # X position reward - reward for progressing through level
        if "x_pos" in reward_config and "x_pos" in info:
            x_config = reward_config["x_pos"]
            current_x = info.get("x_pos", 0)

            # Only reward forward progress (no reward for going backwards)
            if current_x > self.max_x_reached:
                progress = current_x - self.max_x_reached
                self.max_x_reached = current_x

                if x_config.get("mode") == "linear":
                    reward += progress * x_config.get("reward", 0.1)
                else:
                    reward += x_config.get("reward", 0.1)

        # Screen X reward - reward for screen scrolling progress
        if "screen_x" in reward_config and "screen_x" in info:
            screen_config = reward_config["screen_x"]
            if "screen_x" in self.prev_info:
                screen_progress = max(0, info["screen_x"] - self.prev_info["screen_x"])
                if screen_progress > 0:
                    if screen_config.get("mode") == "linear":
                        reward += screen_progress * screen_config.get("reward", 0.001)
                    else:
                        reward += screen_config.get("reward", 0.01)

        # Lives penalty - penalty for losing a life
        if "lives" in reward_config and "lives" in info:
            lives_config = reward_config["lives"]
            if "penalty" in lives_config and "lives" in self.prev_info:
                lives_lost = max(0, self.prev_info["lives"] - info["lives"])
                if lives_lost > 0:
                    reward -= lives_lost * lives_config["penalty"]

        # Health penalty - penalty for taking damage
        # Note: Health can fluctuate (6→9→6 during invincibility), so only penalize
        # decreases when previous health was <= 6 (max health) to avoid false penalties
        if "health" in reward_config and "health" in info:
            health_config = reward_config["health"]
            if "penalty" in health_config and "health" in self.prev_info:
                prev_health = self.prev_info["health"]
                curr_health = info["health"]
                # Only count damage if prev_health was in normal range (<=6)
                # and health actually decreased (not invincibility state changes)
                if prev_health <= 6 and curr_health < prev_health:
                    health_lost = prev_health - curr_health
                    reward -= health_lost * health_config["penalty"]

        # Alive bonus - small reward for staying alive (denser signal)
        if "alive_bonus" in reward_config:
            reward += reward_config["alive_bonus"]

        return reward

    def _check_done(self, info: Dict[str, Any]) -> bool:
        """Check if episode should terminate."""
        done_config = self.training_config.get("done", {}).get("variables", {})

        if "lives" in done_config and "lives" in info:
            lives_config = done_config["lives"]
            if lives_config.get("op") == "less-than":
                ref = lives_config.get("reference", 0)
                if info["lives"] < ref:
                    return True

        return False


class BattletoadsDreamerWrapper(BattletoadsSheepRLWrapper):
    """Battletoads wrapper with Discrete action space for Dreamer v3 compatibility."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # NES controller: A, B, Select, Start, Up, Down, Left, Right
        # Battletoads uses: A (jump), B (attack), directions
        self.action_mappings = [
            # Index: [B, None, Select, Start, Up, Down, Left, Right, A]
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0: No-op
            [0, 0, 0, 0, 0, 0, 0, 1, 0],  # 1: Right
            [0, 0, 0, 0, 0, 0, 1, 0, 0],  # 2: Left
            [0, 0, 0, 0, 1, 0, 0, 0, 0],  # 3: Up
            [0, 0, 0, 0, 0, 1, 0, 0, 0],  # 4: Down
            [1, 0, 0, 0, 0, 0, 0, 0, 0],  # 5: B (attack)
            [0, 0, 0, 0, 0, 0, 0, 0, 1],  # 6: A (jump)
            [1, 0, 0, 0, 0, 0, 0, 1, 0],  # 7: Right + B (running attack)
            [1, 0, 0, 0, 0, 0, 1, 0, 0],  # 8: Left + B (running attack)
            [0, 0, 0, 0, 0, 0, 0, 1, 1],  # 9: Right + A (jump right)
            [0, 0, 0, 0, 0, 0, 1, 0, 1],  # 10: Left + A (jump left)
            [0, 0, 0, 0, 0, 1, 0, 0, 1],  # 11: Down + A (drop kick)
        ]

        self.action_space = spaces.Discrete(len(self.action_mappings))
        self._original_action_space = self._env.action_space

    def step(self, action):
        """Convert discrete action to MultiBinary and execute step."""
        if isinstance(action, np.ndarray):
            if action.ndim == 0:
                action = int(action.item())
            elif action.shape == ():
                action = int(action.item())
            else:
                action = int(action.flat[0])
        elif isinstance(action, list):
            action = int(action[0]) if len(action) > 0 else 0
        else:
            action = int(action)

        action = max(0, min(action, len(self.action_mappings) - 1))
        multibinary_action = np.array(self.action_mappings[action], dtype=np.int8)

        return super().step(multibinary_action)
