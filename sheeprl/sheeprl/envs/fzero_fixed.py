"""F-Zero SNES environment wrapper for SheepRL using stable-retro."""

import os
from typing import Any, Dict, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
import retro
from gymnasium import spaces

from sheeprl.utils.imports import _IS_STABLE_RETRO_AVAILABLE

if not _IS_STABLE_RETRO_AVAILABLE:
    raise ModuleNotFoundError("stable-retro is required for F-Zero environment")


class FZeroSheepRLWrapper(gym.Wrapper):
    """F-Zero wrapper designed specifically for SheepRL compatibility.
    
    This wrapper properly handles:
    - Single environment instances (retro limitation)
    - Dict observation spaces (SheepRL requirement)
    - Proper shape handling for vectorized environments
    - Custom reward shaping and termination conditions
    """
    
    def __init__(
        self,
        id: str = "F-Zero-go",  # SheepRL expects an 'id' parameter
        render_mode: str = "rgb_array",
        screen_size: int = 64,
        frame_stack: int = 1,
        frame_skip: int = 4,
        seed: Optional[int] = None,
        grayscale: bool = False,
        game_path: str = "/root/F-Zero Game Files",
        initial_state: str = "go",
        **kwargs  # Capture any additional kwargs
    ):
        """Initialize F-Zero environment with SheepRL-compatible interface."""
        self.game_path = game_path
        self.initial_state = initial_state.replace(".state", "")  # Remove extension if present
        self.frame_skip = frame_skip
        self.screen_size = screen_size
        self.grayscale = grayscale
        
        # Load configuration files
        import json
        with open(os.path.join(game_path, "training.json"), "r") as f:
            self.training_config = json.load(f)
        
        with open(os.path.join(game_path, "data.json"), "r") as f:
            self.data_config = json.load(f)
        
        # Close any existing retro environment before creating a new one
        try:
            # Try to access and close any existing retro emulator
            import gc
            if hasattr(retro, '_emulator') and retro._emulator is not None:
                retro._emulator.close()
                retro._emulator = None
            gc.collect()
        except:
            pass
            
        # Create base retro environment
        self._env = retro.make(
            game="F-Zero",
            state=self.initial_state,
            use_restricted_actions=retro.Actions.FILTERED,
            render_mode=render_mode,
        )
        
        super().__init__(self._env)
        
        # Set seed if provided
        if seed is not None:
            self.seed(seed)
        
        # Create observation space compatible with SheepRL
        # Use 'rgb' key as per SheepRL conventions
        # Note: Frame stacking is handled by SheepRL's FrameStack wrapper
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
        
    def seed(self, seed: int):
        """Set random seed."""
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        
    def _process_observation(self, obs: np.ndarray) -> np.ndarray:
        """Process observation to match SheepRL expectations."""
        # Resize to screen_size
        obs = cv2.resize(obs, (self.screen_size, self.screen_size), interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale if needed
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            obs = np.expand_dims(obs, axis=-1)
        
        # Convert from HWC to CHW format (SheepRL uses channel-first)
        obs = np.transpose(obs, (2, 0, 1))
        
        return obs
    
    def reset(self, **kwargs):
        """Reset environment and return SheepRL-compatible observation."""
        # Ensure seed is properly passed to retro environment
        if 'seed' in kwargs:
            # Retro environments don't support seed parameter directly
            # but we can set numpy random seed for any random elements
            import numpy as np
            np.random.seed(kwargs['seed'])
        
        obs, info = self._env.reset(**kwargs)
        
        self.episode_step = 0
        self.episode_reward = 0.0
        self.prev_info = info
        
        # Process observation
        processed_obs = self._process_observation(obs)
        
        return {"rgb": processed_obs}, info
    
    def step(self, action):
        """Execute action with frame skipping and custom rewards."""
        total_reward = 0.0
        terminated = False
        truncated = False
        
        # Frame skipping
        for _ in range(self.frame_skip):
            obs, reward, done, trunc, info = self._env.step(action)
            
            # Apply custom reward shaping
            shaped_reward = self._calculate_reward(info, reward)
            total_reward += shaped_reward
            
            if done or trunc:
                terminated = done
                truncated = trunc
                break
        
        # Process observation
        processed_obs = self._process_observation(obs)
        
        # Check custom termination conditions
        if self._check_done(info):
            terminated = True
        
        self.episode_step += 1
        self.episode_reward += total_reward
        self.prev_info = info
        
        return {"rgb": processed_obs}, total_reward, terminated, truncated, info
    
    def _calculate_reward(self, info: Dict[str, Any], base_reward: float) -> float:
        """Calculate shaped reward based on training configuration."""
        reward = base_reward
        
        reward_config = self.training_config.get("reward", {}).get("variables", {})
        
        # Health penalty
        if "health" in reward_config and "health" in info:
            health_config = reward_config["health"]
            if "penalty" in health_config:
                # Penalty for losing health
                if "health" in self.prev_info:
                    health_loss = max(0, self.prev_info["health"] - info["health"])
                    reward -= health_loss * health_config["penalty"]
        
        # Position reward
        if "pos" in reward_config and "pos" in info:
            pos_config = reward_config["pos"]
            if "reward" in pos_config:
                # Reward for advancing position
                if "pos" in self.prev_info:
                    pos_gain = max(0, info["pos"] - self.prev_info["pos"])
                    reward += pos_gain * pos_config["reward"]
        
        # Speed reward - Enhanced with multiple modes
        if "speed" in reward_config and "speed" in info:
            speed_config = reward_config["speed"]
            speed = info["speed"]
            
            mode = speed_config.get("mode", "binary")
            
            if mode == "binary":
                # Original binary reward system (backward compatible)
                if "reward" in speed_config and "op" in speed_config:
                    if speed_config["op"] == "greater-than":
                        ref = speed_config.get("reference", 0)
                        if speed > ref:
                            reward += speed_config["reward"]
                            
            elif mode == "quadratic":
                # Quadratic reward scaling
                max_speed = speed_config.get("max_speed", 500.0)
                base_reward = speed_config.get("base_reward", 0.1)
                scaling_coeff = speed_config.get("scaling_coefficient", 1.0)
                power = speed_config.get("power", 2.0)
                min_threshold = speed_config.get("min_threshold", 0.0)
                
                if speed >= min_threshold and max_speed > 0:
                    # Scale speed to [0, 1] range
                    normalized_speed = min(speed / max_speed, 1.0)
                    # Apply power scaling and coefficient
                    speed_reward = scaling_coeff * base_reward * (normalized_speed ** power)
                    reward += speed_reward
                    
            elif mode == "linear":
                # Linear reward scaling
                max_speed = speed_config.get("max_speed", 500.0)
                base_reward = speed_config.get("base_reward", 0.1)
                min_threshold = speed_config.get("min_threshold", 0.0)
                
                if speed >= min_threshold and max_speed > 0:
                    normalized_speed = min(speed / max_speed, 1.0)
                    speed_reward = base_reward * normalized_speed
                    reward += speed_reward
                    
            elif mode == "exponential":
                # Exponential reward scaling
                max_speed = speed_config.get("max_speed", 500.0)
                base_reward = speed_config.get("base_reward", 0.1)
                scaling_coeff = speed_config.get("scaling_coefficient", 1.0)
                min_threshold = speed_config.get("min_threshold", 0.0)
                
                if speed >= min_threshold and max_speed > 0:
                    normalized_speed = min(speed / max_speed, 1.0)
                    # Exponential scaling: e^(normalized_speed) - 1
                    speed_reward = scaling_coeff * base_reward * (np.exp(normalized_speed) - 1) / (np.e - 1)
                    reward += speed_reward
        
        return reward
    
    def _check_done(self, info: Dict[str, Any]) -> bool:
        """Check if episode should terminate based on training configuration."""
        done_config = self.training_config.get("done", {}).get("variables", {})
        
        # Check health condition
        if "health" in done_config and "health" in info:
            health_config = done_config["health"]
            if health_config.get("op") == "less-than":
                ref = health_config.get("reference", 0)
                if info["health"] < ref:
                    return True
        
        # Check reverse condition
        if "reverse" in done_config and "reverse" in info:
            reverse_config = done_config["reverse"]
            if reverse_config.get("op") == "equal":
                ref = reverse_config.get("reference", 0)
                if info["reverse"] == ref:
                    return True
        
        return False