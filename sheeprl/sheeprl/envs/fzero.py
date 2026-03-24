"""F-Zero SNES environment wrapper for SheepRL using stable-retro."""

import os
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import gymnasium as gym
import numpy as np
import retro
from gymnasium import spaces

from sheeprl.utils.imports import _IS_STABLE_RETRO_AVAILABLE

if not _IS_STABLE_RETRO_AVAILABLE:
    raise ModuleNotFoundError("stable-retro is required for F-Zero environment")


class FZeroWrapper(gym.Wrapper):
    """Wrapper for F-Zero SNES game using stable-retro.
    
    This wrapper handles:
    - Loading custom ROM and state files
    - Applying reward shaping based on JSON configs
    - Frame stacking and preprocessing
    - Action space configuration
    """
    
    def __init__(
        self,
        game_path: str = "/root/F-Zero Game Files",
        initial_state: str = "go.state",
        render_mode: str = "rgb_array",
        frame_stack: int = 4,
        frame_skip: int = 4,
        max_episode_steps: int = 10000,
        capture_video: bool = False,
        video_size: Optional[Tuple[int, int]] = None,
        resize_to: Optional[Tuple[int, int]] = (64, 64),
    ):
        """Initialize F-Zero environment.
        
        Args:
            game_path: Path to F-Zero game files directory
            initial_state: Name of the state file to start from
            render_mode: Rendering mode for the environment
            frame_stack: Number of frames to stack
            frame_skip: Number of frames to skip between actions
            max_episode_steps: Maximum steps per episode
            capture_video: Whether to capture video during evaluation
            video_size: Size for video capture (None uses original size)
        """
        self.game_path = game_path
        self.initial_state = initial_state
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.capture_video = capture_video
        self.video_size = video_size
        self.resize_to = resize_to
        
        # Load configuration files
        import json
        with open(os.path.join(game_path, "training.json"), "r") as f:
            self.training_config = json.load(f)
        
        with open(os.path.join(game_path, "data.json"), "r") as f:
            self.data_config = json.load(f)
        
        # Create base environment
        # Using the installed game in retro data path
        self._env = retro.make(
            game="F-Zero",
            state=initial_state.replace(".state", ""),
            use_restricted_actions=retro.Actions.FILTERED,
            render_mode=render_mode,
        )
        
        super().__init__(self._env)
        
        # Override observation space for frame stacking and resizing
        base_obs_shape = self._env.observation_space.shape
        
        # Determine final observation shape
        if self.resize_to:
            h, w = self.resize_to
            resized_shape = (h, w, 3)
        else:
            resized_shape = base_obs_shape
            
        if frame_stack > 1:
            # Stack frames along channel dimension for CNN compatibility
            h, w, _ = resized_shape
            obs_shape = (h, w, 3 * frame_stack)
            self.frames = np.zeros((frame_stack, h, w, 3), dtype=np.uint8)
        else:
            obs_shape = resized_shape
        
        # Create Dict observation space for Dreamer v3 compatibility
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
        
    def reset(self, **kwargs):
        """Reset the environment and return initial observation."""
        obs, info = self._env.reset(**kwargs)
        
        self.episode_step = 0
        self.episode_reward = 0.0
        self.prev_info = info
        
        # Resize observation if needed
        if self.resize_to:
            obs = cv2.resize(obs, self.resize_to[::-1], interpolation=cv2.INTER_AREA)
        
        # Initialize frame stack
        if self.frame_stack > 1:
            self.frames.fill(0)
            self.frames[-1] = obs
            # Stack frames along channel dimension
            obs = np.concatenate(self.frames, axis=-1)
        
        # Return observation as dict for Dreamer v3 compatibility
        obs_dict = {"rgb": obs}
        
        return obs_dict, info
    
    def step(self, action):
        """Execute action and return step information."""
        total_reward = 0.0
        
        # Frame skipping
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self._env.step(action)
            
            # Apply custom reward shaping
            shaped_reward = self._calculate_reward(info, reward)
            total_reward += shaped_reward
            
            if terminated or truncated:
                break
        
        # Resize observation if needed
        if self.resize_to:
            obs = cv2.resize(obs, self.resize_to[::-1], interpolation=cv2.INTER_AREA)
        
        # Update frame stack
        if self.frame_stack > 1:
            self.frames[:-1] = self.frames[1:]
            self.frames[-1] = obs
            # Stack frames along channel dimension
            obs = np.concatenate(self.frames, axis=-1)
        
        # Check custom termination conditions
        if self._check_done(info):
            terminated = True
        
        # Episode step limit
        self.episode_step += 1
        if self.episode_step >= self.max_episode_steps:
            truncated = True
        
        self.episode_reward += total_reward
        self.prev_info = info
        
        # Return observation as dict for Dreamer v3 compatibility
        obs_dict = {"rgb": obs}
        
        return obs_dict, total_reward, terminated, truncated, info
    
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
        
        # Speed reward (enhanced with multiple modes)
        if "speed" in reward_config and "speed" in info:
            speed_config = reward_config["speed"]
            speed = info["speed"]
            
            # Handle different reward modes
            mode = speed_config.get("mode", "binary")
            
            if mode == "binary":
                # Original binary reward (backward compatibility)
                if "reward" in speed_config and "op" in speed_config:
                    if speed_config["op"] == "greater-than":
                        ref = speed_config.get("reference", 0)
                        if speed > ref:
                            reward += speed_config["reward"]
            
            elif mode == "quadratic":
                # New quadratic scaling
                max_speed = speed_config.get("max_speed", 500.0)
                base_reward = speed_config.get("base_reward", 0.1)
                scaling_coeff = speed_config.get("scaling_coefficient", 1.0)
                power = speed_config.get("power", 2.0)
                min_threshold = speed_config.get("min_threshold", 0.0)
                
                if speed >= min_threshold:
                    normalized_speed = min(speed / max_speed, 1.0)
                    speed_reward = scaling_coeff * base_reward * (normalized_speed ** power)
                    reward += speed_reward
            
            elif mode == "linear":
                # Linear scaling option
                max_speed = speed_config.get("max_speed", 500.0)
                base_reward = speed_config.get("base_reward", 0.1)
                min_threshold = speed_config.get("min_threshold", 0.0)
                
                if speed >= min_threshold:
                    normalized_speed = min(speed / max_speed, 1.0)
                    speed_reward = base_reward * normalized_speed
                    reward += speed_reward
            
            elif mode == "exponential":
                # Exponential scaling option
                max_speed = speed_config.get("max_speed", 500.0)
                base_reward = speed_config.get("base_reward", 0.1)
                scaling_coeff = speed_config.get("scaling_coefficient", 1.0)
                min_threshold = speed_config.get("min_threshold", 0.0)
                
                if speed >= min_threshold:
                    normalized_speed = min(speed / max_speed, 1.0)
                    speed_reward = base_reward * (np.exp(normalized_speed * scaling_coeff) - 1) / (np.exp(scaling_coeff) - 1)
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
    
    def render(self):
        """Render the environment."""
        return self._env.render()
    
    def close(self):
        """Close the environment."""
        self._env.close()


def make_fzero_env(
    cfg: Dict[str, Any],
    seed: int,
    rank: int,
    log_dir: str,
    run_name: str,
    capture_video: bool = False,
) -> gym.Env:
    """Factory function to create F-Zero environment for SheepRL.
    
    Args:
        cfg: Configuration dictionary
        seed: Random seed
        rank: Process rank for distributed training
        log_dir: Directory for logs
        run_name: Name of the run
        capture_video: Whether to capture video
    
    Returns:
        Configured F-Zero environment
    """
    env = FZeroWrapper(
        game_path=cfg.get("game_path", "/root/F-Zero Game Files"),
        initial_state=cfg.get("initial_state", "go.state"),
        render_mode="rgb_array",
        frame_stack=cfg.get("frame_stack", 4),
        frame_skip=cfg.get("frame_skip", 4),
        max_episode_steps=cfg.get("max_episode_steps", 10000),
        capture_video=capture_video,
        video_size=cfg.get("video_size", None),
    )
    
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    
    # Add video recording wrapper if requested
    if capture_video:
        from gymnasium.wrappers import RecordVideo
        video_folder = os.path.join(log_dir, f"{run_name}_videos", f"process_{rank}")
        
        # Use original resolution for video capture
        env = RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda episode_id: True,  # Record every episode
            disable_logger=True,
        )
    
    return env