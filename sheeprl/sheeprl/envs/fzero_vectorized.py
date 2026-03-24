"""F-Zero environment with custom vectorization wrapper for SheepRL compatibility."""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import gymnasium as gym
import numpy as np
import retro
from gymnasium import spaces

from sheeprl.envs.fzero import FZeroWrapper
from sheeprl.utils.imports import _IS_STABLE_RETRO_AVAILABLE

if not _IS_STABLE_RETRO_AVAILABLE:
    raise ModuleNotFoundError("stable-retro is required for F-Zero environment")


class SingleEnvVecWrapper:
    """Wrapper that makes a single environment behave like a vectorized environment.
    
    This wrapper is specifically designed to handle the shape mismatch between
    retro (which only supports single environments) and SheepRL's expectation
    of vectorized environments.
    """
    
    def __init__(self, env: gym.Env):
        """Initialize the wrapper.
        
        Args:
            env: The single environment to wrap
        """
        self.env = env
        self.num_envs = 1
        self.single_observation_space = env.observation_space
        self.single_action_space = env.action_space
        
        # Create vectorized versions of spaces
        if isinstance(self.single_observation_space, spaces.Dict):
            self.observation_space = spaces.Dict({
                key: self._add_batch_dim(space) 
                for key, space in self.single_observation_space.items()
            })
        else:
            self.observation_space = self._add_batch_dim(self.single_observation_space)
            
        self.action_space = self._add_batch_dim(self.single_action_space)
        
    def _add_batch_dim(self, space: spaces.Space) -> spaces.Space:
        """Add batch dimension to a space."""
        if isinstance(space, spaces.Box):
            return spaces.Box(
                low=np.expand_dims(space.low, 0),
                high=np.expand_dims(space.high, 0),
                shape=(1,) + space.shape,
                dtype=space.dtype
            )
        elif isinstance(space, spaces.Discrete):
            # For discrete spaces, we don't change the space itself
            # but will handle the batch dimension in step/reset
            return space
        else:
            return space
    
    def _add_batch_to_obs(self, obs: Union[np.ndarray, Dict]) -> Union[np.ndarray, Dict]:
        """Add batch dimension to observation."""
        if isinstance(obs, dict):
            return {key: np.expand_dims(val, 0) for key, val in obs.items()}
        else:
            return np.expand_dims(obs, 0)
    
    def _remove_batch_from_action(self, action: np.ndarray) -> Any:
        """Remove batch dimension from action."""
        if isinstance(action, np.ndarray) and action.shape[0] == 1:
            return action[0]
        return action
    
    def reset(self, **kwargs) -> Tuple[Any, List[Dict]]:
        """Reset the environment and return initial observation with batch dimension."""
        obs, info = self.env.reset(**kwargs)
        # Add batch dimension
        obs_batch = self._add_batch_to_obs(obs)
        # Return info as a list
        return obs_batch, [info]
    
    def step(self, actions: np.ndarray) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Execute action and return step information with batch dimensions."""
        # Remove batch dimension from action
        action = self._remove_batch_from_action(actions)
        
        # Step the environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Add batch dimensions
        obs_batch = self._add_batch_to_obs(obs)
        reward_batch = np.array([reward], dtype=np.float32)
        terminated_batch = np.array([terminated], dtype=bool)
        truncated_batch = np.array([truncated], dtype=bool)
        
        return obs_batch, reward_batch, terminated_batch, truncated_batch, [info]
    
    def render(self):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment."""
        self.env.close()
    
    def __getattr__(self, name):
        """Forward other attributes to the wrapped environment."""
        return getattr(self.env, name)


class FZeroVectorizedWrapper(FZeroWrapper):
    """F-Zero wrapper that handles vectorization for SheepRL compatibility.
    
    This wrapper extends the base FZeroWrapper to properly handle the shape
    expectations of SheepRL's Dreamer v3 implementation.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the vectorized F-Zero wrapper."""
        super().__init__(*args, **kwargs)
        
        # Store original shapes
        self._single_obs_shape = self.observation_space["rgb"].shape
        
    def reset(self, **kwargs):
        """Reset with proper shape handling."""
        obs_dict, info = super().reset(**kwargs)
        
        # Ensure observation is in the expected format
        # SheepRL expects observations without explicit batch dimension in single env mode
        return obs_dict, info
    
    def step(self, action):
        """Step with proper shape handling."""
        obs_dict, reward, terminated, truncated, info = super().step(action)
        
        # Ensure shapes are correct
        return obs_dict, reward, terminated, truncated, info


def make_fzero_vectorized_env(
    seed: int,
    rank: int,
    log_dir: str,
    run_name: str,
    vector_env_idx: int = 0,
    **kwargs
) -> SingleEnvVecWrapper:
    """Factory function to create vectorized F-Zero environment for SheepRL.
    
    This function creates a single F-Zero environment and wraps it to behave
    like a vectorized environment, which is what SheepRL expects.
    
    Args:
        cfg: Configuration dictionary
        seed: Random seed
        rank: Process rank for distributed training
        log_dir: Directory for logs
        run_name: Name of the run
        vector_env_idx: Index of this environment in the vector
    
    Returns:
        Vectorized F-Zero environment
    """
    # Access the global config through hydra
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    
    # Get configuration from kwargs and environment
    game_path = kwargs.get("game_path", "/root/F-Zero Game Files")
    initial_state = kwargs.get("initial_state", "go.state")
    resize_to = kwargs.get("resize_to", [64, 64])
    video_size = kwargs.get("video_size", None)
    
    # Try to get env config from parent context
    frame_stack = kwargs.get("frame_stack", 4)
    frame_skip = kwargs.get("frame_skip", 4) 
    max_episode_steps = kwargs.get("max_episode_steps", 10000)
    capture_video = kwargs.get("capture_video", False)
    
    # Create single environment
    env = FZeroVectorizedWrapper(
        game_path=game_path,
        initial_state=initial_state,
        render_mode="rgb_array",
        frame_stack=frame_stack,
        frame_skip=frame_skip,
        max_episode_steps=max_episode_steps,
        capture_video=capture_video and rank == 0 and vector_env_idx == 0,
        video_size=video_size,
        resize_to=resize_to,
    )
    
    # Set seeds
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    
    # Add video recording wrapper if requested
    if capture_video and rank == 0 and vector_env_idx == 0:
        from gymnasium.wrappers import RecordVideo
        video_folder = os.path.join(log_dir, f"{run_name}_videos")
        os.makedirs(video_folder, exist_ok=True)
        
        # Calculate episode trigger based on video frequency
        video_freq = kwargs.get("video_freq", 5000)
        steps_per_episode = max_episode_steps
        episode_freq = max(1, video_freq // steps_per_episode)
        
        env = RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda episode_id: episode_id % episode_freq == 0,
            disable_logger=True,
        )
    
    # For single environment mode (sync_env=True), return unwrapped
    sync_env = kwargs.get("sync_env", True)
    if sync_env:
        return env
    
    # For vectorized mode, wrap in SingleEnvVecWrapper
    return SingleEnvVecWrapper(env)