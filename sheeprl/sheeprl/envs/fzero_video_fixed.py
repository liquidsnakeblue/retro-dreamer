"""F-Zero SNES environment wrapper with proper video recording that captures all frames."""

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


class FZeroWrapperWithFullVideoCapture(gym.Wrapper):
    """F-Zero wrapper that properly captures all frames for video recording.
    
    This wrapper handles:
    - Loading custom ROM and state files
    - Applying reward shaping based on JSON configs
    - Frame stacking and preprocessing
    - Action space configuration
    - FULL video capture of all frames (not just observation frames)
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
        video_folder: Optional[str] = None,
        video_fps: int = 60,  # SNES native framerate
        resize_to: Optional[Tuple[int, int]] = (64, 64),
    ):
        """Initialize F-Zero environment with full video capture support.
        
        Args:
            game_path: Path to F-Zero game files directory
            initial_state: Name of the state file to start from
            render_mode: Rendering mode for the environment
            frame_stack: Number of frames to stack
            frame_skip: Number of frames to skip between actions
            max_episode_steps: Maximum steps per episode
            capture_video: Whether to capture video during evaluation
            video_folder: Directory to save videos
            video_fps: FPS for video output (default 60 for SNES)
            resize_to: Size to resize observations to
        """
        self.game_path = game_path
        self.initial_state = initial_state
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.capture_video = capture_video
        self.video_folder = video_folder
        self.video_fps = video_fps
        self.resize_to = resize_to
        
        # Video recording state
        self.video_frames = []
        self.is_recording = False
        self.episode_count = 0
        
        # Create video folder if capturing
        if self.capture_video:
            if not self.video_folder:
                raise ValueError("video_folder must be specified when capture_video=True")
            os.makedirs(self.video_folder, exist_ok=True)
        
        # Load configuration files
        import json
        with open(os.path.join(game_path, "training.json"), "r") as f:
            self.training_config = json.load(f)
        
        with open(os.path.join(game_path, "data.json"), "r") as f:
            self.data_config = json.load(f)
        
        # Create base environment
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
        
        # Start new video recording if enabled
        if self.capture_video:
            if self.is_recording and self.video_frames:
                self._save_video()
            self.video_frames = []
            self.is_recording = True
            # Capture initial frame at full resolution
            frame = self._env.render()
            if frame is not None:
                self.video_frames.append(frame)
        
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
        """Execute action and return step information, capturing all frames."""
        total_reward = 0.0
        
        # Frame skipping with full frame capture
        for i in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self._env.step(action)
            
            # Capture frame BEFORE processing (at full resolution)
            if self.capture_video and self.is_recording:
                frame = self._env.render()
                if frame is not None:
                    self.video_frames.append(frame)
            
            # Apply custom reward shaping
            shaped_reward = self._calculate_reward(info, reward)
            total_reward += shaped_reward
            
            if terminated or truncated:
                break
        
        # Resize observation if needed (for agent, not video)
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
        
        # Save video if episode ended
        if (terminated or truncated) and self.capture_video and self.is_recording:
            self._save_video()
            self.is_recording = False
        
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
        
        # Speed reward
        if "speed" in reward_config and "speed" in info:
            speed_config = reward_config["speed"]
            if "reward" in speed_config and "op" in speed_config:
                if speed_config["op"] == "greater-than":
                    ref = speed_config.get("reference", 0)
                    if info["speed"] > ref:
                        reward += speed_config["reward"]
        
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
    
    def _save_video(self):
        """Save captured frames to video file at proper framerate."""
        if not self.video_frames:
            return
        
        video_name = f"fzero_episode_{self.episode_count:06d}_fullframes.mp4"
        video_path = os.path.join(self.video_folder, video_name)
        
        # Get frame dimensions from first frame
        height, width = self.video_frames[0].shape[:2]
        
        # Use H264 codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, self.video_fps, (width, height))
        
        # Write all frames
        for frame in self.video_frames:
            # OpenCV expects BGR, but gym environments typically provide RGB
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
        
        out.release()
        
        print(f"Saved video to {video_path}")
        print(f"  Total frames: {len(self.video_frames)}")
        print(f"  Duration: {len(self.video_frames) / self.video_fps:.2f} seconds")
        print(f"  Frame rate: {self.video_fps} FPS")
        
        self.video_frames = []
        self.episode_count += 1
    
    def render(self):
        """Render the environment."""
        return self._env.render()
    
    def close(self):
        """Close the environment and save any remaining video."""
        if self.capture_video and self.is_recording and self.video_frames:
            self._save_video()
        self._env.close()


def make_fzero_env_with_proper_video(
    cfg: Dict[str, Any],
    seed: int,
    rank: int,
    log_dir: str,
    run_name: str,
    capture_video: bool = False,
) -> gym.Env:
    """Factory function to create F-Zero environment with proper video capture.
    
    This version captures ALL frames at the proper framerate, not just observation frames.
    
    Args:
        cfg: Configuration dictionary
        seed: Random seed
        rank: Process rank for distributed training
        log_dir: Directory for logs
        run_name: Name of the run
        capture_video: Whether to capture video
    
    Returns:
        Configured F-Zero environment with proper video recording
    """
    video_folder = None
    if capture_video:
        video_folder = os.path.join(log_dir, f"{run_name}_videos", f"process_{rank}")
    
    env = FZeroWrapperWithFullVideoCapture(
        game_path=cfg.get("game_path", "/root/F-Zero Game Files"),
        initial_state=cfg.get("initial_state", "go.state"),
        render_mode="rgb_array",
        frame_stack=cfg.get("frame_stack", 4),
        frame_skip=cfg.get("frame_skip", 4),
        max_episode_steps=cfg.get("max_episode_steps", 10000),
        capture_video=capture_video,
        video_folder=video_folder,
        video_fps=cfg.get("video_fps", 60),  # SNES native framerate
    )
    
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    
    return env