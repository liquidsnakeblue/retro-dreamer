"""Custom video recording wrapper that captures all frames, including those during frame skipping."""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FullFrameVideoRecordingWrapper(gym.Wrapper):
    """Video recording wrapper that captures ALL frames, including those skipped during frame_skip.
    
    This wrapper should be placed INSIDE the F-Zero wrapper to capture frames before they are skipped.
    It records frames at the actual game framerate, not the observation framerate.
    """
    
    def __init__(
        self,
        env: gym.Env,
        video_folder: str,
        episode_trigger: Optional[Callable[[int], bool]] = None,
        video_length: int = 0,
        name_prefix: str = "fzero",
        fps: int = 30,
        disable_logger: bool = True,
    ):
        """Initialize the video recording wrapper.
        
        Args:
            env: The environment to wrap
            video_folder: Directory to save videos
            episode_trigger: Function to determine which episodes to record
            video_length: Maximum length of video (0 for unlimited)
            name_prefix: Prefix for video filenames
            fps: Frames per second for the output video
            disable_logger: Whether to disable logging
        """
        super().__init__(env)
        
        self.video_folder = video_folder
        self.episode_trigger = episode_trigger or (lambda x: True)
        self.video_length = video_length
        self.name_prefix = name_prefix
        self.fps = fps
        self.disable_logger = disable_logger
        
        # Create video folder if it doesn't exist
        os.makedirs(self.video_folder, exist_ok=True)
        
        # Video recording state
        self.recording = False
        self.video_writer = None
        self.recorded_frames = []
        self.episode_id = 0
        self.step_id = 0
        
    def reset(self, **kwargs):
        """Reset the environment and start recording if needed."""
        obs, info = self.env.reset(**kwargs)
        
        # Close previous video if any
        self._close_video_recorder()
        
        # Check if we should record this episode
        if self.episode_trigger(self.episode_id):
            self.recording = True
            self.recorded_frames = []
            if not self.disable_logger:
                print(f"Starting video recording for episode {self.episode_id}")
        
        self.step_id = 0
        self.episode_id += 1
        
        # Capture initial frame
        if self.recording:
            frame = self.env.render()
            if frame is not None:
                self.recorded_frames.append(frame)
        
        return obs, info
    
    def step(self, action):
        """Step the environment and capture frame."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Capture frame after step
        if self.recording:
            frame = self.env.render()
            if frame is not None:
                self.recorded_frames.append(frame)
                
                # Check video length limit
                if self.video_length > 0 and len(self.recorded_frames) >= self.video_length:
                    terminated = True
        
        self.step_id += 1
        
        # Save video when episode ends
        if (terminated or truncated) and self.recording:
            self._save_video()
            self.recording = False
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Pass through render call."""
        return self.env.render()
    
    def _save_video(self):
        """Save the recorded frames to a video file."""
        if not self.recorded_frames:
            return
        
        # Generate filename
        video_name = f"{self.name_prefix}_episode_{self.episode_id:06d}.mp4"
        video_path = os.path.join(self.video_folder, video_name)
        
        # Get frame dimensions from first frame
        height, width = self.recorded_frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, self.fps, (width, height))
        
        # Write frames
        for frame in self.recorded_frames:
            # OpenCV expects BGR, but gym environments typically provide RGB
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
        
        # Release video writer
        out.release()
        
        if not self.disable_logger:
            print(f"Saved video to {video_path} ({len(self.recorded_frames)} frames)")
        
        # Clear frames
        self.recorded_frames = []
    
    def _close_video_recorder(self):
        """Close video recorder if recording."""
        if self.recording and self.recorded_frames:
            self._save_video()
        self.recording = False
    
    def close(self):
        """Close the environment and save any remaining video."""
        self._close_video_recorder()
        super().close()


class FrameSkipVideoWrapper(gym.Wrapper):
    """Modified F-Zero wrapper that captures all frames during frame skipping.
    
    This wrapper integrates video recording that captures every frame,
    not just the observation frames after frame skipping.
    """
    
    def __init__(
        self,
        env: gym.Env,
        frame_skip: int = 4,
        capture_all_frames: bool = False,
        video_folder: Optional[str] = None,
        **kwargs
    ):
        """Initialize the frame skip wrapper with optional full-frame video recording.
        
        Args:
            env: The environment to wrap
            frame_skip: Number of frames to skip between observations
            capture_all_frames: Whether to capture all frames for video
            video_folder: Directory to save videos (required if capture_all_frames=True)
            **kwargs: Additional arguments for the parent wrapper
        """
        super().__init__(env)
        
        self.frame_skip = frame_skip
        self.capture_all_frames = capture_all_frames
        self.video_frames = []
        self.video_folder = video_folder
        self.is_recording = False
        
        if self.capture_all_frames and not self.video_folder:
            raise ValueError("video_folder must be specified when capture_all_frames=True")
        
        # Set up video recording if needed
        if self.capture_all_frames:
            os.makedirs(self.video_folder, exist_ok=True)
            self.episode_count = 0
            self.fps = 60  # SNES native framerate
    
    def reset(self, **kwargs):
        """Reset the environment and prepare for video recording."""
        obs, info = self.env.reset(**kwargs)
        
        if self.capture_all_frames:
            self.video_frames = []
            self.is_recording = True
            # Capture initial frame
            frame = self.env.render()
            if frame is not None:
                self.video_frames.append(frame)
        
        return obs, info
    
    def step(self, action):
        """Execute action with frame skipping and capture all frames."""
        total_reward = 0.0
        
        # Execute action for frame_skip frames
        for i in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            
            # Capture frame if recording
            if self.capture_all_frames and self.is_recording:
                frame = self.env.render()
                if frame is not None:
                    self.video_frames.append(frame)
            
            if terminated or truncated:
                break
        
        # Save video if episode ended
        if (terminated or truncated) and self.capture_all_frames and self.is_recording:
            self._save_video()
            self.is_recording = False
        
        return obs, total_reward, terminated, truncated, info
    
    def _save_video(self):
        """Save captured frames to video file."""
        if not self.video_frames:
            return
        
        video_name = f"fzero_episode_{self.episode_count:06d}_all_frames.mp4"
        video_path = os.path.join(self.video_folder, video_name)
        
        height, width = self.video_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, self.fps, (width, height))
        
        for frame in self.video_frames:
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
        
        out.release()
        print(f"Saved video with all frames to {video_path} ({len(self.video_frames)} frames)")
        
        self.video_frames = []
        self.episode_count += 1
    
    def close(self):
        """Close environment and save any remaining video."""
        if self.capture_all_frames and self.is_recording and self.video_frames:
            self._save_video()
        super().close()