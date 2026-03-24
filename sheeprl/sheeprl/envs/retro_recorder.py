"""Retro environment wrapper for .bk2 file recording."""

import os
import retro
from typing import Any, Dict, Optional
import gymnasium as gym


class RetroRecorder(gym.Wrapper):
    """Wrapper for recording .bk2 files using retro's built-in recording functionality."""
    
    def __init__(self, env: gym.Env, bk2_path: str, record: bool = True):
        """Initialize the RetroRecorder wrapper.
        
        Args:
            env: The environment to wrap
            bk2_path: Path where the .bk2 file should be saved
            record: Whether to enable recording
        """
        super().__init__(env)
        self.bk2_path = bk2_path
        self.record = record
        self.movie = None
        self.recording = False
        
    def reset(self, **kwargs) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment and start recording if enabled."""
        obs, info = self.env.reset(**kwargs)
        
        if self.record:
            # Start recording when environment resets
            self._start_recording()
        
        return obs, info
    
    def step(self, action) -> tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment and record if enabled."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if self.recording and self.movie is not None:
            # Record the action
            self.movie.step()
            
        return obs, reward, terminated, truncated, info
    
    def _start_recording(self):
        """Start recording to .bk2 file."""
        try:
            # Get the underlying retro environment
            if hasattr(self.env, '_env'):
                retro_env = self.env._env
            else:
                retro_env = self.env
            
            # Create movie object for recording
            self.movie = retro.Movie(
                self.bk2_path,
                record=True,
                players=1
            )
            
            # Configure the retro environment to use the movie
            if hasattr(retro_env, 'movie'):
                retro_env.movie = self.movie
            elif hasattr(retro_env, '_movie'):
                retro_env._movie = self.movie
            
            self.recording = True
            print(f"Started recording .bk2 file to: {self.bk2_path}")
            
        except Exception as e:
            print(f"Warning: Could not start .bk2 recording: {e}")
            self.recording = False
    
    def close(self):
        """Close the environment and stop recording."""
        if self.recording and self.movie is not None:
            try:
                self.movie.close()
                print(f"Finished recording .bk2 file: {self.bk2_path}")
            except Exception as e:
                print(f"Warning: Error closing .bk2 file: {e}")
            finally:
                self.recording = False
                self.movie = None
        
        super().close()