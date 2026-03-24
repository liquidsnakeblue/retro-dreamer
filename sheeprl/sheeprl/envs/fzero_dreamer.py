"""F-Zero wrapper with Discrete action space for Dreamer v3 compatibility."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sheeprl.envs.fzero_fixed import FZeroSheepRLWrapper


class FZeroDreamerWrapper(FZeroSheepRLWrapper):
    """F-Zero wrapper that converts MultiBinary actions to Discrete for Dreamer v3.
    
    This wrapper converts the 12-button MultiBinary action space to a Discrete
    action space. We use a simplified action mapping focusing on the most
    important controls for F-Zero.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Define action mappings for F-Zero
        # Simplified action space with only 5 essential actions
        self.action_mappings = [
            # Index: [B, Y, Select, Start, Up, Down, Left, Right, A, X, L, R]
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0: Forward (B)
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 1: Forward + Left
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 2: Forward + Right
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # 3: Forward + Left + L (sharp left)
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # 4: Forward + Right + R (sharp right)
        ]
        
        # Override action space to be Discrete
        self.action_space = spaces.Discrete(len(self.action_mappings))
        self._original_action_space = self._env.action_space
        
    def step(self, action):
        """Convert discrete action to MultiBinary and execute step."""
        # Convert discrete action to MultiBinary
        if isinstance(action, np.ndarray):
            # Handle different numpy array shapes
            if action.ndim == 0:
                # Scalar array
                action = int(action.item())
            elif action.shape == ():
                # Empty shape array
                action = int(action.item())
            else:
                # Array with elements
                action = int(action.flat[0])
        elif isinstance(action, list):
            action = int(action[0]) if len(action) > 0 else 0
        else:
            action = int(action)
            
        # Ensure action is within valid range
        action = max(0, min(action, len(self.action_mappings) - 1))
        
        # Get the MultiBinary action
        multibinary_action = np.array(self.action_mappings[action], dtype=np.int8)
        
        # Call parent's step method with MultiBinary action
        return super().step(multibinary_action)