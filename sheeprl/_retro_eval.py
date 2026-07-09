import os
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('PYGLET_HEADLESS', '1')
import pyglet; pyglet.options['shadow_window'] = False
import torch
original_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = _patched_torch_load
from sheeprl.cli import evaluation
evaluation()
