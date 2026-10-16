"""Import shim for running LIBERO here: CPU rendering, and torch>=2.6 safe loading of
LIBERO's numpy init-state files (allow-list numpy array reconstruction only)."""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import numpy as np
import torch
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct, np.ndarray, np.dtype,
                                      type(np.dtype(np.float64))])
