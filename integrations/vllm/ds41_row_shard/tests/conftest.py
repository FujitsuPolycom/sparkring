"""Session setup: the plugin package on the path and float64 by default.

float64 makes any arithmetic difference between the owned-row and full-row
paths visible; summation order is fixed by ``tp_world``.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
torch.set_default_dtype(torch.float64)
