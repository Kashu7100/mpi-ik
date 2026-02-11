"""Default paths for model files and cache directory."""

from __future__ import annotations

import os
from pathlib import Path

CACHE_DIR = Path(os.environ.get("MPI_IK_CACHE_DIR", Path("~/.cache/mpi_ik").expanduser()))

MANO_MODEL_PATH = CACHE_DIR / "mano.pkl"
SMPL_MODEL_PATH = CACHE_DIR / "smpl.pkl"
SMPLH_MODEL_PATH = CACHE_DIR / "smplh.pkl"
