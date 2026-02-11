"""Shared test fixtures for mpi_ik tests."""

from pathlib import Path

import pytest

from mpi_ik.config import MANO_LEFT_MODEL_PATH as _CACHE_MANO_PATH

# Prefer the cache dir path; fall back to the local model_files/ directory.
_LOCAL_MANO_PATH = Path(__file__).resolve().parent.parent / 'model_files' / 'mano_left.pkl'
MANO_LEFT_MODEL_PATH = _CACHE_MANO_PATH if _CACHE_MANO_PATH.exists() else _LOCAL_MANO_PATH


@pytest.fixture
def mano_model_path() -> Path:
    return MANO_LEFT_MODEL_PATH


def has_mano_model() -> bool:
    return MANO_LEFT_MODEL_PATH.exists()


skip_without_model = pytest.mark.skipif(
    not has_mano_model(),
    reason='MANO model file not found',
)
