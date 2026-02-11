"""mpi_ik - Inverse kinematics solver for MANO, SMPL, and SMPL-H models."""

from mpi_ik.armatures import MANOArmature, SMPLArmature, SMPLHArmature
from mpi_ik.config import CACHE_DIR, MANO_MODEL_PATH, SMPL_MODEL_PATH, SMPLH_MODEL_PATH
from mpi_ik.models import KinematicModel, KinematicPCAWrapper
from mpi_ik.solver import Solver

__all__ = [
    "Solver",
    "KinematicModel",
    "KinematicPCAWrapper",
    "MANOArmature",
    "SMPLArmature",
    "SMPLHArmature",
    "CACHE_DIR",
    "MANO_MODEL_PATH",
    "SMPL_MODEL_PATH",
    "SMPLH_MODEL_PATH",
]
