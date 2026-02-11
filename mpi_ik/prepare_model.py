"""Convert official MANO / SMPL / SMPL-H model files into mpi-ik format.

Can be used as a library or via the ``mpi-ik-prepare`` CLI entry point.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from mpi_ik.config import CACHE_DIR, MANO_MODEL_PATH, SMPL_MODEL_PATH, SMPLH_MODEL_PATH

_DEFAULT_OUTPUT = {
    'mano': MANO_MODEL_PATH,
    'smpl': SMPL_MODEL_PATH,
    'smplh': SMPLH_MODEL_PATH,
}


def prepare_mano_model(input_path: str | Path, output_path: str | Path) -> None:
    """
    Convert the official MANO model into the format expected by mpi-ik.

    Parameters
    ----------
    input_path : str | Path
        Path to the official MANO ``.pkl`` file.
    output_path : str | Path
        Destination path for the converted model.
    """
    with open(input_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    params = {
        'pose_pca_basis': np.array(data['hands_components']),
        'pose_pca_mean': np.array(data['hands_mean']),
        'J_regressor': data['J_regressor'].toarray(),
        'skinning_weights': np.array(data['weights']),
        'mesh_pose_basis': np.array(data['posedirs']),
        'mesh_shape_basis': np.array(data['shapedirs']),
        'mesh_template': np.array(data['v_template']),
        'faces': np.array(data['f']),
        'parents': data['kintree_table'][0].tolist(),
    }
    params['parents'][0] = None
    with open(output_path, 'wb') as f:
        pickle.dump(params, f)


def prepare_smpl_model(input_path: str | Path, output_path: str | Path) -> None:
    """
    Convert the official SMPL model into the format expected by mpi-ik.

    Parameters
    ----------
    input_path : str | Path
        Path to the official SMPL ``.pkl`` file.
    output_path : str | Path
        Destination path for the converted model.
    """
    with open(input_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    params = {
        'pose_pca_basis': np.eye(23 * 3),
        'pose_pca_mean': np.zeros(23 * 3),
        'J_regressor': data['J_regressor'].toarray(),
        'skinning_weights': np.array(data['weights']),
        'mesh_pose_basis': np.array(data['posedirs']),
        'mesh_shape_basis': np.array(data['shapedirs']),
        'mesh_template': np.array(data['v_template']),
        'faces': np.array(data['f']),
        'parents': data['kintree_table'][0].tolist(),
    }
    params['parents'][0] = None
    with open(output_path, 'wb') as f:
        pickle.dump(params, f)


def prepare_smplh_model(input_path: str | Path, output_path: str | Path) -> None:
    """
    Convert the official SMPL-H model into the format expected by mpi-ik.

    Parameters
    ----------
    input_path : str | Path
        Path to the official SMPL-H ``.npz`` file.
    output_path : str | Path
        Destination path for the converted model.
    """
    data = np.load(input_path)
    params = {
        'pose_pca_basis': np.eye(51 * 3),
        'pose_pca_mean': np.zeros(51 * 3),
        'J_regressor': data['J_regressor'],
        'skinning_weights': np.array(data['weights']),
        'mesh_pose_basis': np.array(data['posedirs']),
        'mesh_shape_basis': np.array(data['shapedirs']),
        'mesh_template': np.array(data['v_template']),
        'faces': np.array(data['f']),
        'parents': data['kintree_table'][0].tolist(),
    }
    params['parents'][0] = None
    with open(output_path, 'wb') as f:
        pickle.dump(params, f)


_PREPARE_FUNCS = {
    'mano': prepare_mano_model,
    'smpl': prepare_smpl_model,
    'smplh': prepare_smplh_model,
}


def main() -> None:
    """CLI entry point for model conversion."""
    parser = argparse.ArgumentParser(
        description='Convert official MANO/SMPL/SMPL-H models into mpi-ik format.',
    )
    parser.add_argument(
        '--model-type',
        required=True,
        choices=list(_PREPARE_FUNCS),
        help='Type of model to convert.',
    )
    parser.add_argument(
        '--input',
        required=True,
        type=Path,
        help='Path to the official model file.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Destination path for the converted model (default: ~/.cache/mpi_ik/<type>.pkl).',
    )
    args = parser.parse_args()
    output = args.output or _DEFAULT_OUTPUT[args.model_type]
    output.parent.mkdir(parents=True, exist_ok=True)
    _PREPARE_FUNCS[args.model_type](args.input, output)


if __name__ == '__main__':
    main()
