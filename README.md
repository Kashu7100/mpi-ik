# MPI IK

A simple inverse kinematics solver for MANO hand model, SMPL body model, and SMPL-H body+hand model.

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```python
from mpi_ik import MANO_MODEL_PATH, KinematicModel, KinematicPCAWrapper, MANOArmature, Solver

mesh = KinematicModel(MANO_MODEL_PATH, MANOArmature, scale=1000)
wrapper = KinematicPCAWrapper(mesh, n_pose=12)
solver = Solver(verbose=True)

_, keypoints = mesh.set_params(pose_pca=pose_pca, pose_glb=pose_glb, shape=shape)
# keypoints (21, 3)
params_est = solver.solve(wrapper, keypoints)
```

See `examples/mano_example.py` for a full working example.

## Model Download

Download the official model files from the following sites (registration required):

- **MANO**: https://mano.is.tue.mpg.de/
- **SMPL**: https://smpl.is.tue.mpg.de/
- **SMPL-H**: https://mano.is.tue.mpg.de/ (included with MANO download)

## Model Preparation

Convert official MANO/SMPL/SMPL-H model files into the format expected by this package.
By default, converted models are saved to `~/.cache/mpi_ik/`:

```bash
# Output defaults to ~/.cache/mpi_ik/mano.pkl
mpi-ik-prepare --model-type mano --input path/to/mano_v1_2/models/MANO_LEFT.pkl

# Or specify a custom output path
mpi-ik-prepare --model-type mano --input path/to/mano_v1_2/models/MANO_LEFT.pkl --output custom/path/mano.pkl
```

The cache directory can be overridden with the `MPI_IK_CACHE_DIR` environment variable.

## Acknowledgement

This project is based on [Minimal-IK](https://github.com/CalciferZh/Minimal-IK) by Yuxiao Zhou.
