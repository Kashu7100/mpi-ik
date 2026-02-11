"""MANO inverse kinematics example.

Generates random pose/shape parameters, computes keypoints via forward
kinematics, then recovers the parameters with the LM solver.
"""

import numpy as np

from mpi_ik import MANO_LEFT_MODEL_PATH, KinematicModel, KinematicPCAWrapper, MANOArmature, Solver


def main() -> None:
    np.random.seed(20160923)
    pose_glb = np.zeros([1, 3])

    n_pose = 12
    n_shape = 10
    pose_pca = np.random.normal(size=n_pose)
    shape = np.random.normal(size=n_shape)
    mesh = KinematicModel(MANO_LEFT_MODEL_PATH, MANOArmature, scale=1000)

    wrapper = KinematicPCAWrapper(mesh, n_pose=n_pose)
    solver = Solver(verbose=True)

    _, keypoints = mesh.set_params(
        pose_pca=pose_pca, pose_glb=pose_glb, shape=shape
    )
    print(keypoints.shape)
    params_est = solver.solve(wrapper, keypoints)

    shape_est, pose_pca_est, pose_glb_est = wrapper.decode(params_est)

    print('----------------------------------------------------------------------')
    print('ground truth parameters')
    print('pose pca coefficients:', pose_pca)
    print('pose global rotation:', pose_glb)
    print('shape: pca coefficients:', shape)

    print('----------------------------------------------------------------------')
    print('estimated parameters')
    print('pose pca coefficients:', pose_pca_est)
    print('pose global rotation:', pose_glb_est)
    print('shape: pca coefficients:', shape_est)

    mesh.set_params(pose_pca=pose_pca)
    mesh.save_obj('./gt.obj')
    mesh.set_params(pose_pca=pose_pca_est)
    mesh.save_obj('./est.obj')

    print('ground truth and estimated meshes are saved into gt.obj and est.obj')


if __name__ == '__main__':
    main()
