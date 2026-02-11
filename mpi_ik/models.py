"""Kinematic model and PCA wrapper for MANO/SMPL/SMPL-H meshes."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class ArmatureProtocol(Protocol):
    """Structural type expected for armature classes."""

    n_joints: int
    n_keypoints: int
    keypoints_ext: list[int]


class KinematicModel:
    """Kinematic model that takes model parameters and outputs mesh and keypoints."""

    def __init__(
        self,
        model_path: str | Path,
        armature: type[ArmatureProtocol],
        scale: float = 1,
    ) -> None:
        """
        Parameters
        ----------
        model_path : str | Path
            Path to the model pickle file.
        armature : type[ArmatureProtocol]
            An armature class from ``armatures.py``.
        scale : float, optional
            Scale factor for output coordinates, by default 1.
        """
        with open(model_path, 'rb') as f:
            params = pickle.load(f)

            self.pose_pca_basis: NDArray = params['pose_pca_basis']
            self.pose_pca_mean: NDArray = params['pose_pca_mean']

            self.J_regressor: NDArray = params['J_regressor']

            self.skinning_weights: NDArray = params['skinning_weights']

            self.mesh_pose_basis: NDArray = params['mesh_pose_basis']
            self.mesh_shape_basis: NDArray = params['mesh_shape_basis']
            self.mesh_template: NDArray = params['mesh_template']

            self.faces: NDArray = params['faces']

            self.parents: list = params['parents']

        self.n_shape_params: int = self.mesh_shape_basis.shape[-1]
        self.scale = scale

        self.armature = armature
        self.n_joints = self.armature.n_joints
        self.pose: NDArray = np.zeros((self.n_joints, 3))
        self.shape: NDArray = np.zeros(self.mesh_shape_basis.shape[-1])
        self.verts: NDArray | None = None
        self.J: NDArray | None = None
        self.R: NDArray | None = None
        self.keypoints: NDArray | None = None

        self.J_regressor_ext: NDArray = np.zeros(
            [self.armature.n_keypoints, self.J_regressor.shape[1]]
        )
        self.J_regressor_ext[:self.armature.n_joints] = self.J_regressor
        for i, v in enumerate(self.armature.keypoints_ext):
            self.J_regressor_ext[i + self.armature.n_joints, v] = 1

        self.update()

    def set_params(
        self,
        pose_abs: NDArray | None = None,
        pose_pca: NDArray | None = None,
        pose_glb: NDArray | None = None,
        shape: NDArray | None = None,
    ) -> tuple[NDArray, NDArray]:
        """
        Set model parameters and compute the mesh.

        Do not set *pose_abs* and *pose_pca* at the same time.

        Parameters
        ----------
        pose_abs : NDArray, shape [n_joints, 3], optional
            Absolute pose in axis-angle representation.
        pose_pca : NDArray, optional
            PCA coefficients of the pose.
        pose_glb : NDArray, shape [1, 3], optional
            Global rotation for the model.
        shape : NDArray, shape [n_shape], optional
            Shape coefficients.

        Returns
        -------
        tuple[NDArray, NDArray]
            ``(vertices, keypoints)`` with scale applied.
        """
        if pose_abs is not None:
            self.pose = pose_abs
        elif pose_pca is not None:
            self.pose = np.dot(
                np.expand_dims(pose_pca, 0),
                self.pose_pca_basis[:pose_pca.shape[0]],
            )[0] + self.pose_pca_mean
            self.pose = np.reshape(self.pose, [self.n_joints - 1, 3])
            if pose_glb is None:
                pose_glb = np.zeros([1, 3])
            pose_glb = np.reshape(pose_glb, [1, 3])
            self.pose = np.concatenate([pose_glb, self.pose], 0)
        if shape is not None:
            self.shape = shape
        return self.update()

    def update(self) -> tuple[NDArray, NDArray]:
        """
        Recompute vertices and keypoints from current parameters.

        Returns
        -------
        tuple[NDArray, NDArray]
            ``(vertices, keypoints)`` with scale applied.
        """
        verts = self.mesh_template + self.mesh_shape_basis.dot(self.shape)
        self.J = self.J_regressor.dot(verts)
        self.R = self.rodrigues(self.pose.reshape((-1, 1, 3)))
        G = np.empty((self.n_joints, 4, 4))
        G[0] = self.with_zeros(
            np.hstack((self.R[0], self.J[0, :].reshape([3, 1])))
        )
        for i in range(1, self.n_joints):
            G[i] = G[self.parents[i]].dot(self.with_zeros(
                np.hstack([
                    self.R[i],
                    (self.J[i, :] - self.J[self.parents[i], :]).reshape([3, 1]),
                ])
            ))
        G = G - self.pack(np.matmul(
            G,
            np.hstack([self.J, np.zeros([self.n_joints, 1])])
                .reshape([self.n_joints, 4, 1]),
        ))
        T = np.tensordot(self.skinning_weights, G, axes=[[1], [0]])
        verts = np.hstack((verts, np.ones([verts.shape[0], 1])))

        self.verts = np.matmul(
            T, verts.reshape([-1, 4, 1])
        ).reshape([-1, 4])[:, :3]
        self.keypoints = self.J_regressor_ext.dot(self.verts)

        self.verts *= self.scale
        self.keypoints *= self.scale

        return self.verts.copy(), self.keypoints.copy()

    @staticmethod
    def rodrigues(r: NDArray) -> NDArray:
        """
        Rodrigues' rotation formula (batched axis-angle to rotation matrix).

        Parameters
        ----------
        r : NDArray, shape [batch_size, 1, 3]
            Axis-angle rotation vectors.

        Returns
        -------
        NDArray, shape [batch_size, 3, 3]
            Rotation matrices.
        """
        theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
        theta = np.maximum(theta, np.finfo(np.float64).eps)
        r_hat = r / theta
        cos = np.cos(theta)
        z_stick = np.zeros(theta.shape[0])
        m = np.dstack([
            z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
            r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
            -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick,
        ]).reshape([-1, 3, 3])
        i_cube = np.broadcast_to(
            np.expand_dims(np.eye(3), axis=0), [theta.shape[0], 3, 3]
        )
        A = np.transpose(r_hat, axes=[0, 2, 1])
        B = r_hat
        dot = np.matmul(A, B)
        R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
        return R

    @staticmethod
    def with_zeros(x: NDArray) -> NDArray:
        """
        Append ``[0, 0, 0, 1]`` to a ``[3, 4]`` matrix.

        Parameters
        ----------
        x : NDArray, shape [3, 4]
            Input matrix.

        Returns
        -------
        NDArray, shape [4, 4]
            Homogeneous transform matrix.
        """
        return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))

    @staticmethod
    def pack(x: NDArray) -> NDArray:
        """
        Append zero columns to vectors in a batch.

        Parameters
        ----------
        x : NDArray, shape [batch_size, 4, 1]
            Input vectors.

        Returns
        -------
        NDArray, shape [batch_size, 4, 4]
            Matrices with zero-padded columns.
        """
        return np.dstack((np.zeros((x.shape[0], 4, 3)), x))

    def save_obj(self, path: str | Path) -> None:
        """
        Save the current mesh as a Wavefront OBJ file.

        Parameters
        ----------
        path : str | Path
            Output file path.
        """
        with open(path, 'w') as fp:
            for v in self.verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in self.faces + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


class KinematicPCAWrapper:
    """Wrapper around :class:`KinematicModel` for use with the LM solver."""

    def __init__(self, core: KinematicModel, n_pose: int = 12) -> None:
        """
        Parameters
        ----------
        core : KinematicModel
            The kinematic model instance to wrap.
        n_pose : int, optional
            Number of PCA pose coefficients, by default 12.
        """
        self.core = core
        self.n_pose = n_pose
        self.n_shape = core.n_shape_params
        self.n_glb = 3
        self.n_params = self.n_pose + self.n_shape + self.n_glb

    def run(self, params: NDArray) -> NDArray:
        """
        Evaluate the model at the given parameters.

        Parameters
        ----------
        params : NDArray
            Compact parameter vector.

        Returns
        -------
        NDArray
            Model keypoints.
        """
        shape, pose_pca, pose_glb = self.decode(params)
        return self.core.set_params(
            pose_glb=pose_glb, pose_pca=pose_pca, shape=shape
        )[1]

    def decode(self, params: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """
        Decode a compact parameter vector into semantic components.

        Parameters
        ----------
        params : NDArray
            Compact parameter vector.

        Returns
        -------
        tuple[NDArray, NDArray, NDArray]
            ``(shape, pose_pca, pose_glb)``
        """
        pose_glb = params[:self.n_glb]
        pose_pca = params[self.n_glb:-self.n_shape]
        shape = params[-self.n_shape:]
        return shape, pose_pca, pose_glb
