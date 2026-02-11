"""Tests for the LM solver."""

import numpy as np
import pytest

from mpi_ik.solver import Solver
from tests.conftest import skip_without_model


class LinearModel:
    """A @ x = b  ->  model.run(x) = A @ x."""

    def __init__(self, A: np.ndarray):
        self.A = A
        self.n_params = A.shape[1]

    def run(self, params: np.ndarray) -> np.ndarray:
        return self.A @ params


class QuadraticModel:
    """f(x) = x^2."""

    def __init__(self, n: int):
        self.n_params = n

    def run(self, params: np.ndarray) -> np.ndarray:
        return params ** 2


class TestSolverLinear:
    def test_recovers_solution(self):
        rng = np.random.RandomState(42)
        A = rng.randn(5, 3)
        x_true = rng.randn(3)
        b = A @ x_true

        model = LinearModel(A)
        solver = Solver(max_iter=50, verbose=False)
        x_est = solver.solve(model, b)
        np.testing.assert_allclose(x_est, x_true, atol=1e-3)


class TestSolverNonlinear:
    def test_converges_to_sqrt(self):
        target = np.array([4.0, 9.0, 16.0])
        model = QuadraticModel(3)
        solver = Solver(max_iter=100, verbose=False)
        x_est = solver.solve(model, target, init=np.ones(3))
        # could converge to positive or negative root
        np.testing.assert_allclose(x_est ** 2, target, atol=1e-2)


class TestSolverEarlyStopping:
    def test_already_at_solution(self):
        A = np.eye(3)
        model = LinearModel(A)
        solver = Solver(max_iter=100, verbose=False)
        x_true = np.array([1.0, 2.0, 3.0])
        x_est = solver.solve(model, x_true, init=x_true)
        np.testing.assert_allclose(x_est, x_true, atol=1e-6)


class TestSolverVerbose:
    def test_verbose_does_not_crash(self, capsys):
        A = np.eye(2)
        model = LinearModel(A)
        solver = Solver(max_iter=5, verbose=True)
        solver.solve(model, np.array([1.0, 2.0]))
        captured = capsys.readouterr()
        assert 'iter' in captured.out
        assert 'mse' in captured.out


@skip_without_model
class TestIntegrationMANO:
    def test_roundtrip(self, mano_model_path):
        from mpi_ik import KinematicModel, KinematicPCAWrapper, MANOArmature

        np.random.seed(20160923)
        n_pose = 12
        n_shape = 10
        pose_pca = np.random.normal(size=n_pose)
        shape = np.random.normal(size=n_shape)
        pose_glb = np.zeros([1, 3])

        mesh = KinematicModel(str(mano_model_path), MANOArmature, scale=1000)
        wrapper = KinematicPCAWrapper(mesh, n_pose=n_pose)
        _, keypoints = mesh.set_params(
            pose_pca=pose_pca, pose_glb=pose_glb, shape=shape
        )

        solver = Solver(verbose=False)
        params_est = solver.solve(wrapper, keypoints)
        shape_est, pose_pca_est, pose_glb_est = wrapper.decode(params_est)

        np.testing.assert_allclose(pose_pca_est, pose_pca, atol=1e-1)
        np.testing.assert_allclose(shape_est, shape, atol=1e-1)
