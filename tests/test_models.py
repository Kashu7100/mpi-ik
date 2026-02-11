"""Tests for static math utilities in KinematicModel."""

import numpy as np
import pytest

from mpi_ik.models import KinematicModel


class TestRodrigues:
    def test_zero_vector_gives_identity(self):
        r = np.zeros((1, 1, 3))
        R = KinematicModel.rodrigues(r)
        np.testing.assert_allclose(R[0], np.eye(3), atol=1e-10)

    def test_90_degree_rotation_z(self):
        angle = np.pi / 2
        r = np.array([[[0, 0, angle]]])
        R = KinematicModel.rodrigues(r)
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        np.testing.assert_allclose(R[0], expected, atol=1e-10)

    def test_batch(self):
        r = np.random.randn(5, 1, 3)
        R = KinematicModel.rodrigues(r)
        assert R.shape == (5, 3, 3)

    def test_orthogonality(self):
        r = np.random.randn(10, 1, 3)
        R = KinematicModel.rodrigues(r)
        for i in range(10):
            np.testing.assert_allclose(
                R[i] @ R[i].T, np.eye(3), atol=1e-10,
            )

    def test_determinant_positive(self):
        r = np.random.randn(10, 1, 3)
        R = KinematicModel.rodrigues(r)
        for i in range(10):
            assert np.linalg.det(R[i]) == pytest.approx(1.0, abs=1e-10)


class TestWithZeros:
    def test_shape_and_last_row(self):
        x = np.random.randn(3, 4)
        result = KinematicModel.with_zeros(x)
        assert result.shape == (4, 4)
        np.testing.assert_array_equal(result[3], [0, 0, 0, 1])

    def test_preserves_top_rows(self):
        x = np.random.randn(3, 4)
        result = KinematicModel.with_zeros(x)
        np.testing.assert_array_equal(result[:3], x)


class TestPack:
    def test_shape(self):
        x = np.random.randn(7, 4, 1)
        result = KinematicModel.pack(x)
        assert result.shape == (7, 4, 4)

    def test_zero_columns(self):
        x = np.random.randn(3, 4, 1)
        result = KinematicModel.pack(x)
        np.testing.assert_array_equal(result[:, :, :3], 0)

    def test_last_column(self):
        x = np.random.randn(3, 4, 1)
        result = KinematicModel.pack(x)
        np.testing.assert_array_equal(result[:, :, 3:], x)
