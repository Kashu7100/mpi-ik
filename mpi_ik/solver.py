"""Levenberg-Marquardt inverse kinematics solver."""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class IKModel(Protocol):
    """Protocol for models compatible with the solver."""

    n_params: int

    def run(self, params: NDArray) -> NDArray: ...


class Solver:
    """Levenberg-Marquardt solver for inverse kinematics."""

    def __init__(
        self,
        eps: float = 1e-5,
        max_iter: int = 30,
        mse_threshold: float = 1e-8,
        verbose: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        eps : float, optional
            Epsilon for finite-difference derivative, by default 1e-5.
        max_iter : int, optional
            Maximum number of iterations, by default 30.
        mse_threshold : float, optional
            Early stop when MSE change is below this threshold, by default 1e-8.
        verbose : bool, optional
            Print per-iteration diagnostics, by default False.
        """
        self.eps = eps
        self.max_iter = max_iter
        self.mse_threshold = mse_threshold
        self.verbose = verbose

    def get_derivative(
        self, model: IKModel, params: NDArray, n: int
    ) -> NDArray:
        """
        Compute the partial derivative w.r.t. the *n*-th parameter via
        central finite differences.

        Parameters
        ----------
        model : IKModel
            Model wrapper.
        params : NDArray
            Current parameter vector.
        n : int
            Index of the parameter to differentiate.

        Returns
        -------
        NDArray
            Flattened derivative vector.
        """
        params1 = np.array(params)
        params2 = np.array(params)

        params1[n] += self.eps
        params2[n] -= self.eps

        res1 = model.run(params1)
        res2 = model.run(params2)

        d = (res1 - res2) / (2 * self.eps)

        return d.ravel()

    def solve(
        self,
        model: IKModel,
        target: NDArray,
        init: NDArray | None = None,
        u: float = 1e-3,
        v: float = 1.5,
    ) -> NDArray:
        """
        Solve for model parameters that best fit *target*.

        Parameters
        ----------
        model : IKModel
            Model wrapper to optimise.
        target : NDArray
            Target output to match.
        init : NDArray, optional
            Initial parameter vector, by default zeros.
        u : float, optional
            LM damping factor, by default 1e-3.
        v : float, optional
            LM damping adjustment factor, by default 1.5.

        Returns
        -------
        NDArray
            Optimised parameter vector.
        """
        if init is None:
            init = np.zeros(model.n_params)
        out_n = np.shape(model.run(init).ravel())[0]
        jacobian = np.zeros([out_n, init.shape[0]])

        last_update = 0.0
        last_mse = 0.0
        params = init
        for i in range(self.max_iter):
            t0 = time.monotonic()

            residual = (model.run(params) - target).reshape(out_n, 1)
            mse = np.mean(np.square(residual))

            if abs(mse - last_mse) < self.mse_threshold:
                return params

            for k in range(params.shape[0]):
                jacobian[:, k] = self.get_derivative(model, params, k)

            jtj = np.matmul(jacobian.T, jacobian)
            jtj = jtj + u * np.eye(jtj.shape[0])

            update = last_mse - mse
            delta = np.matmul(
                np.matmul(np.linalg.inv(jtj), jacobian.T), residual
            ).ravel()
            params -= delta

            if update > last_update and update > 0:
                u /= v
            else:
                u *= v

            last_update = update
            last_mse = mse

            if self.verbose:
                elapsed = time.monotonic() - t0
                print(f'iter {i:3d}  mse {mse:.6e}  time {elapsed:.3f}s')

        return params
