"""Sparse Cholesky factorization with CHOLMOD, provided by the cholespy package."""

import numpy as np
from scipy import sparse

from .utils._imports import import_optional_dependency

_INT32_MAX = np.iinfo(np.int32).max


def require_cholmod() -> None:
    """Raise an ImportError if cholespy, which provides CHOLMOD, is not installed."""
    import_optional_dependency("cholespy", extra="It is required for use_cholmod=True.", raise_error=True)


class CholeskyFactor:
    """Cholesky factorization of a sparse symmetric positive definite matrix.

    CHOLMOD reads only one triangle of the matrix, so the matrix has to be
    genuinely symmetric. A ``float32`` matrix is factorized in single
    precision, anything else in double precision.

    Parameters
    ----------
    matrix : scipy.sparse.spmatrix
        Sparse symmetric positive definite matrix of shape (n, n).

    Raises
    ------
    ImportError
        If cholespy is not installed.
    ValueError
        If the matrix is not square, too large for 32-bit indices, or not
        positive definite.
    """

    def __init__(self, matrix) -> None:
        cholespy = import_optional_dependency(
            "cholespy", extra="It is required for use_cholmod=True.", raise_error=True
        )
        a = sparse.csc_matrix(matrix)
        a.sum_duplicates()
        a.sort_indices()
        n = a.shape[0]
        if a.shape != (n, n):
            raise ValueError(f"Cholesky factorization needs a square matrix, got shape {a.shape}.")
        # cholespy takes 32-bit indices; scipy switches to 64-bit ones beyond that range
        if n > _INT32_MAX or a.nnz > _INT32_MAX:
            raise ValueError(
                f"Matrix with {n} rows and {a.nnz} non-zeros is too large for CHOLMOD's 32-bit indices."
            )
        self.dtype = np.dtype(np.float32) if a.dtype == np.float32 else np.dtype(np.float64)
        solver = cholespy.CholeskySolverF if self.dtype == np.float32 else cholespy.CholeskySolverD
        try:
            self._solver = solver(
                n,
                a.indptr.astype(np.int32),
                a.indices.astype(np.int32),
                a.data.astype(self.dtype),
                cholespy.MatrixType.CSC,
            )
        except ValueError as err:
            if "positive definite" not in str(err):
                raise
            raise ValueError(
                "Cholesky factorization failed because the matrix is not positive definite, "
                "for example a Laplace operator without boundary conditions. "
                "Use use_cholmod=False for the LU decomposition instead."
            ) from err
        self.shape = (n, n)

    def solve(self, b) -> np.ndarray:
        """Solve ``A x = b`` with the factorized matrix.

        Parameters
        ----------
        b : array_like or scipy.sparse.spmatrix
            Right hand side of shape (n,) or (n, n_rhs).

        Returns
        -------
        np.ndarray
            Solution of the same shape as ``b``, in the precision of the factor.

        Raises
        ------
        ValueError
            If the first dimension of ``b`` does not match the matrix.
        """
        if sparse.issparse(b):
            b = b.toarray()
        b = np.ascontiguousarray(np.asarray(b), dtype=self.dtype)
        if b.ndim not in (1, 2) or b.shape[0] != self.shape[0]:
            raise ValueError(f"Right hand side of shape {b.shape} does not match a matrix of shape {self.shape}.")
        x = np.empty_like(b)
        self._solver.solve(b, x)
        return x
