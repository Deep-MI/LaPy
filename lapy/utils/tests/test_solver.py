"""Tests for Solver.eigs and Solver.poisson — parameters and 2-D rhs support."""

import numpy as np
import pytest
from scipy import sparse

from ...solver import Solver
from ...tria_mesh import TriaMesh


@pytest.fixture
def tria_mesh():
    return TriaMesh.read_off("data/square-mesh.off")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _assert_evecs_allclose(evecs1, evecs2, **kwargs):
    """Assert eigenvectors agree up to a per-column sign flip.

    The eigenspace returned by ARPACK/eigsh is unique only up to sign (and,
    for degenerate eigenvalues, up to rotation within the subspace). For
    non-degenerate eigenvalues the per-column dot product detects the sign and
    aligns evecs2 before the numerical comparison.
    """
    signs = np.sign(np.einsum("ij,ij->j", evecs1, evecs2))
    signs[signs == 0] = 1  # zero dot product: columns are already orthogonal
    np.testing.assert_allclose(evecs1, evecs2 * signs, **kwargs)


# ---------------------------------------------------------------------------
# eigs — new parameter and sorting tests
# ---------------------------------------------------------------------------


def test_eigs_rng_int_reproducible(tria_mesh):
    """Two calls with the same integer rng seed must return identical results."""
    fem = Solver(tria_mesh, lump=True)
    evals1, evecs1 = fem.eigs(k=4, rng=42)
    evals2, evecs2 = fem.eigs(k=4, rng=42)
    np.testing.assert_allclose(evals1, evals2, rtol=1e-12)
    _assert_evecs_allclose(evecs1, evecs2, rtol=1e-10, atol=1e-12)


def test_eigs_v0_takes_precedence_over_rng(tria_mesh):
    """Explicit v0 must take precedence over rng."""
    fem = Solver(tria_mesh, lump=True)
    v0 = np.random.default_rng(0).standard_normal(len(tria_mesh.v))
    # Same v0 with different rng seeds must give identical results.
    evals1, evecs1 = fem.eigs(k=4, v0=v0, rng=0)
    evals2, evecs2 = fem.eigs(k=4, v0=v0, rng=99)
    np.testing.assert_allclose(evals1, evals2, rtol=1e-12)
    _assert_evecs_allclose(evecs1, evecs2, rtol=1e-10, atol=1e-12)


def test_poisson_scalar_and_1d_return_1d(tria_mesh):
    """Scalar and 1-D rhs must return a 1-D array (backward compatibility)."""
    fem = Solver(tria_mesh, lump=True)
    _, evec = fem.eigs(k=3)

    assert fem.poisson(0.0).ndim == 1
    assert fem.poisson(evec[:, 1]).ndim == 1


def test_poisson_2d_rhs_matches_1d(tria_mesh):
    """2-D rhs must give the same result as repeated independent 1-D solves."""
    fem = Solver(tria_mesh, lump=True)
    _, evec = fem.eigs(k=5)
    rhs = evec[:, 1:5]  # (n_vertices, 4)

    x_batch = fem.poisson(rhs)

    assert x_batch.shape == (len(tria_mesh.v), 4)
    for k in range(4):
        np.testing.assert_allclose(
            x_batch[:, k],
            fem.poisson(rhs[:, k]),
            rtol=1e-6, atol=1e-9,
            err_msg=f"poisson 2-D mismatch at column {k}",
        )


def test_poisson_2d_rhs_with_dirichlet(tria_mesh):
    """2-D rhs with Dirichlet BC must match repeated 1-D solves."""
    fem = Solver(tria_mesh, lump=True)
    _, evec = fem.eigs(k=5)
    rhs = evec[:, 1:4]  # (n_vertices, 3)
    dtup = (np.array([0, 1]), np.array([0.0, 0.0]))

    x_batch = fem.poisson(rhs, dtup=dtup)

    assert x_batch.shape == (len(tria_mesh.v), 3)
    for k in range(3):
        np.testing.assert_allclose(
            x_batch[:, k],
            fem.poisson(rhs[:, k], dtup=dtup),
            rtol=1e-6, atol=1e-9,
            err_msg=f"poisson 2-D Dirichlet mismatch at column {k}",
        )

def test_poisson_with_integrate_false(tria_mesh):
    """poisson with integrate=False must solve A x = h, not A x = B h."""
    fem1 = Solver(tria_mesh, lump=True)
    _, evec = fem1.eigs(k=5)
    rhs = evec[:, 1:4]  # (n_vertices, 3)

    res1 = fem1.poisson(rhs, integrate=False)

    fem2 = Solver(tria_mesh, lump=True)
    fem2.mass = sparse.eye(fem2.stiffness.shape[0], dtype=fem2.stiffness.dtype)
    res2 = fem2.poisson(rhs, integrate=True)

    np.testing.assert_allclose(
        res1, res2, rtol=1e-6, atol=1e-9,
        err_msg="poisson with integrate=False does not match poisson with identity mass matrix",
    )
