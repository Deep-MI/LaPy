"""Tests for Solver.poisson — backward compatibility and 2-D rhs support."""

import numpy as np
import pytest

from ...solver import Solver
from ...tria_mesh import TriaMesh


@pytest.fixture
def tria_mesh():
    return TriaMesh.read_off("data/square-mesh.off")


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

