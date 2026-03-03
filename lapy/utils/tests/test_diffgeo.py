"""Tests for lapy.diffgeo vectorized (multi-function) paths.

Each *_multi_vs_single test verifies that passing a 2-D array of shape
(n_vertices, n_functions) produces exactly the same result as repeated
single-function calls.  The 1-D backward-compatibility path is implicitly
covered because each multi-vs-single test calls the single-function version
internally.

The tria_gradient_divergence_laplacian_multi test additionally verifies the
mathematical correctness of the multi-function paths via the identity
    -div(grad(f_k)) = lambda_k * B * f_k
for FEM eigenfunctions.
"""

import numpy as np
import pytest

from ...diffgeo import (
    tria_compute_divergence,
    tria_compute_divergence2,
    tria_compute_geodesic_f,
    tria_compute_gradient,
    tria_compute_rotated_f,
    tet_compute_divergence,
    tet_compute_gradient,
)
from ...solver import Solver
from ...tria_mesh import TriaMesh
from ...tet_mesh import TetMesh


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tria_mesh():
    return TriaMesh.read_off("data/square-mesh.off")


@pytest.fixture
def tet_mesh():
    T = TetMesh.read_vtk("data/cubeTetra.vtk")
    T.orient_()
    return T


# ---------------------------------------------------------------------------
# TriaMesh — gradient
# ---------------------------------------------------------------------------


def test_tria_compute_gradient_multi_vs_single(tria_mesh):
    """Batch gradient must match repeated single-function calls column-by-column."""
    T = tria_mesh
    _, evec = Solver(T, lump=True).eigs(k=5)
    vfuncs = evec[:, 1:5]  # (n_vertices, 4)

    grad_batch = tria_compute_gradient(T, vfuncs)  # (n_triangles, 4, 3)

    assert grad_batch.shape == (len(T.t), 4, 3)
    for k in range(4):
        np.testing.assert_allclose(
            grad_batch[:, k, :],
            tria_compute_gradient(T, vfuncs[:, k]),
            rtol=1e-12, atol=1e-14,
            err_msg=f"Gradient mismatch at column {k}",
        )


# ---------------------------------------------------------------------------
# TriaMesh — divergence (cotangent formulation)
# ---------------------------------------------------------------------------


def test_tria_compute_divergence_multi_vs_single(tria_mesh):
    """Batch divergence must match repeated single-function calls."""
    T = tria_mesh
    _, evec = Solver(T, lump=True).eigs(k=5)
    vfuncs = evec[:, 1:5]

    grad_batch = tria_compute_gradient(T, vfuncs)        # (T, 4, 3)
    div_batch = tria_compute_divergence(T, grad_batch)   # (V, 4)

    assert div_batch.shape == (len(T.v), 4)
    for k in range(4):
        np.testing.assert_allclose(
            div_batch[:, k],
            tria_compute_divergence(T, tria_compute_gradient(T, vfuncs[:, k])),
            rtol=1e-12, atol=1e-14,
            err_msg=f"Divergence mismatch at column {k}",
        )


def test_tria_gradient_divergence_laplacian_multi(tria_mesh):
    """Mathematical check: -div(grad(f_k)) = lambda_k * B * f_k for eigenfunctions."""
    T = tria_mesh
    fem = Solver(T, lump=True)
    evals, evec = fem.eigs(k=5)
    vfuncs = evec[:, 1:5]

    div_batch = tria_compute_divergence(T, tria_compute_gradient(T, vfuncs))

    for k in range(4):
        np.testing.assert_allclose(
            -div_batch[:, k],
            evals[k + 1] * fem.mass.dot(vfuncs[:, k]),
            rtol=1e-3, atol=1e-7,
            err_msg=f"Laplacian identity failed for eigenfunction {k + 1}",
        )


# ---------------------------------------------------------------------------
# TriaMesh — divergence2 (flux formulation)
# ---------------------------------------------------------------------------


def test_tria_compute_divergence2_multi_vs_single(tria_mesh):
    """Batch divergence2 must match repeated single-function calls."""
    T = tria_mesh
    _, evec = Solver(T, lump=True).eigs(k=5)
    vfuncs = evec[:, 1:5]

    grad_batch = tria_compute_gradient(T, vfuncs)
    div_batch = tria_compute_divergence2(T, grad_batch)

    assert div_batch.shape == (len(T.v), 4)
    for k in range(4):
        np.testing.assert_allclose(
            div_batch[:, k],
            tria_compute_divergence2(T, tria_compute_gradient(T, vfuncs[:, k])),
            rtol=1e-12, atol=1e-14,
            err_msg=f"Divergence2 mismatch at column {k}",
        )


# ---------------------------------------------------------------------------
# TriaMesh — geodesic_f
# ---------------------------------------------------------------------------


def test_tria_compute_geodesic_f_multi_vs_single(tria_mesh):
    """Batch geodesic_f must match repeated single-function calls."""
    T = tria_mesh
    _, evec = Solver(T, lump=True).eigs(k=5)
    vfuncs = evec[:, 1:4]

    gf_batch = tria_compute_geodesic_f(T, vfuncs)

    assert gf_batch.shape == (len(T.v), 3)
    for k in range(3):
        np.testing.assert_allclose(
            gf_batch[:, k],
            tria_compute_geodesic_f(T, vfuncs[:, k]),
            rtol=1e-12, atol=1e-14,
            err_msg=f"Geodesic_f mismatch at column {k}",
        )


# ---------------------------------------------------------------------------
# TriaMesh — rotated_f
# ---------------------------------------------------------------------------


def test_tria_compute_rotated_f_multi_vs_single(tria_mesh):
    """Batch rotated_f must match repeated single-function calls."""
    T = tria_mesh
    _, evec = Solver(T, lump=True).eigs(k=5)
    vfuncs = evec[:, 1:4]

    rf_batch = tria_compute_rotated_f(T, vfuncs)

    assert rf_batch.shape == (len(T.v), 3)
    for k in range(3):
        np.testing.assert_allclose(
            rf_batch[:, k],
            tria_compute_rotated_f(T, vfuncs[:, k]),
            rtol=1e-12, atol=1e-14,
            err_msg=f"Rotated_f mismatch at column {k}",
        )


# ---------------------------------------------------------------------------
# TetMesh — gradient
# ---------------------------------------------------------------------------


def test_tet_compute_gradient_multi_vs_single(tet_mesh):
    """Batch tet gradient must match repeated single-function calls."""
    T = tet_mesh
    _, evec = Solver(T, lump=True).eigs(k=5)
    vfuncs = evec[:, 1:5]

    grad_batch = tet_compute_gradient(T, vfuncs)  # (n_tets, 4, 3)

    assert grad_batch.shape == (len(T.t), 4, 3)
    for k in range(4):
        np.testing.assert_allclose(
            grad_batch[:, k, :],
            tet_compute_gradient(T, vfuncs[:, k]),
            rtol=1e-12, atol=1e-14,
            err_msg=f"Tet gradient mismatch at column {k}",
        )


# ---------------------------------------------------------------------------
# TetMesh — divergence
# ---------------------------------------------------------------------------


def test_tet_compute_divergence_multi_vs_single(tet_mesh):
    """Batch tet divergence must match repeated single-function calls."""
    T = tet_mesh
    _, evec = Solver(T, lump=True).eigs(k=5)
    vfuncs = evec[:, 1:5]

    grad_batch = tet_compute_gradient(T, vfuncs)
    div_batch = tet_compute_divergence(T, grad_batch)

    assert div_batch.shape == (len(T.v), 4)
    for k in range(4):
        np.testing.assert_allclose(
            div_batch[:, k],
            tet_compute_divergence(T, tet_compute_gradient(T, vfuncs[:, k])),
            rtol=1e-12, atol=1e-14,
            err_msg=f"Tet divergence mismatch at column {k}",
        )

