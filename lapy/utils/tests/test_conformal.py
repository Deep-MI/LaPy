"""Tests for lapy.conformal — symmetric constraints and solver agreement."""

import numpy as np
import pytest
from scipy import sparse

from ... import conformal
from ...conformal import linear_beltrami_solver, spherical_conformal_map
from ...tria_mesh import TriaMesh


@pytest.fixture
def sphere():
    """Refined icosahedron projected onto the unit sphere."""
    tria = TriaMesh.read_off("data/icosahedron.off")
    tria.refine_(it=2)
    tria.normalize_()
    return tria


@pytest.fixture
def square():
    """Planar triangulation of the unit square."""
    return TriaMesh.read_off("data/square-mesh.off")


def test_solved_systems_are_symmetric(sphere, monkeypatch):
    """Both systems must still be symmetric after the constraints are imposed.

    The Cholesky solver reads only the lower triangular part of the matrix, so
    a matrix that lost its symmetry would make the result depend on which
    solver backend happens to be installed.
    """
    seen = []
    original = conformal._sparse_symmetric_solve

    def spy(A, b, use_cholmod=False):
        mat = sparse.csc_matrix(A)
        asym = abs(mat - mat.T)
        seen.append(asym.max() if asym.nnz else 0.0)
        return original(A, b, use_cholmod=use_cholmod)

    monkeypatch.setattr(conformal, "_sparse_symmetric_solve", spy)
    spherical_conformal_map(sphere)

    assert len(seen) == 2, "expected the harmonic and the Beltrami solve"
    assert max(seen) == 0.0


def test_linear_beltrami_solver_recovers_identity(square):
    """Without distortion and with the boundary pinned, the map is the identity.

    The assembled matrix reduces to the cotangent Laplacian for ``mu = 0``, and
    that annihilates linear functions on a planar mesh. This pins down both the
    constrained and the free block of the eliminated system.
    """
    mu = np.zeros(square.t.shape[0], dtype=complex)
    boundary = np.concatenate(square.boundary_loops())

    out = linear_beltrami_solver(square, mu, boundary, square.v[boundary, :2])

    assert out.shape == (len(square.v), 2)
    np.testing.assert_array_equal(out[boundary], square.v[boundary, :2])
    np.testing.assert_allclose(out, square.v[:, :2], atol=1e-6)


def test_spherical_conformal_map_lands_on_unit_sphere(sphere):
    """The parameterisation must be a map onto the unit sphere."""
    mapping = spherical_conformal_map(sphere)

    assert mapping.shape == sphere.v.shape
    np.testing.assert_allclose(
        np.linalg.norm(mapping, axis=1), 1.0, rtol=1e-8, atol=1e-10
    )


def test_cholmod_matches_lu(sphere):
    """The Cholesky and the LU backend must agree."""
    pytest.importorskip("sksparse.cholmod")

    lu = spherical_conformal_map(sphere, use_cholmod=False)
    chol = spherical_conformal_map(sphere, use_cholmod=True)

    np.testing.assert_allclose(chol, lu, rtol=1e-8, atol=1e-10)
