"""Tests for lapy.conformal: symmetric constraints and solver agreement."""

import logging

import numpy as np
import pytest
from scipy import sparse

from ... import conformal
from ...conformal import (
    _dirichlet_system,
    _inverse_stereographic_south,
    _sparse_symmetric_solve,
    linear_beltrami_solver,
    spherical_conformal_map,
    spherical_tutte_map,
)
from ...tria_mesh import TriaMesh


def _signed_volume(v, t):
    """Signed volume of a closed surface; the sign flips when it is mirrored."""
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    return np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0


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


def test_solved_systems_are_real_and_symmetric(sphere, monkeypatch):
    """Both systems must be real and still symmetric after the constraints.

    The Cholesky solver reads only the lower triangular part of the matrix, so
    a matrix that lost its symmetry would make the result depend on which
    solver backend happens to be installed. Staying real keeps a real-only
    Cholesky backend usable.
    """
    seen = []
    original = conformal._sparse_symmetric_solve

    def spy(A, b, use_cholmod=False):
        mat = sparse.csc_matrix(A)
        asym = abs(mat - mat.T)
        seen.append(
            (np.iscomplexobj(mat), np.iscomplexobj(b),
             asym.max() if asym.nnz else 0.0)
        )
        return original(A, b, use_cholmod=use_cholmod)

    monkeypatch.setattr(conformal, "_sparse_symmetric_solve", spy)
    spherical_conformal_map(sphere)

    assert len(seen) == 2, "expected the harmonic and the Beltrami solve"
    for complex_a, complex_b, asym in seen:
        assert not complex_a
        assert not complex_b
        assert asym == 0.0


def test_sparse_symmetric_solve_rejects_complex():
    """Complex input must be refused rather than silently mis-factorised."""
    A = sparse.eye(3, format="csc")
    with pytest.raises(ValueError, match="real-valued"):
        _sparse_symmetric_solve(A, np.ones(3, dtype=complex))
    with pytest.raises(ValueError, match="real-valued"):
        _sparse_symmetric_solve(A.astype(complex), np.ones(3))


def test_dirichlet_system_rejects_duplicate_indices():
    """A repeated index prescribes two values for one vertex.

    Accepting it would drop that vertex's column twice while only one of the two
    values survives in the right hand side, so the free block would be solved
    against a load no boundary condition corresponds to.
    """
    A = sparse.eye(4, format="csc")
    target = np.array([[1.0, 0.0], [9.0, 9.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match="unique"):
        _dirichlet_system(A, np.array([1, 1, 3]), target)


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


def test_spherical_conformal_map_distributes_area(sphere):
    """The south pole chart must be built from the rescaled map.

    Reusing the denominator computed before the rescale scales every vertex by
    the wrong factor and collapses almost the whole sphere into a small patch;
    the area ratio was then five orders of magnitude instead of a handful.
    """
    areas = TriaMesh(spherical_conformal_map(sphere), sphere.t).tria_areas()

    assert areas.max() / areas.min() < 10.0


def test_small_radius_is_not_a_degeneracy():
    """A vertex close to the origin must not abort the parameterisation.

    ``1 + S[:, 2]`` is ``2 |z|^2 / (1 + |z|^2)``, so it is quadratically small
    near the origin: on this mesh ``min |z|`` is about 5e-5 while that
    denominator reaches 5e-9, which an absolute zero test rejects even though
    nothing is degenerate.
    """
    tria = TriaMesh.read_off("data/icosahedron.off")
    tria.refine_(it=5)
    tria.normalize_()

    mapping = spherical_conformal_map(tria)

    np.testing.assert_allclose(
        np.linalg.norm(mapping, axis=1), 1.0, rtol=1e-8, atol=1e-10
    )


def test_inverse_stereographic_south_round_trip():
    """Projecting through the south pole and back must be the identity."""
    rng = np.random.default_rng(0)
    points = rng.standard_normal((200, 3))
    points /= np.linalg.norm(points, axis=1)[:, np.newaxis]

    chart = points[:, :2] / (1 + points[:, 2])[:, np.newaxis]

    np.testing.assert_allclose(
        _inverse_stereographic_south(chart), points, rtol=1e-10, atol=1e-12
    )


def test_spherical_conformal_map_preserves_orientation(sphere):
    """The parameterisation must not mirror the surface.

    The Beltrami solve happens in the south pole chart, so inverting it with the
    northern formula returns the sphere mirrored in z. That flips the surface
    orientation and makes the map anti-conformal rather than conformal.
    """
    mapping = spherical_conformal_map(sphere)

    assert _signed_volume(mapping, sphere.t) > 0
    assert _signed_volume(sphere.v, sphere.t) > 0


def test_spherical_tutte_map_is_an_oriented_sphere(sphere):
    """The Tutte map needs only the connectivity and must still give a sphere."""
    mapping = spherical_tutte_map(sphere)

    assert mapping.shape == sphere.v.shape
    assert not np.isnan(mapping).any()
    np.testing.assert_allclose(
        np.linalg.norm(mapping, axis=1), 1.0, rtol=1e-8, atol=1e-10
    )
    assert _signed_volume(mapping, sphere.t) > 0


def test_spherical_tutte_map_rejects_non_genus_zero(square):
    """The Tutte map promises a genus-0 closed surface, so it must check."""
    with pytest.raises(ValueError, match="genus-0"):
        spherical_tutte_map(square)


@pytest.mark.parametrize("mode", ["nan", "raise"])
def test_falls_back_to_tutte_map(sphere, monkeypatch, caplog, mode):
    """A harmonic map that degenerates must fall back instead of raising.

    A collapsed map reaches the caller either as NaN or as the ValueError the
    rescale raises when the southernmost triangle has no extent. Both have to
    route to the Tutte map, which only uses the connectivity.
    """
    original = conformal._rescale_polar_triangles
    calls = []

    def poisoned(z, tria, bigtri):
        calls.append(1)
        if len(calls) == 1:  # the harmonic map
            if mode == "nan":
                return np.full_like(z, np.nan)
            raise ValueError("southernmost triangle radius contains zero entries")
        return original(z, tria, bigtri)

    monkeypatch.setattr(conformal, "_rescale_polar_triangles", poisoned)

    with caplog.at_level(logging.WARNING, logger="lapy.conformal"):
        mapping = spherical_conformal_map(sphere)

    assert "Tutte" in caplog.text
    assert not np.isnan(mapping).any()
    np.testing.assert_allclose(
        np.linalg.norm(mapping, axis=1), 1.0, rtol=1e-8, atol=1e-10
    )


def test_cholmod_matches_lu(sphere):
    """The Cholesky and the LU backend must agree."""
    pytest.importorskip("sksparse.cholmod")

    lu = spherical_conformal_map(sphere, use_cholmod=False)
    chol = spherical_conformal_map(sphere, use_cholmod=True)

    np.testing.assert_allclose(chol, lu, rtol=1e-8, atol=1e-10)
