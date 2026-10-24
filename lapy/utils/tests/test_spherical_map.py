"""Tests for the spectral maps onto the sphere in lapy.diffgeo."""

import numpy as np
import pytest

from ...diffgeo import flipped_triangles, spherical_spectral_map, tria_spherical_project
from ...tria_mesh import TriaMesh


def _box_surface(nx, ny, nz):
    """Boundary of an nx x ny x nz block of unit voxels, two triangles per voxel face."""
    m = np.zeros((nx + 2, ny + 2, nz + 2), dtype=np.int8)
    m[1:-1, 1:-1, 1:-1] = 1
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    corner_shape = np.array(m.shape) + 1
    quads = []
    for axis in range(3):
        b, c = (axis + 1) % 3, (axis + 2) % 3
        step = np.diff(m, axis=axis)
        for sign in (1, -1):
            base = np.argwhere(step == sign)
            base[:, axis] += 1
            corners = np.repeat(base[:, None, :], 4, axis=1)
            corners[:, :, b] += square[:, 0]
            corners[:, :, c] += square[:, 1]
            if sign == 1:
                corners = corners[:, ::-1]
            quads.append(np.ravel_multi_index(corners.reshape(-1, 3).T, corner_shape).reshape(-1, 4))
    used, inv = np.unique(np.concatenate(quads), return_inverse=True)
    quads = inv.reshape(-1, 4)
    t = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    v = np.column_stack(np.unravel_index(used, corner_shape)).astype(float)
    return TriaMesh(v, t)


@pytest.fixture
def unit_sphere():
    """Refined icosahedron projected onto the unit sphere."""
    tria = TriaMesh.read_off("data/icosahedron.off")
    tria.refine_(it=3)
    v = tria.v - tria.v.mean(axis=0)
    return TriaMesh(v / np.linalg.norm(v, axis=1, keepdims=True), tria.t)


def test_repeated_eigenvalue_does_not_fold():
    """A square slab has a double eigenvalue; missing one copy would fold half of the map."""
    box = _box_surface(20, 20, 6)
    sphere = spherical_spectral_map(box)
    np.testing.assert_allclose(np.linalg.norm(sphere, axis=1), 1.0)
    assert not flipped_triangles(sphere, box.t).any()


@pytest.mark.parametrize(
    "v, t",
    [
        ([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]]),
        (
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
            [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]],
        ),
    ],
    ids=["tetrahedron", "octahedron"],
)
def test_tiny_meshes_map_without_folds(v, t):
    """Meshes with fewer vertices than the sparse solver needs are mapped too."""
    tria = TriaMesh(np.array(v, dtype=float), np.array(t))
    tria.orient_()
    sphere = spherical_spectral_map(tria)
    np.testing.assert_allclose(np.linalg.norm(sphere, axis=1), 1.0)
    assert not flipped_triangles(sphere, tria.t).any()


def test_map_does_not_depend_on_position(unit_sphere):
    """The map is intrinsic: a rigidly moved ellipsoid maps to the same sphere up to an orthogonal transform."""
    ellipsoid = TriaMesh(unit_sphere.v * np.array([3.0, 2.0, 1.0]), unit_sphere.t)
    a, b, c = 0.4, -1.1, 2.3
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    moved = TriaMesh(ellipsoid.v @ (rz @ ry @ rx).T + np.array([5.0, -3.0, 12.0]), ellipsoid.t)

    s1 = spherical_spectral_map(ellipsoid)
    s2 = spherical_spectral_map(moved)
    u, _, vt = np.linalg.svd(s1.T @ s2)
    np.testing.assert_allclose(s1 @ (u @ vt), s2, atol=1e-5)
    assert not flipped_triangles(s1, ellipsoid.t).any()


def test_flipped_triangles_finds_a_mirrored_cap(unit_sphere):
    """Mirroring a cap of the sphere reverses its triangles, which then face the centre."""
    v = unit_sphere.v.copy()
    cap = v[:, 2] > 0.7
    v[cap, 0] *= -1
    flipped = flipped_triangles(v, unit_sphere.t)
    inside = np.all(cap[unit_sphere.t], axis=1)
    outside = ~np.any(cap[unit_sphere.t], axis=1)
    assert inside.any()
    assert flipped[inside].all()
    assert not flipped[outside].any()


def test_tria_spherical_project_aligns_brain_axes(unit_sphere):
    """For an ellipsoid longest from back to front, then bottom to top, the axes keep their direction."""
    ellipsoid = TriaMesh(unit_sphere.v * np.array([40.0, 80.0, 60.0]), unit_sphere.t)
    projected = tria_spherical_project(ellipsoid)
    np.testing.assert_allclose(np.linalg.norm(projected.v, axis=1), 100.0)
    for axis in range(3):
        assert projected.v[np.argmax(ellipsoid.v[:, axis]), axis] > 90.0
        assert projected.v[np.argmin(ellipsoid.v[:, axis]), axis] < -90.0
