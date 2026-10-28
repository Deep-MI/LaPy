import numpy as np
import pytest
from scipy.spatial import ConvexHull

from ...intersect import self_intersections
from ...remesh import closest_points, remesh
from ...tria_mesh import TriaMesh
from .test_tria_mesh import _torus


def _irregular_sphere(n=800, seed=0):
    """Convex hull of random points on the unit sphere, oriented outward, with many slivers."""
    p = np.random.default_rng(seed).normal(size=(n, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    t = ConvexHull(p).simplices
    normal = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
    inward = np.einsum("ij,ij->i", normal, p[t].mean(axis=1)) < 0
    t[inward] = t[inward][:, ::-1]
    return TriaMesh(p, t)


def _min_angles(mesh):
    p = mesh.v[mesh.t]
    out = []
    for i in range(3):
        x, y = p[:, (i + 1) % 3] - p[:, i], p[:, (i + 2) % 3] - p[:, i]
        cos = np.einsum("ij,ij->i", x, y) / np.linalg.norm(x, axis=1) / np.linalg.norm(y, axis=1)
        out.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return np.min(out, axis=0)


def _edge_lengths(mesh):
    e = np.vstack([mesh.t[:, [0, 1]], mesh.t[:, [1, 2]], mesh.t[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    return np.linalg.norm(mesh.v[e[:, 1]] - mesh.v[e[:, 0]], axis=1)


def _assert_closed_surface(mesh, genus):
    assert mesh.is_closed() and mesh.is_manifold() and mesh.is_oriented()
    assert mesh.genus() == genus
    assert len(np.unique(np.sort(mesh.t, axis=1), axis=0)) == len(mesh.t)


def test_remesh_gives_even_triangles_on_the_input_surface():
    """
    Remeshing a sphere made of slivers gives triangles with edges near the
    target length and no small angles, with all vertices on the input
    surface and no self-intersections.
    """
    mesh = _irregular_sphere()
    h = 0.1
    new = remesh(mesh, target_length=h)
    _assert_closed_surface(new, 0)
    assert new.volume() > 0
    length = _edge_lengths(new)
    assert np.mean((length > 0.8 * h) & (length < 4 / 3 * h)) > 0.95
    assert _min_angles(mesh).min() < 5 and _min_angles(new).min() > 20
    q, _ = closest_points(mesh, new.v)
    assert np.abs(q - new.v).max() < 1e-9
    assert len(self_intersections(new)) == 0


def test_coarsening_a_thin_tube_keeps_the_genus():
    """
    A torus with a triangular cross-section stays a torus when coarsened.
    Collapses along the tube would pinch it without the link condition, and
    the triangles barely tilt, so the tilt check alone does not catch it.
    """
    v, t = _torus(60, 3, 3.0, 0.1)
    for h in (0.5, 2.0):
        new = remesh(TriaMesh(v, t), target_length=h)
        assert len(new.v) < len(v)
        _assert_closed_surface(new, 1)


def test_super_triangle_does_not_collapse():
    """
    A tetrahedron with one face split at its center. Collapsing an edge of
    that face would fold the face onto itself, so only an edge from the
    center may go, which leaves a tetrahedron.
    """
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1 / 3, 1 / 3, 0]], dtype=float)
    t = np.array([[0, 1, 3], [1, 2, 3], [2, 0, 3], [0, 4, 1], [1, 4, 2], [2, 4, 0]])
    mesh = TriaMesh(v, t)
    assert mesh.volume() > 0
    new = remesh(mesh, target_length=10.0)
    _assert_closed_surface(new, 0)
    assert len(new.v) == 4


def test_tetrahedron_next_to_another_component():
    """
    A tetrahedron cannot be coarsened further, also when the mesh has other
    components; the sphere next to it coarsens down to a tetrahedron too.
    """
    tv = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float) + [5, 0, 0]
    tt = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])
    sphere = _irregular_sphere(200)
    mesh = TriaMesh(np.vstack([sphere.v, tv]), np.vstack([sphere.t, tt + len(sphere.v)]))
    new = remesh(mesh, target_length=10.0)
    _assert_closed_surface(new, 0)
    assert len(new.v) == 8 and new.connected_components()[0] == 2


def test_single_triangle():
    """
    A mesh without interior vertices is refined along its boundary and keeps
    its corners and its area.
    """
    v = np.array([[0, 0, 0], [4, 0, 0], [0, 4, 0]], dtype=float)
    mesh = TriaMesh(v, np.array([[0, 1, 2]]))
    new = remesh(mesh, target_length=1.0)
    assert len(new.v) > 3 and new.is_manifold() and new.is_oriented()
    for corner in v:
        assert np.min(np.linalg.norm(new.v - corner, axis=1)) == 0
    assert np.isclose(new.area(), 8.0)
    assert np.all(new.v[:, 2] == 0)


def test_boundary_stays_in_place_and_2d_stays_2d():
    """
    Remeshing a jittered 2D grid at half its spacing keeps the boundary
    vertices, the area and the 2D vertices.
    """
    rng = np.random.default_rng(1)
    i, j = np.meshgrid(np.arange(11), np.arange(11), indexing="ij")
    v = np.column_stack([i.ravel(), j.ravel()]).astype(float)
    inner = (i.ravel() > 0) & (i.ravel() < 10) & (j.ravel() > 0) & (j.ravel() < 10)
    v[inner] += rng.uniform(-0.3, 0.3, size=(inner.sum(), 2))
    a = (i[:-1, :-1] * 11 + j[:-1, :-1]).ravel()
    t = np.concatenate([np.column_stack([a, a + 11, a + 12]), np.column_stack([a, a + 12, a + 1])])
    mesh = TriaMesh(v, t)
    new = remesh(mesh, target_length=0.5)
    assert new.is_2d()
    v2 = new.get_vertices(original_dim=True)
    assert v2.shape[1] == 2 and len(v2) > 2 * len(v)
    for corner in v[~inner]:
        assert np.min(np.linalg.norm(v2 - corner, axis=1)) == 0
    assert np.isclose(new.area(), 100.0)


def test_unsplit_boundary_keeps_its_vertices():
    """
    Without splitting boundary edges, the boundary keeps exactly its
    vertices, also with edges far longer than the target, where the
    triangles along them are left coarse rather than cut into slivers.
    """
    i, j = np.meshgrid(np.arange(5), np.arange(5), indexing="ij")
    v = np.column_stack([i.ravel(), j.ravel(), np.zeros(25)]).astype(float)
    a = (i[:-1, :-1] * 5 + j[:-1, :-1]).ravel()
    t = np.concatenate([np.column_stack([a, a + 5, a + 6]), np.column_stack([a, a + 6, a + 1])])
    mesh = TriaMesh(v, t)
    on_boundary = np.flatnonzero((i.ravel() % 4 == 0) | (j.ravel() % 4 == 0))
    for h in (0.5, 0.05):
        new = remesh(mesh, target_length=h, split_boundary=False)
        assert new.is_manifold() and new.is_oriented()
        loop = new.boundary_loops()[0]
        assert sorted(map(tuple, new.v[loop])) == sorted(map(tuple, v[on_boundary]))
        assert np.isclose(new.area(), 16.0)
        # no more vertices than a mesh of equilateral triangles of the target size has triangles
        assert len(new.v) < 16.0 / (np.sqrt(3) / 4 * h**2)


def test_remesh_rejects_unoriented_and_nonmanifold_meshes():
    """
    Remeshing needs consistent orientation and a manifold mesh.
    """
    v, t = _torus(20, 8)
    mixed = t.copy()
    mixed[::3] = mixed[::3, ::-1]
    with pytest.raises(ValueError, match="oriented"):
        remesh(TriaMesh(v, mixed))
    # two triangles touching at one vertex
    bow = np.array([[0, 0, 0], [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0]], dtype=float)
    with pytest.raises(ValueError, match="manifold"):
        remesh(TriaMesh(bow, np.array([[0, 2, 1], [0, 3, 4]])))


def test_closest_points_in_every_region():
    """
    Points closest to the inside, each corner and each edge of a triangle.
    """
    mesh = TriaMesh(np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=float), np.array([[0, 1, 2]]))
    query = np.array(
        [[0.5, 0.5, 1], [-1, -1, 0], [3, -1, 2], [-1, 3, 0], [1, -1, 0], [-1, 1, 0], [2, 2, 0]],
        dtype=float,
    )
    expected = np.array(
        [[0.5, 0.5, 0], [0, 0, 0], [2, 0, 0], [0, 2, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        dtype=float,
    )
    q, tria_idx = closest_points(mesh, query)
    np.testing.assert_allclose(q, expected, atol=1e-12)
    assert np.all(tria_idx == 0)
