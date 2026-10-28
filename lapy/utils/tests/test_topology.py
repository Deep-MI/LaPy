"""Tests for lapy.topology: loops, surgery, handles and topology correction."""

import numpy as np
import pytest

from ...intersect import self_intersections
from ...solver import Solver
from ...topology import (
    _nudge,
    capped_surgery,
    disk_cap,
    essential_saddles,
    find_handles,
    fix_topology,
    handle_clusters,
    handle_patches,
    refine_loop_pair,
    shortest_crossing_loop,
    tree_cotree_loops,
)
from ...tria_mesh import TriaMesh
from ._shapes import block_with_bridge, block_with_holes, box, torus, voxel_surface


def _cut(tria, loops):
    """Copy of the mesh cut open along closed loops."""
    cut = TriaMesh(tria.v, tria.t)
    cut.cut_(np.concatenate([np.column_stack([lp, np.roll(lp, -1)]) for lp in loops]))
    return cut


def _n_components(tria):
    _, labels = tria.connected_components()
    return len(np.unique(labels[np.unique(tria.t)]))


def _spaced_holes():
    """Slab with three through-holes, separated by walls three voxels wide."""
    mask = np.ones((11, 3, 2), dtype=bool)
    mask[[1, 5, 9], 1, :] = False
    return voxel_surface(mask)


def _first_eigenfunction(tria):
    return Solver(tria, lump=True).eigs(k=2)[1][:, 1]


# ---------------------------------------------------------------------------
# Loops
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("g", [1, 3])
@pytest.mark.parametrize("root", [0, 17])
def test_tree_cotree_loops_are_non_separating(g, root):
    """Each of the 2g loops lowers the genus by one when cut, and all of them keep the mesh in one piece."""
    tria = block_with_holes(g)
    loops, lengths = tree_cotree_loops(tria, root=root)
    assert len(loops) == 2 * g
    assert np.all(np.diff(lengths) >= 0)
    for lp in loops:
        assert len(np.unique(lp)) == len(lp)
        cut = _cut(tria, [lp])
        assert _n_components(cut) == 1
        assert cut.genus() == g - 1
    assert _n_components(_cut(tria, loops)) == 1


def test_refined_torus_loops_are_meridian_and_inner_equator():
    n, m, big_r, small_r = 40, 16, 3.0, 1.0
    tria = torus(n, m, big_r, small_r)
    loops, _ = tree_cotree_loops(tria)
    a, b, la, lb = refine_loop_pair(tria, loops[-1])
    meridian = m * 2 * small_r * np.sin(np.pi / m)
    inner = n * 2 * (big_r - small_r) * np.sin(np.pi / n)
    assert sorted([la, lb]) == pytest.approx([meridian, inner])
    p = tria.v[a]
    assert la == pytest.approx(np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1).sum())


def test_crossing_loop_search_grows_until_optimal():
    """Starting with a radius far too small, the search still finds the global optimum."""
    tria = torus(60, 12, 6.0, 1.0)
    meridian = np.arange(12)
    crossing, length = shortest_crossing_loop(tria, meridian, radius=0.5)
    assert length == pytest.approx(60 * 2 * 5.0 * np.sin(np.pi / 60))
    p = tria.v[crossing]
    assert length == pytest.approx(np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1).sum())


def test_vertex_weights_route_loops_around_a_region():
    """Expensive vertices on the inner half of the tube push the crossing loop to the outer half."""
    tria = torus(60, 12, 6.0, 1.0)
    radius = np.linalg.norm(tria.v[:, :2], axis=1)
    weights = np.where(radius < 6.0 - 1e-9, 100.0, 1.0)
    crossing, length = shortest_crossing_loop(tria, np.arange(12), vertex_weights=weights)
    assert np.all(radius[crossing] >= 6.0 - 1e-9)
    assert length == pytest.approx(60 * 2 * 6.0 * np.sin(np.pi / 60))


def test_crossing_loop_search_keeps_edges_next_to_expensive_vertices():
    """The inner equator stays optimal when both rings next to it are expensive.

    Each edge of the equator lies in triangles whose third vertex is out of reach of a small
    search radius, so the search must keep triangles with any vertex in reach.
    """
    tria = torus(60, 12, 6.0, 1.0)
    ring = np.arange(len(tria.v)) % 12
    weights = np.where(np.isin(ring, [5, 7]), 50.0, 1.0)
    crossing, length = shortest_crossing_loop(tria, np.arange(12), vertex_weights=weights)
    assert length == pytest.approx(60 * 2 * 5.0 * np.sin(np.pi / 60))
    assert np.all(ring[crossing] == 6)


def test_crossing_loop_rejects_separating_loop():
    tria = box(3, 3, 3)
    # the block spans corners 1 to 4; this ring of side vertices separates top from bottom
    v = tria.v
    side = np.isin(v[:, 0], [1, 4]) | np.isin(v[:, 1], [1, 4])
    ring = np.flatnonzero((v[:, 2] == 2) & side)
    ring = ring[np.argsort(np.arctan2(v[ring, 1] - 2.5, v[ring, 0] - 2.5))]
    assert len(ring) == 12
    with pytest.raises(RuntimeError):
        shortest_crossing_loop(tria, ring)


# ---------------------------------------------------------------------------
# Surgery
# ---------------------------------------------------------------------------


def test_disk_cap_spans_a_circle():
    k, radius = 40, 5.0
    a = 2 * np.pi * np.arange(k) / k
    boundary = np.column_stack([radius * np.cos(a), radius * np.sin(a), np.zeros(k)])
    inner, tris = disk_cap(boundary, 2 * np.pi * radius / k)
    disk = TriaMesh(np.vstack([boundary, inner]), tris)
    assert disk.euler() == 1
    assert disk.is_oriented()
    assert np.count_nonzero(disk.adj_sym.data == 1) // 2 == k
    assert disk.area() == pytest.approx(np.pi * radius**2, rel=0.01)
    assert np.allclose(inner[:, 2], 0.0)


def test_disk_cap_fills_a_concave_polygon_without_folds():
    """An L-shaped boundary, whose centroid lies outside it, is filled flat and without folds."""
    corners = np.array([[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3], [0, 0]], dtype=float)
    sides = zip(corners[:-1], corners[1:], strict=True)
    pts = [np.linspace(a, b, int(np.linalg.norm(b - a) / 0.25), endpoint=False) for a, b in sides]
    boundary = np.column_stack([np.concatenate(pts), np.zeros(sum(len(p) for p in pts))])
    inner, tris = disk_cap(boundary, 0.25)
    disk = TriaMesh(np.vstack([boundary, inner]), tris)
    assert len(inner) > 0
    assert disk.euler() == 1 and disk.is_oriented()
    assert np.allclose(inner[:, 2], 0.0)
    # every triangle faces up, the way the boundary runs, and together they cover the L exactly
    normals = disk.tria_normals()
    assert np.all(normals[:, 2] > 0)
    assert disk.area() == pytest.approx(5.0)


def test_disk_cap_keeps_a_thin_concave_boundary_unfolded():
    """Area decides the triangulation, and smoothing may not push vertices across a thin spike."""
    boundary = np.array([[0, 0, 0], [1, 0, 0], [0.001, 0.001, 0], [0, 1, 0]], dtype=float)
    inner, tris = disk_cap(boundary, 2.0)
    p = np.vstack([boundary, inner])[tris]
    signed = 0.5 * np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])[:, 2]
    assert np.all(signed > 0)
    assert signed.sum() == pytest.approx(0.001)


def test_disk_cap_is_close_to_a_minimal_surface():
    """A saddle-shaped boundary gets a disk with almost no mean curvature, smaller than the cone over its centroid."""
    k = 40
    a = 2 * np.pi * np.arange(k) / k
    boundary = np.column_stack([np.cos(a), np.sin(a), np.cos(2 * a)])
    inner, tris = disk_cap(boundary, 2 * np.pi / k)
    disk = TriaMesh(np.vstack([boundary, inner]), tris)
    fem = Solver(disk, lump=True)
    curvature = np.linalg.norm((fem.stiffness @ disk.v)[k:], axis=1) / fem.mass.diagonal()[k:]
    assert np.median(curvature) < 0.02
    centre = boundary.mean(axis=0)
    cone = 0.5 * np.linalg.norm(np.cross(boundary - centre, np.roll(boundary, -1, axis=0) - centre), axis=1).sum()
    assert disk.area() < 0.95 * cone


@pytest.mark.parametrize("g", [1, 2])
def test_capped_surgery_lowers_genus_and_keeps_other_vertices(g):
    tria = block_with_holes(g)
    loops, _ = tree_cotree_loops(tria)
    out, source = capped_surgery(tria, loops[0])
    assert out.is_closed() and out.is_manifold() and out.is_oriented()
    assert out.genus() == g - 1
    assert np.array_equal(np.sort(source), np.sort(loops[0]))
    off_loop = np.setdiff1d(np.arange(len(tria.v)), loops[0])
    assert np.array_equal(out.v[off_loop], tria.v[off_loop])


def test_disk_cap_leaves_out_triangles_without_area():
    """Unevenly sampled boundary points on a line need no zero-area triangles."""
    boundary = np.array([[0, 0, 0], [0.001, 0, 0], [0.002, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    inner, tris = disk_cap(boundary, 2.0, iterations=0)
    p = np.vstack([boundary, inner])[tris]
    signed = 0.5 * np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])[:, 2]
    assert np.all(signed > 0)
    assert signed.sum() == pytest.approx(1.0)


def test_nudging_cut_sides_does_not_flip_skinny_triangles():
    """Each side of a cut moves only as far as its triangles stay unflipped."""
    v = np.array([[0, 1, 0], [0, 0, 0], [-1, 0, 0], [-1, 2, 0], [1, 2, 0], [1, 0, 0], [-0.001, 0.001, 0]], dtype=float)
    # a flat fan around vertex 0, with vertex 6 just beside the corner at vertex 1
    t = np.array([[0, 1, 6], [0, 6, 2], [6, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 1]])
    copies = np.array([1, 2, 5])
    moved = _nudge(v, t, copies, 0.2)

    def signed(p):
        return np.cross(p[t][:, 1] - p[t][:, 0], p[t][:, 2] - p[t][:, 0])[:, 2]

    assert np.all(np.sign(signed(moved)) == np.sign(signed(v)))
    # the vertex away from the skinny triangle still moves by the full gap
    assert np.linalg.norm(moved[5] - v[5]) == pytest.approx(0.2)


def test_capped_surgery_separates_a_loop_through_every_vertex():
    """On a 3 x 3 torus, a loop can visit all vertices, leaving the cut copies no neighbours off it."""
    tria = torus(3, 3)
    loop = np.arange(9)
    out, source = capped_surgery(tria, loop)
    assert out.is_closed() and out.is_manifold() and out.genus() == 0
    # the copy of each loop vertex has moved away from the vertex itself
    copy = 9 + np.argsort(source)
    assert np.linalg.norm(out.v[loop] - out.v[copy], axis=1).min() > 0.1


# ---------------------------------------------------------------------------
# Handles
# ---------------------------------------------------------------------------


def test_every_hole_found_with_tight_loops():
    handles = find_handles(block_with_holes(3))
    assert len(handles) == 3
    for h in handles:
        # hole perimeter and the cross-section of the 1 x 2 voxel wall around it
        assert sorted([h.length_a, h.length_b]) == pytest.approx([4.0, 6.0])


def test_thin_bridge_is_found_and_its_loops_are_non_separating():
    tria = block_with_bridge()
    (handle,) = find_handles(tria)
    # the arch has a one-voxel cross-section, the hole under it is 4 x 2 voxels
    assert min(handle.length_a, handle.length_b) == pytest.approx(4.0)
    assert max(handle.length_a, handle.length_b) == pytest.approx(12.0)
    for loop in (handle.loop_a, handle.loop_b):
        assert _n_components(_cut(tria, [loop])) == 1
    assert len(np.intersect1d(handle.loop_a, handle.loop_b)) >= 1


def test_find_handles_rejects_non_manifold_input():
    mask = np.zeros((2, 2, 2), dtype=bool)
    mask[0, 0, 0] = mask[1, 1, 1] = True
    with pytest.raises(ValueError):
        find_handles(voxel_surface(mask))


def test_handle_patches_merge_only_close_handles():
    tria = _spaced_holes()
    handles = find_handles(tria)
    labels, handle_patch = handle_patches(tria, handles, radius=1.5)
    assert sorted(handle_patch) == [1, 2, 3]
    assert set(np.unique(labels)) == {0, 1, 2, 3}
    _, handle_patch = handle_patches(tria, handles, radius=3.5)
    assert np.all(handle_patch == 1)


# ---------------------------------------------------------------------------
# Topology correction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tria",
    [torus(), block_with_holes(3), block_with_bridge()],
    ids=["torus", "three-holes", "bridge"],
)
def test_fix_topology_gives_a_clean_sphere(tria):
    out, origin, surgeries = fix_topology(tria)
    assert out.is_closed() and out.is_manifold() and out.is_oriented()
    assert out.genus() == 0
    assert len(surgeries) == tria.genus()
    assert len(self_intersections(out)) == 0
    # only loop vertices move; all others keep index and position
    moved = np.zeros(len(tria.v), dtype=bool)
    for s in surgeries:
        moved[s.loop] = True
    assert np.array_equal(out.v[: len(tria.v)][~moved], tria.v[~moved])
    assert np.array_equal(origin[: len(tria.v)], np.arange(len(tria.v)))


def test_fix_topology_ignores_unused_vertices():
    """An unused vertex at index 0 must not become the root of the loop search."""
    ring = torus()
    tria = TriaMesh(np.vstack([[10.0, 10.0, 10.0], ring.v]), ring.t + 1)
    out, origin, surgeries = fix_topology(tria)
    assert out.genus() == 0
    assert len(surgeries) == 1
    assert origin[0] == 0


def test_choice_decides_between_cut_and_fill():
    """On the bridge, the short loop cuts the arch and the long one fills the hole under it."""
    tria = block_with_bridge()
    volume = tria.volume()

    def classify(v, t, loop):
        # the arch cross-section is the short loop
        p = v[loop]
        return np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1).sum() < 6.0

    def longer(a, b, length_a, length_b, scores, v=None):
        return 0 if length_a > length_b else 1

    cut, _, (s_cut,) = fix_topology(tria, classify=classify)
    fill, _, (s_fill,) = fix_topology(tria, choose=longer, classify=classify)
    assert (s_cut.operation, s_fill.operation) == ("cut", "fill")
    assert cut.volume() < volume < fill.volume()
    assert fill.genus() == 0


# ---------------------------------------------------------------------------
# Essential saddles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tria",
    [torus(), block_with_holes(3), block_with_bridge(), box(4, 4, 2)],
    ids=["torus", "three-holes", "bridge", "box"],
)
def test_two_essential_saddles_per_handle(tria):
    saddles = essential_saddles(tria, _first_eigenfunction(tria))
    assert saddles.min() >= 0
    assert saddles.sum() == 2 * tria.genus()


def test_essential_saddles_reject_pinch_vertices():
    """Two tetrahedra sharing a vertex would give a negative saddle count there."""
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
    t = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3], [0, 4, 5], [0, 6, 4], [0, 5, 6], [4, 6, 5]])
    with pytest.raises(ValueError):
        essential_saddles(TriaMesh(v, t), np.array([0.0, -1, 1, 2, -1, 1, 2]))


def test_separate_holes_form_separate_clusters():
    tria = _spaced_holes()
    handles = find_handles(tria)
    cluster, near = handle_clusters(tria, handles, essential_saddles(tria, _first_eigenfunction(tria)))
    assert [len(s) for s in near] == [2, 2, 2]
    assert sorted(cluster) == [0, 1, 2]


def test_holes_with_shared_walls_form_one_cluster():
    tria = block_with_holes(3)
    handles = find_handles(tria)
    cluster, _ = handle_clusters(tria, handles, essential_saddles(tria, _first_eigenfunction(tria)))
    assert np.all(cluster == 0)
