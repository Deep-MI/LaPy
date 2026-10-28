"""Isotropic remeshing of triangle meshes after Botsch and Kobbelt.

Each iteration splits long edges, collapses short edges, flips edges towards
valence 6, smooths tangentially and projects the vertices back onto the input
surface. Every operation runs in rounds over sets of edges whose
neighborhoods do not overlap, so a round is a few array operations instead of
a loop over the mesh.
"""

from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse, spatial

if TYPE_CHECKING:
    from .tria_mesh import TriaMesh


def remesh(
    tria: "TriaMesh",
    target_length: float | None = None,
    iterations: int = 5,
    max_rounds: int = 10,
    split_boundary: bool = True,
) -> "TriaMesh":
    """Remesh a triangle mesh to near-equilateral triangles of a given edge length.

    This follows the remeshing approach of Botsch and Kobbelt [1]_. Edges
    longer than 4/3 of the target are split at their midpoint, and edges
    shorter than 4/5 of it are collapsed to their midpoint. A collapse is
    only done if it keeps the topology, creates no edge longer than 4/3 of
    the target and tilts no triangle by more than 60 degrees. The topology
    check is the link condition [2]_: the two end points of the edge may
    share no neighbors other than the two vertices opposite the edge. Edges
    are then flipped where that brings vertex valences closer to 6, and the
    vertices are smoothed tangentially and projected back onto the input
    surface. Boundary vertices stay where they are.

    Parameters
    ----------
    tria : TriaMesh
        Oriented manifold triangle mesh.
    target_length : float, default=None
        Target edge length. Defaults to the mean edge length of the input.
    iterations : int, default=5
        Number of split, collapse, flip, smooth and project iterations.
    max_rounds : int, default=10
        Maximum number of rounds per operation and iteration. Later rounds
        do little, and what is left carries over to the next iteration.
    split_boundary : bool, default=True
        Whether long boundary edges are split. Without splitting, the
        boundary keeps exactly its vertices and edges, for example to stitch
        the mesh into a hole.

    Returns
    -------
    TriaMesh
        Remeshed surface, oriented like the input. A 2D mesh stays 2D.

    Raises
    ------
    ValueError
        If the mesh is not manifold or not oriented, or if the target length
        is not a finite positive number.

    References
    ----------
    .. [1] M. Botsch and L. Kobbelt. A remeshing approach to multiresolution
       modeling. In Proceedings of the Eurographics/ACM SIGGRAPH Symposium on
       Geometry Processing, pages 185-192, 2004. doi:10.1145/1057432.1057457
    .. [2] T. K. Dey, H. Edelsbrunner, S. Guha and D. V. Nekhayev. Topology
       preserving edge contraction. Publications de l'Institut Mathematique
       (Beograd), 66(80):23-45, 1999.
    """
    from .tria_mesh import TriaMesh

    if not tria.is_manifold():
        raise ValueError("Remeshing needs a manifold mesh, see TriaMesh.is_manifold.")
    if not tria.is_oriented():
        raise ValueError("Remeshing needs an oriented mesh, see TriaMesh.orient_.")
    v0 = np.asarray(tria.v, dtype=float)
    t0 = np.asarray(tria.t, dtype=np.int64)
    edges, _ = _unique_edges(t0)
    if target_length is None:
        target_length = np.linalg.norm(v0[edges[:, 1]] - v0[edges[:, 0]], axis=1).mean()
    if not np.isfinite(target_length) or target_length <= 0:
        raise ValueError(f"target_length must be a finite positive number, got {target_length}.")
    hi, lo = 4.0 / 3.0 * target_length, 4.0 / 5.0 * target_length
    surface = _Projector(v0, t0)
    v, t = v0.copy(), t0.copy()
    for _ in range(iterations):
        for _ in range(max_rounds):
            v, t, n = _split_round(v, t, hi, split_boundary)
            if n == 0:
                break
        for _ in range(max_rounds):
            v, t, n = _collapse_round(v, t, lo, hi)
            if n == 0:
                break
        for _ in range(max_rounds):
            t, n = _flip_round(v, t)
            if n == 0:
                break
        v = _tangential_smooth(v, t)
        moved = _interior_vertices(v, t)
        v[moved] = surface.closest(v[moved])
    used, t = np.unique(t, return_inverse=True)
    v = v[used]
    if tria.is_2d():
        v = v[:, :2]
    return TriaMesh(v, t.reshape(-1, 3), tria.fsinfo)


def closest_points(tria: "TriaMesh", points: np.ndarray, k: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Find the closest point on a triangle mesh for each query point.

    Only the ``k`` triangles with the nearest centroids are tested, so a
    point must be close to the surface compared with the triangle size.

    Parameters
    ----------
    tria : TriaMesh
        Triangle mesh.
    points : np.ndarray
        Query points of shape (n_points, 3).
    k : int, default=8
        Number of candidate triangles per point.

    Returns
    -------
    closest : np.ndarray
        Closest surface point for each query point, shape (n_points, 3).
    tria_idx : np.ndarray
        Index of the triangle that contains each closest point.

    Raises
    ------
    ValueError
        If ``k`` is not a positive integer.
    """
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}.")
    projector = _Projector(np.asarray(tria.v, dtype=float), np.asarray(tria.t, dtype=np.int64), k)
    return projector.closest(points, return_tria=True)


class _Projector:
    """Closest points on a fixed triangle mesh."""

    def __init__(self, v: np.ndarray, t: np.ndarray, k: int = 8):
        self.p = v[t]
        self.tree = spatial.KDTree(self.p.mean(axis=1))
        self.k = min(k, len(self.p))

    def closest(self, points: np.ndarray, return_tria: bool = False):
        """Return the closest surface points, and their triangles if ``return_tria``."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if len(points) == 0:
            return (points.copy(), np.empty(0, dtype=np.int64)) if return_tria else points.copy()
        _, idx = self.tree.query(points, k=self.k)
        idx = idx.reshape(len(points), -1)
        best = np.full(len(points), np.inf)
        out = points.copy()
        tria_idx = np.zeros(len(points), dtype=np.int64)
        for j in range(idx.shape[1]):
            tri = self.p[idx[:, j]]
            q = _closest_on_triangles(points, tri[:, 0], tri[:, 1], tri[:, 2])
            d = np.einsum("ij,ij->i", q - points, q - points)
            better = d < best
            best[better] = d[better]
            out[better] = q[better]
            tria_idx[better] = idx[better, j]
        return (out, tria_idx) if return_tria else out


def _closest_on_triangles(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Return the closest point to p on each triangle abc (Ericson, Real-Time Collision Detection 5.1.5)."""
    ab, ac = b - a, c - a
    ap, bp, cp = p - a, p - b, p - c
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = va + vb + vc
        q = a + ab * (vb / denom)[:, None] + ac * (vc / denom)[:, None]
        s_ab = d1 / (d1 - d3)
        s_ac = d2 / (d2 - d6)
        s_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    # later regions take precedence, so vertices come after edges
    regions = [
        ((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), b + (c - b) * s_bc[:, None]),
        ((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + ac * s_ac[:, None]),
        ((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + ab * s_ab[:, None]),
        ((d6 >= 0) & (d5 <= d6), c),
        ((d3 >= 0) & (d4 <= d3), b),
        ((d1 <= 0) & (d2 <= 0), a),
    ]
    for mask, point in regions:
        q[mask] = point[mask]
    return q


def _unique_edges(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the sorted vertex pairs of all edges and the edge index of each triangle side.

    Side ``k`` of triangle ``i`` joins ``t[i, k]`` and ``t[i, (k + 1) % 3]``.
    Edges are in lexicographic order.
    """
    n = int(t.max()) + 1
    a = t.ravel()
    b = np.roll(t, -1, axis=1).ravel()
    keys = np.minimum(a, b) * n + np.maximum(a, b)
    ukeys, inv = np.unique(keys, return_inverse=True)
    return np.column_stack(np.divmod(ukeys, n)), inv.reshape(-1, 3)


def _sides(t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the edges, the side to edge map and the number of triangles on each edge."""
    edges, tria_edges = _unique_edges(t)
    count = np.bincount(tria_edges.ravel(), minlength=len(edges))
    return edges, tria_edges, count


def _boundary_mask(n: int, edges: np.ndarray, count: np.ndarray) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    mask[edges[count == 1].ravel()] = True
    return mask


def _interior_vertices(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    edges, _, count = _sides(t)
    inside = np.zeros(len(v), dtype=bool)
    inside[t.ravel()] = True
    inside[_boundary_mask(len(v), edges, count)] = False
    return np.flatnonzero(inside)


def _adjacency(n: int, edges: np.ndarray) -> sparse.csr_matrix:
    i = np.concatenate([edges[:, 0], edges[:, 1]])
    j = np.concatenate([edges[:, 1], edges[:, 0]])
    return sparse.csr_matrix((np.ones(len(i), dtype=np.int32), (i, j)), shape=(n, n))


def _unit_normals(p: np.ndarray) -> np.ndarray:
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return n / norm


def _agree(new: np.ndarray, old: np.ndarray, min_cos: float) -> np.ndarray:
    """Accept new triangles that are not degenerate and tilt by at most acos(min_cos) from the old ones.

    Degenerate old triangles accept any non-degenerate replacement.
    """
    valid = np.isfinite(new).all(axis=1)
    old_bad = ~np.isfinite(old).all(axis=1)
    with np.errstate(invalid="ignore"):
        return valid & (old_bad | (np.einsum("ij,ij->i", new, old) >= min_cos))


def _claim(
    n: int, owner: np.ndarray, member: np.ndarray, priority: np.ndarray, check: np.ndarray | None = None
) -> np.ndarray:
    """Pick candidates that have the lowest priority at the vertices they must own.

    ``owner[i]`` is the candidate that touches vertex ``member[i]``, and
    ``check[i]`` says whether the candidate must have the lowest priority
    there (all entries by default). With all entries checked, accepted
    candidates touch disjoint vertex sets. The candidate with the lowest
    priority is always accepted.
    """
    best = np.full(n, np.inf)
    np.minimum.at(best, member, priority[owner])
    beaten = best[member] != priority[owner]
    if check is not None:
        beaten &= check
    lost = np.zeros(len(priority), dtype=bool)
    lost[owner[beaten]] = True
    return ~lost


def _split_round(
    v: np.ndarray, t: np.ndarray, hi: float, split_boundary: bool = True
) -> tuple[np.ndarray, np.ndarray, int]:
    """Split long edges, at most one per triangle, the longest first."""
    edges, tria_edges, count = _sides(t)
    length = np.linalg.norm(v[edges[:, 1]] - v[edges[:, 0]], axis=1)
    side_len = np.where(length[tria_edges] > hi, length[tria_edges], -1.0)
    k = np.argmax(side_len, axis=1)
    longest = side_len[np.arange(len(t)), k] > 0
    if not split_boundary:
        # a triangle whose longest side stays unsplit is left as it is, as splitting its other
        # sides would only cut it into slivers along that side
        longest &= count[tria_edges[np.arange(len(t)), k]] != 1
    rows = np.flatnonzero(longest)
    if len(rows) == 0:
        return v, t, 0
    wanted = tria_edges[rows, k[rows]]
    votes = np.bincount(wanted, minlength=len(edges))
    split = votes == count
    split[votes == 0] = False
    keep = split[wanted]
    rows, k, wanted = rows[keep], k[rows][keep], wanted[keep]
    new_id = np.full(len(edges), -1, dtype=np.int64)
    se = np.flatnonzero(split)
    new_id[se] = len(v) + np.arange(len(se))
    v = np.vstack([v, 0.5 * (v[edges[se, 0]] + v[edges[se, 1]])])
    a = t[rows, k]
    b = t[rows, (k + 1) % 3]
    c = t[rows, (k + 2) % 3]
    m = new_id[wanted]
    t = t.copy()
    t[rows] = np.column_stack([a, m, c])
    t = np.vstack([t, np.column_stack([m, b, c])])
    return v, t, len(se)


def _collapse_round(
    v: np.ndarray, t: np.ndarray, lo: float, hi: float, min_cos: float = 0.5
) -> tuple[np.ndarray, np.ndarray, int]:
    """Collapse short edges to their midpoints where the result stays a valid surface."""
    n = len(v)
    edges, _, count = _sides(t)
    boundary = _boundary_mask(n, edges, count)
    length = np.linalg.norm(v[edges[:, 1]] - v[edges[:, 0]], axis=1)
    cand = np.flatnonzero((length < lo) & (count == 2) & ~boundary[edges].any(axis=1))
    if len(cand) == 0:
        return v, t, 0
    adj = _adjacency(n, edges)
    deg = np.bincount(edges.ravel(), minlength=n)
    a, b = edges[cand, 0], edges[cand, 1]
    # link condition: exactly the two opposite vertices are common neighbors;
    # an edge between two vertices of valence 3 that passes it belongs to a
    # tetrahedron, which would collapse into two copies of one triangle
    keep = np.asarray(adj[a].multiply(adj[b]).sum(axis=1)).ravel() == 2
    keep &= (deg[a] > 3) | (deg[b] > 3)
    cand, a, b = cand[keep], a[keep], b[keep]
    p = 0.5 * (v[a] + v[b])
    # no new edge longer than hi
    ring = (adj[a] + adj[b]).tocoo()
    r, j = ring.row, ring.col
    ok = np.ones(len(cand), dtype=bool)
    far = (j != a[r]) & (j != b[r]) & (np.linalg.norm(p[r] - v[j], axis=1) >= hi)
    ok[r[far]] = False
    # no triangle around a or b tilts by more than acos(min_cos)
    idx = np.flatnonzero(ok)
    vt = sparse.csr_matrix(
        (np.ones(t.size, dtype=np.int32), (t.ravel(), np.repeat(np.arange(len(t)), 3))),
        shape=(n, len(t)),
    )
    fan = (vt[a[idx]] + vt[b[idx]]).tocoo()
    moved = fan.data == 1
    fr, ft = idx[fan.row[moved]], fan.col[moved]
    corners = v[t[ft]]
    at_end = (t[ft] == a[fr, None]) | (t[ft] == b[fr, None])
    corners[at_end] = p[fr]
    ok[fr[~_agree(_unit_normals(corners), _unit_normals(v[t[ft]]), min_cos)]] = False
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return v, t, 0
    # collapses are independent when neither edge has an end point in the other's ring
    keep_ring = ok[r]
    owner = np.searchsorted(idx, r[keep_ring])
    member = j[keep_ring]
    core = (member == a[r[keep_ring]]) | (member == b[r[keep_ring]])
    priority = np.empty(len(idx))
    priority[np.argsort(length[cand[idx]], kind="stable")] = np.arange(len(idx))
    won = _claim(n, owner, member, priority, check=core)
    a, b, p = a[idx[won]], b[idx[won]], p[idx[won]]
    v = v.copy()
    v[a] = p
    remap = np.arange(n)
    remap[b] = a
    t = remap[t]
    degenerate = (t[:, 0] == t[:, 1]) | (t[:, 1] == t[:, 2]) | (t[:, 2] == t[:, 0])
    return v, t[~degenerate], len(a)


def _flip_round(v: np.ndarray, t: np.ndarray, min_cos: float = 0.5) -> tuple[np.ndarray, int]:
    """Flip interior edges where that brings the four vertex valences closer to 6."""
    n = len(v)
    edges, tria_edges, count = _sides(t)
    boundary = _boundary_mask(n, edges, count)
    side = tria_edges.ravel()
    order = np.argsort(side, kind="stable")
    start = np.concatenate([[0], np.cumsum(count)[:-1]])
    e = np.flatnonzero(count == 2)
    s0, s1 = order[start[e]], order[start[e] + 1]
    t0, k0, t1, k1 = s0 // 3, s0 % 3, s1 // 3, s1 % 3
    a, b = t[t0, k0], t[t0, (k0 + 1) % 3]
    c, d = t[t0, (k0 + 2) % 3], t[t1, (k1 + 2) % 3]
    deg = np.bincount(edges.ravel(), minlength=n)
    da, db, dc, dd = deg[a], deg[b], deg[c], deg[d]
    before = np.abs(da - 6) + np.abs(db - 6) + np.abs(dc - 6) + np.abs(dd - 6)
    after = np.abs(da - 7) + np.abs(db - 7) + np.abs(dc - 5) + np.abs(dd - 5)
    gain = before - after
    quad = np.column_stack([a, b, c, d])
    idx = np.flatnonzero((gain > 0) & (da > 3) & (db > 3) & (c != d) & ~boundary[quad].any(axis=1))
    # c and d must not be joined already
    keys = edges[:, 0] * n + edges[:, 1]
    query = np.minimum(c[idx], d[idx]) * n + np.maximum(c[idx], d[idx])
    pos = np.minimum(np.searchsorted(keys, query), len(keys) - 1)
    idx = idx[keys[pos] != query]
    new0 = _unit_normals(v[quad[idx][:, [0, 3, 2]]])
    new1 = _unit_normals(v[quad[idx][:, [3, 1, 2]]])
    old0, old1 = _unit_normals(v[t[t0[idx]]]), _unit_normals(v[t[t1[idx]]])
    ok = np.ones(len(idx), dtype=bool)
    for x in (new0, new1):
        for y in (old0, old1):
            ok &= _agree(x, y, min_cos)
    idx = idx[ok]
    if len(idx) == 0:
        return t, 0
    priority = np.empty(len(idx))
    priority[np.argsort(-gain[idx], kind="stable")] = np.arange(len(idx))
    won = idx[_claim(n, np.repeat(np.arange(len(idx)), 4), quad[idx].ravel(), priority)]
    t = t.copy()
    t[t0[won]] = np.column_stack([a[won], d[won], c[won]])
    t[t1[won]] = np.column_stack([d[won], b[won], c[won]])
    return t, len(won)


def _tangential_smooth(v: np.ndarray, t: np.ndarray, iterations: int = 2, step: float = 0.99) -> np.ndarray:
    """Move each vertex towards the area weighted centroid of its neighbors, in the tangent plane."""
    n = len(v)
    edges, _, count = _sides(t)
    fixed = _boundary_mask(n, edges, count)
    used = np.zeros(n, dtype=bool)
    used[t.ravel()] = True
    fixed |= ~used
    adj = _adjacency(n, edges).astype(float)
    v = v.copy()
    for _ in range(iterations):
        p = v[t]
        cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        area = np.bincount(t.ravel(), np.repeat(np.linalg.norm(cross, axis=1) / 6.0, 3), minlength=n)
        normal = np.zeros((n, 3))
        for k in range(3):
            np.add.at(normal, t[:, k], cross)
        with np.errstate(divide="ignore", invalid="ignore"):
            normal /= np.linalg.norm(normal, axis=1, keepdims=True)
            centroid = (adj @ (area[:, None] * v)) / (adj @ area)[:, None]
        move = centroid - v
        move -= np.einsum("ij,ij->i", move, normal)[:, None] * normal
        move[fixed] = 0.0
        v += step * move
    return v
