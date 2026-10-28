"""Find and remove the handles of closed triangle meshes.

Each handle of a closed surface, one unit of its genus, comes with two non-trivial loops that
cross each other once. Cutting the surface along either loop and closing both openings with
disks removes the handle. Along one loop this cuts a bridge, along the other it fills a tunnel;
which loop is which depends on how the surface sits in space, so the choice is left to the
caller.

Loops are arrays of vertex indices along a closed path of mesh edges, the first vertex not
repeated at the end. They are not boundary loops: those are found with
:meth:`lapy.TriaMesh.boundary_loops`. Queries such as the genus, the Euler characteristic, the
manifold and orientation tests and critical points are methods of :class:`lapy.TriaMesh`.
"""

import dataclasses

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from .intersect import self_intersections
from .remesh import remesh
from .solver import Solver
from .tria_mesh import TriaMesh

# ---------------------------------------------------------------------------
# Edge helpers
# ---------------------------------------------------------------------------


def _unique_edges(t):
    """Undirected edges, sorted, and the edge index of each triangle side.

    Side ``k`` of a triangle runs from corner ``k`` to corner ``(k + 1) % 3``.
    """
    t = np.asarray(t, dtype=np.int64)
    n = int(t.max()) + 1
    a = t.ravel()
    b = np.roll(t, -1, axis=1).ravel()
    keys = np.minimum(a, b) * n + np.maximum(a, b)
    ukeys, inv = np.unique(keys, return_inverse=True)
    return np.column_stack(np.divmod(ukeys, n)), inv.reshape(-1, 3)


def _edge_index(edges, a, b):
    """Index of the edge between ``a`` and ``b`` for each pair, in the order of ``edges``."""
    n = int(edges.max()) + 1
    keys = edges[:, 0] * n + edges[:, 1]
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    query = np.minimum(a, b) * n + np.maximum(a, b)
    idx = np.minimum(np.searchsorted(keys, query), len(keys) - 1)
    if np.any(keys[idx] != query):
        raise ValueError("Vertex pair is not an edge of the mesh.")
    return idx


def _edge_face_pairs(tria_edges, n_edges):
    """Return the two triangles on each edge of a closed, edge-manifold mesh."""
    side = tria_edges.ravel()
    if np.any(np.bincount(side, minlength=n_edges) != 2):
        raise ValueError("Mesh is not closed and edge-manifold: every edge needs exactly two triangles.")
    order = np.argsort(side, kind="stable")
    return (order // 3).reshape(-1, 2)


def _edge_weights(v, edges, vertex_weights=None):
    """Edge lengths, each multiplied by the larger weight of its two vertices."""
    w = np.linalg.norm(v[edges[:, 1]] - v[edges[:, 0]], axis=1)
    if vertex_weights is not None:
        w = w * np.maximum(vertex_weights[edges[:, 0]], vertex_weights[edges[:, 1]])
    return w


def _edge_graph(n_vertices, edges, weights):
    """Symmetric sparse edge graph for shortest paths.

    Zero weights are raised to a tiny positive value, because scipy's graph routines treat a
    stored zero as a missing edge.
    """
    w = np.asarray(weights, dtype=float)
    pos = w[w > 0]
    w = np.maximum(w, 1e-12 * (pos.mean() if pos.size else 1.0))
    i = np.concatenate([edges[:, 0], edges[:, 1]])
    j = np.concatenate([edges[:, 1], edges[:, 0]])
    return sparse.csr_matrix((np.concatenate([w, w]), (i, j)), shape=(n_vertices, n_vertices))


def _directed_sides(t, a, b):
    """Flat corner index ``3 * triangle + k`` of each side running from ``a`` to ``b``."""
    t = np.asarray(t, dtype=np.int64)
    n = int(t.max()) + 1
    keys = t.ravel() * n + np.roll(t, -1, axis=1).ravel()
    order = np.argsort(keys)
    query = np.asarray(a, dtype=np.int64) * n + np.asarray(b, dtype=np.int64)
    pos = np.minimum(np.searchsorted(keys[order], query), len(keys) - 1)
    found = order[pos]
    if np.any(keys[found] != query):
        raise ValueError("Directed side does not exist; is the mesh oriented?")
    return found


def _next_corner(c):
    """Flat index of the next corner in the same triangle."""
    return c - c % 3 + (c + 1) % 3


def _has_side(t, a, b):
    """Whether some triangle contains the directed side a -> b."""
    try:
        _directed_sides(t, [a], [b])
    except ValueError:
        return False
    return True


def _loop_length(v, loop):
    """Euclidean length of a closed vertex loop."""
    p = v[loop]
    return float(np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1).sum())


def _weighted_length(edges, w, loop):
    return float(w[_edge_index(edges, loop, np.roll(loop, -1))].sum())


def _check_closed_surface(tria):
    """Raise unless the mesh is a closed, connected, oriented manifold."""
    if not (tria.is_closed() and tria.is_manifold() and tria.is_oriented()):
        raise ValueError("Need a closed, oriented manifold mesh.")
    _, labels = tria.connected_components()
    if len(np.unique(labels[np.unique(tria.t)])) != 1:
        raise ValueError("Need a connected mesh.")


def _cut_along_loop(v, t, loop):
    """Cut an oriented mesh along a simple closed loop with :meth:`TriaMesh.cut_`.

    Returns the cut mesh as arrays, the original index of each appended copy, and the left and
    right copy of each loop vertex. The left side holds the triangles with the directed sides
    ``loop[i] -> loop[i + 1]``.
    """
    loop = np.asarray(loop, dtype=np.int64)
    if len(np.unique(loop)) != len(loop):
        raise ValueError("Loop is not simple.")
    nxt = np.roll(loop, -1)
    mesh = TriaMesh(v, t)
    source = mesh.cut_(np.column_stack([loop, nxt]))
    # cut_ keeps the order of the triangles, so sides of the input locate the copies
    flat = mesh.t.ravel()
    left = flat[_directed_sides(t, loop, nxt)]
    right = flat[_next_corner(_directed_sides(t, nxt, loop))]
    return mesh.v, mesh.t, source, left, right


# ---------------------------------------------------------------------------
# Loops
# ---------------------------------------------------------------------------


def _tree_path(pred, i):
    """Vertices from ``i`` up the shortest-path tree to the root."""
    path = [i]
    while pred[i] >= 0:
        i = pred[i]
        path.append(i)
    return path


def _tree_loop(pred, u, v):
    """Close the tree paths to ``u`` and ``v`` with the edge (u, v), dropping their common stem."""
    pu = _tree_path(pred, u)
    pv = _tree_path(pred, v)
    k = 0
    while k < min(len(pu), len(pv)) and pu[-1 - k] == pv[-1 - k]:
        k += 1
    # pu[len(pu) - k] is the lowest common ancestor
    return np.array(pu[: len(pu) - k + 1] + pv[: len(pv) - k][::-1], dtype=np.int64)


def _tree_cotree_loops(v, t, root, w):
    edges, tria_edges = _unique_edges(t)
    faces = _edge_face_pairs(tria_edges, len(edges))
    graph = _edge_graph(len(v), edges, w)

    dist, pred = csgraph.dijkstra(graph, indices=root, return_predecessors=True)
    if np.any(np.isinf(dist[np.unique(edges)])):
        raise ValueError("Mesh is not connected.")
    child = np.flatnonzero(pred >= 0)
    in_tree = np.zeros(len(edges), dtype=bool)
    in_tree[_edge_index(edges, child, pred[child])] = True

    nontree = np.flatnonzero(~in_tree)
    closing = dist[edges[nontree, 0]] + dist[edges[nontree, 1]] + w[nontree]
    # maximum spanning tree as a minimum spanning tree of reversed, positive weights
    reversed_w = closing.max() - closing + 1.0
    fa = faces[nontree].min(axis=1)
    fb = faces[nontree].max(axis=1)
    nt = len(t)
    dual = sparse.csr_matrix((reversed_w, (fa, fb)), shape=(nt, nt))
    cotree = csgraph.minimum_spanning_tree(dual).tocoo()

    pair_keys = fa * nt + fb
    order = np.argsort(pair_keys)
    pos = np.searchsorted(pair_keys[order], cotree.row.astype(np.int64) * nt + cotree.col)
    in_cotree = np.zeros(len(nontree), dtype=bool)
    in_cotree[order[pos]] = True

    loops = [_tree_loop(pred, *edges[e]) for e in nontree[~in_cotree]]
    lengths = np.array([_weighted_length(edges, w, lp) for lp in loops])
    order = np.argsort(lengths)
    return [loops[i] for i in order], lengths[order]


def _root_vertex(t, root):
    """Return the given root vertex, or the lowest vertex index used by a triangle."""
    if root is None:
        return int(np.min(t))
    if not np.any(t == root):
        raise ValueError("Root vertex is not used by any triangle.")
    return int(root)


def tree_cotree_loops(
    tria: TriaMesh, root: int | None = None, vertex_weights: np.ndarray | None = None
) -> tuple[list, np.ndarray]:
    """Compute a greedy system of non-trivial loops through a root vertex.

    A shortest-path tree from the root and a maximum spanning tree of the dual graph, weighted by
    the length of the loop that each dual edge closes, leave 2g edges that belong to neither tree
    (Erickson and Whittlesey 2005). Each of them closes one loop. Cutting the mesh along all of
    them leaves a single disk, so each loop is non-separating and together they span the first
    homology. The loops are returned without the stem they share with the root, so a loop passes
    through the root only when it has to.

    Parameters
    ----------
    tria : TriaMesh
        Closed, connected, edge-manifold triangle mesh of genus g.
    root : int or None, default=None
        Root vertex of the shortest-path tree; by default the lowest vertex index used by a
        triangle.
    vertex_weights : np.ndarray or None, default=None
        Cost multiplier per vertex, shape (n_vertices,). An edge costs its length times the larger
        weight of its two vertices, so large weights keep loops away from a vertex.

    Returns
    -------
    loops : list of np.ndarray
        The 2g loops, each as vertex indices, sorted by length.
    lengths : np.ndarray
        Weighted length of each loop.

    Raises
    ------
    ValueError
        If the mesh is not closed, edge-manifold and connected, or the root is not used by a
        triangle.
    """
    v = np.asarray(tria.v, dtype=float)
    t = np.asarray(tria.t, dtype=np.int64)
    edges, _ = _unique_edges(t)
    return _tree_cotree_loops(v, t, _root_vertex(t, root), _edge_weights(v, edges, vertex_weights))


def _shortest_crossing_loop(v, t, loop, vertex_weights=None, graph=None, radius=None):
    loop = np.asarray(loop, dtype=np.int64)
    if graph is None:
        edges, _ = _unique_edges(t)
        graph = _edge_graph(len(v), edges, _edge_weights(v, edges, vertex_weights))
    if radius is None:
        radius = 8.0 * np.median(graph.data)
    elif not radius > 0:
        raise ValueError("Search radius must be positive.")
    while True:
        dist = csgraph.dijkstra(graph, indices=loop, min_only=True, limit=radius)
        # triangles with any vertex in reach keep every edge between two vertices in reach,
        # which all edges of a crossing loop no longer than twice the radius are
        keep = np.any(dist[t] <= radius, axis=1)
        whole = bool(np.all(keep))
        used = np.unique(t[keep])
        remap = np.full(len(v), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        weights = None if vertex_weights is None else vertex_weights[used]
        try:
            crossing, length = _crossing_loop_exact(v[used], remap[t[keep]], remap[loop], weights)
        except RuntimeError:
            if whole:
                raise
        else:
            if whole or length <= 2.0 * radius:
                return used[crossing], length
        radius *= 2.0


def _crossing_loop_exact(v, t, loop, vertex_weights):
    """Shortest crossing loop on the whole given mesh, see :func:`shortest_crossing_loop`."""
    vc, tc, source, left, right = _cut_along_loop(v, t, loop)
    origin = np.concatenate([np.arange(len(v)), source])
    weights = None if vertex_weights is None else vertex_weights[origin]
    edges, _ = _unique_edges(tc)
    graph = _edge_graph(len(vc), edges, _edge_weights(vc, edges, weights))

    # upper bound: nearest right copy from any left copy, closed along the right boundary
    dist, _, sources = csgraph.dijkstra(graph, indices=left, min_only=True, return_predecessors=True)
    j = int(np.argmin(dist[right]))
    if not np.isfinite(dist[right[j]]):
        raise RuntimeError("No crossing loop found; is the loop non-separating?")
    i = int(np.flatnonzero(left == sources[right[j]])[0])
    seg = _edge_weights(vc, np.column_stack([right, np.roll(right, -1)]), weights)
    arc = seg[min(i, j) : max(i, j)].sum()
    best = dist[right[j]] + min(arc, seg.sum() - arc)

    # the bound is a path length summed in a different order, so allow for rounding
    limit = best * (1 + 1e-9)
    best_path = None
    for p in range(len(loop)):
        d, pred = csgraph.dijkstra(graph, indices=left[p], limit=limit, return_predecessors=True)
        if d[right[p]] <= limit:
            best = limit = d[right[p]]
            best_path = _tree_path(pred, right[p])
    if best_path is None:
        raise RuntimeError("No crossing loop found; is the loop non-separating?")
    # the path runs from the right copy back to the left copy of the same vertex
    crossing = origin[np.array(best_path[:-1][::-1], dtype=np.int64)]
    return crossing, float(best)


def shortest_crossing_loop(
    tria: TriaMesh,
    loop: np.ndarray,
    vertex_weights: np.ndarray | None = None,
    radius: float | None = None,
) -> tuple[np.ndarray, float]:
    """Find the shortest closed loop that crosses a given loop exactly once.

    The mesh is cut along ``loop``; a path in the cut mesh from one copy of a loop vertex to the
    other copy closes into a loop that crosses ``loop`` once. The shortest such path over all loop
    vertices is returned. If ``loop`` is non-separating, so is the result.

    The search runs on the part of the mesh within ``radius`` of ``loop``. A crossing loop of
    length L stays within L / 2 of ``loop``, so a result no longer than twice the radius is the
    global optimum; otherwise the radius is doubled and the search repeated.

    Parameters
    ----------
    tria : TriaMesh
        Closed, oriented, edge-manifold triangle mesh.
    loop : np.ndarray
        Vertex indices of a simple closed loop.
    vertex_weights : np.ndarray or None, default=None
        Cost multiplier per vertex, see :func:`tree_cotree_loops`.
    radius : float or None, default=None
        Initial search radius. Defaults to eight times the median edge cost.

    Returns
    -------
    crossing : np.ndarray
        Vertex indices of the crossing loop.
    length : float
        Its weighted length.

    Raises
    ------
    RuntimeError
        If no loop crosses ``loop`` once, which is the case when ``loop`` separates the mesh.
    """
    v = np.asarray(tria.v, dtype=float)
    t = np.asarray(tria.t, dtype=np.int64)
    return _shortest_crossing_loop(v, t, loop, vertex_weights, radius=radius)


def _refine_loop_pair(v, t, loop, vertex_weights=None, max_rounds=10):
    edges, _ = _unique_edges(t)
    w = _edge_weights(v, edges, vertex_weights)
    graph = _edge_graph(len(v), edges, w)
    a = np.asarray(loop, dtype=np.int64)
    la = _weighted_length(edges, w, a)
    b, lb = _shortest_crossing_loop(v, t, a, vertex_weights, graph)
    for _ in range(max_rounds):
        a2, la2 = _shortest_crossing_loop(v, t, b, vertex_weights, graph)
        b2, lb2 = _shortest_crossing_loop(v, t, a2, vertex_weights, graph)
        if la2 + lb2 >= la + lb - 1e-9:
            break
        a, la, b, lb = a2, la2, b2, lb2
    return a, b, la, lb


def refine_loop_pair(
    tria: TriaMesh,
    loop: np.ndarray,
    vertex_weights: np.ndarray | None = None,
    max_rounds: int = 10,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Shorten a non-separating loop and its crossing loop by alternating crossing searches.

    Starting from ``loop``, compute the shortest loop crossing it once, then the shortest loop
    crossing that one, and so on, as long as the summed length of the pair decreases.

    Parameters
    ----------
    tria : TriaMesh
        Closed, oriented, edge-manifold triangle mesh.
    loop : np.ndarray
        Vertex indices of a simple, non-separating closed loop.
    vertex_weights : np.ndarray or None, default=None
        Cost multiplier per vertex, see :func:`tree_cotree_loops`.
    max_rounds : int, default=10
        Maximum number of alternations.

    Returns
    -------
    a, b : np.ndarray
        The refined pair; each crosses the other once.
    length_a, length_b : float
        Their weighted lengths.
    """
    v = np.asarray(tria.v, dtype=float)
    t = np.asarray(tria.t, dtype=np.int64)
    return _refine_loop_pair(v, t, loop, vertex_weights, max_rounds)


# ---------------------------------------------------------------------------
# Surgery
# ---------------------------------------------------------------------------


def _fan_surgery(v, t, loop):
    """Cut along a loop and close both openings with a triangle fan around a centre vertex."""
    vc, tc, source, left, right = _cut_along_loop(v, t, loop)
    n = len(vc)
    centre = v[loop].mean(axis=0)
    caps = np.concatenate(
        [
            np.column_stack([np.roll(left, -1), left, np.full(len(left), n)]),
            np.column_stack([right, np.roll(right, -1), np.full(len(right), n + 1)]),
        ]
    )
    return np.vstack([vc, centre, centre]), np.vstack([tc, caps]), source


def _min_area_triangulation(points, allow_degenerate=False):
    """Triangulate a closed polygon without new vertices, with the least total area.

    Dynamic programming over all triangulations, as in the first step of Liepa (2003). For a
    planar polygon, convex or not, every triangle outside it adds area, so the result lies
    inside it. Triangles without area, with three points on a line, are left out unless the
    polygon has no triangulation without them. Only among triangulations of the same area, up
    to rounding, nearly degenerate triangles are avoided and short diagonals preferred.
    Triangle ``(i, m, j)`` contains the directed sides ``i -> m``, ``m -> j`` and ``j -> i``.
    """
    k = len(points)
    perimeter = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1).sum()
    tol = 1e-9 * perimeter**2
    cost = np.zeros((k, k))
    tie = np.zeros((k, k))
    best = np.zeros((k, k), dtype=np.int64)
    for d in range(2, k):
        i = np.arange(k - d)
        j = i + d
        m = i[:, None] + np.arange(1, d)[None, :]
        a, b, c = points[i][:, None], points[m], points[j][:, None]
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=2)
        squares = np.sum((b - a) ** 2, axis=2) + np.sum((c - b) ** 2, axis=2) + np.sum((c - a) ** 2, axis=2)
        total = cost[i[:, None], m] + cost[m, j[:, None]] + area
        if not allow_degenerate:
            total = np.where(area > 1e-12 * squares, total, np.inf)
        quality = tie[i[:, None], m] + tie[m, j[:, None]] + squares * (1.0 + 1e3 * (area < 1e-3 * squares))
        # least area first; the quality term only decides between equal areas
        quality = np.where(np.isfinite(total) & (total <= total.min(axis=1, keepdims=True) + tol), quality, np.inf)
        pick = np.argmin(quality, axis=1)
        rows = np.arange(len(i))
        cost[i, j] = total[rows, pick]
        tie[i, j] = quality[rows, pick]
        best[i, j] = m[rows, pick]
    if k > 2 and not np.isfinite(cost[0, k - 1]):
        return _min_area_triangulation(points, allow_degenerate=True)
    tris = []
    stack = [(0, k - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        m = best[i, j]
        tris.append((i, m, j))
        stack += [(i, m), (m, j)]
    return np.array(tris, dtype=np.int64)


def _split_chords(v, t, k):
    """Split each edge between two boundary vertices that is not a boundary edge at its midpoint.

    The first ``k`` vertices form the boundary. Without such chords, no edge of the disk can
    duplicate an edge of the mesh whose hole it closes.
    """
    v, t = v.copy(), t.copy()
    while True:
        edges, tria_edges = _unique_edges(t)
        count = np.bincount(tria_edges.ravel(), minlength=len(edges))
        chords = np.flatnonzero((edges[:, 1] < k) & (count == 2))
        if len(chords) == 0:
            return v, t
        a, b = edges[chords[0]]
        m = len(v)
        v = np.vstack([v, 0.5 * (v[a] + v[b])])
        rows, sides = np.nonzero(tria_edges == chords[0])
        new = []
        for r, s in zip(rows, sides, strict=True):
            p, q, o = t[r, s], t[r, (s + 1) % 3], t[r, (s + 2) % 3]
            t[r] = (p, m, o)
            new.append((m, q, o))
        t = np.vstack([t, new])


def _n_folded_edges(v, t):
    """Count the interior edges whose two triangles face in opposite directions."""
    edges, tria_edges = _unique_edges(t)
    side = tria_edges.ravel()
    count = np.bincount(side, minlength=len(edges))
    order = np.argsort(side, kind="stable")
    start = np.concatenate([[0], np.cumsum(count)[:-1]])
    inner = np.flatnonzero(count == 2)
    fa, fb = order[start[inner]] // 3, order[start[inner] + 1] // 3
    p = v[t]
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    return int(np.count_nonzero(np.einsum("ij,ij->i", normal[fa], normal[fb]) < 0))


def _minimal_surface_step(v, t, k):
    """Move the inner vertices to the harmonic positions for the current cotangent weights.

    One iteration of Pinkall and Polthier (1993); repeated, it converges to a discrete minimal
    surface. The first ``k`` vertices form the fixed boundary. A flat mesh stays as it is.
    """
    from scipy.sparse.linalg import splu

    if len(v) == k:
        return v
    a = Solver(TriaMesh(v, t), lump=False).stiffness.tocsr()
    inner = a[k:]
    v = v.copy()
    v[k:] = splu(inner[:, k:].tocsc()).solve(-(inner[:, :k] @ v[:k]))
    return v


def disk_cap(points: np.ndarray, edge_length: float, iterations: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a disk that spans a closed boundary polygon, close to a minimal surface.

    The polygon is first triangulated with the least area and no new vertices (Liepa 2003),
    which is free of folds for a planar polygon, convex or not, and edges between two boundary
    vertices are split, so that the disk can close a hole without duplicating an edge of the
    mesh around it. Then, ``iterations`` times, the disk is remeshed to the target edge length
    with the boundary kept as it is, see :func:`lapy.remesh.remesh`, and its inner vertices are
    moved to the harmonic positions for the current cotangent weights (Pinkall and Polthier
    1993), which pulls the disk towards a minimal surface. A round that would fold the disk
    more, as smoothing can in thin concave parts, is not applied, and the disk stays as it was.

    Parameters
    ----------
    points : np.ndarray
        Boundary polygon of shape (k, 3), in order.
    edge_length : float
        Target edge length inside the disk.
    iterations : int, default=3
        Rounds of remeshing and minimal surface steps.

    Returns
    -------
    inner : np.ndarray
        Positions of the new inner vertices, shape (m, 3).
    tris : np.ndarray
        Triangles of shape (n, 3); indices below k refer to the boundary points in order, indices
        from k on to the inner vertices. The triangles contain the directed sides
        ``points[i] -> points[i + 1]``; reverse them to close a hole whose triangles contain
        these sides.
    """
    points = np.asarray(points, dtype=float)
    k = len(points)
    v, t = _split_chords(points, _min_area_triangulation(points), k)
    folds = _n_folded_edges(v, t)
    for _ in range(iterations):
        mesh = remesh(TriaMesh(v, t), target_length=edge_length, iterations=3, split_boundary=False)
        # the boundary keeps its vertices, and as the lowest indices also their numbers
        v_new = _minimal_surface_step(np.asarray(mesh.v, dtype=float), np.asarray(mesh.t, dtype=np.int64), k)
        folds_new = _n_folded_edges(v_new, mesh.t)
        # in a thin concave part, smoothing can push vertices across the boundary; keep the
        # last disk that did not fold more
        if folds_new > folds:
            break
        v, t, folds = v_new, np.asarray(mesh.t, dtype=np.int64), folds_new
    return v[k:], t


def _nudge(v, t, copies, gap):
    """Move each copy of a cut loop by ``gap`` towards the mean of its neighbours off the loop.

    A copy without such neighbours moves towards the mean centroid of its triangles, which all
    lie on its own side of the cut. Moves that would flip a triangle are shortened.
    """
    n = len(v)
    edges, _ = _unique_edges(t)
    rows = np.r_[edges[:, 0], edges[:, 1]]
    cols = np.r_[edges[:, 1], edges[:, 0]]
    adj = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    off = np.ones(n)
    off[copies] = 0.0
    nb = adj[copies] @ (v * off[:, None])
    cnt = adj[copies] @ off
    ok = cnt > 0
    d = np.zeros((len(copies), 3))
    d[ok] = nb[ok] / cnt[ok, None] - v[copies[ok]]
    if not np.all(ok):
        incident = sparse.csr_matrix(
            (np.ones(t.size), (t.ravel(), np.repeat(np.arange(len(t)), 3))), shape=(n, len(t))
        )[copies[~ok]]
        centroid = (incident @ v[t].mean(axis=1)) / np.asarray(incident.sum(axis=1))
        d[~ok] = centroid - v[copies[~ok]]
    dist = np.linalg.norm(d, axis=1)
    step = np.minimum(gap, 0.5 * dist)
    with np.errstate(invalid="ignore", divide="ignore"):
        move = np.where(dist[:, None] > 0, d * (step / dist)[:, None], 0.0)
    # halve the moves of copies whose triangles would flip, until none does
    near = t[np.any(np.isin(t, copies), axis=1)]
    p = v[near]
    before = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    scale = np.ones(len(copies))
    for attempt in range(21):
        moved = v.copy()
        moved[copies] += scale[:, None] * move
        p = moved[near]
        flipped = np.einsum("ij,ij->i", before, np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])) < 0
        if not flipped.any():
            return moved
        culprit = np.isin(copies, near[flipped])
        scale[culprit] = 0.0 if attempt == 19 else 0.5 * scale[culprit]
    return moved


def _capped_surgery(v, t, loop, gap=0.2):
    loop = np.asarray(loop, dtype=np.int64)
    vc, tc, source, left, right = _cut_along_loop(v, t, loop)
    h = _loop_length(v, loop) / len(loop)
    sides = np.concatenate([left, right])
    vc = _nudge(vc, tc, sides, gap * h)
    new_v, new_t = [vc], [tc]
    n = len(vc)
    for side in (left, right):
        inner, tris = disk_cap(vc[side], h)
        cap = np.concatenate([side, n + np.arange(len(inner))])[tris]
        # the cap must run against the mesh along the shared boundary edges
        if _has_side(cap, side[0], side[1]) == _has_side(tc, side[0], side[1]):
            cap = cap[:, ::-1]
        new_v.append(inner)
        new_t.append(cap)
        n += len(inner)
    return np.vstack(new_v), np.vstack(new_t), source


def capped_surgery(tria: TriaMesh, loop: np.ndarray, gap: float = 0.2) -> tuple[TriaMesh, np.ndarray]:
    """Cut a closed mesh along a simple loop and close both openings with disks.

    For a non-separating loop this lowers the genus by one. The two copies of the loop are first
    moved apart, each by ``gap`` times the mean loop edge length towards its own side, so that
    the disks do not touch. The disks are close to minimal surfaces, with edges about as long as
    those of the loop, see :func:`disk_cap`. Vertices away from the loop keep their index and
    position.

    Parameters
    ----------
    tria : TriaMesh
        Closed, oriented, edge-manifold triangle mesh.
    loop : np.ndarray
        Vertex indices of a simple closed loop.
    gap : float, default=0.2
        Separation of each side from the loop, relative to the mean loop edge length.

    Returns
    -------
    TriaMesh
        The new mesh: the input vertices, the cut copies, then the inner disk vertices.
    source : np.ndarray
        Input vertex index of each cut copy, in the order they were appended.
    """
    v = np.asarray(tria.v, dtype=float)
    t = np.asarray(tria.t, dtype=np.int64)
    nv, nt, source = _capped_surgery(v, t, loop, gap)
    return TriaMesh(nv, nt), source


# ---------------------------------------------------------------------------
# Handles
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Handle:
    """A handle given by two loops that cross each other once.

    Capped surgery along either loop removes the handle. After earlier handles have been
    removed, a loop can run along both sides of an earlier cut and then visits a vertex twice.

    Parameters
    ----------
    loop_a, loop_b : np.ndarray
        The two loops, as vertex indices of the input mesh.
    length_a, length_b : float
        Their lengths.
    """

    loop_a: np.ndarray
    loop_b: np.ndarray
    length_a: float
    length_b: float


def find_handles(
    tria: TriaMesh, root: int | None = None, max_rounds: int = 10, cap_weight: float = 1e3
) -> list:
    """Find one pair of short crossing loops per handle.

    Repeatedly takes the shortest tree-cotree loop, shortens it together with its crossing loop
    (:func:`refine_loop_pair`), records the pair, and cuts the surface along the shorter loop of
    the pair, closing both openings, which lowers the genus by one. Edges at the centres of these
    closures cost ``cap_weight`` times their length, so later loops run around them on the
    surface.

    Parameters
    ----------
    tria : TriaMesh
        Closed, connected, oriented manifold triangle mesh.
    root : int or None, default=None
        Root vertex for the tree-cotree loops, see :func:`tree_cotree_loops`.
    max_rounds : int, default=10
        Maximum number of refinement rounds per handle.
    cap_weight : float, default=1e3
        Cost multiplier at the centres of the closures.

    Returns
    -------
    list of Handle
        One entry per handle, in the order they were found.

    Raises
    ------
    ValueError
        If the mesh is not a closed, connected, oriented manifold, or the root is not used by a
        triangle.
    RuntimeError
        If a loop runs through the centre of an earlier closure.
    """
    _check_closed_surface(tria)
    v0 = np.asarray(tria.v, dtype=float)
    cv, ct = v0, np.asarray(tria.t, dtype=np.int64)
    root = _root_vertex(ct, root)
    origin = np.arange(len(v0))
    weights = np.ones(len(v0))
    handles = []
    for _ in range(tria.genus()):
        edges, _ = _unique_edges(ct)
        loops, _ = _tree_cotree_loops(cv, ct, root, _edge_weights(cv, edges, weights))
        a, b, la, lb = _refine_loop_pair(cv, ct, loops[0], weights, max_rounds)
        oa, ob = origin[a], origin[b]
        if np.any(oa < 0) or np.any(ob < 0):
            raise RuntimeError("A loop runs through the centre of an earlier closure.")
        handles.append(Handle(oa, ob, _loop_length(v0, oa), _loop_length(v0, ob)))
        cv, ct, source = _fan_surgery(cv, ct, a if la <= lb else b)
        origin = np.concatenate([origin, origin[source], [-1, -1]])
        weights = np.concatenate([weights, weights[source], [cap_weight, cap_weight]])
    return handles


def handle_patches(tria: TriaMesh, handles: list, radius: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    """Group the vertices near each handle into patches.

    A handle's patch holds the vertices within distance ``radius``, along mesh edges, of either of
    its loops. Patches that share a vertex are merged.

    Parameters
    ----------
    tria : TriaMesh
        Triangle mesh.
    handles : list of Handle
        Handles as returned by :func:`find_handles`.
    radius : float, default=1.5
        Distance along mesh edges.

    Returns
    -------
    labels : np.ndarray
        Patch number per vertex, starting at 1; 0 outside all patches.
    handle_patch : np.ndarray
        Patch number of each handle.
    """
    v = np.asarray(tria.v, dtype=float)
    n = len(v)
    k = len(handles)
    if k == 0:
        return np.zeros(n, dtype=np.int64), np.zeros(0, dtype=np.int64)
    edges, _ = _unique_edges(tria.t)
    graph = _edge_graph(n, edges, _edge_weights(v, edges))
    near = []
    for h in handles:
        seeds = np.unique(np.concatenate([h.loop_a, h.loop_b]))
        dist = csgraph.dijkstra(graph, indices=seeds, min_only=True, limit=radius)
        near.append(np.flatnonzero(np.isfinite(dist)))
    rows = np.concatenate([np.full(len(p), i) for i, p in enumerate(near)])
    cols = np.concatenate(near)
    member = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(k, n))
    _, group = csgraph.connected_components(member @ member.T, directed=False)
    # number patches by the order in which their first handle was found
    _, first = np.unique(group, return_index=True)
    rank = np.empty(len(first), dtype=np.int64)
    rank[np.argsort(first)] = np.arange(1, len(first) + 1)
    handle_patch = rank[group]
    labels = np.zeros(n, dtype=np.int64)
    labels[cols] = handle_patch[rows]
    return labels, handle_patch


# ---------------------------------------------------------------------------
# Topology correction
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Surgery:
    """One handle removed by cutting along a loop and closing both openings.

    Parameters
    ----------
    loop : np.ndarray
        The loop that was used, as vertex indices of the input mesh.
    other : np.ndarray
        The other loop of the pair.
    length, other_length : float
        Their lengths.
    operation : str
        "cut" when the loop goes around a bridge and "fill" when it goes around a tunnel, as
        reported by the classifier, or "unknown" without one.
    n_intersections : int, default=0
        Triangle pairs that the surgery left intersecting; zero unless no candidate was free of
        intersections.
    evidence : tuple or None, default=None
        Classifier results for the used and the other loop.
    """

    loop: np.ndarray
    other: np.ndarray
    length: float
    other_length: float
    operation: str
    n_intersections: int = 0
    evidence: tuple | None = None


def shorter_loop(a, b, length_a, length_b, scores=None, v=None) -> int:
    """Choose the shorter loop of a pair, the default rule of :func:`fix_topology`.

    Parameters
    ----------
    a, b : np.ndarray
        The two loops of a handle.
    length_a, length_b : float
        Their lengths.
    scores : tuple or None, default=None
        Classifier results for both loops; not used.
    v : np.ndarray or None, default=None
        Current vertex positions; not used.

    Returns
    -------
    int
        0 for ``a``, 1 for ``b``.
    """
    return 0 if length_a <= length_b else 1


def _cuts(score):
    return getattr(score, "cuts", score)


def _new_intersections(n_before, loop, v, t):
    """Count intersections of the triangles that a surgery created or moved."""
    changed = np.zeros(len(v), dtype=bool)
    changed[n_before:] = True
    changed[loop] = True
    return len(self_intersections(TriaMesh(v, t), subset=np.flatnonzero(np.any(changed[t], axis=1))))


def fix_topology(
    tria: TriaMesh,
    choose=shorter_loop,
    classify=None,
    max_rounds: int = 10,
    gap: float = 0.2,
) -> tuple[TriaMesh, np.ndarray, list]:
    """Remove all handles of a closed surface by capped surgery, one handle at a time.

    Each round finds the shortest tree-cotree loop, refines it with its crossing loop, asks
    ``choose`` which of the two loops to use, and performs :func:`capped_surgery` along it. If the
    new disks intersect the surface, other gaps and then the other loop are tried, and the first
    result without intersections is kept, or else the one with the fewest. Later handles are
    found on the already corrected mesh.

    Parameters
    ----------
    tria : TriaMesh
        Closed, connected, oriented manifold triangle mesh.
    choose : callable, default=shorter_loop
        ``choose(a, b, length_a, length_b, scores, v=...)`` returns 0 or 1 for the loop to cut
        along; ``v`` holds the current vertex positions.
    classify : callable or None, default=None
        ``classify(v, t, loop)`` returns True if surgery along the loop cuts a bridge rather than
        filling a tunnel, or an object with such a ``cuts`` attribute. Its results are passed to
        ``choose`` as ``scores`` and recorded in the result.
    max_rounds : int, default=10
        Maximum refinement rounds per handle, see :func:`refine_loop_pair`.
    gap : float, default=0.2
        Separation of the cut sides, see :func:`capped_surgery`.

    Returns
    -------
    TriaMesh
        The corrected mesh of genus 0.
    origin : np.ndarray
        Input vertex index of each output vertex, -1 for new vertices inside the disks.
    surgeries : list of Surgery
        One entry per removed handle.

    Raises
    ------
    ValueError
        If the mesh is not a closed, connected, oriented manifold.
    """
    _check_closed_surface(tria)
    cv = np.asarray(tria.v, dtype=float)
    ct = np.asarray(tria.t, dtype=np.int64)
    origin = np.arange(len(cv))
    surgeries = []
    for _ in range(tria.genus()):
        edges, _ = _unique_edges(ct)
        loops, _ = _tree_cotree_loops(cv, ct, _root_vertex(ct, None), _edge_weights(cv, edges))
        a, b, la, lb = _refine_loop_pair(cv, ct, loops[0], max_rounds=max_rounds)
        pair, lengths = (a, b), (la, lb)
        scores = None if classify is None else (classify(cv, ct, a), classify(cv, ct, b))
        k = choose(a, b, la, lb, scores, v=cv)
        # preferred loop first, then other gaps, then the other loop; keep the first clean result
        best = None
        for kk, g in [(k, gap), (k, gap / 2), (k, 2 * gap), (1 - k, gap), (1 - k, gap / 2), (1 - k, 2 * gap)]:
            result = _capped_surgery(cv, ct, pair[kk], gap=g)
            n_x = _new_intersections(len(cv), pair[kk], *result[:2])
            if best is None or n_x < best[0]:
                best = (n_x, kk, result)
            if n_x == 0:
                break
        n_x, kk, (nv, nt, source) = best
        operation = "unknown" if scores is None else ("cut" if _cuts(scores[kk]) else "fill")
        evidence = None if scores is None else (scores[kk], scores[1 - kk])
        surgeries.append(
            Surgery(origin[pair[kk]], origin[pair[1 - kk]], lengths[kk], lengths[1 - kk], operation, n_x, evidence)
        )
        n_new = len(nv) - len(cv) - len(source)
        origin = np.concatenate([origin, origin[source], np.full(n_new, -1)])
        cv, ct = nv, nt
    return TriaMesh(cv, ct), origin, surgeries


# ---------------------------------------------------------------------------
# Essential saddles
# ---------------------------------------------------------------------------


def _merges(order, adj):
    """Union-find sweep over vertices in ``order``; number of component merges at each vertex."""
    parent = np.arange(len(order))
    done = np.zeros(len(order), dtype=bool)
    merges = np.zeros(len(order), dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for vtx in order:
        roots = {find(u) for u in adj.indices[adj.indptr[vtx] : adj.indptr[vtx + 1]] if done[u]}
        for r in roots:
            parent[r] = vtx
        merges[vtx] = max(len(roots) - 1, 0)
        done[vtx] = True
    return merges


def essential_saddles(tria: TriaMesh, f: np.ndarray) -> np.ndarray:
    """Find the saddles of a function that no cancellation with a minimum or maximum can remove.

    Ties in ``f`` are broken by vertex index. A saddle of multiplicity m has 2(m + 1) sign changes
    around its link. Each of its m units either merges two sublevel components and cancels with
    a minimum, merges two superlevel components and cancels with a maximum, or is essential. A
    closed surface of genus g has exactly 2g essential saddles, two per handle.

    Parameters
    ----------
    tria : TriaMesh
        Closed manifold triangle mesh.
    f : np.ndarray
        Function value per vertex, shape (n_vertices,).

    Returns
    -------
    np.ndarray
        Number of essential saddle units per vertex, shape (n_vertices,).

    Raises
    ------
    ValueError
        If the mesh is not closed or not manifold. At a pinch vertex the link is not a single
        circle, so saddles cannot be counted from its sign changes.
    """
    if not (tria.is_closed() and tria.is_manifold()):
        raise ValueError("Need a closed manifold mesh.")
    t = np.asarray(tria.t, dtype=np.int64)
    n = len(f)
    order = np.argsort(f, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    changes = np.zeros(n, dtype=np.int64)
    for k in range(3):
        c, a, b = t[:, k], t[:, (k + 1) % 3], t[:, (k + 2) % 3]
        changes += np.bincount(c, weights=(rank[a] > rank[c]) != (rank[b] > rank[c]), minlength=n).astype(np.int64)
    multiplicity = np.maximum(changes // 2 - 1, 0)
    adj = tria.adj_sym.tocsr()
    return multiplicity - _merges(order, adj) - _merges(order[::-1], adj)


def handle_clusters(tria: TriaMesh, handles: list, saddles: np.ndarray, rings: int = 2) -> tuple[np.ndarray, list]:
    """Group handles whose loops pass close to the same essential saddle.

    Parameters
    ----------
    tria : TriaMesh
        Triangle mesh.
    handles : list of Handle
        Handles as returned by :func:`find_handles`.
    saddles : np.ndarray
        Essential saddle count per vertex, as returned by :func:`essential_saddles`.
    rings : int, default=2
        A saddle belongs to a handle if it lies within this many edge rings of one of its loops.

    Returns
    -------
    cluster : np.ndarray
        Cluster index per handle, numbered from 0.
    saddle_vertices : list of np.ndarray
        Essential saddle vertices near each handle.
    """
    n = len(saddles)
    adj = tria.adj_sym.tocsr()
    saddle_ids = np.flatnonzero(saddles > 0)
    near = []
    for h in handles:
        mask = np.zeros(n, dtype=bool)
        mask[np.concatenate([h.loop_a, h.loop_b])] = True
        for _ in range(rings):
            mask |= adj @ mask.astype(float) > 0
        near.append(saddle_ids[mask[saddle_ids]])
    empty = np.empty(0, dtype=np.int64)
    rows = np.concatenate([np.full(len(s), i) for i, s in enumerate(near)] + [empty])
    cols = np.searchsorted(saddle_ids, np.concatenate(near + [empty]))
    k = len(handles)
    link = sparse.csr_matrix((np.ones(len(rows)), (rows, k + cols)), shape=(k + len(saddle_ids),) * 2)
    _, labels = csgraph.connected_components(link, directed=False)
    _, cluster = np.unique(labels[:k], return_inverse=True)
    return cluster, near
