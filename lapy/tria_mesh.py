import logging
import sys
import warnings

import numpy as np
from scipy import sparse

from . import _tria_io as io
from . import polygon

logger = logging.getLogger(__name__)

class TriaMesh:
    """Class representing a triangle mesh.

    This is an efficient implementation of a triangle mesh data structure
    with core functionality using sparse matrices internally (Scipy).

    Parameters
    ----------
    v : array_like
        List of lists of 2 or 3 float coordinates. For 2D vertices (n, 2),
        they will be automatically padded with z=0 to make them 3D internally.
        Must not be empty.
    t : array_like
        List of lists of 3 int of indices (>= 0) into ``v`` array
        Ordering is important: All triangles should be
        oriented in the same way (counter-clockwise, when
        looking from above). Must not be empty.
    fsinfo : dict | None
        FreeSurfer Surface Header Info.

    Attributes
    ----------
    v : array_like
        List of lists of 3 float coordinates (internally always 3D).
    t : array_like
        List of lists of 3 int of indices (>= 0) into ``v`` array.
    adj_sym : csc_matrix
        Symmetric adjacency matrix as csc sparse matrix.
    adj_dir : csc_matrix
        Directed adjacency matrix as csc sparse matrix.
    fsinfo : dict | None
        FreeSurfer Surface Header Info.
    _is_2d : bool
        Internal flag indicating if the mesh was created with 2D vertices.

    Raises
    ------
    ValueError
        If mesh has no triangles or vertices (empty mesh).
        If vertices don't have 2 or 3 coordinates.
        If triangles don't have exactly 3 vertices.
        If triangle indices exceed number of vertices.

    Notes
    -----
    The class has static class methods to read triangle meshes from FreeSurfer,
    OFF, and VTK file formats.

    When 2D vertices are provided, they are internally padded with z=0 to create
    3D vertices. This allows all geometric operations to work correctly while
    maintaining compatibility with 2D mesh data.

    Empty meshes are not allowed. Use class methods (read_fssurf, read_vtk,
    read_off) to load mesh data from files.
    """

    def __init__(self, v, t, fsinfo=None):
        self.v = np.array(v)
        self.t = np.array(t)
        # transpose if necessary
        if self.v.shape[0] < self.v.shape[1]:
            self.v = self.v.T
        # For triangles, check if we need to transpose:
        # Triangles should have shape (n_triangles, 3)
        # If shape[1] != 3 but shape[0] == 3, transpose
        if self.t.shape[1] != 3 and self.t.shape[0] == 3:
            self.t = self.t.T

        # Validate non-empty mesh
        if self.t.size == 0:
            raise ValueError("Mesh has no triangles (empty mesh)")
        if self.v.size == 0:
            raise ValueError("Mesh has no vertices (empty mesh)")

        # Check a few things
        vnum = np.max(self.v.shape)
        if np.max(self.t) >= vnum:
            raise ValueError("Max index exceeds number of vertices")
        if self.t.shape[1] != 3:
            raise ValueError("Triangles should have 3 vertices")

        # Support both 2D and 3D vertices
        if self.v.shape[1] == 2:
            # Pad 2D vertices with z=0 to make them 3D
            self.v = np.column_stack([self.v, np.zeros(self.v.shape[0])])
            self._is_2d = True
        elif self.v.shape[1] == 3:
            self._is_2d = False
        else:
            raise ValueError("Vertices should have 2 or 3 coordinates")

        # Compute adjacency matrices
        self.adj_sym = self._construct_adj_sym()
        self.adj_dir = self._construct_adj_dir()
        self.fsinfo = fsinfo  # place for Freesurfer Header info

    def is_2d(self) -> bool:
        """Check if the mesh was created with 2D vertices.

        Returns
        -------
        bool
            True if mesh was created with 2D vertices, False otherwise.
        """
        return self._is_2d

    def get_vertices(self, original_dim: bool = False) -> np.ndarray:
        """Get mesh vertices.

        Parameters
        ----------
        original_dim : bool, default=False
            If True and mesh was created with 2D vertices, return vertices
            in original 2D format (without z-coordinate). If False, always
            return 3D vertices.

        Returns
        -------
        np.ndarray
            Vertex array of shape (n, 2) or (n, 3) depending on original_dim.
        """
        if original_dim and self._is_2d:
            # Return only x, y coordinates (strip z=0)
            return self.v[:, :2]
        else:
            return self.v

    @classmethod
    def read_fssurf(cls, filename):
        """Load triangle mesh from FreeSurfer surface geometry file.

        Parameters
        ----------
        filename : str
            Filename to load, supporting .pial, .white, .sphere etc.

        Returns
        -------
        TriaMesh
            Loaded triangle mesh.
        """
        return io.read_fssurf(filename)

    @classmethod
    def read_off(cls, filename):
        """Load triangle mesh from OFF txt file.

        Parameters
        ----------
        filename : str
            Filename to load.

        Returns
        -------
        TriaMesh
            Loaded triangle mesh.
        """
        return io.read_off(filename)

    @classmethod
    def read_vtk(cls, filename):
        """Load triangle mesh from VTK txt file.

        Parameters
        ----------
        filename : str
            Filename to load.

        Returns
        -------
        TriaMesh
            Loaded triangle mesh.
        """
        return io.read_vtk(filename)

    def write_vtk(self, filename: str) -> None:
        """Save mesh as VTK file.

        Parameters
        ----------
        filename : str
            Filename to save to.

        Raises
        ------
        IOError
            If file cannot be written.
        """
        io.write_vtk(self, filename)

    def write_fssurf(
        self,
        filename: str,
        image: object | None = None,
        coords_are_voxels: bool | None = None,
    ) -> None:
        """Save as Freesurfer Surface Geometry file (wrap Nibabel).

        Parameters
        ----------
        filename : str
            Filename to save to.
        image : str, object, None
            Path to image, nibabel image object, or image header. If specified, volume_info
            will be extracted from the image header, and by default, vertices are assumed to
            be in voxel coordinates and will be converted to surface RAS (tkr) before saving.
            The expected order of coordinates is (x, y, z) matching the image voxel indices
            in nibabel.
        coords_are_voxels : bool or None, default=None
            Specifies whether vertices are in voxel coordinates. If None (default), the
            behavior is inferred: when image is provided, vertices are assumed to be in
            voxel space and converted to surface RAS; when image is not provided, vertices
            are assumed to already be in surface RAS. Set explicitly to True to force
            conversion (requires image), or False to skip conversion even when image is
            provided (only extracts volume_info).

        Raises
        ------
        ValueError
            If coords_are_voxels is explicitly True but image is None.
            If image header cannot be processed.
        IOError
            If file cannot be written.

        Notes
        -----
        The surface RAS (tkr) transform is obtained from a header that implements
        ``get_vox2ras_tkr()`` (e.g., ``MGHHeader``). For other header types (NIfTI1/2,
        Analyze/SPM, etc.), we attempt conversion via ``MGHHeader.from_header``.
        """
        io.write_fssurf(self, filename, image=image, coords_are_voxels=coords_are_voxels)

    def _construct_adj_sym(self):
        """Construct symmetric adjacency matrix (edge graph) of triangle mesh.

        Operates only on triangles.

        Returns
        -------
        csc_matrix
            The non-directed adjacency matrix
            will be symmetric. Each inner edge (i,j) will have
            the number of triangles that contain this edge.
            Inner edges usually 2, boundary edges 1. Higher
            numbers can occur when there are non-manifold triangles.
            The sparse matrix can be binarized via:
            adj.data = np.ones(adj.data.shape).
        """
        t0 = self.t[:, 0]
        t1 = self.t[:, 1]
        t2 = self.t[:, 2]
        i = np.column_stack((t0, t1, t1, t2, t2, t0)).reshape(-1)
        j = np.column_stack((t1, t0, t2, t1, t0, t2)).reshape(-1)
        dat = np.ones(i.shape)
        n = self.v.shape[0]
        return sparse.csc_matrix((dat, (i, j)), shape=(n, n))

    def _construct_adj_dir(self):
        """Construct directed adjacency matrix (edge graph) of triangle mesh.

        Operates only on triangles.

        Returns
        -------
        csc_matrix
            The directed adjacency matrix is not symmetric if
            boundaries exist or if mesh is non-manifold.
            For manifold meshes, there are only entries with
            value 1. Symmetric entries are inner edges. Non-symmetric
            are boundary edges. The direction prescribes a direction
            on the boundary loops. Adding the matrix to its transpose
            creates the non-directed version.
        """
        t0 = self.t[:, 0]
        t1 = self.t[:, 1]
        t2 = self.t[:, 2]
        i = np.column_stack((t0, t1, t2)).reshape(-1)
        j = np.column_stack((t1, t2, t0)).reshape(-1)
        dat = np.ones(i.shape)
        n = self.v.shape[0]
        return sparse.csc_matrix((dat, (i, j)), shape=(n, n))

    def construct_adj_dir_tidx(self):
        """Construct directed adjacency matrix (edge graph) of triangle mesh.

        The directed adjacency matrix contains the triangle indices (only for
        non-manifold meshes). Operates only on triangles.

        Returns
        -------
        csc_matrix
            Similar to adj_dir, but stores the tria idx+1 instead
            of one in the matrix (allows lookup of vertex to tria).
        """
        if not self.is_oriented():
            raise ValueError(
                "Error: Can only tidx matrix for oriented triangle meshes!"
            )
        t0 = self.t[:, 0]
        t1 = self.t[:, 1]
        t2 = self.t[:, 2]
        i = np.column_stack((t0, t1, t2)).reshape(-1)
        j = np.column_stack((t1, t2, t0)).reshape(-1)
        # store tria idx +1  (zero means no edge here)
        dat = np.repeat(np.arange(1, self.t.shape[0] + 1), 3)
        n = self.v.shape[0]
        return sparse.csc_matrix((dat, (i, j)), shape=(n, n))

    def is_closed(self):
        """Check if triangle mesh is closed (no boundary edges).

        Operates only on triangles

        Returns
        -------
        bool
            True if no boundary edges in adj matrix.
        """
        return 1 not in self.adj_sym.data

    def is_manifold(self):
        """Check if triangle mesh is manifold (no edges with >2 triangles).

        Operates only on triangles

        Returns
        -------
        bool
            True if no edges with > 2 triangles.
        """
        return np.max(self.adj_sym.data) <= 2

    def is_boundary(self) -> np.ndarray:
        """Check which vertices are on the boundary.

        Returns
        -------
        np.ndarray
            Boolean array of shape (n_vertices,) where True indicates the vertex
            is on a boundary edge (an edge that belongs to only one triangle).
        """
        # Boundary edges have value 1 in the symmetric adjacency matrix
        boundary_edges = self.adj_sym == 1
        # Get vertices that are part of any boundary edge
        boundary_vertices = np.zeros(self.v.shape[0], dtype=bool)
        rows, cols = boundary_edges.nonzero()
        boundary_vertices[rows] = True
        boundary_vertices[cols] = True
        return boundary_vertices

    def is_oriented(self) -> bool:
        """Check if triangle mesh is oriented.

        True if all triangles are oriented counter-clockwise, when looking from
        above. Operates only on triangles.

        Returns
        -------
        bool
            True if ``max(adj_directed)=1``.
        """
        return np.max(self.adj_dir.data) == 1

    def euler(self) -> int:
        """Compute the Euler Characteristic.

        The Euler characteristic is the number of vertices minus the number
        of edges plus the number of triangles  (= #V - #E + #T). For example,
        it is 2 for the sphere and 0 for the torus.
        This operates only on triangles array.

        Returns
        -------
        int
            Euler characteristic.
        """
        # v can contain unused vertices so we get vnum from trias
        vnum = len(np.unique(self.t.reshape(-1)))
        tnum = np.max(self.t.shape)
        enum = int(self.adj_sym.nnz / 2)
        return vnum - enum + tnum

    def tria_areas(self) -> np.ndarray:
        """Compute the area of triangles using Heron's formula.

        `Heron's formula <https://en.wikipedia.org/wiki/Heron%27s_formula>`_
        computes the area of a triangle by using the three edge lengths.

        Returns
        -------
        np.ndarray
            Array with areas of each triangle, shape (n_triangles,).
        """
        v0 = self.v[self.t[:, 0], :]
        v1 = self.v[self.t[:, 1], :]
        v2 = self.v[self.t[:, 2], :]
        v1mv0 = v1 - v0
        v2mv1 = v2 - v1
        v0mv2 = v0 - v2
        a = np.sqrt(np.sum(v1mv0 * v1mv0, axis=1))
        b = np.sqrt(np.sum(v2mv1 * v2mv1, axis=1))
        c = np.sqrt(np.sum(v0mv2 * v0mv2, axis=1))
        ph = 0.5 * (a + b + c)
        areas = np.sqrt(ph * (ph - a) * (ph - b) * (ph - c))
        return areas

    def area(self) -> float:
        """Compute the total surface area of triangle mesh.

        Returns
        -------
        float
            Total surface area (sum of all triangle areas).
        """
        areas = self.tria_areas()
        return np.sum(areas)

    def volume(self) -> float:
        """Compute the volume of closed triangle mesh, summing tetrahedra at origin.

        The mesh must be closed (no boundary edges) and oriented (consistent
        triangle orientation).

        Returns
        -------
        float
            Total enclosed volume.

        Raises
        ------
        ValueError
            If mesh is not closed or not oriented.
        """
        if not self.is_closed():
            logger.error("Volume computation requires closed mesh.")
            raise ValueError("Mesh must be closed to compute volume.")
        if not self.is_oriented():
            logger.error("Volume computation requires oriented mesh.")
            raise ValueError("Mesh must be oriented to compute volume.")
        v0 = self.v[self.t[:, 0], :]
        v1 = self.v[self.t[:, 1], :]
        v2 = self.v[self.t[:, 2], :]
        v1mv0 = v1 - v0
        v2mv0 = v2 - v0
        cr = np.cross(v1mv0, v2mv0)
        spatvol = np.sum(v0 * cr, axis=1)
        vol = np.sum(spatvol) / 6.0
        logger.debug("Computed volume %s", vol)
        return vol

    def vertex_degrees(self) -> np.ndarray:
        """Compute the vertex degrees (number of edges at each vertex).

        Returns
        -------
        np.ndarray
            Array of vertex degrees, shape (n_vertices,).
        """
        vdeg = np.bincount(self.t.reshape(-1))
        return vdeg

    def vertex_areas(self):
        """Compute the area associated to each vertex (1/3 of one-ring trias).

        Returns
        -------
        vareas : array
            Array of vertex areas.
        """
        v0 = self.v[self.t[:, 0], :]
        v1 = self.v[self.t[:, 1], :]
        v2 = self.v[self.t[:, 2], :]
        v1mv0 = v1 - v0
        v2mv0 = v2 - v0
        cr = np.cross(v1mv0, v2mv0)
        area = 0.5 * np.sqrt(np.sum(cr * cr, axis=1))
        area3 = np.repeat(area[:, np.newaxis], 3, 1)
        # varea = accumarray(t(:),area3(:))./3;
        vareas = np.bincount(self.t.reshape(-1), area3.reshape(-1)) / 3.0
        return vareas

    def avg_edge_length(self):
        """Compute the average edge length of the mesh.

        Returns
        -------
        float
            Average edge length.
        """
        # get only upper off-diag elements from symmetric adj matrix
        triadj = sparse.triu(self.adj_sym, 1, format="coo")
        edgelens = np.sqrt(
            ((self.v[triadj.row, :] - self.v[triadj.col, :]) ** 2).sum(1)
        )
        return edgelens.mean()

    def tria_normals(self) -> np.ndarray:
        """Compute triangle normals.

        Triangle normals are computed using the cross product of two edges.
        Ordering of triangles is important: normals point outward for
        counter-clockwise oriented triangles when looking from above.

        Returns
        -------
        np.ndarray
            Triangle normals of shape (n_triangles, 3). Each normal is
            normalized to unit length.
        """
        # Compute vertex coordinates and a difference vectors for each triangle:
        v0 = self.v[self.t[:, 0], :]
        v1 = self.v[self.t[:, 1], :]
        v2 = self.v[self.t[:, 2], :]
        v1mv0 = v1 - v0
        v2mv0 = v2 - v0
        # Compute cross product
        n = np.cross(v1mv0, v2mv0)
        ln = np.sqrt(np.sum(n * n, axis=1))
        ln[ln < sys.float_info.epsilon] = 1  # avoid division by zero
        n = n / ln.reshape(-1, 1)
        return n

    def vertex_normals(self) -> np.ndarray:
        """Compute vertex normals.

        Vertex normals are computed by averaging triangle normals around each
        vertex, weighted by the angle that each triangle contributes at that
        vertex. The mesh must be oriented for meaningful results.

        Returns
        -------
        np.ndarray
            Vertex normals of shape (n_vertices, 3). Each normal is
            normalized to unit length.

        Raises
        ------
        ValueError
            If mesh is not oriented.
        """
        if not self.is_oriented():
            raise ValueError(
                "Error: Vertex normals are meaningless for un-oriented triangle meshes!"
            )

        # Compute vertex coordinates and a difference vector for each triangle:
        v0 = self.v[self.t[:, 0], :]
        v1 = self.v[self.t[:, 1], :]
        v2 = self.v[self.t[:, 2], :]
        v1mv0 = v1 - v0
        v2mv1 = v2 - v1
        v0mv2 = v0 - v2
        # Compute cross product at every vertex
        # will all point in the same direction but have
        #   different lengths depending on spanned area
        cr0 = np.cross(v1mv0, -v0mv2)
        cr1 = np.cross(v2mv1, -v1mv0)
        cr2 = np.cross(v0mv2, -v2mv1)
        # Add normals at each vertex (there can be duplicate indices in t at vertex i)
        n = np.zeros(self.v.shape)
        np.add.at(n, self.t[:, 0], cr0)
        np.add.at(n, self.t[:, 1], cr1)
        np.add.at(n, self.t[:, 2], cr2)
        # Normalize normals
        ln = np.sqrt(np.sum(n * n, axis=1))
        ln[ln < sys.float_info.epsilon] = 1  # avoid division by zero
        n = n / ln.reshape(-1, 1)
        return n

    def has_free_vertices(self) -> bool:
        """Check if the vertex list has more vertices than what is used in triangles.

        Free vertices are vertices that are not referenced by any triangle.

        Returns
        -------
        bool
            True if vertex list has more vertices than used in triangles,
            False otherwise.
        """
        vnum = np.max(self.v.shape)
        vnumt = len(np.unique(self.t.reshape(-1)))
        return vnum != vnumt

    def tria_qualities(self) -> np.ndarray:
        """Compute triangle quality for each triangle in mesh.

        Quality measure: q = 4 sqrt(3) A / (e1^2 + e2^2 + e3^2)
        where A is the triangle area and ei the edge length of the three edges.
        Constants are chosen so that q=1 for the equilateral triangle.

        .. note::

            This measure is used by FEMLAB and can also be found in:
            R.E. Bank, PLTMG ..., Frontiers in Appl. Math. (7), 1990.

        Returns
        -------
        np.ndarray
            Array with triangle qualities, shape (n_triangles,). Values range
            from 0 (degenerate) to 1 (equilateral).
        """
        # Compute vertex coordinates and a difference vectors for each triangle:
        v0 = self.v[self.t[:, 0], :]
        v1 = self.v[self.t[:, 1], :]
        v2 = self.v[self.t[:, 2], :]
        v1mv0 = v1 - v0
        v2mv1 = v2 - v1
        v0mv2 = v0 - v2
        # Compute cross product
        n = np.cross(v1mv0, -v0mv2)
        # compute length (2*area)
        ln = np.sqrt(np.sum(n * n, axis=1))
        q = 2.0 * np.sqrt(3) * ln
        es = (v1mv0 * v1mv0).sum(1) + (v2mv1 * v2mv1).sum(1) + (v0mv2 * v0mv2).sum(1)
        es[es < sys.float_info.epsilon] = 1  # avoid division by zero
        return q / es

    def boundary_loops(self) -> list:
        """Compute boundary loops of the mesh.

        Meshes can have 0 or more boundary loops, which are cycles in the directed
        adjacency graph of the boundary edges. The mesh must be manifold and oriented.

        .. note::

            Could fail if loops are connected via a single vertex (like a figure 8).
            That case needs debugging.

        Returns
        -------
        list of list
            List of lists with boundary loops. Each inner list contains vertex
            indices forming a closed loop. Empty list if mesh is closed.

        Raises
        ------
        ValueError
            If mesh is not manifold (edges with more than 2 triangles).
            If mesh is not oriented.
        """
        if not self.is_manifold():
            raise ValueError(
                "Error: tria not manifold (edges with more than 2 triangles)!"
            )
        if self.is_closed():
            return []
        # get directed matrix of only boundary edges
        inneredges = self.adj_sym == 2
        if not self.is_oriented():
            raise ValueError("Error: tria not oriented !")
        adj = self.adj_dir.copy()
        adj[inneredges] = 0
        adj.eliminate_zeros()
        # find loops
        # get first column index with an entry:
        firstcol = np.nonzero(adj.indptr)[0][0] - 1
        loops = []
        # loop while we have more first columns:
        while np.size(firstcol) > 0:
            # start the new loop with this index
            loop = [firstcol]
            # delete this entry from matrix (visited)
            adj.data[adj.indptr[firstcol]] = 0
            # get the next column (=row index of the first entry (and only, hopefully)
            ncol = adj.indices[adj.indptr[firstcol]]
            # as long as loop is not closed walk through it
            while not ncol == firstcol:
                loop.append(ncol)
                adj.data[adj.indptr[ncol]] = 0  # visited
                ncol = adj.indices[adj.indptr[ncol]]
            # get rid of the visited nodes, store loop and check for another one
            adj.eliminate_zeros()
            loops.append(loop)
            nz = np.nonzero(adj.indptr)[0]
            if len(nz) > 0:
                firstcol = nz[0] - 1
            else:
                firstcol = []
        return loops

    def connected_components(self) -> tuple[int, np.ndarray]:
        """Compute connected components of the mesh.

        Uses scipy's connected_components on the symmetric adjacency matrix.

        Returns
        -------
        n_components : int
            Number of connected components.
        labels : np.ndarray
            Label array of shape (n_vertices,) where labels[i] is the component
            ID of vertex i. Component IDs are integers from 0 to n_components-1.
        """
        from scipy.sparse.csgraph import connected_components

        return connected_components(self.adj_sym, directed=False)

    def keep_largest_connected_component_(
        self, clean: bool = True
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Keep only the largest connected component of the mesh.

        Modifies the mesh in-place.

        Parameters
        ----------
        clean : bool, default=True
            If True, remove vertices that are not used in the largest component.
            If False, vertices are kept but triangles from other components are removed,
            creating free vertices.

        Returns
        -------
        vkeep : np.ndarray or None
            Indices (from original list) of kept vertices if clean=True, else None.
        vdel : np.ndarray or None
            Indices of deleted (unused) vertices if clean=True, else None.
        """
        n_components, labels = self.connected_components()
        if n_components <= 1:
            if clean:
                return self.rm_free_vertices_()
            else:
                return None, None
        # Count vertices in each component
        counts = np.bincount(labels)
        # Get label of largest component
        largest_comp_label = np.argmax(counts)
        # Filter triangles: check which component the first vertex of each triangle
        # belongs to (all vertices of a triangle are in the same component)
        t_mask = labels[self.t[:, 0]] == largest_comp_label
        self.t = self.t[t_mask]
        if clean:
            return self.rm_free_vertices_()
        else:
            # Re-init to update adjacency matrices with new t
            self.__init__(self.v, self.t, self.fsinfo)
            return None, None

    def centroid(self) -> tuple[np.ndarray, float]:
        """Compute centroid of triangle mesh as a weighted average of triangle centers.

        The weight is determined by the triangle area.
        (This could be done much faster if a FEM lumped mass matrix M is
        already available where this would be M*v, because it is equivalent
        with averaging vertices weighted by vertex area)

        Returns
        -------
        centroid : np.ndarray
            The centroid coordinates of the mesh, shape (3,).
        totalarea : float
            The total area of the mesh.
        """
        v0 = self.v[self.t[:, 0], :]
        v1 = self.v[self.t[:, 1], :]
        v2 = self.v[self.t[:, 2], :]
        v2mv1 = v2 - v1
        v0mv2 = v0 - v2
        # Compute cross product and area for each triangle:
        cr = np.cross(v2mv1, v0mv2)
        areas = 0.5 * np.sqrt(np.sum(cr * cr, axis=1))
        totalarea = areas.sum()
        areas = areas / totalarea
        centers = (1.0 / 3.0) * (v0 + v1 + v2)
        c = centers * areas[:, np.newaxis]
        return np.sum(c, axis=0), totalarea

    def edges(self, with_boundary: bool = False) -> tuple[np.ndarray, ...]:
        """Compute vertices and adjacent triangle ids for each edge.

        Parameters
        ----------
        with_boundary : bool, default=False
            If True, also return boundary half edges. If False, only return
            interior edges.

        Returns
        -------
        vids : np.ndarray
            Array of shape (n_edges, 2) with starting and end vertex for each
            unique inner edge.
        tids : np.ndarray
            Array of shape (n_edges, 2) with triangle containing the half edge
            from vids[:,0] to vids[:,1] in first column and the neighboring
            triangle in the second column.
        bdrvids : np.ndarray
            If with_boundary is True: Array of shape (n_boundary_edges, 2) with
            each boundary half-edge. Only returned if with_boundary=True.
        bdrtids : np.ndarray
            If with_boundary is True: Array of shape (n_boundary_edges, 1) with
            the associated triangle to each boundary edge. Only returned if
            with_boundary=True.

        Raises
        ------
        ValueError
            If mesh is not oriented.
        """
        if not self.is_oriented():
            raise ValueError(
                "Error: Can only compute edge information for oriented meshes!"
            )
        adjtria = self.construct_adj_dir_tidx().tolil()
        # for boundary edges, we can just remove those edges (implicitly a zero angle)
        bdredges = []
        bdrtrias = []
        if 1 in self.adj_sym.data:
            bdredges = self.adj_sym == 1
            bdrtrias = adjtria[bdredges].toarray().ravel() - 1
            adjtria[bdredges] = 0
        # get transpose adjTria matrix and keep only upper triangular matrices
        adjtria2 = adjtria.transpose()
        adjtriu1 = sparse.triu(adjtria, 0, format="csr")
        adjtriu2 = sparse.triu(adjtria2, 0, format="csr")
        vids = np.array(np.nonzero(adjtriu1), dtype=np.int32).T
        tids = np.empty(vids.shape, dtype=np.int32)
        tids[:, 0] = adjtriu1.data - 1
        tids[:, 1] = adjtriu2.data - 1
        if not with_boundary or bdredges.size == 0:
            return vids, tids
        bdrv = np.array(np.nonzero(bdredges)).T
        nzids = bdrtrias > -1
        bdrv = bdrv[nzids, :]
        bdrtrias = bdrtrias[nzids].reshape(-1, 1)
        return vids, tids, bdrv, bdrtrias

    def curvature(
        self, smoothit: int = 3
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Compute various curvature values at vertices.

        Computes principal curvature directions and values, mean and Gaussian
        curvature at each vertex using the anisotropic polygonal remeshing approach.

        .. note::

            For the algorithm see:
            Pierre Alliez, David Cohen-Steiner, Olivier Devillers,
            Bruno Levy, and Mathieu Desbrun.
            Anisotropic Polygonal Remeshing.
            ACM Transactions on Graphics, 2003.

        Parameters
        ----------
        smoothit : int, default=3
            Number of smoothing iterations on vertex functions.

        Returns
        -------
        u_min : np.ndarray
            Minimal curvature directions, shape (n_vertices, 3).
        u_max : np.ndarray
            Maximal curvature directions, shape (n_vertices, 3).
        c_min : np.ndarray
            Minimal curvature values, shape (n_vertices,).
        c_max : np.ndarray
            Maximal curvature values, shape (n_vertices,).
        c_mean : np.ndarray
            Mean curvature ``(c_min + c_max) / 2.0``, shape (n_vertices,).
        c_gauss : np.ndarray
           Gauss curvature ``c_min * c_max``, shape (n_vertices,).
        normals : np.ndarray
           Vertex normals, shape (n_vertices, 3).
        """
        # import warnings
        # warnings.filterwarnings('error')

        # get edge information for inner edges (vertex ids and tria ids):
        vids, tids = self.edges()
        # compute normals for each tria
        tnormals = self.tria_normals()
        # compute dot product of normals at each edge
        sprod = np.sum(tnormals[tids[:, 0], :] * tnormals[tids[:, 1], :], axis=1)
        # compute unsigned angles (clamp to ensure range)
        angle = np.maximum(sprod, -1)
        angle = np.minimum(angle, 1)
        angle = np.arccos(angle)
        # compute edge vectors and lengths
        edgevecs = self.v[vids[:, 1], :] - self.v[vids[:, 0], :]
        edgelen = np.sqrt(np.sum(edgevecs**2, axis=1))
        # get sign (if normals face towards each other or away, across each edge)
        cp = np.cross(tnormals[tids[:, 0], :], tnormals[tids[:, 1], :])
        si = -np.sign(np.sum(cp * edgevecs, axis=1))
        angle = angle * si
        # normalized edges
        edgelen[edgelen < sys.float_info.epsilon] = 1  # avoid division by zero
        edgevecs = edgevecs / edgelen.reshape(-1, 1)
        # adjust edgelengths so that mean is 1 for numerics
        edgelen = edgelen / np.mean(edgelen)
        # symmetric edge matrix (3x3, upper triangular matrix entries):
        ee = np.empty([edgelen.shape[0], 6])
        ee[:, 0] = edgevecs[:, 0] * edgevecs[:, 0]
        ee[:, 1] = edgevecs[:, 0] * edgevecs[:, 1]
        ee[:, 2] = edgevecs[:, 0] * edgevecs[:, 2]
        ee[:, 3] = edgevecs[:, 1] * edgevecs[:, 1]
        ee[:, 4] = edgevecs[:, 1] * edgevecs[:, 2]
        ee[:, 5] = edgevecs[:, 2] * edgevecs[:, 2]
        # scale angle by edge lengths
        angle = angle * edgelen
        # multiply scaled angle with matrix entries
        ee = ee * angle.reshape(-1, 1)
        # map to vertices
        vnum = self.v.shape[0]
        vv = np.zeros([vnum, 6])
        np.add.at(vv, vids[:, 0], ee)
        np.add.at(vv, vids[:, 1], ee)
        vdeg = np.zeros([vnum])
        np.add.at(vdeg, vids[:, 0], 1)
        np.add.at(vdeg, vids[:, 1], 1)
        # divide by vertex degree (maybe better by edge length sum??)
        # handle division by zero (for isolated vertices)
        vdeg[vdeg == 0] = 1
        vv = vv / vdeg.reshape(-1, 1)
        # smooth vertex functions
        vv = self.smooth_laplace(vfunc=vv, n=smoothit, lambda_=1.0)
        # create vnum 3x3 symmetric matrices at each vertex
        mats = np.empty([vnum, 3, 3])
        mats[:, 0, :] = vv[:, [0, 1, 2]]
        mats[:, [1, 2], 0] = vv[:, [1, 2]]
        mats[:, 1, [1, 2]] = vv[:, [3, 4]]
        mats[:, 2, 1] = vv[:, 4]
        mats[:, 2, 2] = vv[:, 5]
        # compute eigendecomposition (real for symmetric matrices)
        # eigh is better for symmetric matrices
        evals, evecs = np.linalg.eigh(mats)
        # sort evals ascending
        # this is instable in perfectly planar regions
        #  (normal can lie in tangential plane)
        # i = np.argsort(np.abs(evals), axis=1)
        # instead we find direction that aligns with vertex normals as first
        # the other two will be sorted later anyway
        vnormals = self.vertex_normals()
        dprod = -np.abs(np.squeeze(np.sum(evecs * vnormals[:, :, np.newaxis], axis=1)))
        i = np.argsort(dprod, axis=1)
        evals = np.take_along_axis(evals, i, axis=1)
        it = np.tile(i.reshape((vnum, 1, 3)), (1, 3, 1))
        evecs = np.take_along_axis(evecs, it, axis=2)
        # pull min and max curv. dirs
        u_min = np.squeeze(evecs[:, :, 2])
        u_max = np.squeeze(evecs[:, :, 1])
        c_min = evals[:, 1]
        c_max = evals[:, 2]
        normals = np.squeeze(evecs[:, :, 0])
        c_mean = (c_min + c_max) / 2.0
        c_gauss = c_min * c_max
        # enforce that min<max
        i = np.squeeze(np.where(c_min > c_max))
        c_min[i], c_max[i] = c_max[i], c_min[i]
        u_min[i, :], u_max[i, :] = u_max[i, :], u_min[i, :]
        # flip normals to point towards vertex normals
        s = np.sign(np.sum(normals * vnormals, axis=1)).reshape(-1, 1)
        normals = normals * s
        # (here we could also project to tangent plane at vertex (using v_normals)
        # as the normals above are not really good v_normals)
        # flip u_max so that cross(u_min , u_max) aligns with normals
        u_cross = np.cross(u_min, u_max)
        d = np.sum(np.multiply(u_cross, normals), axis=1)
        i = np.squeeze(np.where(d < 0))
        u_max[i, :] = -u_max[i, :]
        return u_min, u_max, c_min, c_max, c_mean, c_gauss, normals

    def curvature_tria(
        self, smoothit: int = 3
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute min and max curvature and directions on triangle faces.

        First computes curvature values on vertices and smooths them.
        Then maps them to triangles (averaging) and projects onto the triangle plane,
        ensuring orthogonality.

        Parameters
        ----------
        smoothit : int, default=3
            Number of smoothing iterations for curvature computation on vertices.

        Returns
        -------
        u_min : np.ndarray
            Minimal curvature direction on triangles, shape (n_triangles, 3).
        u_max : np.ndarray
            Maximal curvature direction on triangles, shape (n_triangles, 3).
            Orthogonal to u_min and triangle normal.
        c_min : np.ndarray
            Minimal curvature values on triangles, shape (n_triangles,).
        c_max : np.ndarray
            Maximal curvature values on triangles, shape (n_triangles,).
        """
        u_min, u_max, c_min, c_max, c_mean, c_gauss, normals = self.curvature(smoothit)

        # pool vertex functions (u_min and u_max) to triangles:
        tumin = self.map_vfunc_to_tfunc(u_min)
        # tumax = self.map_vfunc_to_tfunc(u_max)
        tcmin = self.map_vfunc_to_tfunc(c_min)
        tcmax = self.map_vfunc_to_tfunc(c_max)
        # some Us are almost collinear, strange
        # print(np.max(np.abs(np.sum(tumin * tumax, axis=1))))
        # print(np.sum(tumin * tumax, axis=1))

        # project onto triangle plane:
        e0 = self.v[self.t[:, 1], :] - self.v[self.t[:, 0], :]
        e1 = self.v[self.t[:, 2], :] - self.v[self.t[:, 0], :]
        tn = np.cross(e0, e1)
        tnl = np.sqrt(np.sum(tn * tn, axis=1)).reshape(-1, 1)
        tn = tn / np.maximum(tnl, 1e-8)
        # project tumin back to tria plane and normalize
        tumin2 = tumin - tn * (np.sum(tn * tumin, axis=1)).reshape(-1, 1)
        tuminl = np.sqrt(np.sum(tumin2 * tumin2, axis=1)).reshape(-1, 1)
        tumin2 = tumin2 / np.maximum(tuminl, 1e-8)
        # project tumax back to tria plane and normalize
        #   (will not be orthogonal to tumin)
        # tumax1 = tumax - tn * (np.sum(tn * tumax, axis=1)).reshape(-1, 1)
        # in a second step orthorgonalize to tumin
        # tumax1 = tumax1 - tumin * (np.sum(tumin * tumax1, axis=1)).reshape(-1, 1)
        # normalize
        # tumax1l = np.sqrt(np.sum(tumax1 * tumax1, axis=1)).reshape(-1, 1)
        # tumax1 = tumax1 / np.maximum(tumax1l, 1e-8)
        # or simply create vector that is orthogonal to both normal and tumin
        tumax2 = np.cross(tn, tumin2)
        # if really necessary flip direction if that is true for inputs
        # tumax3 = np.sign(np.sum(np.cross(tumin, tumax) * tn, axis=1)).reshape(-1, 1)
        #           * tumax2
        # I wonder how much changes, if we first map umax to tria and then
        #   find orthogonal umin next?
        return tumin2, tumax2, tcmin, tcmax

    def critical_points(self, vfunc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute critical points (extrema and saddles) of a function on mesh vertices.

        A minimum is a vertex where all neighbor values are larger.
        A maximum is a vertex where all neighbor values are smaller.
        A saddle is a vertex with at least two regions of neighbors with larger values,
        and two with smaller values. Boundary vertices assume a virtual edge outside
        the mesh that closes the boundary loop around the vertex.

        Parameters
        ----------
        vfunc : np.ndarray
            Real-valued function defined on mesh vertices, shape (n_vertices,).

        Returns
        -------
        minima : np.ndarray
            Array of vertex indices that are local minima, shape (n_minima,).
        maxima : np.ndarray
            Array of vertex indices that are local maxima, shape (n_maxima,).
        saddles : np.ndarray
            Array of vertex indices that are saddles (all orders), shape (n_saddles,).
        saddle_orders : np.ndarray
            Array of saddle orders for each saddle vertex (same length as saddles), shape (n_saddles,).
            Order 2 = simple saddle (4 sign flips), order 3+ = higher-order saddles.

        Notes
        -----
        The algorithm works by:

        1. For each vertex, compute difference: neighbor_value - vertex_value
        2. Minima: all differences are positive (all neighbors higher)
        3. Maxima: all differences are negative (all neighbors lower)
        4. Saddles: count sign flips across opposite edges in triangles at vertex

           - Regular point: 2 sign flips
           - Simple saddle (order 2): 4 sign flips
           - Higher-order saddle (order n): 2n sign flips, order = n_flips / 2

        5. Tie-breaker: when two vertices have equal function values, the vertex
           with the higher vertex ID is treated as slightly larger to remove ambiguity.
        """
        vfunc = np.asarray(vfunc)
        if len(vfunc) != self.v.shape[0]:
            raise ValueError("vfunc length must match number of vertices")
        n_vertices = self.v.shape[0]

        # Use SYMMETRIC adjacency matrix to get ALL edges (including boundary edges in both directions)
        # COMPUTE EDGE SIGNS ONCE for all neighbor relationships
        rows, cols = self.adj_sym.nonzero()
        edge_diffs = vfunc[cols] - vfunc[rows]
        edge_signs = np.sign(edge_diffs)
        # Tie-breaker: when function values are equal, vertex with higher ID is larger
        zero_mask = edge_signs == 0
        edge_signs[zero_mask] = np.sign(cols[zero_mask] - rows[zero_mask])
        # Create sparse matrix of edge signs for O(1) lookup
        edge_sign_matrix = sparse.csr_matrix(
            (edge_signs, (rows, cols)),
            shape=(n_vertices, n_vertices)
        )

        # CLASSIFY MINIMA AND MAXIMA
        # Compute min and max edge sign per vertex (row-wise)
        # Note: edge_sign_matrix only contains +1/-1 (never 0 due to tie-breaker)
        # Implicit zeros in sparse matrix represent non-neighbors and can be ignored
        min_signs = edge_sign_matrix.min(axis=1).toarray().flatten()
        max_signs = edge_sign_matrix.max(axis=1).toarray().flatten()
        # Minimum: all neighbor signs are positive (+1)
        # If min in {0,1} and max=1, all actual neighbors are +1 (zeros are non-neighbors)
        is_minimum = (min_signs > -1) & (max_signs == 1)
        # Maximum: all neighbor signs are negative (-1)
        # If min=-1 and max in {-1,0}, all actual neighbors are -1 (zeros are non-neighbors)
        is_maximum = (min_signs == -1) & (max_signs < 1)
        minima = np.where(is_minimum)[0]
        maxima = np.where(is_maximum)[0]

        # COUNT SIGN FLIPS at opposite edge for saddle detection
        sign_flips = np.zeros(n_vertices, dtype=int)
        t0 = self.t[:, 0]
        t1 = self.t[:, 1]
        t2 = self.t[:, 2]
        # For vertex 0, opposite edge is (v1, v2)
        sign0_1 = np.array(edge_sign_matrix[t0, t1]).flatten()
        sign0_2 = np.array(edge_sign_matrix[t0, t2]).flatten()
        sign1_2 = np.array(edge_sign_matrix[t1, t2]).flatten()
        flip0 = (sign0_1 * sign0_2) < 0
        np.add.at(sign_flips, t0[flip0], 1)
        # For vertex 1, opposite edge is (v2, v0)
        # sign(v1→v0) = -sign(v0→v1) = -sign0_1
        flip1 = (sign1_2 * (-sign0_1)) < 0
        np.add.at(sign_flips, t1[flip1], 1)
        # For vertex 2, opposite edge is (v0, v1)
        # sign(v2→v0) = -sign(v0→v2) = -sign0_2
        # sign(v2→v1) = -sign(v1→v2) = -sign1_2
        flip2 = (sign0_2 * sign1_2) < 0
        np.add.at(sign_flips, t2[flip2], 1)

        # CLASSIFY SADDLES
        # Saddles have 4+ flips (regular points have 2, minima/maxima have 0)
        # Boundary vertices can have 3 flips (or more) to be a saddle, assuming an additional flip outside
        # Order = n_flips / 2
        is_saddle = sign_flips >= 3
        saddles = np.where(is_saddle)[0]
        saddle_orders = (sign_flips[saddles] + 1) // 2

        return minima, maxima, saddles, saddle_orders

    def normalize_(self) -> None:
        """Normalize TriaMesh to unit surface area and centroid at the origin.

        Modifies the vertices in-place by:
        1. Translating centroid to origin
        2. Scaling to unit surface area

        Raises
        ------
        ValueError
            If mesh surface area is not positive.
        """
        centroid, area = self.centroid()
        if area <= 0:
            raise ValueError("Mesh surface area must be positive to normalize.")
        self.v = (1.0 / np.sqrt(area)) * (self.v - centroid)

    def rm_free_vertices_(self) -> tuple[np.ndarray, np.ndarray]:
        """Remove unused (free) vertices.

        Free vertices are vertices that are not referenced by any triangle.
        They can produce problems when constructing, e.g., Laplace matrices.

        Modifies the mesh in-place by removing unused vertices and re-indexing
        triangles.

        Returns
        -------
        vkeep : np.ndarray
            Indices (from original list) of kept vertices, shape (n_kept,).
        vdel : np.ndarray
            Indices of deleted (unused) vertices, shape (n_deleted,).

        Raises
        ------
        ValueError
            If triangle indices exceed number of vertices.
        """
        tflat = self.t.reshape(-1)
        vnum = self.v.shape[0]
        if np.max(tflat) >= vnum:
            raise ValueError("Max index exceeds number of vertices")
        # determine which vertices to keep
        vkeep = np.full(vnum, False, dtype=bool)
        vkeep[tflat] = True
        # list of deleted vertices (old indices)
        vdel = np.nonzero(~vkeep)[0]
        # if nothing to delete return
        if len(vdel) == 0:
            return np.arange(vnum), np.array([], dtype=int)
        # delete unused vertices
        vnew = self.v[vkeep, :]
        # create lookup table
        tlookup = np.cumsum(vkeep) - 1
        # reindex tria
        tnew = tlookup[self.t]
        # convert vkeep to index list
        vkeep = np.nonzero(vkeep)[0]
        # set new vertices and tria and re-init adj matrices
        self.__init__(vnew, tnew, self.fsinfo)
        return vkeep, vdel

    def refine_(self, it: int = 1) -> None:
        """Refine the triangle mesh by placing new vertex on each edge midpoint.

        Creates 4 similar triangles from each parent triangle (1-to-4 subdivision).
        Modifies mesh in-place.

        Parameters
        ----------
        it : int, default=1
            Number of refinement iterations. Each iteration quadruples the number
            of triangles.
        """
        for _x in range(it):
            # make symmetric adj matrix to upper triangle
            adjtriu = sparse.triu(self.adj_sym, 0, format="csr")
            # create new vertex index for each edge
            edgeno = adjtriu.data.shape[0]
            vno = self.v.shape[0]
            adjtriu.data = np.arange(vno, vno + edgeno)
            # get vertices at edge midpoints:
            rows, cols = adjtriu.nonzero()
            vnew = 0.5 * (self.v[rows, :] + self.v[cols, :])
            vnew = np.append(self.v, vnew, axis=0)
            # make adj symmetric again
            adjtriu = adjtriu + adjtriu.T
            # create 4 new triangles for each old one
            e1 = np.asarray(adjtriu[self.t[:, 0], self.t[:, 1]].flat)
            e2 = np.asarray(adjtriu[self.t[:, 1], self.t[:, 2]].flat)
            e3 = np.asarray(adjtriu[self.t[:, 2], self.t[:, 0]].flat)
            t1 = np.column_stack((self.t[:, 0], e1, e3))
            t2 = np.column_stack((self.t[:, 1], e2, e1))
            t3 = np.column_stack((self.t[:, 2], e3, e2))
            t4 = np.column_stack((e1, e2, e3))
            tnew = np.reshape(np.concatenate((t1, t2, t3, t4), axis=1), (-1, 3))
            # set new vertices and tria and re-init adj matrices
            self.__init__(vnew, tnew, self.fsinfo)

    def normal_offset_(self, d: float) -> None:
        """Move vertices along their normals by distance d.

        Displaces each vertex along its vertex normal. Useful for creating
        offset surfaces or inflating/deflating meshes.

        Parameters
        ----------
        d : float or np.ndarray
            Move distance. Can be a scalar (same distance for all vertices)
            or array of shape (n_vertices,) for per-vertex distances.

        Raises
        ------
        ValueError
            If mesh is not oriented (required for vertex normals).

        Notes
        -----
        This modifies vertices in-place without re-initializing adjacency matrices.
        """
        n = self.vertex_normals()
        vn = self.v + d * n
        self.v = vn
        # no need to re-init, only changed vertices

    def orient_(self) -> int:
        """Re-orient triangles of manifold mesh to be consistent.

        Re-orients triangles so that vertices are listed counter-clockwise
        when looking from above (outside). Uses a flood-fill algorithm to
        propagate orientation from the first triangle.

        Algorithm:

        1. Construct list for each half-edge with its triangle and edge direction
        2. Drop boundary half-edges and find half-edge pairs
        3. Construct sparse matrix with triangle neighbors, with entry 1 for opposite
           half edges and -1 for parallel half-edges (normal flip across this edge)
        4. Flood mesh from first triangle using neighbor matrix, tracking sign
        5. Negative sign for a triangle indicates it needs to be flipped
        6. If global volume is negative, flip everything (first triangle was wrong)

        Returns
        -------
        int
            Number of triangles flipped.

        Notes
        -----
        This modifies the triangle array in-place to achieve consistent orientation.
        """
        tnew = self.t
        flipped = 0
        if not self.is_oriented():
            logger.info("Orienting the mesh for consistent triangle ordering.")
            # get half edges
            t0 = self.t[:, 0]
            t1 = self.t[:, 1]
            t2 = self.t[:, 2]
            # i,j are beginning and end points of each half edge
            i = np.column_stack((t0, t1, t2)).reshape(-1)
            j = np.column_stack((t1, t2, t0)).reshape(-1)
            # tidx for each half edge
            tidx = np.repeat(np.arange(0, self.t.shape[0]), 3)
            # if edge points from smaller to larger index or not
            dirij = i < j
            ndirij = np.logical_not(dirij)
            ij = np.column_stack((i, j))
            # make sure i < j
            ij[np.ix_(ndirij, [1, 0])] = ij[np.ix_(ndirij, [0, 1])]
            # remove rows with unique (boundary) edges (half-edges without partner)
            u, ind, c = np.unique(ij, axis=0, return_index=True, return_counts=True)
            bidx = ind[c == 1]
            # assert remaining edges have two triangles: min = max =2
            # note if we have only a single triangle or triangle soup
            # this will fail as we have no inner edges.
            if max(c) != 2 or min(c) < 1:
                raise ValueError(
                    "Without boundary edges, all should have two triangles!"
                )
            # inner is a mask for inner edges
            inner = np.ones(ij.shape[0], bool)
            inner[bidx] = False
            # stack i,j,tria_id, edge_direction (smaller to larger vidx) for inner edges
            ijk = np.column_stack((ij, tidx, dirij))[inner, :]
            # sort according to first two columns
            ind = np.lexsort(
                (ijk[:, 0], ijk[:, 1])
            )  # Sort by column 0, then by column 1
            ijks = ijk[ind, :]
            # select both tria indices at each edge and the edge directions
            tdir = ijks.reshape((-1, 8))[:, [2, 6, 3, 7]]
            # compute sign vector (1 if edge points from small to large, else -1)
            tsgn = 2 * np.logical_xor(tdir[:, 2], tdir[:, 3]) - 1
            # append to itself for symmetry
            tsgn = np.append(tsgn, tsgn)
            i = np.append(tdir[:, 0], tdir[:, 1])
            j = np.append(tdir[:, 1], tdir[:, 0])
            # construct sparse tria neighbor matrix where
            #   weights indicate normal flips across edge
            tmat = sparse.csc_matrix((tsgn, (i, j)))
            tdim = max(i) + 1
            tmat = tmat + sparse.eye(tdim)
            # flood by starting with neighbors of tria 0 to fill all trias
            # sadly we still need a loop for this, matrix power would be too slow
            # as we don't really need to compute full matrix, only need first column
            v = tmat[:, 0]
            count = 0
            import time

            startt = time.time()
            while len(v.data) < tdim:
                count = count + 1
                v = tmat * v
                v.data = np.sign(v.data)
            endt = time.time()
            logger.debug(
                "Searched mesh after %d flood iterations (%f sec).", count, endt - startt
            )
            # get tria indices that need flipping:
            idx = v.toarray() == -1
            idx = idx.reshape(-1)
            tnew = self.t
            tnew[np.ix_(idx, [1, 0])] = tnew[np.ix_(idx, [0, 1])]
            self.__init__(self.v, tnew, self.fsinfo)
            flipped = idx.sum()
        # for closed meshes, flip orientation on all trias if volume is negative:
        if self.is_closed():
            logger.debug("Closed mesh detected; ensuring global orientation.")
            if self.volume() < 0:
                tnew[:, [1, 2]] = tnew[:, [2, 1]]
                self.__init__(self.v, tnew, self.fsinfo)
                flipped = tnew.shape[0] - flipped
        return flipped

    def map_tfunc_to_vfunc(self, tfunc: np.ndarray, weighted: bool = False) -> np.ndarray:
        """Map triangle function to vertices by attributing 1/3 to each vertex.

        Distributes triangle values to their three vertices. Useful for converting
        per-triangle data to per-vertex data.

        Parameters
        ----------
        tfunc : np.ndarray
            Float vector or matrix of shape (n_triangles,) or (n_triangles, N)
            with values at triangles.
        weighted : bool, default=False
            If False, weigh only by 1/3 (e.g., to compute vertex areas from
            triangle areas). If True, weigh by triangle area / 3 (e.g., to
            integrate a function defined on the triangles).

        Returns
        -------
        np.ndarray
            Function on vertices, shape (n_vertices,) or (n_vertices, N).

        Raises
        ------
        ValueError
            If tfunc length doesn't match number of triangles.
        """
        if self.t.shape[0] != tfunc.shape[0]:
            raise ValueError(
                "Error: length of tfunc needs to match number of triangles"
            )
        tfunca = np.array(tfunc)
        # make sure tfunc is 2D (even with only 1-dim input)
        if tfunca.ndim == 1:
            tfunca = tfunca[:, np.newaxis]
        if weighted:
            areas = self.tria_areas()[:, np.newaxis]  # to enable broadcasting
            tfunca = tfunca * areas
        vfunc = np.zeros((self.v.shape[0], tfunca.shape[1]))
        np.add.at(vfunc, self.t[:, 0], tfunca)
        np.add.at(vfunc, self.t[:, 1], tfunca)
        np.add.at(vfunc, self.t[:, 2], tfunca)
        return np.squeeze(vfunc / 3.0)

    def map_vfunc_to_tfunc(self, vfunc: np.ndarray) -> np.ndarray:
        """Map vertex function to triangles by averaging vertex values.

        Computes triangle values by averaging the values at the three vertices
        of each triangle.

        Parameters
        ----------
        vfunc : np.ndarray
            Float vector or matrix of shape (n_vertices,) or (n_vertices, N)
            with values at vertices.

        Returns
        -------
        np.ndarray
            Function on triangles, shape (n_triangles,) or (n_triangles, N).

        Raises
        ------
        ValueError
            If vfunc length doesn't match number of vertices.
        """
        if self.v.shape[0] != vfunc.shape[0]:
            raise ValueError("Error: length of vfunc needs to match number of vertices")
        vfunc = np.array(vfunc) / 3.0
        tfunc = np.sum(vfunc[self.t], axis=1)
        return tfunc

    def _construct_smoothing_matrix(self):
        """Construct the row-stochastic matrix for smoothing.

        Returns
        -------
        scipy.sparse.csc_matrix
            Sparse matrix for smoothing.
        """
        areas = self.vertex_areas()[:, np.newaxis]
        adj = self.adj_sym.copy()
        # binarize:
        adj.data = np.ones(adj.data.shape)
        # adjust columns to contain areas of vertex i
        adj2 = adj.multiply(areas)
        # rowsum is the area sum of 1-ring neighbors
        rowsum = np.sum(adj2, axis=1)
        # normalize rows to sum = 1, handling division by zero
        rowsum[rowsum == 0] = 1.0
        adj2 = adj2.multiply(1.0 / rowsum)
        return adj2

    def smooth_vfunc(self, vfunc, n=1):
        """Smooth the mesh or a vertex function iteratively.

        This is the most simple smoothing approach of a weighted
        average of the neighbors.

        .. deprecated::1.4.0
            `smooth_vfunc` will be removed in LaPy 2.0.0, use `smooth_laplace`
            or `smooth_taubin` instead. `smooth_laplace` with `lambda_=1`
            is equivalent to this function.

        Parameters
        ----------
        vfunc : array
            Float vector of values at vertices, if empty, use vertex coordinates.
        n : int, default=1
            Number of iterations for smoothing.

        Returns
        -------
        vfunc : array
            Smoothed surface vertex function.
        """
        warnings.warn(
            "TriaMesh.smooth_vfunc is deprecated, use smooth_laplace or smooth_taubin instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.smooth_laplace(vfunc=vfunc, n=n, lambda_=1.0)

    def smooth_laplace(
        self,
        vfunc: np.ndarray | None = None,
        n: int = 1,
        lambda_: float = 0.5,
        mat: sparse.csc_matrix | None = None,
    ) -> np.ndarray:
        """Smooth the mesh or a vertex function using Laplace smoothing.

        Applies iterative smoothing: v_new = (1-lambda)*v + lambda * M*v
        where M is the vertex-area weighted adjacency matrix.

        Parameters
        ----------
        vfunc : np.ndarray or None, default=None
            Float vector of values at vertices, shape (n_vertices,) or (n_vertices, N).
            If None, uses vertex coordinates for mesh smoothing.
        n : int, default=1
            Number of smoothing iterations.
        lambda_ : float, default=0.5
            Diffusion speed parameter in range [0, 1]. lambda_=1 reduces to
            weighted average of neighboring vertices, while smaller values
            preserve more of the original values.
        mat : scipy.sparse.csc_matrix or None, default=None
            Precomputed smoothing matrix. If None, constructs it internally.

        Returns
        -------
        np.ndarray
            Smoothed vertex function, same shape as input vfunc.

        Raises
        ------
        ValueError
            If vfunc length doesn't match number of vertices.
        """
        if vfunc is None:
            vfunc = self.v
        vfunc = np.array(vfunc)
        if self.v.shape[0] != vfunc.shape[0]:
            raise ValueError("Error: length of vfunc needs to match number of vertices")
        if mat is None:
            mat = self._construct_smoothing_matrix()
        for _ in range(n):
            vfunc = (1.0 - lambda_) * vfunc + lambda_ * mat.dot(vfunc)
        return vfunc

    def smooth_taubin(
        self,
        vfunc: np.ndarray | None = None,
        n: int = 1,
        lambda_: float = 0.5,
        mu: float = -0.53,
    ) -> np.ndarray:
        """Smooth the mesh or a vertex function using Taubin smoothing.

        Taubin smoothing alternates between shrinking (positive lambda) and
        expanding (negative mu) steps to reduce shrinkage while smoothing.
        This is a low-pass filter that better preserves mesh volume.

        Parameters
        ----------
        vfunc : np.ndarray or None, default=None
            Float vector of values at vertices, shape (n_vertices,) or (n_vertices, N).
            If None, uses vertex coordinates for mesh smoothing.
        n : int, default=1
            Number of smoothing iterations (each iteration applies both lambda
            and mu steps).
        lambda_ : float, default=0.5
            Positive diffusion parameter for shrinking step.
        mu : float, default=-0.53
            Negative diffusion parameter for expanding step. Should be negative
            and satisfy mu < -lambda to avoid instability.

        Returns
        -------
        np.ndarray
            Smoothed vertex function, same shape as input vfunc.

        Raises
        ------
        ValueError
            If vfunc length doesn't match number of vertices.

        References
        ----------
        Gabriel Taubin, "A Signal Processing Approach to Fair Surface Design",
        SIGGRAPH 1995.
        """
        if vfunc is None:
            vfunc = self.v
        vfunc = np.array(vfunc)
        if self.v.shape[0] != vfunc.shape[0]:
            raise ValueError("Error: length of vfunc needs to match number of vertices")
        mat = self._construct_smoothing_matrix()
        for _ in range(n):
            vfunc = self.smooth_laplace(vfunc, n=1, lambda_=lambda_, mat=mat)
            vfunc = self.smooth_laplace(vfunc, n=1, lambda_=mu, mat=mat)
        return vfunc

    def smooth_(self, n: int = 1) -> None:
        """Smooth the mesh iteratively in-place using Taubin smoothing.

        Parameters
        ----------
        n : int, default=1
            Number of smoothing iterations.
        """
        vfunc = self.smooth_taubin(self.v, n=n)
        self.v = vfunc
        return

    def level_length(self, vfunc: np.ndarray, level: float | np.ndarray) -> float | np.ndarray:
        """Compute the length of level sets.

        For a scalar function defined on mesh vertices, computes the total length
        of the curve where the function equals the specified level value(s).

        Parameters
        ----------
        vfunc : np.ndarray
            Scalar function values at vertices, shape (n_vertices,). Must be 1D.
        level : float or np.ndarray
            Level set value(s). Can be a single float or array of level values.

        Returns
        -------
        float or np.ndarray
            Length of level set. Returns float for single level, array for
            multiple levels.

        Raises
        ------
        ValueError
            If vfunc is not 1-dimensional.
            If no lengths could be computed.
        """
        if vfunc.ndim > 1:
            raise ValueError(f"vfunc needs to be 1-dim, but is {vfunc.ndim}-dim!")
        levels = np.atleast_1d(level)
        ll = np.empty((levels.size, 0))
        for lnext in levels:
            # get intersecting triangles
            intersect = vfunc[self.t] > lnext
            t_idx = np.where(
                np.logical_or(
                    np.sum(intersect, axis=1) == 1, np.sum(intersect, axis=1) == 2
                )
            )[0]
            # reduce to triangles that intersect with level:
            t_level = self.t[t_idx, :]
            intersect = intersect[t_idx, :]
            # trias have one vertex on one side and two on the other side of the
            # level set. Here we invert trias with two true values, so that single
            # vertex is true
            intersect[np.sum(intersect, axis=1) > 1, :] = np.logical_not(
                intersect[np.sum(intersect, axis=1) > 1, :]
            )
            # get idx within tria with single vertex:
            idx_single = np.argmax(intersect, axis=1)
            idx_o1 = (idx_single + 1) % 3
            idx_o2 = (idx_single + 2) % 3
            # get global idx
            gidx0 = t_level[np.arange(t_level.shape[0]), idx_single]
            gidx1 = t_level[np.arange(t_level.shape[0]), idx_o1]
            gidx2 = t_level[np.arange(t_level.shape[0]), idx_o2]
            # determine fraction along edges (for each triangle)
            xl1 = (lnext - vfunc[gidx0]) / (vfunc[gidx1] - vfunc[gidx0])
            xl2 = (lnext - vfunc[gidx0]) / (vfunc[gidx2] - vfunc[gidx0])
            # determine points on the two edges (for each triangle)
            p1 = (1 - xl1)[:, np.newaxis] * self.v[gidx0] + xl1[:, np.newaxis] * self.v[
                gidx1
            ]
            p2 = (1 - xl2)[:, np.newaxis] * self.v[gidx0] + xl2[:, np.newaxis] * self.v[
                gidx2
            ]
            # compute edge length between the points
            lls = np.sqrt(((p1 - p2) ** 2).sum(1))
            ll = np.append(ll, lls.sum())
        if ll.size > 1:
            return ll
        elif ll.size > 0:
            return ll[0]
        else:
            raise ValueError("No lengths computed, should never get here.")

    @staticmethod
    def __reduce_edges_to_path(
        edges: np.ndarray,
        start_idx: int | None = None,
        get_edge_idx: bool = False,
    ) -> list | tuple[list, list]:
        """Reduce undirected unsorted edges to ordered path(s).

        Converts unordered edge pairs into ordered paths by finding traversals.
        Handles single open paths, closed loops, and multiple disconnected components.
        Always returns lists to handle multiple components uniformly.

        Parameters
        ----------
        edges : np.ndarray
            Array of shape (n, 2) with pairs of positive integer node indices
            representing undirected edges. Node indices can have gaps (i.e., not all
            indices from 0 to max need to appear in the edges). Only nodes that
            actually appear in edges are processed.
        start_idx : int or None, default=None
            Preferred start node. If None, selects an endpoint (degree 1) automatically,
            or arbitrary node for closed loops. Must be a node that appears in edges.
        get_edge_idx : bool, default=False
            If True, also return edge indices for each path segment.

        Returns
        -------
        paths : list[np.ndarray]
            List of ordered paths, one per connected component. For closed loops,
            first node is repeated at end.
        edge_idxs : list[np.ndarray]
            List of edge index arrays, one per component. Only returned if
            get_edge_idx=True.

        Raises
        ------
        ValueError
            If start_idx is invalid.
            If graph has degree > 2 (branching or self-intersections).
        """
        edges = np.array(edges)
        if edges.shape[0] == 0:
            return ([], []) if get_edge_idx else []

        # Build adjacency matrix
        i = np.column_stack((edges[:, 0], edges[:, 1])).reshape(-1)
        j = np.column_stack((edges[:, 1], edges[:, 0])).reshape(-1)
        dat = np.ones(i.shape)
        n = edges.max() + 1
        adj_matrix = sparse.csr_matrix((dat, (i, j)), shape=(n, n))
        degrees = np.asarray(adj_matrix.sum(axis=1)).ravel()

        # Find connected components
        n_comp, labels = sparse.csgraph.connected_components(adj_matrix, directed=False)

        # Build edge index lookup matrix
        edge_dat = np.arange(edges.shape[0]) + 1
        edge_dat = np.column_stack((edge_dat, edge_dat)).reshape(-1)
        eidx_matrix = sparse.csr_matrix((edge_dat, (i, j)), shape=(n, n))

        paths = []
        edge_idxs = []

        for comp_id in range(n_comp):
            comp_nodes = np.where(labels == comp_id)[0]
            if len(comp_nodes) == 0:
                continue

            comp_degrees = degrees[comp_nodes]

            # Skip isolated nodes (degree 0) that exist only due to matrix sizing.
            # When edges use indices [0, 5, 10], we create a matrix of size 11x11.
            # Indices [1,2,3,4,6,7,8,9] don't appear in any edges (have degree 0).
            # connected_components treats each as a separate component, but they're
            # not real nodes - just phantom entries from sizing matrix to max_index+1.
            if np.all(comp_degrees == 0):
                continue

            # SAFETY CHECK: Reject graphs with nodes of degree > 2
            # This single check catches all problematic cases:
            # - Branching structures (Y-shape, star graph)
            # - Self-intersections (figure-8, etc.)
            # - Trees with multiple endpoints
            # Valid graphs have only degree 1 (endpoints) and degree 2 (path nodes)
            max_degree = comp_degrees.max()
            if max_degree > 2:
                high_degree_nodes = comp_nodes[comp_degrees > 2]
                raise ValueError(
                    f"Component {comp_id}: found {len(high_degree_nodes)} node(s) with degree > 2 "
                    f"(max degree: {max_degree}). Degrees: {np.sort(comp_degrees)}. "
                    f"This indicates branching or self-intersecting structure. "
                    f"Only simple paths and simple closed loops are supported."
                )

            # Determine if closed loop: all nodes have degree 2
            is_closed = np.all(comp_degrees == 2)

            # For closed loops: break one edge to convert to open path
            if is_closed:
                # Pick start node
                if start_idx is not None and start_idx in comp_nodes:
                    start = start_idx
                else:
                    start = comp_nodes[0]

                # Find neighbors of start node to break one edge
                neighbors = adj_matrix[start, :].nonzero()[1]
                neighbors_in_comp = [n for n in neighbors if n in comp_nodes]

                if len(neighbors_in_comp) < 2:
                    raise ValueError(f"Component {comp_id}: Closed loop node {start} should have 2 neighbors")

                # Break the edge from start to first neighbor (temporarily)
                adj_matrix = adj_matrix.tolil()
                break_neighbor = neighbors_in_comp[0]
                adj_matrix[start, break_neighbor] = 0
                adj_matrix[break_neighbor, start] = 0
                adj_matrix = adj_matrix.tocsr()

                # Update degrees after breaking edge
                degrees = np.asarray(adj_matrix.sum(axis=1)).ravel()
                comp_degrees = degrees[comp_nodes]

            # Now handle as open path (both originally open and converted closed loops)
            endpoints = comp_nodes[comp_degrees == 1]

            if len(endpoints) != 2:
                raise ValueError(
                    f"Component {comp_id}: Expected 2 endpoints after breaking loop, found {len(endpoints)}"
                )

            # Select start node
            if is_closed:
                # For closed loops, start is already selected above
                pass
            elif start_idx is not None and start_idx in endpoints:
                start = start_idx
            else:
                # For originally open paths, pick first endpoint
                start = endpoints[0]

            # Use shortest_path to order nodes by distance from start
            dist = sparse.csgraph.shortest_path(adj_matrix, indices=start, directed=False)

            if np.isinf(dist[comp_nodes]).any():
                raise ValueError(f"Component {comp_id} is not fully connected.")

            # Sort nodes by distance from start
            path = comp_nodes[dist[comp_nodes].argsort()]

            # For closed loops: append start again to close the loop
            if is_closed:
                path = np.append(path, path[0])

            paths.append(path)

            # Get edge indices if requested
            if get_edge_idx:
                ei = path[:-1]
                ej = path[1:]
                eidx = np.asarray(eidx_matrix[ei, ej] - 1).ravel()
                edge_idxs.append(eidx)

        # Always return lists
        if get_edge_idx:
            return paths, edge_idxs
        else:
            return paths

    def extract_level_paths(
        self,
        vfunc: np.ndarray,
        level: float,
    ) -> list[polygon.Polygon]:
        """Extract level set paths as Polygon objects with triangle/edge metadata.

        For a given scalar function on mesh vertices, extracts all paths where
        the function equals the specified level. Returns polygons with embedded
        metadata about which triangles and edges were intersected.
        Handles single open paths, closed loops, and multiple disconnected components.

        Parameters
        ----------
        vfunc : np.ndarray
            Scalar function values at vertices, shape (n_vertices,). Must be 1D.
        level : float
            Level set value to extract.

        Returns
        -------
        list[polygon.Polygon]
            List of Polygon objects, one for each connected level curve component.
            Each polygon has the following additional attributes:

            - points : np.ndarray of shape (n_points, 3) - 3D coordinates on level curve
            - closed : bool - whether the curve is closed
            - tria_idx : np.ndarray of shape (n_segments,) - triangle index for each segment
            - edge_vidx : np.ndarray of shape (n_points, 2) - mesh vertex indices for edge
            - edge_bary : np.ndarray of shape (n_points,) - barycentric coordinate [0,1]
              along edge where level set intersects (0=first vertex, 1=second vertex)

        Raises
        ------
        ValueError
            If vfunc is not 1-dimensional.
        """
        if vfunc.ndim > 1:
            raise ValueError(f"vfunc needs to be 1-dim, but is {vfunc.ndim}-dim!")

        # Get intersecting triangles
        intersect = vfunc[self.t] > level
        t_idx = np.where(
            np.logical_or(
                np.sum(intersect, axis=1) == 1, np.sum(intersect, axis=1) == 2
            )
        )[0]

        if t_idx.size == 0:
            return []

        # Reduce to triangles that intersect with level
        t_level = self.t[t_idx, :]
        intersect = intersect[t_idx, :]

        # Invert triangles with two true values so single vertex is true
        intersect[np.sum(intersect, axis=1) > 1, :] = np.logical_not(
            intersect[np.sum(intersect, axis=1) > 1, :]
        )

        # Get index within triangle with single vertex
        idx_single = np.argmax(intersect, axis=1)
        idx_o1 = (idx_single + 1) % 3
        idx_o2 = (idx_single + 2) % 3

        # Get global indices
        gidx0 = t_level[np.arange(t_level.shape[0]), idx_single]
        gidx1 = t_level[np.arange(t_level.shape[0]), idx_o1]
        gidx2 = t_level[np.arange(t_level.shape[0]), idx_o2]

        # Sort edge indices (rows are triangles, cols are the two vertex ids)
        # This creates unique edge identifiers
        gg1 = np.sort(
            np.concatenate((gidx0[:, np.newaxis], gidx1[:, np.newaxis]), axis=1)
        )
        gg2 = np.sort(
            np.concatenate((gidx0[:, np.newaxis], gidx2[:, np.newaxis]), axis=1)
        )

        # Concatenate all edges and get unique ones
        gg = np.concatenate((gg1, gg2), axis=0)
        gg_unique = np.unique(gg, axis=0)

        # Generate level set intersection points for unique edges
        # Barycentric coordinate (0=first vertex, 1=second vertex)
        xl = ((level - vfunc[gg_unique[:, 0]])
              / (vfunc[gg_unique[:, 1]] - vfunc[gg_unique[:, 0]]))

        # 3D points on unique edges
        p = ((1 - xl)[:, np.newaxis] * self.v[gg_unique[:, 0]]
             + xl[:, np.newaxis] * self.v[gg_unique[:, 1]])

        # Fill sparse matrix with new point indices (+1 to distinguish from zero)
        a_mat = sparse.csc_matrix(
            (np.arange(gg_unique.shape[0]) + 1, (gg_unique[:, 0], gg_unique[:, 1]))
        )

        # For each triangle, create edge pair via lookup in matrix
        edge_idxs = (np.concatenate((a_mat[gg1[:, 0], gg1[:, 1]],
                                      a_mat[gg2[:, 0], gg2[:, 1]]), axis=0).T - 1)

        # Reduce edges to paths (returns list of paths for multiple components)
        paths, path_edge_idxs = self._TriaMesh__reduce_edges_to_path(edge_idxs, get_edge_idx=True)

        # Build polygon objects for each connected component
        polygons = []
        for path_nodes, path_edge_idx in zip(paths, path_edge_idxs, strict=True):
            # Get 3D coordinates for this path
            poly_v = p[path_nodes, :]

            # Remove duplicate vertices (when levelset hits a vertex almost perfectly)
            dd = ((poly_v[0:-1, :] - poly_v[1:, :]) ** 2).sum(1)
            dd = np.append(dd, 1)  # Never delete last node
            eps = 0.000001
            keep_ids = dd > eps
            poly_v = poly_v[keep_ids, :]

            # Get triangle indices for each edge
            poly_tria_idx = t_idx[path_edge_idx[keep_ids[:-1]]]

            # Get edge vertex indices
            poly_edge_vidx = gg_unique[path_nodes[keep_ids], :]

            # Get barycentric coordinates
            poly_edge_bary = xl[path_nodes[keep_ids]]

            # Create polygon with metadata
            # Use closed=None to let Polygon auto-detect based on endpoints
            # This will automatically remove duplicate endpoint if present
            n_points_before = poly_v.shape[0]
            poly = polygon.Polygon(poly_v, closed=None)
            n_points_after = poly.points.shape[0]

            # If Polygon removed duplicate endpoint, adjust metadata
            if n_points_after < n_points_before:
                # Remove last entry from metadata to match polygon points
                poly_edge_vidx = poly_edge_vidx[:n_points_after]
                poly_edge_bary = poly_edge_bary[:n_points_after]

            poly.tria_idx = poly_tria_idx
            poly.edge_vidx = poly_edge_vidx
            poly.edge_bary = poly_edge_bary

            polygons.append(poly)

        return polygons

    def level_path(
        self,
        vfunc: np.ndarray,
        level: float,
        get_tria_idx: bool = False,
        get_edges: bool = False,
        n_points: int | None = None,
    ) -> tuple[np.ndarray, ...]:
        """Extract levelset of vfunc at a specific level as a path of 3D points.

        For a given real-valued scalar map on the surface mesh (vfunc), this
        function computes the edges that intersect with a given level set (level).
        It then finds the point on each mesh edge where the level set intersects.
        The points are sorted and returned as an ordered array of 3D coordinates
        together with the length of the level set path.

        This implementation uses extract_level_paths internally to compute the
        level set, ensuring consistent handling of closed loops and metadata.

        .. note::

            Works for level sets that represent a single path or closed loop.
            For closed loops, the first and last points are identical (duplicated)
            so you can detect closure via ``np.allclose(path[0], path[-1])``.
            For open paths, the path has two distinct endpoints.
            This function is kept mainly for backward compatibility.

            **For more advanced use cases, consider using extract_level_paths() directly:**

            - Multiple disconnected components: extract_level_paths returns all components
            - Custom resampling: Get Polygon objects and use Polygon.resample() directly
            - Metadata analysis: Access triangle indices and edge information per component
            - Closed loop detection: Polygon objects have a ``closed`` attribute

        Parameters
        ----------
        vfunc : np.ndarray
            Scalar function values at vertices, shape (n_vertices,). Must be 1D.
        level : float
            Level set value to extract.
        get_tria_idx : bool, default=False
            If True, also return array of triangle indices for each path segment.
            For closed loops with n points (including duplicate), returns n-1 triangle
            indices. For open paths with n points, returns n-1 triangle indices.
        get_edges : bool, default=False
            If True, also return arrays of vertex indices (i,j) and relative
            positions for each 3D point along the intersecting edge.
        n_points : int or None, default=None
            If specified, resample level set into n equidistant points. Cannot
            be combined with get_tria_idx=True or get_edges=True.

        Returns
        -------
        path : np.ndarray
            Array of shape (n, 3) containing 3D coordinates of vertices on the
            level path. For closed loops, the last point duplicates the first point,
            so ``np.allclose(path[0], path[-1])`` will be True.
        length : float
            Total length of the level set path. For closed loops, includes the
            closing segment from last to first point.
        tria_idx : np.ndarray
            Array of triangle indices for each segment on the path, shape (n-1,)
            where n is the number of points (including duplicate for closed loops).
            Only returned if get_tria_idx is True.
        edges_vidxs : np.ndarray
            Array of shape (n, 2) with vertex indices (i,j) for each 3D point,
            defining the vertices of the original mesh edge intersecting the
            level set at this point. Only returned if get_edges is True.
        edges_relpos : np.ndarray
            Array of floats defining the relative position for each 3D point
            along the edges of the original mesh (defined by edges_vidxs).
            Value 0 corresponds to first vertex, 1 to second vertex.
            The 3D point is computed as: (1 - relpos) * v_i + relpos * v_j.
            Only returned if get_edges is True.

        Raises
        ------
        ValueError
            If vfunc is not 1-dimensional.
            If multiple disconnected level paths are found (use extract_level_paths).
            If n_points is combined with get_tria_idx=True.
            If n_points is combined with get_edges=True.

        See Also
        --------
        extract_level_paths : Extract multiple disconnected level paths as Polygon objects.
            Recommended for advanced use cases with multiple components, custom resampling,
            or when you need explicit closed/open information.

        Examples
        --------
        >>> # Extract a simple level curve
        >>> path, length = mesh.level_path(vfunc, 0.5)
        >>> print(f"Path has {path.shape[0]} points, length {length:.2f}")

        >>> # Check if the path is closed
        >>> is_closed = np.allclose(path[0], path[-1])
        >>> print(f"Path is closed: {is_closed}")

        >>> # Get triangle indices for each segment
        >>> path, length, tria_idx = mesh.level_path(vfunc, 0.5, get_tria_idx=True)

        >>> # Get complete edge metadata
        >>> path, length, edges, relpos = mesh.level_path(vfunc, 0.5, get_edges=True)

        >>> # Resample to fixed number of points
        >>> path, length = mesh.level_path(vfunc, 0.5, n_points=100)

        >>> # For multiple components, use extract_level_paths instead
        >>> curves = mesh.extract_level_paths(vfunc, 0.5)
        >>> for i, curve in enumerate(curves):
        >>>     print(f"Component {i}: {curve.points.shape[0]} points, closed={curve.closed}")
        """
        # Use extract_level_paths to get polygons
        curves = self.extract_level_paths(vfunc, level)

        # level_path expects single component
        if len(curves) != 1:
            raise ValueError(
                f"Found {len(curves)} disconnected level curves. "
                f"Use extract_level_paths() for multiple components."
            )

        # Get the single curve
        curve = curves[0]

        # Extract data from polygon
        path3d = curve.points
        edges_vidxs = curve.edge_vidx
        edges_relpos = curve.edge_bary

        # Compute length using polygon's length method
        length = curve.length()

        # For closed loops, duplicate the last point to match first point
        # This allows users to detect closure via np.allclose(path[0], path[-1])
        # and maintains backward compatibility with the original level_path behavior
        if curve.closed:
            path3d = np.vstack([path3d, path3d[0:1]])
            edges_vidxs = np.vstack([edges_vidxs, edges_vidxs[0:1]])
            edges_relpos = np.append(edges_relpos, edges_relpos[0])

        # Handle optional resampling
        if n_points:
            if get_tria_idx:
                raise ValueError("n_points cannot be combined with get_tria_idx=True.")
            if get_edges:
                raise ValueError("n_points cannot be combined with get_edges=True.")
            poly = polygon.Polygon(path3d, closed=curve.closed)
            path3d = poly.resample(n_points=n_points, n_iter=3, inplace=False)
            path3d = path3d.get_points()
            if curve.closed:
                path3d = np.vstack([path3d, path3d[0:1]])

        # Build return tuple based on options
        if get_tria_idx:
            tria_idx = curve.tria_idx
            if get_edges:
                return path3d, length, tria_idx, edges_vidxs, edges_relpos
            else:
                return path3d, length, tria_idx
        else:
            if get_edges:
                return path3d, length, edges_vidxs, edges_relpos
            else:
                return path3d, length


