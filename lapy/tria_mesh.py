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
    t : array_like
        List of lists of 3 int of indices (>= 0) into ``v`` array
        Ordering is important: All triangles should be
        oriented in the same way (counter-clockwise, when
        looking from above).
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

    Notes
    -----
    The class has static class methods to read triangle meshes from FreeSurfer,
    OFF, and VTK file formats.

    When 2D vertices are provided, they are internally padded with z=0 to create
    3D vertices. This allows all geometric operations to work correctly while
    maintaining compatibility with 2D mesh data.
    """

    def __init__(self, v, t, fsinfo=None):
        self.v = np.array(v)
        self.t = np.array(t)
        # transpose if necessary
        if self.v.shape[0] < self.v.shape[1]:
            self.v = self.v.T
        if self.t.shape[0] < self.t.shape[1]:
            self.t = self.t.T
        # Check a few things
        vnum = np.max(self.v.shape)
        if self.t.size >0 and np.max(self.t) >= vnum:
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

    def is_2d(self):
        """Check if the mesh was created with 2D vertices.

        Returns
        -------
        bool
            True if mesh was created with 2D vertices, False otherwise.
        """
        return self._is_2d

    def get_vertices(self, original_dim=False):
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

    def write_vtk(self, filename):
        """Save as VTK file.

        Parameters
        ----------
        filename : str
            Filename to save to.
        """
        io.write_vtk(self, filename)

    def write_fssurf(self, filename, image=None):
        """Save as Freesurfer Surface Geometry file (wrap Nibabel).

        Parameters
        ----------
        filename : str
            Filename to save to.
        image : str, object, None
            Path to image, nibabel image object, or image header. If specified, the vertices
            are assumed to be in voxel coordinates and are converted to surface RAS (tkr)
            coordinates before saving. The expected order of coordinates is (x, y, z)
            matching the image voxel indices in nibabel.

        Notes
        -----
        The surface RAS (tkr) transform is obtained from a header that implements
        ``get_vox2ras_tkr()`` (e.g., ``MGHHeader``). For other header types (NIfTI1/2,
        Analyze/SPM, etc.), we attempt conversion via ``MGHHeader.from_header``.
        """
        io.write_fssurf(self, filename, image=image)

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

    def is_oriented(self):
        """Check if triangle mesh is oriented.

        True if all triangles are oriented counter-clockwise, when looking from
        above. Operates only on triangles.

        Returns
        -------
        bool
            True if ``max(adj_directed)=1``.
        """
        return np.max(self.adj_dir.data) == 1

    def euler(self):
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

    def tria_areas(self):
        """Compute the area of triangles using Heron's formula.

        `Heron's formula <https://en.wikipedia.org/wiki/Heron%27s_formula>`_
        computes the area of a triangle by using the three edge lengths.

        Returns
        -------
        areas : array
            Array with areas of each triangle.
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

    def area(self):
        """Compute the total surface area of triangle mesh.

        Returns
        -------
        float
            Total surface area.
        """
        areas = self.tria_areas()
        return np.sum(areas)

    def volume(self):
        """Compute the volume of closed triangle mesh, summing tetrahedra at origin.

        Returns
        -------
        vol : float
            Total enclosed volume.
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

    def vertex_degrees(self):
        """Compute the vertex degrees (number of edges at each vertex).

        Returns
        -------
        vdeg : array
            Array of vertex degrees.
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

    def tria_normals(self):
        """Compute triangle normals.

        Ordering of triangles is important: counterclockwise when looking.

        Returns
        -------
        n : array of shape (n_triangles, 3)
            Triangle normals.
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
        # lni = np.divide(1.0, ln)
        # n[:, 0] *= lni
        # n[:, 1] *= lni
        # n[:, 2] *= lni
        return n

    def vertex_normals(self):
        """Compute vertex normals.

        get_vertex_normals(v,t) computes vertex normals
            Triangle normals around each vertex are averaged, weighted
            by the angle that they contribute.
            Ordering is important: counterclockwise when looking
            at the triangle from above.

        Returns
        -------
        n : array of shape (n_triangles, 3)
            Vertex normals.
        """
        if not self.is_oriented():
            raise ValueError(
                "Error: Vertex normals are meaningless for un-oriented triangle meshes!"
            )
        import sys

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
        # lni = np.divide(1.0, ln)
        # n[:, 0] *= lni
        # n[:, 1] *= lni
        # n[:, 2] *= lni
        return n

    def has_free_vertices(self):
        """Check if the vertex list has more vertices than what is used in tria.

        Returns
        -------
        bool
            Whether vertex list has more vertices or not.
        """
        vnum = np.max(self.v.shape)
        vnumt = len(np.unique(self.t.reshape(-1)))
        return vnum != vnumt

    def tria_qualities(self):
        """Compute triangle quality for each triangle in mesh where.

        q = 4 sqrt(3) A / (e1^2 + e2^2 + e3^2 )
        where A is the triangle area and ei the edge length of the three edges.
        Constants are chosen so that q=1 for the equilateral triangle.

        .. note::

            This measure is used by FEMLAB and can also be found in:
            R.E. Bank, PLTMG ..., Frontiers in Appl. Math. (7), 1990.

        Returns
        -------
        array
            Array with triangle qualities.
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
        return q / es

    def boundary_loops(self):
        """Compute a tuple of boundary loops.

        Meshes can have 0 or more boundary loops, which are cycles in the directed
        adjacency graph of the boundary edges.
        Works on trias only. Could fail if loops are connected via a single
        vertex (like a figure 8). That case needs debugging.

        Returns
        -------
        loops : list of list
            List of lists with boundary loops.
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

    def connected_components(self):
        """Compute connected components of the mesh.

        Returns
        -------
        n_components : int
            Number of connected components.
        labels : array
            Label array of shape (n_vertices,) where labels[i] is the component
            ID of vertex i. Component IDs are integers from 0 to n_components-1.
        """
        from scipy.sparse.csgraph import connected_components

        return connected_components(self.adj_sym, directed=False)

    def keep_largest_connected_component_(self, clean=True):
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
        vkeep : array or None
            Indices (from original list) of kept vertices if clean=True, else None.
        vdel : array or None
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

    def centroid(self):
        """Compute centroid of triangle mesh as a weighted average of triangle centers.

        The weight is determined by the triangle area.
        (This could be done much faster if a FEM lumped mass matrix M is
        already available where this would be M*v, because it is equivalent
        with averaging vertices weighted by vertex area)

        Returns
        -------
        centroid : float
            The centroid of the mesh.
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

    def edges(self, with_boundary=False):
        """Compute vertices and adjacent triangle ids for each edge.

        Parameters
        ----------
        with_boundary : bool
            Also work on boundary half edges, default ignore.

        Returns
        -------
        vids : array
            Column array with starting and end vertex for each unique inner edge.
        tids : array
            2 column array with triangle containing the half edge
            from vids[0,:] to vids [1,:] in first column and the
            neighboring triangle in the second column.
        bdrvids : array
            If with_boundary is true: 2 column array with each
            boundary half-edge.
        bdrtids : array
            If with_boundary is true: 1 column array with the
            associated triangle to each boundary edge.
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
        vids = np.array(np.nonzero(adjtriu1)).T
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

    def curvature(self, smoothit=3):
        """Compute various curvature values at vertices.

        .. note::

            For the algorithm see e.g.
            Pierre Alliez, David Cohen-Steiner, Olivier Devillers,
            Bruno Levy, and Mathieu Desbrun.
            Anisotropic Polygonal Remeshing.
            ACM Transactions on Graphics, 2003.

        Parameters
        ----------
        smoothit : int
            Smoothing iterations on vertex functions.

        Returns
        -------
        u_min : array of shape (vnum, 3)
            Minimal curvature directions.
        u_max : array of shape (vnum, 3)
            Maximal curvature directions.
        c_min : array
            Minimal curvature.
        c_max : array
            Maximal curvature.
        c_mean : array
            Mean curvature ``(c_min + c_max) / 2.0m``.
        c_gauss : array
           Gauss curvature ``c_min * c_maxm``.
        normals : array of shape (vnum, 3)
           Normals.
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

    def curvature_tria(self, smoothit=3):
        """Compute min and max curvature and directions (orthogonal and in tria plane).

        First we compute these values on vertices and then smooth
        there. Finally, they get mapped to the trias (averaging) and projected onto
        the triangle plane, and orthogonalized.

        Parameters
        ----------
        smoothit : int
            Number of smoothing iterations for curvature computation on vertices.

        Returns
        -------
        u_min : array
            Min curvature direction on triangles.
        u_max : array
            Max curvature direction on triangles.
        c_min : array
            Min curvature on triangles.
        c_max : array
            Max curvature on triangles.
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

    def normalize_(self):
        """Normalize TriaMesh to unit surface area and centroid at the origin.

        Modifies the vertices.
        """
        centroid, area = self.centroid()
        if area <= 0:
            raise ValueError("Mesh surface area must be positive to normalize.")
        self.v = (1.0 / np.sqrt(area)) * (self.v - centroid)

    def rm_free_vertices_(self):
        """Remove unused (free) vertices.

        Free vertices are vertices that are not used in any triangle.
        They can produce problems when constructing, e.g., Laplace matrices.

        Modifies the mesh in-place.

        Returns
        -------
        vkeep : array
            Indices (from original list) of kept vertices.
        vdel : array
            Indices of deleted (unused) vertices.
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

    def refine_(self, it=1):
        """Refine the triangle mesh by placing new vertex on each edge midpoint.

        Thus creates 4 similar triangles from one parent triangle.
        Modifies mesh in place.

        Parameters
        ----------
        it : int
            Number of iterations.
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

    def normal_offset_(self, d):
        """Move vertices along normal by distance ``d``.

        normal_offset(d) moves vertices along normal by distance d

        Parameters
        ----------
        d : int | array
            Move distance.
        """
        n = self.vertex_normals()
        vn = self.v + d * n
        self.v = vn
        # no need to re-init, only changed vertices

    def orient_(self):
        """Re-orient triangles of manifold mesh to be consistent.

        Re-orients triangles of manifold mesh to be consistent, so that vertices are
        listed counter-clockwise, when looking from above (outside).

        Algorithm:

        * Construct list for each half-edge with its triangle and edge direction
        * Drop boundary half-edges and find half-edge pairs
        * Construct sparse matrix with triangle neighbors, with entry 1 for opposite
          half edges and -1 for parallel half-edges (normal flip across this edge)
        * Flood mesh from first tria using triangle neighbor matrix and keeping track of
          sign
        * When flooded, negative sign for a triangle indicates it needs to be flipped
        * If global volume is negative, flip everything (first tria was wrong)

        Returns
        -------
        flipped : int
            Number of trias flipped.
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

    def map_tfunc_to_vfunc(self, tfunc, weighted=False):
        """Map tria function to vertices by attributing 1/3 to each vertex of triangle.

        Uses vertices and trias.

        Parameters
        ----------
        tfunc : array
            Float vector or matrix (#t x N) of values at vertices.
        weighted : bool, default=False
            False, weigh only by 1/3, e.g. to compute
            vertex areas from tria areas
            True, weigh by triangle area / 3, e.g. to
            integrate a function defined on the trias,
            for example integrating the "one" function
            will also yield the vertex areas.

        Returns
        -------
        vfunc : array
            Function on vertices vector or matrix (#v x N).
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

    def map_vfunc_to_tfunc(self, vfunc):
        """Map vertex function to triangles by attributing 1/3 to each.

        Uses number of vertices and trias.

        Parameters
        ----------
        vfunc : array
            Float vector or matrix (#t x N) of values at vertices.

        Returns
        -------
        tfunc : array
            Function on trias vector or matrix (#t x N).
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
        csc_matrix
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

    def smooth_laplace(self, vfunc=None, n=1, lambda_=0.5, mat=None):
        """Smooth the mesh or a vertex function using Laplace smoothing.

        Applies v_new = (1-lambda)*v + lambda * M*v
        where M is the vertex-area weighted adjacency matrix.

        Parameters
        ----------
        vfunc : array or None
            Float vector of values at vertices, if None, use vertex coordinates.
        n : int
            Number of iterations.
        lambda_ : float
            Diffusion speed parameter. lambda=1 reduces to the most simple case
            of a weighted average of the values at neighboring vertices,
            while smaller lambdas will include the value at the current vertex.
        mat : csc_matrix, None
            Precomputed smoothing matrix.

        Returns
        -------
        vfunc : array
            Smoothed surface vertex function.
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

    def smooth_taubin(self, vfunc=None, n=1, lambda_=0.5, mu=-0.53):
        """Smooth the mesh or a vertex function using Taubin smoothing.

        Taubin smoothing alternates between shrinking (positive lambda) and
        expanding (negative mu) steps to avoid shrinkage of the mesh.

        Parameters
        ----------
        vfunc : array or None
            Float vector of values at vertices, if None, use vertex coordinates.
        n : int
            Number of iterations (each iteration includes one shrink and one expand step).
        lambda_ : float
            Shrinking factor (0 < lambda < 1).
        mu : float
            Expanding factor (negative, ``|mu| > lambda``).

        Returns
        -------
        vfunc : array
            Smoothed surface vertex function.
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

    def smooth_(self, n=1):
        """Smooth the mesh iteratively in-place using Taubin smoothing.

        Parameters
        ----------
        n : int
            Smoothing iterations.
        """
        vfunc = self.smooth_taubin(self.v, n=n)
        self.v = vfunc
        return

    def level_length(self, vfunc, level):
        """Compute the length of level sets.

        Parameters
        ----------
        vfunc : array
            Float vector of values at vertices (here only scalar function 1D).
        level : float | array
            Level set value or array of level values.

        Returns
        -------
        length : float | array
            Length of level set (or array of lengths).
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
    def __reduce_edges_to_path(edges, start_idx=None, get_edge_idx=False):
        """Reduce undirected unsorted edges (index pairs) to path (index array).

        Parameters
        ----------
        edges : list of shape (n, 2)
            Pairs of positive integer node indices representing a undirected edges.
            All indices from 0 to max(edges) must be used and graph needs to be
            connected. Nodes cannot have more than 2 neighbors. Also needs exactly
            two endnodes (nodes with only one neighbor). Tip for closed loops, cut
            it open before passing to this function by removing one edge.
        start_idx : int, default: None
            Node with only one neighbor to start path. Optional, if not passed the
            node with the smaller index will be selected as start point.
        get_edge_idx : bool, default: False
            Also return index of edge into edges for each path segment. This list
            has length n, while path has length n+1.

        Returns
        -------
        path: array
            Array of length n+1 containing indices as single path from start to
            endpoint.
        edge_idx: array
            Array of length n containing corresponding edge idx into edges for each
            path segment. Only passed if get_edges_idx is True.
        """
        from scipy.sparse.csgraph import shortest_path

        # Extract node indices and create a sparse adjacency matrix
        edges = np.array(edges)
        i = np.column_stack((edges[:, 0], edges[:, 1])).reshape(-1)
        j = np.column_stack((edges[:, 1], edges[:, 0])).reshape(-1)
        dat = np.ones(i.shape)
        n = edges.max() + 1
        adj_matrix = sparse.csr_matrix((dat, (i, j)), shape=(n, n))
        # Find the degree of each node, sum over rows to get degree
        degrees = np.asarray(adj_matrix.sum(axis=1)).ravel()
        endpoints = np.where(degrees == 1)[0]
        if len(endpoints) != 2:
            raise ValueError(
                "The graph does not have exactly two endpoints; invalid input."
            )
        if not start_idx:
            start_idx = endpoints[0]
        else:
            if not np.isin(start_idx, endpoints):
                raise ValueError(
                    f"start_idx {start_idx} must be one of the endpoints {endpoints}."
                )
        # Traverse the graph by computing shortest path
        dist_matrix = shortest_path(
            csgraph=adj_matrix,
            directed=False,
            indices=start_idx,
            return_predecessors=False,
        )
        if np.isinf(dist_matrix).any():
            raise ValueError(
                "Ensure connected graph with indices from 0 to max_idx without gaps."
            )
        # sort indices according to distance form start
        path = dist_matrix.argsort()
        # get edge idx of each segment from original list
        enum = edges.shape[0]
        dat = np.arange(enum) + 1
        dat = np.column_stack((dat, dat)).reshape(-1)
        eidx_matrix = sparse.csr_matrix((dat, (i, j)), shape=(n, n))
        ei = path[0:-1]
        ej = path[(np.arange(path.size - 1) + 1)]
        eidx = np.asarray(eidx_matrix[ei, ej] - 1).ravel()
        if get_edge_idx:
            return path, eidx
        else:
            return path


    def level_path(self, vfunc, level, get_tria_idx=False, get_edges=False,
                   n_points=None):
        """Extract levelset of vfund at a specific level as a path of 3D points.

        For a given real-valued scalar map on the surface mesh (vfunc) this
        function computes the edges that intersect with a given level set (level).
        It then finds the point on each mesh edge where the level set intersects.
        The points are sorted and returned as an ordered array of 3D coordinates
        together with the length of the level set path.

        Note: Only works for level sets that represent a single non-intersecting
        path with exactly one start and one endpoint (e.g. not closed)!

        Additional options: get_tria_idx and get_edges when True will also
        return an array of triangle ids for each path segment, defining the
        triangle i, where the path from i to i+1 passes through (so for a path
        with n points, these will be n-1 triangle ids). Furthermore, an array
        of edge vertex indices of shape (n,2) can be obtained defining the two
        vertices of the intersecting edge in the original mesh for each 3D point
        on the path. A second array is returned, defining the relative position
        of the intersecting point along this edge as a float from 0 (start vertex)
        to 1 (end vertex). This information can, for example, be useful for
        interpolating a second surface map at the new path point coordinates.
        Neither of this information is available when n_points is used to resample
        the path into n_points equidistant new points as the association to edges
        or triangles in the original mesh is lost.

        Parameters
        ----------
        vfunc : array
            Float vector of values at vertices (here only scalar function 1D).
        level : float
            Level set value.
        get_tria_idx : bool, default: False
            Also return a list of triangle indices for each edge, default False.
        get_edges : bool, default: False
            Also return a list of two vertex indices (i,j) for each 3D point and
            a list of the relative position defining the 3D point along that
            edge (i,j) from the original mesh, default False.
        n_points : int
            Resample level set into n equidistant points. Cannot be combined
            with get_tria_idx=True nor with get_edges=True.

        Returns
        -------
        path: array
            Array of shape (n,3) containing 3D coordinates of vertices on a level path.
        length : float
            Length of the level set.
        tria_idx : array
            Array of triangle index for each segment on the path (length n-1
            if the path is length n). Will only be returned, if get_tria_idx is True.
        edges_vidxs : array
            Array of shape (n,2) of vertex indices (i,j) for each 3D point, defining
            the vertices of the original mesh of the edge intersecting the level set
            at this point. Will only be returned if get_edges is True.
        edges_relpos: array
            Array of floats defining the relative position for each 3D point along
            the edges of the original mesh (defined by the two points in edges_vidxs).
            Float value 0 defines first point, and 1 defines end point. So the 3D
            point of the path is computed (1 - relpos) v_i + relpos v_j.
            Will only be returned if get_edges is True.
        """
        if vfunc.ndim > 1:
            raise ValueError(f"vfunc needs to be 1-dim, but is {vfunc.ndim}-dim!")
        # get intersecting triangles
        intersect = vfunc[self.t] > level
        t_idx = np.where(
            np.logical_or(
                np.sum(intersect, axis=1) == 1, np.sum(intersect, axis=1) == 2
            )
        )[0]
        # reduce to triangles that intersect with level:
        t_level = self.t[t_idx, :]
        intersect = intersect[t_idx, :]
        # trias have one vertex on one side and two on the other side of the level set
        # invert trias with two true values, so that single vertex is true
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
        # sort edge indices (rows are trias, cols are the two vertex ids)
        gg1 = np.sort(
            np.concatenate((gidx0[:, np.newaxis], gidx1[:, np.newaxis]), axis=1)
        )
        gg2 = np.sort(
            np.concatenate((gidx0[:, np.newaxis], gidx2[:, np.newaxis]), axis=1)
        )
        # concatenate all and get unique ones
        gg = np.concatenate((gg1, gg2), axis=0)
        gg_unique = np.unique(gg, axis=0)
        # generate level set intersection points for unique edges
        xl = ((level - vfunc[gg_unique[:, 0]])
              / ( vfunc[gg_unique[:, 1]] - vfunc[gg_unique[:, 0]]))
        p = ((1 - xl)[:, np.newaxis] * self.v[gg_unique[:, 0]]
             + xl[:, np.newaxis] * self.v[gg_unique[:, 1]])
        # fill sparse matrix with new point indices (+1 to distinguish from zero)
        a_mat = sparse.csc_matrix(
            (np.arange(gg_unique.shape[0]) + 1, (gg_unique[:, 0], gg_unique[:, 1]))
        )
        # for each tria create one edge via lookup in matrix
        edge_idxs = ( np.concatenate((a_mat[gg1[:, 0], gg1[:, 1]],
                                      a_mat[gg2[:, 0], gg2[:, 1]]), axis=0).T - 1 )
        # lengths computation
        p1 = np.squeeze(p[edge_idxs[:, 0]])
        p2 = np.squeeze(p[edge_idxs[:, 1]])
        llength = np.sqrt(((p1 - p2) ** 2).sum(1)).sum()
        # compute path from unordered, not-directed edge list
        # and return path as list of points, and path length
        if get_tria_idx:
            path, edge_idx = TriaMesh.__reduce_edges_to_path(
                edge_idxs, get_edge_idx=get_tria_idx
            )
            # translate local edge id to global tria id
            tria_idx = t_idx[edge_idx]
        else:
            path = TriaMesh.__reduce_edges_to_path(edge_idxs, get_tria_idx)
        # remove duplicate vertices (happens when levelset hits a vertex almost
        # perfectly)
        path3d = p[path, :]
        dd = ((path3d[0:-1, :] - path3d[1:, :]) ** 2).sum(1)
        # append 1 (never delete last node, if identical to the one before, we delete
        # the one before)
        dd = np.append(dd, 1)
        eps = 0.000001
        keep_ids = dd > eps
        path3d = path3d[keep_ids, :]
        edges_vidxs = gg_unique[path, :]
        edges_vidxs = edges_vidxs[keep_ids, :]
        edges_relpos = xl[path]
        edges_relpos = edges_relpos[keep_ids]
        if get_tria_idx:
            if n_points:
                raise ValueError("n_points cannot be combined with get_tria_idx=True.")
            tria_idx = tria_idx[dd[:-1] > eps]
            if get_edges:
                return path3d, llength, tria_idx, edges_vidxs, edges_relpos
            else:
                return path3d, llength, tria_idx
        else:
            if n_points:
                if get_edges:
                    raise ValueError("n_points cannot be combined with get_edges=True.")
                path3d = polygon.resample(path3d, n_points, n_iter=3)
            if get_edges:
                return path3d, llength, edges_vidxs, edges_relpos
            else:
                return path3d, llength