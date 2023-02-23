import numpy as np
from scipy import sparse

"""

Dependency:
    Scipy 0.10 or later for sparse matrix support


Original Author: Martin Reuter
Date: Feb-01-2019
"""


class TriaMesh:
    """A class representing a triangle mesh"""

    def __init__(self, v, t, fsinfo=None):
        """
        :param    v - vertices   List of lists of 3 float coordinates
                  t - triangles  List of lists of 3 int of indices (>=0) into v array
                                 Ordering is important: All triangles should be
                                 oriented the same way (counter-clockwise, when
                                 looking from above)
                  fsinfo         optional, FreeSurfer Surface Header Info
        """
        self.v = np.array(v)
        self.t = np.array(t)
        # transpose if necessary
        if self.v.shape[0] < self.v.shape[1]:
            self.v = self.v.T
        if self.t.shape[0] < self.t.shape[1]:
            self.t = self.t.T
        # Check a few things
        vnum = np.max(self.v.shape)
        if np.max(self.t) >= vnum:
            raise ValueError("Max index exceeds number of vertices")
        if self.t.shape[1] != 3:
            raise ValueError("Triangles should have 3 vertices")
        if self.v.shape[1] != 3:
            raise ValueError("Vertices should have 3 coordinates")
        # Compute adjacency matrices
        self.adj_sym = self._construct_adj_sym()
        self.adj_dir = self._construct_adj_dir()
        self.fsinfo = fsinfo  # place for Freesurfer Header info

    def _construct_adj_sym(self):
        """
        Constructs symmetric adjacency matrix (edge graph) of triangle mesh t
        Operates only on triangles.
        :return:    Sparse symmetric CSC matrix
                    The non-directed adjacency matrix
                    will be symmetric. Each inner edge (i,j) will have
                    the number of triangles that contain this edge.
                    Inner edges usually 2, boundary edges 1. Higher
                    numbers can occur when there are non-manifold triangles.
                    The sparse matrix can be binarized via:
                    adj.data = np.ones(adj.data.shape)
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
        """
        Constructs directed adjacency matrix (edge graph) of triangle mesh t
        Operates only on triangles.
        :return:    Sparse CSC matrix
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
        """
        Constructs directed adjacency matrix (edge graph) of triangle mesh t
        containing the triangle indices (only for non-manifold meshes)
        Operates only on triangles.
        :return:    Sparse CSC matrix
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
        """
        Check if triangle mesh is closed (no boundary edges)
        Operates only on triangles
        :return:   closed         bool True if no boundary edges in adj matrix
        """
        return 1 not in self.adj_sym.data

    def is_manifold(self):
        """
        Check if triangle mesh is manifold (no edges with >2 triangles)
        Operates only on triangles
        :return:   manifold       bool True if no edges with > 2 triangles
        """
        return np.max(self.adj_sym.data) <= 2

    def is_oriented(self):
        """
        Check if triangle mesh is oriented. True if all triangles are oriented
        counter-clockwise, when looking from above.
        Operates only on triangles
        :return:   oriented       bool True if max(adj_directed)=1
        """
        return np.max(self.adj_dir.data) == 1

    def euler(self):
        """
        Computes the Euler Characteristic (=#V-#E+#T)
        Operates only on triangles
        :return:   euler          Euler Characteristic (2=sphere,0=torus)
        """
        # v can contain unused vertices so we get vnum from trias
        vnum = len(np.unique(self.t.reshape(-1)))
        tnum = np.max(self.t.shape)
        enum = int(self.adj_sym.nnz / 2)
        return vnum - enum + tnum

    def tria_areas(self):
        """
        Computes the area of triangles using Heron's formula
        :return:   areas          ndarray with areas of each triangle
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
        """
        Computes the total surface area of triangle mesh
        :return:   area           Total surface area
        """
        areas = self.tria_areas()
        return np.sum(areas)

    def volume(self):
        """
        Computes the volume of closed triangle mesh, summing tetrahedra at origin
        :return:   volume         Total enclosed volume
        """
        if not self.is_closed():
            return 0.0
        if not self.is_oriented():
            raise ValueError(
                "Error: Can only compute volume for oriented triangle meshes!"
            )
        v0 = self.v[self.t[:, 0], :]
        v1 = self.v[self.t[:, 1], :]
        v2 = self.v[self.t[:, 2], :]
        v1mv0 = v1 - v0
        v2mv0 = v2 - v0
        cr = np.cross(v1mv0, v2mv0)
        spatvol = np.sum(v0 * cr, axis=1)
        vol = np.sum(spatvol) / 6.0
        return vol

    def vertex_degrees(self):
        """
        Computes the vertex degrees (number of edges at each vertex)
        :return:   vdeg           Array of vertex degrees
        """
        vdeg = np.bincount(self.t.reshape(-1))
        return vdeg

    def vertex_areas(self):
        """
        Computes the area associated to each vertex (1/3 of one-ring trias)
        :return:   vareas         Array of vertex areas
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
        """
        Computes the average edge length of the mesh
        :return:   edgelength     Avg. edge length
        """
        # get only upper off-diag elements from symmetric adj matrix
        triadj = sparse.triu(self.adj_sym, 1, format="coo")
        edgelens = np.sqrt(
            ((self.v[triadj.row, :] - self.v[triadj.col, :]) ** 2).sum(1)
        )
        return edgelens.mean()

    def tria_normals(self):
        """
        Computes triangle normals
        Ordering of trias is important: counterclockwise when looking
        :return:  n - normals (num triangles X 3 )
        """
        import sys

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
        """
        get_vertex_normals(v,t) computes vertex normals
            Triangle normals around each vertex are averaged, weighted
            by the angle that they contribute.
            Ordering is important: counterclockwise when looking
            at the triangle from above.
        :return:  n - normals (num vertices X 3 )
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
        # will all point in the same direction but have different lengths depending on spanned area
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
        """
        Checks if the vertex list has more vertices than what is used in tria
        :return:    bool
        """
        vnum = np.max(self.v.shape)
        vnumt = len(np.unique(self.t.reshape(-1)))
        return vnum != vnumt

    def tria_qualities(self):
        """
        Computes triangle quality for each triangle in mesh where
        q = 4 sqrt(3) A / (e1^2 + e2^2 + e3^2 )
        where A is the triangle area and ei the edge length of the three edges.
        This measure is used by FEMLAB and can also be found in:
            R.E. Bank, PLTMG ..., Frontiers in Appl. Math. (7), 1990.
        Constants are chosen so that q=1 for the equilateral triangle.
        :return:    ndarray with triangle qualities
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
        """
        Computes a tuple of boundary loops. Meshes can have 0 or more boundary
        loops, which are cycles in the directed adjacency graph of the boundary
        edges.
        Works on trias only. Could fail if loops are connected via a single
        vertex (like a figure 8). That case needs debugging.
        :return:   loops          List of lists with boundary loops
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
        while not firstcol == []:
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

    def centroid(self):
        """
        Computes centroid of triangle mesh as a weighted average of triangle
        centers. The weight is determined by the triangle area.
        (This could be done much faster if a FEM lumped mass matrix M is
        already available where this would be M*v, because it is equivalent
        with averaging vertices weighted by vertex area)

        :return:    centroid    The centroid of the mesh
                    totalarea   The total area of the mesh

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
        """
        Compute vertices and adjacent triangle ids for each edge

        :param      with_boundary   also work on boundary half edges, default ignore

        :return:    vids            2 column array with starting and end vertex for each
                                    unique inner edge
                    tids            2 column array with triangle containing the half edge
                                    from vids[0,:] to vids [1,:] in first column and the
                                    neighboring triangle in the second column
                    bdrvids         if with_boundary is true: 2 column array with each
                                    boundary half-edge
                    bdrtids         if with_boundary is true: 1 column array with the
                                    associated triangle to each boundary edge
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
        """
        Compute various curvature values at vertices.

        For the algorithm see e.g.
        Pierre Alliez, David Cohen-Steiner, Olivier Devillers, Bruno Levy, and Mathieu Desbrun.
        Anisotropic Polygonal Remeshing.
        ACM Transactions on Graphics, 2003.

        :param      smoothit  smoothing iterations on vertex functions
        :return:    u_min     minimal curvature directions (vnum x 3)
                    u_max     maximal curvature directions (vnum x 3)
                    c_min     minimal curvature
                    c_max     maximal curvature
                    c_mean    mean curvature: (c_min + c_max) / 2.0
                    c_gauss   Gauss curvature: c_min * c_max
                    normals   normals (vnum x 3)
        """
        # import warnings
        # warnings.filterwarnings('error')
        import sys

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
        vdeg[vdeg == 0] = 1
        vv = vv / vdeg.reshape(-1, 1)
        # smooth vertex functions
        vv = self.smooth_vfunc(vv, smoothit)
        # create vnum 3x3 symmetric matrices at each vertex
        mats = np.empty([vnum, 3, 3])
        mats[:, 0, :] = vv[:, [0, 1, 2]]
        mats[:, [1, 2], 0] = vv[:, [1, 2]]
        mats[:, 1, [1, 2]] = vv[:, [3, 4]]
        mats[:, 2, 1] = vv[:, 4]
        mats[:, 2, 2] = vv[:, 5]
        # compute eigendecomposition (real for symmetric matrices)
        evals, evecs = np.linalg.eig(mats)
        evals = np.real(evals)
        evecs = np.real(evecs)
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
        """
        Compute min and max curvature and directions (orthogonal and in tria plane)
        for each triangle. First we compute these values on vertices and then smooth
        there. Finally they get mapped to the trias (averaging) and projected onto
        the triangle plane, and orthogonalized.
        :param smoothit: number of smoothing iterations for curvature computation on vertices
        :return: u_min : min curvature direction on triangles
                 u_max : max curvature direction on triangles
                 c_min : min curvature on triangles
                 c_max : max curvature on triangles
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
        # project tumax back to tria plane and normalize (will not be orthogonal to tumin)
        # tumax1 = tumax - tn * (np.sum(tn * tumax, axis=1)).reshape(-1, 1)
        # in a second step orthorgonalize to tumin
        # tumax1 = tumax1 - tumin * (np.sum(tumin * tumax1, axis=1)).reshape(-1, 1)
        # normalize
        # tumax1l = np.sqrt(np.sum(tumax1 * tumax1, axis=1)).reshape(-1, 1)
        # tumax1 = tumax1 / np.maximum(tumax1l, 1e-8)
        # or simply create vector that is orthogonal to both normal and tumin
        tumax2 = np.cross(tn, tumin2)
        # if really necessary flip direction if that is true for inputs
        # tumax3 = np.sign(np.sum(np.cross(tumin, tumax) * tn, axis=1)).reshape(-1, 1) * tumax2
        # I wonder how much changes, if we first map umax to tria and then find orhtogonal umin next?
        return tumin2, tumax2, tcmin, tcmax

    def normalize_(self):
        """
        Normalizes TriaMesh to unit surface area with a centroid at the origin.
        Modifies the vertices.
        """
        centroid, area = self.centroid()
        self.v = (1.0 / np.sqrt(area)) * (self.v - centroid)

    def rm_free_vertices_(self):
        """
        Remove unused (free) vertices from v and t. These are vertices that are not
        used in any triangle. They can produce problems when constructing, e.g.,
        Laplace matrices.

        Will update v and t in mesh.

        :return:    vkeep          Indices (from original list) of kept vertices
                    vdel           Indices of deleted (unused) vertices
        """
        tflat = self.t.reshape(-1)
        vnum = np.max(self.v.shape)
        if np.max(tflat) >= vnum:
            raise ValueError("Max index exceeds number of vertices")
        # determine which vertices to keep
        vkeep = np.full(vnum, False, dtype=bool)
        vkeep[tflat] = True
        # list of deleted vertices (old indices)
        vdel = np.nonzero(~vkeep)[0]
        # if nothing to delete return
        if len(vdel) == 0:
            return np.arange(vnum), []
        # delete unused vertices
        vnew = self.v[vkeep, :]
        # create lookup table
        tlookup = np.cumsum(vkeep) - 1
        # reindex tria
        tnew = tlookup[self.t]
        # convert vkeep to index list
        vkeep = np.nonzero(vkeep)[0]
        # set new vertices and tria and re-init adj matrices
        self.__init__(vnew, tnew)
        return vkeep, vdel

    def refine_(self, it=1):
        """
        Refines the triangle mesh by placing new vertex on each edge midpoint
        and thus creating 4 similar triangles from one parent triangle.
        :param    it      : iterations (default 1)
        :return:  none, modifies mesh in place
        """
        for x in range(it):
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
            self.__init__(vnew, tnew)

    def normal_offset_(self, d):
        """
        normal_offset(d) moves vertices along normal by distance d
        :param    d    move distance, can be a number or array of vertex length
        :return:  none, modifies vertices in place
        """
        n = self.vertex_normals()
        vn = self.v + d * n
        self.v = vn
        # no need to re-init, only changed vertices

    def orient_(self):
        """
        orient_ re-orients triangles of manifold mesh to be consistent, so that vertices
        are listed counter-clockwise, when looking from above (outside).

        :return:    none, modifies triangles in place and re-inits adj matrices

        Algorithm:
        1. Construct list for each half-edge with its triangle and edge direction
        2. Drop boundary half-edges and find half-edge pairs
        3. Construct sparse matrix with triangle neighbors, with entry 1 for opposite half edges
           and -1 for parallel half-edges (normal flip across this edge)
        4. Flood mesh from first tria using triangle neighbor matrix - keeping track of sign
        5. When flooded, negative sign for a triangle indicates it needs to be flipped
        6. If global volume is negative, flip everything (first tria was wrong)
        """
        tnew = self.t
        flipped = 0
        if not self.is_oriented():
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
            # construct sparse tria neighbor matrix where weights indicate normal flips across edge
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
            print(
                "Searched mesh after {} flooding iterations ({} sec).".format(
                    count, endt - startt
                )
            )
            # get tria indices that need flipping:
            idx = v.toarray() == -1
            idx = idx.reshape(-1)
            tnew = self.t
            tnew[np.ix_(idx, [1, 0])] = tnew[np.ix_(idx, [0, 1])]
            self.__init__(self.v, tnew)
            flipped = idx.sum()
        # flip orientation on all trias if volume is negative:
        if self.volume() < 0:
            tnew[:, [1, 2]] = tnew[:, [2, 1]]
            self.__init__(self.v, tnew)
            flipped = tnew.shape[0] - flipped
        return flipped

    def map_tfunc_to_vfunc(self, tfunc, weighted=False):
        """
        Maps function for each tria to each vertex by attributing 1/3 to each
        Uses vertices and trias.

        :param    tfunc :        Float vector or matrix (#t x N) of values at
                                 vertices
        :param    weighted :     False, weigh only by 1/3, e.g. to compute
                                 vertex areas from tria areas
                                 True, weigh by triangle area / 3, e.g. to
                                 integrate a function defined on the trias,
                                 for example integrating the "one" function
                                 will also yield the vertex areas.

        :return:   vfunc          Function on vertices vector or matrix (#v x N)
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
        """
        Maps function for each vertex to each triangle by attributing 1/3 to each
        Uses number of vertices and trias

        :param    vfunc          Float vector or matrix (#t x N) of values at
                                 vertices

        :return:  tfunc          Function on trias vector or matrix (#t x N)
        """
        if self.v.shape[0] != vfunc.shape[0]:
            raise ValueError("Error: length of vfunc needs to match number of vertices")
        vfunc = np.array(vfunc) / 3.0
        tfunc = np.sum(vfunc[self.t], axis=1)
        return tfunc

    def smooth_vfunc(self, vfunc, n=1):
        """
        Smoothes vector float function on the mesh iteratively

        :param    vfunc :            Float vector of values at vertices,
                                     if empty, use vertex coordinates
        :param    n :                Number of iterations for smoothing

        :return:  vfunc              Smoothed surface vertex function
        """
        if vfunc is None:
            vfunc = self.v
        vfunc = np.array(vfunc)
        if self.v.shape[0] != vfunc.shape[0]:
            raise ValueError("Error: length of vfunc needs to match number of vertices")
        areas = self.vertex_areas()[:, np.newaxis]
        adj = self.adj_sym.copy()
        # binarize:
        adj.data = np.ones(adj.data.shape)
        # adjust columns to contain areas of vertex i
        adj2 = adj.multiply(areas)
        # rowsum is the area sum of 1-ring neighbors
        rowsum = np.sum(adj2, axis=1)
        # normalize rows to sum = 1
        adj2 = adj2.multiply(1.0 / rowsum)
        # apply sparse matrix n times (fast in spite of loop)
        vout = adj2.dot(vfunc)
        for i in range(n - 1):
            vout = adj2.dot(vout)
        return vout

    def smooth_(self, n=1):
        """
        Smoothes mesh in place for a number of iterations
        :param n:   smoothing iterations
        :return:    none, smoothes mesh in place
        """
        vfunc = self.smooth_vfunc(self.v, n)
        self.v = vfunc
        return
