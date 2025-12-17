import logging

import numpy as np
from scipy import sparse

from . import _tet_io as io

logger = logging.getLogger(__name__)

class TetMesh:
    """Class representing a tetrahedral mesh.

    This is an efficient implementation of a tetrahedral mesh data structure
    with core functionality using sparse matrices internally (Scipy).

    Parameters
    ----------
    v : array_like
        List of lists of 3 float coordinates.
    t : array_like
        List of lists of 4 int of indices (>=0) into ``v`` array.
        Ordering is important: so that t0, t1, t2 are oriented
        counterclockwise when looking from above, and t3 is
        on top of that triangle.

    Notes
    -----
    The class has static class methods to read tetrahera meshes from
    `GMSH <https://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format>`_
    and `VTK <https://examples.vtk.org/site/VTKFileFormats/>`_ files.
    """

    def __init__(self, v, t):
        self.v = np.array(v)
        self.t = np.array(t)
        vnum = self.v.shape[0]
        if np.max(self.t) >= vnum:
            raise ValueError("Max index exceeds number of vertices")
        # put more checks here (e.g. the dim 3 conditions on columns)
        # self.orient_()
        self.adj_sym = self.construct_adj_sym()

    @classmethod
    def read_gmsh(cls, filename):
        """Load GMSH tetrahedron mesh.

        Parameters
        ----------
        filename : str
            Filename to load.

        Returns
        -------
        tet : TetMesh
            Object of loaded GMSH tetrahedron mesh.
        """
        return io.read_gmsh(filename)

    @classmethod
    def read_vtk(cls, filename):
        """Load VTK tetrahedron mesh.

        Parameters
        ----------
        filename : str
            Filename to load.

        Returns
        -------
        tet : TetMesh
            Object of loaded VTK tetrahedron mesh.
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

    def construct_adj_sym(self):
        """Create adjacency symmetric matrix.

        The adjacency matrix will be symmetric. Each inner
        edge will get the number of tetrahedra that contain this edge.
        Inner edges are usually 3 or larger, boundary, 2 or 1.
        Works on tetras only.

        Returns
        -------
        adj : csc_matrix
            Symmetric adjacency matrix as csc sparse matrix.
        """
        t1 = self.t[:, 0]
        t2 = self.t[:, 1]
        t3 = self.t[:, 2]
        t4 = self.t[:, 3]
        i = np.hstack((t1, t2, t2, t3, t3, t1, t1, t2, t3, t4, t4, t4))
        j = np.hstack((t2, t1, t3, t2, t1, t3, t4, t4, t4, t1, t2, t3))
        adj = sparse.csc_matrix((np.ones(i.shape, dtype=int), (i, j)))
        return adj

    def has_free_vertices(self):
        """Check if the vertex list has more vertices than what is used in tetra.

        (same implementation as in `~lapy.TriaMesh`)

        Returns
        -------
        bool
            Whether vertex list has more vertices than tetra or not.
        """
        vnum = len(self.v)
        vnumt = len(np.unique(self.t.reshape(-1)))
        return vnum != vnumt

    def is_oriented(self):
        """Check if tet mesh is oriented.

        True if all tetrahedra are oriented
        so that v0,v1,v2 are oriented counterclockwise when looking from above,
        and v3 is on top of that triangle.

        Returns
        -------
        oriented: bool
            True if all tet volumes are positive.
            False if some or all are negative.
            Raises ValueError if degenerate (zero-volume) tets are found.
        """
        # Compute vertex coordinates and a difference vector for each triangle:
        t0 = self.t[:, 0]
        t1 = self.t[:, 1]
        t2 = self.t[:, 2]
        t3 = self.t[:, 3]
        v0 = self.v[t0, :]
        v1 = self.v[t1, :]
        v2 = self.v[t2, :]
        v3 = self.v[t3, :]
        e0 = v1 - v0
        e2 = v2 - v0
        e3 = v3 - v0
        # Compute cross product and 6 * vol for each triangle:
        cr = np.cross(e0, e2)
        vol = np.sum(e3 * cr, axis=1)
        if np.any(vol == 0):
            raise ValueError("Degenerate (zero-volume) tetrahedra detected")
        if np.max(vol) < 0.0:
            #print("All tet orientations are flipped")
            return False
        elif np.min(vol) > 0.0:
            #print("All tet orientations are correct")
            return True
        #print("Orientations are not uniform")
        return False

    def avg_edge_length(self):
        """Get average edge lengths in tet mesh.

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

    def boundary_tria(self, tetfunc=None):
        """Get boundary triangle mesh of tetrahedra.

        It can have multiple connected components.
        Tria will have same vertices (including free vertices),
        so that the tria indices agree with the tet-mesh, in case we want to
        transfer information back, e.g. a FEM boundary condition, or to access
        a TetMesh vertex function with TriaMesh.t indices.

        .. warning::

            Note, that it seems to be returning non-oriented triangle meshes,
            may need some debugging, until then use tria.orient_() after this.

        Parameters
        ----------
        tetfunc : array | None
            List of tetra function values (optional).

        Returns
        -------
        TriaMesh
            TriaMesh of boundary (potentially >1 components).
        triafunc : array
            List of tria function values (only returned if ``tetfunc`` is provided).
        """
        from . import TriaMesh

        # get all triangles
        allt = np.vstack(
            (
                self.t[:, np.array([3, 1, 2])],
                self.t[:, np.array([2, 0, 3])],
                self.t[:, np.array([1, 3, 0])],
                self.t[:, np.array([0, 2, 1])],
            )
        )
        # sort rows so that faces are reorder in ascending order of indices
        allts = np.sort(allt, axis=1)
        # find unique trias without a neighbor
        tria, indices, count = np.unique(
            allts, axis=0, return_index=True, return_counts=True
        )
        tria = allt[indices[count == 1]]
        logger.info("Found %d triangles on boundary.", np.size(tria, 0))
        # if we have tetra function, map these to the boundary triangles
        if tetfunc is not None:
            alltidx = np.tile(np.arange(self.t.shape[0]), 4)
            tidx = alltidx[indices[count == 1]]
            triafunc = tetfunc[tidx]
            return TriaMesh(self.v, tria), triafunc
        return TriaMesh(self.v, tria)

    def rm_free_vertices_(self):
        """Remove unused (free) vertices from v and t.

        These are vertices that are not used in any triangle. They can produce problems
        when constructing, e.g., Laplace matrices.

        Will update v and t in mesh.
        Same implementation as in `~lapy.TriaMesh`.

        Returns
        -------
        vkeep: array
            Indices (from original list) of kept vertices.
        vdel: array
            Indices of deleted (unused) vertices.
        """
        tflat = self.t.reshape(-1)
        vnum = len(self.v)
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
        self.v = vnew
        self.t = tnew
        return vkeep, vdel

    def orient_(self):
        """Ensure that tet mesh is oriented.

        Re-orient tetras so that
        v0,v1,v2 are oriented counterclockwise when looking from above,
        and v3 is on top of that triangle.

        Returns
        -------
        onum : int
            Number of re-oriented tetras.
        """
        # Compute vertex coordinates and a difference vector for each tetra:
        t0 = self.t[:, 0]
        t1 = self.t[:, 1]
        t2 = self.t[:, 2]
        t3 = self.t[:, 3]
        v0 = self.v[t0, :]
        v1 = self.v[t1, :]
        v2 = self.v[t2, :]
        v3 = self.v[t3, :]
        e0 = v1 - v0
        e2 = v2 - v0
        e3 = v3 - v0
        # Compute cross product and 6 * vol for each tetra:
        cr = np.cross(e0, e2)
        vol = np.sum(e3 * cr, axis=1)
        if np.any(vol == 0):
            raise ValueError("Degenerate (zero-volume) tetrahedra detected")
        negtet = vol < 0.0
        negnum = np.sum(negtet)
        if negnum == 0:
            logger.info("Mesh is oriented, nothing to do")
            return 0
        tnew = self.t.copy()
        temp = tnew[negtet, 1].copy()
        tnew[negtet, 1] = tnew[negtet, 2]
        tnew[negtet, 2] = temp
        self.t = tnew
        self.adj_sym = self.construct_adj_sym()
        logger.info("Flipped %d tetrahedra", negnum)
        #self.__init__(self.v, tnew)
        return negnum
