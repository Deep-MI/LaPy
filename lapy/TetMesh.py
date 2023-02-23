#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Dependency:
    Scipy 0.10 or later for sparse matrix support


Original Author: Martin Reuter
Date: Feb-01-2019
"""

import numpy as np
from scipy import sparse


class TetMesh:
    """A class representing a tetraheral mesh"""

    def __init__(self, v, t):
        """
        Inputs:   v - vertices   List of lists of 3 float coordinates
                  t - tetra      List of lists of 4 int of indices (>=0) into v array
                                 Ordering is important: so that t0,t1,t2 are oriented
                                 counterclockwise when looking from above, and t3 is
                                 on top of that triangle.
        """
        self.v = np.array(v)
        self.t = np.array(t)
        vnum = np.max(self.v.shape)
        if np.max(self.t) >= vnum:
            raise ValueError("Max index exceeds number of vertices")
        # put more checks here (e.g. the dim 3 conditions on columns)
        # self.orient_()
        self.adj_sym = self.construct_adj_sym()

    def construct_adj_sym(self):
        """
        The adjacency matrix will be symmetric. Each inner
        edge will get the number of tetrahedra that contain this edge.
        Inner edges are usually 3 or larger, boundary, 2 or 1.
        Works on tetras only.
        :return:    symmetric adjacency matrix as csc sparse matrix
        """
        t1 = self.t[:, 0]
        t2 = self.t[:, 1]
        t3 = self.t[:, 2]
        t4 = self.t[:, 3]
        i = np.column_stack((t1, t2, t2, t3, t3, t1, t1, t2, t3, t4, t4, t4)).reshape(
            -1
        )
        j = np.column_stack((t2, t1, t3, t2, t1, t3, t4, t4, t4, t1, t2, t3)).reshape(
            -1
        )
        adj = sparse.csc_matrix((np.ones(i.shape), (i, j)))
        return adj

    def has_free_vertices(self):
        """
        Checks if the vertex list has more vertices than what is used in tetra
        (same implementation as in TriaMesh)
        :return:    bool
        """
        vnum = np.max(self.v.shape)
        vnumt = len(np.unique(self.t.reshape(-1)))
        return vnum != vnumt

    def is_oriented(self):
        """
        Check if tet mesh is oriented. True if all tetrahedra are oriented
        so that v0,v1,v2 are oriented counterclockwise when looking from above,
        and v3 is on top of that triangle.

        :return:   oriented       bool True if max(adj_directed)=1
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
        if np.max(vol) < 0.0:
            print("All tet orientations are flipped")
            return False
        elif np.min(vol) > 0.0:
            print("All tet orientations are correct")
            return True
        elif np.count_nonzero(vol) < len(vol):
            print("We have degenerated zero-volume tetrahedra")
            return False
        else:
            print("Orientations are not uniform")
            return False

    def avg_edge_length(self):
        """
        Get average edge lengths in tet mesh
        :return:    double  average edge length
        """
        # get only upper off-diag elements from symmetric adj matrix
        triadj = sparse.triu(self.adj_sym, 1, format="coo")
        edgelens = np.sqrt(
            ((self.v[triadj.row, :] - self.v[triadj.col, :]) ** 2).sum(1)
        )
        return edgelens.mean()

    def boundary_tria(self, tetfunc=None):
        """
        Get boundary triangle mesh of tetrahedra (can have multiple connected components).
        Tria will have same vertices (including free vertices), so that the tria indices
        agree with the tet-mesh, in case we want to transfer information back, e.g.
        a FEM boundary condition, or to access a TetMesh vertex function with
        TriaMesh.t indices.

        !! Note, that it seems to be returning non-oriented triangle meshes,
        may need some debugging, until then use tria.orient_() after this. !!

        Inputs:   tetfunc        List of tetra function values (optional)

        :return:  TriaMesh       TriaMesh of boundary (potentially >1 components)
                  triafunc       List of tria function values (if tetfunc passed)
        """
        from .TriaMesh import TriaMesh

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
        print("Found " + str(np.size(tria, 0)) + " triangles on boundary.")
        # if we have tetra function, map these to the boundary triangles
        if tetfunc is not None:
            alltidx = np.tile(np.arange(self.t.shape[0]), 4)
            tidx = alltidx[indices[count == 1]]
            triafunc = tetfunc[tidx]
            return TriaMesh(self.v, tria), triafunc
        return TriaMesh(self.v, tria)

    def rm_free_vertices_(self):
        """
        Remove unused (free) vertices from v and t. These are vertices that are not
        used in any triangle. They can produce problems when constructing, e.g.,
        Laplace matrices.

        Will update v and t in mesh.
        Same implementation as in TriaMesh

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
        self.v = vnew
        self.t = tnew
        return vkeep, vdel

    def orient_(self):
        """
        Ensure that tet mesh is oriented. Re-orient tetras so that
        v0,v1,v2 are oriented counterclockwise when looking from above,
        and v3 is on top of that triangle.

        :return:    int     number of re-oriented tetras
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
        negtet = vol < 0.0
        negnum = np.sum(negtet)
        if negnum == 0:
            print("Mesh is oriented, nothing to do")
            return 0
        tnew = self.t
        # negtet = np.where(negtet)
        temp = self.t[negtet, 1]
        tnew[negtet, 1] = self.t[negtet, 2]
        tnew[negtet, 2] = temp
        onum = np.sum(negtet)
        print("Flipped " + str(onum) + " tetrahedra")
        self.__init__(self.v, tnew)
        return onum
