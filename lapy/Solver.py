import numpy as np
from scipy import sparse


class Solver:
    """A class representing a linear FEM solver for:
     Laplace Eigenvalue problems and Poisson Equation

     Inputs can be geometry classes which have vertices and elements.
     Currently TriaMesh and TetMesh are implemented.
     FEM matrices (stiffness (or A) and mass matrix (or B)) are computed
     during the construction. After that the Eigenvalue solver (eigs) or
     Poisson Solver (poisson) can be called.

     The class has a static member to create the mass matrix of TriaMesh
     for external function that do not need stiffness.
     """

    def __init__(self, geometry, lump=False, aniso=None, aniso_smooth=10, use_cholmod=True):
        """
        Construct the Solver class. Computes linear Laplace FEM stiffness and
        mass matrix for TriaMesh or TetMesh input geometries. For TriaMesh it can also
        construct the anisotropic Laplace.

        Inputs:
            geometry    : is a geometry class, currently either TriaMesh or TetMesh
            aniso       : float, anisotropy for curvature based anisotopic Laplace
                          can also be tuple (a_min, a_max) to differentially affect
                          the min and max curvature directions. E.g. (0,50) will set
                          scaling to 1 into min curv direction even if the max curvature
                          is large in those regions (= isotropic in regions with 
                          large max curv and min curv close to zero= concave cylinder)
            lump        : whether to lump the mass matrix (diagonal), default False
            use_cholmod : try to use the Cholesky decomposition from the cholmod
                          library for improved speed. This requires skikit sparse to
                          be installed. If it cannot be found, we fallback to LU
                          decomposition. 
        """
        self.use_cholmod = use_cholmod
        if type(geometry).__name__ == "TriaMesh":
            if aniso is not None:
                # anisotropic Laplace
                print("TriaMesh with anisotropic Laplace-Beltrami")
                u1, u2, c1, c2 = geometry.curvature_tria(smoothit=aniso_smooth)
                # Diag mat to specify anisotropy strength
                if isinstance(aniso, (list, tuple, set, np.ndarray)):
                    if len(aniso) != 2: 
                        raise ValueError('aniso should be scalar or tuple/array of length 2!')
                    aniso0 = aniso[0]
                    aniso1 = aniso[1]
                else:
                    aniso0 = aniso
                    aniso1 = aniso
                aniso_mat = np.empty((geometry.t.shape[0], 2))
                aniso_mat[:, 1] = np.exp(-aniso1 * np.abs(c1))
                aniso_mat[:, 0] = np.exp(-aniso0 * np.abs(c2))
                a, b = self._fem_tria_aniso(geometry, u1, u2, aniso_mat, lump)
            else:
                print("TriaMesh with regular Laplace-Beltrami")
                a, b = self._fem_tria(geometry, lump)
        elif type(geometry).__name__ == "TetMesh":
            print("TetMesh with regular Laplace")
            a, b = self._fem_tetra(geometry, lump)
        else:
            raise ValueError('Geometry type "' + type(geometry).__name__ + '" unknown')
        self.stiffness = a
        self.mass = b
        self.geotype = type(geometry)

    @staticmethod
    def _fem_tria(tria, lump=False):
        """
        computeABtria(v,t) computes the two sparse symmetric matrices representing
               the Laplace Beltrami Operator for a given triangle mesh using
               the linear finite element method (assuming a closed mesh or
               the Neumann boundary condition).

        Inputs:   v - vertices : list of lists of 3 floats
                  t - triangles: list of lists of 3 int of indices (>=0) into v array

        Outputs:  A - sparse sym. (n x n) positive semi definite numpy matrix
                  B - sparse sym. (n x n) positive definite numpy matrix (inner product)

        Can be used to solve sparse generalized Eigenvalue problem: A x = lambda B x
        or to solve Poisson equation: A x = B f (where f is function on mesh vertices)
        or to solve Laplace equation: A x = 0
        or to model the operator's action on a vector x:   y = B\(Ax)
        """
        import sys
        # Compute vertex coordinates and a difference vector for each triangle:
        t1 = tria.t[:, 0]
        t2 = tria.t[:, 1]
        t3 = tria.t[:, 2]
        v1 = tria.v[t1, :]
        v2 = tria.v[t2, :]
        v3 = tria.v[t3, :]
        v2mv1 = v2 - v1
        v3mv2 = v3 - v2
        v1mv3 = v1 - v3
        # Compute cross product and 4*vol for each triangle:
        cr = np.cross(v3mv2, v1mv3)
        vol = 2 * np.sqrt(np.sum(cr * cr, axis=1))
        # zero vol will cause division by zero below, so set to small value:
        vol_mean = 0.0001 * np.mean(vol)
        vol[vol < sys.float_info.epsilon] = vol_mean
        # compute cotangents for A
        # using that v2mv1 = - (v3mv2 + v1mv3) this can also be seen by
        # summing the local matrix entries in the old algorithm
        a12 = np.sum(v3mv2 * v1mv3, axis=1) / vol
        a23 = np.sum(v1mv3 * v2mv1, axis=1) / vol
        a31 = np.sum(v2mv1 * v3mv2, axis=1) / vol
        # compute diagonals (from row sum = 0)
        a11 = -a12 - a31
        a22 = -a12 - a23
        a33 = -a31 - a23
        # stack columns to assemble data
        local_a = np.column_stack((a12, a12, a23, a23, a31, a31, a11, a22, a33)).reshape(-1)
        i = np.column_stack((t1, t2, t2, t3, t3, t1, t1, t2, t3)).reshape(-1)
        j = np.column_stack((t2, t1, t3, t2, t1, t3, t1, t2, t3)).reshape(-1)
        # Construct sparse matrix:
        # a = sparse.csr_matrix((local_a, (i, j)))
        a = sparse.csc_matrix((local_a, (i, j)))
        # construct mass matrix (sparse or diagonal if lumped)
        if not lump:
            # create b matrix data (account for that vol is 4 times area)
            b_ii = vol / 24
            b_ij = vol / 48
            local_b = np.column_stack((b_ij, b_ij, b_ij, b_ij, b_ij, b_ij,
                                       b_ii, b_ii, b_ii)).reshape(-1)
            b = sparse.csc_matrix((local_b, (i, j)))
        else:
            # when lumping put all onto diagonal  (area/3 for each vertex)
            b_ii = vol / 12
            local_b = np.column_stack((b_ii, b_ii, b_ii)).reshape(-1)
            i = np.column_stack((t1, t2, t3)).reshape(-1)
            b = sparse.csc_matrix((local_b, (i, i)))
        return a, b

    @staticmethod
    def _fem_tria_aniso(tria, u1, u2, aniso_mat, lump=False):
        """
        computeABtria(v,t) computes the two sparse symmetric matrices representing
               the Laplace Beltrami Operator for a given triangle mesh using
               the linear finite element method (assuming a closed mesh or
               the Neumann boundary condition).

        Inputs:   v  - vertices : list of lists of 3 floats
                  t  - triangles: N list of lists of 3 int of indices (>=0) into v array
                  u1 - min curv:  min curvature direction per triangle (Nx3 floats)
                  u2 - max curv:  max curvature direction per triangle (Nx3 floats)
                  aniso_mat  - anisotropy matrix: diagonal elements in u1,u2 basis per
                                  triangle (Nx2 floats)

        Outputs:  A  - sparse sym. (n x n) positive semi definite numpy matrix
                  B  - sparse sym. (n x n) positive definite numpy matrix (inner product)

        Can be used to solve sparse generalized Eigenvalue problem: A x = lambda B x
        or to solve Poisson equation: A x = B f (where f is function on mesh vertices)
        or to solve Laplace equation: A x = 0
        or to model the operator's action on a vector x:   y = B\(Ax)
        """
        import sys
        # Compute vertex coordinates and a difference vector for each triangle:
        t1 = tria.t[:, 0]
        t2 = tria.t[:, 1]
        t3 = tria.t[:, 2]
        v1 = tria.v[t1, :]
        v2 = tria.v[t2, :]
        v3 = tria.v[t3, :]
        v2mv1 = v2 - v1
        v3mv2 = v3 - v2
        v1mv3 = v1 - v3
        # transform edge e to basis U = (U1,U2) via U^T * e
        # Ui is n x 3, e is n x 1, result is n x 2
        uv2mv1 = np.column_stack((np.sum(u1 * v2mv1, axis=1), np.sum(u2 * v2mv1, axis=1)))
        uv3mv2 = np.column_stack((np.sum(u1 * v3mv2, axis=1), np.sum(u2 * v3mv2, axis=1)))
        uv1mv3 = np.column_stack((np.sum(u1 * v1mv3, axis=1), np.sum(u2 * v1mv3, axis=1)))
        # Compute cross product and 4*vol for each triangle:
        cr = np.cross(v3mv2, v1mv3)
        vol = 2 * np.sqrt(np.sum(cr * cr, axis=1))
        # zero vol will cause division by zero below, so set to small value:
        vol_mean = 0.0001 * np.mean(vol)
        vol[vol < sys.float_info.epsilon] = vol_mean
        # compute cotangents for A
        # using that v2mv1 = - (v3mv2 + v1mv3) this can also be seen by
        # summing the local matrix entries in the old algorithm
        # Also: here aniso_mat is the two diagonal entries, not full matrices
        a12 = np.sum(uv3mv2 * aniso_mat * uv1mv3, axis=1) / vol
        a23 = np.sum(uv1mv3 * aniso_mat * uv2mv1, axis=1) / vol
        a31 = np.sum(uv2mv1 * aniso_mat * uv3mv2, axis=1) / vol
        # compute diagonals (from row sum = 0)
        a11 = -a12 - a31
        a22 = -a12 - a23
        a33 = -a31 - a23
        # stack columns to assemble data
        local_a = np.column_stack((a12, a12, a23, a23, a31, a31, a11, a22, a33)).reshape(-1)
        i = np.column_stack((t1, t2, t2, t3, t3, t1, t1, t2, t3)).reshape(-1)
        j = np.column_stack((t2, t1, t3, t2, t1, t3, t1, t2, t3)).reshape(-1)
        # Construct sparse matrix:
        # a = sparse.csr_matrix((local_a, (i, j)))
        a = sparse.csc_matrix((local_a, (i, j)), dtype=np.float32)
        if not lump:
            # create b matrix data (account for that vol is 4 times area)
            b_ii = vol / 24
            b_ij = vol / 48
            local_b = np.column_stack((b_ij, b_ij, b_ij, b_ij, b_ij, b_ij,
                                       b_ii, b_ii, b_ii)).reshape(-1)
            b = sparse.csc_matrix((local_b, (i, j)), dtype=np.float32)
        else:
            # when lumping put all onto diagonal  (area/3 for each vertex)
            b_ii = vol / 12
            local_b = np.column_stack((b_ii, b_ii, b_ii)).reshape(-1)
            i = np.column_stack((t1, t2, t3)).reshape(-1)
            b = sparse.csc_matrix((local_b, (i, i)), dtype=np.float32)
        return a, b

    @staticmethod
    def fem_tria_mass(tria, lump=False):
        """
        Computes the sparse symmetric mass matrix of the
        Laplace Beltrami Operator for a given triangle mesh using the
        linear finite element method (assuming a closed mesh or the
        Neumann boundary condition).
        This is here, because sometimes only a mass matrix is needed and then
        this call is faster than the constructor above.

        Inputs:   v - vertices : list of lists of 3 floats
                  t - triangles: list of lists of 3 int of indices (>=0) into v array
                  lump         : Bool if B matrix should be lumped (diagnoal), default False

        Outputs:  B - sparse sym. (n x n) positive definite numpy matrix (inner product)

        This is only the mass matrix B of the Eigenvalue problem: A x = lambda B x
        Area of the surface mesh can be obtained via B.sum()
        """
        # Compute vertex coordinates and a difference vector for each triangle:
        t1 = tria.t[:, 0]
        t2 = tria.t[:, 1]
        t3 = tria.t[:, 2]
        v1 = tria.v[t1, :]
        v2 = tria.v[t2, :]
        v3 = tria.v[t3, :]
        # v2mv1 = v2 - v1
        v3mv2 = v3 - v2
        v1mv3 = v1 - v3

        # Compute cross product and 4*vol for each triangle:
        cr = np.cross(v3mv2, v1mv3)
        vol = 0.5 * np.sqrt(np.sum(cr * cr, axis=1))
        # zero vol will cause division by zero below, so set to small value:
        vol_mean = 0.001 * np.mean(vol)
        vol[vol == 0] = vol_mean
        # create b matrix data
        if not lump:
            b_ii = vol / 6
            b_ij = vol / 12
            local_b = np.column_stack((b_ij, b_ij, b_ij, b_ij, b_ij, b_ij,
                                       b_ii, b_ii, b_ii)).reshape(-1)
            # stack edge and diag coords for matrix indices
            i = np.column_stack((t1, t2, t2, t3, t3, t1, t1, t2, t3)).reshape(-1)
            j = np.column_stack((t2, t1, t3, t2, t1, t3, t1, t2, t3)).reshape(-1)
            # Construct sparse matrix:
            b = sparse.csc_matrix((local_b, (i, j)))
        else:
            # when lumping put all onto diagonal
            b_ii = vol / 3
            local_b = np.column_stack((b_ii, b_ii, b_ii)).reshape(-1)
            i = np.column_stack((t1, t2, t3)).reshape(-1)
            b = sparse.csc_matrix((local_b, (i, i)))
        return b

    @staticmethod
    def _fem_tetra(tetra, lump=False):
        """
        computeABtetra(v,t) computes the two sparse symmetric matrices representing
               the Laplace Beltrami Operator for a given tetrahedral mesh using
               the linear finite element method (Neumann boundary condition).

        Inputs:   v - vertices : list of lists of 3 floats
                  t - tetras   : list of lists of 4 int of indices (>=0) into v array
                                 Ordering is important: first three vertices for
                                 triangle counter-clockwise when looking at it from
                                 the inside of the tetrahedron
                  lump         : Bool if B matrix should be lumped (diagnoal), default False

        Outputs:  A - sparse sym. (n x n) positive semi definite numpy matrix
                  B - sparse sym. (n x n) positive definite numpy matrix (inner product)

        Can be used to solve sparse generalized Eigenvalue problem: A x = lambda B x
        or to solve Poisson equation: A x = B f (where f is function on mesh vertices)
        or to solve Laplace equation: A x = 0
        or to model the operator's action on a vector x:   y = B\(Ax)
        """
        # Compute vertex coordinates and a difference vector for each triangle:
        t1 = tetra.t[:, 0]
        t2 = tetra.t[:, 1]
        t3 = tetra.t[:, 2]
        t4 = tetra.t[:, 3]
        v1 = tetra.v[t1, :]
        v2 = tetra.v[t2, :]
        v3 = tetra.v[t3, :]
        v4 = tetra.v[t4, :]
        e1 = v2 - v1
        e2 = v3 - v2
        e3 = v1 - v3
        e4 = v4 - v1
        e5 = v4 - v2
        e6 = v4 - v3
        # Compute cross product and 6 * vol for each triangle:
        cr = np.cross(e1, e3)
        vol = np.abs(np.sum(e4 * cr, axis=1))
        # zero vol will cause division by zero below, so set to small value:
        vol_mean = 0.0001 * np.mean(vol)
        vol[vol == 0] = vol_mean
        # compute dot products of edge vectors
        e11 = np.sum(e1 * e1, axis=1)
        e22 = np.sum(e2 * e2, axis=1)
        e33 = np.sum(e3 * e3, axis=1)
        e44 = np.sum(e4 * e4, axis=1)
        e55 = np.sum(e5 * e5, axis=1)
        e66 = np.sum(e6 * e6, axis=1)
        e12 = np.sum(e1 * e2, axis=1)
        e13 = np.sum(e1 * e3, axis=1)
        e14 = np.sum(e1 * e4, axis=1)
        e15 = np.sum(e1 * e5, axis=1)
        e23 = np.sum(e2 * e3, axis=1)
        e25 = np.sum(e2 * e5, axis=1)
        e26 = np.sum(e2 * e6, axis=1)
        e34 = np.sum(e3 * e4, axis=1)
        e36 = np.sum(e3 * e6, axis=1)
        # compute entries for A (negations occur when one edge direction is flipped)
        # these can be computed multiple ways
        # basically for ij, take opposing edge (call it Ek) and two edges from the starting
        # point of Ek to point i (=El) and to point j (=Em), then these are of the
        # scheme:   (El * Ek)  (Em * Ek) - (El * Em) (Ek * Ek)
        # where * is vector dot product
        a12 = (- e36 * e26 + e23 * e66) / vol
        a13 = (- e15 * e25 + e12 * e55) / vol
        a14 = (e23 * e26 - e36 * e22) / vol
        a23 = (- e14 * e34 + e13 * e44) / vol
        a24 = (e13 * e34 - e14 * e33) / vol
        a34 = (- e14 * e13 + e11 * e34) / vol
        # compute diagonals (from row sum = 0)
        a11 = -a12 - a13 - a14
        a22 = -a12 - a23 - a24
        a33 = -a13 - a23 - a34
        a44 = -a14 - a24 - a34
        # stack columns to assemble data
        local_a = np.column_stack(
            (a12, a12, a23, a23, a13, a13, a14, a14, a24, a24, a34, a34,
             a11, a22, a33, a44)).reshape(-1)
        i = np.column_stack((t1, t2, t2, t3, t3, t1, t1, t4, t2, t4, t3, t4,
                             t1, t2, t3, t4)).reshape(-1)
        j = np.column_stack((t2, t1, t3, t2, t1, t3, t4, t1, t4, t2, t4, t3,
                             t1, t2, t3, t4)).reshape(-1)
        local_a = local_a / 6.0
        # Construct sparse matrix:
        # a = sparse.csr_matrix((local_a, (i, j)))
        a = sparse.csc_matrix((local_a, (i, j)))
        if not lump:
            # create b matrix data (account for that vol is 6 times tet volume)
            bii = vol / 60.0
            bij = vol / 120.0
            local_b = np.column_stack(
                (bij, bij, bij, bij, bij, bij, bij, bij, bij, bij, bij, bij,
                 bii, bii, bii, bii)).reshape(-1)
            b = sparse.csc_matrix((local_b, (i, j)))
        else:
            # when lumping put all onto diagonal (volume/4 for each vertex)
            bii = vol / 24.0
            local_b = np.column_stack((bii, bii, bii, bii)).reshape(-1)
            i = np.column_stack((t1, t2, t3, t4)).reshape(-1)
            b = sparse.csc_matrix((local_b, (i, i)))
        return a, b

    @staticmethod
    def _fem_voxels(vox, lump=False):
        """
        computeABvoxels(v,t) computes the two sparse symmetric matrices representing
               the Laplace Beltrami Operator for a given voxel mesh using
               the linear finite element method (Neumann boundary condition).

        Inputs:   v - vertices : list of lists of 3 floats
                  t - voxels   : list of lists of 8 int of indices (>=0) into v array
                                 Ordering: base counter-clockwise, then top counter-
                                 clockwise when looking from above
                  lump         : Bool if B matrix should be lumped (diagnoal), default False

        Outputs:  A - sparse sym. (n x n) positive semi definite numpy matrix
                  B - sparse sym. (n x n) positive definite numpy matrix (inner product)

        Can be used to solve sparse generalized Eigenvalue problem: A x = lambda B x
        or to solve Poisson equation: A x = B f (where f is function on mesh vertices)
        or to solve Laplace equation: A x = 0
        or to model the operator's action on a vector x:   y = B\(Ax)
        """
        tnum = vox.t.shape[0]
        # Linear local matrices on unit voxel
        tb = np.array([[8.0, 4.0, 2.0, 4.0, 4.0, 2.0, 1.0, 2.0],
                       [4.0, 8.0, 4.0, 2.0, 2.0, 4.0, 2.0, 1.0],
                       [2.0, 4.0, 8.0, 4.0, 1.0, 2.0, 4.0, 2.0],
                       [4.0, 2.0, 4.0, 8.0, 2.0, 1.0, 2.0, 4.0],
                       [4.0, 2.0, 1.0, 2.0, 8.0, 4.0, 2.0, 4.0],
                       [2.0, 4.0, 2.0, 1.0, 4.0, 8.0, 4.0, 2.0],
                       [1.0, 2.0, 4.0, 2.0, 2.0, 4.0, 8.0, 4.0],
                       [2.0, 1.0, 2.0, 4.0, 4.0, 2.0, 4.0, 8.0]]) / 216.0
        x = 1.0 / 9.0
        y = 1.0 / 18.0
        z = 1.0 / 36.0
        ta00 = np.array([[x, -x, -y, y, y, -y, -z, z],
                         [-x, x, y, -y, -y, y, z, -z],
                         [-y, y, x, -x, -z, z, y, -y],
                         [y, -y, -x, x, z, -z, -y, y],
                         [y, -y, -z, z, x, -x, -y, y],
                         [-y, y, z, -z, -x, x, y, -y],
                         [-z, z, y, -y, -y, y, x, -x],
                         [z, -z, -y, y, y, -y, -x, x]])
        ta11 = np.array([[x, y, -y, -x, y, z, -z, -y],
                         [y, x, -x, -y, z, y, -y, -z],
                         [-y, -x, x, y, -z, -y, y, z],
                         [-x, -y, y, x, -y, -z, z, y],
                         [y, z, -z, -y, x, y, -y, -x],
                         [z, y, -y, -z, y, x, -x, -y],
                         [-z, -y, y, z, -y, -x, x, y],
                         [-y, -z, z, y, -x, -y, y, x]])
        ta22 = np.array([[x, y, z, y, -x, -y, -z, -y],
                         [y, x, y, z, -y, -x, -y, -z],
                         [z, y, x, y, -z, -y, -x, -y],
                         [y, z, y, x, -y, -z, -y, -x],
                         [-x, -y, -z, -y, x, y, z, y],
                         [-y, -x, -y, -z, y, x, y, z],
                         [-z, -y, -x, -y, z, y, x, y],
                         [-y, -z, -y, -x, y, z, y, x]])
        # here we assume all voxels have the same dimensions (side lengths)
        v0 = vox.v[vox.t[0, 0], :]
        v1 = vox.v[vox.t[0, 1], :]
        v2 = vox.v[vox.t[0, 3], :]
        v3 = vox.v[vox.t[0, 4], :]
        v1mv0 = v1 - v0
        v2mv0 = v2 - v0
        v3mv0 = v3 - v0
        g11 = np.sum(v1mv0 * v1mv0)
        g22 = np.sum(v2mv0 * v2mv0)
        g33 = np.sum(v3mv0 * v3mv0)
        vol = np.sqrt(g11 * g22 * g33)
        a0 = 1.0 / g11
        a1 = 1.0 / g22
        a2 = 1.0 / g33
        if lump:
            local_b = (vol / 8.0) * np.ones([8, 8])
        else:
            local_b = tb * vol
        local_a = vol * (a0 * ta00 + a1 * ta11 + a2 * ta22)
        local_b = np.repeat(local_b[np.newaxis, :, :], tnum, axis=0).reshape(-1)
        local_a = np.repeat(local_a[np.newaxis, :, :], tnum, axis=0).reshape(-1)
        # Construct row and col indices.
        i = np.array([np.tile(x, (8, 1)) for x in vox.t]).reshape(-1)
        j = np.array([np.transpose(np.tile(x, (8, 1))) for x in vox.t]).reshape(-1)
        # Construct sparse matrix:
        a = sparse.csc_matrix((local_a, (i, j)))
        b = sparse.csc_matrix((local_b, (i, j)))
        return a, b

    def eigs(self, k=10):
        """
        Compute linear finite-element method Laplace-Beltrami spectrum

        :return:    eigenvalues     array: k Laplace eigenvalues, sorted.
                                    For closed meshes, or Neumann boundary condition
                                    0 will be first eigenvalue (with constant eigenfunction)
                    eigenfunctions  array: (N x k) with k eigenfunctions (in the columns)
        """
        from scipy.sparse.linalg import LinearOperator, eigsh, splu
        if self.use_cholmod:
            try:
                from sksparse.cholmod import cholesky
            except ImportError:
                self.use_cholmod = False

        if self.use_cholmod:
            print("Solver: Cholesky decomposition from scikit-sparse cholmod ...")
        else:
            print("Solver: spsolve (LU decomposition) ...")
            # turns out it is much faster to use cholesky and pass operator
        sigma = -0.01
        if self.use_cholmod:
            chol = cholesky(self.stiffness - sigma * self.mass)
            op_inv = LinearOperator(matvec=chol, shape=self.stiffness.shape,
                                    dtype=self.stiffness.dtype)
        else:
            lu = splu(self.stiffness - sigma * self.mass)
            op_inv = LinearOperator(matvec=lu.solve, shape=self.stiffness.shape,
                                    dtype=self.stiffness.dtype)
        eigenvalues, eigenvectors = eigsh(self.stiffness, k, self.mass, sigma=sigma, OPinv=op_inv)
        return eigenvalues, eigenvectors

    def poisson(self, h=0.0, dtup=(), ntup=()):
        """
        poissonSolver solves the poisson equation with boundary conditions
               based on the A and B Laplace matrices:  A x = B h

        Inputs:   A - sparse sym. (n x n) positive semi definite numpy matrix
                  B - sparse sym. (n x n) positive definite numpy matrix (inner product)
                      A and B are obtained via computeAB for either triangle or tet mesh
        Optional:
                  h - right hand side, can be constant or array with vertex values
                      Default: 0.0 (Laplace equation A x = 0)
                  dtup - Dirichlet boundary condition as a tuple.
                         Tuple contains index and data arrays of same length.
                         Default: no Dirichlet condition
                  ntup - Neumann boundary condition as a tuple.
                         Tuple contains index and data arrays of same length.
                         Default: Neumann on all boundaries

        Outputs:  x - array with vertex values of solution
        """
        from scipy.sparse.linalg import splu
        if self.use_cholmod:
            try:
                from sksparse.cholmod import cholesky
            except ImportError:
                self.use_cholmod = False
        # check matrices
        dim = self.stiffness.shape[0]
        if self.stiffness.shape != self.mass.shape or self.stiffness.shape[1] != dim:
            raise ValueError('Error: Square input matrices should have same number of rows and columns')
        # create vector h
        if np.isscalar(h):
            h = np.full((dim, 1), h, dtype='float64')
        elif (not np.isscalar(h)) and h.size != dim:
            raise ValueError('h should be either scalar or column vector with row num of A')
        # create vector d
        didx = []
        dvec = []
        ddat = []
        if dtup:
            if len(dtup) != 2:
                raise ValueError('dtup should contain index and data arrays')
            didx = dtup[0]
            ddat = dtup[1]
            if np.unique(didx).size != len(didx):
                raise ValueError('dtup indices need to be unique')
            if not (len(didx) > 0 and len(didx) == len(ddat)):
                raise ValueError('dtup should contain index and data arrays (same lengths > 0)')
            dvec = sparse.csc_matrix((ddat, (didx, np.zeros(len(didx), dtype=np.uint32))), (dim, 1))

        # create vector n
        nvec = 0
        if ntup:
            if len(ntup) != 2:
                raise ValueError('ntup should contain index and data arrays')
            nidx = ntup[0]
            ndat = ntup[1]
            if not (len(nidx) > 0 and len(nidx) == len(ndat)):
                raise ValueError('dtup should contain index and data arrays (same lengths > 0)')
            nvec = sparse.csc_matrix((ndat, (nidx, np.zeros(len(nidx), dtype=np.uint32))), (dim, 1))
        # compute right hand side
        b = self.mass * (h - nvec)
        if len(didx) > 0:
            b = b - self.stiffness * dvec
        # remove Dirichlet Nodes
        mask = []
        if len(didx) > 0:
            mask = np.full(dim, True, dtype=bool)
            mask[didx] = False
            b = b[mask]
            # we need to keep A sparse and do col and row slicing
            # only on the right format:
            if self.stiffness.getformat() == 'csc':
                a = self.stiffness[:, mask].tocsr()
                a = a[mask, :]
                a = a.tocsc()
            elif self.stiffness.getformat() == 'csr':
                a = self.stiffness[mask, :].tocrc()
                a = a[:, mask]
            else:
                raise ValueError('A matrix needs to be sparse CSC or CSR')
        else:
            a = self.stiffness
        # solve A x = b
        print("Matrix Format now: " + a.getformat())
        if self.use_cholmod:
            print("Solver: Cholesky decomposition from scikit-sparse cholmod ...")
            chol = cholesky(a)
            x = chol(b)
        else:
            print("Solver: spsolve (LU decomposition) ...")
            lu = splu(a)
            x = lu.solve(b.astype(np.float32))
        x = np.squeeze(np.array(x))
        # pad Dirichlet nodes
        if len(didx) > 0:
            xfull = np.zeros(dim)
            xfull[mask] = x
            xfull[didx] = ddat
            return xfull
        return x
