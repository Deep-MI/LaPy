import numpy as np


def diagonal(t, x, evecs, evals, n):
    """
    Computes heat kernel diagonal ( K(t,x,x,) )
    for a given time t (can be a vector)
    using only the first n smallest eigenvalues and eigenvectors

    :param   t       time or a row vector of time values
             x       vertex ids for the positions of K(t,x,x)
             evecs   eigenvectors (matrix: vnum x evecsnum)
             evals   vector of eigenvalues (col vector: evecsnum x 1)
             n       number of evecs and vals to use (smaller or equal length)

    :return:  h       matrix, rows: vertices selected in x, cols: times in t

    """
    # maybe add code to check dimensions of input and flip axis if necessary
    h = np.matmul(evecs[x, 0:n] * evecs[x, 0:n], np.exp(-np.matmul(evals[0:n], t)))
    return h


def kernel(t, vfix, evecs, evals, n):
    """
    Computes heat kernel from all points to a fixed point (vfix)
    for a given time t (using only the first n smallest eigenvalues
    and eigenvectors)

    K_t (p,q) = sum_j exp(-eval_j t) evec_j(p) evec_j(q)

    :param
      t      time (can also be a row vector, if passing multiple times)
      vfix   fixed vertex index
      evecs  matrix of eigenvectors (M x N), M = #vertices, N=#eigenvectors
      eval   col vector of eigenvalues (N)
      n      number of eigenvalues/vectors used in heat kernel (n<=N)

    :return:
      h      matrix m rows: all vertices, cols: times in t

    """
    # h = evecs * ( exp(-evals * t) .* repmat(evecs(vfix,:)',1,length(t))  )
    h = np.matmul(evecs[:, 0:n], (np.exp(np.matmul(-evals[0:n], t)) * evecs[vfix, 0:n]))
    return h


def diffusion(geometry, vids, m=1.0, aniso=None, use_cholmod=True):
    """
    Computes heat diffusion from initial vertices in vids using
    backward Euler solution for time t:

      t = m * avg_edge_length^2

    :param
      geometry      TriaMesh or TetMesh, on which to run diffusion
      vids          vertex index or indices where initial heat is applied
      m             factor (default 1) to compute time of heat evolution:
                    t = m * avg_edge_length^2
      use_cholmod   (default True), if Cholmod not installed automatically
                    revert to LU decomposition (slower)

    :return:
      vfunc         heat diffusion at vertices
    """
    if use_cholmod:
        try:
            from sksparse.cholmod import cholesky
        except ImportError:
            use_cholmod = False
    if not use_cholmod:
        from scipy.sparse.linalg import splu
    from .Solver import Solver
    nv = len(geometry.v)
    fem = Solver(geometry, lump=True, aniso=aniso)
    # time of heat evolution:
    t = m * geometry.avg_edge_length() ** 2
    # backward Euler matrix:
    hmat = fem.mass + t * fem.stiffness
    # set initial heat
    b0 = np.zeros((nv,))
    b0[np.array(vids)] = 1.0
    # solve H x = b0
    print("Matrix Format now:  " + hmat.getformat())
    if use_cholmod:
        print("Solver: cholesky decomp - performance optimal ...")
        chol = cholesky(hmat)
        vfunc = chol(b0)
    else:
        print("Package scikit-sparse not found (Cholesky decomp)")
        print("Solver: spsolve (LU decomp) - performance not optimal ...")
        lu = splu(hmat)
        vfunc = lu.solve(np.float32(b0))
    return vfunc
