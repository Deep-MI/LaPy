import numpy as np

from .utils._imports import import_optional_dependency


def diagonal(t, x, evecs, evals, n):
    """Compute heat kernel diagonal ( K(t,x,x,) ).

    for a given time t (can be a vector)
    using only the first n smallest eigenvalues and eigenvectors

    Parameters
    ----------
    t : float or np.ndarray
        time or a row vector of time values
    x : np.ndarray
        vertex ids for the positions of K(t,x,x)
    evecs : np.ndarray
        eigenvectors (matrix: vnum x evecsnum)
    evals : np.ndarray
        vector of eigenvalues (col vector: evecsnum x 1)
    n : int
        number of evecs and vals to use (smaller or equal length)

    Returns
    -------
    h:
        matrix, rows: vertices selected in x, cols: times in t
    """
    # maybe add code to check dimensions of input and flip axis if necessary
    h = np.matmul(evecs[x, 0:n] * evecs[x, 0:n], np.exp(-np.matmul(evals[0:n], t)))
    return h


def kernel(t, vfix, evecs, evals, n):
    """Compute heat kernel from all points to a fixed point (vfix).

    for a given time t (using only the first n smallest eigenvalues
    and eigenvectors)

    K_t (p,q) = sum_j exp(-eval_j t) evec_j(p) evec_j(q)

    Parameters
    ----------
    t : number or np.ndarray
        time (can also be a row vector, if passing multiple times)
    vfix : np.ndarray
        fixed vertex index
    evecs : np.ndarray
        matrix of eigenvectors (M x N), M = #vertices, N=#eigenvectors
    evals : np.ndarray
        col vector of eigenvalues (N)
    n : int
        number of eigenvalues/vectors used in heat kernel (n<=N)

    Returns
    -------
    h : np.ndarray
        matrix m rows: all vertices, cols: times in t
    """
    # h = evecs * ( exp(-evals * t) .* repmat(evecs(vfix,:)',1,length(t))  )
    h = np.matmul(evecs[:, 0:n], (np.exp(np.matmul(-evals[0:n], t)) * evecs[vfix, 0:n]))
    return h


def diffusion(geometry, vids, m=1.0, aniso=None, use_cholmod=True):
    """Compute heat diffusion from initial vertices in vids .

    Uses backward Euler solution for time t:

      t = m * avg_edge_length^2

    Parameters
    ----------
    geometry : TriaMesh or TetMesh
        Object on which to run diffusion
    vids : array_like
        vertex index or indices where initial heat is applied
    m : float, default=1.0
        factor  to compute time of heat evolution:
                    t = m * avg_edge_length^2
    aniso : , Default=None

    use_cholmod : Default=True
        if Cholmod is not found
            revert to LU decomposition (slower)

    Returns
    -------
    vfunc: function
        heat diffusion at vertices
    """
    sksparse = import_optional_dependency("sksparse", raise_error=use_cholmod)
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
    if sksparse is not None:
        print("Solver: Cholesky decomposition from scikit-sparse cholmod ...")
        chol = sksparse.choldmod.cholesky(hmat)
        vfunc = chol(b0)
    else:
        from scipy.sparse.linalg import splu

        print("Solver: spsolve (LU decomposition) ...")
        lu = splu(hmat)
        vfunc = lu.solve(np.float32(b0))
    return vfunc
