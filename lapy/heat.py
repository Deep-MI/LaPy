"""Functions for computing heat kernel and diffusion.

Inputs are eigenvalues and eigenvectors (for heat kernel) and the
mesh geometries (tet or tria mesh) for heat diffusion. 
"""

import importlib
from typing import Optional

import numpy as np

from .utils._imports import import_optional_dependency


def diagonal(t, x, evecs, evals, n):
    """Compute heat kernel diagonal ( K(t,x,x,) ).

    for a given time t (can be a vector)
    using only the first n smallest eigenvalues and eigenvectors

    Parameters
    ----------
    t : float | array
        time or a row vector of time values
    x : array
        vertex ids for the positions of K(t,x,x)
    evecs : array
        eigenvectors (matrix: vnum x evecsnum)
    evals : array
        vector of eigenvalues (col vector: evecsnum x 1)
    n : int
        number of evecs and vals to use (smaller or equal length)

    Returns
    -------
    h : array
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
    t : float | array
        time (can also be a row vector, if passing multiple times)
    vfix : array
        fixed vertex index
    evecs : array
        matrix of eigenvectors (M x N), M = #vertices, N=#eigenvectors
    evals : array
        col vector of eigenvalues (N)
    n : int
        number of eigenvalues/vectors used in heat kernel (n<=N)

    Returns
    -------
    h : array
        matrix m rows: all vertices, cols: times in t
    """
    # h = evecs * ( exp(-evals * t) .* repmat(evecs(vfix,:)',1,length(t))  )
    h = np.matmul(evecs[:, 0:n], (np.exp(np.matmul(-evals[0:n], t)) * evecs[vfix, 0:n]))
    return h


def diffusion(geometry, vids, m=1.0, aniso: Optional[int] = None, use_cholmod=False):
    """Compute the heat diffusion from initial vertices in vids.

    It uses the backward Euler solution: ``t = m * avg_edge_length^2``.

    Parameters
    ----------
    geometry : TriaMesh | TetMesh
        Object on which to run diffusion
    vids : array
        vertex index or indices where initial heat is applied
    m : float, default=1.0
        factor  to compute time of heat evolution:
                    t = m * avg_edge_length^2
    aniso : int
        Number of smoothing iterations for curvature computation on vertices.
    use_cholmod : bool, default=False
        Which solver to use:
            * True : Use Cholesky decomposition from scikit-sparse cholmod
            * False: Use spsolve (LU decomposition)

    Returns
    -------
    vfunc: function
        heat diffusion at vertices
    """
    if use_cholmod:
        sksparse = import_optional_dependency("sksparse", raise_error=True)
        importlib.import_module(".cholmod", sksparse.__name__)
    else:
        sksparse = None
    from . import Solver

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
        print("Solver: Cholesky decomposition from scikit-sparse cholmod ...")
        chol = sksparse.cholmod.cholesky(hmat)
        vfunc = chol(b0)
    else:
        from scipy.sparse.linalg import splu

        print("Solver: spsolve (LU decomposition) ...")
        lu = splu(hmat)
        vfunc = lu.solve(np.float32(b0))
    return vfunc
