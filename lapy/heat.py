"""Functions for computing heat kernel and diffusion.

Inputs are eigenvalues and eigenvectors (for heat kernel) and the
mesh geometries (tet or tria mesh) for heat diffusion.
"""

from __future__ import annotations

import importlib
import logging

import numpy as np

from .utils._imports import import_optional_dependency

logger = logging.getLogger(__name__)


def diagonal(
    t: float | np.ndarray,
    x: np.ndarray,
    evecs: np.ndarray,
    evals: np.ndarray,
    n: int,
) -> np.ndarray:
    """Compute heat kernel diagonal K(t,x,x).

    For a given time t (can be a vector) using only the first n smallest
    eigenvalues and eigenvectors.

    Parameters
    ----------
    t : float or np.ndarray
        Time or array of time values, shape (n_times,).
    x : np.ndarray
        Vertex indices for the positions of K(t,x,x), shape (n_vertices,).
    evecs : np.ndarray
        Eigenvectors matrix, shape (n_vertices, n_eigenvectors).
    evals : np.ndarray
        Vector of eigenvalues, shape (n_eigenvalues,).
    n : int
        Number of eigenvectors and eigenvalues to use (smaller or equal to length).

    Returns
    -------
    np.ndarray
        Heat kernel diagonal values. Shape (n_vertices, n_times) if t is array,
        or (n_vertices, 1) if t is scalar. Rows correspond to vertices selected
        in x, columns to times in t.

    Raises
    ------
    ValueError
        If n exceeds the number of available eigenpairs.
    """
    if n > evecs.shape[1] or n > evals.shape[0]:
        raise ValueError("n exceeds the number of available eigenpairs")
    # maybe add code to check dimensions of input and flip axis if necessary
    h = np.matmul(evecs[x, 0:n] * evecs[x, 0:n], np.exp(-np.matmul(evals[0:n], t)))
    return h


def kernel(
    t: float | np.ndarray,
    vfix: int,
    evecs: np.ndarray,
    evals: np.ndarray,
    n: int,
) -> np.ndarray:
    r"""Compute heat kernel from all points to a fixed point.

    For a given time t, computes K_t(p,q) using only the first n smallest
    eigenvalues and eigenvectors:

    .. math::
        K_t (p,q) = \sum_j \exp(-\lambda_j t) \phi_j(p) \phi_j(q)

    where :math:`\lambda_j` are eigenvalues and :math:`\phi_j` are eigenvectors.

    Parameters
    ----------
    t : float or np.ndarray
        Time (can also be array, if passing multiple times), shape (n_times,).
    vfix : int
        Fixed vertex index.
    evecs : np.ndarray
        Matrix of eigenvectors, shape (n_vertices, n_eigenvectors).
    evals : np.ndarray
        Column vector of eigenvalues, shape (n_eigenvalues,).
    n : int
        Number of eigenvalues/vectors used in heat kernel (n <= n_eigenvectors).

    Returns
    -------
    np.ndarray
        Heat kernel values. Shape (n_vertices, n_times) if t is array,
        or (n_vertices, 1) if t is scalar. Rows correspond to all vertices,
        columns to times in t.

    Raises
    ------
    ValueError
        If n exceeds the number of available eigenpairs.
    """
    # h = evecs * ( exp(-evals * t) .* repmat(evecs(vfix,:)',1,length(t))  )
    if n > evecs.shape[1] or n > evals.shape[0]:
        raise ValueError("n exceeds the number of available eigenpairs")
    h = np.matmul(
        evecs[:, 0:n], (np.exp(np.matmul(-evals[0:n], t)) * evecs[vfix, 0:n])
    )
    return h


def diffusion(
    geometry: object,
    vids: int | np.ndarray | list[int] | list[list[int]] | list[np.ndarray],
    m: float = 1.0,
    aniso: int | None = None,
    use_cholmod: bool = False,
) -> np.ndarray:
    """Compute the heat diffusion from initial vertices in vids.

    Uses the backward Euler solution with time :math:`t = m l^2`, where l
    describes the average edge length.

    Parameters
    ----------
    geometry : TriaMesh or TetMesh
        Geometric object on which to run diffusion.
    vids : int, np.ndarray, list[int], list[list[int]], or list[np.ndarray]
        Vertex index or indices where initial heat is applied.

        The nesting level determines whether one or multiple seed sets are
        computed:

        - **Single seed set** — ``int``, ``np.ndarray``, or ``list[int]``:
          heat is seeded at those vertices and a 1-D result array of shape
          ``(n_vertices,)`` is returned.
        - **Multiple seed sets** — ``list[list[int]]`` or
          ``list[np.ndarray]``: each inner sequence defines one independent
          seed set.  The heat matrix is factorised only once and reused for
          all sets.  A 2-D array of shape ``(n_vertices, n_cases)`` is
          returned, where column ``k`` corresponds to ``vids[k]``.

        .. note::
            The direct output of :func:`~lapy.TriaMesh.boundary_loops` is a
            ``list[list[int]]`` and therefore treated as **multiple seed
            sets** (one diffusion per boundary loop).  To compute the
            distance from *all* boundary vertices at once, concatenate the
            loops first::

                loops = tria.boundary_loops()
                all_bnd = np.concatenate(loops)
                vfunc = diffusion(tria, all_bnd)

    m : float, default=1.0
        Factor to compute time of heat evolution.
    aniso : int, default=None
        Number of smoothing iterations for curvature computation on vertices.
    use_cholmod : bool, default=False
        Which solver to use. If True, use Cholesky decomposition from
        scikit-sparse cholmod. If False, use spsolve (LU decomposition).

    Returns
    -------
    np.ndarray
        Heat diffusion values at vertices.  Shape ``(n_vertices,)`` for a
        single seed set, or ``(n_vertices, n_cases)`` for multiple seed sets.

    Raises
    ------
    ValueError
        If vids contains out-of-range vertex indices.
    ImportError
        If use_cholmod is True but scikit-sparse is not installed.
    """
    if use_cholmod:
        sksparse = import_optional_dependency("sksparse", raise_error=True)
        importlib.import_module(".cholmod", sksparse.__name__)
    else:
        sksparse = None
    from . import Solver

    nv = len(geometry.v)

    # ------------------------------------------------------------------
    # Nesting level determines single vs. multiple seed sets:
    #   Multiple: list[list[int]] or list[np.ndarray]  → scalar_input=False
    #   Single  : int, np.ndarray, or list[int]        → scalar_input=True
    # ------------------------------------------------------------------
    if isinstance(vids, list) and len(vids) > 0 and isinstance(
        vids[0], (list, np.ndarray)
    ):
        scalar_input = False
        vids_list = [np.asarray(v, dtype=int).ravel() for v in vids]
    else:
        scalar_input = True
        vids_list = [np.asarray(vids, dtype=int).ravel()]

    for v in vids_list:
        if np.any(v < 0) or np.any(v >= nv):
            raise ValueError("vids contains out-of-range vertex indices")

    fem = Solver(geometry, lump=True, aniso=aniso)
    # time of heat evolution:
    t = m * geometry.avg_edge_length() ** 2
    # backward Euler matrix (same for all cases):
    hmat = fem.mass + t * fem.stiffness

    # Build rhs matrix — each column is one initial-heat vector
    n_cases = len(vids_list)
    b0 = np.zeros((nv, n_cases))
    for k, v in enumerate(vids_list):
        b0[v, k] = 1.0

    # Solve — both splu.solve and cholmod accept 2-D rhs natively
    logger.debug("Matrix Format: %s", hmat.getformat())
    if use_cholmod:
        logger.info("Solver: Cholesky decomposition from scikit-sparse cholmod")
        chol = sksparse.cholmod.cholesky(hmat)
        vfunc = chol(b0)
    else:
        from scipy.sparse.linalg import splu

        logger.info("Solver: LU decomposition via splu")
        lu = splu(hmat)
        vfunc = lu.solve(b0.astype(np.float64))

    # Return 1-D array for the single-case (backward compatible) path
    if scalar_input:
        return vfunc[:, 0]
    return vfunc
