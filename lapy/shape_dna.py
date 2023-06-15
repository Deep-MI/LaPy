"""Functions for computing and comparing Laplace spectra.

Includes code for solving the anisotropic Laplace-Beltrami eigenvalue
problem as well as functions for normalization and comparison of 
Laplace spectra. 
"""

import numpy as np
import scipy.spatial.distance as di

from . import Solver


def compute_shapedna(
    geom, k=50, lump=False, aniso=None, aniso_smooth=10, use_cholmod=False
):
    """Compute the shapeDNA descriptor for triangle or tetrahedral meshes.

    Parameters
    ----------
    geom : TriaMesh or TetMesh
        Mesh geometry.
    k : int, default=50
        Number of eigenfunctions / eigenvalues.
    lump : bool, default=False
        If True, lump the mass matrix (diagonal).
            (See 'lapy.Solver.Solver' class).
    aniso : float or tuple of shape (2,)
        Anisotropy for curvature based anisotopic Laplace.
            (See 'lapy.Solver.Solver' class).
    aniso_smooth : int
        Number of smoothing iterations for curvature computation on vertices.
            (See 'lapy.Solver.Solver' class).
    use_cholmod : bool, default: False
        If True, attempts to use the Cholesky decomposition for improved execution
        speed. Requires the ``scikit-sparse`` library. If it can not be found, an error
        will be thrown.
        If False, will use slower LU decomposition.

    Returns
    -------
    ev : dict
        A dictionary, including 'Eigenvalues' and 'Eigenvectors' fields.
    """
    # get fem, evals, evecs

    fem = Solver(
        geom, lump=lump, aniso=aniso, aniso_smooth=aniso_smooth, use_cholmod=use_cholmod
    )
    evals, evecs = fem.eigs(k=k)

    # write ev

    evDict = dict()
    evDict["Refine"] = 0
    evDict["Degree"] = 1
    if type(geom).__name__ == "TriaMesh":
        evDict["Dimension"] = 2
    elif type(geom).__name__ == "TetMesh":
        evDict["Dimension"] = 3
    evDict["Elements"] = len(geom.t)
    evDict["DoF"] = len(geom.v)
    evDict["NumEW"] = k
    evDict["Eigenvalues"] = evals
    evDict["Eigenvectors"] = evecs

    return evDict


def normalize_ev(geom, evals, method="geometry"):
    """Normalize a surface or a volume.

    Parameters
    ----------
    geom : TriaMesh or TetMesh
        Mesh geometry.
    evals : array_like
        Set of sorted eigenvalues.
    method : str
        Either "surface", "volume", or "geometry";
        "geometry" will perform surface normalization for
        2D objects, and volume normalization for 3D objects.

    Returns
    -------
    array_like
        Vector of re-weighted eigenvalues.
    """
    if method == "surface":
        vol = geom.area()

        return evals * vol ** np.divide(2.0, 2.0)

    elif method == "volume":
        if type(geom).__name__ == "TriaMesh":
            geom.orient_()

            vol = geom.volume()

        elif type(geom).__name__ == "TetMesh":
            bnd = geom.boundary_tria()

            bnd.orient_()

            vol = bnd.volume()

        return evals * vol ** np.divide(2.0, 3.0)

    elif method == "geometry":
        if type(geom).__name__ == "TriaMesh":
            vol = geom.area()

            return evals * vol ** np.divide(2.0, 2.0)

        elif type(geom).__name__ == "TetMesh":
            bnd = geom.boundary_tria()

            bnd.orient_()

            vol = bnd.volume()

            return evals * vol ** np.divide(2.0, 3.0)


def reweight_ev(evals):
    """Apply linear re-weighting.

    Parameters
    ----------
    evals : array_like
        Set of sorted eigenvalues.

    Returns
    -------
    evals: array_like
        Vector of re-weighted eigenvalues.
    """
    # evals[1:] = evals[1:] / np.arange(1, len(evals))
    evals = evals / np.arange(1, len(evals) + 1)

    return evals


def compute_distance(ev1, ev2, dist="euc"):
    """Compute the shape dissimilarity from two shapeDNA descriptors.

    Parameters
    ----------
    ev1 : array_like
        First set of sorted eigenvalues.
    ev2 : array_like
        Second set of sorted eigenvalues.
    dist : str
        Distance measure; currently only 'euc' (Euclidean).

    Returns
    -------
    * : float
        Distance between the eigenvalue arrays.
    """
    if dist == "euc":
        return di.euclidean(ev1, ev2)
    else:
        print("Only euclidean distance is currently implemented.")
        return
