import numpy as np
import scipy.spatial.distance as di

from .Solver import Solver
from .TetMesh import TetMesh  # noqa: F401
from .TriaMesh import TriaMesh  # noqa: F401

# compute shapeDNA


def compute_shapedna(geom, k=50, lump=False, aniso=None, aniso_smooth=10, use_cholmod=False):
    """
    a function to compute the shapeDNA descriptor for triangle or tetrahedral
    meshes

    Parameters
    ----------
    geom : TriaMesh or TetMesh
        geometry object
    k : int, default=50
        number of eigenfunctions / eigenvalues
    lump : bool, Default=False
        If True, lump the mass matrix (diagonal)
            (See 'lapy.Solver.Solver' class)
    aniso :  float or tuple of shape (2,)
        Anisotropy for curvature based anisotopic Laplace.
            (See 'lapy.Solver.Solver' class)
    aniso_smooth : int
        Number of smoothing iterations for curvature computation on vertices.
            (See 'lapy.Solver.Solver' class)
    use_cholmod : bool, default: False
        If True, attempts to use the Cholesky decomposition for improved execution
        speed. Requires the ``scikit-sparse`` library. If it can not be found, an error 
        will be thrown.
        If False, will use slower LU decomposition.            

    Returns
    -------
    ev : dict
         a dictionary, including 'Eigenvalues' and 'Eigenvectors' fields
    """

    # get fem, evals, evecs

    fem = Solver(geom, lump=lump, aniso=aniso, aniso_smooth=aniso_smooth, use_cholmod=use_cholmod)
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


# function for ev normalization


def normalize_ev(geom, evals, method="geometry"):
    """
    a function for surface / volume normalization

    Parameters
    ----------
    geom : TriaMesh or TetMesh
        geometry object
    evals : array_like
        vector of eigenvalues
    method : str
        either "surface", "volume", or "geometry";
        "geometry" will perform surface normalization for
        2D objects, and volume normalization for 3D objects

    Returns
    -------
    array_like
        vector of reweighted eigenvalues
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


# function for linear reweighting


def reweight_ev(evals):
    """
    a function for linear reweighting

    Parameters
    ----------
    evals : array_like
        vector of eigenvalues

    Returns
    -------
    evals: array_like
        vector of reweighted eigenvalues
    """

    # evals[1:] = evals[1:] / np.arange(1, len(evals))
    evals = evals / np.arange(1, len(evals) + 1)

    return evals


# compute distance


def compute_distance(ev1, ev2, dist="euc"):
    """
    a function to compute the shape asymmetry from two shapeDNA descriptors
    for triangle or tetrahedral meshes

    Parameters
    ----------
    ev1, ev2 : float
        eigenvalues
    dist : str
        distance measure; currently only 'euc' (euclidean)

    Returns
    -------
    * :  double
        a distance measure
    """

    if dist == "euc":
        return di.euclidean(ev1, ev2)
    else:
        print("Only euclidean distance is currently implemented.")
        return
