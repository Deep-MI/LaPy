import numpy as np
import scipy.spatial.distance as di

from .Solver import Solver
from .TetMesh import TetMesh  # noqa: F401
from .TriaMesh import TriaMesh  # noqa: F401


def compute_shapedna(geom, k=50, lump=False, aniso=None, aniso_smooth=10):
    """Computethe shapeDNA descriptor for triangle or tetrahedral mesh.

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

    Returns
    -------
    ev : dict
         a dictionary, including 'Eigenvalues' and 'Eigenvectors' fields
    """
    # get fem, evals, evecs

    fem = Solver(geom, lump=lump, aniso=aniso, aniso_smooth=aniso_smooth)
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


def reweight_ev(evals):
    """Apply linear reweighting.

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


def compute_distance(ev1, ev2, dist="euc"):
    """Compute the shape asymmetry from two shapeDNA descriptors.

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
