from typing import Union

import numpy as np
import scipy.spatial.distance as di

from lapy import configuration, messages
from lapy.Solver import Solver
from lapy.TetMesh import TetMesh
from lapy.TriaMesh import TriaMesh


def compute_shapedna(
    geom: Union[TetMesh, TriaMesh],
    k: int = 50,
    lump: bool = False,
    aniso: float = None,
    aniso_smooth: int = 10,
):
    """
    Compute the ShapeDNA descriptor for triangle or tetrahedron meshes.

    Parameters
    ----------
    geometry : Union[TriaMesh, TetMesh]
        A geometry class, currently either TriaMesh or TetMesh
    lump : bool, optional
        Whether to lump the mass matrix (diagonal), by default False
    aniso : float, optional
        Anisotropy for curvature based anisotopic Laplace can also be tuple
        (a_min, a_max) to differentially affect the min and max curvature
        directions. E.g. (0,50) will set scaling to 1 into min curv
        direction even if the max curvature is large in those regions
        (= isotropic in regions with large max curv and min curv close to
        zero= concave cylinder), by default None
    aniso_smooth : int, optional
        _description_, by default 10

    Returns
    -------
    dict
        ShapeDNA, including 'Eigenvalues' and 'Eigenvectors' fields

    """
    solver = Solver(geom, lump=lump, aniso=aniso, aniso_smooth=aniso_smooth)
    evals, evecs = solver.eigs(k=k)

    evDict = configuration.SHAPEDNA_DEFAULTS.copy()
    if isinstance(geom, TriaMesh):
        evDict["Dimension"] = 2
    elif isinstance(geom, TetMesh):
        evDict["Dimension"] = 3
    evDict["Elements"] = len(geom.t)
    evDict["DoF"] = len(geom.v)
    evDict["NumEW"] = k
    evDict["Eigenvalues"] = evals
    evDict["Eigenvectors"] = evecs

    return evDict


def normalize_ev(
    geom: Union[TriaMesh, TetMesh], evals: np.ndarray, method: str = "geometry"
) -> np.ndarray:
    """
    A function for surface / volume normalization.

    Parameters
    ----------
    geom : Union[TriaMesh, TetMesh]
        Geometry object; either TriaMesh or TetMesh
    evals : np.ndarray
        Vector of eigenvalues
    method : str, optional
        either "surface", "volume", or "geometry"; "geometry" will perform
        surface normalization for 2D objects, and volume normalization for 3D
        objects, by default "geometry"

    Returns
    -------
    np.ndarray
        Vector of reweighted eigenvalues
    """

    if method == "surface":
        vol = geom.area()
        return evals * vol ** np.divide(2.0, 2.0)
    elif method == "volume":
        if isinstance(geom, TriaMesh):
            geom.orient_()
            vol = geom.volume()
        elif isinstance(geom, TetMesh):
            bnd = geom.boundary_tria()
            bnd.orient_()
            vol = bnd.volume()
        return evals * vol ** np.divide(2.0, 3.0)
    elif method == "geometry":
        if isinstance(geom, TriaMesh):
            vol = geom.area()
            return evals * vol ** np.divide(2.0, 2.0)
        elif isinstance(geom, TetMesh):
            bnd = geom.boundary_tria()
            bnd.orient_()
            vol = bnd.volume()
            return evals * vol ** np.divide(2.0, 3.0)
    else:
        raise NotImplementedError(messages.INVALID_NORMALIZATION_METHOD)


def reweight_ev(eigenvalues: np.ndarray) -> np.ndarray:
    """
    Reweights eigenvalues by index.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues

    Returns
    -------
    np.ndarray
        Reweighted eigenvalues
    """
    return eigenvalues / np.arange(1, len(eigenvalues) + 1)


def compute_distance(
    ev1: np.ndarray, ev2: np.ndarray, distance: str = "euc"
) -> float:
    """
    A function to compute the shape asymmetry from two ShapeDNA descriptors
    for triangle or tetrahedron meshes.

    Parameters
    ----------
    ev1, ev2 : np.ndarray
        Eigenvalues vector
    distance : str, optional
        distance measure; currently only 'euc' (euclidean)

    Returns
    -------
    float
        A distance measure
    """
    if distance == "euc":
        return di.euclidean(ev1, ev2)
    else:
        raise NotImplementedError(messages.INVALID_DISTANCE_KEY)
