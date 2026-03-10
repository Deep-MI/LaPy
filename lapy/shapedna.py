"""Functions for computing and comparing Laplace spectra.

Includes code for solving the anisotropic Laplace-Beltrami eigenvalue
problem as well as functions for normalization and comparison of
Laplace spectra (shapeDNA descriptors).

The shapeDNA is a descriptor based on the eigenvalues and eigenvectors
of the Laplace-Beltrami operator and can be used for shape analysis
and comparison.
"""
import logging
from typing import TYPE_CHECKING, Union

import numpy as np
import scipy.spatial.distance as di

from . import Solver

if TYPE_CHECKING:
    from .tet_mesh import TetMesh
    from .tria_mesh import TriaMesh

logger = logging.getLogger(__name__)

def _positive_measure(value: float, name: str) -> float:
    """Validate that a measure is positive.

    Parameters
    ----------
    value : float
        The measure value to validate.
    name : str
        Name of the measure for error messages.

    Returns
    -------
    float
        The validated positive measure value.

    Raises
    ------
    ValueError
        If value is not positive (<=0).
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive for normalization")
    return value


def _boundary_volume(geom: "TetMesh") -> float:
    """Compute the volume enclosed by the boundary of a tetrahedral mesh.

    Parameters
    ----------
    geom : TetMesh
        Tetrahedral mesh geometry.

    Returns
    -------
    float
        Volume enclosed by the oriented boundary surface.

    Raises
    ------
    ValueError
        If boundary volume is not positive.
    """
    bnd = geom.boundary_tria()
    bnd.orient_()
    return _positive_measure(bnd.volume(), "boundary volume")


def _surface_measure(geom: "TriaMesh") -> float:
    """Compute the surface area of a triangle mesh.

    Parameters
    ----------
    geom : TriaMesh
        Triangle mesh geometry.

    Returns
    -------
    float
        Surface area of the mesh.

    Raises
    ------
    ValueError
        If area is not positive.
    """
    area = _positive_measure(geom.area(), "area")
    return area



def compute_shapedna(
    geom: Union["TriaMesh", "TetMesh"],
    k: int = 50,
    lump: bool = False,
    aniso: float | tuple[float, float] | None = None,
    aniso_smooth: int = 10,
    use_cholmod: bool = False,
) -> dict:
    """Compute the shapeDNA descriptor for triangle or tetrahedral meshes.

    The shapeDNA descriptor consists of the eigenvalues and eigenvectors of
    the Laplace-Beltrami operator and can be used for shape analysis and
    comparison.

    Parameters
    ----------
    geom : TriaMesh | TetMesh
        Mesh geometry.
    k : int, default=50
        Number of eigenvalues/eigenvectors to compute.
    lump : bool, default=False
        If True, lump the mass matrix (diagonal). See `lapy.Solver` class.
    aniso : float | tuple of shape (2,) | None, default=None
        Anisotropy for curvature-based anisotropic Laplace. See `lapy.Solver`
        class.
    aniso_smooth : int, default=10
        Number of smoothing iterations for curvature computation on vertices.
        See `lapy.Solver` class.
    use_cholmod : bool, default=False
        If True, attempts to use the Cholesky decomposition for improved
        execution speed. Requires the ``scikit-sparse`` library. If it cannot
        be found, an error will be thrown. If False, will use slower LU
        decomposition.

    Returns
    -------
    dict
        A dictionary with the following keys:

        - 'Refine' : int - Refinement level (0)
        - 'Degree' : int - Polynomial degree (1)
        - 'Dimension' : int - Mesh dimension (2 for TriaMesh, 3 for TetMesh)
        - 'Elements' : int - Number of mesh elements
        - 'DoF' : int - Degrees of freedom (number of vertices)
        - 'NumEW' : int - Number of eigenvalues computed
        - 'Eigenvalues' : np.ndarray - Array of eigenvalues, shape (k,)
        - 'Eigenvectors' : np.ndarray - Array of eigenvectors, shape (n_vertices, k)
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


def normalize_ev(
    geom: Union["TriaMesh", "TetMesh"],
    evals: np.ndarray,
    method: str = "geometry",
) -> np.ndarray:
    """Normalize eigenvalues for a 2D surface or a 3D solid.

    Normalizes eigenvalues to unit surface area or unit volume,
    enabling meaningful comparison between different shapes.

    Parameters
    ----------
    geom : TriaMesh or TetMesh
        Mesh geometry.
    evals : np.ndarray
        Set of sorted eigenvalues, shape (k,).
    method : {'surface', 'volume', 'geometry'}, default='geometry'
        Normalization method:

        - 'surface': Normalize to unit surface area
        - 'volume': Normalize to unit volume
        - 'geometry': Automatically choose surface for TriaMesh, volume for TetMesh

    Returns
    -------
    np.ndarray
        Vector of re-weighted eigenvalues, shape (k,).

    Raises
    ------
    ValueError
        If the method is not one of 'surface', 'volume', or 'geometry'.
        If the geometry type is unsupported for the chosen normalization.
        If method=volume and the volume of a surface is not defined.

    Notes
    -----
    For TriaMesh with 'volume' method, the mesh must be closed and oriented
    to compute a valid enclosed volume.
    """
    geom_type = type(geom).__name__
    if method == "surface":
        return evals * _surface_measure(geom)

    if method == "volume":
        if geom_type == "TriaMesh":
            return evals * _positive_measure(geom.volume(), "volume") ** (2.0 / 3.0)
        if geom_type == "TetMesh":
            return evals * _boundary_volume(geom) ** (2.0 / 3.0)
        raise ValueError("Unsupported geometry type for volume normalization")

    if method == "geometry":
        if geom_type == "TriaMesh":
            return evals * _surface_measure(geom)
        if geom_type == "TetMesh":
            return evals * _boundary_volume(geom) ** (2.0 / 3.0)
        raise ValueError("Unsupported geometry type for geometry normalization")

    raise ValueError(f"Unknown normalization method: {method}")


def reweight_ev(evals: np.ndarray) -> np.ndarray:
    """Apply linear re-weighting to eigenvalues.

    Divides each eigenvalue by its index to reduce the influence of higher
    eigenvalues, which tend to be less stable.

    Parameters
    ----------
    evals : np.ndarray
        Set of sorted eigenvalues, shape (k,).

    Returns
    -------
    np.ndarray
        Vector of re-weighted eigenvalues, shape (k,). Each eigenvalue is
        divided by its 1-based index: ``evals[i] / (i+1)``.

    Notes
    -----
    This reweighting scheme gives less importance to higher eigenvalues, which
    are typically more sensitive to discretization and numerical errors.
    """
    # evals[1:] = evals[1:] / np.arange(1, len(evals))
    evals = evals / np.arange(1, len(evals) + 1)

    return evals


def compute_distance(
    ev1: np.ndarray, ev2: np.ndarray, dist: str = "euc"
) -> float:
    """Compute the shape dissimilarity from two shapeDNA descriptors.

    Computes a distance metric between two sets of eigenvalues to quantify
    the dissimilarity between two shapes.

    Parameters
    ----------
    ev1 : np.ndarray
        First set of sorted eigenvalues, shape (k,).
    ev2 : np.ndarray
        Second set of sorted eigenvalues, shape (k,).
    dist : {'euc'}, default='euc'
        Distance measure. Currently only 'euc' (Euclidean) is implemented.

    Returns
    -------
    float
        Distance between the eigenvalue arrays.

    Raises
    ------
    ValueError
        If dist is not 'euc' (other distance metrics not yet implemented).

    Notes
    -----
    The eigenvalue arrays should have the same length and be normalized and
    reweighted in the same way for meaningful comparison.
    """
    if dist == "euc":
        return di.euclidean(ev1, ev2)
    else:
        logger.warning(
            "Only Euclidean distance is currently implemented; received %s", dist
        )
        raise ValueError(f"Distance metric {dist} is not implemented.")