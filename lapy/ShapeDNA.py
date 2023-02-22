import numpy as np
import scipy.spatial.distance as di

from .Solver import Solver
from .TetMesh import TetMesh
from .TriaMesh import TriaMesh

# compute shapeDNA


def compute_shapedna(geom, k=50, lump=False, aniso=None, aniso_smooth=10):
    """
    a function to compute the shapeDNA descriptor for triangle or tetrahedral
    meshes

    Inputs:     geom        geometry object; either TriaMesh or TetMesh
                k           number of eigenfunctions / eigenvalues
                lump, aniso, aniso_smooth
                            arguments for 'Solver' class

    :return:    ev          a dictionary, including 'Eigenvalues' and
                            'Eigenvectors' fields

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

    # return

    return evDict


# function for ev normalization


def normalize_ev(geom, evals, method="geometry"):
    """
    a function for surface / volume normalization

    Inputs:     geom         geometry object; either TriaMesh or TetMesh
                evals        vector of eigenvalues
                method       either "surface", "volume", or "geometry";
                             "geometry" will perform surface normalization for
                             2D objects, and volume normalization for 3D objects

    :return:    evals        vector of reweighted eigenvalues
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

    Inputs:     evals        vector of eigenvalues

    :return:    evals        vector of reweighted eigenvalues
    """

    # evals[1:] = evals[1:] / np.arange(1, len(evals))
    evals = evals / np.arange(1, len(evals) + 1)

    return evals


# compute distance


def compute_distance(ev1, ev2, dist="euc"):
    """
    a function to compute the shape asymmetry from two shapeDNA descriptors
    for triangle or tetrahedral meshes

    Inputs:     ev1, ev2    eigenvalues
                distance    distance measure; currently only 'euc' (euclidean)

    :return:    dst         a distance measure
    """

    if dist == "euc":
        return di.euclidean(ev1, ev2)
    else:
        print("Only euclidean distance is currently implemented.")
        return
