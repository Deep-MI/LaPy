import numpy as np
import scipy.spatial.distance as di
from .TriaMesh import TriaMesh
from .TetMesh import TetMesh
from .Solver import Solver

# compute shapeDNA
def compute_shapedna(geom, k=50, lump=False, aniso=None, aniso_smooth=10,
    norm=False, rwt=False):
    """
    a function to compute the shapeDNA descriptor for triangle or tetrahedral
    meshes

    Inputs:     geom        geometry; either TriaMesh or TetMesh
                k           number of eigenfunctions / eigenvalues
                lump, aniso, aniso_smooth
                            arguments for 'Solver' class
                norm        whether to perform area / volume normalization
                rwt         whether to perform linear reweighting

    :return:    ev          a dictionary, including 'Eigenvalues' and
                            'Eigenvectors' fields

    """

    # get fem, evals, evecs

    fem = Solver(geom, lump=lump, aniso=aniso, aniso_smooth=aniso_smooth)
    evals, evecs = fem.eigs(k=k)

    # surface / volume normalization

    if norm is True:

        evals = _surf_vol_norm(geom, evals)

    # linear reweighting

    if rwt is True:

        evals = _linear_reweighting(evals)

    # write ev

    evDict = dict()
    evDict['Refine'] = 0
    evDict['Degree'] = 1
    if type(geom).__name__ == "TriaMesh":
        evDict['Dimension'] = 2
    elif type(geom).__name__ == "TetMesh":
        evDict['Dimension'] = 3
    evDict['Elements'] = len(geom.t)
    evDict['DoF'] = len(geom.v)
    evDict['NumEW'] = k
    evDict['Eigenvalues'] = evals
    evDict['Eigenvectors'] = evecs

    # return

    return evDict


# compute shape asysmmetry

def compute_asymmetry(geom1, geom2, dist="euc", k=50, lump=False, aniso=None,
    aniso_smooth=10, norm=False, rwt=False):
    """
    a function to compute the shape asymmetry from two shapeDNA descriptors
    for triangle or tetrahedral meshes

    Inputs:     geom1, geom2
                            geometries; either TriaMesh or TetMesh
                distance    distance measure; currently only 'euc' (euclidean)
                k           number of eigenfunctions / eigenvalues
                lump, aniso, aniso_smooth
                            arguments for 'Solver' class
                norm        whether to perform area / volume normalization
                rwt         whether to perform linear reweighting

    :return:    dst          a distance measure

    """

    ev1 = compute_shapedna(geom1, k=k, lump=lump, aniso=aniso,
        aniso_smooth=aniso_smooth, norm=norm, rwt=rwt)

    ev2 = compute_shapedna(geom2, k=k, lump=lump, aniso=aniso,
        aniso_smooth=aniso_smooth, norm=norm, rwt=rwt)

    if dist == "euc":
        return di.euclidean(ev1["Eigenvalues"][1:], ev2["Eigenvalues"][1:])
    else:
        print("Only euclidean distance is currently implemented.")
        return

# internal function for surface / volume normalization

def _surf_vol_norm(geom, evals):

    if type(geom).__name__ == "TriaMesh":

        vol = geom.area()

        return evals * vol ** np.divide(2.0, 2.0)


    elif type(geom).__name__ == "TetMesh":

        bnd = geom.boundary_tria()

        bnd.orient_()

        vol = bnd.volume()

        return evals * vol ** np.divide(2.0, 3.0)

# internal function linear reweighting

def _linear_reweighting(evals):

    evals[1:] = evals[1:] / np.arange(1, len(evals))

    return evals
