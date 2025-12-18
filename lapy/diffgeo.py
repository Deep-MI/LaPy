"""Differential Geometry Functions for meshes.

This module includes gradient, divergence, curvature, geodesics,
mean curvature flow etc.

Note, the interface is not yet final, some functions are split into
tet and tria versions.
"""

import importlib
import logging
from typing import Union

import numpy as np
from scipy import sparse

from . import Solver, TriaMesh
from .utils._imports import import_optional_dependency

logger = logging.getLogger(__name__)

def compute_gradient(geom: Union["TriaMesh", "TetMesh"], vfunc: np.ndarray) -> np.ndarray:
    """Compute gradient of a vertex function f.

    Parameters
    ----------
    geom : TriaMesh or TetMesh
        Mesh geometry.
    vfunc : np.ndarray
        Scalar function at vertices, shape (n_vertices,).

    Returns
    -------
    np.ndarray
        3D gradient vector at each element, shape (n_elements, 3).

    Raises
    ------
    ValueError
        If unknown geometry type.
    """
    if type(geom).__name__ == "TriaMesh":
        return tria_compute_gradient(geom, vfunc)
    elif type(geom).__name__ == "TetMesh":
        return tet_compute_gradient(geom, vfunc)
    else:
        raise ValueError('Geometry type "' + type(geom).__name__ + '" unknown')


def compute_divergence(geom: Union["TriaMesh", "TetMesh"], vfunc: np.ndarray) -> np.ndarray:
    """Compute divergence of a vertex function f.

    Parameters
    ----------
    geom : TriaMesh or TetMesh
        Mesh geometry.
    vfunc : np.ndarray
        Scalar function at vertices, shape (n_vertices,).

    Returns
    -------
    np.ndarray
        Scalar function of divergence at vertices, shape (n_vertices,).

    Raises
    ------
    ValueError
        If unknown geometry type.
    """
    if type(geom).__name__ == "TriaMesh":
        return tria_compute_divergence(geom, vfunc)
    elif type(geom).__name__ == "TetMesh":
        return tet_compute_divergence(geom, vfunc)
    else:
        raise ValueError('Geometry type "' + type(geom).__name__ + '" unknown')


def compute_rotated_f(geom: "TriaMesh", vfunc: np.ndarray) -> np.ndarray:
    """Compute function whose level sets are orthgonal to the ones of vfunc.

    Parameters
    ----------
    geom : TriaMesh
        Mesh geometry.
    vfunc : np.ndarray
        Scalar function at vertices, shape (n_vertices,).

    Returns
    -------
    np.ndarray
        Rotated function at vertices, shape (n_vertices,).

    Raises
    ------
    ValueError
        If unknown geometry type.
    """
    if type(geom).__name__ == "TriaMesh":
        return tria_compute_rotated_f(geom, vfunc)
    else:
        raise ValueError('Geometry type "' + type(geom).__name__ + '" not implemented')


def compute_geodesic_f(geom: Union["TriaMesh", "TetMesh"], vfunc: np.ndarray) -> np.ndarray:
    """Compute function with normalized gradient (geodesic distance).

    Computes gradient, normalizes it and computes function with this normalized
    gradient by solving the Poisson equation with the divergence of grad.
    This idea is also described in the paper "Geodesics in Heat" for triangles.

    Parameters
    ----------
    geom : TriaMesh or TetMesh
        Mesh geometry.
    vfunc : np.ndarray
        Scalar function at vertices, shape (n_vertices,).

    Returns
    -------
    np.ndarray
        Scalar geodesic function at vertices, shape (n_vertices,).
    """
    gradf = compute_gradient(geom, vfunc)
    # normalize gradient
    gradnorm = gradf / np.sqrt((gradf**2).sum(1))[:, np.newaxis]
    gradnorm = np.nan_to_num(gradnorm)
    divf = compute_divergence(geom, gradnorm)
    fem = Solver(geom, lump=True)
    # as long as div does not care about weighing with a Bi,
    #   we can pass identity instead of B here:
    fem.mass = sparse.eye(fem.stiffness.shape[0], dtype=fem.stiffness.dtype)
    vf = fem.poisson(divf)
    vf -= min(vf)
    return vf


def tria_compute_geodesic_f(tria: TriaMesh, vfunc: np.ndarray) -> np.ndarray:
    """Compute function with normalized gradient (geodesic distance).

    Computes gradient, normalizes it and computes function with this normalized
    gradient by solving the Poisson equation with the divergence of grad.
    This idea is also described in the paper "Geodesics in Heat".

    Parameters
    ----------
    tria : TriaMesh
        Triangle mesh.
    vfunc : np.ndarray
        Scalar function at vertices, shape (n_vertices,).

    Returns
    -------
    np.ndarray
        Scalar geodesic function at vertices, shape (n_vertices,).
    """
    gradf = tria_compute_gradient(tria, vfunc)
    # normalize gradient
    gradnorm = gradf / np.sqrt((gradf**2).sum(1))[:, np.newaxis]
    gradnorm = np.nan_to_num(gradnorm)
    divf = tria_compute_divergence(tria, gradnorm)
    fem = Solver(tria)
    # as long as div does not care about weighing with a Bi,
    #   we can pass identity instead of B here:
    # div is the integrated divergence (so it is already B*div)
    fem.mass = sparse.eye(fem.stiffness.shape[0])
    vf = fem.poisson(divf)
    vf -= min(vf)
    return vf


def tria_compute_gradient(tria: TriaMesh, vfunc: np.ndarray) -> np.ndarray:
    r"""Compute gradient of a vertex function f (for each triangle).

    .. math::
        grad(f) &= [ (f_j - f_i) (v_i-v_k)' + (f_k - f_i) (v_j-v_i)' ] / (2 A) \\
                &= [ f_i (v_k-v_j)' + f_j (v_i-v_k)' +  f_k (v_j-v_i)' ] / (2 A)

    for triangle :math:`(v_i,v_j,v_k)` with area :math:`A`, where (.)' is 90 degrees
    rotated edge, which is equal to cross(n,vec).

    Parameters
    ----------
    tria : TriaMesh
        Triangle mesh.
    vfunc : np.ndarray
        Scalar function at vertices, shape (n_vertices,).

    Returns
    -------
    np.ndarray
        3D gradient vector at triangles, shape (n_triangles, 3).

    Notes
    -----
    Numexpr could speed up this functions if necessary.
    Good background to read:
    http://dgd.service.tu-berlin.de/wordpress/vismathws10/2012/10/17/gradient-of-scalar-functions/
    Mancinelli, Livesu, Puppo, Gradient Field Estimation on Triangle Meshes
    http://pers.ge.imati.cnr.it/livesu/papers/MLP18/MLP18.pdf
    Desbrun ...
    """
    import sys

    v0 = tria.v[tria.t[:, 0], :]
    v1 = tria.v[tria.t[:, 1], :]
    v2 = tria.v[tria.t[:, 2], :]
    e2 = v1 - v0
    e0 = v2 - v1
    e1 = v0 - v2
    # get tria normals in n and 1.0/(2*areas) in lni
    n = np.cross(e2, -e1)
    ln = np.sqrt(np.sum(n * n, axis=1))
    ln[ln < sys.float_info.epsilon] = 1  # avoid division by zero
    lni = np.divide(1.0, ln)[:, np.newaxis]
    n *= lni
    # sum three weighted edges
    c0 = vfunc[tria.t[:, 0], np.newaxis] * e0
    c1 = vfunc[tria.t[:, 1], np.newaxis] * e1
    c2 = vfunc[tria.t[:, 2], np.newaxis] * e2
    # divided by 2 * area and rotate edge sum
    tfunc = lni * np.cross(n, (c0 + c1 + c2))
    return tfunc


def tria_compute_divergence(tria: TriaMesh, tfunc: np.ndarray) -> np.ndarray:
    """Compute integrated divergence of a 3D triangle function f (for each vertex).

    Divergence is the flux density leaving or entering a point.
    Note: this is the integrated divergence, you may want to multiply
    with :math:`B^{-1}` to get back the function in some applications

    Parameters
    ----------
    tria : TriaMesh
        Triangle mesh.
    tfunc : np.ndarray
        3D vector field on triangles, shape (n_triangles, 3).

    Returns
    -------
    np.ndarray
        Scalar function of divergence at vertices, shape (n_vertices,).

    Notes
    -----
    Numexpr could speed up this functions if necessary.
    """
    import sys

    v0 = tria.v[tria.t[:, 0], :]
    v1 = tria.v[tria.t[:, 1], :]
    v2 = tria.v[tria.t[:, 2], :]
    e2 = v1 - v0
    e0 = v2 - v1
    e1 = v0 - v2
    # cross length
    n = np.cross(e2, -e1)
    ln = np.sqrt(np.sum(n * n, axis=1))
    ln[ln < sys.float_info.epsilon] = 1  # avoid division by zero
    # cot = scalar products / cross norm
    # number according to opposite edge num
    cot0 = (e2 * (-e1)).sum(1) / ln
    cot1 = (e0 * (-e2)).sum(1) / ln
    cot2 = (e1 * (-e0)).sum(1) / ln
    # dot products of cot with edges
    c0 = cot0[:, np.newaxis] * e0
    c1 = cot1[:, np.newaxis] * e1
    c2 = cot2[:, np.newaxis] * e2
    # compute vfunc divergence
    x0 = ((c2 - c1) * tfunc).sum(1)
    x1 = ((c0 - c2) * tfunc).sum(1)
    x2 = ((c1 - c0) * tfunc).sum(1)
    # use sparse matrix to add multiple entries of each tria at each of its vertices
    i = np.column_stack((tria.t[:, 0], tria.t[:, 1], tria.t[:, 2])).reshape(-1)
    j = np.zeros((3 * len(tria.t), 1), dtype=int).reshape(-1)
    dat = np.column_stack((x0, x1, x2)).reshape(-1)
    # convert back to nparray 1D
    vfunc = np.squeeze(
        np.asarray(0.5 * sparse.csc_matrix((dat, (i, j))).todense(), dtype=tfunc.dtype)
    )
    return vfunc


def tria_compute_divergence2(tria: TriaMesh, tfunc: np.ndarray) -> np.ndarray:
    """Compute integrated divergence of a 3D triangle function f (for each vertex).

    Divergence is the flux density leaving or entering a point.
    It can be measured by summing the dot product of the vector
    field with the normals to the outer edges of the 1-ring triangles
    around a vertex. Summing < tfunc , e_ij cross n >,
    this is the integrated divergence, you may want to multiply
    with :math:`B^{-1}` to get back the function in some applications.

    Parameters
    ----------
    tria : TriaMesh
        Triangle mesh.
    tfunc : np.ndarray
        3D vector field on triangles, shape (n_triangles, 3).

    Returns
    -------
    np.ndarray
        Scalar function of divergence at vertices, shape (n_vertices,).

    Notes
    -----
    Numexpr could speed-up this functions if necessary.
    """
    import sys

    v0 = tria.v[tria.t[:, 0], :]
    v1 = tria.v[tria.t[:, 1], :]
    v2 = tria.v[tria.t[:, 2], :]
    e2 = v1 - v0
    e0 = v2 - v1
    e1 = v0 - v2
    # cross length
    n = np.cross(e2, -e1)
    ln = np.sqrt(np.sum(n * n, axis=1))
    ln[ln < sys.float_info.epsilon] = 1  # avoid division by zero
    lni = np.divide(1.0, ln)[:, np.newaxis]
    n *= lni
    c0 = np.cross(e0, n)
    c1 = np.cross(e1, n)
    c2 = np.cross(e2, n)
    x0 = (c0 * tfunc).sum(1)
    x1 = (c1 * tfunc).sum(1)
    x2 = (c2 * tfunc).sum(1)
    i = np.column_stack((tria.t[:, 0], tria.t[:, 1], tria.t[:, 2])).reshape(-1)
    j = np.zeros((3 * len(tria.t), 1), dtype=int).reshape(-1)
    dat = np.column_stack((x0, x1, x2)).reshape(-1)
    vfunc = np.squeeze(np.asarray(0.5 * sparse.csc_matrix((dat, (i, j))).todense()))
    return vfunc


def tria_compute_rotated_f(tria: TriaMesh, vfunc: np.ndarray) -> np.ndarray:
    """Compute function whose level sets are orthgonal to the ones of vfunc.

    This is done by rotating the gradient around the normal by 90 degrees,
    then solving the Poisson equations with the divergence of rotated grad.

    Parameters
    ----------
    tria : TriaMesh
        Triangle mesh.
    vfunc : np.ndarray
        Scalar function at vertices, shape (n_vertices,).

    Returns
    -------
    np.ndarray
        Rotated scalar function at vertices, shape (n_vertices,).

    Notes
    -----
    Numexpr could speed up this functions if necessary.
    """
    gradf = tria_compute_gradient(tria, vfunc)
    tn = tria.tria_normals()
    # lg = np.sqrt(np.sum(gradf * gradf, axis=1))
    # lgi = np.divide(1.0,lg)[:,np.newaxis]
    # gradf *= lgi
    gradf = np.cross(tn, gradf)
    divf = tria_compute_divergence(tria, gradf)
    fem = Solver(tria)
    # as long as div does not care about weighing with a Bi,
    #   we can pass identity instead of B here:
    # div is the integrated divergence (so it is already B*div)
    fem.mass = sparse.eye(fem.stiffness.shape[0], dtype=vfunc.dtype)
    # since the solution is ill defined (addition of constant)
    # we specify an arbitrary boundary condition at a single vertex to
    # remove that degree of freedom
    dtup = (np.array([0]), np.array([0.0]))
    vf = fem.poisson(divf, dtup)
    return vf


def tria_mean_curvature_flow(
    tria: TriaMesh,
    max_iter: int = 30,
    stop_eps: float = 1e-13,
    step: float = 1.0,
    use_cholmod: bool = False,
) -> TriaMesh:
    """Flow a triangle mesh along the mean curvature normal.

    Iteratively flows a triangle mesh along mean curvature
    normal (non-singular, see Kazhdan 2012).
    This uses the algorithm described in Kazhdan 2012 "Can mean curvature flow be
    made non-singular" which uses the Laplace-Beltrami operator but keeps the
    stiffness matrix (A) fixed and only adjusts the mass matrix (B) during the
    steps. It will normalize surface area of the mesh and translate the barycenter
    to the origin. Closed meshes will map to the unit sphere.

    Parameters
    ----------
    tria : TriaMesh
        Triangle mesh.
    max_iter : int, default=30
        Maximal number of steps.
    stop_eps : float, default=1e-13
        Stopping threshold.
    step : float, default=1.0
        Euler step size.
    use_cholmod : bool, default=False
        Which solver to use. If True, use Cholesky decomposition from
        scikit-sparse cholmod. If False, use spsolve (LU decomposition).

    Returns
    -------
    TriaMesh
        Triangle mesh after flow.

    Raises
    ------
    ImportError
        If use_cholmod is True but scikit-sparse is not installed.

    Notes
    -----
    Numexpr could speed up this functions if necessary.
    """
    if use_cholmod:
        sksparse = import_optional_dependency("sksparse", raise_error=True)
        importlib.import_module(".cholmod", sksparse.__name__)
    else:
        sksparse = None
    # pre-normalize
    trianorm = TriaMesh(tria.v, tria.t)
    trianorm.normalize_()
    # compute fixed A
    lump = True  # for computation here and inside loop
    fem = Solver(trianorm, lump)
    a_mat = fem.stiffness
    for x in range(max_iter):
        # store last position (for delta computation below)
        vlast = trianorm.v
        # get current mass matrix and Mv
        mass = Solver.fem_tria_mass(trianorm, lump)
        mass_v = mass.dot(trianorm.v)
        # solve (M + step*A) * v = Mv and update vertices
        if use_cholmod:
            logger.info("Solver: Cholesky decomposition from scikit-sparse cholmod")
            factor = sksparse.cholmod.cholesky(mass + step * a_mat)
            trianorm.v = factor(mass_v)
        else:
            # Note, it would be better to do sparse Cholesky (CHOLMOD)
            # as it can be 5-6 times faster
            from scipy.sparse.linalg import spsolve

            logger.info("Solver: spsolve (LU decomposition)")
            trianorm.v = spsolve(mass + step * a_mat, mass_v)
        # normalize updated mesh
        trianorm.normalize_()
        # compute difference
        dv = trianorm.v - vlast
        diff = np.trace(np.square(np.matmul(np.transpose(dv), mass.dot(dv))))
        logger.debug("Step %d delta: %g", x + 1, diff)
        if diff < stop_eps:
            logger.info("Converged after %d iterations.", x + 1)
            break
    return trianorm

def _unit_vector(v: np.ndarray, name: str) -> np.ndarray:
    """Normalize vector to unit length.

    Parameters
    ----------
    v : np.ndarray
        Vector to normalize.
    name : str
        Name of the vector for error messages.

    Returns
    -------
    np.ndarray
        Unit vector.

    Raises
    ------
    ValueError
        If vector is degenerate (zero length).
    """
    norm = np.linalg.norm(v)
    if np.isclose(norm, 0.0):
        raise ValueError(f"{name} is degenerate (zero length)")
    return v / norm

def tria_spherical_project(
    tria: TriaMesh, flow_iter: int = 3, debug: bool = False
) -> TriaMesh:
    """Compute first 3 non-constant eigenfunctions and project spectral embedding onto sphere.

    Computes the first three non-constant eigenfunctions
    and then projects the spectral embedding onto a sphere. This works
    when the first functions have a single closed zero level set,
    splitting the mesh into two domains each. Depending on the original
    shape triangles could get inverted. We also flip the functions
    according to the axes that they are aligned with for the special
    case of brain surfaces in FreeSurfer coordinates.

    Parameters
    ----------
    tria : TriaMesh
        Triangle mesh.
    flow_iter : int, default=3
        Mean curvature flow iterations (3 should be enough).
    debug : bool, default=False
        If True, writes debug eigenvalue file.

    Returns
    -------
    TriaMesh
        Triangle mesh projected onto sphere.

    Raises
    ------
    ValueError
        If mesh is not closed.
        If direction 1 is not anterior-posterior.
        If global normal flip is detected.
        If sphere area fraction is below 0.99.
        If flipped area fraction is above 0.0008.
        If spatial volume (orthogonality) is below 0.6.

    Notes
    -----
    Numexpr could speed up this functions if necessary.
    """
    import math

    if not tria.is_closed():
        raise ValueError("Error: Can only project closed meshes!")

    # sub-function to compute flipped area of trias where normal
    # points towards origin, meaningful for the sphere, centered at zero
    def get_flipped_area(triax):
        vx1 = triax.v[triax.t[:, 0], :]
        vx2 = triax.v[triax.t[:, 1], :]
        vx3 = triax.v[triax.t[:, 2], :]
        v2mv1 = vx2 - vx1
        v3mv1 = vx3 - vx1
        cr = np.cross(v2mv1, v3mv1)
        spatvolx = np.sum(vx1 * cr, axis=1)
        areasx = 0.5 * np.sqrt(np.sum(cr * cr, axis=1))
        areax = np.sum(areasx[np.where(spatvolx < 0)])
        return areax

    fem = Solver(tria, lump=False)
    evals, evecs = fem.eigs(k=4)

    if debug:
        data = dict()
        data["Eigenvalues"] = evals
        data["Eigenvectors"] = evecs
        data["Creator"] = "spherically_project.py"
        data["Refine"] = 0
        data["Degree"] = 1
        data["Dimension"] = 2
        data["Elements"] = tria.t.shape[0]
        data["DoF"] = evecs.shape[0]
        data["NumEW"] = 4
        from .io import write_ev

        write_ev(data, "debug.ev")

    # flip efuncs to align to coordinates consistently
    ev1 = evecs[:, 1]
    # ev1maxi = np.argmax(ev1)
    # ev1mini = np.argmin(ev1)
    # cmax = v[ev1maxi,:]
    # cmin = v[ev1mini,:]
    cmax1 = np.mean(tria.v[ev1 > 0.5 * np.max(ev1), :], 0)
    cmin1 = np.mean(tria.v[ev1 < 0.5 * np.min(ev1), :], 0)
    ev2 = evecs[:, 2]
    cmax2 = np.mean(tria.v[ev2 > 0.5 * np.max(ev2), :], 0)
    cmin2 = np.mean(tria.v[ev2 < 0.5 * np.min(ev2), :], 0)
    ev3 = evecs[:, 3]
    cmax3 = np.mean(tria.v[ev3 > 0.5 * np.max(ev3), :], 0)
    cmin3 = np.mean(tria.v[ev3 < 0.5 * np.min(ev3), :], 0)

    # we trust ev 1 goes from front to back
    l11 = abs(cmax1[1] - cmin1[1])
    l21 = abs(cmax2[1] - cmin2[1])
    l31 = abs(cmax3[1] - cmin3[1])
    if l11 < l21 or l11 < l31:
        logger.error("Direction 1 should be anterior - posterior (%g, %g, %g)", l11, l21, l31)
        raise ValueError("Direction 1 should be anterior - posterior")

    # only flip direction if necessary
    logger.info("ev1 min: %s  max %s", cmin1, cmax1)
    # axis 1 = y is aligned with this function (for brains in FS space)
    v1 = _unit_vector(cmax1 - cmin1, "direction 1")
    if cmax1[1] < cmin1[1]:
        ev1 = -1 * ev1
        logger.debug("inverting direction 1 (anterior - posterior)")
    l1 = abs(cmax1[1] - cmin1[1])

    # for ev2 and ev3 there could be also a swap of the two
    l22 = abs(cmax2[2] - cmin2[2])
    l32 = abs(cmax3[2] - cmin3[2])
    # usually ev2 should be superior inferior, if ev3 is better in that direction, swap
    if l22 < l32:
        logger.debug("swapping direction 2 and 3")
        ev2, ev3 = ev3, ev2
        cmax2, cmax3 = cmax3, cmax2
        cmin2, cmin3 = cmin3, cmin2
    l23 = abs(cmax2[0] - cmin2[0])
    l33 = abs(cmax3[0] - cmin3[0])
    if l33 < l23:
        logger.warning("WARNING: direction 3 wants to swap with 2, but cannot")

    logger.info("ev2 min: %s  max %s", cmin2, cmax2)
    # axis 2 = z is aligned with this function (for brains in FS space)
    v2 = _unit_vector(cmax2 - cmin2, "direction 2")
    if cmax2[2] < cmin2[2]:
        ev2 = -1 * ev2
        logger.debug("inverting direction 2 (superior - inferior)")
    l2 = abs(cmax2[2] - cmin2[2])

    logger.info("ev3 min: %s  max %s", cmin3, cmax3)
    # axis 0 = x is aligned with this function (for brains in FS space)
    v3 = _unit_vector(cmax3 - cmin3, "direction 3")
    if cmax3[0] < cmin3[0]:
        ev3 = -1 * ev3
        logger.debug("inverting direction 3 (right - left)")
    l3 = abs(cmax3[0] - cmin3[0])
    # compute spatial volume of the three direction vectors
    spatvol = abs(np.dot(v1, np.cross(v2, v3)))
    logger.debug("spat vol: %g", spatvol)

    mvol = tria.volume()
    logger.info("orig mesh vol %g", mvol)
    bvol = l1 * l2 * l3
    logger.info("box %g, %g, %g volume: %g", l1, l2, l3, bvol)
    logger.info("box coverage: %g", bvol / mvol)

    # we map evN to -1..0..+1 (keep zero level fixed)
    # I have the feeling that this helps a little with the stretching
    # at the poles, but who knows...
    ev1min = np.amin(ev1)
    ev1max = np.amax(ev1)
    ev1[ev1 < 0] /= -ev1min
    ev1[ev1 > 0] /= ev1max

    ev2min = np.amin(ev2)
    ev2max = np.amax(ev2)
    ev2[ev2 < 0] /= -ev2min
    ev2[ev2 > 0] /= ev2max

    ev3min = np.amin(ev3)
    ev3max = np.amax(ev3)
    ev3[ev3 < 0] /= -ev3min
    ev3[ev3 > 0] /= ev3max

    # set evec as new coordinates (spectral embedding)
    vn = np.empty(tria.v.shape)
    vn[:, 0] = ev3
    vn[:, 1] = ev1
    vn[:, 2] = ev2

    # do a few mean curvature flow euler steps to make more convex
    # three should be sufficient
    if flow_iter > 0:
        tflow = tria_mean_curvature_flow(TriaMesh(vn, tria.t), max_iter=flow_iter)
        vn = tflow.v

    # project to sphere and scaled to have the same scale/origin as FS:
    dist = np.sqrt(np.sum(vn * vn, axis=1))
    vn = 100 * (vn / dist[:, np.newaxis])

    trianew = TriaMesh(vn, tria.t)
    svol = trianew.area() / (4.0 * math.pi * 10000)
    logger.info("sphere area fraction: %g", svol)

    flippedarea = get_flipped_area(trianew) / (4.0 * math.pi * 10000)
    if flippedarea > 0.95:
        logger.error("global normal flip detected: %g", flippedarea)
        raise ValueError("global normal flip")

    logger.info("flipped area fraction: %g", flippedarea)

    if svol < 0.99:
        logger.error("sphere area fraction below threshold .99 > %g", svol)
        raise ValueError("sphere area fraction should be above .99")

    if flippedarea > 0.0008:
        logger.error("flipped area fraction too high (>0.0008): %g", flippedarea)
        raise ValueError("flipped area fraction should be below .0008")

    # here we finally check also the spat vol (orthogonality of direction vectors)
    # we could stop earlier, but most failure cases will be covered by the svol and
    # flipped area which can be better interpreted than spatvol
    if spatvol < 0.6:
        logger.error("spat vol (orthogonality) below threshold 0.6 > %g", spatvol)
        raise ValueError("spat vol (orthogonality) should be above .6")

    return trianew


def tet_compute_gradient(tet: "TetMesh", vfunc: np.ndarray) -> np.ndarray:
    r"""Compute gradient of a vertex function f (for each tetrahedron).

    For a tetrahedron :math:`(v_i, v_j, v_k, v_h)` with volume V we have:
    
    .. math::
        grad(f) &= [ (f_j - f_i) (v_i-v_k) \times (v_h-v_k) \\
                &   + (f_k - f_i) (v_i-v_h) \times (v_j-v_h) \\
                &   + (f_h - f_i) (v_k-v_i) \times (v_j-v_i) ] / (6 V).

    Parameters
    ----------
    tet : TetMesh
        Tetrahedral mesh.
    vfunc : np.ndarray
        Scalar function at vertices, shape (n_vertices,).

    Returns
    -------
    np.ndarray
        3D gradient vector at tetrahedra, shape (n_tetrahedra, 3).

    Notes
    -----
    Numexpr could speed up this functions if necessary.
    """
    import sys

    v0 = tet.v[tet.t[:, 0], :]
    v1 = tet.v[tet.t[:, 1], :]
    v2 = tet.v[tet.t[:, 2], :]
    v3 = tet.v[tet.t[:, 3], :]
    e0 = v1 - v0
    # e1 = v2 - v1 # not needed below
    e2 = v0 - v2
    e3 = v3 - v0
    e4 = v3 - v1
    e5 = v3 - v2
    # Compute cross product and  1 / (6 * vol) for each tetrahedron:
    cr = np.cross(e0, e2)
    vol = np.abs(np.sum(e3 * cr, axis=1))
    vol[vol < sys.float_info.epsilon] = 1  # avoid division by zero
    voli = np.divide(1.0, vol)[:, np.newaxis]
    # sum weighted edges
    # c0 = vfunc[t[:,0],np.newaxis] * np.cross(,)
    c1 = (vfunc[tet.t[:, 1], np.newaxis] - vfunc[tet.t[:, 0], np.newaxis]) * np.cross(
        e2, e5
    )
    c2 = (vfunc[tet.t[:, 2], np.newaxis] - vfunc[tet.t[:, 0], np.newaxis]) * np.cross(
        e3, e4
    )
    c3 = (vfunc[tet.t[:, 3], np.newaxis] - vfunc[tet.t[:, 0], np.newaxis]) * np.cross(
        -e2, e0
    )
    # divided by parallelepiped vol
    tfunc = voli * (c1 + c2 + c3)
    return tfunc


def tet_compute_divergence(tet: "TetMesh", tfunc: np.ndarray) -> np.ndarray:
    """Compute integrated divergence of a 3D tetra function f (for each vertex).

    Divergence is the flux density leaving or entering a point.
    It can be measured by summing the dot product of the vector
    field with the normals to the outer faces of the 1-ring tetras
    around a vertex. Summing < tfunc , n_tria_oposite_v >,
    this is the integrated divergence, you may want to multiply
    with :math:`B^{-1}` to get back the function in some applications.

    Parameters
    ----------
    tet : TetMesh
        Tetrahedral mesh.
    tfunc : np.ndarray
        3D vector field on tetrahedra, shape (n_tetrahedra, 3).

    Returns
    -------
    np.ndarray
        Scalar function of divergence at vertices, shape (n_vertices,).
    """
    v0 = tet.v[tet.t[:, 0], :]
    v1 = tet.v[tet.t[:, 1], :]
    v2 = tet.v[tet.t[:, 2], :]
    v3 = tet.v[tet.t[:, 3], :]
    e0 = v1 - v0
    e1 = v2 - v1
    e2 = v2 - v0
    e3 = v3 - v0
    e4 = v3 - v1
    # 2-times-area-length-normals opposite vertex i
    n0 = np.cross(e1, e4)
    n1 = np.cross(e3, e2)
    n2 = np.cross(e0, e3)
    n3 = np.cross(e2, e0)
    # sum contributions to vertices
    x0 = (n0 * tfunc).sum(1)
    x1 = (n1 * tfunc).sum(1)
    x2 = (n2 * tfunc).sum(1)
    x3 = (n3 * tfunc).sum(1)
    i = np.column_stack((tet.t[:, 0], tet.t[:, 1], tet.t[:, 2], tet.t[:, 3])).reshape(
        -1
    )
    j = np.zeros((4 * len(tet.t), 1), dtype=int).reshape(-1)
    dat = np.column_stack((x0, x1, x2, x3)).reshape(-1)
    vfunc = -np.squeeze(
        np.asarray(
            (1.0 / 6.0) * sparse.csc_matrix((dat, (i, j))).todense(),
            dtype=tfunc.dtype,
        )
    )
    return vfunc
