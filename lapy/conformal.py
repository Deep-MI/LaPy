"""Computes spherical conformal mappings of triangle meshes.

Functions are adopted from Matlab code at
https://github.com/garyptchoi/spherical-conformal-map
with this
Copyright (c) 2013-2020, Gary Pui-Tung Choi
https://math.mit.edu/~ptchoi
and has been distributed with the Apache 2 License.

Notes
-----
If you use this code in your own work, please cite the following paper:

[1] P. T. Choi, K. C. Lam, and L. M. Lui,
"FLASH: Fast Landmark Aligned Spherical Harmonic Parameterization for Genus-0
Closed Brain Surfaces."
SIAM Journal on Imaging Sciences, vol. 8, no. 1, pp. 67-94, 2015.
"""
import importlib
import logging
from typing import Any

import numpy as np
from scipy import sparse
from scipy.optimize import minimize

from . import Solver, TriaMesh
from .utils._imports import import_optional_dependency

logger = logging.getLogger(__name__)

def _ensure_planar_mesh(tria: TriaMesh, context: str) -> None:
    """Ensure mesh is planar (all z-coordinates near zero).

    Parameters
    ----------
    tria : TriaMesh
        Triangle mesh to check.
    context : str
        Context string for error message.

    Raises
    ------
    ValueError
        If mesh is not planar (z-coordinate variation exceeds 0.001).
    """
    if np.amax(tria.v[:, 2]) - np.amin(tria.v[:, 2]) > 0.001:
        logger.error("%s: Mesh should be on the complex plane.", context)
        raise ValueError("Mesh is not planar")

def _ensure_nonzero(value: float, name: str) -> None:
    """Ensure value is not close to zero.

    Parameters
    ----------
    value : float
        Value to check.
    name : str
        Name of the value for error message.

    Raises
    ------
    ValueError
        If value is close to zero (division by zero risk).
    """
    if np.isclose(value, 0.0):
        raise ValueError(f"{name} is degenerate (division by zero)")

def _ensure_nonzero_array(values: np.ndarray, name: str) -> None:
    """Ensure array contains no values close to zero.

    Parameters
    ----------
    values : np.ndarray
        Array to check.
    name : str
        Name of the array for error message.

    Raises
    ------
    ValueError
        If any value in array is close to zero.
    """
    if np.any(np.isclose(values, 0.0)):
        raise ValueError(f"{name} contains zero entries and cannot be used as a denominator")

def _ensure_genus_zero(tria: TriaMesh) -> None:
    """Ensure the mesh is a genus-0 closed surface.

    Parameters
    ----------
    tria : TriaMesh
        Triangle mesh to check.

    Raises
    ------
    ValueError
        If the Euler characteristic is not 2.
    """
    if tria.euler() != 2:
        logger.error("The mesh is not a genus-0 closed surface.")
        raise ValueError("Invalid input: Mesh must be genus-0.")

def _dirichlet_system(
        A: sparse.spmatrix,
        idx: np.ndarray,
        target: np.ndarray
) -> tuple[sparse.csc_matrix, np.ndarray]:
    """Impose Dirichlet conditions on ``idx`` while keeping ``A`` symmetric.

    Zeros both the rows *and* the columns of the constrained vertices, puts 1
    on their diagonal, and moves the contribution of the removed columns to the
    right hand side. The assembled matrix is symmetric and must stay that way:
    a Cholesky solver reads only the lower triangular part of its input, so a
    matrix whose rows alone were eliminated would be factorised as if the
    columns had been eliminated too, giving a different solution than ``splu``.

    The matrix and the right hand side are built together because they have to
    agree: the right hand side has to be formed from the *unconstrained* matrix,
    before the columns are dropped.

    The unconstrained right hand side is assumed to be zero, which holds for
    every system in this module.

    Parameters
    ----------
    A : sparse.spmatrix
        Unconstrained symmetric matrix of shape (n, n).
    idx : np.ndarray
        Indices of the constrained vertices, shape (n_fixed,). Must be unique,
        since each one prescribes the value in the matching row of ``target``.
    target : np.ndarray
        Prescribed values, shape (n_fixed,) or (n_fixed, n_rhs).

    Returns
    -------
    sparse.csc_matrix
        Constrained matrix of shape (n, n), still symmetric.
    np.ndarray
        Matching right hand side, shape (n,) or (n, n_rhs) following ``target``.

    Raises
    ------
    ValueError
        If ``idx`` contains a repeated index.
    """
    n = A.shape[0]
    idx = np.asarray(idx, dtype=np.intp)
    if np.unique(idx).size != idx.size:
        # A repeated index prescribes two values for one vertex. It would also
        # subtract that vertex's column twice while only one of the two values
        # survives in the right hand side, so the free block would be solved
        # against a load no boundary condition corresponds to.
        raise ValueError(
            "constrained indices must be unique; a repeated index prescribes "
            "two values for the same vertex"
        )
    target = np.asarray(target)

    rhs = -np.asarray(A[:, idx] @ target)
    rhs[idx] = target

    keep = np.ones(n, dtype=A.dtype)
    keep[idx] = 0
    mask = sparse.diags(keep)
    ones = sparse.csc_matrix(
        (np.ones(idx.size, dtype=A.dtype), (idx, idx)), shape=(n, n)
    )
    out = (mask @ A @ mask + ones).tocsc()
    out.eliminate_zeros()
    return out, rhs

def spherical_conformal_map(tria: TriaMesh, use_cholmod: bool = False) -> np.ndarray:
    """Linear method for computing spherical conformal map of a genus-0 closed surface.

    Parameters
    ----------
    tria : TriaMesh
        A triangular mesh object representing a genus-0 closed surface.
    use_cholmod : bool, default=False
        Which solver to use. If True, use Cholesky decomposition from
        scikit-sparse cholmod. If False, use spsolve (LU decomposition).

    Returns
    -------
    np.ndarray
        Vertex coordinates of shape (n_vertices, 3) of the spherical conformal
        parameterization.

    Raises
    ------
    ValueError
        If mesh is not genus-0 (Euler characteristic != 2).
        If edge lengths are degenerate.
        If projection contains NaN values.
    ImportError
        If use_cholmod is True but scikit-sparse is not installed.
    """
    # Ensure the input mesh has genus-0 topology
    _ensure_genus_zero(tria)

    # Find the "big triangle" by selecting the most regularly shaped triangle
    bigtri = np.argmax(tria.tria_qualities())
    # If it turns out that the spherical parameterization result is homogeneous
    # you can try to change bigtri to the id of some other triangles with good quality

    # Solve the Laplace equation on the big triangle
    nv = tria.v.shape[0]
    S = Solver(tria)
    M = S.stiffness.tocsc()

    # Fixed vertices of the big triangle, which ends up at the north pole
    p0, p1, p2 = tria.t[bigtri, :]
    north_fixed = tria.t[bigtri, :]

    # Compute the local coordinates for the big triangle
    # arbitrarily set first two points
    x0, y0, x1, y1 = 0, 0, 1, 0
    a = tria.v[p1, :] - tria.v[p0, :]
    b = tria.v[p2, :] - tria.v[p0, :]
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    _ensure_nonzero(norm_a, "edge a")
    _ensure_nonzero(norm_b, "edge b")
    cross_ab = np.linalg.norm(np.cross(a, b))
    denominator = norm_a * norm_b
    _ensure_nonzero(denominator, "area denominator")
    sin1 = cross_ab / denominator
    ori_h = norm_b * sin1
    ratio = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2) / norm_a
    y2 = ori_h * ratio  # y-coordinate for the third vertex
    x2_square = norm_b ** 2 * ratio ** 2 - y2 ** 2
    # Allow for small negative values due to floating-point rounding.
    eps = 1e-12
    if x2_square < -eps:
        raise ValueError(f"Negative value encountered in x2 calculation: {x2_square}")
    if x2_square < 0:
        x2_square = 0.0
    x2 = np.sqrt(x2_square)
    # should be around (0.5, sqrt(3)/2) if we found an equilateral bigtri

    # Solve Laplace's equation to compute the harmonic map, pinning the three
    # vertices of the big triangle to their planar positions. The two planar
    # coordinates are two real right hand side columns; packing them into one
    # complex column would only double the work of a real solve.
    target = np.array([[x0, y0], [x1, y1], [x2, y2]], dtype=np.float64)
    M, rhs = _dirichlet_system(M, north_fixed, target)

    z = _sparse_symmetric_solve(M, rhs, use_cholmod=use_cholmod)
    z = z[:, 0] + 1j * z[:, 1]
    z = z - np.mean(z, axis=0)

    # A harmonic map that failed on a very bad triangulation shows up in one of
    # two ways: as NaN after the projection, or as a collapsed southernmost
    # triangle, which makes the rescale raise. Both mean the same thing, so
    # both fall back to the Tutte map, which only uses the connectivity.
    try:
        z = _rescale_polar_triangles(z, tria, bigtri)
        S = inverse_stereographic(z)
        failure = "the projection contains NaN values" if np.isnan(np.sum(S)) else None
    except ValueError as err:
        failure = str(err)

    if failure is not None:
        logger.warning(
            "Harmonic map failed (%s); falling back to the spherical Tutte map.",
            failure,
        )
        z = _spherical_tutte_z(tria, bigtri, use_cholmod=use_cholmod)
        S = inverse_stereographic(z)
        if np.isnan(np.sum(S)):
            raise ValueError("Projection contains NaN values!")

    # Fix near the south pole to reduce distortion
    order = np.argsort(S[:, 2])

    # number of points near the south pole to be fixed
    # simply set it to be 1/10 of the total number of vertices (can be changed)
    # In case the spherical parameterization is not good, change 10 to
    # something smaller (e.g. 2)
    fixnum = np.maximum(round(nv / 10), 3)
    south_fixed = order[: np.minimum(nv, fixnum)]

    # South pole stereographic projection, w = z / |z|^2. This has to be built
    # from the rescaled z: the denominator 1 + S[:, 2] computed before the
    # rescale belongs to a different sphere and scales every vertex by the
    # wrong factor.
    absz = np.abs(z)
    _ensure_nonzero_array(absz, "south pole stereographic radius")
    P = np.column_stack(
        (z.real / absz ** 2, z.imag / absz ** 2, np.zeros(nv))
    )

    # Compute Beltrami coefficients for the current parameterization (value per triangle)
    triasouth = TriaMesh(P, tria.t)
    mu = beltrami_coefficient(triasouth, tria.v)

    # compose the map with another quasi-conformal map to cancel the distortion
    mapping = linear_beltrami_solver(
        triasouth, mu, south_fixed, P[south_fixed, :], use_cholmod=use_cholmod
    )

    if np.isnan(np.sum(mapping)):
        # if the result has NaN entries, then most probably the number of
        # boundary constraints is not large enough
        # increase the number of boundary constrains and run again
        logger.warning(
            "South pole composed map contains NaN values; retrying with more fixed vertices."
        )
        fixnum *= 5  # again, this number can be changed
        south_fixed = order[: np.minimum(nv, fixnum)]
        mapping = linear_beltrami_solver(
            triasouth, mu, south_fixed, P[south_fixed, :], use_cholmod=use_cholmod
        )
        if np.isnan(np.sum(mapping)):
            logger.warning("Retry still contains NaNs; falling back to stereographic result.")
            mapping = P  # use the old result

    # inverse south pole stereographic projection
    mapping = _inverse_stereographic_south(mapping)
    return mapping


def _rescale_polar_triangles(
        z: np.ndarray,
        tria: TriaMesh,
        bigtri: int
) -> np.ndarray:
    """Rescale a planar map so both polar triangles end up a similar size.

    The big triangle ends up at the north pole and the triangle closest to the
    origin at the south pole; scaling ``z`` by the geometric mean of their side
    lengths balances the two.

    Parameters
    ----------
    z : np.ndarray
        Planar map as complex numbers, shape (n_vertices,).
    tria : TriaMesh
        The mesh ``z`` was computed on.
    bigtri : int
        Index of the triangle pinned at the north pole.

    Returns
    -------
    np.ndarray
        Rescaled planar map, shape (n_vertices,).
    """
    # Find the index of the southernmost triangle
    absz = np.abs(z)
    index = np.argsort(absz[tria.t[:, 0]] +
                       absz[tria.t[:, 1]] +
                       absz[tria.t[:, 2]])
    inner = index[0]
    if inner == bigtri:
        inner = index[1]

    # Compute side lengths of northernmost and southernmost triangles. The
    # southern one is measured in the south pole chart w = z / |z|^2, where
    # |w_a - w_b| = |z_a - z_b| / (|z_a| |z_b|), so it follows from z without
    # projecting onto the sphere and dividing by 1 + S[:, 2] = 2 |z|^2 /
    # (1 + |z|^2), which is quadratically small near the origin.
    NorthTriSide = (np.abs(z[tria.t[bigtri, 0]] - z[tria.t[bigtri, 1]]) +
                    np.abs(z[tria.t[bigtri, 1]] - z[tria.t[bigtri, 2]]) +
                    np.abs(z[tria.t[bigtri, 2]] - z[tria.t[bigtri, 0]])) / 3.0

    i0, i1, i2 = tria.t[inner, :]
    _ensure_nonzero_array(absz[[i0, i1, i2]], "southernmost triangle radius")
    SouthTriSide = (np.abs(z[i0] - z[i1]) / (absz[i0] * absz[i1]) +
                    np.abs(z[i1] - z[i2]) / (absz[i1] * absz[i2]) +
                    np.abs(z[i2] - z[i0]) / (absz[i2] * absz[i0])) / 3.0

    # rescale to get the best distribution
    return z * np.sqrt(NorthTriSide * SouthTriSide) / NorthTriSide


def _spherical_tutte_z(
        tria: TriaMesh,
        bigtri: int = 0,
        use_cholmod: bool = False
) -> np.ndarray:
    """Compute the planar Tutte map, rescaled like the harmonic one.

    See :func:`spherical_tutte_map` for the parameters; this returns the map on
    the complex plane rather than on the sphere, which is what the south pole
    step of :func:`spherical_conformal_map` needs.
    """
    nv = tria.v.shape[0]
    t = tria.t

    # Tutte (uniform weight) Laplacian. Each directed half edge contributes 1/2
    # in both directions, so every edge of the closed mesh carries weight 1.
    i = np.concatenate((t[:, 0], t[:, 1], t[:, 2]))
    j = np.concatenate((t[:, 1], t[:, 2], t[:, 0]))
    half = np.full(i.shape[0], 0.5)
    w = sparse.csc_matrix(
        (np.concatenate((half, half)),
         (np.concatenate((i, j)), np.concatenate((j, i)))),
        shape=(nv, nv),
    )
    # Assemble as D - W rather than W - D so the matrix is positive semidefinite
    # like lapy's stiffness and a Cholesky backend can factorise it. Flipping
    # the sign of the whole system leaves the solution unchanged.
    m = (sparse.diags(np.asarray(w.sum(axis=1)).ravel()) - w).tocsc()

    # Pin the big triangle to the three cube roots of unity
    fixed = t[bigtri, :]
    angles = 2.0 * np.pi * np.arange(3) / 3.0
    target = np.column_stack((np.cos(angles), np.sin(angles)))
    m, rhs = _dirichlet_system(m, fixed, target)

    z = _sparse_symmetric_solve(m, rhs, use_cholmod=use_cholmod)
    z = z[:, 0] + 1j * z[:, 1]
    z = z - np.mean(z)

    return _rescale_polar_triangles(z, tria, bigtri)


def spherical_tutte_map(
        tria: TriaMesh,
        bigtri: int = 0,
        use_cholmod: bool = False
) -> np.ndarray:
    """Compute the spherical Tutte map of a genus-0 closed surface.

    Same construction as :func:`spherical_conformal_map`, with the cotangent
    Laplacian replaced by the Tutte Laplacian. It ignores the vertex positions
    and uses only the connectivity, so it still produces a valid sphere where
    the harmonic map breaks down on a badly shaped triangulation.
    :func:`spherical_conformal_map` falls back to it for exactly that reason.

    Parameters
    ----------
    tria : TriaMesh
        A triangular mesh object representing a genus-0 closed surface.
    bigtri : int, default=0
        Index of the triangle to pin at the north pole.
    use_cholmod : bool, default=False
        Which solver to use. If True, use Cholesky decomposition from
        scikit-sparse cholmod. If False, use spsolve (LU decomposition).

    Returns
    -------
    np.ndarray
        Vertex coordinates of shape (n_vertices, 3) on the unit sphere.

    Raises
    ------
    ValueError
        If mesh is not genus-0 (Euler characteristic != 2).
    ImportError
        If use_cholmod is True but scikit-sparse is not installed.
    """
    _ensure_genus_zero(tria)
    return inverse_stereographic(
        _spherical_tutte_z(tria, bigtri, use_cholmod=use_cholmod)
    )


def mobius_area_correction_spherical(
    tria: TriaMesh, mapping: np.ndarray
) -> tuple[np.ndarray, Any]:
    r"""Find an improved Mobius transformation to reduce distortion.

    This helps reducing the area distortion of
    a spherical conformal parameterization using the method in
    Choi et al, SIAM Journal on Imaging Sciences, 2020.

    Parameters
    ----------
    tria : TriaMesh
        Genus-0 closed triangle mesh.
    mapping : np.ndarray
        Vertex coordinates of shape (n_vertices, 3) representing a spherical conformal
        parameterization.

    Returns
    -------
    map_mobius : np.ndarray
        Vertex coordinates of shape (n_vertices, 3) updated to minimize area distortion.
    result : object
        Optimization result object containing optimal parameters (x) for the Mobius
        transformation, where

        .. math::
            f(z) = \frac{az+b}{cz+d} = \frac{(x[0]+x[1]*1j)*z+(x[2]+x[3]*1j)}{(x[4]+x[5]*1j)*z+(x[6]+x[7]*1j)}.
    """  # noqa: E501
    # Compute normalized triangle areas
    area_t = tria.tria_areas()
    area_t = area_t / area_t.sum()

    # Project the sphere onto the complex plane using stereographic projection
    z = stereographic(mapping)

    def area_map(xx: np.ndarray) -> np.ndarray:
        """
        Compute the area distribution from the Möbius-transformed mapping.

        Parameters
        ----------
            xx (np.ndarray): A length-8 array of Möbius transformation parameters.

        Returns
        -------
            np.ndarray: Normalized triangle areas after applying the transformation.
        """
        v = inverse_stereographic(((xx[0] + xx[1] * 1j) * z + (xx[2] + xx[3] * 1j)) /
                                  ((xx[4] + xx[5] * 1j) * z + (xx[6] + xx[7] * 1j)))
        area_v = TriaMesh(v, tria.t).tria_areas()
        return area_v / area_v.sum()

    def d_area(xx: np.ndarray) -> float:
        """
        Objective function: Mean absolute log area distortion after the Möbius transformation.

        Parameters
        ----------
            xx (np.ndarray): A length-8 array of Möbius transformation parameters.

        Returns
        -------
            float: Mean of the absolute log area distortion where finite.
        """
        a = np.abs(np.log(area_map(xx) / area_t))
        return (a[np.isfinite(a)]).mean()

    # Initial guess for the Möbius transformation parameters
    x0 = np.array([1, 0, 0, 0, 0, 0, 1, 0])

    # Bounds for optimization parameters to keep transformation bounded
    bnds = ((-100, 100), (-100, 100), (-100, 100), (-100, 100),
            (-100, 100), (-100, 100), (-100, 100), (-100, 100))

    # Perform optimization to find the optimal Möbius transformation
    # Optimization (may further supply gradients for better result, not yet implemented)
    # options = optimoptions('fmincon','Display','iter');
    # x = fmincon(d_area,x0,[],[],[],[],lb,ub,[],options);
    options = {"disp": True}
    result = minimize(d_area, x0, bounds=bnds, options=options)
    x = result.x

    # Apply the optimized Möbius transformation
    fz = ((x[0] + x[1]* 1j) * z + (x[2] + x[3]* 1j)) / ((x[4] + x[5]* 1j)* z + (x[6] + x[7]* 1j))
    map_mobius = inverse_stereographic(fz)

    return map_mobius, x


def beltrami_coefficient(tria: TriaMesh, mapping: np.ndarray) -> np.ndarray:
    """Compute the Beltrami coefficient of a given mapping.

    The Beltrami coefficient is a complex-valued function that characterizes the
    distortion of a mapping in terms of conformality.

    Parameters
    ----------
    tria : TriaMesh
        Genus-0 closed triangle mesh.
        Should be planar mapping on complex plane.
    mapping : np.ndarray
        Vertex coordinates of shape (n_vertices, 3) representing the spherical conformal
        parameterization.

    Returns
    -------
    np.ndarray
        Complex Beltrami coefficient per triangle, shape (n_triangles,).

    Raises
    ------
    ValueError
        If mesh is not planar.
    """
    # Ensure the triangulation is planar
    _ensure_planar_mesh(tria, "Beltrami coefficient")

    # Extract 2D vertex positions and compute triangle edges
    v0 = tria.v[tria.t[:, 0], :][:, :-1]
    v1 = tria.v[tria.t[:, 1], :][:, :-1]
    v2 = tria.v[tria.t[:, 2], :][:, :-1]
    e0 = v2 - v1
    e1 = v0 - v2
    e2 = v1 - v0

    # Compute double areas of triangles
    areas2 = np.cross(e0, e1)  # Returns the z-component of the cross product (scalar)

    # Create Dx and Dy sparse matrices (summing area-normalized edge coordinates)
    nf = tria.t.shape[0]  # Number of triangles
    tids = np.arange(nf)
    i = np.column_stack((tids, tids, tids)).reshape(-1)
    j = tria.t.reshape(-1)
    datx = (
        np.column_stack((e0[:, 1], e1[:, 1], e2[:, 1])) / areas2[:, np.newaxis]
    ).reshape(-1)
    daty = -(
        np.column_stack((e0[:, 0], e1[:, 0], e2[:, 0])) / areas2[:, np.newaxis]
    ).reshape(-1)
    nv = tria.v.shape[0]  # Number of vertices
    Dx = sparse.csr_matrix((datx, (i, j)), shape=(nf, nv))
    Dy = sparse.csr_matrix((daty, (i, j)), shape=(nf, nv))

    # Compute partial derivatives of the mapping
    dXdu = Dx.dot(mapping[:, 0])
    dXdv = Dy.dot(mapping[:, 0])
    dYdu = Dx.dot(mapping[:, 1])
    dYdv = Dy.dot(mapping[:, 1])
    dZdu = Dx.dot(mapping[:, 2])
    dZdv = Dy.dot(mapping[:, 2])

    # Compute coefficients of the first fundamental form
    E = dXdu ** 2 + dYdu ** 2 + dZdu ** 2  # Length of the first derivative wrt u
    G = dXdv ** 2 + dYdv ** 2 + dZdv ** 2  # Length of the first derivative wrt v
    F = dXdu * dXdv + dYdu * dYdv + dZdu * dZdv  # Mixed derivative term

    # Compute Beltrami coefficient
    mu = (E - G + 2j * F) / (E + G + 2.0 * np.sqrt(E * G - F ** 2))

    return mu


def linear_beltrami_solver(
        tria: TriaMesh,
        mu: np.ndarray,
        landmark: np.ndarray,
        target: np.ndarray,
        use_cholmod: bool = False
) -> np.ndarray:
    """Solve the Linear Beltrami equation for a given mesh and target.

    Parameters
    ----------
    tria : TriaMesh
        Genus-0 closed triangle mesh.
        Should be planar mapping on complex plane.
    mu : np.ndarray
        Beltrami coefficients describing distortion at each triangle, shape (n_triangles,).
    landmark : np.ndarray
        Indices of fixed landmark vertices, shape (n_landmarks,).
    target : np.ndarray
        2D target positions for the landmark vertices, shape (n_landmarks, 2).
    use_cholmod : bool, default=False
        Which solver to use. If True, use Cholesky decomposition from
        scikit-sparse cholmod. If False, use spsolve (LU decomposition).

    Returns
    -------
    np.ndarray
        Mapping of all vertices to 2D coordinates, shape (n_vertices, 2),
        aligned to the given landmarks.

    Raises
    ------
    ValueError
        If mesh is not planar.
        If triangle areas are degenerate.
        If Beltrami denominator is close to zero.
    ImportError
        If use_cholmod is True but scikit-sparse is not installed.
    """
    # Ensure the triangulation is planar
    _ensure_planar_mesh(tria, "Linear Beltrami solver")

    # Compute coefficients for the Beltrami equation
    denominator = 1.0 - np.abs(mu) ** 2
    _ensure_nonzero_array(np.abs(denominator), "Beltrami denominator (1 - |mu|^2)")
    af = (1.0 - 2 * np.real(mu) + np.abs(mu) ** 2) / denominator
    bf = -2.0 * np.imag(mu) / denominator
    gf = (1.0 + 2 * np.real(mu) + np.abs(mu) ** 2) / denominator

    # Extract vertices and indices for triangles (drop 3rd dimension)
    t0 = tria.t[:, 0]
    t1 = tria.t[:, 1]
    t2 = tria.t[:, 2]
    v0 = tria.v[t0, :][:, :-1]
    v1 = tria.v[t1, :][:, :-1]
    v2 = tria.v[t2, :][:, :-1]

    # Calculate vertex components to determine areas
    uxv0 = v1[:, 1] - v2[:, 1]
    uyv0 = v2[:, 0] - v1[:, 0]
    uxv1 = v2[:, 1] - v0[:, 1]
    uyv1 = v0[:, 0] - v2[:, 0]
    uxv2 = v0[:, 1] - v1[:, 1]
    uyv2 = v1[:, 0] - v0[:, 0]

    c0 = np.sqrt(uxv0 ** 2 + uyv0 ** 2)
    c1 = np.sqrt(uxv1 ** 2 + uyv1 ** 2)
    c2 = np.sqrt(uxv2 ** 2 + uyv2 ** 2)
    s = 0.5 * (c0 + c1 + c2)
    area2 = 2 * np.sqrt(s * (s - c0) * (s - c1) * (s - c2))
    _ensure_nonzero_array(area2, "triangle area")

    v00 = (af * uxv0 * uxv0 + 2 * bf * uxv0 * uyv0 + gf * uyv0 * uyv0) / area2
    v11 = (af * uxv1 * uxv1 + 2 * bf * uxv1 * uyv1 + gf * uyv1 * uyv1) / area2
    v22 = (af * uxv2 * uxv2 + 2 * bf * uxv2 * uyv2 + gf * uyv2 * uyv2) / area2
    v01 = (
        af * uxv1 * uxv0 + bf * uxv1 * uyv0 + bf * uxv0 * uyv1 + gf * uyv1 * uyv0
    ) / area2
    v12 = (
        af * uxv2 * uxv1 + bf * uxv2 * uyv1 + bf * uxv1 * uyv2 + gf * uyv2 * uyv1
    ) / area2
    v20 = (
        af * uxv0 * uxv2 + bf * uxv0 * uyv2 + bf * uxv2 * uyv0 + gf * uyv0 * uyv2
    ) / area2

    # Create a symmetric sparse matrix A. af, bf and gf are built from the real
    # and imaginary parts of mu, so every entry of A is real.
    i = np.column_stack((t0, t1, t2, t0, t1, t1, t2, t2, t0)).reshape(-1)
    j = np.column_stack((t0, t1, t2, t1, t0, t2, t1, t0, t2)).reshape(-1)
    dat = np.column_stack((v00, v11, v22, v01, v01, v12, v12, v20, v20)).reshape(-1)
    nv = tria.v.shape[0]
    A = sparse.csc_matrix((dat, (i, j)), shape=(nv, nv))

    # Impose the landmark positions, keeping A symmetric. The two target
    # coordinates are two real right hand side columns.
    A, b = _dirichlet_system(A, landmark, target[:, :2])

    # Solve the sparse linear system
    mapping = _sparse_symmetric_solve(A, b, use_cholmod=use_cholmod)
    return mapping


def _sparse_symmetric_solve(
        A: sparse.spmatrix,
        b: np.ndarray,
        use_cholmod: bool = False
) -> np.ndarray:
    """Solve the real sparse symmetric linear system of equations Ax = b.

    Depending on the availability of the `scikit-sparse` package, it uses either:
    - Cholesky decomposition (via scikit-sparse) for performance-optimal solving.
    - LU decomposition (via SciPy) if scikit-sparse is not available.

    ``A`` has to be genuinely symmetric for the two branches to agree: the
    Cholesky solver reads only the lower triangular part of ``A``, whereas
    ``splu`` reads all of it. Use :func:`_dirichlet_system` to impose boundary
    conditions without destroying the symmetry.

    Parameters
    ----------
    A : sparse.spmatrix
        Real, sparse, symmetric coefficient matrix of shape (n, n).
    b : np.ndarray
        Real right hand side of shape (n,) or (n, n_rhs).
    use_cholmod : bool, default=False
        Which solver to use. If True, use Cholesky decomposition from
        scikit-sparse cholmod. If False, use spsolve (LU decomposition).

    Returns
    -------
    np.ndarray
        Solution ``x`` with the same shape as ``b``.

    Raises
    ------
    ValueError
        If ``A`` or ``b`` is complex.
    ImportError
        If use_cholmod is True but scikit-sparse is not installed.
    RuntimeError
        Propagated from ``splu`` when ``A`` is singular.
    sksparse.cholmod.CholmodNotPositiveDefiniteError
        Propagated from CHOLMOD when ``A`` is not positive definite and
        use_cholmod is True.
    """
    if np.iscomplexobj(A) or np.iscomplexobj(b):
        raise ValueError(
            "_sparse_symmetric_solve is real-valued; pass the two planar "
            "coordinates as two right hand side columns instead of as one "
            "complex column"
        )
    b = np.ascontiguousarray(b, dtype=np.float64)
    if use_cholmod:
        sksparse = import_optional_dependency("sksparse", raise_error=True)
        importlib.import_module(".cholmod", sksparse.__name__)
        logger.info("Solver: Cholesky decomposition (scikit-sparse cholmod)")
        chol = sksparse.cholmod.cholesky(sparse.csc_matrix(A))
        x = chol(b)
    else:
        from scipy.sparse.linalg import splu
        logger.info("Solver: LU decomposition (spsolve)")
        lu = splu(sparse.csc_matrix(A))
        x = lu.solve(b)
    return np.asarray(x).reshape(b.shape)


def stereographic(u: np.ndarray) -> np.ndarray:
    """Map points on a sphere to the complex plane using the stereographic projection.

    Parameters
    ----------
    u : np.ndarray
        Points on the sphere as (x, y, z) coordinates, shape (n_points, 3).

    Returns
    -------
    np.ndarray
        Mapped points as complex numbers on the complex plane, shape (n_points,).

    Raises
    ------
    ValueError
        If stereographic denominator (1 - z) is close to zero.
    """
    # Map sphere to complex plane
    # u has three columns (x,y,z)
    # return z as array of complex numbers
    x = u[:, 0]
    y = u[:, 1]
    z = u[:, 2]
    denom = 1 - z
    _ensure_nonzero_array(np.abs(denom), "stereographic denominator (1 - z)")
    v = np.empty(u.shape[:-1], dtype=complex)
    v.real = (x / denom).flatten()
    v.imag = (y / denom).flatten()
    return v


def inverse_stereographic(u: np.ndarray) -> np.ndarray:
    """Compute mapping from the complex plane to the sphere using inverse stereographic projection.

    Parameters
    ----------
    u : np.ndarray
        Input points in the complex plane. Can be:
        - Array of shape (n_points, 2), representing real and imaginary parts.
        - Array of complex numbers of shape (n_points,).

    Returns
    -------
    np.ndarray
        Mapped points on the sphere as (x, y, z) coordinates, shape (n_points, 3).
    """
    if np.iscomplexobj(u):
        x = u.real
        y = u.imag
    else:
        x = u[:, 0]
        y = u[:, 1]
    z = 1 + x**2 + y**2
    v = np.column_stack((2*x / z, 2*y / z, (-1 + x**2 + y**2) / z))
    return v


def _inverse_stereographic_south(u: np.ndarray) -> np.ndarray:
    """Map the complex plane back to the sphere through the *south* pole.

    :func:`inverse_stereographic` inverts the projection from the north pole,
    the one :func:`stereographic` performs. The chart used for the south pole
    step of :func:`spherical_conformal_map` is ``w = z / |z|^2`` instead, and
    its inverse differs by the sign of the third coordinate: ``|w| = 1 / |z|``,
    so ``(|w|^2 - 1) / (|w|^2 + 1) = -(|z|^2 - 1) / (|z|^2 + 1)``. Inverting
    with the northern formula would return the sphere mirrored in ``z``, an
    orientation-reversing parameterisation.

    Parameters
    ----------
    u : np.ndarray
        Input points in the complex plane. Can be:
        - Array of shape (n_points, 2), representing real and imaginary parts.
        - Array of complex numbers of shape (n_points,).

    Returns
    -------
    np.ndarray
        Mapped points on the sphere as (x, y, z) coordinates, shape (n_points, 3).
    """
    v = inverse_stereographic(u)
    v[:, 2] = -v[:, 2]
    return v
