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
from typing import Any, Union

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

from . import Solver, TriaMesh
from .utils._imports import import_optional_dependency


def spherical_conformal_map(tria: TriaMesh, use_cholmod: bool = False) -> np.ndarray:
    """Linear method for computing spherical conformal map of a genus-0 closed surface.

    Parameters
    ----------
    tria : TriaMesh
        A triangular mesh object representing a genus-0 closed surface.
    use_cholmod : bool, default=False
        Which solver to use:
            * True : Use Cholesky decomposition from scikit-sparse cholmod.
            * False: Use spsolve (LU decomposition).

    Returns
    -------
    mapping: np.ndarray
        Vertex coordinates as NumPy array of shape (n, 3) of the spherical conformal
        parameterization.
    """
    # Ensure the input mesh has genus-0 topology
    if tria.euler() != 2:
        print("ERROR: The mesh is not a genus-0 closed surface.")
        raise ValueError("Invalid input: Mesh must be genus-0.")

    # Find the "big triangle" by selecting the most regularly shaped triangle
    tquals = tria.tria_qualities()
    bigtri = np.argmax(tquals)
    # If it turns out that the spherical parameterization result is homogeneous
    # you can try to change bigtri to the id of some other triangles with good quality

    # Solve the Laplace equation on the big triangle
    nv = tria.v.shape[0]
    S = Solver(tria)
    M = S.stiffness.astype(complex)

    # Fixed vertices of the big triangle
    p0, p1, p2 = tria.t[bigtri, 0], tria.t[bigtri, 1], tria.t[bigtri, 2]
    fixed = tria.t[bigtri, :]

    # Make rows/columns of fixed indices zero, and set the diagonal to 1
    mrow, mcol, mval = sparse.find(M[fixed, :])
    M = M - sparse.csc_matrix((mval, (fixed[mrow], mcol)), shape=(nv, nv)) \
        + sparse.csc_matrix(((1, 1, 1), (fixed, fixed)), shape=(nv, nv))

    # Compute the local coordinates for the big triangle
    # arbitrarily set first two points
    x0, y0, x1, y1 = 0, 0, 1, 0
    a = tria.v[p1, :] - tria.v[p0, :]
    b = tria.v[p2, :] - tria.v[p0, :]
    sin1 = np.linalg.norm(np.cross(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    ori_h = np.linalg.norm(b) * sin1
    ratio = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2) / np.linalg.norm(a)
    y2 = ori_h * ratio  # y-coordinate for the third vertex
    x2 = np.sqrt(np.linalg.norm(b) ** 2 * ratio ** 2 - y2 ** 2)
    # should be around (0.5, sqrt(3)/2) if we found an equilateral bigtri

    # Solve Laplace's equation to compute the harmonic map
    c = np.zeros((nv, 1))
    c[p0], c[p1], c[p2] = x0, x1, x2
    d = np.zeros((nv, 1))
    d[p0], d[p1], d[p2] = y0, y1, y2
    rhs = np.empty(c.shape[:-1], dtype=complex)
    rhs.real = c.flatten()
    rhs.imag = d.flatten()

    z = _sparse_symmetric_solve(M, rhs, use_cholmod=use_cholmod)
    z = np.squeeze(np.array(z))
    z = z - np.mean(z, axis=0)

    # Apply inverse stereographic projection
    S = inverse_stereographic(z)

    # Rescale the mapping for better area distribution
    w = np.empty(S.shape[:-1], dtype=complex)
    w.real = (S[:, 0] / (1 + S[:, 2])).flatten()
    w.imag = (S[:, 1] / (1 + S[:, 2])).flatten()

    # Find the index of the southernmost triangle
    index = np.argsort(np.abs(z[tria.t[:, 0]]) +
                       np.abs(z[tria.t[:, 1]]) +
                       np.abs(z[tria.t[:, 2]]))
    inner = index[0]
    if inner == bigtri:
        inner = index[1]

    # Compute side lengths of northernmost and southernmost triangles
    NorthTriSide = (np.abs(z[tria.t[bigtri, 0]] - z[tria.t[bigtri, 1]]) +
                    np.abs(z[tria.t[bigtri, 1]] - z[tria.t[bigtri, 2]]) +
                    np.abs(z[tria.t[bigtri, 2]] - z[tria.t[bigtri, 0]])) / 3.0

    SouthTriSide = (np.abs(w[tria.t[inner, 0]] - w[tria.t[inner, 1]]) +
                    np.abs(w[tria.t[inner, 1]] - w[tria.t[inner, 2]]) +
                    np.abs(w[tria.t[inner, 2]] - w[tria.t[inner, 0]])) / 3.0

    # rescale to get the best distribution
    z = z * np.sqrt(NorthTriSide * SouthTriSide) / NorthTriSide

    # Final inverse stereographic projection
    S = inverse_stereographic(z)
    if np.isnan(np.sum(S)):
        raise ValueError("Error: Projection contains NaN values!")
        # could revert to spherical tutte map here

    # Fix near the south pole to reduce distortion
    idx = np.argsort(S[:, 2])

    # number of points near the south pole to be fixed
    # simply set it to be 1/10 of the total number of vertices (can be changed)
    # In case the spherical parameterization is not good, change 10 to
    # something smaller (e.g. 2)
    fixnum = np.maximum(round(nv / 10), 3)
    fixed = idx[: np.minimum(nv, fixnum)]

    # South pole stereographic projection
    P = np.column_stack((S[:, 0] / (1 + S[:, 2]), S[:, 1] / (1 + S[:, 2]), np.zeros(nv)))

    # Compute Beltrami coefficients for the current parameterization (value per triangle)
    triasouth = TriaMesh(P, tria.t)
    mu = beltrami_coefficient(triasouth, tria.v)

    # compose the map with another quasi-conformal map to cancel the distortion
    mapping = linear_beltrami_solver(
        triasouth, mu, fixed, P[fixed, :], use_cholmod=use_cholmod
    )

    if np.isnan(np.sum(mapping)):
        # if the result has NaN entries, then most probably the number of
        # boundary constraints is not large enough
        # increase the number of boundary constrains and run again
        print("South pole composed map contains NaN values!")
        fixnum *= 5  # again, this number can be changed
        fixed = idx[: np.minimum(nv, fixnum)]
        mapping = linear_beltrami_solver(
            triasouth, mu, fixed, P[fixed, :], use_cholmod=use_cholmod
        )
        if np.isnan(np.sum(mapping)):
            mapping = P  # use the old result

    # inverse south pole stereographic projection
    mapping = inverse_stereographic(mapping)
    return mapping


def mobius_area_correction_spherical(tria: TriaMesh, mapping: np.ndarray) -> tuple[np.ndarray, Any]:
    r"""
    Find an improved Mobius transformation to reduce distortion.

    This helps reducing the area distortion of
    a spherical conformal parameterization using the method in
    Choi et al, SIAM Journal on Imaging Sciences, 2020.

    Parameters
    ----------
    tria : TriaMesh
        Genus-0 closed triangle mesh.
    mapping : np.ndarray
        A NumPy array of shape (n, 3) representing vertex coordinates of a spherical conformal parameterization.

    Returns
    -------
    map_mobius: np.ndarray
        A NumPy array of shape (n, 3) with vertex coordinates updated to minimize area distortion.
    result: np.ndarray
        Optimal parameters (x) for the Mobius transformation, where

        .. math::
            f(z) = \frac{az+b}{cz+d} = \frac{(x(1)+x(2)*1j)*z+(x(3)+x(4)*1j)}{(x(5)+x(6)*1j)*z+(x(7)+x(8)*1j)}.
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
    """
    Compute the Beltrami coefficient of a given mapping.

    The Beltrami coefficient is a complex-valued function that characterizes the
    distortion of a mapping in terms of conformality.

    Parameters
    ----------
    tria : TriaMesh
        Genus-0 closed triangle mesh.
        Should be planar mapping on complex plane.
    mapping : np.ndarray
        A numpy array of shape (n, 3) representing the coordinates of the spherical conformal
        parameterization.

    Returns
    -------
    mu : np.ndarray
        Complex Beltrami coefficient per triangle.
    """
    # Ensure the triangulation is planar
    if np.amax(tria.v[:, 2]) - np.amin(tria.v[:, 2]) > 0.001:
        print("ERROR: Mesh should be on the complex plane ...")
        raise ValueError('Mesh is not planar')

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
        use_cholmod=False
) -> np.ndarray:
    """
    Solve the Linear Beltrami equation for a given mesh and target.

    Parameters
    ----------
    tria : TriaMesh
        Genus-0 closed triangle mesh.
        Should be planar mapping on complex plane.
    mu : np.ndarray
        A numpy array containing Beltrami coefficients, describing distortion at each vertex.
    landmark : np.ndarray
        A numpy array of indices specifying the fixed landmark vertices.
    target : np.ndarray
        A numpy array of shape (len(landmark), 2), specifying the 2D target positions
            for the landmark vertices.
    use_cholmod : bool, default=False
        Which solver to use:
            * True : Attempt to use Cholesky decomposition from scikit-sparse cholmod.
            * False: Use spsolve (LU decomposition).

    Returns
    -------
    mapping : np.ndarray
        An array of shape (n, 2) representing the mapping of all vertices in the
            triangulation to 2D coordinates, aligned to the given landmarks.
    """
    # Ensure the triangulation is planar
    if np.amax(tria.v[:, 2]) - np.amin(tria.v[:, 2]) > 0.001:
        print("ERROR: Mesh should be on the complex plane ...")
        raise ValueError('Mesh is not planar')

    # Compute coefficients for the Beltrami equation
    af = (1.0 - 2 * np.real(mu) + np.abs(mu) ** 2) / (1.0 - np.abs(mu) ** 2)
    bf = -2.0 * np.imag(mu) / (1.0 - np.abs(mu) ** 2)
    gf = (1.0 + 2 * np.real(mu) + np.abs(mu) ** 2) / (1.0 - np.abs(mu) ** 2)

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

    # Create a symmetric sparse matrix A
    i = np.column_stack((t0, t1, t2, t0, t1, t1, t2, t2, t0)).reshape(-1)
    j = np.column_stack((t0, t1, t2, t1, t0, t2, t1, t0, t2)).reshape(-1)
    dat = np.column_stack((v00, v11, v22, v01, v01, v12, v12, v20, v20)).reshape(-1)
    nv = tria.v.shape[0]
    A = sparse.csc_matrix((dat, (i, j)), shape=(nv, nv), dtype=complex)

    # Convert target to complex and set up the b vector
    targetc = target[:, 0] + 1j * target[:, 1]
    b = -A[:, landmark] * targetc
    b[landmark] = targetc

    # Modify A matrix to incorporate alignment with landmarks
    mrow, mcol, mval = sparse.find(A[landmark, :])
    Azero = sparse.csc_matrix((mval, (landmark[mrow], mcol)), shape=(nv, nv))
    A = A - Azero
    mrow, mcol, mval = sparse.find(A[:, landmark])
    Azero = sparse.csc_matrix((mval, (mrow, landmark[mcol])), shape=(nv, nv))
    Aones = sparse.csr_matrix(
        (np.ones(landmark.shape[0]), (landmark, landmark)), shape=(nv, nv)
    )
    A = A - Azero + Aones
    A.eliminate_zeros()

    # Solve the sparse linear system
    x = _sparse_symmetric_solve(A, b, use_cholmod=use_cholmod)

    # Extract the mapping as real and imaginary components
    mapping = np.squeeze(np.array(x))
    mapping = np.column_stack((np.real(mapping), np.imag(mapping)))
    return mapping


def _sparse_symmetric_solve(
        A: csr_matrix,
        b: Union[np.ndarray, csr_matrix],
        use_cholmod: bool = False
) -> np.ndarray:
    """
    Solve a sparse symmetric linear system of equations Ax = b.

    Depending on the availability of the `scikit-sparse` package, it uses either:
    - Cholesky decomposition (via scikit-sparse) for performance-optimal solving.
    - LU decomposition (via SciPy) if scikit-sparse is not available.

    Parameters
    ----------
    A : csc_matrix of shape (n, n)
        The sparse, symmetric coefficient matrix (in CSR format).
    b : (Union[np.ndarray, csr_matrix])
        The right-hand-side vector or matrix.
    use_cholmod : bool, default=False
        Which solver to use:
            * True : Attempt to use Cholesky decomposition from scikit-sparse cholmod.
            * False: Use spsolve (LU decomposition).

    Returns
    -------
    x: np.ndarray
        The solution vector `x` for the system Ax = b.
    """
    if use_cholmod:
        sksparse = import_optional_dependency("sksparse", raise_error=True)
        importlib.import_module(".cholmod", sksparse.__name__)
    else:
        sksparse = None
    if use_cholmod:
        print("Solver: Cholesky decomposition (scikit-sparse cholmod) ...")
        chol = sksparse.cholmod.cholesky(A)
        x = chol(b)
    else:
        from scipy.sparse.linalg import splu

        print("Solver: LU decomposition (spsolve) ...")
        lu = splu(A)
        x = lu.solve(b)
    return x


def stereographic(u: np.ndarray) -> np.ndarray:
    """
    Map points on a sphere to the complex plane using the stereographic projection.

    Parameters
    ----------
    u : np.ndarray
        A numpy array of shape (n, 3), where each row represents a point on the sphere
        as (x, y, z) coordinates.

    Returns
    -------
    np.ndarray:
        A numpy array of shape (n,) containing the mapped points as complex numbers
        on the complex plane.
    """
    # Map sphere to complex plane
    # u has three columns (x,y,z)
    # return z as array of complex numbers
    x = u[:, 0]
    y = u[:, 1]
    z = u[:, 2]
    v = np.empty(u.shape[:-1], dtype=complex)
    v.real = (x / (1-z)).flatten()
    v.imag = (y / (1-z)).flatten()
    return v


def inverse_stereographic(u: np.ndarray) -> np.ndarray:
    """
    Compute mapping from the complex plane to the sphere using the inverse stereographic projection.

    Parameters
    ----------
    u : Union[np.ndarray, list[complex], list[tuple[float, float]]]
        Input points in the complex plane. Can be:
            - A numpy array of shape (n, 2), representing real and imaginary parts.
            - A list of complex numbers.
            - A list of tuples representing (real, imaginary) coordinates.

    Returns
    -------
    np.ndarray:
        A numpy array of shape (n, 3) containing the mapped points on the sphere as
        (x, y, z) coordinates.
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
