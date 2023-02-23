# Adopted from Matlab code at
# https://github.com/garyptchoi/spherical-conformal-map
# with this
# Copyright (c) 2013-2020, Gary Pui-Tung Choi
# https://math.mit.edu/~ptchoi
# and has been distributed with the Apache 2 License

# If you use this code in your own work, please cite the following paper:
# [1] P. T. Choi, K. C. Lam, and L. M. Lui,
# "FLASH: Fast Landmark Aligned Spherical Harmonic Parameterization for Genus-0
#  Closed Brain Surfaces."
# SIAM Journal on Imaging Sciences, vol. 8, no. 1, pp. 67-94, 2015.


import numpy as np
from scipy import sparse
from scipy.optimize import minimize

from .Solver import Solver
from .TriaMesh import TriaMesh
from .utils._imports import import_optional_dependency


def spherical_conformal_map(tria):
    """
    A linear method for computing spherical conformal map of a genus-0 closed surface

    Input:   TriaMesh (vertices and faces)
    Output:
    mapped_vertices: nv x 3 vertex coordinates of the spherical conformal parameterization

    If you use this code in your own work, please cite the following paper:
    [1] P. T. Choi, K. C. Lam, and L. M. Lui,
       "FLASH: Fast Landmark Aligned Spherical Harmonic Parameterization for Genus-0 Closed Brain Surfaces."
       SIAM Journal on Imaging Sciences, vol. 8, no. 1, pp. 67-94, 2015.

    Adopted from Matlab code at
    https://github.com/garyptchoi/spherical-conformal-map
    with this
    Copyright (c) 2013-2020, Gary Pui-Tung Choi
    https://math.mit.edu/~ptchoi
    and has been distributed with the Apache 2 License
    """
    # Check whether the input mesh is spherical topology (genus-0)
    if tria.euler() != 2:
        print("ERROR: The mesh is not a genus-0 closed surface ..")
        raise ValueError("not genus-0")

    # Find the most regular triangle as the "big triangle"
    tquals = tria.tria_qualities()
    bigtri = np.argmax(tquals)
    # print(bigtri, tquals[bigtri])
    # If it turns out that the spherical parameterization result is homogeneous
    # you can try to change bigtri to the id of some other triangles with good quality

    # North pole step: Compute spherical map by solving laplace equation on a big triangle
    nv = tria.v.shape[0]
    S = Solver(tria)
    M = S.stiffness.astype(complex)

    p0 = tria.t[bigtri, 0]
    p1 = tria.t[bigtri, 1]
    p2 = tria.t[bigtri, 2]
    fixed = tria.t[bigtri, :]

    # set all rows and cols with fixed vidxs to zero
    # and set diag entries to 1
    mrow, mcol, mval = sparse.find(M[fixed, :])
    M = (
        M
        - sparse.csc_matrix((mval, (fixed[mrow], mcol)), shape=(nv, nv))
        + sparse.csc_matrix(((1, 1, 1), (fixed, fixed)), shape=(nv, nv))
    )

    # find embedding of the bigtria (boundary condition later)
    # arbitrarily set first two points
    x0, y0, x1, y1 = 0, 0, 1, 0
    a = tria.v[p1, :] - tria.v[p0, :]
    b = tria.v[p2, :] - tria.v[p0, :]
    sin1 = np.linalg.norm(np.cross(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    ori_h = np.linalg.norm(b) * sin1
    ratio = np.sqrt(((x0 - x1) ** 2 + (y0 - y1) ** 2)) / np.linalg.norm(a)
    y2 = ori_h * ratio  # compute the coordinates of the third vertex
    x2 = np.sqrt(np.linalg.norm(b) ** 2 * ratio**2 - y2**2)
    # should be around (0.5, sqrt(3)/2) if we found an equilateral bigtri

    # Solve the Laplace equation to obtain a harmonic map
    c = np.zeros((nv, 1))
    c[p0], c[p1], c[p2] = x0, x1, x2
    d = np.zeros((nv, 1))
    d[p0], d[p1], d[p2] = y0, y1, y2
    rhs = np.empty(c.shape[:-1], dtype=complex)
    rhs.real = c.flatten()
    rhs.imag = d.flatten()

    z = sparse_symmetric_solve(M, rhs)
    z = np.squeeze(np.array(z))
    z = z - np.mean(z, axis=0)

    # inverse stereographic projection (not scaled well)
    S = inverse_stereographic(z)

    # Find optimal big triangle size
    w = np.empty(S.shape[:-1], dtype=complex)
    w.real = (S[:, 0] / (1 + S[:, 2])).flatten()
    w.imag = (S[:, 1] / (1 + S[:, 2])).flatten()

    # find the index of the southernmost triangle
    index = np.argsort(
        np.abs(z[tria.t[:, 0]]) + np.abs(z[tria.t[:, 1]]) + np.abs(z[tria.t[:, 2]])
    )
    inner = index[0]
    if inner == bigtri:
        inner = index[1]

    # Compute the size of the northern most and the southern most triangles
    NorthTriSide = (
        np.abs(z[tria.t[bigtri, 0]] - z[tria.t[bigtri, 1]])
        + np.abs(z[tria.t[bigtri, 1]] - z[tria.t[bigtri, 2]])
        + np.abs(z[tria.t[bigtri, 2]] - z[tria.t[bigtri, 0]])
    ) / 3.0

    SouthTriSide = (
        np.abs(w[tria.t[inner, 0]] - w[tria.t[inner, 1]])
        + np.abs(w[tria.t[inner, 1]] - w[tria.t[inner, 2]])
        + np.abs(w[tria.t[inner, 2]] - w[tria.t[inner, 0]])
    ) / 3.0

    # rescale to get the best distribution
    z = z * np.sqrt(NorthTriSide * SouthTriSide) / NorthTriSide

    # inverse stereographic projection (now distributed well)
    S = inverse_stereographic(z)

    if np.isnan(np.sum(S)):
        raise ValueError("Error: projection contains nan value(s)!")
        # could revert to spherical tutte map here

    # South pole step
    idx = np.argsort(S[:, 2])

    # number of points near the south pole to be fixed
    # simply set it to be 1/10 of the total number of vertices (can be changed)
    # In case the spherical parameterization is not good, change 10 to
    # something smaller (e.g. 2)
    fixnum = np.maximum(round(nv / 10), 3)
    fixed = idx[0 : np.minimum(nv, fixnum)]

    # south pole stereographic projection
    P = np.column_stack(
        (S[:, 0] / (1 + S[:, 2]), S[:, 1] / (1 + S[:, 2]), np.zeros(nv))
    )

    # compute the Beltrami coefficient (value per triangle)
    triasouth = TriaMesh(P, tria.t)
    mu = beltrami_coefficient(triasouth, tria.v)

    # compose the map with another quasi-conformal map to cancel the distortion
    mapping = linear_beltrami_solver(triasouth, mu, fixed, P[fixed, :])

    if np.isnan(np.sum(mapping)):
        # if the result has NaN entries, then most probably the number of
        # boundary constraints is not large enough
        # increase the number of boundary constrains and run again
        print("South pole compsed map has nan value(s)!")
        fixnum = fixnum * 5  # again, this number can be changed
        fixed = idx[0 : np.minimum(nv, fixnum)]
        mapping = linear_beltrami_solver(triasouth, mu, fixed, P[fixed, :])
        if np.isnan(np.sum(mapping)):
            mapping = P  # use the old result

    # inverse south pole stereographic projection
    mapping = inverse_stereographic(mapping)
    return mapping


def mobius_area_correction_spherical(tria, mapping):
    """
    Find an optimal Mobius transformation for reducing the area distortion of a spherical conformal parameterization
    using the method in [1].

    Input:
    tria : TriaMesh (vertices, triangle) of genus-0 closed triangle mesh
    mapping: nv x 3 vertex coordinates of the spherical conformal parameterization

    Output:
    map_mobius: nv x 3 vertex coordinates of the updated spherical conformal parameterization
    x: the optimal parameters for the Mobius transformation, where
       f(z) = \frac{az+b}{cz+d}
            = ((x(1)+x(2)*1j)*z+(x(3)+x(4)*1j))/((x(5)+x(6)*1j)*z+(x(7)+x(8)*1j))

    If you use this code in your own work, please cite the following paper:
    [1] G. P. T. Choi, Y. Leung-Liu, X. Gu, and L. M. Lui,
        "Parallelizable global conformal parameterization of simply-connected surfaces via partial welding."
        SIAM Journal on Imaging Sciences, 2020.

    Adopted by Martin Reuter from Matlab code at
    https://github.com/garyptchoi/spherical-conformal-map
    with this
    Copyright (c) 2019-2020, Gary Pui-Tung Choi
    https://scholar.harvard.edu/choi
    and has been distributed with the Apache 2 License
    """

    # Compute the tria areas with normalization
    area_t = tria.tria_areas()
    area_t = area_t / area_t.sum()
    # Project the sphere onto the plane
    z = stereographic(mapping)

    def area_map(xx):
        v = inverse_stereographic(
            ((xx[0] + xx[1] * 1j) * z + (xx[2] + xx[3] * 1j))
            / ((xx[4] + xx[5] * 1j) * z + (xx[6] + xx[7] * 1j))
        )
        area_v = TriaMesh(v, tria.t).tria_areas()
        return area_v / area_v.sum()

    # objective function: mean(abs(log(area_map/area_t)))
    def d_area(xx):
        a = np.abs(np.log(area_map(xx) / area_t))
        return (a[np.isfinite(a)]).mean()

    # Optimization setup
    x0 = np.array([1, 0, 0, 0, 0, 0, 1, 0])  # initial guess
    # lower and upper bounds
    bnds = (
        (-100, 100),
        (-100, 100),
        (-100, 100),
        (-100, 100),
        (-100, 100),
        (-100, 100),
        (-100, 100),
        (-100, 100),
    )
    # Optimization (may further supply gradients for better result, not yet implemented)
    # options = optimoptions('fmincon','Display','iter');
    # x = fmincon(d_area,x0,[],[],[],[],lb,ub,[],options);
    options = {"disp": True}
    result = minimize(d_area, x0, bounds=bnds, options=options)
    x = result.x
    # obtain the conformal parameterization with area distortion corrected
    fz = ((x[0] + x[1] * 1j) * z + (x[2] + x[3] * 1j)) / (
        (x[4] + x[5] * 1j) * z + (x[6] + x[7] * 1j)
    )
    map_mobius = inverse_stereographic(fz)
    return map_mobius, result


def beltrami_coefficient(tria, mapping):
    """
    Compute the Beltrami coefficient of a mapping.
    If you use this code in your own work, please cite the following paper:
    [1] P. T. Choi, K. C. Lam, and L. M. Lui,
    "FLASH: Fast Landmark Aligned Spherical Harmonic Parameterization for Genus-0 Closed Brain Surfaces."
    SIAM Journal on Imaging Sciences, vol. 8, no. 1, pp. 67-94, 2015.

    Adopted by Martin Reuter from Matlab code at
    https://github.com/garyptchoi/spherical-conformal-map
    with this
    Copyright (c) 2013-2020, Gary Pui-Tung Choi
    https://math.mit.edu/~ptchoi
    and has been distributed with the Apache 2 License
    """
    # here we should be in the plane
    if np.amax(tria.v[:, 2]) - np.amin(tria.v[:, 2]) > 0.001:
        print("ERROR: mesh should be on complex plane ..")
        raise ValueError("not planar")

    # get 2d vetrices, edges and area
    v0 = (tria.v[tria.t[:, 0], :])[:, :-1]
    v1 = (tria.v[tria.t[:, 1], :])[:, :-1]
    v2 = (tria.v[tria.t[:, 2], :])[:, :-1]
    e0 = v2 - v1
    e1 = v0 - v2
    e2 = v1 - v0
    # double areas
    areas2 = np.cross(e0, e1)  # returns z-component is length

    # create tria,vertex matrices (summing area normalized edge coords)
    nf = tria.t.shape[0]
    tids = np.arange(nf)
    i = np.column_stack((tids, tids, tids)).reshape(-1)
    j = tria.t.reshape(-1)
    datx = (
        np.column_stack((e0[:, 1], e1[:, 1], e2[:, 1])) / areas2[:, np.newaxis]
    ).reshape(-1)
    daty = -(
        np.column_stack((e0[:, 0], e1[:, 0], e2[:, 0])) / areas2[:, np.newaxis]
    ).reshape(-1)
    nv = tria.v.shape[0]
    Dx = sparse.csr_matrix((datx, (i, j)), shape=(nf, nv))
    Dy = sparse.csr_matrix((daty, (i, j)), shape=(nf, nv))

    dXdu = Dx.dot(mapping[:, 0])
    dXdv = Dy.dot(mapping[:, 0])
    dYdu = Dx.dot(mapping[:, 1])
    dYdv = Dy.dot(mapping[:, 1])
    dZdu = Dx.dot(mapping[:, 2])
    dZdv = Dy.dot(mapping[:, 2])

    E = dXdu**2 + dYdu**2 + dZdu**2
    G = dXdv**2 + dYdv**2 + dZdv**2
    F = dXdu * dXdv + dYdu * dYdv + dZdu * dZdv
    mu = (E - G + 2j * F) / (E + G + 2.0 * np.sqrt(E * G - F**2))

    return mu


def linear_beltrami_solver(tria, mu, landmark, target):
    """
    Linear Beltrami solver
    If you use this code in your own work, please cite the following paper:
    [1] P. T. Choi, K. C. Lam, and L. M. Lui,
    "FLASH: Fast Landmark Aligned Spherical Harmonic Parameterization for Genus-0 Closed Brain Surfaces."
    SIAM Journal on Imaging Sciences, vol. 8, no. 1, pp. 67-94, 2015.

    Adopted by Martin Reuter from Matlab code at
    https://github.com/garyptchoi/spherical-conformal-map
    with this
    Copyright (c) 2013-2020, Gary Pui-Tung Choi
    https://math.mit.edu/~ptchoi
    and has been distributed with the Apache 2 License
    """

    # here we should be in the plane
    if np.amax(tria.v[:, 2]) - np.amin(tria.v[:, 2]) > 0.001:
        print("ERROR: mesh should be on complex plane ..")
        raise ValueError("not planar")

    af = (1.0 - 2 * np.real(mu) + np.abs(mu) ** 2) / (1.0 - np.abs(mu) ** 2)
    bf = -2.0 * np.imag(mu) / (1.0 - np.abs(mu) ** 2)
    gf = (1.0 + 2 * np.real(mu) + np.abs(mu) ** 2) / (1.0 - np.abs(mu) ** 2)

    # get 2D vertices (drop 3rd dim)
    t0 = tria.t[:, 0]
    t1 = tria.t[:, 1]
    t2 = tria.t[:, 2]
    v0 = (tria.v[t0, :])[:, :-1]
    v1 = (tria.v[t1, :])[:, :-1]
    v2 = (tria.v[t2, :])[:, :-1]

    uxv0 = v1[:, 1] - v2[:, 1]
    uyv0 = v2[:, 0] - v1[:, 0]
    uxv1 = v2[:, 1] - v0[:, 1]
    uyv1 = v0[:, 0] - v2[:, 0]
    uxv2 = v0[:, 1] - v1[:, 1]
    uyv2 = v1[:, 0] - v0[:, 0]

    c0 = np.sqrt(uxv0**2 + uyv0**2)
    c1 = np.sqrt(uxv1**2 + uyv1**2)
    c2 = np.sqrt(uxv2**2 + uyv2**2)
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

    # create symmetric A
    i = np.column_stack((t0, t1, t2, t0, t1, t1, t2, t2, t0)).reshape(-1)
    j = np.column_stack((t0, t1, t2, t1, t0, t2, t1, t0, t2)).reshape(-1)
    dat = np.column_stack((v00, v11, v22, v01, v01, v12, v12, v20, v20)).reshape(-1)
    nv = tria.v.shape[0]
    A = sparse.csc_matrix((dat, (i, j)), shape=(nv, nv), dtype=complex)

    # convert target to complex and set b vector
    targetc = target[:, 0] + 1j * target[:, 1]
    b = -A[:, landmark] * targetc
    b[landmark] = targetc

    # set all rows and columns in landmark to zero and put diag 1
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

    x = sparse_symmetric_solve(A, b)

    mapping = np.squeeze(np.array(x))
    mapping = np.column_stack((np.real(mapping), np.imag(mapping)))
    return mapping


def sparse_symmetric_solve(A, b, use_cholmod=True):
    sksparse = import_optional_dependency("sksparse", raise_error=use_cholmod)
    if sksparse is not None:
        print("Solver: Cholesky decomposition from scikit-sparse cholmod ...")
        chol = sksparse.cholmod.cholesky(A)
        x = chol(b)
    else:
        from scipy.sparse.linalg import splu

        print("Solver: spsolve (LU decomposition) ...")
        lu = splu(A)
        x = lu.solve(b)
    return x


def stereographic(u):
    # Map sphere to complex plane
    # u has three columns (x,y,z)
    # return z as array of complex numbers
    x = u[:, 0]
    y = u[:, 1]
    z = u[:, 2]
    v = np.empty(u.shape[:-1], dtype=complex)
    v.real = (x / (1 - z)).flatten()
    v.imag = (y / (1 - z)).flatten()
    return v


def inverse_stereographic(u):
    # Computes mapping from complex plane to sphere
    # u can be complex array, or two columns (real,img)
    # returns v as (nv x 3) coordinates on sphere
    if np.iscomplexobj(u):
        x = u.real
        y = u.imag
    else:
        x = u[:, 0]
        y = u[:, 1]
    z = 1 + x**2 + y**2
    v = np.column_stack((2 * x / z, 2 * y / z, (-1 + x**2 + y**2) / z))
    return v
