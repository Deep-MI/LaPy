"""Test surfaces with known topology."""

import numpy as np

from ...tria_mesh import TriaMesh


def voxel_surface(mask):
    """Boundary of a binary voxel volume, two triangles per voxel face, normals pointing outward.

    Voxel corners sit at integer coordinates. Voxels that touch only along an edge or at a corner
    give non-manifold edges or pinch vertices.
    """
    m = np.pad(np.asarray(mask, dtype=np.int8), 1)
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    corner_shape = np.array(m.shape) + 1
    quads = []
    for axis in range(3):
        b, c = (axis + 1) % 3, (axis + 2) % 3
        step = np.diff(m, axis=axis)
        for sign in (1, -1):
            base = np.argwhere(step == sign)
            base[:, axis] += 1
            corners = np.repeat(base[:, None, :], 4, axis=1)
            corners[:, :, b] += square[:, 0]
            corners[:, :, c] += square[:, 1]
            if sign == 1:
                corners = corners[:, ::-1]
            quads.append(np.ravel_multi_index(corners.reshape(-1, 3).T, corner_shape).reshape(-1, 4))
    used, inv = np.unique(np.concatenate(quads), return_inverse=True)
    quads = inv.reshape(-1, 4)
    t = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    v = np.column_stack(np.unravel_index(used, corner_shape)).astype(float)
    return TriaMesh(v, t)


def box(nx, ny, nz):
    """Surface of a solid block of nx x ny x nz voxels."""
    return voxel_surface(np.ones((nx, ny, nz), dtype=bool))


def block_with_holes(g):
    """Voxel slab with g separate square through-holes, a surface of genus g."""
    mask = np.ones((2 * g + 1, 3, 2), dtype=bool)
    mask[1 : 2 * g : 2, 1, :] = False
    return voxel_surface(mask)


def block_with_bridge():
    """Solid voxel block with a one-voxel-thick arch on top, a surface of genus 1."""
    mask = np.zeros((10, 6, 9), dtype=bool)
    mask[:, :, :5] = True
    mask[2, 2, 5:8] = True
    mask[7, 2, 5:8] = True
    mask[2:8, 2, 7] = True
    return voxel_surface(mask)


def torus(n=40, m=16, big_r=3.0, small_r=1.0):
    """Quad-grid torus with n segments around the axis and m around the tube."""
    i, j = np.meshgrid(np.arange(n), np.arange(m), indexing="ij")
    u = 2 * np.pi * i / n
    w = 2 * np.pi * j / m
    v = np.column_stack(
        [
            ((big_r + small_r * np.cos(w)) * np.cos(u)).ravel(),
            ((big_r + small_r * np.cos(w)) * np.sin(u)).ravel(),
            (small_r * np.sin(w)).ravel(),
        ]
    )
    a = (i * m + j).ravel()
    b = (((i + 1) % n) * m + j).ravel()
    c = (((i + 1) % n) * m + (j + 1) % m).ravel()
    d = (i * m + (j + 1) % m).ravel()
    return TriaMesh(v, np.concatenate([np.column_stack([a, b, c]), np.column_stack([a, c, d])]))
