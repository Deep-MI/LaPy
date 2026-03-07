"""Tests for lapy.heat — backward compatibility and multi-case diffusion."""

import numpy as np
import pytest

from ...heat import diffusion
from ...tria_mesh import TriaMesh


@pytest.fixture
def tria_mesh():
    return TriaMesh.read_off("data/square-mesh.off")


# ---------------------------------------------------------------------------
# Backward compatibility — single vids set must still return 1-D
# ---------------------------------------------------------------------------


def test_diffusion_single_returns_1d(tria_mesh):
    """int, list[int], or 1-D array vids must return a 1-D (n_vertices,) array."""
    T = tria_mesh
    bvert = np.concatenate(T.boundary_loops())

    assert diffusion(T, bvert, m=1).shape == (len(T.v),)
    assert diffusion(T, bvert.tolist(), m=1).shape == (len(T.v),)
    assert diffusion(T, 0, m=1).shape == (len(T.v),)


# ---------------------------------------------------------------------------
# Multi-case — list of arrays/lists returns 2-D, each column matches single call
# ---------------------------------------------------------------------------


def test_diffusion_multi_matches_single(tria_mesh):
    """Multi-case diffusion columns must match independent single calls."""
    T = tria_mesh
    seeds = [np.concatenate(T.boundary_loops()), np.array([0]), np.array([1, 2])]

    u_batch = diffusion(T, seeds, m=1)

    assert u_batch.shape == (len(T.v), 3)
    for k, s in enumerate(seeds):
        np.testing.assert_allclose(
            u_batch[:, k], diffusion(T, s, m=1), rtol=1e-6, atol=1e-9
        )


def test_diffusion_boundary_loops_multi(tria_mesh):
    """boundary_loops() output (list[list[int]]) triggers multi-case path."""
    T = tria_mesh
    loops = T.boundary_loops()

    u_batch = diffusion(T, loops, m=1)

    assert u_batch.ndim == 2
    assert u_batch.shape == (len(T.v), len(loops))


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_diffusion_out_of_range_raises(tria_mesh):
    """Out-of-range vertex indices must raise ValueError."""
    T = tria_mesh
    with pytest.raises(ValueError, match="out-of-range"):
        diffusion(T, np.array([len(T.v)]), m=1)

    with pytest.raises(ValueError, match="out-of-range"):
        diffusion(T, [np.array([0]), np.array([len(T.v)])], m=1)
