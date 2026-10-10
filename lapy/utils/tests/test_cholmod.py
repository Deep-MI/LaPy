"""The CHOLMOD backend (use_cholmod=True) must agree with the LU backend everywhere it is used."""

import numpy as np
import pytest

from ...diffgeo import tria_mean_curvature_flow
from ...heat import diffusion
from ...solver import Solver
from ...tet_mesh import TetMesh
from ...tria_mesh import TriaMesh

pytest.importorskip("cholespy")


@pytest.fixture
def cube_tria():
    """Closed triangle surface of a cube."""
    return TriaMesh.read_vtk("data/cubeTria.vtk")


@pytest.fixture
def cube_tet():
    """Tetrahedral mesh of a cube."""
    return TetMesh.read_vtk("data/cubeTetra.vtk")


@pytest.fixture
def square():
    """Open planar triangulation of the unit square."""
    return TriaMesh.read_off("data/square-mesh.off")


@pytest.mark.parametrize("mesh", ["cube_tria", "cube_tet"])
def test_eigs_match_lu(mesh, request):
    """Shift-invert eigenvalues agree on a closed surface and on a tetrahedral mesh."""
    geom = request.getfixturevalue(mesh)
    lu = Solver(geom, use_cholmod=False).eigs(k=8, rng=0)[0]
    chol = Solver(geom, use_cholmod=True).eigs(k=8, rng=0)[0]
    np.testing.assert_allclose(chol, lu, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("integrate", [True, False])
def test_poisson_matches_lu(square, integrate):
    """Poisson solutions with Dirichlet and Neumann conditions agree, for one and several right hand sides."""
    rng = np.random.default_rng(0)
    nv = len(square.v)
    bnd = np.concatenate(square.boundary_loops())
    dtup = (bnd[::2], np.sin(square.v[bnd[::2], 0]))
    ntup = (bnd[1::2], np.cos(square.v[bnd[1::2], 1]))
    for h in (rng.normal(size=nv), rng.normal(size=(nv, 3))):
        lu = Solver(square, use_cholmod=False).poisson(h, dtup, ntup, integrate=integrate)
        chol = Solver(square, use_cholmod=True).poisson(h, dtup, ntup, integrate=integrate)
        assert chol.shape == lu.shape == h.shape
        np.testing.assert_allclose(chol, lu, rtol=1e-8, atol=1e-10)


def test_poisson_in_single_precision(square):
    """A float32 solver is factorized in single precision and stays close to the double precision result."""
    bnd = np.concatenate(square.boundary_loops())
    dtup = (bnd, np.zeros(len(bnd)))
    h = np.ones(len(square.v))
    ref = Solver(square, use_cholmod=False).poisson(h, dtup)
    chol = Solver(square, use_cholmod=True, dtype=np.float32).poisson(h, dtup)
    assert chol.dtype == np.float32
    np.testing.assert_allclose(chol, ref, rtol=1e-4, atol=1e-6)


def test_diffusion_matches_lu(cube_tria):
    """Heat diffusion from several seed sets agrees."""
    seeds = [[0, 1], [5], [10, 20, 30]]
    lu = diffusion(cube_tria, seeds, use_cholmod=False)
    chol = diffusion(cube_tria, seeds, use_cholmod=True)
    np.testing.assert_allclose(chol, lu, rtol=1e-8, atol=1e-10)


def test_mean_curvature_flow_matches_lu(cube_tria):
    """Mean curvature flow agrees step by step."""
    lu = tria_mean_curvature_flow(cube_tria, max_iter=5, use_cholmod=False)
    chol = tria_mean_curvature_flow(cube_tria, max_iter=5, use_cholmod=True)
    np.testing.assert_allclose(chol.v, lu.v, rtol=1e-8, atol=1e-10)


def test_not_positive_definite_is_reported(cube_tria):
    """Without boundary conditions the Laplace operator of a closed surface is singular."""
    with pytest.raises(ValueError, match="not positive definite"):
        Solver(cube_tria, use_cholmod=True).poisson(np.ones(len(cube_tria.v)))
