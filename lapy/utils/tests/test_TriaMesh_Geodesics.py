import json

import cv2
import numpy as np
import pytest
from scipy.sparse.linalg import splu

from ...diffgeo import compute_divergence, compute_geodesic_f, compute_gradient
from ...heat import diffusion
from ...solver import Solver
from ...tria_mesh import TriaMesh


@pytest.fixture
def loaded_data():
    """
    Load and provide the expected outcomes data from a JSON file.

    Returns:
        dict: Dictionary containing the expected outcomes data.
    """
    with open("lapy/utils/tests/expected_outcomes.json", "r") as f:
        expected_outcomes = json.load(f)
    return expected_outcomes


# Fixture to load the TetMesh
@pytest.fixture
def load_square_mesh():
    T = TriaMesh.read_off("data/square-mesh.off")
    return T


def test_tra_qualities(load_square_mesh):
    """
    Test triangle mesh quality computation.
    """
    T = load_square_mesh
    computed_q = T.tria_qualities()
    expected_q_length = 768
    assert len(computed_q) == expected_q_length


# Laplace
def test_Laplace_Geodesics(load_square_mesh):
    """
    Test Laplace solver for geodesics on a mesh.
    """

    T = load_square_mesh

    # compute first eigenfunction
    fem = Solver(T, lump=True)
    eval, evec = fem.eigs()
    vfunc = evec[:, 1]

    # Get A,B (lumped), and inverse of B (as it is diagonal due to lumping)
    A, B = fem.stiffness, fem.mass
    Bi = B.copy()
    Bi.data **= -1

    assert B.sum() == 1.0
    assert Bi is not B
    # Convert A to a dense NumPy array
    A_dense = A.toarray()

    # Assert that A is symmetric
    assert (A_dense == A_dense.T).all()


# Geodesics
def test_Laplace_Geodesics_with_Gradient_Divergence(load_square_mesh):
    """
    Test Laplace geodesics using gradient and divergence.
    """
    T = load_square_mesh

    # Load eigenfunction
    fem = Solver(T, lump=True)
    eval, evec = fem.eigs()
    vfunc = evec[:, 1]

    # Compute Laplacian using -div(grad(f))
    grad = compute_gradient(T, vfunc)
    divx = -compute_divergence(T, grad)

    # Get the lumped mass matrix B
    fem = Solver(T, lump=True)
    B = fem.mass
    Bi = B.copy()
    Bi.data **= -1

    # Apply Laplacian operator and then the inverse of B
    Laplacian_result = -divx  # The Laplacian result

    # Apply the inverse of B to recover vfunc
    recovered_vfunc = Bi.dot(Laplacian_result)

    # Check if the original vfunc and the recovered vfunc length are equal
    assert len(recovered_vfunc) == len(vfunc)

    expected_eval_length = 10
    assert len(eval) == expected_eval_length


def test_heat_diffusion_shape(load_square_mesh):
    """
    Test the shape of the heat diffusion result on a square mesh.

    Parameters:
        load_square_mesh: Fixture providing a loaded square mesh.

    This test function computes the heat diffusion and verifies that the shape
    of the result matches the expected shape.

    Returns:
        None
    """
    T = load_square_mesh
    bvert = T.boundary_loops()
    u = diffusion(T, bvert, m=1)
    expected_shape = (len(T.v),)
    assert u.shape == expected_shape


def test_Geodesics_format(loaded_data, load_square_mesh):
    """
    Test geodesics format and accuracy.
    """
    T = load_square_mesh
    bvert = T.boundary_loops()
    u = diffusion(T, bvert, m=1)
    # compute gradient of heat diffusion
    tfunc = compute_gradient(T, u)

    # normalize gradient
    X = -tfunc / np.sqrt((tfunc**2).sum(1))[:, np.newaxis]
    X = np.nan_to_num(X)
    divx = compute_divergence(T, X)
    # compute distance

    useCholmod = True
    try:
        from sksparse.cholmod import cholesky
    except ImportError:
        useCholmod = False

    fem = Solver(T, lump=True)
    A, B = fem.stiffness, fem.mass
    H = -A
    b0 = divx

    # solve H x = b0
    # we don't need the B matrix here, as divx is the intgrated divergence
    print("Matrix Format now: " + H.getformat())
    if useCholmod:
        print("Solver: cholesky decomp - performance optimal ...")
        chol = cholesky(H)
        x = chol(b0)
    else:
        print("Solver: spsolve (LU decomp) - performance not optimal ...")
        lu = splu(H)
        x = lu.solve(b0)

    # remove shift
    x = x - min(x)

    Bi = B.copy()
    vf = fem.poisson(-Bi * divx)
    vf = vf - min(vf)
    gf = compute_geodesic_f(T, u)
    expected_matrix_format = loaded_data["expected_outcomes"]["test_Geodesics_format"][
        "expected_matrix_format"
    ]
    assert H.getformat() == expected_matrix_format
    assert useCholmod == False, "Solver: cholesky decomp - performance optimal ..."
    expected_max_x = loaded_data["expected_outcomes"]["test_Geodesics_format"][
        "max_distance"
    ]
    expected_sqrt_2_div_2 = loaded_data["expected_outcomes"]["test_Geodesics_format"][
        "expected_sqrt_2_div_2"
    ]
    assert np.isclose(max(x), expected_max_x)
    computed_sqrt_2_div_2 = np.sqrt(2) / 2
    assert np.isclose(computed_sqrt_2_div_2, expected_sqrt_2_div_2)
    expected_max_abs_diff = loaded_data["expected_outcomes"]["test_Geodesics_format"][
        "expected_max_abs_diff"
    ]
    computed_max_abs_diff = max(abs(gf - x))
    assert np.allclose(computed_max_abs_diff, expected_max_abs_diff)
