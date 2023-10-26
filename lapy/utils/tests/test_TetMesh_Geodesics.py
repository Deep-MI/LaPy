import json

import numpy as np
import pytest
from scipy.sparse.linalg import splu

from ...diffgeo import compute_divergence, compute_geodesic_f, compute_gradient
from ...heat import diffusion
from ...solver import Solver
from ...tet_mesh import TetMesh


# Fixture to load the TetMesh
@pytest.fixture
def load_tet_mesh():
    T = TetMesh.read_vtk("data/cubeTetra.vtk")
    return T


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


# Test if the mesh is oriented
def test_is_oriented(load_tet_mesh):
    T = load_tet_mesh
    assert not T.is_oriented(), "Mesh is already oriented"


# Test orienting the mesh
def test_orient_mesh(load_tet_mesh):
    T = load_tet_mesh
    T.orient_()
    assert T.is_oriented(), "Mesh is not oriented"


# Test solving the Laplace eigenvalue problem
def test_solve_eigenvalue_problem(load_tet_mesh):
    T = load_tet_mesh
    fem = Solver(T, lump=True)

    num_eigenvalues = 10
    evals, evecs = fem.eigs(num_eigenvalues)

    assert len(evals) == num_eigenvalues
    assert evecs.shape == (len(T.v), num_eigenvalues)


def test_evals_evec_dimension(load_tet_mesh, loaded_data):
    T = load_tet_mesh

    expected_evals_len = loaded_data["expected_outcomes"]["test_TetMesh_Geodesics"][
        "expected_evals_len"
    ]

    fem = Solver(T, lump=True)
    evals, evecs = fem.eigs(expected_evals_len)
    assert len(evals) == expected_evals_len
    assert np.shape(evecs) == (9261, 10)


# Geodesics

T = TetMesh.read_vtk("data/cubeTetra.vtk")
tria = T.boundary_tria()
bvert = np.unique(tria.t)
u = diffusion(T, bvert, m=1)


def test_gradients_normalization_and_divergence(load_tet_mesh, loaded_data):
    # Compute gradients
    T = load_tet_mesh
    tfunc = compute_gradient(T, u)

    # Define the expected shape of tfunc (gradient)
    expected_tfunc_shape = (48000, 3)

    # Assert that the shape of tfunc matches the expected shape
    assert tfunc.shape == expected_tfunc_shape

    # Flip and normalize
    X = -tfunc / np.sqrt((tfunc**2).sum(1))[:, np.newaxis]

    # Define the expected shape of X (normalized gradient)
    expected_X_shape = (48000, 3)

    # Assert that the shape of X matches the expected shape
    assert X.shape == expected_X_shape

    # Load the expected maximum and minimum values for each column of X
    expected_max_col_values = loaded_data["expected_outcomes"][
        "test_TetMesh_Geodesics"
    ]["expected_max_col_values"]
    expected_min_col_values = loaded_data["expected_outcomes"][
        "test_TetMesh_Geodesics"
    ]["expected_min_col_values"]

    # Assert maximum and minimum values of each column of X match the expected values
    for col in range(X.shape[1]):
        assert np.allclose(np.max(X[:, col]), expected_max_col_values[col], atol=1e-6)
        assert np.allclose(np.min(X[:, col]), expected_min_col_values[col], atol=1e-6)

    # Compute divergence
    divx = compute_divergence(T, X)

    # Define the expected shape of divx (divergence)
    expected_divx_shape = (9261,)

    # Assert that the shape of divx matches the expected shape
    assert divx.shape == expected_divx_shape


# get gradients
tfunc = compute_gradient(T, u)
# flip and normalize
X = -tfunc / np.sqrt((tfunc**2).sum(1))[:, np.newaxis]
X = np.nan_to_num(X)
# compute divergence
divx = compute_divergence(T, X)

# compute distance
useCholmod = True
try:
    from sksparse.cholmod import cholesky
except ImportError:
    useCholmod = False


fem = Solver(T, lump=True)
A, B = fem.stiffness, fem.mass  # computed above when creating Solver

H = A
b0 = -divx

# solve H x = b0
# print("Matrix Format now: "+H.getformat())
if useCholmod:
    print("Solver: cholesky decomp - performance optimal ...")
    chol = cholesky(H)
    x = chol(b0)
else:
    print("Solver: spsolve (LU decomp) - performance not optimal ...")
    lu = splu(H)
    x = lu.solve(b0)

x = x - np.min(x)


# get heat diffusion
gu = compute_geodesic_f(T, u)

v1func = T.v[:, 0] * T.v[:, 0] + T.v[:, 1] * T.v[:, 1] + T.v[:, 2] * T.v[:, 2]
grad = compute_gradient(T, v1func)
glength = np.sqrt(np.sum(grad * grad, axis=1))
# fcols=glength
A, B = fem.stiffness, fem.mass
Bi = B.copy()
Bi.data **= -1
divx2 = Bi * divx


def test_tetMesh_Geodesics_format(loaded_data):
    """
    Test if matrix format, solver settings, max distance,
    and computed values match the expected outcomes.

    Parameters:
    - loaded_data (dict): Dictionary containing loaded test data.

    Raises:
    - AssertionError: If any test condition is not met.
    """
    expected_matrix_format = loaded_data["expected_outcomes"]["test_TetMesh_Geodesics"][
        "expected_matrix_format"
    ]
    assert H.getformat() == expected_matrix_format
    assert np.shape(x) == (9261,)
    assert not useCholmod, "Solver: cholesky decomp - performance optimal ..."
    expected_max_x = loaded_data["expected_outcomes"]["test_TetMesh_Geodesics"][
        "max_distance"
    ]
    expected_sqrt_3 = loaded_data["expected_outcomes"]["test_TetMesh_Geodesics"][
        "expected_sqrt"
    ]
    assert np.isclose(max(x), expected_max_x)
    computed_sqrt_3 = 0.5 * np.sqrt(3.0)
    assert np.isclose(computed_sqrt_3, expected_sqrt_3)
    assert np.shape(glength) == (48000,)
    expected_divx = loaded_data["expected_outcomes"]["test_TetMesh_Geodesics"][
        "expected_divx"
    ]
    assert len(divx2[5000:5010]) == len(expected_divx)
    assert not np.all(divx2[5000:5010] == expected_divx), "divergence is equal"
