import json
from pathlib import Path

import pytest

from ...io import write_ev
from ...solver import Solver
from ...tet_mesh import TetMesh
from ...tria_mesh import TriaMesh


# Fixture to load the TetMesh
@pytest.fixture
def load_tria_mesh():
    tria = TriaMesh.read_vtk("data/cubeTria.vtk")
    return tria


@pytest.fixture
def load_tet_mesh():
    tetra = TetMesh.read_vtk("data/cubeTetra.vtk")
    return tetra


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


def test_visualization_triangle_mesh(load_tria_mesh, loaded_data):
    """
    Test visualization of a triangle mesh using expected outcomes.

    Parameters:
    - load_tria_mesh (fixture): Fixture for loading a triangle mesh.
    - loaded_data (fixture): Fixture for loading expected outcomes.

    Raises:
    - AssertionError: If any test assertions fail.
    """
    tria = load_tria_mesh
    fem = Solver(tria)
    evals, evecs = fem.eigs(k=3)
    evDict = dict()
    evDict["Refine"] = 0
    evDict["Degree"] = 1
    evDict["Dimension"] = 2
    evDict["Elements"] = len(tria.t)
    evDict["DoF"] = len(tria.v)
    evDict["NumEW"] = 3
    evDict["Eigenvalues"] = evals
    evDict["Eigenvectors"] = evecs
    write_ev("data/cubeTria.ev", evDict)
    output_file = Path("data/cubeTria.ev")
    assert output_file.exists()  # Check if the output file exists
    expected_elements = loaded_data["expected_outcomes"][
        "test_visualization_triangle_mesh"
    ]["expected_elements"]
    expected_dof = loaded_data["expected_outcomes"]["test_visualization_triangle_mesh"][
        "expected_dof"
    ]
    expected_ev = loaded_data["expected_outcomes"]["test_visualization_triangle_mesh"][
        "expected_ev"
    ]

    expected_evec_shape = (2402, 3)
    assert evDict["Elements"] == expected_elements
    assert evDict["DoF"] == expected_dof
    assert evals == pytest.approx(expected_ev, rel=1e-5, abs=1e-5)
    assert evecs.shape == expected_evec_shape


def test_visualization_tetrahedral_mesh(load_tet_mesh, loaded_data):
    """
    Test visualization of a tetrahedral mesh using expected outcomes.

    Parameters:
    - load_tet_mesh (fixture): Fixture for loading a tetrahedral mesh.
    - loaded_data (fixture): Fixture for loading expected outcomes.

    Raises:
    - AssertionError: If any test assertions fail.
    """
    tetra = load_tet_mesh
    fem = Solver(tetra)
    evals, evecs = fem.eigs(k=3)
    evDict = dict()
    evDict["Refine"] = 0
    evDict["Degree"] = 1
    evDict["Dimension"] = 2
    evDict["Elements"] = len(tetra.t)
    evDict["DoF"] = len(tetra.v)
    evDict["NumEW"] = 3
    evDict["Eigenvalues"] = evals
    evDict["Eigenvectors"] = evecs
    write_ev("data/cubeTetra.ev", evDict)
    output_file = Path("data/cubeTetra.ev")
    assert output_file.exists()  # Check if the output file exists
    expected_elements = loaded_data["expected_outcomes"][
        "test_visualization_tetrahedral_mesh"
    ]["expected_elements"]
    expected_dof = loaded_data["expected_outcomes"][
        "test_visualization_tetrahedral_mesh"
    ]["expected_dof"]
    expected_ev = loaded_data["expected_outcomes"][
        "test_visualization_tetrahedral_mesh"
    ]["expected_ev"]
    expected_evec_shape = (9261, 3)
    assert evDict["Elements"] == expected_elements
    assert evDict["DoF"] == expected_dof
    assert evals == pytest.approx(expected_ev, rel=1e-5, abs=1e-5)
    assert evecs.shape == expected_evec_shape
