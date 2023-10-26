import json

import numpy as np
import pytest

from ...shapedna import compute_distance, compute_shapedna, normalize_ev, reweight_ev
from ...tet_mesh import TetMesh
from ...tria_mesh import TriaMesh

tria = TriaMesh.read_vtk("data/cubeTria.vtk")
tet = TetMesh.read_vtk("data/cubeTetra.vtk")


@pytest.fixture
def loaded_data():
    """
    Load expected outcomes data from a JSON file as a dictionary.
    """
    with open("lapy/utils/tests/expected_outcomes.json", "r") as f:
        expected_outcomes = json.load(f)
    return expected_outcomes


def test_compute_shapedna(loaded_data):
    """
    Test compute_shapedna function for triangular mesh.

    Args:
        loaded_data (dict): Expected outcomes loaded from a JSON file.

    Raises:
        AssertionError: If computed eigenvalues don't match expected values within tolerance.
        AssertionError: If eigenvalues' dtype isn't float32.
    """
    ev = compute_shapedna(tria, k=3)

    expected_Eigenvalues = np.array(
        loaded_data["expected_outcomes"]["test_compute_shapedna"][
            "expected_eigenvalues"
        ]
    )
    tolerance = loaded_data["expected_outcomes"]["test_compute_shapedna"]["tolerance"]
    assert np.allclose(ev["Eigenvalues"], expected_Eigenvalues, atol=tolerance)
    assert ev["Eigenvalues"].dtype == np.float32


def test_normalize_ev_geometry(loaded_data):
    """
    Test normalize_ev() using 'geometry' method for a triangular mesh.

    Args:
        loaded_data (dict): Expected outcomes from a JSON file.

    Raises:
        AssertionError: If normalized eigenvalues don't match expected values within tolerance.
        AssertionError: If normalized eigenvalues' dtype isn't float32.
    """
    ev = compute_shapedna(tria, k=3)

    expected_normalized_values = np.array(
        loaded_data["expected_outcomes"]["test_normalize_ev_geometry"][
            "expected_normalized_values"
        ]
    )
    tolerance = loaded_data["expected_outcomes"]["test_normalize_ev_geometry"][
        "tolerance"
    ]
    normalized_eigenvalues = normalize_ev(tria, ev["Eigenvalues"], method="geometry")
    assert np.allclose(
        normalized_eigenvalues, expected_normalized_values, atol=tolerance
    )
    assert normalized_eigenvalues.dtype == np.float32


def test_reweight_ev(loaded_data):
    """
    Test reweighted_ev() and validate reweighted eigenvalues' data type.

    Args:
        loaded_data (dict): Expected outcomes from a JSON file.

    Raises:
        AssertionError: If reweighted eigenvalues don't match expected values within tolerance.
        AssertionError: If reweighted eigenvalues' dtype isn't float32.
    """
    ev = compute_shapedna(tria, k=3)

    expected_reweighted_values = np.array(
        loaded_data["expected_outcomes"]["test_reweight_ev"][
            "expected_reweighted_values"
        ]
    )
    tolerance = loaded_data["expected_outcomes"]["test_reweight_ev"]["tolerance"]
    reweighted_eigenvalues = reweight_ev(ev["Eigenvalues"])
    tolerance = 1e-4
    assert np.allclose(
        reweighted_eigenvalues, expected_reweighted_values, atol=tolerance
    )


def test_compute_distance(loaded_data):
    """
    Test compute_distance() for eigenvalues and validate the computed distance.

    Args:
        loaded_data (dict): Expected outcomes from a JSON file.

    Raises:
        AssertionError: If computed distance doesn't match the expected value.
    """
    ev = compute_shapedna(tria, k=3)

    expected_compute_distance = loaded_data["expected_outcomes"][
        "test_compute_distance"
    ]["expected_compute_distance"]
    # compute distance for tria eigenvalues (trivial case)
    computed_distance = compute_distance(ev["Eigenvalues"], ev["Eigenvalues"])
    assert computed_distance == expected_compute_distance


# Repeating test steps for a tetrahedral mesh


def test_compute_shapedna_tet(loaded_data):
    """
    Test compute_shapedna for a tetrahedral mesh.

    Args:
        loaded_data (dict): Expected outcomes from a JSON file.

    Raises:
        AssertionError: If computed eigenvalues don't match expected values within tolerance.
        AssertionError: If eigenvalues' dtype isn't float32.
    """
    evTet = compute_shapedna(tet, k=3)

    expected_eigen_values = np.array(
        loaded_data["expected_outcomes"]["test_compute_shapedna_tet"][
            "expected_eigen_values"
        ]
    )
    tolerance = loaded_data["expected_outcomes"]["test_compute_shapedna_tet"][
        "tolerance"
    ]
    evTet = compute_shapedna(tet, k=3)
    assert np.allclose(evTet["Eigenvalues"], expected_eigen_values, atol=tolerance)
    assert evTet["Eigenvalues"].dtype == np.float32


def test_normalize_ev_geometry_tet(loaded_data):
    """
    Test normalize_ev() using 'geometry' method for a tetrahedral mesh.

    Args:
        loaded_data (dict): Expected outcomes from a JSON file.

    Raises:
        AssertionError: If normalized eigenvalues don't match expected values within tolerance.
        AssertionError: If normalized eigenvalues' dtype isn't float32.
    """
    evTet = compute_shapedna(tet, k=3)

    expected_normalized_values = np.array(
        loaded_data["expected_outcomes"]["test_normalize_ev_geometry_tet"][
            "expected_normalized_values"
        ]
    )
    tolerance = loaded_data["expected_outcomes"]["test_normalize_ev_geometry_tet"][
        "tolerance"
    ]
    # volume / surface / geometry normalization of tet eigenvalues
    normalized_eigenvalues = normalize_ev(tet, evTet["Eigenvalues"], method="geometry")

    assert np.allclose(
        normalized_eigenvalues, expected_normalized_values, atol=tolerance
    )
    assert normalized_eigenvalues.dtype == np.float32


def test_reweight_ev_tet(loaded_data):
    """
    Test reweighted_ev() for tetrahedral meshes and validate reweighted eigenvalues' data type.

    Args:
        loaded_data (dict): Expected outcomes from a JSON file.

    Raises:
        AssertionError: If reweighted eigenvalues don't match expected values within tolerance.
    """
    evTet = compute_shapedna(tet, k=3)

    expected_reweighted_values = np.array(
        loaded_data["expected_outcomes"]["test_reweight_ev_tet"][
            "expected_reweighted_values"
        ]
    )
    tolerance = loaded_data["expected_outcomes"]["test_reweight_ev_tet"]["tolerance"]
    # Linear reweighting of tet eigenvalues
    reweighted_eigenvalues = reweight_ev(evTet["Eigenvalues"])
    assert np.allclose(
        reweighted_eigenvalues, expected_reweighted_values, atol=tolerance
    )


def test_compute_distance_tet(loaded_data):
    """
    Test compute_distance() for eigenvalues of tetrahedral meshes and validate computed distance.

    Args:
        loaded_data (dict): Expected outcomes from a JSON file.

    Raises:
        AssertionError: If computed distance doesn't match the expected value.
    """
    evTet = compute_shapedna(tet, k=3)

    # compute distance for tria eigenvalues (trivial case)
    computed_distance = compute_distance(evTet["Eigenvalues"], evTet["Eigenvalues"])
    expected_compute_distance = loaded_data["expected_outcomes"][
        "test_compute_distance_tet"
    ]["exp_compute_distance"]

    # Compare the computed distance with the expected distance using a tolerance
    assert computed_distance == expected_compute_distance
