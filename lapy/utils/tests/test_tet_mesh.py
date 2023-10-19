import json
import logging

import numpy as np
import pytest

from ...tet_mesh import TetMesh


@pytest.fixture
def tet_mesh_fixture():
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
            [0.5, 0.5, 0.5],
        ]
    )
    tets = np.array(
        [
            [0, 5, 8, 1],
            [0, 4, 5, 8],
            [2, 5, 6, 8],
            [1, 5, 2, 8],
            [6, 7, 3, 8],
            [6, 3, 2, 8],
            [0, 3, 4, 8],
            [3, 7, 4, 8],
            [0, 1, 2, 8],
            [0, 2, 3, 8],
            [4, 6, 5, 8],
            [4, 7, 6, 8],
        ]
    )

    return TetMesh(points, tets)


@pytest.fixture
def loaded_data():
    """
    Load expected outcomes data from a JSON file as a dictionary.
    """
    with open("lapy/utils/tests/expected_outcomes.json", "r") as f:
        expected_outcomes = json.load(f)
    return expected_outcomes


def test_has_free_vertices(tet_mesh_fixture):
    """
    Testing tet mesh has free vertices or not
    """
    mesh = tet_mesh_fixture
    result = mesh.has_free_vertices()
    expected_result = False
    assert result == expected_result


def test_rm_free_vertices(tet_mesh_fixture, loaded_data):
    """
    Testing removing free vertices from tet mesh
    """
    mesh = tet_mesh_fixture
    updated_vertices, deleted_vertices = mesh.rm_free_vertices_()
    expected_vertices = np.array(
        loaded_data["expected_outcomes"]["test_tet_mesh"]["expected_vertices"]
    )
    expected_removed_vertices = np.array([])
    assert np.array_equal(
        updated_vertices, expected_vertices
    ), f"{updated_vertices}, {deleted_vertices}"
    assert np.array_equal(deleted_vertices, expected_removed_vertices)


def test_is_oriented(tet_mesh_fixture):
    """
    Testing whether test mesh orientations are consistent
    """
    mesh = tet_mesh_fixture
    result = mesh.is_oriented()
    expected_result = False
    assert (
        result == expected_result
    ), f"Expected is_oriented result {expected_result}, but got {result}"


def test_avg_edge_length(tet_mesh_fixture):
    """
    Testing computation of average edge length
    """
    expected_result = 1.0543647924813107
    mesh = tet_mesh_fixture
    result = mesh.avg_edge_length()

    assert (
        result == expected_result
    ), f"Expected average edge length {expected_result}, but got {result}"


def test_boundary_trai(tet_mesh_fixture):
    """
    Test computation of boundary triangles from tet mesh.

    - `BT.t` represents the array of boundary triangles.
    - `.shape[0]` counts the number of boundary triangles.
    """
    mesh = tet_mesh_fixture
    boundary_tria_mesh = mesh.boundary_tria()

    expected_num_traingles = 12
    assert boundary_tria_mesh.t.shape[0] == expected_num_traingles

    # print(f"Found {boundary_tria_mesh.t.shape[0]} triangles on boundary.")

    # Check if the boundary triangle mesh is not oriented (this should fail)
    result = boundary_tria_mesh.is_oriented()
    expected_result = False
    assert (
        result == expected_result
    ), f"Expected is_oriented result {expected_result}, but got {result}"


def test_avg_edge_length(tet_mesh_fixture):
    """
    Testing the computatoin of average edge length for tetrahedral mesh
    """
    mesh = tet_mesh_fixture
    result = mesh.avg_edge_length()

    expected_avg_edge_length = 1.0543647924813107

    assert np.isclose(result, expected_avg_edge_length)


def test_boundary_is_oriented(tet_mesh_fixture):
    """
    Test orientation consistency in boundary of tetrahedral mesh.    
    """
    mesh = tet_mesh_fixture

    # Get the boundary triangle mesh
    boundary_mesh = mesh.boundary_tria()

    # Check if the boundary triangle mesh has consistent orientations
    result = boundary_mesh.is_oriented()

    expected_result = False

    assert result == expected_result


def test_orient_and_check_oriented(tet_mesh_fixture):
    """
    Test orienting the tetrahedral mesh for consistency.
    """
    mesh = tet_mesh_fixture

    # Correct the orientation of the tetrahedral mesh
    flipped_tetrahedra = mesh.orient_()

    # Check if the orientations of the tetrahedra are consistent
    result = mesh.is_oriented()

    expected_flipped_tetrahedra = 1
    expected_oriented_result = True

    # print(f"{flipped_tetrahedra}")

    assert flipped_tetrahedra == expected_flipped_tetrahedra
    assert result == expected_oriented_result


def test_correct_orientations_and_boundary(tet_mesh_fixture):
    """
    Testing correcting orientation and checking boundary surface orientation
    """
    mesh = tet_mesh_fixture

    # Correct the orientation of the tetrahedral mesh
    flipped_tetrahedra = mesh.orient_()

    # Check if the orientations of the tetrahedra are consistent
    result_oriented = mesh.is_oriented()
    expected_oriented_result = True
    assert result_oriented == expected_oriented_result

    # Extract the boundary surface
    boundary_surface = mesh.boundary_tria()
    print(f"{boundary_surface}")

    # Check if the orientations of the boundary surface are consistent
    result_boundary_oriented = boundary_surface.is_oriented()
    print(f"{result_boundary_oriented}")
    expected_boundary_oriented_result = True
    assert result_boundary_oriented == expected_boundary_oriented_result


def test_boundary_surface_volume(tet_mesh_fixture):
    """
    Testing computation of volume for the boundary surface mesh
    """
    mesh = tet_mesh_fixture

    # Correct the orientation of the tetrahedral mesh
    mesh.orient_()

    # Extract the boundary surface
    boundary_surface = mesh.boundary_tria()

    # Compute the volume of the boundary surface
    result_volume = boundary_surface.volume()
    expected_volume = 1.0

    assert np.isclose(result_volume, expected_volume)
