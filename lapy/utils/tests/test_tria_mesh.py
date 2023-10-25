import json

import numpy as np
import pytest

from ...tria_mesh import TriaMesh


@pytest.fixture
def tria_mesh_fixture():
    """
    fixture is a method to parse parameters to the class
    only once so that it is not required in each test case
    """
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [1, 0, 1],
        ]
    )
    trias = np.array(
        [
            [0, 1, 2],
            [2, 3, 0],
            [4, 5, 6],
            [6, 7, 4],
            [0, 4, 7],
            [7, 3, 0],
            [0, 4, 5],
            [5, 1, 0],
            [1, 5, 6],
            [6, 2, 1],
            [3, 7, 6],
            [6, 2, 3],
        ]
    )
    return TriaMesh(points, trias)


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


def test_is_closed(tria_mesh_fixture):
    """
    testing whether the function is_closed() returns True
    """

    mesh = tria_mesh_fixture
    result = mesh.is_closed()
    assert result is True


def test_is_manifold(tria_mesh_fixture):
    """
    Testing whether the function is_manifold() returns 1
    """
    mesh = tria_mesh_fixture
    result = mesh.is_manifold()
    expected_result = True
    assert (
        result == expected_result
    ), f"Expected is_manifold result {expected_result}, but got {result}"


def test_is_oriented(tria_mesh_fixture):
    """
    Testing whether the function is_oriented() returns True
    """
    T = tria_mesh_fixture
    result = T.is_oriented()
    expected_result = False
    assert result == expected_result, f"returning {result}"


def test_euler(tria_mesh_fixture, loaded_data):
    """
    Testing whether the function euler() is equal to 2
    """
    mesh = tria_mesh_fixture
    expected_euler_value = loaded_data["expected_outcomes"]["test_tria_mesh"][
        "expected_euler_value"
    ]
    result = mesh.euler()
    assert (
        result == expected_euler_value
    ), f"Expected Euler characteristic 2, but got {result}"


def test_tria_areas(tria_mesh_fixture, loaded_data):
    """
    np.testing.assert_array_almost_equal raises an AssertionError if two objects i.e tria_areas and expected_area
    are not equal up to desired precision.
    """
    expected_area = np.array(
        loaded_data["expected_outcomes"]["test_tria_mesh"]["expected_area"]
    )

    mesh = tria_mesh_fixture
    result = mesh.tria_areas()
    np.testing.assert_array_almost_equal(result, expected_area)


def test_area(tria_mesh_fixture, loaded_data):
    """
    Testing whether the function area() return is almost equal to expected area
    """
    mesh = tria_mesh_fixture
    result = mesh.area()
    expected_mesh_area = float(
        loaded_data["expected_outcomes"]["test_tria_mesh"]["expected_mesh_area"]
    )
    assert result == pytest.approx(expected_mesh_area)


def test_volume(tria_mesh_fixture):
    """
    Testing the volume calculation of the mesh for an unoriented mesh
    """
    # Assuming that tria_mesh_fixture is unoriented
    try:
        tria_mesh_fixture.volume()
    except ValueError as e:
        assert "Can only compute volume for oriented triangle meshes!" in str(e)
    else:
        assert False  # The function should raise a ValueError


# Define the test case for non-oriented mesh
def test_volume_oriented(tria_mesh_fixture):
    """
    This test is verifying that the T.volume() function raises a ValueError
    with the error message when the input TriaMesh object is not correctly oriented.
    The test will always pass by matching an error because the volume inside the closed mesh,
    however, requires the mesh to be correctly oriented
    """
    # Use the appropriate exception that T.volume() raises
    with pytest.raises(
        ValueError, match="Error: Can only compute volume for oriented triangle meshes!"
    ):
        tria_mesh_fixture.volume()


def test_vertex_degrees(tria_mesh_fixture, loaded_data):
    """
    Testing the calculation of vertex degrees
    """
    mesh = tria_mesh_fixture
    result = mesh.vertex_degrees()
    expected_vertex_degrees = np.array(
        loaded_data["expected_outcomes"]["test_tria_mesh"]["expected_vertex_degrees"]
    )
    np.testing.assert_array_equal(result, expected_vertex_degrees)


def test_vertex_areas(tria_mesh_fixture, loaded_data):
    """
    Testing the calculation of vertex areas
    """
    expected_vertex_area = np.array(
        loaded_data["expected_outcomes"]["test_tria_mesh"]["expected_vertex_area"]
    )
    mesh = tria_mesh_fixture
    result = mesh.vertex_areas()
    np.testing.assert_almost_equal(result, expected_vertex_area)
    # Verify that the sum of vertex areas is approximately equal to the total surface area
    vertex_areas_sum = np.sum(mesh.vertex_areas())
    total_surface_area = mesh.area()
    assert np.isclose(vertex_areas_sum, total_surface_area)


def test_avg_edge_length(tria_mesh_fixture, loaded_data):
    """
    Testing the calculation of average edge length
    """
    mesh = tria_mesh_fixture
    expected_edge_length = float(
        loaded_data["expected_outcomes"]["test_tria_mesh"]["expected_edge_length"]
    )
    result = mesh.avg_edge_length()
    assert np.isclose(
        result, expected_edge_length
    ), f"Average edge length {result} is not equal to expected {expected_edge_length}"


def test_tria_normals(tria_mesh_fixture, loaded_data):
    """
    Testing whether the shape of tria_normals array is equal to
    (n_triangles,3)
    """
    expected_triangle_normals = np.array(
        loaded_data["expected_outcomes"]["test_tria_mesh"]["expected_triangle_normals"]
    )
    mesh = tria_mesh_fixture
    result = mesh.tria_normals()
    np.testing.assert_allclose(result, expected_triangle_normals)


def test_tria_qualities(tria_mesh_fixture, loaded_data):
    """
    Testing the calculation of triangle qualities
    """
    mesh = tria_mesh_fixture
    result = mesh.tria_qualities()
    expected_triangle = np.array(
        loaded_data["expected_outcomes"]["test_tria_mesh"]["expected_triangle"]
    )
    np.testing.assert_almost_equal(result, expected_triangle)


def test_has_free_vertices(tria_mesh_fixture):
    """
    Testing the detection of free vertices
    """
    mesh = tria_mesh_fixture
    result = mesh.has_free_vertices()

    expected_result = False
    assert (
        result == expected_result
    ), f"Expected has_free_vertices {expected_result}, but got {result}"


def test_rm_free_vertices(tria_mesh_fixture, loaded_data):
    """
    Testing the removal of free vertices
    """
    updated_vertices, deleted_vertices = tria_mesh_fixture.rm_free_vertices_()
    expected_vertices = np.array(
        loaded_data["expected_outcomes"]["test_tria_mesh"]["expected_vertices"]
    )
    expected_deleted_vertices = np.array([])
    assert np.array_equal(updated_vertices, expected_vertices)
    assert np.array_equal(deleted_vertices, expected_deleted_vertices)


# Define the test case for orient_ function
def test_orient(tria_mesh_fixture, loaded_data):
    """
    Testing the orienting of the mesh
    """
    # Call the tria_mesh_fixture.orient_() method to re-orient the mesh consistently
    mesh = tria_mesh_fixture
    flipped = mesh.orient_()

    # Check if the returned 'flipped' count matches the expected count i.e 6
    expected_flips = np.array(
        loaded_data["expected_outcomes"]["test_tria_mesh"]["expected_flips"]
    )
    assert flipped == expected_flips


def test_is_oriented_(tria_mesh_fixture):
    """
    Testing the check for mesh orientation
    """
    mesh = tria_mesh_fixture
    "orient the mesh consistently so that all triangle normals point outwards."
    mesh.orient_()
    result = mesh.is_oriented()
    expected_result = True
    assert (
        result == expected_result
    ), f"Expected is_oriented result {expected_result}, but got {result}"


## Compute volume (works only for oriented meshes)


def test_volume_(tria_mesh_fixture):
    """
    Testing the computation of enclosed volume for oriented mesh
    """
    mesh = tria_mesh_fixture
    mesh.orient_()
    result = mesh.volume()
    expected_result = 1.0
    assert (
        result == expected_result
    ), f"Expected volume result {expected_result}, but got {result}"


def test_vertex_normals(tria_mesh_fixture, loaded_data):
    """
    Testing the computation of vertex normals for oriented mesh
    """

    # Calling tria_mesh_fixture.orient_() will modify the tria_mesh_fixture in-place and
    # return the number of flipped triangles. However, it won't return a new instance of TriaMesh, so assigning
    # the result to mesh like mesh = tria_mesh_fixture.orient_() would not work as expected.

    # Ensure the mesh is oriented before computing vertex normals
    tria_mesh_fixture.orient_()
    mesh = tria_mesh_fixture
    result = mesh.vertex_normals()
    expected_result = np.array(
        loaded_data["expected_outcomes"]["test_tria_mesh"]["expected_result"]
    )
    np.testing.assert_allclose(result, expected_result)


def test_normal_offset(tria_mesh_fixture, loaded_data):
    """
    Testing the normal offset operation
    """

    # Orient the mesh before applying normal offset
    mesh = tria_mesh_fixture
    mesh.orient_()

    # Get the initial vertex coordinates
    mesh.v.copy()

    # Calculate the distance 'd' for the offset
    d = 0.2 * mesh.avg_edge_length()

    # Act: Perform the 'normal_offset_' operation
    mesh.normal_offset_(d)


def test_boundary_mesh(tria_mesh_fixture):
    # Original mesh with closed boundaries

    original_mesh = tria_mesh_fixture

    # Create a boundary mesh by dropping triangles
    boundary_mesh = TriaMesh(original_mesh.v, original_mesh.t[2:, :])

    # Check if the boundary mesh has the correct number of vertices and triangles
    assert boundary_mesh.v.shape[0] == original_mesh.v.shape[0]
    assert boundary_mesh.t.shape[0] == original_mesh.t.shape[0] - 2


def test_refine_and_boundary_loops(tria_mesh_fixture, loaded_data):
    """
    Testing boundary loops after refining the mesh
    """
    # Create a boundary mesh by dropping triangles
    tria_mesh_fixture.orient_()

    original_mesh = tria_mesh_fixture

    boundary_mesh = TriaMesh(original_mesh.v, original_mesh.t[2:, :])

    # Refine the boundary mesh
    boundary_mesh.refine_()

    # Get the boundary loops of the refined mesh
    boundary_loops = boundary_mesh.boundary_loops()

    # Check if there is only one boundary loop
    assert len(boundary_loops) == 1

    # Check the vertices along the boundary loop
    expected_boundary_loop = loaded_data["expected_outcomes"]["test_tria_mesh"][
        "expected_boundary_loop"
    ]

    assert boundary_loops[0] == expected_boundary_loop
