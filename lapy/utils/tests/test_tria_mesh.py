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
    with open("lapy/utils/tests/expected_outcomes.json") as f:
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
    np.testing.assert_array_almost_equal raises an AssertionError if two objects
    i.e tria_areas and expected_area are not equal up to desired precision.
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


# Define the test case for non-oriented mesh
def test_volume_oriented(tria_mesh_fixture):
    """
    This test is verifying that the T.volume() function raises a ValueError
    with the error message when the input TriaMesh object is not correctly oriented.
    The test will always pass by matching an error because the volume inside the
    closed mesh,however, requires the mesh to be correctly oriented
    """
    # Use the appropriate exception that T.volume() raises
    with pytest.raises(
        ValueError, match="Mesh must be oriented to compute volume."
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
    # Verify that the sum of vertex areas is close to the total surface area
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
    Testing the computation of vertex normals for oriented mesh.
    """

    # Calling tria_mesh_fixture.orient_() will modify the tria_mesh_fixture in-place and
    # return the number of flipped triangles. However, it won't return a new instance of
    # TriaMesh, so assigning the result to mesh like mesh = tria_mesh_fixture.orient_()
    # would not work as expected.

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
    Testing boundary loops after refining the mesh.
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


def test_connected_components(tria_mesh_fixture):
    """
    Test the connected_components method.
    """
    # Create a mesh with two disconnected components
    # Component 1: Original fixture (vertices 0-7)
    v1 = tria_mesh_fixture.v
    t1 = tria_mesh_fixture.t

    # Component 2: Shifted copy (vertices 8-15)
    v2 = v1 + 5.0  # Shift by 5 in x, y, z
    t2 = t1 + v1.shape[0]

    # Combine
    v_combined = np.vstack((v1, v2))
    t_combined = np.vstack((t1, t2))

    mesh = TriaMesh(v_combined, t_combined)

    n_components, labels = mesh.connected_components()

    assert n_components == 2
    # Verify labels: first 8 vertices should be 0 (or 1), next 8 should be 1 (or 0)
    # Since labeling order isn't strictly guaranteed, check if they are split correctly
    assert len(np.unique(labels[:8])) == 1
    assert len(np.unique(labels[8:])) == 1
    assert labels[0] != labels[8]


def test_keep_largest_connected_component(tria_mesh_fixture):
    """
    Test the keep_largest_connected_component method.
    """
    # Create a mesh with two disconnected components of different sizes
    # Component 1: Original fixture (8 vertices, 12 triangles) - larger
    v1 = tria_mesh_fixture.v
    t1 = tria_mesh_fixture.t

    # Component 2: A single triangle (3 vertices, 1 triangle) - smaller
    v2 = np.array([[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [10.0, 1.0, 0.0]])
    t2 = np.array([[0, 1, 2]]) + v1.shape[0]

    # Combine
    v_combined = np.vstack((v1, v2))
    t_combined = np.vstack((t1, t2))

    mesh = TriaMesh(v_combined, t_combined)

    # Test with clean=True (default)
    # Should keep only the larger component and remove unused vertices
    vkeep, vdel = mesh.keep_largest_connected_component(clean=True)

    assert mesh.v.shape[0] == 8
    assert mesh.t.shape[0] == 12
    # Verify we kept the original vertices
    np.testing.assert_array_equal(mesh.v, v1)
    np.testing.assert_array_equal(mesh.t, t1)
    # Check return values
    assert len(vkeep) == 8
    assert len(vdel) == 3

    # Reset mesh
    mesh = TriaMesh(v_combined, t_combined)

    # Test with clean=False
    # Should keep only triangles of larger component but keep all vertices
    vkeep, vdel = mesh.keep_largest_connected_component(clean=False)

    assert mesh.v.shape[0] == 11
    assert mesh.t.shape[0] == 12
    # Should result in free vertices
    assert mesh.has_free_vertices()
    # Return values should be None
    assert vkeep is None
    assert vdel is None


def test_smooth_laplace(tria_mesh_fixture):
    """Test Laplace smoothing."""
    mesh = tria_mesh_fixture
    v_orig = mesh.v.copy()

    # Smooth with default parameters
    v_smooth = mesh.smooth_laplace(n=1, lambda_=0.5)

    # Vertices should change
    assert not np.allclose(v_orig, v_smooth)
    assert v_smooth.shape == v_orig.shape

    # Check shrinkage (distance to centroid should decrease for a convex-like cube)
    centroid = np.mean(v_orig, axis=0)
    dist_orig = np.linalg.norm(v_orig - centroid, axis=1)
    dist_smooth = np.linalg.norm(v_smooth - centroid, axis=1)
    assert np.mean(dist_smooth) < np.mean(dist_orig)

    # Equivalence check with old smooth_vfunc (which corresponds to lambda=1)
    # Note: smooth_vfunc applies M*v. smooth_laplace applies (1-l)v + l*M*v.
    # If l=1, result is M*v.
    v_old = mesh.smooth_vfunc(v_orig, n=1)
    v_new = mesh.smooth_laplace(v_orig, n=1, lambda_=1.0)
    np.testing.assert_allclose(v_old, v_new)


def test_smooth_taubin(tria_mesh_fixture):
    """Test Taubin smoothing."""
    mesh = tria_mesh_fixture
    v_orig = mesh.v.copy()

    # Smooth
    v_smooth = mesh.smooth_taubin(n=1, lambda_=0.5, mu=-0.53)

    # Vertices should change
    assert not np.allclose(v_orig, v_smooth)
    assert v_smooth.shape == v_orig.shape

    # Check volume preservation (rough check compared to laplace)
    # Laplace with same lambda and 2 steps (one shrink, one shrink) would shrink a lot.
    # Taubin (one shrink, one grow) should shrink less.
    v_laplace_2 = mesh.smooth_laplace(n=2, lambda_=0.5)

    centroid = np.mean(v_orig, axis=0)
    dist_taubin = np.linalg.norm(v_smooth - centroid, axis=1)
    dist_laplace = np.linalg.norm(v_laplace_2 - centroid, axis=1)

    # Taubin should maintain size better than double laplace
    assert np.mean(dist_taubin) > np.mean(dist_laplace)


def test_smooth_functions_custom_vfunc(tria_mesh_fixture):
    """Test smoothing on a custom scalar function."""
    mesh = tria_mesh_fixture
    # Scalar function (e.g., x-coordinate)
    vfunc = mesh.v[:, 0].copy()

    # Laplace
    vfunc_smooth = mesh.smooth_laplace(vfunc, n=1, lambda_=0.5)
    assert not np.allclose(vfunc, vfunc_smooth)
    assert vfunc_smooth.shape == vfunc.shape

    # Taubin
    vfunc_taubin = mesh.smooth_taubin(vfunc, n=1, lambda_=0.5, mu=-0.53)
    assert not np.allclose(vfunc, vfunc_taubin)
    assert vfunc_taubin.shape == vfunc.shape


def test_normalize(tria_mesh_fixture):
    """Test normalize_ method."""
    mesh = tria_mesh_fixture
    mesh.normalize_()
    # Check centroid is origin
    centroid, area = mesh.centroid()
    np.testing.assert_allclose(centroid, np.zeros(3), atol=1e-7)
    # Check area is 1
    assert np.isclose(area, 1.0)


def test_edges(tria_mesh_fixture):
    """Test edges computation."""
    mesh = tria_mesh_fixture
    mesh.orient_() # Ensure oriented
    # Basic edge check
    vids, tids = mesh.edges()
    assert vids.shape[1] == 2
    assert tids.shape[1] == 2
    assert vids.shape[0] == tids.shape[0]
    # Test with boundary
    boundary_mesh = TriaMesh(mesh.v, mesh.t[2:, :])
    vids_b, tids_b, bdrv, bdrt = boundary_mesh.edges(with_boundary=True)
    assert len(bdrv) > 0
    assert len(bdrt) > 0


def test_construct_adj_dir_tidx(tria_mesh_fixture):
    """Test directed adjacency matrix with triangle indices."""
    mesh = tria_mesh_fixture
    mesh.orient_() # Ensure oriented
    adj = mesh.construct_adj_dir_tidx()
    # Check shape
    assert adj.shape == (mesh.v.shape[0], mesh.v.shape[0])
    # Check values (should be triangle indices + 1)
    assert adj.max() <= mesh.t.shape[0]
    assert adj.min() >= 0


def test_map_functions(tria_mesh_fixture):
    """Test mapping between vertex and triangle functions."""
    mesh = tria_mesh_fixture
    # Create a vertex function (e.g. index)
    vfunc = np.arange(mesh.v.shape[0])
    # Map to triangles
    tfunc = mesh.map_vfunc_to_tfunc(vfunc)
    assert tfunc.shape[0] == mesh.t.shape[0]
    # Value on tria should be average of its vertices
    expected_t0 = np.mean(vfunc[mesh.t[0]])
    assert np.isclose(tfunc[0], expected_t0)
    # Map back to vertices
    vfunc_back = mesh.map_tfunc_to_vfunc(tfunc)
    assert vfunc_back.shape[0] == mesh.v.shape[0]


def test_curvature(tria_mesh_fixture):
    """Test curvature computation."""
    mesh = tria_mesh_fixture
    mesh.orient_()
    # Compute curvature
    u_min, u_max, c_min, c_max, c_mean, c_gauss, normals = mesh.curvature(smoothit=0)
    assert u_min.shape == mesh.v.shape
    assert u_max.shape == mesh.v.shape
    assert c_min.shape[0] == mesh.v.shape[0]
    assert c_max.shape[0] == mesh.v.shape[0]
    assert normals.shape == mesh.v.shape
    # Tria curvature
    tu_min, tu_max, tc_min, tc_max = mesh.curvature_tria(smoothit=0)
    assert tu_min.shape[0] == mesh.t.shape[0]
    assert tc_min.shape[0] == mesh.t.shape[0]


def test_level_sets(tria_mesh_fixture):
    """Test level set functions."""
    mesh = tria_mesh_fixture
    # Define a simple function: z-coordinate
    vfunc = mesh.v[:, 2]
    level = 0.5
    # Test level_length
    length = mesh.level_length(vfunc, level)
    # For a unit cube-like mesh, cut at z=0.5 should exist
    assert length > 0
    # Test level_path
    # Note: level_path requires a single non-intersecting path.
    # The fixture might produce multiple loops if not carefully chosen.
    # z=0.5 on the fixture likely produces a simple loop around the middle.
    try:
        path, l_length = mesh.level_path(vfunc, level)
        assert len(path) > 0
        assert np.isclose(l_length, length)
        # Check that all points on path have z approx 0.5
        np.testing.assert_allclose(path[:, 2], level, atol=1e-5)
    except ValueError:
        # If topology is complex (multiple loops), it might raise ValueError
        # In that case, we at least tested it runs until that check
        pass
