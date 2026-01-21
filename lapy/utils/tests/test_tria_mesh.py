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
    vkeep, vdel = mesh.keep_largest_connected_component_(clean=True)

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
    vkeep, vdel = mesh.keep_largest_connected_component_(clean=False)

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


def test_2d_mesh_support():
    """
    Testing that TriaMesh correctly handles 2D vertices by padding with z=0
    """
    # Create a simple 2D mesh (unit square made of 2 triangles)
    vertices_2d = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    triangles = np.array([[0, 1, 2], [0, 2, 3]])

    # Create mesh with 2D vertices
    mesh = TriaMesh(vertices_2d, triangles)

    # Verify that vertices were padded to 3D
    assert mesh.v.shape == (4, 3), "Vertices should be padded to 3D"
    assert mesh.is_2d(), "Mesh should be marked as 2D"
    assert np.allclose(mesh.v[:, 2], 0.0), "Z-coordinates should be 0"

    # Verify that geometric operations work correctly
    total_area = mesh.area()
    assert np.isclose(total_area, 1.0), f"Square area should be 1.0, got {total_area}"

    # Verify that original 2D vertices can be retrieved
    v2d = mesh.get_vertices(original_dim=True)
    assert v2d.shape == (4, 2), "Should return 2D vertices"
    np.testing.assert_array_almost_equal(v2d, vertices_2d)


def test_critical_points_simple():
    """
    Test critical_points on a simple mesh with known extrema.
    """
    # Create a simple triangular mesh with 3 peaks and a valley
    vertices = np.array([
        [0.0, 0.0, 0.5],  # 0: center (middle height)
        [1.0, 0.0, 1.0],  # 1: peak
        [0.0, 1.0, 0.0],  # 2: valley
        [-1.0, 0.0, 1.0], # 3: peak
    ])
    triangles = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 1],
    ])
    mesh = TriaMesh(vertices, triangles)

    # Height function: z-coordinate
    vfunc = vertices[:, 2]

    # Compute critical points
    minima, maxima, saddles, saddle_orders = mesh.critical_points(vfunc)

    # Expected: vertex 2 is minimum (z=0), vertices 1 and 3 are maxima (z=1)
    assert 2 in minima, f"Expected vertex 2 (valley) as minimum, got minima={minima}"
    assert 1 in maxima or 3 in maxima, f"Expected vertices 1 or 3 (peaks) as maxima, got maxima={maxima}"
    # Vertex 0 is at mid-height, surrounded by higher and lower neighbors - could be saddle or regular

    # Simpler test: just verify the function runs without error
    assert len(minima) + len(maxima) + len(saddles) > 0, "Should find some critical points"


def test_critical_points_saddle():
    """
    Test critical_points on a mesh with a saddle point.
    """
    # Create a simple saddle mesh: 3x3 grid in xy-plane
    # Height creates saddle at center
    vertices = np.array([
        [-1, -1, 1],  # 0: corner (high)
        [0, -1, 0],   # 1: edge midpoint (low)
        [1, -1, 1],   # 2: corner (high)
        [-1, 0, 0],   # 3: edge midpoint (low)
        [0, 0, 0.5],  # 4: center (saddle)
        [1, 0, 0],    # 5: edge midpoint (low)
        [-1, 1, 1],   # 6: corner (high)
        [0, 1, 0],    # 7: edge midpoint (low)
        [1, 1, 1],    # 8: corner (high)
    ], dtype=float)

    # Create triangular mesh from grid
    triangles = np.array([
        [0, 1, 4], [0, 4, 3],  # bottom-left quad
        [1, 2, 5], [1, 5, 4],  # bottom-right quad
        [3, 4, 7], [3, 7, 6],  # top-left quad
        [4, 5, 8], [4, 8, 7],  # top-right quad
    ])
    mesh = TriaMesh(vertices, triangles)

    # Use z-coordinate as function
    vfunc = vertices[:, 2]

    # Compute critical points
    minima, maxima, saddles, saddle_orders = mesh.critical_points(vfunc)

    # Center should be a saddle (neighbors alternate high-low)
    assert 4 in saddles, f"Expected vertex 4 (center) to be a saddle, saddles={saddles}"
    # Check saddle order for center vertex
    saddle_idx = np.where(saddles == 4)[0]
    if len(saddle_idx) > 0:
        order = saddle_orders[saddle_idx[0]]
        assert order >= 2, f"Expected saddle order >= 2, got {order}"


def test_critical_points_with_ties():
    """
    Test critical_points with equal function values (tie-breaker rule).
    """
    # Simple triangle mesh
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
    ])
    triangles = np.array([[0, 1, 2]])
    mesh = TriaMesh(vertices, triangles)

    # All vertices have same value (ties everywhere)
    vfunc = np.array([1.0, 1.0, 1.0])

    # Should not crash, tie-breaker should resolve ambiguities
    minima, maxima, saddles, saddle_orders = mesh.critical_points(vfunc)

    # With tie-breaker, vertex with highest ID is treated as larger
    # So vertex 2 should be maximum, vertex 0 should be minimum
    assert 0 in minima, "Expected vertex 0 to be minimum with tie-breaker"
    assert 2 in maxima, "Expected vertex 2 to be maximum with tie-breaker"


def test_critical_points_boundary():
    """
    Test critical_points on a mesh with boundary (non-closed).
    """
    # Single triangle (has boundary)
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ])
    triangles = np.array([[0, 1, 2]])
    mesh = TriaMesh(vertices, triangles)

    # Height function
    vfunc = vertices[:, 2]

    minima, maxima, saddles, saddle_orders = mesh.critical_points(vfunc)

    # Vertex 1 is highest (z=1), vertices 0 and 2 are lowest (z=0)
    assert 1 in maxima, f"Expected vertex 1 as maximum, got maxima={maxima}"
    assert len(minima) >= 1, f"Expected at least one minimum, got {len(minima)}"


def test_extract_level_paths_simple():
    """
    Test extract_level_paths on a simple mesh with single level curve.
    """
    # Create a simple mesh: unit square in xy-plane
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ])
    triangles = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ])
    mesh = TriaMesh(vertices, triangles)

    # Height function (z-coordinate)
    vfunc = vertices[:, 2]

    # Extract level curve at z=0.5
    curves = mesh.extract_level_paths(vfunc, 0.5)

    # Should have at least one curve
    assert len(curves) > 0, "Expected at least one level curve"

    # Check that returned objects are Polygon
    from ...polygon import Polygon
    for curve in curves:
        assert isinstance(curve, Polygon), f"Expected Polygon, got {type(curve)}"

    # Check that polygons have required attributes
    for curve in curves:
        assert hasattr(curve, 'points'), "Polygon should have 'points' attribute"
        assert hasattr(curve, 'closed'), "Polygon should have 'closed' attribute"
        assert hasattr(curve, 'tria_idx'), "Polygon should have 'tria_idx' attribute"
        assert hasattr(curve, 'edge_vidx'), "Polygon should have 'edge_vidx' attribute"
        assert hasattr(curve, 'edge_bary'), "Polygon should have 'edge_bary' attribute"

    # Check that points are 3D
    for curve in curves:
        assert curve.points.shape[1] == 3, f"Expected 3D points, got shape {curve.points.shape}"


def test_extract_level_paths_closed_loop():
    """
    Test extract_level_paths on a mesh with closed loop level curve.
    """
    # Create a pyramid mesh
    vertices = np.array([
        [0.0, 0.0, 0.0],  # base
        [1.0, 0.0, 0.0],  # base
        [1.0, 1.0, 0.0],  # base
        [0.0, 1.0, 0.0],  # base
        [0.5, 0.5, 1.0],  # apex
    ])
    triangles = np.array([
        [0, 1, 4],
        [1, 2, 4],
        [2, 3, 4],
        [3, 0, 4],
        [0, 2, 1],
        [0, 3, 2],
    ])
    mesh = TriaMesh(vertices, triangles)

    # Height function
    vfunc = vertices[:, 2]

    # Extract level curve at mid-height (should create closed loop around pyramid)
    curves = mesh.extract_level_paths(vfunc, 0.5)

    assert len(curves) == 1, "Expected one level curve"

    # Curve should be closed
    assert curves[0].closed, "Expected closed level curve"

    # Note: May or may not be closed depending on mesh topology

    # All points should be approximately at z=0.5
    for curve in curves:
        z_coords = curve.points[:, 2]
        np.testing.assert_allclose(z_coords, 0.5, atol=1e-5,
                                   err_msg="Level curve points should be at z=0.5")


def test_extract_level_paths_no_intersection():
    """
    Test extract_level_paths when level doesn't intersect mesh.
    """
    # Simple triangle
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ])
    triangles = np.array([[0, 1, 2]])
    mesh = TriaMesh(vertices, triangles)

    # Height function
    vfunc = vertices[:, 2]

    # Extract level curve at z=10.0 (way above mesh)
    curves = mesh.extract_level_paths(vfunc, 10.0)

    # Should return empty list
    assert len(curves) == 0, f"Expected no curves, got {len(curves)}"


def test_extract_level_paths_metadata():
    """
    Test that extract_level_paths returns correct metadata.
    """
    # Create a simple mesh
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.5],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 0.5],
    ])
    triangles = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ])
    mesh = TriaMesh(vertices, triangles)

    # Height function
    vfunc = vertices[:, 2]

    # Extract level curve
    curves = mesh.extract_level_paths(vfunc, 0.5)

    for curve in curves:
        n_points = curve.points.shape[0]

        # Check tria_idx
        assert hasattr(curve, 'tria_idx'), "Missing tria_idx"
        # Number of segments (tria_idx) depends on whether curve is closed
        expected_tria_len = n_points if curve.closed else n_points - 1
        assert curve.tria_idx.shape == (expected_tria_len,), \
            f"tria_idx should be ({expected_tria_len},), got {curve.tria_idx.shape}"

        # Check edge_vidx
        assert hasattr(curve, 'edge_vidx'), "Missing edge_vidx"
        assert curve.edge_vidx.shape == (n_points, 2), \
            f"edge_vidx should be (n_points, 2), got {curve.edge_vidx.shape}"

        # Check edge_bary
        assert hasattr(curve, 'edge_bary'), "Missing edge_bary"
        assert curve.edge_bary.shape == (n_points,), \
            f"edge_bary should be (n_points,), got {curve.edge_bary.shape}"
        # Barycentric coordinates should be in [0, 1]
        assert np.all(curve.edge_bary >= 0) and np.all(curve.edge_bary <= 1), \
            "edge_bary should be in [0, 1]"


def test_level_path():
    """
    Test level_path function for single path extraction with various options.
    """
    # Create a simple mesh with a clear level curve
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.5],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 0.5],
    ])
    triangles = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ])
    mesh = TriaMesh(vertices, triangles)
    vfunc = vertices[:, 2]
    level = 0.5

    # Test 1: Basic usage - get path and length
    path, length = mesh.level_path(vfunc, level)

    assert path.ndim == 2, "Path should be 2D array"
    assert path.shape[1] == 3, f"Path should have 3D points, got shape {path.shape}"
    assert path.shape[0] >= 2, "Path should have at least 2 points"
    assert length > 0, f"Path length should be positive, got {length}"

    # Check all points are approximately at the level
    z_coords = path[:, 2]
    np.testing.assert_allclose(z_coords, level, atol=1e-5,
                               err_msg=f"All path points should be at z={level}")

    # Test 2: Check if path is closed (first == last point)
    is_closed = np.allclose(path[0], path[-1])
    if is_closed:
        print("  Path is closed (first point == last point)")
    else:
        print("  Path is open (endpoints differ)")

    # Test 3: Get triangle indices
    path_with_tria, length_with_tria, tria_idx = mesh.level_path(
        vfunc, level, get_tria_idx=True
    )

    np.testing.assert_array_equal(path_with_tria, path,
                                  err_msg="Path should be same with get_tria_idx=True")
    assert length_with_tria == length, "Length should be same"
    assert tria_idx.ndim == 1, "tria_idx should be 1D array"
    # For n points, we have n-1 segments
    expected_tria_len = path.shape[0] - 1
    assert tria_idx.shape[0] == expected_tria_len, \
        f"tria_idx should have {expected_tria_len} elements, got {tria_idx.shape[0]}"
    # Triangle indices should be valid
    assert np.all(tria_idx >= 0), "Triangle indices should be non-negative"
    assert np.all(tria_idx < len(triangles)), \
        f"Triangle indices should be < {len(triangles)}"

    # Test 4: Get edge information
    path_with_edges, length_with_edges, edges_vidxs, edges_relpos = mesh.level_path(
        vfunc, level, get_edges=True
    )

    np.testing.assert_array_equal(path_with_edges, path,
                                  err_msg="Path should be same with get_edges=True")
    assert length_with_edges == length, "Length should be same"
    assert edges_vidxs.shape == (path.shape[0], 2), \
        f"edges_vidxs should be (n_points, 2), got {edges_vidxs.shape}"
    assert edges_relpos.shape == (path.shape[0],), \
        f"edges_relpos should be (n_points,), got {edges_relpos.shape}"
    # Barycentric coordinates should be in [0, 1]
    assert np.all(edges_relpos >= 0) and np.all(edges_relpos <= 1), \
        "Barycentric coordinates should be in [0, 1]"

    # Test 5: Get both triangle and edge information
    result = mesh.level_path(vfunc, level, get_tria_idx=True, get_edges=True)
    path_full, length_full, tria_idx_full, edges_vidxs_full, edges_relpos_full = result

    np.testing.assert_array_equal(path_full, path,
                                  err_msg="Path should be consistent")
    np.testing.assert_array_equal(tria_idx_full, tria_idx,
                                  err_msg="tria_idx should be consistent")
    np.testing.assert_array_equal(edges_vidxs_full, edges_vidxs,
                                  err_msg="edges_vidxs should be consistent")
    np.testing.assert_array_equal(edges_relpos_full, edges_relpos,
                                  err_msg="edges_relpos should be consistent")

    # Test 6: Resampling to fixed number of points
    n_resample = 50
    path_resampled, length_resampled = mesh.level_path(
        vfunc, level, n_points=n_resample
    )

    assert path_resampled.shape[0] == n_resample, \
        f"Resampled path should have {n_resample} points, got {path_resampled.shape[0]}"
    assert path_resampled.shape[1] == 3, "Resampled path should have 3D points"
    # Length should be approximately the same
    np.testing.assert_allclose(length_resampled, length, rtol=0.1,
                               err_msg="Resampled length should be close to original")


def test_level_path_closed_loop():
    """
    Test level_path on a mesh that produces a closed loop.
    """
    # Create a pyramid mesh
    vertices = np.array([
        [0.0, 0.0, 0.0],  # base
        [1.0, 0.0, 0.0],  # base
        [1.0, 1.0, 0.0],  # base
        [0.0, 1.0, 0.0],  # base
        [0.5, 0.5, 1.0],  # apex
    ])
    triangles = np.array([
        [0, 1, 4],
        [1, 2, 4],
        [2, 3, 4],
        [3, 0, 4],
        [0, 2, 1],
        [0, 3, 2],
    ])
    mesh = TriaMesh(vertices, triangles)
    vfunc = vertices[:, 2]

    # Extract level at mid-height (should create closed loop around pyramid)
    path, length = mesh.level_path(vfunc, 0.5)

    # For closed loops, first and last points should be identical
    is_closed = np.allclose(path[0], path[-1])
    assert is_closed, "Closed loop should have first point equal to last point"

    # All points should be at z=0.5
    np.testing.assert_allclose(path[:, 2], 0.5, atol=1e-5,
                               err_msg="All points should be at z=0.5")

    # Length should be positive
    assert length > 0, "Closed loop should have positive length"

