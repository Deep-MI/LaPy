"""Tests for the polygon module."""

import numpy as np
import pytest

from ... import Polygon


class TestPolygonClass:
    """Test cases for the Polygon class."""

    def test_init_2d(self):
        """Test initialization with 2D points."""
        points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        poly = Polygon(points, closed=True)

        assert poly.is_2d(), "Should be 2D polygon"
        assert poly.is_closed(), "Should be closed"
        assert poly.n_points() == 4, "Should have 4 points"

    def test_init_3d(self):
        """Test initialization with 3D points."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        poly = Polygon(points, closed=False)

        assert not poly.is_2d(), "Should be 3D polygon"
        assert not poly.is_closed(), "Should be open"
        assert poly.n_points() == 3, "Should have 3 points"

    def test_init_empty_raises(self):
        """Test that empty points raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            Polygon(np.array([]))

    def test_init_invalid_dimensions_raises(self):
        """Test that invalid dimensions raise ValueError."""
        with pytest.raises(ValueError, match="2 or 3 coordinates"):
            Polygon(np.array([[0.0], [1.0]]))

    def test_length_open(self):
        """Test length computation for open polygon."""
        points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        poly = Polygon(points, closed=False)
        length = poly.length()

        expected = 2.0  # 1.0 + 1.0
        assert np.isclose(length, expected), f"Expected {expected}, got {length}"

    def test_length_closed(self):
        """Test length computation for closed polygon (square)."""
        square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        poly = Polygon(square, closed=True)
        length = poly.length()

        expected = 4.0
        assert np.isclose(length, expected), f"Expected {expected}, got {length}"

    def test_centroid_open(self):
        """Test centroid for open polygon."""
        points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        poly = Polygon(points, closed=False)
        centroid = poly.centroid()

        expected = np.array([0.5, 0.5])
        assert np.allclose(centroid, expected), f"Expected {expected}, got {centroid}"

    def test_centroid_closed_2d(self):
        """Test area-weighted centroid for closed 2D polygon."""
        # Unit square centered at origin
        square = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
        poly = Polygon(square, closed=True)
        centroid = poly.centroid()

        expected = np.array([0.0, 0.0])
        assert np.allclose(centroid, expected, atol=1e-10), \
            f"Expected {expected}, got {centroid}"

    def test_centroid_closed_3d(self):
        """Test centroid for closed 3D polygon (simple average)."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        poly = Polygon(points, closed=True)
        centroid = poly.centroid()

        expected = np.mean(points, axis=0)
        assert np.allclose(centroid, expected), f"Expected {expected}, got {centroid}"

    def test_area_closed_2d(self):
        """Test area computation for closed 2D polygon."""
        square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        poly = Polygon(square, closed=True)
        area = poly.area()

        expected = 1.0
        assert np.isclose(area, expected), f"Expected {expected}, got {area}"

    def test_area_open_raises(self):
        """Test that area computation raises for open polygon."""
        points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        poly = Polygon(points, closed=False)

        with pytest.raises(ValueError, match="closed polygon"):
            poly.area()

    def test_area_3d_raises(self):
        """Test that area computation raises for 3D polygon."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        poly = Polygon(points, closed=True)

        with pytest.raises(ValueError, match="2D polygons"):
            poly.area()

    def test_resample_open(self):
        """Test resampling an open polygon."""
        points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        poly = Polygon(points, closed=False)
        resampled = poly.resample(n_points=10, inplace=False)

        assert resampled.n_points() == 10, "Should have 10 points"
        assert not resampled.is_closed(), "Should remain open"
        assert np.allclose(resampled.get_points()[0], [0.0, 0.0]), \
            "First point should match"
        assert np.allclose(resampled.get_points()[-1], [1.0, 1.0]), \
            "Last point should match"

    def test_resample_closed(self):
        """Test resampling a closed polygon."""
        square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        poly = Polygon(square, closed=True)
        resampled = poly.resample(n_points=12, inplace=False)

        assert resampled.n_points() == 12, "Should have 12 points"
        assert resampled.is_closed(), "Should remain closed"
        # Check perimeter
        length = resampled.length()
        assert np.isclose(length, 4.0, atol=1e-10), \
            f"Perimeter should be 4.0, got {length}"

    def test_resample_inplace(self):
        """Test in-place resampling."""
        points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        poly = Polygon(points, closed=False)
        result = poly.resample(n_points=10, inplace=True)

        assert result is poly, "Should return self when inplace=True"
        assert poly.n_points() == 10, "Should have 10 points after in-place resample"

    def test_resample_iterative(self):
        """Test that iterative resampling improves uniformity."""
        # Create path with non-uniform spacing
        points = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 0.0], [1.0, 1.0]])
        poly = Polygon(points, closed=False)

        result1 = poly.resample(n_points=20, n_iter=1, inplace=False)
        result5 = poly.resample(n_points=20, n_iter=5, inplace=False)

        # Calculate spacing uniformity
        pts1 = result1.get_points()
        pts5 = result5.get_points()
        dists1 = np.sqrt(np.sum(np.diff(pts1, axis=0)**2, axis=1))
        dists5 = np.sqrt(np.sum(np.diff(pts5, axis=0)**2, axis=1))

        assert np.std(dists5) <= np.std(dists1), \
            "More iterations should improve uniformity"

    def test_smooth_laplace_open(self):
        """Test Laplace smoothing on open polygon."""
        # Create a jagged path
        points = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 0.0], [1.5, 0.5], [2.0, 0.0]])
        poly = Polygon(points, closed=False)
        smoothed = poly.smooth_laplace(n=3, lambda_=0.5, inplace=False)

        assert smoothed.n_points() == poly.n_points(), \
            "Should preserve number of points"
        # First and last points should remain unchanged for open polygon
        assert np.allclose(smoothed.get_points()[0], points[0]), \
            "First point should remain unchanged (up to numerical precision)"
        assert np.allclose(smoothed.get_points()[-1], points[-1]), \
            "Last point should remain unchanged (up to numerical precision)"

    def test_smooth_laplace_closed(self):
        """Test Laplace smoothing on closed polygon."""
        # Create a slightly irregular square
        square = np.array([
            [0.0, 0.0], [1.0, 0.1], [1.0, 1.0], [0.1, 1.0]
        ])
        poly = Polygon(square, closed=True)
        smoothed = poly.smooth_laplace(n=5, lambda_=0.5, inplace=False)

        assert smoothed.n_points() == 4, "Should preserve number of points"
        assert smoothed.is_closed(), "Should remain closed"

    def test_smooth_laplace_inplace(self):
        """Test in-place Laplace smoothing."""
        points = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 0.0]])
        poly = Polygon(points, closed=False)
        original_points = poly.get_points().copy()
        result = poly.smooth_laplace(n=1, lambda_=0.5, inplace=True)

        assert result is poly, "Should return self when inplace=True"
        assert not np.allclose(poly.get_points(), original_points), \
            "Points should be modified in-place"

    def test_smooth_taubin(self):
        """Test Taubin smoothing on polygon."""
        # Create a jagged path
        points = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 0.0], [1.5, 0.5], [2.0, 0.0]])
        poly = Polygon(points, closed=False)
        smoothed = poly.smooth_taubin(n=3, lambda_=0.5, mu=-0.53, inplace=False)

        assert smoothed.n_points() == poly.n_points(), \
            "Should preserve number of points"
        # Taubin should preserve overall shape better than pure Laplace
        assert not np.allclose(smoothed.get_points(), points), \
            "Points should be smoothed"

    def test_smooth_taubin_inplace(self):
        """Test in-place Taubin smoothing."""
        points = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 0.0]])
        poly = Polygon(points, closed=False)
        result = poly.smooth_taubin(n=1, inplace=True)

        assert result is poly, "Should return self when inplace=True"

    def test_smooth_two_point_open(self):
        """Test smoothing on two-point open polygon."""
        points = np.array([[0.0, 0.0], [1.0, 1.0]])
        poly = Polygon(points, closed=False)
        smoothed = poly.smooth_laplace(n=5, lambda_=0.5, inplace=False)

        # Both points should remain unchanged (boundary points are fixed)
        assert np.allclose(smoothed.get_points(), points), \
            "Two-point open polygon should not change when smoothed"

    def test_smooth_two_point_closed(self):
        """Test smoothing on two-point closed polygon."""
        points = np.array([[0.0, 0.0], [2.0, 2.0]])
        poly = Polygon(points, closed=True)
        smoothed = poly.smooth_laplace(n=5, lambda_=0.5, inplace=False)

        # Points should converge to their midpoint
        midpoint = np.mean(points, axis=0)
        assert np.allclose(smoothed.get_points(), midpoint, atol=1e-10), \
            "Two-point closed polygon should converge to midpoint"

    def test_smooth_three_point_open(self):
        """Test smoothing on three-point open polygon."""
        points = np.array([[0.0, 0.0], [0.5, 2.0], [1.0, 0.0]])
        poly = Polygon(points, closed=False)
        smoothed = poly.smooth_laplace(n=5, lambda_=0.5, inplace=False)

        # First and last points should remain fixed
        assert np.allclose(smoothed.get_points()[0], points[0]), \
            "First point should remain fixed in open polygon"
        assert np.allclose(smoothed.get_points()[-1], points[-1]), \
            "Last point should remain fixed in open polygon"
        # Middle point should be smoothed (moved toward average of neighbors)
        assert smoothed.get_points()[1, 1] < points[1, 1], \
            "Middle point should be smoothed downward"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
