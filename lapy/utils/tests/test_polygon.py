"""Tests for the polygon module."""

import numpy as np
import pytest

from ... import polygon


class TestResample:
    """Test cases for the polygon.resample function."""

    def test_resample_open_path(self):
        """Test resampling a 2D open path."""
        # Create a simple L-shaped path
        path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        resampled = polygon.resample(path, n_points=10, closed=False)

        assert resampled.shape == (10, 2), "Output shape should be (10, 2)"
        assert np.allclose(resampled[0], [0.0, 0.0]), "First point should match"
        assert np.allclose(resampled[-1], [1.0, 1.0]), "Last point should match"

    def test_resample_closed_path(self):
        """Test resampling a 2D closed path (square)."""
        # Create a unit square (4 vertices, no duplication)
        square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        resampled = polygon.resample(square, n_points=12, closed=True)

        assert resampled.shape == (12, 2), "Output shape should be (12, 2)"

        # Check that path covers the full perimeter
        resampled_with_wrap = np.vstack([resampled, resampled[0]])
        dists = np.sqrt(np.sum(np.diff(resampled_with_wrap, axis=0)**2, axis=1))
        total_length = np.sum(dists)

        assert np.allclose(total_length, 4.0, atol=1e-10), \
            f"Closed square should have perimeter 4.0, got {total_length}"

    def test_resample_closed_vs_open(self):
        """Test that closed parameter makes a difference."""
        square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

        resampled_open = polygon.resample(square, n_points=12, closed=False)
        resampled_closed = polygon.resample(square, n_points=12, closed=True)

        # Calculate path lengths
        dists_open = np.sqrt(np.sum(np.diff(resampled_open, axis=0)**2, axis=1))
        total_open = np.sum(dists_open)

        resampled_closed_wrap = np.vstack([resampled_closed, resampled_closed[0]])
        dists_closed = np.sqrt(np.sum(np.diff(resampled_closed_wrap, axis=0)**2, axis=1))
        total_closed = np.sum(dists_closed)

        # Closed path should cover full perimeter, open path should not
        assert total_open < total_closed, "Open path should be shorter"
        assert np.allclose(total_closed, 4.0, atol=1e-10), "Closed perimeter should be 4.0"

    def test_resample_iterative(self):
        """Test that iterative resampling improves spacing uniformity."""
        # Create a path with non-uniform spacing
        path = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 0.0], [1.0, 1.0]])

        result_single = polygon.resample(path, n_points=20, n_iter=1, closed=False)
        result_multiple = polygon.resample(path, n_points=20, n_iter=5, closed=False)

        # Calculate spacing uniformity
        dists_single = np.sqrt(np.sum(np.diff(result_single, axis=0)**2, axis=1))
        dists_multiple = np.sqrt(np.sum(np.diff(result_multiple, axis=0)**2, axis=1))

        # Multiple iterations should have more uniform spacing (lower std dev)
        assert np.std(dists_multiple) <= np.std(dists_single), \
            "Iterative resampling should improve uniformity"



if __name__ == "__main__":
    pytest.main([__file__, "-v"])

