"""Polygon class for open and closed polygon paths.

This module provides a Polygon class for processing 2D and 3D polygon paths with
various geometric operations including resampling, smoothing, and metric computations.
"""

import logging
import sys

import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)


class Polygon:
    """Class representing a polygon path (open or closed).

    This class handles 2D and 3D polygon paths with operations for resampling,
    smoothing, and computing geometric properties like length, centroid, and area.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (n, d) containing coordinates of polygon vertices
        in order, where d is 2 or 3 for 2D (x, y) or 3D (x, y, z) paths.
        For closed polygons, the last point should not duplicate the first point.
    closed : bool, default=False
        If True, treats the path as a closed polygon. If False, treats it as
        an open polyline.

    Attributes
    ----------
    points : np.ndarray
        Polygon vertex coordinates, shape (n_points, d).
    closed : bool
        Whether the polygon is closed or open.
    _is_2d : bool
        Internal flag indicating if polygon is 2D (True) or 3D (False).

    Raises
    ------
    ValueError
        If points array is empty.
        If points don't have 2 or 3 coordinates.

    Examples
    --------
    >>> import numpy as np
    >>> from lapy.polygon import Polygon
    >>> # Create a 2D closed polygon (square)
    >>> square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    >>> poly = Polygon(square, closed=True)
    >>> poly.is_2d()
    True
    >>> poly.length()
    4.0
    >>> # Create a 3D open path
    >>> path_3d = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]])
    >>> poly3d = Polygon(path_3d, closed=False)
    >>> poly3d.is_2d()
    False
    """

    def __init__(self, points: np.ndarray, closed: bool = False):
        self.points = np.array(points)
        self.closed = closed

        # Validate non-empty polygon
        if self.points.size == 0:
            raise ValueError("Polygon has no points (empty)")

        # Transpose if necessary
        if self.points.shape[0] < self.points.shape[1]:
            self.points = self.points.T

        # Support both 2D and 3D points
        if self.points.shape[1] == 2:
            self._is_2d = True
        elif self.points.shape[1] == 3:
            self._is_2d = False
        else:
            raise ValueError("Points should have 2 or 3 coordinates")

    def is_2d(self) -> bool:
        """Check if the polygon is 2D.

        Returns
        -------
        bool
            True if polygon is 2D, False if 3D.
        """
        return self._is_2d

    def is_closed(self) -> bool:
        """Check if the polygon is closed.

        Returns
        -------
        bool
            True if polygon is closed, False if open.
        """
        return self.closed

    def n_points(self) -> int:
        """Get number of points in polygon.

        Returns
        -------
        int
            Number of points.
        """
        return self.points.shape[0]

    def get_points(self) -> np.ndarray:
        """Get polygon points.

        Returns
        -------
        np.ndarray
            Point array of shape (n, 2) or (n, 3).
        """
        return self.points

    def length(self) -> float:
        """Compute total length of polygon path.

        For closed polygons, includes the segment from last to first point.

        Returns
        -------
        float
            Total path length.
        """
        if self.closed:
            points_closed = np.vstack([self.points, self.points[0]])
            edge_vecs = np.diff(points_closed, axis=0)
        else:
            edge_vecs = np.diff(self.points, axis=0)

        edge_lens = np.sqrt((edge_vecs**2).sum(axis=1))
        return edge_lens.sum()

    def centroid(self) -> np.ndarray:
        """Compute centroid of polygon.

        For open polygons or closed 3D polygons, returns the simple arithmetic mean
        of all vertex coordinates.

        For closed 2D polygons, returns the area-weighted centroid (geometric center
        of mass). The area weighting accounts for the shape's geometry, ensuring the
        centroid lies at the balance point of the polygon as if it were a uniform
        plate. This differs from the simple average of vertices, which would not
        account for how vertices are distributed around the polygon's boundary.

        Returns
        -------
        np.ndarray
            Centroid coordinates, shape (2,) or (3,).

        Notes
        -----
        For closed 2D polygons, uses the standard formula:
        C_x = (1 / (6*A)) * sum((x_i + x_{i+1}) * (x_i * y_{i+1} - x_{i+1} * y_i))
        C_y = (1 / (6*A)) * sum((y_i + y_{i+1}) * (x_i * y_{i+1} - x_{i+1} * y_i))
        where A is the polygon area.
        """
        if not self.closed or not self._is_2d:
            # Simple average for open polygons or 3D closed polygons
            return np.mean(self.points, axis=0)

        # Area-weighted centroid for closed 2D polygons
        x = self.points[:, 0]
        y = self.points[:, 1]
        # Append first point to close polygon
        x_closed = np.append(x, x[0])
        y_closed = np.append(y, y[0])
        # Shoelace formula components
        cross = x_closed[:-1] * y_closed[1:] - x_closed[1:] * y_closed[:-1]
        area = 0.5 * np.abs(cross.sum())

        if area < sys.float_info.epsilon:
            # Degenerate case: zero area
            return np.mean(self.points, axis=0)

        cx = np.sum((x_closed[:-1] + x_closed[1:]) * cross) / (6.0 * area)
        cy = np.sum((y_closed[:-1] + y_closed[1:]) * cross) / (6.0 * area)
        return np.array([cx, cy])

    def area(self) -> float:
        """Compute area enclosed by closed 2D polygon.

        Uses the shoelace formula. Only valid for closed 2D polygons.

        Returns
        -------
        float
            Enclosed area (always positive).

        Raises
        ------
        ValueError
            If polygon is not closed or not 2D.
        """
        if not self.closed:
            raise ValueError("Area computation requires closed polygon.")
        if not self._is_2d:
            raise ValueError("Area computation only valid for 2D polygons.")

        x = self.points[:, 0]
        y = self.points[:, 1]
        # Append first point to close polygon
        x_closed = np.append(x, x[0])
        y_closed = np.append(y, y[0])
        # Shoelace formula
        area = 0.5 * np.abs(
            np.sum(x_closed[:-1] * y_closed[1:] - x_closed[1:] * y_closed[:-1])
        )
        return area

    def resample(
        self, n_points: int = 100, n_iter: int = 1, inplace: bool = False
    ) -> "Polygon":
        """Resample polygon to have equidistant points.

        Creates n_points that are approximately equidistantly spaced along
        the cumulative Euclidean distance. Uses linear interpolation.

        Parameters
        ----------
        n_points : int, default=100
            Number of points in resampled polygon. Must be at least 2.
        n_iter : int, default=1
            Number of resampling iterations. Higher values (e.g., 3-5) provide
            better equidistant spacing. Must be at least 1.
        inplace : bool, default=False
            If True, modify this polygon in-place. If False, return new polygon.

        Returns
        -------
        Polygon
            Resampled polygon. Returns self if inplace=True, new instance otherwise.
        """
        def _resample_once(p: np.ndarray, n: int, is_closed: bool) -> np.ndarray:
            """Single resampling pass."""
            if is_closed:
                p_closed = np.vstack([p, p[0]])
                d = np.cumsum(
                    np.r_[0, np.sqrt((np.diff(p_closed, axis=0) ** 2).sum(axis=1))]
                )
                d_sampled = np.linspace(0, d.max(), n + 1)[:-1]
            else:
                d = np.cumsum(np.r_[0, np.sqrt((np.diff(p, axis=0) ** 2).sum(axis=1))])
                d_sampled = np.linspace(0, d.max(), n)
                p_closed = p

            n_dims = p.shape[1]
            return np.column_stack(
                [np.interp(d_sampled, d, p_closed[:, i]) for i in range(n_dims)]
            )

        # Perform resampling n_iter times
        points_resampled = _resample_once(self.points, n_points, self.closed)
        for _ in range(n_iter - 1):
            points_resampled = _resample_once(points_resampled, n_points, self.closed)

        if inplace:
            self.points = points_resampled
            return self
        else:
            return Polygon(points_resampled, closed=self.closed)

    def _construct_smoothing_matrix(self) -> sparse.csc_matrix:
        """Construct smoothing matrix for Laplace smoothing.

        Creates a row-stochastic matrix where each point is connected to
        its neighbors (previous and next point). For open polygons, boundary
        points (first and last) are kept fixed.

        The method handles polygons of any size:

        - For open polygons with 2 points: Both boundary points remain fixed
          (identity matrix), so smoothing has no effect.
        - For open polygons with 3+ points: Boundary points are fixed, interior
          points are averaged with their neighbors.
        - For closed polygons with 2 points: Each point is averaged with its
          neighbor, causing them to converge to their midpoint.
        - For closed polygons with 3+ points: All points are averaged with
          their neighbors in a circular manner.

        Returns
        -------
        scipy.sparse.csc_matrix
            Sparse smoothing matrix.
        """
        n = self.points.shape[0]

        if self.closed:
            # For closed polygons, connect last to first
            i = np.arange(n)
            j_prev = np.roll(np.arange(n), 1)
            j_next = np.roll(np.arange(n), -1)

            # Create adjacency with neighbors
            i_all = np.concatenate([i, i])
            j_all = np.concatenate([j_prev, j_next])
            data = np.ones(len(i_all))

            adj = sparse.csc_matrix((data, (i_all, j_all)), shape=(n, n))

            # Normalize rows to create stochastic matrix
            row_sum = np.array(adj.sum(axis=1)).ravel()
            row_sum[row_sum == 0] = 1.0  # Avoid division by zero
            adj = adj.multiply(1.0 / row_sum[:, np.newaxis])
        else:
            # For open polygons, use LIL format for easier construction
            adj = sparse.lil_matrix((n, n))

            # Set identity for boundary points (they stay fixed)
            adj[0, 0] = 1.0
            adj[n - 1, n - 1] = 1.0

            # For interior points, connect to neighbors
            for i in range(1, n - 1):
                adj[i, i - 1] = 0.5
                adj[i, i + 1] = 0.5

            # Convert to CSC for efficient operations
            adj = adj.tocsc()

        return adj

    def smooth_laplace(
        self,
        n: int = 1,
        lambda_: float = 0.5,
        inplace: bool = False,
    ) -> "Polygon":
        """Smooth polygon using Laplace smoothing.

        Applies iterative smoothing: p_new = (1-lambda)*p + lambda * M*p
        where M is the neighbor-averaging matrix.

        Parameters
        ----------
        n : int, default=1
            Number of smoothing iterations.
        lambda_ : float, default=0.5
            Diffusion speed parameter in range [0, 1].
        inplace : bool, default=False
            If True, modify this polygon in-place. If False, return new polygon.

        Returns
        -------
        Polygon
            Smoothed polygon. Returns self if inplace=True, new instance otherwise.
        """
        mat = self._construct_smoothing_matrix()
        points_smooth = self.points.copy()

        for _ in range(n):
            points_smooth = (1.0 - lambda_) * points_smooth + lambda_ * mat.dot(
                points_smooth
            )

        if inplace:
            self.points = points_smooth
            return self
        else:
            return Polygon(points_smooth, closed=self.closed)

    def smooth_taubin(
        self,
        n: int = 1,
        lambda_: float = 0.5,
        mu: float = -0.53,
        inplace: bool = False,
    ) -> "Polygon":
        """Smooth polygon using Taubin smoothing.

        Alternates between shrinking (positive lambda) and expanding (negative mu)
        steps to reduce shrinkage while smoothing.

        Parameters
        ----------
        n : int, default=1
            Number of smoothing iterations.
        lambda_ : float, default=0.5
            Positive diffusion parameter for shrinking step.
        mu : float, default=-0.53
            Negative diffusion parameter for expanding step.
        inplace : bool, default=False
            If True, modify this polygon in-place. If False, return new polygon.

        Returns
        -------
        Polygon
            Smoothed polygon. Returns self if inplace=True, new instance otherwise.
        """
        mat = self._construct_smoothing_matrix()
        points_smooth = self.points.copy()

        for _ in range(n):
            # Lambda step (shrinking)
            points_smooth = (1.0 - lambda_) * points_smooth + lambda_ * mat.dot(
                points_smooth
            )
            # Mu step (expanding)
            points_smooth = (1.0 - mu) * points_smooth + mu * mat.dot(points_smooth)

        if inplace:
            self.points = points_smooth
            return self
        else:
            return Polygon(points_smooth, closed=self.closed)

