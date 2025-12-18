"""Functions for open and closed polygon paths.

This module provides utilities for resampling 2D and 3D polygon paths with
equidistant spacing. These functions are useful for processing level set
paths extracted from triangle meshes or any other polyline data.

Functions
---------
resample
    Resample a 2D or 3D polygon path to have a specified number of equidistant points.
"""

import numpy as np


def resample(path: np.ndarray, n_points: int = 100, n_iter: int = 1, closed: bool = False) -> np.ndarray:
    """Resample a 2D or 3D polygon path to have equidistant points.

    This function resamples a polygon path (open or closed) by creating
    n_points that are approximately equidistantly spaced along the cumulative
    Euclidean distance of the path. The resampling is performed using linear
    interpolation independently for each coordinate. Optionally, the resampling
    can be performed iteratively to achieve better numerical stability and more
    accurate equidistant spacing.

    Parameters
    ----------
    path : np.ndarray
        Array of shape (n, d) containing coordinates of the polygon vertices
        in order, where d is 2 or 3 for 2D (x, y) or 3D (x, y, z) paths.
        For closed polygons, the last point should not duplicate the first point.
    n_points : int, default=100
        Number of points in the resampled path. Must be at least 2.
    n_iter : int, default=1
        Number of resampling iterations to perform. The default value of 1
        performs a single resampling pass. Higher values (e.g., 3-5) provide
        better equidistant spacing but increase computation time. Must be at
        least 1. Iterative resampling is particularly useful for paths with
        highly variable point density or strong curvature.
    closed : bool, default=False
        If True, treats the path as a closed polygon and includes the segment
        from the last point back to the first point in the resampling. If False,
        treats the path as an open polyline. For closed paths, the output will
        not duplicate the first point at the end.

    Returns
    -------
    np.ndarray
        Array of shape (n_points, d) containing the resampled coordinates
        with approximately equidistant spacing along the path, where d
        matches the input dimensionality.

    Notes
    -----
    The function computes cumulative Euclidean distances between successive
    points and uses linear interpolation to place new points at equally spaced
    distance values.

    When n_iter > 1, the resampling is applied iteratively. Each iteration
    refines the point distribution by resampling the result from the previous
    iteration. The iterative process converges to an approximately equidistant
    point distribution, with each iteration making the spacing more uniform.

    For closed paths (closed=True), the total path length includes the distance
    from the last point back to the first point, and resampled points are
    distributed along this closed loop. The returned points form a closed path
    without duplicating the first point at the end.

    Examples
    --------
    >>> import numpy as np
    >>> from lapy import polygon
    >>> # Create a simple 3D open path and resample once
    >>> path_3d = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]])
    >>> resampled_3d = polygon.resample(path_3d, n_points=10)
    >>> resampled_3d.shape
    (10, 3)
    >>> # Create a 2D closed path (e.g., a square without duplicating first point)
    >>> square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    >>> resampled_square = polygon.resample(square, n_points=40, closed=True)
    >>> resampled_square.shape
    (40, 2)
    >>> # Iterative resampling with closed path
    >>> path_2d = np.array([[0, 0], [0.1, 0], [1, 0], [1, 1]])
    >>> resampled_2d = polygon.resample(path_2d, n_points=20, n_iter=5, closed=True)
    >>> resampled_2d.shape
    (20, 2)
    """
    def _resample_once(p: np.ndarray, n: int, is_closed: bool) -> np.ndarray:
        """Single resampling pass."""
        if is_closed:
            # For closed paths, append the first point to close the loop
            p_closed = np.vstack([p, p[0]])
            # Cumulative Euclidean distance between successive polygon points
            d = np.cumsum(np.r_[0, np.sqrt((np.diff(p_closed, axis=0) ** 2).sum(axis=1))])
            # Get linearly spaced points along the cumulative Euclidean distance
            # Exclude the endpoint (d.max()) to avoid duplicating the first point
            d_sampled = np.linspace(0, d.max(), n + 1)[:-1]
        else:
            # For open paths, use the original behavior
            d = np.cumsum(np.r_[0, np.sqrt((np.diff(p, axis=0) ** 2).sum(axis=1))])
            d_sampled = np.linspace(0, d.max(), n)
            p_closed = p

        # Interpolate each coordinate dimension
        n_dims = p.shape[1]
        return np.column_stack([
            np.interp(d_sampled, d, p_closed[:, i]) for i in range(n_dims)
        ])

    # Perform resampling n_iter times
    path_resampled = _resample_once(path, n_points, closed)
    for _ in range(n_iter - 1):
        path_resampled = _resample_once(path_resampled, n_points, closed)
    return path_resampled
