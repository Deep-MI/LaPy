"""
Stores string messages for the :mod:`lapy` modules.
"""
INVALID_DISTANCE_KEY: str = "Only euclidean distance is currently implemented."
INVALID_NORMALIZATION_METHOD: str = "{method} is not a valid eigenvalues normalization method! Value must be  one of: geometry, surface, or volume."
INVALID_ANISO_VALUE: str = "aniso should be scalar or tuple/array of length 2!"
INVALID_GEOMETRY_TYPE: str = "Unknown geometry type: {geometry}! Must be an instance of either TriaMesh or TetMesh."
CHOLESKY_SOLVER: str = (
    "Solver: Cholesky decomposition from scikit-sparse cholmod..."
)
LU_SOLVER: str = "Solver: spsolve (LU decomposition)..."
NOT_SQUARE: str = (
    "Square input matrices should have same number of rows and columns."
)
INVALID_POISSON_H: str = (
    "h should be either scalar or column vector with row num of A."
)
BAD_DTUP_NTUP: str = "dtup and ntup should contain index and data arrays of the same positive length, as well as unique values."
BAD_STIFFNESS_FORMAT: str = "A matrix needs to be sparse CSC or CSR."
