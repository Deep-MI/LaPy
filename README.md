[![PyPI version](https://badge.fury.io/py/lapy.svg)](https://pypi.org/project/lapy/)
# LaPy

LaPy is an open-source Python library for differential geometry and finite-element computations on triangle and tetrahedral meshes. It provides mesh data structures and fast, vectorized algorithms to compute differential operators, curvature flows, spectral descriptors and to solve PDEs such as Laplace, Poisson and Heat equations on surfaces and volumes.

Key design goals:
- Pure Python 3 implementation with heavy use of NumPy and SciPy for performance.
- Sparse, memory-efficient mesh data structures and vectorized algorithms.
- Utilities for IO, visualization, and common geometry processing tasks.

## Table of contents
- [Features](#features)
- [Quick start](#quick-start)
- [Installation](#installation)
- [Solver backends](#solver-backends)
- [Examples and documentation](#examples-and-documentation)
- [Development and testing](#development-and-testing)
- [References](#references)

## Features
- `TriaMesh`: triangle mesh class with orientation checking, boundary handling, normals, smoothing and quality metrics; efficient edge and adjacency representations; IO for OFF, VTK and related formats.
- `TetMesh`: tetrahedral mesh utilities, boundary handling and IO.
- `Solver`: FEM routines producing stiffness and mass matrices, sparse eigenvalue solvers, Poisson and heat equation solvers, and support for anisotropic operators.
- `diffgeo`: gradient, divergence, mean-curvature flow and related differential operators.
- `heat`: heat kernel and diffusion utilities, geodesics via heat method.
- `shapedna`: compute ShapeDNA (Laplace spectra) for shape descriptors.
- `conformal`: conformal mapping methods for genus-0 surfaces.
- `io`: read/write vertex functions and eigenvector files.
- `plot`: lightweight Plotly wrappers for interactive visualization.

## Quick start

Install the released package:
```
python3 -m pip install lapy
```

Import and inspect main classes:
```
import lapy as lp
help(lp.TriaMesh)
help(lp.Solver)
```

A minimal example (compute eigenpairs of a triangular mesh):
```
import lapy as lp
mesh = lp.TriaMesh.from_off('examples/data/sample.off')  # or other supported reader
solver = lp.Solver(mesh)
vals, vecs = solver.eigensystem(k=20)  # compute 20 smallest nontrivial eigenpairs
```

## Installation

Install the development version to a chosen source location:
```
python3 -m pip install --user --src /my/preferred/location --editable git+https://github.com/Deep-MI/Lapy.git#egg=lapy
```

### Dependencies and optional backends
- Core: Python 3, NumPy, SciPy.
- Optional (recommended): scikit-sparse (for CHOLMOD) to accelerate Cholesky sparse solves.
Note: CHOLMOD (via scikit-sparse) is not currently installable via plain `pip` on all platforms; use `conda` when possible. If `use_cholmod=True` is requested, LaPy attempts to import CHOLMOD and will raise if it is unavailable. Install ordering: install `numpy` and `scipy` first, then `scikit-sparse`.

## Solver backends
- Default: SciPy sparse LU/QR routines.
- Optional: CHOLMOD (faster for symmetric positive definite systems). Toggle with `use_cholmod=True` when constructing `Solver`.

## Examples and documentation
- Example Jupyter notebooks are available in the `examples` directory demonstrating common workflows (mesh IO, curvature, diffusion, ShapeDNA, geodesics).
- Full API documentation: https://deep-mi.org/LaPy

## Development and testing
- The project includes unit tests and example notebooks. Use the development installation command above and run tests with your preferred test runner (e.g., `pytest`).
- Contributions and issues are welcome via the repository issue tracker.

## References
If you use LaPy in publications, please cite:

1. Reuter M, Wolter F-E, Peinecke N. "Laplace-Beltrami spectra as 'Shape-DNA' of surfaces and solids." Computer-Aided Design. 2006;38(4):342–366. http://dx.doi.org/10.1016/j.cad.2005.10.011

2. Wachinger C, Golland P, Kremen W, Fischl B, Reuter M. "BrainPrint: a discriminative characterization of brain morphology." NeuroImage. 2015;109:232–248. http://dx.doi.org/10.1016/j.neuroimage.2015.01.032 http://www.ncbi.nlm.nih.gov/pubmed/25613439

Additional algorithmic sources:
- Crane K, Weischedel C, Wardetzky M. "Geodesics in heat." ACM Trans. Graph. (use for heat-based geodesics) https://doi.org/10.1145/2516971.2516977
- Kazhdan M, Solomon J, Ben-Chen M. "Can Mean-Curvature Flow be Modified to be Non-singular?" Comput. Graph. Forum (2012) (for non-singular mean curvature flow) https://doi.org/10.1111/j.1467-8659.2012.03179.x
- Choi PT, Lam KC, Lui LM. "FLASH: Fast Landmark Aligned Spherical Harmonic Parameterization for Genus-0 Closed Brain Surfaces." SIAM J. Imaging Sci. (for conformal mapping methods) https://doi.org/10.1137/130950008

## Website
- Lab / project page: https://deep-mi.org

## License
- See the repository `LICENSE` file for license terms.

## Contact
- Report issues or feature requests via the repository issue tracker on GitHub.
