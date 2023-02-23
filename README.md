# LaPy

LaPy is a package to compute spectral features (Laplace-Beltrami operator) on
tetrahedral and triangle meshes. It is written purely in python 3 without
sacrificing speed as almost all loops are vectorized, drawing upon efficient
and sparse mesh data structures. It is basically a port of the C++ ShapeDNA
project with extended differential geometry capabilities.

## Contents:

- TriaMesh: a class for triangle meshes offering various operations, such as
  fixing orientation, smoothing, curvature, boundary, quality, normals, and
  various efficient mesh datastructure (edges, adjacency matrices)
- TetMesh: a class for tetrahedral meshes (orientation, boundary ...)
- TriaIO, TetIO: for both tets and trias from off, vtk, etc. formats
- FuncIO: import/export vertex functions and eigenvector files
- Solver: a class for linear FEM computation (Laplace stiffness and mass
  matrix, fast and sparse eigenvalue solver, anisotropic Laplace, Poisson)
- DiffGeo: compute gradients, divergence, mean curvature flow, etc.
- Heat: for heat kernel and diffusion
- ShapeDNA: compute the ShapeDNA descriptor of surfaces and solids
- Plot: functions for interactive visualization (wrapping plotly)

## ToDo:

- Add unit tests and automated testing (e.g. travis)
- Add command line scripts for various functions

## Usage:

The LaPy package is a comprehensive collection of scripts, so we refer to the
'help' function and docstring of each module / function / class for usage info.
For example:

```
import lapy as lp
help(lp.TriaMesh)
help(lp.Solver)
```

In the `examples` subdirectory, we provide several Jupyter notebooks that
illustrate prototypical use cases of the toolbox.

## Installation:

Use the following code to download, build and install a package from this
repository into your local Python package directory:

`python3 -m pip install lapy`

Use the following code to install the dev package in editable mode to a location of
your choice:

`python3 -m pip install --user --src /my/preferred/location --editable git+https://github.com/Deep-MI/Lapy.git#egg=lapy`

Several functions, e.g. the Solver, require a sparse matrix decomposition, for which either the LU decomposition (from scipy sparse) or the faster Cholesky decomposition (from scikit-sparse cholmod) can be used. If the parameter flag use_cholmod is True, the code will try to import cholmod from the scikit-sparse package and will fall back to LU if the import fails. If you would like to use cholmod, you need to install scikit-sparse separately. It cannot be listed among LaPy's dependencies as that causes errors with pip. scikit-sparse requires numpy and scipy to be installed separately beforehand.

## References:

If you use this software for a publication please cite both these papers:

[1] Laplace-Beltrami spectra as 'Shape-DNA' of surfaces and solids. Reuter M, Wolter F-E, Peinecke N Computer-Aided Design. 2006;38(4):342-366. http://dx.doi.org/10.1016/j.cad.2005.10.011

[2] BrainPrint: a discriminative characterization of brain morphology. Wachinger C, Golland P, Kremen W, Fischl B, Reuter M Neuroimage. 2015;109:232-48. http://dx.doi.org/10.1016/j.neuroimage.2015.01.032 http://www.ncbi.nlm.nih.gov/pubmed/25613439

[1] introduces the FEM methods and the Laplace spectra for shape analysis, while [2] focusses on medical applications.

For Geodesics please cite:

[3] Crane K, Weischedel C, Wardetzky M. Geodesics in heat: A new approach to computing distance based on heat flow. ACM Transactions on Graphics. https://doi.org/10.1145/2516971.2516977

For non-singular mean curvature flow please cite:

[4] Kazhdan M, Solomon J, Ben-Chen M. 2012. Can Mean-Curvature Flow be Modified to be Non-singular? Comput. Graph. Forum 31, 5, 1745–1754.
https://doi.org/10.1111/j.1467-8659.2012.03179.x

We also invite you to check out our lab webpage at https://deep-mi.org
