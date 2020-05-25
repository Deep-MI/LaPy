# LaPy

LaPy is a package to compute spectral features (Laplace-Beltrami operator) on
tetrahedral and triangle meshes. It is written purely in python 3 without
sacrificing speed as almost all loops are vectorized, drawing upon efficient
and sparse mesh datastructures. 

## Contents:

- TriaMesh: a class for triangle meshes offering various operations, such as 
  fixing orientation, smoothing, curvature, boundary, quality, normals, and 
  various efficient mesh datastructure (edges, adjacency matrices)
- TetMesh: a class for tetrahedral meshes (orientation, boundary ...)
- IO: for both tets and trias from off, vtk ..., as well as functions
- Solver: a class for linear FEM computation (Laplace stiffness and mass
  matrix, fast and sparse eigenvalue solver, anisotropic Laplace, Poission)
- DiffGeo: compute gradients, divergence, mean curvature flow, etc. 
- Heat: for heat kernel and diffusion
- Plot: functions for interactive visualization (wrapping plotly)

## ToDo

- Add and improve documentation
- Add unit tests and automated testing (e.g. travis)
- Add command line scripts for various functions
- Integrate ShapeDNA and BrainPrint

## Installation

Use the following code to download, build and install a package from this 
repository into your local Python package directory:

`pip3 install --user git+https://github.com/Deep-MI/LaPy.git#egg=lapy`

Use the following code to install the package in editable mode to a location of
your choice:

`pip3 install --user --src /my/preferred/location --editable git+https://github.com/Deep-MI/Lapy.git#egg=lapy`

## References

If you use this software for a publication please cite both these papers:

[1] Laplace-Beltrami spectra as 'Shape-DNA' of surfaces and solids. Reuter M, Wolter F-E, Peinecke N Computer-Aided Design. 2006;38(4):342-366. http://dx.doi.org/10.1016/j.cad.2005.10.011

[2] BrainPrint: a discriminative characterization of brain morphology. Wachinger C, Golland P, Kremen W, Fischl B, Reuter M Neuroimage. 2015;109:232-48. http://dx.doi.org/10.1016/j.neuroimage.2015.01.032 http://www.ncbi.nlm.nih.gov/pubmed/25613439

[1] introduces the FEM methods and the Laplace spectra for shape analysis, while [2] focusses on medical applications.

Also see our Lab webpage at https://deep-mi.org 

