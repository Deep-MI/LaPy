.. include:: ./links.inc

**LaPy**
========

.. toctree::
   :hidden:

   api/index
   generated/tutorials/index
   changes/index

LaPy is an open-source Python package for differential geometry on triangle
and tetrahedra meshes. It includes an FEM solver to estimate the Laplace,
Poisson or Heat equations. Further functionality includes the computations
of gradients, divergence, mean-curvature flow, conformal mappings, 
geodesics, ShapeDNA (Laplace spectra), and IO and plotting methods. 

LaPy is written purely in Python 3 without sacrificing speed as almost all
loops are vectorized, drawing upon efficient and sparse mesh data structures.


Install
-------

LaPy is available on `Pypi <project pypi_>`_ and on
`conda-forge <project conda_>`_.

.. tab-set::

    .. tab-item:: Pypi

        .. code-block:: bash

            pip install lapy

    .. tab-item:: Conda

        .. code-block:: bash

            conda install -c conda-forge lapy

    .. tab-item:: Source

        .. code-block:: bash

            pip install git+https://github.com/Deep-MI/LaPy

License
-------

``lapy`` is licensed under the `MIT license`_.
A full copy of the license can be found `on GitHub <project license_>`_.
