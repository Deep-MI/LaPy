.. include:: ./links.inc

**LaPy**
========

.. toctree::
   :hidden:

   api/index
   generated/tutorials/index
   changes/index

LaPy is a package to compute spectral features (Laplace-Beltrami operator) on
tetrahedral and triangle meshes. It is written purely in python 3 without
sacrificing speed as almost all loops are vectorized, drawing upon efficient
and sparse mesh data structures. It is basically a port of the C++ ShapeDNA
project with extended differential geometry capabilities.

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
