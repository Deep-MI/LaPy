.. NOTE: we use cross-references to highlight new functions and classes.
   Please follow the examples below, so the changelog page will have a link to
   the function/class documentation.

.. NOTE: there are 3 separate sections for changes, based on type:
   - "Enhancements" for new features
   - "Bugs" for bug fixes
   - "API changes" for backward-incompatible changes

.. NOTE: You can use the :pr:`xx` and :issue:`xx` role to x-ref to a GitHub PR
   or issue from this project.

.. include:: ./authors.inc

.. _latest:

Version 1.0
===========

API changes
-----------

- Classes: TriaMesh, TetMesh, and Solver are still available directly at top level and imported directly from lapy.
- Mesh IO: mesh classes have been extended with IO class member functions and TriaIO and TetIO have been deprecated. Use read\_* and write\_* class members to load and write mehses, for example, TriaMesh.read_vtk() to import a VTK triangle mesh file. This simplifies IO greatly.
- Module names have been changed to comply with PEP8 conventions (lower case and underscore). For example, DiffGeo to diff_geo, FuncIO to func_io, and Plot to plot, etc. 

Bugs
----

- Fixed numpy deprecation issue in import_vfunc and import_ev functions.

Enhancements
------------

- Comply with the numpy convention for docstrings (by `Andreas Girodi`_, `Kersten Diers`_ and `Martin Reuter`_ in :pr:`19` and :pr:`21`)
- Add initial documentation build (by `Mathieu Scheltienne`_ in :pr:`22`)


Authors
-------

* `Andreas Girodi`_
* `Kersten Diers`_
* `Martin Reuter`_
* `Mathieu Scheltienne`_
