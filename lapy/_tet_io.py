"""Functions for IO of Tetrahedra Meshes.

Should be called via the TetMesh member functions.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .tet_mesh import TetMesh

logger = logging.getLogger(__name__)

def read_gmsh(filename: str) -> "TetMesh":
    """Load GMSH tetrahedron mesh, MSH 2 ASCII format.

    Parameters
    ----------
    filename : str
        Filename to load.

    Returns
    -------
    TetMesh
        Object of loaded GMSH tetrahedron mesh.

    Raises
    ------
    OSError
        If file is not found or not readable.
    ValueError
        If the extension is not ``.msh``, the file is binary or not version 2,
        a section is malformed, or the file holds no tetrahedra.
    """
    from . import TetMesh
    from ._gmsh_io import read_gmsh as _read

    points, cells = _read(filename, want="tetra")
    logger.info(
        " --> DONE ( V: %d , T: %d )", points.shape[0], cells["tetra"].shape[0]
    )
    return TetMesh(points, cells["tetra"])

def read_vtk(filename: str) -> "TetMesh":
    """Load VTK tetrahedron mesh.

    Parameters
    ----------
    filename : str
        Filename to load.

    Returns
    -------
    TetMesh
        Object of loaded VTK tetrahedron mesh.

    Raises
    ------
    OSError
        If file is not found or not readable.
    ValueError
        If ASCII keyword is not found.
        If DATASET POLYDATA or DATASET UNSTRUCTURED_GRID is not found.
        If POINTS keyword is malformed.
        If file does not contain tetrahedra data.
    """
    verbose = 1
    if verbose > 0:
        logger.info("--> VTK format         ... ")
    try:
        f = open(filename)
    except OSError:
        logger.error("[file not found or not readable]")
        raise
    # skip comments
    line = f.readline()
    while line[0] == "#":
        line = f.readline()
    # search for ASCII keyword in first 5 lines:
    count = 0
    while count < 5 and not line.startswith("ASCII"):
        line = f.readline()
        # print line
        count = count + 1
    if not line.startswith("ASCII"):
        msg = "[ASCII keyword not found] --> FAILED\n"
        logger.error(msg)
        raise ValueError(msg)
    # expect Dataset Polydata line after ASCII:
    line = f.readline()
    if not line.startswith("DATASET POLYDATA") and not line.startswith(
        "DATASET UNSTRUCTURED_GRID"
    ):
        msg = (
            f"[read: {line} expected DATASET POLYDATA or DATASET UNSTRUCTURED_GRID]"
            f" --> FAILED\n"
        )
        logger.error(msg)
        raise ValueError(msg)
    # read number of points
    line = f.readline()
    larr = line.split()
    if larr[0] != "POINTS" or (larr[2] != "float" and larr[2] != "double"):
        msg = f"[read: {line} expected POINTS # float or POINTS # double ] --> FAILED\n"
        logger.error(msg)
        raise ValueError(msg)
    pnum = int(larr[1])
    # read points as chunk
    v = np.fromfile(f, "float32", 3 * pnum, " ")
    v = v.reshape(pnum, 3)
    # expect polygon or tria_strip line
    line = f.readline()
    larr = line.split()
    if larr[0] == "POLYGONS" or larr[0] == "CELLS":
        tnum = int(larr[1])
        ttnum = int(larr[2])
        npt = float(ttnum) / tnum
        if npt != 5.0:
            msg = f"[having: {npt} data per tetra, expected 4+1] --> FAILED\n"
            logger.error(msg)
            raise ValueError(msg)
        t = np.fromfile(f, "int", ttnum, " ")
        t = t.reshape(tnum, 5)
        if t[tnum - 1][0] != 4:
            msg = "[can only read tetras] --> FAILED\n"
            logger.error(msg)
            raise ValueError(msg)
        t = np.delete(t, 0, 1)
    else:
        msg = f"[read: {line} expected POLYGONS or CELLS] --> FAILED\n"
        logger.error(msg)
        raise ValueError(msg)
    f.close()
    logger.info(" --> DONE ( V: %d , T: %d )", v.shape[0], t.shape[0])
    from . import TetMesh

    return TetMesh(v, t)


def write_vtk(tet: "TetMesh", filename: str) -> None:
    """Save VTK file.

    Parameters
    ----------
    tet : TetMesh
        Tetrahedron mesh to save.
    filename : str
        Filename to save to.

    Raises
    ------
    OSError
        If file is not writable.
    """
    # open file
    try:
        f = open(filename, "w")
    except OSError:
        logger.error("[File %s not writable]", filename)
        raise
    # check data structure
    # ...
    # Write
    f.write("# vtk DataFile Version 1.0\n")
    f.write("vtk output\n")
    f.write("ASCII\n")
    f.write("DATASET POLYDATA\n")
    f.write("POINTS " + str(np.shape(tet.v)[0]) + " float\n")
    for i in range(np.shape(tet.v)[0]):
        f.write(" ".join(map(str, tet.v[i, :])))
        f.write("\n")
    f.write(
        "POLYGONS " + str(np.shape(tet.t)[0]) + " " + str(5 * np.shape(tet.t)[0]) + "\n"
    )
    for i in range(np.shape(tet.t)[0]):
        f.write(" ".join(map(str, np.append(4, tet.t[i, :]))))
        f.write("\n")
    f.close()
