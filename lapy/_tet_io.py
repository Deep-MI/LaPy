"""Functions for IO of Tetrahedra Meshes.

Should be called via the TetMesh member functions.
"""

import logging
import os.path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .tet_mesh import TetMesh

logger = logging.getLogger(__name__)

def read_gmsh(filename: str) -> "TetMesh":
    """Load GMSH tetrahedron mesh.

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
        If file format is invalid or binary format is encountered.
    """
    extension = os.path.splitext(filename)[1]
    verbose = 1
    if verbose > 0:
        logger.info("--> GMSH format         ... ")
    if extension != ".msh":
        msg = "[no .msh file] --> FAILED\n"
        logger.error(msg)
        raise ValueError(msg)
    try:
        f = open(filename)
    except OSError:
        logger.error("[file not found or not readable]")
        raise
    line = f.readline()
    if not line.startswith("$MeshFormat"):
        msg = "[$MeshFormat keyword not found] --> FAILED\n"
        logger.error(msg)
        f.close()
        raise ValueError(msg)
    line = f.readline()
    larr = line.split()
    ver = float(larr[0])
    ftype = int(larr[1])
    datatype = int(larr[2])
    logger.debug(
        "Msh file ver %s, ftype %s, datatype %s",
        ver,
        ftype,
        datatype,
    )
    if ftype != 0:
        msg = "[binary format not implemented] --> FAILED\n"
        logger.error(msg)
        f.close()
        raise ValueError(msg)
    line = f.readline()
    if not line.startswith("$EndMeshFormat"):
        msg = "[$EndMeshFormat keyword not found] --> FAILED\n"
        logger.error(msg)
        f.close()
        raise ValueError(msg)
    line = f.readline()
    if not line.startswith("$Nodes"):
        msg = "[$Nodes keyword not found] --> FAILED\n"
        logger.error(msg)
        f.close()
        raise ValueError(msg)
    pnum = int(f.readline())
    # read (nodes X 4) matrix as chunk
    # drop first column
    v = np.fromfile(f, "float32", 4 * pnum, " ")
    v.shape = (pnum, 4)
    v = np.delete(v, 0, 1)
    line = f.readline()
    if not line.startswith("$EndNodes"):
        msg = "[$EndNodes keyword not found] --> FAILED\n"
        logger.error(msg)
        f.close()
        raise ValueError(msg)
    line = f.readline()
    if not line.startswith("$Elements"):
        msg = "[$Elements keyword not found] --> FAILED\n"
        logger.error(msg)
        f.close()
        raise ValueError(msg)
    tnum = int(f.readline())
    pos = f.tell()
    line = f.readline()
    f.seek(pos)
    larr = line.split()
    if int(larr[1]) != 4:
        logger.debug("larr: %s", larr)
        msg = "[can only read tetras] --> FAILED\n"
        logger.error(msg)
        f.close()
        raise ValueError(msg)
    # read (nodes X ?) matrix
    t = np.fromfile(f, "int", tnum * len(larr), " ")
    t.shape = (tnum, len(larr))
    t = np.delete(t, np.s_[0 : len(larr) - 4], 1)
    line = f.readline()
    if not line.startswith("$EndElements"):
        logger.debug("Line: %s", line)
        msg = "[$EndElements keyword not found] --> FAILED\n"
        logger.error(msg)
        f.close()
        raise ValueError(msg)
    f.close()
    logger.info(" --> DONE ( V: %d , T: %d )", v.shape[0], t.shape[0])
    from . import TetMesh

    return TetMesh(v, t)


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
    v.shape = (pnum, 3)
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
        t.shape = (tnum, 5)
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
