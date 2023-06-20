"""Functions for IO of Tetrahedra Meshes.

Should be called via the TetMesh member functions.
"""

import os.path

import numpy as np


def read_gmsh(filename):
    """Load GMSH tetrahedron mesh.

    Args:
        filename (str): Filename to load.

    Returns:
        TetMesh: Object of loaded  GMSH tetrahedron mesh.

    Raises:

    
    """
    extension = os.path.splitext(filename)[1]
    verbose = 1
    if verbose > 0:
        print("--> GMSH format         ... ")
    if extension != ".msh":
        print("[no .msh file] --> FAILED\n")
        return
    try:
        f = open(filename, "r")
    except IOError:
        print("[file not found or not readable]\n")
        return
    line = f.readline()
    if not line.startswith("$MeshFormat"):
        print("[$MeshFormat keyword not found] --> FAILED\n")
        f.close()
        return
    line = f.readline()
    larr = line.split()
    ver = float(larr[0])
    ftype = int(larr[1])
    datatype = int(larr[2])
    print(
        "Msh file ver ",
        ver,
        " , ftype ",
        ftype,
        " , datatype ",
        datatype,
        "\n",
    )
    if ftype != 0:
        print("[binary format not implemented] --> FAILED\n")
        f.close()
        return
    line = f.readline()
    if not line.startswith("$EndMeshFormat"):
        print("[$EndMeshFormat keyword not found] --> FAILED\n")
        f.close()
        return
    line = f.readline()
    if not line.startswith("$Nodes"):
        print("[$Nodes keyword not found] --> FAILED\n")
        f.close()
        return
    pnum = int(f.readline())
    # read (nodes X 4) matrix as chunk
    # drop first column
    v = np.fromfile(f, "float32", 4 * pnum, " ")
    v.shape = (pnum, 4)
    v = np.delete(v, 0, 1)
    line = f.readline()
    if not line.startswith("$EndNodes"):
        print("[$EndNodes keyword not found] --> FAILED\n")
        f.close()
        return
    line = f.readline()
    if not line.startswith("$Elements"):
        print("[$Elements keyword not found] --> FAILED\n")
        f.close()
        return
    tnum = int(f.readline())
    pos = f.tell()
    line = f.readline()
    f.seek(pos)
    larr = line.split()
    if int(larr[1]) != 4:
        print("larr: ", larr, "\n")
        print("[can only read tetras] --> FAILED\n")
        f.close()
        return
    # read (nodes X ?) matrix
    t = np.fromfile(f, "int", tnum * len(larr), " ")
    t.shape = (tnum, len(larr))
    t = np.delete(t, np.s_[0 : len(larr) - 4], 1)
    line = f.readline()
    if not line.startswith("$EndElements"):
        print("Line: ", line, " \n")
        print("[$EndElements keyword not found] --> FAILED\n")
        f.close()
        return
    f.close()
    print(" --> DONE ( V: " + str(v.shape[0]) + " , T: " + str(t.shape[0]) + " )\n")
    from . import TetMesh

    return TetMesh(v, t)


def read_vtk(filename):
    """Load VTK tetrahedron mesh.

    Args:
        filename (str): Filename to load.

    Returns:
        TetMesh: Object of loaded  VTK tetrahedron mesh.

    Raises:

    
    """
    verbose = 1
    if verbose > 0:
        print("--> VTK format         ... ")
    try:
        f = open(filename, "r")
    except IOError:
        print("[file not found or not readable]\n")
        return
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
        print("[ASCII keyword not found] --> FAILED\n")
        return
    # expect Dataset Polydata line after ASCII:
    line = f.readline()
    if not line.startswith("DATASET POLYDATA") and not line.startswith(
        "DATASET UNSTRUCTURED_GRID"
    ):
        print(
            "[read: "
            + line
            + " expected DATASET POLYDATA or DATASET UNSTRUCTURED_GRID] --> FAILED\n"
        )
        return
    # read number of points
    line = f.readline()
    larr = line.split()
    if larr[0] != "POINTS" or (larr[2] != "float" and larr[2] != "double"):
        print(
            "[read: "
            + line
            + " expected POINTS # float or POINTS # double ] --> FAILED\n"
        )
        return
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
            print(
                "[having: " + str(npt) + " data per tetra, expected 4+1] --> FAILED\n"
            )
            return
        t = np.fromfile(f, "int", ttnum, " ")
        t.shape = (tnum, 5)
        if t[tnum - 1][0] != 4:
            print("[can only read tetras] --> FAILED\n")
            return
        t = np.delete(t, 0, 1)
    else:
        print("[read: " + line + " expected POLYGONS or CELLS] --> FAILED\n")
        return
    f.close()
    print(" --> DONE ( V: " + str(v.shape[0]) + " , T: " + str(t.shape[0]) + " )\n")
    from . import TetMesh

    return TetMesh(v, t)


def write_vtk(tet, filename):
    """Save VTK file.

    Args:
        tet (TetMesh): Tetrahedron mesh to save.
        filename (str): Filename to save to.

    Returns:

    Raises:

    
    """
    # open file
    try:
        f = open(filename, "w")
    except IOError:
        print("[File " + filename + " not writable]")
        return
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
