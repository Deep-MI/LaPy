#!/usr/bin/env python
# -*- coding: latin-1 -*-


import numpy as np

from .TriaMesh import TriaMesh


def import_fssurf(infile):
    """
    Load triangle mesh from FreeSurfer surface geometry file
    :return:    TriaMesh
    """
    verbose = 1
    if verbose > 0:
        print("--> FS Surf format     ... ")
    try:
        # here we use our copy to also support surfaces from dev (and maybe v7*?)
        # these have an empty line and mess up Nibabel
        # once this is fixed in nibabel we can switch back
        from .read_geometry import read_geometry

        surf = read_geometry(infile, read_metadata=True)
    except IOError:
        print("[file not found or not readable]\n")
        return

    return TriaMesh(surf[0], surf[1], fsinfo=surf[2])


def import_off(infile):
    """
    Load triangle mesh from OFF txt file
    :return:    TriaMesh
    """
    verbose = 1
    if verbose > 0:
        print("--> OFF format         ... ")
    try:
        f = open(infile, "r")
    except IOError:
        print("[file not found or not readable]\n")
        return
    line = f.readline()
    while line[0] == "#":
        line = f.readline()
    if not line.startswith("OFF"):
        print("[OFF keyword not found] --> FAILED\n")
        f.close()
        return
    # expect tria and vertex sizes after OFF line:
    line = f.readline()
    larr = line.split()
    pnum = int(larr[0])
    tnum = int(larr[1])
    # print(" tnum: {} pnum: {}".format(tnum,pnum))
    # read points as chunch
    v = np.fromfile(f, "float32", 3 * pnum, " ")
    v.shape = (pnum, 3)
    # read trias as chunch
    t = np.fromfile(f, "int", 4 * tnum, " ")
    t.shape = (tnum, 4)
    # print(" t0: {} ".format(t[0, :]))
    # make sure first column is equal to 3 (trias)
    # max0 = np.amax(t[:, 0])
    # print(" max: {}".format(max0))
    if np.amax(t[:, 0]) != 3:
        print("[no triangle data] --> FAILED\n")
        f.close()
        return
    t = t[:, 1:]
    f.close()
    print(" --> DONE ( V: " + str(v.shape[0]) + " , T: " + str(t.shape[0]) + " )\n")
    return TriaMesh(v, t)


def import_vtk(infile):
    """
    Load triangle mesh from VTK txt file
    :return:    TriaMesh
    """
    verbose = 1
    if verbose > 0:
        print("--> VTK format         ... ")
    try:
        f = open(infile, "r")
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
        if npt != 4.0:
            print(
                "[having: "
                + str(npt)
                + " data per tria, expected trias 3+1] --> FAILED\n"
            )
            return
        t = np.fromfile(f, "int", ttnum, " ")
        t.shape = (tnum, 4)
        if t[tnum - 1][0] != 3:
            print("[can only read triangles] --> FAILED\n")
            return
        t = np.delete(t, 0, 1)
    elif larr[0] == "TRIANGLE_STRIPS":
        tnum = int(larr[1])
        # ttnum = int(larr[2])
        tt = []
        for i in range(tnum):
            larr = f.readline().split()
            if len(larr) == 0:
                print("[error reading triangle strip (i)] --> FAILED\n")
                return
            n = int(larr[0])
            if len(larr) != n + 1:
                print("[error reading triangle strip (ii)] --> FAILED\n")
                return
            # create triangles from strip
            # note that larr tria info starts at index 1
            for ii in range(2, n):
                if ii % 2 == 0:
                    tria = [larr[ii - 1], larr[ii], larr[ii + 1]]
                else:
                    tria = [larr[ii], larr[ii - 1], larr[ii + 1]]
                tt.append(tria)
        t = np.array(tt)
    else:
        print("[read: " + line + " expected POLYGONS or TRIANGLE_STRIPS] --> FAILED\n")
        return
    f.close()
    print(" --> DONE ( V: " + str(v.shape[0]) + " , T: " + str(t.shape[0]) + " )\n")
    return TriaMesh(v, t)


def import_gmsh(infile):
    """
    Load GMSH tetra mesh ASCII Format
    http://geuz.org/gmsh/doc/texinfo/gmsh.html#MSH-ASCII-file-format
    .. moduleauthor:: Nico Schloemer <nico.schloemer@gmail.com>
    LICENSE MIT
    https://github.com/nschloe/meshio
    """

    import logging
    import struct

    import numpy

    num_nodes_per_cell = {
        "vertex": 1,
        "line": 2,
        "triangle": 3,
        "quad": 4,
        "tetra": 4,
        "hexahedron": 8,
        "wedge": 6,
        "pyramid": 5,
        #
        "line3": 3,
        "triangle6": 6,
        "quad9": 9,
        "tetra10": 10,
        "hexahedron27": 27,
        "prism18": 18,
        "pyramid14": 14,
        "line4": 4,
        "quad16": 16,
    }

    # Translate meshio types to gmsh codes
    # http://geuz.org/gmsh/doc/texinfo/gmsh.html#MSH-ASCII-file-format
    _gmsh_to_meshio_type = {
        15: "vertex",
        1: "line",
        2: "triangle",
        3: "quad",
        4: "tetra",
        5: "hexahedron",
        6: "wedge",
        7: "pyramid",
        8: "line3",
        9: "triangle6",
        10: "quad9",
        11: "tetra10",
        12: "hexahedron27",
        13: "prism18",
        14: "pyramid14",
        26: "line4",
        36: "quad16",
    }
    _meshio_to_gmsh_type = {v: k for k, v in _gmsh_to_meshio_type.items()}

    verbose = 1
    if verbose > 0:
        print("--> GMSH format         ... ")

    try:
        f = open(infile, "r")
    except IOError:
        print("[file not found or not readable]\n")
        return

    # Initialize the data optional data fields
    points = []
    cells = {}
    field_data = {}
    cell_data = {}
    point_data = {}

    has_additional_tag_data = False
    is_ascii = None
    int_size = 4
    data_size = None
    while True:
        line = f.readline()
        if not line:
            # EOF
            break
        assert line[0] == "$"
        environ = line[1:].strip()
        if environ == "MeshFormat":
            line = f.readline()
            # Split the line
            # 2.2 0 8
            # into its components.
            str_list = list(filter(None, line.split()))
            assert str_list[0][0] == "2", "Need mesh format 2"
            assert str_list[1] in ["0", "1"]
            is_ascii = str_list[1] == "0"
            data_size = int(str_list[2])
            if not is_ascii:
                # The next line is the integer 1 in bytes. Useful to check
                # endianness. Just assert that we get 1 here.
                one = f.read(int_size)
                assert struct.unpack("i", one)[0] == 1
                line = f.readline()
                assert line == "\n"
            line = f.readline()
            assert line.strip() == "$EndMeshFormat"
        elif environ == "PhysicalNames":
            line = f.readline()
            num_phys_names = int(line)
            for _ in range(num_phys_names):
                line = f.readline()
                key = line.split(" ")[2].replace('"', "").replace("\n", "")
                phys_group = int(line.split(" ")[1])
                field_data[key] = phys_group
            line = f.readline()
            assert line.strip() == "$EndPhysicalNames"
        elif environ == "Nodes":
            # The first line is the number of nodes
            line = f.readline()
            num_nodes = int(line)
            if is_ascii:
                points = numpy.fromfile(f, count=num_nodes * 4, sep=" ").reshape(
                    (num_nodes, 4)
                )
                # The first number is the index
                points = points[:, 1:]
            else:
                # binary
                num_bytes = num_nodes * (int_size + 3 * data_size)
                assert numpy.int32(0).nbytes == int_size
                assert numpy.float64(0.0).nbytes == data_size
                dtype = [("index", numpy.int32), ("x", numpy.float64, (3,))]
                data = numpy.fromstring(f.read(num_bytes), dtype=dtype)
                assert (data["index"] == range(1, num_nodes + 1)).all()
                # vtk numpy support requires contiguous data
                points = numpy.ascontiguousarray(data["x"])
                line = f.readline()
                assert line == "\n"

            line = f.readline()
            assert line.strip() == "$EndNodes"
        else:
            assert environ == "Elements", "Unknown environment '{}'.".format(environ)
            # The first line is the number of elements
            line = f.readline()
            total_num_cells = int(line)
            if is_ascii:
                for _ in range(total_num_cells):
                    line = f.readline()
                    data = [int(k) for k in filter(None, line.split())]
                    t = _gmsh_to_meshio_type[data[1]]
                    num_nodes_per_elem = num_nodes_per_cell[t]

                    if t not in cells:
                        cells[t] = []
                    cells[t].append(data[-num_nodes_per_elem:])

                    # data[2] gives the number of tags. The gmsh manual
                    # <http://gmsh.info/doc/texinfo/gmsh.html#MSH-ASCII-file-format>
                    # says:
                    # >>>
                    # By default, the first tag is the number of the physical
                    # entity to which the element belongs; the second is the
                    # number of the elementary geometrical entity to which the
                    # element belongs; the third is the number of mesh
                    # partitions to which the element belongs, followed by the
                    # partition ids (negative partition ids indicate ghost
                    # cells). A zero tag is equivalent to no tag. Gmsh and most
                    # codes using the MSH 2 format require at least the first
                    # two tags (physical and elementary tags).
                    # <<<
                    num_tags = data[2]
                    if t not in cell_data:
                        cell_data[t] = []
                    cell_data[t].append(data[3 : 3 + num_tags])

                # convert to numpy arrays
                for key in cells:
                    cells[key] = numpy.array(cells[key], dtype=int)
                for key in cell_data:
                    cell_data[key] = numpy.array(cell_data[key], dtype=int)
            else:
                # binary
                num_elems = 0
                while num_elems < total_num_cells:
                    # read element header
                    elem_type = struct.unpack("i", f.read(int_size))[0]
                    t = _gmsh_to_meshio_type[elem_type]
                    num_nodes_per_elem = num_nodes_per_cell[t]
                    num_elems0 = struct.unpack("i", f.read(int_size))[0]
                    num_tags = struct.unpack("i", f.read(int_size))[0]
                    # assert num_tags >= 2

                    # read element data
                    num_bytes = 4 * (num_elems0 * (1 + num_tags + num_nodes_per_elem))
                    shape = (num_elems0, 1 + num_tags + num_nodes_per_elem)
                    b = f.read(num_bytes)
                    data = numpy.fromstring(b, dtype=numpy.int32).reshape(shape)

                    if t not in cells:
                        cells[t] = []
                    cells[t].append(data[:, -num_nodes_per_elem:])

                    if t not in cell_data:
                        cell_data[t] = []
                    cell_data[t].append(data[:, 1 : num_tags + 1])

                    num_elems += num_elems0

                # collect cells
                for key in cells:
                    cells[key] = numpy.vstack(cells[key])

                # collect cell data
                for key in cell_data:
                    cell_data[key] = numpy.vstack(cell_data[key])

                line = f.readline()
                assert line == "\n"

            line = f.readline()
            assert line.strip() == "$EndElements"

            # Subtract one to account for the fact that python indices are
            # 0-based.
            for key in cells:
                cells[key] -= 1

            # restrict to the standard two data items
            output_cell_data = {}
            for key in cell_data:
                if cell_data[key].shape[1] > 2:
                    has_additional_tag_data = True
                output_cell_data[key] = {}
                if cell_data[key].shape[1] > 0:
                    output_cell_data[key]["physical"] = cell_data[key][:, 0]
                if cell_data[key].shape[1] > 1:
                    output_cell_data[key]["geometrical"] = cell_data[key][:, 1]
            cell_data = output_cell_data

    if has_additional_tag_data:
        logging.warning("The file contains tag data that couldn't be processed.")

    return points, cells, point_data, cell_data, field_data


def export_vtk(tria, outfile):
    """
    Save VTK file
    usage: exportVTK(TriaMesh,outfile)
    """
    # open file
    try:
        f = open(outfile, "w")
    except IOError:
        print("[File " + outfile + " not writable]")
        return
    # check data structure
    # ...
    # Write
    f.write("# vtk DataFile Version 1.0\n")
    f.write("vtk output\n")
    f.write("ASCII\n")
    f.write("DATASET POLYDATA\n")
    f.write("POINTS " + str(np.shape(tria.v)[0]) + " float\n")
    for i in range(np.shape(tria.v)[0]):
        f.write(" ".join(map(str, tria.v[i, :])))
        f.write("\n")
    f.write(
        "POLYGONS "
        + str(np.shape(tria.t)[0])
        + " "
        + str(4 * np.shape(tria.t)[0])
        + "\n"
    )
    for i in range(np.shape(tria.t)[0]):
        f.write(" ".join(map(str, np.append(3, tria.t[i, :]))))
        f.write("\n")
    f.close()


def export_fssurf(tria, outfile):
    """
    Save Freesurfer Surface Geometry file (wrap Nibabel)
    """
    # open file
    try:
        from nibabel.freesurfer.io import write_geometry

        write_geometry(outfile, tria.v, tria.t, volume_info=tria.fsinfo)
    except IOError:
        print("[File " + outfile + " not writable]")
        return
