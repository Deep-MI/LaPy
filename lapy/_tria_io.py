"""Functions for IO of Triangle Meshes.

Should be called via the TriaMesh member functions.
"""

from logging import getLogger
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from .tria_mesh import TriaMesh

logger = getLogger(__name__)

def read_fssurf(filename: str) -> "TriaMesh":
    """Load triangle mesh from FreeSurfer surface geometry file.

    Parameters
    ----------
    filename : str
        Filename to load.

    Returns
    -------
    TriaMesh
        Loaded triangle mesh.

    Raises
    ------
    OSError
        If file is not found or not readable.
    """
    logger.debug("--> FS Surf format     ... ")
    try:
        # here we use our copy to also support surfaces from dev (and maybe v7*?)
        # these have an empty line and mess up Nibabel
        # once this is fixed in nibabel we can switch back
        from ._read_geometry import read_geometry

        surf = read_geometry(filename, read_metadata=True)
    except OSError:
        logger.error("[file not found or not readable]")
        raise
    from . import TriaMesh

    return TriaMesh(surf[0], surf[1], fsinfo=surf[2])


def read_off(filename: str) -> "TriaMesh":
    """Load triangle mesh from OFF txt file.

    Parameters
    ----------
    filename : str
        Filename to load.

    Returns
    -------
    TriaMesh
        Loaded triangle mesh.

    Raises
    ------
    OSError
        If file is not found or not readable.
        If OFF keyword is not found.
        If file does not contain triangle data.
    """
    logger.debug("--> OFF format         ... ")
    try:
        f = open(filename)
    except OSError:
        logger.error("[file not found or not readable]")
        raise
    line = f.readline()
    while line[0] == "#":
        line = f.readline()
    if not line.startswith("OFF"):
        msg = "[OFF keyword not found] --> FAILED\n"
        logger.error(msg)
        f.close()
        raise OSError(msg)
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
        msg = "[no triangle data] --> FAILED\n"
        logger.error(msg)
        f.close()
        raise OSError(msg)
    t = t[:, 1:]
    f.close()
    logger.info(" --> DONE ( V: %d , T: %d )", v.shape[0], t.shape[0])
    from . import TriaMesh

    return TriaMesh(v, t)


def read_vtk(filename: str) -> "TriaMesh":
    """Load triangle mesh from VTK txt file.

    Parameters
    ----------
    filename : str
        Filename to load.

    Returns
    -------
    TriaMesh
        Loaded triangle mesh.

    Raises
    ------
    OSError
        If file is not found or not readable.
        If ASCII keyword is not found.
        If DATASET POLYDATA or DATASET UNSTRUCTURED_GRID is not found.
        If POINTS keyword is malformed.
        If file does not contain triangle data.
    """
    logger.debug("--> VTK format         ... ")
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
        raise OSError(msg)
    # expect Dataset Polydata line after ASCII:
    line = f.readline()
    if not line.startswith("DATASET POLYDATA") and not line.startswith(
        "DATASET UNSTRUCTURED_GRID"
    ):
        msg = (
            f"[read: {line} expected DATASET POLYDATA or DATASET UNSTRUCTURED_GRID] "
            f"--> FAILED\n"
        )
        logger.error(msg)
        raise OSError(msg)
    # read number of points
    line = f.readline()
    larr = line.split()
    if larr[0] != "POINTS" or (larr[2] != "float" and larr[2] != "double"):
        msg = f"[read: {line} expected POINTS # float or POINTS # double ] --> FAILED\n"
        logger.error(msg)
        raise OSError(msg)
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
            msg = f"[having: {npt} data per tria, expected trias 3+1] --> FAILED\n"
            logger.error(msg)
            raise OSError(msg)
        t = np.fromfile(f, "int", ttnum, " ")
        t.shape = (tnum, 4)
        if t[tnum - 1][0] != 3:
            msg = "[can only read triangles] --> FAILED\n"
            logger.error(msg)
            raise OSError(msg)
        t = np.delete(t, 0, 1)
    elif larr[0] == "TRIANGLE_STRIPS":
        tnum = int(larr[1])
        # ttnum = int(larr[2])
        tt = []
        for _i in range(tnum):
            larr = f.readline().split()
            if len(larr) == 0:
                msg = "[error reading triangle strip (i)] --> FAILED\n"
                logger.error(msg)
                raise OSError(msg)
            n = int(larr[0])
            if len(larr) != n + 1:
                msg = "[error reading triangle strip (ii)] --> FAILED\n"
                logger.error(msg)
                raise OSError(msg)
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
        msg = f"[read: {line} expected POLYGONS or TRIANGLE_STRIPS] --> FAILED\n"
        logger.error(msg)
        raise OSError(msg)
    f.close()
    logger.info(" --> DONE ( V: %d , T: %d )", v.shape[0], t.shape[0])
    from . import TriaMesh

    return TriaMesh(v, t)


def read_gmsh(filename: str) -> tuple[np.ndarray, dict, dict, dict, dict]:
    """Load GMSH tetra mesh ASCII Format.

    Parameters
    ----------
    filename : str
        Filename to load.

    Returns
    -------
    points : np.ndarray
        Array of point coordinates, shape (n_points, 3).
    cells : dict
        Dictionary mapping cell type strings to arrays of cell vertex indices.
        Each array has shape (n_cells_of_type, n_vertices_per_cell).
    point_data : dict
        Dictionary of point data arrays.
    cell_data : dict
        Dictionary mapping cell type strings to dictionaries of data arrays.
        Contains 'physical' and 'geometrical' tags where available.
    field_data : dict
        Dictionary mapping physical region names to their integer tags.

    Raises
    ------
    OSError
        If file is not found or not readable.

    Notes
    -----
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
    _meshio_to_gmsh_type = {v: k for k, v in _gmsh_to_meshio_type.items()}  # noqa: F841

    logger.debug("--> GMSH format         ... ")

    try:
        f = open(filename)
    except OSError:
        logger.error("[file not found or not readable]")
        raise

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
            assert environ == "Elements", f"Unknown environment '{environ}'."
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
        logger.warning("The file contains tag data that couldn't be processed.")

    return points, cells, point_data, cell_data, field_data


def write_vtk(tria: "TriaMesh", filename: str) -> None:
    """Save VTK file.

    Parameters
    ----------
    tria : TriaMesh
        Triangle mesh to save.
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
    f.write("POINTS " + str(np.shape(tria.v)[0]) + " float\n")
    for i in range(np.shape(tria.v)[0]):
        f.write(" ".join(map(str, tria.v[i, :])))
        f.write("\n")
    f.write(f"POLYGONS {np.shape(tria.t)[0]} {4 * np.shape(tria.t)[0]}\n")
    for i in range(np.shape(tria.t)[0]):
        f.write(" ".join(map(str, np.append(3, tria.t[i, :]))))
        f.write("\n")
    f.close()


def write_fssurf(tria: "TriaMesh", filename: str, image: Optional[object] = None) -> None:
    """Save Freesurfer Surface Geometry file (wrap Nibabel).

    Parameters
    ----------
    tria : TriaMesh
        Triangle mesh to save.
    filename : str
        Filename to save to.
    image : str, object, or None, default=None
        Path to image, nibabel image object, or image header. If specified, the vertices
        are assumed to be in voxel coordinates and are converted to surface RAS (tkr)
        coordinates before saving. The expected order of coordinates is (x, y, z)
        matching the image voxel indices in nibabel.

    Raises
    ------
    OSError
        If file is not writable.
    TypeError
        If image header cannot be converted to provide get_vox2ras_tkr() method.

    Notes
    -----
    The surface RAS (tkr) transform is obtained from a header that implements
    ``get_vox2ras_tkr()`` (e.g., ``MGHHeader``). For other header types (NIfTI1/2,
    Analyze/SPM, etc.), we attempt conversion via ``MGHHeader.from_header``.
    """
    # open file
    try:
        from nibabel.freesurfer.io import write_geometry

        v = tria.v
        if image is not None:
            import nibabel as nib
            from nibabel.affines import apply_affine
            from nibabel.freesurfer.mghformat import MGHHeader

            # Accept: path -> image, image -> header, or header directly
            if isinstance(image, str):
                img = nib.load(image)
                header = img.header
            elif hasattr(image, "header"):
                # nibabel SpatialImage-like object
                header = image.header
            else:
                # assume header-like object
                header = image

            # If header doesn't provide tkr transform, try converting to MGHHeader
            if not hasattr(header, "get_vox2ras_tkr"):
                try:
                    header = MGHHeader.from_header(header)
                except Exception as e:
                    raise TypeError(
                        "write_fssurf(..., image=...) expected a nibabel image, a path to an "
                        "image, or a header that provides get_vox2ras_tkr() (or is convertible "
                        "via MGHHeader.from_header)."
                    ) from e

            v = apply_affine(header.get_vox2ras_tkr(), v)

        write_geometry(filename, v, tria.t, volume_info=tria.fsinfo)
    except OSError:
        logger.error("[File %s not writable]", filename)
        raise
