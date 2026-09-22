"""Shared reader for GMSH meshes, MSH 2 ASCII format.

Both :mod:`lapy._tria_io` and :mod:`lapy._tet_io` parse the same file format,
so the parsing lives here once and each of them keeps only the thin wrapper
that turns the result into its own mesh type.

.. moduleauthor:: Nico Schloemer <nico.schloemer@gmail.com>
LICENSE MIT
https://github.com/nschloe/meshio
http://geuz.org/gmsh/doc/texinfo/gmsh.html#MSH-ASCII-file-format
"""

import logging
import os
import re
from itertools import islice

import numpy as np

logger = logging.getLogger(__name__)

# gmsh element code -> name, and the vertex count of each name.
GMSH_TO_NAME = {
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
NAME_TO_GMSH = {v: k for k, v in GMSH_TO_NAME.items()}
NODES_PER_CELL = {
    "vertex": 1,
    "line": 2,
    "triangle": 3,
    "quad": 4,
    "tetra": 4,
    "hexahedron": 8,
    "wedge": 6,
    "pyramid": 5,
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


def _fail(msg: str) -> None:
    """Log and raise in the format the other readers use."""
    msg = f"[{msg}] --> FAILED\n"
    logger.error(msg)
    raise ValueError(msg)


def _expect(line: str, keyword: str) -> None:
    """Fail on a malformed file.

    Not an assert: these validate untrusted file content, and assert is
    stripped under ``python -O``, which would let a truncated file through and
    only surface much later as a wrong mesh.
    """
    if line.strip() != keyword:
        _fail(f"{keyword} keyword not found")


def _holds_type(block: str, name: str) -> bool:
    """Say whether an element block contains a cell type, without parsing it.

    Each record starts ``number type ...``, so the type is the second token of
    a line. Scanning the raw text for that is several times cheaper than
    converting the whole block to integers, which lets a caller that asked for
    triangles reject a volume mesh before paying for the conversion.
    """
    return re.search(rf"(?m)^\s*\d+\s+{NAME_TO_GMSH[name]}\s", block) is not None


def _cells_from_block(flat: np.ndarray, num_cells: int) -> dict:
    """Split a flat MSH 2 element block into 0-based node indices per cell type.

    A record is ``number type n_tags <tags...> <nodes...>``, so records only
    share a width when the block holds a single element type with a single tag
    count. That is the common case and reshapes in one step; a mixed block
    falls back to walking the records.

    Parameters
    ----------
    flat : np.ndarray
        All integers of the element block, in file order.
    num_cells : int
        Number of records the block declares.

    Returns
    -------
    dict
        Maps an element name to its (n_cells, n_vertices) index array.
    """
    if num_cells == 0:
        return {}

    width, remainder = divmod(flat.size, num_cells)
    if remainder == 0:
        rows = flat.reshape(num_cells, width)
        etype, n_tags = int(rows[0, 1]), int(rows[0, 2])
        name = GMSH_TO_NAME[etype]
        uniform = (
            3 + n_tags + NODES_PER_CELL[name] == width
            and bool((rows[:, 1] == etype).all())
            and bool((rows[:, 2] == n_tags).all())
        )
        if uniform:
            return {name: rows[:, -NODES_PER_CELL[name] :] - 1}

    # Mixed block: walk the records. tolist() keeps the walk in plain python,
    # which is far cheaper than numpy scalar indexing in a loop, and the node
    # indices are extended into one flat list per type rather than appended as
    # rows, so the final conversion is a single reshape.
    values = flat.tolist()
    chunks: dict = {}
    pos, size = 0, len(values)
    while pos < size:
        name = GMSH_TO_NAME[values[pos + 1]]
        n_nodes = NODES_PER_CELL[name]
        end = pos + 3 + values[pos + 2] + n_nodes
        chunks.setdefault(name, []).extend(values[end - n_nodes : end])
        pos = end
    return {
        k: np.array(v, dtype=np.int64).reshape(-1, NODES_PER_CELL[k]) - 1
        for k, v in chunks.items()
    }


def read_gmsh(filename: str, want: str | None = None) -> tuple[np.ndarray, dict]:
    """Read points and cells from a GMSH file, MSH 2 ASCII format.

    Parameters
    ----------
    filename : str
        Filename to load, must end in ``.msh``.
    want : str or None, default=None
        Cell type the caller needs, for example ``"triangle"`` or ``"tetra"``.
        When given, a file that cannot supply it is rejected as early as
        possible: the node coordinates are skipped rather than parsed until the
        element block has been checked, and the check itself is a text scan
        rather than a full conversion. Reading a volume mesh as a triangle mesh
        therefore costs a fraction of a full read.

    Returns
    -------
    points : np.ndarray
        Vertex coordinates, shape (n_points, 3).
    cells : dict
        Maps an element name to its (n_cells, n_vertices) index array.

    Raises
    ------
    OSError
        If file is not found or not readable.
    ValueError
        If the extension is not ``.msh``, the file is binary or not version 2,
        a section is malformed, or ``want`` is not present.
    """
    if os.path.splitext(filename)[1] != ".msh":
        _fail("no .msh file")

    logger.debug("--> GMSH format         ... ")

    try:
        f = open(filename)
    except OSError:
        logger.error("[file not found or not readable]")
        raise

    points = None
    node_offset = None
    num_nodes = 0
    cells: dict = {}

    with f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith("$"):
                _fail(f"section keyword expected, found '{line.strip()}'")
            section = line[1:].strip()

            if section == "MeshFormat":
                larr = list(filter(None, f.readline().split()))
                if larr[0][0] != "2":
                    _fail(f"need mesh format 2, found {larr[0]}")
                if larr[1] != "0":
                    # The file is opened in text mode, so the binary layout
                    # cannot be read from this handle at all.
                    _fail("binary format not implemented")
                _expect(f.readline(), "$EndMeshFormat")

            elif section == "Nodes":
                num_nodes = int(f.readline())
                if want is not None and not cells:
                    # Skip for now. If the element block turns out not to hold
                    # `want` we never pay to parse these coordinates, and
                    # skipping is about thirty times cheaper than parsing.
                    node_offset = f.tell()
                    for _ in islice(f, num_nodes):
                        pass
                else:
                    points = _read_points(f, num_nodes)
                _expect(f.readline(), "$EndNodes")

            elif section == "Elements":
                num_cells = int(f.readline())
                block = "".join(islice(f, num_cells))
                if want is not None:
                    first = block.split("\n", 1)[0].split()
                    known = len(first) > 1 and int(first[1]) == NAME_TO_GMSH[want]
                    if not known and not _holds_type(block, want):
                        _fail(f"file holds no {want} elements")
                cells = _cells_from_block(
                    np.array(block.split(), dtype=np.int64), num_cells
                )
                _expect(f.readline(), "$EndElements")

            else:
                # Skip sections we do not use, such as $PhysicalNames,
                # $Periodic or $NodeData. Erroring out instead would reject
                # perfectly good files.
                end = f"$End{section}"
                while True:
                    line = f.readline()
                    if not line:
                        _fail(f"{end} keyword not found")
                    if line.strip() == end:
                        break

        if points is None and node_offset is not None:
            f.seek(node_offset)
            points = _read_points(f, num_nodes)

    if points is None:
        _fail("$Nodes section not found")
    if want is not None and want not in cells:
        _fail(f"file holds no {want} elements")

    return points, cells


def _read_points(f, num_nodes: int) -> np.ndarray:
    """Read a node block as (n, 3), dropping the leading index column."""
    points = np.fromfile(f, "float32", num_nodes * 4, " ").reshape(num_nodes, 4)
    return np.ascontiguousarray(points[:, 1:])
