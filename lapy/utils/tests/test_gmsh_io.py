"""Tests for the shared GMSH reader."""

import numpy as np
import pytest

from ...tet_mesh import TetMesh
from ...tria_mesh import TriaMesh

GMSH_CODE = {3: 2, 4: 4}  # vertices per cell -> gmsh element code


def _write_msh(path, points, cells):
    """Write MSH 2 ASCII, the only variant the reader accepts.

    ``cells`` is a list of index arrays; the element code follows from width.
    """
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$Nodes", str(len(points))]
    lines += [
        f"{i} {p[0]:.17g} {p[1]:.17g} {p[2]:.17g}" for i, p in enumerate(points, 1)
    ]
    lines += ["$EndNodes"]
    records, n = [], 0
    for block in cells:
        code = GMSH_CODE[block.shape[1]]
        for row in block:
            n += 1
            nodes = " ".join(str(int(x) + 1) for x in row)
            # number type n_tags physical elementary <nodes>
            records.append(f"{n} {code} 2 1 1 {nodes}")
    lines += ["$Elements", str(n)] + records + ["$EndElements", ""]
    path.write_text("\n".join(lines))


@pytest.fixture
def icosahedron():
    return TriaMesh.read_off("data/icosahedron.off")


def test_read_gmsh_round_trip(icosahedron, tmp_path):
    """Reading back a written triangle mesh must reproduce it exactly."""
    path = tmp_path / "tria.msh"
    _write_msh(path, icosahedron.v, [icosahedron.t])

    tria = TriaMesh.read_gmsh(str(path))

    np.testing.assert_array_equal(tria.t, icosahedron.t)
    np.testing.assert_allclose(tria.v, icosahedron.v, rtol=1e-6, atol=1e-7)
    assert tria.is_closed()


def test_read_gmsh_picks_the_requested_type(icosahedron, tmp_path):
    """One file holding both types must serve both readers."""
    path = tmp_path / "mixed.msh"
    tetra = np.array([[0, 1, 2, 3]])
    _write_msh(path, icosahedron.v, [icosahedron.t, tetra])

    np.testing.assert_array_equal(TriaMesh.read_gmsh(str(path)).t, icosahedron.t)
    np.testing.assert_array_equal(TetMesh.read_gmsh(str(path)).t, tetra)


def test_read_gmsh_rejects_a_file_without_the_wanted_type(icosahedron, tmp_path):
    """Asking for triangles in a volume mesh must fail, and vice versa."""
    tria_only = tmp_path / "tria.msh"
    _write_msh(tria_only, icosahedron.v, [icosahedron.t])
    tet_only = tmp_path / "tet.msh"
    _write_msh(tet_only, icosahedron.v, [np.array([[0, 1, 2, 3]])])

    with pytest.raises(ValueError, match="no triangle elements"):
        TriaMesh.read_gmsh(str(tet_only))
    with pytest.raises(ValueError, match="no tetra elements"):
        TetMesh.read_gmsh(str(tria_only))


def test_read_gmsh_skips_sections_it_does_not_use(icosahedron, tmp_path):
    """$PhysicalNames and friends must not make a good file unreadable."""
    path = tmp_path / "extra.msh"
    _write_msh(path, icosahedron.v, [icosahedron.t])
    text = path.read_text().replace(
        "$Nodes",
        '$PhysicalNames\n1\n2 1 "surface"\n$EndPhysicalNames\n'
        "$Periodic\n0\n$EndPeriodic\n$Nodes",
        1,
    )
    path.write_text(text)

    np.testing.assert_array_equal(TriaMesh.read_gmsh(str(path)).t, icosahedron.t)


def test_read_gmsh_rejects_binary(tmp_path):
    """Binary MSH cannot be read from a text handle, so say so rather than crash."""
    path = tmp_path / "binary.msh"
    path.write_text("$MeshFormat\n2.2 1 8\n")

    with pytest.raises(ValueError, match="binary format not implemented"):
        TriaMesh.read_gmsh(str(path))
