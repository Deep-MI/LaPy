"""Tests for the GMSH triangle reader."""

import numpy as np
import pytest

from ...tria_mesh import TriaMesh


def _write_msh(path, tria):
    """Write a mesh as MSH 2 ASCII, the only variant the reader accepts."""
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$Nodes", str(len(tria.v))]
    lines += [
        f"{i} {p[0]:.17g} {p[1]:.17g} {p[2]:.17g}" for i, p in enumerate(tria.v, 1)
    ]
    lines += ["$EndNodes", "$Elements", str(len(tria.t))]
    # elm-number elm-type(2=triangle) n-tags physical elementary nodes
    lines += [
        f"{i} 2 2 1 1 {t[0] + 1} {t[1] + 1} {t[2] + 1}"
        for i, t in enumerate(tria.t, 1)
    ]
    lines += ["$EndElements", ""]
    path.write_text("\n".join(lines))


def test_read_gmsh_round_trip(tmp_path):
    """Reading back a written mesh must reproduce it exactly."""
    expected = TriaMesh.read_off("data/icosahedron.off")
    path = tmp_path / "icosahedron.msh"
    _write_msh(path, expected)

    tria = TriaMesh.read_gmsh(str(path))

    np.testing.assert_array_equal(tria.t, expected.t)
    np.testing.assert_allclose(tria.v, expected.v, rtol=1e-12, atol=1e-14)
    assert tria.is_closed()


def test_read_gmsh_rejects_binary(tmp_path):
    """Binary MSH cannot be read from a text handle, so say so rather than crash."""
    path = tmp_path / "binary.msh"
    path.write_text("$MeshFormat\n2.2 1 8\n")

    with pytest.raises(ValueError, match="binary format not implemented"):
        TriaMesh.read_gmsh(str(path))
