"""Tests for the XYZ reader/writer and the format registry."""

from pathlib import Path

import numpy as np
import pytest

from pychem.core.molecule import Molecule
from pychem.io.registry import get_reader, supported_extensions
from pychem.io.xyz import read_xyz_frames, write_xyz

DATA = Path(__file__).parent / "data" / "xyz"


def test_read_water():
    mol = get_reader(DATA / "water.xyz")
    assert isinstance(mol, Molecule)
    assert mol.n_atoms == 3
    assert mol.name == "water"
    assert [a.symbol for a in mol.iter_atoms()] == ["O", "H", "H"]
    # H-O-H angle ~104.5° in the reference geometry
    o = mol.coords[0]
    v1 = mol.coords[1] - o
    v2 = mol.coords[2] - o
    angle = np.degrees(
        np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    )
    assert angle == pytest.approx(104.5, abs=0.2)


def test_read_extended_xyz_metadata():
    mol = get_reader(DATA / "extended.xyz")
    assert mol.n_atoms == 2
    assert "lattice" in mol.metadata
    assert "properties" in mol.metadata
    assert mol.formula == "H2"


def test_read_multiframe_first_frame():
    mol = get_reader(DATA / "water_traj.xyz")
    assert mol.n_atoms == 3
    frames = read_xyz_frames(DATA / "water_traj.xyz")
    assert len(frames) == 2
    assert np.allclose(frames[1].coords[0], [0.5, 0.1, 0.2173])


def test_write_roundtrip(tmp_path):
    mol = get_reader(DATA / "water.xyz")
    out = tmp_path / "out.xyz"
    write_xyz(out, mol, comment="roundtrip")
    reread = get_reader(out)
    assert reread.formula == mol.formula
    assert np.allclose(reread.coords, mol.coords)
    assert reread.metadata.get("comment") == "roundtrip"


def test_read_blank_comment_line():
    # An empty comment (blank line after the count) is valid in XYZ.
    mol = get_reader(DATA / "blank_comment.xyz")
    assert mol.n_atoms == 2
    assert mol.formula == "H2"


def test_unknown_extension_raises():
    with pytest.raises(ValueError):
        get_reader(Path("file.unknown"))


def test_registry_lists_xyz():
    assert "xyz" in supported_extensions()
