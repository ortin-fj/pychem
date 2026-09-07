"""Tests for the PDB reader (structures only, first model)."""

from pathlib import Path

import numpy as np
import pytest

from pychem.core.molecule import Molecule
from pychem.io.registry import get_reader, supported_extensions

DATA = Path(__file__).parent / "data" / "xyz"


def test_read_pdb_water():
    mol = get_reader(DATA / "water.pdb")
    assert isinstance(mol, Molecule)
    assert mol.n_atoms == 3
    assert mol.name == "water"
    assert mol.symbols == ["O", "H", "H"]
    assert mol.formula == "H2O"

    # Same reference geometry as water.xyz: H-O-H angle ~104.5°.
    o = mol.coords[0]
    v1 = mol.coords[1] - o
    v2 = mol.coords[2] - o
    angle = np.degrees(
        np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    )
    assert angle == pytest.approx(104.5, abs=0.5)


def test_read_pdb_returns_structure_not_trajectory():
    # Even a multi-model PDB is read as a single structure (Molecule), never
    # as a Trajectory.
    from pychem.core.trajectory import Trajectory

    mol = get_reader(DATA / "two_models.pdb")
    assert isinstance(mol, Molecule)
    assert not isinstance(mol, Trajectory)
    # Only the first MODEL is used: the +5 Å second model (with the extra
    # sodium) is ignored.
    assert mol.n_atoms == 3
    assert mol.symbols == ["O", "H", "H"]
    assert np.allclose(mol.coords[0], [0.0, 0.0, 0.1173], atol=1e-3)


def test_pdb_is_registered():
    assert "pdb" in supported_extensions()
    assert "ent" in supported_extensions()


def test_pdb_missing_atoms_raises(tmp_path):
    bad = tmp_path / "empty.pdb"
    bad.write_text("HEADER    no atoms here\nEND\n", encoding="ascii")
    with pytest.raises(ValueError):
        get_reader(bad)
