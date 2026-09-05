"""Tests for the core models (Atom, Molecule)."""

import numpy as np
import pytest

from pychem.core.elements import element
from pychem.core.molecule import Molecule


def _water() -> Molecule:
    return Molecule(
        symbols=["O", "H", "H"],
        coords=np.array(
            [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]
        ),
        name="water",
    )


def test_basic_properties():
    mol = _water()
    assert mol.n_atoms == 3
    assert mol.formula == "H2O"
    assert mol.positions.shape == (3, 3)


def test_coords_shape_mismatch_raises():
    with pytest.raises(ValueError):
        Molecule(symbols=["O", "H"], coords=np.zeros((3, 3)))
    with pytest.raises(ValueError):
        Molecule(symbols=["O", "H"], coords=np.zeros((2, 2)))


def test_atom_iteration_and_get():
    mol = _water()
    atoms = list(mol.iter_atoms())
    assert [a.symbol for a in atoms] == ["O", "H", "H"]
    assert [a.index for a in atoms] == [0, 1, 2]
    assert np.allclose(atoms[1].position, mol.coords[1])

    atom = mol.get_atom(2)
    assert atom.symbol == "H"
    with pytest.raises(IndexError):
        mol.get_atom(3)


def test_element_lookup():
    o = element("O")
    assert o.number == 8
    assert o.color == (255, 13, 13)
    unknown = element("Xx")
    assert unknown.number == 0  # neutral/default element


def test_copy_is_independent():
    mol = _water()
    copied = mol.copy()
    copied.coords[0, 0] = 99.0
    assert mol.coords[0, 0] != 99.0
    assert copied.formula == mol.formula
