"""Tests for covalent-radius bond guessing."""

from pathlib import Path

import numpy as np

from pychem.core.bonds import guess_bonds
from pychem.io import get_reader

DATA = Path(__file__).parent / "data" / "xyz"


def _degrees(bonds: np.ndarray, n_atoms: int) -> np.ndarray:
    flat = bonds.ravel()
    if flat.size == 0:
        return np.zeros(n_atoms, dtype=int)
    return np.bincount(flat, minlength=n_atoms)


def test_water_two_oh_bonds():
    water = get_reader(DATA / "water.xyz")
    bonds = guess_bonds(water)
    assert bonds.shape == (2, 2)
    # Only O-H bonds
    assert {tuple(sorted(b)) for b in bonds} == {(0, 1), (0, 2)}


def test_benzene_ring_and_ch_bonds():
    benzene = get_reader(DATA / "benzene.xyz")
    bonds = guess_bonds(benzene)
    assert len(bonds) == 12  # 6 C-C + 6 C-H

    degrees = _degrees(bonds, benzene.n_atoms)
    # The 6 carbons have degree 3; the 6 hydrogens degree 1
    assert (degrees[:6] == 3).all()
    assert (degrees[6:] == 1).all()


def test_single_atom_no_bonds():
    neon = get_reader(DATA / "ne.xyz")
    bonds = guess_bonds(neon)
    assert bonds.shape == (0, 2)
