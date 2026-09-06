"""Tests for covalent-radius bond guessing."""

from pathlib import Path

import numpy as np
import pytest

from pychem.core.bonds import guess_bonds, update_dynamic_bonds
from pychem.core.molecule import Molecule
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


def _h2(distance: float) -> Molecule:
    return Molecule(
        symbols=["H", "H"],
        coords=np.array([[-distance / 2, 0.0, 0.0], [distance / 2, 0.0, 0.0]]),
    )


def test_dynamic_bonds_hysteresis():
    # Form: d <= 0.62*1.10 = 0.682; keep (previous): d <= 0.62*1.25 = 0.775
    assert update_dynamic_bonds(_h2(0.58), None).shape == (1, 2)  # forms
    # From empty, 0.72 (> form) does NOT form...
    assert update_dynamic_bonds(_h2(0.72), None).size == 0
    # ...but if it was bonded before, it is kept (no flicker).
    b1 = update_dynamic_bonds(_h2(0.58), None)
    b2 = update_dynamic_bonds(_h2(0.72), b1)
    assert b2.shape == (1, 2)
    # 0.82 > break threshold -> bond disappears
    b3 = update_dynamic_bonds(_h2(0.82), b2)
    assert b3.size == 0
    # 0.62 <= form threshold -> bond reappears
    b4 = update_dynamic_bonds(_h2(0.62), b3)
    assert b4.shape == (1, 2)
    assert b4[0].tolist() == [0, 1]
