"""Tests for volumetric file loading (cube and Molden)."""

from pathlib import Path

import numpy as np
import pytest

from pychem.io.volumetric import load_volume_file, read_cube

DATA = Path(__file__).parent / "data" / "volumetric"
CUBE = DATA / "single_point.mo1a.cube"
MOLDEN = DATA / "single_point.molden"


def test_read_cube_molecule_and_grid():
    molecule, volumes = read_cube(CUBE)
    assert molecule.n_atoms == 16
    assert len(volumes) == 1
    vol = volumes[0]
    assert vol.shape == (40, 40, 40)
    assert vol.spacing.shape == (3,)
    assert vol.max_abs == pytest.approx(2.1687, abs=0.01)
    # values are not all zero and mostly positive for this MO
    assert np.abs(vol.values).sum() > 0


def test_read_cube_coordinates_in_angstrom():
    """Cube geometry must be converted Bohr -> Å (regression).

    Gaussian/ORCA cube files store the atom positions and the grid in Bohr;
    before the conversion the atoms appeared ~1.9x too far apart and no bonds
    were formed. The geometry must come out in Å and match the Molden
    reference of the same molecule (also in Å after its own AU conversion).
    """
    from pychem.core.bonds import guess_bonds

    cube_mol, _ = read_cube(CUBE)
    molden = load_volume_file(MOLDEN)

    # Å-scale coordinates: the molecule fits well within ±10 Å.
    assert np.abs(cube_mol.coords).max() < 10.0
    # Same molecule and ordering as the Molden reference geometry.
    assert cube_mol.symbols == molden.molecule.symbols
    np.testing.assert_allclose(cube_mol.coords, molden.molecule.coords, atol=1e-6)
    # Bond guessing now recovers the structure (the reported bug).
    assert len(guess_bonds(cube_mol)) == 17


def test_load_volume_file_cube():
    vf = load_volume_file(CUBE)
    assert len(vf.labels) == 1
    assert vf.molecule.n_atoms == 16
    vol = vf.volume(0)
    assert vol.shape == (40, 40, 40)


def test_load_molden_metadata_and_evaluate():
    vf = load_volume_file(MOLDEN)
    assert vf.molecule.n_atoms == 16
    # 109 basis functions -> 109 MO blocks in this file
    assert len(vf.labels) == 109
    vol = vf.volume(30)  # evaluate one orbital on the grid
    assert vol.shape == (40, 40, 40)
    assert vol.max_abs > 0


def test_read_cube_multiblock(tmp_path):
    """A cube with two appended grids reads as two scalar fields."""
    text = Path(CUBE).read_text(encoding="utf-8").splitlines()
    header = "\n".join(text[:22])  # comments + counts/axes + atoms
    # Original file has a couple of trailing tokens; take exactly 64000 values
    tokens = [float(tok) for line in text[22:] for tok in line.split()][:64000]

    def fmt(vals):
        lines = []
        for i in range(0, len(vals), 6):
            lines.append(" ".join(f"{v:.6E}" for v in vals[i : i + 6]))
        return "\n".join(lines)

    body = fmt(tokens)
    out = tmp_path / "multi.cube"
    out.write_text(f"{header}\n{body}\n{body}\n", encoding="utf-8")

    molecule, volumes = read_cube(out)
    assert molecule.n_atoms == 16
    assert len(volumes) == 2
    assert volumes[0].shape == (40, 40, 40)
    np.testing.assert_allclose(volumes[0].values, volumes[1].values)


def test_molden_atomic_orbitals_per_atom():
    """[GTO] shells are assigned to their real atom (regression) and the AO
    groups match the expected basis (heavy: s/p; H: s only)."""
    from pychem.io.volumetric import atomic_orbitals_for_atom

    vf = load_volume_file(MOLDEN)
    assert vf.basis is not None  # only Molden carries a basis
    assert load_volume_file(CUBE).basis is None

    # Every basis function belongs to exactly one atom: 109 in total.
    total = 0
    for atom in range(vf.molecule.n_atoms):
        groups = atomic_orbitals_for_atom(vf.basis, atom)
        total += sum(len(aos) for aos in groups.values())
    assert total == len(vf.labels) == 109

    # Carbon (index 0): s6+s3+p3+s1+p1 -> 3 s + 6 p = 9 functions.
    groups = atomic_orbitals_for_atom(vf.basis, 0)
    assert sorted(groups) == [0, 1]
    assert len(groups[0]) == 3 and len(groups[1]) == 6

    # Hydrogen (index 3): two contracted s shells -> 2 s functions, no p.
    groups_h = atomic_orbitals_for_atom(vf.basis, 3)
    assert sorted(groups_h) == [0]
    assert len(groups_h[0]) == 2


def test_evaluate_atomic_orbital_grid():
    """A single basis function evaluates on a box around its atom."""
    from pychem.io.volumetric import (
        atomic_orbitals_for_atom,
        evaluate_atomic_orbital,
    )

    vf = load_volume_file(MOLDEN)
    groups = atomic_orbitals_for_atom(vf.basis, 0)
    for ao in (groups[0][0], groups[1][0]):  # first s and first px
        vol = evaluate_atomic_orbital(vf.basis, ao)
        assert vol.shape == (40, 40, 40)
        assert vol.max_abs > 0
        assert vol.label  # non-empty label
