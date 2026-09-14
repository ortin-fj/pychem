"""Tests for covalent-radius bond guessing."""

import subprocess
import sys
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


# -- spatial-grid search regressions ----------------------------------------
#
# The search used to compare every pair, which materialised an ``n x n x 3``
# temporary: on the 11070-atom PDB below that was ~2.9 GB and ~3.2 s. It now
# bins the atoms in a grid and only compares neighbours. These tests pin down
# both the exact same *result* and the bounded memory.


def _naive_guess_bonds(molecule, tolerance: float = 0.15) -> np.ndarray:
    """Reference all-pairs implementation (the pre-grid behaviour)."""
    n = molecule.n_atoms
    if n < 2:
        return np.empty((0, 2), dtype=np.int64)
    from pychem.core.elements import element

    radii = np.array([element(s).covalent_radius for s in molecule.symbols])
    coords = molecule.coords
    rsum = radii[:, None] + radii[None, :]
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    i_idx, j_idx = np.where(
        (distances <= rsum * (1.0 + tolerance))
        & (np.arange(n)[:, None] < np.arange(n)[None, :])
    )
    return np.column_stack([i_idx, j_idx]).astype(np.int64)


@pytest.mark.parametrize(
    "name", ["water.xyz", "benzene.xyz", "sn2_opt.xyz", "water_traj.xyz"]
)
def test_guess_bonds_matches_the_all_pairs_reference(name):
    molecule = get_reader(DATA / name)
    assert np.array_equal(guess_bonds(molecule), _naive_guess_bonds(molecule))


def test_guess_bonds_grid_matches_reference_on_awkward_geometries():
    """Ties, duplicates and empty cells must not change the result."""
    rng = np.random.default_rng(7)
    for size, scale, layout in (
        (40, 3.0, "spread"),
        (40, 0.8, "clustered"),
        (30, 0.0, "integer grid"),
        (25, 0.01, "duplicates"),
    ):
        symbols = [str(rng.choice(["H", "C", "N", "O", "S"])) for _ in range(size)]
        if layout == "integer grid":
            coords = rng.integers(0, 3, size=(size, 3)).astype(float)
        else:
            coords = rng.normal(0.0, scale, size=(size, 3))
        molecule = Molecule(symbols=symbols, coords=coords)
        assert np.array_equal(
            guess_bonds(molecule), _naive_guess_bonds(molecule)
        ), f"grid search differs from reference for {layout}"


def test_guess_bonds_on_a_large_pdb_stays_within_a_memory_budget():
    """An 11070-atom PDB must fit in well under a gigabyte.

    The all-pairs version needed ~2.9 GB just for the coordinate differences,
    so this runs the real reader and search in a subprocess with a hard
    address-space cap: a regression to O(n²) fails with MemoryError instead
    of silently making the GUI swap.
    """
    script = (
        "import resource;"
        "resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30));"
        "from pychem.core.bonds import guess_bonds;"
        "from pychem.io.pdb import read_pdb;"
        "b = guess_bonds(read_pdb(r'%s'));"
        "print(len(b))"
    ) % (DATA / "4P00.pdb")
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert int(result.stdout.strip().splitlines()[-1]) == 11405
