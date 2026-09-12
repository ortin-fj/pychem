"""Tests for volumetric file loading (cube and Molden)."""

from pathlib import Path

import numpy as np
import pytest

from pychem.io.volumetric import (
    cube_kind,
    load_volume_file,
    read_cube,
    read_esp_pair,
)

DATA = Path(__file__).parent / "data" / "volumetric"
CUBE = DATA / "single_point.mo1a.cube"
MOLDEN = DATA / "single_point.molden"
DENSITY_CUBE = DATA / "opt.eldens.cube"
ESP_CUBE = DATA / "0.esp.cube"


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


def test_read_cube_field_is_anchored_to_the_nuclei():
    """Regression: the voxel field must not read as a transposed grid.

    The values are stored with the *last* grid axis varying fastest, so
    reshaping in F order ("first axis fastest") transposes the field: the
    density then peaks *between* atoms (rho(Br) dropped to 0.003 instead of
    being the grid maximum) and every isosurface came out sheared and shifted
    away from the molecule.
    """
    molecule, volumes = read_cube(DENSITY_CUBE)
    density = volumes[0]

    at_atoms = density.sample_at(molecule.coords)
    rho = dict(zip(molecule.symbols, at_atoms))
    # An electron density decreases with the nuclear charge.
    assert rho["Br"] > rho["Cl"] > rho["C"] > 5.0 * rho["H"]

    # Its global maximum is a heavy nucleus too (within one voxel of it).
    heavy = molecule.symbols.index("Br")
    node = np.unravel_index(np.argmax(density.values), density.shape)
    node_position = density.origin + np.asarray(node) * density.spacing
    distance = float(np.linalg.norm(node_position - molecule.coords[heavy]))
    assert distance < float(np.linalg.norm(density.spacing))


def test_read_cube_potential_field_uses_the_same_order():
    """The ESP cube shares the axis order (its divergence sits on Br)."""
    molecule, volumes = read_cube(ESP_CUBE)
    potential = volumes[0]

    heavy = molecule.symbols.index("Br")
    node = np.unravel_index(np.argmax(potential.values), potential.shape)
    node_position = potential.origin + np.asarray(node) * potential.spacing
    distance = float(np.linalg.norm(node_position - molecule.coords[heavy]))
    assert distance < float(np.linalg.norm(potential.spacing))
    # Nuclear attraction: the potential is huge (and positive) at the nucleus.
    assert potential.values.max() > 10.0


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


# ---------------------------------------------------------------------------
# Electrostatic-potential maps (density + potential cube pair)
# ---------------------------------------------------------------------------


def test_cube_kind_reads_the_second_line_description():
    assert cube_kind("Total electron density") == "density"
    assert cube_kind("Electrostatic Potential") == "potential"
    assert cube_kind("Electrostatic potential (a.u.)") == "potential"
    assert cube_kind("scf electron density") == "density"
    assert cube_kind("MO 1") == "unknown"
    assert cube_kind("") == "unknown"


def test_read_esp_pair_is_order_independent():
    """The density and the potential are told apart by their own labels."""
    pair = read_esp_pair([ESP_CUBE, DENSITY_CUBE])
    assert pair.density_path == DENSITY_CUBE
    assert pair.potential_path == ESP_CUBE
    assert pair.density.label == "Total electron density"
    assert pair.potential.label == "Electrostatic Potential"

    # Both grids describe the same box, so the potential can be sampled on
    # the density envelope.
    np.testing.assert_allclose(pair.density.origin, pair.potential.origin)
    np.testing.assert_allclose(pair.density.spacing, pair.potential.spacing)
    assert pair.density.shape == pair.potential.shape == (40, 40, 40)

    swapped = read_esp_pair([DENSITY_CUBE, ESP_CUBE])
    assert swapped.density_path == pair.density_path
    assert swapped.potential_path == pair.potential_path


def test_read_esp_pair_requires_both_cubes():
    with pytest.raises(ValueError, match="Total electron density"):
        read_esp_pair([DENSITY_CUBE, DENSITY_CUBE])
    with pytest.raises(ValueError, match="Electrostatic Potential"):
        read_esp_pair([ESP_CUBE])


def test_esp_sample_data_matches_its_structure():
    """The ESP cubes and sn2_opt.xyz are the same geometry (Å)."""
    from pychem.io import get_reader

    xyz = get_reader(Path(__file__).parent / "data" / "xyz" / "sn2_opt.xyz")
    pair = read_esp_pair([DENSITY_CUBE, ESP_CUBE])
    density_molecule, _ = read_cube(DENSITY_CUBE)

    assert density_molecule.symbols == xyz.symbols
    np.testing.assert_allclose(density_molecule.coords, xyz.coords, atol=1e-4)
    np.testing.assert_allclose(
        read_cube(ESP_CUBE)[0].coords, xyz.coords, atol=1e-4
    )
    assert pair.density.suggest_density_isovalue() == pytest.approx(1e-3)


def test_potential_sampled_on_the_density_envelope():
    """Sampling the potential inside its grid stays within its own range."""
    pair = read_esp_pair([DENSITY_CUBE, ESP_CUBE])
    points = pair.density.origin + pair.density.spacing * np.array(
        [[10, 10, 10], [20, 20, 20], [30, 5, 25]]
    )
    values = pair.potential.sample_at(points)

    assert np.all(np.isfinite(values))
    assert values.min() >= pair.potential.values.min() - 1e-9
    assert values.max() <= pair.potential.values.max() + 1e-9

