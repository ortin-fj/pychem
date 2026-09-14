"""Tests for the structure-editing helpers (``pychem.core.editing``).

The umbrella geometry is checked against a methyl-halide model (the SN2
motif): reactants/products are pyramidal, the transition state is planar and
over-rotating collapses the interest atoms onto the axis.
"""

import numpy as np
import pytest

from pychem.core.editing import (
    UmbrellaGroup,
    has_bond,
    sorted_unique_pairs,
    toggle_bond,
)
from pychem.core.molecule import Molecule

_TETRAHEDRAL = np.radians(109.4712206)
_CH_RADIUS = 1.09


def _methyl_halide() -> Molecule:
    """CH3Br-like model: C at the origin, Br on +z, three H around -z."""
    coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.94]]
    for phi in (90.0, 210.0, 330.0):
        rad = np.radians(phi)
        coords.append(
            [
                _CH_RADIUS * np.sin(_TETRAHEDRAL) * np.cos(rad),
                _CH_RADIUS * np.sin(_TETRAHEDRAL) * np.sin(rad),
                _CH_RADIUS * np.cos(_TETRAHEDRAL),
            ]
        )
    return Molecule(
        symbols=["C", "Br", "H", "H", "H"], coords=np.array(coords, dtype=float)
    )


# -- bonds --------------------------------------------------------------------


def test_toggle_bond_adds_then_removes():
    bonds = toggle_bond(None, 4, 0)
    assert bonds.tolist() == [[0, 4]]
    assert has_bond(bonds, 0, 4) and has_bond(bonds, 4, 0)
    assert not has_bond(bonds, 0, 1)

    removed = toggle_bond(bonds, 0, 4)
    assert removed.shape == (0, 2)
    assert not has_bond(removed, 4, 0)


def test_sorted_unique_pairs_normalises_rows():
    raw = np.array([[3, 1], [1, 3], [2, 2], [0, 2]], dtype=np.int64)
    normalised = sorted_unique_pairs(raw)
    assert normalised.tolist() == [[0, 2], [1, 3]]
    assert normalised.dtype == np.int64


# -- umbrella group -----------------------------------------------------------


def test_umbrella_planar_state_is_an_equilateral_triangle_in_the_c_plane():
    mol = _methyl_halide()
    group = UmbrellaGroup.capture(mol, 0, 1, [2, 3, 4])
    assert group.theta_deg == pytest.approx(109.4712206, abs=1e-6)

    planar = group.positions(90.0)
    c = mol.coords[0]
    # All three H lie in the plane through C perpendicular to the C->Br axis.
    axis = np.array([0.0, 0.0, 1.0])
    assert np.allclose((planar - c) @ axis, 0.0, atol=1e-9)
    # Equilateral triangle centred on the reference atom.
    sides = [
        float(np.linalg.norm(planar[i] - planar[j]))
        for i, j in ((0, 1), (1, 2), (0, 2))
    ]
    assert max(sides) - min(sides) < 1e-9
    assert np.allclose(planar.mean(axis=0), c, atol=1e-9)
    # H-C-H opened to 120 deg at the planar state.
    u, v = planar[0] - c, planar[1] - c
    cos = float(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v))
    assert np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))) == pytest.approx(120.0)


def test_umbrella_keeps_reference_interest_distances_constant():
    mol = _methyl_halide()
    group = UmbrellaGroup.capture(mol, 0, 1, [2, 3, 4])
    c = mol.coords[0]
    for angle in (0.0, 30.0, 70.53, 90.0, 109.47, 150.0, 180.0):
        radii = np.linalg.norm(group.positions(angle) - c, axis=1)
        assert np.allclose(radii, group.radii, atol=1e-9)
        assert np.allclose(radii, _CH_RADIUS, atol=1e-9)


def test_umbrella_over_rotation_collapses_on_the_axis():
    mol = _methyl_halide()
    group = UmbrellaGroup.capture(mol, 0, 1, [2, 3, 4])
    axis = np.array([0.0, 0.0, 1.0])

    collapsed = group.positions(0.0)
    spread = np.max(np.linalg.norm(collapsed - collapsed[0], axis=1))
    assert spread < 1e-9
    assert np.allclose(collapsed[0], mol.coords[0] + _CH_RADIUS * axis, atol=1e-9)

    # The opposite extreme collapses on the other side of the reference.
    other = group.positions(180.0)
    assert np.max(np.linalg.norm(other - other[0], axis=1)) < 1e-9
    assert np.allclose(other[0], mol.coords[0] - _CH_RADIUS * axis, atol=1e-9)


def test_umbrella_sweeps_reactant_transition_state_and_product():
    mol = _methyl_halide()
    group = UmbrellaGroup.capture(mol, 0, 1, [2, 3, 4])
    original = mol.coords[[2, 3, 4]].copy()

    # At the captured angle the atoms are back where they started.
    assert np.allclose(group.positions(group.theta_deg), original, atol=1e-9)

    # The product (inverted) state only flips the coordinate along the axis.
    product = group.positions(180.0 - group.theta_deg)
    assert np.allclose(product[:, :2], original[:, :2], atol=1e-9)
    assert np.allclose(product[:, 2], -original[:, 2], atol=1e-9)


def test_umbrella_apply_writes_the_molecule_in_place():
    mol = _methyl_halide()
    group = UmbrellaGroup.capture(mol, 0, 1, [2, 3, 4])
    group.apply(mol, 90.0)
    assert np.allclose(mol.coords[[2, 3, 4]], group.positions(90.0))
    # Reference and axis atoms never move.
    assert np.allclose(mol.coords[[0, 1]], [[0, 0, 0], [0, 0, 1.94]])


def test_angle_from_displacement_uses_the_axis_component():
    mol = _methyl_halide()
    group = UmbrellaGroup.capture(mol, 0, 1, [2, 3, 4])
    axis = np.array([0.0, 0.0, 1.0])
    # From the planar state, pushing along +axis reduces theta to 0...
    assert group.angle_from_displacement(90.0, _CH_RADIUS * axis) == pytest.approx(
        0.0, abs=1e-6
    )
    # ...and along -axis reaches the opposite collapse.
    assert group.angle_from_displacement(
        90.0, -_CH_RADIUS * axis
    ) == pytest.approx(180.0, abs=1e-6)
    # A displacement perpendicular to the axis does not change the angle.
    assert group.angle_from_displacement(
        90.0, np.array([0.3, 0.0, 0.0])
    ) == pytest.approx(90.0, abs=1e-6)


def test_umbrella_rejects_degenerate_definitions():
    mol = _methyl_halide()
    with pytest.raises(ValueError):
        UmbrellaGroup.capture(mol, 0, 1, [])
    with pytest.raises(ValueError):
        UmbrellaGroup.capture(mol, 0, 0, [2, 3, 4])
    with pytest.raises(ValueError):
        UmbrellaGroup.capture(mol, 0, 1, [0, 2])
    with pytest.raises(ValueError):
        UmbrellaGroup.capture(mol, 0, 1, [1, 2])
    with pytest.raises(ValueError):
        UmbrellaGroup.capture(mol, 0, 99, [2, 3, 4])


# -- model --------------------------------------------------------------------


def test_molecule_bonds_field_is_validated_and_copied():
    mol = Molecule(
        symbols=["O", "H", "H"],
        coords=np.zeros((3, 3)),
        bonds=np.array([[1, 0], [0, 2]], dtype=np.int64),
    )
    assert mol.bonds.tolist() == [[0, 1], [0, 2]]

    clone = mol.copy()
    assert clone.bonds is not mol.bonds
    clone.bonds[0] = [0, 2]
    assert mol.bonds[0].tolist() == [0, 1]

    with pytest.raises(ValueError):
        Molecule(
            symbols=["O", "H"],
            coords=np.zeros((2, 3)),
            bonds=np.array([[0, 5]], dtype=np.int64),
        )
