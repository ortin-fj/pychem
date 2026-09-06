"""Tests for the pure-geometry measurement helpers (Bonds/Angles/Dihedrals)."""

import numpy as np
import pytest

from pychem.core.geometry import (
    angle_degrees,
    bond_length,
    dihedral_degrees,
)

# Water geometry (matches tests/data/xyz/water.xyz).
_O = (0.0, 0.0, 0.1173)
_H1 = (0.0, 0.7572, -0.4692)
_H2 = (0.0, -0.7572, -0.4692)


def test_bond_length_water_oh():
    assert bond_length(_O, _H1) == pytest.approx(0.958, abs=0.002)
    assert bond_length(_O, _H2) == pytest.approx(0.958, abs=0.002)


def test_angle_water_hoh():
    assert angle_degrees(_H1, _O, _H2) == pytest.approx(104.48, abs=0.1)


def test_angle_right_angle():
    assert angle_degrees([1, 0, 0], [0, 0, 0], [0, 1, 0]) == pytest.approx(90.0)
    assert angle_degrees([1, 0, 0], [0, 0, 0], [-1, 0, 0]) == pytest.approx(180.0)


def test_dihedral_sign_and_magnitude():
    # A dihedral of -90 degrees (p4 lifted out of the plane).
    a, b, c = (0, 0, 0), (1, 0, 0), (1, 1, 0)
    d_up = (1, 1, 1)
    assert dihedral_degrees(a, b, c, d_up) == pytest.approx(-90.0, abs=0.5)


def test_dihedral_trans_is_180():
    # Coplanar atoms on opposite sides of the central bond -> 180 degrees.
    a, b, c, d = (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)
    assert abs(dihedral_degrees(a, b, c, d)) == pytest.approx(180.0, abs=0.5)


def test_dihedral_cis_is_0():
    # Coplanar atoms on the same side of the central bond -> 0 degrees.
    a, b, c, d = (0, 0, 1), (0, 0, 0), (1, 0, 0), (1, 0, -1)
    assert dihedral_degrees(a, b, c, d) == pytest.approx(0.0, abs=0.5)


def test_anchors_are_finite_and_near_the_atoms():
    from pychem.core.geometry import (
        angle_anchor,
        bond_anchor,
        dihedral_anchor,
    )

    for anchor in (
        bond_anchor(_O, _H1),
        angle_anchor(_H1, _O, _H2),
        dihedral_anchor((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)),
    ):
        assert np.all(np.isfinite(anchor))
