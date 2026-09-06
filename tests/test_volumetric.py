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
