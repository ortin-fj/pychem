"""Tests for the Trajectory model and the XYZ trajectory reader."""

from pathlib import Path

import numpy as np
import pytest

from pychem.core.trajectory import Trajectory
from pychem.io import get_reader, read_xyz_trajectory
from pychem.io.xyz import read_xyz_frames

DATA = Path(__file__).parent / "data" / "xyz"


def test_read_xyz_trajectory_basics():
    traj = read_xyz_trajectory(DATA / "water_traj.xyz")
    assert traj.n_frames == 2
    assert traj.n_atoms == 3
    assert traj.name == "water_traj"
    assert traj.get_frame(0).symbols == ["O", "H", "H"]
    # get_frame returns independent copies
    f = traj.get_frame(0)
    f.coords[0, 0] = 999.0
    assert traj.frames[0, 0, 0] != 999.0


def test_trajectory_from_frames_consistent():
    frames = read_xyz_frames(DATA / "water_traj.xyz")
    traj = Trajectory.from_molecules(frames, name="w")
    assert traj.n_frames == 2


def test_trajectory_from_frames_mismatch_raises():
    water = get_reader(DATA / "water.xyz")
    benzene = get_reader(DATA / "benzene.xyz")
    with pytest.raises(ValueError):
        Trajectory.from_molecules([water, benzene])


def test_sample_interpolation_midpoint():
    traj = read_xyz_trajectory(DATA / "water_traj.xyz")
    mid = traj.sample(0.5, interpolate=True)
    expected = (traj.frames[0] + traj.frames[1]) / 2.0
    assert np.allclose(mid.coords, expected)


def test_sample_no_interpolation_steps_to_floor():
    traj = read_xyz_trajectory(DATA / "water_traj.xyz")
    assert np.allclose(
        traj.sample(0.9, interpolate=False).coords, traj.frames[0]
    )
    assert np.allclose(
        traj.sample(1.0, interpolate=False).coords, traj.frames[1]
    )


def test_sample_clamps_to_range():
    traj = read_xyz_trajectory(DATA / "water_traj.xyz")
    assert np.allclose(traj.sample(-3.0).coords, traj.frames[0])
    assert np.allclose(traj.sample(50.0).coords, traj.frames[1])


def test_single_frame_trajectory():
    neon = get_reader(DATA / "ne.xyz")
    traj = Trajectory.from_molecules([neon])
    assert traj.n_frames == 1
    assert np.allclose(traj.sample(0.0).coords, neon.coords)


def test_shape_validation_raises():
    # atoms per frame do not match the symbols
    with pytest.raises(ValueError):
        Trajectory(symbols=["H", "H"], frames=np.zeros((2, 3, 3)))
    # last dimension is not 3
    with pytest.raises(ValueError):
        Trajectory(symbols=["H", "H"], frames=np.zeros((2, 2, 2)))
    # empty trajectory
    with pytest.raises(ValueError):
        Trajectory(symbols=["H", "H"], frames=np.zeros((0, 2, 3)))


def test_energies_and_times_validation():
    with pytest.raises(ValueError):
        Trajectory(
            symbols=["H", "H"],
            frames=np.zeros((2, 2, 3)),
            energies=np.zeros(3),  # wrong length
        )
