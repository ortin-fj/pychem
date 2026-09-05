"""Trajectory data model: a sequence of molecular frames.

All frames share the same atom order and symbols (topology), so they can be
stored as a single array of shape (n_frames, n_atoms, 3). Optional per-frame
energies and times allow playback and energy-profile plots later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from pychem.core.molecule import Molecule


@dataclass
class Trajectory:
    """Ordered molecular frames with constant topology."""

    symbols: list[str]
    frames: np.ndarray  # shape (n_frames, n_atoms, 3), Å
    name: str = ""
    energies: np.ndarray | None = None  # shape (n_frames,), optional
    times: np.ndarray | None = None  # shape (n_frames,), optional
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frames = np.asarray(self.frames, dtype=float)
        n_atoms = len(self.symbols)
        if frames.ndim != 3 or frames.shape[2] != 3 or frames.shape[1] != n_atoms:
            raise ValueError(
                f"frames must have shape (n_frames, {n_atoms}, 3); got {frames.shape}"
            )
        if frames.shape[0] == 0:
            raise ValueError("a trajectory needs at least one frame")
        self.frames = frames
        for label, arr in (("energies", self.energies), ("times", self.times)):
            if arr is not None:
                arr = np.asarray(arr, dtype=float)
                if arr.shape != (frames.shape[0],):
                    raise ValueError(
                        f"{label} must have shape ({frames.shape[0]},); got {arr.shape}"
                    )
                setattr(self, label, arr)

    @property
    def n_frames(self) -> int:
        return self.frames.shape[0]

    @property
    def n_atoms(self) -> int:
        return self.frames.shape[1]

    # -- frames -------------------------------------------------------------

    def get_frame(self, index: int) -> Molecule:
        """Return frame ``index`` as a :class:`Molecule`."""
        if not 0 <= index < self.n_frames:
            raise IndexError(f"frame index out of range: {index}")
        return self._molecule_from(index, self.frames[index])

    def iter_frames(self) -> Iterator[Molecule]:
        for i in range(self.n_frames):
            yield self.get_frame(i)

    def _molecule_from(self, index: int, coords: np.ndarray) -> Molecule:
        metadata: dict[str, object] = dict(self.metadata)
        if self.energies is not None:
            metadata["energy"] = float(self.energies[index])
        if self.times is not None:
            metadata["time"] = float(self.times[index])
        return Molecule(
            symbols=list(self.symbols),
            coords=np.array(coords, dtype=float, copy=True),
            name=f"{self.name} [frame {index}]",
            metadata=metadata,
        )

    # -- sampling / interpolation -------------------------------------------

    def sample(self, position: float, *, interpolate: bool = True) -> Molecule:
        """Return the pose at a continuous ``position`` (in frame units).

        With ``interpolate=True`` the atom coordinates are linearly
        interpolated between the surrounding frames; otherwise the pose snaps
        to the frame at ``floor(position)``. The position is clamped to
        ``[0, n_frames - 1]``.
        """
        n = self.n_frames
        if n == 1:
            return self.get_frame(0)
        pos = min(max(float(position), 0.0), float(n - 1))
        lo = int(np.floor(pos))
        hi = min(lo + 1, n - 1)

        if not interpolate or hi == lo:
            return self.get_frame(lo)

        frac = pos - lo
        coords = self.frames[lo] * (1.0 - frac) + self.frames[hi] * frac
        return self._molecule_from(lo, coords)

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_molecules(cls, molecules: list[Molecule], name: str = "") -> "Trajectory":
        """Build a trajectory from a list of equally-topologied molecules."""
        if not molecules:
            raise ValueError("cannot build a trajectory from an empty list")
        first = molecules[0]
        symbols = list(first.symbols)
        n_atoms = len(symbols)

        coords = np.empty((len(molecules), n_atoms, 3), dtype=float)
        energies: list[float] = []
        times: list[float] = []

        for i, mol in enumerate(molecules):
            if list(mol.symbols) != symbols:
                raise ValueError(
                    f"molecule {i} has different symbols/topology than frame 0"
                )
            coords[i] = mol.coords
            if "energy" in mol.metadata:
                energies.append(float(mol.metadata["energy"]))
            if "time" in mol.metadata:
                times.append(float(mol.metadata["time"]))

        traj = cls(
            symbols=symbols,
            frames=coords,
            name=name or first.name,
            metadata=dict(first.metadata),
        )
        if len(energies) == len(molecules):
            traj.energies = np.asarray(energies)
        if len(times) == len(molecules):
            traj.times = np.asarray(times)
        return traj

    def __repr__(self) -> str:  # pragma: no cover - debugging
        return (
            f"Trajectory(name={self.name!r}, n_frames={self.n_frames}, "
            f"n_atoms={self.n_atoms})"
        )
