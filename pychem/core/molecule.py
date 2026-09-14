"""Data models for molecules and trajectories.

The core uses flat objects (dataclasses + NumPy arrays) that depend neither
on the source format nor on the rendering engine. This lets different readers
produce the same models and the viewer consume them uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from pychem.core.elements import Element, element


@dataclass
class Atom:
    """A single atom with its cartesian position (Å)."""

    symbol: str
    position: np.ndarray  # shape (3,), Å
    index: int = 0

    @property
    def element(self) -> Element:
        return element(self.symbol)

    @property
    def x(self) -> float:
        return float(self.position[0])

    @property
    def y(self) -> float:
        return float(self.position[1])

    @property
    def z(self) -> float:
        return float(self.position[2])


def _hill_formula(counts: dict[str, int]) -> str:
    """Empirical formula in Hill order (C, H, then the rest alphabetically)."""
    order = []
    for key in ("C", "H"):
        if key in counts:
            order.append(key)
    for key in sorted(counts):
        if key not in order:
            order.append(key)

    parts: list[str] = []
    for key in order:
        n = counts[key]
        parts.append(key if n == 1 else f"{key}{n}")
    return "".join(parts)


@dataclass
class Molecule:
    """A set of atoms forming a molecular structure or a frame.

    ``bonds`` is an optional explicit connectivity as an ``(m, 2)`` array of
    atom indices with ``i < j``. ``None`` means "no explicit connectivity":
    consumers (e.g. the renderer) guess it from the geometry. Editing the
    structure in the GUI turns it into an explicit list, which can be
    authored with :mod:`pychem.core.editing`.
    """

    symbols: list[str]
    coords: np.ndarray  # shape (n_atoms, 3), Å
    name: str = ""
    charge: int = 0
    multiplicity: int = 1
    metadata: dict[str, object] = field(default_factory=dict)
    bonds: np.ndarray | None = None  # (m, 2) int, i < j, or None (auto)

    def __post_init__(self) -> None:
        coords = np.asarray(self.coords, dtype=float)
        n_atoms = len(self.symbols)
        if coords.ndim != 2 or coords.shape != (n_atoms, 3):
            raise ValueError(
                f"coords must have shape ({n_atoms}, 3); got {coords.shape}"
            )
        self.coords = coords
        if self.bonds is not None:
            arr = np.asarray(self.bonds, dtype=np.int64).reshape(-1, 2)
            if arr.size:
                arr = np.sort(arr, axis=1)
                if (arr[:, 0] < 0).any() or (arr[:, 1] >= n_atoms).any():
                    raise ValueError("bond indices out of range")
            self.bonds = arr

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)

    @property
    def positions(self) -> np.ndarray:
        """Cartesian positions with shape (n_atoms, 3)."""
        return self.coords

    @property
    def formula(self) -> str:
        counts: dict[str, int] = {}
        for symbol in self.symbols:
            counts[symbol] = counts.get(symbol, 0) + 1
        return _hill_formula(counts)

    def iter_atoms(self) -> Iterator[Atom]:
        """Iterate over the atoms as :class:`Atom` objects."""
        for i, (symbol, pos) in enumerate(zip(self.symbols, self.coords)):
            yield Atom(symbol=symbol, position=pos, index=i)

    def get_atom(self, index: int) -> Atom:
        if not 0 <= index < self.n_atoms:
            raise IndexError(f"atom index out of range: {index}")
        return Atom(self.symbols[index], self.coords[index], index=index)

    def copy(self, *, deep: bool = True) -> "Molecule":
        """Copy the molecule. Copies the coordinates by default."""
        coords = self.coords.copy() if deep else self.coords
        bonds = None
        if self.bonds is not None:
            bonds = self.bonds.copy() if deep else self.bonds
        return Molecule(
            symbols=list(self.symbols),
            coords=coords,
            name=self.name,
            charge=self.charge,
            multiplicity=self.multiplicity,
            metadata=dict(self.metadata),
            bonds=bonds,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging
        return (
            f"Molecule(name={self.name!r}, formula={self.formula}, "
            f"n_atoms={self.n_atoms})"
        )
