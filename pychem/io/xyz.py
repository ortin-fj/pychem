"""Reader and writer for the XYZ format (and extended XYZ).

Supports:
- Classic XYZ (count, comment, ``S x y z`` lines).
- Extended XYZ with annotations in the comment line (e.g.
  ``Lattice="..."``), kept in ``Molecule.metadata``.
- Multi-frame files: molecule reading returns the first frame; trajectory
  reading will arrive in the trajectory module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pychem.core.molecule import Molecule
from pychem.core.trajectory import Trajectory
from pychem.io.registry import Format, _registry

_EXTENSIONS = ("xyz",)


def _parse_frame(raw: list[str], start: int) -> tuple[Molecule | None, int]:
    """Parse one frame starting at line ``start`` of ``raw``.

    Returns (molecule, index of the next unprocessed line) or (None, start)
    if there is no valid header at that position. The comment line may be
    empty; blank lines separating frames are ignored.
    """
    n = len(raw)
    header = raw[start].strip() if start < n else ""
    if not header.isdigit():
        return None, start

    n_atoms = int(header)
    if n_atoms < 1:
        raise ValueError(f"Invalid XYZ frame: n_atoms={n_atoms}")
    # Comment line + n_atoms atom lines
    if start + 1 + n_atoms >= len(raw):
        raise ValueError(
            f"Invalid XYZ frame: n_atoms={n_atoms}, not enough lines"
        )

    comment = raw[start + 1].strip()
    symbols: list[str] = []
    coords = np.empty((n_atoms, 3), dtype=float)

    for i in range(n_atoms):
        parts = raw[start + 2 + i].split()
        if len(parts) < 4:
            raise ValueError(
                f"Malformed atom line (atom {i + 1}): "
                f"{raw[start + 2 + i]!r}"
            )
        symbol = parts[0].capitalize()
        symbols.append(symbol)
        try:
            coords[i] = [float(v) for v in parts[1:4]]
        except ValueError as exc:
            raise ValueError(
                f"Non-numeric coordinates in atom {i + 1}: "
                f"{raw[start + 2 + i]!r}"
            ) from exc

    metadata: dict[str, object] = {}
    if comment:
        metadata["comment"] = comment
        # key="value" annotations typical of extended XYZ.
        for token in comment.split():
            if "=" in token:
                key, value = token.split("=", 1)
                metadata[key.strip().lower()] = value.strip().strip('"')

    molecule = Molecule(symbols=symbols, coords=coords, name="", metadata=metadata)
    return molecule, start + 2 + n_atoms


def _parse_all(text: str) -> list[Molecule]:
    """Parse all frames in the text (useful for trajectories)."""
    raw = text.splitlines()
    frames: list[Molecule] = []
    pos = 0
    while pos < len(raw):
        if not raw[pos].strip():
            pos += 1
            continue
        molecule, pos = _parse_frame(raw, pos)
        if molecule is None:
            break
        frames.append(molecule)
    return frames


def read_xyz(path: str | Path) -> Molecule:
    """Read the first frame of an XYZ file as a :class:`Molecule`."""
    frames = _parse_all(Path(path).read_text(encoding="utf-8"))
    if not frames:
        raise ValueError(f"No valid XYZ frame found in {path}")
    molecule = frames[0]
    molecule.name = Path(path).stem
    return molecule


def read_xyz_frames(path: str | Path) -> list[Molecule]:
    """Read all frames of an XYZ file (independent molecules)."""
    frames = _parse_all(Path(path).read_text(encoding="utf-8"))
    if not frames:
        raise ValueError(f"No valid XYZ frame found in {path}")
    return frames


def read_xyz_trajectory(path: str | Path) -> Trajectory:
    """Read all frames of an XYZ file as a :class:`Trajectory`.

    Raises ``ValueError`` if the frames do not share the same topology
    (symbols/order), which is required for a trajectory.
    """
    frames = read_xyz_frames(path)
    return Trajectory.from_molecules(frames, name=Path(path).stem)


def write_xyz(path: str | Path, molecule: Molecule, *, comment: str = "") -> Path:
    """Write a molecule in classic XYZ format. Returns the path."""
    path = Path(path)
    if not comment:
        comment = molecule.name or molecule.formula
    lines = [str(molecule.n_atoms), comment]
    for symbol, pos in zip(molecule.symbols, molecule.coords):
        lines.append(f"{symbol:<3s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


_registry.register(
    Format(
        extensions=_EXTENSIONS,
        description="XYZ and extended XYZ (structures and trajectories)",
        read=read_xyz,
        write=write_xyz,
    )
)
