"""Reader for the Protein Data Bank (PDB) format (structures only).

``ATOM``/``HETATM`` records become atoms of a single :class:`Molecule`.
Multi-model files (NMR bundles, PDB "trajectories") are intentionally read
as a *single structure*: only the first ``MODEL`` is used and any later
model is ignored. Coordinates are in ångström, as in the PDB standard.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pychem.core.elements import element
from pychem.core.molecule import Molecule
from pychem.io.registry import Format, _registry

_EXTENSIONS = ("pdb", "ent")

# Fixed-width columns of a standard PDB ATOM/HETATM record (0-based slices).
_RECORD = slice(0, 6)
_ATOM_NAME = slice(12, 16)  # fallback element source
_X = slice(30, 38)
_Y = slice(38, 46)
_Z = slice(46, 54)
_ELEMENT = slice(76, 78)


def _record_type(line: str) -> str:
    """Uppercased record name of a PDB line (or ``""`` for short lines)."""
    return line[_RECORD].strip().upper()


def _element_symbol(line: str) -> str:
    """Element of an atom record.

    The dedicated element column (77-78) is used when present; otherwise the
    element is inferred from the leading letters of the atom name.
    """
    symbol = line[_ELEMENT].strip()
    if not symbol:
        name = line[_ATOM_NAME].strip()
        symbol = "".join(ch for ch in name if ch.isalpha())[:2]
    return element(symbol).symbol


def _coordinates(line: str) -> list[float]:
    """Cartesian coordinates of an atom record (fixed columns, then fallback).

    Files are normally column-aligned; when they are not, the last three
    numeric tokens of the line are used instead.
    """
    try:
        return [
            float(line[_X]),
            float(line[_Y]),
            float(line[_Z]),
        ]
    except ValueError:
        numbers: list[float] = []
        for token in line.split():
            try:
                numbers.append(float(token))
            except ValueError:
                continue
        if len(numbers) < 3:
            raise ValueError(
                f"Could not read coordinates from PDB record: {line!r}"
            )
        return numbers[-3:]


def _first_model_records(lines: list[str]) -> list[str]:
    """Atom records of the structure (first ``MODEL`` only, if any)."""
    records: list[str] = []
    model_seen = False
    for line in lines:
        record = _record_type(line)
        if record == "MODEL":
            if model_seen:
                break  # already past the first model
            model_seen = True
        elif record == "ENDMDL":
            if model_seen:
                break  # end of the first model: structure is complete
        elif record in ("ATOM", "HETATM"):
            records.append(line)
    return records


def read_pdb(path: str | Path) -> Molecule:
    """Read a PDB file as a single structure (:class:`Molecule`)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    records = _first_model_records(text.splitlines())
    if not records:
        raise ValueError(f"No ATOM/HETATM records found in {path}")

    symbols: list[str] = []
    coords = np.empty((len(records), 3), dtype=float)
    for i, line in enumerate(records):
        symbols.append(_element_symbol(line))
        coords[i] = _coordinates(line)

    molecule = Molecule(symbols=symbols, coords=coords, name=path.stem)
    return molecule


_registry.register(
    Format(
        extensions=_EXTENSIONS,
        description="Protein Data Bank (structures; first model only)",
        read=read_pdb,
    )
)
