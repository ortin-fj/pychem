"""Volumetric data loading: Gaussian cube and Molden orbitals.

- ``.cube``: the grid is stored directly (one or more scalar fields).
- ``.molden``: ORCA-style Molden files store basis + MO coefficients, so
  each orbital is *evaluated on a grid* on demand (the current basis here is
  limited to s/p shells, which covers the test data).

Both are wrapped in a :class:`VolumetricFile` that provides the molecule (to
show it like a normal structure) plus a list of orbitals that can be computed
lazily and rendered as isosurfaces.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from pychem.core.elements import symbol_from_number
from pychem.core.molecule import Molecule
from pychem.core.volume import VolumeData

BOHR = 0.529177210903  # Å per Bohr

# default isosurface level as a fraction of the field maximum
DEFAULT_ISO_FRACTION = 0.08


@dataclass
class VolumetricFile:
    """A structure plus lazily-available volumetric fields (orbitals)."""

    molecule: Molecule
    labels: list[str]
    compute: Callable[[int], VolumeData]  # compute(i) -> VolumeData
    name: str = ""
    metadata: dict[str, object] = field(default_factory=dict)
    # Present only for Molden files: the parsed (s/p) basis that lets the GUI
    # also visualize individual atomic orbitals.
    basis: object | None = None

    def volume(self, index: int) -> VolumeData:
        return self.compute(index)

    def __repr__(self) -> str:  # pragma: no cover - debugging
        return f"VolumetricFile(name={self.name!r}, fields={len(self.labels)})"


# ---------------------------------------------------------------------------
# Gaussian cube
# ---------------------------------------------------------------------------


def _read_numbers(text: str) -> list[float]:
    return [float(tok) for tok in text.split()]


def read_cube(path: str | Path) -> tuple[Molecule, list[VolumeData]]:
    """Read a Gaussian cube file -> (molecule, list of scalar fields).

    Gaussian/ORCA cube files store the atom positions, the grid origin and
    the axis vectors in atomic units (Bohr). They are converted to Ångstrom
    here so ``Molecule``/``VolumeData`` keep the project-wide Å convention
    (the Molden reader applies the same conversion to its ``[Atoms] AU``
    block). The scalar field values themselves are left untouched: they are
    orbital/density values, not distances.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) < 7:
        raise ValueError(f"Invalid cube file (too short): {path}")

    comment2 = lines[1].strip()
    counts = lines[2].split()
    natoms = abs(int(float(counts[0])))
    origin = np.array([float(v) for v in counts[1:4]]) * BOHR

    dims: list[int] = []
    spacing = np.zeros(3)
    for i, line in enumerate(lines[3:6]):
        parts = line.split()
        dims.append(int(float(parts[0])))
        spacing[i] = abs(float(parts[i + 1])) * BOHR  # diagonal axis vector

    symbols: list[str] = []
    coords = np.empty((natoms, 3))
    for a in range(natoms):
        parts = lines[6 + a].split()
        number = int(float(parts[0]))
        coords[a] = [float(v) for v in parts[2:5]]
        symbols.append(symbol_from_number(number) or "X")
    coords *= BOHR  # the atomic coordinates are given in Bohr

    n_values = int(np.prod(dims))
    numbers = [
        float(tok)
        for line in lines[6 + natoms :]
        for tok in line.split()
    ]
    if len(numbers) < n_values:
        raise ValueError(
            f"Cube file truncated: expected {n_values} values, got {len(numbers)}"
        )

    # One or more scalar fields appended after the header (each block is
    # exactly nx*ny*nz values; a small trailing remainder is ignored).
    n_blocks = len(numbers) // n_values
    raw_all = np.asarray(numbers[: n_blocks * n_values])

    molecule = Molecule(
        symbols=symbols,
        coords=coords,
        name=Path(path).stem,
    )
    volumes: list[VolumeData] = []
    for block in range(n_blocks):
        raw = raw_all[block * n_values : (block + 1) * n_values]
        values = raw.reshape(tuple(dims), order="F")  # x fastest in cube files
        label = comment2 or "MO 1"
        if n_blocks > 1:
            label = f"{label} ({block + 1})" if block else comment2 or "MO 1"
        volumes.append(
            VolumeData(
                values=values,
                origin=origin,
                spacing=spacing,
                label=label,
                metadata={"block": block},
            )
        )
    return molecule, volumes


# ---------------------------------------------------------------------------
# Molden (s/p basis) -> orbitals evaluated on a grid
# ---------------------------------------------------------------------------


@dataclass
class _Shell:
    atom: int
    l: int
    exps: np.ndarray
    coeffs: np.ndarray


@dataclass
class _AO:
    shell: int
    axis: int  # -1 for s, 0/1/2 for px/py/pz


@dataclass
class _MoldenData:
    molecule: Molecule
    shells: list[_Shell]
    aos: list[_AO]
    mo_labels: list[str]
    mo_coeffs: list[np.ndarray]

    @property
    def n_basis(self) -> int:
        return len(self.aos)


@dataclass(frozen=True)
class AtomicOrbital:
    """One atomic basis function (AO) available for visualization.

    ``index`` is the position of the function in the basis AO list, so a unit
    coefficient at that position selects exactly this function in the grid
    evaluator. ``axis`` is -1 for an s function and 0/1/2 for the px/py/pz
    cartesian components (the angular momenta the evaluator supports).
    """

    index: int
    atom: int
    l: int
    axis: int
    n_prim: int
    shell: int


_ANGULAR = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}
_N_FUNCS = {"s": 1, "p": 3, "d": 6, "f": 10, "g": 15}


def _parse_shell_type(letter: str) -> tuple[int, int]:
    """Return (l, number of cartesian functions)."""
    if letter not in _ANGULAR:
        raise ValueError(f"Unsupported Molden shell type '{letter}'")
    return _ANGULAR[letter], _N_FUNCS[letter]


def read_molden(path: str | Path) -> _MoldenData:
    """Parse a Molden file: atoms, s/p basis and MO coefficients."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()

    section = ""
    atoms_symbols: list[str] = []
    atoms_coords: list[list[float]] = []
    atoms_au = False
    shells: list[_Shell] = []
    mo_syms: list[str] = []
    mo_coeffs: list[list[tuple[int, float]]] = []
    current_mo: list[tuple[int, float]] | None = None
    current_mo_sym = ""

    shell_atom = 0
    pending: _Shell | None = None  # while collecting primitives
    prim_lines_left = 0
    shell_exps: list[float] = []
    shell_co: list[float] = []

    def flush_pending() -> None:
        nonlocal pending, prim_lines_left, shell_exps, shell_co
        if pending is not None:
            shells.append(
                _Shell(
                    atom=pending.atom,
                    l=pending.l,
                    exps=np.asarray(shell_exps),
                    coeffs=np.asarray(shell_co),
                )
            )
        pending = None
        prim_lines_left = 0
        shell_exps = []
        shell_co = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("["):
            flush_pending()
            section = stripped
            continue
        if not stripped:
            continue

        if section == "[Atoms] AU" or section == "[Atoms] Angstrom":
            parts = stripped.split()
            if len(parts) >= 6:
                symbol = parts[0]
                try:
                    number = int(float(parts[2]))
                except ValueError:
                    number = symbol_from_number(symbol) or 0
                atoms_symbols.append(symbol)
                atoms_coords.append([float(v) for v in parts[3:6]])
                if section.endswith("AU"):
                    atoms_au = True
        elif section == "[GTO]":
            # Atom block header: "N" or "N 0" (ORCA writes the 1-based atomic
            # index followed by a second integer field). It is never matched
            # while the primitives of a shell are still pending (those are two
            # floats), which keeps it unambiguous.
            parts = stripped.split()
            is_block_header = (
                bool(parts)
                and parts[0].isdigit()
                and (
                    len(parts) == 1
                    or re.fullmatch(r"[+-]?\d+", parts[1]) is not None
                )
                and (pending is None or prim_lines_left == 0)
            )
            if is_block_header:
                flush_pending()
                shell_atom = int(parts[0]) - 1
                continue
            m = re.match(r"^([spdfgh])\s+(\d+)\s+([\d.eE+-]+)\s*$", stripped)
            if m:
                flush_pending()
                l, _ = _parse_shell_type(m.group(1))
                pending = _Shell(atom=shell_atom, l=l, exps=np.array([]), coeffs=np.array([]))
                prim_lines_left = int(m.group(2))
                continue
            if pending is not None and prim_lines_left > 0:
                ex, cf = stripped.split()
                shell_exps.append(float(ex))
                shell_co.append(float(cf))
                prim_lines_left -= 1
        elif section == "[MO]":
            if stripped.startswith("Sym="):
                flush_mo = True
                if current_mo is not None:
                    mo_syms.append(current_mo_sym)
                    mo_coeffs.append(current_mo)
                current_mo_sym = stripped.split("=", 1)[1].strip()
                current_mo = []
                continue
            m = re.match(r"^(\d+)\s+([-\d.eE+]+)$", stripped)
            if m and current_mo is not None:
                current_mo.append((int(m.group(1)) - 1, float(m.group(2))))

    flush_pending()
    if current_mo is not None:
        mo_syms.append(current_mo_sym)
        mo_coeffs.append(current_mo)

    # Only s/p shells are supported by the evaluator.
    for sh in shells:
        if sh.l > 1:
            raise ValueError(
                "This Molden evaluator only supports s and p basis functions"
            )

    # Build the AO list following Molden ordering (s then px,py,pz).
    aos: list[_AO] = []
    for si, sh in enumerate(shells):
        if sh.l == 0:
            aos.append(_AO(shell=si, axis=-1))
        else:
            for ax in range(3):
                aos.append(_AO(shell=si, axis=ax))

    if atoms_au:
        coords = np.asarray(atoms_coords) * BOHR
    else:
        coords = np.asarray(atoms_coords)

    molecule = Molecule(
        symbols=atoms_symbols,
        coords=coords,
        name=Path(path).stem,
    )

    # MO coefficient vectors, densified to nbasis
    nbasis = len(aos)
    mo_vectors: list[np.ndarray] = []
    for block in mo_coeffs:
        vec = np.zeros(nbasis)
        for idx, val in block:
            if 0 <= idx < nbasis:
                vec[idx] = val
        mo_vectors.append(vec)

    return _MoldenData(
        molecule=molecule,
        shells=shells,
        aos=aos,
        mo_labels=mo_syms,
        mo_coeffs=mo_vectors,
    )


def _evaluate_molden_orbital_v2(
    data: _MoldenData, coeff: np.ndarray, lo: np.ndarray, hi: np.ndarray, n: int
) -> VolumeData:
    axes = [np.linspace(lo[d], hi[d], n) for d in range(3)]
    grid = np.meshgrid(*axes, indexing="ij")
    X, Y, Z = (g.ravel() for g in grid)
    out = np.zeros_like(X)
    coords = data.molecule.coords

    for i, ao in enumerate(data.aos):
        sh = data.shells[ao.shell]
        c = float(coeff[i])
        if abs(c) < 1e-12:
            continue
        cx, cy, cz = coords[sh.atom]
        dx = X - cx
        dy = Y - cy
        dz = Z - cz
        r2 = dx * dx + dy * dy + dz * dz
        prim_sum = np.zeros_like(X)
        for expo, cc in zip(sh.exps, sh.coeffs):
            a = expo / (BOHR * BOHR)
            prim = cc * (2.0 * a / math.pi) ** 0.75
            if sh.l == 1:
                prim *= (4.0 * a) ** 0.5
            prim_sum += prim * np.exp(-a * r2)
        if sh.l == 0:
            out += c * prim_sum
        else:
            comp = (dx, dy, dz)[ao.axis]
            out += c * prim_sum * comp

    return VolumeData(
        values=out.reshape((n, n, n), order="C"),
        origin=np.asarray([axes[0][0], axes[1][0], axes[2][0]]),
        spacing=np.asarray(
            [
                axes[0][1] - axes[0][0],
                axes[1][1] - axes[1][0],
                axes[2][1] - axes[2][0],
            ]
        ),
        label="orbital",
    )


# ---------------------------------------------------------------------------
# Atomic orbitals (basis functions of an individual atom)
# ---------------------------------------------------------------------------

_L_NAME = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g"}
_AXIS_NAME = ("x", "y", "z")


def atomic_orbitals_for_atom(
    basis: _MoldenData, atom: int
) -> dict[int, list[AtomicOrbital]]:
    """Basis functions centered on ``atom``, grouped by angular momentum.

    Returns ``{l: [AtomicOrbital, ...]}`` with ``l`` in ascending order; the
    AOs of a shell keep the cartesian file order (s, or px/py/pz).
    """
    from collections import defaultdict

    by_shell: dict[int, list[int]] = defaultdict(list)
    for idx, ao in enumerate(basis.aos):
        by_shell[ao.shell].append(idx)

    groups: dict[int, list[AtomicOrbital]] = defaultdict(list)
    for sh_idx, sh in enumerate(basis.shells):
        if sh.atom != atom:
            continue
        for ao_idx in by_shell.get(sh_idx, []):
            ao = basis.aos[ao_idx]
            groups[sh.l].append(
                AtomicOrbital(
                    index=ao_idx,
                    atom=atom,
                    l=sh.l,
                    axis=ao.axis,
                    n_prim=len(sh.exps),
                    shell=sh_idx,
                )
            )
    return {l: groups[l] for l in sorted(groups)}


def atomic_orbital_label(ao: AtomicOrbital) -> str:
    """Short label for an AO, e.g. ``"px (3)"`` or ``"s (6)"``."""
    name = _L_NAME.get(ao.l, str(ao.l))
    if ao.l == 0:
        return f"{name} ({ao.n_prim})"
    axis = _AXIS_NAME[ao.axis] if 0 <= ao.axis < 3 else str(ao.axis)
    return f"{name}{axis} ({ao.n_prim})"


def evaluate_atomic_orbital(
    basis: _MoldenData,
    ao: AtomicOrbital,
    *,
    half_width: float = 2.5,
    n: int = 40,
) -> VolumeData:
    """Evaluate a single atomic basis function on a box around its atom.

    Reuses the molecular-orbital grid evaluator with a unit coefficient
    vector selecting exactly ``ao``, so both visualizations share the same
    conventions (units, normalization, grid sampling).
    """
    if not 0 <= ao.index < basis.n_basis:
        raise IndexError(f"no AO with index {ao.index}")
    coeff = np.zeros(basis.n_basis)
    coeff[ao.index] = 1.0
    center = basis.molecule.coords[ao.atom]
    volume = _evaluate_molden_orbital_v2(
        basis, coeff, center - half_width, center + half_width, n
    )
    volume.label = atomic_orbital_label(ao)
    return volume


def load_volume_file(path: str | Path) -> VolumetricFile:
    """Load a cube or Molden file as a :class:`VolumetricFile`."""
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".cube":
        molecule, volumes = read_cube(p)
        volumes_by_index = {0: volumes[0]}
        labels = [v.label or f"MO {i + 1}" for i, v in enumerate(volumes)]

        def compute(index: int) -> VolumeData:
            vol = volumes_by_index.get(index)
            if vol is None and 0 <= index < len(volumes):
                vol = volumes[index]
            if vol is None:
                raise IndexError(f"no volume with index {index}")
            return vol

        return VolumetricFile(molecule=molecule, labels=labels, compute=compute, name=p.stem)

    if suffix == ".molden":
        data = read_molden(p)
        labels = [f"MO {i + 1}" for i in range(len(data.mo_labels))]

        def compute(index: int) -> VolumeData:
            if not 0 <= index < len(data.mo_coeffs):
                raise IndexError(f"no orbital with index {index}")
            return _evaluate_molden_orbital_v2(
                data, data.mo_coeffs[index], data.molecule.coords.min(axis=0) - 3.0,
                data.molecule.coords.max(axis=0) + 3.0, 40,
            )

        return VolumetricFile(
            molecule=data.molecule, labels=labels, compute=compute, name=p.stem, basis=data
        )

    raise ValueError(f"Unsupported volumetric file: {p.name}")
