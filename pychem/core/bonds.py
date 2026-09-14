"""Connectivity geometry: bond guessing from distances.

Used as the default in visualization when the input format does not provide
explicit connectivity (as with XYZ).
"""

from __future__ import annotations

import numpy as np

from pychem.core.elements import element
from pychem.core.molecule import Molecule

# Extra fraction over the sum of covalent radii that is accepted as a bond
# (typical values 0.1-0.3 in the literature; 0.15 is a good starting point
# for organic molecules).
_DEFAULT_TOLERANCE = 0.15


def _covalent_radii(symbols: list[str]) -> np.ndarray:
    """Covalent radius of every atom (one lookup per *distinct* element)."""
    unique, inverse = np.unique(np.asarray(symbols), return_inverse=True)
    table = np.array([element(str(s)).covalent_radius for s in unique], dtype=float)
    return table[inverse]


def _candidate_pairs(
    coords: np.ndarray, radii: np.ndarray, tolerance: float
) -> np.ndarray:
    """Atom pairs that *may* be bonds, as ``(i, j)`` with ``i < j``.

    A bond needs ``d <= (r_i + r_j) * (1 + tolerance)``, so nothing farther
    apart than ``2 * r_max * (1 + tolerance)`` can ever qualify. Binning the
    atoms into a cubic grid of exactly that size means only the 27
    neighbouring cells have to be compared, which turns the search from
    O(n²) into O(n) and removes the giant ``n x n`` temporaries.

    This is only a *filter*: callers still apply the exact distance test on
    the returned candidates, so the result is identical to the naive
    all-pairs computation.
    """
    n = coords.shape[0]
    if n < 2:
        return np.empty((0, 2), dtype=np.int64)

    cutoff = 2.0 * float(radii.max()) * (1.0 + tolerance)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        return np.empty((0, 2), dtype=np.int64)

    # Grid geometry: cell size == cutoff guarantees that any pair within
    # bonding distance lies in the same or an adjacent cell.
    origin = coords.min(axis=0)
    extent = coords.max(axis=0) - origin
    dims = np.maximum(np.floor(extent / cutoff).astype(np.int64) + 1, 1)
    cells = np.floor((coords - origin) / cutoff).astype(np.int64)
    np.clip(cells, 0, dims - 1, out=cells)
    linear = (cells[:, 0] * dims[1] + cells[:, 1]) * dims[2] + cells[:, 2]

    # Atoms grouped by cell: `uniq[k]` owns the slice [starts[k], ends[k]).
    order = np.argsort(linear, kind="stable")
    uniq, starts = np.unique(linear[order], return_index=True)
    ends = np.append(starts[1:], n)

    i_parts: list[np.ndarray] = []
    j_parts: list[np.ndarray] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                tx = cells[:, 0] + dx
                ty = cells[:, 1] + dy
                tz = cells[:, 2] + dz
                # Stay inside the grid (no wrap-around across axes).
                inside = (
                    (tx >= 0)
                    & (tx < dims[0])
                    & (ty >= 0)
                    & (ty < dims[1])
                    & (tz >= 0)
                    & (tz < dims[2])
                )
                if not inside.any():
                    continue
                target = (tx * dims[1] + ty) * dims[2] + tz
                pos = np.searchsorted(uniq, target)
                pos_clipped = np.minimum(pos, uniq.size - 1)
                valid = inside & (pos < uniq.size) & (uniq[pos_clipped] == target)
                atoms = np.nonzero(valid)[0]
                if atoms.size == 0:
                    continue
                lo = starts[pos[atoms]]
                count = ends[pos[atoms]] - lo
                non_empty = count > 0
                atoms, lo, count = atoms[non_empty], lo[non_empty], count[non_empty]
                if atoms.size == 0:
                    continue
                # Expand each (atom, neighbour-cell) pair into one pair per
                # atom of that cell, without a Python loop.
                total = int(count.sum())
                cumulative = np.cumsum(count)
                # Index inside each cell's slice (0, 1, ... count-1).
                inner = np.arange(total, dtype=np.int64) - np.repeat(
                    cumulative - count, count
                )
                i_parts.append(np.repeat(atoms, count))
                j_parts.append(order[np.repeat(lo, count) + inner])

    if not i_parts:
        return np.empty((0, 2), dtype=np.int64)

    i_all = np.concatenate(i_parts)
    j_all = np.concatenate(j_parts)
    # Every pair is generated twice (once from each end); keep i < j only.
    forward = i_all < j_all
    pairs = np.column_stack([i_all[forward], j_all[forward]])
    if pairs.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    pairs = np.unique(pairs, axis=0)
    return pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))].astype(np.int64)


def guess_bonds(
    molecule: Molecule, tolerance: float = _DEFAULT_TOLERANCE
) -> np.ndarray:
    """Guess bonds from overlapping covalent radii.

    Returns an array of pairs ``(i, j)`` with ``i < j``, atom indices.
    Cost O(n): only neighbouring grid cells are compared (see
    :func:`_candidate_pairs`), which keeps large structures such as a
    11000-atom PDB at ~0.1 s instead of minutes.
    """
    n = molecule.n_atoms
    if n < 2:
        return np.empty((0, 2), dtype=np.int64)

    radii = _covalent_radii(molecule.symbols)
    coords = np.asarray(molecule.coords, dtype=float)

    candidates = _candidate_pairs(coords, radii, tolerance)
    if candidates.size == 0:
        return candidates.astype(np.int64)

    # Exact test, applied only to the candidates (the authoritative rule).
    distances = np.linalg.norm(
        coords[candidates[:, 0]] - coords[candidates[:, 1]], axis=1
    )
    limit = (radii[candidates[:, 0]] + radii[candidates[:, 1]]) * (1.0 + tolerance)
    return candidates[distances <= limit].astype(np.int64)


# Hysteresis tolerances for dynamic bonds (fractions over the covalent radius
# sum). Breaking requires a larger distance than forming, which avoids the
# bond flickering on/off when a distance hovers near the threshold.
_DYNAMIC_BREAK_TOLERANCE = 0.25
_DYNAMIC_FORM_TOLERANCE = 0.10


def update_dynamic_bonds(
    molecule: Molecule,
    previous: np.ndarray | None,
    *,
    break_tolerance: float = _DYNAMIC_BREAK_TOLERANCE,
    form_tolerance: float = _DYNAMIC_FORM_TOLERANCE,
) -> np.ndarray:
    """Update bonds frame-to-frame using hysteresis on the distances.

    - Pairs bonded in ``previous`` are kept while their distance is within
      ``(r_i + r_j) * (1 + break_tolerance)``.
    - Non-bonded pairs become bonded once their distance drops within
      ``(r_i + r_j) * (1 + form_tolerance)`` (``break > form`` creates the
      hysteresis band).

    If ``previous`` is ``None`` or empty, only bond *formation* applies, so a
    new bond appears once the atoms get close enough.

    Returns the sorted pair array ``(i, j)`` with ``i < j``. Like
    :func:`guess_bonds`, the search runs on a spatial grid (O(n)) instead of
    all pairs; the hysteresis rules below are unchanged, so the result is
    identical to comparing every pair. This matters during trajectory
    playback, where the bonds are recomputed on every frame.
    """
    n = molecule.n_atoms
    if n < 2:
        return np.empty((0, 2), dtype=np.int64)

    radii = _covalent_radii(molecule.symbols)
    coords = np.asarray(molecule.coords, dtype=float)

    # A pair kept from `previous` survives up to the (larger) break
    # tolerance and a new one forms at the (smaller) form tolerance, so the
    # widest of the two defines the search radius.
    search_tolerance = max(break_tolerance, form_tolerance)
    all_pairs = _candidate_pairs(coords, radii, search_tolerance)
    if all_pairs.size == 0:
        return all_pairs.astype(np.int64)

    d = np.linalg.norm(coords[all_pairs[:, 0]] - coords[all_pairs[:, 1]], axis=1)
    base = radii[all_pairs[:, 0]] + radii[all_pairs[:, 1]]

    if previous is None or previous.size == 0:
        prev = set()
    else:
        prev = {tuple(sorted((int(a), int(b)))) for a, b in previous}

    keep_mask = np.array(
        [tuple(p) in prev for p in all_pairs], dtype=bool
    )
    kept = all_pairs[keep_mask & (d <= base * (1.0 + break_tolerance))]
    formed = all_pairs[~keep_mask & (d <= base * (1.0 + form_tolerance))]

    combined = np.vstack([kept, formed]) if (kept.size or formed.size) else np.empty((0, 2), dtype=np.int64)
    if combined.size == 0:
        return combined.astype(np.int64)
    # Unique and sorted rows
    combined = np.unique(combined, axis=0)
    return combined[np.lexsort((combined[:, 1], combined[:, 0]))].astype(np.int64)
