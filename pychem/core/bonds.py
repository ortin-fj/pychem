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


def guess_bonds(
    molecule: Molecule, tolerance: float = _DEFAULT_TOLERANCE
) -> np.ndarray:
    """Guess bonds from overlapping covalent radii.

    Returns an array of pairs ``(i, j)`` with ``i < j``, atom indices.
    Cost O(n²); can be accelerated with a spatial grid if needed for large
    systems.
    """
    n = molecule.n_atoms
    if n < 2:
        return np.empty((0, 2), dtype=np.int64)

    radii = np.array([element(s).covalent_radius for s in molecule.symbols])
    coords = molecule.coords

    # Pairwise radius sums: R[i, j] = r_i + r_j
    rsum = radii[:, None] + radii[None, :]
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

    i_idx, j_idx = np.where(
        (distances <= rsum * (1.0 + tolerance)) & (np.arange(n)[:, None] < np.arange(n)[None, :])
    )
    return np.column_stack([i_idx, j_idx]).astype(np.int64)


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

    Returns the sorted pair array ``(i, j)`` with ``i < j``. Naive O(n²);
    a spatial grid / neighbour list should be used for very large systems.
    """
    n = molecule.n_atoms
    if n < 2:
        return np.empty((0, 2), dtype=np.int64)

    radii = np.array([element(s).covalent_radius for s in molecule.symbols])
    coords = molecule.coords
    rsum = radii[:, None] + radii[None, :]
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

    i_idx, j_idx = np.where(np.arange(n)[:, None] < np.arange(n)[None, :])
    all_pairs = np.column_stack([i_idx, j_idx])
    d = distances[i_idx, j_idx]
    base = rsum[i_idx, j_idx]

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
