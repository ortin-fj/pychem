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
