"""Tests for the volumetric data model (pychem.core.volume)."""

import numpy as np
import pytest

from pychem.core.volume import DENSITY_ENVELOPE_LEVEL, VolumeData


def _linear_field() -> VolumeData:
    """A field that is linear in x/y/z, so trilinear interpolation is exact."""
    origin = np.array([-1.0, 2.0, 0.5])
    spacing = np.array([0.25, 0.5, 0.1])
    shape = (7, 5, 4)
    axes = [origin[d] + spacing[d] * np.arange(shape[d]) for d in range(3)]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    return VolumeData(
        values=2.0 + 3.0 * x - y + 0.5 * z, origin=origin, spacing=spacing
    )


def _gradient(point: np.ndarray) -> np.ndarray:
    return 2.0 + 3.0 * point[:, 0] - point[:, 1] + 0.5 * point[:, 2]


def test_sample_at_reproduces_a_linear_field():
    volume = _linear_field()
    rng = np.random.default_rng(0)
    low = volume.origin
    high = volume.origin + volume.spacing * (np.asarray(volume.shape) - 1)

    points = rng.uniform(low, high, size=(25, 3))
    values = volume.sample_at(points)

    assert values.shape == (25,)
    np.testing.assert_allclose(values, _gradient(points), atol=1e-10)


def test_sample_at_is_exact_at_grid_points_and_single_point_input():
    volume = _linear_field()
    node = volume.origin + volume.spacing * np.array([2, 3, 1])

    # A single position may be given without the (n, 3) shape.
    values = volume.sample_at(node)
    assert values.shape == (1,)
    np.testing.assert_allclose(values, _gradient(node.reshape(1, 3)))


def test_sample_at_clamps_points_outside_the_grid():
    """Points beyond the box use the boundary value (no holes/NaNs)."""
    volume = _linear_field()
    low = volume.origin
    high = volume.origin + volume.spacing * (np.asarray(volume.shape) - 1)

    outside = np.vstack([low - 10.0, high + 10.0])
    values = volume.sample_at(outside)

    assert np.all(np.isfinite(values))
    np.testing.assert_allclose(values[0], _gradient(low.reshape(1, 3))[0])
    np.testing.assert_allclose(values[1], _gradient(high.reshape(1, 3))[0])


def test_sample_at_rejects_bad_shapes():
    volume = _linear_field()
    with pytest.raises(ValueError, match=r"shape \(n, 3\)"):
        volume.sample_at(np.zeros((4, 2)))


def test_suggest_density_isovalue_uses_the_physical_envelope():
    """The envelope is the small e/Bohr^3 level, not a quantile of the grid."""
    origin = np.zeros(3)
    spacing = np.ones(3)
    # A density-like field: tiny almost everywhere, huge near one nucleus.
    values = np.full((4, 4, 4), 1e-5)
    values[0, 0, 0] = 96.4  # e.g. carbon in e/Bohr^3
    volume = VolumeData(values=values, origin=origin, spacing=spacing)

    assert volume.suggest_density_isovalue() == pytest.approx(
        DENSITY_ENVELOPE_LEVEL
    )
    # Far below the density near the nucleus, hence a small closed envelope.
    assert volume.suggest_density_isovalue() < 0.01


def test_suggest_density_isovalue_caps_tiny_fields():
    volume = VolumeData(
        values=np.full((3, 3, 3), 0.05), origin=np.zeros(3), spacing=np.ones(3)
    )
    # 1e-2 * max < 1e-3, so the physical target is capped for this scale.
    assert volume.suggest_density_isovalue() == pytest.approx(5e-4)


def test_suggest_density_isovalue_on_an_empty_field():
    volume = VolumeData(
        values=np.zeros((2, 2, 2)), origin=np.zeros(3), spacing=np.ones(3)
    )
    assert volume.suggest_density_isovalue() == pytest.approx(
        DENSITY_ENVELOPE_LEVEL
    )
