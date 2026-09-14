"""Tests for the click/drag distinction in atom selection."""

import numpy as np
import pytest

from pychem.rendering.molecular_renderer import _is_click, screen_delta_to_world


def test_click_small_movement_is_click():
    assert _is_click((100, 100), (102, 103)) is True
    assert _is_click((100, 100), (100, 100)) is True


def test_drag_movement_is_not_click():
    # A typical rotation drag exceeds the 6 px threshold
    assert _is_click((100, 100), (180, 130)) is False
    assert _is_click((100, 100), (100, 160)) is False


def test_boundary_tolerance():
    # Exactly at the limit (6 px) it is considered a click
    assert _is_click((0, 0), (6, 6)) is True
    assert _is_click((0, 0), (7, 0)) is False


def test_screen_delta_maps_to_the_view_plane():
    # Camera looking down -z with +y up: screen right is +x, screen up is +y.
    direction = (0.0, 0.0, -1.0)
    up = (0.0, 1.0, 0.0)

    right = screen_delta_to_world(direction, up, 0.01, 100.0, 0.0)
    assert np.allclose(right, [1.0, 0.0, 0.0], atol=1e-9)

    upward = screen_delta_to_world(direction, up, 0.01, 0.0, 100.0)
    assert np.allclose(upward, [0.0, 1.0, 0.0], atol=1e-9)

    # The displacement always stays in the plane of the screen.
    oblique = screen_delta_to_world(direction, up, 0.02, 33.0, -17.0)
    assert abs(float(np.dot(oblique, direction))) < 1e-9
