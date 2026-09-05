"""Tests for the click/drag distinction in atom selection."""

import pytest

from pychem.rendering.molecular_renderer import _is_click


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
