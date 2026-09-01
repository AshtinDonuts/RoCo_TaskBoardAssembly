"""Unit tests for the R-wrist distal-tip aperture ruler overlay."""
from __future__ import annotations

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

try:
    from controllers.r_wrist_laser import _draw_aperture_ruler
except Exception as exc:  # Isaac/omni deps not available in plain pytest
    pytest.skip(f"r_wrist_laser import failed: {exc}", allow_module_level=True)


def test_draw_aperture_ruler_marks_image():
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    _draw_aperture_ruler(
        img,
        (80.0, 120.0),
        (240.0, 120.0),
        0.016,
        label="16.0 mm",
    )
    assert int(img.max()) > 0
    assert int(img[120, 160].max()) > 0
