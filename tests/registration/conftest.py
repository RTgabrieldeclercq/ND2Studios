"""Synthetic-ground-truth fixtures for the registration engine tests (V1.56)."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter


def _textured_image(H: int = 256, W: int = 256, seed: int = 0) -> np.ndarray:
    """A 12-bit textured image: smoothed random field → structure phase
    correlation and ECC can lock onto (pure white noise has no usable peak)."""
    rng = np.random.default_rng(seed)
    field = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), sigma=3.0)
    field -= field.min()
    field /= max(field.max(), 1e-6)
    return (field * 3500.0 + 200.0).astype(np.uint16)


@pytest.fixture(scope="session")
def ref_image() -> np.ndarray:
    """A single reference image (uint16, textured)."""
    return _textured_image()


@pytest.fixture(scope="session")
def ref_image_factory():
    """Callable returning a fresh textured image for a given seed."""
    return _textured_image
