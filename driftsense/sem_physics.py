"""SEM image formation - v0 placeholder.

v0 applies independent additive Gaussian noise per capture, which is
acceptable ONLY for this crude bootstrap generator (PLAN.md S3, A1.1).
TECH-SPEC.md S4.2 mandates a full forward model - beam PSF, scan drift,
charging, Poisson shot noise, detector chain - which replaces this module's
internals on A3.1 without changing the call signature below, so nothing
downstream (layouts.py callers, generate_dataset.py) needs to change.
"""

import numpy as np


def sem_forward(clean, rng, noise_std=0.03):
    """Apply v0 noise to a clean [0, 1] float32 image.

    clean: float32 array in [0, 1].
    rng: np.random.Generator - use an INDEPENDENT stream per capture
      (reference vs search), never the same rng or seed for both.
    noise_std: Gaussian sigma in normalized [0,1] units.

    Returns uint8 array in [0, 255].
    """
    noisy = clean + rng.normal(0.0, noise_std, size=clean.shape).astype(np.float32)
    noisy = np.clip(noisy, 0.0, 1.0)
    return (noisy * 255.0).round().astype(np.uint8)
