"""SEM image formation (TECH-SPEC.md S4.2).

Stages implemented so far: 3 (SE yield / edge brightening) and 5 (beam
PSF) - A2.3. `sem_forward`'s detector-noise stage is still the v0 Gaussian
placeholder (A1.1); scan distortion, charging, shading and the switch to
Poisson shot noise land in A3.1 without changing any call signature here,
so nothing downstream (generate_dataset.py) needs to change again.
"""

import cv2
import numpy as np


def apply_edge_brightening(clean, k_edge=0.6, lambda_esc=3.0, threshold=0.4):
    """Stage 3: secondary-electron yield boost near a material edge.

        delta = delta_mat * (1 + k_edge * exp(-d / lambda_esc))

    This is the single mandatory augmentation the brief names explicitly
    ("apply edge-brightening to mimic real SEM behaviour") - see
    CITATIONS.md, SE yield with edge brightening.

    clean: float32 [0, 1], the rendered geometry before blur/noise.
    k_edge: peak fractional brightness boost right at an edge.
    lambda_esc: SE escape-depth length scale, in pixels of THIS image's
      own resolution - decides how fast the boost fades away from an edge.
    threshold: intensity above which a pixel counts as "material" for
      finding edges - the renderer's features are near-binary, so a mid
      threshold cleanly separates structure from background.

    Returns float32, NOT yet clipped to [0, 1] - the boost can push above
    1.0 right at a bright edge, same as real SEM edge saturation; clip
    after calling.
    """
    mask = (clean > threshold).astype(np.uint8)
    if not mask.any() or mask.all():
        return clean  # no edges in this patch (e.g. pure block or pure background)
    dist_in = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_out = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
    d = np.where(mask > 0, dist_in, dist_out).astype(np.float32)
    boost = 1.0 + k_edge * np.exp(-d / lambda_esc)
    return clean * boost


def apply_beam_psf(clean, sigma_beam=1.5, skirt_weight=0.05, skirt_sigma_mult=4.0):
    """Stage 5: finite electron beam spot size blurs the signal before it is
    ever sampled - Gaussian core plus a wide, low-weight Gaussian standing
    in for the Lorentzian beam tail (TECH-SPEC.md S3.6's own suggested
    simplification when a true Lorentzian is fiddly). See CITATIONS.md,
    Beam PSF.

    sigma_beam: core spot size, in pixels of THIS capture's own
    resolution - reference and search each call this with their own
    value, since the same physical spot maps to a different pixel count
    at each magnification (divide by m for the search-resolution call,
    same convention as every other length parameter in this project).
    """
    core = cv2.GaussianBlur(clean, (0, 0), sigma_beam)
    skirt = cv2.GaussianBlur(clean, (0, 0), sigma_beam * skirt_sigma_mult)
    return (1.0 - skirt_weight) * core + skirt_weight * skirt


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
