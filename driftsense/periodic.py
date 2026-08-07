"""Periodic / aperiodic decomposition — TECH-SPEC §3.5. Member B.

The core contribution.

The argument
-----------
A perfectly periodic pattern is, by definition, identical at every lattice site.
So the periodic component of the image carries **zero** information about *which*
site you are looking at. It contributes equally to the correlation score
everywhere, which is precisely why plain template matching produces a hundred
tied peaks on a DRAM array.

So remove it. Keep only the reciprocal-lattice peaks in the Fourier domain,
inverse-transform to get the periodic component, and subtract. What remains —
array boundaries, periphery blocks, dummy fill, missing contacts, particles,
the illumination envelope — is exactly the content capable of resolving
position, and nothing else.

The elegant part
----------------
When the layout really is perfectly periodic, the residual is pure noise and
this stage contributes nothing. That is not a failure, it is *correct*: such a
pair is genuinely ambiguous and no algorithm can resolve it. Residual energy
therefore doubles as a principled ambiguity measure — the method degrades
exactly where the problem becomes unsolvable, and it can say so.

⚠ Citation note (TECH-SPEC §3.5): do **not** cite Moisan (2011) "Periodic plus
smooth image decomposition" for this. Its "periodic" means periodic *boundary
extension*, an unrelated decomposition. The right framing is standard
reciprocal-lattice / Fourier crystallography.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["Decomposition", "decompose", "lattice_frequencies", "residual_energy_ratio"]


class Decomposition:
    """Periodic and aperiodic parts of one image."""

    def __init__(self, periodic: np.ndarray, aperiodic: np.ndarray, ratio: float):
        self.periodic = periodic
        self.aperiodic = aperiodic
        self.ratio = ratio          # aperiodic energy / total energy, in [0, 1]


def lattice_frequencies(peaks: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Convert spectral peak offsets (in shifted-FFT bins) to cycles per pixel.

    Working in cycles/pixel makes the lattice image-size independent, so the
    *same* frequency list can be applied to the 1000x1000 search image and to
    the ~100x100 template. That only works because the template has already
    been rescaled into search-pixel units by `matching.build_template`; its
    lattice is therefore the search image's lattice.
    """
    if len(peaks) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    h, w = shape[:2]
    return np.stack([peaks[:, 0] / float(w), peaks[:, 1] / float(h)], axis=1)


def _lattice_mask(shape: tuple[int, int], freqs: np.ndarray,
                  radius: float, keep_dc: bool = True) -> np.ndarray:
    """Binary mask over the shifted spectrum keeping only lattice peaks.

    Each peak gets a small disc rather than a single bin: a finite image window
    convolves every spectral line with the window transform, so real lattice
    energy is spread over a few bins. Too small a disc leaves periodic residue
    behind (which then dominates the residual correlation); too large a disc
    starts eating the aperiodic content we are trying to isolate.
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    cy, cx = h // 2, w // 2
    r = max(1, int(round(radius)))

    if keep_dc:
        cv2.circle(mask, (cx, cy), r, 1.0, -1)

    for fx, fy in freqs:
        for sgn in (1.0, -1.0):                 # spectrum is conjugate-symmetric
            bx = int(round(cx + sgn * fx * w))
            by = int(round(cy + sgn * fy * h))
            if 0 <= bx < w and 0 <= by < h:
                cv2.circle(mask, (bx, by), r, 1.0, -1)
    return mask


def decompose(img: np.ndarray, freqs: np.ndarray,
              radius: float | None = None) -> Decomposition:
    """Split `img` into lattice-periodic and aperiodic parts.

    v1 of TECH-SPEC §3.5 — Fourier synthesis. (v2, unit-cell folding, is more
    robust when the lattice drifts slowly across the field, and is only worth
    building if the schedule allows.)
    """
    img = np.asarray(img, dtype=np.float32)
    h, w = img.shape[:2]
    if len(freqs) == 0:
        return Decomposition(np.zeros_like(img), img.copy(), 1.0)

    if radius is None:
        # scale with image size so template and search are treated consistently
        radius = max(1.5, 0.006 * min(h, w))

    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    F = np.fft.fftshift(np.fft.fft2(img * win))
    mask = _lattice_mask((h, w), freqs, radius)

    periodic = np.fft.ifft2(np.fft.ifftshift(F * mask)).real.astype(np.float32)
    # Undo the analysis window where it is safe to; near the border the window
    # goes to zero and the division explodes, so clamp it.
    safe = np.maximum(win, 0.08)
    periodic = periodic / safe
    aperiodic = img - periodic

    # Taper the residual at the border: the Hann window makes the outer ring
    # meaningless, and if left in it dominates the residual correlation.
    aperiodic = aperiodic * win

    e_tot = float(np.sum(img.astype(np.float64) ** 2)) + 1e-12
    e_ap = float(np.sum(aperiodic.astype(np.float64) ** 2))
    ratio = float(np.clip(e_ap / e_tot, 0.0, 1.0))
    return Decomposition(periodic, aperiodic.astype(np.float32), ratio)


#: Residual-energy operating point, measured on dev pairs rather than guessed.
#:
#: The absolute ratio is small and easy to misjudge: matching runs on the
#: high-pass band, whose energy sits almost entirely *in* the lattice peaks, so
#: removing them removes ~97-99% of it. Observed values run 0.001 for a
#: genuinely degenerate pair up to ~0.03 for one with a periphery block in the
#: reference window — a factor of ~20 separation, but nowhere near the 0.02-0.17
#: range an untrained eye would assume. Setting the gate from intuition instead
#: of measurement switched the whole decomposition stage off silently: it ran,
#: cost time, and changed no answer.
RESIDUAL_GATE_LO = 0.004
RESIDUAL_GATE_HI = 0.020


def residual_gate(ratio: float) -> float:
    """Map an aperiodic energy ratio onto a [0, 1] trust weight.

    0 means "this pair has no aperiodic content — the residual is noise, and
    correlating it would invent a confident answer where none is warranted".
    1 means "there is real non-repeating structure here to lock on to".
    """
    span = max(RESIDUAL_GATE_HI - RESIDUAL_GATE_LO, 1e-9)
    return float(np.clip((ratio - RESIDUAL_GATE_LO) / span, 0.0, 1.0))


def residual_energy_ratio(dec: Decomposition) -> float:
    """Aperiodic energy fraction — a principled ambiguity measure.

    Low means the layout is essentially pure lattice: there is nothing in the
    image capable of distinguishing one site from another, and any confident
    answer would be a fabrication. High means real structure exists to lock on
    to. This is confidence feature #2 in `decide.py`.
    """
    return dec.ratio
