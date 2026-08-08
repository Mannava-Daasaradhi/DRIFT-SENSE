"""Image loading and preprocessing — TECH-SPEC §3.0.

Member B. Everything downstream (spectral, matching, periodic) consumes the
`Bands` produced here, so reference and search images are guaranteed to have been
treated identically.

Design notes
------------
The two captures in a pair come from different magnifications, different doses and
different detector settings. Raw intensity is therefore *not* comparable between
them, and correlating raw intensity directly is the first thing that goes wrong.

Three defences, in order:

1. Robust percentile normalization instead of min/max — invariant to detector gain
   and offset, and immune to a single saturated pixel rescaling the whole frame.
2. A high-pass band `hp = I - blur(I)`. This removes the low-order shading and
   charging field, which differ between captures and carry almost no positional
   information. Matching runs on `hp`.
3. A gradient-magnitude channel. SEM contrast is dominated by edge brightening
   (secondary-electron escape near feature edges), so the edge map is the most
   stable thing across two captures of the same structure. Scored separately and
   fused with the intensity channel in `matching.py`.

The removed low-frequency field `lf` is kept, not discarded: it is aperiodic by
construction and feeds the disambiguation stage in `periodic.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = ["Bands", "load_gray", "robust_normalize", "band_split",
           "gradient_magnitude", "preprocess", "smooth_lowpass"]


@dataclass
class Bands:
    """The preprocessed representation of one image.

    Attributes
    ----------
    img : float32 2-D, robustly normalized to roughly [0, 1]
    hp  : float32 2-D, high-pass structure band — matching runs on this
    lf  : float32 2-D, the removed low-frequency field (shading / charging / defocus)
    grad: float32 2-D, gradient magnitude of `hp`, normalized
    sp  : float32 2-D, *gently* high-passed band — spectral analysis runs on this
    """

    img: np.ndarray
    hp: np.ndarray
    lf: np.ndarray
    grad: np.ndarray
    sp: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.img.shape[:2]


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load_gray(path: str) -> np.ndarray:
    """Load any image as a float32 2-D array.

    Deliberately permissive (PLAN.md Rule 4): accepts PNG / TIF / JPG / BMP,
    8-bit or 16-bit or float, grayscale / RGB / RGBA / palette, any dimensions.
    Colour is collapsed with ITU-R BT.601 luminance weights rather than a plain
    mean, because that is what a detector-response weighting looks like.

    Raises
    ------
    IOError
        If the file cannot be decoded at all. `localize.py` catches this and
        falls back; it is not allowed to propagate to the caller of the CLI.
    """
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        # cv2.imread returns None for unicode paths on Windows and for anything
        # it cannot decode. Retry through numpy so unicode paths still work.
        try:
            buf = np.fromfile(path, dtype=np.uint8)
            arr = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        except Exception:  # pragma: no cover - defensive
            arr = None
    if arr is None:
        raise IOError(f"could not read image: {path!r}")

    arr = np.asarray(arr)
    if arr.ndim == 3:
        if arr.shape[2] == 4:            # RGBA / BGRA — drop alpha, do not blend
            arr = arr[:, :, :3]
        if arr.shape[2] == 3:            # OpenCV gives BGR
            b, g, r = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            arr = 0.299 * r.astype(np.float32) + \
                  0.587 * g.astype(np.float32) + \
                  0.114 * b.astype(np.float32)
        else:                            # 1- or 2-channel oddity
            arr = arr[:, :, 0]
    elif arr.ndim != 2:
        raise IOError(f"unsupported image shape {arr.shape} in {path!r}")

    arr = arr.astype(np.float32, copy=False)
    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(arr)


# --------------------------------------------------------------------------- #
# normalization and band splitting
# --------------------------------------------------------------------------- #

def robust_normalize(img: np.ndarray, lo_pct: float = 1.0,
                     hi_pct: float = 99.0) -> np.ndarray:
    """Map the [lo_pct, hi_pct] percentile range onto [0, 1] and clip.

    Percentiles rather than min/max: a single hot pixel or a charging streak
    would otherwise compress the whole image into a narrow band, and the two
    captures would end up on different intensity scales.
    """
    img = img.astype(np.float32, copy=False)
    lo, hi = np.percentile(img, [lo_pct, hi_pct])
    span = float(hi - lo)
    if not np.isfinite(span) or span < 1e-8:
        # Constant (or near-constant) image — nothing to stretch.
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - lo) / span, 0.0, 1.0).astype(np.float32)


def _sigma_for(shape: tuple[int, int], frac: float = 0.06) -> float:
    """Blur sigma as a fraction of the smaller image dimension.

    Scale-relative rather than an absolute pixel count, so the same code gives
    the same *physical* cutoff on a 1000 px search image and on a 100 px
    template (PLAN.md Rule 4 — never hardcode a size).
    """
    n = min(shape[0], shape[1])
    return max(2.0, frac * float(n))


def smooth_lowpass(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian low-pass, computed at a resolution matched to the cutoff.

    Both low-pass fields this module needs are *very* wide: `band_split` uses
    sigma ~= 6% of the frame and the spectral band uses 30%, which on a
    1000 px search image is sigma = 60 and sigma = 300. `cv2.GaussianBlur` sizes
    its kernel from sigma, so those become ~361- and ~1801-tap separable
    convolutions, and they dominated the entire runtime: 2.44 s of a 4.27 s
    pair, against 0.50 s for every `matchTemplate` call combined.

    A low-pass band by definition contains no detail finer than its own cutoff,
    so evaluating it at full resolution is wasted work. Decimate by roughly
    sigma/4, blur with the correspondingly small sigma, and resize back. The
    residual error is far below the noise floor of the images, and the whole
    preprocessing chain drops from ~2.6 s to a few milliseconds.
    """
    img = np.asarray(img, dtype=np.float32)
    h, w = img.shape[:2]
    sigma = float(max(sigma, 1e-3))

    k = int(max(1, min(sigma / 4.0, min(h, w) / 16.0)))
    if k <= 1:
        return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma,
                                borderType=cv2.BORDER_REFLECT)

    small = cv2.resize(img, (max(2, w // k), max(2, h // k)),
                       interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=sigma / k, sigmaY=sigma / k,
                             borderType=cv2.BORDER_REFLECT)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def band_split(img: np.ndarray, sigma: float | None = None
               ) -> tuple[np.ndarray, np.ndarray]:
    """Split into (high-pass structure, low-frequency field).

    `lf` is the illumination / charging / defocus envelope: smooth, aperiodic,
    and different between the two captures. `hp = img - lf` is what matching
    runs on. `lf` is not thrown away — it is aperiodic content and therefore
    weakly informative for disambiguation (TECH-SPEC §3.5).
    """
    if sigma is None:
        sigma = _sigma_for(img.shape)
    lf = smooth_lowpass(img, sigma)
    hp = img - lf
    return hp.astype(np.float32), lf.astype(np.float32)


def gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude, robustly normalized.

    SEM images are edge-contrast dominated, so this channel survives dose and
    gain differences between captures better than raw intensity does.
    """
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return robust_normalize(mag, 1.0, 99.0)


def preprocess(img: np.ndarray, sigma: float | None = None) -> Bands:
    """Run the full §3.0 chain on an already-loaded float array.

    Applied identically to the search image and to the constructed template —
    if the two are preprocessed differently, ZNCC scores drift and the peak
    ordering becomes unreliable.
    """
    norm = robust_normalize(img)
    hp, lf = band_split(norm, sigma)
    grad = gradient_magnitude(hp)

    # A separate, much gentler high-pass for the FFT stage.
    #
    # `hp` uses sigma ~= 6% of the frame, which is right for matching but wrong
    # for spectral analysis: it removes everything below ~16 cycles/frame, and
    # the reference image only shows ~10 periods of the layout, so its own
    # fundamentals live right in that stop-band. On a FinFET reference the gate
    # frequency is at ~2 cycles/frame and is erased completely. Using `hp` for
    # the FFT costs roughly two thirds of the scale estimates.
    sp_sigma = max(8.0, 0.30 * float(min(norm.shape[:2])))
    sp = norm - smooth_lowpass(norm, sp_sigma)
    return Bands(img=norm, hp=hp, lf=lf, grad=grad, sp=sp.astype(np.float32))


def load_and_preprocess(path: str, sigma: float | None = None) -> Bands:
    """Convenience: `load_gray` then `preprocess`."""
    return preprocess(load_gray(path), sigma)
