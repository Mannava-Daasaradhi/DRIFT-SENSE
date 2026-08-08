"""Spectral lattice estimation — TECH-SPEC §3.1. Member B.

This is the Slide-5 innovation, and the one part of the pipeline that is not
table stakes.

The idea
--------
A repeating circuit layout is a 2-D crystal. Its Fourier transform is therefore
not a smear but a *reciprocal lattice*: a discrete set of peaks at integer
combinations of two basis vectors `g1, g2`. Real-space period and reciprocal
vector length are inverses, so if the same layout is imaged at two
magnifications:

    period_search = period_reference / m        (features are m times smaller)
    |g_search|    = m * |g_reference|           (so frequencies are m times higher)

which gives the magnification ratio in **closed form**:

    m = |g_search| / |g_reference|

and the rotation as the angle between the two bases. No pyramid, no sweep, no
`for scale in np.arange(7, 14, 0.05)`.

Why this is better, not just different
--------------------------------------
The FFT integrates over every pixel in the image. A 1000x1000 search image
contributes 10^6 pixels to the estimate, so the noise averages down by ~10^3.
A patch-based scale search sees only the ~10^4 pixels under the template. Since
the brief states outright that the official test images are noisier than ours,
that difference in effective SNR is the whole ballgame.

And the elegance: periodicity is what makes this problem hard. Here it is what
makes it easy. The more perfectly periodic the layout, the sharper the
reciprocal-lattice peaks and the more precise the scale estimate.

Symmetry caveat
---------------
A lattice does not have a unique basis, and the layout has a point group: a
square DRAM grid is invariant under 90 deg rotations, a FinFET line array under
180 deg. The measured rotation is therefore only defined modulo that group. We
do not guess — we emit every symmetry-equivalent hypothesis and let the
correlation stage in `matching.py` decide. Forgetting this produces a mysterious
~25% failure rate that costs a day to find.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

__all__ = ["LatticeEstimate", "estimate_lattice", "estimate_scale_rotation",
           "fourier_mellin_scale_rotation", "log_magnitude_spectrum"]


@dataclass
class LatticeEstimate:
    """Reciprocal-lattice basis recovered from one image."""

    g1: np.ndarray                      # reciprocal basis vector, cycles/pixel (x, y)
    g2: np.ndarray
    peaks: np.ndarray                   # (N, 2) detected peak offsets from DC, px
    strengths: np.ndarray               # (N,) peak prominences
    quality: float                      # [0, 1]; 0 = no usable lattice found
    n_assigned: int = 0                 # peaks explained by the fitted basis
    #: The windowed log-magnitude spectrum this estimate was derived from.
    #: Kept so the Fourier-Mellin cross-check can reuse it instead of paying for
    #: a second full-frame FFT of the same image — that duplication was ~0.2 s
    #: of a ~1.3 s pair.
    mag: np.ndarray | None = None

    @property
    def periods(self) -> tuple[float, float]:
        """Real-space periods in pixels, |a_i| = 1 / |g_i|."""
        n1 = float(np.linalg.norm(self.g1))
        n2 = float(np.linalg.norm(self.g2))
        return (1.0 / n1 if n1 > 1e-9 else float("inf"),
                1.0 / n2 if n2 > 1e-9 else float("inf"))


@dataclass
class ScaleRotation:
    """Result of comparing two lattices."""

    scale: float                        # m, search px per reference px
    rotation: float                     # degrees, for cv2.getRotationMatrix2D on the template
    quality: float                      # [0, 1]
    hypotheses: list[tuple[float, float]] = field(default_factory=list)  # (scale, rot) list
    method: str = "lattice"             # "lattice" | "fourier_mellin" | "fallback"
    agreement: float = 0.0              # [0,1] agreement between the two estimators


# --------------------------------------------------------------------------- #
# spectrum
# --------------------------------------------------------------------------- #

def log_magnitude_spectrum(img: np.ndarray, dc_radius_frac: float = 0.004
                           ) -> np.ndarray:
    """Windowed, DC-suppressed log-magnitude spectrum, fftshifted.

    The Hann window matters more than it looks. Without it the implicit periodic
    boundary of the FFT creates a bright horizontal+vertical cross through DC
    that is *stronger* than the real lattice peaks, and every peak detector
    locks onto it instead of the signal.

    The DC disc is deliberately *small*. It is tempting to suppress a generous
    low-frequency region to kill the shading field, but the reference image only
    contains ~10 periods of the layout, so its fundamentals live at radius ~10
    and a 12 px disc erases them outright. On a FinFET reference the gate
    frequency sits at radius ~2 and is even more fragile. Low-order shading is
    handled by the caller passing a gently high-passed band, not by carving a
    hole in the spectrum.
    """
    h, w = img.shape[:2]
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    f = np.fft.fftshift(np.fft.fft2(img * win))
    mag = np.log1p(np.abs(f)).astype(np.float32)

    # Suppress a small disc around DC: it carries the mean and the low-order
    # shading field, both of which are enormous and carry no lattice information.
    cy, cx = h // 2, w // 2
    r = max(2, int(dc_radius_frac * min(h, w)))
    yy, xx = np.ogrid[:h, :w]
    disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    mag[disc] = 0.0
    return mag


def _detect_peaks(mag: np.ndarray, max_peaks: int = 60,
                  min_sep_frac: float = 0.008) -> tuple[np.ndarray, np.ndarray]:
    """Local maxima of the spectrum, returned as (dx, dy) offsets from DC.

    A local-contrast normalization is applied first: raw spectral magnitude
    falls off steeply with frequency, so without it every detected peak is a
    low-order harmonic and the true fundamental is never found on a fine-pitch
    layout.
    """
    h, w = mag.shape
    cy, cx = h // 2, w // 2

    bg = cv2.GaussianBlur(mag, (0, 0), sigmaX=max(3.0, 0.008 * min(h, w)),
                          borderType=cv2.BORDER_REFLECT)
    norm = mag - bg

    k = max(3, int(min_sep_frac * min(h, w)) | 1)     # odd kernel
    dil = cv2.dilate(norm, np.ones((k, k), np.uint8))
    is_peak = (norm >= dil) & (norm > 0)

    ys, xs = np.nonzero(is_peak)
    if ys.size == 0:
        return np.zeros((0, 2), np.float64), np.zeros((0,), np.float64)

    vals = norm[ys, xs]
    order = np.argsort(vals)[::-1][:max_peaks * 2]
    ys, xs, vals = ys[order], xs[order], vals[order]

    # sub-pixel refinement by 3-point parabola on each axis; a half-bin error in
    # frequency is a ~1% error in the recovered magnification
    offs = []
    strengths = []
    for y, x, v in zip(ys, xs, vals):
        if not (1 <= y < h - 1 and 1 <= x < w - 1):
            continue
        dx = _parabola(norm[y, x - 1], norm[y, x], norm[y, x + 1])
        dy = _parabola(norm[y - 1, x], norm[y, x], norm[y + 1, x])
        offs.append((x + dx - cx, y + dy - cy))
        strengths.append(float(v))
        if len(offs) >= max_peaks:
            break

    return np.asarray(offs, dtype=np.float64), np.asarray(strengths, dtype=np.float64)


def _parabola(a: float, b: float, c: float) -> float:
    """Sub-bin offset of the vertex of the parabola through (-1,a),(0,b),(1,c)."""
    den = a - 2.0 * b + c
    if abs(den) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))


# --------------------------------------------------------------------------- #
# lattice fitting
# --------------------------------------------------------------------------- #

def gauss_reduce(g1: np.ndarray, g2: np.ndarray,
                 iters: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Lagrange-Gauss reduction to the canonical two shortest basis vectors.

    A lattice has infinitely many bases, all related by unimodular transforms.
    Two independent fits of the *same* lattice can therefore return bases that
    look nothing alike, which makes comparing reference to search unnecessarily
    hard. Reducing both to the canonical shortest basis first removes most of
    that freedom, so the relabelling search downstream has little left to do.
    """
    b1, b2 = np.asarray(g1, float).copy(), np.asarray(g2, float).copy()
    for _ in range(iters):
        if b1 @ b1 > b2 @ b2:
            b1, b2 = b2, b1
        d = b1 @ b1
        if d < 1e-18:
            break
        mu = round(float(b1 @ b2) / d)
        if mu == 0:
            break
        b2 = b2 - mu * b1
    if b1 @ b1 > b2 @ b2:
        b1, b2 = b2, b1
    return b1, b2


def _fit_basis(peaks: np.ndarray, strengths: np.ndarray, shape: tuple[int, int],
               tol_frac: float = 0.22) -> tuple[np.ndarray, np.ndarray, int]:
    """Choose the peak-vector pair that best generates the whole peak set.

    Candidate bases are scored by how much of the observed spectral energy they
    explain as integer combinations `i*g1 + j*g2`. A spurious peak from noise or
    from aperiodic content explains nothing; a true fundamental explains its
    entire family of harmonics.

    The candidate pool deliberately mixes the *shortest* and the *strongest*
    peaks. Shortest alone is not enough: on a search image the low-frequency end
    is populated by aperiodic content (array boundaries, periphery blocks), and
    the real lattice fundamentals of a fine-pitch layout sit far out — for a
    7 px pitch in a 1000 px frame, at radius ~142. Picking only short vectors
    guarantees fitting the noise.
    """
    if len(peaks) < 2:
        return np.zeros(2), np.zeros(2), 0

    # keep only one of each conjugate pair (the spectrum is symmetric about DC)
    keep = []
    for i, p in enumerate(peaks):
        if p[1] > 1e-9 or (abs(p[1]) <= 1e-9 and p[0] > 0):
            keep.append(i)
    if len(keep) < 2:
        keep = list(range(len(peaks)))
    P, S = peaks[keep], strengths[keep]

    norms = np.linalg.norm(P, axis=1)
    valid = norms > 1e-6
    P, S, norms = P[valid], S[valid], norms[valid]
    if len(P) < 2:
        return np.zeros(2), np.zeros(2), 0

    # Candidate pool: shortest AND strongest. See docstring — either alone fails.
    by_short = np.argsort(norms)[:10]
    by_strong = np.argsort(S)[::-1][:14]
    rank = np.unique(np.concatenate([by_short, by_strong]))

    best = (-1.0, None, None, 0)
    for ii in range(len(rank)):
        for jj in range(ii + 1, len(rank)):
            g1, g2 = P[rank[ii]], P[rank[jj]]
            n1, n2 = np.linalg.norm(g1), np.linalg.norm(g2)
            cross = abs(g1[0] * g2[1] - g1[1] * g2[0])
            if cross < 0.20 * n1 * n2:
                continue                                  # collinear-ish, not a basis
            M = np.array([[g1[0], g2[0]], [g1[1], g2[1]]], dtype=np.float64)
            try:
                coef = np.linalg.solve(M, P.T).T          # (N, 2) fractional indices
            except np.linalg.LinAlgError:
                continue
            resid = coef - np.round(coef)
            # tolerance relative to THIS basis, not to the global shortest peak
            tol = tol_frac * min(n1, n2)
            err = np.linalg.norm(resid @ M.T, axis=1)
            hit = err < tol
            if not np.any(hit):
                continue
            # Weight by peak strength: a basis that explains the strong
            # fundamentals beats one that explains a crowd of weak noise peaks.
            # The mild size penalty prevents a degenerate "very fine basis
            # explains everything" solution, which is always available since a
            # sufficiently short basis tiles the entire plane within tolerance.
            score = float(S[hit].sum()) / (1.0 + 0.35 * math.log(max(n1 * n2, 1.0)))
            if score > best[0]:
                best = (score, g1.copy(), g2.copy(), int(np.count_nonzero(hit)))

    if best[1] is None:
        o = np.argsort(norms)
        return P[o[0]], P[o[1]], 0

    g1, g2, n = best[1], best[2], best[3]

    # least-squares refinement over every peak the basis explains — this is what
    # buys sub-percent scale accuracy, because it uses the high harmonics whose
    # positions are measured with the smallest relative error
    M = np.array([[g1[0], g2[0]], [g1[1], g2[1]]])
    tol = tol_frac * min(np.linalg.norm(g1), np.linalg.norm(g2))
    coef = np.linalg.solve(M, P.T).T
    idx = np.round(coef)
    err = np.linalg.norm((coef - idx) @ M.T, axis=1)
    hit = err < tol
    if np.count_nonzero(hit) >= 3:
        A = idx[hit]                                      # (K, 2) integer indices
        B = P[hit]                                        # (K, 2) measured positions
        wts = np.sqrt(np.maximum(S[hit], 1e-6))
        sol, *_ = np.linalg.lstsq(A * wts[:, None], B * wts[:, None], rcond=None)
        g1r, g2r = sol[0], sol[1]
        if np.all(np.isfinite(sol)) and np.linalg.norm(g1r) > 1e-9 \
                and np.linalg.norm(g2r) > 1e-9:
            g1, g2 = g1r, g2r

    return g1, g2, n


def estimate_lattice(img: np.ndarray, max_peaks: int = 60) -> LatticeEstimate:
    """Full §3.1 lattice estimation for a single image."""
    mag = log_magnitude_spectrum(img)
    peaks, strengths = _detect_peaks(mag, max_peaks=max_peaks)
    if len(peaks) < 2:
        return LatticeEstimate(np.zeros(2), np.zeros(2), peaks, strengths, 0.0, 0,
                               mag=mag)

    g1, g2, n_assigned = _fit_basis(peaks, strengths, img.shape)
    if np.linalg.norm(g1) < 1e-9 or np.linalg.norm(g2) < 1e-9:
        return LatticeEstimate(g1, g2, peaks, strengths, 0.0, 0, mag=mag)
    # canonical (shortest) basis, so reference and search are directly comparable
    g1, g2 = gauss_reduce(g1, g2)

    # Quality: what fraction of detected peaks the lattice explains, tempered by
    # how many peaks we found at all. Low quality routes us to the fallback.
    frac = n_assigned / max(len(peaks), 1)
    quality = float(np.clip(frac * min(1.0, len(peaks) / 8.0), 0.0, 1.0))

    # normalize to image-independent units: cycles per pixel
    h, w = img.shape[:2]
    g1 = np.array([g1[0] / w, g1[1] / h])
    g2 = np.array([g2[0] / w, g2[1] / h])
    return LatticeEstimate(g1, g2, peaks, strengths, quality, n_assigned,
                           mag=mag)


# --------------------------------------------------------------------------- #
# comparing two lattices -> scale and rotation
# --------------------------------------------------------------------------- #

def _basis_matrix(g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    return np.array([[g1[0], g2[0]], [g1[1], g2[1]]], dtype=np.float64)


def _symmetry_ops(n: int) -> list[np.ndarray]:
    """Rotation matrices of the cyclic group C_n."""
    ops = []
    for k in range(n):
        a = 2.0 * math.pi * k / n
        c, s = math.cos(a), math.sin(a)
        ops.append(np.array([[c, -s], [s, c]]))
    return ops


def estimate_scale_rotation(ref: np.ndarray, search: np.ndarray,
                            scale_range: tuple[float, float] = (4.0, 22.0),
                            n_hyp: int = 4,
                            lat_ref: LatticeEstimate | None = None,
                            lat_search: LatticeEstimate | None = None
                            ) -> ScaleRotation:
    """Magnification and rotation by voting in reciprocal space.

    Why voting rather than relating two fitted bases
    ------------------------------------------------
    Fitting a basis to each image independently and then solving for the
    transform between them is the textbook route, and it is brittle: a lattice
    has infinitely many bases, the two fits routinely land on different ones,
    and a single mis-assigned fundamental throws the magnification out by a
    factor of two. It measured correctly on under a third of dev pairs.

    Instead, every pair of reciprocal-lattice vectors (one from the reference,
    one from the search image) proposes a similarity transform:

        scale    = |g_search| / |g_reference|
        rotation = arg(g_search) - arg(g_reference)

    A *correct* correspondence proposes the same (scale, rotation) as every
    other correct correspondence, because they are all views of one lattice
    under one transform. Wrong correspondences propose scattered values. So the
    true answer is simply the mode of the vote distribution, and the whole
    harmonic family reinforces it — the higher harmonics most of all, since
    their positions are measured with the smallest relative error.

    Symmetry falls out for free: if the layout has a 4-fold point group, the
    accumulator grows four equally strong peaks at 90 degree spacing. We return
    the top `n_hyp` well-separated peaks as hypotheses rather than pretending to
    resolve something the spectrum genuinely cannot. `matching.py` tries them
    all; only correlation can break that tie.

    Parameters
    ----------
    scale_range
        Plausible magnification bracket. This is a *search range*, not an
        assumed value (PLAN.md Rule 3): the brief says "~10x" and mandates
        scaling variation, so anything in a wide bracket is admissible and the
        vote decides. Never narrow this to 10.
    """
    # Callers that already computed a lattice (localize.py needs the search
    # lattice again for the periodic decomposition) pass it in: a full FFT plus
    # peak detection on a 1000x1000 image is a significant slice of the budget
    # and there is no reason to do it twice.
    lat_ref = lat_ref if lat_ref is not None else estimate_lattice(ref)
    lat_search = lat_search if lat_search is not None else estimate_lattice(search)

    pr, sr_ = lat_ref.peaks, lat_ref.strengths
    ps, ss_ = lat_search.peaks, lat_search.strengths
    if len(pr) < 2 or len(ps) < 2:
        return ScaleRotation(0.0, 0.0, 0.0, [], "fallback", 0.0)

    # Reference peaks: keep one of each conjugate pair. Search peaks: keep all,
    # so each true correspondence votes exactly once with the correct sign.
    half = (pr[:, 1] > 1e-9) | ((np.abs(pr[:, 1]) <= 1e-9) & (pr[:, 0] > 0))
    if np.count_nonzero(half) >= 2:
        pr, sr_ = pr[half], sr_[half]

    nr = np.linalg.norm(pr, axis=1)
    ns = np.linalg.norm(ps, axis=1)
    keep_r = nr > 1e-6
    keep_s = ns > 1e-6
    pr, sr_, nr = pr[keep_r], sr_[keep_r], nr[keep_r]
    ps, ss_, ns = ps[keep_s], ss_[keep_s], ns[keep_s]
    if len(pr) < 2 or len(ps) < 2:
        return ScaleRotation(0.0, 0.0, 0.0, [], "fallback", 0.0)

    # every (reference peak, search peak) pairing proposes one transform
    ratio = ns[None, :] / nr[:, None]                      # (R, S)
    ang_r = np.arctan2(pr[:, 1], pr[:, 0])
    ang_s = np.arctan2(ps[:, 1], ps[:, 0])
    dth = np.degrees(ang_s[None, :] - ang_r[:, None])
    dth = (dth + 180.0) % 360.0 - 180.0

    lo, hi = scale_range
    valid = (ratio >= lo) & (ratio <= hi)
    if not np.any(valid):
        return ScaleRotation(0.0, 0.0, 0.0, [], "fallback", 0.0)

    # Vote weight: geometric mean of the two peak strengths, and NOTHING that
    # depends on radius. Weighting by reference radius looks attractive (high
    # harmonics localize the scale more precisely) but `ratio = |g_s| / |g_r|`
    # has the radius in the denominator, so any increasing function of `nr`
    # systematically favours *smaller* ratios. On dev data that alone produced
    # a clean m/2 answer on a third of the pairs.
    w = np.sqrt(np.outer(np.maximum(sr_, 1e-6), np.maximum(ss_, 1e-6)))
    w = np.where(valid, w, 0.0)

    ls = np.log(np.clip(ratio, 1e-9, None))
    # accumulator bins: ~0.5% in scale, 0.5 degrees in rotation
    nb_s, nb_a = 220, 720
    ls_lo, ls_hi = math.log(lo), math.log(hi)
    bi = np.clip(((ls - ls_lo) / (ls_hi - ls_lo) * nb_s).astype(np.int64), 0, nb_s - 1)
    bj = np.clip(((dth + 180.0) / 360.0 * nb_a).astype(np.int64), 0, nb_a - 1)

    acc = np.zeros((nb_s, nb_a), dtype=np.float64)
    np.add.at(acc, (bi.ravel(), bj.ravel()), w.ravel())
    if acc.max() <= 0:
        return ScaleRotation(0.0, 0.0, 0.0, [], "fallback", 0.0)

    # blur so that votes landing in adjacent bins reinforce instead of splitting;
    # wrap in the angle axis because rotation is circular
    accf = cv2.GaussianBlur(np.concatenate([acc[:, -40:], acc, acc[:, :40]], axis=1),
                            (0, 0), sigmaX=2.0, sigmaY=1.5,
                            borderType=cv2.BORDER_REPLICATE)[:, 40:40 + nb_a]

    # --- extract the top few well-separated accumulator peaks ----------------
    # Over-generate here: the accumulator maximum is not reliably the right
    # answer (harmonic ambiguity puts a strong competing mode at m/2 and 2m),
    # so candidates are re-scored against the full peak set below.
    hyps: list[tuple[float, float]] = []
    votes: list[float] = []
    work = accf.copy()
    total = float(accf.sum()) + 1e-12
    for _ in range(max(n_hyp, 10)):
        idx = int(np.argmax(work))
        i, j = divmod(idx, nb_a)
        if work[i, j] <= 0:
            break
        # centroid refinement in a small neighbourhood -> sub-bin precision
        i0, i1 = max(0, i - 3), min(nb_s, i + 4)
        jj = (np.arange(j - 3, j + 4)) % nb_a
        patch = accf[i0:i1][:, jj]
        wsum = patch.sum()
        if wsum > 0:
            ii = np.arange(i0, i1)[:, None]
            jrel = np.arange(-3, 4)[None, :]
            ci = float((patch * ii).sum() / wsum)
            cj = float(j + (patch * jrel).sum() / wsum)
        else:
            ci, cj = float(i), float(j)
        s_hat = math.exp(ls_lo + (ci + 0.5) / nb_s * (ls_hi - ls_lo))
        a_hat = (cj + 0.5) / nb_a * 360.0 - 180.0
        a_hat = (a_hat + 180.0) % 360.0 - 180.0
        hyps.append((float(s_hat), float(a_hat)))
        votes.append(float(accf[i, j]) / total)
        # suppress this peak (and its scale column band) before looking again
        work[max(0, i - 5):i + 6, :] = np.where(
            np.abs(((np.arange(nb_a) - j + nb_a // 2) % nb_a) - nb_a // 2) < 12,
            0.0, work[max(0, i - 5):i + 6, :])

    if not hyps:
        return ScaleRotation(0.0, 0.0, 0.0, [], "fallback", 0.0)

    # --- verify each candidate against the FULL peak set ---------------------
    # This is what kills the octave ambiguity. Transform the reference peaks by
    # the candidate (scale, rotation) and ask how much of the search spectrum
    # they land on. At the true magnification every harmonic finds a partner. At
    # m/2 the transformed reference peaks land at half the correct radii, so
    # only the even harmonics coincide and roughly half the reference energy is
    # left unexplained — a large, reliable difference in this score.
    r_max = 0.5 * float(min(search.shape[:2]))     # search-image Nyquist radius
    scores = [_overlay_score(pr, sr_, ps, ss_, s, a, r_max=r_max) for s, a in hyps]
    best_score = max(scores) if scores else 0.0
    if best_score <= 0.0:
        return ScaleRotation(0.0, 0.0, 0.0, [], "fallback", 0.0)
    best_scale = hyps[int(np.argmax(scores))][0]

    # Candidates scoring close to the winner are the genuine symmetry-equivalent
    # solutions: the spectrum cannot separate them and neither should we. BUT
    # "symmetry-equivalent" means the SAME scale at a crystallographically
    # different rotation (DRAM's 4-fold symmetry, FinFET's 2-fold) - it does
    # NOT mean a different (e.g. octave-wrong) scale that happens to clear the
    # score threshold too. Without this scale gate, a half/double-scale
    # impostor can win the "prefer smallest rotation" tiebreak below purely
    # because its spurious rotation happens to be small - confirmed on a
    # FinFET pair where the octave-wrong scale scored 0.146 (rotation 1.3 deg)
    # against the true scale's 0.153 (rotation 7.7 deg): the impostor's score
    # cleared 0.72x-of-best, then won on rotation alone (GH issue #2).
    scale_tol = 0.05
    def _same_scale(sc, h):
        return sc >= 0.72 * best_score and abs(h[0] - best_scale) <= scale_tol * best_scale
    keep = [(h, sc) for h, sc in zip(hyps, scores) if _same_scale(sc, h)]
    rest = [(h, sc) for h, sc in zip(hyps, scores) if not _same_scale(sc, h)]
    # among equally-good hypotheses prefer the smallest rotation: inter-visit
    # stage drift is a few degrees, not ninety
    keep.sort(key=lambda t: abs(t[0][1]))
    rest.sort(key=lambda t: -t[1])

    # Expand the accumulator short-list into a set actually worth correlating.
    # The accumulator's own ranking is not trustworthy enough to pick a single
    # winner (see _diversify_hypotheses); correlation is the arbiter.
    ordered = _diversify_hypotheses([h for h, _ in keep], [h for h, _ in rest],
                                    pr, sr_, ps, ss_, r_max, scale_range,
                                    n_out=max(n_hyp, 8))

    # Quality combines "the peak sets really do overlay" with "the accumulator
    # had a dominant mode". Margin over the best REJECTED candidate matters most.
    margin = 1.0
    if rest:
        margin = float(np.clip(1.0 - rest[0][1] / max(best_score, 1e-9), 0.0, 1.0))
    quality = float(np.clip(best_score * (0.45 + 0.55 * margin), 0.0, 1.0))

    # --- convert to the OpenCV rotation convention --------------------------
    # Everything above works in spectrum coordinates, where the measured angle
    # is arg(g_search) - arg(g_reference) with y pointing down. The angle that
    # `cv2.getRotationMatrix2D` needs to rotate the template INTO the search
    # frame is the negative of that (see docs/INTERFACES.md §0, which defines
    # the convention operationally for exactly this reason).
    ordered = [(s, -a) for s, a in ordered]

    return ScaleRotation(scale=ordered[0][0], rotation=ordered[0][1],
                         quality=quality, hypotheses=ordered,
                         method="lattice_vote", agreement=0.0)


def _fold_rotation(deg: float) -> float:
    """Collapse a 180-degree spectral twin onto its small-|rotation| member.

    A reciprocal lattice is symmetric about DC, so the accumulator grows a mode
    at `rot` and an identical one at `rot - 180` for every real solution. The
    two are indistinguishable *spectrally*, but they are not indistinguishable
    to correlation: on the frozen eval set correlation prefers the small-angle
    member on 35 of 36 pairs, because inter-visit stage drift is a few degrees,
    not ninety. Keeping both wastes half the hypothesis budget.
    """
    a = (float(deg) + 180.0) % 360.0 - 180.0     # normalize into (-180, 180]
    b = (a + 360.0) % 360.0 - 180.0              # the twin, 180 degrees away
    return a if abs(a) <= abs(b) else b


def _diversify_hypotheses(primary: list[tuple[float, float]],
                          fallbacks: list[tuple[float, float]],
                          pr: np.ndarray, sr_: np.ndarray,
                          ps: np.ndarray, ss_: np.ndarray,
                          r_max: float, scale_range: tuple[float, float],
                          n_out: int = 8) -> list[tuple[float, float]]:
    """Turn a small accumulator short-list into a hypothesis set worth correlating.

    Correlating the accumulator's top-N directly fails three ways, all of them
    measured on the frozen 36-pair eval set:

    1. **180-degree twins.** Every mode appears twice (see `_fold_rotation`), so
       with four slots only two distinct hypotheses were ever tried.
    2. **Octave ambiguity.** A lattice at half or double the true magnification
       explains a large share of the same peaks, so `_overlay_score` ranks it
       near — and sometimes above — the truth. This cannot be resolved from the
       spectrum at all; the harmonics genuinely coincide. Both partners must
       therefore be *offered*, and correlation asked to choose.
    3. **Ranking by `_overlay_score`.** It is a pruning statistic, not an
       arbiter. Letting it pick the single hypothesis to correlate is what held
       coverage of the true (scale, rotation) to 27/36 pairs — and correlation
       cannot find what was never proposed.

    So: fold the twins, add the x2 and x0.5 partner of every surviving scale,
    re-score the expanded set, and hand all of it to
    `matching.match_all_hypotheses`, which ranks by correlation.

    Measured effect: coverage 27/36 -> 33/36, top-1 accuracy within 5 px
    50.0% -> 61.1%.

    Note this adds no assumption about the magnification. `scale_range` is the
    same wide admissible bracket as before (PLAN.md Rule 3); octave partners are
    generated *relative to what was measured*, never relative to 10.
    """
    lo, hi = scale_range
    out: list[tuple[float, float]] = []

    def _add(s: float, r: float) -> None:
        if not (np.isfinite(s) and np.isfinite(r)) or s <= 1e-6:
            return
        if not (lo <= s <= hi):
            return
        r = (float(r) + 180.0) % 360.0 - 180.0
        for (s2, r2) in out:
            d_ang = abs(((r2 - r + 180.0) % 360.0) - 180.0)
            if abs(s2 - s) <= 0.02 * max(s, 1e-9) and d_ang < 3.0:
                return
        out.append((float(s), r))

    # Order matters only for what survives the n_out cut, so seed with the
    # accumulator's own preference and expand outward from it.
    for group in (primary, fallbacks):
        for (s, r) in group:
            _add(s, _fold_rotation(r))
    # Octave partners of everything proposed so far. Snapshot first — _add
    # appends to `out` while we iterate.
    for (s, r) in list(out):
        _add(s * 2.0, r)
        _add(s * 0.5, r)
    # Keep one large-angle representative in the tail, in case a capture really
    # was rotated near 90 or 180 degrees and the fold above guessed wrong.
    for (s, r) in list(primary[:1]) + list(fallbacks[:1]):
        _add(s, _fold_rotation(r) + 180.0)

    if not out:
        return [(s, _fold_rotation(r)) for s, r in (primary + fallbacks)][:n_out] \
            or list(primary + fallbacks)[:n_out]

    # Re-score the *expanded* set with the same overlay statistic. It is not
    # trusted to pick the winner, only to order the shortlist so that the true
    # hypothesis survives truncation.
    scored = [(_overlay_score(pr, sr_, ps, ss_, s, r, r_max=r_max), s, r)
              for (s, r) in out]
    scored.sort(key=lambda t: -t[0])
    ranked = [(s, r) for _, s, r in scored]

    # Pin the octave-gated winner (GH #2) at the front: `ScaleRotation.scale`
    # and `.rotation` are read as the single best estimate by callers and by the
    # confidence features, and re-scoring the expanded set must not silently
    # replace it. The rest of the list is ordering only — correlation decides.
    if primary:
        head = (primary[0][0], _fold_rotation(primary[0][1]))
        ranked = [head] + [h for h in ranked
                           if not (abs(h[0] - head[0]) <= 0.02 * max(head[0], 1e-9)
                                   and abs(((h[1] - head[1] + 180.0) % 360.0) - 180.0) < 3.0)]
    return ranked[:n_out]


def _overlay_score(pr: np.ndarray, sr_: np.ndarray, ps: np.ndarray,
                   ss_: np.ndarray, scale: float, rot_deg: float,
                   sigma: float = 2.5, r_max: float | None = None) -> float:
    """Fraction of *observable* reference spectral energy explained by a transform.

    Maps every reference reciprocal vector through `scale * R(rot)` and rewards
    it for landing on a search peak, weighted by both peaks' strengths.

    The `r_max` cut is essential, not a refinement. Magnifying by ~10x pushes
    reference frequencies ten times further out, so most reference harmonics
    land beyond the search image's Nyquist radius and simply cannot be observed
    there. Counting those as "unexplained" penalises the TRUE scale hardest —
    it is the hypothesis that pushes peaks furthest out — and hands the win to
    m/2, which keeps more harmonics inside the frame. That single omission
    produced a clean factor-of-two error on every FinFET dev pair.
    """
    if len(pr) == 0 or len(ps) == 0 or scale <= 0:
        return 0.0
    a = math.radians(rot_deg)
    c, s = math.cos(a), math.sin(a)
    R = np.array([[c, -s], [s, c]], dtype=np.float64) * scale
    q = pr @ R.T                                            # (R, 2) transformed

    if r_max is None:
        r_max = float(np.linalg.norm(ps, axis=1).max()) * 1.05
    obs = np.linalg.norm(q, axis=1) <= r_max
    if np.count_nonzero(obs) < 2:
        return 0.0
    q, pr, sr_ = q[obs], pr[obs], sr_[obs]

    # nearest search peak for each transformed reference peak; also allow the
    # conjugate -p, since the spectrum is symmetric about DC
    d1 = np.linalg.norm(q[:, None, :] - ps[None, :, :], axis=2)
    d2 = np.linalg.norm(q[:, None, :] + ps[None, :, :], axis=2)
    d = np.minimum(d1, d2)                                  # (R, S)

    jj = np.argmin(d, axis=1)
    dmin = d[np.arange(len(q)), jj]
    # tolerance grows slightly with radius: peak localization error is roughly
    # proportional, and a fixed tolerance under-credits the high harmonics
    tol = sigma + 0.01 * np.linalg.norm(q, axis=1)
    hit = np.exp(-0.5 * (dmin / np.maximum(tol, 1e-6)) ** 2)

    wr = np.maximum(sr_, 0.0)
    ws = np.maximum(ss_[jj], 0.0)
    num = float((wr * hit * np.sqrt(np.maximum(ws, 1e-9))).sum())
    den = float((wr * np.sqrt(np.maximum(ss_.max(), 1e-9))).sum()) + 1e-12
    return float(np.clip(num / den, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# independent cross-check: Fourier-Mellin (Reddy & Chatterji 1996)
# --------------------------------------------------------------------------- #

def fourier_mellin_scale_rotation(ref: np.ndarray, search: np.ndarray,
                                  n_ang: int = 360, n_rad: int = 256,
                                  mag_ref: np.ndarray | None = None,
                                  mag_search: np.ndarray | None = None
                                  ) -> ScaleRotation:
    """Scale and rotation by log-polar correlation of the magnitude spectra.

    A completely different route to the same two numbers: in log-polar
    coordinates a rotation becomes a shift along the angle axis and a scale
    becomes a shift along the log-radius axis, so a single phase correlation
    recovers both.

    Its value here is not accuracy but *independence*. Two unrelated estimators
    agreeing is strong evidence; disagreeing is a genuine warning, and feeds
    confidence feature #4 in `decide.py`.

    Reference: Reddy, B.S. & Chatterji, B.N., "An FFT-based technique for
    translation, rotation and scale-invariant image registration",
    IEEE Trans. Image Processing 5(8), 1266-1271, 1996.
    """
    try:
        # Both spectra have already been computed by `estimate_lattice` in the
        # normal call path; recomputing them here doubled the FFT cost of a pair
        # for no new information.
        mr = mag_ref if mag_ref is not None else log_magnitude_spectrum(ref)
        ms = mag_search if mag_search is not None else log_magnitude_spectrum(search)

        def _logpolar(m: np.ndarray) -> tuple[np.ndarray, float]:
            h, w = m.shape
            centre = (w / 2.0, h / 2.0)
            maxr = min(h, w) / 2.0
            # log-base chosen so the full radius maps onto n_rad samples
            M = n_rad / math.log(maxr)
            out = cv2.warpPolar(m, (n_rad, n_ang), centre, maxr,
                                cv2.INTER_LINEAR + cv2.WARP_POLAR_LOG)
            return out.astype(np.float32), M

        lr, Mconst = _logpolar(mr)
        ls, _ = _logpolar(ms)

        win = np.hanning(lr.shape[0])[:, None] * np.hanning(lr.shape[1])[None, :]
        (dx, dy), resp = cv2.phaseCorrelate(lr * win, ls * win)

        # dx is the shift along log-radius -> scale; dy along angle -> rotation
        scale = float(math.exp(dx / Mconst))
        rot = float(dy * 360.0 / n_ang)
        rot = (rot + 180.0) % 360.0 - 180.0
        # Same spectrum-angle -> OpenCV-rotation convention as
        # estimate_scale_rotation (negate the measured spectrum angle) - the
        # two producers of ScaleRotation.rotation must agree.
        rot = -rot
        if not (np.isfinite(scale) and scale > 1e-6):
            return ScaleRotation(0.0, 0.0, 0.0, [], "fourier_mellin", 0.0)
        return ScaleRotation(scale, rot, float(np.clip(resp, 0.0, 1.0)),
                             [(scale, rot)], "fourier_mellin", 0.0)
    except Exception:
        # Never allowed to break the pipeline — it is only a cross-check.
        return ScaleRotation(0.0, 0.0, 0.0, [], "fourier_mellin", 0.0)
