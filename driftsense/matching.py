"""Template construction and dense correlation — TECH-SPEC §3.2 / §3.3. Member B.

The one rule that matters here
------------------------------
**Do not take argmax.**

On a periodic layout the correlation surface has hundreds of near-identical
maxima, and `argmax` silently picks an arbitrary one. The brief's tie-break rule
("if more than one matching region is found, return the one closest to the
center") is an admission that the problem is ambiguous by construction — you
cannot comply with a rule about multiple matches if you only ever produce one.

So this module returns a full ranked peak list with scores. `decide.py` performs
the statistical tie test and applies the centre rule. Everything downstream
depends on the peak list being complete and honestly scored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .preprocess import Bands, preprocess

__all__ = ["Candidate", "Template", "build_template", "correlate",
           "find_peaks", "match_all_hypotheses"]


@dataclass
class Candidate:
    """One peak on the fused correlation surface."""

    x: float                 # centre of the matched region, search-image px
    y: float
    score: float             # fused ZNCC-like score, roughly [-1, 1]
    scale: float             # the hypothesis that produced it
    rotation: float
    tpl_size: int = 0        # side of the template used, px

    def as_dict(self, rank: int) -> dict:
        return {"x": float(self.x), "y": float(self.y),
                "score": float(self.score), "rank": int(rank)}


@dataclass
class Template:
    """A downscaled, rotated, preprocessed reference patch."""

    hp: np.ndarray
    grad: np.ndarray
    img: np.ndarray
    scale: float
    rotation: float

    @property
    def size(self) -> tuple[int, int]:
        return self.hp.shape[:2]


# --------------------------------------------------------------------------- #
# template construction (§3.2)
# --------------------------------------------------------------------------- #

def _inscribed_side(side: float, rot_deg: float, margin: float = 0.985) -> int:
    """Largest axis-aligned square that stays inside a rotated square.

    Rotating a square template leaves undefined corners. Cropping a fixed
    fraction (say 1/1.45, the worst case at 45 degrees) throws away a third of
    the template even when the rotation is 2 degrees, and a smaller template
    means a weaker, more ambiguous correlation peak. So the crop is sized to the
    actual angle: at 3 degrees we keep ~95% of the side, not 69%.
    """
    a = math.radians(abs(rot_deg) % 90.0)
    denom = math.cos(a) + math.sin(a)
    return max(8, int(side * margin / max(denom, 1e-6)))


def build_template(ref_bands: Bands, scale: float, rot_deg: float,
                   min_size: int = 12, max_size: int = 400) -> Template | None:
    """Downscale the reference by `1/scale`, rotate by `rot_deg`, re-preprocess.

    `INTER_AREA` is not an arbitrary choice: area averaging is what a real
    lower-magnification capture physically does — one detector pixel integrates
    the signal over a larger footprint on the wafer. It is also how a sane
    generator downsamples. Bilinear or Lanczos would leave a different aliasing
    signature in the template than in the search image, and ZNCC would pay for
    the mismatch.

    The template is preprocessed with the *same* function as the search image.
    If the two are normalized differently, the correlation surface tilts and
    peak ordering stops meaning anything.
    """
    if scale <= 1e-6 or not np.isfinite(scale):
        return None
    h, w = ref_bands.img.shape[:2]
    th, tw = int(round(h / scale)), int(round(w / scale))
    if th < min_size or tw < min_size:
        return None
    th, tw = min(th, max_size), min(tw, max_size)

    # Downscale on the *normalized* image, then redo the band split, rather
    # than downscaling the bands: blurring and decimation do not commute, and
    # doing it the other way leaves the template's high-pass cutoff at a
    # different spatial frequency from the search image's.
    small = cv2.resize(ref_bands.img, (tw, th), interpolation=cv2.INTER_AREA)

    if abs(rot_deg) > 1e-3:
        M = cv2.getRotationMatrix2D((tw / 2.0 - 0.5, th / 2.0 - 0.5), rot_deg, 1.0)
        small = cv2.warpAffine(small, M, (tw, th), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
        keep = _inscribed_side(min(th, tw), rot_deg)
        keep = min(keep, th, tw)
        y0, x0 = (th - keep) // 2, (tw - keep) // 2
        small = small[y0:y0 + keep, x0:x0 + keep]

    if min(small.shape[:2]) < min_size:
        return None

    b = preprocess(small)
    return Template(hp=b.hp, grad=b.grad, img=b.img, scale=scale, rotation=rot_deg)


# --------------------------------------------------------------------------- #
# correlation (§3.3)
# --------------------------------------------------------------------------- #

def correlate(search: Bands, tpl: Template, grad_weight: float = 0.45
              ) -> np.ndarray | None:
    """Fused ZNCC surface over the search image.

    Two channels are correlated and blended:

    * `hp`   — high-pass intensity, illumination invariant
    * `grad` — gradient magnitude

    SEM contrast is dominated by edge brightening, so the gradient channel is
    the more stable of the two across two captures taken at different dose and
    gain. It is also the more ambiguous on a periodic layout, since edges repeat
    perfectly. Blending keeps the robustness without surrendering the intensity
    channel's slightly better ability to discriminate.
    """
    sh, sw = search.hp.shape[:2]
    th, tw = tpl.hp.shape[:2]
    if th > sh or tw > sw or th < 4 or tw < 4:
        return None

    try:
        c1 = cv2.matchTemplate(search.hp, tpl.hp, cv2.TM_CCOEFF_NORMED)
        c2 = cv2.matchTemplate(search.grad, tpl.grad, cv2.TM_CCOEFF_NORMED)
    except cv2.error:
        return None

    out = (1.0 - grad_weight) * c1 + grad_weight * c2
    return np.nan_to_num(out, nan=-1.0, posinf=-1.0, neginf=-1.0)


def _subpixel(surface: np.ndarray, x: int, y: int) -> tuple[float, float]:
    """2-D quadratic fit on the 3x3 neighbourhood (TECH-SPEC §3.3 step 5)."""
    h, w = surface.shape
    if not (1 <= x < w - 1 and 1 <= y < h - 1):
        return 0.0, 0.0
    cx0, cx, cx1 = surface[y, x - 1], surface[y, x], surface[y, x + 1]
    cy0, cy1 = surface[y - 1, x], surface[y + 1, x]
    dxden = cx0 - 2.0 * cx + cx1
    dyden = cy0 - 2.0 * cx + cy1
    dx = 0.5 * (cx0 - cx1) / dxden if abs(dxden) > 1e-12 else 0.0
    dy = 0.5 * (cy0 - cy1) / dyden if abs(dyden) > 1e-12 else 0.0
    return float(np.clip(dx, -1.0, 1.0)), float(np.clip(dy, -1.0, 1.0))


def find_peaks(surface: np.ndarray, tpl_shape: tuple[int, int],
               k: int = 50, nms_frac: float = 0.5,
               scale: float = 0.0, rotation: float = 0.0) -> list[Candidate]:
    """Full peak list with non-maximum suppression and sub-pixel refinement.

    NMS radius is half the template size: two candidates closer than that
    overlap so heavily that they are the same match, not two.
    """
    th, tw = tpl_shape
    if surface.size == 0:
        return []

    r = max(1, int(nms_frac * min(th, tw) * 0.5))
    ksz = 2 * r + 1
    dil = cv2.dilate(surface, np.ones((ksz, ksz), np.uint8))
    is_peak = surface >= dil

    ys, xs = np.nonzero(is_peak)
    if ys.size == 0:
        return []
    vals = surface[ys, xs]
    order = np.argsort(vals)[::-1][:k]

    out: list[Candidate] = []
    for i in order:
        x, y = int(xs[i]), int(ys[i])
        dx, dy = _subpixel(surface, x, y)
        # matchTemplate indexes the template's TOP-LEFT; convert to the CENTRE
        # convention fixed in docs/INTERFACES.md §0. This is where the classic
        # off-by-half-template-size bug lives.
        out.append(Candidate(x=x + dx + tw / 2.0,
                             y=y + dy + th / 2.0,
                             score=float(vals[i]),
                             scale=scale, rotation=rotation,
                             tpl_size=int(min(th, tw))))
    return out


# --------------------------------------------------------------------------- #
# multi-hypothesis driver
# --------------------------------------------------------------------------- #

def match_all_hypotheses(ref: Bands, search: Bands,
                         hypotheses: list[tuple[float, float]],
                         k_per_hyp: int = 50,
                         max_hypotheses: int = 6,
                         search_freqs: np.ndarray | None = None,
                         residual_weight: float = 0.75
                         ) -> tuple[list[Candidate], dict]:
    """Correlate under every (scale, rotation) hypothesis and pool the peaks.

    The spectral stage cannot resolve the layout's point-group symmetry, so it
    hands over 2-4 equally valid hypotheses. Only correlation can break that
    tie, and it does so here: whichever hypothesis produces the strongest peak
    wins. Trying just the first one costs roughly a 25% failure rate on
    square-symmetric DRAM grids.

    Returns the pooled candidate list (best first) and a small diagnostics dict.
    """
    all_c: list[Candidate] = []
    per_hyp: list[dict] = []

    # Decompose the search image ONCE, outside the hypothesis loop.
    #
    # The residual surface is fused with the correlation surface BEFORE peaks
    # are extracted, not after. That ordering matters more than it looks: a
    # 1000x1000 search image over a ~10 px lattice contains ~10,000 lattice
    # sites, and on raw ZNCC the true site is frequently not in the top 50. Any
    # scheme that extracts a short candidate list first and rescores it
    # afterwards can only reorder sites it already kept, and the true one is
    # often not among them.
    dec_s = None
    if search_freqs is not None and len(search_freqs):
        try:
            from .periodic import decompose
            dec_s = decompose(search.hp, search_freqs)
            if dec_s.ratio <= 1e-6:
                dec_s = None
        except Exception:
            dec_s = None

    residual_ratio = 0.0
    for (s, rot) in hypotheses[:max_hypotheses]:
        tpl = build_template(ref, s, rot)
        if tpl is None:
            continue
        surf = correlate(search, tpl)
        if surf is None:
            continue

        if dec_s is not None:
            try:
                from .periodic import decompose, residual_gate
                dec_t = decompose(tpl.hp, search_freqs)
                # Gate on how much aperiodic structure the TEMPLATE actually
                # has. None means the pair is genuinely degenerate: correlating
                # noise against noise would invent a confident winner, so the
                # weight collapses to zero and the tie survives for the centre
                # rule to handle. The method degrades exactly where the problem
                # becomes unsolvable.
                gate = residual_gate(dec_t.ratio)
                residual_ratio = max(residual_ratio, gate)
                if gate > 0.0 and dec_t.aperiodic.shape[0] <= dec_s.aperiodic.shape[0] \
                        and dec_t.aperiodic.shape[1] <= dec_s.aperiodic.shape[1]:
                    rsurf = cv2.matchTemplate(dec_s.aperiodic, dec_t.aperiodic,
                                              cv2.TM_CCOEFF_NORMED)
                    rsurf = np.nan_to_num(rsurf, nan=0.0, posinf=0.0, neginf=0.0)
                    if rsurf.shape == surf.shape:
                        w = residual_weight * gate
                        surf = (1.0 - w) * surf + w * rsurf
            except Exception:
                pass

        cands = find_peaks(surf, tpl.hp.shape[:2], k=k_per_hyp,
                           scale=s, rotation=rot)
        if not cands:
            continue
        all_c.extend(cands)
        per_hyp.append({"scale": float(s), "rotation": float(rot),
                        "best": float(cands[0].score), "n": len(cands)})

    if not all_c:
        return [], {"hypotheses": per_hyp, "residual_ratio": residual_ratio}

    all_c.sort(key=lambda c: -c.score)

    # Pool across hypotheses, then suppress duplicates: the same physical site
    # is usually found under several hypotheses and must not be counted twice
    # when the tie test asks how many distinct candidates there are.
    kept: list[Candidate] = []
    for c in all_c:
        rad = max(4.0, 0.5 * c.tpl_size)
        if all((c.x - o.x) ** 2 + (c.y - o.y) ** 2 > rad * rad for o in kept):
            kept.append(c)
        if len(kept) >= k_per_hyp:
            break

    best_hyp = max(per_hyp, key=lambda d: d["best"]) if per_hyp else {}
    return kept, {"hypotheses": per_hyp, "best_hypothesis": best_hyp,
                  "residual_ratio": residual_ratio}


def rescore_with_residual(ref: Bands, search: Bands, cands: list[Candidate],
                          search_freqs: np.ndarray,
                          weight: float = 0.75) -> tuple[list[Candidate], float]:
    """Re-rank candidates using the aperiodic residual — TECH-SPEC §3.5.

    The periodic component scores identically at every lattice site, so it
    cannot break a tie; the residual is the only part that can. Here the
    candidates from `match_all_hypotheses` are rescored on the residual
    channel and the two scores are blended.

    The blend weight is *gated by how much residual there actually is*. On a
    perfectly periodic pair the residual is noise, correlating it would
    manufacture a confident-looking but arbitrary winner, and the gate collapses
    the weight to zero — leaving the tie intact so the centre rule handles it,
    which is the correct behaviour. Returns the rescored list and the residual
    energy ratio (confidence feature #2).
    """
    from .periodic import decompose

    if not cands or len(search_freqs) == 0:
        return cands, 0.0

    # one template per distinct hypothesis actually present in the candidate list
    hyps = []
    for c in cands:
        key = (round(c.scale, 4), round(c.rotation, 3))
        if key not in [h[0] for h in hyps]:
            hyps.append((key, c))
    if not hyps:
        return cands, 0.0

    dec_s = decompose(search.hp, search_freqs)
    if dec_s.ratio <= 1e-6:
        return cands, 0.0

    surfaces: dict[tuple, np.ndarray] = {}
    tpl_ratio_best = 0.0
    for key, c in hyps:
        tpl = build_template(ref, c.scale, c.rotation)
        if tpl is None:
            continue
        dec_t = decompose(tpl.hp, search_freqs)
        tpl_ratio_best = max(tpl_ratio_best, dec_t.ratio)
        th, tw = dec_t.aperiodic.shape[:2]
        if th > dec_s.aperiodic.shape[0] or tw > dec_s.aperiodic.shape[1]:
            continue
        try:
            surf = cv2.matchTemplate(dec_s.aperiodic, dec_t.aperiodic,
                                     cv2.TM_CCOEFF_NORMED)
        except cv2.error:
            continue
        surfaces[key] = (np.nan_to_num(surf, nan=0.0, posinf=0.0, neginf=0.0),
                         th, tw)

    if not surfaces:
        return cands, 0.0

    # Gate: no residual structure in the template -> no usable evidence.
    gate = float(np.clip((tpl_ratio_best - 0.02) / 0.15, 0.0, 1.0))
    if gate <= 0.0:
        return cands, tpl_ratio_best

    w = weight * gate
    out: list[Candidate] = []
    for c in cands:
        key = (round(c.scale, 4), round(c.rotation, 3))
        entry = surfaces.get(key)
        rs = 0.0
        if entry is not None:
            surf, th, tw = entry
            xi = int(round(c.x - tw / 2.0))
            yi = int(round(c.y - th / 2.0))
            if 0 <= yi < surf.shape[0] and 0 <= xi < surf.shape[1]:
                rs = float(surf[yi, xi])
        out.append(Candidate(x=c.x, y=c.y,
                             score=(1.0 - w) * c.score + w * rs,
                             scale=c.scale, rotation=c.rotation,
                             tpl_size=c.tpl_size))

    out.sort(key=lambda c: -c.score)
    return out, tpl_ratio_best


# --------------------------------------------------------------------------- #
# fallback: coarse-to-fine scale sweep (§3.1 fallback, B2.3)
# --------------------------------------------------------------------------- #

def scale_sweep(ref: Bands, search: Bands,
                lo: float = 7.0, hi: float = 14.0,
                coarse_steps: int = 15, rotations: tuple[float, ...] = (0.0,),
                k: int = 50) -> tuple[list[Candidate], dict]:
    """Brute-force scale search, used when the spectral estimate is unusable.

    Slower and less accurate than the spectral route, but it always terminates
    and always returns something. Reached when the layout has no detectable
    lattice at all — a blurred, low-contrast, or unexpected capture.

    Note this is a *search range*, not an assumed value: nothing here hardcodes
    10 (PLAN.md Rule 3). The range is deliberately wide.
    """
    best: tuple[float, float, float] | None = None       # (score, scale, rot)
    for s in np.linspace(lo, hi, coarse_steps):
        for rot in rotations:
            tpl = build_template(ref, float(s), float(rot))
            if tpl is None:
                continue
            surf = correlate(search, tpl)
            if surf is None:
                continue
            v = float(surf.max())
            if best is None or v > best[0]:
                best = (v, float(s), float(rot))

    if best is None:
        return [], {"method": "scale_sweep", "found": False}

    # fine pass around the coarse winner
    _, s0, rot0 = best
    step = (hi - lo) / max(coarse_steps - 1, 1)
    for s in np.linspace(s0 - step, s0 + step, 9):
        if s <= 0:
            continue
        tpl = build_template(ref, float(s), rot0)
        if tpl is None:
            continue
        surf = correlate(search, tpl)
        if surf is None:
            continue
        v = float(surf.max())
        if v > best[0]:
            best = (v, float(s), rot0)

    _, s_best, rot_best = best
    tpl = build_template(ref, s_best, rot_best)
    if tpl is None:
        return [], {"method": "scale_sweep", "found": False}
    surf = correlate(search, tpl)
    if surf is None:
        return [], {"method": "scale_sweep", "found": False}
    cands = find_peaks(surf, tpl.hp.shape[:2], k=k, scale=s_best, rotation=rot_best)
    return cands, {"method": "scale_sweep", "found": True,
                   "scale": s_best, "rotation": rot_best}
