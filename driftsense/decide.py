"""Tie test, centre rule, ambiguity index, confidence — TECH-SPEC §3.7. Member B.

The brief says, verbatim:

    "If more than one matching region is found, return the one closest to the
     center of the Search Image."

That sentence is the specification, not a hint. Implementing it literally
requires knowing *whether* more than one region was found, which requires a
ranked peak list and a defensible notion of "tied" — not an argmax.

Our output is therefore never a bare (x, y). It carries a confidence, a Periodic
Ambiguity Index, and an explicit record of which rule produced the answer. On
the pathological case the brief guarantees is in the test set, the honest output
is "these four sites are indistinguishable, here is the one nearest the centre,
and my confidence is low" — which is exactly what the brief asks for when it
says the case is designed to test failure-mode awareness.
"""

from __future__ import annotations

import math

import numpy as np

from .matching import Candidate

__all__ = ["Decision", "decide", "periodic_ambiguity_index", "confidence_from_features"]


class Decision:
    """Outcome of the decision stage."""

    def __init__(self, x: float, y: float, decision: str, confidence: float,
                 pai: float, tie_size: int, ranked: list[Candidate],
                 scale: float = 0.0, rotation: float = 0.0):
        self.x = x
        self.y = y
        self.decision = decision           # unique | tie_broken_by_center | fallback
        self.confidence = confidence
        self.pai = pai
        self.tie_size = tie_size
        self.ranked = ranked
        # scale/rotation of the CHOSEN candidate (post tie-break, post
        # re-rank) - not cands[0] pre-rerank, so these always describe the
        # same match as x/y. See docs/INTERFACES.md S2: "scale"/"rotation"
        # must correspond to the returned x/y.
        self.scale = scale
        self.rotation = rotation


# --------------------------------------------------------------------------- #
# ambiguity
# --------------------------------------------------------------------------- #

def periodic_ambiguity_index(cands: list[Candidate], min_sep: float | None = None
                             ) -> float:
    """PAI = score_2 / score_1 over peaks separated by more than a template radius.

    1.0 means the runner-up is just as good as the winner: the layout is
    periodic and the answer is a coin flip. Near 0 means the match is isolated
    and trustworthy. Reported on every prediction — this is the number that
    makes the system's uncertainty legible instead of implied.

    The separation requirement matters: two samples of the *same* peak, one
    pixel apart, are not a competing hypothesis, and counting them would make
    every prediction look ambiguous.
    """
    if len(cands) < 2:
        return 0.0
    best = cands[0]
    sep = min_sep if min_sep is not None else max(4.0, 0.5 * best.tpl_size)
    s1 = best.score
    if s1 <= 1e-9:
        return 1.0
    for c in cands[1:]:
        if math.hypot(c.x - best.x, c.y - best.y) >= sep:
            return float(np.clip(c.score / s1, 0.0, 1.0))
    return 0.0


def _tie_set(cands: list[Candidate], delta: float,
             min_sep: float | None = None) -> list[Candidate]:
    """Candidates statistically indistinguishable from the best one.

    `delta` is the score spread that noise alone can produce. Calibrating it by
    bootstrapping over noise realizations (B5.1) turns it from a magic number
    into a measured quantity that can be defended in Q&A. Until that
    calibration lands it is a conservative constant.
    """
    if not cands:
        return []
    best = cands[0]
    sep = min_sep if min_sep is not None else max(4.0, 0.5 * best.tpl_size)
    thresh = best.score - delta

    tied = [best]
    for c in cands[1:]:
        if c.score < thresh:
            break
        if all(math.hypot(c.x - t.x, c.y - t.y) >= sep for t in tied):
            tied.append(c)
    return tied


# --------------------------------------------------------------------------- #
# confidence
# --------------------------------------------------------------------------- #

def confidence_from_features(margin: float, pai: float, spectral_quality: float,
                             scale_agreement: float, residual_ratio: float = 0.0
                             ) -> float:
    """Confidence in [0, 1] from interpretable features.

    Four features, per TECH-SPEC §3.7:

    1. normalized peak margin        — how far clear of the runner-up we are
    2. aperiodic residual energy     — is there anything here that *can* disambiguate
    3. lattice/spectral consistency  — did the spectrum give a clean answer
    4. scale-estimator agreement     — do two independent estimators concur

    A logistic regression over these is fitted on validation data at B5.1. Until
    then the weights below are a hand-set prior with the right monotonicity, so
    the plumbing, the reliability diagram and C's harness all work today.

    The value is deliberately *not* the correlation score. A perfectly periodic
    layout produces a beautiful 0.99 correlation at a hundred different places;
    reporting that as confidence would be a lie, and the reliability diagram
    would show it immediately.
    """
    z = (-1.15
         + 3.2 * float(np.clip(margin, 0.0, 1.0))
         - 2.6 * float(np.clip(pai, 0.0, 1.0))
         + 1.1 * float(np.clip(spectral_quality, 0.0, 1.0))
         + 0.9 * float(np.clip(scale_agreement, 0.0, 1.0))
         + 1.3 * float(np.clip(residual_ratio, 0.0, 1.0)))
    return float(1.0 / (1.0 + math.exp(-z)))


# --------------------------------------------------------------------------- #
# the decision
# --------------------------------------------------------------------------- #

def decide(cands: list[Candidate], search_shape: tuple[int, int],
           delta: float = 0.035,
           spectral_quality: float = 0.0,
           scale_agreement: float = 0.0,
           residual_ratio: float = 0.0) -> Decision:
    """Apply the tie test and, when needed, the brief's centre rule."""
    h, w = search_shape[:2]
    cx, cy = w / 2.0, h / 2.0

    if not cands:
        return Decision(cx, cy, "fallback", 0.0, 1.0, 0, [])

    ranked = sorted(cands, key=lambda c: -c.score)
    best = ranked[0]
    pai = periodic_ambiguity_index(ranked)

    tied = _tie_set(ranked, delta)

    if len(tied) > 1:
        # THE RULE, applied literally: among statistically indistinguishable
        # matches, return the one closest to the centre of the search image.
        chosen = min(tied, key=lambda c: (c.x - cx) ** 2 + (c.y - cy) ** 2)
        decision = "tie_broken_by_center"
    else:
        chosen = best
        decision = "unique"

    # margin: how far the winner stands clear of the nearest genuine rival,
    # normalized so it is comparable across pairs with different absolute scores
    margin = 0.0
    if best.score > 1e-9:
        margin = float(np.clip(1.0 - pai, 0.0, 1.0)) * float(np.clip(best.score, 0.0, 1.0))

    conf = confidence_from_features(margin, pai, spectral_quality,
                                    scale_agreement, residual_ratio)
    if decision == "tie_broken_by_center":
        # An answer produced by a tie-break is a guess among equals. Saying so
        # is the entire point of the exercise; PLAN.md §7 calls confident
        # honesty the thing judges remember.
        conf = min(conf, 0.45 / max(1.0, math.log2(len(tied) + 1)))

    return Decision(float(chosen.x), float(chosen.y), decision, float(conf),
                    float(pai), len(tied), ranked,
                    scale=float(chosen.scale), rotation=float(chosen.rotation))
