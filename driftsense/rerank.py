"""Siamese re-ranker — TECH-SPEC §3.6. Member C.

**Strictly optional at inference.** `localize.py` must produce a complete answer
with torch absent, the weights missing, or this module raising (PLAN.md Rule 1).
Nothing here is allowed to gate the pipeline; it only reorders candidates that
the classical stages already produced.

What it is for
--------------
After the spectral and correlation stages, the true site is inside the candidate
list on ~89% of eval pairs but is rank-0 on only ~72%. The gap is pairs where an
alias correlates marginally better than the truth. That is a *ranking* problem
over a short list, which is exactly what a small Siamese network is good at, and
it is why the re-ranker sits here rather than replacing anything.

The design decision that matters
--------------------------------
Hard negatives are the **lattice-alias positions**, taken straight from
`meta["alias_positions"]`. They look nearly identical to the positive, which is
precisely what forces the network to learn the aperiodic cues — array
boundaries, periphery blocks, dummy fill, defects — instead of the periodic
texture that is common to every site.

And the network is trained **only on pairs with genuine aperiodic content**. On a
degenerate pair there is no learnable signal at all; training on them teaches the
network to manufacture confidence where none is warranted. Degenerate pairs are
handled by the brief's centre rule, not by a CNN. This is a deliberate scope
limit, not an oversight — see Slide 5.
"""

from __future__ import annotations

import os
import sys

import numpy as np

__all__ = ["PATCH", "build_model", "rerank", "load_model", "embed_patch",
           "default_weights_path"]

#: Both patches are resampled to this size before they reach the network, so the
#: checkpoint is independent of the magnification actually measured at runtime.
PATCH = 64

_MODEL = None            # lazily loaded, cached across calls
_LOAD_FAILED = False     # never retry a load that already failed


def _torch():
    import torch
    return torch


def _build():
    """Small Siamese encoder + MLP head. ~200k parameters, well under 5 MB."""
    torch = _torch()
    nn = torch.nn

    class Encoder(nn.Module):
        def __init__(self, ch=(1, 16, 32, 64, 96)):
            super().__init__()
            blocks = []
            for i in range(len(ch) - 1):
                blocks += [nn.Conv2d(ch[i], ch[i + 1], 3, padding=1, bias=False),
                           nn.BatchNorm2d(ch[i + 1]),
                           nn.ReLU(inplace=True),
                           nn.MaxPool2d(2)]
            self.body = nn.Sequential(*blocks)
            self.pool = nn.AdaptiveAvgPool2d(1)

        def forward(self, x):
            return self.pool(self.body(x)).flatten(1)

    class SiameseReranker(nn.Module):
        """Shared encoder, feature concatenation, MLP head -> one logit.

        The head sees `[f_t, f_c, |f_t - f_c|, f_t * f_c]` rather than just the
        concatenation. The difference and product terms make "these two patches
        are the same structure" directly representable, instead of something the
        MLP has to discover from scratch on a few thousand examples.
        """

        def __init__(self, dim=96):
            super().__init__()
            self.enc = Encoder()
            self.head = nn.Sequential(
                nn.Linear(dim * 4, 128), nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, 64), nn.ReLU(inplace=True),
                nn.Linear(64, 1))

        def forward(self, t, c):
            ft, fc = self.enc(t), self.enc(c)
            return self.head(_torch().cat([ft, fc, (ft - fc).abs(), ft * fc], 1)).squeeze(1)

    return SiameseReranker()


def build_model():
    """Construct an untrained re-ranker. Requires torch; used by `train.py`."""
    return _build()


# --------------------------------------------------------------------------- #
# patch preparation — must match between training and inference
# --------------------------------------------------------------------------- #

def embed_patch(patch: np.ndarray) -> np.ndarray:
    """Resize to PATCH x PATCH and standardize to zero mean, unit variance.

    Per-patch standardization rather than a global normalization: the two
    captures differ in dose and detector gain, so any absolute intensity scale
    the network might learn would not transfer. This is the same reasoning that
    puts robust percentile normalization in `preprocess.py`.
    """
    import cv2
    a = np.asarray(patch, dtype=np.float32)
    if a.ndim != 2 or a.size == 0:
        raise ValueError(f"expected a 2-D patch, got shape {a.shape}")
    a = cv2.resize(a, (PATCH, PATCH), interpolation=cv2.INTER_AREA)
    mu, sd = float(a.mean()), float(a.std())
    return (a - mu) / (sd if sd > 1e-6 else 1.0)


def default_weights_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "weights", "reranker.pt")


def load_model(weights_path: str | None = None):
    """Load the checkpoint once and cache it. Returns None if unavailable.

    Every failure path returns None rather than raising: a missing checkpoint,
    a torch build without the right ops, or an architecture mismatch after a
    refactor must all degrade to "classical ordering stands", never to a crash.
    """
    global _MODEL, _LOAD_FAILED
    if _MODEL is not None:
        return _MODEL
    if _LOAD_FAILED:
        return None
    try:
        torch = _torch()
        # Single-threaded: torch's multi-threaded reduction order for conv/matmul
        # is not fixed, so the same weights can score candidates slightly
        # differently across machines with different core counts - enough to
        # flip a close decision. This patch is 64x64 and the model is ~200k
        # params, so the speed cost is negligible; the reproducibility is not.
        torch.set_num_threads(1)
        path = weights_path or default_weights_path()
        if not os.path.isfile(path):
            _LOAD_FAILED = True
            return None
        model = _build()
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        model.eval()
        _MODEL = model
        return model
    except Exception as e:
        print(f"[warn] re-ranker weights not usable: {type(e).__name__}: {e}",
              file=sys.stderr)
        _LOAD_FAILED = True
        return None


# --------------------------------------------------------------------------- #
# the hook Member B calls (INTERFACES.md §2)
# --------------------------------------------------------------------------- #

def rerank(template_patch: np.ndarray,
           candidate_patches: list[np.ndarray],
           weights_path: str | None = None) -> list[float] | None:
    """One score per candidate; higher means more likely to be the true match.

    Returns None if the model is unavailable, which `localize.py` reads as
    "keep the classical ordering". Runs on CPU — their machine may not have a
    GPU, and 32 candidates at 64x64 is a few milliseconds either way.
    """
    model = load_model(weights_path)
    if model is None or not candidate_patches:
        return None
    try:
        torch = _torch()
        t = embed_patch(template_patch)
        cs = [embed_patch(c) for c in candidate_patches]
        tt = torch.from_numpy(np.repeat(t[None, None], len(cs), axis=0)).float()
        cc = torch.from_numpy(np.stack(cs)[:, None]).float()
        with torch.no_grad():
            logits = model(tt, cc)
        return [float(v) for v in logits.cpu().numpy()]
    except Exception as e:
        print(f"[warn] re-ranker inference failed: {type(e).__name__}: {e}",
              file=sys.stderr)
        return None
