"""Figures — Member C.

Every figure the deck needs, generated from `results.json` rather than from a
live pipeline run, so the numbers on a slide and the numbers in the repository
cannot drift apart.

Four of these carry the presentation (PLAN.md §7):

* `plot_pair`           — the Slide-6 result visual the brief asks for by name:
                          reference, search, predicted location, true location.
* `plot_spectra`        — two spectra side by side with the lattice peaks marked
                          and the magnification read off the ratio. Slide 5.
* `plot_decomposition`  — search / periodic / aperiodic residual. Slide 5.
* `plot_reliability`    — predicted confidence against observed accuracy. The
                          one claim in the deck that cannot be faked, and the
                          direct answer to the brief's "failure mode awareness".

Matplotlib only — no seaborn, no styles that need a network fetch. Everything
renders headless (Agg) so `evaluate.py` works over SSH and in CI.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")                     # headless; must precede pyplot import
import matplotlib.pyplot as plt           # noqa: E402
import numpy as np                        # noqa: E402

__all__ = ["plot_pair", "plot_accuracy_bars", "plot_error_cdf", "plot_reliability",
           "plot_spectra", "plot_decomposition", "plot_accuracy_vs_noise",
           "save"]

# One colour per meaning, used identically in every figure. Readable on a
# projector and distinguishable in greyscale print.
C_OURS = "#1f77b4"
C_BASE = "#d62728"
C_TRUE = "#2ca02c"
C_PRED = "#ff7f0e"
C_ALIAS = "#9467bd"


def save(fig, out_dir: str, name: str, dpi: int = 200) -> str:
    """Write a figure and return its path. Creates `out_dir` if needed."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Slide 6 — the result visual the brief asks for by name
# --------------------------------------------------------------------------- #

def plot_pair(reference: np.ndarray, search: np.ndarray,
              pred_xy: tuple[float, float], true_xy: tuple[float, float],
              aliases: list | None = None, title: str = "",
              confidence: float | None = None, decision: str = "",
              tpl_size: float | None = None):
    """Reference, search, predicted location, true location — one figure.

    Slide 6 of the i4C template asks for exactly this set of four things, so
    the figure is built to that list rather than to what happens to look nice.

    Alias sites are drawn faintly. On a degenerate pair they are the whole
    story: they show the judge *why* the answer is a coin flip, which is what
    turns a wrong answer into an honest failure case.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2),
                             gridspec_kw={"width_ratios": [1, 1.45]})

    axes[0].imshow(reference, cmap="gray")
    axes[0].set_title("Reference (high magnification)", fontsize=11)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    ax = axes[1]
    ax.imshow(search, cmap="gray")

    for i, a in enumerate(aliases or []):
        ax.plot(a[0], a[1], "o", ms=3, mfc="none", mec=C_ALIAS, mew=0.8,
                alpha=0.55, label="lattice alias" if i == 0 else None)

    err = float(np.hypot(pred_xy[0] - true_xy[0], pred_xy[1] - true_xy[1]))
    ax.plot(true_xy[0], true_xy[1], "+", ms=18, mew=2.6, color=C_TRUE,
            label=f"true ({true_xy[0]:.1f}, {true_xy[1]:.1f})")
    ax.plot(pred_xy[0], pred_xy[1], "x", ms=16, mew=2.6, color=C_PRED,
            label=f"predicted ({pred_xy[0]:.1f}, {pred_xy[1]:.1f})")

    if tpl_size:
        ax.add_patch(plt.Rectangle((pred_xy[0] - tpl_size / 2, pred_xy[1] - tpl_size / 2),
                                   tpl_size, tpl_size, fill=False,
                                   ec=C_PRED, lw=1.4, ls="--"))

    bits = [f"error = {err:.2f} px"]
    if confidence is not None:
        bits.append(f"confidence = {confidence:.2f}")
    if decision:
        bits.append(decision.replace("_", " "))
    ax.set_title("Search image (1000x1000)   —   " + "   |   ".join(bits), fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)

    if title:
        fig.suptitle(title, fontsize=13, y=0.99)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Slides 3 and 5 — why we beat plain template matching
# --------------------------------------------------------------------------- #

def plot_accuracy_bars(ours: dict, baseline: dict,
                       tols=(1, 2, 5, 10), title: str = "Accuracy vs tolerance"):
    """Grouped bars, ours against the `cv2.matchTemplate` baseline.

    Slides 3 and 5 both ask why the approach beats template matching. A chart
    answers that better than a paragraph, and it is the only form of the answer
    a judge can verify in three seconds.
    """
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    x = np.arange(len(tols))
    w = 0.38
    ax.bar(x - w / 2, [ours[t] for t in tols], w, label="DRIFT-SENSE", color=C_OURS)
    ax.bar(x + w / 2, [baseline[t] for t in tols], w,
           label="cv2.matchTemplate @ fixed 10x", color=C_BASE)

    for i, t in enumerate(tols):
        ax.text(i - w / 2, ours[t] + 1.2, f"{ours[t]:.0f}", ha="center", fontsize=9)
        ax.text(i + w / 2, baseline[t] + 1.2, f"{baseline[t]:.0f}", ha="center", fontsize=9)

    ax.set_xticks(x, [f"<= {t} px" for t in tols])
    ax.set_ylabel("pairs within tolerance (%)")
    ax.set_ylim(0, 105)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_error_cdf(errs_ours, errs_base, title: str = "Localization error CDF"):
    """Cumulative error distribution — shows the whole story, not four points."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for errs, lab, col in ((errs_ours, "DRIFT-SENSE", C_OURS),
                           (errs_base, "cv2.matchTemplate @ fixed 10x", C_BASE)):
        e = np.sort(np.asarray(errs, dtype=float))
        if e.size == 0:
            continue
        ax.step(e, 100.0 * np.arange(1, e.size + 1) / e.size, where="post",
                label=lab, color=col, lw=2)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("Euclidean error (px, symlog)")
    ax.set_ylabel("pairs within error (%)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Slide 6 — calibration
# --------------------------------------------------------------------------- #

def plot_reliability(confidences, correct, n_bins: int = 8,
                     title: str = "Confidence reliability"):
    """Predicted confidence against observed accuracy, plus ECE.

    A system that reports 100% accuracy on this problem is either lying or has
    not understood it. This plot is the evidence that our uncertainty is honest
    rather than decorative: points near the diagonal mean a stated confidence of
    0.3 really does correspond to being right about 30% of the time.

    Returns `(fig, ece)`.
    """
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    xs, ys, ns = [], [], []
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
        if not np.any(m):
            continue
        xs.append(float(conf[m].mean()))
        ys.append(float(corr[m].mean()))
        ns.append(int(m.sum()))
        ece += (m.sum() / max(len(conf), 1)) * abs(ys[-1] - xs[-1])

    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1.2, label="perfect calibration")
    if xs:
        ax.plot(xs, ys, "o-", color=C_OURS, lw=2, ms=7, label="observed")
        for x, y, n in zip(xs, ys, ns):
            ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                        xytext=(6, -11), fontsize=8, color="#444")
    ax.set_xlabel("predicted confidence")
    ax.set_ylabel("observed accuracy (within tolerance)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"{title}   (ECE = {ece:.3f})")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig, float(ece)


# --------------------------------------------------------------------------- #
# Slide 5 — the two figures that carry the innovation claim
# --------------------------------------------------------------------------- #

def plot_spectra(mag_ref: np.ndarray, mag_search: np.ndarray,
                 peaks_ref=None, peaks_search=None,
                 scale: float | None = None, crop: float = 0.32):
    """Two log-magnitude spectra with lattice peaks marked.

    The point of the figure is that the magnification is *read off* the ratio of
    reciprocal-lattice vector lengths rather than searched for. Both panels are
    cropped around DC by the same fraction, so the search image's peaks visibly
    sit further out by exactly the magnification factor.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6))
    for ax, mag, pk, lab in ((axes[0], mag_ref, peaks_ref, "Reference"),
                             (axes[1], mag_search, peaks_search, "Search")):
        h, w = mag.shape[:2]
        cy, cx = h // 2, w // 2
        r = int(min(h, w) * crop / 2)
        ax.imshow(mag[cy - r:cy + r, cx - r:cx + r], cmap="magma")
        if pk is not None and len(pk):
            p = np.asarray(pk, dtype=float)
            keep = (np.abs(p[:, 0]) < r) & (np.abs(p[:, 1]) < r)
            ax.plot(p[keep, 0] + r, p[keep, 1] + r, "o", ms=7, mfc="none",
                    mec="#00e5ff", mew=1.4)
        ax.set_title(f"{lab} — log |FFT|", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    if scale:
        fig.suptitle("Reciprocal lattice: magnification read from the vector-length "
                     f"ratio,  m = |g_search| / |g_ref| = {scale:.3f}", fontsize=12)
    fig.tight_layout()
    return fig


def plot_decomposition(search: np.ndarray, periodic: np.ndarray,
                       aperiodic: np.ndarray, true_xy=None, ratio: float | None = None):
    """Search image -> periodic component -> aperiodic residual.

    One image that explains the entire contribution: the periodic part is
    identical at every lattice site and therefore cannot disambiguate anything,
    so it is subtracted, and what remains is exactly the content capable of
    resolving position.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
    for ax, im, lab in ((axes[0], search, "Search image"),
                        (axes[1], periodic, "Periodic component (lattice)"),
                        (axes[2], aperiodic, "Aperiodic residual — the only part that can disambiguate")):
        v = np.asarray(im, dtype=float)
        lo, hi = np.percentile(v, [1, 99])
        ax.imshow(v, cmap="gray", vmin=lo, vmax=hi if hi > lo else lo + 1e-6)
        ax.set_title(lab, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        if true_xy is not None:
            ax.plot(true_xy[0], true_xy[1], "+", ms=15, mew=2.2, color=C_TRUE)
    if ratio is not None:
        fig.suptitle(f"Aperiodic energy fraction = {ratio:.4f}   "
                     "(low = genuinely ambiguous, and the method says so)", fontsize=11)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Slides 6 / 7 — robustness under the noise their test set will have
# --------------------------------------------------------------------------- #

def plot_accuracy_vs_noise(levels, acc_ours, acc_base=None, tol: int = 5):
    """Accuracy against noise level — the brief says their test set is noisier."""
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(levels, acc_ours, "o-", color=C_OURS, lw=2, label="DRIFT-SENSE")
    if acc_base is not None:
        ax.plot(levels, acc_base, "s--", color=C_BASE, lw=2, label="cv2.matchTemplate")
    ax.set_xlabel("noise level (dose divisor — higher is noisier)")
    ax.set_ylabel(f"pairs within {tol} px (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Robustness to sensor noise")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig
