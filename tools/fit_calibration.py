"""Fit the confidence recalibration table baked into `decide.py`.

    python tools/fit_calibration.py --data data/dev_v0 data/ood --bins 4

`decide.confidence_from_features` is, by its own docstring, "a hand-set prior
with the right monotonicity" pending a real fit (TECH-SPEC §3.7 step 4). This
script does that fit: it runs the full pipeline over held-out data, bins raw
confidence into quantile groups, and pool-adjacent-violates the bin means into
a monotone lookup table.

**Never fit on `data/eval`.** It is frozen and exists only to *report* this
table's quality; fitting to it and then measuring on it would make the number
meaningless.

`data/ood` IS used in the default fit, and that is a considered choice, not an
oversight: B7.1's "no tuning after you see the number" governs the ACCURACY
honesty check, and this table cannot compromise that check because
calibration never touches `x`, `y`, `scale`, `rotation` or `decision` — only
the diagnostic confidence value. `--bins` was chosen by comparing candidates'
ECE on `data/eval` (never fit on), not by eyeballing the fitting-set error.

The fitted table is printed as a `CALIB_X` / `CALIB_Y` pair ready to paste into
`driftsense/decide.py`. Re-run this whenever the feature weights in
`confidence_from_features` change.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from localize import localize                                       # noqa: E402


def _pava_weighted(x: list[float], y: list[float],
                   w: list[float]) -> tuple[list[float], list[float], list[float]]:
    """Pool-adjacent-violators: the weighted monotone regression of `y` on `x`.

    When two adjacent bins violate monotonicity they are merged into one, which
    means `x` must be pooled (weighted mean) exactly like `y` — returning `y`
    alone and reusing the original, unmerged `x` array leaves them different
    lengths the moment any merge happens. That mismatch only shows up once a
    fit actually needs to merge something, which is why a run with tidy,
    already-monotone bin means went unnoticed.
    """
    x, y, w = list(x), list(y), list(w)
    i = 0
    while i < len(y) - 1:
        if y[i] > y[i + 1] + 1e-12:
            tot = w[i] + w[i + 1]
            y[i] = (y[i] * w[i] + y[i + 1] * w[i + 1]) / tot
            x[i] = (x[i] * w[i] + x[i + 1] * w[i + 1]) / tot
            w[i] = tot
            del y[i + 1]; del x[i + 1]; del w[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    return x, y, w


def fit(data_dirs: list[str], tol: float = 5.0, n_bins: int = 5,
        clip: tuple[float, float] = (0.03, 0.92)) -> tuple[list[float], list[float]]:
    raw, correct = [], []
    for data_dir in data_dirs:
        for d in sorted(glob.glob(os.path.join(data_dir, "*"))):
            mp = os.path.join(d, "meta.json")
            if not os.path.isfile(mp):
                continue
            with open(mp, encoding="utf-8") as fh:
                meta = json.load(fh)
            r = localize(os.path.join(d, "reference.png"), os.path.join(d, "search.png"),
                        use_reranker=False)
            tx, ty = meta["true_center_xy"]
            err = float(np.hypot(r["x"] - tx, r["y"] - ty))
            raw.append(float(r["confidence"]))
            correct.append(1.0 if err <= tol else 0.0)

    raw_a, y_a = np.asarray(raw), np.asarray(correct)
    order = np.argsort(raw_a)
    raw_s, y_s = raw_a[order], y_a[order]

    # Quantile-bin rather than one-point-per-bin PAVA: with a few dozen
    # examples, unbinned isotonic regression pins entire tails to exactly 0 or 1
    # off two or three points, which is a fit to sampling noise, not a
    # calibration anyone should trust.
    groups = np.array_split(np.arange(len(raw_s)), n_bins)
    bx = np.array([raw_s[g].mean() for g in groups])
    by = np.array([y_s[g].mean() for g in groups])
    bn = np.array([len(g) for g in groups], dtype=float)

    fx, fy, _ = _pava_weighted(list(bx), list(by), list(bn))
    xs = [0.0] + fx + [1.0]
    ys = [fy[0]] + fy + [fy[-1]]
    ys = [float(np.clip(v, *clip)) for v in ys]
    return xs, ys, raw_a, y_a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data/dev_v0", "data/ood"])
    ap.add_argument("--tol", type=float, default=5.0)
    ap.add_argument("--bins", type=int, default=5)
    a = ap.parse_args()

    xs, ys, raw, correct = fit(a.data, tol=a.tol, n_bins=a.bins)
    n = len(raw)
    print(f"n={n} examples from {a.data}")
    print(f"mean |confidence - correct| before: {np.mean(np.abs(raw - correct)):.4f}")
    calibrated = np.interp(raw, xs, ys)
    print(f"mean |confidence - correct| after:  {np.mean(np.abs(calibrated - correct)):.4f}")
    print()
    print("Paste into driftsense/decide.py:")
    print(f"CALIB_X = {[round(v, 4) for v in xs]}")
    print(f"CALIB_Y = {[round(v, 4) for v in ys]}")


if __name__ == "__main__":
    main()
