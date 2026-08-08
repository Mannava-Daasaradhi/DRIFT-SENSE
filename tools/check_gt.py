"""Ground-truth sanity check — does `true_center_xy` actually mean what we agreed?

Runs the dumbest possible oracle: take the reference, shrink it by the *true*
magnification from `meta.json`, rotate it by the *true* rotation, and plain
`cv2.matchTemplate` it into the search image. With the true transform handed to
it, argmax must land on `true_center_xy`.

If this fails, the coordinate convention in docs/INTERFACES.md §0 is being
violated somewhere — almost always the off-by-half-template-size error, or an
(x, y) / (row, col) swap. Catching it now is worth days later.

    python tools/check_gt.py --data data/dev_b --tol 3
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import numpy as np


def oracle_predict(ref: np.ndarray, search: np.ndarray, m: float,
                   rot_deg: float) -> tuple[float, float, float]:
    """Predict the centre using the TRUE scale and rotation. Returns (x, y, score)."""
    th = max(8, int(round(ref.shape[0] / m)))
    tw = max(8, int(round(ref.shape[1] / m)))
    tpl = cv2.resize(ref, (tw, th), interpolation=cv2.INTER_AREA)

    if abs(rot_deg) > 1e-6:
        M = cv2.getRotationMatrix2D((tw / 2.0 - 0.5, th / 2.0 - 0.5), rot_deg, 1.0)
        tpl = cv2.warpAffine(tpl, M, (tw, th), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
        # crop the inscribed square so rotation-induced border does not bias ZNCC
        keep = int(min(th, tw) / 1.45)
        y0, x0 = (th - keep) // 2, (tw - keep) // 2
        tpl = tpl[y0:y0 + keep, x0:x0 + keep]
        th = tw = keep

    res = cv2.matchTemplate(search, tpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(res)
    # matchTemplate reports the template TOP-LEFT; INTERFACES.md §0 wants the CENTRE
    return loc[0] + tw / 2.0, loc[1] + th / 2.0, mx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dev_b")
    ap.add_argument("--tol", type=float, default=3.0)
    args = ap.parse_args()

    dirs = sorted(d for d in glob.glob(os.path.join(args.data, "*"))
                  if os.path.isfile(os.path.join(d, "meta.json")))
    if not dirs:
        raise SystemExit(f"no pairs found under {args.data}")

    ok = 0
    errs = []
    offsets = []            # signed (dx, dy) for pairs that found the right site
    swapped = []            # error if (x, y) had been (y, x) — the swap hypothesis
    for d in dirs:
        meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
        ref = cv2.imread(os.path.join(d, "reference.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        search = cv2.imread(os.path.join(d, "search.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)

        x, y, score = oracle_predict(ref, search, meta["magnification_ratio"],
                                     meta["rotation_deg"])
        tx, ty = meta["true_center_xy"]
        e = float(np.hypot(x - tx, y - ty))
        errs.append(e)
        swapped.append(float(np.hypot(x - ty, y - tx)))
        if e <= args.tol:
            offsets.append((x - tx, y - ty))

        # a hit on a lattice alias is not a ground-truth bug, it is the problem itself
        alias_hit = any(np.hypot(x - ax, y - ay) <= args.tol
                        for ax, ay in meta.get("alias_positions") or [])
        good = e <= args.tol
        if good:
            ok += 1
        flag = "OK   " if good else ("ALIAS" if alias_hit else "FAIL ")
        print(f"{flag} {meta['pair_id']:<16} err={e:7.2f}px  score={score:.3f}  "
              f"{meta['ambiguity_class']}")

    errs = np.array(errs)
    print(f"\n{ok}/{len(dirs)} within {args.tol} px   "
          f"median={np.median(errs):.2f}  mean={errs.mean():.2f}")

    # Two DIFFERENT questions, which must not be conflated:
    #
    #   (a) Is the coordinate convention right?  Look only at the pairs that
    #       landed on the correct site at all. If the convention were wrong —
    #       an (x,y)/(row,col) swap, a half-template offset, a rotation sign
    #       flip — there would be NO sub-pixel hits, because every hit would
    #       carry the same systematic bias.
    #
    #   (b) Is the pair solvable by plain template matching?  Usually not, and
    #       that is the entire point of the project. A large error here is a
    #       result, not a bug.
    # A convention bug is a SYSTEMATIC offset, so test for one directly rather
    # than counting sub-pixel hits. The count-based test used here previously
    # was measuring the wrong thing: scan distortion and shot noise put many
    # perfectly-correct pairs at 1-2 px, so a healthy dataset could show only a
    # handful of true sub-pixel hits and trip a "CONVENTION IS SUSPECT" banner
    # that sent people hunting a bug that was not there.
    n_hit = len(offsets)
    if n_hit >= 5:
        off = np.asarray(offsets, dtype=float)
        mdx, mdy = float(np.median(off[:, 0])), float(np.median(off[:, 1]))
        bias = float(np.hypot(mdx, mdy))
        print(f"convention check: {n_hit}/{len(dirs)} pairs landed on the true site; "
              f"median signed offset = ({mdx:+.2f}, {mdy:+.2f}) px")
        swap_better = float(np.median(swapped)) < 0.5 * float(np.median(errs))

        if swap_better:
            print("\n!! (x, y) LOOKS SWAPPED — errors drop sharply when x and y are\n"
                  "   exchanged. Fix the axis order before doing anything else.")
        elif bias > 1.5:
            print(f"\n!! SYSTEMATIC {bias:.2f} px BIAS on pairs that found the right site.\n"
                  "   That is a convention bug, not noise. Check the half-template-size\n"
                  "   offset from matchTemplate's top-left convention, then the\n"
                  "   rotation sign.")
        else:
            print("convention OK — no systematic bias. The remaining large errors are\n"
                  "   periodic ambiguity (this oracle is a plain argmax and lands on\n"
                  "   whichever lattice site scores highest), which is the problem the\n"
                  "   project solves, not a bug in the data.")
    else:
        print(f"convention check: only {n_hit} pairs landed within {args.tol} px — too "
              f"few to test for a systematic bias.\n"
              "   This is inconclusive, NOT a failure. Re-run on a larger or less\n"
              "   ambiguous split before concluding anything about the convention.")


if __name__ == "__main__":
    main()
