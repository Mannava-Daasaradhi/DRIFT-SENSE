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
    for d in dirs:
        meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
        ref = cv2.imread(os.path.join(d, "reference.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        search = cv2.imread(os.path.join(d, "search.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)

        x, y, score = oracle_predict(ref, search, meta["magnification_ratio"],
                                     meta["rotation_deg"])
        tx, ty = meta["true_center_xy"]
        e = float(np.hypot(x - tx, y - ty))
        errs.append(e)

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
    sub_px = errs[errs < 1.0]
    conv_ok = len(sub_px) >= max(3, 0.15 * len(dirs))
    print(f"convention check: {len(sub_px)}/{len(dirs)} pairs sub-pixel "
          f"(median of those = {np.median(sub_px):.3f} px)" if len(sub_px)
          else "convention check: NO sub-pixel hits at all")

    if not conv_ok:
        print("\n!! COORDINATE CONVENTION IS SUSPECT — zero or near-zero sub-pixel hits.\n"
              "   Check, in this order: (x,y) vs (row,col); the half-template-size\n"
              "   offset from matchTemplate's top-left convention; the rotation sign.\n"
              "   Do not build on this data until it passes.")
    else:
        print("convention OK — remaining error is periodic ambiguity, "
              "which is the problem we are solving, not a bug.")


if __name__ == "__main__":
    main()
