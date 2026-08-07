"""B1.2 acceptance test â€” does the closed-form scale estimate actually work?

Checklist bar: recovered `m` within 2% of `meta["magnification_ratio"]` on at
least 25 of 30 pairs. If this passes, the Slide-5 innovation claim is real.

    python tools/test_spectral.py --data data/dev_b
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from driftsense.preprocess import load_gray, preprocess          # noqa: E402
from driftsense.spectral import (estimate_lattice,               # noqa: E402
                                 estimate_scale_rotation,
                                 fourier_mellin_scale_rotation)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dev_b")
    ap.add_argument("--tol", type=float, default=2.0, help="percent")
    ap.add_argument("--fm", action="store_true", help="also run Fourier-Mellin")
    args = ap.parse_args()

    dirs = sorted(d for d in glob.glob(os.path.join(args.data, "*"))
                  if os.path.isfile(os.path.join(d, "meta.json")))
    if not dirs:
        raise SystemExit(f"no pairs under {args.data}")

    ok = 0
    rot_ok = 0
    errs, rerrs, times, hyp_ok = [], [], [], []
    for d in dirs:
        meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
        R = preprocess(load_gray(os.path.join(d, "reference.png")))
        S = preprocess(load_gray(os.path.join(d, "search.png")))

        t0 = time.perf_counter()
        sr = estimate_scale_rotation(R.sp, S.sp)
        times.append((time.perf_counter() - t0) * 1e3)

        m_true = meta["magnification_ratio"]
        pct = 100.0 * abs(sr.scale - m_true) / m_true if sr.scale > 0 else 999.0
        errs.append(pct)
        good = pct <= args.tol
        ok += good

        # rotation: correct if ANY symmetry hypothesis is near the truth, since
        # the lattice genuinely cannot resolve the point group on its own
        rt = meta["rotation_deg"]
        best_rot = min((abs(((h[1] - rt) + 180) % 360 - 180) for h in sr.hypotheses),
                       default=999.0)
        rerrs.append(best_rot)
        rot_good = best_rot <= 1.5
        rot_ok += rot_good

        # Being mis-RANKED is far less serious than being missing: matching.py
        # correlates against every hypothesis, so as long as the true transform
        # is somewhere in the list, the pipeline can still recover it.
        in_hyps = any(abs(h[0] - m_true) / m_true * 100 <= args.tol and
                      abs(((h[1] - rt) + 180) % 360 - 180) <= 1.5
                      for h in sr.hypotheses)
        hyp_ok.append(in_hyps)

        line = (f"{'OK ' if good else 'BAD'} {meta['pair_id']:<16} "
                f"m_true={m_true:6.3f} m_est={sr.scale:6.3f} ({pct:5.2f}%)  "
                f"rot_true={rt:+6.2f} best_hyp_err={best_rot:5.2f} "
                f"[{len(sr.hypotheses)} hyps] q={sr.quality:.2f}")
        if args.fm:
            fm = fourier_mellin_scale_rotation(R.sp, S.sp)
            line += f"  FM_m={fm.scale:6.3f}"
        print(line)

    errs = np.array(errs)
    n = len(dirs)
    print(f"\nSCALE     {ok}/{n} within {args.tol}%   "
          f"median err = {np.median(errs):.2f}%   mean = {errs.mean():.2f}%")
    print(f"ROTATION  {rot_ok}/{n} within 1.5 deg (best symmetry hypothesis)   "
          f"median = {np.median(rerrs):.2f} deg")
    print(f"IN-HYPS   {sum(hyp_ok)}/{n} have the TRUE (scale,rot) somewhere "
          f"in the hypothesis list")
    print(f"TIME      median {np.median(times):.1f} ms per pair")
    bar = 25 / 30 * n
    print(f"\n{'PASS' if ok >= bar else 'FAIL'} â€” checklist bar is "
          f"{bar:.0f}/{n} within {args.tol}%")


if __name__ == "__main__":
    main()
