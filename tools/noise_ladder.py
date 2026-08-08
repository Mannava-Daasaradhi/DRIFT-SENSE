"""Noise-ladder robustness sweep (MEMBER-A-CHECKLIST.md A6.1).

Generates the SAME 20 pairs' geometry at 5 increasing noise levels (dose
divided by 1, 2, 4, 8, 16 - Poisson relative noise scales as
1/sqrt(dose), so this covers a 4x range in SNR) and runs them through the
real localize.py pipeline, producing the accuracy-vs-SNR evidence for
Slides 6/7: robustness under the higher noise the brief says their test
set will have.

Usage: python tools/noise_ladder.py [--num 20] [--out data/noise_ladder]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate_dataset import build_pair  # noqa: E402

DIVISORS = [1, 2, 4, 8, 16]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num", type=int, default=20)
    ap.add_argument("--out", default="data/noise_ladder")
    ap.add_argument("--seed", type=int, default=555)
    args = ap.parse_args()

    import cv2
    py = sys.executable
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    results = {}
    for div in DIVISORS:
        out_dir = os.path.join(args.out, f"div_{div}")
        os.makedirs(out_dir, exist_ok=True)
        errs = []
        for i in range(args.num):
            style = "dram" if i % 2 == 0 else "finfet"
            ref_u8, search_u8, meta = build_pair(style, i, args.seed, noise_divisor=float(div))
            pair_dir = os.path.join(out_dir, meta["pair_id"])
            os.makedirs(pair_dir, exist_ok=True)
            cv2.imwrite(os.path.join(pair_dir, "reference.png"), ref_u8)
            cv2.imwrite(os.path.join(pair_dir, "search.png"), search_u8)
            with open(os.path.join(pair_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

            out = subprocess.run(
                [py, os.path.join(root, "localize.py"),
                 os.path.join(pair_dir, "reference.png"), os.path.join(pair_dir, "search.png")],
                capture_output=True, text=True, timeout=30)
            x, y = map(float, out.stdout.strip().split(","))
            tx, ty = meta["true_center_xy"]
            errs.append(math.hypot(x - tx, y - ty))

        n = len(errs)
        w5 = sum(1 for e in errs if e <= 5) / n
        w10 = sum(1 for e in errs if e <= 10) / n
        median = sorted(errs)[n // 2]
        results[div] = {"within_5px": w5, "within_10px": w10, "median_err_px": median}
        print(f"dose/{div:2d}  within_5px={w5:.0%}  within_10px={w10:.0%}  "
              f"median_err={median:6.2f}px", file=sys.stderr)

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {os.path.join(args.out, 'results.json')}", file=sys.stderr)


if __name__ == "__main__":
    main()
