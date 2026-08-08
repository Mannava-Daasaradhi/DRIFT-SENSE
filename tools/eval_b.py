"""Member B's own scoreboard — a stand-in for C's `evaluate.py`.

C owns the real evaluation harness, the figures and the baseline chart. This is
a minimal local version so B can tune without waiting for it, and so B can break
results out by `ambiguity_class` (a miss on a degenerate pair is the expected
outcome, not a bug, and blending the two hides the whole story).

    python tools/eval_b.py --data data/dev_b
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from localize import localize                                    # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dev_b")
    ap.add_argument("--no-reranker", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    dirs = sorted(d for d in glob.glob(os.path.join(args.data, "*"))
                  if os.path.isfile(os.path.join(d, "meta.json")))
    if not dirs:
        raise SystemExit(f"no pairs under {args.data}")

    rows = []
    for d in dirs:
        meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
        r = localize(os.path.join(d, "reference.png"),
                     os.path.join(d, "search.png"),
                     use_reranker=not args.no_reranker)
        tx, ty = meta["true_center_xy"]
        err = float(np.hypot(r["x"] - tx, r["y"] - ty))

        # Did we land on a lattice-equivalent site rather than the true one?
        # On a degenerate pair that IS the expected outcome.
        aliases = meta.get("alias_positions") or []
        alias_hit = any(np.hypot(r["x"] - ax, r["y"] - ay) <= 5.0 for ax, ay in aliases)

        m_err = (abs(r["scale"] - meta["magnification_ratio"])
                 / meta["magnification_ratio"] * 100.0) if r["scale"] > 0 else 999.0

        rows.append(dict(pid=meta["pair_id"], style=meta["style"],
                         amb=meta.get("ambiguity_class") or "unknown",
                         err=err, conf=r["confidence"], pai=r["pai"],
                         dec=r["decision"], ms=r["time_ms"],
                         alias=alias_hit, m_err=m_err))
        if not args.quiet:
            print(f"{meta['pair_id']:<16} err={err:8.2f}px  conf={r['confidence']:.3f} "
                  f"pai={r['pai']:.3f}  m_err={m_err:5.2f}%  {r['decision']:<22} "
                  f"{rows[-1]['amb']}")

    errs = np.array([r["err"] for r in rows])
    ms = np.array([r["ms"] for r in rows])
    n = len(rows)

    def acc(sel, tol):
        s = [r for r in rows if sel(r)]
        return (100.0 * sum(1 for r in s if r["err"] <= tol) / len(s)) if s else float("nan")

    print(f"\n{'='*72}\nOVERALL  n={n}   median={np.median(errs):.2f}px  "
          f"mean={errs.mean():.2f}px")
    print(f"TIME     median={np.median(ms):.0f}ms  worst={ms.max():.0f}ms")
    print(f"{'':<22}" + "".join(f"{'<='+str(t)+'px':>10}" for t in (1, 2, 5, 10)))
    print(f"{'all':<22}" + "".join(f"{acc(lambda r: True, t):>9.1f}%" for t in (1, 2, 5, 10)))

    print("\nby ambiguity_class (this breakdown is the scientific story):")
    for a in ("unique", "weakly_ambiguous", "degenerate", "unknown"):
        sel = [r for r in rows if r["amb"] == a]
        if not sel:
            continue
        print(f"{'  ' + a + f' (n={len(sel)})':<22}"
              + "".join(f"{acc(lambda r, a=a: r['amb'] == a, t):>9.1f}%"
                        for t in (1, 2, 5, 10)))

    print("\nby style:")
    for s in ("dram", "finfet"):
        sel = [r for r in rows if r["style"] == s]
        if not sel:
            continue
        print(f"{'  ' + s + f' (n={len(sel)})':<22}"
              + "".join(f"{acc(lambda r, s=s: r['style'] == s, t):>9.1f}%"
                        for t in (1, 2, 5, 10)))

    dec = defaultdict(int)
    for r in rows:
        dec[r["dec"]] += 1
    print(f"\ndecisions: {dict(dec)}")
    print(f"alias-hit rate: {100.0*sum(r['alias'] for r in rows)/n:.1f}%")
    good_m = [r["m_err"] for r in rows if r["m_err"] < 900]
    print(f"scale error: median {np.median(good_m):.2f}%  "
          f"({sum(1 for v in good_m if v <= 2)}/{n} within 2%)")

    # Confidence should be higher when we are right than when we are wrong.
    # If it is not, the reliability diagram will expose it and the confidence
    # claim on Slide 5 collapses.
    hit = [r["conf"] for r in rows if r["err"] <= 5]
    miss = [r["conf"] for r in rows if r["err"] > 5]
    if hit and miss:
        print(f"confidence separation: mean(correct)={np.mean(hit):.3f} "
              f"vs mean(wrong)={np.mean(miss):.3f}")


if __name__ == "__main__":
    main()
