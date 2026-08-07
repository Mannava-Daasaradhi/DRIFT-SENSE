"""Adversarial tests for `localize.py` — PLAN.md Rules 1, 2, 4, 5.

These matter more than another percent of accuracy. An import error or an
unhandled exception on their machine zeroes the entire Phase-2 score, and no
amount of algorithmic cleverness recovers from that.

Every single case below must exit 0 and print one parseable `x,y` line.

    python tools/test_robustness.py --pair data/dev_b/dram_00000
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def run(args: list[str], cwd: str | None = None) -> tuple[int, str, str]:
    p = subprocess.run([PY, os.path.join(ROOT, "localize.py")] + args,
                       capture_output=True, text=True, cwd=cwd or ROOT)
    return p.returncode, p.stdout, p.stderr


def check(name: str, args: list[str], cwd: str | None = None,
          expect_json: bool = False) -> bool:
    code, out, err = run(args, cwd)
    ok = True
    reason = ""

    if code != 0:
        ok, reason = False, f"exit code {code}"
    elif expect_json:
        # --json prints the full diagnostics dict INSTEAD of the x,y line
        # (docs/INTERFACES.md §3). Graders get the default form; C's tooling
        # uses this one.
        try:
            import json as _json
            obj = _json.loads(out)
            required = {"x", "y", "confidence", "pai", "candidates", "scale",
                        "rotation", "decision", "time_ms"}
            missing = required - set(obj)
            if missing:
                ok, reason = False, f"missing keys {sorted(missing)}"
        except Exception as e:
            ok, reason = False, f"not valid JSON ({e})"
    else:
        lines = [l for l in out.strip().splitlines() if l.strip()]
        if len(lines) != 1:
            ok, reason = False, f"stdout has {len(lines)} lines, expected exactly 1"
        else:
            try:
                parts = lines[0].split(",")
                x, y = float(parts[0]), float(parts[1])
                if not (np.isfinite(x) and np.isfinite(y)):
                    ok, reason = False, "non-finite coordinate"
            except Exception as e:
                ok, reason = False, f"unparseable stdout {lines[0]!r} ({e})"

    shown = out.strip() if len(out.strip()) < 60 else out.strip()[:57] + "..."
    print(f"{'PASS' if ok else 'FAIL'}  {name}"
          + (f"   <- {reason}" if not ok else f"   -> {shown}"))
    if not ok and err.strip():
        print("      stderr tail:", err.strip().splitlines()[-1][:160])
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="data/dev_b/dram_00000")
    args = ap.parse_args()

    pair = os.path.join(ROOT, args.pair) if not os.path.isabs(args.pair) else args.pair
    ref = os.path.join(pair, "reference.png")
    search = os.path.join(pair, "search.png")
    if not os.path.isfile(ref):
        raise SystemExit(f"no pair at {pair} — run tools/devgen.py first")

    tmp = tempfile.mkdtemp(prefix="driftsense_rb_")
    R = cv2.imread(ref, cv2.IMREAD_GRAYSCALE)
    S = cv2.imread(search, cv2.IMREAD_GRAYSCALE)

    # --- Rule 4: arbitrary formats, dtypes, channel counts, sizes -----------
    paths = {}
    cv2.imwrite(os.path.join(tmp, "r.jpg"), R, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(os.path.join(tmp, "s.jpg"), S, [cv2.IMWRITE_JPEG_QUALITY, 92])
    paths["jpg"] = (os.path.join(tmp, "r.jpg"), os.path.join(tmp, "s.jpg"))

    cv2.imwrite(os.path.join(tmp, "r16.tif"), (R.astype(np.uint16) * 257))
    cv2.imwrite(os.path.join(tmp, "s16.tif"), (S.astype(np.uint16) * 257))
    paths["16-bit TIFF"] = (os.path.join(tmp, "r16.tif"), os.path.join(tmp, "s16.tif"))

    rgba = cv2.cvtColor(R, cv2.COLOR_GRAY2BGRA)
    srgba = cv2.cvtColor(S, cv2.COLOR_GRAY2BGRA)
    cv2.imwrite(os.path.join(tmp, "r.png"), rgba)
    cv2.imwrite(os.path.join(tmp, "s.png"), srgba)
    paths["RGBA PNG"] = (os.path.join(tmp, "r.png"), os.path.join(tmp, "s.png"))

    cv2.imwrite(os.path.join(tmp, "r.bmp"), R)
    cv2.imwrite(os.path.join(tmp, "s.bmp"), S)
    paths["grayscale BMP"] = (os.path.join(tmp, "r.bmp"), os.path.join(tmp, "s.bmp"))

    # a path with a space in it — classic Windows failure
    spaced = os.path.join(tmp, "a folder with spaces")
    os.makedirs(spaced, exist_ok=True)
    cv2.imwrite(os.path.join(spaced, "r.png"), R)
    cv2.imwrite(os.path.join(spaced, "s.png"), S)
    paths["path with spaces"] = (os.path.join(spaced, "r.png"),
                                 os.path.join(spaced, "s.png"))

    # --- Rule 2: garbage in, coordinate out ---------------------------------
    with open(os.path.join(tmp, "corrupt.png"), "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + os.urandom(400))
    cv2.imwrite(os.path.join(tmp, "tiny.png"), np.array([[128]], dtype=np.uint8))
    # reference LARGER than the search image
    cv2.imwrite(os.path.join(tmp, "big.png"),
                cv2.resize(R, (S.shape[1] * 2, S.shape[0] * 2)))

    results = []

    print("--- Rule 5: both CLI conventions ---")
    results.append(check("positional args", [ref, search]))
    results.append(check("--ref / --search", ["--ref", ref, "--search", search]))
    results.append(check("--json", ["--ref", ref, "--search", search, "--json"],
                         expect_json=True))

    print("\n--- Rule 4: arbitrary formats and dtypes ---")
    for name, (a, b) in paths.items():
        results.append(check(name, [a, b]))

    print("\n--- Rule 2: never raises, always exits 0 ---")
    results.append(check("corrupt PNG", [os.path.join(tmp, "corrupt.png"), search]))
    results.append(check("1x1 image", [os.path.join(tmp, "tiny.png"), search]))
    results.append(check("reference larger than search",
                         [os.path.join(tmp, "big.png"), search]))
    results.append(check("nonexistent path", [os.path.join(tmp, "nope.png"), search]))
    results.append(check("search image is the corrupt one",
                         [ref, os.path.join(tmp, "corrupt.png")]))

    print("\n--- runs from a different working directory ---")
    results.append(check("cwd = temp dir", [ref, search], cwd=tmp))

    print("\n--- Rule 1: no-torch path ---")
    # Simulate torch being absent by blocking the import in a child process.
    blocker = os.path.join(tmp, "torch.py")
    with open(blocker, "w", encoding="utf-8") as f:
        f.write("raise ImportError('simulated: torch not installed')\n")
    env = dict(os.environ, PYTHONPATH=tmp)
    p = subprocess.run([PY, os.path.join(ROOT, "localize.py"), ref, search],
                       capture_output=True, text=True, cwd=ROOT, env=env)
    ok = p.returncode == 0 and len(p.stdout.strip().splitlines()) == 1
    print(f"{'PASS' if ok else 'FAIL'}  torch import fails -> classical path"
          f"   -> {p.stdout.strip()}")
    results.append(ok)

    n_ok = sum(results)
    print(f"\n{'='*60}\n{n_ok}/{len(results)} robustness checks passed")
    if n_ok != len(results):
        print("A failure here can zero the entire Phase-2 score. Fix before anything else.")


if __name__ == "__main__":
    main()
