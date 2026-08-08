#!/usr/bin/env python3
"""Generate the two missing Slide-5 figures:
  figures/fft_lattice.png       — two spectra, peaks marked, magnification ratio
  figures/decomposition.png     — search / periodic / aperiodic residual

Run from the repo root:
  python3 tools/gen_slide5_figures.py --pair data/eval/dram_00022
"""
import argparse, json, os, sys
import numpy as np
import inspect

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from driftsense.preprocess import load_gray, preprocess
from driftsense.spectral import estimate_scale_rotation
from driftsense.periodic import decompose
from driftsense import viz
from driftsense.spectral import estimate_lattice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="data/eval/dram_00022")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()

    pair = os.path.join(ROOT, a.pair)
    rp = os.path.join(pair, "reference.png")
    sp = os.path.join(pair, "search.png")
    meta_path = os.path.join(pair, "meta.json")

    ref_raw = load_gray(rp)
    srch_raw = load_gray(sp)
    with open(meta_path) as f:
        meta = json.load(f)

    ref_b = preprocess(ref_raw)
    srch_b = preprocess(srch_raw)

    # ── FFT / spectral figure ──────────────────────────────────────────────────
    print("computing spectral estimate...")
    scale = meta.get("magnification_ratio", 10.0)
    peaks_ref = peaks_search = None
    lat_ref = lat_search = None
    try:
        lat_ref = estimate_lattice(ref_b.hp)
        lat_search = estimate_lattice(srch_b.hp)
        if lat_ref is not None and lat_search is not None:
            scale = float(np.linalg.norm(lat_search.basis[0]) / np.linalg.norm(lat_ref.basis[0]))
            # peaks as (dx, dy) from centre for viz
            peaks_ref = [(float(p[0]), float(p[1])) for p in (lat_ref.peaks if hasattr(lat_ref, 'peaks') else [])]
            peaks_search = [(float(p[0]), float(p[1])) for p in (lat_search.peaks if hasattr(lat_search, 'peaks') else [])]
    except Exception as e:
        print(f"  spectral estimate: {e}, using meta scale")
        lat_ref = lat_search = None

    # Compute log-magnitude spectra
    def log_mag(img):
        h, w = img.shape[:2]
        win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
        F = np.fft.fftshift(np.abs(np.fft.fft2(img.astype(np.float32) * win)))
        return np.log1p(F)

    mag_ref = log_mag(ref_b.hp)
    mag_srch = log_mag(srch_b.hp)

    fig = viz.plot_spectra(mag_ref, mag_srch,
                           peaks_ref=peaks_ref, peaks_search=peaks_search,
                           scale=scale)
    out = viz.save(fig, os.path.join(ROOT, a.out), "fft_lattice.png")
    print(f"wrote {out}")

    # ── Decomposition figure ───────────────────────────────────────────────────
    print("computing aperiodic decomposition...")
    img = srch_b.hp.astype(np.float32)
    h, w = img.shape[:2]
    try:
        # Build reciprocal-lattice frequency array from lattice estimate
        if lat_search is not None and hasattr(lat_search, 'freqs'):
            freqs = lat_search.freqs
        else:
            # Fallback: build freq grid from meta lattice periods
            periods = meta.get("lattice_period_search_px", [30, 30])
            px, py = float(periods[0]), float(periods[1])
            # Fundamental frequencies in cycles/pixel
            freqs = np.array([[1.0/px, 0.0], [0.0, 1.0/py]])
        decomp = decompose(img, freqs)
        periodic_img = decomp.periodic if hasattr(decomp, 'periodic') else decomp[0]
        aperiodic_img = decomp.aperiodic if hasattr(decomp, 'aperiodic') else decomp[1]
        ratio = float(np.var(aperiodic_img) / (np.var(img) + 1e-10))
    except Exception as e:
        print(f"  decompose fallback: {e}")
        # Pure-Fourier fallback
        F = np.fft.fft2(img)
        periods = meta.get("lattice_period_search_px", [30, 30])
        px, py = float(periods[0]), float(periods[1])
        mask = np.zeros((h, w), dtype=np.float32)
        cy, cx = h // 2, w // 2
        for ny in range(-6, 7):
            for nx in range(-6, 7):
                fy = int(round(cy + ny * h / py))
                fx = int(round(cx + nx * w / px))
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        yy, xx = fy + dy, fx + dx
                        if 0 <= yy < h and 0 <= xx < w:
                            mask[yy, xx] = 1.0
        periodic_F = np.fft.ifftshift(np.fft.fftshift(F) * mask)
        periodic_img = np.real(np.fft.ifft2(periodic_F))
        aperiodic_img = img - periodic_img
        ratio = float(np.var(aperiodic_img) / (np.var(img) + 1e-10))

    tx, ty = meta["true_center_xy"]
    fig = viz.plot_decomposition(srch_b.hp, periodic_img, aperiodic_img,
                                 true_xy=(tx, ty), ratio=ratio)
    out = viz.save(fig, os.path.join(ROOT, a.out), "decomposition.png")
    print(f"wrote {out}")
    print("done.")


if __name__ == "__main__":
    main()
