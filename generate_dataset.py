"""DRIFT-SENSE synthetic pair generator (v0 - crude bootstrap, see PLAN.md S3).

Generates (reference, search) image pairs for the Navigation-Error Recovery
localization task: a small high-magnification Reference capture and a
1000x1000 low-magnification Search capture in which the reference pattern
appears shrunk by ~magnification_ratio somewhere inside.

v0 status (this file): plain DRAM/FinFET line-and-contact geometry, no
supersampling, no SEM physics beyond independent Gaussian noise per
capture, no rotation. This intentionally ships fast so Member B and C are
never blocked (PLAN.md's "Day-1 unblocking trick"). It is replaced in
place by the full physics model (TECH-SPEC.md S4) across A2.x-A3.x without
changing the pair-on-disk interface in docs/INTERFACES.md.

Usage:
    python generate_dataset.py --style dram --num 30 --out data/eval --seed 42
    python generate_dataset.py --style both --num 500 --out data/train --seed 1
"""

import argparse
import json
import os

import numpy as np

from driftsense.layouts import render_dram, render_finfet
from driftsense.sem_physics import sem_forward

REF_SIZE = 1000
SEARCH_SIZE = 1000


def _rng_for(seed, pair_index, tag):
    """Deterministic, independent sub-stream keyed by (seed, pair, purpose)."""
    return np.random.default_rng([seed, pair_index, tag])


def _make_defects(layout_rng, n_range=(3, 9), zero_prob=0.15):
    """Random aperiodic blobs, in WORLD coordinates inside the reference window.

    Placing these only within [0, REF_SIZE) x [0, REF_SIZE) - the exact
    region the reference crop covers - and rendering them consistently
    (scaled) into both reference and search is what breaks the otherwise
    perfect lattice symmetry and makes the true location findable at all.
    A zero-defect pair is a deliberately, genuinely degenerate case: no
    algorithm can disambiguate it, which is required test-set content
    (TECH-SPEC.md S4.3, ambiguity_class == "degenerate").
    """
    if layout_rng.uniform() < zero_prob:
        return []
    n = layout_rng.integers(n_range[0], n_range[1] + 1)
    defects = []
    for _ in range(n):
        wx = layout_rng.uniform(0, REF_SIZE)
        wy = layout_rng.uniform(0, REF_SIZE)
        radius = layout_rng.uniform(3, 10)
        sign = layout_rng.choice([-1, 1])
        defects.append((wx, wy, radius, sign))
    return defects


def _placement(layout_rng, m):
    """Pick where the reference's true centre lands inside the search canvas.

    Returns (true_center_search_xy, search_origin_world_xy). World
    coordinates are defined so that 1 world unit == 1 reference pixel and
    the reference crop's own top-left corner is world (0, 0); a search
    pixel (u, v) then sits at world (search_origin + (u, v) * m).
    """
    margin = REF_SIZE / (2 * m) + 10
    sx = layout_rng.uniform(margin, SEARCH_SIZE - margin)
    sy = layout_rng.uniform(margin, SEARCH_SIZE - margin)
    true_center_world = np.array([REF_SIZE / 2, REF_SIZE / 2])
    search_origin_world = true_center_world - np.array([sx, sy]) * m
    return (sx, sy), search_origin_world


def _alias_positions(style, true_center_xy, pitch_search_xy, size=SEARCH_SIZE, margin=20):
    """Lattice-equivalent positions of the true centre, axis-aligned (v0 has no rotation).

    DRAM is periodic in both axes; FinFET only along the fin pitch (x) -
    the gate bar(s) are not periodic, so a y-shift does not produce a
    lattice-equivalent match.
    """
    sx, sy = true_center_xy
    px, py = pitch_search_xy
    aliases = []
    # Bounded to a local neighborhood, not every lattice point on the canvas:
    # with a small search-pitch a fully periodic array can have thousands of
    # mathematically valid aliases, which is real but useless to enumerate in
    # full - B's tie-break only needs the near ones (TECH-SPEC.md S3.7 "the
    # small set of lattice-alias positions").
    max_periods = 5
    k_range = range(-max_periods, max_periods + 1)
    if style == "dram":
        j_range = range(-max_periods, max_periods + 1)
        for k in k_range:
            for j in j_range:
                if k == 0 and j == 0:
                    continue
                ax, ay = sx + k * px, sy + j * py
                if margin <= ax <= size - margin and margin <= ay <= size - margin:
                    aliases.append([round(float(ax), 2), round(float(ay), 2)])
    else:
        for k in k_range:
            if k == 0:
                continue
            ax = sx + k * px
            if margin <= ax <= size - margin:
                aliases.append([round(float(ax), 2), round(float(sy), 2)])
    return aliases


def build_pair(style, pair_index, seed, noiseless=False):
    """Build one (reference_u8, search_u8, meta) triple. Pure function - no disk I/O.

    noiseless=True skips the sensor-noise stage entirely, for
    tools/check_gt.py's coordinate-convention self-check.
    """
    layout_rng = _rng_for(seed, pair_index, 0)
    m = layout_rng.uniform(9.0, 11.0)
    defects_world = _make_defects(layout_rng)
    (sx, sy), search_origin_world = _placement(layout_rng, m)

    if style == "dram":
        pitch_x_ref = layout_rng.uniform(45, 90)
        pitch_y_ref = layout_rng.uniform(45, 90)
        line_width_ref = layout_rng.uniform(3, 6)
        contact_radius_ref = layout_rng.uniform(4, 8)
        phase_x_ref = layout_rng.uniform(0, pitch_x_ref)
        phase_y_ref = layout_rng.uniform(0, pitch_y_ref)

        pitch_x_s = pitch_x_ref / m
        pitch_y_s = pitch_y_ref / m
        phase_x_s = (phase_x_ref - search_origin_world[0]) / m
        phase_y_s = (phase_y_ref - search_origin_world[1]) / m

        defects_ref = [(wx, wy, r, s) for wx, wy, r, s in defects_world]
        defects_s = [((wx - search_origin_world[0]) / m,
                      (wy - search_origin_world[1]) / m,
                      max(1.0, r / m), s) for wx, wy, r, s in defects_world]

        ref_clean = render_dram(REF_SIZE, pitch_x_ref, pitch_y_ref, line_width_ref,
                                 contact_radius_ref, phase_x_ref, phase_y_ref, defects_ref)
        search_clean = render_dram(SEARCH_SIZE, pitch_x_s, pitch_y_s,
                                    max(1.0, line_width_ref / m), max(1.0, contact_radius_ref / m),
                                    phase_x_s, phase_y_s, defects_s)
        lattice_period_search_px = [pitch_x_s, pitch_y_s]
        sem_params = {
            "pitch_x_ref": pitch_x_ref, "pitch_y_ref": pitch_y_ref,
            "line_width_ref": line_width_ref, "contact_radius_ref": contact_radius_ref,
        }
    elif style == "finfet":
        pitch_fin_ref = layout_rng.uniform(20, 40)
        fin_width_ref = layout_rng.uniform(3, 6)
        n_gates = layout_rng.integers(1, 3)
        gate_width_ref = layout_rng.uniform(15, 30)
        gate_ys_ref = sorted(layout_rng.uniform(0.25 * REF_SIZE, 0.75 * REF_SIZE, size=n_gates).tolist())
        phase_x_ref = layout_rng.uniform(0, pitch_fin_ref)

        pitch_fin_s = pitch_fin_ref / m
        phase_x_s = (phase_x_ref - search_origin_world[0]) / m
        gate_ys_s = [(gy - search_origin_world[1]) / m for gy in gate_ys_ref]

        defects_ref = [(wx, wy, r, s) for wx, wy, r, s in defects_world]
        defects_s = [((wx - search_origin_world[0]) / m,
                      (wy - search_origin_world[1]) / m,
                      max(1.0, r / m), s) for wx, wy, r, s in defects_world]

        ref_clean = render_finfet(REF_SIZE, pitch_fin_ref, fin_width_ref, gate_ys_ref,
                                   gate_width_ref, phase_x_ref, defects_ref)
        search_clean = render_finfet(SEARCH_SIZE, pitch_fin_s, max(1.0, fin_width_ref / m),
                                      gate_ys_s, max(1.0, gate_width_ref / m), phase_x_s, defects_s)
        lattice_period_search_px = [pitch_fin_s, None]
        sem_params = {
            "pitch_fin_ref": pitch_fin_ref, "fin_width_ref": fin_width_ref,
            "n_gates": int(n_gates), "gate_width_ref": gate_width_ref,
        }
    else:
        raise ValueError(f"unknown style {style!r}")

    seed_ref = int(_rng_for(seed, pair_index, 1).integers(0, 2**31 - 1))
    seed_search = int(_rng_for(seed, pair_index, 2).integers(0, 2**31 - 1))
    noise_std_ref = 0.02
    noise_std_search = 0.05
    sem_params.update({"noise_std_ref": noise_std_ref, "noise_std_search": noise_std_search,
                        "v0_placeholder": True})

    if noiseless:
        ref_u8 = (np.clip(ref_clean, 0, 1) * 255.0).round().astype(np.uint8)
        search_u8 = (np.clip(search_clean, 0, 1) * 255.0).round().astype(np.uint8)
    else:
        ref_u8 = sem_forward(ref_clean, np.random.default_rng(seed_ref), noise_std_ref)
        search_u8 = sem_forward(search_clean, np.random.default_rng(seed_search), noise_std_search)

    n_defects = len(defects_world)
    if n_defects == 0:
        ambiguity_class = "degenerate"
    elif n_defects <= 2:
        ambiguity_class = "weakly_ambiguous"
    else:
        ambiguity_class = "unique"

    meta = {
        "pair_id": f"{style}_{pair_index:05d}",
        "style": style,
        "true_center_xy": [round(float(sx), 3), round(float(sy), 3)],
        "magnification_ratio": round(float(m), 4),
        "rotation_deg": 0.0,
        "lattice_period_search_px": lattice_period_search_px,
        "alias_positions": _alias_positions(style, (sx, sy), (lattice_period_search_px[0],
                            lattice_period_search_px[1] or lattice_period_search_px[0])),
        "ambiguity_class": ambiguity_class,
        "aperiodic_content": ["defect"] if n_defects > 0 else [],
        "sem_params": sem_params,
        "seeds": {"reference": seed_ref, "search": seed_search},
    }
    return ref_u8, search_u8, meta


def generate(style, num, out_dir, seed):
    import cv2

    os.makedirs(out_dir, exist_ok=True)
    styles = []
    for i in range(num):
        if style == "both":
            styles.append("dram" if i % 2 == 0 else "finfet")
        else:
            styles.append(style)

    for i, s in enumerate(styles):
        ref_u8, search_u8, meta = build_pair(s, i, seed)
        pair_dir = os.path.join(out_dir, meta["pair_id"])
        os.makedirs(pair_dir, exist_ok=True)
        cv2.imwrite(os.path.join(pair_dir, "reference.png"), ref_u8)
        cv2.imwrite(os.path.join(pair_dir, "search.png"), search_u8)
        with open(os.path.join(pair_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    return len(styles)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic DRAM/FinFET reference+search pairs for "
                     "Navigation-Error Recovery localization (DRIFT-SENSE v0).")
    parser.add_argument("--style", choices=["dram", "finfet", "both"], required=True,
                         help="Die architecture to generate. 'both' alternates per pair.")
    parser.add_argument("--num", type=int, required=True, help="Number of pairs to generate.")
    parser.add_argument("--out", required=True, help="Output directory (created if missing).")
    parser.add_argument("--seed", type=int, required=True, help="Master seed; reproducible per pair.")
    args = parser.parse_args()

    n = generate(args.style, args.num, args.out, args.seed)
    print(f"wrote {n} pairs to {args.out}")


if __name__ == "__main__":
    main()
