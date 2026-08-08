"""DRIFT-SENSE synthetic pair generator (TECH-SPEC.md S4).

Generates (reference, search) image pairs for the Navigation-Error Recovery
localization task: a small high-magnification Reference capture and a
1000x1000 low-magnification Search capture in which the reference pattern
appears shrunk by ~magnification_ratio somewhere inside.

Status: the full ten-stage SEM forward model is implemented (see
driftsense/sem_physics.py) - supersampled analytic geometry, edge
brightening, beam PSF, scan distortion, charging, shading, Poisson shot
noise, and the detector chain - plus inter-capture rotation (+-3 deg) and
an objective aperiodic-content knob driving ambiguity_class
(MEMBER-A-CHECKLIST.md A1-A3). The layout geometry itself (DRAM
lines/contacts, FinFET fins/gates) remains intentionally simple; only the
imaging physics is "full".

Usage:
    python generate_dataset.py --style dram --num 30 --out data/eval --seed 42
    python generate_dataset.py --style both --num 500 --out data/train --seed 1
"""

import argparse
import json
import os

import cv2
import numpy as np

from driftsense.layouts import render_dram, render_finfet
from driftsense.sem_physics import apply_beam_psf, apply_edge_brightening, sem_forward

REF_SIZE = 1000
SEARCH_SIZE = 1000
ZERO_APERIODIC_PROB = 0.15


def _rotate_point(x, y, pivot_xy, angle_deg):
    """Rotate (x, y) the same way cv2.warpAffine(cv2.getRotationMatrix2D(...))
    would move image CONTENT - i.e. a point that was at (x, y) before the
    warp ends up here after it. Used to keep alias_positions valid once the
    search canvas itself gets rotated (see build_pair).
    """
    M = cv2.getRotationMatrix2D(pivot_xy, angle_deg, 1.0)
    vec = np.array([x, y, 1.0])
    rx, ry = M @ vec
    return float(rx), float(ry)


def _rng_for(seed, pair_index, tag):
    """Deterministic, independent sub-stream keyed by (seed, pair, purpose)."""
    return np.random.default_rng([seed, pair_index, tag])


def _make_aperiodic_content(layout_rng):
    """Aperiodic content in WORLD coordinates inside the reference window,
    driven by a single aperiodic_content_level knob (TECH-SPEC.md S4.3).

    Two kinds, both placed only within [0, REF_SIZE) x [0, REF_SIZE) - the
    exact region the reference crop covers - and rendered consistently
    (scaled) into both reference and search:

    - A periphery/array-boundary BLOCK: a solid rectangular region standing
      in for TECH-SPEC's "array boundary", "periphery block" or "dummy
      fill" content. This is the primary disambiguation signal - it must
      survive being divided by magnification_ratio (~10x) and still be a
      large, high-contrast feature in the search image. A handful of
      sub-10-world-unit defect dots (v0's original approach) do NOT survive
      that division: at m~10 they shrink to well under 1 search pixel and
      contribute essentially nothing to correlation, which is exactly why
      Member B's localizer - verified working correctly on an exaggerated
      test case - could not disambiguate any of the original "unique"
      pairs. See MEMBER-A-CHECKLIST.md A4.1.
    - A few bigger defect blobs (missing/added contacts, particles) as
      secondary texture, sized so each is still individually visible
      post-scaling (>= ~1.5 search px radius at m~10).

    level == 0 (with probability ZERO_APERIODIC_PROB) means NEITHER is
    present - a deliberately, genuinely degenerate pair that no algorithm
    can disambiguate, which is required test-set content.

    Returns (level, block, defects, aperiodic_energy_fraction).
    """
    if layout_rng.uniform() < ZERO_APERIODIC_PROB:
        return 0.0, None, [], 0.0

    level = layout_rng.uniform(0.15, 1.0)

    block_frac = 0.30 * level
    side = max(40.0, (block_frac ** 0.5) * REF_SIZE)
    side = min(side, REF_SIZE * 0.9)
    bx0 = layout_rng.uniform(0, REF_SIZE - side)
    by0 = layout_rng.uniform(0, REF_SIZE - side)
    value = float(layout_rng.choice([0.15, 0.85]))
    block = (bx0, by0, bx0 + side, by0 + side, value)
    block_area = side * side

    n = int(layout_rng.integers(2, 6))
    defects = []
    defect_area = 0.0
    for _ in range(n):
        wx = layout_rng.uniform(0, REF_SIZE)
        wy = layout_rng.uniform(0, REF_SIZE)
        radius = layout_rng.uniform(15, 35)
        sign = layout_rng.choice([-1, 1])
        defects.append((wx, wy, radius, sign))
        defect_area += np.pi * radius * radius

    fraction = min(1.0, (block_area + defect_area) / (REF_SIZE * REF_SIZE))
    return level, block, defects, fraction


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


def build_pair(style, pair_index, seed, noiseless=False, ood=False, noise_divisor=1.0):
    """Build one (reference_u8, search_u8, meta) triple. Pure function - no disk I/O.

    noiseless=True skips the sensor-noise stage entirely, for
    tools/check_gt.py's coordinate-convention self-check.

    ood=True (A5.2): a generator config NEVER used for tuning - unseen
    pitches, wider rotation, a wider magnification bracket, and 2-3x the
    noise (lower dose). One-shot honesty test: do not tune B's or C's
    parameters against this set.

    noise_divisor (A6.1, noise-ladder sweep): divides dose by this factor,
    applied AFTER the normal randomized dose draw so geometry, placement,
    and every other parameter stay byte-identical across a sweep - only
    SNR changes. Not drawn from layout_rng on purpose, for that reason.
    """
    layout_rng = _rng_for(seed, pair_index, 0)
    m = layout_rng.uniform(8.0, 13.0) if ood else layout_rng.uniform(9.0, 11.0)
    rotation_deg = layout_rng.uniform(-6.0, 6.0) if ood else layout_rng.uniform(-3.0, 3.0)
    level, block_world, defects_world, aperiodic_fraction = _make_aperiodic_content(layout_rng)
    (sx, sy), search_origin_world = _placement(layout_rng, m)

    # SEM physics stages 3 (SE yield / edge brightening) and 5 (beam PSF),
    # TECH-SPEC.md S4.2 - A2.3. Parameters are in REFERENCE-pixel units,
    # like every other length in this generator; dividing by m for the
    # search-resolution call is the same convention as pitch/line_width/etc.
    k_edge = layout_rng.uniform(0.4, 1.0)
    lambda_esc_ref = layout_rng.uniform(2.0, 6.0)
    sigma_beam_ref = layout_rng.uniform(2.0, 5.0)
    # Differential defocus (A3.2): the two captures' focus quality is NOT
    # simply the scale-driven pixel-size difference - each capture session
    # gets its own independent jitter on top of that.
    defocus_jitter_ref = layout_rng.uniform(0.85, 1.15)
    defocus_jitter_search = layout_rng.uniform(0.85, 1.15)

    def _defects_to(scale_origin=None):
        if scale_origin is None:
            return [(wx, wy, r, s) for wx, wy, r, s in defects_world]
        ox, oy, div = scale_origin
        return [((wx - ox) / div, (wy - oy) / div, max(1.0, r / div), s)
                for wx, wy, r, s in defects_world]

    def _block_to(scale_origin=None):
        if block_world is None:
            return None
        x0, y0, x1, y1, value = block_world
        if scale_origin is None:
            return (x0, y0, x1, y1, value)
        ox, oy, div = scale_origin
        return ((x0 - ox) / div, (y0 - oy) / div, (x1 - ox) / div, (y1 - oy) / div, value)

    search_scale_origin = (search_origin_world[0], search_origin_world[1], m)

    if style == "dram":
        pitch_x_ref = layout_rng.uniform(95, 150) if ood else layout_rng.uniform(45, 90)
        pitch_y_ref = layout_rng.uniform(95, 150) if ood else layout_rng.uniform(45, 90)
        line_width_ref = layout_rng.uniform(3, 6)
        contact_radius_ref = layout_rng.uniform(4, 8)
        phase_x_ref = layout_rng.uniform(0, pitch_x_ref)
        phase_y_ref = layout_rng.uniform(0, pitch_y_ref)

        pitch_x_s = pitch_x_ref / m
        pitch_y_s = pitch_y_ref / m
        phase_x_s = (phase_x_ref - search_origin_world[0]) / m
        phase_y_s = (phase_y_ref - search_origin_world[1]) / m

        ref_clean = render_dram(REF_SIZE, pitch_x_ref, pitch_y_ref, line_width_ref,
                                 contact_radius_ref, phase_x_ref, phase_y_ref,
                                 _defects_to(), _block_to())
        search_clean = render_dram(SEARCH_SIZE, pitch_x_s, pitch_y_s,
                                    max(1.0, line_width_ref / m), max(1.0, contact_radius_ref / m),
                                    phase_x_s, phase_y_s,
                                    _defects_to(search_scale_origin), _block_to(search_scale_origin))
        lattice_period_search_px = [pitch_x_s, pitch_y_s]
        sem_params = {
            "pitch_x_ref": pitch_x_ref, "pitch_y_ref": pitch_y_ref,
            "line_width_ref": line_width_ref, "contact_radius_ref": contact_radius_ref,
        }
    elif style == "finfet":
        pitch_fin_ref = layout_rng.uniform(45, 65) if ood else layout_rng.uniform(20, 40)
        fin_width_ref = layout_rng.uniform(3, 6)
        n_gates = layout_rng.integers(1, 3)
        gate_width_ref = layout_rng.uniform(15, 30)
        gate_ys_ref = sorted(layout_rng.uniform(0.25 * REF_SIZE, 0.75 * REF_SIZE, size=n_gates).tolist())
        phase_x_ref = layout_rng.uniform(0, pitch_fin_ref)
        epi_width_ref = layout_rng.uniform(20, 45)

        pitch_fin_s = pitch_fin_ref / m
        phase_x_s = (phase_x_ref - search_origin_world[0]) / m
        gate_ys_s = [(gy - search_origin_world[1]) / m for gy in gate_ys_ref]

        ref_clean = render_finfet(REF_SIZE, pitch_fin_ref, fin_width_ref, gate_ys_ref,
                                   gate_width_ref, phase_x_ref, _defects_to(), _block_to(),
                                   epi_width_ref)
        search_clean = render_finfet(SEARCH_SIZE, pitch_fin_s, max(1.0, fin_width_ref / m),
                                      gate_ys_s, max(1.0, gate_width_ref / m), phase_x_s,
                                      _defects_to(search_scale_origin), _block_to(search_scale_origin),
                                      max(1.0, epi_width_ref / m))
        lattice_period_search_px = [pitch_fin_s, None]
        sem_params = {
            "pitch_fin_ref": pitch_fin_ref, "fin_width_ref": fin_width_ref,
            "n_gates": int(n_gates), "gate_width_ref": gate_width_ref,
            "epi_width_ref": epi_width_ref,
        }
    else:
        raise ValueError(f"unknown style {style!r}")

    # A3.2: rotation is a rigid stage/sample transform between the two
    # captures, logically prior to every imaging artifact below - so it is
    # applied to the raw rendered geometry, not after edge brightening/PSF.
    # Reference stays canonical (rotation_deg describes search relative to
    # reference); rotating around the true-centre PIVOT keeps true_center_xy
    # valid without recomputing it. This must match the EXACT convention
    # operationally defined in docs/INTERFACES.md S1 (same
    # cv2.getRotationMatrix2D + warpAffine construction B's tools/check_gt.py
    # uses to verify it) - see MEMBER-A-CHECKLIST.md A3.2.
    if abs(rotation_deg) > 1e-9:
        rot_matrix = cv2.getRotationMatrix2D((sx, sy), rotation_deg, 1.0)
        search_clean = cv2.warpAffine(search_clean, rot_matrix, (SEARCH_SIZE, SEARCH_SIZE),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Stage 3 then stage 5, in that order (TECH-SPEC.md S4.2: the beam blurs
    # what edge-brightening already brightened, not the other way round).
    # lambda_esc/sigma_beam are divided by m for the search capture, same as
    # every other length parameter - the same physical scale covers fewer
    # search pixels than reference pixels - then each gets its own
    # independent defocus jitter (A3.2).
    ref_clean = apply_edge_brightening(ref_clean, k_edge, lambda_esc_ref)
    ref_clean = np.clip(apply_beam_psf(ref_clean, sigma_beam_ref * defocus_jitter_ref), 0.0, 1.0)
    search_clean = apply_edge_brightening(search_clean, k_edge, max(0.5, lambda_esc_ref / m))
    search_clean = np.clip(apply_beam_psf(
        search_clean, max(0.4, (sigma_beam_ref / m) * defocus_jitter_search)), 0.0, 1.0)

    seed_ref = int(_rng_for(seed, pair_index, 1).integers(0, 2**31 - 1))
    seed_search = int(_rng_for(seed, pair_index, 2).integers(0, 2**31 - 1))

    # Stages 6-10 (A3.1). Reference: high dose (low noise), small warp -
    # a careful high-magnification capture. Search: lower dose (higher
    # noise, per the brief's explicit "their test search images are
    # noisier"), larger warp - a faster, lower-quality navigation pass.
    dose_ref = layout_rng.uniform(150.0, 300.0)
    dose_search = layout_rng.uniform(30.0, 90.0)
    if ood:
        # Poisson relative noise ~ 1/sqrt(dose), so "2-3x the noise" needs
        # dose divided by (2-3)**2 = 4-9x, not 2-3x.
        ood_noise_divisor = layout_rng.uniform(4.0, 9.0)
        dose_ref /= ood_noise_divisor
        dose_search /= ood_noise_divisor
    dose_ref /= noise_divisor
    dose_search /= noise_divisor
    warp_amp_ref = layout_rng.uniform(0.5, 2.0)
    warp_amp_search = layout_rng.uniform(1.0, 3.0)
    charging_amplitude = layout_rng.uniform(0.1, 0.3)
    shading_amplitude = layout_rng.uniform(0.1, 0.25)

    sem_params.update({"dose_ref": round(float(dose_ref), 2),
                        "dose_search": round(float(dose_search), 2),
                        "warp_amplitude_ref": round(float(warp_amp_ref), 4),
                        "warp_amplitude_search": round(float(warp_amp_search), 4),
                        "charging_amplitude": round(float(charging_amplitude), 4),
                        "shading_amplitude": round(float(shading_amplitude), 4),
                        "aperiodic_content_level": round(float(level), 4),
                        "aperiodic_energy_fraction": round(float(aperiodic_fraction), 5),
                        "ood": bool(ood),
                        "k_edge": round(float(k_edge), 4),
                        "lambda_esc_ref": round(float(lambda_esc_ref), 4),
                        "sigma_beam_ref": round(float(sigma_beam_ref), 4),
                        "defocus_jitter_ref": round(float(defocus_jitter_ref), 4),
                        "defocus_jitter_search": round(float(defocus_jitter_search), 4)})

    if noiseless:
        ref_u8 = (np.clip(ref_clean, 0, 1) * 255.0).round().astype(np.uint8)
        search_u8 = (np.clip(search_clean, 0, 1) * 255.0).round().astype(np.uint8)
    else:
        ref_u8 = sem_forward(ref_clean, np.random.default_rng(seed_ref), dose=dose_ref,
                              warp_amplitude=warp_amp_ref, charging_amplitude=charging_amplitude,
                              shading_amplitude=shading_amplitude)
        search_u8 = sem_forward(search_clean, np.random.default_rng(seed_search), dose=dose_search,
                                 warp_amplitude=warp_amp_search, charging_amplitude=charging_amplitude,
                                 shading_amplitude=shading_amplitude)

    # Objective criterion (TECH-SPEC.md S4.1), not a defect-count vibe: how much
    # of the reference window's area is aperiodic content that survives the
    # /m division into the search image. Thresholds calibrated so a 36-pair
    # draw lands close to A5.1's target composition (~12 unique / ~16
    # weakly_ambiguous / ~8 degenerate) - level 0 pairs are provably
    # unsolvable, mid-range pairs are solvable but not trivially so.
    if aperiodic_fraction <= 0.0:
        ambiguity_class = "degenerate"
    elif aperiodic_fraction <= 0.148:
        ambiguity_class = "weakly_ambiguous"
    else:
        ambiguity_class = "unique"

    aperiodic_content = []
    if block_world is not None:
        aperiodic_content.append("periphery_block")
    if defects_world:
        aperiodic_content.append("defect")

    # Aliases are OTHER points on the same (pre-rotation) axis-aligned
    # lattice - once the search canvas is rotated around the true-centre
    # pivot, they move with it exactly like every other pixel did.
    aliases = _alias_positions(style, (sx, sy), (lattice_period_search_px[0],
                                lattice_period_search_px[1] or lattice_period_search_px[0]))
    if abs(rotation_deg) > 1e-9:
        aliases = [list(_rotate_point(ax, ay, (sx, sy), rotation_deg)) for ax, ay in aliases]
        aliases = [[round(ax, 2), round(ay, 2)] for ax, ay in aliases]

    meta = {
        "pair_id": f"{style}_{pair_index:05d}",
        "style": style,
        "true_center_xy": [round(float(sx), 3), round(float(sy), 3)],
        "magnification_ratio": round(float(m), 4),
        "rotation_deg": round(float(rotation_deg), 4),
        "lattice_period_search_px": lattice_period_search_px,
        "alias_positions": aliases,
        "ambiguity_class": ambiguity_class,
        "aperiodic_content": aperiodic_content,
        "sem_params": sem_params,
        "seeds": {"reference": seed_ref, "search": seed_search},
    }
    return ref_u8, search_u8, meta


def generate(style, num, out_dir, seed, ood=False):
    import sys

    import cv2

    if os.path.isdir(out_dir) and os.listdir(out_dir):
        print(f"warning: {out_dir!r} already exists and is not empty - "
              f"writing into it (existing pair_ids will be overwritten, "
              f"anything else left alone)", file=sys.stderr)
    os.makedirs(out_dir, exist_ok=True)
    styles = []
    for i in range(num):
        if style == "both":
            styles.append("dram" if i % 2 == 0 else "finfet")
        else:
            styles.append(style)

    for i, s in enumerate(styles):
        ref_u8, search_u8, meta = build_pair(s, i, seed, ood=ood)
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
    parser.add_argument("--ood", action="store_true",
                         help="A5.2: out-of-distribution config never used for tuning - "
                              "unseen pitches, +-6deg rotation, m in [8,13], 2-3x the noise.")
    args = parser.parse_args()

    n = generate(args.style, args.num, args.out, args.seed, ood=args.ood)
    print(f"wrote {n} pairs to {args.out}" + (" (OOD config)" if args.ood else ""))


if __name__ == "__main__":
    main()
