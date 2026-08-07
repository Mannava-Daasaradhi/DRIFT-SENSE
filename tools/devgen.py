"""DEV-ONLY synthetic pair generator — Member B's scaffold.

⚠️  THIS IS NOT THE SUBMISSION GENERATOR.  Member A owns `generate_dataset.py`
    and `driftsense/layouts.py` / `driftsense/sem_physics.py`. This file exists
    only so that B is not idle while waiting for A1.1, and so that B always has a
    dataset with *analytically exact* ground truth to regression-test against.
    It deliberately lives in `tools/`, not in the package.

Why it still bothers to be geometrically correct
------------------------------------------------
The one thing B cannot debug without is a pair whose true centre is known to
floating-point precision. So the layout here is a continuous function of world
coordinates, and both captures *sample* that function on their own grid:

    search    pixel p -> world w = R(theta) @ (p - c) / s_S + o_S
    reference pixel p -> world w =            (p - c) / s_R + w_c

with `s_R = m * s_S`. The true centre is then the exact closed-form projection of
`w_c` into search-pixel space — no template matching, no argmax, no guessing.

Per TECH-SPEC §4.1 each magnification is rendered *separately* from the analytic
layout rather than by downsampling one giant canvas, so the two captures get
genuinely independent noise. The physics here is crude on purpose (that is A's
30% of the score); the geometry is not.

Usage
-----
    python tools/devgen.py --style both --num 30 --out data/dev_b --seed 7
"""

from __future__ import annotations

import argparse
import json
import math
import os

import cv2
import numpy as np

SS = 2  # supersample factor for anti-aliasing


# --------------------------------------------------------------------------- #
# analytic layouts, evaluated on arbitrary world coordinates
# --------------------------------------------------------------------------- #

def _stripes(coord: np.ndarray, pitch: float, width: float,
             soft: float) -> tuple[np.ndarray, np.ndarray]:
    """Soft-edged periodic stripes. Returns (fill, distance-to-edge)."""
    r = np.abs(((coord % pitch) + pitch * 0.5) % pitch - pitch * 0.5)
    fill = np.clip((width * 0.5 - r) / soft + 0.5, 0.0, 1.0)
    return fill, np.abs(r - width * 0.5)


def _edge_glow(dist: np.ndarray, lam: float) -> np.ndarray:
    """Crude stand-in for secondary-electron edge brightening.

    Real edge brightening is A's job (`sem_physics.py`, TECH-SPEC §4.2 stage 3).
    A rim is included here only so B's gradient channel is exercised by something
    other than a hard binary edge.
    """
    return np.exp(-dist / max(lam, 1e-6))


def render_dram(wx: np.ndarray, wy: np.ndarray, p: dict) -> np.ndarray:
    """DRAM: horizontal word-lines x vertical bit-lines, contact at each crossing."""
    soft = p["soft"]
    wl, dwl = _stripes(wy, p["pitch_wl"], p["w_wl"], soft)
    bl, dbl = _stripes(wx, p["pitch_bl"], p["w_bl"], soft)

    img = 0.30 * wl + 0.30 * bl
    img += 0.22 * _edge_glow(np.minimum(dwl, dbl), p["lam"]) * np.maximum(wl, bl)

    # contact/via dot at every word-line / bit-line intersection
    cx = np.abs(((wx % p["pitch_bl"]) + p["pitch_bl"] * .5) % p["pitch_bl"] - p["pitch_bl"] * .5)
    cy = np.abs(((wy % p["pitch_wl"]) + p["pitch_wl"] * .5) % p["pitch_wl"] - p["pitch_wl"] * .5)
    rad = np.hypot(cx, cy)
    dot = np.clip((p["r_contact"] - rad) / soft + 0.5, 0.0, 1.0)

    # --- aperiodic content: some contacts are missing (defects) ---------------
    if p["missing"]:
        ix = np.floor(wx / p["pitch_bl"] + 0.5).astype(np.int64)
        iy = np.floor(wy / p["pitch_wl"] + 0.5).astype(np.int64)
        h = ((ix * 73856093) ^ (iy * 19349663) ^ p["defect_seed"]) & 0xFFFF
        dot = dot * (h > int(p["missing"] * 0xFFFF))

    img += 0.45 * dot
    return img


def render_finfet(wx: np.ndarray, wy: np.ndarray, p: dict) -> np.ndarray:
    """FinFET: dense vertical fins crossed by one or two horizontal gate bars."""
    soft = p["soft"]
    fin, dfin = _stripes(wx, p["pitch_fin"], p["w_fin"], soft)

    # --- aperiodic content: fin-cut regions (short breaks in some fins) -------
    if p["fin_cut"]:
        ix = np.floor(wx / p["pitch_fin"] + 0.5).astype(np.int64)
        iy = np.floor(wy / (p["pitch_fin"] * 6.0)).astype(np.int64)
        h = ((ix * 83492791) ^ (iy * 29563577) ^ p["defect_seed"]) & 0xFFFF
        fin = fin * (h > int(p["fin_cut"] * 0xFFFF))

    img = 0.34 * fin + 0.20 * _edge_glow(dfin, p["lam"]) * fin

    # gate bars: 1 or 2 horizontal bars per gate period
    gate = np.zeros_like(wx)
    for k in range(p["n_gate"]):
        off = p["gate_off"] + k * p["gate_sep"]
        g, dg = _stripes(wy - off, p["pitch_gate"], p["w_gate"], soft)
        gate = np.maximum(gate, g + 0.55 * _edge_glow(dg, p["lam"]) * g)
    img += 0.40 * gate

    # source/drain epi: slightly brighter blocks between the gates
    sd, _ = _stripes(wy - p["gate_off"] - p["pitch_gate"] * 0.5,
                     p["pitch_gate"], p["pitch_gate"] * 0.45, soft)
    img += 0.10 * sd * fin
    return img


def _apply_aperiodic(img: np.ndarray, wx: np.ndarray, wy: np.ndarray,
                     p: dict) -> np.ndarray:
    """Array boundary + periphery block — the content that actually disambiguates.

    At `aperiodic_level == 0` none of this is drawn and the pair is genuinely
    degenerate: every lattice site is indistinguishable and only the centre rule
    from the brief can answer it. That case must exist in B's test data, because
    it is guaranteed to exist in Applied Materials' test set.
    """
    lvl = p["aperiodic_level"]
    if lvl <= 0.0:
        return img

    # array boundary: the array stops, a blank street runs across the die
    if p["boundary_at"] is not None:
        bx = p["boundary_at"]
        street = np.clip((p["street_w"] * .5 - np.abs(wx - bx)) / p["soft"] + .5, 0, 1)
        img = img * (1.0 - street) + 0.06 * street
        beyond = np.clip((wx - bx - p["street_w"] * .5) / p["soft"] + .5, 0, 1)
        img = img * (1.0 - 0.55 * lvl * beyond)

    # Periphery block: non-array circuitry — coarse, irregular, and deliberately
    # NOT periodic. A sinusoidal fill would just be a second lattice and would
    # disambiguate nothing; the texture is hashed on the coarse cell index so it
    # has no repeating period at all.
    if p["periph_at"] is not None:
        px0, py0, pw, ph = p["periph_at"]
        inside = ((wx > px0) & (wx < px0 + pw) & (wy > py0) & (wy < py0 + ph))
        s = p["periph_s"]
        ix = np.floor(wx / s).astype(np.int64)
        iy = np.floor(wy / (s * 1.3)).astype(np.int64)
        h = ((ix * 2654435761) ^ (iy * 40503) ^ p["defect_seed"]) & 0xFFFF
        coarse = (h / 65535.0)
        img = np.where(inside, 0.10 + 0.6 * lvl * coarse, img)

    return img


LAYOUTS = {"dram": render_dram, "finfet": render_finfet}


# --------------------------------------------------------------------------- #
# sampling a capture out of the analytic layout
# --------------------------------------------------------------------------- #

def _rot(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def sample_capture(style: str, p: dict, size: int, px_per_world: float,
                   world_centre: np.ndarray, theta: float) -> np.ndarray:
    """Render one capture by evaluating the layout on that capture's own grid.

    `theta` rotates the *sampling axes* relative to world axes, which is what a
    stage rotation between two visits physically does.
    """
    n = size * SS
    # pixel centres of the supersampled grid, in capture-pixel units
    u = (np.arange(n, dtype=np.float64) + 0.5) / SS - size * 0.5
    UX, UY = np.meshgrid(u, u)
    R = _rot(theta)
    wx = (R[0, 0] * UX + R[0, 1] * UY) / px_per_world + world_centre[0]
    wy = (R[1, 0] * UX + R[1, 1] * UY) / px_per_world + world_centre[1]

    img = LAYOUTS[style](wx, wy, p)
    img = _apply_aperiodic(img, wx, wy, p)

    # area-average the supersample down to the final pixel grid
    img = img.reshape(size, SS, size, SS).mean(axis=(1, 3))
    return np.clip(img, 0.0, None).astype(np.float32)


def sem_noise(clean: np.ndarray, rng: np.random.Generator, dose: float,
              sigma_beam: float, read_noise: float) -> np.ndarray:
    """Crude detection chain: blur -> Poisson -> read noise -> 8-bit quantize.

    Poisson (not Gaussian) because shot noise is the dominant SEM noise source.
    A's real forward model has ten stages; this has four. Its only job is to make
    B's matching code face a non-trivial SNR.
    """
    img = cv2.GaussianBlur(clean, (0, 0), sigmaX=sigma_beam, sigmaY=sigma_beam,
                           borderType=cv2.BORDER_REFLECT)
    lam = np.clip(img, 0.0, None) * dose + 1.0
    sig = rng.poisson(lam).astype(np.float32) / dose
    sig += rng.normal(0.0, read_noise, size=sig.shape).astype(np.float32)
    lo, hi = np.percentile(sig, [0.5, 99.5])
    sig = np.clip((sig - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (sig * 255.0 + 0.5).astype(np.uint8)


# --------------------------------------------------------------------------- #
# one pair
# --------------------------------------------------------------------------- #

def make_pair(style: str, rng: np.random.Generator, pair_id: str,
              search_size: int = 1000, ref_size: int = 1000,
              aperiodic_level: float | None = None,
              noise_boost: float = 1.0,
              clean: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    m = float(rng.uniform(9.0, 11.0))                 # magnification ratio, never 10.000
    theta_deg = float(rng.uniform(-3.0, 3.0))         # stage rotation between visits
    theta = math.radians(theta_deg)
    if aperiodic_level is None:
        aperiodic_level = float(rng.choice([0.0, 0.0, 0.35, 0.7, 1.0]))

    # search-image sampling density, in search px per world unit
    s_S = 1.0
    s_R = m * s_S

    p: dict = {
        "soft": 0.8, "lam": 1.2, "aperiodic_level": aperiodic_level,
        "defect_seed": int(rng.integers(0, 1 << 30)),
        "periph_s": float(rng.uniform(6.0, 12.0)),
        "street_w": float(rng.uniform(10.0, 20.0)),
        "boundary_at": None, "periph_at": None,
    }
    if style == "dram":
        p.update(pitch_wl=float(rng.uniform(7.0, 12.0)),
                 pitch_bl=float(rng.uniform(7.0, 12.0)),
                 r_contact=float(rng.uniform(1.2, 2.2)),
                 missing=0.06 * aperiodic_level)
        p["w_wl"] = p["pitch_wl"] * rng.uniform(0.32, 0.48)
        p["w_bl"] = p["pitch_bl"] * rng.uniform(0.32, 0.48)
    else:
        p.update(pitch_fin=float(rng.uniform(6.0, 10.0)),
                 n_gate=int(rng.integers(1, 3)),
                 gate_off=float(rng.uniform(0.0, 40.0)),
                 fin_cut=0.05 * aperiodic_level)
        p["w_fin"] = p["pitch_fin"] * rng.uniform(0.30, 0.45)
        p["pitch_gate"] = p["pitch_fin"] * rng.uniform(4.0, 7.0)
        p["w_gate"] = p["pitch_gate"] * rng.uniform(0.18, 0.30)
        p["gate_sep"] = p["pitch_gate"] * 0.5

    o_S = np.array([rng.uniform(-500, 500), rng.uniform(-500, 500)])  # world centre of search

    # --- pick the target site, keeping the whole footprint inside the frame ---
    foot = ref_size / m                       # reference footprint in search px
    margin = foot * 0.80
    tc = np.array([rng.uniform(margin, search_size - margin),
                   rng.uniform(margin, search_size - margin)])   # true centre, search px

    # invert the search sampling map to get the world point at that search pixel
    R = _rot(theta)
    w_c = R @ ((tc - search_size * 0.5) / s_S) + o_S

    # --- aperiodic content ----------------------------------------------------
    # Placement matters more than presence. A periphery block in the far corner
    # of the SEARCH image disambiguates nothing, because the REFERENCE only sees
    # a `foot`-wide window around the target site. Content that is not inside
    # that window is invisible to the matcher. So for a pair to be genuinely
    # solvable, the content has to be placed near w_c, not merely somewhere.
    half = foot * 0.5 / s_S                   # reference footprint half-width, world units
    aperiodic_content: list[str] = []
    in_footprint = False
    if aperiodic_level > 0:
        if rng.random() < 0.75:
            # array boundary street crossing (or just missing) the reference window
            near = rng.random() < aperiodic_level
            off = rng.uniform(-half * 0.5, half * 0.5) if near else rng.uniform(-400, 400)
            p["boundary_at"] = float(w_c[0] + off)
            aperiodic_content.append("array_boundary")
            in_footprint |= abs(off) < half
        # A vertical array boundary constrains x only — every lattice site along y
        # remains an equally good match. Genuine disambiguation needs a feature
        # bounded in BOTH axes, so when we want a solvable pair we always place
        # the periphery block. (This cost B an hour of confusion on Aug 7: 30/30
        # "unique" pairs were still unsolvable because the only aperiodic content
        # was a vertical street.)
        near = rng.random() < aperiodic_level
        if near or rng.random() < 0.5:
            # Size it so it cannot swallow the whole reference window: a block
            # covering the entire footprint leaves no array to register against
            # and the pair becomes unsolvable again for a different reason.
            if near:
                bw, bh = float(rng.uniform(.4 * half, 1.1 * half)), \
                         float(rng.uniform(.4 * half, 1.1 * half))
            else:
                bw, bh = float(rng.uniform(60, 220)), float(rng.uniform(60, 220))
            if near:
                cx = w_c[0] + rng.uniform(-half * .5, half * .5)
                cy = w_c[1] + rng.uniform(-half * .5, half * .5)
            else:
                cx, cy = o_S[0] + rng.uniform(-450, 450), o_S[1] + rng.uniform(-450, 450)
            p["periph_at"] = (float(cx - bw * .5), float(cy - bh * .5), bw, bh)
            aperiodic_content.append("periphery_block")
            in_footprint |= (abs(cx - w_c[0]) < half + bw * .5 and
                             abs(cy - w_c[1]) < half + bh * .5)
        if style == "dram" and p["missing"] > 0:
            aperiodic_content.append("missing_contacts")
        if style == "finfet" and p["fin_cut"] > 0:
            aperiodic_content.append("fin_cuts")

    clean_S = sample_capture(style, p, search_size, s_S, o_S, theta)
    clean_R = sample_capture(style, p, ref_size, s_R, w_c, 0.0)

    # two captures, two independent RNG streams (mandatory per the brief)
    # `clean` is for verifying the coordinate convention only (tools/check_gt.py):
    # it removes the noise so that any residual error is unambiguously a geometry bug.
    seed_r, seed_s = int(rng.integers(0, 1 << 31)), int(rng.integers(0, 1 << 31))
    dose_r, dose_s = (1e5, 1e5) if clean else (140.0, 40.0 / noise_boost)
    rn_r, rn_s = (0.0, 0.0) if clean else (0.010, 0.030 * noise_boost)
    ref = sem_noise(clean_R, np.random.default_rng(seed_r),
                    dose=dose_r, sigma_beam=rng.uniform(0.6, 1.1), read_noise=rn_r)
    search = sem_noise(clean_S, np.random.default_rng(seed_s),
                       dose=dose_s, sigma_beam=rng.uniform(0.7, 1.3), read_noise=rn_s)

    # --- lattice aliases, in closed form: we drew the lattice, so we know it ---
    if style == "dram":
        a1_w, a2_w = np.array([p["pitch_bl"], 0.0]), np.array([0.0, p["pitch_wl"]])
    else:
        a1_w, a2_w = np.array([p["pitch_fin"], 0.0]), np.array([0.0, p["pitch_gate"]])
    Rinv = _rot(-theta)
    a1, a2 = Rinv @ a1_w * s_S, Rinv @ a2_w * s_S

    aliases = []
    span = int(search_size / max(np.linalg.norm(a1), np.linalg.norm(a2), 1.0)) + 2
    for i in range(-span, span + 1):
        for j in range(-span, span + 1):
            if i == 0 and j == 0:
                continue
            q = tc + i * a1 + j * a2
            if margin <= q[0] <= search_size - margin and \
               margin <= q[1] <= search_size - margin:
                aliases.append([round(float(q[0]), 3), round(float(q[1]), 3)])

    # Label by what is actually VISIBLE to the matcher, not by the knob setting.
    # (A owns the real objective criterion in `generate_dataset.py`; this is the
    #  dev-set approximation of it.)
    if aperiodic_level <= 0.0:
        amb = "degenerate"          # no aperiodic content anywhere: unsolvable by construction
    elif in_footprint:
        amb = "unique"              # structural content inside the reference window
    else:
        amb = "weakly_ambiguous"    # only defect-level cues (dropped contacts / fin cuts)

    meta = {
        "pair_id": pair_id,
        "style": style,
        "true_center_xy": [round(float(tc[0]), 3), round(float(tc[1]), 3)],
        "magnification_ratio": round(m, 5),
        # Angle to pass to cv2.getRotationMatrix2D when rotating the downscaled
        # REFERENCE so that it aligns with the SEARCH image. See INTERFACES.md §0.
        "rotation_deg": round(theta_deg, 4),
        "lattice_period_search_px": [round(float(np.linalg.norm(a1)), 4),
                                     round(float(np.linalg.norm(a2)), 4)],
        "alias_positions": aliases,
        "ambiguity_class": amb,
        "aperiodic_content": aperiodic_content,
        "sem_params": {"aperiodic_level": aperiodic_level, "noise_boost": noise_boost},
        "seeds": {"reference": seed_r, "search": seed_s},
        "_generator": "tools/devgen.py (Member B dev scaffold, NOT the submission generator)",
    }
    return ref, search, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--style", choices=["dram", "finfet", "both"], default="both")
    ap.add_argument("--num", type=int, default=30)
    ap.add_argument("--out", default="data/dev_b")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--search-size", type=int, default=1000)
    ap.add_argument("--ref-size", type=int, default=1000)
    ap.add_argument("--noise-boost", type=float, default=1.0,
                    help="multiply search-image noise; >1 emulates their harder test set")
    ap.add_argument("--aperiodic", type=float, default=None,
                    help="pin aperiodic_content_level in [0,1]; 0 = degenerate")
    ap.add_argument("--clean", action="store_true",
                    help="no noise — for verifying the coordinate convention only")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    for i in range(args.num):
        style = args.style if args.style != "both" else ("dram" if i % 2 == 0 else "finfet")
        pid = f"{style}_{i:05d}"
        ref, search, meta = make_pair(style, rng, pid,
                                      search_size=args.search_size,
                                      ref_size=args.ref_size,
                                      aperiodic_level=args.aperiodic,
                                      noise_boost=args.noise_boost,
                                      clean=args.clean)
        d = os.path.join(args.out, pid)
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(os.path.join(d, "reference.png"), ref)
        cv2.imwrite(os.path.join(d, "search.png"), search)
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"{pid}  m={meta['magnification_ratio']:.3f}  "
              f"rot={meta['rotation_deg']:+.2f}  tc={meta['true_center_xy']}  "
              f"{meta['ambiguity_class']}")
    print(f"\nwrote {args.num} pairs to {args.out}")


if __name__ == "__main__":
    main()
