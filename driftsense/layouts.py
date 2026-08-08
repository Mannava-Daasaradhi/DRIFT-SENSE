"""DRAM / FinFET analytic layout synthesis (v0 - crude geometry, no SEM physics yet).

Each layout is rendered directly at whatever pixel resolution it's asked
for - reference at high-magnification pitch, search at pitch/m - per
TECH-SPEC.md S4.1: never build one canvas and downsample it, since that
correlates the two captures' noise and defeats the independent-sensor-noise
requirement.

Both renderers supersample internally (TECH-SPEC.md S4.1/S4.2 stage 4,
pulled forward from A2.2): drawing thin, non-integer-pitch lines directly
at final resolution makes each line's cv2 rasterization round to its own
nearest output pixel independently at every scale, and reference vs search
round *differently* since they're quantized on different pixel grids -
with no blur to smooth that over, the two renders of "the same" pattern
stop agreeing pixel-for-pixel and template matching degrades sharply even
though the underlying geometry is correct. Supersample-then-area-average
is the standard anti-aliasing fix and is what a beam PSF will do for real
once A3.1 lands.

Coordinate convention: (x, y), x -> right, y -> down, origin at the
top-left corner of pixel (0, 0). See docs/INTERFACES.md S2.
"""

import cv2
import numpy as np

SUPERSAMPLE = 4


def render_dram(size, pitch_x, pitch_y, line_width, contact_radius,
                 phase_x=0.0, phase_y=0.0, defects=None, block=None):
    """Render a DRAM-style word-line/bit-line grid with a contact at every crossing.

    size: output canvas is size x size.
    pitch_x, pitch_y: bit-line / word-line spacing, in OUTPUT pixels.
    line_width, contact_radius: also in OUTPUT pixels - the caller divides
      all of these by the magnification ratio when rendering the
      search-resolution copy of the same pattern.
    phase_x, phase_y: offset of the first line from pixel 0, in OUTPUT pixels.
    defects: optional list of (x, y, radius, sign) in OUTPUT pixels - bright
      (sign>0) or dark/missing-contact (sign<0) blobs that break perfect
      periodicity. See generate_dataset.py for how these are derived.
    block: optional (x0, y0, x1, y1, value) in OUTPUT pixels - a periphery/
      array-boundary region, see apply_block.

    Returns a float32 array in [0, 1], shape (size, size).
    """
    ss = SUPERSAMPLE
    S = size * ss
    img = np.zeros((S, S), dtype=np.float32)
    px, py = pitch_x * ss, pitch_y * ss
    lw = max(1, round(line_width * ss))

    # Material baseline is deliberately NOT 1.0 - real SEM flat regions sit at
    # a moderate grey, with only edges brightening toward saturation
    # (apply_edge_brightening). Painting lines fully saturated here would
    # leave that stage with no headroom to have any visible effect.
    material = 0.55

    x = (phase_x * ss) % px
    while x < S:
        cv2.line(img, (round(x), 0), (round(x), S - 1), material, lw, cv2.LINE_AA)
        x += px

    y = (phase_y * ss) % py
    while y < S:
        cv2.line(img, (0, round(y)), (S - 1, round(y)), material, lw, cv2.LINE_AA)
        y += py

    r = max(1, round(contact_radius * ss))
    yc = (phase_y * ss) % py
    while yc < S:
        xc = (phase_x * ss) % px
        while xc < S:
            cv2.circle(img, (round(xc), round(yc)), r, material, -1, cv2.LINE_AA)
            xc += px
        yc += py

    apply_block(img, block, scale=ss)
    _apply_defects(img, defects, scale=ss)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return np.clip(img, 0.0, 1.0)


def render_finfet(size, pitch_fin, fin_width, gate_ys, gate_width,
                   phase_x=0.0, defects=None, block=None, epi_width=0.0):
    """Render a FinFET-style dense parallel-fin grid crossed by horizontal gate bars.

    size: output canvas is size x size.
    pitch_fin, fin_width: fin spacing / width, in OUTPUT pixels.
    gate_ys: list of gate-bar centre y positions, in OUTPUT pixels. Gate
      bars are NOT periodic (1-2 of them), so they are what makes the
      y-axis unique even though the x-axis (fin pitch) is periodic.
    gate_width: gate bar thickness, in OUTPUT pixels.
    phase_x: fin phase offset, in OUTPUT pixels.
    epi_width: source/drain epi region extent flanking each gate, in OUTPUT
      pixels - a slightly brighter band, per MEMBER-A-CHECKLIST.md A2.1.
    defects, block: see render_dram.

    Returns a float32 array in [0, 1], shape (size, size).
    """
    ss = SUPERSAMPLE
    S = size * ss
    img = np.zeros((S, S), dtype=np.float32)
    pfin = pitch_fin * ss
    lw = max(1, round(fin_width * ss))
    gw = max(1, round(gate_width * ss))
    ew = round(epi_width * ss)

    # Epi regions are background, drawn first, so fins and gates sit on top.
    if ew > 0:
        for gy in gate_ys:
            gy_s = gy * ss
            for sign in (-1, 1):
                y0 = gy_s + sign * gw / 2.0
                y1 = y0 + sign * ew
                ylo, yhi = sorted((y0, y1))
                if yhi < 0 or ylo > S:
                    continue
                cv2.rectangle(img, (0, round(max(0, ylo))), (S - 1, round(min(S, yhi))),
                              0.35, -1, cv2.LINE_AA)

    x = (phase_x * ss) % pfin
    while x < S:
        cv2.line(img, (round(x), 0), (round(x), S - 1), 0.5, lw, cv2.LINE_AA)
        x += pfin

    for gy in gate_ys:
        gy_s = gy * ss
        if -gw <= gy_s <= S + gw:
            cv2.rectangle(img, (0, round(gy_s - gw / 2)), (S - 1, round(gy_s + gw / 2)),
                          0.6, -1, cv2.LINE_AA)

    apply_block(img, block, scale=ss)
    _apply_defects(img, defects, scale=ss)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return np.clip(img, 0.0, 1.0)


def _apply_defects(img, defects, scale=1):
    if not defects:
        return
    size = img.shape[0]
    for x, y, radius, sign in defects:
        xs, ys, rs = x * scale, y * scale, max(1, round(radius * scale))
        if -rs <= xs <= size + rs and -rs <= ys <= size + rs:
            color = 1.0 if sign > 0 else 0.0
            cv2.circle(img, (round(xs), round(ys)), rs, color, -1, cv2.LINE_AA)


def apply_block(img, block, scale=1):
    """Paint a solid rectangular non-periodic region - array boundary / periphery
    block / dummy fill (TECH-SPEC.md S4.3). block = (x0, y0, x1, y1, value) in
    OUTPUT (unscaled) coordinates; scale matches the internal supersample factor
    used by render_dram/render_finfet. This is the PRIMARY disambiguation
    signal in v0 - see generate_dataset.py's aperiodic_content_level knob.
    """
    if block is None:
        return
    x0, y0, x1, y1, value = block
    size = img.shape[0]
    pt0 = (max(0, round(x0 * scale)), max(0, round(y0 * scale)))
    pt1 = (min(size - 1, round(x1 * scale)), min(size - 1, round(y1 * scale)))
    cv2.rectangle(img, pt0, pt1, float(value), -1, cv2.LINE_AA)
