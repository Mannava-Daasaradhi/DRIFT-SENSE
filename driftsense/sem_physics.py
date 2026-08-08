"""SEM image formation (TECH-SPEC.md S4.2).

All ten forward-model stages are implemented: 3 (SE yield / edge
brightening) and 5 (beam PSF) landed in A2.3; 6-10 (scan distortion,
charging, shading, Poisson shot noise, detector chain) land here in A3.1,
replacing the v0 Gaussian-noise placeholder. Stage 4 (downsample) is
handled inside driftsense/layouts.py's internal supersampling, and stage 1
(geometry)/2 (edge distance) live there and in apply_edge_brightening's
own distance transform respectively - see CITATIONS.md for the physical
justification of each stage.
"""

import cv2
import numpy as np


def apply_edge_brightening(clean, k_edge=0.6, lambda_esc=3.0, threshold=0.4):
    """Stage 3: secondary-electron yield boost near a material edge.

        delta = delta_mat * (1 + k_edge * exp(-d / lambda_esc))

    This is the single mandatory augmentation the brief names explicitly
    ("apply edge-brightening to mimic real SEM behaviour") - see
    CITATIONS.md, SE yield with edge brightening.

    clean: float32 [0, 1], the rendered geometry before blur/noise.
    k_edge: peak fractional brightness boost right at an edge.
    lambda_esc: SE escape-depth length scale, in pixels of THIS image's
      own resolution - decides how fast the boost fades away from an edge.
    threshold: intensity above which a pixel counts as "material" for
      finding edges - the renderer's features are near-binary, so a mid
      threshold cleanly separates structure from background.

    Returns float32, NOT yet clipped to [0, 1] - the boost can push above
    1.0 right at a bright edge, same as real SEM edge saturation; clip
    after calling.
    """
    mask = (clean > threshold).astype(np.uint8)
    if not mask.any() or mask.all():
        return clean  # no edges in this patch (e.g. pure block or pure background)
    dist_in = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_out = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
    d = np.where(mask > 0, dist_in, dist_out).astype(np.float32)
    boost = 1.0 + k_edge * np.exp(-d / lambda_esc)
    return clean * boost


def apply_beam_psf(clean, sigma_beam=1.5, skirt_weight=0.05, skirt_sigma_mult=4.0):
    """Stage 5: finite electron beam spot size blurs the signal before it is
    ever sampled - Gaussian core plus a wide, low-weight Gaussian standing
    in for the Lorentzian beam tail (TECH-SPEC.md S3.6's own suggested
    simplification when a true Lorentzian is fiddly). See CITATIONS.md,
    Beam PSF.

    sigma_beam: core spot size, in pixels of THIS capture's own
    resolution - reference and search each call this with their own
    value, since the same physical spot maps to a different pixel count
    at each magnification (divide by m for the search-resolution call,
    same convention as every other length parameter in this project).
    """
    core = cv2.GaussianBlur(clean, (0, 0), sigma_beam)
    skirt = cv2.GaussianBlur(clean, (0, 0), sigma_beam * skirt_sigma_mult)
    return (1.0 - skirt_weight) * core + skirt_weight * skirt


def apply_scan_distortion(img, rng, warp_amplitude=2.0, jitter_ar=0.6, jitter_std=0.35,
                           vibration_amplitude=0.6, vibration_freq=0.03):
    """Stage 6: motion-stage drift between the beam and the sample, in three
    components that are physically distinct time-scales of the same
    underlying problem this project exists to help recover from:

    - thermal drift: a slow, smooth (low-order polynomial) warp of the
      whole scan field.
    - per-row x-jitter: an AR(1) process, since scan-coil jitter on one row
      is correlated with the row just scanned before it, not independent
      noise.
    - fab vibration: a low-amplitude sinusoid.

    Displaces the SAMPLING GRID via cv2.remap rather than shifting pixel
    values directly, so the result is properly interpolated. See
    CITATIONS.md, Drift and scan distortion.
    """
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ny, nx = (yy - h / 2.0) / (h / 2.0), (xx - w / 2.0) / (w / 2.0)

    a = rng.normal(0.0, 1.0, size=6)
    drift_x = warp_amplitude * (a[0] * nx + a[1] * ny + a[2] * nx * ny)
    drift_y = warp_amplitude * (a[3] * nx + a[4] * ny + a[5] * nx * ny)

    jitter = np.zeros(h, dtype=np.float32)
    innovations = rng.normal(0.0, jitter_std, size=h).astype(np.float32)
    for i in range(1, h):
        jitter[i] = jitter_ar * jitter[i - 1] + innovations[i]

    phase = rng.uniform(0.0, 2.0 * np.pi)
    row_vibration = vibration_amplitude * np.sin(
        2.0 * np.pi * vibration_freq * np.arange(h, dtype=np.float32) + phase)

    map_x = (xx + drift_x + (jitter + row_vibration)[:, None]).astype(np.float32)
    map_y = (yy + drift_y).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REFLECT)


def apply_charging(img, rng, material_threshold=0.35, field_amplitude=0.2, streak_prob=0.12):
    """Stage 7: unremoved surface charge on non-conductive regions modulates
    local secondary-electron yield over a slow spatial scale - a smooth,
    low-frequency multiplicative field - plus, occasionally, a bright
    horizontal streak from a sudden discharge event.

    This crude renderer has no separate material-ID map (TECH-SPEC.md S4.1
    notes this as a known gap), so "dielectric" is approximated as
    background/substrate: pixels below material_threshold, i.e. not part
    of a drawn metal/contact/fin/gate feature. See CITATIONS.md, Charging.
    """
    h, w = img.shape
    dielectric = img < material_threshold
    coarse = rng.normal(1.0, field_amplitude, size=(6, 6)).astype(np.float32)
    field = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    field = cv2.GaussianBlur(field, (0, 0), max(h, w) * 0.06)
    out = img.copy()
    out[dielectric] = out[dielectric] * field[dielectric]
    if rng.uniform() < streak_prob:
        y0 = int(rng.integers(0, h))
        y1 = min(h, y0 + max(1, int(round(rng.uniform(1, 3)))))
        out[y0:y1, :] = np.clip(out[y0:y1, :] + rng.uniform(0.25, 0.55), 0.0, 1.5)
    return out


def apply_shading(img, rng, amplitude=0.2):
    """Stage 8: low-order polynomial illumination field - working-distance
    and detector-geometry vignetting. Not a sample physical mechanism, a
    property of the imaging system, so no material mask is needed here.
    """
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ny, nx = (yy - h / 2.0) / (h / 2.0), (xx - w / 2.0) / (w / 2.0)
    a, b, c = rng.normal(0.0, 1.0, size=3)
    field = 1.0 + amplitude * (a * nx + b * ny + c * (nx ** 2 + ny ** 2 - 2.0 / 3.0))
    return img * np.clip(field, 0.5, 1.5).astype(np.float32)


def apply_shot_noise(signal, rng, dose):
    """Stage 9: Poisson shot noise - the dominant SEM noise source, since SE
    detection is an electron-counting process with variance equal to the
    mean count, NOT an additive Gaussian. Lower dose means lower SNR
    (SNR ~ sqrt(dose), the Rose criterion). See CITATIONS.md, Shot noise.
    """
    counts = rng.poisson(np.clip(signal, 0.0, None) * dose)
    return (counts / dose).astype(np.float32)


def apply_detector(signal, rng, gain=1.0, read_noise_std=0.008):
    """Stage 10: detector gain, additive Gaussian read noise (an electronic
    noise floor, not sample-related), saturation clip, 8-bit quantization.
    See CITATIONS.md, Detector - Timischl et al. 2012's 5-stage detector
    signal chain covers this alongside stage 9's shot noise.
    """
    out = signal * gain + rng.normal(0.0, read_noise_std, size=signal.shape).astype(np.float32)
    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0).round().astype(np.uint8)


def sem_forward(clean, rng, dose=180.0, warp_amplitude=2.0, charging_amplitude=0.2,
                 shading_amplitude=0.2, gain=1.0, read_noise_std=0.008):
    """Chains stages 6-10 with ONE rng stream, in the physical order
    TECH-SPEC.md S4.2 specifies: the beam has already scanned a distorted
    grid before charge can accumulate on it, illumination shading is a
    property of the whole optical path, and noise is a detection process
    applied last, not a property of the surface.

    clean: float32 [0, 1], post edge-brightening and beam PSF.
    rng: np.random.Generator - use an INDEPENDENT stream per capture
      (reference vs search); never share one rng or seed between them.
    dose: higher dose = lower relative shot noise. Reference is captured
      at high dose (low noise); search at lower dose (higher noise) -
      the brief states their test search images are noisier, so this
      asymmetry is intentional, not a bug.

    Returns uint8 array in [0, 255].
    """
    out = apply_scan_distortion(clean, rng, warp_amplitude=warp_amplitude)
    out = apply_charging(out, rng, field_amplitude=charging_amplitude)
    out = apply_shading(out, rng, amplitude=shading_amplitude)
    out = apply_shot_noise(out, rng, dose=dose)
    return apply_detector(out, rng, gain=gain, read_noise_std=read_noise_std)
