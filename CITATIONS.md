# DRIFT-SENSE — Citations for the SEM Forward Model

Every augmentation choice, noise model, and structural parameter in
`driftsense/sem_physics.py` and `driftsense/layouts.py` is justified here
with 2-3 references and a sentence naming the physical mechanism it models
- not just that the mechanism exists (DATASET-INSTRUCT.png citation
requirement; TECH-SPEC.md S6).

**Verification rule (MEMBER-A-CHECKLIST.md A1.3):** a reference is only
listed as ✅ Verified if Member A has personally opened it and confirmed the
DOI resolves to the real paper. Everything else is a ⏳ candidate from
TECH-SPEC.md S6's starting list and must not be treated as approved or
quoted in the slide deck until verified. Status legend:

- ✅ Verified — opened, DOI confirmed resolving to the correct paper.
- ⏳ Candidate — from TECH-SPEC.md S6's starting list, not yet personally opened.

Progress: 2/10 stages have a verified reference as of Aug 7 (Day 1). Full
verification of every stage is due by Gate 2 (A5.3, Aug 11 EOD per PLAN.md)
- this file is a **living document**, updated across A2.x-A5.x as the
physics model itself moves from v0 (crude, current) to full (TECH-SPEC S4).

---

## General prior art — synthetic SEM image generation

✅ **Verified** — Cizmar, P., Vladár, A.E., Ming, B., Postek, M.T.
"Simulated SEM images for resolution measurement." *Scanning* 30(5),
381-391, 2008. DOI: [10.1002/sca.20120](https://doi.org/10.1002/sca.20120)
(NIST ARTIMAGEN project). Opened Aug 7, 2026 - DOI redirects to the Wiley
Online Library record for this exact title/author/year; independently
corroborated via PubMed (PMID 18613028) and ResearchGate listings.

Directly relevant prior art: NIST's ARTIMAGEN is, like this project's
generator, a from-scratch synthetic SEM image renderer used to produce
ground-truth-known test images for algorithm evaluation rather than relying
on real (unlabeled) micrographs - the same justification for why
DRIFT-SENSE generates rather than sources its dataset.

---

## Stage-by-stage justification (TECH-SPEC.md S4.2)

### 1. Geometry — analytic layout at supersample, material ID map
⏳ Itoh, K. *VLSI Memory Chip Design*, Springer, 2001 (DRAM array pitch/geometry).
⏳ Hisamoto, D. et al. "FinFET - a self-aligned double-gate MOSFET scalable
to 20 nm." *IEEE Trans. Electron Devices* 47(12), 2320-2325, 2000 (FinFET fin/gate geometry).

Models: the actual repeating unit-cell geometry (word-line/bit-line pitch
for DRAM, fin/gate pitch for FinFET) that the rest of the forward model
images. Not yet personally verified - queued for A2.1/A2.2 when the real
(non-crude) layout parameters are drawn from literature pitch figures.

### 2. Edge distance / signed distance to material boundary
No separate citation - this is a standard image-processing primitive
(`cv2.distanceTransform`), not a physical mechanism. It exists solely to
feed stage 3's edge-brightening term.

### 3. SE yield with edge brightening ⚠️ mandatory augmentation (brief names it explicitly)
⏳ Reimer, L. *Scanning Electron Microscopy: Physics of Image Formation and
Microanalysis*, 2nd ed., Springer, 1998 (SE yield, edge effect).
⏳ Goldstein, J. et al. *Scanning Electron Microscopy and X-Ray
Microanalysis*, 4th ed., Springer, 2018 (edge/topographic contrast).
⏳ Seiler, H. "Secondary electron emission in the scanning electron
microscope." *J. Appl. Phys.* 54(11), R1-R18, 1983.

Models: secondary-electron yield rises near a topographic edge because more
of the interaction volume is within escape depth of a surface, producing
the bright rims real SEM images show at every feature edge - this is why
`delta = delta_mat * (1 + k_edge * exp(-d/lambda_esc))` uses an exponential
falloff with distance-to-edge `d`, not a uniform brightness boost.

### 4. Downsample — area-average the supersample to pixel resolution
No separate citation - standard anti-aliasing (box filter / area averaging),
not a physical mechanism of the sample. Implemented in v0 as 4x
supersample + `cv2.resize(..., INTER_AREA)` (`driftsense/layouts.py`),
pulled forward from its originally-planned A2.2 slot because the crude
geometry renderer needs it to avoid rasterization-quantization artifacts
between independently-rendered reference/search resolutions (see
`tools/check_gt.py`'s module docstring for the diagnosis).

### 5. Beam PSF — Gaussian core + Lorentzian skirt
⏳ Reimer 1998 (as above) - beam broadening and interaction volume.
⏳ Joy, D.C. *Monte Carlo Modeling for Electron Microscopy and
Microanalysis*, Oxford, 1995 (beam-sample interaction, scattering).

Models: the electron beam is not an infinitesimal point - finite spot size
(Gaussian core) plus long-range scattered electrons (Lorentzian skirt/tail)
blur the true surface signal before it is ever sampled. Not yet implemented
- v0 has no PSF at all (see the rasterization-noise note above); lands in A3.1.

### 6. Scan distortion — thermal drift, per-row jitter, vibration
⏳ Sutton, M.A. et al. "Scanning Electron Microscopy for Quantitative Small
and Large Deformation Measurements, Parts I & II." *Experimental Mechanics*
47, 2007 (SEM drift and spatial distortion, and its correction).

Models: this is literally the phenomenon DRIFT-SENSE exists to help
recover from - motion-stage thermal drift (slow polynomial warp), scan-coil
jitter (AR(1)-correlated row shift), and fab vibration (sinusoidal
component) all displace the sampling grid between the reference and search
captures. Not yet implemented in v0 (`rotation_deg` is fixed at 0.0; no
warp field yet) - lands in A3.1/A3.2.

### 7. Charging — low-frequency field on dielectric regions
⏳ Cazaux, J. — charge compensation of insulating samples in SEM (specific
paper TBD; candidate author only, not yet a verified single citation - see
open item below).

Models: unremoved surface charge on dielectric regions modulates local
secondary-electron yield over a slow spatial scale, producing the smooth
brightness gradients (and occasional bright streaks) real SEM images of
oxide/dielectric regions show. Not implemented in v0.

### 8. Shading — low-order polynomial illumination field
No separate citation - standard imaging-system vignetting/shading model
from working-distance and detector-geometry variation, not sample physics.
Not implemented in v0.

### 9. Shot noise — Poisson, not Gaussian
✅ **Verified** — Timischl, F., Date, M., Nemoto, S. "A statistical model of
signal-noise in scanning electron microscopy." *Scanning* 34(3), 137-144,
2012. DOI: [10.1002/sca.20282](https://doi.org/10.1002/sca.20282). Opened
Aug 7, 2026 - DOI redirects to the Wiley Online Library record for this
exact title/author/year; independently corroborated via PubMed (PMID
21898458) and Semantic Scholar.
⏳ Rose, A. *Vision: Human and Electronic*, Plenum, 1973 (SNR ∝ √N, the
Rose criterion).

Models: SE detection is a electron-counting process, so its dominant noise
source is shot noise - variance equal to the mean count - not an additive
Gaussian. `driftsense/sem_physics.py`'s v0 stub uses Gaussian noise
ONLY as a temporary placeholder (explicitly flagged `v0_placeholder: True`
in every pair's `meta.json`); the Poisson model (`rng.poisson(dose*delta)/dose`)
replaces it in A3.1 without changing the module's call signature.

### 10. Detector — gain, read noise, saturation, quantization
⏳ Timischl et al. 2012 (as above) - covers the full detector signal chain,
not just the shot-noise stage.

Models: after the electron count is converted to a voltage, the detector
adds its own Gaussian read noise, has a finite dynamic range (saturation
clipping), and the final digitization is 8-bit quantization - three
distinct, independently-citable steps this project treats as one stage for
brevity, matching the Timischl et al. 5-stage detector cascade model.

---

## Known trap — do not cite for the periodic/aperiodic decomposition

TECH-SPEC.md S3.5 flags this explicitly and it belongs here as a standing
reminder, even though the decomposition itself is Member B's code
(`driftsense/periodic.py`): **Moisan, L. (2011), "Periodic plus smooth
image decomposition,"** is a real, well-known paper, but its "periodic"
means *periodic boundary extension* (a trick for FFT edge artifacts), not
*lattice-periodic content* (repeating device structure). It is the wrong
citation for this project's core disambiguation idea. The correct citation
family is standard reciprocal-lattice / Fourier crystallography treatment -
still ⏳ open, to be selected and verified before Gate 2.

---

## Open items before this file is frozen (A5.3, due Aug 11 EOD)

- [ ] Personally verify all remaining ⏳ candidates above, or replace them.
- [ ] Find and verify a specific Cazaux charging paper (currently author-only).
- [ ] Select and verify a reciprocal-lattice/crystallography citation for
      the periodic/aperiodic decomposition (Moisan 2011 trap, above).
- [ ] Cross-check this file against Slide 4 and Slide 9 content once drafted
      - they must agree exactly (DATASET-INSTRUCT.png, TECH-SPEC.md S5).
