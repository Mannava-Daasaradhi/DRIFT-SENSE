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

Progress: 10/10 stages have at least one verified reference. All ten
forward-model stages are implemented (`driftsense/sem_physics.py`). One
optional secondary reference (Rose 1973, stage 9) remains an unverified
candidate - it sits behind institutional auth Member A can't clear - but
every stage's primary claim is now personally DOI-verified.

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
✅ **Verified** — Itoh, K. *VLSI Memory Chip Design*, Springer Series in
Advanced Microelectronics 5, 2001. DOI:
[10.1007/978-3-662-04478-0](https://doi.org/10.1007/978-3-662-04478-0)
(DRAM array pitch/geometry). Opened Aug 8, 2026 - DOI redirects to the
Springer Nature Link record for this exact title/author/year (login-gated
full text, redirect target confirms the record).
✅ **Verified** — Hisamoto, D., Lee, W.C., Kedzierski, J. et al. "FinFET - a
self-aligned double-gate MOSFET scalable to 20 nm." *IEEE Trans. Electron
Devices* 47(12), 2320-2325, 2000. DOI:
[10.1109/16.887014](https://doi.org/10.1109/16.887014). Opened Aug 9, 2026
- DOI redirects to the correct IEEE Xplore record; independently
corroborated via Semantic Scholar and ResearchGate.

Models: the actual repeating unit-cell geometry (word-line/bit-line pitch
for DRAM, fin/gate pitch for FinFET) that the rest of the forward model
images. `driftsense/layouts.py`'s DRAM and FinFET pitch/gate-count ranges
are still generic placeholders, not drawn from either paper's specific
figures - a known simplification, not a citation gap.

### 2. Edge distance / signed distance to material boundary
No separate citation - this is a standard image-processing primitive
(`cv2.distanceTransform`), not a physical mechanism. It exists solely to
feed stage 3's edge-brightening term.

### 3. SE yield with edge brightening ⚠️ mandatory augmentation (brief names it explicitly)
✅ **Verified** — Reimer, L. *Scanning Electron Microscopy: Physics of Image
Formation and Microanalysis*, 2nd ed., Springer Series in Optical Sciences
45, 1998. DOI: [10.1007/978-3-540-38967-5](https://doi.org/10.1007/978-3-540-38967-5)
(SE yield, edge effect). Opened Aug 9, 2026 - DOI redirects to the correct
Springer record; independently corroborated via ResearchGate and the
German National Library catalogue.
✅ **Verified** — Seiler, H. "Secondary electron emission in the scanning
electron microscope." *J. Appl. Phys.* 54(11), R1-R18, 1983. DOI:
[10.1063/1.332840](https://doi.org/10.1063/1.332840). Opened Aug 9, 2026 -
DOI redirects to the correct AIP Publishing record.
✅ **Verified** — Goldstein, J.I., Newbury, D.E., Michael, J.R. et al.
*Scanning Electron Microscopy and X-Ray Microanalysis*, 4th ed., Springer,
2018. DOI: [10.1007/978-1-4939-6676-9](https://doi.org/10.1007/978-1-4939-6676-9)
(edge/topographic contrast). Opened Aug 8, 2026 - DOI redirects to the
Springer Nature Link record for this exact title/author/year; independently
corroborated via the Semantic Scholar and Oxford Academic book-review
listings found alongside it.

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
✅ Reimer 1998 (as above, stage 3) - beam broadening and interaction volume.
✅ **Verified** — Joy, D.C. *Monte Carlo Modeling for Electron Microscopy
and Microanalysis*, Oxford Series in Optical and Imaging Sciences, Oxford
University Press, 1995. DOI:
[10.1093/oso/9780195088748.001.0001](https://doi.org/10.1093/oso/9780195088748.001.0001)
(beam-sample interaction, scattering). Opened Aug 8, 2026 - DOI redirects
to the correct Oxford Academic book record.

Models: the electron beam is not an infinitesimal point - finite spot size
(Gaussian core) plus long-range scattered electrons (Lorentzian skirt/tail)
blur the true surface signal before it is ever sampled. Implemented in
`apply_beam_psf` (A2.3): Gaussian core plus a wide, low-weight second
Gaussian standing in for the Lorentzian tail.

### 6. Scan distortion — thermal drift, per-row jitter, vibration
✅ **Verified** — Sutton, M.A., Li, N., Joy, D.C. et al. "Scanning Electron
Microscopy for Quantitative Small and Large Deformation Measurements Part
I: SEM Imaging at Magnifications from 200 to 10,000." *Experimental
Mechanics* 47, 775-787, 2007. DOI:
[10.1007/s11340-007-9042-z](https://doi.org/10.1007/s11340-007-9042-z).
Companion: Sutton, M.A., Li, N., Garcia, D. et al. Part II, *Experimental
Mechanics* 47, 789-804, 2007. DOI:
[10.1007/s11340-007-9041-0](https://doi.org/10.1007/s11340-007-9041-0).
Opened Aug 9, 2026 - both DOIs redirect to the correct Springer records;
independently corroborated via ResearchGate and HAL listings.

Models: this is literally the phenomenon DRIFT-SENSE exists to help
recover from - motion-stage thermal drift (slow polynomial warp), scan-coil
jitter (AR(1)-correlated row shift, since jitter on one row is correlated
with the row just scanned, not independent noise), and fab vibration
(sinusoidal component) all displace the sampling grid between the
reference and search captures. Implemented in `apply_scan_distortion`
(A3.1) via `cv2.remap` on a displaced sampling grid, with independent RNG
per capture.

### 7. Charging — low-frequency field on dielectric regions
✅ **Verified** — Cazaux, J. "Charging in scanning electron microscopy
'from inside and outside'." *Scanning* 26(4), 181-203, 2004. DOI:
[10.1002/sca.4950260406](https://doi.org/10.1002/sca.4950260406). Opened
Aug 9, 2026 - DOI redirects to the Wiley Online Library record for this
exact title/author/year; independently corroborated via ResearchGate.

Models: unremoved surface charge on dielectric regions modulates local
secondary-electron yield over a slow spatial scale, producing the smooth
brightness gradients (and occasional bright streaks from a sudden
discharge event) real SEM images of oxide/dielectric regions show.
Implemented in `apply_charging` (A3.1); this crude renderer has no
separate material-ID map, so "dielectric" is approximated as
background/substrate (below a fixed intensity threshold) rather than a
true material classification - a known simplification, not yet a full
metal/dielectric/contact map (see stage 1).

### 8. Shading — low-order polynomial illumination field
No separate citation - standard imaging-system vignetting/shading model
from working-distance and detector-geometry variation, not sample physics.
Implemented in `apply_shading` (A3.1).

### 9. Shot noise — Poisson, not Gaussian
✅ **Verified** — Timischl, F., Date, M., Nemoto, S. "A statistical model of
signal-noise in scanning electron microscopy." *Scanning* 34(3), 137-144,
2012. DOI: [10.1002/sca.20282](https://doi.org/10.1002/sca.20282). Opened
Aug 7, 2026 - DOI redirects to the Wiley Online Library record for this
exact title/author/year; independently corroborated via PubMed (PMID
21898458) and Semantic Scholar.
⏳ Rose, A. *Vision: Human and Electronic*, Plenum, 1973 (SNR ∝ √N, the
Rose criterion). Still unverified: found the Chapter 1 PDF hosted at
UMich, but it sits behind University of Michigan's Shibboleth
institutional login, which Member A cannot clear. Kept as a secondary/
optional reference, not the stage's load-bearing citation - Timischl 2012
(below) is.

Models: SE detection is an electron-counting process, so its dominant
noise source is shot noise - variance equal to the mean count - not an
additive Gaussian. Implemented in `apply_shot_noise` (A3.1):
`rng.poisson(dose * signal) / dose`, replacing the v0 Gaussian placeholder
entirely.

### 10. Detector — gain, read noise, saturation, quantization
✅ Timischl et al. 2012 (as above, stage 9) - covers the full detector
signal chain, not just the shot-noise stage.
✅ **Verified** — Scharf, D. "Secondary Electron Detectors, Image Quality &
Contrast." *Microscopy and Microanalysis* 4(S2), 256-257, 1998. DOI:
[10.1017/S1431927600021401](https://doi.org/10.1017/S1431927600021401).
Opened Aug 8, 2026 - DOI redirects to the correct Oxford Academic/MAM
record (confirmed title, author, volume, year on the landing page). Note:
this is a 2-page conference abstract, not a full paper - listed as
secondary support for the detector-chain concept (scintillator/PMT gain
stage), not as the quantitative model, which remains Timischl 2012.

Models: after the electron count is converted to a voltage, the detector
adds its own Gaussian read noise (an electronic noise floor, not
sample-related), has a finite dynamic range (saturation clipping), and the
final digitization is 8-bit quantization - three distinct,
independently-citable steps this project treats as one stage for brevity,
matching the Timischl et al. 5-stage detector cascade model. Implemented
in `apply_detector` (A3.1).

---

## Known trap — do not cite for the periodic/aperiodic decomposition

TECH-SPEC.md S3.5 flags this explicitly and it belongs here as a standing
reminder, even though the decomposition itself is Member B's code
(`driftsense/periodic.py`): **Moisan, L. (2011), "Periodic plus smooth
image decomposition,"** is a real, well-known paper, but its "periodic"
means *periodic boundary extension* (a trick for FFT edge artifacts), not
*lattice-periodic content* (repeating device structure). It is the wrong
citation for this project's core disambiguation idea. The correct citation,
selected and verified:

✅ **Verified** — Zaefferer, S. "New developments of computer-aided
crystallographic analysis in transmission electron microscopy." *J. Appl.
Cryst.* 33, 10-25, 2000. DOI:
[10.1107/S0021889899010894](https://doi.org/10.1107/S0021889899010894).
Opened Aug 8, 2026 - DOI redirects to the correct IUCr *Journal of Applied
Crystallography* record; independently corroborated by the Wiley Online
Library and ResearchGate listings for the same title/author/year/volume.

Models: this is the actual family of technique DRIFT-SENSE's own
`driftsense/spectral.py` reimplements for a different purpose - indexing a
diffraction/FFT pattern's peaks against a reciprocal lattice to recover the
real-space lattice's scale and orientation, the same peak-pair voting this
project's scale/rotation estimator performs on the search image's spatial
frequency spectrum. A real crystallographic auto-indexing method, not a
coincidentally-similar-sounding trick like Moisan's periodic-boundary
decomposition above.

---

## Open items before this file is frozen (A5.3, due Aug 11 EOD)

- [x] ~~Personally verify Itoh 2001 (DRAM geometry, stage 1)~~ - done, DOI
      10.1007/978-3-662-04478-0.
- [x] ~~Personally verify Goldstein 2018 (stage 3)~~ - done, DOI
      10.1007/978-1-4939-6676-9.
- [x] ~~Personally verify Joy 1995 (stage 5)~~ - done, DOI
      10.1093/oso/9780195088748.001.0001.
- [ ] Rose 1973 (stage 9, secondary/optional reference) - found the book
      and a chapter PDF but it sits behind University of Michigan's
      Shibboleth institutional login, which I can't clear. Left as ⏳
      rather than falsely marked done. Not load-bearing: Timischl 2012 is
      stage 9's verified primary reference.
- [x] ~~Verify a stage 10 detector-chain reference~~ - done, Scharf 1998
      (secondary support; Timischl 2012 remains the primary/quantitative
      reference for the stage).
- [x] ~~Find and verify a specific Cazaux charging paper~~ - done, stage 7.
- [x] ~~Select and verify a reciprocal-lattice/crystallography citation for
      the periodic/aperiodic decomposition (Moisan 2011 trap, above)~~ -
      done: Zaefferer 2000, DOI 10.1107/S0021889899010894.
- [ ] Cross-check this file against Slide 4 and Slide 9 content once drafted
      - they must agree exactly (DATASET-INSTRUCT.png, TECH-SPEC.md S5).
