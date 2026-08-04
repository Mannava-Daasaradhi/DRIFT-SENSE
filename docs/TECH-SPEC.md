# DRIFT-SENSE — Technical Specification

Companion to [`PLAN.md`](./PLAN.md). This is the implementation contract.

---

## 1. Repository layout

```
DRIFT-SENSE/
├── README.md                  # C — setup, usage, results. Written last, by a non-author.
├── requirements.txt           # C — pip freeze from the clean venv
├── CITATIONS.md               # A — every augmentation choice, 2-3 verified refs each
├── generate_dataset.py        # A — standalone CLI, mandatory repo item #2
├── localize.py                # B — standalone CLI, mandatory repo item #3, THE critical file
├── train.py                   # C — trains the re-ranker, mandatory repo item #5
├── evaluate.py                # C — 30+ pair harness, metrics, figures
├── weights/reranker.pt        # C — mandatory repo item #4 (keep under 5 MB)
├── driftsense/
│   ├── layouts.py             # A — DRAM / FinFET geometry synthesis
│   ├── sem_physics.py         # A — SEM image formation forward model
│   ├── spectral.py            # B — FFT lattice estimation, Fourier-Mellin cross-check
│   ├── matching.py            # B — correlation, peak list, NMS, sub-pixel
│   ├── periodic.py            # B — periodic / aperiodic decomposition
│   ├── decide.py              # B — tie test, centre rule, confidence calibration
│   ├── rerank.py              # C — Siamese re-ranker (optional at inference)
│   └── viz.py                 # C — figures
├── data/                      # generated; gitignore all but one sample pair
├── figures/
└── docs/                      # PLAN.md, TECH-SPEC.md, slides_content.md
```

---

## 2. Frozen interfaces

### 2.1 Pair format (A produces, B and C consume)

```
data/<split>/<pair_id>/
    reference.png     # high magnification capture
    search.png        # 1000x1000, ~10x lower magnification
    meta.json
```

```jsonc
{
  "pair_id": "dram_00017",
  "style": "dram",                          // "dram" | "finfet"
  "true_center_xy": [412.37, 688.02],       // sub-pixel, in SEARCH image coords
  "magnification_ratio": 9.83,              // NOT always 10
  "rotation_deg": 1.42,
  "lattice_period_search_px": [7.4, 9.1],
  "alias_positions": [[404.9, 688.0], ...], // lattice-equivalent sites
  "ambiguity_class": "unique",              // unique | weakly_ambiguous | degenerate
  "aperiodic_content": ["array_boundary", "defect"],
  "sem_params": { "...": "..." },
  "seeds": { "reference": 12345, "search": 67890 }
}
```

`alias_positions` and `ambiguity_class` are what let us report accuracy honestly
instead of pretending the degenerate cases are failures of the algorithm.

### 2.2 Localizer API (B produces, C consumes)

```python
def localize(reference_path: str, search_path: str,
             use_reranker: bool = True) -> dict:
    """Returns the dict described in PLAN.md §3. Never raises."""
```

### 2.3 CLI contract for `localize.py`

Both invocation styles must work — we do not know which they will use:

```bash
python localize.py reference.png search.png
python localize.py --ref reference.png --search search.png
python localize.py --ref r.png --search s.png --json      # full diagnostics
```

Default stdout is exactly one line, nothing else:

```
412.4,688.0
```

Rationale: an automated grader most likely parses stdout. Anything printed
alongside the coordinate risks breaking their parser. Route all logging to stderr.

---

## 3. Localization algorithm — Spectral Lattice Disambiguation

### 3.0 Preprocessing

1. Load as float32 grayscale. Convert colour with luminance weights; accept any dtype.
2. Robust normalization to the 1st–99th percentile range — invariant to detector
   gain and offset differences between the two captures.
3. Split into two bands:
   - `hp` = I − GaussianBlur(I, σ_large): structure, illumination-invariant. **Matching runs on this.**
   - `lf` = the removed low-frequency field: shading and defocus envelope. Aperiodic
     and therefore weakly informative for disambiguation — kept for §3.5.
4. Also compute gradient magnitude `|∇I|`. SEM contrast is dominated by edge
   brightening, so the edge map is more stable across the two captures than raw
   intensity. Score on both channels and fuse.

### 3.1 Spectral lattice estimation — closed-form scale and rotation

This is the Slide-5 innovation. Do not brute-force the scale.

1. Apply a 2D Hann window to reference and search (suppresses the spectral cross
   artifact from non-periodic image boundaries).
2. `M = log(1 + |FFT2(I)|)`, `fftshift`, mask out a small disc around DC.
3. Detect local maxima; keep conjugate-symmetric pairs. For DRAM expect two strong
   orthogonal fundamentals (word-line and bit-line pitch) plus harmonics. For FinFET
   expect one dominant fin-pitch fundamental plus a weaker gate frequency.
4. Fit a reciprocal-lattice basis `(g1, g2)`: take the two shortest non-collinear
   peak vectors that generate the remaining peaks within tolerance. Refine by
   least-squares over all assigned harmonics.

**Magnification**, in closed form. A capture at 10× lower magnification has features
10× smaller in pixels, so its spatial period is 10× smaller and its frequency 10×
higher:

```
period_S = period_R / m      =>      m = |g_S| / |g_R|
```

**Rotation:** `θ = angle(g_S basis) − angle(g_R basis)`, resolved modulo the lattice
symmetry group — 90° for a square DRAM grid, 180° for FinFET line arrays. Symmetry
does not resolve itself; emit the small set of symmetry-equivalent hypotheses
(typically 2 or 4) and carry all of them into §3.3.

**Independent cross-check:** run log-polar Fourier–Mellin registration (Reddy &
Chatterji 1996) on the two magnitude spectra. It returns scale and rotation by a
completely different route. Two independent estimators agreeing is a strong
confidence signal; disagreement raises a flag and widens the hypothesis set.

**Fallback:** if lattice peaks are weak or absent, fall back to a coarse-to-fine
scale sweep over `m ∈ [7, 14]`. Slower, always terminates. Never crash.

> Why this is genuinely better: the FFT integrates over the entire image, so the
> scale estimate averages out noise across a million pixels. A local patch-based
> scale search sees only ~10⁴ noisy pixels. Under the high-noise conditions the test
> set is explicitly designed to have, this difference is large.

### 3.2 Template construction

Resize the reference by `1/m` with area averaging — this matches how a real
lower-magnification capture integrates signal over a larger pixel footprint, and how
their generator almost certainly downsamples. Benchmark `INTER_AREA` against
Lanczos on our validation set and keep the winner. Rotate by `−θ`. Apply the §3.0
preprocessing to the result so template and search are treated identically.

### 3.3 Dense correlation → full peak list

**Do not take `argmax`.** That single decision is what separates this submission
from the median one.

1. ZNCC over the search image for each (scale, rotation) hypothesis —
   `cv2.matchTemplate(..., TM_CCOEFF_NORMED)`, ~50–100 ms at 1000×1000 with a
   100×100 template.
2. Fuse the intensity-channel and gradient-channel score maps.
3. Non-maximum suppression with radius ≈ half the template size.
4. Keep the top `K ≈ 50` peaks with their scores.
5. Sub-pixel refinement by 2D quadratic fit on the 3×3 neighbourhood:

```
dx = 0.5·(C[y,x−1] − C[y,x+1]) / (C[y,x−1] − 2·C[y,x] + C[y,x+1])
```

### 3.4 Lattice-consistency structuring

Fit a lattice to the recovered peak positions. If the top peaks lie on a lattice
matching the period measured in §3.1, we are provably in the degenerate periodic
regime — and we now know it *analytically* rather than guessing from a threshold.

This is the principled basis for the ambiguity report, and it is what makes the
confidence output defensible rather than a tuned magic number.

### 3.5 Periodic / aperiodic decomposition — the core contribution

The periodic component is, by definition, the part that cannot disambiguate. So
remove it and match on what remains.

**v1 — Fourier synthesis (implement first).** Build a frequency-domain mask keeping
only reciprocal-lattice points and their harmonics (small disc around each).
Inverse-transform → `S_periodic`. Then `S_aperiodic = S − S_periodic`. Same
procedure on the template.

**v2 — unit-cell folding (if time allows).** Warp into lattice coordinates, average
over all cells to get the mean unit cell, re-tile. More robust when the lattice
drifts slowly across the field.

The residual isolates exactly the information capable of resolving position: array
boundaries, periphery blocks, dummy fill, defects, and the illumination envelope
from §3.0. Correlate residuals for each candidate from §3.3, weighted by a residual-
energy mask so that flat regions do not dominate the score.

> **The elegant part:** when the layout is perfectly periodic, the residual is pure
> noise and this stage contributes nothing — which is *correct*, because that case is
> genuinely ambiguous. Residual energy therefore doubles as a principled ambiguity
> measure. The method degrades exactly where the problem becomes unsolvable.

⚠️ **Citation trap for Member A:** Moisan (2011), "Periodic plus smooth image
decomposition," is a well-known paper but its "periodic" means *periodic boundary
extension*, not *lattice-periodic content*. It is a different decomposition. Do not
cite it for this. Cite standard reciprocal-lattice / Fourier crystallography
treatment instead.

### 3.6 Siamese re-ranker (optional at inference)

- **Input:** template patch and candidate patch, both resized to 64×64.
- **Architecture:** shared CNN encoder, 4–5 conv blocks, ~200 k parameters, feature
  concatenation, MLP head → logit. Keep the checkpoint under 5 MB.
- **Training data — this is the important design decision:** positives are the true
  location; **hard negatives are the lattice-alias positions**. They look nearly
  identical to the positive, which is precisely what forces the network to learn the
  aperiodic cues instead of the periodic texture. Easy negatives are random offsets.
- **Train only on pairs with genuine aperiodic content.** On degenerate pairs there
  is no learnable signal, and training on them teaches the network to hallucinate
  confidence. Degenerate cases are handled by the centre rule, not by the network.
  State this explicitly on Slide 5 — it is a mature design choice.
- **Strictly optional at runtime.** `localize.py` must produce a full answer with
  torch absent (PLAN.md Rule 1). The re-ranker adjusts candidate ordering; it never
  gates the pipeline.

### 3.7 Decision, tie-break, confidence

1. Fuse scores: ZNCC + gradient channel + aperiodic residual + re-ranker logit.
2. **Tie test.** Calibrate δ by bootstrapping over noise realizations on the
   validation set — the score spread induced by noise alone. Tie set =
   `{c : score(c) ≥ score_best − δ}`.
3. If `|tie set| > 1`, **return the member closest to the search-image centre.**
   Literal compliance with the brief. Record `decision = "tie_broken_by_center"`.
4. **Confidence** from a logistic regression over four features — normalized peak
   margin, aperiodic residual energy ratio, lattice-consistency of the peak set, and
   agreement between the two independent scale estimators. Fit on validation data.
   Four interpretable features, one explainable model, genuinely calibrated. Verify
   with a reliability diagram.
5. **Periodic Ambiguity Index** = `score_2 / score_1` over peaks separated by more
   than the template radius. Reported alongside every prediction.
6. Optional final sub-pixel polish: local ECC refinement (`cv2.findTransformECC`)
   initialized from the estimated scale and rotation.

**Timing budget:** target under 1 s per pair on CPU. Report the measured figure on
Slides 6 and 7.

---

## 4. Dataset generator — SEM forward model

### 4.1 Render at each magnification separately

Do **not** build one giant 10 000 × 10 000 canvas and downsample it. Because the
layouts are analytic (lines, bars, contacts at known pitches), render the reference
directly at high magnification and the search image directly at 1000 × 1000 with the
pitch expressed in search pixels. Faster, no memory pressure, exact, and — critically
— it lets the SEM model run with the correct per-magnification dose, pixel size and
PSF. Supersample 4× and area-downsample for anti-aliasing.

> **Slide-worthy correctness point.** If you generate one noisy high-resolution image
> and downsample it by 10×, the noise averages down by a factor of 10 *and* the search
> image's noise becomes correlated with the reference's. That violates the brief's
> explicit "independent sensor noise" requirement and produces an unrealistically
> clean search image. Many teams will do exactly this. Say so on Slide 4.

### 4.2 The forward model, in order

Order matters physically: the beam blurs, the scan warps the sampling grid, and
*then* electrons are counted. Noise is applied last because it is a detection
process, not a property of the surface.

| # | Stage | Model |
|---|-------|-------|
| 1 | Geometry | Analytic layout at 4× supersample → material ID map |
| 2 | Edge distance | Signed distance to nearest material edge, `d(x)`; local surface tilt |
| 3 | SE yield | `δ = δ_mat · (1 + k_edge·exp(−d/λ)) · sec(θ_tilt)^n` — edge brightening plus topographic contrast |
| 4 | Downsample | Area-average the supersample → true signal at pixel resolution |
| 5 | Beam PSF | Gaussian core (σ_beam) + small Lorentzian skirt for the beam tail |
| 6 | Scan distortion | Slow thermal drift (low-order polynomial warp) + per-row jitter (AR(1) correlated x-shift) + sinusoidal vibration |
| 7 | Charging | Low-frequency multiplicative field on dielectric regions; occasional bright streaks |
| 8 | Shading | Low-order polynomial illumination field (working distance / detector geometry) |
| 9 | Shot noise | `N ~ Poisson(dose · δ)` — the dominant SEM noise source, **not** Gaussian |
| 10 | Detector | Gain, Gaussian read noise, saturation clipping, 8-bit quantization |

**Two captures, two seeds.** Run stages 5–10 twice with independent RNG streams:
once for the reference (high magnification, high dose, low noise) and once for the
search image (low magnification, lower dose per unit area, **higher noise** — the
brief says the test search images are noisier, so design for it).

### 4.3 Layout parameters to randomize

**DRAM:** word-line pitch and width, bit-line pitch and width, contact radius,
contact-present probability (defects), sub-array break positions.

**FinFET:** fin pitch and width, gate bar count (1–2), gate width and position,
source/drain epi regions, fin-cut regions.

**Both — and this is the important one: `aperiodic_content_level`.** A knob
controlling array boundaries, periphery blocks, dummy fill and defect density. At
zero, the pair is *genuinely degenerate* and only the centre rule can answer it. At
high values it is uniquely solvable. We need the full range: it is our training
signal, our honest evaluation axis, and our reproduction of the pathological case
their test set is guaranteed to contain.

**Inter-capture perturbations:** rotation ±3°, magnification ratio `m ∈ [9.0, 11.0]`,
differential defocus, differential drift warp.

### 4.4 CLI

```bash
python generate_dataset.py --style dram --num 30 --out data/eval --seed 42
python generate_dataset.py --style both --num 500 --out data/train --seed 1
```

`--style` accepts `dram | finfet | both`, per repository requirement #2.

---

## 5. Evaluation

### 5.1 Metrics

- Mean / median Euclidean error in pixels
- Accuracy within {1, 2, 5, 10} px tolerance
- **Broken out by `ambiguity_class`** — reporting one blended number hides the
  entire scientific story
- **Alias-hit rate:** predictions landing on a lattice-equivalent site rather than
  the true one. On degenerate pairs this is the expected outcome, not a bug.
- Wall-clock per pair (Slides 6 and 7)
- **Confidence calibration** — reliability diagram, expected calibration error

### 5.2 Mandatory baseline

Plain `cv2.matchTemplate` at a fixed assumed 10× scale, `argmax`. Slides 3 and 5
both ask why we beat template matching; a chart answers it better than a claim.
Expect the baseline to fail on rotation, on non-exact scale, and on high-periodicity
cases — exactly the three axes we handle.

### 5.3 Out-of-distribution validation

Hold out a generator configuration never used for training or tuning: unseen
pitches, higher noise, larger rotation. Because the brief states their test set is
noisier than ours, tuning to our own noise level is the most likely way to lose.

---

## 6. Citation candidates

**Member A: verify every one of these — DOI and a working link — before use. This
is a starting list, not an approved one.** A fabricated reference in front of
Applied Materials engineers costs more than a missing one.

**SEM physics and image formation**
- Reimer, L. *Scanning Electron Microscopy: Physics of Image Formation and Microanalysis*, 2nd ed., Springer, 1998 — SE yield, edge effect
- Goldstein, J. et al. *Scanning Electron Microscopy and X-Ray Microanalysis*, 4th ed., Springer, 2018 — edge and topographic contrast
- Seiler, H. "Secondary electron emission in the scanning electron microscope." *J. Appl. Phys.* 54(11), R1–R18, 1983
- Joy, D.C. *Monte Carlo Modeling for Electron Microscopy and Microanalysis*, Oxford, 1995

**Synthetic SEM image generation** — directly relevant prior art, cite it
- Cizmar, P., Vladár, A.E., Ming, B., Postek, M.T. "Simulated SEM images for resolution measurement." *Scanning* 30(5), 381–391, 2008 (NIST ARTIMAGEN)

**Noise statistics**
- Timischl, F., Date, M., Nemoto, S. "A statistical model of signal–noise in scanning electron microscopy." *Scanning* 34(3), 137–144, 2012
- Rose, A. *Vision: Human and Electronic*, Plenum, 1973 — SNR ∝ √N, the Rose criterion

**Drift and scan distortion**
- Sutton, M.A. et al. "Scanning Electron Microscopy for Quantitative Small and Large Deformation Measurements, Parts I & II." *Experimental Mechanics* 47, 2007 — SEM drift and spatial distortion, and its correction

**Charging**
- Cazaux, J. — work on charge compensation of insulating samples in SEM

**Device structure**
- Hisamoto, D. et al. "FinFET — a self-aligned double-gate MOSFET scalable to 20 nm." *IEEE Trans. Electron Devices* 47(12), 2320–2325, 2000
- Itoh, K. *VLSI Memory Chip Design*, Springer, 2001 — DRAM array architecture
- IRDS (International Roadmap for Devices and Systems) — contemporary pitch figures

**Computer vision**
- Reddy, B.S. & Chatterji, B.N. "An FFT-based technique for translation, rotation and scale-invariant image registration." *IEEE Trans. Image Processing* 5(8), 1266–1271, 1996 — Fourier–Mellin
- Lewis, J.P. "Fast Normalized Cross-Correlation." *Vision Interface*, 1995
- Lowe, D.G. "Distinctive Image Features from Scale-Invariant Keypoints." *IJCV* 60(2), 2004 — useful for arguing why keypoint matching fails on periodic texture
- Bertinetto, L. et al. "Fully-Convolutional Siamese Networks for Object Tracking." *ECCVW*, 2016 — basis for the re-ranker
- Sun, J. et al. "LoFTR: Detector-Free Local Feature Matching with Transformers." *CVPR*, 2021

Each augmentation choice in `CITATIONS.md` needs 2–3 references and a sentence
stating *what physical mechanism it models*, not merely that it exists.

---

## 7. Environment

System Python is 3.14 — **torch has no wheels for it.** Pin 3.12.

```powershell
# from the repo root
uv venv --python 3.12 .venv
.venv\Scripts\activate
uv pip install numpy scipy opencv-python pillow matplotlib scikit-image tqdm
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Hardware: RTX 4090 Laptop, 16 GB — comfortably sufficient. The re-ranker is small
enough to train in minutes.

Before submission: `uv pip freeze > requirements.txt` from the clean venv, then
clone into a fresh directory and run the README steps verbatim.
