# DRIFT-SENSE — Navigation-Error Recovery

Locate a high-magnification **reference** pattern inside a ~10x lower-magnification
**search** image of a repeating semiconductor layout, and return the centre `(x, y)`
in search-image pixels.

Submission for the i4C / Applied Materials hackathon problem *Drift-Sense:
Navigation-Error Recovery*. The repository contains a physically-grounded synthetic
SEM dataset generator, the localization algorithm, an evaluation harness with the
mandatory `cv2.matchTemplate` baseline, and an optional CNN re-ranker with its
training script.

---

## The idea in one paragraph

Every die on a wafer carries the same repeating circuit, so the structure at the
wrong location looks almost exactly like the structure at the right one. Template
matching therefore produces hundreds of near-identical correlation peaks and
`argmax` silently picks an arbitrary one.

**Everyone else fights the periodicity. We measure it.** A repeating layout is a
2-D crystal, so its Fourier transform is a discrete *reciprocal lattice*. The ratio
of reciprocal-vector lengths between the two captures **is** the magnification, in
closed form — no pyramid, no scale sweep. We then *subtract* the periodic component
to expose the aperiodic residual (array boundaries, periphery blocks, defects),
which is the only content that can actually distinguish one lattice site from
another. And when the residual carries nothing — a genuinely degenerate pair — the
method says so, reports low confidence, and falls back to the tie-break rule the
brief specifies.

---

## Quick start

Requires **Python 3.12**. (3.13/3.14 have no torch wheels; torch is optional here,
but 3.12 is the version everything is verified on.)

```bash
git clone https://github.com/Mannava-Daasaradhi/DRIFT-SENSE.git
cd DRIFT-SENSE

python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 1. Generate a sample image pair

```bash
python generate_dataset.py --style dram --num 1 --out data/smoke --seed 0
```

This writes `data/smoke/dram_00000/` containing `reference.png`, `search.png` and
`meta.json` (which records the true centre — the ground truth).

### 2. Run the localizer on it

```bash
python localize.py data/smoke/dram_00000/reference.png data/smoke/dram_00000/search.png
```

**stdout is exactly one line and nothing else:**

```
412.4,688.0
```

That is the predicted centre `(x, y)` in search-image pixels. Every log line,
warning and traceback goes to stderr, and the exit code is always `0` — so an
automated grader doing `float(stdout.split(',')[0])` always gets a number.

### 3. Reproduce the reported results

```bash
python generate_dataset.py --style both --num 36 --out data/eval --seed 1337
python evaluate.py --data data/eval
```

---

## Results

Frozen 36-pair evaluation set (`--style both --num 36 --seed 1337`, tagged
`eval-set-frozen`) and a 30-pair **out-of-distribution** set generated from a
configuration never used for tuning — unseen pitches, 2-3x the noise, rotation to
±6°, `m ∈ [8, 13]`.

The brief states the official test set will be **noisier** than ours, so the OOD
column is the more honest predictor of Phase-2 performance.

### Accuracy — within 5 px of the true centre

| Set | DRIFT-SENSE | `cv2.matchTemplate` @ fixed 10x | median error | baseline median |
|---|---|---|---|---|
| `data/eval` (36 pairs) | **61.1%** | 58.3% | **2.76 px** | 3.83 px |
| `data/ood` (30 pairs) | **63.3%** | 33.3% | **2.91 px** | 22.82 px |

On the in-distribution set the baseline is competitive, because that draw happens
to sit near its two built-in assumptions: magnification close to 10 and rotation
close to 0. **Move off those assumptions and it collapses** — on the OOD set the
baseline's median error is 22.8 px against our 2.9 px, and it solves half as many
pairs. That gap is the entire argument for measuring the scale instead of assuming
it.

### Broken out by difficulty — this is the part that matters

A single blended accuracy number hides the whole story, so we never report one.

| `ambiguity_class` | `data/eval` | `data/ood` | what it means |
|---|---|---|---|
| `unique` | **85.7%** | **93.3%** | enough aperiodic content to identify the site |
| `weakly_ambiguous` | 62.5% | 71.4% | some disambiguating structure, not much |
| `degenerate` | 0.0% | 0.0% | **no algorithm can solve these** — see below |

**The degenerate row is 0% and that is the correct answer, not a bug.** Those pairs
are generated with `aperiodic_content_level = 0`: the layout is a perfect lattice,
every site is pixel-for-pixel equivalent, and the true centre is unrecoverable in
principle. The brief deliberately includes such a case ("at least one highly
periodic array region where correct localization is genuinely difficult — this is
specifically designed to test failure mode awareness"). On those pairs we detect
the degeneracy, report `decision` accordingly, and cap confidence. A submission
claiming high accuracy on this subset is either lucky or not measuring it.

### Other measurements

| | |
|---|---|
| Time per pair | **~0.6 s** median, 0.68 s worst, on CPU (1000x1000 search, ~100x100 template) |
| Magnification estimate | median error **0.26%**, 34/36 within 2% |
| Alias-hit rate | 0.0% |
| Confidence calibration | ECE **0.079** on `data/eval` (fitted only on `data/dev_v0` + `data/ood`) |
| Re-ranker checkpoint | 0.56 MB, ~217k parameters |

Figures land in `figures/` (accuracy vs baseline, error CDF, confidence
reliability diagram, success case, honest failure case — each also generated
against `data/ood`, suffixed `_ood`, which is the more compelling comparison
since the baseline's assumptions happen to nearly hold on the in-distribution
draw).

---

## On the CNN re-ranker — it is off by default, deliberately

`driftsense/rerank.py` is a small Siamese network trained with **lattice-alias hard
negatives**: the negatives are the lattice-equivalent sites, which look nearly
identical to the positive, so the only way to separate them is the aperiodic
content. It reaches 0.987 validation AUC.

It is nonetheless **disabled by default**, because we measured what it does
end-to-end rather than trusting the AUC:

| | `data/eval` (tuned on) | `data/ood` (held out) |
|---|---|---|
| classical only | 61.1% | **63.3%** |
| with re-ranker | **63.9%** | 50.0% |

It gains 2.8 points on the set its fusion weight was tuned against and loses 13.3
on the held-out configuration — the `unique` subset drops from 93.3% to 73.3%.
That is overfitting to our own generator, which `docs/PLAN.md` §6 lists as the top
project risk, and the OOD set exists precisely to catch it. The classical core
carries no learned priors and does not degrade that way.

Everything ships and it can be switched on:

```bash
python localize.py --ref r.png --search s.png --reranker
python train.py --data data/train --epochs 25 --out weights/reranker.pt
```

We would rather present a measured negative result than a number that looks better
on our own data and worse on theirs.

---

## Command reference

### `generate_dataset.py` — synthetic pair generator

```bash
python generate_dataset.py --style {dram,finfet,both} --num N --out DIR --seed S [--ood]
```

Renders each magnification **separately** from the analytic layout rather than
downsampling one big canvas — otherwise the search image's noise is a decimated
copy of the reference's, which violates the brief's independent-sensor-noise
requirement and produces an unrealistically clean search image.

The forward model runs in physical order (`driftsense/sem_physics.py`): SE yield
with edge brightening → area downsample → beam PSF → scan drift, row jitter and
vibration → charging → shading → **Poisson** shot noise → detector gain, read
noise, quantization. Every stage is justified in [`CITATIONS.md`](CITATIONS.md).

`--ood` selects the held-out configuration used for the honesty check above.

### `localize.py` — the inference script

```bash
python localize.py reference.png search.png              # positional
python localize.py --ref r.png --search s.png            # named
python localize.py --ref r.png --search s.png --json     # full diagnostics
python localize.py --ref r.png --search s.png --reranker # enable the CNN
```

Both invocation styles work because we do not control how it will be called.

Guarantees, verified by `python tools/test_robustness.py` (15 checks):

- **Runs with zero ML dependencies.** torch absent, weights missing, or the model
  raising — the classical answer stands.
- **Never raises.** Any internal failure returns the search-image centre with
  `confidence = 0.0` and `decision = "fallback"`, exit code `0`.
- **No assumption about image size, dtype, channel count or file format.**
  PNG/TIF/JPG/BMP, 8- or 16-bit, grayscale/RGB/RGBA, any dimensions.
- **Nothing but the coordinate on stdout.**

### `evaluate.py` — harness, baseline and figures

```bash
python evaluate.py --data data/eval --out figures --results results.json
python evaluate.py --data data/ood --no-figures        # numbers only
```

Scores the localizer and the `cv2.matchTemplate` baseline through the identical
code path, breaks results out by `ambiguity_class` and architecture style, and
writes every figure plus a full per-pair `results.json`.

### `train.py` — reproduce the re-ranker checkpoint

```bash
python generate_dataset.py --style both --num 200 --out data/train --seed 1
python train.py --data data/train --epochs 25 --out weights/reranker.pt
```

Splits **by pair**, never by example — patches from one pair share a template and
overlap heavily, so an example-wise split leaks the answer and reports a
meaningless AUC. Degenerate pairs are excluded from training: there is nothing to
learn on them, and including them teaches the network to manufacture confidence.

---

## Repository layout

```
README.md                  this file
requirements.txt           pip freeze from the development venv
CITATIONS.md               every augmentation choice, with verified references
generate_dataset.py        synthetic pair generator            (mandatory item 2)
localize.py                inference script                    (mandatory item 3)
train.py                   re-ranker training                  (mandatory item 5)
evaluate.py                metrics, baseline comparison, figures
weights/reranker.pt        trained checkpoint, 0.56 MB         (mandatory item 4)
driftsense/
  layouts.py               DRAM / FinFET geometry synthesis
  sem_physics.py           SEM image-formation forward model
  preprocess.py            normalization, band split, gradient channel
  spectral.py              reciprocal-lattice estimation, Fourier-Mellin check
  matching.py              template construction, correlation, peak list, NMS
  periodic.py              periodic / aperiodic decomposition
  decide.py                tie test, centre rule, confidence
  rerank.py                optional Siamese re-ranker
  viz.py                   figures
tools/                     development checks (ground truth, robustness, spectral)
docs/                      PLAN.md, TECH-SPEC.md, INTERFACES.md, checklists
data/                      generated pairs (gitignored; one sample pair committed)
figures/                   generated figures
```

`data/` is gitignored apart from one committed sample pair — every split is
reproducible from its seed with the commands above.

---

## How the algorithm works

1. **Preprocess.** Robust 1st–99th percentile normalization (invariant to detector
   gain and offset between two captures), a high-pass structure band, and a
   gradient-magnitude channel — SEM contrast is edge-dominated, so the edge map is
   the most stable thing across two captures at different dose.

2. **Measure scale and rotation from the reciprocal lattice.** Detect spectral
   peaks in both images; every (reference peak, search peak) pairing proposes one
   `(scale, rotation)`, and correct correspondences all propose the same value
   while wrong ones scatter. The mode of that vote is the answer. The FFT
   integrates over all 10⁶ pixels, so the estimate averages noise down far better
   than a patch-based scale search over ~10⁴ pixels — which matters exactly
   because the test set is noisy.

3. **Offer, do not guess.** The spectrum genuinely cannot resolve the layout's
   point-group symmetry or the octave ambiguity (a lattice at half the true
   magnification explains many of the same peaks). So eight hypotheses are
   proposed, ranked cheaply at half resolution, and the survivors are scored in
   full. Correlation is the arbiter, not the spectrum.

4. **Correlate, and never `argmax`.** ZNCC on both channels, non-maximum
   suppression, sub-pixel quadratic refinement, and a **full ranked peak list**.

5. **Subtract the periodic component.** Keep only reciprocal-lattice points in the
   Fourier domain, inverse-transform, subtract. The residual isolates exactly the
   content capable of resolving position. When the layout is perfectly periodic the
   residual is noise and this stage contributes nothing — which is *correct*,
   because that case is genuinely ambiguous. Residual energy therefore doubles as a
   principled ambiguity measure.

6. **Decide, and be honest.** A tie test over the peak list; if several matches are
   indistinguishable *and* the residual cannot separate them, return the one
   closest to the search-image centre exactly as the brief specifies, and report
   low confidence. The output is never a bare `(x, y)`.

Full detail: [`docs/TECH-SPEC.md`](docs/TECH-SPEC.md).

---

## Development environment

Verified on Python 3.12.13, OpenCV 5.0.0, NumPy 2.5.1, SciPy 1.18.0,
scikit-image 0.26.0, torch 2.13.0+cpu (Windows 11). All results above were measured
on **CPU**; no GPU is required to run or to evaluate, and the re-ranker trains on
CPU in about a minute.
