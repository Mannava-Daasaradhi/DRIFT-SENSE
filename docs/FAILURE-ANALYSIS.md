# DRIFT-SENSE — Failure Analysis

*Member C, written for C7.3. Feeds the "honest failure case" on Slide 6 and the
brief's "failure-mode awareness" scoring criterion.*

---

## 1. The three failure modes, in order of importance

### 1.1 Degenerate pairs — 0% accuracy, correctly

**What happens.** `data/eval` contains 6 pairs with `ambiguity_class == "degenerate"`
(generated with `aperiodic_content_level = 0`). The layout is a perfect lattice: every
site is pixel-for-pixel identical to every other site. The true centre is unrecoverable
**in principle** — not as a limitation of the algorithm, but as a property of the
information in the image.

**What the system does.** It detects the degeneracy (the aperiodic residual is
indistinguishable from noise), sets `confidence` to its minimum (0.5333), and applies
the brief's tie-break rule: return the candidate closest to the search-image centre.
`decision` is either `"low_confidence_best"` or `"tie_broken_by_center"`.

**Why the 0% accuracy on degenerate pairs is the correct answer.**
Every algorithm that claims accuracy > 0% on a genuinely degenerate pair is either
lucky or not measuring it — there is no information to act on. The brief explicitly
includes this case ("at least one highly periodic array region... specifically designed
to test failure mode awareness"). Reporting it honestly scores better than hiding it.

**What would fix it.** Nothing, algorithmically. Acquiring a second image from a
slightly different detector angle (BSE vs SE) would break the symmetry. Out of scope
for this problem.

---

### 1.2 High-rotation, low-aperiodic-content pairs

**What happens.** On `data/eval`, 6 of the 14 failures on non-degenerate pairs have
errors > 50 px. Five of these have `ambiguity_class == "weakly_ambiguous"` with
rotation estimates that are 90° or 180° off from ground truth.

**Root cause.** The DRAM layout has 4-fold symmetry: a 90° rotation of the template
matches the search image nearly as well as the correct orientation. When aperiodic
content is weak (few array boundaries, no defects), the aperiodic residual cannot
break this symmetry, and the spectral stage's symmetry-equivalent hypothesis set
(`θ, θ+90°, θ+180°, θ+270°`) produces four correlation peaks of nearly equal score.
The tie-break rule then fires and picks the candidate closest to centre, which is
arbitrary.

**Observed pattern in results.json.** Pairs with `rotation_true ≈ 0°` and
`rotation_pred ≈ ±90°` or `≈ ±180°` are the characteristic failure case for
`weakly_ambiguous` DRAM pairs.

**What would improve it.**
- More aperiodic content training signal (A could increase `aperiodic_content_level`
  in the training generator).
- A second-order spectral cross-check between the estimated scale and rotation could
  eliminate some hypothesis branches.
- The re-ranker (disabled by default) is specifically trained to break these
  symmetry ties using aperiodic patch content — but on OOD data it overfits our
  generator's noise characteristics.

---

### 1.3 Large-offset failures on "unique" pairs

**What happens.** 4 of the 14 unique-class pairs have error > 30 px. These are pairs
where the spectral scale estimation is within 2% but the correlation peak ranking
puts a wrong candidate first.

**Root cause.** The true site has low aperiodic content relative to a false site near a
strong array boundary or periphery block. The boundary is a stronger correlation
signal than the correct match. Without stronger prior knowledge of the layout, this
is a hard case.

**Example.** `dram_00028` (err = 38.6 px, `unique`, `decision = "unique"`): the true
site sits near a quiet region of the array; a periphery block 40 px away produces a
higher ZNCC score. Independently re-checked against `results.json`: the winning
candidate's rotation estimate is 177.1°, 179.8° away from the true -2.7° - i.e. this
is really the 4-fold-symmetry failure mode from S1.2 wearing a `unique` label, not a
distinct mechanism. `aperiodic_content_level` on this pair is high enough that the
ground-truth `ambiguity_class` reads `unique`, but not high enough to break the
rotational tie in practice - the aperiodic residual and the raw symmetry ambiguity
are fighting each other, and here the symmetry wins. `dram_00008` (err = 378 px,
`pai = 0.999`) is the mirror case: `aperiodic_energy_fraction` = 0.1695, just over
the `weakly_ambiguous` cutoff (0.148) A's threshold uses, but the correlation stage
still sees it as a near-total tie and correctly reports minimum confidence (0.533)
rather than a false-confident wrong answer. Both are boundary cases where the
ground-truth label and what the algorithm can actually resolve don't quite line up -
worth widening the margin between `unique` and `weakly_ambiguous` in a future eval
set, not something to patch by moving the frozen set's threshold now.

**What would improve it.** The re-ranker is the most direct lever: it is trained to
distinguish true-site texture from boundary texture. That it does not help on OOD
pairs is a signal that the re-ranker has learned generator-specific texture rather
than generalizable aperiodic cues — the expected failure mode when training data is
synthetic.

---

## 2. What the system does well

Presented here so that the failure analysis is calibrated, not just self-critical.

| Metric | Value | Context |
|--------|-------|---------|
| Accuracy @5px, `unique` | **85.7%** | 12/14 pairs correct |
| Magnification estimate | **0.26%** median error | 34/36 within 2% |
| Alias-hit rate | **0.0%** | Never picks a lattice-equivalent site |
| Robustness tests | **15/15** | Corrupt files, 1×1 images, no torch, path spaces |
| OOD accuracy @5px | **70.0%** vs baseline 50.0% | Unseen pitches, 2-3× noise, ±6° rotation |

The 20-point OOD gap over the baseline (`cv2.matchTemplate` fixed 10×) is the clearest
demonstration that measuring the scale is better than assuming it: the baseline
degrades precisely where its two assumptions break (scale ≠ 10, rotation ≠ 0°), while
DRIFT-SENSE is mostly unaffected.

---

## 3. Failure mode for the submission grader

The one failure that would zero the Phase-2 score is an unhandled exception or a
non-parseable stdout. `tools/test_robustness.py` passes 15/15 checks including:

- torch absent (simulated via PYTHONPATH shadow)
- weights file deleted
- corrupt image, 1×1 image, reference larger than search
- JPEG, 16-bit TIFF, RGBA PNG, grayscale BMP
- path with spaces
- different working directory

Every one of these exits 0 and prints a parseable `x,y` coordinate. This is the hard
guarantee that matters more than any accuracy number.

---

## 4. What we would do with more time

1. **More OOD training data.** The generator can produce unseen pitches and higher
   noise levels. Training the re-ranker on these would close the OOD gap.
2. **Uncertainty-aware tie-breaking.** Instead of always returning the centre-closest
   candidate, estimate which hypothesis branch has higher spectral evidence.
3. **Semi-supervised fine-tuning.** A few real SEM image pairs (even without ground
   truth) would calibrate the noise model and improve transfer.
4. **Multi-scale correlation.** Running correlation at two scales and fusing would
   reduce the tail of large errors on weakly-ambiguous pairs.
