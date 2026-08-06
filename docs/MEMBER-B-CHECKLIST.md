# MEMBER B — Localization Core

**You own the accuracy score, and you own `localize.py` — the single most important file in
the repository.** From the brief: *"The localization inference script is the most important
file in your repository. Applied Materials will run it directly on their test image pairs to
compute your Phase 2 score. An unrunnable script cannot be scored."*

**Your files (nobody else edits these):**
```
driftsense/spectral.py     # FFT lattice estimation, Fourier-Mellin cross-check
driftsense/matching.py     # correlation, peak list, NMS, sub-pixel
driftsense/periodic.py     # periodic / aperiodic decomposition
driftsense/decide.py       # tie test, centre rule, confidence
localize.py                # THE critical file
```

**Read before you start:** `TECH-SPEC.md` §3 in full. That section is your implementation
contract. `PLAN.md` §2 rules 1–8 are non-negotiable and most of them are yours.

**You are shielded from documentation and slide work.** If someone asks you to write README
prose, say no and point at this line. C writes the README on purpose — a non-author catches
the missing setup steps.

---

## The eight rules that are yours to enforce

| # | Rule | Consequence if broken |
|---|---|---|
| 1 | `localize.py` runs with **zero ML deps installed**. Torch optional. | One import error zeroes the whole Phase-2 score |
| 2 | `localize.py` **never raises.** Top-level try/except returns search-image centre, `confidence=0` | A crash on pair 7 costs you pairs 8–30 |
| 3 | **Never hardcode `10`.** Scale is measured, always | Brief says "~10x"; their set will not be 10.000 |
| 4 | Never hardcode size, dtype, or channel count. PNG/TIF/JPG/BMP, grey or colour, any dims | You don't control their file format |
| 5 | Accept **both** CLIs: positional *and* `--ref/--search` | Cheap insurance against invocation mismatch |
| 6 | Handle **both DRAM and FinFET** | Their test set covers both; picking one is a 50% loss |
| 7 | Validate on harder data than you tune on | Their test set is explicitly noisier |
| 8 | Fresh-machine test before submission | That's where projects die |

---

## Legend

- 🔒 **BLOCKED BY** — cannot start until someone ships something.
- 🚦 **BLOCKING** — someone is idle until you ship.
- ✅ **Done when** — the objective test.

---

# DAY 0 — Wed Aug 6 (evening, ~3 h)

## B0.1 · Environment (30 min)
- [ ] Python **3.12** venv (3.14 has no torch wheels — and you may want torch later for the
      re-ranker interface even though C trains it).
  ```powershell
  uv venv --python 3.12 .venv
  .venv\Scripts\activate
  uv pip install numpy scipy opencv-python pillow matplotlib scikit-image tqdm
  ```
- ✅ Done when: `python -c "import cv2; print(cv2.__version__)"` works.

## B0.2 · Interface-freeze call with A and C (45 min)
🚦 **BLOCKING BOTH OTHERS — happens tonight.**
- [ ] Confirm the pair format A will write (`TECH-SPEC.md` §2.1).
- [ ] **Nail the coordinate convention out loud:** `(x, y)`, x → right, y → down, origin at the
      top-left of pixel (0,0), value is the **centre** of the matched region, sub-pixel float.
      Not (row, col). Not top-left corner of the match. Say it twice.
- [ ] **You define the `localize()` return dict.** Write it into `docs/INTERFACES.md` yourself:
  ```python
  localize(reference_path: str, search_path: str, use_reranker: bool = True) -> {
      "x": float, "y": float,       # centre, search-image pixels
      "confidence": float,          # [0,1], calibrated
      "pai": float,                 # Periodic Ambiguity Index
      "candidates": [               # ranked, for C's diagnostics and figures
          {"x": float, "y": float, "score": float, "rank": int}, ...
      ],
      "scale": float, "rotation": float,
      "decision": "unique" | "tie_broken_by_center" | "fallback",
      "time_ms": float,
  }
  ```
- [ ] Agree the **stdout contract** for `localize.py`: exactly one line, `412.4,688.0`, nothing
      else, all logging to **stderr**. An automated grader most likely parses stdout.
- ✅ Done when: `docs/INTERFACES.md` is on `main`.

## 🚦 B0.3 · SHIP THE `localize()` STUB TONIGHT (1 h)
🚦 **BLOCKING C** — C's evaluation harness needs something callable, tonight, to build against.
- [ ] `localize.py` that loads both images, ignores them, and returns the **search-image centre**
      with the full dict populated and `decision="fallback"`, `confidence=0.0`.
- [ ] Both CLI conventions already wired (Rule 5 — do it now, not on Aug 13).
- [ ] Top-level try/except already in place (Rule 2 — same reason).
- [ ] `python localize.py ref.png search.png` prints exactly `500.0,500.0`.
- ✅ Done when: C can `from localize import localize` and score 30 pairs against it.
      **Message C the moment it's pushed.**

---

# DAY 1 — Thu Aug 7

## B1.1 · Preprocessing module (2 h) — can start before A's data lands
🔒 Partially blocked by **A1.1 (v0 pairs, due 13:00)**. Write it against synthetic sine
gratings you make yourself in the morning; swap to A's pairs after lunch.

Per `TECH-SPEC.md` §3.0:
- [ ] Load as **float32 grayscale**; colour → luminance weights; accept any dtype (Rule 4).
- [ ] Robust normalize to the **1st–99th percentile** range — invariant to detector gain/offset
      differences between the two captures.
- [ ] Band split: `hp = I - GaussianBlur(I, sigma_large)` (matching runs on this),
      `lf` = the removed low-frequency field (kept for §3.5).
- [ ] Gradient magnitude `|∇I|`. SEM contrast is dominated by edge brightening, so the edge map
      is more stable across captures than raw intensity. You will score on both and fuse.
- ✅ Done when: feeding the same layout at two different gains produces near-identical `hp`.

## B1.2 · FFT lattice estimation — prototype (3 h)
🔒 **BLOCKED BY A1.1** (needs real periodic images). Start ~13:30.

This is the Slide-5 innovation. Everything else in the pipeline is table stakes.
- [ ] 2D **Hann window** both images (kills the spectral cross artifact from image boundaries).
- [ ] `M = log(1 + |fftshift(fft2(I))|)`, mask a small disc around DC.
- [ ] Detect local maxima, keep conjugate-symmetric pairs.
      DRAM → two strong orthogonal fundamentals + harmonics. FinFET → one dominant fin-pitch
      fundamental + weaker gate frequency.
- [ ] Fit reciprocal-lattice basis `(g1, g2)`: the two shortest non-collinear peak vectors that
      generate the remaining peaks within tolerance; refine by least squares over all harmonics.
- [ ] **Closed-form magnification:** `m = |g_S| / |g_R|`.
      (Lower magnification → features 10× smaller in pixels → period 10× smaller → frequency 10× higher.)
- ✅ **Success criterion for tonight:** on A's v0 pairs, your recovered `m` is within **2%** of
  `meta["magnification_ratio"]` on ≥25/30 pairs. If that works, the entire innovation claim on
  Slide 5 is real. Tell the team — it's a morale moment and C can start drafting the FFT slide.

## B1.3 · Rotation + symmetry hypotheses (1 h)
- [ ] `theta = angle(g_S basis) - angle(g_R basis)`.
- [ ] **Resolve modulo the lattice symmetry group** — 90° for a square DRAM grid, 180° for FinFET
      line arrays. Symmetry does **not** resolve itself: emit the small set of symmetry-equivalent
      hypotheses (typically 2 or 4) and carry **all** of them forward into matching. Forgetting
      this produces a mysterious 25% failure rate that costs a day to diagnose.

---

# DAY 2 — Fri Aug 8

## B2.1 · Template construction (1 h)
- [ ] Resize reference by `1/m` with **`INTER_AREA`** — area averaging matches how a real
      lower-magnification capture integrates signal over a larger pixel footprint, and how A's
      generator downsamples.
- [ ] Benchmark `INTER_AREA` vs Lanczos on the dev set; keep the winner and note the number
      (it's a nice Slide-4 detail).
- [ ] Rotate by `-theta`. Apply **the same §3.0 preprocessing** to the template as to the search
      image — template and search must be treated identically or ZNCC drifts.

## B2.2 · Dense correlation → FULL PEAK LIST (3 h)
> **Do not take `argmax`. That single decision is what separates this submission from the
> median one.** Read `PLAN.md` §1 Lever 1 again if you're tempted.

- [ ] ZNCC per (scale, rotation) hypothesis: `cv2.matchTemplate(..., TM_CCOEFF_NORMED)`.
      ~50–100 ms at 1000×1000 with a 100×100 template.
- [ ] Fuse intensity-channel and gradient-channel score maps.
- [ ] **Non-maximum suppression**, radius ≈ half the template size.
- [ ] Keep top **K ≈ 50** peaks with scores.
- [ ] **Sub-pixel refinement** by 2D quadratic fit on the 3×3 neighbourhood:
      `dx = 0.5*(C[y,x-1] - C[y,x+1]) / (C[y,x-1] - 2*C[y,x] + C[y,x+1])`, same for dy.
- [ ] Convert peak (which is the template's **top-left** in `matchTemplate` output) to the
      **centre** convention: `x_centre = x_peak + tw/2`, `y_centre = y_peak + th/2`.
      ⚠️ This is where the off-by-half-template bug lives. Verify against A's clean pairs.
- ✅ Done when: on a low-ambiguity clean pair, peak #1 is within 1 px of `true_center_xy`.

## B2.3 · Scale-sweep fallback (1 h) — Rule 3 and Rule 2 insurance
- [ ] If lattice peaks are weak or absent (blurry, low-contrast, or a layout you didn't expect),
      fall back to a coarse-to-fine scale sweep over `m ∈ [7, 14]`. Slower, always terminates.
- [ ] Never crash, never return nothing.

## B2.4 · Wire the real pipeline into `localize.py` (1 h)
- [ ] Replace the stub's guts, keep the exact same return dict and stdout contract.
- 🚦 Tell C: "localize.py now does real work, re-run the harness." C's accuracy chart will move
  off zero for the first time.

---

# DAY 3 — Sat Aug 9  ⟶ GATE 1 DAY

## B3.1 · Lattice-consistency structuring (2 h)
Per `TECH-SPEC.md` §3.4:
- [ ] Fit a lattice to the recovered **peak positions**.
- [ ] If the top peaks lie on a lattice matching the period measured in §3.1, you are **provably**
      in the degenerate periodic regime — and you now know it *analytically*, not from a
      threshold you tuned. This is the principled basis for the whole ambiguity report and is
      what makes your confidence output defensible instead of a magic number.
- [ ] Cross-check the fitted peak lattice against `meta["lattice_period_search_px"]` on dev pairs.

## B3.2 · Tie test + centre rule (2 h) — free marks, do not skip
The brief: *"If more than one matching region is found, return the one closest to the center of
the Search Image."* Implement it **literally**.
- [ ] Tie set = `{c : score(c) >= score_best - delta}`.
- [ ] For now set `delta` by hand; you'll calibrate it properly on Day 5.
- [ ] If `|tie set| > 1` → return the member **closest to the search-image centre**, and record
      `decision = "tie_broken_by_center"`.
- [ ] **Periodic Ambiguity Index** = `score_2 / score_1` over peaks separated by more than the
      template radius. Report it on every prediction.
- ✅ Done when: on a `degenerate` pair, `decision == "tie_broken_by_center"` and the returned
  point is the alias nearest the image centre — verify by hand against `meta["alias_positions"]`.

## 🚦 B3.3 · GATE 1 — full pipeline runs on 30 pairs — BY 18:00
🚦 **BLOCKING the team's gate decision.**
- [ ] `localize.py` produces a sane answer on all 30 of A's pairs, no exceptions raised,
      under ~2 s each.
- [ ] Report the accuracy number to the team, however bad it is.

> ## 🚧 GATE 1 — end of Sat Aug 9
> Pipeline runs end-to-end and prints an accuracy number. If it does not, **the CNN is cut**
> and C reallocates to evaluation and docs. The classical core alone is a winning submission;
> a half-trained CNN is not.

---

# DAY 4 — Sun Aug 10

## B4.1 · Periodic / aperiodic decomposition — THE core contribution (4 h)
🔒 Best done **after A4.2 lands** (`alias_positions` in meta) so you can verify against truth,
but you can start without it.

The periodic component is, by definition, the part that *cannot* disambiguate. So remove it and
match on what remains.

- [ ] **v1 — Fourier synthesis (do this first):** build a frequency-domain mask keeping only
      reciprocal-lattice points and their harmonics (small disc around each). Inverse-transform →
      `S_periodic`. Then `S_aperiodic = S - S_periodic`. Same procedure on the template.
- [ ] Correlate **residuals** for each candidate from §3.3, weighted by a residual-energy mask so
      that flat regions don't dominate the score.
- [ ] **v2 — unit-cell folding (only if time allows):** warp into lattice coordinates, average
      over all cells to get the mean unit cell, re-tile. More robust when the lattice drifts
      slowly across the field. Skip this if Day 5 is at risk.
- 🚦 **Save the three-panel image — search / periodic / aperiodic residual — to `figures/`.**
      C needs it for Slide 5; `PLAN.md` §7 calls it one of the four moments judges remember.
- ✅ **The elegant property to state on Slide 5:** when the layout is perfectly periodic the
  residual is pure noise and this stage contributes nothing — which is *correct*, because that
  case is genuinely ambiguous. Residual energy therefore doubles as a principled ambiguity
  measure. The method degrades exactly where the problem becomes unsolvable.

## B4.2 · Fourier–Mellin cross-check (1.5 h)
- [ ] Log-polar registration (Reddy & Chatterji 1996) on the two magnitude spectra. Returns scale
      and rotation by a **completely different route**.
- [ ] Two independent estimators agreeing is a strong confidence feature; disagreement raises a
      flag and widens the hypothesis set.
- [ ] Expose `scale_agreement` as a number — it's confidence feature #4.

## 🚦 B4.3 · Freeze the `candidates` list format (30 min)
🚦 **BLOCKING C's re-ranker integration.** C needs to know exactly what he's re-ranking.
- [ ] Confirm `candidates` is populated, ranked, with `{x, y, score, rank}` per entry.
- [ ] Define the hook: `rerank(template_patch, candidate_patches) -> list[float]`, called only
      if torch imports successfully, and **only reorders** — it never gates the pipeline.
- [ ] Tell C the signature.

---

# DAY 5 — Mon Aug 11  ⟶ GATE 2 DAY

## B5.1 · Score fusion + confidence calibration (3 h)
🔒 **BLOCKED BY A5.1** (frozen eval set, due 14:00) for the final numbers — but **tune on
`data/dev_v1`, never on `data/eval`**. Tuning on the eval set is how you convince yourself
you're winning and then lose.

- [ ] Fuse: ZNCC + gradient channel + aperiodic residual + (optional) re-ranker logit.
- [ ] **Calibrate `delta`** by bootstrapping over noise realizations on the validation set —
      i.e. measure the score spread induced by noise *alone*, and set the tie threshold to that.
      This turns a magic number into a measured quantity you can defend in Q&A.
- [ ] **Confidence** = logistic regression over four interpretable features:
      1. normalized peak margin
      2. aperiodic residual energy ratio
      3. lattice-consistency of the peak set
      4. agreement between the two independent scale estimators
      Fit on validation data. Four features, one explainable model, genuinely calibrated.
- 🚦 Hand C the confidence outputs — he needs them for the **reliability diagram**, one of the
  four judge-memorable moments and the direct answer to "failure mode awareness".

## B5.2 · Push for the Gate 2 number (2 h)
- [ ] Target: **≥90% within 5 px on the `unique` subset**, and **correct centre-rule behaviour on
      the `degenerate` subset** (measured as: prediction lands on the alias nearest the centre).
- [ ] Break failures down by `ambiguity_class` before "fixing" anything. A miss on a `degenerate`
      pair is not a bug — it's the expected outcome and you report it as such.

> ## 🚧 GATE 2 — end of Mon Aug 11
> If short of target: **stop adding features and debug.** No new ideas after tonight.

---

# DAY 6 — Tue Aug 12

## B6.1 · BULLETPROOF `localize.py` — highest-value day of your week (4 h)
Everything here is worth more than another 1% of accuracy.

- [ ] **Rule 1 — no-torch path.** Uninstall torch in a scratch venv and run. It must produce a
      full answer. `try: import torch / except ImportError: reranker = None`. Also handle
      "torch present but weights file missing".
- [ ] **Rule 2 — never raises.** Wrap `main()` in try/except. On any exception: print the
      search-image centre, `confidence=0`, `decision="fallback"`, exit code **0**.
      Test by feeding it: a corrupt PNG, a 1×1 image, a 4-channel RGBA, a 16-bit TIFF, a
      reference larger than the search image, and a path that doesn't exist.
- [ ] **Rule 4 — arbitrary inputs.** PNG / TIF / JPG / BMP, grayscale or colour, any dimensions.
      Do not assume 1000×1000 anywhere. Grep your own code for the literals `1000`, `100`, `10`.
- [ ] **Rule 5 — both CLIs**, plus `--json` for full diagnostics.
- [ ] **stdout discipline:** exactly one line, `x,y`, one decimal. Everything else to stderr.
      Run `python localize.py a.png b.png 2>/dev/null` and confirm the output is parseable by
      `float(s.split(',')[0])`.
- [ ] **Rule 6 — both layouts.** Run the full eval on DRAM-only and FinFET-only subsets
      separately. If either is much worse, fix it today.

## B6.2 · Timing (1.5 h)
- [ ] Target **< 1 s per pair on CPU** (not GPU — assume their machine has none).
- [ ] Profile. The usual wins: reduce the number of (scale, rotation) hypotheses once the
      spectral estimate is confident; run correlation at half resolution then refine.
- [ ] Report the measured median and worst-case ms to C for Slides 6 and 7.

---

# DAY 7 — Wed Aug 13  ⟶ CODE FREEZE

## B7.1 · OOD validation — the honesty test (2 h)
🔒 **BLOCKED BY A5.2** (`data/ood/`, shipped Aug 11).
- [ ] Run on the OOD set **once**. Do not tune afterwards. Whatever number comes out is the
      number you report as robustness evidence.
- [ ] If it's catastrophically worse than `data/eval`, you have overfit to A's physics — the
      classical core shouldn't, so investigate any hardcoded constant you missed (Rule 3).

## B7.2 · Support C's fresh-machine test (2 h)
🔒 **BLOCKED BY C** starting the test.
- [ ] Be on call. Every failure is yours to fix within the hour.
- [ ] Watch specifically for: a missing `__init__.py`, an import that works only from repo root,
      a relative weights path, an OpenCV version difference.

## B7.3 · Final read-through of `localize.py` (1 h)
- [ ] Docstrings on every public function.
- [ ] Delete dead experiment code. A reviewer opens this file first.
- [ ] Confirm the top-level try/except is still the outermost thing in the file.

> ## 🚧 GATE 3 — end of Wed Aug 13 · CODE FREEZE
> **No algorithm changes after tonight, enforced by A and C, not by you.**
> `PLAN.md` §6: "Every hackathon loses a team to a 'small improvement' committed at 2am on
> deadline day." Do not be that person on your own most important file.

---

# DAYS 8–9 — Thu Aug 14 / Fri Aug 15

## B8.1 · Own the content of Slides 3 and 5 (3 h)
C builds the deck; you supply the substance.
- [ ] **Slide 3 — Idea Description:** both architectures handled; spectral lattice approach;
      **why it beats plain template matching** on periodic layouts. C has the baseline chart.
- [ ] **Slide 5 — Innovation:** the three claims, in this order —
      1. **Closed-form scale and rotation from the reciprocal lattice.** Everyone else
         brute-forces a pyramid. The FFT integrates over the whole image, so the scale estimate
         averages noise over a million pixels; a local patch search sees ~10⁴ noisy pixels. Under
         the high-noise conditions their test set is *designed* to have, that gap is large.
      2. **Periodic/aperiodic decomposition** — subtract the part that can't disambiguate.
      3. **Ambiguity awareness** — enumerate all near-tied peaks, test statistically, then apply
         the centre rule. Report calibrated confidence, never a bare (x, y).
- [ ] The one-line pitch: *"Everyone else fights the periodicity. We measure it, exploit it for
      scale and rotation, subtract it to expose the aperiodic signal that actually disambiguates,
      and when nothing can disambiguate we say so and fall back to the rule in the spec."*

## B8.2 · Demo video technical run (1 h)
- [ ] Drive `localize.py` on a sample pair for C's screen recording. Two takes.

**Aug 16 is buffer. Target done on Aug 15.**

---

## Your dependency summary

| You are blocked by | What | When you're unblocked |
|---|---|---|
| **A** | Crude v0 pairs (`A1.1`) — you cannot prototype the FFT without periodic images | **Aug 7, 13:00** |
| A | Physics-grade pairs (`A3.3`) — realistic noise for tuning | Aug 9, 18:00 |
| A | `alias_positions` + `ambiguity_class` (`A4.2`) — to verify the tie-break against truth | Aug 10 evening |
| A | Frozen eval set (`A5.1`) — for final calibration numbers | **Aug 11, 14:00** |
| A | OOD set (`A5.2`) — the one-shot honesty test | Aug 11 EOD |
| C | Evaluation harness (`C1.x`) — so you see accuracy without writing your own scorer | Aug 7 EOD |
| C | Fresh-machine test results | Aug 13 |

| You are blocking | What they need | Your deadline |
|---|---|---|
| **C** | `localize()` stub with the frozen return dict (`B0.3`) | **Aug 6 tonight** |
| C | Real pipeline wired in, so the accuracy chart moves (`B2.4`) | Aug 8 EOD |
| Team | Gate 1: runs on 30 pairs (`B3.3`) | Aug 9, 18:00 |
| C | Decomposition three-panel figure (`B4.1`) | Aug 10 EOD |
| C | `candidates` format + `rerank()` signature (`B4.3`) | Aug 10 EOD |
| C | Confidence values for the reliability diagram (`B5.1`) | Aug 11 EOD |
| C | Timing numbers for Slides 6 & 7 (`B6.2`) | Aug 12 EOD |
| C | Slide 3 & 5 content (`B8.1`) | Aug 14 |

**Morning-of-Aug-7 note:** you are blocked until A ships at 13:00. Spend the morning on
`B1.1` preprocessing against your own synthetic sine gratings — do not sit idle waiting.
