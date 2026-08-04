# DRIFT-SENSE — Execution Plan

**Deadline:** Aug 16, 2026 · **Start:** Aug 3, 2026 · **Team size:** 3
**Companion doc:** [`TECH-SPEC.md`](./TECH-SPEC.md) — algorithm + generator + interfaces

---

## 1. Where the win actually is

The brief hides four scoring levers that most teams will walk past. Our whole
strategy is to take all four.

### Lever 1 — The tie-break rule is an admission

> *"If more than one matching region is found, return the one closest to the center
> of the Search Image."*

Applied Materials is telling us the problem is **ambiguous by construction**. A
tiled periodic layout means the reference genuinely appears at many lattice-
equivalent positions, and no algorithm can tell them apart from structure alone.

Nearly every team will call `cv2.matchTemplate(...).argmax()` and take one point.
That silently picks an arbitrary alias. The spec asks for something different:
enumerate **all** near-tied peaks, decide whether they are *statistically* tied,
and only then apply the center rule. We implement the rule literally. Free marks.

### Lever 2 — The test set contains a deliberately unsolvable case

> *"Include at least one highly periodic array region where correct localization is
> genuinely difficult — this is specifically designed to test failure mode awareness."*

They are not scoring whether you get that one right. They are scoring whether your
algorithm **knows** it is uncertain. So our output is never a bare `(x, y)` — it
carries a calibrated confidence and a Periodic Ambiguity Index, and the evaluation
report includes a **reliability diagram** proving the confidence is honest.

A team that reports 100% accuracy on this problem is either lying or has not
understood it. Our honesty is a competitive advantage, not a weakness — Slide 6
explicitly asks for an *honest failure case*.

### Lever 3 — 30% of the score is augmentation, and it must be cited

Most teams will write `img + np.random.normal(0, 15)`. That is not how an SEM
forms an image, and it will not earn 30%.

We build a **physically-grounded SEM forward model**: secondary-electron yield with
edge enhancement → beam point-spread function → scan drift and line jitter →
charging → Poisson shot noise → detector gain, read noise, quantization. Every
stage maps to a real physical mechanism with a real citation. This is the single
highest-value-per-hour item in the project.

### Lever 4 — "A novel approach to the 10x scale difference problem" (Slide 5)

Brute-force pyramid search over scales is what everyone does. We solve scale and
rotation in **closed form** from the Fourier spectrum: a periodic layout has a
reciprocal lattice, and the ratio of lattice vector magnitudes between reference
and search *is* the magnification ratio. Periodicity — the thing that makes the
problem hard — becomes the thing that makes it fast.

### The pitch, in one line

> Everyone else fights the periodicity. We measure it, exploit it for scale and
> rotation, subtract it to expose the aperiodic signal that actually disambiguates,
> and when nothing can disambiguate we say so and fall back to the rule in the spec.

---

## 2. Non-negotiable engineering rules

These exist because an unrunnable script scores zero regardless of how clever it is.

| # | Rule | Why |
|---|------|-----|
| 1 | `localize.py` **must run with zero ML dependencies installed.** Torch is optional; if weights or torch are missing, the classical core still returns an answer. | Their machine is not our machine. A single import error zeroes the entire Phase-2 score. |
| 2 | `localize.py` **must never raise.** Top-level try/except returns the search-image center as a last-resort answer with `confidence=0`. | A crash on pair 7 of 30 must not cost us pairs 8–30. |
| 3 | **Never hardcode `10`.** Scale is always measured, never assumed. | The brief says "~10x" and mandates scaling variation. Their test set will not be exactly 10.000x. |
| 4 | **Never hardcode image size, dtype, or channel count.** Accept PNG/TIF/JPG/BMP, grayscale or colour, any dimensions. | We do not control their file format. |
| 5 | **Accept both CLI conventions:** `localize.py ref.png search.png` *and* `localize.py --ref ref.png --search search.png`. | "Must run without manual edits." Cheap insurance against an invocation mismatch. |
| 6 | **Handle both DRAM and FinFET** regardless of which we generate. | The test set explicitly covers both. Choosing one style for *generation* is allowed; choosing one for *localization* is a 50% loss. |
| 7 | **Validate on harder data than we train on.** Higher noise, unseen pitches, unseen rotations. | Explicitly stated: their test set is noisier than ours. |
| 8 | **Fresh-machine test before submission.** Clean venv, `pip install -r requirements.txt`, clone, run. | Required by the README criterion, and it is where projects die. |

---

## 3. Team split

Roles are cut along the score weights so each person owns an outcome, not a pile
of files. Interfaces are frozen on Day 1 so all three work in parallel from hour one.

### Member A — Dataset & Physics · *owns the 30% augmentation score*

**Files:** `driftsense/layouts.py`, `driftsense/sem_physics.py`, `generate_dataset.py`, `CITATIONS.md`

- DRAM and FinFET layout synthesis with randomized pitch, duty cycle, contact
  geometry — plus **controlled aperiodic content** (array boundaries, dummy fill,
  periphery, defects) at a tunable rate. That knob is what makes some pairs
  solvable and some genuinely ambiguous, and we need both.
- The full SEM forward model (see TECH-SPEC §4).
- Ground-truth metadata per pair: true centre at sub-pixel precision, true scale,
  true rotation, **and the full list of lattice-alias positions** plus an
  `ambiguity_class` label.
- `CITATIONS.md` — 2–3 verified references per augmentation choice. **Verify every
  DOI yourself.** A fabricated citation in front of Applied Materials engineers is
  worse than no citation.

**Deliverable that defines success:** a frozen 30+ pair evaluation set, plus a
harder out-of-distribution set, plus a citation document that survives scrutiny.

### Member B — Localization Core · *owns the accuracy score*

**Files:** `driftsense/spectral.py`, `matching.py`, `periodic.py`, `decide.py`, `localize.py`

- Spectral lattice estimation → closed-form scale + rotation (TECH-SPEC §3.1).
- Correlation, full peak list, non-max suppression, sub-pixel refinement.
- Periodic/aperiodic decomposition — the core disambiguation idea (TECH-SPEC §3.5).
- Tie test, centre rule, confidence output.
- `localize.py` — **the single most important file in the repository.** Applied
  Materials runs this and nothing else to compute our Phase-2 score.

This is the highest-risk, highest-value path. Member B should be the strongest
algorithms person and should be shielded from documentation work.

### Member C — Learning, Evaluation & Presentation

**Files:** `driftsense/rerank.py`, `viz.py`, `train.py`, `evaluate.py`, `README.md`, slides

- Evaluation harness and metrics, including the **baseline comparison against plain
  `cv2.matchTemplate`** — Slide 3 and Slide 5 both ask why we beat template
  matching, and a chart answers it better than a paragraph.
- Siamese re-ranker CNN trained with **lattice-alias hard negatives** (TECH-SPEC §3.6).
  This ticks the DL-weights and training-script repository boxes and supplies the
  "AI" narrative — while remaining strictly optional at inference time.
- All figures. Confidence reliability diagram. The periodic/aperiodic decomposition
  visual. Success case and honest failure case.
- README written by the person who did **not** write the localizer — that is how you
  catch missing setup steps.
- Fresh-machine test, `pip freeze`, repo hygiene, video, slides.

### Shared, frozen on Day 1

Nobody may change these without telling the other two:

```python
# The pair format A produces and B/C consume
pairs/<id>/reference.png, search.png, meta.json

# The function B produces and C consumes
localize(reference_path, search_path) -> {
    "x": float, "y": float,          # centre in search-image pixels
    "confidence": float,             # [0, 1], calibrated
    "pai": float,                    # Periodic Ambiguity Index
    "candidates": [ ... ],           # ranked, for diagnostics
    "scale": float, "rotation": float,
    "decision": "unique" | "tie_broken_by_center" | "fallback",
    "time_ms": float,
}
```

**Day-1 unblocking trick:** Member A ships a deliberately crude generator (clean
lines, Gaussian noise, no physics) within the first few hours. B and C are then
never blocked. A replaces it with the real physics model over the following days,
and because the interface is frozen, nothing downstream breaks.

---

## 4. Timeline

Fourteen days. Four phases, three hard gates.

### Phase 0 — Foundation · Aug 3–4

| Who | Task |
|-----|------|
| All | Freeze the two interfaces above. Create Python 3.12 venv (**not 3.14 — no torch wheels**). Repo skeleton with stub modules that import and run end-to-end with dummy logic. Public GitHub repo created. |
| A | Crude v0 generator shipped. B and C unblocked. |
| B | FFT lattice estimation prototype on v0 data — prove scale recovery works. |
| C | Eval harness skeleton + `cv2.matchTemplate` baseline wired up. |

**Exit condition:** `generate_dataset.py` → `localize.py` → `evaluate.py` runs
end-to-end and prints a (terrible) accuracy number. Everything after this is
improving a number that already exists.

### Phase 1 — Core v1 · Aug 5–7

| Who | Task |
|-----|------|
| A | Full SEM physics model: SE edge yield, beam PSF, Poisson shot noise, detector chain. Both DRAM and FinFET layouts. Citations drafted. |
| B | Spectral scale/rotation + ZNCC + full peak list + NMS + sub-pixel + centre tie-break. |
| C | Metrics suite, baseline comparison chart, first figures. |

> **GATE 1 — Aug 7 · End-to-end working.**
> If the pipeline does not run cleanly on 30 pairs by end of Aug 7, **cut the CNN
> entirely** and spend the recovered time on the classical core. The classical core
> alone is a winning submission; a half-trained CNN is not.

### Phase 2 — Differentiators · Aug 8–10

| Who | Task |
|-----|------|
| A | Ambiguity-controlled generation. Augmentation sweeps. **Freeze the official 30+ eval set** and a harder OOD set. Citations finalized and DOI-verified. |
| B | Periodic/aperiodic decomposition. Multi-band agreement scoring. Confidence calibration. Fourier–Mellin cross-check. |
| C | Train the Siamese re-ranker with lattice hard negatives. Reliability diagram. Decomposition visual. |

> **GATE 2 — Aug 10 · Accuracy target.**
> ≥90% within 5 px on the unambiguous subset, and correct centre-rule behaviour on
> the ambiguous subset. If we are short, stop adding features and debug.

### Phase 3 — Harden · Aug 11–13

| Who | Task |
|-----|------|
| A | Noise-ladder robustness sweep — accuracy vs SNR curve. Sample data committed. |
| B | Bulletproof `localize.py`: no-torch path, never-raises wrapper, both CLI conventions, arbitrary formats/sizes. Timing optimization. |
| C | README from scratch. `pip freeze` → `requirements.txt`. **Fresh-machine clone-and-run test.** Failure analysis writeup. |

> **GATE 3 — Aug 13 · Code freeze.**
> After this date, documentation and slides only. No algorithm changes. Every
> hackathon loses a team to a "small improvement" committed at 2am on deadline day.

### Phase 4 — Present · Aug 14–16

| Who | Task |
|-----|------|
| C lead, all support | Nine slides per the i4C template. Success case + honest failure case visuals. Demo video. Final repo review. |
| All | Aug 16 is **buffer**. Target completion Aug 15. |

---

## 5. Slide-by-slide ownership

The PPT structure is dictated by the brief. Mapping it now prevents a scramble.

| Slide | Content | Owner | Depends on |
|-------|---------|-------|------------|
| 1 Team Details | Names, roles, college, contact | C | — |
| 2 Problem Statement | Why navigation-error recovery matters; drift, thermal, vibration | C | — |
| 3 Idea Description | Both architectures; spectral lattice approach; why it beats template matching | B | Gate 1 |
| 4 Proposed Solution | Generator design, SEM physics, pipeline diagram, **citations inline** | A + B | Phase 2 |
| 5 Innovation | Closed-form scale, aperiodic decomposition, ambiguity awareness | B | Phase 2 |
| 6 Results | Accuracy on 30+, time per pair, success case, **honest failure case** | C | Gate 2 |
| 7 Tech & Feasibility | Stack, RTX 4090 Laptop, generation time, inference ms, model size | C | Phase 3 |
| 8 GitHub & Video | Repo link, demo video | C | Gate 3 |
| 9 References | Full citation list, matching `CITATIONS.md` | A | Phase 2 |

Slide 4 and Slide 9 must agree exactly — the brief requires the PPT citations to
correspond to the repository citation document.

---

## 6. Risk register

| Risk | Severity | Mitigation |
|------|----------|------------|
| Their generator differs from ours; we overfit to our own physics | **High** | Measure everything, assume nothing (Rule 3). Hold out a generator config never used in training. Validate on the OOD set. Classical core has no learned priors to overfit. |
| `localize.py` fails on their machine | **High** | Rules 1, 2, 4, 5. Fresh-machine test at Gate 3. No-torch fallback path is mandatory, not optional. |
| Python 3.14 (system default) has no torch wheels | Medium | Pin a 3.12 venv on Day 0. Document the exact version in README. |
| CNN does not converge in time | Medium | Gate 1 kill switch. Re-ranker is architecturally optional — removing it degrades accuracy slightly, not catastrophically. |
| Fabricated or unverifiable citations | Medium | Member A verifies every DOI/link personally. Candidate list in TECH-SPEC §6 is a *starting point*, not pre-approved. |
| Scope creep past Aug 13 | Medium | Gate 3 code freeze, enforced by whoever is not the author. |
| Three people editing one repo | Low | Feature branches, PR review by one other member, `main` always runnable. |

---

## 7. What "standing out" concretely looks like

Judges see many decks. These are the four moments that will be remembered:

1. **The FFT slide.** Two spectra side by side, lattice peaks marked, magnification
   read straight off the ratio. It looks like physics, not like a tutorial.
2. **The decomposition visual.** Search image → periodic component → aperiodic
   residual, with the true match popping visibly out of the residual. One image
   that explains the entire contribution.
3. **The reliability diagram.** Predicted confidence vs observed accuracy, near the
   diagonal. Proof of failure-mode awareness, which the brief explicitly asks for
   and which cannot be faked.
4. **The honest failure slide.** A genuinely ambiguous pair, our four tied
   candidates shown, the centre rule applied, confidence correctly reported low.
   Confident honesty reads as mastery. Claimed perfection reads as a bug.

Everything else is table stakes.
