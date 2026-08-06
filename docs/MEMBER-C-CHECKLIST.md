# MEMBER C — Learning, Evaluation, Documentation & Presentation

**You own everything the judges actually see.** A and B can build a perfect system and still
score badly if the deck doesn't prove it, the README doesn't run, or the repo is missing a
mandatory item. You are also the person who catches that, because you did not write the code.

**Your files (nobody else edits these):**
```
driftsense/rerank.py       # Siamese re-ranker (strictly optional at inference)
driftsense/viz.py          # all figures
train.py                   # mandatory repo item #5
evaluate.py                # 30+ pair harness, metrics, figures
weights/reranker.pt        # mandatory repo item #4, keep under 5 MB
README.md                  # mandatory repo item #1
requirements.txt           # mandatory repo item #6
docs/slides_content.md     # Component 1 — the PPT
```

**The mandatory repo checklist is yours to guard.** From the brief, the repo must be **public**
and contain all seven of: README, dataset generator, localization script, DL weights, training
script, requirements.txt, citation document. Missing one is a scored loss regardless of accuracy.

---

## Legend

- 🔒 **BLOCKED BY** — cannot start until someone ships something.
- 🚦 **BLOCKING** — someone is idle until you ship.
- ✅ **Done when** — the objective test.

---

# DAY 0 — Wed Aug 6 (evening, ~3 h)

## C0.1 · Environment — you're the only one who needs torch (45 min)
- [ ] Python **3.12** (system 3.14 has **no torch wheels** — this bites you, not A or B).
  ```powershell
  uv venv --python 3.12 .venv
  .venv\Scripts\activate
  uv pip install numpy scipy opencv-python pillow matplotlib scikit-image tqdm
  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  ```
- ✅ Done when: `python -c "import torch; print(torch.cuda.is_available())"` prints `True`
  (RTX 4090 Laptop). If it prints False, fix it tonight — not on Aug 12.

## C0.2 · Interface-freeze call with A and B (45 min)
- [ ] You are the **consumer** of both interfaces, so you have veto power over anything awkward.
- [ ] Confirm you can read A's `meta.json` and call B's `localize()`.
- [ ] Push back if the coordinate convention isn't stated in writing: `(x, y)`, x → right,
      y → down, origin top-left of pixel (0,0), value = **centre** of match, sub-pixel float.
- ✅ Done when: `docs/INTERFACES.md` is on `main` and you've read it.

## C0.3 · Create and configure the public GitHub repo (45 min)
🚦 **BLOCKING nothing today, but it must exist before anyone pushes.**
- [ ] **Public** repository. Not private-with-a-plan-to-flip-it-later. Verify in an incognito
      window that it loads while logged out.
- [ ] `.gitignore`: `data/`, `.venv/`, `__pycache__/`, `*.pyc` — but **allow one sample pair**
      (repo requirement) and **allow `weights/*.pt`** (mandatory item #4, so it must be committed).
- [ ] Branch protection off; three people need to move fast. Convention: feature branches,
      PR reviewed by one other member, `main` always runnable.
- [ ] MIT licence, sensible repo name.

## C0.4 · Repo skeleton (45 min)
- [ ] Create the full directory tree from `TECH-SPEC.md` §1 with empty/stub files, so nobody
      argues about where a file goes at 1 a.m. on Aug 12.
- [ ] `driftsense/__init__.py` **must exist** — a missing `__init__.py` is the classic
      works-on-my-machine failure that only shows up in the fresh-machine test.

---

# DAY 1 — Thu Aug 7

## C1.1 · Evaluation harness (3 h)
🔒 **BLOCKED BY B0.3** (the `localize()` stub — shipped last night) and **A1.1** (v0 pairs, 13:00).
Write the harness in the morning against the stub with hand-made fake pairs; point it at A's
real data after lunch.

- [ ] `evaluate.py --data data/eval --out figures/` walks pair folders, calls `localize()`,
      compares to `meta["true_center_xy"]`.
- [ ] Metrics, per `TECH-SPEC.md` §5.1:
  - mean / median Euclidean error in px
  - **accuracy within {1, 2, 5, 10} px** tolerance
  - **broken out by `ambiguity_class`** — one blended number hides the entire scientific story
  - **alias-hit rate**: predictions landing on a lattice-equivalent site rather than the true
    one. On `degenerate` pairs this is the *expected outcome*, not a bug — report it that way.
  - wall-clock per pair (median and worst case)
- [ ] Write results to `results.json` so figures regenerate without re-running localization.
- [ ] Handle `ambiguity_class == null` gracefully — A won't add it until Aug 10.
- 🚦 ✅ **Done when: it prints a (terrible) accuracy number.** Announce it. From this point on,
  the whole team is improving a number that already exists — that psychological shift matters.

## C1.2 · The mandatory baseline (1.5 h)
`TECH-SPEC.md` §5.2. Slides 3 and 5 both ask why you beat template matching; a chart answers
that better than a paragraph.
- [ ] `baseline_matchtemplate()` — downscale the reference by a **fixed assumed 10.0×**, plain
      `cv2.matchTemplate`, `argmax`. That's it. This is what most competing teams will submit.
- [ ] Run it through the identical harness so the comparison is apples to apples.
- [ ] Expect it to fail on rotation, on non-exact scale, and on high-periodicity cases — exactly
      the three axes your system handles. Confirm that's what you observe.
- ✅ Done when: you have two accuracy numbers in one table. **Show A and B.** It sets the bar.

## C1.3 · Repo hygiene pass (30 min)
- [ ] Confirm A's and B's Day-0 commits are on `main` and `main` still runs.

---

# DAY 2 — Fri Aug 8

## C2.1 · Figure infrastructure (2 h)
Build `driftsense/viz.py` now so that on Aug 13 you're regenerating figures, not writing
plotting code.
- [ ] `plot_pair(ref, search, pred_xy, true_xy, aliases)` — the standard result visual.
      Brief, Slide 6: *"Visual: reference image, search image, your predicted location, true
      location."* Draw exactly that. Predicted = one colour, true = another, aliases = faint.
- [ ] `plot_accuracy_bars(results, baseline_results)` — you vs baseline, grouped by tolerance.
- [ ] `plot_error_cdf(results)`.
- [ ] Consistent style: one colour per method, readable at projector size, labelled axes with
      units. Save at 200 dpi PNG **and** keep the source arrays in `results.json`.

## C2.2 · Re-ranker architecture + training scaffold (3 h)
🔒 Partially blocked by **A3.3** (physics pairs, Aug 9). Build the code today, train tomorrow.

Per `TECH-SPEC.md` §3.6:
- [ ] **Input:** template patch and candidate patch, both resized to **64×64**.
- [ ] **Architecture:** shared CNN encoder, 4–5 conv blocks, ~200 k params, feature
      concatenation, MLP head → single logit. **Checkpoint must stay under 5 MB.**
- [ ] `train.py` with argparse, a fixed seed, and a printed final validation number. This is
      mandatory repo item #5 — *"include the training script that reproduces your training
      process."* It must actually reproduce.
- [ ] Save to `weights/reranker.pt` (mandatory item #4) and **load it automatically** in
      inference — the brief says weights "must be loaded automatically by your inference script".
- ✅ Done when: `python train.py --epochs 1` completes on v0 data without crashing.

---

# DAY 3 — Sat Aug 9  ⟶ GATE 1 DAY

## C3.1 · Gate 1 support (2 h)
- [ ] Run the full harness the moment B says the real pipeline is wired in (`B2.4`).
- [ ] Produce the first proper accuracy table: yours vs baseline, broken out by style.
- [ ] Circulate it. This table is the skeleton of Slide 6.

> ## 🚧 GATE 1 — end of Sat Aug 9
> Pipeline runs end-to-end on 30 pairs and prints an accuracy number.
> **If it does not: the CNN is cut entirely.** You drop `C2.2`/`C4.1` and move full time to
> evaluation, figures and documentation. `PLAN.md` is explicit: the classical core alone is a
> winning submission; a half-trained CNN is not. Make this call honestly — it's your call to
> make, since it's your workstream that gets cut.

## C3.2 · Start the README (2 h) — yes, this early
🔒 Nothing blocks a first draft.
Brief, item #1: *"A reviewer must be able to clone your repo, generate a sample image pair, and
run the localization algorithm from the README **without contacting you**."*
- [ ] Write it as a **stranger following instructions**, not as a summary of what exists.
- [ ] Sections: what this is → requirements (**state Python 3.12 explicitly**) → install →
      generate a sample pair → run localization on it → expected output → run the full eval →
      results table → repo map → citations link.
- [ ] Every command copy-pasteable. No `<your-path-here>` placeholders in the critical path.
- 📌 You write this **because you didn't write the localizer.** That's the whole point — an
  author's README silently omits the three steps they do from muscle memory.

---

# DAY 4 — Sun Aug 10

## C4.1 · Train the re-ranker with lattice hard negatives (4 h)
🔒 **BLOCKED BY A4.2** (`alias_positions` in `meta.json`) — pull hard negatives straight from
that field rather than detecting them yourself.

This is the design decision that makes the "AI" narrative real rather than decorative:
- [ ] **Positives:** the true location.
- [ ] **Hard negatives:** the **lattice-alias positions**. They look nearly identical to the
      positive, which is precisely what forces the network to learn the *aperiodic* cues instead
      of the periodic texture.
- [ ] **Easy negatives:** random offsets.
- [ ] **Train only on pairs with genuine aperiodic content.** On `degenerate` pairs there is no
      learnable signal, and training on them teaches the network to hallucinate confidence.
      Degenerate cases are handled by the centre rule, not the network.
      **State this explicitly on Slide 5** — it reads as a mature design choice, not a limitation.
- [ ] Train on the 4090. This model is small; minutes, not hours.
- [ ] Report val AUC on held-out pairs.
- ✅ Done when: `weights/reranker.pt` is **< 5 MB**, committed, and B's optional hook loads it.

## C4.2 · Integrate the re-ranker into B's pipeline (1 h)
🔒 **BLOCKED BY B4.3** (candidates format + `rerank()` signature).
- [ ] Implement exactly the signature B defined. It **reorders** candidates; it must never gate
      the pipeline and must never be required.
- [ ] **Test the no-torch path yourself** — in a scratch venv with torch uninstalled, confirm
      `localize.py` still returns a full answer. This is `PLAN.md` Rule 1 and it's worth more
      than the re-ranker's entire accuracy contribution.
- [ ] Measure accuracy **with and without** the re-ranker. If the delta is negative, say so and
      default it off. An honest negative result is fine; a silently harmful model is not.

---

# DAY 5 — Mon Aug 11  ⟶ GATE 2 DAY

## C5.1 · Reliability diagram + calibration (2.5 h)
🔒 **BLOCKED BY B5.1** (confidence outputs).

`PLAN.md` §7 names this one of the four things judges remember, and the brief explicitly asks
for failure-mode awareness. Calibration is the one claim in this deck that cannot be faked.
- [ ] Bin predictions by predicted confidence; plot predicted vs observed accuracy.
- [ ] Compute **expected calibration error (ECE)**.
- [ ] If the curve is far off the diagonal, tell B **today** — Gate 2 is tonight and after it
      nobody adds features.

## C5.2 · Full evaluation on the frozen set (2 h)
🔒 **BLOCKED BY A5.1** (frozen eval set, due 14:00). Do not run final numbers before it lands.
- [ ] Full run on `data/eval` (36 pairs). Full run of the baseline on the same set.
- [ ] Produce the Slide-6 table: accuracy at {1,2,5,10} px, broken out by `ambiguity_class`
      and by architecture style, alias-hit rate, median and worst-case ms per pair.
- [ ] **Pick the two hero images now:**
  - one clean **SUCCESS** case (brief asks for it by name)
  - one **HONEST FAILURE** case (brief asks for it by name) — pick a `degenerate` pair, show all
    tied candidates, show the centre rule being applied, show confidence correctly reported low.
    `PLAN.md`: *"Confident honesty reads as mastery. Claimed perfection reads as a bug."*

> ## 🚧 GATE 2 — end of Mon Aug 11
> ≥90% within 5 px on the `unique` subset; correct centre-rule behaviour on `degenerate`.
> If short — everyone stops adding features and debugs.

---

# DAY 6 — Tue Aug 12

## C6.1 · Remaining figures (3 h)
🔒 Needs `B4.1`'s decomposition panels and `A6.1`'s noise-ladder pairs.
- [ ] **The FFT figure** (Slide 5): two spectra side by side, lattice peaks marked, magnification
      read straight off the ratio of lattice vector lengths. It should look like physics.
- [ ] **The decomposition figure** (Slide 5): search → periodic component → aperiodic residual,
      with the true match visibly popping out of the residual. One image that explains the
      entire contribution.
- [ ] **Accuracy vs SNR curve** from A's noise ladder (Slide 6/7) — direct evidence of robustness
      under the higher noise their test set will have.
- [ ] **The augmentation stage strip** from A (Slide 4): clean layout → edge-brightened →
      drifted → noisy. This one image sells the 30% augmentation score.
- [ ] Baseline-vs-ours bar chart (Slides 3 and 5).

## C6.2 · Slide deck first full pass (3 h)
Use the i4C Idea Submission Template. Nine slides, fixed by the brief:

| # | Slide | Content | Source |
|---|---|---|---|
| 1 | Team Details | Team name, members, roles, college, contact | You |
| 2 | Problem Statement | Why navigation-error recovery matters — drift, thermal expansion, vibration, mechanical slack | You, from the Background text |
| 3 | Idea Description | Which architecture(s), which algorithm, **why better than template matching** | **B** |
| 4 | Proposed Solution | Generator design, SEM noise models, augmentation, localization method, pipeline diagram input-pair → (x,y), **citations inline** | **A + B** |
| 5 | Innovation | Closed-form 10× scale from the reciprocal lattice, aperiodic decomposition, ambiguity awareness | **B** |
| 6 | Results | Accuracy on your 30+ cases, time per pair, **success case + honest failure case** with ref/search/predicted/true visual | You |
| 7 | Tech & Feasibility | Python stack, RTX 4090 Laptop, dataset generation time, inference ms/pair, model size | You |
| 8 | GitHub & Video | Public repo link (**mandatory**), demo video link (optional but recommended) | You |
| 9 | References | Full citation list | **A** |

- [ ] ⚠️ **Slide 4 and Slide 9 must agree exactly.** The brief requires PPT citations to
      correspond to the repository citation document. Diff them literally.
- [ ] Chase A and B for their slide content **today**, not on Aug 14.

---

# DAY 7 — Wed Aug 13  ⟶ CODE FREEZE + THE TEST THAT MATTERS

## 🔴 C7.1 · FRESH-MACHINE CLONE-AND-RUN TEST (3 h) — do this first, at 09:00
This is where projects die. Budget the whole morning; you will find three things wrong.

- [ ] `uv pip freeze > requirements.txt` **from the clean venv** (mandatory repo item #6).
      Strip anything with a local file path in it.
- [ ] On a genuinely different machine, or at minimum a fresh directory + brand-new venv with
      **nothing** pre-installed:
  1. `git clone <public url>` — clone the **public HTTPS URL while logged out**, not your local folder
  2. create the venv, `pip install -r requirements.txt`
  3. follow **your own README, verbatim, without improvising** — if you have to think, the README
     is wrong; fix the README, don't fix it in your head
  4. `python generate_dataset.py --style dram --num 1 --out data/smoke --seed 0`
  5. `python localize.py data/smoke/<id>/reference.png data/smoke/<id>/search.png`
  6. confirm stdout is exactly one parseable `x,y` line
- [ ] **Then the adversarial pass** — this is how Applied Materials will actually run it:
  - [ ] **torch uninstalled entirely.** Must still work (Rule 1).
  - [ ] weights file deleted. Must still work.
  - [ ] run from a **different working directory** than the repo root
  - [ ] a path with a **space** in it
  - [ ] a JPG, a 16-bit TIFF, an RGBA PNG, a grayscale BMP
  - [ ] a corrupt image, a 1×1 image, a reference **larger** than the search image
  - [ ] every one of these must exit 0 and print a coordinate (Rule 2)
- [ ] Anything that breaks goes to A or B **immediately** — they're on call today for exactly this.
- ✅ Done when: you did it twice, the second time with zero manual intervention.

## C7.2 · Mandatory repo item audit (1 h)
Tick all seven off against the brief, in the GitHub web UI, logged out:
- [ ] 1 · `README.md` — complete setup, clone-to-run without contacting you
- [ ] 2 · Dataset generator — standalone, documented, accepts style / count / output dir, records ground truth
- [ ] 3 · Localization script — standalone, takes ref path + search path, outputs (x, y), no manual edits
- [ ] 4 · DL weights — `.pt`, committed, loaded automatically, < 5 MB
- [ ] 5 · Training script — reproduces training
- [ ] 6 · `requirements.txt` — full freeze
- [ ] 7 · `CITATIONS.md` — matches the PPT citations
- [ ] Repo is **public** (verify logged out, incognito)

## C7.3 · Failure analysis writeup (1.5 h)
- [ ] A short `docs/FAILURE-ANALYSIS.md`: where the system fails, why, and what would fix it.
      This feeds Slide 6's honest failure case and answers the brief's "failure mode awareness"
      requirement in writing as well as in the deck.

> ## 🚧 GATE 3 — end of Wed Aug 13 · CODE FREEZE
> **You and A enforce this on B, not the other way round.** After tonight: documentation and
> slides only. No algorithm changes.

---

# DAY 8 — Thu Aug 14

## C8.1 · Finish the deck (4 h)
- [ ] All nine slides complete with real numbers and real figures — no placeholders, no "TBD".
- [ ] Every number on Slides 6 and 7 traceable to `results.json`. If you can't source it, cut it.
- [ ] Slide 6 shows the honest failure case. Do not quietly drop it because it looks bad —
      the brief asks for it by name and it is a scored differentiator.
- [ ] Read Slides 4 and 9 side by side one final time; the citation lists must match.

## C8.2 · Demo video (2 h)
Optional per the brief, but explicitly recommended, and it costs two hours.
- [ ] 2–3 minutes: generate a pair → run `localize.py` → show the overlay of predicted vs true →
      show one ambiguous case and the honest low confidence.
- [ ] Terminal font large enough to read on a projector.
- [ ] Upload unlisted, **test the link in a private window**, put it on Slide 8.

---

# DAY 9 — Fri Aug 15 — SUBMISSION DAY (Aug 16 is buffer only)

## C9.1 · Final review (2 h)
- [ ] Fresh-machine test **one more time**, on the final commit hash. Things broke after Aug 13
      more often than anyone expects.
- [ ] Repo public, links in the deck live, video link live — all verified logged out.
- [ ] Export the deck to **PDF as well as PPT** (submission accepts PPT/PDF; PDF can't reflow
      your fonts on someone else's machine).
- [ ] Filenames and team name consistent across every artefact.

## C9.2 · Submit (1 h)
- [ ] Component 1 — PPT/PDF via the i4C template.
- [ ] Component 2 — public GitHub link.
- [ ] Screenshot the submission confirmation.
- [ ] **Submit on Aug 15. Aug 16 is buffer, not the plan.**

---

## Your dependency summary

| You are blocked by | What | When you're unblocked |
|---|---|---|
| **B** | `localize()` stub with the frozen return dict (`B0.3`) | **Aug 6 tonight** |
| **A** | Crude v0 pairs (`A1.1`) — harness has nothing to score without them | **Aug 7, 13:00** |
| B | Real pipeline wired in (`B2.4`) — accuracy chart moves off zero | Aug 8 EOD |
| A | Physics-grade pairs (`A3.3`) — realistic re-ranker training data | Aug 9, 18:00 |
| **A** | `alias_positions` (`A4.2`) — **the hard negatives; the whole re-ranker design needs this** | Aug 10 evening |
| B | `candidates` format + `rerank()` signature (`B4.3`) | Aug 10 EOD |
| B | Confidence outputs (`B5.1`) — reliability diagram | Aug 11 EOD |
| **A** | Frozen eval set (`A5.1`) — every final number | **Aug 11, 14:00** |
| A | `CITATIONS.md` final (`A5.3`) — Slides 4 & 9 | Aug 11 EOD |
| B | Decomposition + FFT figures (`B4.1`) | Aug 10–11 |
| A | Noise-ladder pairs (`A6.1`) — accuracy vs SNR curve | Aug 12 |
| B | Timing numbers (`B6.2`) | Aug 12 EOD |
| A + B | Slide content for 3, 4, 5, 9 | Aug 13–14 |

| You are blocking | What they need | Your deadline |
|---|---|---|
| A and B | Public repo + skeleton (`C0.3`, `C0.4`) | Aug 6 tonight |
| A and B | Working eval harness — their scoreboard (`C1.1`) | Aug 7 EOD |
| A and B | Baseline number — the bar to beat (`C1.2`) | Aug 7 EOD |
| B | Calibration feedback, in time for Gate 2 (`C5.1`) | Aug 11 EOD |
| A and B | Fresh-machine failures to fix (`C7.1`) | **Aug 13 morning** |

**Note on your Aug 7 morning:** you're blocked on A's data until 13:00. Build the harness
against hand-made fake pairs and B's stub — do not wait.
