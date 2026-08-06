# MEMBER A — Dataset, SEM Physics & Citations

**You own the 30% augmentation score.** That is the single largest named slice of the
rubric, and it is the one most teams will lose by writing `img + np.random.normal()`.

**Your files (nobody else edits these):**
```
driftsense/layouts.py
driftsense/sem_physics.py
generate_dataset.py
CITATIONS.md
data/            (you generate; C consumes)
```

**Your deliverables, in one line each:**
1. `generate_dataset.py` — mandatory repo item #2, must accept style/count/outdir.
2. A physically grounded SEM forward model, not additive Gaussian noise.
3. A **frozen 30+ pair official eval set** + a harder out-of-distribution set.
4. `CITATIONS.md` — 2–3 **personally DOI-verified** references per augmentation choice.

**Read before you start:** `TECH-SPEC.md` §2.1 (pair format), §4 (forward model), §6
(citation candidates — a starting list, *not* an approved one).

---

## Legend used in this file

- 🔒 **BLOCKED BY** — you cannot start until someone else ships something.
- 🚦 **BLOCKING** — someone is waiting on you. Ship on time or two people idle.
- ✅ **Done when** — the objective test. If you can't run the test, it isn't done.

---

# DAY 0 — Wed Aug 6 (evening, ~3 h)

## A0.1 · Environment (30 min)
- [ ] Install Python **3.12** (system default is 3.14 — torch has no 3.14 wheels).
- [ ] From repo root:
  ```powershell
  uv venv --python 3.12 .venv
  .venv\Scripts\activate
  uv pip install numpy scipy opencv-python pillow matplotlib scikit-image tqdm
  ```
- [ ] You do **not** need torch. Skip it. That's C's problem.
- ✅ Done when: `python -c "import cv2, numpy, scipy, skimage; print('ok')"` prints ok.

## A0.2 · Sit in the interface-freeze call with B and C (45 min)
🚦 **BLOCKING B AND C — this call must happen tonight.**

Agree and write down, verbatim, in `docs/INTERFACES.md`:
- [ ] Pair-on-disk format exactly as `TECH-SPEC.md` §2.1: `data/<split>/<pair_id>/reference.png`, `search.png`, `meta.json`.
- [ ] Every key name in `meta.json`. Argue about names **now**, never again.
- [ ] Coordinate convention — **state it explicitly**: `true_center_xy` is `[x, y]` in
      **search-image pixel coordinates**, origin at the **top-left corner of pixel (0,0)**,
      x → right, y → down, and the value is the **centre of the matched region**, sub-pixel float.
      Half the bugs in this project will come from someone assuming (row, col).
- [ ] Search image is exactly **1000 × 1000**. Reference is roughly **1000 × 1000** at high
      mag, appearing as a ~100 × 100 patch inside search.
- [ ] The `localize()` return dict (that one is B's to define; you just need to not break it).
- ✅ Done when: `docs/INTERFACES.md` is committed to `main` and all three have read it.

## A0.3 · Skeleton your three modules with importable stubs (45 min)
- [ ] `driftsense/layouts.py` — `render_dram(...)`, `render_finfet(...)` returning a float32 array.
- [ ] `driftsense/sem_physics.py` — `sem_forward(clean, params, rng)` returning uint8.
- [ ] `generate_dataset.py` — argparse skeleton with `--style {dram,finfet,both} --num --out --seed`.
- ✅ Done when: `python generate_dataset.py --style dram --num 1 --out data/smoke --seed 0`
  runs and writes a folder, even if the image is a black square.

## A0.4 · Commit and push
- [ ] Feature branch `a/dataset`, PR into `main`, one reviewer (B or C), merge.
- [ ] `.gitignore` must contain `data/` **except** one sample pair (repo requirement).

---

# DAY 1 — Thu Aug 7

## 🚦 A1.1 · CRUDE v0 GENERATOR — SHIP BY 13:00. HARD DEADLINE. (3 h)
**This is the most schedule-critical task any of you has this week.** B and C are both
idle until pairs exist on disk. Ugly is fine. Wrong physics is fine. *Late is not fine.*

- [ ] DRAM layout, no supersampling, no physics:
  - horizontal word-lines at pitch `p_w`, vertical bit-lines at pitch `p_b`, bright dot at each crossing
  - draw with numpy slicing or `cv2.line`, values in [0,1]
- [ ] Render the **search image directly at 1000×1000** with pitch expressed in *search* pixels.
      Render the **reference separately** at ~10× that pitch. **Do not build one huge canvas
      and downsample it** (see TECH-SPEC §4.1 — this is a Slide-4 talking point, and doing it
      the wrong way correlates the noise between the two images, violating the brief).
- [ ] Pick a random true centre, crop the reference region from the *same analytic layout*
      at high magnification.
- [ ] Independent noise: two separate `np.random.default_rng(seed_ref)` / `(seed_search)` streams.
      Gaussian is acceptable **for v0 only**.
- [ ] Write `meta.json` with **every key** from `TECH-SPEC.md` §2.1 present — fill unknowns with
      `null` or `[]`. B and C write code against key *names*; missing keys break them later.
- [ ] Generate `data/dev_v0/` with 30 pairs, seed 42. Commit **one** sample pair, gitignore the rest.
- [ ] Message B and C in the group chat: "v0 is on main, 30 pairs at data/dev_v0".
- ✅ Done when: B can run his prototype on your pairs and C's harness can read your `meta.json`.

## A1.2 · Verify your own ground truth before anyone trusts it (1 h)
Do this the same day. A ground-truth bug discovered on Aug 12 destroys three days of B's tuning.
- [ ] Write `tools/check_gt.py`: for each pair, downscale the reference by the true ratio,
      run plain `cv2.matchTemplate`, and check the argmax lands within 2 px of `true_center_xy`
      on a **noise-free** pair.
- [ ] Run on 30 clean pairs. Expect ≥ 28/30. If not, **your coordinate convention is wrong** —
      fix it now, off-by-half-template-size is the classic error.
- ✅ Done when: check passes and you have said so out loud to B and C.

## A1.3 · Start the citation hunt (1.5 h, runs in background all week)
- [ ] Create `CITATIONS.md` with one section per augmentation stage (the 10 rows of TECH-SPEC §4.2).
- [ ] For tonight, find and **open** the Cizmar 2008 ARTIMAGEN paper and the Timischl 2012
      noise paper. Paste the real DOI. Confirm the link resolves in a browser.
- ⚠️ **Rule: if you have not personally opened it, it does not go in the file.** A hallucinated
  DOI in front of Applied Materials engineers is worse than having one fewer citation.

---

# DAY 2 — Fri Aug 8

## A2.1 · FinFET layout (2 h)
- [ ] Dense parallel **vertical fin lines** at pitch `p_fin`, width `w_fin`.
- [ ] **One or two horizontal gate bars** crossing them (per the brief's table).
- [ ] Source/drain epi regions as slightly brighter blocks between gates.
- [ ] Fin-cut regions — short breaks in some fins. These are aperiodic content; keep the knob.
- ✅ Done when: `--style finfet` and `--style both` produce visually correct layouts, and
  a human can tell DRAM from FinFET at a glance.

## A2.2 · Supersampling + analytic geometry (1.5 h)
- [ ] Render at **4× supersample**, area-average down. This is your anti-aliasing.
- [ ] Produce a **material ID map** (metal / dielectric / contact) alongside the intensity —
      the physics model needs to know what material each pixel is.
- [ ] Compute a **signed distance to nearest edge** map `d(x)`. `cv2.distanceTransform` on the
      material boundary mask is the cheap way. You need this for edge brightening.
- ✅ Done when: zooming into a diagonal-ish edge shows smooth greyscale, not staircase pixels.

## A2.3 · SEM physics stages 3–5 (2.5 h)
Implement in `sem_physics.py`, **in this order** (order is physical, TECH-SPEC §4.2):
- [ ] **Stage 3 — SE yield with edge brightening:**
      `delta = delta_mat * (1 + k_edge * exp(-d / lambda_esc))`
      This is the single mandatory augmentation the brief names explicitly
      ("Apply edge-brightening to mimic real SEM behaviour"). Get it visibly right.
- [ ] **Stage 4 — area-downsample** the supersample to final pixel grid.
- [ ] **Stage 5 — beam PSF:** Gaussian core `sigma_beam` + small Lorentzian skirt (beam tail).
      Implement the skirt as a second wide Gaussian at ~5% weight if a true Lorentzian is fiddly.
- ✅ Done when: side-by-side of before/after shows bright rims on every feature edge. Save that
  image to `figures/` — **C needs it for Slide 4.**

---

# DAY 3 — Sat Aug 9  ⟶ GATE 1 DAY

## A3.1 · SEM physics stages 6–10 (3 h)
- [ ] **6 · Scan distortion:** low-order polynomial warp (thermal drift) + per-row x-jitter as an
      AR(1) process + a low-amplitude sinusoid (fab vibration). Use `cv2.remap`.
- [ ] **7 · Charging:** smooth low-frequency multiplicative field applied only to dielectric
      regions; with small probability, a bright horizontal streak.
- [ ] **8 · Shading:** low-order polynomial illumination field across the frame.
- [ ] **9 · Shot noise:** `signal = rng.poisson(dose * delta) / dose`. **Poisson, not Gaussian.**
      This is the dominant SEM noise source and saying so on Slide 4 is free marks.
- [ ] **10 · Detector:** gain, additive Gaussian read noise, saturation clip, 8-bit quantize.
- [ ] **Run stages 5–10 twice with independent RNG streams** — once for reference (high dose,
      low noise), once for search (lower dose per unit area, **higher noise**). The brief states
      their test search images are noisier; bake that asymmetry in.
- ✅ Done when: `meta.json` records the full `sem_params` dict used, and re-running with the same
  seed reproduces byte-identical PNGs.

## A3.2 · Inter-capture perturbations (1 h)
- [ ] Rotation between captures: ±3°.
- [ ] Magnification ratio sampled from `m ∈ [9.0, 11.0]` — **never exactly 10.0**. Record the
      true `m` in meta. B's whole Slide-5 innovation is measuring this rather than assuming it.
- [ ] Differential defocus (different `sigma_beam` per capture).
- [ ] Differential drift warp.

## 🚦 A3.3 · Ship the physics generator to main — BY 18:00 (30 min)
🚦 **BLOCKING C:** C cannot train the re-ranker on v0 toy data; it needs realistic pairs.
- [ ] Regenerate `data/dev_v1/` — 60 pairs, `--style both`.
- [ ] Announce in chat.

> ## 🚧 GATE 1 — end of Sat Aug 9
> **Whole team:** `generate_dataset.py` → `localize.py` → `evaluate.py` runs end-to-end on
> 30 pairs and prints an accuracy number. Any number. If this doesn't happen tonight, the team
> **cuts the CNN entirely** and C moves onto evaluation and docs full time.
> **Your part of the gate:** 60 physics-grade pairs on disk with valid meta. Nothing else.

---

# DAY 4 — Sun Aug 10

## A4.1 · Ambiguity control — the knob that makes the project honest (3 h)
This is what lets the team report accuracy *by difficulty class* instead of one blended,
meaningless number. It is also the reproduction of the pathological case the brief guarantees
is in their test set.

- [ ] Add `aperiodic_content_level ∈ [0, 1]` controlling:
  - array boundary / edge-of-array visible in frame
  - periphery block (non-array circuitry) in one corner
  - dummy fill patterns
  - defects: missing contacts, bridged lines, particles
- [ ] At level 0 the pair is **genuinely degenerate** — only the centre rule can answer it.
      At level 1 it is uniquely solvable. Generate the whole range.
- [ ] Label each pair `ambiguity_class ∈ {unique, weakly_ambiguous, degenerate}` using an
      objective criterion (e.g. aperiodic energy fraction of the search image), not a vibe.
- ✅ Done when: you can point at one pair and say "no algorithm on earth can solve this one,
  and here is why" — that pair is Slide 6's honest failure case.

## 🚦 A4.2 · Alias positions in meta — B AND C BOTH NEED THIS (2 h)
🚦 **BLOCKING B's ambiguity reporting and C's hard-negative training. Ship by Sun evening.**
- [ ] For each pair, compute **all lattice-equivalent positions** of the true centre that fall
      inside the search image: `true_center + i*a1 + j*a2` for integer i, j, where `(a1, a2)` is
      the real-space lattice basis you generated with. You know these exactly — you drew them.
- [ ] Write them to `alias_positions` in `meta.json`.
- [ ] Also record `lattice_period_search_px`.
- ✅ Done when: C can pull hard negatives straight from `meta["alias_positions"]` without
  running any detection of his own. **Tell C the moment this lands.**

---

# DAY 5 — Mon Aug 11  ⟶ GATE 2 DAY

## 🚦 A5.1 · FREEZE THE OFFICIAL EVAL SET — BY 14:00 (1.5 h)
🚦 **BLOCKING C's final numbers, every figure, and Slide 6.** After this, the set never changes.
- [ ] `python generate_dataset.py --style both --num 36 --out data/eval --seed 1337`
- [ ] Composition, deliberately balanced:
  - 18 DRAM / 18 FinFET
  - ~12 `unique`, ~16 `weakly_ambiguous`, ~8 `degenerate`
  - at least 2 pairs that are hopeless by construction (aperiodic level 0)
- [ ] **Tag the commit** `eval-set-frozen`. Announce: "eval set frozen, do not regenerate."
- [ ] Commit **one** sample pair into the repo (repo requirement); gitignore the rest.

## A5.2 · Out-of-distribution set (1.5 h)
The brief says their test set is noisier than yours. Tuning to your own noise level is the
single most likely way this team loses.
- [ ] `data/ood/` — 30 pairs with a generator config **never used for tuning**:
      unseen pitches, 2–3× the noise, rotation up to ±6°, `m ∈ [8, 13]`.
- [ ] Do not let B or C tune against this. It is a one-shot honesty test at the end.

## A5.3 · CITATIONS.md — complete and verified (3 h)
- [ ] Every one of the 10 forward-model stages gets **2–3 references** and, critically,
      **one sentence naming the physical mechanism it models** — not just "noise exists".
- [ ] Every DOI opened by you, in a browser, today.
- [ ] ⚠️ **Trap, from TECH-SPEC §3.5:** do **not** cite Moisan (2011) "Periodic plus smooth image
      decomposition" for the lattice decomposition. Its "periodic" means periodic *boundary
      extension*, a completely different thing. An Applied Materials reviewer will know this.
      Cite standard reciprocal-lattice / Fourier crystallography treatment instead.
- [ ] Structure the file so C can paste it into Slide 9 unmodified. Slide 4 and Slide 9 must
      agree exactly — the brief requires PPT citations to match the repo citation document.

> ## 🚧 GATE 2 — end of Mon Aug 11
> **Team target:** ≥90% within 5 px on the `unique` subset, correct centre-rule behaviour on
> the `degenerate` subset. If short: everyone stops adding features and debugs.
> **Your part:** frozen eval set + OOD set + verified citations. After tonight your generator
> is feature-complete.

---

# DAY 6 — Tue Aug 12

## A6.1 · Noise-ladder robustness sweep (2.5 h)
- [ ] Generate 5 versions of the same 20 pairs at increasing noise (dose ÷ 1, 2, 4, 8, 16).
      **Same seeds for geometry**, different noise seeds — so the only variable is SNR.
- [ ] Hand to C to run through `evaluate.py`.
- ✅ Deliverable: an **accuracy vs SNR curve**. This is the direct evidence for "robust under
  the higher noise their test set will have". Slide 6 or 7.

## A6.2 · Generation-time benchmark (30 min)
- [ ] Time 30-pair generation, record seconds/pair. C needs the number for Slide 7.

## A6.3 · Docstrings and generator documentation (2 h)
Repo requirement #2 says "**a documented** Python script".
- [ ] Module docstring in `generate_dataset.py` explaining the forward model in ~15 lines.
- [ ] Every physics stage function: what it models, what the parameter ranges are, which
      citation justifies it (`# see CITATIONS.md §Shot noise`).
- [ ] `--help` output that a stranger can act on.

---

# DAY 7 — Wed Aug 13  ⟶ CODE FREEZE

## A7.1 · Final generator hardening (2 h)
- [ ] `--style both` verified. `--num 1` verified. `--seed` reproducibility verified twice.
- [ ] Output dir created if missing; existing dir not silently overwritten without a warning.
- [ ] No absolute paths, no `C:\Users\prana\...` anywhere. Test from a different folder.

## A7.2 · Support C's fresh-machine test (1 h)
🔒 **BLOCKED BY C4.x** — C runs the clone-and-run; you fix whatever breaks on your side.
- [ ] Be available. Fix generator import errors immediately.

## A7.3 · Slide 9 content + Slide 4 citation inserts (2 h)
- [ ] Hand C a ready-to-paste reference list and the per-augmentation citation callouts for Slide 4.
- [ ] Hand C your figures: clean layout → edge-brightened → drifted → noisy, as a stage strip.
      This one image sells the entire 30% augmentation score.

> ## 🚧 GATE 3 — end of Wed Aug 13 · CODE FREEZE
> No algorithm or generator changes after tonight. Docs and slides only.
> Every hackathon loses a team to a "tiny improvement" pushed at 2 a.m. on deadline day.

---

# DAYS 8–9 — Thu Aug 14 / Fri Aug 15

## A8.1 · Support C on slides (2 h)
- [ ] Review Slides 4 and 9 for factual errors. You are the only person who can catch a wrong
      physics claim.
- [ ] Be ready to answer, in the video or a Q&A: "why Poisson and not Gaussian?", "what is
      edge brightening?", "why render each magnification separately?"

## A8.2 · Final repo review (1 h)
- [ ] `CITATIONS.md` renders correctly on GitHub. Every link clicked once more.
- [ ] The one committed sample pair is present and viewable in the GitHub web UI.

**Aug 16 is buffer. Target done on Aug 15.**

---

## Your dependency summary — who is waiting on whom

| You are blocked by | What | When |
|---|---|---|
| All three | Interface freeze in `docs/INTERFACES.md` | Aug 6 night |
| C | Fresh-machine test results | Aug 13 |
| *(nothing else)* | You are the least blocked person on the team. | |

| You are blocking | What they need | Your deadline |
|---|---|---|
| **B and C** | Crude v0 pairs on disk (`A1.1`) | **Aug 7, 13:00 — hardest deadline in the project** |
| C | Physics-grade pairs for re-ranker training (`A3.3`) | Aug 9, 18:00 |
| **B and C** | `alias_positions` + `ambiguity_class` in meta (`A4.2`) | Aug 10 evening |
| C | Frozen official eval set (`A5.1`) | **Aug 11, 14:00** |
| C | `CITATIONS.md` final, for Slides 4 & 9 (`A5.3`) | Aug 11 EOD |
| C | Noise-ladder pairs (`A6.1`) | Aug 12 |

**If you slip only one deadline all week, do not let it be A1.1 or A5.1.**
