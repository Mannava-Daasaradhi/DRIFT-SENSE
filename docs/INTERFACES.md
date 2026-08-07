# DRIFT-SENSE — Frozen Interfaces

> Status: drafted by Member A from `TECH-SPEC.md` / `PLAN.md` (Aug 7, ahead of the
> Day-0 live freeze call not having happened yet). **B and C: read this and reply
> with any objection before you write code against it.** Once nobody objects,
> treat it as frozen — nobody changes it unilaterally after that (`PLAN.md` §3).

---

## 1. Pair-on-disk format (A produces, B and C consume)

```
data/<split>/<pair_id>/
    reference.png     # high magnification capture
    search.png        # 1000 x 1000, ~10x lower magnification
    meta.json
```

- `search.png` is **exactly 1000 × 1000**.
- `reference.png` is roughly 1000×1000 at high magnification; the pattern it
  depicts appears as a **~100×100 patch** inside `search.png` (ratio = magnification).

## 2. Coordinate convention — stated explicitly so nobody assumes `(row, col)`

- All coordinates are `(x, y)`, **x → right, y → down**, origin at the
  **top-left corner of pixel (0,0)**.
- A coordinate is always the **centre** of the matched region, as a **sub-pixel float**.
- This applies to `true_center_xy`, `alias_positions`, and the `x, y` returned by `localize()`.

## 3. `meta.json` schema

```jsonc
{
  "pair_id": "dram_00017",
  "style": "dram",                          // "dram" | "finfet"
  "true_center_xy": [412.37, 688.02],       // sub-pixel, SEARCH image coords, see §2
  "magnification_ratio": 9.83,              // measured, never assumed 10.0
  "rotation_deg": 1.42,
  "lattice_period_search_px": [7.4, 9.1],
  "alias_positions": [[404.9, 688.0]],      // lattice-equivalent sites, filled by A4.2
  "ambiguity_class": "unique",              // "unique" | "weakly_ambiguous" | "degenerate" | null until A4.1
  "aperiodic_content": ["array_boundary", "defect"],
  "sem_params": { "...": "..." },
  "seeds": { "reference": 12345, "search": 67890 }
}
```

Unknown fields (before the corresponding generator task lands) are filled with
`null` or `[]`, never omitted — B and C write code against key *names*, and a
missing key breaks them later, a wrong-typed value does not.

## 4. Localizer API (B produces, A and C consume)

```python
def localize(reference_path: str, search_path: str,
             use_reranker: bool = True) -> dict:
    """Never raises. Returns:
    {
        "x": float, "y": float,          # centre, SEARCH image pixels, see §2
        "confidence": float,             # [0, 1], calibrated
        "pai": float,                    # Periodic Ambiguity Index
        "candidates": [                  # ranked, for diagnostics/figures
            {"x": float, "y": float, "score": float, "rank": int}, ...
        ],
        "scale": float, "rotation": float,
        "decision": "unique" | "tie_broken_by_center" | "fallback",
        "time_ms": float,
    }
    """
```

## 5. `localize.py` CLI contract

Both invocation styles must work:

```bash
python localize.py reference.png search.png
python localize.py --ref reference.png --search search.png
python localize.py --ref r.png --search s.png --json      # full diagnostics
```

Default stdout is **exactly one line, nothing else**: `412.4,688.0`.
All logging goes to **stderr** — an automated grader most likely parses stdout.

## 6. `generate_dataset.py` CLI contract

```bash
python generate_dataset.py --style {dram,finfet,both} --num <int> --out <dir> --seed <int>
```

## 7. Non-negotiable engineering rules (PLAN.md §2 — repeated here for visibility)

1. `localize.py` runs with **zero ML deps installed**; torch optional.
2. `localize.py` **never raises** — top-level try/except returns the search-image
   centre with `confidence=0` as a last resort.
3. **Never hardcode `10`** for magnification — always measured.
4. Never hardcode image size, dtype, or channel count.
5. Accept both CLI conventions (§5).
6. Handle both DRAM and FinFET.
7. Validate on data harder than what you tuned on.
8. Fresh-machine test before submission.

---

**Open items pending the actual 3-way call:** none known yet — this file mirrors
`TECH-SPEC.md` §2 exactly. If B or C want to change a key name or the dict shape,
raise it now; every downstream file is about to be written against this.
