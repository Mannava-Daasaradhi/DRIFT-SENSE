#!/usr/bin/env python
"""Train the Siamese re-ranker — mandatory repo item #5. Member C.

    python train.py --data data/train --epochs 25 --out weights/reranker.pt

Reproduces the checkpoint in `weights/reranker.pt` from generated pairs alone.
Fixed seed, no hidden state, no manual steps.

The training set is built from ground truth, not from the localizer
--------------------------------------------------------------------
Every example comes out of `meta.json`:

* **positive**       — the patch at `true_center_xy`.
* **hard negative**  — patches at `alias_positions`, the lattice-equivalent
                       sites. These look almost identical to the positive, which
                       is the entire point: the only way to separate them is the
                       aperiodic content, so that is what the network is forced
                       to learn.
* **easy negative**  — a random offset far from any lattice site, so the model
                       still learns the trivial rejection cheaply.

Pairs whose `ambiguity_class` is `degenerate` are **excluded**. On those there is
genuinely nothing to learn — every site is identical by construction — and
including them would teach the network to emit confident scores from noise. That
is a deliberate scope limit and is stated on Slide 5.

Because the re-ranker only ever reorders (`INTERFACES.md` §2), a mediocre model
costs a little accuracy and never correctness. `PLAN.md` Gate 1 allows cutting it
entirely; the classical core stands alone.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from driftsense.preprocess import load_gray, preprocess       # noqa: E402
from driftsense.rerank import PATCH, build_model, embed_patch  # noqa: E402


def _crop(img: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray | None:
    """Centre crop of side `size`, or None if it falls outside the image."""
    h, w = img.shape[:2]
    x0, y0 = int(round(cx - size / 2.0)), int(round(cy - size / 2.0))
    if x0 < 0 or y0 < 0 or x0 + size > w or y0 + size > h or size < 8:
        return None
    return np.ascontiguousarray(img[y0:y0 + size, x0:x0 + size])


def build_dataset(data_dir: str, max_alias: int = 6, rng: random.Random | None = None):
    """(template, candidate, label) triples from every usable pair.

    Returns lists of float32 PATCH x PATCH arrays, already standardized.
    """
    rng = rng or random.Random(0)
    dirs = sorted(d for d in glob.glob(os.path.join(data_dir, "*"))
                  if os.path.isfile(os.path.join(d, "meta.json")))
    if not dirs:
        raise SystemExit(f"no pairs under {data_dir!r} — generate a training split first:\n"
                         f"  python generate_dataset.py --style both --num 200 "
                         f"--out {data_dir} --seed 1")

    T, C, Y, groups = [], [], [], []
    skipped_degenerate = 0

    for gi, d in enumerate(dirs):
        with open(os.path.join(d, "meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)

        if (meta.get("ambiguity_class") or "") == "degenerate":
            skipped_degenerate += 1
            continue

        m = float(meta.get("magnification_ratio") or 0.0)
        if m <= 1e-6:
            continue

        ref = preprocess(load_gray(os.path.join(d, "reference.png")))
        search = preprocess(load_gray(os.path.join(d, "search.png")))

        # Template exactly as `matching.build_template` makes it: INTER_AREA
        # downscale of the normalized reference, then the same band split. If
        # training patches are prepared differently from inference patches, the
        # network learns a preprocessing artefact instead of the structure.
        rh, rw = ref.img.shape[:2]
        th, tw = max(8, int(round(rh / m))), max(8, int(round(rw / m)))
        small = cv2.resize(ref.img, (tw, th), interpolation=cv2.INTER_AREA)
        tpl = preprocess(small).hp
        size = int(min(tpl.shape[:2]))
        if size < 12:
            continue
        tpl_p = embed_patch(tpl)

        tx, ty = meta["true_center_xy"]

        # Jitter the positive. At inference the candidate patch is cropped at a
        # correlation peak refined to sub-pixel and then rounded, under a
        # template built from an *estimated* magnification — so the true patch
        # arrives a pixel or two off centre and slightly rescaled. Training only
        # on exactly-centred crops taught the network to key on that perfect
        # alignment: it reached 0.99 validation AUC and still changed the final
        # answer on none of 36 eval pairs, because no inference patch ever
        # looked like a training positive.
        n_pos = 0
        for jx, jy in [(0.0, 0.0)] + [(rng.uniform(-2.5, 2.5), rng.uniform(-2.5, 2.5))
                                      for _ in range(2)]:
            pos = _crop(search.hp, tx + jx, ty + jy, size)
            if pos is None:
                continue
            T.append(tpl_p); C.append(embed_patch(pos)); Y.append(1.0); groups.append(gi)
            n_pos += 1
        if n_pos == 0:
            continue

        aliases = list(meta.get("alias_positions") or [])
        rng.shuffle(aliases)
        n_hard = 0
        for ax, ay in aliases:
            if np.hypot(ax - tx, ay - ty) < size * 0.5:
                continue                     # overlaps the positive; not a negative
            # Same jitter on the hard negatives, so the network cannot separate
            # positive from alias by alignment sharpness alone.
            neg = _crop(search.hp, ax + rng.uniform(-2.5, 2.5),
                        ay + rng.uniform(-2.5, 2.5), size)
            if neg is None:
                continue
            T.append(tpl_p); C.append(embed_patch(neg)); Y.append(0.0); groups.append(gi)
            n_hard += 1
            if n_hard >= max_alias:
                break

        # A couple of easy negatives so the model does not see only near-misses.
        sh, sw = search.hp.shape[:2]
        for _ in range(2):
            for _try in range(20):
                rx, ry = rng.uniform(0, sw), rng.uniform(0, sh)
                if np.hypot(rx - tx, ry - ty) < size * 1.5:
                    continue
                neg = _crop(search.hp, rx, ry, size)
                if neg is not None:
                    T.append(tpl_p); C.append(embed_patch(neg))
                    Y.append(0.0); groups.append(gi)
                    break

    if not T:
        raise SystemExit("no usable training examples were built — check that "
                         "meta.json has true_center_xy and alias_positions.")
    print(f"built {len(T)} examples from {len(set(groups))} pairs "
          f"({int(sum(Y))} positive, {len(Y) - int(sum(Y))} negative); "
          f"skipped {skipped_degenerate} degenerate pairs")
    return (np.stack(T)[:, None].astype(np.float32),
            np.stack(C)[:, None].astype(np.float32),
            np.asarray(Y, dtype=np.float32),
            np.asarray(groups, dtype=np.int64))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train the DRIFT-SENSE Siamese re-ranker.")
    p.add_argument("--data", default="data/train", help="training pairs directory")
    p.add_argument("--out", default="weights/reranker.pt", help="checkpoint path")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    a = p.parse_args(argv)

    try:
        import torch
    except ImportError:
        print("torch is not installed. The re-ranker is OPTIONAL — localize.py runs "
              "without it (PLAN.md Rule 1). Install torch only if you want to "
              "reproduce the checkpoint.", file=sys.stderr)
        return 1

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {dev}")

    T, C, Y, G = build_dataset(a.data, rng=random.Random(a.seed))

    # Split by PAIR, never by example. Patches from one pair share a template and
    # overlap heavily; splitting by example would leak the answer across the
    # split and report a validation AUC that means nothing.
    pairs = np.unique(G)
    rng = np.random.default_rng(a.seed)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * a.val_frac))
    val_pairs = set(pairs[:n_val].tolist())
    vm = np.array([g in val_pairs for g in G])
    print(f"train {int((~vm).sum())} examples / {len(pairs) - n_val} pairs   |   "
          f"val {int(vm.sum())} examples / {n_val} pairs")

    to = lambda x: torch.from_numpy(x).to(dev)
    Tt, Ct, Yt = to(T[~vm]), to(C[~vm]), to(Y[~vm])
    Tv, Cv, Yv = to(T[vm]), to(C[vm]), to(Y[vm])

    model = build_model().to(dev)
    n_par = sum(q.numel() for q in model.parameters())
    print(f"parameters: {n_par:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(a.epochs, 1))
    # Positives are outnumbered ~7:1 by negatives; without the weight the model
    # scores everything negative and still looks accurate.
    pos_w = torch.tensor([max((Yt == 0).sum().item(), 1) / max((Yt == 1).sum().item(), 1)],
                         device=dev)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)

    best_auc, best_state = -1.0, None
    n = Tt.shape[0]
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, a.batch):
            idx = perm[i:i + a.batch]
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(Tt[idx], Ct[idx]), Yt[idx])
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()

        model.eval()
        with torch.no_grad():
            lv = model(Tv, Cv).cpu().numpy()
        auc = _auc(Yv.cpu().numpy(), lv)
        print(f"epoch {ep:3d}/{a.epochs}   loss {tot / max(n, 1):.4f}   val AUC {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    torch.save(best_state or model.state_dict(), a.out)
    mb = os.path.getsize(a.out) / 1e6
    print(f"\nbest val AUC {best_auc:.4f}   ->  {a.out}  ({mb:.2f} MB)")
    if mb > 5.0:
        print("[warn] checkpoint exceeds the 5 MB budget in TECH-SPEC §3.6",
              file=sys.stderr)
    return 0


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    """ROC AUC via the rank identity — no sklearn dependency."""
    y = np.asarray(y).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


if __name__ == "__main__":
    sys.exit(main())
