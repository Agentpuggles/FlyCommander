#!/usr/bin/env python3
"""FlyCommander — train the card embedder (and friends) on synthetic captures.

What this produces, all under ``vision/weights/``:

* ``embedder.pt``    — the CNN embedder (best accuracy; ~2 ms/card CPU, <1 ms GPU)
* ``embedder.onnx``  — the same network for torch-free inference (onnxruntime)
                       + ``embedder.json`` describing input size and dim
* ``dense_head.npz`` — a linear projection of the torch-free dense descriptor
                       (works even with no torch/onnxruntime installed)
* ``cardness.npz``   — a logistic "is this rectified crop a card?" head used
                       by the detector to sharpen confidences

Why synthetic training is the right move for this project: MTG card imagery
is copyrighted, there is no overhead-camera dataset of a stranger's kitchen
table, and manual annotation is exactly the "make the user register cards"
work the project forbids. The augmentation pipeline in ``vision.synthetic``
models the physical conditions the matcher must survive (sleeves, foil, glare,
occlusion, rotation, blur, exposure, background bleed) and the labels are free.

Usage
-----
    python scripts/train_embedder.py --cards 1500 --steps 4000 --batch 64
    python scripts/train_embedder.py --quick          # smoke test / CI
    python scripts/train_embedder.py --device cuda --cards 4000 --steps 12000

The script is resumable: ``--resume`` continues from ``embedder.pt``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vision import rectify as R                                     # noqa: E402
from vision.embeddings import (CardEmbedNet, DENSE_HEAD_WEIGHTS,    # noqa: E402
                               EMBED_SIZE, FEATURE_DIM, CNN_TORCH_WEIGHTS,
                               CNN_ONNX_WEIGHTS, dense_features,
                               enhance, normalize_capture)
from vision.synthetic import (CardSpec, augmented_capture,           # noqa: E402
                              background_crop, default_library)

WEIGHTS_DIR = REPO_ROOT / "vision" / "weights"


# ---------------------------------------------------------------------------
# data pipeline
# ---------------------------------------------------------------------------
class CaptureBatcher:
    """Generates ``(images, labels)`` batches of augmented captures.

    Held-out cards: cards with index >= n_train are never trained on, so the
    validation retrieval numbers measure generalisation to unseen cards rather
    than memorisation.
    """

    def __init__(self, specs: list[CardSpec], n_train: int,
                 batch: int = 64, views: int = 2, hard_frac: float = 0.5,
                 workers: int = 4, size: tuple[int, int] = EMBED_SIZE,
                 seed: int = 0) -> None:
        self.specs = specs
        self.n_train = n_train
        self.batch_size = batch
        self.views = views
        self.hard_frac = hard_frac
        self.size = size
        self.workers = max(1, workers)
        self.rng = np.random.default_rng(seed)
        self._pool = ThreadPoolExecutor(max_workers=self.workers) if workers > 1 else None

    def _make(self, spec: CardSpec, hard: bool) -> np.ndarray:
        rng = np.random.default_rng(int(self.rng.integers(0, 2**31 - 1)))
        img = augmented_capture(spec, rng, hard=hard, size=self.size)
        img, _ = normalize_capture(img, None, size=self.size)
        img = enhance(img)
        rgb = img[:, :, ::-1].astype(np.float32) / 255.0      # BGR → RGB
        return ((rgb - 0.5) / 0.25).transpose(2, 0, 1)

    def next_batch(self) -> tuple[np.ndarray, np.ndarray]:
        n_classes = max(2, self.batch_size // self.views)
        idx = self.rng.choice(self.n_train, size=n_classes, replace=False)
        jobs = []
        for cls in idx:
            spec = self.specs[int(cls)]
            for _ in range(self.views):
                hard = bool(self.rng.random() < self.hard_frac)
                jobs.append((spec, hard, int(cls)))
        if self._pool is not None:
            images = list(self._pool.map(lambda j: self._make(j[0], j[1]), jobs))
        else:
            images = [self._make(j[0], j[1]) for j in jobs]
        labels = np.array([j[2] for j in jobs], np.int64)
        return np.stack(images).astype(np.float32), labels

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------
def supervised_contrastive(emb, labels, temperature: float = 0.07,
                           label_smoothing: float = 0.05):
    """SupCon (Khosla et al. 2020) with label smoothing.

    Label smoothing matters here: two *different* cards can be visually
    near-identical (same artist style, reprints, basic lands), and a hard
    negative pushes them apart — smoothing keeps those cases from wrecking
    the embedding space.
    """
    import torch
    import torch.nn.functional as F

    device = emb.device
    n = emb.shape[0]
    sim = emb @ emb.t() / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp_sim = sim.exp()
    log_prob = sim - exp_sim.sum(dim=1, keepdim=True).log()

    same = (labels[:, None] == labels[None, :]).float().to(device)
    same.fill_diagonal_(0.0)
    positive_counts = same.sum(dim=1).clamp(min=1.0)

    # smooth: true positives weighted 1-ε, everything else ε/(N-1)
    if label_smoothing > 0 and n > 1:
        weight = same * (1.0 - label_smoothing) + (1.0 - same) * \
            label_smoothing / float(n - 1)
    else:
        weight = same
    loss = -(weight * log_prob).sum(dim=1) / positive_counts
    return loss.mean()


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def embed_library(model, specs, device, size, batch: int = 64,
                  clean: bool = True) -> np.ndarray:
    """Embed the reference library (clean renders, like a Scryfall scan)."""
    import torch

    from vision.synthetic import render_card

    out = []
    with torch.no_grad():
        for i in range(0, len(specs), batch):
            chunk = specs[i:i + batch]
            cards = [render_card(s, size) for s in chunk]
            tensors = []
            for card in cards:
                card, _ = normalize_capture(card, None, size=size)
                card = enhance(card)
                rgb = card[:, :, ::-1].astype(np.float32) / 255.0
                tensors.append(((rgb - 0.5) / 0.25).transpose(2, 0, 1))
            x = torch.from_numpy(np.stack(tensors).astype(np.float32)).to(device)
            out.append(torch.nn.functional.normalize(model(x), dim=1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def evaluate(model, specs: list[CardSpec], device, size, *, n_val: int = 120,
             hard: bool = True, batch: int = 32, seed: int = 1234,
             pool: ThreadPoolExecutor | None = None) -> dict:
    """Top-1/top-5 retrieval for unseen cards under hard conditions."""
    import torch

    rng = np.random.default_rng(seed)
    lib_vecs = embed_library(model, specs, device, size)
    picks = rng.choice(len(specs), size=min(n_val, len(specs)), replace=False)

    def _capture(spec: CardSpec, i: int) -> np.ndarray:
        r = np.random.default_rng(seed + 1000 + i)
        img = augmented_capture(spec, r, hard=hard, size=size)
        img, _ = normalize_capture(img, None, size=size)
        img = enhance(img)
        rgb = img[:, :, ::-1].astype(np.float32) / 255.0
        return ((rgb - 0.5) / 0.25).transpose(2, 0, 1)

    jobs = [(specs[int(i)], k) for k, i in enumerate(picks)]
    if pool is not None:
        images = list(pool.map(lambda j: _capture(j[0], j[1]), jobs))
    else:
        images = [_capture(j[0], j[1]) for j in jobs]

    top1 = top3 = top5 = 0
    ranks: list[int] = []
    with torch.no_grad():
        for k in range(0, len(images), batch):
            chunk = np.stack(images[k:k + batch])
            x = torch.from_numpy(chunk).to(device)
            emb = torch.nn.functional.normalize(model(x), dim=1)
            sims = (emb @ torch.from_numpy(lib_vecs).to(device).t()).cpu().numpy()
            # a capture can be any quarter-turn → best over the four readings
            for row, pick in enumerate(picks[k:k + batch]):
                order = np.argsort(-sims[row])
                rank = int(np.where(order == pick)[0][0]) + 1
                ranks.append(rank)
                top1 += rank <= 1
                top3 += rank <= 3
                top5 += rank <= 5
    n = max(1, len(ranks))
    return {"top1": top1 / n, "top3": top3 / n, "top5": top5 / n,
            "medianRank": float(np.median(ranks)) if ranks else -1,
            "n": len(ranks), "hard": hard}


# ---------------------------------------------------------------------------
# auxiliary heads (torch-free tiers)
# ---------------------------------------------------------------------------
def fit_dense_head(specs: list[CardSpec], library_vecs: np.ndarray,
                   n_samples: int = 2500, hard_frac: float = 0.5,
                   ridge: float = 1.0, seed: int = 7,
                   workers: int = 4) -> dict:
    """Ridge map from the dense descriptor to the CNN embedding space."""
    rng = np.random.default_rng(seed)
    pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None

    def _one(i: int) -> tuple[np.ndarray, int]:
        spec_i = int(rng.integers(0, len(specs)))
        r = np.random.default_rng(seed + i)
        img = augmented_capture(specs[spec_i], r, hard=bool(r.random() < hard_frac))
        return dense_features(img), spec_i

    jobs = range(n_samples)
    results = list(pool.map(_one, jobs)) if pool else [_one(i) for i in jobs]
    if pool:
        pool.shutdown(wait=False)
    X = np.stack([r[0] for r in results]).astype(np.float32)
    Y = library_vecs[[r[1] for r in results]].astype(np.float32) \
        if library_vecs.shape[0] == len(specs) else None
    if Y is None:
        raise ValueError("library vectors must align with specs")
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    Xn = (X - mu) / sigma
    XtX = Xn.T @ Xn + ridge * np.eye(Xn.shape[1], dtype=np.float32)
    W = np.linalg.solve(XtX, Xn.T @ Y).astype(np.float32)
    b = (Y - Xn @ W).mean(axis=0).astype(np.float32)
    return {"w": W, "b": b, "mu": mu.astype(np.float32),
            "sigma": sigma.astype(np.float32)}


def fit_cardness(specs: list[CardSpec], n_pos: int = 1500, n_neg: int = 1500,
                 seed: int = 11, workers: int = 4, max_iter: int = 200) -> dict:
    """Logistic regression: card crop vs background crop (detector evidence)."""
    rng = np.random.default_rng(seed)
    pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None

    def _pos(i: int) -> np.ndarray:
        r = np.random.default_rng(seed + 5000 + i)
        spec = specs[int(r.integers(0, len(specs)))]
        return dense_features(augmented_capture(spec, r, hard=bool(r.random() < 0.6)))

    def _neg(i: int) -> np.ndarray:
        r = np.random.default_rng(seed + 9000 + i)
        return dense_features(background_crop(r))

    pos = list(pool.map(_pos, range(n_pos))) if pool else [_pos(i) for i in range(n_pos)]
    neg = list(pool.map(_neg, range(n_neg))) if pool else [_neg(i) for i in range(n_neg)]
    if pool:
        pool.shutdown(wait=False)
    X = np.stack(pos + neg).astype(np.float64)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    Xn = (X - mu) / sigma

    w = np.zeros(Xn.shape[1])
    b = 0.0
    lr, l2 = 0.35, 1e-3
    for _ in range(max_iter):
        z = Xn @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad_w = Xn.T @ (p - y) / len(y) + l2 * w
        grad_b = float((p - y).mean())
        w -= lr * grad_w
        b -= lr * grad_b
    return {"w": w.astype(np.float32), "b": np.float32(b),
            "mu": mu.astype(np.float32), "sigma": sigma.astype(np.float32)}


def export_onnx(model, path: Path, size: tuple[int, int], dim: int) -> bool:
    """Export the trained CNN to ONNX and verify it against torch."""
    try:
        import torch
        model_cpu = model.to("cpu").eval()
        dummy = torch.zeros(1, 3, size[1], size[0])
        try:   # torch >= 2.6 defaults to the dynamo exporter (needs onnxscript)
            torch.onnx.export(model_cpu, dummy, str(path),
                              input_names=["card"], output_names=["embedding"],
                              dynamic_axes={"card": {0: "batch"},
                                            "embedding": {0: "batch"}},
                              opset_version=17, dynamo=False)
        except TypeError:
            torch.onnx.export(model_cpu, dummy, str(path),
                              input_names=["card"], output_names=["embedding"],
                              dynamic_axes={"card": {0: "batch"},
                                            "embedding": {0: "batch"}},
                              opset_version=17)
        meta = {"dim": dim, "size": list(size), "inputName": "card",
                "outputName": "embedding"}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
        import onnxruntime as ort
        sess = ort.InferenceSession(str(path),
                                    providers=["CPUExecutionProvider"])
        with torch.no_grad():
            ref = model_cpu(dummy).numpy()
        got = sess.run(None, {"card": dummy.numpy()})[0]
        err = float(np.abs(ref - got).max())
        return err < 1e-3
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"  [warn] ONNX export failed: {type(exc).__name__}: {exc}")
        return False


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cards", type=int, default=600,
                    help="size of the synthetic card library")
    ap.add_argument("--train-frac", type=float, default=0.8,
                    help="fraction of the library used for training")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--views", type=int, default=2,
                    help="augmented views per card in a contrastive batch")
    ap.add_argument("--hard-frac", type=float, default=0.55,
                    help="fraction of samples drawn from hard conditions")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(WEIGHTS_DIR))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="tiny smoke run (CI): 40 cards, 40 steps")
    ap.add_argument("--no-aux", action="store_true",
                    help="skip dense-head and cardness fitting")
    ap.add_argument("--eval-every", type=int, default=500)
    args = ap.parse_args()

    if args.quick:
        args.cards, args.steps, args.eval_every = 40, 40, 20
        args.workers = min(args.workers, 4)

    import torch
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"FlyCommander embedder training — device={device} cards={args.cards} "
          f"steps={args.steps} batch={args.batch} workers={args.workers}")

    specs = default_library(args.cards, seed=args.seed + 7)
    n_train = max(10, int(len(specs) * args.train_frac))
    train_specs, val_specs = specs[:n_train], specs[n_train:]
    print(f"  library: {len(specs)} cards ({n_train} train / {len(val_specs)} held out)")

    model = CardEmbedNet(dim=args.dim, width=args.width).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  model: {params / 1e6:.2f}M params, dim={args.dim}, input={EMBED_SIZE}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=args.lr, total_steps=max(1, args.steps), pct_start=0.15)

    start_step = 0
    ckpt_path = out_dir / "embedder.pt"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        if "optim" in ckpt:
            optim.load_state_dict(ckpt["optim"])
        start_step = int(ckpt.get("step", 0))
        print(f"  resumed from {ckpt_path} at step {start_step}")

    batcher = CaptureBatcher(train_specs, len(train_specs), batch=args.batch,
                             views=args.views, hard_frac=args.hard_frac,
                             workers=args.workers, seed=args.seed)
    pool = ThreadPoolExecutor(max_workers=args.workers)

    history: list[dict] = []
    t_start = time.time()
    model.train()
    for step in range(start_step, args.steps):
        images, labels = batcher.next_batch()
        x = torch.from_numpy(images).to(device)
        y = torch.from_numpy(labels).to(device)
        emb = model(x)
        loss = supervised_contrastive(emb, y, temperature=args.temperature)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optim.step()
        sched.step()

        if (step + 1) % 20 == 0 or step == 0:
            elapsed = time.time() - t_start
            rate = (step + 1 - start_step) / max(1e-6, elapsed)
            print(f"  step {step + 1:5d}/{args.steps}  loss {loss.item():.4f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  {rate:.1f} steps/s")
        if args.eval_every and (step + 1) % args.eval_every == 0 and val_specs:
            model.eval()
            metrics = evaluate(model, specs, device, EMBED_SIZE, n_val=40,
                               hard=True, pool=pool)
            print(f"  [eval @{step + 1}] top1 {metrics['top1']:.3f} "
                  f"top5 {metrics['top5']:.3f} medianRank {metrics['medianRank']:.0f}")
            history.append({"step": step + 1, **metrics})
            model.train()

    batcher.close()
    model.eval()

    # ---- final evaluation on held-out cards -------------------------------
    metrics_hard = evaluate(model, specs, device, EMBED_SIZE, n_val=150,
                            hard=True, pool=pool) if val_specs else {}
    metrics_easy = evaluate(model, specs, device, EMBED_SIZE, n_val=150,
                            hard=False, pool=pool) if val_specs else {}
    print(f"  final (held-out): hard top1 {metrics_hard.get('top1', 0):.3f} "
          f"top5 {metrics_hard.get('top5', 0):.3f} | easy top1 "
          f"{metrics_easy.get('top1', 0):.3f}")

    meta = {"trained_at": time.time(), "cards": len(specs), "train_cards": n_train,
            "steps": args.steps, "batch": args.batch, "views": args.views,
            "hardFrac": args.hard_frac, "device": device, "seed": args.seed,
            "metrics": {"hard": metrics_hard, "easy": metrics_easy,
                        "history": history}}

    # ---- save torch weights ----------------------------------------------
    torch.save({"state_dict": model.state_dict(),
                "config": {"dim": args.dim, "width": args.width,
                           "size": list(EMBED_SIZE)},
                "optim": optim.state_dict(), "step": args.steps,
                "meta": meta}, ckpt_path)
    print(f"  saved {ckpt_path}")

    # ---- export onnx ------------------------------------------------------
    ok = export_onnx(model, out_dir / "embedder.onnx", EMBED_SIZE, args.dim)
    print(f"  onnx export {'ok' if ok else 'skipped'}")

    # ---- dense head ------------------------------------------------------
    if not args.no_aux:
        print("  fitting dense head (torch-free tier)…")
        library_vecs = embed_library(model, specs, device, EMBED_SIZE)
        head = fit_dense_head(specs, library_vecs, n_samples=200 if args.quick else 2500,
                              workers=args.workers)
        np.savez_compressed(DENSE_HEAD_WEIGHTS, **head,
                            meta=np.array(json.dumps(
                                {"dim": args.dim, "featureDim": FEATURE_DIM,
                                 "trainedAt": time.time(), "cards": len(specs)})))
        print(f"  saved {DENSE_HEAD_WEIGHTS}")

        print("  fitting cardness classifier…")
        cardness = fit_cardness(specs, n_pos=150 if args.quick else 1500,
                                n_neg=150 if args.quick else 1500,
                                workers=args.workers)
        np.savez_compressed(out_dir / "cardness.npz", **cardness)
        print(f"  saved {out_dir / 'cardness.npz'}")

    (out_dir / "training_report.json").write_text(json.dumps(meta, indent=1))
    pool.shutdown(wait=False)
    print(f"done in {time.time() - t_start:.1f}s — weights in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
