"""Обучение U-Net + baseline'ы + оценка на тестовых сценах.

python -m geoseg.train --epochs 30
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from .models.strategies import (N_CLASSES, PixelForestStrategy, SpectralIndexStrategy, UNetStrategy, normalize,  # noqa: E402
                                spectral_features)
from .models.unet import UNet  # noqa: E402
from .pipeline.tiling import Mosaic, make_windows  # noqa: E402
from .synth import CLASSES, PALETTE, make_dataset  # noqa: E402

DATA = Path("data/synthetic")
IMG = Path("docs/img")
MODELS = Path("models")


def load_split(split: str) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for p in sorted((DATA / split).glob("scene_*[0-9].tif")):
        with rasterio.open(p) as src:
            img = src.read().astype(np.float32) / 10000
        with rasterio.open(str(p).replace(".tif", "_mask.tif")) as src:
            m = src.read(1)
        out.append((img, m))
    return out


def random_batch(rng, scenes, bs, size):
    xs, ys = [], []
    for _ in range(bs):
        img, m = scenes[rng.integers(len(scenes))]
        r, c = rng.integers(0, img.shape[1] - size + 1), rng.integers(0, img.shape[2] - size + 1)
        x, y = img[:, r:r + size, c:c + size], m[r:r + size, c:c + size]
        k = rng.integers(4)
        x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k)
        if rng.random() < 0.5:
            x, y = x[:, :, ::-1], y[:, ::-1]
        x = x * rng.uniform(0.9, 1.1) + rng.normal(0, 0.005)  # фотометрическая аугментация
        xs.append(normalize(np.ascontiguousarray(x)))
        ys.append(np.ascontiguousarray(y))
    return torch.from_numpy(np.stack(xs)).float(), torch.from_numpy(np.stack(ys)).long()


def confusion(y, p, k=N_CLASSES):
    return np.bincount(k * y.ravel().astype(int) + p.ravel(), minlength=k * k).reshape(k, k)


def scores(cm):
    tp = np.diag(cm)
    iou = tp / np.maximum(cm.sum(0) + cm.sum(1) - tp, 1)
    return {"pixel_acc": float(tp.sum() / cm.sum()), "miou": float(iou.mean()),
            "iou": {c: round(float(v), 4) for c, v in zip(CLASSES, iou)},
            "f1": {c: round(float(2 * t / max(cm[i].sum() + cm[:, i].sum(), 1)), 4) for i, (c, t) in enumerate(zip(CLASSES, tp))}}


def predict_scene(strategy, img, tile=256, overlap=32):
    h, w = img.shape[1:]
    mos = Mosaic(h, w, N_CLASSES, tile, overlap)
    wins = make_windows(h, w, tile, overlap)
    if hasattr(strategy, "predict_proba_batch"):
        probs = strategy.predict_proba_batch(np.stack([img[:, wi.row:wi.row + wi.height, wi.col:wi.col + wi.width] for wi in wins]))
        for wi, p in zip(wins, probs):
            mos.add(wi, p)
    else:
        for wi in wins:
            mos.add(wi, strategy.predict_proba(img[:, wi.row:wi.row + wi.height, wi.col:wi.col + wi.width]))
    return mos.mask()


def evaluate(strategy, scenes):
    cm = np.zeros((N_CLASSES, N_CLASSES), np.int64)
    t = time.time()
    for img, m in scenes:
        cm += confusion(m, predict_scene(strategy, img))
    s = scores(cm)
    s["sec_per_mpx"] = round((time.time() - t) / (sum(m.size for _, m in scenes) / 1e6), 3)
    return s, cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--base", type=int, default=16)
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(2)
    if not (DATA / "train").exists():
        make_dataset(DATA)
    MODELS.mkdir(exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)
    train, val, test = load_split("train"), load_split("val"), load_split("test")
    counts = np.bincount(np.concatenate([m.ravel() for _, m in train]), minlength=N_CLASSES)
    freq = counts / counts.sum()
    cw = torch.tensor((1 / np.sqrt(freq)) / (1 / np.sqrt(freq)).mean(), dtype=torch.float32)

    model = UNet(n_classes=N_CLASSES, base=a.base)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=a.epochs * a.steps)
    loss_fn = nn.CrossEntropyLoss(weight=cw)
    rng = np.random.default_rng(0)
    hist, best, t0 = [], -1, time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        losses = []
        for _ in range(a.steps):
            x, y = random_batch(rng, train, a.bs, a.size)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        model.eval()
        ckpt = {"state_dict": model.state_dict(), "config": {"base": a.base}, "version": f"unet-b{a.base}-ep{ep}",
                "classes": CLASSES}
        torch.save(ckpt, MODELS / "_tmp.pt")
        vs, _ = evaluate(UNetStrategy(MODELS / "_tmp.pt"), val)
        hist.append({"epoch": ep, "loss": float(np.mean(losses)), "val_miou": vs["miou"], "val_acc": vs["pixel_acc"],
                     "sec": round(time.time() - t0, 1)})
        print(json.dumps(hist[-1]), flush=True)
        if vs["miou"] > best:
            best = vs["miou"]
            ckpt["version"] = f"unet-b{a.base}"
            torch.save(ckpt, MODELS / "unet_landcover.pt")
    (MODELS / "_tmp.pt").unlink()

    # baseline: RandomForest по пикселю
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    feats = np.concatenate([spectral_features(img).reshape(6, -1).T for img, _ in train])
    labels = np.concatenate([m.ravel() for _, m in train])
    idx = rng.choice(len(labels), 200_000, replace=False)
    rf = RandomForestClassifier(n_estimators=60, max_depth=16, n_jobs=2, random_state=0).fit(feats[idx], labels[idx])
    joblib.dump(rf, MODELS / "pixel_rf.joblib", compress=3)

    results, cms = {}, {}
    for s in (SpectralIndexStrategy(), PixelForestStrategy(rf), UNetStrategy(MODELS / "unet_landcover.pt")):
        results[s.name], cms[s.name] = evaluate(s, test)
        print(s.name, results[s.name]["miou"], flush=True)

    report = {"dataset": {"train_scenes": len(train), "val_scenes": len(val), "test_scenes": len(test),
                          "scene_size": list(train[0][1].shape), "class_freq": dict(zip(CLASSES, np.round(freq, 4).tolist()))},
              "model": {"arch": "U-Net", "base": a.base, "params": n_params, "epochs": a.epochs, "steps": a.steps,
                        "batch": a.bs, "crop": a.size, "class_weights": cw.round(decimals=3).tolist()},
              "train_seconds": round(time.time() - t0, 1), "history": hist, "test": results,
              "confusion_unet": cms["unet"].tolist()}
    (MODELS / "metrics.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))

    # графики
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot([h["epoch"] for h in hist], [h["loss"] for h in hist])
    ax[0].set(title="Train loss (weighted CE)", xlabel="epoch")
    ax[1].plot([h["epoch"] for h in hist], [h["val_miou"] for h in hist], label="val mIoU")
    ax[1].plot([h["epoch"] for h in hist], [h["val_acc"] for h in hist], label="val pixel acc")
    ax[1].set(title="Validation", xlabel="epoch")
    ax[1].legend()
    for x in ax:
        x.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMG / "training_curves.png", dpi=130)
    plt.close(fig)

    cm = cms["unet"].astype(float)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cmn, cmap="Blues")
    ax.set_xticks(range(N_CLASSES), CLASSES, rotation=45)
    ax.set_yticks(range(N_CLASSES), CLASSES)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8, color="w" if cmn[i, j] > 0.5 else "k")
    ax.set(title="U-Net: нормированная матрица ошибок (test)", xlabel="предсказано", ylabel="истина")
    fig.tight_layout()
    fig.savefig(IMG / "confusion_unet.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    xs = np.arange(N_CLASSES)
    for k, name in enumerate(results):
        ax.bar(xs + (k - 1) * 0.27, [results[name]["iou"][c] for c in CLASSES], 0.27, label=f"{name} (mIoU={results[name]['miou']:.3f})")
    ax.set_xticks(xs, CLASSES)
    ax.set(title="IoU по классам на тесте", ylim=(0, 1))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(IMG / "iou_by_class.png", dpi=130)
    plt.close(fig)

    unet = UNetStrategy(MODELS / "unet_landcover.pt")
    rows = []
    for img, m in test[:3]:
        rgb = (np.clip(img[:3].transpose(1, 2, 0) * 2.5, 0, 1) * 255).astype(np.uint8)
        rows.append(np.concatenate([rgb, PALETTE[m], PALETTE[predict_scene(unet, img)],
                                    PALETTE[predict_scene(results and PixelForestStrategy(rf), img)]], 1))
    plt.imsave(IMG / "examples.png", np.concatenate(rows, 0))
    print(json.dumps({k: v["miou"] for k, v in results.items()}))


if __name__ == "__main__":
    main()
