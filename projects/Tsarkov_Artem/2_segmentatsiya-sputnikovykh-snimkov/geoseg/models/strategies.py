"""Модели сегментации — паттерны Strategy + Adapter.

SegmentationStrategy — единый интерфейс predict_proba(tile[C,H,W]) -> [K,H,W].
* UNetStrategy        — обученный U-Net (PyTorch), основной вариант;
* SegFormerStrategy   — адаптер к transformers.SegformerForSemanticSegmentation
                        (другой формат входа/выхода приводится к общему интерфейсу);
* SpectralIndexStrategy — baseline на правилах по индексам NDVI/NDWI;
* PixelForestStrategy — baseline: RandomForest по спектру пикселя (без контекста).
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

N_CLASSES = 6


def normalize(tile: np.ndarray, stats: dict | None = None) -> np.ndarray:
    """tile: [4,H,W] в долях отражения -> стандартизированный по каналам."""
    mean = np.array((stats or {}).get("mean", [0.17, 0.17, 0.14, 0.31]), np.float32)[:, None, None]
    std = np.array((stats or {}).get("std", [0.12, 0.09, 0.08, 0.13]), np.float32)[:, None, None]
    return (tile - mean) / std


def spectral_features(tile: np.ndarray) -> np.ndarray:
    r, g, b, n = tile
    ndvi = (n - r) / (n + r + 1e-6)
    ndwi = (g - n) / (g + n + 1e-6)
    return np.stack([r, g, b, n, ndvi, ndwi], 0)


class SegmentationStrategy(ABC):
    name = "abstract"
    version = "0"

    @abstractmethod
    def predict_proba(self, tile: np.ndarray) -> np.ndarray: ...

    def predict(self, tile: np.ndarray) -> np.ndarray:
        return self.predict_proba(tile).argmax(0).astype(np.uint8)


class UNetStrategy(SegmentationStrategy):
    name = "unet"

    def __init__(self, weights: str | Path, device: str = "cpu") -> None:
        import torch

        from .unet import UNet

        ckpt = torch.load(weights, map_location=device, weights_only=False)
        self.cfg = ckpt["config"]
        self.model = UNet(n_classes=N_CLASSES, base=self.cfg["base"]).to(device).eval()
        self.model.load_state_dict(ckpt["state_dict"])
        self.stats = ckpt.get("stats")
        self.version = ckpt.get("version", "unet")
        self.device = device
        self.torch = torch

    def predict_proba(self, tile: np.ndarray) -> np.ndarray:
        return self.predict_proba_batch(tile[None])[0]

    def predict_proba_batch(self, tiles: np.ndarray) -> np.ndarray:
        x = np.stack([normalize(t, self.stats) for t in tiles])
        with self.torch.no_grad():
            return self.torch.softmax(self.model(self.torch.from_numpy(x).to(self.device)), 1).cpu().numpy()


class SegFormerStrategy(SegmentationStrategy):
    """Адаптер Hugging Face SegFormer: 3 канала на входе, выход в 1/4 разрешения → апсемплинг."""

    name = "segformer"

    def __init__(self, checkpoint: str = "nvidia/mit-b0") -> None:
        import torch
        from transformers import SegformerForSemanticSegmentation  # type: ignore

        self.torch = torch
        self.model = SegformerForSemanticSegmentation.from_pretrained(checkpoint, num_labels=N_CLASSES,
                                                                      ignore_mismatched_sizes=True).eval()
        self.version = checkpoint

    def predict_proba(self, tile: np.ndarray) -> np.ndarray:
        rgb = normalize(tile)[:3][None]  # SegFormer ожидает RGB
        with self.torch.no_grad():
            logits = self.model(pixel_values=self.torch.from_numpy(rgb)).logits
            logits = self.torch.nn.functional.interpolate(logits, size=tile.shape[1:], mode="bilinear")
        return self.torch.softmax(logits, 1)[0].numpy()


class SpectralIndexStrategy(SegmentationStrategy):
    """Классический ГИС-подход: пороги по NDVI/NDWI/яркости. Не различает лес/поле и дороги/здания."""

    name = "spectral_rules"
    version = "rules-1"

    def predict_proba(self, tile: np.ndarray) -> np.ndarray:
        f = spectral_features(tile)
        ndvi, ndwi, bright = f[4], f[5], tile[:3].mean(0)
        out = np.zeros((N_CLASSES,) + tile.shape[1:], np.float32)
        water = ndwi > 0.3
        veg = (ndvi > 0.45) & ~water
        dense = veg & (tile[1] < 0.11)
        out[2][water] = 1
        out[1][dense] = 1
        out[5][veg & ~dense] = 1
        built = ~water & ~veg & (ndvi < 0.08) & (bright > 0.2)
        out[3][built & (bright > 0.3)] = 1
        out[4][built & (bright <= 0.3)] = 1
        out[0][out.sum(0) == 0] = 1
        return out


class PixelForestStrategy(SegmentationStrategy):
    name = "pixel_rf"

    def __init__(self, model=None, path: str | Path | None = None) -> None:
        import joblib

        self.model = model if model is not None else joblib.load(path)
        self.version = "rf-1"

    def predict_proba(self, tile: np.ndarray) -> np.ndarray:
        f = spectral_features(tile).reshape(6, -1).T
        p = self.model.predict_proba(f).T.reshape((-1,) + tile.shape[1:])
        return p.astype(np.float32)


def make_strategy(name: str, weights: str | None = None) -> SegmentationStrategy:
    if name == "unet":
        return UNetStrategy(weights or "models/unet_landcover.pt")
    if name == "segformer":
        return SegFormerStrategy(weights or "nvidia/mit-b0")
    if name == "spectral_rules":
        return SpectralIndexStrategy()
    if name == "pixel_rf":
        return PixelForestStrategy(path=weights or "models/pixel_rf.joblib")
    raise ValueError(f"unknown model {name}")


def load_meta(path: str | Path) -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}
