"""Нарезка растра на тайлы с перекрытием и обратная сборка с плавным смешиванием.

Перекрытие + весовое окно (Hann) убирают «швы» на границах тайлов, где у
свёрточной сети мало контекста.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TileWindow:
    row: int
    col: int
    height: int
    width: int

    @property
    def key(self) -> str:
        return f"r{self.row}_c{self.col}"


def make_windows(h: int, w: int, tile: int = 256, overlap: int = 32) -> list[TileWindow]:
    step = tile - overlap
    rows = list(range(0, max(h - tile, 0) + 1, step))
    cols = list(range(0, max(w - tile, 0) + 1, step))
    if rows[-1] + tile < h:
        rows.append(h - tile)
    if cols[-1] + tile < w:
        cols.append(w - tile)
    return [TileWindow(r, c, min(tile, h), min(tile, w)) for r in rows for c in cols]


def blend_weights(tile: int, overlap: int) -> np.ndarray:
    if overlap <= 0:
        return np.ones((tile, tile), np.float32)
    ramp = np.ones(tile, np.float32)
    edge = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, overlap + 2)[1:-1])
    ramp[:overlap], ramp[-overlap:] = edge, edge[::-1]
    return np.outer(ramp, ramp) + 1e-3


class Mosaic:
    """Аккумулятор вероятностей тайлов -> итоговая маска всего растра."""

    def __init__(self, h: int, w: int, n_classes: int, tile: int, overlap: int) -> None:
        self.acc = np.zeros((n_classes, h, w), np.float32)
        self.wsum = np.zeros((h, w), np.float32)
        self.wt = blend_weights(tile, overlap)

    def add(self, win: TileWindow, proba: np.ndarray) -> None:
        wt = self.wt[: win.height, : win.width]
        self.acc[:, win.row:win.row + win.height, win.col:win.col + win.width] += proba * wt
        self.wsum[win.row:win.row + win.height, win.col:win.col + win.width] += wt

    def mask(self) -> np.ndarray:
        return (self.acc / np.maximum(self.wsum, 1e-6)).argmax(0).astype(np.uint8)
