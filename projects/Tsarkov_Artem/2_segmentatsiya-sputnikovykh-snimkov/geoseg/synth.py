"""Процедурный генератор многоспектральных «спутниковых» сцен с попиксельной разметкой.

Зачем: открытые датасеты (LandCover.ai, DeepGlobe, LoveDA) весят десятки ГБ и
требуют регистрации; для воспроизводимого CI и демо нужен лёгкий источник
данных с той же структурой (4 канала R,G,B,NIR, GeoTIFF с CRS, маска классов).
Адаптер geoseg.datasets.LandCoverAiAdapter позволяет подключить реальный
LandCover.ai без изменения остального кода.

Классы: 0 — прочее/открытый грунт, 1 — лес, 2 — вода, 3 — здания, 4 — дороги, 5 — сельхозугодья.
Спектральные сигнатуры подобраны по типичным отражательным способностям
(вода — низкий NIR, растительность — высокий NIR и т.д.), а «трудные» пары
специально пересекаются: вспаханное поле ≈ грунт, зелёные посевы ≈ лес
(отличаются текстурой и формой), крыши ≈ дороги.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

CLASSES = ["other", "forest", "water", "building", "road", "farmland"]
PALETTE = np.array([[190, 180, 150], [34, 110, 50], [40, 90, 200], [220, 60, 60], [90, 90, 90], [235, 205, 80]], dtype=np.uint8)

SIGNATURES = {  # R, G, B, NIR
    "other": [(0.30, 0.27, 0.22, 0.35), (0.24, 0.22, 0.18, 0.30)],
    "forest": [(0.04, 0.08, 0.04, 0.38), (0.05, 0.10, 0.05, 0.44)],
    "water": [(0.03, 0.06, 0.09, 0.02), (0.06, 0.09, 0.10, 0.04)],
    "building": [(0.36, 0.34, 0.34, 0.31), (0.42, 0.20, 0.15, 0.30), (0.20, 0.21, 0.24, 0.22)],
    "road": [(0.22, 0.22, 0.23, 0.24), (0.27, 0.26, 0.25, 0.28)],
    "farmland": [(0.08, 0.15, 0.07, 0.45), (0.25, 0.20, 0.15, 0.30), (0.35, 0.32, 0.18, 0.38), (0.12, 0.18, 0.08, 0.40)],
}


def _noise(rng, h, w, scale: float, octaves: int = 4) -> np.ndarray:
    out = np.zeros((h, w))
    amp = 1.0
    for o in range(octaves):
        s = max(scale / (2 ** o), 1.0)
        out += amp * ndi.gaussian_filter(rng.standard_normal((h, w)), s)
        amp *= 0.5
    out -= out.mean()
    return out / (out.std() + 1e-9)


def _thick_polyline(mask: np.ndarray, pts: np.ndarray, width: int) -> None:
    h, w = mask.shape
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        xs, ys = np.linspace(x0, x1, n).astype(int), np.linspace(y0, y1, n).astype(int)
        ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        mask[ys[ok], xs[ok]] = True
    if width > 1:
        mask[:] = ndi.binary_dilation(mask, iterations=width // 2)


def _meander(rng, h, w, n=12, wobble=40) -> np.ndarray:
    if rng.random() < 0.5:
        xs = np.linspace(-10, w + 10, n)
        ys = np.cumsum(rng.normal(0, wobble / 3, n)) + rng.uniform(0.2, 0.8) * h
    else:
        ys = np.linspace(-10, h + 10, n)
        xs = np.cumsum(rng.normal(0, wobble / 3, n)) + rng.uniform(0.2, 0.8) * w
    return np.stack([xs, ys], 1)


def generate_scene(rng: np.random.Generator, h: int = 512, w: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (image float32 [4,H,W] в долях отражения, mask uint8 [H,W])."""
    mask = np.zeros((h, w), np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]

    # сельхозугодья: регион + нарезка на поля (повернутая решётка)
    farm_region = _noise(rng, h, w, 60) > rng.uniform(-0.3, 0.4)
    ang = rng.uniform(0, np.pi)
    u = xx * np.cos(ang) + yy * np.sin(ang)
    v = -xx * np.sin(ang) + yy * np.cos(ang)
    fw, fh = rng.uniform(40, 90), rng.uniform(30, 70)
    field_id = (np.floor(u / fw).astype(int) * 1000 + np.floor(v / fh).astype(int))
    mask[farm_region] = 5

    forest = (_noise(rng, h, w, 25) > rng.uniform(0.0, 0.6)) & ~(farm_region & (rng.random() < 0.7))
    mask[forest] = 1

    # жилая застройка: регион, внутри — сетка домов
    urban = _noise(rng, h, w, 50) > rng.uniform(0.6, 1.2)
    bmask = np.zeros_like(urban)
    for _ in range(int(urban.sum() / 250)):
        y, x = rng.integers(0, h), rng.integers(0, w)
        if urban[y, x]:
            bh, bw = rng.integers(5, 16, 2)
            bmask[y:y + bh, x:x + bw] = True
    mask[bmask] = 3

    roads = np.zeros((h, w), bool)
    for _ in range(rng.integers(1, 4)):
        _thick_polyline(roads, _meander(rng, h, w, n=6, wobble=15), int(rng.integers(3, 6)))
    if urban.any():  # улицы в застройке
        for _ in range(rng.integers(2, 5)):
            r = np.zeros((h, w), bool)
            _thick_polyline(r, _meander(rng, h, w, n=4, wobble=5), 3)
            roads |= r & ndi.binary_dilation(urban, iterations=10)
    mask[roads] = 4

    water = _noise(rng, h, w, 40) > rng.uniform(1.4, 2.2)
    if rng.random() < 0.6:
        river = np.zeros((h, w), bool)
        _thick_polyline(river, _meander(rng, h, w), int(rng.integers(6, 16)))
        water |= river
    mask[water] = 2

    # --- рендер спектра
    img = np.zeros((4, h, w), np.float32)
    for ci, name in enumerate(CLASSES):
        sel = mask == ci
        if not sel.any():
            continue
        sigs = np.array(SIGNATURES[name])
        if name == "farmland":  # каждое поле — своя культура
            ids, inv = np.unique(field_id[sel], return_inverse=True)
            choice = rng.integers(0, len(sigs), len(ids))
            base = sigs[choice[inv]] * rng.uniform(0.85, 1.15, (len(ids), 1))[inv]
            stripes = 1 + 0.06 * np.sin(u[sel] / rng.uniform(1.5, 3.0))
            img[:, sel] = (base * stripes[:, None]).T
        elif name == "building":
            lab, n = ndi.label(sel)
            choice = rng.integers(0, len(sigs), n + 1)
            img[:, sel] = sigs[choice[lab[sel]]].T
        else:
            img[:, sel] = sigs[rng.integers(0, len(sigs))][:, None]
        tex = {"forest": 0.25, "other": 0.10, "water": 0.03, "farmland": 0.04, "building": 0.05, "road": 0.05}[name]
        img[:, sel] *= (1 + tex * _noise(rng, h, w, 1.5, 2)[sel])[None]
    # тени деревьев/зданий, атмосфера, PSF и шум сенсора
    shade = ndi.shift(((mask == 1) | (mask == 3)).astype(float), (2, 2), order=0)
    img *= (1 - 0.25 * shade * ((mask != 1) & (mask != 3)))[None]
    img *= (1 + 0.15 * _noise(rng, h, w, 120, 1))[None]               # неравномерная освещённость / облачные тени
    img += rng.uniform(0.0, 0.06) * np.array([1.0, 0.8, 0.6, 0.3])[:, None, None]  # дымка (сильнее в синем)
    img = ndi.gaussian_filter(img, sigma=(0, 0.7, 0.7))
    img += rng.normal(0, 0.012, img.shape)
    return np.clip(img, 0, 1).astype(np.float32), mask


def write_geotiff(path: str | Path, img: np.ndarray, mask: np.ndarray | None = None, origin=(400000.0, 6200000.0),
                  res: float = 1.0, crs: str = "EPSG:32637") -> None:
    import rasterio
    from rasterio.transform import from_origin

    tr = from_origin(origin[0], origin[1], res, res)
    prof = dict(driver="GTiff", height=img.shape[1], width=img.shape[2], count=4, dtype="uint16", crs=crs,
                transform=tr, compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write((img * 10000).astype(np.uint16))
        dst.descriptions = ("red", "green", "blue", "nir")
    if mask is not None:
        mp = dict(prof, count=1, dtype="uint8")
        with rasterio.open(str(path).replace(".tif", "_mask.tif"), "w", **mp) as dst:
            dst.write(mask[None])


def make_dataset(out: str | Path = "data/synthetic", n_train=48, n_val=8, n_test=12, size=512, seed=0) -> None:
    rng = np.random.default_rng(seed)
    out = Path(out)
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        d = out / split
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img, m = generate_scene(rng, size, size)
            write_geotiff(d / f"scene_{i:03d}.tif", img, m, origin=(400000.0 + i * size, 6200000.0))


if __name__ == "__main__":
    make_dataset()
