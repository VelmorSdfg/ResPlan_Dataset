#!/usr/bin/env python
"""
resplan_dataset.py — PyTorch Dataset для пар image/mask, сгенерированных
`resplan_to_masks.py`.

Ожидаемая раскладка (см. README):

    masks_out/
    ├── images/{id}.png       вход: чертёж, uint8, 0 = стена, 255 = лист
    ├── masks/{id}.png        метка: uint8, значение пикселя = ID класса 0..11
    ├── train.txt / val.txt   списки файлов вида "14433.png"
    └── class_mapping.json    таксономия (источник истины, не дублируем в коде)

Ключевые инварианты, которые здесь соблюдаются:

* аугментации применяются СИНХРОННО: `transform(image=..., mask=...)`, один
  вызов на пару — иначе геометрия входа и метки разъезжается;
* маска НИКОГДА не нормализуется и интерполируется только `INTER_NEAREST` —
  её значения это ID классов, билинейка смешала бы 3 и 5 в несуществующий 4;
* при поворотах/сдвигах пустые углы заполняются осмысленно: лист (255) для
  входа и фон (0) для маски. Дефолтный reflect-борд отзеркалил бы куски стен
  и создал бы фантомную геометрию, которой на плане нет;
* фотометрия идёт ДО `Normalize`: CLAHE и ISONoise требуют uint8.

Зависимости:
    pip install torch albumentations opencv-python-headless
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Callable, List, Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Статистика ImageNet: чертёж серый, но его реплицируют в 3 канала под
# предобученный энкодер, поэтому нужна трёхканальная статистика.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PAPER_VALUE = 255  # чем заполнять углы входа при геометрии
BACKGROUND_ID = 0  # чем заполнять углы маски
WALL_ID = 8        # класс стен; synthetic_clutter не рисует поверх него


# ---------------------------------------------------------------------------
# Image-only Morphological
# ---------------------------------------------------------------------------

class ImageOnlyMorphological(A.Morphological):
    """A.Morphological — DualTransform: по умолчанию эрозия/дилатация
    применяется И к маске, сдвигая границы классов и ломая попиксельное
    соответствие пары. Нам нужен эффект утолщения/утончения ТОЛЬКО линий на
    изображении, поэтому применение к маске переопределяем на тождество."""

    def apply_to_mask(self, mask, *args, **params):
        return mask


# ---------------------------------------------------------------------------
# Таксономия — читаем из class_mapping.json, не дублируем
# ---------------------------------------------------------------------------

def load_taxonomy(root: str = "masks_out") -> tuple[dict[int, str], int]:
    """Возвращает (id → имя класса, число классов) из class_mapping.json."""
    path = os.path.join(root, "class_mapping.json")
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    id_to_name = {int(k): v for k, v in meta["id_to_name"].items()}
    return id_to_name, max(id_to_name) + 1


def class_weights(root: str = "masks_out", num_classes: Optional[int] = None,
                  scheme: str = "median") -> torch.Tensor:
    """Веса классов из pixel_frequency.json для взвешенного CrossEntropy.

    scheme:
      * "median"  — median-frequency balancing (мягче, обычно устойчивее);
      * "inverse" — 1 / частота, нормированная на среднее (агрессивнее).
    Классы, ни разу не встретившиеся, получают вес 0.
    """
    id_to_name, n = load_taxonomy(root)
    if num_classes is None:
        num_classes = n
    with open(os.path.join(root, "pixel_frequency.json"), "r", encoding="utf-8") as f:
        share = json.load(f)["share"]
    freq = np.zeros(num_classes, dtype=np.float64)
    for cid, name in id_to_name.items():
        if cid < num_classes:
            freq[cid] = share.get(name, 0.0)
    seen = freq > 0
    w = np.zeros(num_classes, dtype=np.float64)
    if scheme == "median":
        w[seen] = np.median(freq[seen]) / freq[seen]
    elif scheme == "inverse":
        inv = 1.0 / freq[seen]
        w[seen] = inv / inv.mean()
    else:
        raise ValueError(f"неизвестная schema весов: {scheme}")
    return torch.tensor(w, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Кастомные аугментации (применяются к image ДО Albumentations)
# ---------------------------------------------------------------------------
# Эти два эффекта имитируют «грязь» реального UGC-скана и рисуются вручную
# через cv2, потому что в Albumentations прямых аналогов нет. Оба:
#   * трогают ТОЛЬКО image, маску не принимают и не меняют;
#   * работают на uint8 (H, W, C) — что серый [.,.,1], что RGB [.,.,3];
#   * принимают np.random.Generator, чтобы val_heavy был детерминирован.

def random_blots(image: np.ndarray,
                 rng: Optional[np.random.Generator] = None,
                 p: float = 0.15) -> np.ndarray:
    """2–4 полупрозрачных эллипса (кляксы/загрязнения), alpha-blend с входом."""
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() > p:
        return image
    out = image.copy()
    h, w = image.shape[:2]
    for _ in range(int(rng.integers(2, 5))):        # 2..4
        overlay = out.copy()
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        ax = int(rng.integers(max(2, w // 20), max(3, w // 6)))
        ay = int(rng.integers(max(2, h // 20), max(3, h // 6)))
        angle = int(rng.integers(0, 180))
        val = int(rng.integers(60, 180))            # серое пятно
        color = (val,) * image.shape[2]
        cv2.ellipse(overlay, (cx, cy), (ax, ay), angle, 0, 360, color, -1)
        alpha = float(rng.uniform(0.1, 0.4))        # полупрозрачно
        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
    return out


def synthetic_clutter(image: np.ndarray,
                      mask: np.ndarray,
                      rng: Optional[np.random.Generator] = None,
                      p: float = 0.3,
                      wall_id: int = WALL_ID) -> np.ndarray:
    """Имитация подписей на чертеже: числа (площади) в 2–3 местах + 1–2
    размерные линии со стрелками. Числа рисуются только в «пустых» зонах,
    НЕ поверх стен, — иначе мы бы дорисовывали то, что модель обязана
    считать структурой. Размерные линии кладутся в поля у самого края
    (там фон), что тоже не задевает стены."""
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() > p:
        return image
    out = image.copy()
    h, w = image.shape[:2]
    wall = (mask == wall_id)

    def is_free(x0, y0, bw, bh):
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x0 + bw), min(h, y0 + bh)
        if x1 <= x0 or y1 <= y0:
            return False
        return not wall[y0:y1, x0:x1].any()

    # --- числа-подписи площадей ---
    for _ in range(int(rng.integers(2, 4))):        # 2..3
        txt = f"{int(rng.integers(5, 40))}.{int(rng.integers(0, 9))}"
        scale = float(rng.uniform(0.3, 0.6))
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        for _try in range(12):                      # ищем свободное место
            x = int(rng.integers(2, max(3, w - tw - 2)))
            y = int(rng.integers(th + 2, h - 2))
            if is_free(x, y - th, tw, th + 4):
                col = int(rng.integers(40, 120))
                cv2.putText(out, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            scale, (col,) * image.shape[2], 1, cv2.LINE_AA)
                break

    # --- размерные линии со стрелками по краям ---
    for _ in range(int(rng.integers(1, 3))):        # 1..2
        col = int(rng.integers(40, 120))
        color = (col,) * image.shape[2]
        if rng.random() < 0.5:                       # горизонтальная у верх/низ края
            y = int(rng.integers(4, 14) if rng.random() < 0.5
                    else rng.integers(h - 14, h - 4))
            x0 = int(rng.integers(5, w // 2)); x1 = int(rng.integers(w // 2, w - 5))
            cv2.arrowedLine(out, (x0, y), (x1, y), color, 1, tipLength=0.03)
            cv2.arrowedLine(out, (x1, y), (x0, y), color, 1, tipLength=0.03)
        else:                                        # вертикальная у лев/прав края
            x = int(rng.integers(4, 14) if rng.random() < 0.5
                    else rng.integers(w - 14, w - 4))
            y0 = int(rng.integers(5, h // 2)); y1 = int(rng.integers(h // 2, h - 5))
            cv2.arrowedLine(out, (x, y0), (x, y1), color, 1, tipLength=0.03)
            cv2.arrowedLine(out, (x, y1), (x, y0), color, 1, tipLength=0.03)
    return out


class PreAugment:
    """Связка кастомных image-only эффектов, применяемых ДО Albumentations.

    Вызывается как ``pre(image, mask, rng) -> image``. mask нужен только
    synthetic_clutter (чтобы не рисовать по стенам) и никогда не меняется.
    """

    def __init__(self, blots_p: float = 0.15, clutter_p: float = 0.3):
        self.blots_p = blots_p
        self.clutter_p = clutter_p

    def __call__(self, image: np.ndarray, mask: np.ndarray,
                 rng: np.random.Generator) -> np.ndarray:
        image = synthetic_clutter(image, mask, rng, p=self.clutter_p)
        image = random_blots(image, rng, p=self.blots_p)
        return image


# ---------------------------------------------------------------------------
# Наборы аугментаций
# ---------------------------------------------------------------------------

def build_train_transform(size: Optional[int] = None,
                          in_channels: int = 3) -> A.Compose:
    """Аугментации для train (рецепт в духе MitUNet).

    Геометрия применяется к паре синхронно, маска — строго NEAREST и
    заполнение фоном. Фотометрия трогает только вход.
    """
    geom_kw = dict(
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,  # ID классов не интерполируем
        border_mode=cv2.BORDER_CONSTANT,
        fill=PAPER_VALUE,         # пустые углы входа = белый лист
        fill_mask=BACKGROUND_ID,  # пустые углы маски = фон
    )

    tfs: List[A.BasicTransform] = []
    if size is not None:
        tfs.append(A.Resize(size, size,
                            interpolation=cv2.INTER_LINEAR,
                            mask_interpolation=cv2.INTER_NEAREST))

    # --- геометрия ---
    tfs.append(A.Affine(
        scale=(0.9, 1.1),
        rotate=(-15, 15),
        translate_percent=(-0.0625, 0.0625),
        p=0.7,
        **geom_kw,
    ))
    # Упругие искажения: слабые, иначе тонкие стены рвутся в лапшу.
    tfs.append(A.OneOf([
        A.ElasticTransform(alpha=40.0, sigma=6.0, **geom_kw),
        A.GridDistortion(num_steps=5, distort_limit=0.2, **geom_kw),
    ], p=0.2))
    # Перспектива: тот же geom_kw (fill=255 / fill_mask=0 / NEAREST), иначе в
    # углах после warp появилась бы фантомная геометрия — та же ловушка, что
    # была бы с BORDER_REFLECT_101 на повороте.
    tfs.append(A.Perspective(scale=(0.05, 0.15), **geom_kw, p=0.3))

    # --- фотометрия (только вход; до Normalize, т.к. нужен uint8) ---
    tfs.append(A.RandomBrightnessContrast(
        brightness_limit=0.2, contrast_limit=0.2, p=0.5))
    tfs.append(A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2))

    noise: List[A.BasicTransform] = [
        A.GaussNoise(std_range=(0.02, 0.08), p=1.0),
    ]
    if in_channels == 3:
        # ISONoise определён только для трёхканального RGB.
        noise.append(A.ISONoise(color_shift=(0.01, 0.05),
                                intensity=(0.1, 0.5), p=1.0))
    tfs.append(A.OneOf(noise, p=0.3))

    # --- деградация качества (UGC-устойчивость; только вход) ---
    tfs += _degradation_block()

    tfs += _finalize(in_channels)
    return A.Compose(tfs)


def _degradation_block(heavy: bool = False) -> List[A.BasicTransform]:
    """Имитация артефактов реального скана/фото чертежа. Все трансформы
    трогают ТОЛЬКО изображение (Morphological — через image-only подкласс,
    CoarseDropout — с fill_mask=None). heavy=True усиливает деградацию для
    стресс-набора val_heavy."""
    if heavy:
        return [
            A.ImageCompression(quality_range=(20, 60), p=0.8),
            A.OneOf([A.MotionBlur(blur_limit=9),
                     A.Defocus(radius=(2, 5))], p=0.6),
            ImageOnlyMorphological(scale=(1, 3), operation="dilation", p=0.3),
            ImageOnlyMorphological(scale=(1, 2), operation="erosion", p=0.3),
            A.CoarseDropout(num_holes_range=(3, 7),
                            hole_height_range=(15, 40),
                            hole_width_range=(15, 40),
                            fill="random", fill_mask=None, p=0.5),
        ]
    return [
        A.ImageCompression(quality_range=(40, 90), p=0.3),       # JPEG-артефакты
        A.OneOf([A.MotionBlur(blur_limit=5),
                 A.Defocus(radius=(1, 3))], p=0.2),
        # расплывшиеся / истончённые линии (image-only подкласс)
        ImageOnlyMorphological(scale=(1, 3), operation="dilation", p=0.15),
        ImageOnlyMorphological(scale=(1, 2), operation="erosion", p=0.15),
        # потёртости/пятна: fill_mask=None -> маска не трогается
        A.CoarseDropout(num_holes_range=(2, 5),
                        hole_height_range=(10, 30),
                        hole_width_range=(10, 30),
                        fill="random", fill_mask=None, p=0.2),
    ]


def build_val_transform(size: Optional[int] = None,
                        in_channels: int = 3) -> A.Compose:
    """Валидация: никакой геометрии и фотометрии — только приведение к тензору.

    Аугментировать val нельзя: метрика должна считаться на неизменённых данных,
    иначе она мерит не модель, а конкретный случайный сид.
    """
    tfs: List[A.BasicTransform] = []
    if size is not None:
        tfs.append(A.Resize(size, size,
                            interpolation=cv2.INTER_LINEAR,
                            mask_interpolation=cv2.INTER_NEAREST))
    tfs += _finalize(in_channels)
    return A.Compose(tfs)


def build_val_heavy_transform(size: Optional[int] = None,
                              in_channels: int = 3) -> A.Compose:
    """Стресс-валидация: та же выборка, что val, но с усиленной деградацией
    качества — оценка UGC-устойчивости. Детерминизм обеспечивается на уровне
    Dataset (per-index seed + set_random_seed), а не здесь: набор трансформов
    один, воспроизводимость даёт сид.

    Перспектива берётся по верхней границе scale; компрессия жёстче
    (quality 20+), блюр сильнее. Геометрия по-прежнему с fill=255/fill_mask=0
    и NEAREST для маски."""
    geom_kw = dict(
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_CONSTANT,
        fill=PAPER_VALUE,
        fill_mask=BACKGROUND_ID,
    )
    tfs: List[A.BasicTransform] = []
    if size is not None:
        tfs.append(A.Resize(size, size,
                            interpolation=cv2.INTER_LINEAR,
                            mask_interpolation=cv2.INTER_NEAREST))
    tfs.append(A.Perspective(scale=(0.12, 0.15), **geom_kw, p=0.8))
    tfs.append(A.RandomBrightnessContrast(
        brightness_limit=0.3, contrast_limit=0.3, p=0.7))
    tfs += _degradation_block(heavy=True)
    tfs += _finalize(in_channels)
    return A.Compose(tfs)


def _finalize(in_channels: int) -> List[A.BasicTransform]:
    """Normalize + ToTensorV2. Normalize по контракту Albumentations
    применяется ТОЛЬКО к image, маску не трогает."""
    if in_channels == 3:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        # Одноканальный вариант: усредняем статистику ImageNet по каналам.
        mean = (float(np.mean(IMAGENET_MEAN)),)
        std = (float(np.mean(IMAGENET_STD)),)
    return [
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(transpose_mask=False),  # image → [C,H,W], mask остаётся [H,W]
    ]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ResPlanSegmentation(Dataset):
    """Пары «чертёж → маска классов» из masks_out/.

    Параметры
    ---------
    root : путь к каталогу с images/, masks/, train.txt, val.txt
    split : "train" | "val" — какой список файлов читать
    variant : "train" | "val" | "val_heavy" — какой набор аугментаций.
        None → берётся равным split. "val_heavy" читает список val, но
        прогоняет усиленную деградацию (детерминированно).
    transform : A.Compose; None → набор по умолчанию для variant
    pre_transform : callable(image, mask, rng)->image, применяется ДО
        Albumentations (кастомные кляксы/подписи). None → по умолчанию для
        variant. Передайте False, чтобы выключить.
    deterministic : фиксировать per-index сид (вход и pre_transform
        воспроизводимы). None → True для val_heavy, иначе False.
    in_channels : 3 (реплика серого в RGB под предобученный энкодер) или 1
    size : сторона квадрата для ресайза; None — оставить как есть
    ids : явный список файлов, переопределяет split (для k-fold и отладки)

    Возвращает
    ----------
    image : float32 [C,H,W], нормализован статистикой ImageNet
    mask  : int64   [H,W],   значения = ID классов, готово для CrossEntropy/Dice
    """

    def __init__(self,
                 root: str = "masks_out",
                 split: str = "train",
                 variant: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 deterministic: Optional[bool] = None,
                 in_channels: int = 3,
                 size: Optional[int] = None,
                 ids: Optional[Sequence[str]] = None):
        if split not in ("train", "val"):
            raise ValueError(f"split должен быть 'train' или 'val', получено {split!r}")
        variant = variant or split
        if variant not in ("train", "val", "val_heavy"):
            raise ValueError(f"variant должен быть train/val/val_heavy, получено {variant!r}")
        if in_channels not in (1, 3):
            raise ValueError(f"in_channels должен быть 1 или 3, получено {in_channels}")

        self.root = root
        self.split = split
        self.variant = variant
        self.in_channels = in_channels
        self.images_dir = os.path.join(root, "images")
        self.masks_dir = os.path.join(root, "masks")

        if not os.path.isdir(self.images_dir):
            raise FileNotFoundError(
                f"нет каталога {self.images_dir}. Картинки-чертежи генерируются "
                f"флагом --images: python resplan_to_masks.py --images")
        if not os.path.isdir(self.masks_dir):
            raise FileNotFoundError(f"нет каталога {self.masks_dir}")

        if ids is not None:
            self.files = list(ids)
        else:
            list_path = os.path.join(root, f"{split}.txt")
            if not os.path.isfile(list_path):
                raise FileNotFoundError(f"нет файла со списком {list_path}")
            with open(list_path, "r", encoding="utf-8") as f:
                self.files = [ln.strip() for ln in f if ln.strip()]

        self.id_to_name, self.num_classes = load_taxonomy(root)

        if transform is None:
            transform = {
                "train":     build_train_transform,
                "val":       build_val_transform,
                "val_heavy": build_val_heavy_transform,
            }[variant](size, in_channels)
        self.transform = transform

        # pre_transform по умолчанию: клаттер/кляксы на train и val_heavy,
        # ничего на чистом val. False = явно выключить.
        if pre_transform is None:
            if variant == "train":
                pre_transform = PreAugment(blots_p=0.15, clutter_p=0.3)
            elif variant == "val_heavy":
                pre_transform = PreAugment(blots_p=0.5, clutter_p=0.5)
            else:
                pre_transform = None
        self.pre_transform = pre_transform or None

        # детерминизм: по умолчанию только val_heavy (нужна воспроизводимая
        # оценка); train — случаен, чистый val рандома и так не содержит.
        self.deterministic = (variant == "val_heavy") if deterministic is None \
            else deterministic

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        name = self.files[idx]
        img_path = os.path.join(self.images_dir, name)
        msk_path = os.path.join(self.masks_dir, name)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"не читается вход: {img_path}")
        mask = cv2.imread(msk_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"не читается маска: {msk_path}")
        if mask.ndim != 2:
            raise ValueError(f"маска должна быть одноканальной: {msk_path}, shape={mask.shape}")
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"размеры пары не совпадают: {name} image={image.shape} mask={mask.shape}")

        if self.in_channels == 3:
            # Реплика серого в RGB: нужна и предобученному энкодеру, и ISONoise.
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = image[:, :, None]

        # Per-index сид: делает вход воспроизводимым (val_heavy) — и
        # pre_transform, и Albumentations сеются одним числом от имени файла.
        if self.deterministic:
            seed = int(hashlib.md5(name.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            if hasattr(self.transform, "set_random_seed"):
                self.transform.set_random_seed(seed)
        else:
            rng = np.random.default_rng()

        # Кастомные image-only эффекты ДО Albumentations (маска не меняется).
        if self.pre_transform is not None:
            image = self.pre_transform(image, mask, rng)

        # ОДИН вызов на пару — синхронная геометрия для image и mask.
        out = self.transform(image=image, mask=mask)
        image_t, mask_t = out["image"], out["mask"]

        # CrossEntropy/Dice ждут индексы классов в int64.
        if not torch.is_tensor(mask_t):
            mask_t = torch.from_numpy(np.asarray(mask_t))
        mask_t = mask_t.long()

        return image_t, mask_t
