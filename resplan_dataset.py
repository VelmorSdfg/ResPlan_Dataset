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

    tfs += _finalize(in_channels)
    return A.Compose(tfs)


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
    transform : A.Compose; если None — берётся набор по умолчанию для split
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
                 transform: Optional[Callable] = None,
                 in_channels: int = 3,
                 size: Optional[int] = None,
                 ids: Optional[Sequence[str]] = None):
        if split not in ("train", "val"):
            raise ValueError(f"split должен быть 'train' или 'val', получено {split!r}")
        if in_channels not in (1, 3):
            raise ValueError(f"in_channels должен быть 1 или 3, получено {in_channels}")

        self.root = root
        self.split = split
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
            transform = (build_train_transform(size, in_channels) if split == "train"
                         else build_val_transform(size, in_channels))
        self.transform = transform

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

        # ОДИН вызов на пару — синхронная геометрия для image и mask.
        out = self.transform(image=image, mask=mask)
        image_t, mask_t = out["image"], out["mask"]

        # CrossEntropy/Dice ждут индексы классов в int64.
        if not torch.is_tensor(mask_t):
            mask_t = torch.from_numpy(np.asarray(mask_t))
        mask_t = mask_t.long()

        return image_t, mask_t
