#!/usr/bin/env python
"""
audit_dataset.py — проверка целостности сгенерированных пар в masks_out/.

Прогонять после каждой перегенерации данных. Проверяет:

1. Целостность пар   — сироты, рассинхрон размеров, наличие всех файлов
                       из train.txt / val.txt на диске, дубликаты в списках.
2. Утечку train/val  — ни один id не должен попасть в оба списка.
3. Вырожденные примеры — маски без единого класса кроме фона, картинки без
                       единого тёмного пикселя, ID вне допустимого диапазона.
4. Покрытие холста   — сколько места реально занимает контент. Полезно, чтобы
                       заметить, что разрешение тратится впустую.

Запуск:
    python audit_dataset.py --root masks_out
"""
from __future__ import annotations
import argparse, glob, json, os

import numpy as np
import cv2


def bbox(binary: np.ndarray):
    ys, xs = np.where(binary)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def read_list(path: str):
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def main():
    ap = argparse.ArgumentParser(description="Аудит целостности пар в masks_out/")
    ap.add_argument("--root", default="masks_out")
    ap.add_argument("--limit", type=int, default=0,
                    help="сканировать только N пар (0 = все)")
    args = ap.parse_args()

    root = args.root
    with open(os.path.join(root, "class_mapping.json"), encoding="utf-8") as f:
        meta = json.load(f)
    id_to_name = {int(k): v for k, v in meta["id_to_name"].items()}
    max_id = max(id_to_name)

    imgs = {os.path.basename(p) for p in glob.glob(os.path.join(root, "images", "*.png"))}
    msks = {os.path.basename(p) for p in glob.glob(os.path.join(root, "masks", "*.png"))}

    problems = 0

    print("=" * 66)
    print("1. ЦЕЛОСТНОСТЬ ПАР")
    print("=" * 66)
    print(f"картинок: {len(imgs)}   масок: {len(msks)}")
    oi, om = imgs - msks, msks - imgs
    print(f"картинок без маски : {len(oi)} {sorted(oi)[:5]}")
    print(f"масок без картинки : {len(om)} {sorted(om)[:5]}")
    problems += len(oi) + len(om)

    train = read_list(os.path.join(root, "train.txt"))
    val = read_list(os.path.join(root, "val.txt"))
    print(f"train.txt: {len(train)}   val.txt: {len(val)}   сумма: {len(train)+len(val)}")
    for nm, lst in (("train", train), ("val", val)):
        mi = [f for f in lst if f not in imgs]
        mm = [f for f in lst if f not in msks]
        dup = len(lst) - len(set(lst))
        print(f"  {nm}: нет картинки {len(mi)}, нет маски {len(mm)}, дубликатов {dup}")
        problems += len(mi) + len(mm) + dup
    orphan_listed = (imgs & msks) - set(train) - set(val)
    print(f"на диске, но ни в одном списке: {len(orphan_listed)}")

    print()
    print("=" * 66)
    print("2. УТЕЧКА TRAIN/VAL")
    print("=" * 66)
    leak = set(train) & set(val)
    print(f"общих файлов: {len(leak)}  {sorted(leak)[:10]}")
    print("ВЕРДИКТ:", "чисто" if not leak else "!!! УТЕЧКА !!!")
    problems += len(leak)

    common = sorted(imgs & msks)
    if args.limit:
        common = common[:args.limit]

    print()
    print("=" * 66)
    print("3-4. СКАН ПАР")
    print("=" * 66)
    empty_mask, empty_ink, mismatch, bad_id = [], [], [], []
    ink_frac, fg_frac, sizes = [], [], {}
    marg = {"сверху": [], "снизу": [], "слева": [], "справа": []}

    for n, f in enumerate(common):
        im = cv2.imread(os.path.join(root, "images", f), cv2.IMREAD_GRAYSCALE)
        mk = cv2.imread(os.path.join(root, "masks", f), cv2.IMREAD_UNCHANGED)
        if im is None or mk is None:
            mismatch.append(f); continue
        if im.shape != mk.shape:
            mismatch.append(f); continue
        sizes[im.shape] = sizes.get(im.shape, 0) + 1
        H, W = mk.shape

        u = np.unique(mk)
        if u.max() > max_id or u.min() < 0:
            bad_id.append(f)
        ink = im < 128
        fg = mk != 0
        if not fg.any():
            empty_mask.append(f)
        if not ink.any():
            empty_ink.append(f)
        ink_frac.append(ink.mean()); fg_frac.append(fg.mean())

        b = bbox(fg)
        if b:
            x0, y0, x1, y1 = b
            marg["слева"].append(x0); marg["сверху"].append(y0)
            marg["справа"].append(W - 1 - x1); marg["снизу"].append(H - 1 - y1)
        if (n + 1) % 4000 == 0:
            print(f"  ... {n+1}/{len(common)}")

    problems += len(empty_mask) + len(empty_ink) + len(mismatch) + len(bad_id)

    print(f"просканировано: {len(ink_frac)}   размеры: {sizes}")
    print(f"маски только с фоном      : {len(empty_mask)} {empty_mask[:5]}")
    print(f"картинки без тёмных пикс. : {len(empty_ink)} {empty_ink[:5]}")
    print(f"рассинхрон размеров       : {len(mismatch)} {mismatch[:5]}")
    print(f"ID вне 0..{max_id}             : {len(bad_id)} {bad_id[:5]}")
    if ink_frac:
        a, b_ = np.array(ink_frac), np.array(fg_frac)
        print(f"чернил на входе, %  : средн={a.mean()*100:.2f} медиана={np.median(a)*100:.2f}")
        print(f"не-фона в маске, %  : средн={b_.mean()*100:.2f} медиана={np.median(b_)*100:.2f}")
        print("поля вокруг контента маски, px:")
        for k, v in marg.items():
            v = np.array(v, float)
            print(f"   {k:8s} средн={v.mean():6.2f} медиана={np.median(v):6.1f}")

    print()
    print("=" * 66)
    print("ИТОГ:", "проблем не найдено" if problems == 0 else f"НАЙДЕНО ПРОБЛЕМ: {problems}")
    print("=" * 66)
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
