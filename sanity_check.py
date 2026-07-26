#!/usr/bin/env python
"""
sanity_check.py — проверка «может ли пайплайн вообще обучаться».

Берём N примеров и учим на них маленькую модель до переобучения. Смысл теста:
если модель НЕ способна запомнить два десятка картинок, значит где-то баг —
разъехались пары image/mask, перепутаны каналы, побиты метки, — и искать его
надо ДО того, как потратить часы на полное обучение.

Как читать результат
--------------------
Смотреть надо на IoU ПО КЛАССАМ, а не на общий macro-mIoU:

* крупные классы (background, living, bedroom, bathroom, kitchen, wall,
  balcony) должны дойти до ~0.95+. Особенно важен `wall`: он тонкий, и
  рассинхрон входа с маской даже на 1-2 px обрушил бы именно его;
* редкие тонкие классы (storage, stair, door, front_door) на маленькой
  выборке и без взвешенного лосса остаются около нуля — это НЕ баг, а
  ожидаемое поведение при дисбалансе ~200x (см. pixel_frequency.json).

Аугментации намеренно выключены: тест на запоминание, случайные искажения
мешали бы диагностике.

Запуск:
    pip install torch segmentation-models-pytorch matplotlib
    python sanity_check.py --samples 20 --epochs 80
"""
from __future__ import annotations
import argparse, os, sys, time, json

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resplan_dataset import (ResPlanSegmentation, build_val_transform,
                             load_taxonomy, IMAGENET_MEAN, IMAGENET_STD)

# Классы, по которым судим о здоровье пайплайна (крупные, пространственно
# значимые) и которые заведомо страдают от дисбаланса на малой выборке.
BIG = {"background", "living", "bedroom", "bathroom", "kitchen", "wall", "balcony"}
THIN = {"storage", "stair", "window", "door", "front_door"}

PALETTE = {0:(0,0,0),1:(217,217,217),2:(102,194,165),3:(252,141,98),
           4:(141,160,203),5:(179,179,179),6:(163,124,82),7:(158,154,200),
           8:(255,217,47),9:(166,216,84),10:(231,138,195),11:(166,54,3)}


def colorize(m: np.ndarray) -> np.ndarray:
    out = np.zeros((*m.shape, 3), np.uint8)
    for cid, rgb in PALETTE.items():
        out[m == cid] = rgb[::-1]      # BGR
    return out


def denormalize(t: torch.Tensor) -> np.ndarray:
    x = t.numpy().transpose(1, 2, 0) * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return cv2.cvtColor(np.clip(x * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser(description="Sanity-check пайплайна через переобучение")
    ap.add_argument("--root", default="masks_out")
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--assets", default="assets", help="куда класть кривые и предсказания")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import segmentation_models_pytorch as smp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    id_to_name, num_classes = load_taxonomy(args.root)
    names = [id_to_name[i] for i in range(num_classes)]

    full = ResPlanSegmentation(args.root, split="train")
    ds = ResPlanSegmentation(args.root, split="train",
                             ids=full.files[:args.samples],
                             transform=build_val_transform())  # без аугментаций
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"примеров: {len(ds)}  эпох: {args.epochs}  классов: {num_classes}")

    try:
        model = smp.Unet(args.encoder, encoder_weights="imagenet",
                         in_channels=3, classes=num_classes)
        print(f"энкодер: {args.encoder} (imagenet)")
    except Exception as e:
        print(f"веса imagenet недоступны ({type(e).__name__}), учим с нуля")
        model = smp.Unet(args.encoder, encoder_weights=None,
                         in_channels=3, classes=num_classes)

    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def evaluate():
        """(средний по сэмплам mIoU, IoU по классам, какие классы присутствуют)."""
        model.eval()
        inter = np.zeros(num_classes); union = np.zeros(num_classes)
        present = np.zeros(num_classes, bool); per_sample = []
        with torch.no_grad():
            for x, y in DataLoader(ds, batch_size=args.batch_size, num_workers=0):
                pred = model(x).argmax(1)
                for i in range(pred.shape[0]):
                    p, g = pred[i], y[i]; ious = []
                    for c in range(num_classes):
                        gm = (g == c)
                        if not gm.any():
                            continue           # класса нет в разметке — не штрафуем
                        present[c] = True
                        pm = (p == c)
                        it = (gm & pm).sum().item(); un = (gm | pm).sum().item()
                        inter[c] += it; union[c] += un
                        ious.append(it / un if un else 1.0)
                    per_sample.append(float(np.mean(ious)))
        pc = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
        return float(np.mean(per_sample)), pc, present

    hist_loss, hist_iou = [], []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); tot = nb = 0
        for x, y in dl:
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        hist_loss.append((ep, tot / nb))
        if ep % 5 == 0 or ep in (1, args.epochs):
            m, _, _ = evaluate()
            hist_iou.append((ep, m))
            print(f"эпоха {ep:3d}/{args.epochs}  loss={tot/nb:.4f}  "
                  f"mIoU={m:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    miou, pc, present = evaluate()
    print(f"\nФИНАЛ: macro mIoU = {miou:.4f}")
    print("\n=== IoU ПО КЛАССАМ ===")
    for c in range(num_classes):
        if present[c]:
            print(f"  {names[c]:12s} {pc[c]:.4f}")
        else:
            print(f"  {names[c]:12s}   —  (нет в выборке)")

    big_idx = [c for c in range(num_classes) if present[c] and names[c] in BIG]
    thin_idx = [c for c in range(num_classes) if present[c] and names[c] in THIN]
    big_mean = float(np.nanmean(pc[big_idx])) if big_idx else float("nan")
    thin_mean = float(np.nanmean(pc[thin_idx])) if thin_idx else float("nan")
    print(f"\nсредний IoU по КРУПНЫМ классам : {big_mean:.4f}")
    print(f"средний IoU по ТОНКИМ/РЕДКИМ   : {thin_mean:.4f}")

    healthy = big_mean > 0.9
    print("\nВЕРДИКТ:", "ПАЙПЛАЙН ЗДОРОВ — крупные классы запоминаются (>0.9)"
          if healthy else
          "ПОДОЗРЕНИЕ НА БАГ — даже крупные классы не запоминаются")
    if healthy and thin_mean < 0.5:
        print("Низкий IoU тонких/редких классов — следствие дисбаланса, "
              "а не ошибки пайплайна. Лечится взвешенным лоссом: "
              "resplan_dataset.class_weights().")

    os.makedirs(args.assets, exist_ok=True)
    with open(os.path.join(args.assets, "sanity_check.json"), "w", encoding="utf-8") as f:
        json.dump({"miou": miou,
                   "per_class": {names[c]: (None if not present[c] else float(pc[c]))
                                 for c in range(num_classes)},
                   "big_mean": big_mean, "thin_mean": thin_mean,
                   "loss": hist_loss, "iou": hist_iou}, f, ensure_ascii=False, indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot([e for e, _ in hist_loss], [v for _, v in hist_loss], color="#c0392b")
    ax[0].set_yscale("log"); ax[0].set_xlabel("эпоха"); ax[0].grid(alpha=.3)
    ax[0].set_title(f"CrossEntropy loss, {args.samples} примеров (log)")
    ax[1].plot([e for e, _ in hist_iou], [v for _, v in hist_iou],
               marker="o", ms=4, color="#2471a3")
    ax[1].axhline(0.9, ls="--", c="gray", label="порог 0.9")
    ax[1].set_ylim(0, 1); ax[1].set_xlabel("эпоха"); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[1].set_title("train mIoU")
    plt.tight_layout()
    plt.savefig(os.path.join(args.assets, "overfit_curve.png"), dpi=110)

    rows = []
    model.eval()
    with torch.no_grad():
        for i in range(min(3, len(ds))):
            x, y = ds[i]
            pred = model(x.unsqueeze(0)).argmax(1)[0].numpy().astype(np.uint8)
            sep = np.full((x.shape[1], 4, 3), 160, np.uint8)
            rows.append(np.hstack([denormalize(x), sep,
                                   colorize(y.numpy().astype(np.uint8)), sep,
                                   colorize(pred)]))
    if rows:
        gap = np.full((10, rows[0].shape[1], 3), 245, np.uint8)
        sheet = np.vstack([t for pair in zip(rows, [gap] * len(rows)) for t in pair][:-1])
        cv2.imwrite(os.path.join(args.assets, "overfit_preds.png"), sheet)
    print(f"\nартефакты: {args.assets}/overfit_curve.png, "
          f"{args.assets}/overfit_preds.png, {args.assets}/sanity_check.json")


if __name__ == "__main__":
    main()
