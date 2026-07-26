#!/usr/bin/env python
"""
resplan_to_masks.py
===================
Прогоняет весь датасет ResPlan (ResPlan.pkl) в многоклассовые PNG-маски
сегментации и делит выборку на train/val.

Каждый выходной PNG — одноканальный uint8, где значение пикселя = ID класса
(0 = фон). Это стандартный формат метки для семантической сегментации
(CrossEntropy/Dice/Tversky его едят напрямую).

────────────────────────────────────────────────────────────────────────────
ГЛАВНОЕ: маппинг классов вынесен в CLASS_ID сверху. Он НЕ захардкожен по коду —
меняешь только этот словарь, чтобы согласовать таксономию с Floor Plan CIS:
  • слить классы   -> дать двум исходным ключам ОДИН id (storage+stair -> 6)
  • игнорировать   -> убрать ключ из словаря (его геометрия просто не рисуется)
  • переупорядочить-> поправить DRAW_ORDER (кто рисуется поверх кого)
────────────────────────────────────────────────────────────────────────────

Запуск:
    python resplan_to_masks.py --res 256 --val-frac 0.15 --color
    python resplan_to_masks.py --res 512 --limit 200      # быстрый пробный прогон
"""
from __future__ import annotations
import argparse, json, os, pickle, sys, types, hashlib
import numpy as np
import cv2
from shapely import affinity

# --- resplan_utils тянет geopandas/matplotlib/networkx только ради рисования
#     и графа; нам нужна лишь geometry_to_mask. Подставляем заглушки, чтобы
#     импортировать ИХ настоящую растеризацию без установки тяжёлых пакетов. ---
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
for _m in ["geopandas", "matplotlib", "matplotlib.pyplot", "networkx"]:
    sys.modules.setdefault(_m, types.ModuleType(_m))
from resplan_utils import geometry_to_mask, CATEGORY_COLORS  # noqa: E402

# ============================================================================
#                        ТАКСОНОМИЯ  (единственное, что правишь)
# ============================================================================
# исходный ключ-слой в ResPlan  ->  целочисленный ID класса (0 зарезервирован
# под фон). Несколько ключей с одним ID = слияние классов.
CLASS_ID: dict[str, int] = {
    "living":     1,
    "bedroom":    2,
    "bathroom":   3,
    "kitchen":    4,
    "balcony":    5,
    "storage":    6,
    "stair":      7,
    "wall":       8,
    "window":     9,
    "door":      10,
    "front_door":11,
    # Осознанно НЕ включены (контекст участка, не помещения):
    #   inner, land, garden, parking, pool
    # Чтобы добавить/слить/выкинуть — правь строки выше. Пример слияния:
    #   "storage": 6, "stair": 6,   # обе кладовки/лестницы в один класс "прочее"
}

# Порядок отрисовки: слой рисуется поверх предыдущих (поздний перекрывает).
# Комнаты — снизу, тонкие архитектурные элементы (стены/окна/двери) — сверху,
# иначе стены «тонут» под заливкой комнат.
DRAW_ORDER: list[str] = [
    "living", "bedroom", "bathroom", "kitchen", "balcony", "storage", "stair",
    "wall", "window", "door", "front_door",
]

BACKGROUND_ID = 0
# ============================================================================


def build_id_to_name() -> dict[int, str]:
    """id -> человекочитаемое имя (для легенды/палитры). При слиянии склеивает."""
    inv: dict[int, list[str]] = {}
    for k, v in CLASS_ID.items():
        inv.setdefault(v, []).append(k)
    names = {BACKGROUND_ID: "background"}
    names.update({i: "+".join(ks) for i, ks in inv.items()})
    return names


def content_bounds(plan: dict, keys: list[str]):
    """Bounding box по всем рисуемым слоям плана (чтобы ничего не обрезать)."""
    x1 = y1 = float("inf"); x2 = y2 = float("-inf")
    for k in keys:
        g = plan.get(k)
        if g is None or getattr(g, "is_empty", True):
            continue
        bx1, by1, bx2, by2 = g.bounds
        x1, y1 = min(x1, bx1), min(y1, by1)
        x2, y2 = max(x2, bx2), max(y2, by2)
    if x1 == float("inf"):
        return None
    return x1, y1, x2, y2


def make_transform(bounds, res: int, margin: int):
    """Возвращает f(geom)->geom: вписывает content-bbox в квадрат res×res
    с сохранением пропорций, центрирует и переворачивает Y (архитектурный вид,
    row0 = верх)."""
    x1, y1, x2, y2 = bounds
    w, h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    s = (res - 2 * margin) / max(w, h)
    ox = margin + (res - 2 * margin - w * s) / 2.0
    oy = margin + (res - 2 * margin - h * s) / 2.0

    def tf(g):
        g = affinity.translate(g, xoff=-x1, yoff=-y1)
        g = affinity.scale(g, xfact=s, yfact=s, origin=(0, 0))
        g = affinity.translate(g, xoff=ox, yoff=oy)
        g = affinity.scale(g, xfact=1.0, yfact=-1.0, origin=(0, res / 2.0))  # flip Y
        return g
    return tf


def plan_transform(plan: dict, res: int, margin: int):
    """ЕДИНАЯ геометрическая трансформация для маски И для картинки-чертежа.

    Оба рендера обязаны звать именно её: один и тот же content-bbox гарантирует,
    что вход и метка совпадают попиксельно. Возвращает (tf | None, keys).
    """
    keys = [k for k in DRAW_ORDER if k in CLASS_ID]
    b = content_bounds(plan, keys)
    if b is None:
        return None, keys
    return make_transform(b, res, margin), keys


def plan_to_label(plan: dict, res: int, margin: int) -> np.ndarray:
    """Собирает многоклассовую маску (uint8, значение = ID класса)."""
    label = np.full((res, res), BACKGROUND_ID, dtype=np.uint8)
    tf, keys = plan_transform(plan, res, margin)
    if tf is None:
        return label
    for k in keys:
        g = plan.get(k)
        if g is None or getattr(g, "is_empty", True):
            continue
        m = geometry_to_mask(tf(g), shape=(res, res))  # бинарная маска слоя [0/255]
        if m.max() == 0:
            continue
        label[m > 0] = CLASS_ID[k]
    return label


# --- Рендер «как на реальном чертеже» -------------------------------------
# Слои, читающиеся как несущая структура (заливка тёмным).
STRUCTURE_KEYS = ["wall"]
# Слои, читающиеся как ПРОЁМЫ: они не рисуются, а ВЫРЕЗАЮТСЯ из стен,
# оставляя разрыв в стене — именно так дверь/окно выглядят на плане.
OPENING_KEYS = ["door", "window", "front_door"]

PAPER_VALUE = 255  # белый лист: фон и комнаты
WALL_VALUE = 0     # тёмные стены


def plan_to_image(plan: dict, res: int, margin: int,
                  opening_dilate: int = 0) -> np.ndarray:
    """Рендерит план как чертёж: тёмные стены на белом листе, двери/окна —
    разрывы в стенах. Комнаты НЕ раскрашиваются по типам (иначе задача
    вырождается: модель считывала бы цвет вместо геометрии).

    Использует ту же plan_transform, что и plan_to_label → попиксельное
    соответствие входа и маски.
    """
    img = np.full((res, res), PAPER_VALUE, dtype=np.uint8)
    tf, _ = plan_transform(plan, res, margin)
    if tf is None:
        return img

    def raster(keys):
        acc = np.zeros((res, res), dtype=np.uint8)
        for k in keys:
            g = plan.get(k)
            if g is None or getattr(g, "is_empty", True):
                continue
            acc = np.maximum(acc, geometry_to_mask(tf(g), shape=(res, res)))
        return acc

    wall = raster(STRUCTURE_KEYS)
    opening = raster(OPENING_KEYS)

    # По умолчанию расширение выключено (0). Замер на выборке показал, что
    # двери/окна прорезают стену насквозь и без него: остаточных перемычек
    # 0 px, а IoU(чернила входа, класс wall в маске) = 1.000 против 0.967
    # при расширении в 1 px. То есть расширение лишь срезало однопиксельные
    # торцы стен у проёмов, создавая пиксели, размеченные как wall, но
    # выглядящие на входе пустым листом. Ручка оставлена на случай данных
    # с более грубой геометрией, где перемычки всё же появятся.
    if opening_dilate > 0 and opening.max() > 0:
        opening = cv2.dilate(opening, np.ones((3, 3), np.uint8),
                             iterations=opening_dilate)

    wall[opening > 0] = 0        # вырезаем проёмы из стен
    img[wall > 0] = WALL_VALUE   # то, что осталось от стен — тёмным
    return img


def hex_to_bgr(h: str):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def colorize(label: np.ndarray, id_to_name: dict[int, str]) -> np.ndarray:
    """Цветной превью-PNG (BGR) по палитре resplan_utils.CATEGORY_COLORS."""
    out = np.zeros((*label.shape, 3), np.uint8)
    for cid, name in id_to_name.items():
        if cid == BACKGROUND_ID:
            continue
        key = name.split("+")[0]  # при слиянии берём цвет первого исходного слоя
        out[label == cid] = hex_to_bgr(CATEGORY_COLORS.get(key, "#000000"))
    return out


def split_train_val(ids: list, val_frac: float, seed: int):
    """Детерминированный сплит по хэшу id (стабилен между запусками и res)."""
    train, val = [], []
    for i in ids:
        h = int(hashlib.md5(f"{seed}:{i}".encode()).hexdigest(), 16)
        (val if (h % 10000) / 10000.0 < val_frac else train).append(i)
    return train, val


def main():
    ap = argparse.ArgumentParser(description="ResPlan -> многоклассовые PNG-маски + train/val")
    ap.add_argument("--pkl", default=os.path.join(_HERE, "ResPlan.pkl"))
    ap.add_argument("--out", default=os.path.join(_HERE, "masks_out"))
    ap.add_argument("--res", type=int, default=256, help="сторона квадратной маски (px)")
    ap.add_argument("--margin", type=int, default=4, help="паддинг от края, px")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--color", action="store_true", help="дополнительно писать цветной превью")
    ap.add_argument("--images", action="store_true",
                    help="рендерить входные картинки-чертежи в images/ (пары к маскам)")
    ap.add_argument("--opening-dilate", type=int, default=0,
                    help="на сколько расширять двери/окна при вырезании проёма в стене "
                         "(0 = точно по геометрии, вход и маска совпадают идеально)")
    ap.add_argument("--limit", type=int, default=0, help="обработать только N планов (отладка)")
    args = ap.parse_args()

    id_to_name = build_id_to_name()

    print(f"Загружаю {args.pkl} ...")
    with open(args.pkl, "rb") as f:
        data = pickle.load(f)
    if args.limit:
        data = data[:args.limit]
    print(f"Планов к обработке: {len(data)}  |  res={args.res}  |  классов={len(set(CLASS_ID.values()))}")

    masks_dir = os.path.join(args.out, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    if args.color:
        color_dir = os.path.join(args.out, "masks_color")
        os.makedirs(color_dir, exist_ok=True)
    if args.images:
        images_dir = os.path.join(args.out, "images")
        os.makedirs(images_dir, exist_ok=True)

    # Сохраняем маппинг рядом с масками — для воспроизводимости и согласования с CIS.
    with open(os.path.join(args.out, "class_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({
            "class_id_from_source_key": CLASS_ID,
            "id_to_name": {str(k): v for k, v in id_to_name.items()},
            "draw_order": DRAW_ORDER,
            "background_id": BACKGROUND_ID,
            "resolution": args.res,
        }, f, ensure_ascii=False, indent=2)

    ids, px_per_class = [], np.zeros(256, dtype=np.int64)
    for i, plan in enumerate(data):
        pid = plan.get("id", i)
        label = plan_to_label(plan, args.res, args.margin)
        cv2.imwrite(os.path.join(masks_dir, f"{pid}.png"), label)
        if args.images:
            img = plan_to_image(plan, args.res, args.margin, args.opening_dilate)
            cv2.imwrite(os.path.join(images_dir, f"{pid}.png"), img)
        if args.color:
            cv2.imwrite(os.path.join(color_dir, f"{pid}.png"), colorize(label, id_to_name))
        u, c = np.unique(label, return_counts=True)
        px_per_class[u] += c
        ids.append(pid)
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(data)} ...")

    train, val = split_train_val(ids, args.val_frac, args.seed)
    with open(os.path.join(args.out, "train.txt"), "w") as f:
        f.write("\n".join(f"{p}.png" for p in train))
    with open(os.path.join(args.out, "val.txt"), "w") as f:
        f.write("\n".join(f"{p}.png" for p in val))

    # Частотность по пикселям всего сплита -> сразу пригодится для весов loss.
    total = px_per_class.sum()
    freq = {id_to_name.get(int(cid), str(cid)): int(px_per_class[cid])
            for cid in np.nonzero(px_per_class)[0]}
    with open(os.path.join(args.out, "pixel_frequency.json"), "w", encoding="utf-8") as f:
        json.dump({"total_pixels": int(total),
                   "pixels_per_class": freq,
                   "share": {k: v / total for k, v in freq.items()}}, f,
                  ensure_ascii=False, indent=2)

    print("\nГотово.")
    print(f"  маски:   {masks_dir}")
    if args.images:
        print(f"  входы:   {images_dir}  (пары images/{{id}}.png <-> masks/{{id}}.png)")
    print(f"  train/val: {len(train)} / {len(val)}  (val_frac={args.val_frac})")
    print(f"  маппинг: {os.path.join(args.out, 'class_mapping.json')}")
    print(f"  частоты: {os.path.join(args.out, 'pixel_frequency.json')}")
    print("\nДоли пикселей по классам:")
    for name, c in sorted(freq.items(), key=lambda x: -x[1]):
        print(f"    {name:18s} {100*c/total:6.2f}%")


if __name__ == "__main__":
    main()
