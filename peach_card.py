#!/usr/bin/env python3
"""Замена серого фона карточки на персиковый с фактурой ткани.

Запуск:  python3 peach_card.py [card.png] [output_dir]
Зависимости: pillow numpy opencv-python-headless rembg onnxruntime
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ======================= ПАРАМЕТРЫ =======================
SRC = sys.argv[1] if len(sys.argv) > 1 else "card.png"
OUT_DIR = Path(sys.argv[2] if len(sys.argv) > 2 else "output")
FONT_PATH = Path(__file__).with_name("fonts") / "Montserrat-VF.ttf"
SEED = 7

# Фон
BG_TOP = "#FFE3D4"
BG_BOTTOM = "#FFC9B5"
VIGNETTE = 0.05            # макс. затемнение по углам (0.05 = 5%)

# Ткань
WEAVE_PERIOD = 5.0         # шаг нити, px (на 1500 px ширины)
WEAVE_STRENGTH = 0.018     # амплитуда переплетения (доля яркости)
FOLD_STRENGTH = 0.012      # амплитуда мягких складок
FOLD_SCALE = 260           # размер складок, px
FABRIC_MAX_CONTRAST = 0.055  # потолок p1..p99 размаха фактуры (4–6%)

# Тень
SHADOW_COLOR = "#8A5A44"
SHADOW_OPACITY = 0.28
SHADOW_BLUR = 30           # px
SHADOW_OFFSET = (14, 26)   # вправо, вниз, px

# Текст
TEAL_SRC = "#5EACA8"       # бирюзовый в исходнике (замерено)
TEAL_NEW = "#3E9C94"       # во что перекрашиваем
DARK_NEW = None            # None = оставить исходный тёмно-серый
PLAQUE_FILL = "#FFFFFF"
PLAQUE_OPACITY = 0.85

# Подзаголовок
SUB_OLD_BOX = (400, 315, 1100, 405)    # где стоит «для животных» (стереть)
SUB_TEXT = "для кошек и собак"
SUB_SIZE = 79              # кегль, подобран под исходник (Montserrat Bold)
SUB_TRACKING = 1.4         # межбуквенный, px
SUB_TOP = 335              # верх строчных, px
SUB_CENTER_X = 750
SUB_MAX_W = 1500 - 2 * 90  # если шире — уменьшаем кегль

# Зоны с графикой/текстом (x0, y0, x1, y1, мин. альфа) на исходнике 1500×2000.
# Вне этих зон и вне товара всё считается фоном.
INK_BOXES = [
    (370, 140, 1140, 285, 0.05),    # РАСЧЕСКА
    (60, 490, 425, 820, 0.05),      # кнопка + «Видеообзор»
    (1008, 822, 1264, 974, 0.35, "lines"),  # линия выноски: только прямые штрихи
    (1090, 990, 1440, 1275, 0.05),  # «2 в 1», «дешеддер + колтунорез»
    (50, 1610, 1085, 1870, 0.05),   # плашка
]
SOFT_SHADOW_TINT = True    # слабые серые полутени текста → тёплый тон тени
PLAQUE_BOX = (50, 1610, 1085, 1870)
REF_SIZE = (1500, 2000)
# =========================================================


def hex2rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255


def srgb_to_lin(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rel_lum(rgb):
    r, g, b = srgb_to_lin(rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    la, lb = sorted([rel_lum(a), rel_lum(b)], reverse=True)
    return (la + 0.05) / (lb + 0.05)


def scale_box(b, sx, sy):
    return (int(b[0] * sx), int(b[1] * sy), int(b[2] * sx), int(b[3] * sy)) + tuple(b[4:])


# ---------- 1. товар ----------
def product_alpha(img):
    from rembg import new_session, remove
    m = remove(img, session=new_session("isnet-general-use"), only_mask=True)
    a = np.asarray(m, np.float32) / 255
    a = np.where(a < 0.04, 0, np.where(a > 0.96, 1, a))
    # убрать мелкий мусор
    n, lab, st, _ = cv2.connectedComponentsWithStats((a > 0.5).astype(np.uint8))
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] > 5000
    near = cv2.dilate(keep[lab].astype(np.uint8), np.ones((15, 15), np.uint8))
    return a * near


def old_background(P, A):
    lum = P.mean(2)
    chroma = P.max(2) - P.min(2)
    known = ((A < 0.01) & (chroma < 0.05) & (lum > 0.80)).astype(np.float32)
    known = cv2.erode(known, np.ones((9, 9), np.uint8))
    k = (0, 0)
    w = cv2.GaussianBlur(known, k, 40)
    B = np.stack([cv2.GaussianBlur(P[..., c] * known, k, 40) for c in range(3)], 2)
    return B / np.maximum(w, 1e-4)[..., None]


# ---------- 2. ткань ----------
def fabric(h, w, s, rng):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    per = WEAVE_PERIOD * s

    def threads(coord, along, n_len):
        jit = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 6 * s)
        coord = coord + jit / (jit.std() + 1e-6) * 0.6 * s
        t = coord / per
        idx = np.floor(t).astype(int)
        fr = t - idx
        nthr = idx.max() + 2
        bright = rng.normal(0, 1, nthr).astype(np.float32)
        thick = rng.uniform(0.7, 1.4, nthr).astype(np.float32)
        # «слабы» вдоль нити
        slub = cv2.GaussianBlur(rng.normal(0, 1, (nthr, n_len)).astype(np.float32), (int(150 * s) | 1, 1), 25 * s)
        slub /= slub.std() + 1e-6
        prof = np.sin(np.pi * fr) ** thick[idx]
        return prof * (1 + 0.35 * bright[idx] + 0.25 * slub[idx, along.astype(int)]), idx

    warp, ix = threads(x, y, h)   # вертикальные
    weft, iy = threads(y, x, w)   # горизонтальные
    over = ((ix + iy) % 2) == 0
    weave = np.where(over, warp, weft)
    weave += 0.15 * cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 0.7)
    weave = (weave - weave.mean()) / weave.std()

    fs = FOLD_SCALE * s
    folds = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), fs / 2.5)
    folds = folds / folds.std()
    warpf = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), fs)
    warpf /= warpf.std()
    folds += 0.6 * np.sin((x * 0.8 + y * 0.6) / fs * 2.2 + 1.8 * warpf)
    folds = (folds - folds.mean()) / folds.std()

    tex = WEAVE_STRENGTH * weave / 2 + FOLD_STRENGTH * folds / 2
    lo, hi = np.percentile(tex, [1, 99])
    if hi - lo > FABRIC_MAX_CONTRAST:
        tex *= FABRIC_MAX_CONTRAST / (hi - lo)
    return 1 + tex


def background(h, w, s, rng):
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    bg = hex2rgb(BG_TOP) * (1 - t) + hex2rgb(BG_BOTTOM) * t
    bg = np.broadcast_to(bg, (h, w, 3)).copy()
    tex = fabric(h, w, s, rng)
    bg *= tex[..., None]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((x / w - 0.5) / 0.5) ** 2 + ((y / h - 0.5) / 0.5) ** 2) / np.sqrt(2)
    v = np.clip((r - 0.45) / 0.55, 0, 1)
    bg *= (1 - VIGNETTE * v * v * (3 - 2 * v))[..., None]
    return np.clip(bg, 0, 1), tex


# ---------- 3. графика/текст ----------
def unmix_ink(P, B, A, boxes, dark_ref):
    teal = hex2rgb(TEAL_SRC)
    d = P - B
    best_a = np.zeros(P.shape[:2], np.float32)
    best_k = np.zeros(P.shape[:2], np.int8)
    best_res = np.full(P.shape[:2], np.inf, np.float32)
    for k, C in enumerate([teal, dark_ref]):
        v = C - B
        a = np.clip((d * v).sum(2) / np.maximum((v * v).sum(2), 1e-4), 0, 1)
        res = ((d - a[..., None] * v) ** 2).sum(2)
        m = res < best_res
        best_a[m], best_k[m], best_res[m] = a[m], k, res[m]
    zone = np.zeros_like(best_a)
    thr = np.ones_like(best_a)
    for b in boxes:
        zone[b[1]:b[3], b[0]:b[2]] = 1
        thr[b[1]:b[3], b[0]:b[2]] = b[4]
    prod = cv2.dilate((A > 0.02).astype(np.uint8), np.ones((5, 5), np.uint8))
    raw = np.where(best_a < thr, 0, best_a) * zone
    best_a = raw * (1 - prod)
    for b in boxes:
        if len(b) > 5 and b[5] == "lines":
            sub = raw[b[1]:b[3], b[0]:b[2]]
            bw = (sub > 0.5).astype(np.uint8)
            kh = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((1, 41), np.uint8))
            kv = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((41, 1), np.uint8))
            # только строки/столбцы, где реально идёт линия (±1 px на сглаживание)
            rows = cv2.dilate((kh.sum(1) > 0.5 * kh.shape[1] * 0.3).astype(np.uint8)[:, None], np.ones((3, 1), np.uint8))
            cols = cv2.dilate((kv.sum(0) > 0.5 * kv.shape[0] * 0.3).astype(np.uint8)[None, :], np.ones((1, 3), np.uint8))
            keep = (rows | cols) * (sub > 0).astype(np.uint8)
            best_a[b[1]:b[3], b[0]:b[2]] = sub * keep
    return best_a, best_k


def plaque_mask(ink_a, box):
    x0, y0, x1, y1 = box
    sub = (ink_a[y0:y1, x0:x1] > 0.4).astype(np.uint8)
    cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m = np.zeros(ink_a.shape, np.uint8)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(m, [c + [x0, y0]], -1, 1, -1)
    return cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.0)


def render_sub(h, w, s):
    size = SUB_SIZE * s
    while True:
        f = ImageFont.truetype(str(FONT_PATH), max(int(round(size)), 8))
        try:
            f.set_variation_by_name(b"Bold")
        except Exception:
            f.set_variation_by_name("Bold")
        tr = SUB_TRACKING * size / SUB_SIZE
        widths = [f.getlength(c) for c in SUB_TEXT]
        total = sum(widths) + tr * (len(SUB_TEXT) - 1)
        if total <= SUB_MAX_W * s or size < 20:
            break
        size *= 0.97
    top = f.getbbox("для")[1]
    layer = Image.new("L", (w, h), 0)
    dr = ImageDraw.Draw(layer)
    x = SUB_CENTER_X * s - total / 2
    for c, cw in zip(SUB_TEXT, widths):
        dr.text((x, SUB_TOP * s - top), c, font=f, fill=255)
        x += cw + tr
    print(f"подзаголовок: кегль {size / s:.1f}, ширина {total / s:.0f}px")
    return np.asarray(layer, np.float32) / 255


def lerp(a, b, t):
    return a * (1 - t[..., None]) + b * t[..., None]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img = Image.open(SRC).convert("RGB")
    w, h = img.size
    sx, sy = w / REF_SIZE[0], h / REF_SIZE[1]
    s = (sx + sy) / 2
    P = np.asarray(img, np.float32) / 255
    rng = np.random.default_rng(SEED)

    print("вырезаю товар (rembg)…")
    A = product_alpha(img)
    B = old_background(P, A)

    heading = scale_box(INK_BOXES[0][:4], sx, sy)
    hp = P[heading[1]:heading[3], heading[0]:heading[2]].reshape(-1, 3)
    dark_ref = np.median(hp[hp.mean(1) < 0.35], 0)

    boxes = [scale_box(b, sx, sy) for b in INK_BOXES]
    ink_a, ink_k = unmix_ink(P, B, A, boxes, dark_ref)
    ob = scale_box(SUB_OLD_BOX, sx, sy)
    ink_a[ob[1]:ob[3], ob[0]:ob[2]] = 0      # стереть «для животных»

    bg, tex = background(h, w, s, rng)
    out = bg.copy()

    # тень
    M = np.float32([[1, 0, SHADOW_OFFSET[0] * s], [0, 1, SHADOW_OFFSET[1] * s]])
    sh = cv2.warpAffine(A, M, (w, h))
    sh = cv2.GaussianBlur(sh, (0, 0), SHADOW_BLUR * s / 2)
    out = lerp(out, hex2rgb(SHADOW_COLOR), sh * SHADOW_OPACITY)

    # плашка
    pm = plaque_mask(ink_a, scale_box(PLAQUE_BOX, sx, sy))
    out = lerp(out, hex2rgb(PLAQUE_FILL), pm * PLAQUE_OPACITY)

    # графика: бирюза → TEAL_NEW, тёмное → как было
    teal_new = hex2rgb(TEAL_NEW)
    dark_new = hex2rgb(DARK_NEW) if DARK_NEW else dark_ref
    ink_col = np.where((ink_k == 0)[..., None], teal_new, dark_new)
    if SOFT_SHADOW_TINT:
        k = np.clip((ink_a - 0.12) / 0.3, 0, 1)
        warm = lerp(np.broadcast_to(hex2rgb(SHADOW_COLOR), ink_col.shape), ink_col, k)
        ink_col = np.where((ink_k == 1)[..., None], warm, ink_col)
    out = out * (1 - ink_a[..., None]) + ink_col * ink_a[..., None]

    # новый подзаголовок
    sub = render_sub(h, w, s)
    out = lerp(out, teal_new, sub)

    # товар (с очисткой серой каймы)
    F = np.clip((P - (1 - A[..., None]) * B) / np.maximum(A, 0.05)[..., None], 0, 1)
    F = np.where((A > 0.97)[..., None], P, F)
    out = lerp(out, F, A)

    res = Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8))
    res.save(OUT_DIR / "card_peach.png")
    res.save(OUT_DIR / "card_peach.jpg", quality=92, subsampling=0)
    pw = 300
    res.resize((pw, round(h * pw / w)), Image.LANCZOS).save(OUT_DIR / "card_peach_preview.png")

    # ---------- отчёт ----------
    lo, hi = np.percentile(tex, [1, 99])
    print(f"фактура: размах яркости p1..p99 = {(hi - lo) * 100:.1f}%")
    sm = np.asarray(res.resize((pw, round(h * pw / w)), Image.LANCZOS), np.float32)
    bgz = sm[int(sm.shape[0] * .74):int(sm.shape[0] * .8), int(pw * .78):int(pw * .95)]
    print(f"превью 300px: σ яркости фона = {bgz.mean(2).std():.2f} (из 255)")

    def bg_at(box):
        b = scale_box(box, sx, sy)
        return bg[b[1]:b[3], b[0]:b[2]].reshape(-1, 3).mean(0)

    plaque_bg = hex2rgb(PLAQUE_FILL) * PLAQUE_OPACITY + bg_at(PLAQUE_BOX) * (1 - PLAQUE_OPACITY)
    checks = [
        ("«РАСЧЕСКА»", dark_new, bg_at(INK_BOXES[0]), True),
        ("«для кошек и собак»", teal_new, bg_at(SUB_OLD_BOX), True),
        ("«Видеообзор»", teal_new, bg_at((70, 750, 420, 810)), True),
        ("«2 в 1»", dark_new, bg_at((1100, 1000, 1390, 1110)), True),
        ("«дешеддер + колтунорез»", teal_new, bg_at((1090, 1150, 1420, 1265)), False),
        ("текст в плашке", teal_new, plaque_bg, False),
    ]
    print("\nконтраст WCAG (норма: 4.5 мелкий, 3.0 крупный):")
    for name, fg, bgc, large in checks:
        c = contrast(fg, bgc)
        need = 3.0 if large else 4.5
        print(f"  {name:28s} {c:4.2f}:1  {'OK' if c >= need else 'НИЖЕ НОРМЫ'} (порог {need})")

    # подсказка: какой бирюзовый проходит 4.5
    base = hex2rgb(TEAL_NEW)
    worst = min((bgc for *_, bgc, _ in checks), key=rel_lum)
    for k in np.linspace(1, 0.3, 71):
        if contrast(base * k, worst) >= 4.5:
            print("  бирюзовый для 4.5:1 →", "#%02X%02X%02X" % tuple((base * k * 255).round().astype(int)))
            break
    print("\nготово:", ", ".join(str(OUT_DIR / n) for n in
                               ["card_peach.png", "card_peach.jpg", "card_peach_preview.png"]))


if __name__ == "__main__":
    main()
