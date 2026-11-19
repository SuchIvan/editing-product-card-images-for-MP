#!/usr/bin/env python3
"""v2: объёмный «студийный» фон — ткань со складками, свет, тени, отсвет.

Запуск:  python3 peach_card_v2.py [card.png] [output_dir]
Вырезание товара, перенос надписей и замена подзаголовка — из peach_card.py.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import peach_card as pc
from peach_card import hex2rgb, lerp, contrast, rel_lum, scale_box

# ======================= ПАРАМЕТРЫ =======================
SRC = sys.argv[1] if len(sys.argv) > 1 else "card.png"
OUT_DIR = Path(sys.argv[2] if len(sys.argv) > 2 else "output")
SEED = 11

# Тональный диапазон
BG_CENTER = "#F2E0C8"      # светлый бежевый в центре
BG_EDGE = "#D4A574"        # карамель по краям и в углах
BG_BOTTOM = "#B98355"      # низ карточки
BOTTOM_MIX = 0.75          # насколько низ уходит в BG_BOTTOM

# Свет
LIGHT_POS = (0.30, 0.27)   # доля ширины/высоты (левый верх, за верхом расчёски)
LIGHT_RADIUS = 0.70        # доля ширины
LIGHT_GAIN = 0.11          # +11% яркости в центре пятна

# Виньетка
VIGNETTE = 0.17            # затемнение краёв
VIGNETTE_BOTTOM_CORNERS = 0.20  # дополнительно — нижние углы (итог не выше ~20%)

# Складки (диагональ от левого верхнего к правому нижнему)
FOLD_COUNT = 4
FOLD_RIDGE = 0.07          # гребни светлее на 7%
FOLD_VALLEY = 0.11         # впадины темнее на 11%
FOLD_WIDTH = (55, 115)     # ширина складки, px
FOLD_CALM_UNDER_TEXT = 0.35  # множитель складок под текстом

# Мелкое зерно ткани (матовый лён)
GRAIN_STRENGTH = 0.022
GRAIN_MAX = 0.035

# Тени (цвет общий)
SHADOW_COLOR = "#5E3B22"
CONTACT_OPACITY = 0.50
CONTACT_BLUR = 10          # px
CONTACT_OFFSET = 5         # px от света
CONTACT_FROM_Y = 950       # контактная тень только у нижней (лежащей) части
CAST_OPACITY = 0.25
CAST_BLUR = 50             # px
CAST_DISTANCE = 55         # px от света

# Отражённый свет на металле
REFLECT_COLOR = "#C98A52"
REFLECT_STRENGTH = 0.14

# Масштаб товара
PRODUCT_SCALE = 1.08       # 1.05–1.10; уменьшается сам, если залезает на тексты
PRODUCT_ANCHOR = (720, 1010)
SAFE_GAP = 18              # мин. зазор до текстов, px

# Текст
TEAL_NEW = "#3E9C94"
AUTO_CONTRAST = True       # затемнить бирюзовый до прохождения WCAG
GLASS_UNDER_DESC = True    # «стекло» под «дешеддер + колтунорез»
GLASS_BOX = (1078, 1140, 1432, 1278)
GLASS_OPACITY = 0.35
GLASS_RADIUS = 26
GLASS_STROKE = "#FFFFFF"
GLASS_STROKE_OPACITY = 0.7

# Линия выноски (рисуется заново поверх товара, т.к. товар масштабируется)
LINE_H = (1019.0, 833.4, 1254.6, 838.3)
LINE_V = (1249.6, 833.4, 1254.6, 965.6)

TEXT_BOXES_CALM = [(370, 140, 1140, 285), (300, 310, 1200, 410), (60, 490, 425, 820),
                   (1080, 990, 1440, 1280), (50, 1610, 1085, 1870)]
# =========================================================


def smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def tonal_base(h, w, s):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    nx, ny = (x / w - 0.5) * 2, (y / h - 0.5) * 2
    r = np.sqrt(nx ** 2 + ny ** 2) / np.sqrt(2)
    t = smooth((r - 0.22) / 0.78)[..., None]
    bg = hex2rgb(BG_CENTER) * (1 - t) + hex2rgb(BG_EDGE) * t
    tb = smooth((y / h - 0.62) / 0.38)[..., None] * BOTTOM_MIX
    bg = bg * (1 - tb) + hex2rgb(BG_BOTTOM) * tb
    return bg, x, y


def light_and_vignette(x, y, w, h):
    lx, ly = LIGHT_POS[0] * w, LIGHT_POS[1] * h
    d = np.sqrt((x - lx) ** 2 + (y - ly) ** 2) / (LIGHT_RADIUS * w)
    light = 1 + LIGHT_GAIN * np.exp(-2.2 * d ** 2)
    nx, ny = (x / w - 0.5) * 2, (y / h - 0.5) * 2
    edge = smooth((np.maximum(np.abs(nx) ** 3 + np.abs(ny) ** 3, 0) ** (1 / 3) - 0.55) / 0.45)
    bc = smooth(np.sqrt((np.abs(nx) - 1) ** 2 + (ny - 1) ** 2) * -1.6 + 1)  # нижние углы
    vig = 1 - VIGNETTE * edge - (VIGNETTE_BOTTOM_CORNERS - 0.0) * bc * 0.5
    return light * np.clip(vig, 1 - 0.2, 1)


def folds(x, y, w, h, s, rng, calm):
    d = np.array([w, h], np.float32) / np.hypot(w, h)       # вдоль диагонали
    u = x * d[1] - y * d[0]                                   # поперёк складок
    v = x * d[0] + y * d[1]                                   # вдоль складок
    L = np.hypot(w, h)
    wob = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 140 * s)
    u = u + wob / (wob.std() + 1e-6) * 22 * s
    umin, umax = u.min(), u.max()
    centers = np.linspace(umin, umax, FOLD_COUNT + 2)[1:-1]
    centers += rng.uniform(-0.12, 0.12, FOLD_COUNT) * (umax - umin) / (FOLD_COUNT + 1)
    H = np.zeros_like(x)
    for c in centers:
        wd = rng.uniform(*FOLD_WIDTH) * s
        amp = rng.uniform(0.75, 1.0)
        # огибающая вдоль складки: затухает к концам
        v0, v1 = rng.uniform(-0.1, 0.25) * L, rng.uniform(0.75, 1.1) * L
        env = smooth((v - v0) / (0.25 * L)) * smooth((v1 - v) / (0.25 * L))
        ridge = np.exp(-((u - c) / wd) ** 2)
        valley = np.exp(-((u - c - 1.15 * wd) / (0.7 * wd)) ** 2)  # резче, со стороны от света
        H += amp * env * (ridge - valley)
    H = H / (np.abs(H).max() + 1e-6)
    shade = np.where(H > 0, FOLD_RIDGE * H, FOLD_VALLEY * H)
    # направленный член: склон к свету чуть светлее
    gy, gx = np.gradient(cv2.GaussianBlur(H, (0, 0), 6 * s))
    ldir = np.array([-1.0, -1.0]) / np.sqrt(2)
    dirt = gx * ldir[0] + gy * ldir[1]
    shade += 0.035 * dirt / (np.abs(dirt).max() + 1e-6)
    return 1 + shade * calm, H


def calm_mask(h, w, sx, sy, s):
    m = np.zeros((h, w), np.float32)
    for b in TEXT_BOXES_CALM:
        b = scale_box(b, sx, sy)
        m[b[1]:b[3], b[0]:b[2]] = 1
    m = cv2.GaussianBlur(m, (0, 0), 35 * s)
    m = np.clip(m * 1.6, 0, 1)
    return 1 - (1 - FOLD_CALM_UNDER_TEXT) * m


def fill_rect_aa(h, w, rect, ss=4):
    x0, y0, x1, y1 = rect
    X0, Y0 = int(np.floor(x0)) - 1, int(np.floor(y0)) - 1
    X1, Y1 = int(np.ceil(x1)) + 1, int(np.ceil(y1)) + 1
    img = Image.new("L", ((X1 - X0) * ss, (Y1 - Y0) * ss), 0)
    ImageDraw.Draw(img).rectangle([(x0 - X0) * ss, (y0 - Y0) * ss, (x1 - X0) * ss - 1, (y1 - Y0) * ss - 1], fill=255)
    small = np.asarray(img.resize((X1 - X0, Y1 - Y0), Image.BOX), np.float32) / 255
    out = np.zeros((h, w), np.float32)
    out[Y0:Y1, X0:X1] = small
    return out


def rounded_aa(h, w, box, rad, stroke=0.0, ss=4):
    x0, y0, x1, y1 = box
    pad = 4
    W, H = x1 - x0 + 2 * pad, y1 - y0 + 2 * pad
    fill = Image.new("L", (W * ss, H * ss), 0)
    line = Image.new("L", (W * ss, H * ss), 0)
    r = [pad * ss, pad * ss, (W - pad) * ss, (H - pad) * ss]
    ImageDraw.Draw(fill).rounded_rectangle(r, rad * ss, fill=255)
    if stroke:
        ImageDraw.Draw(line).rounded_rectangle(r, rad * ss, outline=255, width=max(1, int(stroke * ss)))
    f = np.zeros((h, w), np.float32)
    l = np.zeros((h, w), np.float32)
    f[y0 - pad:y1 + pad, x0 - pad:x1 + pad] = np.asarray(fill.resize((W, H), Image.BOX), np.float32) / 255
    l[y0 - pad:y1 + pad, x0 - pad:x1 + pad] = np.asarray(line.resize((W, H), Image.BOX), np.float32) / 255
    return f, l


def transform_product(F, A, scale, anchor):
    h, w = A.shape
    ax, ay = anchor
    M = np.float32([[scale, 0, ax * (1 - scale)], [0, scale, ay * (1 - scale)]])
    pre = cv2.warpAffine(F * A[..., None], M, (w, h), flags=cv2.INTER_CUBIC)
    a2 = np.clip(cv2.warpAffine(A, M, (w, h), flags=cv2.INTER_CUBIC), 0, 1)
    F2 = np.clip(pre / np.maximum(a2, 1e-3)[..., None], 0, 1)
    return F2, a2


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img = Image.open(SRC).convert("RGB")
    w, h = img.size
    sx, sy = w / pc.REF_SIZE[0], h / pc.REF_SIZE[1]
    s = (sx + sy) / 2
    P = np.asarray(img, np.float32) / 255
    rng = np.random.default_rng(SEED)

    print("вырезаю товар (rembg)…")
    A = pc.product_alpha(img)
    B = pc.old_background(P, A)

    heading = scale_box(pc.INK_BOXES[0][:4], sx, sy)
    hp = P[heading[1]:heading[3], heading[0]:heading[2]].reshape(-1, 3)
    dark = np.median(hp[hp.mean(1) < 0.35], 0)

    boxes = [scale_box(b, sx, sy) for b in pc.INK_BOXES]
    ink_a, ink_k = pc.unmix_ink(P, B, A, boxes, dark)
    ob = scale_box(pc.SUB_OLD_BOX, sx, sy)
    ink_a[ob[1]:ob[3], ob[0]:ob[2]] = 0          # старое «для животных»
    lb = boxes[2]
    ink_a[lb[1]:lb[3], lb[0]:lb[2]] = 0          # линию выноски нарисуем заново

    # --- товар: очистка каймы, убрать кусок линии с ручки, масштаб ---
    lh = [v * k for v, k in zip(LINE_H, (sx, sy, sx, sy))]
    lv = [v * k for v, k in zip(LINE_V, (sx, sy, sx, sy))]
    line_a = np.maximum(fill_rect_aa(h, w, lh), fill_rect_aa(h, w, lv))
    line_m = (line_a > 0.02).astype(np.uint8)
    # rembg приклеил к ручке кусок линии и тени: вокруг линии всё, что похоже на фон или линию, — не товар
    win = cv2.dilate(line_m, np.ones((int(60 * s) | 1,) * 2, np.uint8)) > 0
    bgish = np.abs(P - B).max(2) < 0.10
    near_line = cv2.dilate(line_m, np.ones((int(9 * s) | 1,) * 2, np.uint8)) > 0
    A = np.where(win & (bgish | near_line), 0, A)
    # дырку от линии на ручке закрываем морфологией и дорисовываем цвет (inpaint)
    # линия горизонтальная: пиксель — товар, если товар и выше, и ниже полосы,
    # причём сдвиг идёт вдоль наклона края ручки (оценка по строкам над/под линией)
    yl = int((lh[1] + lh[3]) / 2)
    d = int(15 * s)
    xs_ = [np.where(A[yy, int(lh[0]):int(lh[2])] > 0.5)[0] for yy in (yl - 2 * d, yl + 2 * d)]
    slope = ((xs_[1].max() - xs_[0].max()) / (4 * d)) if all(len(v) for v in xs_) else 0.0
    sh = lambda a, k: cv2.warpAffine(a, np.float32([[1, 0, k * slope * d], [0, 1, k * d]]), (w, h))
    Ac = np.minimum(sh(A, 1), sh(A, -1))
    hole = (near_line & (Ac > 0.02)).astype(np.uint8)
    A = np.where(hole > 0, Ac, A)
    F = np.clip((P - (1 - A[..., None]) * B) / np.maximum(A, 0.05)[..., None], 0, 1)
    F = np.where((A > 0.97)[..., None], P, F)
    F = cv2.inpaint((F * 255).astype(np.uint8), hole, 7, cv2.INPAINT_TELEA).astype(np.float32) / 255

    # защищённая зона: тексты (до масштаба уже проверено, что товар их не касается)
    text_zone = np.zeros((h, w), np.uint8)
    for b in boxes[:2] + boxes[3:]:
        text_zone[b[1]:b[3], b[0]:b[2]] = 1
    sb = scale_box((300, 140, 1200, 410), sx, sy)
    text_zone[sb[1]:sb[3], sb[0]:sb[2]] = 1
    if GLASS_UNDER_DESC:
        gb = scale_box(GLASS_BOX, sx, sy)
        text_zone[gb[1]:gb[3], gb[0]:gb[2]] = 1
    text_zone = cv2.dilate(text_zone, np.ones((int(2 * SAFE_GAP * s) | 1,) * 2, np.uint8))
    anchor = (PRODUCT_ANCHOR[0] * sx, PRODUCT_ANCHOR[1] * sy)
    scale = PRODUCT_SCALE
    while True:
        F2, A2 = transform_product(F, A, scale, anchor)
        ys, xs = np.where(A2 > 0.1)
        inside = xs.min() > 10 * s and xs.max() < w - 10 * s and ys.min() > 10 * s and ys.max() < h - 10 * s
        hit = ((A2 > 0.1) & (text_zone > 0)).sum()
        if (hit == 0 and inside) or scale <= 1.0:
            break
        scale -= 0.01
    print(f"масштаб товара: ×{scale:.2f} (bbox x {xs.min()}–{xs.max()}, y {ys.min()}–{ys.max()})")

    # --- фон ---
    bg, x, y = tonal_base(h, w, s)
    lv_ = light_and_vignette(x, y, w, h)
    calm = calm_mask(h, w, sx, sy, s)
    fold_mul, H = folds(x, y, w, h, s, rng, calm)
    pc.FOLD_STRENGTH = 0.0
    pc.WEAVE_STRENGTH = GRAIN_STRENGTH
    pc.FABRIC_MAX_CONTRAST = GRAIN_MAX
    weave = pc.fabric(h, w, s, rng) - 1
    noise = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 0.8 * s)
    noise = noise / noise.std() * weave.std()
    grain = 1 + 0.45 * weave + 0.75 * noise          # матовое зерно поверх нитей
    bg = np.clip(bg * (lv_ * fold_mul * grain)[..., None], 0, 1)
    out = bg.copy()

    # --- тени (от света) ---
    lx, ly = LIGHT_POS[0] * w, LIGHT_POS[1] * h
    ys_, xs_ = np.where(A2 > 0.5)
    cx, cy = xs_.mean(), ys_.mean()
    dv = np.array([cx - lx, cy - ly], np.float32)
    dv /= np.linalg.norm(dv)
    shc = hex2rgb(SHADOW_COLOR)

    def shifted(a, dist, blur):
        M = np.float32([[1, 0, dv[0] * dist * s], [0, 1, dv[1] * dist * s]])
        return cv2.GaussianBlur(cv2.warpAffine(a, M, (w, h)), (0, 0), blur * s / 2)

    cast = shifted(A2, CAST_DISTANCE, CAST_BLUR)
    out = lerp(out, shc, np.clip(cast * CAST_OPACITY, 0, 1))
    low = A2 * smooth((y - CONTACT_FROM_Y * sy) / (150 * sy))
    contact = shifted(low, CONTACT_OFFSET, CONTACT_BLUR)
    out = lerp(out, shc, np.clip(contact * CONTACT_OPACITY * 1.4, 0, CONTACT_OPACITY))

    # --- плашка грумеров ---
    pm = pc.plaque_mask(ink_a, scale_box(pc.PLAQUE_BOX, sx, sy))
    out = lerp(out, hex2rgb(pc.PLAQUE_FILL), pm * pc.PLAQUE_OPACITY)

    # --- стекло под «дешеддер + колтунорез» ---
    if GLASS_UNDER_DESC:
        gb = scale_box(GLASS_BOX, sx, sy)
        gf, gl = rounded_aa(h, w, gb, GLASS_RADIUS * s, stroke=1.6 * s)
        out = lerp(out, np.ones(3, np.float32), gf * GLASS_OPACITY)
        out = lerp(out, hex2rgb(GLASS_STROKE), gl * GLASS_STROKE_OPACITY)

    # --- цвет бирюзы с проверкой контраста ---
    def bg_at(box, src):
        b = scale_box(box, sx, sy)
        return src[b[1]:b[3], b[0]:b[2]].reshape(-1, 3).mean(0)

    teal_checks = {
        "«для кошек и собак»": (pc.SUB_OLD_BOX, True),
        "«Видеообзор»": ((70, 750, 420, 810), True),
        "«дешеддер + колтунорез»": ((1090, 1150, 1420, 1265), False),
    }
    plaque_bg = bg_at((250, 1690, 1030, 1800), out)
    teal = hex2rgb(TEAL_NEW)
    if AUTO_CONTRAST:
        for k in np.linspace(1, 0.3, 141):
            c = teal * k
            ok = all(contrast(c, bg_at(b, out)) >= (3.0 if lg else 4.5) for b, lg in teal_checks.values())
            ok = ok and contrast(c, plaque_bg) >= 4.5
            if ok:
                break
        teal = c
    teal_hex = "#%02X%02X%02X" % tuple((teal * 255).round().astype(int))

    pre_text = out.copy()
    # --- надписи ---
    ink_col = np.where((ink_k == 0)[..., None], teal, dark)
    k = np.clip((ink_a - 0.12) / 0.3, 0, 1)
    warm = lerp(np.broadcast_to(shc, ink_col.shape), ink_col, k)
    ink_col = np.where((ink_k == 1)[..., None], warm, ink_col)
    out = out * (1 - ink_a[..., None]) + ink_col * ink_a[..., None]
    sub = pc.render_sub(h, w, s)
    out = lerp(out, teal, sub)

    # --- товар + отражённый свет на металле ---
    hsv = cv2.cvtColor((F2 * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    sat, val = hsv[..., 1] / 255, hsv[..., 2] / 255
    metal = (A2 > 0.5) & (sat < 0.18) & (val > 0.28)
    metal = cv2.GaussianBlur(metal.astype(np.float32), (0, 0), 1.5)
    facing = 0.5 + 0.5 * smooth((y - 900 * sy) / (500 * sy))   # снизу ткань отсвечивает сильнее
    refl = metal * facing * REFLECT_STRENGTH
    screen = 1 - (1 - F2) * (1 - hex2rgb(REFLECT_COLOR))
    F2 = lerp(F2, screen, refl)
    out = lerp(out, np.clip(F2, 0, 1), A2)

    # --- линия выноски поверх ---
    out = lerp(out, dark, line_a)

    res = Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8))
    res.save(OUT_DIR / "card_peach_v2.png")
    res.save(OUT_DIR / "card_peach_v2.jpg", quality=92, subsampling=0)
    pw = 300
    prev = res.resize((pw, round(h * pw / w)), Image.LANCZOS)
    prev.save(OUT_DIR / "card_peach_v2_preview.png")

    # до/после
    cw = 600
    ch = round(h * cw / w)
    cmp_ = Image.new("RGB", (cw * 2 + 30, ch + 60), "white")
    cmp_.paste(img.resize((cw, ch), Image.LANCZOS), (0, 60))
    cmp_.paste(res.resize((cw, ch), Image.LANCZOS), (cw + 30, 60))
    dr = ImageDraw.Draw(cmp_)
    from PIL import ImageFont
    f = ImageFont.truetype(str(pc.FONT_PATH), 34)
    dr.text((cw // 2, 30), "до", font=f, fill=(40, 40, 40), anchor="mm")
    dr.text((cw + 30 + cw // 2, 30), "после", font=f, fill=(40, 40, 40), anchor="mm")
    cmp_.save(OUT_DIR / "card_peach_v2_compare.png")

    # --- отчёт ---
    print(f"бирюзовый: {TEAL_NEW} → {teal_hex}" if AUTO_CONTRAST else f"бирюзовый: {teal_hex}")
    pv = np.asarray(prev.convert("L"), np.float32)
    zone = pv[int(pv.shape[0] * .765):int(pv.shape[0] * .80), int(pw * .05):int(pw * .95)]
    print(f"складки в превью 300px: размах яркости фона p5..p95 = {np.ptp(np.percentile(zone, [5, 95])):.1f} из 255")
    print("\nконтраст WCAG (крупный ≥3.0, мелкий ≥4.5):")
    rows = [("«РАСЧЕСКА»", dark, bg_at(pc.INK_BOXES[0][:4], bg), True),
            ("«2 в 1»", dark, bg_at((1100, 1000, 1390, 1110), bg), True)]
    for name, (b, lg) in teal_checks.items():
        rows.append((name, teal, bg_at(b, pre_text), lg))
    rows.append(("текст в плашке", teal, plaque_bg, False))
    for name, fg, bgc, lg in rows:
        c = contrast(fg, bgc)
        need = 3.0 if lg else 4.5
        print(f"  {name:28s} {c:4.2f}:1  {'OK' if c >= need else 'НИЖЕ НОРМЫ'} (порог {need})")
    print("\nготово:", ", ".join(str(OUT_DIR / n) for n in [
        "card_peach_v2.png", "card_peach_v2.jpg", "card_peach_v2_preview.png", "card_peach_v2_compare.png"]))


if __name__ == "__main__":
    main()
