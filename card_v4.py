#!/usr/bin/env python3
"""v4: как v3, без «Видеообзора», + строка «Двусторонняя, нержавеющая сталь»,
классический щит, подписи крупнее и жирнее, подзаголовок плотнее и выше.

Запуск:  python3 card_v4.py [card.png] [output_dir]
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import peach_card as pc
from peach_card import hex2rgb, lerp, contrast

# ======================= ПАРАМЕТРЫ =======================
SRC = sys.argv[1] if len(sys.argv) > 1 else "card.png"
OUT_DIR = Path(sys.argv[2] if len(sys.argv) > 2 else "output")
K_TEXT = 1.18              # увеличение всех надписей
K_PROD = 1.18              # увеличение расчёски
MIN_GAP = 20               # мин. зазор текст ↔ товар, px

# Положение блоков (px, холст 1500×2000)
HEAD_TOP = 130
SUB_GAP = 42               # от низа «РАСЧЕСКА» до верха подзаголовка (было 80)
SUB_SPACE = 0.70           # ширина пробела в подзаголовке (1.0 = обычная)
TAG_TEXT = "Двусторонняя, нержавеющая сталь"
TAG_SIZE = 64              # меньше подзаголовка (93), крупнее подписей (54)
TAG_WEIGHT = b"SemiBold"
TAG_GAP = 34               # от низа подзаголовка до верха строки
TAG_COLOR = None           # None = тёмно-серый как у заголовка; "teal" = бирюзовый
PROD_TOP = 470
PROD_CX = 750
RIGHT_TOP = 985            # верх «2 в 1»
RIGHT_CX = 1262            # центр блока «2 в 1 / дешеддер + колтунорез»
LINE_Y = 834               # горизонталь выноски (на бирюзовой ручке)
LINE_X0 = 1095
LINE_W = 5.9

# Значки
BADGE_TOP = 1648
BADGE_D = 115              # диаметр круга
BADGE_CAPTION_GAP = 22
CAPTION_SIZE = 54          # было 48
CAPTION_LH = 62            # межстрочный
CAPTION_WEIGHT = b"Bold"   # было SemiBold
CAPTION_COLOR = None       # None = тёмно-серый как у заголовка
COL_CENTERS = (265, 750, 1235)
COL_WIDTH = 480            # макс. ширина подписи
CAPTIONS = ["Рекомендовано грумерами", "Не травмирует кожу", "Здоровая шелковистая шерсть"]
ICONS = ["check", "shield", "heart"]

TEAL = None                # None = исходный бирюзовый (замер)
SUB_TEXT = "для кошек и собак"

# Боксы элементов на исходнике
BOX_HEAD = (380, 150, 1135, 280)
BOX_VIDEO = (70, 495, 420, 815)
BOX_2V1 = (1095, 995, 1385, 1112)
BOX_DESC = (1080, 1150, 1425, 1275)
OLD_LINE_H = (1019.0, 833.4, 1254.6, 838.3)
OLD_LINE_V = (1249.6, 833.4, 1254.6, 965.6)
# =========================================================


def layer_from(ink_a, ink_col, box):
    x0, y0, x1, y1 = box
    return ink_a[y0:y1, x0:x1].copy(), ink_col[y0:y1, x0:x1].copy()


def scale_layer(a, c, k):
    h, w = a.shape
    size = (round(w * k), round(h * k))
    a2 = np.clip(cv2.resize(a, size, interpolation=cv2.INTER_CUBIC), 0, 1)
    pre = cv2.resize(c * a[..., None], size, interpolation=cv2.INTER_CUBIC)
    c2 = np.clip(pre / np.maximum(a2, 1e-3)[..., None], 0, 1)
    return a2, c2


def tight(a, c, thr=0.02):
    ys, xs = np.where(a > thr)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return a[y0:y1, x0:x1], c[y0:y1, x0:x1]


def paste(out, mask_acc, a, c, x, y):
    h, w = a.shape
    x, y = int(round(x)), int(round(y))
    out[y:y + h, x:x + w] = lerp(out[y:y + h, x:x + w], c, a)
    mask_acc[y:y + h, x:x + w] = np.maximum(mask_acc[y:y + h, x:x + w], a)


def rect_aa(h, w, x0, y0, x1, y1, ss=4):
    img = Image.new("L", (w, h))
    big = Image.new("L", (int((x1 - x0 + 4) * ss), int((y1 - y0 + 4) * ss)))
    X0, Y0 = int(x0) - 2, int(y0) - 2
    ImageDraw.Draw(big).rectangle([(x0 - X0) * ss, (y0 - Y0) * ss, (x1 - X0) * ss - 1, (y1 - Y0) * ss - 1], fill=255)
    img.paste(big.resize((big.width // ss, big.height // ss), Image.BOX), (X0, Y0))
    return np.asarray(img, np.float32) / 255


def icon_layer(kind, D, ss=6):
    S = D * ss
    im = Image.new("L", (S, S), 0)
    gl = Image.new("L", (S, S), 0)
    ImageDraw.Draw(im).ellipse([0, 0, S - 1, S - 1], fill=255)
    d = ImageDraw.Draw(gl)
    c = S / 2
    if kind == "check":
        pts = [(c - 0.24 * S, c + 0.01 * S), (c - 0.07 * S, c + 0.18 * S), (c + 0.25 * S, c - 0.16 * S)]
        wd = int(0.085 * S)
        d.line(pts, fill=255, width=wd, joint="curve")
        for p in (pts[0], pts[-1]):
            d.ellipse([p[0] - wd / 2, p[1] - wd / 2, p[0] + wd / 2, p[1] + wd / 2], fill=255)
    elif kind == "shield":
        # классический щит: плоский верх, прямые борта, дуги к острому низу; белый контур + белая галочка
        top, bot, hw = c - 0.28 * S, c + 0.32 * S, 0.24 * S
        side = top + 0.25 * S                       # до сюда борта прямые
        t = np.linspace(0, 1, 80)
        right = [(c + hw * (1 - tt ** 1.6), side + (bot - side) * (1 - (1 - tt) ** 1.15)) for tt in t]
        poly = [(c - hw, top), (c + hw, top)] + right + [(2 * c - x, y) for x, y in right[::-1]]
        sw = int(0.06 * S)
        d.polygon(poly, fill=255)
        # внутренность — снова прозрачна (виден бирюзовый круг), остаётся контур толщиной sw
        inner = []
        cx_, cy_ = c, (top + bot) / 2 - 0.02 * S
        for x, y in poly:
            vx, vy = x - cx_, y - cy_
            n = np.hypot(vx, vy)
            inner.append((x - vx / n * sw * 1.15, y - vy / n * sw * 1.15))
        d.polygon(inner, fill=0)
        ck = [(c - 0.115 * S, c - 0.005 * S), (c - 0.03 * S, c + 0.08 * S), (c + 0.12 * S, c - 0.085 * S)]
        wd = int(0.058 * S)
        d.line(ck, fill=255, width=wd, joint="curve")
        for p in (ck[0], ck[-1]):
            d.ellipse([p[0] - wd / 2, p[1] - wd / 2, p[0] + wd / 2, p[1] + wd / 2], fill=255)
    elif kind == "heart":
        t = np.linspace(0, 2 * np.pi, 400)
        x = 16 * np.sin(t) ** 3
        y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
        sc = 0.0175 * S
        d.polygon(list(zip(c + x * sc, c + 0.02 * S + y * sc)), fill=255)
    a = np.asarray(im.resize((D, D), Image.LANCZOS), np.float32) / 255
    g = np.asarray(gl.resize((D, D), Image.LANCZOS), np.float32) / 255
    return a, g


def render_line(h, w, text, size, weight, tracking, space_scale, top, cx=750, cap_top=False):
    """Строка по центру; space_scale сужает пробелы; top — верх строчных (или прописных при cap_top)."""
    f = ImageFont.truetype(str(pc.FONT_PATH), int(round(size)))
    f.set_variation_by_name(weight)
    adv = [f.getlength(ch) * (space_scale if ch == " " else 1.0) for ch in text]
    total = sum(adv) + tracking * (len(text) - 1)
    ref = f.getbbox("Д" if cap_top else "для")[1]
    layer = Image.new("L", (w, h), 0)
    dr = ImageDraw.Draw(layer)
    x = cx - total / 2
    for ch, a in zip(text, adv):
        if ch != " ":
            dr.text((x, top - ref), ch, font=f, fill=255)
        x += a + tracking
    return np.asarray(layer, np.float32) / 255


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img = Image.open(SRC).convert("RGB")
    w, h = img.size
    assert (w, h) == (1500, 2000), "раскладка рассчитана на 1500×2000"
    P = np.asarray(img, np.float32) / 255

    print("вырезаю товар (rembg)…")
    A = pc.product_alpha(img)
    B = pc.old_background(P, A)

    hp = P[BOX_HEAD[1]:BOX_HEAD[3], BOX_HEAD[0]:BOX_HEAD[2]].reshape(-1, 3)
    dark = np.median(hp[hp.mean(1) < 0.35], 0)
    teal = hex2rgb(TEAL) if TEAL else hex2rgb(pc.TEAL_SRC)
    cap_col = hex2rgb(CAPTION_COLOR) if CAPTION_COLOR else dark

    boxes = [b + (0.05,) for b in (BOX_HEAD, BOX_VIDEO, BOX_2V1, BOX_DESC)]
    ink_a, ink_k = pc.unmix_ink(P, B, A, boxes, dark)
    ink_col = np.where((ink_k == 0)[..., None], hex2rgb(pc.TEAL_SRC), dark)
    # серые полутени у букв (если есть) остаются серыми — фон тот же

    # --- товар: убрать с ручки кусок старой линии выноски ---
    line_m = np.maximum(rect_aa(h, w, *OLD_LINE_H), rect_aa(h, w, *OLD_LINE_V)) > 0.02
    line_m = line_m.astype(np.uint8)
    win = cv2.dilate(line_m, np.ones((61, 61), np.uint8)) > 0
    bgish = np.abs(P - B).max(2) < 0.10
    near = cv2.dilate(line_m, np.ones((9, 9), np.uint8)) > 0
    A = np.where(win & (bgish | near), 0, A)
    yl, d = int((OLD_LINE_H[1] + OLD_LINE_H[3]) / 2), 15
    xs_ = [np.where(A[yy, int(OLD_LINE_H[0]):int(OLD_LINE_H[2])] > 0.5)[0] for yy in (yl - 2 * d, yl + 2 * d)]
    slope = (xs_[1].max() - xs_[0].max()) / (4 * d)
    sh = lambda a, k: cv2.warpAffine(a, np.float32([[1, 0, k * slope * d], [0, 1, k * d]]), (w, h))
    Ac = np.minimum(sh(A, 1), sh(A, -1))
    hole = (near & (Ac > 0.02)).astype(np.uint8)
    A = np.where(hole > 0, Ac, A)
    F = np.clip((P - (1 - A[..., None]) * B) / np.maximum(A, 0.05)[..., None], 0, 1)
    F = np.where((A > 0.97)[..., None], P, F)
    F = cv2.inpaint((F * 255).astype(np.uint8), hole, 7, cv2.INPAINT_TELEA).astype(np.float32) / 255

    # родная мягкая тень товара на фоне (множитель), чтобы не выглядел наклеенным
    shade = np.clip((P / np.maximum(B, 1e-3)).mean(2), 0, 1)
    keep = cv2.dilate((A > 0.02).astype(np.uint8), np.ones((121, 121), np.uint8)).astype(np.float32)
    txt = np.zeros((h, w), np.float32)
    for b in (BOX_HEAD, BOX_VIDEO, BOX_2V1, BOX_DESC, (50, 1610, 1085, 1870), (380, 300, 1110, 420), (1000, 815, 1270, 980)):
        txt[b[1]:b[3], b[0]:b[2]] = 1
    # в зонах текста убираем только сами буквы/линии, а тень под ними восстанавливаем inpaint'ом
    letters = (txt > 0) & (np.abs(P - B).max(2) > 0.035)
    letters = cv2.dilate(letters.astype(np.uint8), np.ones((7, 7), np.uint8))
    sm = cv2.resize(shade, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    lm = cv2.resize(letters, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST)
    lm = cv2.dilate(lm, np.ones((3, 3), np.uint8))
    fill = cv2.inpaint((sm * 255).astype(np.uint8), lm, 5, cv2.INPAINT_TELEA).astype(np.float32) / 255
    fill = cv2.resize(fill, (w, h), interpolation=cv2.INTER_CUBIC)
    shade = np.where(letters > 0, fill, shade)
    shade = 1 - (1 - np.clip(shade, 0, 1)) * keep * (A < 0.02)
    shade = cv2.GaussianBlur(shade, (0, 0), 2)

    ys, xs = np.where(A > 0.1)
    bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max(), ys.max()
    cx = (bx0 + bx1) / 2
    M = np.float32([[K_PROD, 0, PROD_CX - cx * K_PROD], [0, K_PROD, PROD_TOP - by0 * K_PROD]])
    pre = cv2.warpAffine(F * A[..., None], M, (w, h), flags=cv2.INTER_CUBIC)
    A2 = np.clip(cv2.warpAffine(A, M, (w, h), flags=cv2.INTER_CUBIC), 0, 1)
    F2 = np.clip(pre / np.maximum(A2, 1e-3)[..., None], 0, 1)
    shade2 = cv2.warpAffine(shade, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=1.0)
    ys, xs = np.where(A2 > 0.1)
    print(f"расчёска ×{K_PROD}: x {xs.min()}–{xs.max()}, y {ys.min()}–{ys.max()}")

    # --- холст: исходный фон без надписей и товара ---
    out = B.copy()
    # родное зерно фона: замеряем на чистом участке и генерируем такое же (без «призраков» старых надписей)
    patch = P[1300:1550, 1150:1450]
    sigma = (patch - cv2.GaussianBlur(patch, (0, 0), 3)).std()
    noise = np.random.default_rng(3).normal(0, 1, (h, w)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 0.6)
    out += (noise / noise.std() * sigma)[..., None]
    out *= shade2[..., None]
    occ = np.zeros((h, w), np.float32)                   # маска всех надписей

    # заголовок
    a, c = tight(*scale_layer(*layer_from(ink_a, ink_col, BOX_HEAD), K_TEXT))
    paste(out, occ, a, c, 750 - a.shape[1] / 2, HEAD_TOP)
    head_bottom = HEAD_TOP + a.shape[0]

    # подзаголовок — шрифтом, ×K_TEXT, без уменьшения
    sub = render_line(h, w, SUB_TEXT, 79 * K_TEXT, b"Bold", 1.4 * K_TEXT, SUB_SPACE, head_bottom + SUB_GAP)
    out = lerp(out, teal, sub)
    occ = np.maximum(occ, sub)
    ys_, xs_ = np.where(sub > 0.02)
    sub_box = (xs_.min(), ys_.min(), xs_.max(), ys_.max())
    print(f"подзаголовок: кегль {79 * K_TEXT:.0f}, ширина {xs_.max() - xs_.min()}px, y {ys_.min()}–{ys_.max()}")

    # «Двусторонняя, нержавеющая сталь» (на месте «Видеообзора», «Видеообзор» убран)
    tag = render_line(h, w, TAG_TEXT, TAG_SIZE, TAG_WEIGHT, 0, 1.0, sub_box[3] + TAG_GAP, cap_top=True)
    tag_col = teal if TAG_COLOR == "teal" else (hex2rgb(TAG_COLOR) if TAG_COLOR else dark)
    out = lerp(out, tag_col, tag)
    occ = np.maximum(occ, tag)
    ys_, xs_ = np.where(tag > 0.02)
    vid_box = (xs_.min(), ys_.min(), xs_.max(), ys_.max())
    print(f"строка «{TAG_TEXT}»: кегль {TAG_SIZE}, ширина {xs_.max() - xs_.min()}px, y {ys_.min()}–{ys_.max()}")
    if xs_.min() < 30 or xs_.max() > w - 30:
        print("  ВНИМАНИЕ: строка почти у края")

    # «2 в 1» и «дешеддер + колтунорез» (сохраняем взаимное положение)
    a1, c1 = tight(*scale_layer(*layer_from(ink_a, ink_col, BOX_2V1), K_TEXT))
    a2, c2 = tight(*scale_layer(*layer_from(ink_a, ink_col, BOX_DESC), K_TEXT))
    gap = (1164 - 1100) * K_TEXT
    x1 = RIGHT_CX - a1.shape[1] / 2
    paste(out, occ, a1, c1, x1, RIGHT_TOP)
    dx = ((1093 + 1411) / 2 - (1109 + 1367) / 2) * K_TEXT
    x2 = RIGHT_CX + dx - a2.shape[1] / 2
    y2 = RIGHT_TOP + a1.shape[0] + gap
    paste(out, occ, a2, c2, x2, y2)
    right_box = (int(min(x1, x2)), RIGHT_TOP, int(max(x1 + a1.shape[1], x2 + a2.shape[1])), int(y2 + a2.shape[0]))

    # --- значки ---
    f = ImageFont.truetype(str(pc.FONT_PATH), CAPTION_SIZE)
    f.set_variation_by_name(CAPTION_WEIGHT)
    problems = []
    cap = Image.new("L", (w, h), 0)
    dr = ImageDraw.Draw(cap)
    asc = f.getbbox("Нн")[1]
    badge_bottom = 0
    for cx_, text, kind in zip(COL_CENTERS, CAPTIONS, ICONS):
        ia, ig = icon_layer(kind, BADGE_D)
        x0, y0 = int(cx_ - BADGE_D / 2), BADGE_TOP
        out[y0:y0 + BADGE_D, x0:x0 + BADGE_D] = lerp(out[y0:y0 + BADGE_D, x0:x0 + BADGE_D], teal, ia)
        out[y0:y0 + BADGE_D, x0:x0 + BADGE_D] = lerp(out[y0:y0 + BADGE_D, x0:x0 + BADGE_D], np.ones(3, np.float32), ig * ia)
        occ[y0:y0 + BADGE_D, x0:x0 + BADGE_D] = np.maximum(occ[y0:y0 + BADGE_D, x0:x0 + BADGE_D], ia)
        # перенос по словам в ширину колонки, кегль не трогаем
        lines, cur = [], ""
        for word in text.split():
            t = (cur + " " + word).strip()
            if f.getlength(t) <= COL_WIDTH:
                cur = t
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
        for wd in lines:
            if f.getlength(wd) > COL_WIDTH:
                problems.append(f"«{wd}» шире колонки ({f.getlength(wd):.0f} > {COL_WIDTH})")
        y = BADGE_TOP + BADGE_D + BADGE_CAPTION_GAP
        for ln in lines:
            dr.text((cx_ - f.getlength(ln) / 2, y - asc), ln, font=f, fill=255)
            y += CAPTION_LH
        badge_bottom = max(badge_bottom, y - CAPTION_LH + f.getbbox("Ну")[3] - asc)
        print(f"значок «{text}»: {len(lines)} стр. → {' / '.join(lines)}")
    capa = np.asarray(cap, np.float32) / 255
    out = lerp(out, cap_col, capa)
    occ = np.maximum(occ, capa)
    if badge_bottom > h - 20:
        problems.append(f"подписи значков доходят до y={badge_bottom}, поле снизу < 20 px")

    # --- проверка: текст не на товаре ---
    near_prod = cv2.dilate((A2 > 0.1).astype(np.uint8), np.ones((2 * MIN_GAP + 1,) * 2, np.uint8))
    for name, b in [("подзаголовок", sub_box), ("Двусторонняя…", vid_box), ("2 в 1 / дешеддер", right_box)]:
        hit = near_prod[b[1]:b[3], b[0]:b[2]] & (occ[b[1]:b[3], b[0]:b[2]] > 0.1)
        if hit.any():
            problems.append(f"{name} ближе {MIN_GAP}px к товару")
    hit = near_prod[BADGE_TOP:, :].astype(bool) & (occ[BADGE_TOP:, :] > 0.1)
    if hit.any():
        problems.append("значки касаются товара")
    ys, xs = np.where(A2 > 0.1)
    if xs.min() < 5 or xs.max() > w - 6 or ys.max() > h - 6:
        problems.append("расчёска обрезана краем")

    # --- товар и линия выноски ---
    out = lerp(out, F2, A2)
    corner_x = RIGHT_CX + (1252 - (1109 + 1367) / 2) * K_TEXT
    vy1 = RIGHT_TOP - (1007 - 966) * K_TEXT
    la = np.maximum(rect_aa(h, w, LINE_X0, LINE_Y - LINE_W / 2, corner_x + LINE_W / 2, LINE_Y + LINE_W / 2),
                    rect_aa(h, w, corner_x - LINE_W / 2, LINE_Y - LINE_W / 2, corner_x + LINE_W / 2, vy1))
    out = lerp(out, dark, la)

    res = Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8))
    res.save(OUT_DIR / "card_v4.png")
    cw = 600
    ch = round(h * cw / w)
    cmp_ = Image.new("RGB", (cw * 2 + 30, ch + 60), "white")
    cmp_.paste(Image.open(OUT_DIR / "card_v3.png").convert("RGB").resize((cw, ch), Image.LANCZOS), (0, 60))
    cmp_.paste(res.resize((cw, ch), Image.LANCZOS), (cw + 30, 60))
    fd = ImageFont.truetype(str(pc.FONT_PATH), 34)
    ImageDraw.Draw(cmp_).text((cw // 2, 30), "card_v3", font=fd, fill=(40, 40, 40), anchor="mm")
    ImageDraw.Draw(cmp_).text((cw + 30 + cw // 2, 30), "card_v4", font=fd, fill=(40, 40, 40), anchor="mm")
    cmp_.save(OUT_DIR / "card_v4_compare.png")

    print(f"контраст подписей {contrast(cap_col, B[1800, 750]):.2f}:1, "
          f"бирюзы на фоне {contrast(teal, B[400, 750]):.2f}:1")
    print("ПРОБЛЕМЫ:\n  " + "\n  ".join(problems) if problems else "всё влезло, наложений нет")
    print("готово:", OUT_DIR / "card_v4.png", OUT_DIR / "card_v4_compare.png")


if __name__ == "__main__":
    main()
