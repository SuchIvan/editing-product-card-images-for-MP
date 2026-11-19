#!/usr/bin/env python3
"""v5: без «2 в 1»/выноски, «Двусторонняя, нержавеющая сталь» слева у ручки, тени,
значки крупнее, подзаголовок плотнее.

Запуск:  python3 card_v5.py [card.png] [output_dir]
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
SUB_GAP = 22               # от низа «РАСЧЕСКА» до верха подзаголовка (v4: 42)
SUB_SPACE = 0.55           # ширина пробела в подзаголовке (v4: 0.70)
TAG_LINES = ["Двусторонняя,", "нержавеющая сталь"]
TAG_X = 60                 # ≥ 40 от левого края
TAG_TOP = 520              # верх прописных первой строки
MARGIN = 40                # мин. отступ от краёв карточки
PROD_TOP = 400

# Тень товара
SHADOW_COLOR = "#7A6E64"   # тёплый серый
SHADOW_OPACITY = 0.27
SHADOW_BLUR = 30           # px
SHADOW_OFFSET = (16, 30)   # вправо, вниз
CONTACT_OPACITY = 0.40
CONTACT_BLUR = 9           # px
CONTACT_OFFSET = (2, 5)
CONTACT_FROM = 0.45        # контактная тень — только нижняя часть (лезвия)
PROD_CX = 750

# Значки
BADGE_D = 172              # ×1.5 (v4: 115); верх значков считается от низа карточки
BADGE_CAPTION_GAP = 24
CAPTION_SIZE = 54          # ×1.5 = 81 не помещается в 3 колонки — см. отчёт
CAPTION_LH = 64            # межстрочный
CAPTION_WEIGHT = b"ExtraBold"  # жирнее (v4: Bold)
CAPTION_COLOR = None       # None = тёмно-серый как у заголовка
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
    F = np.clip((P - (1 - A[..., None]) * B) / np.maximum(A, 0.05)[..., None], 0, 1)
    F = np.where((A > 0.97)[..., None], P, F)
    # полоса старой линии (+ запас): альфа и цвет берём вдоль наклона края ручки сверху и снизу,
    # так край ручки продолжается ровно, без тёмного штриха и серого хвоста
    band = (cv2.dilate(line_m, np.ones((23, 9), np.uint8)) > 0) & win
    d = 20   # сдвиг больше полосы: оба образца лежат вне вырезанной линии
    Ac = np.minimum(sh(A, 1), sh(A, -1))
    Fp = lambda k: np.stack([sh(F[..., i] * A, k) for i in range(3)], 2)
    pre = 0.5 * (Fp(1) + Fp(-1))
    asum = 0.5 * (sh(A, 1) + sh(A, -1))
    Fc = np.clip(pre / np.maximum(asum, 1e-3)[..., None], 0, 1)
    A = np.where(band, Ac, A)
    F = np.where(band[..., None], Fc, F)

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

    # --- тени товара (свет сверху-слева): падающая + контактная под лезвиями ---
    shc = hex2rgb(SHADOW_COLOR)
    def shifted(a, dx, dy, blur):
        M_ = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.GaussianBlur(cv2.warpAffine(a, M_, (w, h)), (0, 0), blur / 2)
    cast = shifted(A2, *SHADOW_OFFSET, SHADOW_BLUR)
    yy = np.arange(h, dtype=np.float32)[:, None]
    ptop, pbot = ys.min(), ys.max()
    low = A2 * np.clip((yy - (ptop + CONTACT_FROM * (pbot - ptop))) / (0.12 * (pbot - ptop)), 0, 1)
    contact = shifted(low, *CONTACT_OFFSET, CONTACT_BLUR)
    out = lerp(out, shc, np.clip(cast * SHADOW_OPACITY + contact * CONTACT_OPACITY, 0, 0.6))

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


    # «Двусторонняя, / нержавеющая сталь» — слева от расчёски на уровне ручки, шрифтом нижних подписей
    f = ImageFont.truetype(str(pc.FONT_PATH), CAPTION_SIZE)
    f.set_variation_by_name(CAPTION_WEIGHT)
    asc = f.getbbox("Нн")[1]
    tagl = Image.new("L", (w, h), 0)
    dt = ImageDraw.Draw(tagl)
    for i, ln in enumerate(TAG_LINES):
        dt.text((TAG_X, TAG_TOP + i * CAPTION_LH - asc), ln, font=f, fill=255)
    tag = np.asarray(tagl, np.float32) / 255
    out = lerp(out, cap_col, tag)
    occ = np.maximum(occ, tag)
    ys_, xs_ = np.where(tag > 0.02)
    vid_box = (xs_.min(), ys_.min(), xs_.max(), ys_.max())
    print(f"«{' / '.join(TAG_LINES)}»: x {xs_.min()}–{xs_.max()}, y {ys_.min()}–{ys_.max()}")

    # --- значки: колонки по ширине самой длинной строки, равные промежутки ---
    problems = []

    def wrap(text):
        lines, cur = [], ""
        for word in text.split():
            t = (cur + " " + word).strip()
            if cur and f.getlength(t) > COL_WIDTH:
                lines.append(cur)
                cur = word
            else:
                cur = t
        return lines + [cur]
    wrapped = [wrap(t) for t in CAPTIONS]
    widths = [max(max(f.getlength(l) for l in ls), BADGE_D) for ls in wrapped]
    gap_cols = (w - 2 * MARGIN - sum(widths)) / (len(widths) - 1)
    if gap_cols < 30:
        problems.append(f"между подписями значков всего {gap_cols:.0f}px")
    x, centers = MARGIN, []
    for wd in widths:
        centers.append(x + wd / 2)
        x += wd + gap_cols
    cap_h = (max(len(l) for l in wrapped) - 1) * CAPTION_LH + (f.getbbox("Ну")[3] - asc)
    badge_top = int(h - MARGIN - cap_h - BADGE_CAPTION_GAP - BADGE_D)
    cap = Image.new("L", (w, h), 0)
    dr = ImageDraw.Draw(cap)
    for cx_, lines, kind in zip(centers, wrapped, ICONS):
        ia, ig = icon_layer(kind, BADGE_D)
        x0 = int(round(cx_ - BADGE_D / 2))
        sl = (slice(badge_top, badge_top + BADGE_D), slice(x0, x0 + BADGE_D))
        out[sl] = lerp(out[sl], teal, ia)
        out[sl] = lerp(out[sl], np.ones(3, np.float32), ig * ia)
        occ[sl] = np.maximum(occ[sl], ia)
        y = badge_top + BADGE_D + BADGE_CAPTION_GAP
        for ln in lines:
            dr.text((cx_ - f.getlength(ln) / 2, y - asc), ln, font=f, fill=255)
            y += CAPTION_LH
        print(f"значок {kind}: центр x {cx_:.0f} → {' / '.join(lines)}")
    capa = np.asarray(cap, np.float32) / 255
    out = lerp(out, cap_col, capa)
    occ = np.maximum(occ, capa)
    print(f"значки: круг {BADGE_D}px, верх {badge_top}, низ подписей {badge_top + BADGE_D + BADGE_CAPTION_GAP + cap_h}, "
          f"промежуток колонок {gap_cols:.0f}px")

    # --- проверки ---
    near_prod = cv2.dilate((A2 > 0.1).astype(np.uint8), np.ones((2 * MIN_GAP + 1,) * 2, np.uint8))
    for name, b in [("подзаголовок", sub_box), ("Двусторонняя…", vid_box)]:
        if (near_prod[b[1]:b[3] + 1, b[0]:b[2] + 1] & (occ[b[1]:b[3] + 1, b[0]:b[2] + 1] > 0.1)).any():
            problems.append(f"{name} ближе {MIN_GAP}px к товару")
    if (near_prod[badge_top:, :].astype(bool) & (occ[badge_top:, :] > 0.1)).any():
        problems.append(f"значки ближе {MIN_GAP}px к товару")
    if xs.min() < MARGIN or xs.max() > w - 1 - MARGIN or ys.min() < MARGIN:
        problems.append(f"расчёска ближе {MARGIN}px к краю")
    ys2, xs2 = np.where(occ > 0.05)
    if xs2.min() < MARGIN or xs2.max() > w - 1 - MARGIN or ys2.max() > h - 1 - MARGIN:
        problems.append(f"текст ближе {MARGIN}px к краю")

    # --- товар ---
    out = lerp(out, F2, A2)

    res = Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8))
    res.save(OUT_DIR / "card_v5.png")
    cw = 600
    ch = round(h * cw / w)
    cmp_ = Image.new("RGB", (cw * 2 + 30, ch + 60), "white")
    cmp_.paste(Image.open(OUT_DIR / "card_v4.png").convert("RGB").resize((cw, ch), Image.LANCZOS), (0, 60))
    cmp_.paste(res.resize((cw, ch), Image.LANCZOS), (cw + 30, 60))
    fd = ImageFont.truetype(str(pc.FONT_PATH), 34)
    ImageDraw.Draw(cmp_).text((cw // 2, 30), "card_v4", font=fd, fill=(40, 40, 40), anchor="mm")
    ImageDraw.Draw(cmp_).text((cw + 30 + cw // 2, 30), "card_v5", font=fd, fill=(40, 40, 40), anchor="mm")
    cmp_.save(OUT_DIR / "card_v5_compare.png")

    print(f"контраст подписей {contrast(cap_col, B[1800, 750]):.2f}:1, "
          f"бирюзы на фоне {contrast(teal, B[400, 750]):.2f}:1")
    print("ПРОБЛЕМЫ:\n  " + "\n  ".join(problems) if problems else "всё влезло, наложений нет")
    print("готово:", OUT_DIR / "card_v5.png", OUT_DIR / "card_v5_compare.png")


if __name__ == "__main__":
    main()
