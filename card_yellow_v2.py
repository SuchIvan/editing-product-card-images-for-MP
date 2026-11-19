#!/usr/bin/env python3
"""v2 жёлтой карточки: акценты ярким цветом ручки на тёмно-серых скруглённых плашках,
иконки — жёлтые круги с тёмно-серым знаком, нижние подписи тёмно-серые.

Карточка для жёлтой расчёски по образцу card_v8 (бирюзовой).

Расчёска НЕ вырезается: фото с чистым белым фоном кладётся на фон образца умножением
(белое фото → ровно фон образца, родные полутени фото сохраняются). Маска rembg нужна
только для расчётов: масштаб, выноски, проверки наложений, контактная тень.

Запуск:  python3 card_yellow_v2.py [yellow.png] [card.png] [output_dir]
"""
import colorsys
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import card_v8 as v8
import peach_card as pc
from peach_card import hex2rgb, lerp, contrast

# ======================= ПАРАМЕТРЫ =======================
YELLOW = sys.argv[1] if len(sys.argv) > 1 else "yellow.png"
SAMPLE = sys.argv[2] if len(sys.argv) > 2 else "card.png"     # исходник бирюзовой (фон, «РАСЧЕСКА», «2 в 1»)
OUT_DIR = Path(sys.argv[3] if len(sys.argv) > 3 else "output")

# Куда вписать расчёску: та же рамка, что у бирюзовой в card_v8 (x 52–1448, y 420–1549)
FIT_BOX = (52, 420, 1448, 1549)
MAX_UPSCALE_WARN = 1.25     # выше — предупреждаем о мутности

ACCENT_MIN_CONTRAST = 4.5   # мин. контраст акцента к фону (WCAG AA)

# Контактная тень (без большого ореола)
SHADOW_COLOR = "#6B6560"
SHADOW_OPACITY = 0.22       # мягкий контур по всему товару
SHADOW_BLUR = 10
SHADOW_OFFSET = (4, 7)
CONTACT_OPACITY = 0.45      # под самыми нижними точками (лезвия, низ ручки)
CONTACT_BLUR = 8
CONTACT_OFFSET = (2, 4)
CONTACT_ZONE = 0.07

PLAQUE_COLOR = None         # None = тёмно-серый как у «РАСЧЕСКА»
PLAQUE_PAD = (26, 14)       # поля плашки по X и Y, px
PLAQUE_RADIUS = 18
TAG_X = 72                  # левая подпись правее, чтобы плашка была ≥ 40 px от края
TAG_PLAQUE_TO_LINE = 14     # зазор плашка «нержавеющая сталь» → горизонталь выноски
SUB_PLAQUE_SHIFT = 24       # подзаголовок с плашкой ниже, чтобы плашка не касалась «РАСЧЕСКА»
RIGHT_DOWN = 40             # правый блок ниже: у выноски виден отрезок от ручки до угла
RIGHT_SHIFT = -26           # сдвиг правого блока, чтобы плашка «дешеддер» была ≥ 40 px от края
SHADOW_ON = True
SIL_CLOSE = 41              # px: закрыть промежутки между зубьями для внешнего контура
LINE_ON_HANDLE = 70         # сколько px правой выноски лежит на ручке
# =========================================================


def accent_from_handle(Y, A, bg):
    g, b = Y[..., 1], Y[..., 2]
    m = (A > 0.9) & (g > 0.55) & (b < 0.4) & (g - b > 0.4)
    base = np.median(Y[m], 0)
    hh, ll, ss = colorsys.rgb_to_hls(*base)
    acc = base
    for L in np.linspace(ll, 0.05, 400):
        acc = np.array(colorsys.hls_to_rgb(hh, L, ss), np.float32)
        if contrast(acc, bg) >= ACCENT_MIN_CONTRAST:
            break
    return base, acc


def hexs(c):
    return "#%02X%02X%02X" % tuple((np.asarray(c) * 255).round().astype(int))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # --- образец: фон, тёмный цвет, растровые надписи ---
    simg = Image.open(SAMPLE).convert("RGB")
    w, h = simg.size
    P = np.asarray(simg, np.float32) / 255
    print("маска образца (для отделения надписей)…")
    As = pc.product_alpha(simg)
    B = pc.old_background(P, As)
    hp = P[v8.BOX_HEAD[1]:v8.BOX_HEAD[3], v8.BOX_HEAD[0]:v8.BOX_HEAD[2]].reshape(-1, 3)
    dark = np.median(hp[hp.mean(1) < 0.35], 0)
    boxes = [b + (0.05,) for b in (v8.BOX_HEAD, v8.BOX_2V1, v8.BOX_DESC)]
    ink_a, ink_k = pc.unmix_ink(P, B, As, boxes, dark)

    # --- жёлтое фото ---
    yimg = Image.open(YELLOW).convert("RGB")
    Y = np.asarray(yimg, np.float32) / 255
    print("маска жёлтой расчёски (только для расчётов)…")
    Ay = pc.product_alpha(yimg)
    border = np.concatenate([Y[:20].reshape(-1, 3), Y[-20:].reshape(-1, 3), Y[:, :20].reshape(-1, 3), Y[:, -20:].reshape(-1, 3)])
    white = np.median(border, 0)
    print(f"фон фото: {hexs(white)}; фон образца: {hexs(B[1800, 750])}")
    Yn = np.clip(Y / white, 0, 1)                       # белый фото → 1.0

    bg_ref = B[1300:1500, 200:1300].reshape(-1, 3).mean(0)
    base, acc = accent_from_handle(Y, Ay, bg_ref)
    print(f"цвет ручки {hexs(base)} (контраст {contrast(base, bg_ref):.2f}:1) → акцент {hexs(acc)} "
          f"(контраст {contrast(acc, bg_ref):.2f}:1)")

    # --- масштаб и положение: вписать в рамку образца, пропорционально ---
    ys, xs = np.where(Ay > 0.1)
    bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max(), ys.max()
    fx0, fy0, fx1, fy1 = FIT_BOX
    k = min((fx1 - fx0) / (bx1 - bx0), (fy1 - fy0) / (by1 - by0))
    ox = (fx0 + fx1) / 2 - (bx0 + bx1) / 2 * k
    oy = (fy0 + fy1) / 2 - (by0 + by1) / 2 * k
    M = np.float32([[k, 0, ox], [0, k, oy]])
    Yn2 = cv2.warpAffine(Yn, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(1, 1, 1))
    Yn2 = np.clip(Yn2, 0, 1)
    A2 = np.clip(cv2.warpAffine(Ay, M, (w, h), flags=cv2.INTER_LINEAR), 0, 1)
    ys, xs = np.where(A2 > 0.1)
    print(f"масштаб ×{k:.3f} (одинаково по X и Y), расчёска x {xs.min()}–{xs.max()}, y {ys.min()}–{ys.max()}")
    blur_note = None
    if k > MAX_UPSCALE_WARN:
        blur_note = f"фото увеличено ×{k:.2f} — будет мягче исходника"
    # резкость: доля высоких частот до/после (для отчёта)
    lap0 = cv2.Laplacian(cv2.cvtColor((Y * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.CV_32F)[Ay > 0.9].var()
    lap1 = cv2.Laplacian(cv2.cvtColor((Yn2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.CV_32F)[A2 > 0.9].var()
    print(f"резкость (дисперсия лапласиана) исходник {lap0:.0f} → на карточке {lap1:.0f}")

    # --- фон образца + зерно, тень, фото умножением ---
    # Фон под бирюзовой расчёской в образце восстановлен интерполяцией и там темнее (до −9%):
    # через промежутки между зубьями он проступал тёмной полосой. Берём гладкую модель фона —
    # полином 2-й степени по чистым участкам образца (вне расчёски и надписей).
    clean = (cv2.dilate((As > 0.02).astype(np.uint8), np.ones((81, 81), np.uint8)) == 0)
    for bx in (v8.BOX_HEAD, v8.BOX_VIDEO, v8.BOX_2V1, v8.BOX_DESC, (50, 1610, 1085, 1870), (380, 300, 1110, 420), (1000, 815, 1270, 980)):
        clean[bx[1] - 20:bx[3] + 20, bx[0] - 20:bx[2] + 20] = False
    clean &= np.abs(P - B).max(2) < 0.03
    gy, gx = np.mgrid[0:h:8, 0:w:8]
    cm = clean[::8, ::8]
    X = lambda xx, yy_: np.stack([np.ones_like(xx), xx, yy_, xx * xx, xx * yy_, yy_ * yy_], -1)
    xn, yn = gx / w, gy / h
    coef = np.linalg.lstsq(X(xn[cm], yn[cm]), P[::8, ::8][cm], rcond=None)[0]
    fy, fx = np.mgrid[0:h, 0:w].astype(np.float32)
    B = (X(fx / w, fy / h) @ coef).astype(np.float32)
    print(f"фон: гладкая модель по {cm.sum()} точкам, диапазон {B.min() * 255:.0f}–{B.max() * 255:.0f}")
    out = B.copy()
    patch = P[1300:1550, 1150:1450]
    sigma = (patch - cv2.GaussianBlur(patch, (0, 0), 3)).std()
    noise = cv2.GaussianBlur(np.random.default_rng(3).normal(0, 1, (h, w)).astype(np.float32), (0, 0), 0.6)
    out += (noise / noise.std() * sigma)[..., None]

    def shifted(a, off, blur):
        M_ = np.float32([[1, 0, off[0]], [0, 1, off[1]]])
        return cv2.GaussianBlur(cv2.warpAffine(a, M_, (w, h)), (0, 0), blur / 2)
    yy = np.arange(h, dtype=np.float32)[:, None]
    ptop, pbot = ys.min(), ys.max()
    ph = pbot - ptop
    # Точный внешний контур: маска товара, у которой закрыты промежутки между зубьями и все дырки.
    # Тень строится от этого силуэта и кладётся только СНАРУЖИ него — на зубья, в промежутки
    # между ними и под рамку она не попадает (там остаётся чистое фото).
    sil = (A2 > 0.5).astype(np.uint8)
    sil = cv2.morphologyEx(sil, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SIL_CLOSE, SIL_CLOSE)))
    ff = sil.copy()
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 2)
    sil = ((ff != 2)).astype(np.float32)                # заливка дырок
    silf = sil
    outside = 1 - cv2.GaussianBlur(cv2.dilate(sil, np.ones((5, 5), np.uint8)), (0, 0), 1.0)
    low = silf * np.clip((yy - (pbot - CONTACT_ZONE * ph)) / (0.03 * ph), 0, 1)
    sa = shifted(silf, SHADOW_OFFSET, SHADOW_BLUR) * SHADOW_OPACITY + shifted(low, CONTACT_OFFSET, CONTACT_BLUR) * CONTACT_OPACITY
    sa = np.clip(sa, 0, 0.6) * np.clip(outside, 0, 1)   # тень строго за пределами контура
    if not SHADOW_ON:
        sa[:] = 0
    out = lerp(out, hex2rgb(SHADOW_COLOR), sa)
    out = out * Yn2                                     # расчёска «на белом»: белый фото = фон образца

    occ = np.zeros((h, w), np.float32)
    yel = base                                          # яркий цвет ручки — для текста на плашках и иконок
    plq = hex2rgb(PLAQUE_COLOR) if PLAQUE_COLOR else dark
    ink_col = np.where((ink_k == 0)[..., None], yel, dark)   # бирюзовое в растре → жёлтое
    plaque_boxes = []

    def plaque_under(alpha):
        ys_p, xs_p = np.where(alpha > 0.05)
        box = (xs_p.min() - PLAQUE_PAD[0], ys_p.min() - PLAQUE_PAD[1], xs_p.max() + 1 + PLAQUE_PAD[0], ys_p.max() + 1 + PLAQUE_PAD[1])
        ss = 4
        x0, y0, x1, y1 = box
        im_ = Image.new("L", ((x1 - x0 + 4) * ss, (y1 - y0 + 4) * ss), 0)
        ImageDraw.Draw(im_).rounded_rectangle([2 * ss, 2 * ss, (x1 - x0 + 2) * ss - 1, (y1 - y0 + 2) * ss - 1],
                                             PLAQUE_RADIUS * ss, fill=255)
        small = np.asarray(im_.resize((x1 - x0 + 4, y1 - y0 + 4), Image.BOX), np.float32) / 255
        m = np.zeros((h, w), np.float32)
        m[y0 - 2:y1 + 2, x0 - 2:x1 + 2] = small
        plaque_boxes.append(box)
        return m

    # --- заголовок, подзаголовок (как в v8) ---
    a, c = v8.tight(*v8.scale_layer(*v8.layer_from(ink_a, ink_col, v8.BOX_HEAD), v8.K_TEXT))
    v8.paste(out, occ, a, c, 750 - a.shape[1] / 2, v8.HEAD_TOP)
    head_bottom = v8.HEAD_TOP + a.shape[0]
    sub = v8.render_line(h, w, v8.SUB_TEXT, v8.SUB_SIZE, b"Bold", 1.4 * v8.SUB_SIZE / 79, v8.SUB_SPACE,
                         head_bottom + v8.SUB_GAP + SUB_PLAQUE_SHIFT)
    pm = plaque_under(sub)
    out = lerp(out, plq, pm)
    out = lerp(out, yel, sub)
    occ = np.maximum(occ, np.maximum(sub, pm))
    sy_, sx_ = np.where(sub > 0.02)
    sub_box = (sx_.min(), sy_.min(), sx_.max(), sy_.max())

    # --- левая подпись «нержавеющая / сталь» (шрифт как «дешеддер + колтунорез») ---
    corner_y = 753                                      # как в v8
    tag_bottom = corner_y - v8.LINE_W / 2 - TAG_PLAQUE_TO_LINE - PLAQUE_PAD[1]
    ft = ImageFont.truetype(str(pc.FONT_PATH), int(round(v8.TAG_SIZE * v8.K_TEXT)))
    ft.set_variation_by_name(v8.TAG_WEIGHT)
    tr, lh = v8.TAG_TRACKING * v8.K_TEXT, v8.TAG_LH * v8.K_TEXT
    xh = ft.getbbox("к")[1]
    base_h = ft.getbbox("к")[3] - xh
    tag_top = tag_bottom - base_h - (len(v8.TAG_LINES) - 1) * lh

    def line_layer(text, x0, y_top):
        lay = Image.new("L", (w, h), 0)
        dd = ImageDraw.Draw(lay)
        x = x0
        for ch in text:
            dd.text((x, y_top - xh), ch, font=ft, fill=255)
            x += ft.getlength(ch) + tr
        return np.asarray(lay, np.float32) / 255
    l1 = line_layer(v8.TAG_LINES[0], TAG_X, tag_top)
    c1 = np.where(l1.max(0) > 0.1)[0]
    x_l, x_r = int(c1.min()), int(c1.max()) + 1
    l2 = line_layer(v8.TAG_LINES[1], 0, tag_top + lh)
    c2 = np.where(l2.max(0) > 0.1)[0]
    l2 = cv2.warpAffine(l2, np.float32([[1, 0, (x_l + x_r) / 2 - (c2.min() + c2.max() + 1) / 2], [0, 1, 0]]), (w, h))
    tag = np.maximum(l1, l2)
    pm = plaque_under(tag)
    out = lerp(out, plq, pm)
    out = lerp(out, yel, tag)
    occ = np.maximum(occ, np.maximum(tag, pm))
    ty_, tx_ = np.where(tag > 0.02)
    tag_box = (tx_.min(), ty_.min(), tx_.max(), ty_.max())

    # левая выноска: вниз на правом конце до ЛЕЗВИЯ (первый металл под дугой рамки)
    vx = int(round(x_r - v8.LINE_W / 2))
    colm = Yn2[corner_y:, vx]
    first_prod = int(np.where(A2[corner_y:, vx] > 0.5)[0].min())
    seg = colm[first_prod:first_prod + 40]
    metal_here = ((seg.mean(1) > 0.8) & ((seg.max(1) - seg.min(1)) < 0.08)).sum() > 3
    end_y = corner_y + first_prod + 8
    on_what = "лезвие" if metal_here else "рамку (лезвий под этим столбцом сверху не видно)"
    left_line = np.maximum(
        v8.rect_aa(h, w, x_l, corner_y - v8.LINE_W / 2, x_r, corner_y + v8.LINE_W / 2),
        v8.rect_aa(h, w, x_r - v8.LINE_W, corner_y - v8.LINE_W / 2, x_r, end_y))
    print(f"левая выноска: x {x_l}–{x_r} на y {corner_y}, вниз до y {end_y} — упирается в {on_what}")

    # --- справа «2 в 1» + «дешеддер + колтунорез» (растр образца, бирюза → акцент) ---
    a1, c1_ = v8.tight(*v8.scale_layer(*v8.layer_from(ink_a, ink_col, v8.BOX_2V1), v8.K_TEXT))
    a2_, c2_ = v8.tight(*v8.scale_layer(*v8.layer_from(ink_a, ink_col, v8.BOX_DESC), v8.K_TEXT))
    RCX = v8.RIGHT_CX + RIGHT_SHIFT
    RTOP = v8.RIGHT_TOP + RIGHT_DOWN
    x1 = RCX - a1.shape[1] / 2
    v8.paste(out, occ, a1, c1_, x1, RTOP)
    dx = ((1093 + 1411) / 2 - (1109 + 1367) / 2) * v8.K_TEXT
    x2 = RCX + dx - a2_.shape[1] / 2
    y2 = RTOP + a1.shape[0] + (1164 - 1100) * v8.K_TEXT
    da = np.zeros((h, w), np.float32)
    xi, yi = int(round(x2)), int(round(y2))
    da[yi:yi + a2_.shape[0], xi:xi + a2_.shape[1]] = a2_
    pm = plaque_under(da)
    out = lerp(out, plq, pm)
    occ = np.maximum(occ, pm)
    v8.paste(out, occ, a2_, c2_, x2, y2)
    right_box = (int(min(x1, x2)), RTOP, int(max(x1 + a1.shape[1], x2 + a2_.shape[1])), int(y2 + a2_.shape[0]))

    # правая выноска: горизонталь начинается на ручке, угол над «2 в 1», вниз к надписи
    corner_x = RCX + (1252 - (1109 + 1367) / 2) * v8.K_TEXT
    vy1 = RTOP - (1007 - 966) * v8.K_TEXT
    line_y = None
    for yv in range(int(vy1) - 50, int(ptop) + 40, -2):          # снизу вверх: первая строка, где ручка близко
        cols = np.where(A2[yv, :int(corner_x)] > 0.5)[0]
        if len(cols) and 90 <= corner_x - cols.max() < 260:
            # вертикаль и горизонталь справа от ручки не должны задевать товар
            if not (A2[yv - 30:int(vy1), int(corner_x) - 25:int(corner_x) + 25] > 0.1).any():
                line_y = yv
                break
    if line_y is None:
        line_y = v8.LINE_Y
    edge = int(np.where(A2[line_y, :int(corner_x)] > 0.5)[0].max())
    line_x0 = edge - LINE_ON_HANDLE
    print(f"правая выноска: y {line_y}, от x {line_x0} (край ручки {edge}) до угла {corner_x:.0f}, вниз до {vy1:.0f}")
    la = np.maximum(v8.rect_aa(h, w, line_x0, line_y - v8.LINE_W / 2, corner_x + v8.LINE_W / 2, line_y + v8.LINE_W / 2),
                    v8.rect_aa(h, w, corner_x - v8.LINE_W / 2, line_y - v8.LINE_W / 2, corner_x + v8.LINE_W / 2, vy1))

    # --- значки (как в v8, иконки акцентом) ---
    f = ImageFont.truetype(str(pc.FONT_PATH), v8.CAPTION_SIZE)
    f.set_variation_by_name(v8.CAPTION_WEIGHT)
    asc = f.getbbox("Нн")[1]
    problems = []

    def wrap(text):
        lines, cur = [], ""
        for word in text.split():
            t = (cur + " " + word).strip()
            if cur and f.getlength(t) > v8.COL_WIDTH:
                lines.append(cur)
                cur = word
            else:
                cur = t
        return lines + [cur]
    wrapped = [wrap(t) for t in v8.CAPTIONS]
    widths = [max(max(f.getlength(l) for l in ls), v8.BADGE_D) for ls in wrapped]
    gap_cols = (w - 2 * v8.MARGIN - sum(widths)) / (len(widths) - 1)
    x, centers = v8.MARGIN, []
    for wd in widths:
        centers.append(x + wd / 2)
        x += wd + gap_cols
    cap_h = (max(len(l) for l in wrapped) - 1) * v8.CAPTION_LH + (f.getbbox("Ну")[3] - asc)
    badge_top = int(h - v8.MARGIN - cap_h - v8.BADGE_CAPTION_GAP - v8.BADGE_D)
    cap = Image.new("L", (w, h), 0)
    dr = ImageDraw.Draw(cap)
    for cx_, lines, kind in zip(centers, wrapped, v8.ICONS):
        ia, ig = v8.icon_layer(kind, v8.BADGE_D)
        x0 = int(round(cx_ - v8.BADGE_D / 2))
        sl = (slice(badge_top, badge_top + v8.BADGE_D), slice(x0, x0 + v8.BADGE_D))
        out[sl] = lerp(out[sl], yel, ia)
        out[sl] = lerp(out[sl], dark, ig * ia)          # знак тёмно-серый на жёлтом круге
        occ[sl] = np.maximum(occ[sl], ia)
        y = badge_top + v8.BADGE_D + v8.BADGE_CAPTION_GAP
        for ln in lines:
            dr.text((cx_ - f.getlength(ln) / 2, y - asc), ln, font=f, fill=255)
            y += v8.CAPTION_LH
    capa = np.asarray(cap, np.float32) / 255
    out = lerp(out, dark, capa)
    occ = np.maximum(occ, capa)

    # --- проверки ---
    MG = v8.MIN_GAP
    near_prod = cv2.dilate((A2 > 0.1).astype(np.uint8), np.ones((2 * MG + 1,) * 2, np.uint8))
    for name, b in [("подзаголовок", sub_box), ("нержавеющая сталь", tag_box), ("2 в 1 / дешеддер", right_box)]:
        if (near_prod[b[1]:b[3] + 1, b[0]:b[2] + 1] & (occ[b[1]:b[3] + 1, b[0]:b[2] + 1] > 0.1)).any():
            problems.append(f"{name} ближе {MG}px к товару")
    if (near_prod[badge_top:, :].astype(bool) & (occ[badge_top:, :] > 0.1)).any():
        problems.append(f"значки ближе {MG}px к товару")
    hl = left_line.copy()
    hl[int(corner_y + v8.LINE_W):, :] = 0
    if (near_prod.astype(bool) & (hl > 0.1)).any():
        problems.append("горизонталь левой выноски задевает товар")
    if ys.min() - sub_box[3] < v8.SUB_PROD_GAP:
        problems.append(f"зазор подзаголовок → расчёска {ys.min() - sub_box[3]}px < {v8.SUB_PROD_GAP}")
    if xs.min() < v8.MARGIN or xs.max() > w - 1 - v8.MARGIN:
        problems.append("расчёска ближе 40 px к краю")
    ys2, xs2 = np.where(occ > 0.05)
    if xs2.min() < v8.MARGIN or xs2.max() > w - 1 - v8.MARGIN or ys2.max() > h - 1 - v8.MARGIN:
        problems.append("текст ближе 40 px к краю")
    if blur_note:
        problems.append(blur_note)

    # --- линии поверх ---
    out = lerp(out, dark, np.maximum(la, left_line))

    res = Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8))

    # --- проверка лезвий: карточка против исходного фото (тот же масштаб) ---
    bys, bxs = np.where(sil[int(ptop + 0.35 * ph):] > 0)
    by0_ = int(ptop + 0.35 * ph) + bys.min()
    bb = (max(bxs.min() - 20, 0), by0_ - 10, min(int(xs.max() * 0.62), w), min(int(pbot) + 40, h))   # левая нижняя часть — зубья
    region = np.zeros((h, w), bool)
    region[bb[1]:bb[3], bb[0]:bb[2]] = True
    inside = region & (sil > 0) & (np.maximum(la, left_line) < 0.01)    # без линий-указателей
    print(f"тень внутри контура (лезвия, промежутки): max {sa[inside].max() * 100:.2f}%")
    expect = B * Yn2                                   # чистое фото на фоне образца, без тени
    diff = np.abs(out - expect).max(2)[inside]
    print(f"отличие карточки от чистого фото на лезвиях: 99-й перцентиль {np.percentile(diff, 99) * 255:.1f}, "
          f"max {diff.max() * 255:.1f} из 255 (зерно фона ±{sigma * 255 * 3:.1f})")
    crop_photo = Image.fromarray((np.clip(Yn2[bb[1]:bb[3], bb[0]:bb[2]], 0, 1) * 255).astype(np.uint8))
    crop_card = res.crop(bb)
    cwid = crop_card.width
    chk = Image.new("RGB", (cwid * 2 + 30, crop_card.height + 60), "white")
    chk.paste(crop_photo, (0, 60))
    chk.paste(crop_card, (cwid + 30, 60))
    fdd = ImageFont.truetype(str(pc.FONT_PATH), 30)
    ImageDraw.Draw(chk).text((cwid // 2, 30), "исходное фото yellow.png", font=fdd, fill=(40, 40, 40), anchor="mm")
    ImageDraw.Draw(chk).text((cwid + 30 + cwid // 2, 30), "card_yellow_v2.png", font=fdd, fill=(40, 40, 40), anchor="mm")
    chk.save(OUT_DIR / "card_yellow_v2_blades_check.png")
    res.save(OUT_DIR / "card_yellow_v2.png")
    cw, ch = 600, round(h * 600 / w)
    cmp_ = Image.new("RGB", (cw * 2 + 30, ch + 60), "white")
    cmp_.paste(Image.open(OUT_DIR / "card_yellow.png").convert("RGB").resize((cw, ch), Image.LANCZOS), (0, 60))
    cmp_.paste(res.resize((cw, ch), Image.LANCZOS), (cw + 30, 60))
    fd = ImageFont.truetype(str(pc.FONT_PATH), 34)
    ImageDraw.Draw(cmp_).text((cw // 2, 30), "card_yellow", font=fd, fill=(40, 40, 40), anchor="mm")
    ImageDraw.Draw(cmp_).text((cw + 30 + cw // 2, 30), "card_yellow_v2", font=fd, fill=(40, 40, 40), anchor="mm")
    cmp_.save(OUT_DIR / "card_yellow_v2_compare.png")
    print("ПРОБЛЕМЫ:\n  " + "\n  ".join(problems) if problems else "всё влезло, наложений нет")
    print(f"жёлтый {hexs(yel)} на плашке {hexs(plq)}: контраст {contrast(yel, plq):.2f}:1; "
          f"тёмно-серый знак на жёлтом круге: {contrast(dark, yel):.2f}:1")
    print("готово:", OUT_DIR / "card_yellow_v2.png", OUT_DIR / "card_yellow_v2_compare.png")


if __name__ == "__main__":
    main()
