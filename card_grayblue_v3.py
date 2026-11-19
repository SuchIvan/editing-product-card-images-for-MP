#!/usr/bin/env python3
"""Карточка для серо-голубой расчёски (фото взято из готовой карточки grayblue.png, см. clean_card.py) по образцу card_v8 (бирюзовой).

Расчёска НЕ вырезается: фото с чистым белым фоном кладётся на фон образца умножением
(белое фото → ровно фон образца, родные полутени фото сохраняются). Маска rembg нужна
только для расчётов: масштаб, выноски, проверки наложений, контактная тень.

Запуск:  python3 card_grayblue.py [grayblue_clean.png] [card.png] [output_dir]
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import card_v8 as v8
import peach_card as pc
from peach_card import hex2rgb, lerp, contrast

# ======================= ПАРАМЕТРЫ =======================
PHOTO = sys.argv[1] if len(sys.argv) > 1 else "grayblue_clean.png"
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

SHADOW_ON = False           # v2: без тени
ZOOM_BLADES = (50, 1150, 330, 1350)    # фрагменты для проверки 300%
ZOOM_HANDLE = (1000, 680, 1300, 900)
ERODE_PX = 3                # сжатие маски внутрь, px (на карточке)
FEATHER = 1.0               # сглаживание края маски, px
SRC_CARD = "grayblue.png"   # исходная карточка — для точной маски
CHROMA_BOOST = 1.3          # чуть больше голубизны, чем у ручки, чтобы не ушло в серый
SHARPEN = 0.6               # сила лёгкого повышения резкости после увеличения (0 = выкл)
SIL_CLOSE = 41              # px: закрыть промежутки между зубьями для внешнего контура
PAD_ROWS = (496, 507)       # строки линии на рифлёной площадке
PAD_X0, PAD_X1 = 584, 621   # рифлёная площадка в исходнике grayblue.png (по X)
LINE_GAP = 25               # v3: правая выноска начинается на фоне, в 25 px от края ручки
# =========================================================


def accent_from_handle(Y, A, bg):
    # пиксели ручки: насыщенные, оттенок синий/голубой (180–230°)
    mx, mn = Y.max(2), Y.min(2)
    sat = (mx - mn) / np.maximum(mx, 1e-3)
    hsv = cv2.cvtColor((Y * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.float32) * 2
    m = (A > 0.9) & (sat > 0.15) & (sat < 0.6) & (mx > 0.6) & (hue > 200) & (hue < 235)   # серо-голубая ручка
    base = np.median(Y[m], 0)
    # затемняем в Lab: тон и «серость» ручки сохраняются (насыщенность HLS не раздувается)
    lab = cv2.cvtColor(base.astype(np.float32)[None, None], cv2.COLOR_RGB2Lab)[0, 0]
    acc = base
    for L in np.linspace(lab[0], 10, 400):
        acc = np.clip(cv2.cvtColor(np.array([[[L, lab[1] * CHROMA_BOOST, lab[2] * CHROMA_BOOST]]], np.float32),
                                   cv2.COLOR_Lab2RGB)[0, 0], 0, 1)
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

    # --- синее фото ---
    yimg = Image.open(PHOTO).convert("RGB")
    Y = np.asarray(yimg, np.float32) / 255
    print("маска серо-голубой расчёски (только для расчётов)…")
    # точная маска: rembg по исходной карточке, линия-указатель на ручке закрыта заливкой маски
    from clean_card import LINE_RECTS
    Ay = pc.product_alpha(Image.open(SRC_CARD).convert("RGB"))
    lm = np.zeros(Ay.shape, np.uint8)
    for x0_, y0_, x1_, y1_ in LINE_RECTS:
        lm[y0_:y1_, x0_:x1_] = 1
    # линия на ручке занимает строки ~499–504 исходника: край ручки и цвет в этих строках
    # восстанавливаем линейной интерполяцией между чистыми строками LY0 (над линией) и LY1 (под ней)
    hx0, _, hx1, _ = LINE_RECTS[0]
    xr = LINE_RECTS[1][0] - 2                       # левее вертикали линии
    LY0, LY1 = 496, 521         # v3: полоса шире — захватывает светлый прямоугольник под линией
    def edge_x(row):
        r = np.where(Ay[row, hx0:xr] > 0.5)[0]
        return hx0 + (r.max() if len(r) else 0)
    eu, ed = edge_x(LY0), edge_x(LY1)
    for yy in range(LY0 + 1, LY1):
        f_ = (yy - LY0) / (LY1 - LY0)
        ex = eu + (ed - eu) * f_
        xs_r = np.arange(hx0, xr + 30)
        Ay[yy, hx0:xr + 30] = np.clip(ex + 0.5 - xs_r, 0, 1)
    print(f"край ручки у линии: x {eu} (y {LY0}) → {ed} (y {LY1})")
    Ay[:, LINE_RECTS[1][0]:LINE_RECTS[1][2]] *= (np.arange(Ay.shape[0]) < LINE_RECTS[1][1])[:, None]   # вертикаль линии — не товар
    n_, lab_, st_, _ = cv2.connectedComponentsWithStats((Ay > 0.5).astype(np.uint8))
    Ay = Ay * cv2.dilate((lab_ == 1 + np.argmax(st_[1:, cv2.CC_STAT_AREA])).astype(np.uint8), np.ones((5, 5), np.uint8))
    border = np.concatenate([Y[:20].reshape(-1, 3), Y[-20:].reshape(-1, 3), Y[:, :20].reshape(-1, 3), Y[:, -20:].reshape(-1, 3)])
    white = np.median(border, 0)
    print(f"фон фото: {hexs(white)}; фон образца: {hexs(B[1800, 750])}")
    Yn = np.clip(Y / white, 0, 1)                       # белый фото → 1.0
    # пиксели расчёски — прямо из исходной карточки, нормированные на её локальный фон (без следов чистки)
    Po = np.asarray(Image.open(SRC_CARD).convert("RGB"), np.float32) / 255
    Bo = pc.old_background(Po, Ay)
    Yn = np.clip(Po / np.maximum(Bo, 1e-3), 0, 1)

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
    # снятие каймы: цвет края без примеси старого фона (фон в Yn = 1.0)
    Fsrc = np.clip((Yn - (1 - Ay[..., None])) / np.maximum(Ay, 0.05)[..., None], 0, 1)
    Fsrc = np.where((Ay > 0.85)[..., None], Yn, Fsrc)        # внутри — исходные пиксели, кайму снимаем только на краю
    # v3: восстановление в исходном разрешении (насечки там чёткие).
    # Ровная серая накладка справа от рифлёной площадки: вертикальная интерполяция между LY0 и LY1.
    # Рифлёная площадка (x < PAD_X1): копия строк той же площадки со сдвигом по вертикали на период узора,
    # без поворота и масштаба — направление и размер насечек сохраняются; сдвиг выбирается по совпадению
    # с нетронутыми строками над и под линией.
    Fref = Fsrc.copy()
    xg = np.arange(PAD_X1, xr + 30, dtype=np.float32)
    for yy in range(LY0 + 1, LY1):
        f_ = (yy - LY0) / (LY1 - LY0)
        ex = eu + (ed - eu) * f_

        def samp(row, dxs):          # вдоль наклона края, но не заходя в рифлёную площадку
            xs_s = np.clip(xg + dxs, PAD_X1 + 2, Fsrc.shape[1] - 1)
            return np.stack([np.interp(xs_s, np.arange(Fsrc.shape[1]), Fref[row, :, c]) for c in range(3)], 1)
        g_ = np.clip((f_ - 0.7) / 0.3, 0, 1)      # ровная накладка продолжается сверху, к LY1 — плавный переход
        fill_ = samp(LY0, eu - ex) * (1 - g_) + samp(LY1, ed - ex) * g_
        if yy > PAD_ROWS[1]:      # ниже линии у края площадки — плавно в исходник, без вертикального шва
            ramp = np.clip((xg - PAD_X1) / 8.0, 0, 1)[:, None]
            fill_ = fill_ * ramp + Fref[yy, PAD_X1:xr + 30] * (1 - ramp)
        Fsrc[yy, PAD_X1:xr + 30] = fill_
    PL0, PL1 = PAD_ROWS
    rows = np.arange(PL0 + 1, PL1)
    px0 = PAD_X0
    ring_rows = np.r_[PL0 - 3:PL0 + 1, PL1:PL1 + 4]
    best = None
    for dy in list(range(-30, -(PL1 - PL0) + 1)) + list(range(PL1 - PL0, 31)):
        for dx in (-2, -1, 0, 1, 2):
            d = ((Fsrc[ring_rows, px0:PAD_X1] - Fsrc[ring_rows + dy, px0 + dx:PAD_X1 + dx]) ** 2).mean()
            # кандидат должен лежать в нетронутой части площадки
            if best is None or d < best[0]:
                best = (d, dy, dx)
    _, dy, dx = best
    patch = Fsrc[rows + dy, px0 + dx:PAD_X1 + dx].copy()
    # выравнивание цвета: среднее заплатки = среднему соседних строк
    ref = 0.5 * (Fsrc[PL0, px0:PAD_X1].mean(0) + Fsrc[PL1, px0:PAD_X1].mean(0))
    patch += ref - patch.reshape(-1, 3).mean(0)
    # мягкий шов по X с ровной частью
    wx = np.clip((PAD_X1 - np.arange(px0, PAD_X1)) / 6.0, 0, 1)[None, :, None]
    Fsrc[rows, px0:PAD_X1] = np.clip(patch * wx + Fsrc[rows, px0:PAD_X1] * (1 - wx), 0, 1)
    # шов площадка/ровная часть: убрать одиночные тёмные точки
    sl_y, sl_x = slice(PL0 - 1, PL1 + 2), slice(PAD_X1 - 6, PAD_X1 + 4)
    sub8 = (Fsrc[sl_y, sl_x] * 255).astype(np.uint8)
    Fsrc[sl_y, sl_x] = cv2.medianBlur(sub8, 3).astype(np.float32) / 255
    print(f"рифлёная площадка: заплатка со сдвигом ({dx}, {dy}) px исходника, ошибка {np.sqrt(best[0]) * 255:.1f}")
    pre_ = cv2.warpAffine(Fsrc * Ay[..., None], M, (w, h), flags=cv2.INTER_LANCZOS4)
    Araw = np.clip(cv2.warpAffine(Ay, M, (w, h), flags=cv2.INTER_LINEAR), 0, 1)
    Yn2 = np.clip(pre_ / np.maximum(Araw, 1e-3)[..., None], 0, 1)
    if SHARPEN > 0:   # исходник маленький: лёгкая нерезкая маска после увеличения
        bl = cv2.GaussianBlur(Yn2, (0, 0), 1.2)
        Yn2 = np.clip(Yn2 + SHARPEN * (Yn2 - bl), 0, 1)
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
    # маска сжата внутрь на ERODE_PX и сглажена — по краю не остаётся пикселей старого фона
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ERODE_PX + 1, 2 * ERODE_PX + 1))
    Afin = cv2.GaussianBlur(cv2.erode(Araw, ker), (0, 0), FEATHER)
    out = lerp(out, Yn2, Afin)


    occ = np.zeros((h, w), np.float32)
    ink_col = np.where((ink_k == 0)[..., None], acc, dark)   # бирюзовое в растре → акцент

    # --- заголовок, подзаголовок (как в v8) ---
    a, c = v8.tight(*v8.scale_layer(*v8.layer_from(ink_a, ink_col, v8.BOX_HEAD), v8.K_TEXT))
    v8.paste(out, occ, a, c, 750 - a.shape[1] / 2, v8.HEAD_TOP)
    head_bottom = v8.HEAD_TOP + a.shape[0]
    sub = v8.render_line(h, w, v8.SUB_TEXT, v8.SUB_SIZE, b"Bold", 1.4 * v8.SUB_SIZE / 79, v8.SUB_SPACE,
                         head_bottom + v8.SUB_GAP)
    out = lerp(out, acc, sub)
    occ = np.maximum(occ, sub)
    sy_, sx_ = np.where(sub > 0.02)
    sub_box = (sx_.min(), sy_.min(), sx_.max(), sy_.max())

    # --- левая подпись «нержавеющая / сталь» (шрифт как «дешеддер + колтунорез») ---
    corner_y = 753                                      # как в v8
    tag_bottom = corner_y - v8.TAG_LINE_GAP
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
    l1 = line_layer(v8.TAG_LINES[0], v8.TAG_X, tag_top)
    c1 = np.where(l1.max(0) > 0.1)[0]
    x_l, x_r = int(c1.min()), int(c1.max()) + 1
    l2 = line_layer(v8.TAG_LINES[1], 0, tag_top + lh)
    c2 = np.where(l2.max(0) > 0.1)[0]
    l2 = cv2.warpAffine(l2, np.float32([[1, 0, (x_l + x_r) / 2 - (c2.min() + c2.max() + 1) / 2], [0, 1, 0]]), (w, h))
    tag = np.maximum(l1, l2)
    out = lerp(out, acc, tag)
    occ = np.maximum(occ, tag)
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
    # правый блок: опускаем, пока у выноски не будет видимого отрезка от ручки до угла (≥ 90 px)
    corner_x = v8.RIGHT_CX + (1252 - (1109 + 1367) / 2) * v8.K_TEXT

    def find_line(rtop):
        vy1_ = rtop - (1007 - 966) * v8.K_TEXT
        for yv in range(int(vy1_) - 50, int(ptop) + 40, -2):      # снизу вверх
            cols = np.where(A2[yv, :int(corner_x)] > 0.5)[0]
            if len(cols) and 90 <= corner_x - cols.max() < 260 and \
                    not (A2[yv - 30:int(vy1_), int(corner_x) - 25:int(corner_x) + 25] > 0.1).any():
                return yv, vy1_
        return None, vy1_
    right_down = 0
    for right_down in range(0, 241, 10):
        rtop = v8.RIGHT_TOP + right_down
        line_y, vy1 = find_line(rtop)
        if line_y is not None:
            break
    print(f"правый блок опущен на {right_down}px")
    x1 = v8.RIGHT_CX - a1.shape[1] / 2
    v8.paste(out, occ, a1, c1_, x1, rtop)
    dx = ((1093 + 1411) / 2 - (1109 + 1367) / 2) * v8.K_TEXT
    x2 = v8.RIGHT_CX + dx - a2_.shape[1] / 2
    y2 = rtop + a1.shape[0] + (1164 - 1100) * v8.K_TEXT
    v8.paste(out, occ, a2_, c2_, x2, y2)
    right_box = (int(min(x1, x2)), rtop, int(max(x1 + a1.shape[1], x2 + a2_.shape[1])), int(y2 + a2_.shape[0]))
    if line_y is None:
        line_y = v8.LINE_Y
    edge = int(np.where(A2[line_y, :int(corner_x)] > 0.5)[0].max())
    line_x0 = edge + LINE_GAP
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
        out[sl] = lerp(out[sl], acc, ia)
        out[sl] = lerp(out[sl], np.ones(3, np.float32), ig * ia)
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
    res.save(OUT_DIR / "card_grayblue_v3.png")
    # --- до/после, 300%: лезвия и ручка (с местом бывшего пятна справа от ручки) ---
    before = Image.open(OUT_DIR / "card_grayblue_v2.png").convert("RGB")
    crops = [("ручка", ZOOM_HANDLE)]
    tiles = []
    for name, (cx0, cy0, cx1, cy1) in crops:
        bw, bh = (cx1 - cx0) * 3, (cy1 - cy0) * 3
        pair = Image.new("RGB", (bw * 2 + 20, bh + 50), "white")
        pair.paste(before.crop((cx0, cy0, cx1, cy1)).resize((bw, bh), Image.NEAREST), (0, 50))
        pair.paste(res.crop((cx0, cy0, cx1, cy1)).resize((bw, bh), Image.NEAREST), (bw + 20, 50))
        fdd = ImageFont.truetype(str(pc.FONT_PATH), 28)
        ImageDraw.Draw(pair).text((bw // 2, 25), f"до ({name}, 300%)", font=fdd, fill=(40, 40, 40), anchor="mm")
        ImageDraw.Draw(pair).text((bw + 20 + bw // 2, 25), f"после ({name}, 300%)", font=fdd, fill=(40, 40, 40), anchor="mm")
        tiles.append(pair)
    zw = max(tl.width for tl in tiles)
    zoom = Image.new("RGB", (zw, sum(tl.height for tl in tiles) + 20 * (len(tiles) - 1)), "white")
    yz = 0
    for tl in tiles:
        zoom.paste(tl, (0, yz))
        yz += tl.height + 20
    zoom.save(OUT_DIR / "card_grayblue_v3_zoom.png")
    # остатки старого фона по краю: пиксели в кольце 1–6 px снаружи маски, заметно отличающиеся от фона
    ring = (cv2.dilate((Afin > 0.5).astype(np.uint8), np.ones((13, 13), np.uint8)) > 0) & (Afin < 0.02)
    rim = np.abs(out - B).max(2)[ring]
    print(f"кольцо вокруг расчёски: отклонение от фона p99 {np.percentile(rim, 99) * 255:.1f}, max {rim.max() * 255:.1f} из 255")
    cw, ch = 600, round(h * 600 / w)
    cmp_ = Image.new("RGB", (cw * 2 + 30, ch + 60), "white")
    cmp_.paste(before.resize((cw, ch), Image.LANCZOS), (0, 60))
    cmp_.paste(res.resize((cw, ch), Image.LANCZOS), (cw + 30, 60))
    fd = ImageFont.truetype(str(pc.FONT_PATH), 34)
    ImageDraw.Draw(cmp_).text((cw // 2, 30), "card_grayblue_v2", font=fd, fill=(40, 40, 40), anchor="mm")
    ImageDraw.Draw(cmp_).text((cw + 30 + cw // 2, 30), "card_grayblue_v3", font=fd, fill=(40, 40, 40), anchor="mm")
    cmp_.save(OUT_DIR / "card_grayblue_v3_compare.png")
    print("ПРОБЛЕМЫ:\n  " + "\n  ".join(problems) if problems else "всё влезло, наложений нет")
    print("готово:", OUT_DIR / "card_grayblue_v3.png", OUT_DIR / "card_grayblue_v3_zoom.png", OUT_DIR / "card_grayblue_v3_compare.png")


if __name__ == "__main__":
    main()
