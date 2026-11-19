#!/usr/bin/env python3
"""Карточка для жёлтой расчёски по образцу card_v8 (бирюзовой).

Расчёска НЕ вырезается: фото с чистым белым фоном кладётся на фон образца умножением
(белое фото → ровно фон образца, родные полутени фото сохраняются). Маска rembg нужна
только для расчётов: масштаб, выноски, проверки наложений, контактная тень.

Запуск:  python3 card_yellow.py [yellow.png] [card.png] [output_dir]
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
    low = A2 * np.clip((yy - (pbot - CONTACT_ZONE * ph)) / (0.03 * ph), 0, 1)
    sa = shifted(A2, SHADOW_OFFSET, SHADOW_BLUR) * SHADOW_OPACITY + shifted(low, CONTACT_OFFSET, CONTACT_BLUR) * CONTACT_OPACITY
    sa = np.clip(sa, 0, 0.6) * (1 - A2)               # тень только на фоне, не на самом товаре
    out = lerp(out, hex2rgb(SHADOW_COLOR), sa)
    out = out * Yn2                                     # расчёска «на белом»: белый фото = фон образца

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
    x1 = v8.RIGHT_CX - a1.shape[1] / 2
    v8.paste(out, occ, a1, c1_, x1, v8.RIGHT_TOP)
    dx = ((1093 + 1411) / 2 - (1109 + 1367) / 2) * v8.K_TEXT
    x2 = v8.RIGHT_CX + dx - a2_.shape[1] / 2
    y2 = v8.RIGHT_TOP + a1.shape[0] + (1164 - 1100) * v8.K_TEXT
    v8.paste(out, occ, a2_, c2_, x2, y2)
    right_box = (int(min(x1, x2)), v8.RIGHT_TOP, int(max(x1 + a1.shape[1], x2 + a2_.shape[1])), int(y2 + a2_.shape[0]))

    # правая выноска: горизонталь начинается на ручке, угол над «2 в 1», вниз к надписи
    corner_x = v8.RIGHT_CX + (1252 - (1109 + 1367) / 2) * v8.K_TEXT
    vy1 = v8.RIGHT_TOP - (1007 - 966) * v8.K_TEXT
    line_y = None
    for yv in range(int(vy1) - 60, int(ptop) + 40, -2):          # снизу вверх: первая строка, где ручка близко
        cols = np.where(A2[yv, :int(corner_x)] > 0.5)[0]
        if len(cols) and corner_x - cols.max() < 260:
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
    res.save(OUT_DIR / "card_yellow.png")
    cw, ch = 600, round(h * 600 / w)
    cmp_ = Image.new("RGB", (cw * 2 + 30, ch + 60), "white")
    cmp_.paste(Image.open(OUT_DIR / "card_v8.png").convert("RGB").resize((cw, ch), Image.LANCZOS), (0, 60))
    cmp_.paste(res.resize((cw, ch), Image.LANCZOS), (cw + 30, 60))
    fd = ImageFont.truetype(str(pc.FONT_PATH), 34)
    ImageDraw.Draw(cmp_).text((cw // 2, 30), "бирюзовая (card_v8)", font=fd, fill=(40, 40, 40), anchor="mm")
    ImageDraw.Draw(cmp_).text((cw + 30 + cw // 2, 30), "card_yellow", font=fd, fill=(40, 40, 40), anchor="mm")
    cmp_.save(OUT_DIR / "card_yellow_compare.png")
    print("ПРОБЛЕМЫ:\n  " + "\n  ".join(problems) if problems else "всё влезло, наложений нет")
    print("готово:", OUT_DIR / "card_yellow.png", OUT_DIR / "card_yellow_compare.png")


if __name__ == "__main__":
    main()
