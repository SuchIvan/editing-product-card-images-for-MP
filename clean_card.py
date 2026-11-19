#!/usr/bin/env python3
"""Чистка готовой карточки: убрать все надписи/плашки/выноску, оставить расчёску на белом.

Фон карточки моделируется гладким полиномом и делится на него: фон → ровно белый (1.0),
расчёска остаётся как есть (не вырезается). Всё, что дальше 8 px от расчёски, — белое.
Линия-указатель, заходящая на ручку, убирается заливкой по соседям (inpaint).

Запуск:  python3 clean_card.py grayblue.png grayblue_clean.png
"""
import sys

import cv2
import numpy as np
from PIL import Image

import peach_card as pc

SRC = sys.argv[1] if len(sys.argv) > 1 else "grayblue.png"
DST = sys.argv[2] if len(sys.argv) > 2 else "grayblue_clean.png"
LINE_RECTS = [(596, 497, 753, 506), (745, 497, 754, 622)]   # grayblue.png: горизонталь и вертикаль
KEEP_AROUND = 8        # px вокруг расчёски, где пиксели фото сохраняются (край, родная полутень)


def main():
    img = Image.open(SRC).convert("RGB")
    P = np.asarray(img, np.float32) / 255
    h, w = P.shape[:2]
    A = pc.product_alpha(img)

    # линия-указатель старой карточки: прямоугольники по замеру (x0, y0, x1, y1), с запасом на сглаживание
    line = np.zeros((h, w), np.uint8)
    for x0, y0, x1, y1 in LINE_RECTS:
        line[y0:y1, x0:x1] = 1
    thin = line
    print(f"линия-указатель: {int(thin.sum())} px")

    # альфа товара без линии: на строках линии берём минимум соседей сверху/снизу
    # альфа под линией — заливкой по соседям (край ручки продолжается без выемки и без «наплыва»)
    A = cv2.inpaint((A * 255).astype(np.uint8), line, 5, cv2.INPAINT_TELEA).astype(np.float32) / 255
    n, lab, st, _ = cv2.connectedComponentsWithStats((A > 0.5).astype(np.uint8))
    keep = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    prod = (lab == keep).astype(np.uint8)
    prod = cv2.morphologyEx(prod, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # гладкая модель фона (полином 2-й степени, с отбраковкой надписей)
    far = cv2.dilate(prod, np.ones((61, 61), np.uint8)) == 0
    gy, gx = np.mgrid[0:h:4, 0:w:4]
    Xf = lambda x, y: np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], -1)
    xn, yn = gx / w, gy / h
    m = far[::4, ::4]
    vals = P[::4, ::4]
    for _ in range(4):
        coef = np.linalg.lstsq(Xf(xn[m], yn[m]), vals[m], rcond=None)[0]
        res = np.abs(Xf(xn, yn) @ coef - vals).max(-1)
        m = far[::4, ::4] & (res < 0.03)
    fy, fx = np.mgrid[0:h, 0:w].astype(np.float32)
    Bfit = (Xf(fx / w, fy / h) @ coef).astype(np.float32)
    print(f"фон: модель по {m.sum()} точкам, {Bfit.min() * 255:.0f}–{Bfit.max() * 255:.0f}")

    N = np.clip(P / np.maximum(Bfit, 1e-3), 0, 1)
    # линия на ручке → заливка по соседям
    near = cv2.dilate(prod, np.ones((2 * KEEP_AROUND + 7,) * 2, np.uint8)) > 0
    on_prod = (line > 0) & near                      # линия на ручке и в её полутени — заливка
    N8 = (N * 255).astype(np.uint8)
    N8 = cv2.inpaint(N8, cv2.dilate(on_prod.astype(np.uint8), np.ones((3, 3), np.uint8)), 4, cv2.INPAINT_TELEA)
    N = N8.astype(np.float32) / 255
    # всё вне расчёски (+KEEP_AROUND) → белое; плавный переход 2 px
    zone = cv2.dilate(prod, np.ones((2 * KEEP_AROUND + 1,) * 2, np.uint8)).astype(np.float32)
    zone = cv2.GaussianBlur(zone, (0, 0), 1.0)
    N = N * zone[..., None] + (1 - zone[..., None])
    # остатки линии вне расчёски внутри зоны KEEP_AROUND
    off = (line > 0) & (on_prod == 0)
    N[off] = 1.0
    Image.fromarray((N * 255 + 0.5).astype(np.uint8)).save(DST)
    left = ((np.abs(N - 1).max(2) > 0.04) & (zone < 0.01)).sum()
    print(f"готово: {DST} ({w}×{h}); тёмных пикселей вне расчёски: {left}")


if __name__ == "__main__":
    main()
