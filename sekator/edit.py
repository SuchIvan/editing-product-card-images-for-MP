import sys
from PIL import Image, ImageFont, ImageDraw
import numpy as np
src, out, s2 = sys.argv[1], sys.argv[2], float(sys.argv[3])
im = Image.open(src).convert('RGB')
a = np.asarray(im).astype(float)
# erase old text: vertical interpolation
y0, y1, x0, x1 = 619, 668, 38, 274
for x in range(x0, x1):
    top, bot = a[y0, x], a[y1, x]
    for y in range(y0+1, y1):
        t = (y-y0)/(y1-y0); a[y, x] = top*(1-t)+bot*t
im = Image.fromarray(a.clip(0,255).astype('uint8'))
W, H = im.size
cx = 148.5
lines = [('ровный срез', 39.0, 628-9), ('острая заточка', s2, None)]
mask = Image.new('L', (W, H), 0); d = ImageDraw.Draw(mask)
y = None; boxes=[]
for i,(txt, sz, top) in enumerate(lines):
    f = ImageFont.truetype('m400.ttf', sz)
    bb = f.getbbox(txt); w = bb[2]-bb[0]
    if top is None: top = prev_base + sz*0.30
    x = cx - w/2 - bb[0]
    d.text((x, top), txt, font=f, fill=255)
    prev_base = top + f.getmetrics()[0]
    boxes.append((x, x+w))
xl = min(b[0] for b in boxes); xr = max(b[1] for b in boxes)
xs = np.arange(W)
t = np.clip((xs-xl)/(xr-xl), 0, 1)[None,:,None]
c1 = np.array([248,255,248]); c2 = np.array([205,212,204])
col = c1*(1-t)+c2*t
m = np.asarray(mask).astype(float)[:,:,None]/255
b = np.asarray(im).astype(float)
res = b*(1-m)+col*m
Image.fromarray(res.clip(0,255).astype('uint8')).save(out, quality=95) if out.endswith(('.webp','.jpg')) else Image.fromarray(res.clip(0,255).astype('uint8')).save(out)
print(boxes)
