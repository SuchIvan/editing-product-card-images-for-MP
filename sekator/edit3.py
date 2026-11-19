import sys
from PIL import Image, ImageFont, ImageDraw, ImageFilter
import numpy as np
src, out, mode = sys.argv[1], sys.argv[2], sys.argv[3]   # mode: one | two
base = Image.open(src).convert('RGB'); W, H = base.size
a = np.asarray(base).astype(float)
def erase(y0, y1, x0, x1):
    for x in range(x0, x1):
        top, bot = a[y0, x].copy(), a[y1, x].copy()
        for y in range(y0+1, y1):
            t = (y-y0)/(y1-y0); a[y, x] = top*(1-t)+bot*t
erase(619, 668, 38, 274)          # ровный срез
cm = np.asarray(base).astype(float)[736:815, 580:736].copy()  # "15 СМ" patch
erase(732, 994, 578, 845)         # 15 СМ + правые надписи
img = Image.fromarray(a.clip(0,255).astype('uint8'))

SZ = 42.9
F = ImageFont.truetype('m700.ttf', SZ)
asc = F.getmetrics()[0]
xh = F.getbbox('х')  # x-height ink box

def white_text(img, txt, x, base_y, c1=(248,255,248), c2=(228,234,226)):
    m = Image.new('L', img.size, 0); ImageDraw.Draw(m).text((x, base_y-asc), txt, font=F, fill=255)
    bb = F.getbbox(txt); x0, x1 = x+bb[0], x+bb[2]
    t = np.clip((np.arange(W)-x0)/max(1,(x1-x0)), 0, 1)[None,:,None]
    col = np.array(c1)*(1-t)+np.array(c2)*t
    mm = np.asarray(m).astype(float)[:,:,None]/255
    b = np.asarray(img).astype(float)
    return Image.fromarray((b*(1-mm)+col*mm).clip(0,255).astype('uint8'))

def badge(img, lines, x, y, padx=24, h1=66, lh=52):
    ws = [F.getbbox(t)[2]-F.getbbox(t)[0] for t in lines]
    bw = max(ws)+2*padx; bh = h1 + lh*(len(lines)-1)
    # тень
    sh = Image.new('L', img.size, 0); ImageDraw.Draw(sh).rounded_rectangle((x, y+4, x+bw, y+bh+4), 11, fill=110)
    sh = sh.filter(ImageFilter.GaussianBlur(6))
    img = Image.composite(Image.new('RGB', img.size, (5,15,8)), img, sh)
    # градиент как у «универсальный»
    t = np.linspace(0,1,bh)[:,None,None]
    grad = np.array([247,209,137])*(1-t)+np.array([163,130,67])*t
    grad = np.repeat(grad, bw, 1).clip(0,255).astype('uint8')
    gi = Image.fromarray(grad)
    m = Image.new('L', (bw, bh), 0); ImageDraw.Draw(m).rounded_rectangle((0,0,bw-1,bh-1), 11, fill=255)
    img.paste(gi, (x, y), m)
    d = ImageDraw.Draw(img)
    for i,(txt,w) in enumerate(zip(lines, ws)):
        cy = y + h1/2 + i*lh            # центр строчных букв
        by = cy + (xh[3]-xh[1])/2        # базовая линия
        tx = x + (bw-w)/2 - F.getbbox(txt)[0]
        d.text((tx, by-asc), txt, font=F, fill=(22,27,12))
    return img, bw, bh

# левый блок
LX = 18
img = white_text(img, 'ровный срез', LX, 652)
if mode == 'one':
    img, bw, bh = badge(img, ['острая заточка'], LX-6, 670)
else:
    img, bw, bh = badge(img, ['острая', 'заточка'], LX-6, 670)

# правый блок
RX = 519
cp = Image.fromarray(cm.clip(0,255).astype('uint8'))
cmask = Image.fromarray(((cm.mean(2) > 120)*255).astype('uint8')).filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(0.6))
img.paste(cp, (RX-5, 736), cmask)   # «15 СМ» сдвинут влево к колонке
img = white_text(img, 'диаметр среза', RX, 868)
img = white_text(img, 'рукоядка', RX, 940)
img = white_text(img, 'под обе руки', RX, 992)
img.save(out)
