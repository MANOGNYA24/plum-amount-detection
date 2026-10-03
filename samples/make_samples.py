"""Generates sample receipt images (clean + noisy) for testing/demo.
Run: python samples/make_samples.py"""
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path(__file__).parent
FONT_PATHS = ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
              "/System/Library/Fonts/Menlo.ttc", "C:/Windows/Fonts/consola.ttf"]


def font(size):
    for p in FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def receipt(lines, path, noisy=False):
    f = font(28)
    img = Image.new("L", (700, 60 + 48 * len(lines)), 255)
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        d.text((40, 30 + 48 * i), line, fill=0, font=f)
    if noisy:
        rnd = random.Random(42)
        px = img.load()
        for _ in range(img.width * img.height // 40):          # salt & pepper
            x, y = rnd.randrange(img.width), rnd.randrange(img.height)
            px[x, y] = rnd.choice([0, 255])
        img = img.rotate(2, expand=True, fillcolor=255)          # slight skew
        img = img.filter(ImageFilter.GaussianBlur(0.8))          # camera blur
    img.convert("RGB").save(path, quality=70)
    print("wrote", path)


receipt(["CITY CARE HOSPITAL", "Consultation Fee  500", "Medicines  700",
         "Total: INR 1200", "Paid: 1000", "Due: 200"], OUT / "bill_clean.jpg")
receipt(["CITY CARE HOSPITAL", "T0tal: Rs l200", "Pald: 1000", "Due: 200"],
        OUT / "bill_noisy.jpg", noisy=True)
receipt(["Thank you for visiting", "Get well soon!"], OUT / "no_amounts.jpg")
