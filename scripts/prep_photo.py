"""
Prepare the dp (a flat illustration) for clean ASCII conversion:
  1. convert to grayscale
  2. bilateral-smooth away paper texture while keeping drawn lines sharp
  3. stretch tones so background lands at white and hair/lines stay dark
  4. darken the line work (difference-of-gaussians ridges)
  5. crop square around the subject

Output: source-prepped.png (grayscale), consumed by make_ascii_svg.py.

    python scripts/prep_photo.py <input.png> [output.png]
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
INP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "pfp-source.png")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "pfp-prep.png")

LINE_WEIGHT = 0.6     # how hard drawn lines are pushed toward black

# 1. load grayscale
gray = cv2.cvtColor(np.array(Image.open(INP).convert("RGB")), cv2.COLOR_RGB2GRAY)

# 2. smooth texture, keep edges
smooth = gray
for _ in range(3):
    smooth = cv2.bilateralFilter(smooth, 9, 40, 9)

# 3. tone stretch: bright 2% -> white, dark 2% -> black.
#    For a flat illustration on white background this keeps the bg at 255
#    and spreads the subject's dark lines across the full range.
lo, hi = np.percentile(smooth, [2, 98])
tone = np.clip((smooth.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)

# 4. dark-on-light ridges -> darken
fine = cv2.GaussianBlur(smooth, (0, 0), 1.5).astype(np.float32)
coarse = cv2.GaussianBlur(smooth, (0, 0), 6).astype(np.float32)
lines = np.clip((coarse - fine) / 40.0, 0, 1)
out = np.clip(tone - LINE_WEIGHT * lines, 0, 1) * 255

# 5. force background to pure white, square crop around the subject
mask = smooth < 245          # subject = non-white
ys, xs = np.where(mask)
if len(xs) == 0:             # fallback: use whole image
    xs, ys = [0, gray.shape[1] - 1], [0, gray.shape[0] - 1]
side = max(xs.max() - xs.min(), ys.max() - ys.min()) + 60
cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
canvas = np.full((side, side), 255, np.uint8)
x0, y0 = cx - side // 2, cy - side // 2
sx0, sy0 = max(x0, 0), max(y0, 0)
sx1, sy1 = min(x0 + side, out.shape[1]), min(y0 + side, out.shape[0])
canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = out[sy0:sy1, sx0:sx1].astype(np.uint8)

Image.fromarray(canvas, mode="L").save(OUT)
print("wrote", OUT, canvas.shape)
