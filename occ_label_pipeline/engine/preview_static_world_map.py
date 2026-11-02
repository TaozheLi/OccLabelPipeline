"""Render a labeled top-view preview of the globally fused static cloud."""
from occ_label_pipeline.context import CFG, CONFIG_PATH, PROJECT_ROOT, ENGINE_ROOT, OUTPUT as CAPTURE_ROOT, DATA_ROOT, TRACKER_ROOT, NFRAMES
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from visualize_point_labels import PALETTE
p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
a = p.parse_args()
with np.load(a.root / 'static_world.npz') as d:
    x = d['xyz']
    labels = d['semantics']
lo = x[:, :2].min(0)
hi = x[:, :2].max(0)
size = 1800
scale = (size - 80) / max(hi - lo)
uv = ((x[:, :2] - lo) * scale + 40).astype(int)
rgb = np.rint(PALETTE[labels] * 255).astype(np.uint8)
canvas = np.full((size, size, 3), 14, np.uint8)
order = np.argsort(x[:, 2])
canvas[size - 1 - uv[order, 0], uv[order, 1]] = rgb[order]
im = Image.fromarray(canvas)
draw = ImageDraw.Draw(im)
draw.text((20, 15), 'World static cloud | PTv3 static + stationary tracked objects | 0.1 m', fill='white')
im.save(a.root / 'topview.png')
