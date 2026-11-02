"""Show six synchronized camera images beside an ego-centric Occ3D top view."""
from __future__ import annotations
from occ_label_pipeline.context import CFG, CONFIG_PATH, PROJECT_ROOT, ENGINE_ROOT, OUTPUT as CAPTURE_ROOT, DATA_ROOT, TRACKER_ROOT, NFRAMES
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from generate_occ import CLASSES, PALETTE
CAMERA_LAYOUT = (('FL', 'F', 'FR'), ('BL', 'B', 'BR'))

def top_view(semantics: np.ndarray, visibility: np.ndarray, bounds: np.ndarray, voxel: float, extent: float, size: int, show_free: bool=False) -> Image.Image:
    """Render +X forward/up and +Y left/left, centered on the ego vehicle."""
    if semantics.shape != visibility.shape or semantics.ndim != 3:
        raise ValueError('semantics and visibility must be matching 3D arrays')
    lo = bounds[:3]
    ix0 = max(0, int(np.floor((-extent - lo[0]) / voxel)))
    ix1 = min(semantics.shape[0], int(np.ceil((extent - lo[0]) / voxel)))
    iy0 = max(0, int(np.floor((-extent - lo[1]) / voxel)))
    iy1 = min(semantics.shape[1], int(np.ceil((extent - lo[1]) / voxel)))
    sem = semantics[ix0:ix1, iy0:iy1]
    vis = visibility[ix0:ix1, iy0:iy1]
    occupied = (sem != 17) & vis
    known_free = (sem == 17) & vis
    has_occ = occupied.any(axis=2)
    reverse_z = occupied[:, :, ::-1].argmax(axis=2)
    top_z = sem.shape[2] - 1 - reverse_z
    label = np.take_along_axis(sem, top_z[:, :, None], axis=2)[:, :, 0]
    free_column = known_free.any(axis=2)
    rgb = np.full((*has_occ.shape, 3), (28, 30, 35), dtype=np.uint8)
    if show_free:
        rgb[free_column] = (55, 60, 68)
    palette = np.asarray(PALETTE, dtype=np.float64)
    if palette.max() <= 1.0:
        palette = np.rint(palette * 255)
    rgb[has_occ] = palette[label[has_occ]].astype(np.uint8)
    rgb = rgb[::-1, ::-1]
    image = Image.fromarray(rgb, 'RGB').resize((size, size), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image, 'RGBA')
    scale = size / (2 * extent)
    center = size / 2
    for metres in np.arange(-extent, extent + 1e-06, 5.0):
        px = center - metres * scale
        py = center - metres * scale
        draw.line((px, 0, px, size), fill=(255, 255, 255, 40), width=1)
        draw.line((0, py, size, py), fill=(255, 255, 255, 40), width=1)
    half_w, half_l = (0.95 * scale, 2.4 * scale)
    draw.rounded_rectangle((center - half_w, center - half_l, center + half_w, center + half_l), radius=4, fill=(245, 245, 245, 230), outline=(15, 15, 15, 255), width=2)
    draw.polygon([(center, center - half_l - 22), (center - 10, center - half_l - 5), (center + 10, center - half_l - 5)], fill=(255, 70, 55, 255))
    draw.text((center + half_w + 7, center - half_l), 'EGO', fill=(255, 255, 255, 255))
    draw.text((center + 12, 8), '+X FRONT', fill=(255, 255, 255, 255))
    draw.text((8, center + 8), '+Y LEFT', fill=(255, 255, 255, 255))
    draw.text((size - 70, center + 8), '-Y', fill=(255, 255, 255, 255))
    return image

def camera_montage(frame: Path, tile: tuple[int, int]) -> Image.Image:
    width, height = tile
    canvas = Image.new('RGB', (3 * width, 2 * height), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for row, names in enumerate(CAMERA_LAYOUT):
        for column, name in enumerate(names):
            image = Image.open(frame / f'{name}.png').convert('RGB')
            if name in ('B', 'BL', 'BR'):
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            image.thumbnail((width, height), Image.Resampling.LANCZOS)
            x = column * width + (width - image.width) // 2
            y = row * height + (height - image.height) // 2
            canvas.paste(image, (x, y))
            draw.rectangle((column * width, row * height, column * width + 56, row * height + 25), fill=(0, 0, 0))
            label = f'{name} (display mirrored)' if name in ('B', 'BL', 'BR') else name
            draw.text((column * width + 8, row * height + 5), label, fill=(255, 255, 255))
    return canvas

def compose(frame: Path, labels: Path, output: Path | None, mask: str, extent: float, bev_size: int) -> Image.Image:
    metadata = json.loads(labels.with_name('metadata.json').read_text(encoding='utf-8'))
    with np.load(labels, allow_pickle=False) as data:
        semantics = data['semantics']
        visibility = data[f'mask_{mask}']
    bev = top_view(semantics, visibility, np.asarray(metadata['bounds'], float), float(metadata['voxel_size']), extent, bev_size)
    tile = (bev_size * 2 // 3, bev_size // 2)
    cameras = camera_montage(frame, tile)
    header, footer = (44, 88)
    canvas = Image.new('RGB', (cameras.width + bev.width, max(cameras.height, bev.height) + header + footer), (18, 20, 24))
    canvas.paste(cameras, (0, header))
    canvas.paste(bev, (cameras.width, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 14), f'Frame {frame.name} — synchronized cameras', fill=(245, 245, 245))
    draw.text((cameras.width + 14, 14), f'Occ top view — front up — ±{extent:g} m — mask_{mask}', fill=(245, 245, 245))
    present = sorted((int(v) for v in np.unique(semantics[visibility & (semantics != 17)])))
    palette = np.asarray(PALETTE, float)
    if palette.max() <= 1.0:
        palette *= 255
    x, y = (14, header + max(cameras.height, bev.height) + 14)
    for class_id in present:
        color = tuple(palette[class_id].astype(np.uint8).tolist())
        draw.rectangle((x, y, x + 13, y + 13), fill=color)
        text = CLASSES[class_id]
        draw.text((x + 18, y), text, fill=(225, 225, 225))
        x += 24 + 7 * len(text)
        if x > canvas.width - 160:
            x, y = (14, y + 24)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output)
    return canvas

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('frame', help='Frame ID, e.g. 000141')
    parser.add_argument('--data-root', type=Path, default=Path('customize_data/40frame'))
    parser.add_argument('--occ-root', type=Path, default=Path('sparse_occ_output'))
    parser.add_argument('--mask', choices=('lidar', 'camera'), default='lidar')
    parser.add_argument('--extent', type=float, default=20.0, help='Metres around ego shown in BEV')
    parser.add_argument('--bev-size', type=int, default=720)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.extent <= 0 or args.bev_size < 300:
        parser.error('--extent must be positive and --bev-size at least 300')
    output = args.output or args.occ_root / args.frame / 'camera_occ_topview.png'
    compose(args.data_root / args.frame, args.occ_root / args.frame / 'labels.npz', output, args.mask, args.extent, args.bev_size)
    print(f'Wrote {output}')
if __name__ == '__main__':
    main()
