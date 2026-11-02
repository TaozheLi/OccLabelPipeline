"""Project PCD or KITTI point clouds into 2D images.

Outputs a PNG plus a NumPy ``.npz`` sidecar.  The sidecar contains a
``point_index`` array whose pixels refer to rows in the original PCD file
(-1 means that no point was projected into that pixel).
"""
from __future__ import annotations
from occ_label_pipeline.context import CFG, CONFIG_PATH, PROJECT_ROOT, ENGINE_ROOT, OUTPUT as CAPTURE_ROOT, DATA_ROOT, TRACKER_ROOT, NFRAMES
import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from PIL import Image
PCD_DTYPES = {('F', 4): '<f4', ('F', 8): '<f8', ('I', 1): 'i1', ('I', 2): '<i2', ('I', 4): '<i4', ('U', 1): 'u1', ('U', 2): '<u2', ('U', 4): '<u4'}

def read_pcd(path: Path) -> Tuple[Dict[str, np.ndarray], dict]:
    """Read ASCII or uncompressed binary PCD while preserving point order."""
    header_lines = []
    with path.open('rb') as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError('Invalid PCD: DATA line was not found')
            decoded = line.decode('ascii', errors='strict').strip()
            header_lines.append(decoded)
            if decoded.upper().startswith('DATA '):
                payload = stream.read()
                break
    header = {}
    for line in header_lines:
        if not line or line.startswith('#'):
            continue
        key, *values = line.split()
        header[key.upper()] = values
    fields = header.get('FIELDS') or header.get('FIELD')
    if not fields:
        raise ValueError('Invalid PCD: FIELDS is missing')
    sizes = list(map(int, header.get('SIZE', ['4'] * len(fields))))
    types = [v.upper() for v in header.get('TYPE', ['F'] * len(fields))]
    counts = list(map(int, header.get('COUNT', ['1'] * len(fields))))
    points = int((header.get('POINTS') or header.get('WIDTH') or ['0'])[0])
    data_kind = header['DATA'][0].lower()
    if not len(fields) == len(sizes) == len(types) == len(counts):
        raise ValueError('FIELDS, SIZE, TYPE and COUNT lengths do not match')
    expanded_names = []
    for name, count in zip(fields, counts):
        expanded_names.extend([name] if count == 1 else [f'{name}_{i}' for i in range(count)])
    if data_kind == 'ascii':
        matrix = np.loadtxt(payload.splitlines(), dtype=np.float64, ndmin=2)
        if matrix.shape[1] != len(expanded_names):
            raise ValueError('ASCII PCD column count does not match its header')
        arrays = {name: matrix[:, i] for i, name in enumerate(expanded_names)}
    elif data_kind == 'binary':
        dtype_fields = []
        for name, size, kind, count in zip(fields, sizes, types, counts):
            try:
                base = np.dtype(PCD_DTYPES[kind, size])
            except KeyError as exc:
                raise ValueError(f'Unsupported PCD field type: TYPE={kind} SIZE={size}') from exc
            dtype_fields.append((name, base) if count == 1 else (name, base, (count,)))
        records = np.frombuffer(payload, dtype=np.dtype(dtype_fields), count=points)
        arrays = {}
        for name, count in zip(fields, counts):
            if count == 1:
                arrays[name] = records[name]
            else:
                for i in range(count):
                    arrays[f'{name}_{i}'] = records[name][:, i]
    elif data_kind == 'binary_compressed':
        raise ValueError('binary_compressed PCD is not supported; convert it to ASCII or binary first')
    else:
        raise ValueError(f'Unsupported DATA encoding: {data_kind}')
    actual_points = len(next(iter(arrays.values())))
    if points and actual_points != points:
        raise ValueError(f'Header says {points} points, but file contains {actual_points}')
    metadata = {k.lower(): v for k, v in header.items()}
    metadata['point_count'] = actual_points
    return (arrays, metadata)

def read_kitti_bin(path: Path) -> Tuple[Dict[str, np.ndarray], dict]:
    """Read KITTI's float32 [x, y, z, reflectance] point format."""
    raw = np.fromfile(path, dtype='<f4')
    if raw.size % 4:
        raise ValueError(f'Invalid KITTI point cloud: {path} does not contain groups of 4 float32 values')
    points = raw.reshape(-1, 4)
    fields = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2], 'intensity': points[:, 3]}
    return (fields, {'format': 'kitti-bin', 'point_count': len(points), 'fields': ['x', 'y', 'z', 'intensity']})

def read_point_cloud(path: Path) -> Tuple[Dict[str, np.ndarray], dict]:
    suffix = path.suffix.lower()
    if suffix == '.pcd':
        fields, metadata = read_pcd(path)
        metadata['format'] = 'pcd'
        return (fields, metadata)
    if suffix == '.bin':
        return read_kitti_bin(path)
    raise ValueError(f"Unsupported point-cloud extension '{suffix}'; expected .pcd or KITTI .bin")

def xyz_and_values(fields: Dict[str, np.ndarray], color_by: str) -> Tuple[np.ndarray, np.ndarray]:
    for axis in ('x', 'y', 'z'):
        if axis not in fields:
            raise ValueError(f"PCD has no '{axis}' field")
    xyz = np.column_stack((fields['x'], fields['y'], fields['z'])).astype(np.float64)
    if color_by == 'range':
        values = np.linalg.norm(xyz, axis=1)
    elif color_by in fields:
        values = np.asarray(fields[color_by], dtype=np.float64)
    else:
        available = ', '.join(sorted(fields))
        raise ValueError(f"Unknown --color-by field '{color_by}'. Available: {available}")
    return (xyz, values)

def rasterize(uv: np.ndarray, depth: np.ndarray, values: np.ndarray, valid: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-buffer points into an image, keeping the closest point per pixel."""
    ids = np.flatnonzero(valid)
    u = np.floor(uv[valid, 0]).astype(np.int64)
    v = np.floor(uv[valid, 1]).astype(np.int64)
    d = depth[valid]
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height) & np.isfinite(d) & (d > 0)
    u, v, d, ids = (u[inside], v[inside], d[inside], ids[inside])
    flat = v * width + u
    order = np.lexsort((d, flat))
    flat, ids, d = (flat[order], ids[order], d[order])
    first = np.r_[True, flat[1:] != flat[:-1]]
    flat, ids, d = (flat[first], ids[first], d[first])
    point_index = np.full(width * height, -1, dtype=np.int64)
    depth_image = np.full(width * height, np.nan, dtype=np.float32)
    value_image = np.full(width * height, np.nan, dtype=np.float32)
    point_index[flat] = ids
    depth_image[flat] = d
    value_image[flat] = values[ids]
    shape = (height, width)
    return (value_image.reshape(shape), depth_image.reshape(shape), point_index.reshape(shape))

def project_range(xyz: np.ndarray, width: int, height: int, fov_up: float, fov_down: float):
    distance = np.linalg.norm(xyz, axis=1)
    yaw = np.arctan2(xyz[:, 1], xyz[:, 0])
    pitch = np.arcsin(np.divide(xyz[:, 2], distance, out=np.zeros_like(distance), where=distance > 0))
    up, down = (math.radians(fov_up), math.radians(fov_down))
    total = up - down
    u = 0.5 * (1.0 - yaw / math.pi) * width
    v = (up - pitch) / total * height
    valid = np.isfinite(xyz).all(axis=1) & (distance > 0) & (pitch <= up) & (pitch >= down)
    return (np.column_stack((u, v)), distance, valid)

def project_bev(xyz: np.ndarray, width: int, height: int, bounds):
    xmin, xmax, ymin, ymax = bounds
    u = (xyz[:, 1] - ymin) / (ymax - ymin) * width
    v = (xmax - xyz[:, 0]) / (xmax - xmin) * height
    depth = np.nanmax(xyz[:, 2]) - xyz[:, 2] + 1e-06
    valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 0] >= xmin) & (xyz[:, 0] < xmax) & (xyz[:, 1] >= ymin) & (xyz[:, 1] < ymax)
    return (np.column_stack((u, v)), depth, valid)

def project_camera(xyz: np.ndarray, width: int, height: int, intrinsics: np.ndarray, extrinsics: np.ndarray):
    homogeneous = np.column_stack((xyz, np.ones(len(xyz))))
    camera = (extrinsics @ homogeneous.T).T[:, :3]
    z = camera[:, 2]
    pixels_h = (intrinsics @ camera.T).T
    uv = np.divide(pixels_h[:, :2], z[:, None], out=np.zeros((len(xyz), 2)), where=z[:, None] != 0)
    valid = np.isfinite(camera).all(axis=1) & (z > 0)
    return (uv, z, valid)

def read_kitti_calibration(calib_dir: Path, camera: int) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Return KITTI rectified-camera transform and 3x4 projection matrix."""
    if camera not in (0, 1, 2, 3):
        raise ValueError('KITTI camera must be 0, 1, 2, or 3')

    def read_file(path: Path) -> Dict[str, np.ndarray]:
        result = {}
        for line in path.read_text().splitlines():
            if ':' not in line:
                continue
            key, values = line.split(':', 1)
            try:
                result[key.strip()] = np.fromstring(values, sep=' ', dtype=np.float64)
            except ValueError:
                continue
        return result
    velo_path = calib_dir / 'calib_velo_to_cam.txt'
    camera_path = calib_dir / 'calib_cam_to_cam.txt'
    if not velo_path.exists() or not camera_path.exists():
        raise ValueError(f'{calib_dir} must contain calib_velo_to_cam.txt and calib_cam_to_cam.txt')
    velo = read_file(velo_path)
    cam = read_file(camera_path)
    projection_key = f'P_rect_0{camera}'
    for source, key, size in ((velo, 'R', 9), (velo, 'T', 3), (cam, 'R_rect_00', 9), (cam, projection_key, 12)):
        if key not in source or source[key].size != size:
            raise ValueError(f'Missing or malformed KITTI calibration value: {key}')
    velo_to_cam = np.eye(4)
    velo_to_cam[:3, :3] = velo['R'].reshape(3, 3)
    velo_to_cam[:3, 3] = velo['T']
    rectification = np.eye(4)
    rectification[:3, :3] = cam['R_rect_00'].reshape(3, 3)
    rectified_velo_to_camera = rectification @ velo_to_cam
    projection = cam[projection_key].reshape(3, 4)
    metadata = {'camera': camera, 'velo_to_rectified_camera': rectified_velo_to_camera.tolist(), 'projection': projection.tolist()}
    return (rectified_velo_to_camera, projection, metadata)

def project_kitti_camera(xyz: np.ndarray, transform: np.ndarray, projection: np.ndarray):
    homogeneous = np.column_stack((xyz, np.ones(len(xyz))))
    camera_h = (transform @ homogeneous.T).T
    image_h = (projection @ camera_h.T).T
    denominator = image_h[:, 2]
    uv = np.divide(image_h[:, :2], denominator[:, None], out=np.zeros((len(xyz), 2)), where=denominator[:, None] != 0)
    depth = camera_h[:, 2]
    valid = np.isfinite(camera_h).all(axis=1) & np.isfinite(uv).all(axis=1) & (depth > 0)
    return (uv, depth, valid)

def colored_image(values: np.ndarray, percentile_low: float, percentile_high: float) -> Tuple[np.ndarray, dict]:
    finite = np.isfinite(values)
    rgb = np.zeros((*values.shape, 3), dtype=np.uint8)
    if not finite.any():
        return (rgb, {'value_min': None, 'value_max': None})
    lo, hi = np.percentile(values[finite], [percentile_low, percentile_high])
    if hi <= lo:
        hi = lo + 1.0
    t = np.zeros_like(values, dtype=np.float64)
    t[finite] = np.clip((values[finite] - lo) / (hi - lo), 0.0, 1.0)
    stops = np.array([[24, 36, 82], [28, 144, 153], [246, 211, 69], [194, 45, 42]])
    position = t * (len(stops) - 1)
    left = np.floor(position).astype(np.int64)
    left = np.clip(left, 0, len(stops) - 2)
    fraction = (position - left)[..., None]
    mapped = stops[left] * (1.0 - fraction) + stops[left + 1] * fraction
    rgb[finite] = mapped[finite].astype(np.uint8)
    return (rgb, {'value_min': float(lo), 'value_max': float(hi)})

def overlay_points(background: Image.Image, colors: np.ndarray, point_index: np.ndarray, radius: int, alpha: float) -> np.ndarray:
    """Overlay projected point colors on an RGB image without changing the index map."""
    base = np.asarray(background.convert('RGB'), dtype=np.float32)
    layer = np.zeros_like(base)
    occupied = np.zeros(point_index.shape, dtype=bool)
    source = point_index >= 0
    height, width = source.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            src_y0, src_y1 = (max(0, -dy), min(height, height - dy))
            src_x0, src_x1 = (max(0, -dx), min(width, width - dx))
            dst_y0, dst_y1 = (src_y0 + dy, src_y1 + dy)
            dst_x0, dst_x1 = (src_x0 + dx, src_x1 + dx)
            mask = source[src_y0:src_y1, src_x0:src_x1]
            region = layer[dst_y0:dst_y1, dst_x0:dst_x1]
            region[mask] = colors[src_y0:src_y1, src_x0:src_x1][mask]
            occupied[dst_y0:dst_y1, dst_x0:dst_x1] |= mask
    result = base.copy()
    result[occupied] = (1.0 - alpha) * base[occupied] + alpha * layer[occupied]
    return np.clip(result, 0, 255).astype(np.uint8)

def load_matrix(path: Path, shape: tuple) -> np.ndarray:
    matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.shape != shape:
        raise ValueError(f'{path} must contain a {shape[0]}x{shape[1]} matrix')
    return matrix

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Input .pcd file or KITTI .bin point cloud')
    parser.add_argument('--output', type=Path, default=Path('projection.png'))
    parser.add_argument('--mode', choices=('range', 'bev', 'camera', 'kitti-camera'), default='range')
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=64)
    parser.add_argument('--color-by', default='range', help='range or a numeric PCD field, e.g. intensity or z')
    parser.add_argument('--fov-up', type=float, default=15.0, help='Range-image upper vertical FOV in degrees')
    parser.add_argument('--fov-down', type=float, default=-25.0, help='Range-image lower vertical FOV in degrees')
    parser.add_argument('--bounds', type=float, nargs=4, metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'), default=(-50.0, 50.0, -50.0, 50.0), help='BEV bounds in point-cloud units')
    parser.add_argument('--intrinsics', type=Path, help='Camera 3x3 intrinsic matrix text file')
    parser.add_argument('--extrinsics', type=Path, help='World/LiDAR-to-camera 4x4 matrix text file')
    parser.add_argument('--kitti-calib-dir', type=Path, help='Directory containing KITTI calib_cam_to_cam.txt and calib_velo_to_cam.txt')
    parser.add_argument('--camera', type=int, choices=(0, 1, 2, 3), default=2, help='KITTI camera number (default: 2, left color camera)')
    parser.add_argument('--image', type=Path, help='Background camera image for KITTI overlay')
    parser.add_argument('--point-radius', type=int, default=2, help='Overlay point radius in pixels')
    parser.add_argument('--alpha', type=float, default=0.85, help='Point overlay opacity from 0 to 1')
    parser.add_argument('--percentiles', type=float, nargs=2, default=(1.0, 99.0), metavar=('LOW', 'HIGH'))
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        raise SystemExit('--width and --height must be positive')
    fields, source_meta = read_point_cloud(args.input)
    xyz, values = xyz_and_values(fields, args.color_by)
    background = None
    if args.mode == 'kitti-camera':
        if not args.kitti_calib_dir or not args.image:
            raise SystemExit('kitti-camera mode requires --kitti-calib-dir and --image')
        background = Image.open(args.image)
        args.width, args.height = background.size
    projection_meta = {'mode': args.mode, 'width': args.width, 'height': args.height, 'color_by': args.color_by}
    if args.mode == 'range':
        if args.fov_up <= args.fov_down:
            raise SystemExit('--fov-up must be greater than --fov-down')
        uv, depth, valid = project_range(xyz, args.width, args.height, args.fov_up, args.fov_down)
        projection_meta.update(fov_up=args.fov_up, fov_down=args.fov_down)
    elif args.mode == 'bev':
        xmin, xmax, ymin, ymax = args.bounds
        if xmin >= xmax or ymin >= ymax:
            raise SystemExit('Invalid --bounds: minima must be less than maxima')
        uv, depth, valid = project_bev(xyz, args.width, args.height, args.bounds)
        projection_meta['bounds'] = list(args.bounds)
    elif args.mode == 'camera':
        if not args.intrinsics or not args.extrinsics:
            raise SystemExit('camera mode requires --intrinsics and --extrinsics')
        intrinsics = load_matrix(args.intrinsics, (3, 3))
        extrinsics = load_matrix(args.extrinsics, (4, 4))
        uv, depth, valid = project_camera(xyz, args.width, args.height, intrinsics, extrinsics)
        projection_meta.update(intrinsics=intrinsics.tolist(), extrinsics=extrinsics.tolist())
    else:
        transform, projection, kitti_meta = read_kitti_calibration(args.kitti_calib_dir, args.camera)
        uv, depth, valid = project_kitti_camera(xyz, transform, projection)
        projection_meta.update(kitti_meta)
        projection_meta['background_image'] = str(args.image)
    value_image, depth_image, point_index = rasterize(uv, depth, values, valid, args.width, args.height)
    rgb, scale_meta = colored_image(value_image, *args.percentiles)
    if background is not None:
        if args.point_radius < 0:
            raise SystemExit('--point-radius cannot be negative')
        if not 0.0 <= args.alpha <= 1.0:
            raise SystemExit('--alpha must be between 0 and 1')
        rgb = overlay_points(background, rgb, point_index, args.point_radius, args.alpha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode='RGB').save(args.output)
    sidecar = args.output.with_suffix('.npz')
    np.savez_compressed(sidecar, point_index=point_index, depth=depth_image, value=value_image)
    metadata_path = args.output.with_suffix('.json')
    metadata_path.write_text(json.dumps({'source': str(args.input), 'point_cloud': source_meta, 'projection': projection_meta, 'color_scale': scale_meta, 'visible_points': int(np.count_nonzero(point_index >= 0)), 'sidecar': sidecar.name}, indent=2) + '\n')
    print(f'Wrote {args.output}')
    print(f'Wrote {sidecar} (point_index, depth, value)')
    print(f'Wrote {metadata_path}')
if __name__ == '__main__':
    main()
