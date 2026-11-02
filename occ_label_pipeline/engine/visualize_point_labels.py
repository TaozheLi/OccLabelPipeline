"""Visualize point_labels.npz semantic pseudo-labels with Open3D."""
from occ_label_pipeline.context import CFG, CONFIG_PATH, PROJECT_ROOT, ENGINE_ROOT, OUTPUT as CAPTURE_ROOT, DATA_ROOT, TRACKER_ROOT, NFRAMES
import argparse
from pathlib import Path
import numpy as np
CLASSES = ['others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation', 'free']
PALETTE = np.array([[80, 80, 80], [112, 128, 144], [220, 20, 60], [255, 127, 80], [255, 158, 0], [233, 150, 70], [255, 61, 99], [0, 0, 230], [47, 79, 79], [255, 140, 0], [255, 99, 71], [0, 207, 191], [175, 0, 75], [75, 0, 75], [166, 112, 64], [222, 184, 135], [0, 175, 0], [240, 240, 240]], dtype=float) / 255.0
UNLABELED_COLOR = np.array([0.18, 0.18, 0.18])

def parse_classes(values):
    if values is None:
        return None
    result = []
    for value in values:
        normalized = value.strip().lower().replace('-', '_')
        if normalized.lstrip('-').isdigit():
            class_id = int(normalized)
        elif normalized in CLASSES:
            class_id = CLASSES.index(normalized)
        else:
            raise ValueError(f'Unknown class {value!r}')
        if not 0 <= class_id <= 17:
            raise ValueError(f'Class ID must be 0–17, got {class_id}')
        result.append(class_id)
    return sorted(set(result))

def load_points(path, classes=None, min_confidence=0.0, show_unlabeled=False, max_points=None, seed=0):
    """Load, validate, filter, and optionally sample a point-label NPZ."""
    with np.load(path, allow_pickle=False) as data:
        required = {'xyz', 'semantics'}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"NPZ is missing: {', '.join(sorted(missing))}")
        xyz = np.asarray(data['xyz'], dtype=np.float64)
        labels = np.asarray(data['semantics'])
        confidence = np.asarray(data['confidence'], dtype=np.float64) if 'confidence' in data else np.ones(len(labels))
        observed = np.asarray(data['observed']).astype(bool) if 'observed' in data else labels >= 0
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError('xyz must have shape [N,3]')
    if labels.ndim != 1 or confidence.ndim != 1 or observed.ndim != 1:
        raise ValueError('semantics, confidence, and observed must be 1D')
    if not len(xyz) == len(labels) == len(confidence) == len(observed):
        raise ValueError('All arrays must have the same point count')
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('semantics must contain integer IDs')
    if not np.isfinite(xyz).all() or not np.isfinite(confidence).all():
        raise ValueError('xyz/confidence contains NaN or infinity')
    if np.any((labels < -1) | (labels > 17)):
        raise ValueError('semantics IDs must be -1 or 0–17')
    if np.any((confidence < 0) | (confidence > 1)):
        raise ValueError('confidence must be in [0,1]')
    keep = confidence >= min_confidence
    if not show_unlabeled:
        keep &= observed & (labels >= 0)
    if classes is not None:
        keep &= np.isin(labels, classes)
    ids = np.flatnonzero(keep)
    if max_points is not None and len(ids) > max_points:
        ids = np.random.default_rng(seed).choice(ids, max_points, replace=False)
        ids.sort()
    selected_labels = labels[ids].astype(np.int16)
    colors = np.tile(UNLABELED_COLOR, (len(ids), 1))
    labeled = selected_labels >= 0
    colors[labeled] = PALETTE[selected_labels[labeled]]
    return (xyz[ids], selected_labels, confidence[ids], colors)

def show_open3d(xyz, colors, point_size, window_title='Occ3D point pseudo-labels'):
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError('Install Open3D: python -m pip install open3d') from exc
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
    viewer = o3d.visualization.Visualizer()
    if not viewer.create_window(window_name=window_title, width=1280, height=800):
        raise RuntimeError('Could not create an Open3D window; use a graphical desktop session')
    try:
        viewer.add_geometry(cloud)
        viewer.add_geometry(frame, reset_bounding_box=False)
        options = viewer.get_render_option()
        options.background_color = np.array([0.06, 0.06, 0.08])
        options.point_size = point_size
        control = viewer.get_view_control()
        control.set_front([-1, 0, -0.35])
        control.set_up([0, 0, 1])
        control.set_zoom(0.45)
        print('Mouse drag: rotate; wheel: zoom; H: help; Q/Esc: close.', flush=True)
        print('Axes are red X, green Y, blue Z.', flush=True)
        viewer.run()
    finally:
        viewer.destroy_window()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('npz', type=Path, help='Path to point_labels.npz')
    parser.add_argument('--classes', nargs='+', help='Occ3D names or IDs to display')
    parser.add_argument('--min-confidence', type=float, default=0.0)
    parser.add_argument('--show-unlabeled', action='store_true', help='Show ID -1 points in dark gray')
    parser.add_argument('--window-title', default='Occ3D point pseudo-labels')
    parser.add_argument('--point-size', type=float, default=2.0)
    parser.add_argument('--max-points', type=int, help='Randomly downsample the displayed points')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--inspect', action='store_true', help='Print the selected class counts without opening Open3D')
    args = parser.parse_args()
    if not 0 <= args.min_confidence <= 1:
        parser.error('--min-confidence must be between 0 and 1')
    if not np.isfinite(args.point_size) or args.point_size <= 0:
        parser.error('--point-size must be positive')
    if args.max_points is not None and args.max_points <= 0:
        parser.error('--max-points must be positive')
    try:
        xyz, labels, confidence, colors = load_points(args.npz, parse_classes(args.classes), args.min_confidence, args.show_unlabeled, args.max_points, args.seed)
        with np.load(args.npz, allow_pickle=False) as source_data:
            has_confidence = 'confidence' in source_data
        print(f'{len(xyz):,} points selected')
        if not has_confidence:
            print('No model confidence stored; colors show semantic labels only.')
        for label, count in zip(*np.unique(labels, return_counts=True)):
            name = 'unlabeled' if label == -1 else CLASSES[label]
            mean = confidence[labels == label].mean()
            suffix = f'  mean confidence {mean:.3f}' if has_confidence else ''
            print(f'  {label:2d} {name:24s} {count:8,d}' + suffix)
        if not len(xyz):
            print('No points match the filters; no window opened.')
        elif not args.inspect:
            show_open3d(xyz, colors, args.point_size, args.window_title)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f'Error: {exc}\n')
if __name__ == '__main__':
    main()
