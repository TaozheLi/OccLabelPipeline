"""View Occ3D-schema labels with Open3D. No surface reconstruction is run."""
from occ_label_pipeline.context import CFG, CONFIG_PATH, PROJECT_ROOT, ENGINE_ROOT, OUTPUT as CAPTURE_ROOT, DATA_ROOT, TRACKER_ROOT, NFRAMES
import argparse
import json
from pathlib import Path
import numpy as np
CLASSES = ['others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation', 'free']
PALETTE = np.array([[80, 80, 80], [112, 128, 144], [220, 20, 60], [255, 127, 80], [255, 158, 0], [233, 150, 70], [255, 61, 99], [0, 0, 230], [47, 79, 79], [255, 140, 0], [255, 99, 71], [0, 207, 191], [175, 0, 75], [75, 0, 75], [166, 112, 64], [222, 184, 135], [0, 175, 0], [240, 240, 240]], dtype=float) / 255.0

def class_ids(values):
    result = []
    for value in values:
        if value in CLASSES:
            result.append(CLASSES.index(value))
        elif value.isdigit() and 0 <= int(value) <= 17:
            result.append(int(value))
        else:
            raise ValueError(f'Unknown class {value!r}. Choose from {CLASSES} or IDs 0–17.')
    return result

def load_labels(path, metadata=None, bounds=None, voxel_size=None, mask='lidar', classes=None):
    """Return occupied centers, labels, bounds, voxel size, and class names.

    Axes are NPZ [x,y,z]. Unknown space is not inferred from class 0.
    --mask none intentionally bypasses visibility; class 17 is still excluded.
    """
    path = Path(path)
    meta_path = Path(metadata) if metadata else path.with_name('metadata.json')
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if metadata and (not meta_path.exists()):
        raise ValueError(f'Metadata not found: {meta_path}')
    if bounds is None:
        bounds = meta.get('bounds')
    if voxel_size is None:
        voxel_size = meta.get('voxel_size')
    if bounds is None or voxel_size is None:
        raise ValueError('Provide sibling metadata.json or both --bounds and --voxel-size.')
    bounds = np.asarray(bounds, dtype=float)
    voxel_size = float(voxel_size)
    if bounds.shape != (6,) or not np.isfinite(bounds).all() or (not np.isfinite(voxel_size)) or (voxel_size <= 0) or np.any(bounds[3:] <= bounds[:3]):
        raise ValueError('Bounds must be finite increasing XYZ limits; voxel size must be positive.')
    if meta.get('free_class_id', 17) != 17:
        raise ValueError('Expected Occ3D-nuScenes free class ID 17.')
    with np.load(path, allow_pickle=False) as data:
        if 'semantics' not in data:
            raise ValueError('NPZ must contain semantics.')
        sem = data['semantics']
        if sem.ndim != 3 or not np.issubdtype(sem.dtype, np.integer):
            raise ValueError('semantics must be a 3D integer array with axis order [x,y,z].')
        if not np.allclose((bounds[3:] - bounds[:3]) / voxel_size, sem.shape):
            raise ValueError(f'Grid bounds/resolution do not match semantics shape {sem.shape}.')
        if np.any((sem < 0) | (sem > 17)):
            raise ValueError('Expected semantic IDs 0–17; unexpected IDs found.')
        keep = sem != 17
        if mask != 'none':
            key = 'mask_' + mask
            if key not in data:
                raise ValueError(f'{key} missing. Use --mask none only to bypass visibility explicitly.')
            visibility = data[key]
            if visibility.shape != sem.shape or not np.isin(visibility, [0, 1]).all():
                raise ValueError(f'{key} must be a binary array matching semantics.')
            keep &= visibility.astype(bool)
        if classes is not None:
            keep &= np.isin(sem, classes)
        indices = np.argwhere(keep)
        labels = sem[tuple(indices.T)]
    centers = bounds[:3] + (indices + 0.5) * voxel_size
    names = [meta.get('classes', {}).get(str(i), name) for i, name in enumerate(CLASSES)]
    return (centers, labels, bounds, voxel_size, names)

def show(centers, labels, bounds, voxel_size, mode, point_size):
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError('Install Open3D in this environment: python -m pip install open3d') from exc
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(centers)
    cloud.colors = o3d.utility.Vector3dVector(PALETTE[labels])
    geometry = cloud
    if mode == 'voxels':
        geometry = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(cloud, voxel_size, bounds[:3], bounds[3:])
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
    box = o3d.geometry.AxisAlignedBoundingBox(bounds[:3], bounds[3:])
    box.color = (0.45, 0.45, 0.45)
    viewer = o3d.visualization.Visualizer()
    if not viewer.create_window(window_name='Occ3D occupancy — ' + mode, width=1280, height=800):
        raise RuntimeError('Open3D could not open a window. Run locally in a graphical desktop session.')
    try:
        for item in (geometry, frame, box):
            if not viewer.add_geometry(item):
                raise RuntimeError('Open3D could not add geometry to the viewer.')
        render = viewer.get_render_option()
        render.background_color = np.array([0.08, 0.08, 0.1])
        render.point_size = point_size
        view = viewer.get_view_control()
        view.set_lookat((bounds[:3] + bounds[3:]) / 2)
        view.set_front([0.7, -0.7, 0.65])
        view.set_up([0, 0, 1])
        view.set_zoom(0.75)
        print('Mouse drag: rotate; wheel: zoom; H: Open3D help; Q/Esc: close.', flush=True)
        print('Coordinate axes: red X, green Y, blue Z (meters).', flush=True)
        viewer.run()
    finally:
        viewer.destroy_window()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('npz', type=Path, help='Path to labels.npz')
    parser.add_argument('--metadata', type=Path, help='Defaults to metadata.json next to NPZ')
    parser.add_argument('--bounds', nargs=6, type=float, metavar=('XMIN', 'YMIN', 'ZMIN', 'XMAX', 'YMAX', 'ZMAX'))
    parser.add_argument('--voxel-size', type=float, help='Override metadata resolution in meters')
    parser.add_argument('--mask', choices=['lidar', 'camera', 'none'], default='lidar')
    parser.add_argument('--classes', nargs='+', help='Only these class names or numeric IDs')
    parser.add_argument('--mode', choices=['voxels', 'points'], default='voxels')
    parser.add_argument('--point-size', type=float, default=3, help='Point mode display size')
    parser.add_argument('--inspect', action='store_true', help='Print counts without importing Open3D or opening a window')
    args = parser.parse_args()
    try:
        if not np.isfinite(args.point_size) or args.point_size <= 0:
            raise ValueError('--point-size must be positive and finite.')
        centers, labels, bounds, voxel_size, names = load_labels(args.npz, args.metadata, args.bounds, args.voxel_size, args.mask, class_ids(args.classes) if args.classes else None)
        print(f'{len(centers):,} occupied voxels; resolution {voxel_size:g} m; visibility: {args.mask}')
        for label, count in zip(*np.unique(labels, return_counts=True)):
            rgb = np.rint(PALETTE[label] * 255).astype(int).tolist()
            print(f'  {label:2d} {names[label]:24s} {count:8,d}   RGB {rgb}')
        if not len(centers):
            print('No occupied voxels match the selection; no window opened.')
        elif not args.inspect:
            show(centers, labels, bounds, voxel_size, args.mode, args.point_size)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f'Error: {exc}\n')
if __name__ == '__main__':
    main()
