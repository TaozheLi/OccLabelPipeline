"""Experimental dense occupancy generation; Occ3D-nuScenes taxonomy and layout.

No network, torch, MMCV, or nuscenes-devkit required. Reads nuScenes JSON tables.
Geometry is reconstructed supervision, not an exact reproduction of official GT.
"""
from occ_label_pipeline.context import CFG, CONFIG_PATH, PROJECT_ROOT, ENGINE_ROOT, OUTPUT as CAPTURE_ROOT, DATA_ROOT, TRACKER_ROOT, NFRAMES
import argparse
import json
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
CLASSES = ['others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation', 'free']
TAXONOMY = 'Occ3D-nuScenes'
RECONSTRUCTION_EVENTS = []
FINE_TO_COARSE = np.array([0, 0, 7, 7, 7, 0, 7, 0, 0, 1, 0, 0, 8, 0, 2, 3, 3, 4, 5, 0, 0, 6, 9, 10, 11, 12, 13, 14, 15, 0, 16, 0], dtype=np.uint8)
PALETTE = np.array([[80, 80, 80], [112, 128, 144], [220, 20, 60], [255, 127, 80], [255, 158, 0], [233, 150, 70], [255, 61, 99], [0, 0, 230], [47, 79, 79], [255, 140, 0], [255, 99, 71], [0, 207, 191], [175, 0, 75], [75, 0, 75], [166, 112, 64], [222, 184, 135], [0, 175, 0], [240, 240, 240]], dtype=np.uint8)

def transform(record):
    """nuScenes uses w,x,y,z quaternions; returns local-to-parent transform."""
    q = np.asarray(record['rotation'], dtype=float)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = (R, record['translation'])
    return T

def apply(points, T):
    return points @ T[:3, :3].T + T[:3, 3]

class Grid:

    def __init__(self, bounds=(-40, -40, -1, 40, 40, 5.4), voxel=0.4):
        self.bounds = np.asarray(bounds, dtype=float)
        self.lo, self.hi = (self.bounds[:3], self.bounds[3:])
        self.voxel = float(voxel)
        counts = (self.hi - self.lo) / voxel
        if voxel <= 0 or np.any(counts <= 0) or (not np.allclose(counts, np.round(counts))):
            raise ValueError('Bounds must contain a positive integral number of voxels')
        self.shape = tuple((int(n) for n in np.round(counts)))

    def index(self, points):
        return np.floor((points - self.lo) / self.voxel).astype(np.int64)

    def inside(self, points):
        return np.isfinite(points).all(axis=1) & (points >= self.lo).all(axis=1) & (points < self.hi).all(axis=1)

    def centers(self, indices):
        return self.lo + (np.asarray(indices) + 0.5) * self.voxel

def trace_rays(origin, endpoints, grid, occupied=None):
    """Exact voxel DDA, clipped to ROI. Endpoint voxels excluded from free paths.

    Returns visited free-path cells and per-ray visibility. If occupied is
    supplied, rays stop on the first blocking voxel before their endpoint.
    Boundary-only contacts (edges/corners) are not counted as voxel traversal.
    """
    seen = np.zeros(grid.shape, dtype=bool)
    visible = np.ones(len(endpoints), dtype=bool)
    for start in range(0, len(endpoints), 8192):
        ends = endpoints[start:start + 8192]
        origin = np.asarray(origin, dtype=float)
        d = ends - origin
        parallel = np.abs(d) < 1e-12
        inv = np.divide(1.0, d, out=np.full_like(d, np.inf), where=~parallel)
        with np.errstate(invalid='ignore'):
            a, b = ((grid.lo - origin) * inv, (grid.hi - origin) * inv)
        lower, upper = (np.minimum(a, b), np.maximum(a, b))
        lower = np.where(parallel, -np.inf, lower)
        upper = np.where(parallel, np.inf, upper)
        outside = (parallel & ((origin < grid.lo) | (origin >= grid.hi))).any(axis=1)
        t0 = np.maximum(0.0, lower.max(axis=1))
        t1 = np.minimum(1.0, upper.min(axis=1))
        active = ~outside & (t1 > t0) & np.isfinite(ends).all(axis=1)
        safe_t = np.where(active, t0 + 1e-10, 0.0)
        cell = grid.index(origin + safe_t[:, None] * d)
        end_cell = grid.index(np.where(np.isfinite(ends), ends, origin))
        end_inside = grid.inside(ends)
        step = np.sign(d).astype(int)
        boundary = grid.lo + (cell + (step > 0)) * grid.voxel
        with np.errstate(invalid='ignore'):
            next_t = (boundary - origin) * inv
        next_t = np.where(parallel, np.inf, next_t)
        delta = grid.voxel * np.abs(inv)
        clear = np.ones(len(ends), dtype=bool)
        for _ in range(sum(grid.shape) + 3):
            active &= (cell >= 0).all(axis=1) & (cell < grid.shape).all(axis=1)
            at_end = end_inside & (cell == end_cell).all(axis=1)
            active &= ~at_end
            ids = np.flatnonzero(active)
            if not len(ids):
                break
            if occupied is not None:
                blocked = occupied[tuple(cell[ids].T)]
                clear[ids[blocked]] = False
                active[ids[blocked]] = False
                ids = ids[~blocked]
            seen[tuple(cell[ids].T)] = True
            crossing = next_t.min(axis=1)
            active &= crossing < t1 - 1e-12
            axes = np.isclose(next_t, crossing[:, None], atol=1e-12, rtol=0) & active[:, None]
            cell += axes * step
            next_t = np.where(axes, next_t + delta, next_t)
        visible[start:start + len(ends)] = clear & ~outside & (t1 >= t0)
    return (seen, visible)

def voxel_labels(points, labels, grid):
    keep = grid.inside(points)
    ijk, labels = (grid.index(points[keep]), labels[keep])
    result = np.full(grid.shape, 17, dtype=np.uint8)
    if not len(ijk):
        return result
    flat = np.ravel_multi_index(ijk.T, grid.shape)
    keys, count = np.unique(flat * 18 + labels, return_counts=True)
    vox, cls = (keys // 18, keys % 18)
    order = np.lexsort((cls, -count, vox))
    ordered = vox[order]
    first = np.r_[True, ordered[1:] != ordered[:-1]]
    winners = order[first]
    result.ravel()[vox[winners]] = cls[winners]
    return result

class NuScenesTables:

    def __init__(self, root, version):
        self.root = Path(root)
        self.tables = {}
        for name in ('scene', 'sample', 'sample_data', 'sample_annotation', 'instance', 'category', 'sensor', 'calibrated_sensor', 'ego_pose', 'lidarseg'):
            path = self.root / version / (name + '.json')
            if not path.exists():
                raise FileNotFoundError(f'Missing {path}. Extract mini AND lidarseg-mini into the same root.')
            self.tables[name] = {r['token']: r for r in json.loads(path.read_text())}
        self.by_sample = {}
        for sd in self.tables['sample_data'].values():
            if sd['is_key_frame']:
                cs = self.get('calibrated_sensor', sd['calibrated_sensor_token'])
                sensor = self.get('sensor', cs['sensor_token'])
                self.by_sample.setdefault(sd['sample_token'], {})[sensor['channel']] = sd
        self.annotations = {}
        for ann in self.tables['sample_annotation'].values():
            self.annotations.setdefault(ann['sample_token'], []).append(ann)

    def get(self, table, token):
        return self.tables[table][token]

    def sensor_world(self, sd):
        return transform(self.get('ego_pose', sd['ego_pose_token'])) @ transform(self.get('calibrated_sensor', sd['calibrated_sensor_token']))

    def frames(self, scene):
        token, frames = (scene['first_sample_token'], [])
        while token:
            sample = self.get('sample', token)
            frames.append(sample)
            token = sample['next']
        return frames

    def boxes(self, sample):
        result = {}
        for ann in self.annotations.get(sample['token'], []):
            instance = self.get('instance', ann['instance_token'])
            name = self.get('category', instance['category_token'])['name']
            if name.startswith(('vehicle.', 'human.', 'animal')):
                result[ann['instance_token']] = ann
        return result

    def read_frame(self, sample):
        sd = self.by_sample[sample['token']]['LIDAR_TOP']
        path = self.root / sd['filename']
        if path.stat().st_size % 20:
            raise ValueError(f'Invalid nuScenes five-float point file: {path}')
        points = np.fromfile(path, dtype='<f4').reshape(-1, 5)[:, :3].astype(float)
        label_record = self.get('lidarseg', sd['token'])
        fine = np.fromfile(self.root / label_record['filename'], dtype=np.uint8)
        if len(points) != len(fine) or np.any(fine >= 32):
            raise ValueError(f"Invalid/misaligned lidarseg labels for {sd['token']}")
        valid = np.isfinite(points).all(axis=1) & (fine != 0) & (fine != 31)
        points, fine = (points[valid], fine[valid])
        world = apply(points, self.sensor_world(sd))
        boxes = self.boxes(sample)
        owner = np.full(len(points), '', dtype=object)
        for token, ann in boxes.items():
            local = apply(world, np.linalg.inv(transform(ann)))
            half = np.asarray(ann['size'])[[1, 0, 2]] / 2
            inside = (np.abs(local) <= half).all(axis=1)
            owner[inside] = token
        movable = np.isin(fine, list(range(1, 9)) + list(range(14, 24)))
        return dict(world=world, labels=FINE_TO_COARSE[fine], owner=owner, movable=movable, boxes=boxes, sd=sd)

def align_frame(source, target, world_to_ego, is_target=False):
    world = source['world'].copy()
    keep = np.ones(len(world), bool) if is_target else ~source['movable']
    for token, box in source['boxes'].items():
        mask = source['owner'] == token
        keep[mask] = False
        if token in target['boxes']:
            motion = transform(target['boxes'][token]) @ np.linalg.inv(transform(box))
            world[mask] = apply(world[mask], motion)
            keep[mask] = True
    return (apply(world[keep], world_to_ego), source['labels'][keep])

def densify(points, labels, voxel, depth=8, max_distance=0.8):
    """Contain native Open3D aborts in a worker; preserve measured points on failure."""
    if len(points) < 30:
        return (points, labels)
    event = dict(input_points=len(points), depth=depth, status='fallback')
    with tempfile.TemporaryDirectory(prefix='occ-poisson-') as directory:
        source, destination = (Path(directory) / 'input.npz', Path(directory) / 'output.npz')
        np.savez(source, points=points, labels=labels)
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--poisson-worker', str(source), str(destination), str(voxel), str(depth), str(max_distance)]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=600)
            event['returncode'] = result.returncode
            if result.returncode != 0:
                event['reason'] = (result.stderr or result.stdout)[-4000:]
            elif not destination.exists():
                event['reason'] = 'Worker did not produce an output file'
            else:
                with np.load(destination, allow_pickle=False) as output:
                    p, l = (output['points'], output['labels'])
                if p.ndim != 2 or p.shape[1] != 3 or len(p) != len(l) or (not np.isfinite(p).all()):
                    raise ValueError('Invalid reconstruction output')
                event.update(status='completed', output_points=len(p))
                RECONSTRUCTION_EVENTS.append(event)
                return (p, l)
        except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
            event['reason'] = str(exc)
    RECONSTRUCTION_EVENTS.append(event)
    print(f"Poisson worker failed (exit {event.get('returncode', 'timeout/error')}); keeping {len(points)} accumulated points for this component. Details will be saved in metadata.json.", flush=True)
    return (points, labels)

def densify_native(points, labels, voxel, depth=8, max_distance=0.8):
    """Poisson mesh vertices with nearest-point semantics and support trimming.

    Called separately for static background and each tracked actor to avoid
    surfaces connecting cars to the road. Measured points are always retained.
    """
    if len(points) < 30:
        return (points, labels)
    import open3d as o3d
    from scipy.spatial import cKDTree
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pc = pc.voxel_down_sample(voxel / 2)
    if len(pc.points) < 30:
        return (points, labels)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4, max_nn=30))
    try:
        pc.orient_normals_consistent_tangent_plane(10)
        mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=depth, n_threads=1)
    except RuntimeError as exc:
        print(f'Poisson skipped for degenerate component: {exc}', flush=True)
        return (points, labels)
    if not len(mesh.vertices) or not len(mesh.triangles):
        return (points, labels)
    mesh.remove_vertices_by_mask(np.asarray(density) < np.quantile(density, 0.05))
    if not len(mesh.triangles):
        return (points, labels)
    count = min(1000000, max(len(mesh.vertices), int(mesh.get_surface_area() / (voxel / 3) ** 2)))
    o3d.utility.random.seed(0)
    vertices = np.asarray(mesh.sample_points_uniformly(number_of_points=max(count, 1)).points)
    distance, nearest = cKDTree(points).query(vertices)
    good = distance <= max_distance
    return (np.concatenate([points, vertices[good]]), np.concatenate([labels, labels[nearest[good]]]))

def camera_visibility(db, sample, ego_to_world, semantics, observed, grid):
    """Voxel-center visibility using calibrated camera frusta + voxel ray occlusion."""
    indices = np.argwhere(observed)
    centers = grid.centers(indices)
    occupied = semantics != 17
    camera_mask = np.zeros(grid.shape, bool)
    for channel, sd in db.by_sample[sample['token']].items():
        if not channel.startswith('CAM_'):
            continue
        camera_to_ego = np.linalg.inv(ego_to_world) @ db.sensor_world(sd)
        cam = apply(centers, np.linalg.inv(camera_to_ego))
        K = np.asarray(db.get('calibrated_sensor', sd['calibrated_sensor_token'])['camera_intrinsic'])
        proj = cam @ K.T
        uv = proj[:, :2] / np.maximum(cam[:, 2:3], 1e-09)
        valid = (cam[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < sd['width']) & (uv[:, 1] >= 0) & (uv[:, 1] < sd['height'])
        candidate = np.flatnonzero(valid)
        _, clear = trace_rays(camera_to_ego[:3, 3], centers[candidate], grid, occupied)
        camera_mask[tuple(indices[candidate[clear]].T)] = True
    return camera_mask

def write_ply(path, semantics, grid):
    ijk = np.argwhere(semantics != 17)
    labels = semantics[tuple(ijk.T)]
    xyz = grid.centers(ijk)
    dtype = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('semantic_label', 'u1')]
    data = np.empty(len(xyz), dtype=dtype)
    for i, name in enumerate(('x', 'y', 'z')):
        data[name] = xyz[:, i]
    for i, name in enumerate(('red', 'green', 'blue')):
        data[name] = PALETTE[labels, i]
    data['semantic_label'] = labels
    header = f'ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n'
    header += ''.join((f"property {('float' if name in ('x', 'y', 'z') else 'uchar')} {name}\n" for name, _ in dtype))
    with path.open('wb') as f:
        f.write((header + 'end_header\n').encode())
        f.write(data.tobytes())

def generate(db, frames, target_index, args, grid):
    RECONSTRUCTION_EVENTS.clear()
    target = db.read_frame(frames[target_index])
    ego_to_world = transform(db.get('ego_pose', target['sd']['ego_pose_token']))
    world_to_ego = np.linalg.inv(ego_to_world)
    groups = {}
    for index in range(max(0, target_index - args.window), min(len(frames), target_index + args.window + 1)):
        source = target if index == target_index else db.read_frame(frames[index])
        for token in [''] + list(source['boxes']):
            mask = source['owner'] == token
            if token == '':
                mask &= ~source['movable']
            if not mask.any():
                continue
            sub = dict(source, world=source['world'][mask], labels=source['labels'][mask], owner=source['owner'][mask], movable=source['movable'][mask])
            p, l = align_frame(sub, target, world_to_ego, index == target_index)
            near = ((p >= grid.lo - args.surface_support) & (p < grid.hi + args.surface_support)).all(axis=1)
            groups.setdefault(token, []).append((p[near], l[near]))
    points, labels = ([], [])
    for parts in groups.values():
        p = np.concatenate([pair[0] for pair in parts])
        l = np.concatenate([pair[1] for pair in parts])
        if args.surface == 'poisson':
            p, l = densify(p, l, grid.voxel, args.poisson_depth, args.surface_support)
        points.append(p)
        labels.append(l)
    current = apply(target['world'], world_to_ego)
    points.append(current)
    labels.append(target['labels'])
    semantics = voxel_labels(np.concatenate(points), np.concatenate(labels), grid)
    measured = voxel_labels(current, target['labels'], grid)
    semantics[measured != 17] = measured[measured != 17]
    origin = (world_to_ego @ db.sensor_world(target['sd']))[:3, 3]
    free, _ = trace_rays(origin, current, grid)
    semantics[free & (measured == 17)] = 17
    observed = free | (semantics != 17)
    cameras = camera_visibility(db, frames[target_index], ego_to_world, semantics, observed, grid)
    return (semantics, observed, cameras)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataroot', type=Path, required=True)
    parser.add_argument('--version', default='v1.0-mini')
    parser.add_argument('--scene', default='scene-0061', help='Scene name or all')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=1, help='Target frames per scene; 0 means all')
    parser.add_argument('--window', type=int, default=5, help='Labeled keyframes before/after each target')
    parser.add_argument('--surface', choices=['poisson', 'none'], default='poisson')
    parser.add_argument('--poisson-depth', type=int, default=8)
    parser.add_argument('--surface-support', type=float, default=0.8, help='Maximum mesh distance from input points, meters')
    parser.add_argument('--output', type=Path, default=Path('occ-mini'))
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if min(args.start, args.max_frames, args.window) < 0 or args.surface_support <= 0:
        parser.error('Frame indices/window must be nonnegative; surface support must be positive')
    if not 4 <= args.poisson_depth <= 11:
        parser.error('--poisson-depth must be between 4 and 11')
    if args.surface == 'poisson':
        if any((importlib.util.find_spec(name) is None for name in ('open3d', 'scipy'))):
            parser.error('Poisson needs open3d and scipy; install requirements.txt or use --surface none')
    db = NuScenesTables(args.dataroot, args.version)
    scenes = [s for s in db.tables['scene'].values() if args.scene in ('all', s['name'])]
    if not scenes:
        parser.error(f'Scene {args.scene} not found')
    grid = Grid()
    for scene in scenes:
        frames = db.frames(scene)
        if args.start >= len(frames):
            raise ValueError(f"--start exceeds frame count of {scene['name']}")
        stop = min(len(frames), args.start + args.max_frames) if args.max_frames else len(frames)
        for i in range(args.start, stop):
            directory = args.output / scene['name'] / frames[i]['token']
            if (directory / 'labels.npz').exists() and (not args.overwrite):
                raise FileExistsError(f'{directory}/labels.npz exists; select new output or --overwrite')
            print(f"Generating {scene['name']} frame {i}/{len(frames) - 1}", flush=True)
            semantic, lidar, camera = generate(db, frames, i, args, grid)
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(directory / 'labels.npz', semantics=semantic, mask_lidar=lidar, mask_camera=camera)
            write_ply(directory / 'occupied.ply', semantic, grid)
            metadata = dict(taxonomy=TAXONOMY, classes=dict(enumerate(CLASSES)), class_to_id={name: i for i, name in enumerate(CLASSES)}, free_class_id=17, unknown_representation='mask_lidar == false', coordinate_frame='target LIDAR timestamp ego', bounds=grid.bounds.tolist(), voxel_size=grid.voxel, shape=grid.shape, sample_token=frames[i]['token'], scene=scene['name'], version=args.version, surface=args.surface, window=args.window, poisson_depth=args.poisson_depth, surface_support=args.surface_support, source=str(args.dataroot.resolve()), reconstruction_events=list(RECONSTRUCTION_EVENTS), observed_voxels=int(lidar.sum()), occupied_voxels=int((semantic != 17).sum()), note='Experimental Occ3D-schema labels; not official Occ3D reproduction. Free rays use current scan; occupancy is fused/reconstructed. Camera mask uses voxel-center frustum and voxel occlusion.')
            (directory / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
            print(f'Wrote {directory}/labels.npz', flush=True)
if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--poisson-worker':
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except (ImportError, ValueError, OSError):
            pass
        source, destination, voxel, depth, support = sys.argv[2:]
        with np.load(source, allow_pickle=False) as data:
            p, l = densify_native(data['points'], data['labels'], float(voxel), int(depth), float(support))
        np.savez(destination, points=p, labels=l)
    else:
        main()
