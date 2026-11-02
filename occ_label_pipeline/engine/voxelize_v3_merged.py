"""Project reviewed world-static surfaces, merge tracked objects, voxelize all frames."""
from occ_label_pipeline.context import CFG, CONFIG_PATH, PROJECT_ROOT, ENGINE_ROOT, OUTPUT as CAPTURE_ROOT, DATA_ROOT, TRACKER_ROOT, NFRAMES
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import json
import argparse
import time
import hashlib
import traceback
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage
from numba import njit
from generate_occ import Grid, apply, voxel_labels, CLASSES
from ego_mask import inside_body, mask_arrays
ROOT = PROJECT_ROOT
BASE = ROOT / str(CAPTURE_ROOT / 'occ_pipeline_v3_0')
OUT = BASE / 'occ_static_dynamic_voxelized_v3'
WORLD = BASE / 'world_ground_completed_v3/completed_world.npz'
TRACK = BASE / 'tracked_completion_all_block100/tracking'
PRED = ROOT / str(CAPTURE_ROOT / 'predictions/frames')
BPA = BASE / 'ball_pivoting_all_block100/blocks'
FUSION = BASE / 'tracked_completion_all_block100/blocks'
DETS = ROOT / str(CAPTURE_ROOT / 'training_labels_v1/detections')
STATIC = [0, 1, 8, 11, 12, 13, 14, 15, 16]
DYNAMIC = [2, 3, 4, 5, 6, 7, 9, 10]
CLASS_IDS = dict(zip(['bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian', 'trailer', 'truck'], DYNAMIC))
GRID = Grid(CFG['bounds'], CFG['voxel_size_m'])
EGO = dict(enabled=True, bounds_grid_m=CFG['ego_bounds_grid_m'])
SHIFT = np.eye(4)
SHIFT[2, 3] = float(CFG['lidar_height_m'])
XYZ = LABELS = MANUAL = ADDED_GROUND = UNKNOWN_KNN = TREE = CAMERAS = None
DYNAMIC_LAST = False
INTERIOR_ROOT = None

def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)

@njit(cache=True)
def free_rays(origin, ends, lo, hi, voxel, shape):
    """Measured rays only; exclude each return endpoint from free evidence."""
    seen = np.zeros(shape, np.bool_)
    for r in range(len(ends)):
        direction = ends[r] - origin
        begin = 0.0
        finish = 1.0
        outside = False
        for axis in range(3):
            if abs(direction[axis]) < 1e-12:
                if origin[axis] < lo[axis] or origin[axis] >= hi[axis]:
                    outside = True
            else:
                t0 = (lo[axis] - origin[axis]) / direction[axis]
                t1 = (hi[axis] - origin[axis]) / direction[axis]
                begin = max(begin, min(t0, t1))
                finish = min(finish, max(t0, t1))
        if outside or finish <= begin:
            continue
        cell = np.floor((origin + (begin + 1e-10) * direction - lo) / voxel).astype(np.int64)
        target = np.floor((ends[r] - lo) / voxel).astype(np.int64)
        target_inside = ((ends[r] >= lo) & (ends[r] < hi)).all()
        step = np.zeros(3, np.int64)
        cross = np.full(3, np.inf)
        delta = np.full(3, np.inf)
        for axis in range(3):
            if abs(direction[axis]) >= 1e-12:
                step[axis] = 1 if direction[axis] > 0 else -1
                edge = lo[axis] + (cell[axis] + (1 if step[axis] > 0 else 0)) * voxel
                cross[axis] = (edge - origin[axis]) / direction[axis]
                delta[axis] = voxel / abs(direction[axis])
        for _ in range(shape[0] + shape[1] + shape[2] + 3):
            if (cell < 0).any() or cell[0] >= shape[0] or cell[1] >= shape[1] or (cell[2] >= shape[2]):
                break
            if target_inside and (cell == target).all():
                break
            seen[cell[0], cell[1], cell[2]] = True
            crossing = cross.min()
            if crossing >= finish - 1e-12:
                break
            for axis in range(3):
                if abs(cross[axis] - crossing) <= 1e-12:
                    cell[axis] += step[axis]
                    cross[axis] += delta[axis]
    return seen

def presence(points):
    result = np.zeros(GRID.shape, bool)
    indices = GRID.index(points[GRID.inside(points)])
    if len(indices):
        result[tuple(indices.T)] = True
    return result

def raw_dynamic(xyz, semantics, confidence, boxes):
    """Detector-supported original points; do not fill all voxels inside a box."""
    labels = np.full(len(xyz), -1, np.int16)
    identities = np.full(len(xyz), -1, np.int32)
    ground = np.isin(semantics, [11, 12, 13, 14]) & (confidence >= 0.45)
    for box in sorted(boxes, key=lambda b: b['confidence']):
        if box['model_class'] not in CLASS_IDS or box['confidence'] < 0.45:
            continue
        yaw = box['yaw_rad']
        c, s = (np.cos(yaw), np.sin(yaw))
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        inside = (np.abs((xyz - box['center']) @ rotation) <= np.asarray(box['dimensions']) / 2).all(1)
        inside &= ~ground
        labels[inside] = CLASS_IDS[box['model_class']]
        identities[inside] = box['track_id']
    return (labels, identities)

def merge_semantics(static_sem, manual_sem, dynamic_sem, dynamic_last=False):
    manual_mask = manual_sem != 17
    dynamic_mask = dynamic_sem != 17
    removed = np.zeros(static_sem.shape, bool)
    if dynamic_last:
        background = static_sem.copy()
        background[manual_mask] = manual_sem[manual_mask]
        assert np.isin(background, STATIC + [17]).all(), 'Dynamic class leaked into static processing'
        components, _ = ndimage.label(np.isin(background, STATIC), structure=np.ones((3, 3, 3), np.uint8))
        sizes = np.bincount(components.ravel())
        small = sizes < 5
        small[0] = False
        removed = small[components]
        background[removed] = 17
        sem = background.copy()
        sem[dynamic_mask] = dynamic_sem[dynamic_mask]
        manual_mask &= ~removed & ~dynamic_mask
        removed &= ~dynamic_mask
        assert np.array_equal(sem[dynamic_mask], dynamic_sem[dynamic_mask])
        return (sem, manual_mask, removed, background)
    sem = static_sem.copy()
    sem[dynamic_mask] = dynamic_sem[dynamic_mask]
    sem[manual_mask] = manual_sem[manual_mask]
    return (sem, manual_mask, removed, static_sem)

def process(name):
    dest = OUT / name
    if (dest / 'metadata.json').exists() and (dest / 'labels.npz').exists():
        return json.loads((dest / 'metadata.json').read_text())['counts']
    index = int(name) - 1
    block_start = index // 100 * 100 + 1
    block = f'{block_start:06d}_{min(block_start + 99, NFRAMES):06d}'
    pose = json.loads((DETS / (name + '.json')).read_text())
    world_from_lidar = np.asarray(pose['T_world_from_lidar'])
    grid_from_world = SHIFT @ np.linalg.inv(world_from_lidar)
    candidate_ids = np.asarray(TREE.query_ball_point(world_from_lidar[:3, 3], 57.0), np.int64)
    static = apply(XYZ[candidate_ids], grid_from_world)
    keep = GRID.inside(static) & ~inside_body(static, EGO)
    static = static[keep]
    selected = candidate_ids[keep]
    static_labels = LABELS[selected]
    static_sem = voxel_labels(static, static_labels, GRID)
    manual_sem = voxel_labels(static[MANUAL[selected]], static_labels[MANUAL[selected]], GRID)
    manual_mask = manual_sem != 17
    ground_added = presence(static[ADDED_GROUND[selected]])
    unknown_knn = presence(static[UNKNOWN_KNN[selected]])
    with np.load(PRED / name / 'point_labels.npz') as data:
        raw = data['xyz']
        raw_sem = data['semantics']
        raw_conf = data['confidence']
    measured_grid = apply(raw, SHIFT)
    valid_raw = ~inside_body(measured_grid, EGO)
    measured_grid = measured_grid[valid_raw]
    boxes = json.loads((TRACK / (name + '.json')).read_text())['boxes']
    if DYNAMIC_LAST:
        boxes = boxes + [dict(b, track_id=-2 - index) for index, b in enumerate(pose['detections']) if b['model_class'] == 'construction_vehicle' and b['confidence'] >= 0.45]
    labels, ids = raw_dynamic(raw, raw_sem, raw_conf, boxes)
    valid = (labels >= 0) & valid_raw
    dynamic_points = [apply(raw[valid], SHIFT)]
    dynamic_labels = [labels[valid]]
    dynamic_ids = [ids[valid]]
    dynamic_sources = [np.zeros(int(valid.sum()), np.uint8)]
    for stage, path in [(1, FUSION / block / name / 'added_points.npz'), (2, BPA / block / name / 'bpa_added_points.npz')]:
        with np.load(path) as data:
            pts = apply(data['xyz'], SHIFT)
            valid = np.isin(data['semantics'], DYNAMIC) & ~inside_body(pts, EGO)
            dynamic_points.append(pts[valid])
            dynamic_labels.append(data['semantics'][valid])
            dynamic_ids.append(data['track_id'][valid])
            dynamic_sources.append(np.full(int(valid.sum()), stage, np.uint8))
    dynamic_points = np.concatenate(dynamic_points)
    dynamic_labels = np.concatenate(dynamic_labels)
    dynamic_ids = np.concatenate(dynamic_ids)
    dynamic_sources = np.concatenate(dynamic_sources)
    dynamic_sem = voxel_labels(dynamic_points, dynamic_labels, GRID)
    baseline_dynamic_sem = dynamic_sem.copy()
    interior_new = np.zeros(GRID.shape, bool)
    interior_sem = np.full(GRID.shape, 17, np.uint8)
    interior_point_count = 0
    if INTERIOR_ROOT is not None:
        with np.load(INTERIOR_ROOT / name / 'interior_added_points.npz') as data:
            pts = apply(data['xyz'], SHIFT)
            valid = np.isin(data['semantics'], DYNAMIC) & ~inside_body(pts, EGO)
            added_pts = pts[valid]
            added_labels = data['semantics'][valid]
            added_ids = data['track_id'][valid]
        interior_point_count = len(added_pts)
        interior_sem = voxel_labels(added_pts, added_labels, GRID)
        interior_new = (baseline_dynamic_sem == 17) & (interior_sem != 17)
        dynamic_sem[interior_new] = interior_sem[interior_new]
        dynamic_points = np.concatenate([dynamic_points, added_pts])
        dynamic_labels = np.concatenate([dynamic_labels, added_labels])
        dynamic_ids = np.concatenate([dynamic_ids, added_ids])
        dynamic_sources = np.concatenate([dynamic_sources, np.full(len(added_pts), 3, np.uint8)])
        assert np.array_equal(dynamic_sem[baseline_dynamic_sem != 17], baseline_dynamic_sem[baseline_dynamic_sem != 17]), 'Interior fill changed existing dynamic class'
    dynamic_mask = dynamic_sem != 17
    semantic_conflict = dynamic_mask & (static_sem != 17) & (dynamic_sem != static_sem)
    sem, manual_mask, static_removed, static_sem = merge_semantics(static_sem, manual_sem, dynamic_sem, DYNAMIC_LAST)
    occupied = sem != 17
    measured = presence(measured_grid)
    ray_free = free_rays(SHIFT[:3, 3], measured_grid.astype(float), GRID.lo, GRID.hi, GRID.voxel, GRID.shape)
    free_conflict = ray_free & occupied
    free = ray_free & ~occupied & ~static_removed
    known = occupied | free
    camera = np.zeros(GRID.shape, bool)
    track_ids = np.full(GRID.shape, -1, np.int32)
    inside = GRID.inside(dynamic_points)
    if inside.any():
        flat = np.ravel_multi_index(GRID.index(dynamic_points[inside]).T, GRID.shape)
        labs = dynamic_labels[inside]
        identities = dynamic_ids[inside]
        winner = sem.ravel()[flat] == labs
        if INTERIOR_ROOT is not None:
            winner &= (dynamic_sources[inside] != 3) | interior_new.ravel()[flat]
        pairs = np.column_stack([flat[winner], identities[winner]])
        pairs, votes = np.unique(pairs, axis=0, return_counts=True)
        if len(pairs):
            order = np.lexsort((pairs[:, 1], -votes, pairs[:, 0]))
            pairs = pairs[order]
            first = np.r_[True, pairs[1:, 0] != pairs[:-1, 0]]
            chosen = pairs[first]
            track_ids.ravel()[chosen[:, 0]] = chosen[:, 1]
    arrays = dict(semantics=sem, mask_lidar=known, mask_camera=camera, mask_current_measured=measured & occupied, mask_inferred=occupied & ~measured, mask_world_static=(static_sem != 17) & ~dynamic_mask, mask_manual_override=manual_mask, mask_ground_interpolated=ground_added & occupied, mask_unknown_knn_assigned=unknown_knn & occupied, mask_static_removed_premerge=static_removed, mask_dynamic_interior=interior_new & occupied, mask_dynamic_interior_point_support=(interior_sem != 17) & (interior_sem == sem) & occupied, mask_dynamic_tracked=dynamic_mask & np.isin(sem, DYNAMIC), mask_dynamic_detected_only=presence(dynamic_points[dynamic_ids < -1]) & dynamic_mask & (sem == 5), mask_dynamic_temporal=presence(dynamic_points[dynamic_sources == 1]) & dynamic_mask, mask_dynamic_bpa=presence(dynamic_points[dynamic_sources == 2]) & dynamic_mask, mask_static_dynamic_conflict=semantic_conflict, mask_ray_free_conflict=free_conflict, dynamic_track_ids=track_ids)
    if DYNAMIC_LAST:
        arrays['mask_dynamic_tracked'][arrays['mask_dynamic_detected_only']] = False
        for key in ['mask_ground_interpolated', 'mask_unknown_knn_assigned']:
            arrays[key][dynamic_mask] = False
    removed = mask_arrays(arrays, {'bounds': GRID.bounds.tolist(), 'voxel_size': CFG['voxel_size_m']}, EGO)
    if DYNAMIC_LAST:
        protected = dynamic_mask & ~arrays['mask_ego_excluded']
        assert np.array_equal(arrays['semantics'][protected], dynamic_sem[protected]), 'Dynamic labels changed after final merge'
        assert not arrays['mask_lidar'][static_removed & ~arrays['mask_ego_excluded']].any()
    assert not (arrays['mask_camera'] & ~arrays['mask_lidar']).any()
    assert not ((arrays['semantics'] != 17) & ~arrays['mask_lidar']).any()
    assert not arrays['mask_lidar'][arrays['mask_ego_excluded']].any()
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest / 'labels.tmp.npz'
    np.savez_compressed(tmp, **arrays)
    tmp.replace(dest / 'labels.npz')
    counts = dict(frame=name, projected_static_points=len(static), dynamic_points=len(dynamic_points), dynamic_last=DYNAMIC_LAST, static_removed_before_merge=int(static_removed.sum()), interior_added_points=interior_point_count, interior_new_dynamic_voxels=int((interior_new & ~arrays['mask_ego_excluded']).sum()), interior_new_occupied_voxels=int((interior_new & (static_sem == 17) & ~arrays['mask_ego_excluded']).sum()), protected_dynamic_voxels=int((dynamic_mask & ~arrays['mask_ego_excluded']).sum()) if DYNAMIC_LAST else 0, dynamic_overwritten_voxels=int(((sem != dynamic_sem) & dynamic_mask & ~arrays['mask_ego_excluded']).sum()), occupied_voxels=int((arrays['mask_lidar'] & (arrays['semantics'] != 17)).sum()), free_voxels=int((arrays['mask_lidar'] & (arrays['semantics'] == 17)).sum()), unknown_voxels=int((~arrays['mask_lidar']).sum()), static_dynamic_conflicts=int(semantic_conflict.sum()), ray_free_conflicts=int(free_conflict.sum()), ego_removed=removed)
    metadata = dict(frame=name, bounds=GRID.bounds.tolist(), voxel_size=CFG['voxel_size_m'], classes=dict(enumerate(CLASSES)), free_class_id=17, T_grid_from_lidar=SHIFT.tolist(), T_world_from_grid=(world_from_lidar @ np.linalg.inv(SHIFT)).tolist(), counts=counts, version='pipeline_v3.0_dynamic_last' if DYNAMIC_LAST else 'pipeline_v3.0', labels_are_pseudo=True, static_source=str(WORLD), conflict_policy='All dynamic semantic classes, including stationary and low-speed targets, merged last and protected above manual/static labels; static cleanup before merge; no KNN/cleanup after merge' if DYNAMIC_LAST else 'human static edit > tracked dynamic > ordinary world static; occupied overrides ray-free conflicts with explicit provenance', dynamic_last=DYNAMIC_LAST, interior_source=str(INTERIOR_ROOT) if INTERIOR_ROOT is not None else None, interior_policy='Convex-hull shape-prior fill; existing dynamic classes and track IDs take priority over interior points' if INTERIOR_ROOT is not None else 'disabled', detector_only_dynamic_classes=[5] if DYNAMIC_LAST else [], negative_track_ids='-1: none; <=-2: frame-local detector-only support, not a persistent track', free_policy='only raw current-frame LiDAR rays; no rays to synthetic/fused endpoints', note='LiDAR-only. Camera mask unavailable/all false; no camera calibration is used. Not manually verified GT.')
    write(dest / 'metadata.json', metadata)
    return counts

def main():
    global XYZ, LABELS, MANUAL, ADDED_GROUND, UNKNOWN_KNN, TREE, CAMERAS, WORLD, OUT, DYNAMIC_LAST, INTERIOR_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument('--world', type=Path, default=WORLD)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--attributes-world', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--dynamic-last', action='store_true')
    parser.add_argument('--frames', nargs='+')
    parser.add_argument('--interior-root', type=Path)
    args = parser.parse_args()
    WORLD = args.world
    OUT = args.output
    DYNAMIC_LAST = args.dynamic_last
    INTERIOR_ROOT = args.interior_root
    if INTERIOR_ROOT is not None and (not DYNAMIC_LAST):
        parser.error('Interior fill requires --dynamic-last')
    OUT.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (OUT / 'runner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.time()
    write(OUT / 'status.json', dict(stage='loading_world_map'))
    with np.load(WORLD) as data:
        keep = np.isin(data['semantics'], STATIC)
        original_xyz = data['xyz']
        XYZ = original_xyz[keep].astype(float)
        LABELS = data['semantics'][keep]
        MANUAL = data['manual_override_mask'][keep] if 'manual_override_mask' in data else np.zeros(len(XYZ), bool)
        ground_key = 'is_ground_interpolated' if 'is_ground_interpolated' in data else 'is_added' if 'is_added' in data else None
        ADDED_GROUND = data[ground_key][keep] if ground_key else np.zeros(len(XYZ), bool)
        UNKNOWN_KNN = data['mask_unknown_knn_assigned'][keep] if 'mask_unknown_knn_assigned' in data else np.zeros(len(XYZ), bool)
    if args.attributes_world:
        with np.load(args.attributes_world) as data:
            assert np.array_equal(original_xyz, data['xyz']), 'Point provenance coordinates/order differ from reviewed cloud'
            if 'is_ground_interpolated' in data:
                ADDED_GROUND = data['is_ground_interpolated'][keep]
            if 'mask_unknown_knn_assigned' in data:
                UNKNOWN_KNN = data['mask_unknown_knn_assigned'][keep]
    del original_xyz, keep
    TREE = cKDTree(XYZ)
    config = json.loads((ROOT / str(CONFIG_PATH)).read_text())
    CAMERAS = {}
    for c in CAMERAS.values():
        c['T'] = c['T'] @ np.linalg.inv(SHIFT)
    names = args.frames or sorted((p.name for p in PRED.iterdir() if (p / 'point_labels.npz').exists()))
    if not args.frames:
        assert len(names) == NFRAMES
    for name in names:
        assert (TRACK / (name + '.json')).exists()
    write(OUT / 'run_config.json', dict(frames=len(names), dynamic_last=DYNAMIC_LAST, interior_source=str(INTERIOR_ROOT) if INTERIOR_ROOT is not None else None, world_source=str(WORLD), world_sha256=hashlib.sha256(WORLD.read_bytes()).hexdigest(), world_static_points=len(XYZ), bounds=GRID.bounds.tolist(), voxel_size=CFG['voxel_size_m'], workers=args.workers, attributes_world=str(args.attributes_world), supervision='known merged occupancy + measured ray free; unknown/ego excluded', calibration_source=str(ROOT / str(CONFIG_PATH)), dynamic_priority='all dynamics merged last; no later semantic processing' if DYNAMIC_LAST else 'human static edits override predicted dynamics; conflicts retained', no_additional_temporal_or_knn_or_road_fill=True))
    import shutil
    snap = OUT / 'source'
    snap.mkdir(exist_ok=True)
    for file in ['voxelize_v3_merged.py', 'ego_mask.py', 'generate_occ.py']:
        shutil.copy2(ENGINE_ROOT / file, snap / file)
    free_rays(SHIFT[:3, 3], np.array([[1.0, 1.0, 1.0]]), GRID.lo, GRID.hi, 0.4, GRID.shape)
    names = [n for n in ['000700', '001000'] if n in names] + [n for n in names if n not in ['000700', '001000']]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('fork')) as pool:
        for i, result in enumerate(pool.map(process, names, chunksize=1), 1):
            rows.append(result)
            write(OUT / 'status.json', dict(stage='voxelizing', completed=i, total=len(names), frame=result['frame'], elapsed_s=time.time() - started))
            if i % 20 == 0:
                print('Voxelized', i, '/', len(names), flush=True)
    write(OUT / 'frame_statistics.json', rows)
    report = dict(stage='complete', frames=len(rows), dynamic_last=DYNAMIC_LAST, dynamic_overwritten_voxels=sum((r['dynamic_overwritten_voxels'] for r in rows)), elapsed_s=time.time() - started, **{k: sum((r[k] for r in rows)) for k in ['occupied_voxels', 'free_voxels', 'unknown_voxels', 'static_dynamic_conflicts', 'ray_free_conflicts']})
    write(OUT / 'report.json', report)
    write(OUT / 'status.json', report)
if __name__ == '__main__':
    try:
        main()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        write(OUT / 'failure.json', dict(error=traceback.format_exc()))
        raise
