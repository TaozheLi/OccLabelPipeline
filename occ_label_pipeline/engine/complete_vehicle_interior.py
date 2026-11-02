"""Experimental convex-hull vehicle interior fill, isolated from static processing.
Not a watertight BPA claim: a convex envelope is a shape prior and may bridge
concavities. Infer only strictly inside supported hulls; never fill whole boxes.
"""
from occ_label_pipeline.context import CFG, CONFIG_PATH, PROJECT_ROOT, ENGINE_ROOT, OUTPUT as CAPTURE_ROOT, DATA_ROOT, TRACKER_ROOT, NFRAMES
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse, json, time, traceback, multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from numba import njit
from complete_tracked_points import rot, keys, CLASSES
from voxelize_v3_merged import raw_dynamic, SHIFT, EGO, inside_body, GRID, apply, voxel_labels
ROOT = PROJECT_ROOT
BASE = ROOT / str(CAPTURE_ROOT / 'occ_pipeline_v3_0')
TRACK = BASE / 'tracked_completion_all_block100'
BPA = BASE / 'ball_pivoting_all_block100'
PRED = ROOT / str(CAPTURE_ROOT / 'predictions/frames')
DETS = ROOT / str(CAPTURE_ROOT / 'training_labels_v1/detections')
OUT = BASE / 'vehicle_interior_hull_preview100'
VEHICLES = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle']

@njit(cache=True)
def strictly_inside(points, equations, margin):
    keep = np.ones(len(points), np.bool_)
    for i in range(len(points)):
        for j in range(len(equations)):
            if equations[j, 0] * points[i, 0] + equations[j, 1] * points[i, 1] + equations[j, 2] * points[i, 2] + equations[j, 3] > -margin:
                keep[i] = False
                break
    return keep

def interior_candidates(local, center, R, dims, spacing=0.1, margin=0.05):
    if len(local) < 30:
        return (np.empty((0, 3), np.float32), {'status': 'skipped_sparse'})
    span = np.ptp(local, axis=0)
    if np.any(span < np.array([0.6, 0.4, 0.4])):
        return (np.empty((0, 3), np.float32), {'status': 'skipped_thin_support', 'observed_spans_m': span.tolist()})
    try:
        hull = ConvexHull(local)
    except QhullError:
        return (np.empty((0, 3), np.float32), {'status': 'skipped_degenerate'})
    if hull.volume < 0.05:
        return (np.empty((0, 3), np.float32), {'status': 'skipped_small_volume', 'hull_volume_m3': hull.volume})
    world = local[hull.vertices] @ R.T + center
    lo = np.floor(world.min(0) / spacing).astype(int)
    hi = np.floor(world.max(0) / spacing).astype(int)
    if np.prod(hi - lo + 1) > 1000000:
        return (np.empty((0, 3), np.float32), {'status': 'skipped_large_candidate_grid'})
    grid = np.indices(tuple(hi - lo + 1)).reshape(3, -1).T + lo
    xyz = ((grid + 0.5) * spacing).astype(np.float32)
    loc = (xyz - center) @ R
    keep = (np.abs(loc) <= dims / 2).all(1) & strictly_inside(loc, hull.equations, margin)
    return (xyz[keep], {'status': 'complete', 'hull_volume_m3': float(hull.volume), 'box_volume_m3': float(np.prod(dims)), 'observed_spans_m': span.tolist(), 'hull_vertices': len(hull.vertices), 'hull_triangles': len(hull.simplices), 'candidate_interior_cells': int(keep.sum())})

def write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)

def process(name):
    dest = OUT / name
    if (dest / 'report.json').exists():
        return json.loads((dest / 'report.json').read_text())
    index = int(name) - 1
    start = index // 100 * 100 + 1
    block = f'{start:06d}_{min(start + 99, NFRAMES):06d}'
    with np.load(PRED / name / 'point_labels.npz') as d:
        raw = d['xyz']
        sem = d['semantics']
        conf = d['confidence']
    with np.load(TRACK / 'blocks' / block / name / 'added_points.npz') as d:
        temporal = d['xyz']
        tids = d['track_id']
        tsem = d['semantics']
    with np.load(BPA / 'blocks' / block / name / 'bpa_added_points.npz') as d:
        surface = d['xyz']
        sids = d['track_id']
        ssem = d['semantics']
    boxes = json.loads((TRACK / 'tracking' / (name + '.json')).read_text())['boxes']
    pose = json.loads((DETS / (name + '.json')).read_text())
    boxes = boxes + [dict(b, track_id=-2 - i) for i, b in enumerate(pose['detections']) if b['model_class'] == 'construction_vehicle' and b['confidence'] >= 0.45]
    raw_labels, raw_ids = raw_dynamic(raw, sem, conf, boxes)
    existing = np.concatenate([raw, temporal, surface])
    occupied = set(map(tuple, keys(existing)))
    added = []
    labels = []
    identities = []
    objects = []
    for box in sorted(boxes, key=lambda b: -b['confidence']):
        if box['model_class'] not in VEHICLES or box['confidence'] < 0.45:
            continue
        tid = box['track_id']
        R = rot(box['yaw_rad'])
        center = np.asarray(box['center'])
        dims = np.asarray(box['dimensions'])
        lab = 5 if box['model_class'] == 'construction_vehicle' else CLASSES[box['model_class']]
        current = raw[(raw_ids == tid) & (raw_labels == lab)]
        accumulated = temporal[(tids == tid) & (tsem == lab)]
        bpa = surface[(sids == tid) & (ssem == lab)]
        points = np.concatenate([current, accumulated, bpa])
        local = (points - center) @ R
        local = local[(np.abs(local) <= dims / 2).all(1)]
        if len(local):
            _, indices = np.unique(np.floor(local / 0.1).astype(np.int64), axis=0, return_index=True)
            local = local[indices]
        candidates, details = interior_candidates(local, center, R, dims)
        new = []
        if len(candidates):
            candidates = candidates[~inside_body(apply(candidates, SHIFT), EGO)]
        candidate_keys = keys(candidates)
        for pt, cell in zip(candidates, candidate_keys):
            k = tuple(cell)
            if k in occupied:
                continue
            occupied.add(k)
            new.append(pt)
        count = len(new)
        if count:
            added.append(np.asarray(new, np.float32))
            labels.append(np.full(count, lab, np.int16))
            identities.append(np.full(count, tid, np.int32))
        objects.append({'frame': name, 'track_id': tid, 'class_name': box['model_class'], 'speed_kmh': box['speed_kmh'], 'current_points': len(current), 'temporal_points': len(accumulated), 'bpa_points': len(bpa), 'hull_input_unique_cells': len(local), 'interior_added_points': count, **details})
    added = np.concatenate(added) if added else np.empty((0, 3), np.float32)
    labels = np.concatenate(labels) if labels else np.empty(0, np.int16)
    identities = np.concatenate(identities) if identities else np.empty(0, np.int32)
    assert not set(map(tuple, keys(existing))).intersection(map(tuple, keys(added)))
    assert len(set(map(tuple, keys(added)))) == len(added)
    dest.mkdir(exist_ok=True)
    np.savez_compressed(dest / 'interior_added_points.npz', xyz=added, semantics=labels, track_id=identities, provenance=np.full(len(added), 3, np.uint8))
    raw_dynamic_count = int((raw_labels >= 0).sum())
    before = raw_dynamic_count + len(temporal) + len(surface)
    report = {'frame': name, 'raw_points': len(raw), 'raw_dynamic_points': raw_dynamic_count, 'temporal_added_points': len(temporal), 'bpa_added_points': len(surface), 'dynamic_point_records_before': before, 'interior_added_points': len(added), 'dynamic_point_records_after': before + len(added), 'whole_frame_records_before': len(existing), 'whole_frame_records_after': len(existing) + len(added), 'objects': objects, 'added_by_class': {cls: sum((r['interior_added_points'] for r in objects if r['class_name'] == cls)) for cls in VEHICLES}}
    write(dest / 'report.json', report)
    return report

def main():
    global OUT
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=OUT)
    p.add_argument('--start', type=int, default=700)
    p.add_argument('--count', type=int, default=100)
    p.add_argument('--workers', type=int, default=6)
    a = p.parse_args()
    OUT = a.output
    OUT.mkdir(exist_ok=True)
    import fcntl
    lock = (OUT / 'runner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.time()
    rows = []
    strictly_inside(np.zeros((1, 3)), np.array([[1.0, 0.0, 0.0, -1.0]]), 0.05)
    names = [f'{i:06d}' for i in range(a.start, a.start + a.count)]
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('fork')) as pool:
        for i, row in enumerate(pool.map(process, names), 1):
            rows.append(row)
            write(OUT / 'status.json', {'stage': 'filling_vehicle_interiors', 'completed': i, 'total': len(names), 'elapsed_s': time.time() - started})
    report = {'stage': 'complete', 'frames': len(rows), 'range': [a.start, a.start + a.count - 1], 'elapsed_s': time.time() - started, 'method': 'closed convex hull of observed + motion-aware accumulated + BPA object points; sample strictly >=0.05m inside hull, 0.1m LiDAR-aligned grid, clip to detection box', 'vehicle_classes': VEHICLES, 'pedestrians_bicycles_motorcycles_filled': False, 'count_scope': 'sum of frame-local point records; no global unique-point claim', 'dedup': 'new 0.1m cells, checked against all raw/temporal/BPA points and preceding object fills in each frame', 'original_files_modified': False, **{key: sum((r[key] for r in rows)) for key in ['raw_points', 'raw_dynamic_points', 'temporal_added_points', 'bpa_added_points', 'dynamic_point_records_before', 'interior_added_points', 'dynamic_point_records_after', 'whole_frame_records_before', 'whole_frame_records_after']}, 'by_class': {cls: sum((r['added_by_class'][cls] for r in rows)) for cls in VEHICLES}, 'objects_attempted': sum((len(r['objects']) for r in rows)), 'objects_with_fill': sum((o['interior_added_points'] > 0 for r in rows for o in r['objects'])), 'limitations': 'Convex envelope may bridge wheel wells/concavities and fills cabin air by prior; not measured LiDAR or watertight BPA reconstruction.'}
    write(OUT / 'frame_statistics.json', rows)
    write(OUT / 'report.json', report)
    write(OUT / 'status.json', report)
    print(json.dumps(report, indent=2))
if __name__ == '__main__':
    try:
        main()
    except Exception:
        OUT.mkdir(exist_ok=True)
        write(OUT / 'failure.json', {'error': traceback.format_exc()})
        raise
