"""After BPA completion, fuse semantic-static and speed-stationary points in SLAM world."""
from occ_label_pipeline.context import CFG, CONFIG_PATH, PROJECT_ROOT, ENGINE_ROOT, OUTPUT as CAPTURE_ROOT, DATA_ROOT, TRACKER_ROOT, NFRAMES
import os, json, time, traceback, subprocess, sys, argparse
from pathlib import Path
import numpy as np
from complete_tracked_points import rot, apply
ROOT = PROJECT_ROOT
BASE = ROOT / str(CAPTURE_ROOT / 'occ_pipeline_v3_0')
FUSION = BASE / 'tracked_completion_all_block100'
BPA = BASE / 'ball_pivoting_all_block100'
OUT = BASE / 'static_world_completed_v3'
STATIC = [1, 8, 11, 12, 13, 14, 15, 16]
START = time.time()
PARTS = 64
MIN_CONFIDENCE = 0.45
NO_VIEWER = False
DEFER_ALL_DYNAMIC = False

def write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)

def status(stage, **kw):
    write(OUT / 'status.json', dict(stage=stage, elapsed_s=time.time() - START, **kw))

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (OUT / 'runner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUT / 'report.json').exists():
        return
    while not (BPA / 'report.json').exists():
        for parent in [BPA, FUSION]:
            if (parent / 'failure.json').exists():
                raise RuntimeError('Upstream stage failed: ' + str(parent))
        status('waiting_for_all_surface_completion', upstream=str(BPA))
        time.sleep(5)
    write(OUT / 'config.json', {'static_classes': STATIC, 'minimum_ptv3_confidence': MIN_CONFIDENCE, 'stationary_rule': 'Reuse per-block fusion stationary states (<0.5 km/h for 5 observations, hysteresis); speed alone never checked for exact zero', 'world_pose_source': str(ROOT / str(CAPTURE_ROOT / 'training_labels_v1/detections')), 'grid_m': 0.1, 'moving_box_points_removed_from_semantic_static': True, 'ego_mask': 'provisional raw-lidar box [-2,-1,-1.35,2,1,.35]', 'sources': {'1': 'PTv3 static', '2': 'stationary tracked original point', '4': 'stationary temporal accumulation', '8': 'stationary BPA surface'}, 'dedup': 'one voxel/frame/class vote, majority over frames; ties choose lower class ID; world voxel centers stored', 'moving_points_included': False, 'defer_all_dynamic': DEFER_ALL_DYNAMIC, 'speed_stationary_objects_in_static_map': not DEFER_ALL_DYNAMIC})
    staging = OUT / 'partitions'
    staging.mkdir(exist_ok=True)
    for p in staging.glob('*.bin'):
        p.unlink()
    original_count = 0
    category_counts = {'semantic_static': 0, 'stationary_original': 0, 'stationary_temporal': 0, 'stationary_bpa': 0}
    processed = 0
    excluded_chunks = []
    for block in sorted((FUSION / 'blocks').iterdir()):
        if not (block / 'report.json').exists():
            continue
        key = block.name
        rows = json.loads((block / 'object_counts.json').read_text())
        byframe = {}
        for r in rows:
            byframe.setdefault(r['frame'], []).append(r)
        names = sorted((p.name for p in block.iterdir() if (p / 'completed_points.npz').exists()))
        for name in names:
            det = json.loads((ROOT / f'{CAPTURE_ROOT}/training_labels_v1/detections/{name}.json').read_text())
            T = np.array(det['T_world_from_lidar'])
            boxes = {b['track_id']: b for b in json.loads((FUSION / 'tracking' / f'{name}.json').read_text())['boxes']}
            with np.load(ROOT / f'{CAPTURE_ROOT}/predictions/frames/{name}/point_labels.npz') as data:
                xyz = data['xyz']
                sem = data['semantics']
                conf = data['confidence']
            ground = np.isin(sem, [11, 12, 13, 14]) & (conf >= 0.45)
            keep = np.isin(sem, STATIC) & (conf >= MIN_CONFIDENCE)
            lab = sem.copy()
            src = np.ones(len(xyz), np.int32)
            stationary_ids = set()
            if DEFER_ALL_DYNAMIC:
                for index, b in enumerate(det['detections']):
                    if b['model_class'] == 'construction_vehicle' and b['confidence'] >= 0.45:
                        boxes[-2 - index] = dict(b, track_id=-2 - index)
                blocked = np.zeros(len(xyz), bool)
                for b in boxes.values():
                    if b['confidence'] < 0.45 or b['model_class'] not in ['car', 'truck', 'bus', 'trailer', 'pedestrian', 'bicycle', 'motorcycle', 'construction_vehicle']:
                        continue
                    inside = (np.abs((xyz - b['center']) @ rot(b['yaw_rad'])) <= np.array(b['dimensions']) / 2).all(1)
                    blocked |= inside & ~ground
                keep[blocked] = False
                q = np.floor(apply(xyz[blocked], T) / 0.1).astype(np.int64) + (1 << 20)
                q = q.astype(np.uint64)
                excluded_chunks.append(np.unique(q[:, 0] << np.uint64(42) | q[:, 1] << np.uint64(21) | q[:, 2]))
            else:
                for r in byframe.get(name, []):
                    b = boxes[r['track_id']]
                    inside = (np.abs((xyz - b['center']) @ rot(b['yaw_rad'])) <= np.array(b['dimensions']) / 2).all(1)
                    if r['state'] == 'moving':
                        keep[inside] = False
                    elif r['state'] == 'stationary':
                        stationary_ids.add(r['track_id'])
                        selected = inside & ~ground
                        keep[selected] = True
                        lab[selected] = __import__('complete_tracked_points').CLASSES[b['model_class']]
                        src[selected] = 2
            from ego_mask import inside_body
            shifted = xyz.copy(); shifted[:, 2] += CFG['lidar_height_m']
            ego = inside_body(shifted, {'enabled': True, 'bounds_grid_m': CFG['ego_bounds_grid_m']})
            keep &= ~ego
            points = [xyz[keep]]
            labels = [lab[keep]]
            sources = [src[keep]]
            category_counts['semantic_static'] += int(((src == 1) & keep).sum())
            category_counts['stationary_original'] += int(((src == 2) & keep).sum())
            if not DEFER_ALL_DYNAMIC:
                with np.load(block / name / 'added_points.npz') as d:
                    selected = (d['motion_state'] == 0) & np.isin(d['track_id'], list(stationary_ids))
                    points.append(d['xyz'][selected])
                    labels.append(d['semantics'][selected])
                    sources.append(np.full(int(selected.sum()), 4, np.int32))
                    category_counts['stationary_temporal'] += int(selected.sum())
                with np.load(BPA / 'blocks' / key / name / 'bpa_added_points.npz') as d:
                    selected = np.isin(d['track_id'], list(stationary_ids))
                    points.append(d['xyz'][selected])
                    labels.append(d['semantics'][selected])
                    sources.append(np.full(int(selected.sum()), 8, np.int32))
                    category_counts['stationary_bpa'] += int(selected.sum())
            points = np.concatenate(points)
            labels = np.concatenate(labels)
            sources = np.concatenate(sources)
            original_count += len(points)
            cells = np.floor(apply(points, T) / 0.1).astype(np.int32)
            a = np.column_stack([cells, labels, sources]).astype(np.int32)
            order = np.lexsort((a[:, 3], a[:, 2], a[:, 1], a[:, 0]))
            a = a[order]
            starts = np.r_[0, np.flatnonzero(np.any(a[1:, :4] != a[:-1, :4], axis=1)) + 1]
            reduced = a[starts].copy()
            reduced[:, 4] = np.bitwise_or.reduceat(a[:, 4], starts)
            buckets = (reduced[:, 0].astype(np.int64) * 73856093 ^ reduced[:, 1].astype(np.int64) * 19349663 ^ reduced[:, 2].astype(np.int64) * 83492791) % PARTS
            for partition in np.unique(buckets):
                with (staging / f'{partition:02d}.bin').open('ab') as f:
                    reduced[buckets == partition].tofile(f)
            processed += 1
            if processed % 10 == 0:
                status('projecting_static_points', completed=processed, total=NFRAMES, selected_point_observations=original_count)
            if processed % 100 == 0:
                print('Projected', processed, flush=True)
    if DEFER_ALL_DYNAMIC:
        keys = np.unique(np.concatenate(excluded_chunks))
        np.savez_compressed(OUT / 'excluded_dynamic_world_keys.npz', keys=keys)
        assert category_counts['stationary_original'] == category_counts['stationary_temporal'] == category_counts['stationary_bpa'] == 0
        del excluded_chunks
    chunks = []
    for partition in range(PARTS):
        path = staging / f'{partition:02d}.bin'
        if not path.exists():
            continue
        a = np.fromfile(path, dtype=np.int32).reshape(-1, 5)
        order = np.lexsort((a[:, 3], a[:, 2], a[:, 1], a[:, 0]))
        a = a[order]
        starts = np.r_[0, np.flatnonzero(np.any(a[1:, :4] != a[:-1, :4], axis=1)) + 1]
        votes = np.diff(np.r_[starts, len(a)])
        u = a[starts].copy()
        u[:, 4] = np.bitwise_or.reduceat(a[:, 4], starts)
        order = np.lexsort((u[:, 3], -votes, u[:, 2], u[:, 1], u[:, 0]))
        u = u[order]
        votes = votes[order]
        first = np.r_[True, np.any(u[1:, :3] != u[:-1, :3], axis=1)]
        u = u[first]
        votes = votes[first]
        dest = staging / f'{partition:02d}.npz'
        np.savez_compressed(dest, xyz=(u[:, :3].astype(np.float32) + 0.5) * 0.1, semantics=u[:, 3].astype(np.int16), source_bits=u[:, 4].astype(np.uint8), support_frames=votes.astype(np.int32))
        chunks.append(dest)
        status('deduplicating_world', completed_partitions=partition + 1, total_partitions=PARTS)
    arrays = {key: [] for key in ['xyz', 'semantics', 'source_bits', 'support_frames']}
    for p in chunks:
        with np.load(p) as d:
            for key in arrays:
                arrays[key].append(d[key])
    arrays = {k: np.concatenate(v) for k, v in arrays.items()}
    arrays['manual_override_mask'] = np.zeros(len(arrays['semantics']), bool)
    np.savez_compressed(OUT / 'static_world.npz', **arrays)
    from visualize_point_labels import PALETTE
    dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    n = len(arrays['xyz'])
    with (OUT / 'static_world.ply').open('wb') as f:
        f.write(f'ply\nformat binary_little_endian 1.0\nelement vertex {n}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'.encode())
        for start in range(0, n, 500000):
            xyz = arrays['xyz'][start:start + 500000]
            rgb = np.rint(PALETTE[arrays['semantics'][start:start + 500000]] * 255).astype(np.uint8)
            out = np.empty(len(xyz), dtype=dtype)
            for j, k in enumerate(['x', 'y', 'z']):
                out[k] = xyz[:, j]
            for j, k in enumerate(['red', 'green', 'blue']):
                out[k] = rgb[:, j]
            out.tofile(f)
    report = {'stage': 'complete', 'frames': processed, 'selected_point_observations': original_count, 'source_counts_before_world_dedup': category_counts, 'unique_world_points': n, 'dedup_m': 0.1, 'class_counts': {str(int(k)): int(v) for k, v in zip(*np.unique(arrays['semantics'], return_counts=True))}, 'elapsed_s': time.time() - START, 'defer_all_dynamic': DEFER_ALL_DYNAMIC, 'note': 'All dynamic semantic classes, including speed-stationary targets, excluded before static fusion' if DEFER_ALL_DYNAMIC else 'World voxel centers, class majority of frame votes; moving tracked regions excluded; static semantic errors may remain.'}
    write(OUT / 'report.json', report)
    write(OUT / 'status.json', report)
    subprocess.run([sys.executable, str(ENGINE_ROOT / 'preview_static_world_map.py'), '--root', str(OUT)], check=True)
    if os.environ.get('DISPLAY') and (not NO_VIEWER):
        log = (OUT / 'viewer.log').open('w')
        subprocess.Popen([sys.executable, str(ENGINE_ROOT / 'visualize_point_labels.py'), str(OUT / 'static_world.npz'), '--point-size', '2', '--max-points', '3000000'], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--minimum-confidence', type=float, default=0.45)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--no-viewer', action='store_true')
    parser.add_argument('--defer-all-dynamic', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.minimum_confidence <= 1:
        parser.error('confidence must be 0..1')
    OUT = args.output
    MIN_CONFIDENCE = args.minimum_confidence
    NO_VIEWER = args.no_viewer
    DEFER_ALL_DYNAMIC = args.defer_all_dynamic
    try:
        main()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        write(OUT / 'failure.json', {'error': traceback.format_exc()})
        raise
