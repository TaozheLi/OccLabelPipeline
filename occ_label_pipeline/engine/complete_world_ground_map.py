"""Whole-map ground completion in world coordinates, preserving annotation indices."""
from occ_label_pipeline.context import CFG, CONFIG_PATH, PROJECT_ROOT, ENGINE_ROOT, OUTPUT as CAPTURE_ROOT, DATA_ROOT, TRACKER_ROOT, NFRAMES
import json, time, hashlib, traceback, argparse
from pathlib import Path
import numpy as np
from scipy.spatial import QhullError
from experiment_static_surface_completion import ground_fill
ROOT = PROJECT_ROOT
BASE = ROOT / str(CAPTURE_ROOT / 'occ_pipeline_v3_0')
SOURCE = BASE / 'static_world_manual_v3/manual_knn_labels.npz'
OUT = BASE / 'world_ground_completed_v3'
NAMES = {11: 'driveable_surface', 12: 'other_flat', 13: 'sidewalk', 14: 'terrain'}
START = time.time()

def write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)

def pack(x):
    q = np.floor(x / 0.1).astype(np.int64) + (1 << 20)
    if not ((q >= 0) & (q < 1 << 21)).all():
        raise ValueError('Coordinates exceed voxel key range')
    q = q.astype(np.uint64)
    return q[:, 0] << np.uint64(42) | q[:, 1] << np.uint64(21) | q[:, 2]

def main():
    global SOURCE, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    SOURCE = args.source
    OUT = args.output
    if OUT.exists():
        raise ValueError('Fresh output required')
    OUT.mkdir(parents=True)
    revision = SOURCE.stat().st_mtime_ns
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    with np.load(SOURCE) as d:
        xyz = d['xyz'].copy()
        sem = d['semantics'].copy()
        manual = d['manual_override_mask'].copy()
        removed = d['mask_removed_isolated'].copy() if 'mask_removed_isolated' in d else np.zeros(len(sem), bool)
    if SOURCE.stat().st_mtime_ns != revision:
        raise RuntimeError('Annotation changed while reading snapshot')
    all_keys = pack(xyz)
    existing_all, original_indices = np.unique(all_keys, return_index=True)
    existing = np.unique(all_keys[~removed])
    all_added = []
    all_labels = []
    stats = []
    processed = 0
    write(OUT / 'status.json', {'stage': 'indexing_ground_tiles', 'input_points': len(xyz)})
    for cls, name in NAMES.items():
        ids = np.flatnonzero(sem == cls)
        points = xyz[ids]
        record = {'class_id': cls, 'class_name': name, 'input_points': len(points), 'added_points': 0, 'tiles': 0, 'failed_tiles': 0}
        if len(points) < 30:
            stats.append(record)
            continue
        tile = np.floor(points[:, :2] / 8.0).astype(np.int32)
        order = np.lexsort((tile[:, 1], tile[:, 0]))
        points = points[order]
        tile = tile[order]
        starts = np.r_[0, np.flatnonzero(np.any(tile[1:] != tile[:-1], axis=1)) + 1]
        ends = np.r_[starts[1:], len(points)]
        lookup = {tuple(tile[a]): (a, b) for a, b in zip(starts, ends)}
        parts = []
        for cell, (a, b) in lookup.items():
            pieces = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    interval = lookup.get((cell[0] + dx, cell[1] + dy))
                    if interval:
                        pieces.append(points[interval[0]:interval[1]])
            support = np.concatenate(pieces)
            lo = np.array(cell) * 8.0
            hi = lo + 8.0
            mask = (support[:, :2] >= lo - 0.7).all(1) & (support[:, :2] <= hi + 0.7).all(1)
            support = support[mask]
            if len(support) < 30:
                continue
            try:
                new, triangles = ground_fill(support, cls)
            except QhullError:
                record['failed_tiles'] += 1
                continue
            if len(new):
                new = new.astype(np.float32)
                new = new[(new[:, :2] >= lo).all(1) & (new[:, :2] < hi).all(1)]
                if len(new):
                    k = pack(new)
                    _, indices = np.unique(k, return_index=True)
                    new = new[indices]
                    k = k[indices]
                    pos = np.searchsorted(existing, k)
                    valid = pos < len(existing)
                    collision = np.zeros(len(k), bool)
                    collision[valid] = existing[pos[valid]] == k[valid]
                    new = new[~collision]
                    if len(new):
                        parts.append(new)
            record['tiles'] += 1
            processed += 1
            if processed % 10 == 0:
                write(OUT / 'status.json', {'stage': 'filling_world_ground', 'class': name, 'processed_tiles': processed, 'elapsed_s': time.time() - START})
        if parts:
            new = np.concatenate(parts)
            all_added.append(new)
            all_labels.append(np.full(len(new), cls, np.int16))
            record['added_points'] = len(new)
        stats.append(record)
        write(OUT / 'class_progress.json', stats)
        print(name, record, flush=True)
    added = np.concatenate(all_added) if all_added else np.empty((0, 3), np.float32)
    labels = np.concatenate(all_labels) if all_labels else np.empty(0, np.int16)
    if len(added):
        from scipy.spatial import cKDTree
        scores = np.empty(len(added))
        offset = 0
        for r in stats:
            n = r['added_points']
            if n:
                scores[offset:offset + n] = cKDTree(xyz[sem == r['class_id']]).query(added[offset:offset + n])[0]
            offset += n
        k = pack(added)
        order = np.lexsort((labels, scores, k))
        ks = k[order]
        first = np.r_[True, ks[1:] != ks[:-1]]
        keep = order[first]
        added = added[keep]
        labels = labels[keep]
    assert len(np.unique(pack(added))) == len(added)
    reassigned = np.zeros(len(xyz), bool)
    reassigned_labels = np.empty(0, np.int16)
    if len(added):
        codes = pack(added)
        positions = np.searchsorted(existing_all, codes)
        matches = positions < len(existing_all)
        matches[matches] = existing_all[positions[matches]] == codes[matches]
        indices = original_indices[positions[matches]]
        assert removed[indices].all()
        reassigned[indices] = True
        reassigned_labels = labels[matches].copy()
        sem[indices] = reassigned_labels
        added = added[~matches]
        labels = labels[~matches]
    np.savez_compressed(OUT / 'added_ground_points.npz', xyz=added, semantics=labels, source=np.array('local_world_ground_height_interpolation'), source_cloud_sha256=np.array(digest))
    np.savez_compressed(OUT / 'completed_world.npz', xyz=np.concatenate([xyz, added]), semantics=np.r_[sem, labels], manual_override_mask=np.r_[manual, np.zeros(len(added), bool)], is_added=np.r_[np.zeros(len(xyz), bool), np.ones(len(added), bool)], is_ground_interpolated=np.r_[reassigned, np.ones(len(added), bool)], mask_ground_reassigned=np.r_[reassigned, np.zeros(len(added), bool)], mask_removed_isolated=np.r_[removed, np.zeros(len(added), bool)], original_point_index=np.r_[np.arange(len(xyz), dtype=np.int32), np.full(len(added), -1, np.int32)], source_cloud_sha256=np.array(digest))
    for r in stats:
        r['added_points_final'] = int((labels == r['class_id']).sum())
        r['reassigned_points'] = int((reassigned_labels == r['class_id']).sum())
    report = {'stage': 'complete', 'input_points': len(xyz), 'added_points': len(added), 'reassigned_existing_points': int(reassigned.sum()), 'removed_points_still_unknown': int((removed & (sem < 0)).sum()), 'completed_points': len(xyz) + len(added), 'classes': stats, 'source': str(SOURCE), 'source_sha256': digest, 'source_modified': False, 'original_indices_preserved': True, 'scope': 'entire merged world map, not per-frame totals', 'method': '8m tiles with .7m halo, same-class 2.5D triangulation, max triangle edge .6m, support distance .3m, slope <=25deg or terrain45deg, XY-height spread <=.3m', 'grid_m': 0.1, 'interior_fill': False, 'elapsed_s': time.time() - START}
    write(OUT / 'report.json', report)
    write(OUT / 'status.json', report)
    subprocess = __import__('subprocess')
if __name__ == '__main__':
    try:
        main()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        write(OUT / 'failure.json', {'error': traceback.format_exc()})
        raise
