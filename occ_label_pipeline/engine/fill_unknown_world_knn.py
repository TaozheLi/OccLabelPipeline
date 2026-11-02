"""Assign existing unknown coordinates from frozen labeled neighbors; never create points."""
from occ_label_pipeline.context import CFG, CONFIG_PATH, PROJECT_ROOT, ENGINE_ROOT, OUTPUT as CAPTURE_ROOT, DATA_ROOT, TRACKER_ROOT, NFRAMES
import argparse, json, time, hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

def infer(xyz, sem, manual, k=24, radius=0.6, minimum=5, ratio=0.7, progress=None):
    known = np.flatnonzero((sem >= 0) & (sem < 17))
    unknown = np.flatnonzero((sem == -1) & ~manual)
    out = sem.copy()
    mask = np.zeros(len(sem), bool)
    agreement = np.zeros(len(sem), np.float32)
    if not len(known) or not len(unknown):
        return (out, mask, agreement)
    tree = cKDTree(xyz[known])
    n = min(k, len(known))
    for start in range(0, len(unknown), 20000):
        ids = unknown[start:start + 20000]
        dist, near = tree.query(xyz[ids], k=n, distance_upper_bound=radius, workers=4)
        if n == 1:
            dist = dist[:, None]
            near = near[:, None]
        valid = near < len(known)
        safe = np.minimum(near, len(known) - 1)
        neighbors = known[safe]
        s = sem[neighbors]
        valid &= ~np.isin(s, [11, 12, 13, 14]) | (np.abs(xyz[neighbors, 2] - xyz[ids, 2, None]) <= 0.25)
        votes = np.column_stack([((s == c) & valid).sum(1) for c in range(17)])
        count = valid.sum(1)
        winner = votes.argmax(1)
        score = votes.max(1) / np.maximum(count, 1)
        accept = (count >= minimum) & (score >= ratio)
        target = ids[accept]
        out[target] = winner[accept]
        mask[target] = True
        agreement[target] = score[accept]
        if progress:
            progress(min(start + len(ids), len(unknown)), len(unknown), int(mask.sum()))
    return (out, mask, agreement)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Output must be new')
    a.output.mkdir(parents=True)
    start = time.time()
    source_revision = a.source.stat().st_mtime_ns
    with np.load(a.source, allow_pickle=False) as d:
        arrays = {k: d[k].copy() for k in d.files}
    if a.source.stat().st_mtime_ns != source_revision:
        raise RuntimeError('Input changed while loading')
    xyz = arrays['xyz']
    sem = arrays['semantics']
    manual = arrays.get('manual_override_mask', np.zeros(len(sem), bool))

    def progress(done, total, filled):
        (a.output / 'status.json').write_text(json.dumps({'stage': 'knn_unknown_assignment', 'processed': done, 'total': total, 'assigned': filled, 'elapsed_s': time.time() - start}))
    progress(0, int(((sem == -1) & ~manual).sum()), 0)
    new, mask, agreement = infer(xyz, sem, manual, progress=progress)
    assert np.array_equal(new[sem >= 0], sem[sem >= 0])
    assert not (mask & manual).any()
    assert np.all(sem[mask] == -1)
    arrays['semantics'] = new
    arrays['semantics_before_unknown_knn'] = sem
    arrays['mask_unknown_knn_assigned'] = mask
    arrays['unknown_knn_vote_agreement'] = agreement
    if 'observed' in arrays:
        arrays['observed'] = arrays['observed'] | mask
    np.savez_compressed(a.output / 'completed_world.npz', **arrays)
    report = {'stage': 'complete', 'source': str(a.source.resolve()), 'source_mtime_ns': source_revision, 'coordinates': len(xyz), 'coordinates_added': 0, 'known_before': int((sem >= 0).sum()), 'unknown_before': int((sem == -1).sum()), 'protected_unknown': int(((sem == -1) & manual).sum()), 'knn_assigned': int(mask.sum()), 'known_after': int((new >= 0).sum()), 'unknown_after': int((new == -1).sum()), 'existing_labels_changed': 0, 'manual_override_preserved': True, 'single_pass': True, 'k': 24, 'radius_m': 0.6, 'minimum_neighbors': 5, 'minimum_vote_ratio': 0.7, 'ground_vote_height_gate_m': 0.25, 'assigned_by_class': {str(int(c)): int(v) for c, v in zip(*np.unique(new[mask], return_counts=True))}, 'elapsed_s': time.time() - start, 'note': 'KNN-inferred pseudo-labels, not manual ground truth or model confidence; sparse/ambiguous points remain unknown.'}
    for name in ['report.json', 'status.json']:
        (a.output / name).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
if __name__ == '__main__':
    main()
