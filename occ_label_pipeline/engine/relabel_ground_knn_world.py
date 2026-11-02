"""Conservative ground-class KNN correction on small semantic components."""
from occ_label_pipeline.context import CFG, CONFIG_PATH, PROJECT_ROOT, ENGINE_ROOT, OUTPUT as CAPTURE_ROOT, DATA_ROOT, TRACKER_ROOT, NFRAMES
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from clean_world_isolated import component_sizes
from complete_world_ground_map import pack
GROUND = [11, 12, 13, 14]

def proposals(xyz, labels, ground_ids, candidates, k=24, radius=0.6, height=0.25, minimum=12, ratio=0.8, classes=None):
    classes = GROUND if classes is None else list(classes)
    tree = cKDTree(xyz[ground_ids])
    out = labels.copy()
    changed = np.zeros(len(labels), bool)
    for start in range(0, len(candidates), 20000):
        ids = candidates[start:start + 20000]
        distance, nearest = tree.query(xyz[ids], k=min(k + 1, len(ground_ids)), distance_upper_bound=radius)
        distance = np.asarray(distance)
        nearest = np.asarray(nearest)
        if nearest.ndim == 1:
            nearest = nearest[:, None]
            distance = distance[:, None]
        valid = nearest < len(ground_ids)
        safe = np.minimum(nearest, max(0, len(ground_ids) - 1))
        global_ids = ground_ids[safe]
        valid &= global_ids != ids[:, None]
        valid &= np.abs(xyz[global_ids, 2] - xyz[ids, 2, None]) <= height
        votes = np.column_stack([((labels[global_ids] == c) & valid).sum(1) for c in classes])
        total = valid.sum(1)
        best = votes.argmax(1)
        winner = np.array(classes)[best]
        support = votes.max(1) / np.maximum(total, 1)
        own = ((labels[global_ids] == labels[ids, None]) & valid).sum(1) / np.maximum(total, 1)
        accept = (total >= minimum) & (support >= ratio) & (own <= 0.2) & (winner != labels[ids])
        out[ids[accept]] = winner[accept]
        changed[ids[accept]] = True
    return (out, changed)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--all-classes', action='store_true')
    a = p.parse_args()
    classes = list(range(17)) if a.all_classes else GROUND
    height = 0.6 if a.all_classes else 0.25
    if a.output.exists():
        raise ValueError('Fresh output required')
    a.output.mkdir(parents=True)
    start = time.time()
    with np.load(a.source) as d:
        arrays = {k: d[k].copy() for k in d.files}
    xyz = arrays['xyz']
    sem = arrays['semantics']
    manual = arrays['manual_override_mask']
    candidates = []
    component_stats = []
    for cls in classes:
        if a.all_classes and cls in [2, 3, 4, 5, 6, 7, 9, 10]:
            component_stats.append({'class_id': cls, 'candidate_points': 0, 'protected_dynamic': True})
            continue
        ids = np.flatnonzero(sem == cls)
        if not len(ids):
            continue
        codes, inverse = np.unique(pack(xyz[ids]), return_inverse=True)
        sizes, _ = component_sizes(codes)
        selected = ids[(sizes[inverse] < 20) & ~manual[ids]]
        candidates.append(selected)
        component_stats.append({'class_id': cls, 'candidate_points': len(selected)})
        (a.output / 'status.json').write_text(json.dumps({'stage': 'semantic_component_analysis', 'class_id': cls, 'elapsed_s': time.time() - start}))
    candidates = np.concatenate(candidates) if candidates else np.empty(0, np.int64)
    ground = np.flatnonzero(np.isin(sem, classes))
    if len(candidates) and len(ground):
        new, changed = proposals(xyz, sem, ground, candidates, height=height, classes=classes)
    else:
        new = sem.copy()
        changed = np.zeros(len(sem), bool)
    assert not (changed & manual).any()
    assert not changed[~np.isin(sem, classes)].any()
    assert not changed[np.isin(sem, [2, 3, 4, 5, 6, 7, 9, 10])].any()
    arrays['semantics_before_allclass_knn' if a.all_classes else 'semantics_before_ground_knn'] = sem
    arrays['semantics'] = new
    arrays['mask_allclass_knn_reassigned' if a.all_classes else 'mask_ground_knn_reassigned'] = changed
    np.savez_compressed(a.output / 'relabeled_world.npz', **arrays)
    transitions = [{'from': int(c), 'to': int(d), 'points': int(((sem == c) & (new == d) & changed).sum())} for c in classes for d in classes if c != d and ((sem == c) & (new == d) & changed).any()]
    report = {'stage': 'complete', 'input': str(a.source), 'candidate_points': len(candidates), 'reassigned_points': int(changed.sum()), 'coordinates_added': 0, 'protected_manual': True, 'protected_dynamic_classes': [2, 3, 4, 5, 6, 7, 9, 10], 'component_max_exclusive': 20, 'connectivity': 26, 'k': 24, 'radius_m': 0.6, 'height_gate_m': height, 'classes': classes, 'scope': 'all_occupied_classes' if a.all_classes else 'ground_only', 'minimum_neighbors': 12, 'minimum_class_ratio': 0.8, 'maximum_own_class_ratio': 0.2, 'transitions': transitions, 'candidate_classes': component_stats, 'elapsed_s': time.time() - start}
    (a.output / 'report.json').write_text(json.dumps(report, indent=2))
    (a.output / 'status.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
if __name__ == '__main__':
    main()
