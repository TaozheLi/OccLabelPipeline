"""Sparse all-class 26-connected component cleanup, preserving world-point indices."""
from occ_label_pipeline.context import CFG, CONFIG_PATH, PROJECT_ROOT, ENGINE_ROOT, OUTPUT as CAPTURE_ROOT, DATA_ROOT, TRACKER_ROOT, NFRAMES
import json, time, argparse, hashlib
from pathlib import Path
import numpy as np
from numba import njit
from complete_world_ground_map import pack

@njit(cache=True)
def find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i

@njit(cache=True)
def join(parent, sizes, left, right):
    for i, j in zip(left, right):
        a = find(parent, i)
        b = find(parent, j)
        if a == b:
            continue
        if sizes[a] < sizes[b]:
            a, b = (b, a)
        parent[b] = a
        sizes[a] += sizes[b]

@njit(cache=True)
def roots(parent):
    result = np.empty(len(parent), np.int32)
    for i in range(len(parent)):
        result[i] = find(parent, i)
    return result

def component_sizes(codes, progress=None):
    codes = np.asarray(codes, np.int64)
    n = len(codes)
    parent = np.arange(n, dtype=np.int32)
    sizes = np.ones(n, np.int32)
    offsets = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if (dx, dy, dz) < (0, 0, 0):
                    offsets.append(dx * (1 << 42) + dy * (1 << 21) + dz)
    for index, offset in enumerate(offsets, 1):
        for begin in range(0, n, 1000000):
            query = codes[begin:begin + 1000000] + offset
            positions = np.searchsorted(codes, query)
            valid = positions < n
            valid[valid] = codes[positions[valid]] == query[valid]
            left = np.flatnonzero(valid).astype(np.int32) + begin
            right = positions[valid].astype(np.int32)
            join(parent, sizes, left, right)
        if progress:
            progress(index, len(offsets))
    root = roots(parent)
    return (sizes[root], root)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--minimum-voxels', type=int, default=5)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Fresh output required')
    if a.minimum_voxels < 1:
        raise ValueError('minimum-voxels must be positive')
    a.output.mkdir(parents=True)
    start = time.time()
    digest = hashlib.sha256(a.source.read_bytes()).hexdigest()
    with np.load(a.source) as d:
        xyz = d['xyz']
        sem = d['semantics'].copy()
        manual = d['manual_override_mask'].copy()
    ids = np.flatnonzero(sem >= 0)
    codes, inverse = np.unique(pack(xyz[ids]), return_inverse=True)

    def progress(i, n):
        (a.output / 'status.json').write_text(json.dumps({'stage': 'connected_components', 'neighbor_offsets_done': i, 'total_offsets': n, 'valid_voxels': len(codes), 'elapsed_s': time.time() - start}))
    size, root = component_sizes(codes, progress)
    small = size < a.minimum_voxels
    removed = np.zeros(len(xyz), bool)
    removed[ids] = small[inverse]
    old = sem.copy()
    sem[removed] = -1
    np.savez_compressed(a.output / 'cleaned_world.npz', xyz=xyz, semantics=sem, manual_override_mask=manual & ~removed, manual_override_before_cleanup=manual, mask_removed_isolated=removed, semantics_before_cleanup=old, source_sha256=np.array(digest))
    report = {'stage': 'complete', 'input_points': len(xyz), 'valid_points_before': len(ids), 'removed_label_points': int(removed.sum()), 'removed_voxels': int(small.sum()), 'removed_components': len(np.unique(root[small])), 'minimum_voxels': a.minimum_voxels, 'connectivity': 26, 'grid_m': 0.1, 'scope': 'all labeled classes jointly; no semantic boundary splitting', 'coordinates_deleted': 0, 'removed_state': 'unknown (-1), eligible for supported later ground reassignment', 'by_previous_class': {str(int(c)): int(n) for c, n in zip(*np.unique(old[removed], return_counts=True))}, 'source': str(a.source), 'source_sha256': digest, 'elapsed_s': time.time() - start}
    (a.output / 'report.json').write_text(json.dumps(report, indent=2))
    (a.output / 'status.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
if __name__ == '__main__':
    main()
