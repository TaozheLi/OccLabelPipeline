"""Exclude a configurable ego body box; excluded space is unknown, never free."""
from occ_label_pipeline.context import CFG, CONFIG_PATH, PROJECT_ROOT, ENGINE_ROOT, OUTPUT as CAPTURE_ROOT, DATA_ROOT, TRACKER_ROOT, NFRAMES
import argparse
import json
from pathlib import Path
import numpy as np

def inside_body(points, config):
    if not config or not config.get('enabled', False):
        return np.zeros(len(points), dtype=bool)
    bounds = np.asarray(config['bounds_grid_m'], dtype=float)
    if bounds.shape != (6,) or not np.isfinite(bounds).all() or np.any(bounds[3:] <= bounds[:3]):
        raise ValueError('ego_mask.bounds_grid_m requires finite xmin,ymin,zmin,xmax,ymax,zmax')
    return ((points >= bounds[:3]) & (points < bounds[3:])).all(axis=1)

def mask_arrays(arrays, metadata, config):
    sem = arrays['semantics']
    indices = np.indices(sem.shape).reshape(3, -1).T
    centers = np.asarray(metadata['bounds'][:3]) + (indices + 0.5) * metadata['voxel_size']
    mask = inside_body(centers, config).reshape(sem.shape)
    removed = int((mask & arrays['mask_lidar'].astype(bool) & (sem != 17)).sum())
    for key, value in arrays.items():
        if value.shape == sem.shape and key.startswith('mask_'):
            value[mask] = False
    sem[mask] = 17
    if 'dynamic_track_ids' in arrays:
        arrays['dynamic_track_ids'][mask] = -1
    arrays['mask_ego_excluded'] = mask
    return removed

def process(input_root, output_root, config):
    source, out = (Path(input_root).resolve(), Path(output_root).resolve())
    if not config.get('enabled', False):
        return source
    if out.exists():
        raise ValueError('Use a new ego-mask output directory')
    paths = sorted(source.glob('*/labels.npz'))
    if not paths:
        raise ValueError('No input labels')
    inside_body(np.zeros((0, 3)), config)
    out.mkdir(parents=True)
    total = 0
    for i, path in enumerate(paths, 1):
        metadata = json.loads(path.with_name('metadata.json').read_text())
        with np.load(path) as data:
            arrays = {k: data[k].copy() for k in data.files}
        count = mask_arrays(arrays, metadata, config)
        total += count
        metadata['ego_mask'] = {**config, 'removed_occupied_voxels': count, 'source_root': str(source), 'excluded_space': 'unknown'}
        dest = out / path.parent.name
        dest.mkdir()
        np.savez_compressed(dest / 'labels.npz', **arrays)
        (dest / 'metadata.json').write_text(json.dumps(metadata, indent=2))
        (out / 'status.json').write_text(json.dumps({'stage': 'complete' if i == len(paths) else 'processing', 'completed': i, 'total': len(paths), 'removed_occupied_voxels': total}))
        if i % 100 == 0:
            print(f'Ego mask {i} / {len(paths)}', flush=True)
    (out / 'report.json').write_text(json.dumps({'frames': len(paths), 'removed_occupied_voxels': total, 'config': config, 'source_root': str(source)}, indent=2))
    return out
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    process(args.input_root, args.output_root, json.loads(Path(args.config).read_text())['ego_mask'])
