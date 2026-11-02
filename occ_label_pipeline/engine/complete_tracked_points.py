"""Stage 3 pilot: motion-aware accumulation of observed tracked-object surfaces."""
from occ_label_pipeline.context import CFG, CONFIG_PATH, PROJECT_ROOT, ENGINE_ROOT, OUTPUT as CAPTURE_ROOT, DATA_ROOT, TRACKER_ROOT, NFRAMES
import json, math, csv, argparse
from pathlib import Path
import numpy as np
ROOT = PROJECT_ROOT
SOURCE = ROOT / str(CAPTURE_ROOT / 'occ_pipeline_v3_0/mctrack_people_vehicle_speed_preview100')
OUTPUT = ROOT / str(CAPTURE_ROOT / 'occ_pipeline_v3_0/tracked_surface_completion100_box_supported')
CLASSES = {'car': 4, 'truck': 10, 'bus': 3, 'trailer': 9, 'pedestrian': 7, 'bicycle': 2, 'motorcycle': 6}

def rot(yaw):
    c, s = (np.cos(yaw), np.sin(yaw))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])

def apply(x, t):
    return x @ t[:3, :3].T + t[:3, 3]

def keys(x):
    return np.floor(x / 0.1).astype(np.int64)

def unique(x):
    return np.unique(keys(x), axis=0, return_index=True)[1]

def main():
    global SOURCE, OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    SOURCE = args.source
    OUTPUT = args.output
    if OUTPUT.exists():
        raise ValueError('Fresh output required')
    OUTPUT.mkdir(parents=True)
    frames = json.loads((SOURCE / 'tracks.json').read_text())
    tracks = {}
    raw = {}
    timestamps = {}
    for i, f in enumerate(frames):
        name = f['frame']
        timestamps[i] = f['timestamp']
        d = json.loads((ROOT / f'{CAPTURE_ROOT}/training_labels_v1/detections/{name}.json').read_text())
        T = np.array(d['T_world_from_lidar'])
        with np.load(ROOT / f'{CAPTURE_ROOT}/predictions/frames/{name}/point_labels.npz') as p:
            xyz = p['xyz'].copy()
            sem = p['semantics'].copy()
            conf = p['confidence'].copy()
        raw[i] = (xyz, sem, conf, T, name)
        for b in f['boxes']:
            if b['model_class'] not in CLASSES or b['confidence'] < 0.45:
                continue
            R = rot(b['yaw_rad'])
            local = (xyz - b['center']) @ R
            mask = (np.abs(local) <= np.array(b['dimensions']) / 2).all(1) & ~(np.isin(sem, [11, 12, 13, 14]) & (conf >= 0.45))
            selected = np.flatnonzero(mask)
            if len(selected) < 5:
                continue
            item = dict(b, frame_index=i, points=xyz[selected], point_indices=selected, T=T, center_world=apply(np.array([b['center']]), T)[0])
            tracks.setdefault(b['track_id'], []).append(item)
    for observations in tracks.values():
        low = 0
        state = 'uncertain'
        segment = 0
        previous = None
        for o in observations:
            i = o['frame_index']
            if previous is not None and (timestamps[i] - timestamps[previous['frame_index']] > 0.25 or o['model_class'] != previous['model_class']):
                low = 0
                state = 'uncertain'
                segment += 1
            before = state
            if o['speed_kmh'] > 1.0:
                state = 'moving'
                low = 0
            elif o['speed_kmh'] < 0.5:
                low += 1
                if low >= 5:
                    state = 'stationary'
            else:
                low = 0
            if state != before:
                segment += 1
            o.update(state=state, segment=segment)
            previous = o
    byframe = {i: [] for i in raw}
    for observations in tracks.values():
        for o in observations:
            byframe[o['frame_index']].append((o, observations))
    rows = []
    frame_rows = []
    for i, (xyz, sem, conf, T, name) in raw.items():
        occupied = {tuple(k) for k in keys(xyz)}
        add = []
        labels = []
        ids = []
        states = []
        source_frames = []
        source_rows = []
        for target, observations in byframe[i]:
            if target['state'] == 'uncertain':
                continue
            candidates = []
            origins = []
            rowids = []
            begin = max(0, min(i - (1 if target['model_class'] == 'pedestrian' else 5), len(frames) - (3 if target['model_class'] == 'pedestrian' else 10)))
            end = begin + (3 if target['model_class'] == 'pedestrian' else 10)
            for source in observations:
                j = source['frame_index']
                if j == i or source['segment'] != target['segment'] or source['state'] != target['state']:
                    continue
                if source['model_class'] != target['model_class']:
                    continue
                if target['state'] == 'stationary':
                    if np.linalg.norm(source['center_world'] - target['center_world']) > 1.0:
                        continue
                    mapped = apply(apply(source['points'], source['T']), np.linalg.inv(T))
                else:
                    if not begin <= j < end:
                        continue
                    local = (source['points'] - source['center']) @ rot(source['yaw_rad'])
                    world_source = source['T'][:3, :3] @ rot(source['yaw_rad'])[:, 0]
                    world_target = T[:3, :3] @ rot(target['yaw_rad'])[:, 0]
                    if np.dot(world_source, world_target) < 0.5:
                        continue
                    mapped = local @ rot(target['yaw_rad']).T + target['center']
                inside = (np.abs((mapped - target['center']) @ rot(target['yaw_rad'])) <= np.array(target['dimensions']) / 2).all(1)
                candidates.append(mapped[inside])
                origins.extend([j] * int(inside.sum()))
                rowids.extend(source['point_indices'][inside].tolist())
            n = 0
            if candidates:
                pts = np.concatenate(candidates)
                origins = np.array(origins)
                rowids = np.array(rowids)
                keep = unique(pts)
                for k in keep:
                    key = tuple(keys(pts[k:k + 1])[0])
                    if key in occupied:
                        continue
                    occupied.add(key)
                    add.append(pts[k])
                    labels.append(CLASSES[target['model_class']])
                    ids.append(target['track_id'])
                    states.append(0 if target['state'] == 'stationary' else 1)
                    source_frames.append(origins[k])
                    source_rows.append(rowids[k])
                    n += 1
            rows.append(dict(frame=name, track_id=target['track_id'], class_name=target['model_class'], state=target['state'], speed_kmh=target['speed_kmh'], current_semantic_points=len(target['points']), added_points=n))
        dest = OUTPUT / name
        dest.mkdir()
        added = np.array(add, dtype=np.float32).reshape(-1, 3)
        np.savez_compressed(dest / 'added_points.npz', xyz=added, semantics=np.array(labels, np.int16), track_id=np.array(ids, np.int32), motion_state=np.array(states, np.int8), source_frame_index=np.array(source_frames, np.int16), source_prediction_row=np.array(source_rows, np.int32))
        np.savez_compressed(dest / 'completed_points.npz', xyz=np.concatenate([xyz, added]), semantics=np.r_[sem, np.array(labels, np.int16)], is_added=np.r_[np.zeros(len(xyz), bool), np.ones(len(added), bool)])
        frame_rows.append(dict(frame=name, original_points=len(xyz), added_points=len(added), completed_points=len(xyz) + len(added)))
        (OUTPUT / 'status.json').write_text(json.dumps({'stage': 'processing', 'completed': i + 1, 'total': len(frames)}))
        if (i + 1) % 20 == 0:
            print('Completed', i + 1, flush=True)
    (OUTPUT / 'object_counts.json').write_text(json.dumps(rows))
    with (OUTPUT / 'frame_counts.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(frame_rows[0]))
        w.writeheader()
        w.writerows(frame_rows)
    summary = {'stage': 'complete', 'frames': len(frames), 'original_points': sum((r['original_points'] for r in frame_rows)), 'added_points': sum((r['added_points'] for r in frame_rows)), 'added_stationary': sum((r['added_points'] for r in rows if r['state'] == 'stationary')), 'added_moving': sum((r['added_points'] for r in rows if r['state'] == 'moving')), 'by_class': {c: sum((r['added_points'] for r in rows if r['class_name'] == c)) for c in CLASSES}, 'dedup_resolution_m': 0.1, 'count_scope': 'sum per-frame points; repeated real-world surfaces across frames are counted repeatedly', 'algorithm': 'accumulate observed points inside tracked boxes, excluding confident ground labels; categories from detector; no box interior synthesis', 'stationary_rule': '5 consecutive observations <.5 km/h; moving >1; hysteresis; gaps >.25 s split; stationary world-center gate 1m', 'moving_window_frames': {'pedestrian': 3, 'vehicle': 10}, 'source': str(SOURCE), 'original_data_modified': False, 'target_box_supported_points': sum((r['current_semantic_points'] for r in rows)), 'note': 'PTv3 has insufficient dynamic semantics; detector supplies added-point classes. Box extraction may include occluders/background. Only eligible confident tracked observations; inferred points have separate provenance, not measured target-frame returns.'}
    summary['completed_points'] = summary['original_points'] + summary['added_points']
    (OUTPUT / 'report.json').write_text(json.dumps(summary, indent=2))
    (OUTPUT / 'status.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
if __name__ == '__main__':
    main()
