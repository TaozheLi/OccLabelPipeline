"""Append provenance-tagged BPA surface points after each completed 100-frame block."""
from occ_label_pipeline.context import CFG, CONFIG_PATH, PROJECT_ROOT, ENGINE_ROOT, OUTPUT as CAPTURE_ROOT, DATA_ROOT, TRACKER_ROOT, NFRAMES
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import json, time, csv, traceback, shutil
from pathlib import Path
import numpy as np
import open3d as o3d
from complete_tracked_points import rot, keys, unique, CLASSES
from experiment_tracked_surface_reconstruction import sample_mesh
ROOT = PROJECT_ROOT
SOURCE = ROOT / str(CAPTURE_ROOT / 'occ_pipeline_v3_0/tracked_completion_all_block100')
OUT = ROOT / str(CAPTURE_ROOT / 'occ_pipeline_v3_0/ball_pivoting_all_block100')
PRED = ROOT / str(CAPTURE_ROOT / 'predictions/frames')
START = time.time()

def write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (OUT / 'runner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write(OUT / 'config.json', dict(source=str(SOURCE), method='Ball Pivoting', radii_m=[0.15, 0.25, 0.4], normal_radius_m=0.4, normal_neighbors=30, minimum_object_points=30, dedup_m=0.1, max_distance_from_input_m=0.3, max_edge_m={'vehicle': 1.0, 'pedestrian_bicycle_motorcycle': 0.5}, interior_fill=False, classes=list(CLASSES), box_confidence=0.45, provenance={'0': 'original frame', '1': 'motion-aware multiframe accumulation', '2': 'BPA reconstructed surface'}))
    snap = OUT / 'source'
    snap.mkdir(exist_ok=True)
    for name in ['run_ball_pivoting_completion.py', 'experiment_tracked_surface_reconstruction.py', 'complete_tracked_points.py']:
        shutil.copy2(ENGINE_ROOT / name, snap / name)
    frames = sorted((p.name for p in PRED.iterdir() if (p / 'point_labels.npz').exists()))
    assert len(frames) == NFRAMES
    reports = []
    count = 0
    for start in range(0, len(frames), 100):
        names = frames[start:start + 100]
        key = names[0] + '_' + names[-1]
        block = SOURCE / 'blocks' / key
        dest = OUT / 'blocks' / key
        dest.mkdir(parents=True, exist_ok=True)
        while not (block / 'report.json').exists():
            if (SOURCE / 'failure.json').exists():
                raise RuntimeError('Upstream fusion failed')
            write(OUT / 'status.json', dict(stage='waiting_for_fused_block', block=key, completed=count, total=NFRAMES, elapsed_s=time.time() - START))
            time.sleep(5)
        if (dest / 'report.json').exists():
            reports.append(json.loads((dest / 'report.json').read_text()))
            count += len(names)
            continue
        object_records = json.loads((block / 'object_counts.json').read_text())
        byframe = {f: [] for f in names}
        for r in object_records:
            byframe[r['frame']].append(r)
        frame_rows = []
        object_rows = []
        for name in names:
            with np.load(PRED / name / 'point_labels.npz') as data:
                raw = data['xyz']
                sem = data['semantics']
                conf = data['confidence']
            with np.load(block / name / 'added_points.npz') as data:
                temporal = data['xyz']
                temporal_ids = data['track_id']
                temporal_sem = data['semantics']
            boxes = {b['track_id']: b for b in json.loads((SOURCE / 'tracking' / f'{name}.json').read_text())['boxes']}
            occupied = set(map(tuple, keys(np.concatenate([raw, temporal]))))
            new = []
            new_ids = []
            new_sem = []
            for r in byframe[name]:
                b = boxes[r['track_id']]
                R = rot(b['yaw_rad'])
                c = np.array(b['center'])
                dims = np.array(b['dimensions'])
                local = (raw - c) @ R
                selected = (np.abs(local) <= dims / 2).all(1) & ~(np.isin(sem, [11, 12, 13, 14]) & (conf >= 0.45))
                base = np.concatenate([local[selected], (temporal[temporal_ids == r['track_id']] - c) @ R])
                base = base[unique(base)]
                record = dict(frame=name, track_id=r['track_id'], class_name=r['class_name'], input_unique_cells=len(base), added_points=0)
                if len(base) < 30:
                    record['status'] = 'skipped_sparse'
                    object_rows.append(record)
                    continue
                try:
                    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(base))
                    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.4, max_nn=30))
                    cloud.orient_normals_consistent_tangent_plane(min(10, len(base) - 1))
                    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(cloud, o3d.utility.DoubleVector([0.15, 0.25, 0.4]))
                    points, triangles = sample_mesh(mesh, base, dims, 0.5 if b['model_class'] in ['pedestrian', 'bicycle', 'motorcycle'] else 1.0)
                    points = points @ R.T + c
                    for point in points:
                        k = tuple(keys(point[None])[0])
                        if k in occupied:
                            continue
                        occupied.add(k)
                        new.append(point)
                        new_ids.append(r['track_id'])
                        new_sem.append(CLASSES[b['model_class']])
                        record['added_points'] += 1
                    record.update(status='complete', triangles_used=triangles)
                except RuntimeError as error:
                    record.update(status='skipped_geometry_failure', error=str(error))
                object_rows.append(record)
            added = np.array(new, np.float32).reshape(-1, 3)
            frameout = dest / name
            frameout.mkdir(exist_ok=True)
            np.savez_compressed(frameout / 'bpa_added_points.npz', xyz=added, semantics=np.array(new_sem, np.int16), track_id=np.array(new_ids, np.int32), provenance=np.full(len(added), 2, np.uint8))
            np.savez_compressed(frameout / 'completed_points.npz', xyz=np.concatenate([raw, temporal, added]), semantics=np.concatenate([sem, temporal_sem, np.array(new_sem, np.int16)]), provenance=np.r_[np.zeros(len(raw), np.uint8), np.ones(len(temporal), np.uint8), np.full(len(added), 2, np.uint8)])
            frame_rows.append(dict(frame=name, original_points=len(raw), temporal_added=len(temporal), bpa_added=len(added), completed_points=len(raw) + len(temporal) + len(added)))
            count += 1
            write(OUT / 'status.json', dict(stage='surface_reconstruction', completed=count, total=NFRAMES, block=key, block_bpa_added=sum((x['bpa_added'] for x in frame_rows)), elapsed_s=time.time() - START))
        with (dest / 'frame_counts.csv').open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(frame_rows[0]))
            w.writeheader()
            w.writerows(frame_rows)
        write(dest / 'object_counts.json', object_rows)
        report = {'block': key, 'frames': len(names), 'original_points': sum((r['original_points'] for r in frame_rows)), 'temporal_added': sum((r['temporal_added'] for r in frame_rows)), 'bpa_added': sum((r['bpa_added'] for r in frame_rows)), 'by_class': {c: sum((r['added_points'] for r in object_rows if r['class_name'] == c)) for c in CLASSES}, 'skipped_objects': sum((r['status'] != 'complete' for r in object_rows))}
        write(dest / 'report.json', report)
        reports.append(report)
        write(OUT / 'block_reports.json', reports)
        print('Finished', key, 'BPA added', report['bpa_added'], flush=True)
    report = {'stage': 'complete', 'frames': NFRAMES, 'blocks': len(reports), 'elapsed_s': time.time() - START, **{k: sum((r[k] for r in reports)) for k in ['original_points', 'temporal_added', 'bpa_added']}, 'by_class': {c: sum((r['by_class'][c] for r in reports)) for c in CLASSES}, 'count_scope': 'sum of per-frame points; added BPA points unique at 0.1m in each target frame, not globally unique surfaces', 'interior_fill': False, 'original_files_modified': False}
    report['completed_points'] = report['original_points'] + report['temporal_added'] + report['bpa_added']
    write(OUT / 'report.json', report)
    write(OUT / 'status.json', report)
if __name__ == '__main__':
    try:
        main()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        write(OUT / 'failure.json', {'error': traceback.format_exc()})
        raise
