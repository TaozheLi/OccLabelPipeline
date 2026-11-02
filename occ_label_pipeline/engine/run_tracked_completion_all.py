"""Run continuous MCTrack, then independent 100-frame motion-aware completion blocks."""
from occ_label_pipeline.context import CFG, CONFIG_PATH, PROJECT_ROOT, ENGINE_ROOT, OUTPUT as CAPTURE_ROOT, DATA_ROOT, TRACKER_ROOT, NFRAMES
import sys, json, math, time, subprocess, csv, traceback, hashlib, shutil
from pathlib import Path
import numpy as np
import yaml
ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT / str(TRACKER_ROOT)))
from tracker.base_tracker import Base3DTracker
from tracker.frame import Frame
from tracker.bbox import BBox
OUT = ROOT / str(CAPTURE_ROOT / 'occ_pipeline_v3_0/tracked_completion_all_block100')
DETS = ROOT / str(CAPTURE_ROOT / 'training_labels_v1/detections')
PRED = ROOT / str(CAPTURE_ROOT / 'predictions/frames')
DATA = Path(str(DATA_ROOT))
START = time.time()

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
    names = sorted((p.name for p in DATA.iterdir() if (p / 'lidar.pcd').exists()))
    assert len(names) == NFRAMES
    for name in names:
        assert (DETS / f'{name}.json').is_file(), name
        assert (PRED / name / 'point_labels.npz').is_file(), name
    records = [json.loads((DETS / f'{name}.json').read_text()) for name in names]
    times = np.array([r['timestamp'] for r in records])
    assert np.isfinite(times).all() and (np.diff(times) > 0).all()
    cfg = yaml.safe_load((TRACKER_ROOT / 'config/nuscenes.yaml').read_text())
    dt0 = float(np.median(np.diff(times)))
    cfg['FRAME_RATE'] = 1 / dt0
    cfg['DETECTOR'] = 'transfusion_lidar_custom'
    write(OUT / 'run_config.json', {'frames': len(names), 'block_size': 100, 'blocks': (NFRAMES + 99) // 100, 'tail_frames': NFRAMES % 100, 'predictions': str(PRED), 'detections': str(DETS), 'dedup_m': 0.1, 'moving_window': {'vehicle': 10, 'pedestrian': 3}, 'tracking_continuous': True, 'source_points_cross_block': False, 'class_source': 'detector; exclude confident PTv3 ground', 'original_data_modified': False, 'mc_track_commit': CFG.get('tracking_revision', 'user-provided')})
    snapshot = OUT / 'source'
    snapshot.mkdir(exist_ok=True)
    for name in ['run_tracked_completion_all.py', 'complete_tracked_points.py']:
        shutil.copy2(ENGINE_ROOT / name, snapshot / name)
    tracked = OUT / 'tracking'
    tracked.mkdir(exist_ok=True)
    if not (tracked / 'complete.json').exists():
        tracker = Base3DTracker(cfg)
        all_results = []
        write(tracked / 'tracker_config.json', cfg)
        for index, r in enumerate(records):
            T = np.array(r['T_world_from_lidar'])
            inv = np.linalg.inv(T)
            dt = times[index] - times[index - 1] if index else dt0
            tracker.cfg['FRAME_RATE'] = 1 / dt
            for tr in tracker.all_trajs.values():
                tr.frame_rate = 1 / dt
                for v in vars(tr).values():
                    if hasattr(v, 'dt'):
                        v.dt = dt
            frame = Frame(index, timestamp=r['timestamp'])
            for d in r['detections']:
                cls = d['model_class']
                if cls not in cfg['CATEGORY_LIST']:
                    continue
                c = T[:3, :3] @ np.array(d['center']) + T[:3, 3]
                direction = T[:3, :3] @ np.array([math.cos(d['yaw_rad']), math.sin(d['yaw_rad']), 0])
                yaw = math.atan2(direction[1], direction[0])
                v = T[:3, :3] @ np.r_[d.get('velocity_lidar_mps', [0, 0]), 0]
                b = BBox(index, dict(category=cls, detection_score=d['confidence'], lwh=d['dimensions'], global_xyz=c.tolist(), global_orientation=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)], global_yaw=yaw, global_velocity=v[:2].tolist(), global_acceleration=[0, 0], bbox_image={}))
                b.global_xyz_last = (c - np.r_[v[:2] * dt, 0]).tolist()
                b.global_xyz_lwh_yaw_last = b.global_xyz_last + b.lwh + [yaw]
                frame.bboxes.append(b)
            outputs = tracker.track_single_frame(frame)
            boxes = []
            for identity, b in outputs.items():
                a = np.array(b.global_xyz_lwh_yaw_fusion)
                c = inv[:3, :3] @ a[:3] + inv[:3, 3]
                direction = inv[:3, :3] @ np.array([math.cos(a[6]), math.sin(a[6]), 0])
                yaw = math.atan2(direction[1], direction[0])
                boxes.append(dict(track_id=int(identity), model_class=b.category, center=c.tolist(), dimensions=a[3:6].tolist(), yaw_rad=yaw, confidence=float(b.det_score), velocity_world_mps=np.asarray(b.global_velocity_fusion).tolist(), speed_kmh=float(np.linalg.norm(b.global_velocity_fusion) * 3.6)))
            record = {'frame': r['frame'], 'timestamp': r['timestamp'], 'boxes': boxes}
            all_results.append(record)
            write(tracked / f"{r['frame']}.json", record)
            if index % 10 == 0:
                status('tracking', completed=index + 1, total=len(names))
            if index % 100 == 0:
                print('Tracking', index + 1, '/', len(names), flush=True)
        write(tracked / 'tracks.json', all_results)
        write(tracked / 'complete.json', {'frames': len(names)})
    else:
        all_results = json.loads((tracked / 'tracks.json').read_text())
    totals = []
    frame_counts = []
    for block, start in enumerate(range(0, len(names), 100), 1):
        subset = all_results[start:start + 100]
        key = f"{subset[0]['frame']}_{subset[-1]['frame']}"
        source = OUT / 'block_inputs' / key
        source.mkdir(parents=True, exist_ok=True)
        write(source / 'tracks.json', subset)
        dest = OUT / 'blocks' / key
        if not (dest / 'report.json').exists():
            status('completion', block=block, total_blocks=(NFRAMES + 99) // 100, completed_frames=start, total_frames=len(names), active_block=key)
            if dest.exists():
                dest.rename(dest.with_name(key + '_interrupted_' + str(int(time.time()))))
            with (source / 'completion.log').open('w') as log:
                subprocess.run([sys.executable, str(ENGINE_ROOT / 'complete_tracked_points.py'), '--source', str(source), '--output', str(dest)], stdout=log, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
        report = json.loads((dest / 'report.json').read_text())
        totals.append(dict(block=key, **report))
        with (dest / 'frame_counts.csv').open() as f:
            frame_counts.extend(list(csv.DictReader(f)))
        write(OUT / 'block_reports.json', totals)
        print('Block', block, 'blocks', 'added', report['added_points'], flush=True)
        status('completion', completed_blocks=block, total_blocks=(NFRAMES + 99) // 100, completed_frames=start + len(subset), total_frames=len(names), added_points=sum((x['added_points'] for x in totals)))
    summary = {'stage': 'complete', 'frames': len(names), 'blocks': len(totals), 'original_points': sum((r['original_points'] for r in totals)), 'added_points': sum((r['added_points'] for r in totals)), 'added_stationary': sum((r['added_stationary'] for r in totals)), 'added_moving': sum((r['added_moving'] for r in totals)), 'by_class': {c: sum((r['by_class'][c] for r in totals)) for c in totals[0]['by_class']}, 'count_scope': 'summed voxel-deduplicated per-frame added points, not globally unique world points', 'elapsed_s': time.time() - START}
    summary['completed_points'] = summary['original_points'] + summary['added_points']
    with (OUT / 'frame_counts.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(frame_counts[0]))
        w.writeheader()
        w.writerows(frame_counts)
    write(OUT / 'report.json', summary)
    write(OUT / 'status.json', summary)
if __name__ == '__main__':
    try:
        main()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        write(OUT / 'failure.json', {'error': traceback.format_exc()})
        status('failed')
        raise
