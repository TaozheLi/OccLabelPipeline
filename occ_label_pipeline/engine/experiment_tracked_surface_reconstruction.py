"""Compare local surface reconstruction on tracked, accumulated object clouds."""
from occ_label_pipeline.context import CFG, CONFIG_PATH, PROJECT_ROOT, ENGINE_ROOT, OUTPUT as CAPTURE_ROOT, DATA_ROOT, TRACKER_ROOT, NFRAMES
import json, math
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from complete_tracked_points import rot, keys, unique
ROOT = PROJECT_ROOT
BASE = ROOT / str(CAPTURE_ROOT / 'occ_pipeline_v3_0')
FUSED = BASE / 'tracked_surface_completion100_box_supported'
TRACK = BASE / 'mctrack_people_vehicle_speed_preview100'
OUT = BASE / 'surface_reconstruction_pilot'

def sample_mesh(mesh, base, dims, maxedge):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    parts = []
    kept = []
    tree = cKDTree(base)
    for face in triangles:
        a, b, c = vertices[face]
        edge = max(np.linalg.norm(a - b), np.linalg.norm(a - c), np.linalg.norm(b - c))
        if edge > maxedge:
            continue
        area = np.linalg.norm(np.cross(b - a, c - a)) * 0.5
        if area < 1e-08:
            continue
        steps = max(1, int(np.ceil(edge / 0.05)))
        uv = np.array([(i / steps, j / steps) for i in range(steps + 1) for j in range(steps + 1 - i)])
        pts = a + uv[:, 0, None] * (b - a) + uv[:, 1, None] * (c - a)
        pts = pts[(np.abs(pts) <= dims / 2).all(1)]
        if len(pts):
            parts.append(pts)
            kept.append(face)
    if not parts:
        return (np.empty((0, 3)), 0)
    pts = np.concatenate(parts)
    pts = pts[unique(pts)]
    distance = tree.query(pts, k=1)[0]
    pts = pts[distance <= 0.3]
    occupied = set(map(tuple, keys(base)))
    pts = pts[[tuple(k) not in occupied for k in keys(pts)]]
    return (pts, len(kept))

def main():
    if OUT.exists():
        raise ValueError('Fresh output required')
    OUT.mkdir(parents=True)
    rows = json.loads((FUSED / 'object_counts.json').read_text())
    selection = []
    for frame in ['000700', '000749', '000799']:
        for cls in ['car', 'pedestrian']:
            candidates = [r for r in rows if r['frame'] == frame and r['class_name'] == cls and (r['added_points'] > 0)]
            selection.extend(sorted(candidates, key=lambda r: r['current_semantic_points'] + r['added_points'], reverse=True)[:2])
    results = []
    for n, r in enumerate(selection, 1):
        frame = r['frame']
        b = next((b for b in json.loads((TRACK / frame / 'boxes.json').read_text()) if b['track_id'] == r['track_id']))
        c = np.array(b['center'])
        R = rot(b['yaw_rad'])
        dims = np.array(b['dimensions'])
        with np.load(ROOT / f'{CAPTURE_ROOT}/predictions/frames/{frame}/point_labels.npz') as p:
            local = (p['xyz'] - c) @ R
            mask = (np.abs(local) <= dims / 2).all(1) & ~(np.isin(p['semantics'], [11, 12, 13, 14]) & (p['confidence'] >= 0.45))
            current = local[mask]
        with np.load(FUSED / frame / 'added_points.npz') as p:
            added = (p['xyz'][p['track_id'] == r['track_id']] - c) @ R
        base = np.concatenate([current, added])
        base = base[unique(base)]
        name = f"{frame}_{b['model_class']}_{b['track_id']}"
        dest = OUT / name
        dest.mkdir()
        np.savez_compressed(dest / 'input.npz', xyz_local=base, current_local=current, multiframe_added_local=added)
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(base))
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.4, max_nn=30))
        try:
            cloud.orient_normals_consistent_tangent_plane(min(10, len(base) - 1))
        except RuntimeError:
            pass
        for method in ['alpha_shape', 'ball_pivoting']:
            record = dict(object=name, frame=frame, track_id=b['track_id'], class_name=b['model_class'], current_points=len(current), multiframe_added=len(added), baseline_unique_cells=len(base), method=method)
            try:
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(cloud, 0.4 if b['model_class'] == 'car' else 0.25) if method == 'alpha_shape' else o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(cloud, o3d.utility.DoubleVector([0.15, 0.25, 0.4]))
                new, faces = sample_mesh(mesh, base, dims, 1.0 if b['model_class'] == 'car' else 0.5)
                np.savez_compressed(dest / f'{method}.npz', added_local=new, completed_local=np.concatenate([base, new]), added_lidar=new @ R.T + c)
                o3d.io.write_triangle_mesh(str(dest / f'{method}.ply'), mesh)
                record.update(added_points=len(new), completed_unique_cells=len(base) + len(new), used_triangles=faces, status='complete')
            except (RuntimeError, ValueError) as error:
                record.update(added_points=0, status='failed', error=str(error))
            results.append(record)
        (OUT / 'status.json').write_text(json.dumps({'stage': 'reconstructing', 'completed_objects': n, 'total_objects': len(selection)}))
        print('Object', n, '/', len(selection), flush=True)
    report = {'stage': 'complete', 'objects': len(selection), 'frames': ['000700', '000749', '000799'], 'sampling': 'deterministic triangle surface sampling, 0.1m output grid; reject long edges, outside box and >0.3m from input points', 'no_interior_fill': True, 'original_data_modified': False, 'baseline': 'current box points minus confident ground + existing motion-aware multiframe additions; 0.1m unique cells', 'count_scope': 'sum over selected object-frame examples, not all 100 frames; methods are alternatives, do not add their totals together', 'results': results, 'by_method': {m: {'baseline_points': sum((r['baseline_unique_cells'] for r in results if r['method'] == m)), 'added_points': sum((r['added_points'] for r in results if r['method'] == m)), 'by_class': {c: sum((r['added_points'] for r in results if r['method'] == m and r['class_name'] == c)) for c in ['car', 'pedestrian']}} for m in ['alpha_shape', 'ball_pivoting']}}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2))
    (OUT / 'status.json').write_text(json.dumps({'stage': 'complete', 'objects': len(selection)}))
    print(json.dumps(report['by_method'], indent=2))
if __name__ == '__main__':
    main()
