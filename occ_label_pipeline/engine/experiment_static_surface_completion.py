"""Independent local static-class completion preview; never edits annotation input."""
from occ_label_pipeline.context import CFG, CONFIG_PATH, PROJECT_ROOT, ENGINE_ROOT, OUTPUT as CAPTURE_ROOT, DATA_ROOT, TRACKER_ROOT, NFRAMES
import json, hashlib
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay, cKDTree, QhullError
from experiment_tracked_surface_reconstruction import sample_mesh
from complete_tracked_points import unique, keys
ROOT = PROJECT_ROOT
BASE = ROOT / str(CAPTURE_ROOT / 'occ_pipeline_v3_0')
SOURCE = BASE / 'static_world_manual_v3/manual_knn_labels.npz'
OUT = BASE / 'ground_classes_completion_pilot'
NAMES = {11: 'driveable_surface', 12: 'other_flat', 13: 'sidewalk', 14: 'terrain'}

def ground_fill(points, cls):
    xykeys = np.floor(points[:, :2] / 0.1).astype(int)
    order = np.lexsort((xykeys[:, 1], xykeys[:, 0]))
    xykeys = xykeys[order]
    p = points[order]
    starts = np.r_[0, np.flatnonzero(np.any(xykeys[1:] != xykeys[:-1], axis=1)) + 1]
    ends = np.r_[starts[1:], len(p)]
    support = []
    for a, b in zip(starts, ends):
        if np.ptp(p[a:b, 2]) <= 0.3:
            support.append([*p[a:b, :2].mean(0), np.median(p[a:b, 2])])
    support = np.array(support)
    if len(support) < 4:
        return (np.empty((0, 3)), 0)
    tri = Delaunay(support[:, :2])
    lo = support[:, :2].min(0)
    hi = support[:, :2].max(0)
    x = np.arange(np.floor(lo[0] / 0.1), np.ceil(hi[0] / 0.1) + 1) * 0.1 + 0.05
    y = np.arange(np.floor(lo[1] / 0.1), np.ceil(hi[1] / 0.1) + 1) * 0.1 + 0.05
    xx, yy = np.meshgrid(x, y)
    xy = np.column_stack([xx.ravel(), yy.ravel()])
    idx = tri.find_simplex(xy)
    inside = idx >= 0
    xy = xy[inside]
    idx = idx[inside]
    v = support[tri.simplices[idx]]
    edges = np.maximum.reduce([np.linalg.norm(v[:, 0, :2] - v[:, 1, :2], axis=1), np.linalg.norm(v[:, 1, :2] - v[:, 2, :2], axis=1), np.linalg.norm(v[:, 0, :2] - v[:, 2, :2], axis=1)])
    norm = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    angle = np.arctan2(np.linalg.norm(norm[:, :2], axis=1), np.abs(norm[:, 2]))
    good = (edges <= 0.6) & (angle <= np.deg2rad(45 if cls == 14 else 25))
    xy = xy[good]
    idx = idx[good]
    weights = np.einsum('nij,nj->ni', tri.transform[idx, :2], xy - tri.transform[idx, 2])
    weights = np.c_[weights, 1 - weights.sum(1)]
    z = (weights * support[tri.simplices[idx], 2]).sum(1)
    result = np.c_[xy, z]
    result = result[cKDTree(points).query(result)[0] <= 0.3]
    return (result, int(good.sum()))

def main():
    if OUT.exists():
        raise ValueError('Fresh output required')
    OUT.mkdir(parents=True)
    revision = SOURCE.stat().st_mtime_ns
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    with np.load(SOURCE) as d:
        xyz = d['xyz']
        sem = d['semantics']
        manual = d['manual_override_mask'] if 'manual_override_mask' in d else None
    if SOURCE.stat().st_mtime_ns != revision:
        raise RuntimeError('Annotation changed during read; retry snapshot')
    results = []
    for cls, name in NAMES.items():
        ids = np.flatnonzero(sem == cls)
        if len(ids) < 30:
            results.append(dict(class_id=cls, class_name=name, status='skipped_sparse', available_points=len(ids), added_points=0))
            continue
        ground = cls in [11, 12, 13, 14]
        size = 8.0 if ground else 4.0
        tile = np.floor(xyz[ids, :2 if ground else 3] / size).astype(int)
        cells, counts = np.unique(tile, axis=0, return_counts=True)
        cell = cells[counts.argmax()]
        selected = ids[(tile == cell).all(1)]
        points = xyz[selected]
        center = (points.min(0) + points.max(0)) / 2
        local = points - center
        dims = points.max(0) - points.min(0) + 0.002
        record = dict(class_id=cls, class_name=name, available_points=len(ids), input_points=len(points), patch_size_m=size, method='local_ground_triangulation' if ground else 'ball_pivoting')
        try:
            if ground:
                new, faces = ground_fill(points, cls)
            else:
                cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(local))
                cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.4, max_nn=30))
                cloud.orient_normals_consistent_tangent_plane(min(10, len(local) - 1))
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(cloud, o3d.utility.DoubleVector([0.15, 0.25, 0.4]))
                new, faces = sample_mesh(mesh, local, dims, 1.0)
                new += center
            if len(new):
                new = new.astype(np.float32)
                new = new[unique(new)]
                lo = points.min(0) - 0.4
                hi = points.max(0) + 0.4
                near = xyz[(xyz >= lo).all(1) & (xyz <= hi).all(1)]
                existing = set(map(tuple, keys(near)))
                new = new[[tuple(k) not in existing for k in keys(new)]]
            dest = OUT / name
            dest.mkdir()
            np.savez_compressed(dest / 'comparison.npz', original=points, added=new.astype(np.float32), semantics=np.full(len(new), cls, np.int16))
            record.update(status='complete', added_points=len(new), completed_points=len(points) + len(new), triangles_used=faces)
        except (RuntimeError, QhullError, ValueError) as error:
            record.update(status='failed', added_points=0, error=str(error))
        results.append(record)
        print(name, record.get('input_points'), record['added_points'], flush=True)
    report = {'stage': 'complete', 'source': str(SOURCE), 'source_sha256': digest, 'source_revision': revision, 'source_modified': False, 'scope': 'One dense representative spatial patch per available static class; not full-map totals', 'grid_m': 0.1, 'interior_fill': False, 'results': results, 'original_points': sum((r.get('input_points', 0) for r in results)), 'added_points': sum((r['added_points'] for r in results))}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
if __name__ == '__main__':
    main()
