"""nuScenes TransFusion-L adapter with pose-aligned history and bounded 3D point assignment."""
from occ_label_pipeline.context import CFG, CONFIG_PATH, PROJECT_ROOT, ENGINE_ROOT, OUTPUT as CAPTURE_ROOT, DATA_ROOT, TRACKER_ROOT, NFRAMES
from copy import deepcopy
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from lidar_segmentation import sha256
from pcd_to_2d import read_point_cloud
CLASS_TO_OCC = {'car': 4, 'truck': 10, 'construction_vehicle': 5, 'bus': 3, 'trailer': 9, 'barrier': 1, 'motorcycle': 6, 'bicycle': 2, 'pedestrian': 7, 'traffic_cone': 8}

def velocity_fields(box, inverse):
    if len(box) < 9:
        return {}
    velocity = np.asarray(box[7:9], float) @ np.asarray(inverse)[:2, :2].T
    if not np.isfinite(velocity).all():
        return {}
    return {'velocity_lidar_mps': velocity.tolist(), 'velocity_lidar_kmh': (velocity * 3.6).tolist(), 'speed_kmh': float(np.linalg.norm(velocity) * 3.6), 'velocity_frame': 'current_raw_lidar'}

def assign_3d_boxes(xyz, labels, confidence, detections, minimum_points=5):
    labels, confidence = (labels.copy(), confidence.copy())
    claimed = np.zeros(len(xyz), bool)
    accepted = []
    for box in sorted(detections, key=lambda item: item['confidence'], reverse=True):
        center = np.asarray(box['center'], float)
        dimensions = np.asarray(box['dimensions'], float)
        if center.shape != (3,) or dimensions.shape != (3,) or (not np.isfinite(center).all()) or (not np.isfinite(dimensions).all()) or np.any(dimensions <= 0):
            raise ValueError('Expected finite center and positive box dimensions')
        delta = xyz - center
        c, s = (np.cos(box['yaw_rad']), np.sin(box['yaw_rad']))
        local = np.column_stack((delta[:, 0] * c + delta[:, 1] * s, -delta[:, 0] * s + delta[:, 1] * c, delta[:, 2]))
        inside = (np.abs(local) <= dimensions / 2).all(1) & ~claimed
        ids = np.flatnonzero(inside)
        if len(ids) < minimum_points:
            continue
        labels[ids] = box['occ3d_class_id']
        confidence[ids] = np.maximum(confidence[ids], box['confidence'])
        claimed[ids] = True
        accepted.append({**box, 'assigned_points': len(ids)})
    return (labels, confidence, claimed, accepted)

class TransFusionDetector:

    def __init__(self, project, config, slam_poses, data_root, device='cuda'):
        import torch
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('TransFusion requires CUDA')
        root = (project / config['root']).resolve()
        sys.path.insert(0, str(root))
        from easydict import EasyDict
        from pcdet.config import cfg_from_yaml_file
        from pcdet.datasets.dataset import DatasetTemplate
        from pcdet.models import build_network, load_data_to_gpu
        from pcdet.utils.common_utils import create_logger
        cfg = EasyDict()
        old = os.getcwd()
        try:
            os.chdir(root / 'tools')
            cfg_from_yaml_file(str((project / config['model_config']).resolve()), cfg)
        finally:
            os.chdir(old)
        for processor in cfg.DATA_CONFIG.DATA_PROCESSOR:
            if processor.NAME == 'shuffle_points':
                processor.SHUFFLE_ENABLED['test'] = False
        self.dataset = DatasetTemplate(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False, root_path=data_root, logger=create_logger())
        self.model = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), self.dataset)
        checkpoint = project / config['checkpoint']
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)['model_state']
        self.model._load_state_dict(state, strict=True)
        self.model = self.model.cuda().eval()
        self.torch = torch
        self.load_to_gpu = load_data_to_gpu
        self.names = list(cfg.CLASS_NAMES)
        if set(self.names) != set(CLASS_TO_OCC):
            raise ValueError('Expected nuScenes detection classes')
        self.data_root = data_root
        self.poses = slam_poses
        self.frames = sorted((p.name for p in data_root.iterdir() if (p / 'lidar.pcd').is_file()))
        self.indices = {name: i for i, name in enumerate(self.frames)}
        self.sweeps = int(config.get('sweeps', 10))
        self.max_age = float(config.get('maximum_sweep_age_s', 0.5))
        self.threshold = float(config.get('confidence', 0.3))
        self.minimum_points = int(config.get('minimum_points', 5))
        self.feature = config.get('intensity_field', 'reflectivity')
        self.scale = float(config.get('intensity_scale', 1.0))
        if self.sweeps < 1 or self.max_age <= 0 or self.scale <= 0 or (not 0 < self.threshold <= 1):
            raise ValueError('Invalid 3D detector parameters')
        self.T = np.asarray(config.get('T_model_from_lidar', np.eye(4)), float)
        if self.T.shape != (4, 4) or not np.isfinite(self.T).all() or (not np.allclose(self.T[3], [0, 0, 0, 1])) or (not np.allclose(self.T[:3, :3].T @ self.T[:3, :3], np.eye(3))) or (not np.allclose(self.T[2, :3], [0, 0, 1])):
            raise ValueError('T_model_from_lidar must be a yaw-only rigid transform')
        self.inverse = np.linalg.inv(self.T)
        self.rotation = np.arctan2(self.T[1, 0], self.T[0, 0])
        self.metadata = {'backend': 'transfusion_lidar', 'checkpoint_sha256': sha256(checkpoint), 'model_config_sha256': sha256(project / config['model_config']), 'intensity_field': self.feature, 'intensity_scale': self.scale, 'requested_sweeps': self.sweeps, 'maximum_sweep_age_s': self.max_age, 'T_model_from_lidar': self.T.tolist(), 'confidence_threshold': self.threshold}

    @lru_cache(maxsize=10)
    def cloud(self, name):
        frame = self.data_root / name
        fields, _ = read_point_cloud(frame / 'lidar.pcd')
        xyz = np.column_stack([fields[k] for k in ['x', 'y', 'z']])
        feature = np.asarray(fields[self.feature], np.float32)
        valid = np.isfinite(xyz).all(1) & np.isfinite(feature)
        stamp = float(json.loads((frame / 'meta.json').read_text())['lidar_time'])
        return (xyz[valid], feature[valid] / self.scale, stamp)

    def sweep_points(self, name):
        index = self.indices[name]
        current_xyz, current_strength, current_time = self.cloud(name)
        chunks = [np.column_stack((current_xyz, current_strength, np.zeros(len(current_xyz))))]
        sources = [name]
        for source_index in range(index - 1, max(-1, index - self.sweeps), -1):
            source_name = self.frames[source_index]
            source_xyz, strength, stamp = self.cloud(source_name)
            age = current_time - stamp
            if age < 0:
                raise ValueError('Nonmonotonic capture timestamps')
            if age > self.max_age:
                break
            if name not in self.poses or source_name not in self.poses:
                raise ValueError('Pose-aligned multi-sweep detection requires SLAM poses')
            keep = ~((np.abs(source_xyz[:, 0]) < 1) & (np.abs(source_xyz[:, 1]) < 1))
            T = np.linalg.inv(self.poses[name]) @ self.poses[source_name]
            points = source_xyz[keep] @ T[:3, :3].T + T[:3, 3]
            chunks.append(np.column_stack((points, strength[keep], np.full(len(points), age))))
            sources.append(source_name)
        points = np.concatenate(chunks).astype(np.float32)
        points[:, :3] = points[:, :3] @ self.T[:3, :3].T + self.T[:3, 3]
        return (points, sources)

    def __call__(self, frame, xyz, intensity):
        name = frame.name
        points, sources = self.sweep_points(name)
        batch = self.dataset.collate_batch([self.dataset.prepare_data({'points': points, 'frame_id': name})])
        self.load_to_gpu(batch)
        self.torch.cuda.synchronize()
        start = time.perf_counter()
        with self.torch.inference_mode():
            pred, _ = self.model(batch)
        self.torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        boxes = pred[0]['pred_boxes'].cpu().numpy()
        scores = pred[0]['pred_scores'].cpu().numpy()
        labels = pred[0]['pred_labels'].cpu().numpy()
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            if score < self.threshold:
                continue
            cls = self.names[int(label) - 1]
            center = box[:3] @ self.inverse[:3, :3].T + self.inverse[:3, 3]
            detections.append({'model_class': cls, 'occ3d_class_id': CLASS_TO_OCC[cls], 'confidence': float(score), 'center': center.tolist(), 'dimensions': box[3:6].tolist(), 'yaw_rad': float(box[6] - self.rotation), **velocity_fields(box, self.inverse)})
        return (detections, {'model': self.metadata, 'inference_seconds': elapsed, 'sweep_frames': sources, 'input_points': len(points), 'raw_detections': len(boxes), 'accepted_detections': len(detections), 'gpu_peak_allocated_mb': self.torch.cuda.max_memory_allocated() / 1024 ** 2})
