"""Per-point PTv3 inference and Occ3D camera/LiDAR evidence fusion."""
from __future__ import annotations
from occ_label_pipeline.context import CFG, CONFIG_PATH, PROJECT_ROOT, ENGINE_ROOT, OUTPUT as CAPTURE_ROOT, DATA_ROOT, TRACKER_ROOT, NFRAMES
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
NUSCENES_NAMES = ('barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation')

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def validate_evidence(values, count):
    values = np.asarray(values, np.float32)
    if values.shape != (count, 17) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Semantic evidence must be finite nonnegative [N,17]')
    if np.any(values.sum(1) > 1.0001):
        raise ValueError('Semantic evidence mass must not exceed one per point')
    return values

def lidar_only_evidence(probabilities, config):
    """Assign PTv3 semantic evidence directly, without image inference or fusion."""
    probabilities = validate_evidence(probabilities, len(probabilities))
    threshold = float(config.get('minimum_lidar_confidence', 0.45))
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Invalid LiDAR confidence threshold')
    confidence = probabilities.max(1)
    accepted = (probabilities.sum(1) > 0) & (confidence >= threshold)
    labels = np.full(len(probabilities), -1, dtype=np.int16)
    labels[accepted] = probabilities[accepted].argmax(1)
    confidence[~accepted] = 0
    return (labels, confidence, np.zeros(len(labels), dtype=bool), accepted, {'mode': 'lidar_only', 'image_inference_enabled': False, 'fusion_enabled': False, 'camera_supported_points': 0, 'lidar_supported_points': int(accepted.sum()), 'minimum_lidar_confidence': threshold})

def fuse_point_evidence(camera, lidar, config):
    """Average modality evidence; absent/rejected modalities have zero weight."""
    camera = validate_evidence(camera, len(camera))
    lidar = validate_evidence(lidar, len(camera))
    camera_weight = float(config.get('camera_weight', 1.0))
    lidar_weight = float(config.get('lidar_weight', 1.0))
    threshold = float(config.get('minimum_lidar_confidence', 0.45))
    fused_threshold = float(config.get('minimum_fused_confidence', 0.45))
    if not np.isfinite([camera_weight, lidar_weight, threshold, fused_threshold]).all() or camera_weight <= 0 or lidar_weight <= 0 or (not 0 <= threshold <= 1) or (not 0 <= fused_threshold <= 1):
        raise ValueError('Invalid LiDAR fusion weights/confidence thresholds')
    camera_seen = camera.sum(1) > 0
    lidar_seen = (lidar.sum(1) > 0) & (lidar.max(1) >= threshold)
    numerator = camera * (camera_weight * camera_seen[:, None])
    numerator += lidar * (lidar_weight * lidar_seen[:, None])
    denominator = camera_weight * camera_seen + lidar_weight * lidar_seen
    evidence = np.divide(numerator, denominator[:, None], out=np.zeros_like(numerator), where=denominator[:, None] > 0)
    confidence = evidence.max(1)
    accepted = (denominator > 0) & (confidence >= fused_threshold)
    labels = np.full(len(camera), -1, np.int16)
    labels[accepted] = evidence[accepted].argmax(1)
    confidence[~accepted] = 0
    return (labels, confidence, camera_seen, lidar_seen, {'camera_supported_points': int(camera_seen.sum()), 'lidar_supported_points': int(lidar_seen.sum()), 'both_supported_points': int((camera_seen & lidar_seen).sum()), 'lidar_only_points': int((~camera_seen & lidar_seen).sum()), 'disagreeing_points': int((camera_seen & lidar_seen & (camera.argmax(1) != lidar.argmax(1))).sum()), 'fused_labeled_points': int(accepted.sum()), 'camera_weight': camera_weight, 'lidar_weight': lidar_weight, 'minimum_lidar_confidence': threshold, 'minimum_fused_confidence': fused_threshold})

class PointceptSegmenter:
    """Load once; restore voxel predictions to every original finite LiDAR point."""

    def __init__(self, project: Path, config: dict, device='cuda'):
        import torch
        root = project / config['root']
        sys.path.insert(0, str(root.resolve()))
        from pointcept.utils.config import Config
        from pointcept.models import build_model
        from pointcept.datasets.transform import Compose
        self.torch = torch
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('PTv3 requires CUDA; run outside the Codex sandbox with the LiDAR environment')
        self.device = torch.device(device)
        config_path = project / config['model_config']
        checkpoint_path = project / config['checkpoint']
        cfg = Config.fromfile(str(config_path))
        if tuple(cfg.data.names) != NUSCENES_NAMES or cfg.model.num_classes != 16:
            raise ValueError('Expected the official nuScenes 16-class semantic checkpoint/config')
        cfg.model.backbone.enable_flash = False
        transforms = deepcopy(cfg.data.val.transform)
        grid_count = 0
        for spec in transforms:
            if spec['type'] == 'GridSample':
                spec['return_inverse'] = True
                grid_count += 1
            elif spec['type'] == 'Collect':
                spec['keys'] = tuple(spec['keys']) + ('inverse',)
        if grid_count != 1:
            raise ValueError('Expected one validation GridSample transform')
        self.transform = Compose(transforms)
        self.model = build_model(cfg.model)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state = {key.removeprefix('module.'): value for key, value in checkpoint['state_dict'].items()}
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.to(self.device).eval()
        self.field = str(config.get('intensity_field', 'reflectivity'))
        self.scale = float(config.get('intensity_scale', 255.0))
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('intensity_scale must be finite and positive')
        self.seed = int(config.get('sampling_seed', 0))
        self.metadata = {'backend': 'pointcept_ptv3', 'training_dataset': 'nuScenes', 'checkpoint': str(checkpoint_path), 'checkpoint_sha256': sha256(checkpoint_path), 'model_config_sha256': sha256(config_path), 'intensity_field': self.field, 'intensity_scale': self.scale, 'sampling_seed': self.seed, 'class_mapping': {str(i): i + 1 for i in range(16)}, 'inference': 'single validation voxel pass; inverse restoration; no test augmentation'}

    def __call__(self, xyz: np.ndarray, fields: dict) -> np.ndarray:
        if xyz.ndim != 2 or xyz.shape[1] != 3 or (not np.isfinite(xyz).all()):
            raise ValueError('Model coordinates must be finite [N,3] in raw LiDAR metres')
        if self.field not in fields:
            raise ValueError(f'Point cloud is missing configured feature: {self.field}')
        strength = np.asarray(fields[self.field], np.float32).reshape(-1, 1)
        if len(strength) != len(xyz) or not np.isfinite(strength).all():
            raise ValueError('Intensity features must match input point order and be finite')
        if not len(xyz):
            return np.zeros((0, 17), np.float32)
        strength = np.clip(strength / self.scale, 0, 1)
        data = {'coord': xyz.astype(np.float32), 'strength': strength, 'segment': np.full(len(xyz), -1, np.int64)}
        state = np.random.get_state()
        try:
            np.random.seed(self.seed)
            data = self.transform(data)
        finally:
            np.random.set_state(state)
        inverse = data.pop('inverse').long().to(self.device)
        data.pop('segment', None)
        data.pop('origin_segment', None)
        data = {key: value.to(self.device) if self.torch.is_tensor(value) else value for key, value in data.items()}
        with self.torch.inference_mode():
            logits = self.model(data)['seg_logits']
            probability = logits.float().softmax(-1)[inverse].cpu().numpy()
        if probability.shape != (len(xyz), 16):
            raise ValueError('Model inverse mapping did not restore the original point count')
        result = np.zeros((len(xyz), 17), np.float32)
        result[:, 1:] = probability
        return validate_evidence(result, len(xyz))

def save_point_predictions(path, xyz, original_indices, probabilities, fused_labels=None, fused_confidence=None, **sources):
    """Retain a stable link to PCD rows after rejecting nonfinite coordinates."""
    labels = probabilities.argmax(1).astype(np.int16)
    labels[probabilities.sum(1) == 0] = -1
    arrays = {'xyz': xyz, 'point_index': original_indices, 'semantics': labels, 'confidence': probabilities.max(1), 'probabilities': probabilities, **sources}
    if fused_labels is not None:
        arrays.update(fused_semantics=fused_labels, fused_confidence=fused_confidence)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
