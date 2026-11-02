"""Portable configuration resolution and validation."""
import json,os
from pathlib import Path

def load_config(path):
    path=Path(path).expanduser().resolve();cfg=json.loads(path.read_text())
    def resolve(value):
        p=Path(os.path.expandvars(value)).expanduser()
        return str((path.parent/p).resolve() if not p.is_absolute() else p.resolve())
    for key in ['data_root','output_root','tracking_root','manual_static_world']:
        if cfg.get(key):cfg[key]=resolve(cfg[key])
    for section,keys in [('lidar_segmentation',['root','model_config','checkpoint']),('detector_3d',['root','model_config','checkpoint'])]:
        for key in keys:cfg[section][key]=resolve(cfg[section][key])
    if Path(cfg['data_root'])==Path(cfg['output_root']):raise ValueError('Output must differ from raw data directory')
    if not cfg.get('camera_mask',False) is False:raise ValueError('This LiDAR-only release does not calculate camera masks')
    if cfg['lidar_height_m']<=0:raise ValueError('lidar_height_m must be positive')
    if cfg['voxel_size_m']<=0:raise ValueError('voxel_size_m must be positive')
    if len(cfg['bounds'])!=6 or any(cfg['bounds'][i]>=cfg['bounds'][i+3] for i in range(3)):raise ValueError('Invalid bounds')
    if cfg['intensity_mode'] not in ['reflectivity','intensity_proxy_clip255']:raise ValueError('Explicit intensity_mode required')
    return cfg,path
