"""Command-line entry points. Import GPU engines only in worker processes."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from .config import load_config

def main():
    p=argparse.ArgumentParser(prog='occ-label',description='LiDAR-first semantic occupancy labels')
    p.add_argument('command',choices=['validate','doctor','run','status','view','annotate'])
    p.add_argument('--config',type=Path,required=True)
    a=p.parse_args();cfg,config_path=load_config(a.config)
    package=Path(__file__).resolve().parent;env=dict(os.environ,OCC_PIPELINE_CONFIG=str(config_path))
    env['PYTHONPATH']=os.pathsep.join([str(package.parent),str(package/'engine'),env.get('PYTHONPATH','')])
    output=Path(cfg['output_root']);base=output/'occ_pipeline_v3_0'
    if a.command=='validate':
        missing=[]
        for path in [cfg['data_root'],cfg['tracking_root'],*(cfg[s][k] for s in ['lidar_segmentation','detector_3d'] for k in ['root','model_config','checkpoint'])]:
            if not Path(path).exists():missing.append(path)
        if missing:p.error('Missing inputs/dependencies:\n'+'\n'.join(missing))
        if not (Path(cfg['tracking_root'])/'config/nuscenes.yaml').exists():p.error('MCTrack config/nuscenes.yaml missing')
        print('Configuration and input paths valid. CUDA extensions not tested; run doctor.');return
    if a.command=='status':
        for name in ['status.json','preparation_status.json','segmentation_status.json','detection_status.json']:
            path=base/name
            if path.exists():print(name,path.read_text())
        for path in base.glob('**/failure.json'):print('FAILURE',path,path.read_text())
        return
    if a.command=='doctor':
        code="import sys,torch; print('Python',sys.version); print('Torch',torch.__version__); assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0)); import kiss_icp,open3d,numba,scipy; from occ_label_pipeline.context import CFG; sys.path.insert(0,CFG['lidar_segmentation']['root']); import pointcept; sys.path.insert(0,CFG['detector_3d']['root']); import pcdet; from pcdet.ops.iou3d_nms import iou3d_nms_cuda; from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_stack_cuda; print('Core imports and detector CUDA extension imports passed; validate model-specific forward separately.')"
        subprocess.run([sys.executable,'-c',code],env=env,check=True);return
    if a.command=='run':
        subprocess.run([sys.executable,'-m','occ_label_pipeline.runner'],env=env,check=True);return
    if a.command=='annotate':
        source=base/'dynamic_last_v1/background_world/static_world.npz'
        if not source.exists():p.error('Static world not ready; run with manual_review=true first')
        dest=Path(cfg['manual_static_world']) if cfg.get('manual_static_world') else base/'dynamic_last_v1/manual/manual_knn_labels.npz'
        subprocess.run([sys.executable,str(package/'engine/annotate_cloud_knn.py'),str(source),'--output',str(dest),'--initialize-from-input','--show-label-colors'],env=env,check=True)
        return
    occ=base/'dynamic_last_v1/occ_full_interior_v1'
    if not (base/'status.json').exists() or json.loads((base/'status.json').read_text())['stage']!='complete':p.error('View requires the completed capture')
    subprocess.run([sys.executable,str(package/'engine/view_occ_images_topview.py'),'--data-root',str(base/'input_frames'),'--occ-root',str(occ),'--fps','10','--autoplay'],env=env,check=True)
if __name__=='__main__':main()
