"""Per-sweep segmentation or multi-sweep detection, with provenance-safe resume."""
import argparse,json,time
from pathlib import Path
import numpy as np
from .context import CFG,PROJECT_ROOT,OUTPUT,DATA_ROOT

def main():
    from pcd_to_2d import read_point_cloud
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['segmentation','detection']);a=p.parse_args()
    base=OUTPUT/'occ_pipeline_v3_0';poses=json.loads((base/'slam/poses.json').read_text());start=time.time()
    if a.stage=='segmentation':
        from lidar_segmentation import PointceptSegmenter
        model=PointceptSegmenter(PROJECT_ROOT,CFG['lidar_segmentation'],'cuda');out=OUTPUT/'predictions/frames'
    else:
        from detector_3d import TransFusionDetector
        model=TransFusionDetector(PROJECT_ROOT,CFG['detector_3d'],{p['frame']:np.array(p['T_world_from_lidar']) for p in poses},DATA_ROOT,'cuda');out=OUTPUT/'training_labels_v1/detections'
    out.mkdir(parents=True,exist_ok=True)
    for i,pose in enumerate(poses,1):
        name=pose['frame'];dest=out/name/'point_labels.npz' if a.stage=='segmentation' else out/(name+'.json')
        if not dest.exists():
            fields,_=read_point_cloud(DATA_ROOT/name/'lidar.pcd');x=np.column_stack([fields[k] for k in 'xyz']);finite=np.isfinite(x).all(1);fields={k:np.asarray(v)[finite] for k,v in fields.items()};x=x[finite]
            dest.parent.mkdir(parents=True,exist_ok=True)
            if a.stage=='segmentation':
                prob=model(x,fields);tmp=dest.with_name('point_labels.tmp.npz');np.savez_compressed(tmp,xyz=x.astype(np.float32),point_index=np.flatnonzero(finite),semantics=prob.argmax(1).astype(np.int16),confidence=prob.max(1));tmp.replace(dest)
            else:
                detections,metadata=model(DATA_ROOT/name,x,fields['reflectivity']);dest.write_text(json.dumps({'frame':name,'timestamp':pose['lidar_time'],'T_world_from_lidar':pose['T_world_from_lidar'],'detections':detections,'metadata':metadata}))
        if i%10==0 or i==len(poses):
            (base/(a.stage+'_status.json')).write_text(json.dumps({'stage':'complete' if i==len(poses) else a.stage,'completed':i,'total':len(poses),'elapsed_s':time.time()-start}));print(a.stage,i,'/',len(poses),flush=True)
if __name__=='__main__':main()
