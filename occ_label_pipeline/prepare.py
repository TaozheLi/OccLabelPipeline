"""Create a measured-only standardized capture and estimate LiDAR odometry."""
import csv,json,time
from pathlib import Path
import numpy as np
CAMERAS={'FL':'front left','F':'front mid','FR':'front right','BL':'back left','B':'back','BR':'back right'}

def rotate_xyz(xyz,yaw_deg):
    theta=np.deg2rad(yaw_deg);c,s=np.cos(theta),np.sin(theta)
    rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    return (xyz@rotation.T).astype(np.float32)

def write_pcd(path,xyz,strength,intensity):
    arr=np.empty(len(xyz),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('reflectivity','<f4'),('intensity','<f4')])
    for i,k in enumerate('xyz'):arr[k]=xyz[:,i]
    arr['reflectivity']=strength;arr['intensity']=intensity
    header=f'VERSION .7\nFIELDS x y z reflectivity intensity\nSIZE 4 4 4 4 4\nTYPE F F F F F\nCOUNT 1 1 1 1 1\nWIDTH {len(xyz)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(xyz)}\nDATA binary\n'
    with path.open('wb') as f:f.write(header.encode());f.write(arr.tobytes())

def main():
    from .context import CFG,OUTPUT,DATA_ROOT
    from pcd_to_2d import read_point_cloud
    raw=Path(CFG['data_root']);base=OUTPUT/'occ_pipeline_v3_0';DATA_ROOT.mkdir(parents=True,exist_ok=True)
    started=time.time();cameras={};records=[]
    def status(stage,**kw):(base/'preparation_status.json').write_text(json.dumps({'stage':stage,'elapsed_s':time.time()-started,**kw},indent=2))
    if (raw/'lidar.csv').exists():
        source=list(csv.DictReader((raw/'lidar.csv').open()));kind='npz_csv'
        if (raw/'cameras.csv').exists():
            images=list(csv.DictReader((raw/'cameras.csv').open()))
            for key,role in CAMERAS.items():
                rows=sorted([r for r in images if r['camera']==role],key=lambda r:int(r['wall_time_ns']))
                if rows:cameras[key]=(np.array([int(r['wall_time_ns']) for r in rows],np.int64),rows)
    else:
        source=sorted(p for p in raw.iterdir() if p.is_dir() and (p/'lidar.pcd').exists());kind='frame_pcd'
    if len(source)<2:raise ValueError('At least two LiDAR sweeps required')
    stamps=[]
    for i,item in enumerate(source,1):
        name=f'{i:06d}';dest=DATA_ROOT/name;dest.mkdir(exist_ok=True);matches={}
        if kind=='npz_csv':
            stamp=int(item['ros_stamp_ns'])/1e9;src=raw/item['file']
            with np.load(src) as d:
                x=d['xyz'];intensity=d[CFG.get('intensity_field','intensity')]
            for key,(times,rows) in cameras.items():
                t=round(stamp*1e9);j=int(np.searchsorted(times,t));j=min([k for k in [j-1,j] if 0<=k<len(times)],key=lambda k:abs(int(times[k])-t))
                delta=(int(times[j])-t)/1e9
                if abs(delta)<=CFG.get('camera_sync_tolerance_s',.1):matches[key]=(raw/rows[j]['file'],delta)
        else:
            src=item/'lidar.pcd';meta=json.loads((item/'meta.json').read_text());stamp=float(meta['lidar_time'])
            fields,_=read_point_cloud(src);x=np.column_stack([fields[k] for k in 'xyz']);intensity=fields[CFG.get('intensity_field','reflectivity')]
            for key in CAMERAS:
                p=item/(key+'.png')
                if p.exists():matches[key]=(p,0.)
        stamps.append(stamp)
        if i>1 and stamp<=stamps[-2]:raise ValueError('LiDAR timestamps must increase strictly')
        xyz=rotate_xyz(x,CFG.get('yaw_to_vehicle_deg',0))
        strength=np.clip(intensity,0,255) if CFG['intensity_mode']=='intensity_proxy_clip255' else intensity
        if not (dest/'lidar.pcd').exists():write_pcd(dest/'lidar.pcd',xyz,strength,intensity)
        for key,(image,dt) in matches.items():
            link=dest/(key+'.png')
            if not link.exists():link.symlink_to(image.resolve())
        record={'frame':name,'lidar_time':stamp,'source':str(src),'camera_matches':{k:{'source':str(v[0]),'offset_s':v[1]} for k,v in matches.items()}}
        (dest/'meta.json').write_text(json.dumps(record));records.append(record)
        if i%20==0:status('adapting',completed=i,total=len(source))
    manifest={'frames':len(records),'data_format':kind,'yaw_to_vehicle_deg':CFG.get('yaw_to_vehicle_deg',0),'records':records,'intensity_mode':CFG['intensity_mode'],'camera_mask_available':False,'deskew':False,'point_timing_available':False,'manual_labels_transferred':False}
    (base/'capture_manifest.json').write_text(json.dumps(manifest,indent=2))
    from kiss_icp.config import load_config
    from kiss_icp.kiss_icp import KissICP
    cfg=load_config(None);cfg.data.max_range=80;cfg.data.min_range=2;cfg.data.deskew=False;cfg.mapping.voxel_size=.5;odometry=KissICP(cfg);poses=[]
    for i,r in enumerate(records,1):
        fields,_=read_point_cloud(DATA_ROOT/r['frame']/'lidar.pcd');x=np.column_stack([fields[k] for k in 'xyz']).astype(float);x=x[np.isfinite(x).all(1)]
        odometry.register_frame(x,np.empty(0));poses.append({'frame':r['frame'],'lidar_time':r['lidar_time'],'T_world_from_lidar':np.array(odometry.last_pose).tolist()})
        if i%10==0:status('kiss_icp',completed=i,total=len(records))
    slam=base/'slam';slam.mkdir(exist_ok=True);(slam/'poses.json').write_text(json.dumps(poses));status('complete',frames=len(records))
if __name__=='__main__':main()
