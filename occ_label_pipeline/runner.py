"""Ordered, capture-independent pipeline; no viewer starts automatically."""
import json,sys,subprocess,time,hashlib,traceback,shutil
from pathlib import Path
from .context import CFG,CONFIG_PATH,ENGINE_ROOT,OUTPUT,DATA_ROOT
BASE=OUTPUT/'occ_pipeline_v3_0';BRANCH=BASE/'dynamic_last_v1'
def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def status(stage,**kw):write(BASE/'status.json',{'stage':stage,**kw})
def run(stage,args):
    status(stage)
    with (BASE/(stage+'.log')).open('a') as log:subprocess.run([sys.executable,*map(str,args)],stdout=log,stderr=subprocess.STDOUT,check=True)
def engine(stage,name,*args):run(stage,[ENGINE_ROOT/name,*args])
def complete(path):
    p=path/'report.json'
    return p.exists() and json.loads(p.read_text()).get('stage','complete')=='complete'
def main():
    import fcntl
    BASE.mkdir(parents=True,exist_ok=True)
    lock=(BASE/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Resume only with exactly the same resolved recipe and weight files.
    provenance={'config':CFG,'checkpoint_hashes':{k:hashlib.sha256(Path(CFG[k]['checkpoint']).read_bytes()).hexdigest() for k in ['lidar_segmentation','detector_3d']}}
    p=BASE/'run_provenance.json'
    if p.exists() and json.loads(p.read_text())!=provenance:raise ValueError('Recipe/checkpoint changed: use a fresh output directory')
    write(p,provenance);write(BASE/'config.json',CFG)
    if not (BASE/'slam/poses.json').exists():run('prepare_and_slam',['-m','occ_label_pipeline.prepare'])
    frames=json.loads((BASE/'capture_manifest.json').read_text())['frames']
    run('segmentation',['-m','occ_label_pipeline.predict','segmentation'])
    run('detection',['-m','occ_label_pipeline.predict','detection'])
    fusion=BASE/'tracked_completion_all_block100';bpa=BASE/'ball_pivoting_all_block100'
    if not complete(fusion):engine('tracking_and_temporal','run_tracked_completion_all.py')
    if not complete(bpa):engine('bpa','run_ball_pivoting_completion.py')
    BRANCH.mkdir(exist_ok=True);background=BRANCH/'background_world'
    if not complete(background):engine('static_world','fuse_completed_static_world.py','--minimum-confidence',CFG.get('static_confidence',.7),'--output',background,'--no-viewer','--defer-all-dynamic')
    source=background/'static_world.npz'
    manual=Path(CFG['manual_static_world']) if CFG.get('manual_static_world') else BRANCH/'manual/manual_knn_labels.npz'
    if CFG.get('manual_review') and not manual.exists():
        status('awaiting_manual_review',input=str(source),output=str(manual))
        print('Static world ready. Run occ-label annotate, save, then repeat occ-label run.',flush=True)
        return
    if CFG.get('manual_static_world') or CFG.get('manual_review'):
        from .manual import transfer
        source=BRANCH/'background_reviewed.npz'
        digest=hashlib.sha256(manual.read_bytes()).hexdigest();audit=BRANCH/'manual_source.json'
        if audit.exists() and json.loads(audit.read_text())['sha256']!=digest:raise ValueError('Manual annotations changed after transfer; use a fresh output')
        if not source.exists():transfer(background,manual,source)
        write(audit,{'sha256':digest,'source':str(manual)})
    tasks=[('knn','relabel_ground_knn_world.py',['--all-classes','--source',source,'--output',BRANCH/'knn']),('clean','clean_world_isolated.py',['--source',BRANCH/'knn/relabeled_world.npz','--output',BRANCH/'clean','--minimum-voxels','5']),('ground','complete_world_ground_map.py',['--source',BRANCH/'clean/cleaned_world.npz','--output',BRANCH/'ground']),('unknown_knn','fill_unknown_world_knn.py',['--source',BRANCH/'ground/completed_world.npz','--output',BRANCH/'unknown_knn'])]
    for stage,name,args in tasks:
        dest=BRANCH/stage
        if not complete(dest):
            if dest.exists():dest.rename(dest.with_name(stage+'_interrupted_'+str(time.time_ns())))
            engine(stage,name,*args)
    interior=BRANCH/'interior_full_v1';interior.mkdir(exist_ok=True)
    if not complete(interior):engine('vehicle_interior','complete_vehicle_interior.py','--output',interior,'--start','1','--count',frames,'--workers',CFG.get('cpu_workers',6))
    occ=BRANCH/'occ_full_interior_v1'
    if not complete(occ):engine('voxelize','voxelize_v3_merged.py','--world',BRANCH/'unknown_knn/completed_world.npz','--output',occ,'--workers',CFG.get('cpu_workers',6),'--dynamic-last','--interior-root',interior)
    r=json.loads((occ/'report.json').read_text());assert r['frames']==frames and r['dynamic_overwritten_voxels']==0
    assert len(list(occ.glob('*/labels.npz')))==frames
    status('complete',frames=frames,output=str(occ),dynamic_overwritten_voxels=0)
    print('Completed:',occ,flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        BASE.mkdir(parents=True,exist_ok=True);status('failed',error=traceback.format_exc());raise
