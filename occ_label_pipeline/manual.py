"""Transfer only explicitly supplied, same-world static annotation edits."""
import json
import numpy as np

def transfer(background,annotation,output):
    from complete_world_ground_map import pack
    with np.load(background/'static_world.npz') as d:xyz=d['xyz'];sem=d['semantics'].copy()
    with np.load(annotation) as d:
        k=d['manual_override_mask']&(np.isin(d['semantics'],[0,1,8,11,12,13,14,15,16])|(d['semantics']==-1));mx=d['xyz'][k];ml=d['semantics'][k]
    with np.load(background/'excluded_dynamic_world_keys.npz') as d:excluded=d['keys']
    keys=pack(xyz);order=np.argsort(keys);keys=keys[order];mk=pack(mx);pos=np.searchsorted(keys,mk);safe=np.minimum(pos,len(keys)-1);found=(pos<len(keys))&(keys[safe]==mk)
    conflict=np.isin(mk,excluded);allowed=~(conflict&~np.isin(ml,[-1,11,12,13,14]));matched=found&allowed;manual=np.zeros(len(sem),bool)
    ids=order[safe[matched]];sem[ids]=ml[matched];manual[ids]=True;append=~found&~conflict
    np.savez_compressed(output,xyz=np.concatenate([xyz,mx[append]]),semantics=np.r_[sem,ml[append]],manual_override_mask=np.r_[manual,np.ones(int(append.sum()),bool)])
