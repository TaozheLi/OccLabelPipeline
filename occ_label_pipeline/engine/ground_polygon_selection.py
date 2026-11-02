"""Screen polygon selection and local lower-surface filtering, independent of GUI."""
import numpy as np
from scipy.spatial import cKDTree
from matplotlib.path import Path

def polygon_indices(xyz,matrix,width,height,vertices):
    poly=np.asarray(vertices,float)
    if poly.ndim!=2 or poly.shape[1]!=2 or len(poly)<3:raise ValueError('Polygon needs at least three vertices')
    # Reject self-crossing polygons instead of silently selecting an ambiguous area.
    from shapely.geometry import Polygon
    shape=Polygon(poly)
    if not shape.is_valid or shape.area<9:raise ValueError('Polygon crosses itself or is too small')
    path=Path(np.vstack([poly,poly[0]]));lo=poly.min(0);hi=poly.max(0);out=[]
    for start in range(0,len(xyz),250000):
        clip=xyz[start:start+250000]@matrix[:,:3].T+matrix[:,3];front=clip[:,3]>1e-8
        ndc=clip[:,:3]/np.where(front,clip[:,3],1)[:,None]
        uv=np.column_stack([(ndc[:,0]+1)*width/2,(1-ndc[:,1])*height/2])
        valid=front&(np.abs(ndc[:,2])<=1)&(uv>=lo).all(1)&(uv<=hi).all(1)
        ids=np.flatnonzero(valid)
        out.append(start+ids[path.contains_points(uv[ids],radius=1e-6)])
    return np.concatenate(out).astype(np.int64)

def ground_filter(xyz,ids,tolerance=.35,cell_size=1.,support_radius=2., return_offsets=False):
    """Keep points near a local lower envelope, not a global constant Z plane.

    Cell 10th-percentile height with neighbor median suppresses sparse outliers.
    No guarantee on multiple road decks or cells containing only elevated returns.
    """
    if not np.isfinite(tolerance) or tolerance<=0:raise ValueError('Height tolerance must be positive')
    ids=np.asarray(ids,np.int64)
    if not len(ids):return (ids,np.empty(0)) if return_offsets else ids
    points=xyz[ids];keys=np.floor(points[:,:2]/cell_size).astype(np.int64)
    cells,inv=np.unique(keys,axis=0,return_inverse=True)
    order=np.argsort(inv,kind='stable');cuts=np.r_[0,np.cumsum(np.bincount(inv))]
    low=np.array([np.quantile(points[order[cuts[i]:cuts[i+1]],2],.1) for i in range(len(cells))])
    tree=cKDTree(cells*cell_size+cell_size/2)
    near=tree.query_ball_point(cells*cell_size+cell_size/2,support_radius)
    reference=np.array([np.quantile(low[n],.35) for n in near])
    delta=points[:,2]-reference[inv]
    if return_offsets:return ids,delta
    return ids[(delta>=-tolerance)&(delta<=tolerance)]
