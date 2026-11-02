#!/usr/bin/env python3
"""Manual seed + bounded KNN annotation in an Open3D desktop window."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree
from visualize_occ import CLASSES, PALETTE
from ground_polygon_selection import polygon_indices, ground_filter


def neighbors(tree, xyz, seed, k, radius):
    if k < 1 or radius <= 0: raise ValueError('K and radius must be positive')
    distance, ids = tree.query(xyz[seed], k=min(k,len(xyz)), distance_upper_bound=radius)
    ids=np.atleast_1d(ids);distance=np.atleast_1d(distance)
    return ids[np.isfinite(distance)&(ids<len(xyz))].astype(np.int64)


def rectangle_indices(xyz, view_projection, width, height, start, end, angle_deg=0.):
    """All points in a screen rectangle, including occluded points; chunked."""
    x0,x1=sorted([start[0],end[0]]);y0,y1=sorted([start[1],end[1]])
    cx=(x0+x1)/2;cy=(y0+y1)/2
    theta=np.deg2rad(angle_deg);cos=np.cos(theta);sin=np.sin(theta)
    result=[]
    for offset in range(0,len(xyz),250000):
        points=xyz[offset:offset+250000]
        clip=points@view_projection[:,:3].T+view_projection[:,3]
        front=clip[:,3]>1e-8
        ndc=clip[:,:3]/np.where(front,clip[:,3],1)[:,None]
        u=(ndc[:,0]+1)*width/2;v=(1-ndc[:,1])*height/2
        du=u-cx;dv=v-cy
        u=cx+cos*du+sin*dv;v=cy-sin*du+cos*dv
        keep=front&(np.abs(ndc[:,2])<=1)&(u>=x0)&(u<=x1)&(v>=y0)&(v<=y1)
        result.append(np.flatnonzero(keep)+offset)
    return np.concatenate(result).astype(np.int64)


def annotation_colors(base,labels,predicted=None,confidence=None,show_labels=True,
                      show_predictions=True,highlight_unknown=True,highlight_unconfirmed=False):
    colors=base.copy()
    prediction_valid=np.zeros(len(labels),bool)
    if predicted is not None and show_predictions:
        prediction_valid=(predicted>=0)&(predicted<17)&(confidence>=.45)
        colors[prediction_valid]=PALETTE[predicted[prediction_valid]]
    if show_labels:
        manual=labels>=0;colors[manual]=PALETTE[labels[manual]]
    if highlight_unknown:
        unknown=(labels<0)
        if not highlight_unconfirmed:unknown&=~prediction_valid
        colors[unknown]=[1.,1.,1.]
    return colors


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--predictions',type=Path)
    p.add_argument('--initialize-from-input',action='store_true',help='Edit input semantics as prelabels; erasing becomes unknown')
    p.add_argument('--max-display-points',type=int,default=None,help='Render a deterministic subset; selection and saved labels still use all points')
    p.add_argument('--k',type=int,default=300)
    p.add_argument('--radius',type=float,default=1.)
    p.add_argument('--show-label-colors',action='store_true')
    p.add_argument('--check',action='store_true')
    a=p.parse_args()
    if a.max_display_points is not None and a.max_display_points<1:p.error('max-display-points must be positive')
    if a.initialize_from_input and a.predictions:p.error('Use input prelabels or external predictions, not both')
    if a.k<1 or a.radius<=0: p.error('K and radius must be positive')
    source=a.input.resolve();output=(a.output or source.with_name('manual_knn_labels.npz')).resolve()
    if source==output:p.error('Output must differ from the input')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    with np.load(source) as d:
        xyz=d['xyz'].astype(np.float64)
        intensity=d['reflectivity'].astype(float) if 'reflectivity' in d else np.ones(len(xyz))
        initial=d['semantics'].astype(np.int16).copy() if a.initialize_from_input and 'semantics' in d else None
        inherited_manual=d['manual_override_mask'].astype(bool).copy() if a.initialize_from_input and 'manual_override_mask' in d else np.zeros(len(xyz),bool)
    if a.initialize_from_input and initial is None:p.error('Input semantics required for prelabel editing')
    if initial is not None and (initial.shape!=(len(xyz),) or np.any((initial < -1)|(initial>16))):p.error('Expected input labels -1..16')
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all() or not len(xyz):p.error('Expected nonempty finite XYZ coordinates')
    labels=initial.copy() if initial is not None else np.full(len(xyz),-1,np.int16)
    display_ids=np.arange(0,len(xyz),max(1,int(np.ceil(len(xyz)/a.max_display_points)))) if a.max_display_points else np.arange(len(xyz))
    predicted=None;prediction_confidence=None
    if a.predictions:
        with np.load(a.predictions) as d:
            if str(d['source_cloud_sha256'].item())!=digest: p.error('Prediction cloud does not match input')
            predicted=d['semantics'].copy();prediction_confidence=d['confidence'].copy()
            if len(predicted)!=len(xyz):p.error('Prediction count mismatch')
    saved_seeds=[]
    if output.exists():
        with np.load(output) as d:
            if str(d['source_sha256'].item())!=digest or len(d['semantics'])!=len(xyz):p.error('Saved labels do not match input cloud')
            labels[:]=d['semantics']
            if 'seed_records' in d: saved_seeds=json.loads(str(d['seed_records'].item()))
    revision=output.stat().st_mtime_ns if output.exists() else 0
    print('Building KNN index...',flush=True);tree=cKDTree(xyz)
    if a.check:
        print(f'{len(xyz):,} points; initial seed neighbors: {len(neighbors(tree,xyz,0,a.k,a.radius))}; output: {output}');return
    import open3d as o3d
    from open3d.visualization import gui,rendering
    app=gui.Application.instance;app.initialize();w=app.create_window('Manual LiDAR KNN annotation',1500,900)
    panel=gui.Vert(8,gui.Margins(12,12,12,12));scene=gui.SceneWidget();scene.scene=rendering.Open3DScene(w.renderer);scene.scene.set_background([0,0,0,1])
    panel.add_child(gui.Label('Shift + left click: select a seed'))
    panel.add_child(gui.Label('Select class, then Apply KNN.'))
    select=gui.Combobox()
    names=CLASSES[:17]+['Erase / unlabeled']
    for name in names:select.add_item(name)
    select.selected_index=11;panel.add_child(select)
    panel.add_child(gui.Label('K nearest points (including seed)'));kn=gui.NumberEdit(gui.NumberEdit.INT);kn.int_value=a.k;panel.add_child(kn)
    panel.add_child(gui.Label('Maximum distance (meters)'));radius=gui.NumberEdit(gui.NumberEdit.DOUBLE);radius.double_value=a.radius;panel.add_child(radius)
    overwrite=gui.Checkbox('Overwrite existing labels');panel.add_child(overwrite)
    show_labels=gui.Checkbox('Show label colors');show_labels.checked=a.show_label_colors;panel.add_child(show_labels)
    show_predictions=gui.Checkbox('Show model predictions (not confirmed)');show_predictions.checked=predicted is not None;panel.add_child(show_predictions)
    panel.add_child(gui.Label('Input labels are prelabels; saved corrections are tracked.' if initial is not None else 'Manual labels override prediction colors.'))
    if a.max_display_points:panel.add_child(gui.Label(f'Displaying {len(display_ids):,} / {len(xyz):,} points; edits use ALL points.'))
    highlight_unknown=gui.Checkbox('Show unlabeled points in white');highlight_unknown.checked=True;panel.add_child(highlight_unknown)
    highlight_unconfirmed=gui.Checkbox('Also whiten unconfirmed predictions');panel.add_child(highlight_unconfirmed)
    top_mode=gui.Checkbox('Top view');panel.add_child(top_mode)
    polygon_mode=gui.Checkbox('Ground polygon mode');panel.add_child(polygon_mode)
    panel.add_child(gui.Label('Wheel to zoom; click polygon vertices.'))
    z_filter=gui.Checkbox('Filter overlapping layers by local Z');z_filter.checked=True;panel.add_child(z_filter)
    panel.add_child(gui.Label('Ground height tolerance (+/- meters)'))
    z_tolerance=gui.NumberEdit(gui.NumberEdit.DOUBLE);z_tolerance.double_value=.5;panel.add_child(z_tolerance)
    region_mode=gui.Checkbox('Rectangle selection mode');panel.add_child(region_mode)
    panel.add_child(gui.Label('Rectangle mode locks the camera.\nDrag corners to resize; hidden points included.'))
    lock_view=gui.Checkbox('Lock camera view');panel.add_child(lock_view)
    panel.add_child(gui.Label('Rectangle angle (degrees, clockwise)'))
    rectangle_angle=gui.NumberEdit(gui.NumberEdit.DOUBLE);rectangle_angle.double_value=0;panel.add_child(rectangle_angle)
    panel.add_child(gui.Label('Sphere radius (meters)'))
    sphere_radius=gui.NumberEdit(gui.NumberEdit.DOUBLE);sphere_radius.double_value=2.;panel.add_child(sphere_radius)
    status=gui.Label('No seed selected');panel.add_child(status)
    state={'seed':None,'dirty':False,'undo':[],'revision':revision,'pending':False,'seeds':saved_seeds,'region':np.empty(0,np.int64),'drag_start':None,'rectangle':None,'polygon':[],'polygon_candidates':None,'polygon_labels':[]}
    lo,hi=np.percentile(intensity,[2,98]);grey=np.clip((intensity-lo)/max(hi-lo,1e-6),0,1)
    base=np.repeat(grey[:,None],3,axis=1)
    cloud=o3d.geometry.PointCloud();cloud.points=o3d.utility.Vector3dVector(xyz[display_ids])
    material=rendering.MaterialRecord();material.shader='defaultUnlit';material.point_size=3
    def redraw():
        colors=annotation_colors(base,labels,predicted,prediction_confidence,
                                 show_labels.checked,show_predictions.checked,
                                 highlight_unknown.checked,highlight_unconfirmed.checked)
        if len(state['region']): colors[state['region']]=[1.,.85,.1]
        cloud.colors=o3d.utility.Vector3dVector(colors[display_ids])
        if scene.scene.has_geometry('points'):scene.scene.remove_geometry('points')
        scene.scene.add_geometry('points',cloud,material);scene.force_redraw()
    show_labels.set_on_checked(lambda checked: redraw())
    show_predictions.set_on_checked(lambda checked: redraw())
    highlight_unknown.set_on_checked(lambda checked: redraw())
    highlight_unconfirmed.set_on_checked(lambda checked: redraw())
    def message(text):status.text=text;print(text,flush=True)
    seed_count=gui.Label('');panel.add_child(seed_count)
    def refresh_seeds():
        counts={}
        for point,value in state['seeds']:
            name=CLASSES[value] if value>=0 else 'erase'
            counts[name]=counts.get(name,0)+1
        seed_count.text='Seeds: '+str(len(state['seeds']))+'\n'+'\n'.join(f'{name}: {count}' for name,count in counts.items())
        if scene.scene.has_geometry('seed_markers'):scene.scene.remove_geometry('seed_markers')
        if state['seeds']:
            markers=o3d.geometry.TriangleMesh()
            for point,value in state['seeds']:
                sphere=o3d.geometry.TriangleMesh.create_sphere(.2,resolution=6)
                sphere.translate(xyz[point]);sphere.paint_uniform_color(PALETTE[value] if value>=0 else [1,1,1]);markers+=sphere
            scene.scene.add_geometry('seed_markers',markers,material)
        scene.force_redraw()
    def add_seed():
        if state['seed'] is None:message('Select a seed with Shift + click first');return
        point=state['seed'];value=select.selected_index;value=-1 if value==17 else value
        state['seeds']=[(i,c) for i,c in state['seeds'] if i!=point]+[(point,value)]
        state['dirty']=True;refresh_seeds();message('Seed added. Select another point to add more.')
    def remove_seed():
        if state['seeds']:
            state['seeds'].pop();state['dirty']=True;refresh_seeds()
    def apply():
        if not state['seeds']:message('Add seeds to the queue first');return
        k=kn.int_value;r=radius.double_value
        if k<1 or not np.isfinite(r) or r<=0:message('K and radius must be positive');return
        best={}
        for seed,value in state['seeds']:
            ids=neighbors(tree,xyz,seed,k,r)
            for point,dist in zip(ids,np.linalg.norm(xyz[ids]-xyz[seed],axis=1)):
                if point not in best or dist<best[point][0]:best[point]=(dist,value)
        ids=np.array(list(best),dtype=np.int64)
        values=np.array([best[i][1] for i in ids],dtype=np.int16)
        keep=(labels[ids]!=values)
        if not overwrite.checked:keep&=(labels[ids]<0)|(values<0)
        ids=ids[keep];values=values[keep]
        if not len(ids):message('No changes; enable overwrite to replace labels');return
        state['undo'].append((ids,labels[ids].copy()));state['undo']=state['undo'][-100:]
        labels[ids]=values;state['dirty']=True;redraw();message(f'Changed {len(ids)} points from {len(state["seeds"])} seeds')
    def clear_region():
        state['rectangle']=None
        state['polygon']=[];state['polygon_candidates']=None
        for label in state['polygon_labels']:scene.remove_3d_label(label)
        state['polygon_labels']=[]
        if scene.scene.has_geometry('polygon_vertices'):scene.scene.remove_geometry('polygon_vertices')
        if scene.scene.has_geometry('polygon_outline'):scene.scene.remove_geometry('polygon_outline')
        state['region']=np.empty(0,np.int64)
        if scene.scene.has_geometry('rectangle'):scene.scene.remove_geometry('rectangle')
        if scene.scene.has_geometry('sphere_selection'):scene.scene.remove_geometry('sphere_selection')
        redraw()
    def select_sphere():
        if state['seed'] is None:message('Shift + click a point to choose sphere center');return
        r=sphere_radius.double_value
        if not np.isfinite(r) or r<=0:message('Sphere radius must be positive');return
        clear_region()
        center=xyz[state['seed']]
        state['region']=np.asarray(tree.query_ball_point(center,r),dtype=np.int64)
        mesh=o3d.geometry.TriangleMesh.create_sphere(r,resolution=20);mesh.translate(center)
        wire=o3d.geometry.LineSet.create_from_triangle_mesh(mesh);wire.paint_uniform_color([0.,1.,1.])
        lm=rendering.MaterialRecord();lm.shader='unlitLine';lm.line_width=1
        scene.scene.add_geometry('sphere_selection',wire,lm)
        region_mode.checked=False
        redraw();message(f'Sphere radius {r:g} m: selected {len(state["region"]):,} points')
    def apply_region():
        ids=state['region'];value=select.selected_index;value=-1 if value==17 else value
        if not len(ids):message('Preview your polygon, rectangle or sphere first');return
        keep=labels[ids]!=value
        if not overwrite.checked:keep&=(labels[ids]<0)|(value<0)
        ids=ids[keep]
        if not len(ids):message('No changes; enable overwrite to replace labels');return
        state['undo'].append((ids,labels[ids].copy()));state['undo']=state['undo'][-100:]
        labels[ids]=value;state['dirty']=True;clear_region();message(f'Region: assigned {len(ids):,} points')
    for title,callback in [('Preview sphere at selected point',select_sphere),('Assign selected region to class',apply_region),('Clear region selection',clear_region)]:
        b=gui.Button(title);b.set_on_clicked(callback);panel.add_child(b)
    def draw_polygon():
        for name in ('polygon_outline','polygon_vertices'):
            if scene.scene.has_geometry(name):scene.scene.remove_geometry(name)
        for label in state['polygon_labels']:scene.remove_3d_label(label)
        state['polygon_labels']=[]
        vertices=state['polygon']
        if not vertices:scene.force_redraw();return
        # Put guides in front of the cloud, not behind the ground plane.
        world=np.array([scene.scene.camera.unproject(x,y,.1,scene.frame.width,scene.frame.height) for x,y in vertices],float)
        if not np.isfinite(world).all():return
        markers=o3d.geometry.TriangleMesh()
        for i,(point,(x,y)) in enumerate(zip(world,vertices)):
            adjacent=scene.scene.camera.unproject(x+6,y,.1,scene.frame.width,scene.frame.height)
            radius=max(float(np.linalg.norm(adjacent-point)),1e-5)
            ball=o3d.geometry.TriangleMesh.create_sphere(radius,resolution=8);ball.translate(point)
            ball.paint_uniform_color([1,.2,.1] if i==0 else [1,.9,0]);markers+=ball
            position=scene.scene.camera.unproject(x+10,y-10,.1,scene.frame.width,scene.frame.height)
            label=scene.add_3d_label(position,str(i+1));label.color=gui.Color(1,1,1);label.scale=1.3;state['polygon_labels'].append(label)
        scene.scene.add_geometry('polygon_vertices',markers,material)
        if len(vertices)>1:
            # Render screen-width ribbons as triangles: avoids driver-dependent LineSet rendering.
            edges=[[i,i+1] for i in range(len(vertices)-1)]
            if len(vertices)>2:edges.append([len(vertices)-1,0])
            mesh_vertices=[];triangles=[]
            for a,b in edges:
                a=np.asarray(vertices[a],float);b=np.asarray(vertices[b],float)
                delta=b-a;length=np.linalg.norm(delta)
                if length<1:continue
                normal=np.array([-delta[1],delta[0]])/length*2.5
                quad=[a+normal,b+normal,b-normal,a-normal]
                offset=len(mesh_vertices)
                mesh_vertices.extend(scene.scene.camera.unproject(x,y,.1,scene.frame.width,scene.frame.height) for x,y in quad)
                triangles.extend([[offset,offset+1,offset+2],[offset,offset+2,offset+3],
                                  [offset+2,offset+1,offset],[offset+3,offset+2,offset]])
            if triangles:
                outline=o3d.geometry.TriangleMesh()
                outline.vertices=o3d.utility.Vector3dVector(mesh_vertices)
                outline.triangles=o3d.utility.Vector3iVector(triangles)
                outline.paint_uniform_color([0,1,1])
                scene.scene.add_geometry('polygon_outline',outline,material)
        scene.force_redraw()
    def finish_polygon():
        if not top_mode.checked:message('Enable Top view before ground polygon selection');return
        if select.selected_index not in (11,13,14):message('Ground polygons support driveable_surface, sidewalk, terrain');return
        matrix=np.asarray(scene.scene.camera.get_projection_matrix())@np.asarray(scene.scene.camera.get_view_matrix())
        try:
            ids=polygon_indices(xyz,matrix,scene.frame.width,scene.frame.height,state['polygon'])
            state['polygon_candidates']=ids
            filtered=ground_filter(xyz,ids,z_tolerance.double_value) if z_filter.checked else ids
        except ValueError as e:message(str(e));return
        state['region']=filtered;redraw();message(f'Polygon: {len(ids):,} points -> {len(filtered):,} after Z filter')
    def remove_polygon_vertex():
        if state['polygon']:state['polygon'].pop();state['region']=np.empty(0,np.int64);draw_polygon();redraw()
    for title,callback in [('Finish polygon / Preview Z filter',finish_polygon),('Remove last polygon vertex',remove_polygon_vertex)]:
        b=gui.Button(title);b.set_on_clicked(callback);panel.add_child(b)
    def undo():
        if not state['undo']:return
        ids,old=state['undo'].pop();labels[ids]=old;state['dirty']=True;redraw();message(f'Undid {len(ids)} point changes')
    def save():
        current=output.stat().st_mtime_ns if output.exists() else 0
        if current!=state['revision']:message('Output changed externally; refusing overwrite');return
        output.parent.mkdir(parents=True,exist_ok=True)
        if output.exists():
            import shutil
            b=output.parent/'backups';b.mkdir(exist_ok=True);shutil.copy2(output,b/f'{output.stem}_{time.time_ns()}.npz')
        temp=output.with_name(output.name+'.tmp')
        with temp.open('wb') as f:np.savez_compressed(f,semantics=labels,xyz=xyz.astype(np.float32),source_sha256=np.array(digest),source_path=np.array(str(source)),classes=np.array(CLASSES[:17]),seed_records=np.array(json.dumps(state['seeds'])),voxel_annotation=np.array('manual_seed_bounded_knn'),manual_override_mask=(inherited_manual|(labels!=initial)) if initial is not None else (labels>=0),local_manual_override_mask=(labels!=initial) if initial is not None else (labels>=0),inherited_manual_override_mask=inherited_manual,input_labels_are_prelabels=np.array(initial is not None))
        os.replace(temp,output);state['revision']=output.stat().st_mtime_ns;state['dirty']=False;message(f'Saved {int((labels>=0).sum()):,} labels')
    for title,callback in [('Add selected seed to class',add_seed),('Remove last seed',remove_seed),('Apply KNN to all seeds',apply),('Undo label changes',undo),('Save labels + seeds',save)]:
        button=gui.Button(title);button.set_on_clicked(callback);panel.add_child(button)
    panel.add_child(gui.Label('KNN uses XYZ distance only.\nIt does not recognize objects.\nUse a small radius near boundaries.\nToggle Show label colors to inspect labels.'))
    scroll=gui.ScrollableVert();scroll.add_child(panel)
    w.add_child(scroll);w.add_child(scene)
    def draw_rectangle(start,end,angle):
        corners=np.array([start,(end[0],start[1]),end,(start[0],end[1])],float)
        center=(np.array(start)+np.array(end))/2
        theta=np.deg2rad(angle);rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        corners=(corners-center)@rot.T+center
        vertices=np.array([scene.scene.camera.unproject(x,y,.99,scene.frame.width,scene.frame.height) for x,y in corners],float)
        span=np.ptp(vertices,axis=0)
        if scene.scene.has_geometry('rectangle'):scene.scene.remove_geometry('rectangle')
        if np.isfinite(vertices).all() and np.linalg.norm(span)>1e-6 and np.count_nonzero(span>1e-7)>=2:
            lines=o3d.geometry.LineSet();lines.points=o3d.utility.Vector3dVector(vertices);lines.lines=o3d.utility.Vector2iVector([[0,1],[1,2],[2,3],[3,0]]);lines.paint_uniform_color([1,.85,0])
            lm=rendering.MaterialRecord();lm.shader='unlitLine';lm.line_width=2
            try:scene.scene.add_geometry('rectangle',lines,lm)
            except RuntimeError:message('Rectangle outline unavailable')
    def rotate_rectangle():
        if state['rectangle'] is None:message('Draw a rectangle first');return
        angle=rectangle_angle.double_value
        if not np.isfinite(angle):message('Angle must be finite');return
        start,end=state['rectangle'];draw_rectangle(start,end,angle)
        matrix=np.asarray(scene.scene.camera.get_projection_matrix())@np.asarray(scene.scene.camera.get_view_matrix())
        state['region']=rectangle_indices(xyz,matrix,scene.frame.width,scene.frame.height,start,end,angle)
        redraw();message(f'Angle {angle:g} degrees | {len(state["region"]):,} points selected')
    rotate_button=gui.Button('Apply rectangle angle');rotate_button.set_on_clicked(rotate_rectangle);panel.add_child(rotate_button)
    def zoom_top(factor,anchor=None):
        if not top_mode.checked or 'ortho' not in state:return
        width=scene.frame.width;height=scene.frame.height
        if width<=0 or height<=0:return
        if anchor is None:anchor=(width/2,height/2)
        camera=scene.scene.camera
        world=[camera.unproject(x,y,.99,width,height) for x,y in state['polygon']]
        left,right,bottom,top=state['ortho']
        ax=left+(right-left)*anchor[0]/width
        ay=top-(top-bottom)*anchor[1]/height
        factor=float(np.clip(factor,1/max(right-left,1e-9),2000/max(right-left,1e-9)))
        left=ax+(left-ax)*factor;right=ax+(right-ax)*factor
        bottom=ay+(bottom-ay)*factor;top=ay+(top-ay)*factor
        state['ortho']=(left,right,bottom,top)
        camera.set_projection(rendering.Camera.Projection.Ortho,left,right,bottom,top,.1,3000.)
        if world:
            matrix=np.asarray(camera.get_projection_matrix())@np.asarray(camera.get_view_matrix())
            points=np.asarray(world);clip=points@matrix[:,:3].T+matrix[:,3];ndc=clip[:,:3]/clip[:,3,None]
            state['polygon']=[(float((v[0]+1)*width/2),float((1-v[1])*height/2)) for v in ndc]
            draw_polygon()
        scene.force_redraw()
    zoom_row=gui.Horiz(5)
    for title,factor in [('Top zoom +',.8),('Top zoom -',1.25)]:
        b=gui.Button(title);b.set_on_clicked(lambda f=factor:zoom_top(f));zoom_row.add_child(b)
    panel.add_child(zoom_row)
    def mouse(event):
        if top_mode.checked and event.type==gui.MouseEvent.Type.WHEEL and (polygon_mode.checked or (not region_mode.checked and not lock_view.checked)):
            steps=float(np.clip(event.wheel_dy,-5,5))
            zoom_top(float(np.exp(-.15*steps)),(event.x-scene.frame.x,event.y-scene.frame.y))
            return gui.Widget.EventCallbackResult.CONSUMED
        if polygon_mode.checked:
            if event.type==gui.MouseEvent.Type.BUTTON_DOWN and event.is_button_down(gui.MouseButton.LEFT):
                pos=(event.x-scene.frame.x,event.y-scene.frame.y)
                state['polygon'].append(pos);state['region']=np.empty(0,np.int64)
                draw_polygon();message(f'Polygon vertices: {len(state["polygon"])}; Finish polygon to preview')
            return gui.Widget.EventCallbackResult.CONSUMED
        if region_mode.checked:
            pos=(max(0,min(scene.frame.width,event.x-scene.frame.x)),max(0,min(scene.frame.height,event.y-scene.frame.y)))
            if event.type==gui.MouseEvent.Type.BUTTON_DOWN and event.is_button_down(gui.MouseButton.LEFT):
                anchor=pos
                if state['rectangle'] is not None and abs(rectangle_angle.double_value)<1e-8:
                    a,b=state['rectangle']
                    corners=[a,(b[0],a[1]),b,(a[0],b[1])]
                    distances=[np.linalg.norm(np.array(pos)-c) for c in corners]
                    corner=int(np.argmin(distances))
                    if distances[corner]<=18:anchor=corners[(corner+2)%4]
                state['drag_start']=anchor
                state['region']=np.empty(0,np.int64)
                return gui.Widget.EventCallbackResult.CONSUMED
            if state['drag_start'] is not None and event.type in (gui.MouseEvent.Type.DRAG,gui.MouseEvent.Type.BUTTON_UP):
                start=state['drag_start'];end=pos
                width=abs(end[0]-start[0]);height=abs(end[1]-start[1])
                if scene.scene.has_geometry('rectangle'):scene.scene.remove_geometry('rectangle')
                if width < 3 or height < 3:
                    if event.type==gui.MouseEvent.Type.BUTTON_UP:
                        state['drag_start']=None
                        message('Selection too small; drag a rectangle at least 3 pixels wide and high')
                    return gui.Widget.EventCallbackResult.CONSUMED
                draw_rectangle(start,end,rectangle_angle.double_value)
                if event.type==gui.MouseEvent.Type.BUTTON_UP:
                    matrix=np.asarray(scene.scene.camera.get_projection_matrix())@np.asarray(scene.scene.camera.get_view_matrix())
                    state['region']=rectangle_indices(xyz,matrix,scene.frame.width,scene.frame.height,start,end,rectangle_angle.double_value)
                    state['rectangle']=(start,end)
                    state['drag_start']=None;redraw();message(f'Selected {len(state["region"]):,} points; choose class and Assign')
                scene.force_redraw()
                return gui.Widget.EventCallbackResult.CONSUMED
            return gui.Widget.EventCallbackResult.CONSUMED

        if event.type==gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(gui.KeyModifier.SHIFT) and event.is_button_down(gui.MouseButton.LEFT):
            if state['pending']:return gui.Widget.EventCallbackResult.HANDLED
            x=int(event.x-scene.frame.x);y=int(event.y-scene.frame.y);width=scene.frame.width;height=scene.frame.height
            state['pending']=True
            def depth_ready(image):
                depth=np.asarray(image);seed=None
                if 0<=y<depth.shape[0] and 0<=x<depth.shape[1] and depth[y,x]<1:
                    world=scene.scene.camera.unproject(x,y,float(depth[y,x]),width,height);_,seed=tree.query(world)
                def update():
                    state['pending']=False
                    if seed is None:message('No point at cursor');return
                    state['seed']=int(seed)
                    if scene.scene.has_geometry('seed'):scene.scene.remove_geometry('seed')
                    marker=o3d.geometry.TriangleMesh.create_sphere(.18);marker.translate(xyz[seed]);marker.paint_uniform_color([1,.1,.1]);scene.scene.add_geometry('seed',marker,material)
                    message(f'Seed {seed} | XYZ {xyz[seed].round(2)}')
                app.post_to_main_thread(w,update)
            scene.scene.scene.render_to_depth_image(depth_ready)
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.CONSUMED if (lock_view.checked or (top_mode.checked and event.type in (gui.MouseEvent.Type.DRAG,gui.MouseEvent.Type.BUTTON_DOWN,gui.MouseEvent.Type.BUTTON_UP))) else gui.Widget.EventCallbackResult.IGNORED
    scene.set_on_mouse(mouse)
    scene.set_on_key(lambda event: gui.Widget.EventCallbackResult.CONSUMED if (region_mode.checked or polygon_mode.checked or lock_view.checked or top_mode.checked) else gui.Widget.EventCallbackResult.IGNORED)
    def toggle_rectangle(checked):
        if checked:polygon_mode.checked=False;clear_region()
        state['drag_start']=None
        if not checked and state['rectangle'] is not None:clear_region()
    region_mode.set_on_checked(toggle_rectangle)
    def set_top(checked):
        clear_region();polygon_mode.checked=False;region_mode.checked=False
        b=cloud.get_axis_aligned_bounding_box();center=b.get_center()
        scene.setup_camera(60,b,center)
        if checked:
            scene.look_at(center,center+np.array([0,0,1000.]),[0,1,0])
            half=max(b.get_extent()[0],b.get_extent()[1],10)*.55
            aspect=max(scene.frame.width,1)/max(scene.frame.height,1)
            state['ortho']=(-half*aspect,half*aspect,-half,half)
            scene.scene.camera.set_projection(rendering.Camera.Projection.Ortho,*state['ortho'],.1,3000.)
        scene.force_redraw()
    top_mode.set_on_checked(set_top)
    def set_polygon(checked):
        clear_region()
        if checked:
            if not top_mode.checked:top_mode.checked=True;set_top(True);polygon_mode.checked=True
            region_mode.checked=False
            if select.selected_index not in (11,13,14):select.selected_index=11
            message('Camera locked: click vertices in boundary order, then Finish polygon')
    polygon_mode.set_on_checked(set_polygon)
    def layout(ctx):
        r=w.content_rect;scroll.frame=gui.Rect(r.x,r.y,320,r.height);scene.frame=gui.Rect(r.x+320,r.y,r.width-320,r.height)
    w.set_on_layout(layout)
    def close():
        if not state['dirty']:return True
        dialog=gui.Dialog('Unsaved labels');box=gui.Vert(8,gui.Margins(15,15,15,15));box.add_child(gui.Label('Save labels before closing?'))
        row=gui.Horiz(8)
        def save_close():
            save()
            if not state['dirty']:w.close_dialog();w.set_on_close(lambda:True);w.close()
        def discard():w.close_dialog();w.set_on_close(lambda:True);w.close()
        for text,callback in [('Save and close',save_close),('Discard',discard),('Cancel',w.close_dialog)]:
            b=gui.Button(text);b.set_on_clicked(callback);row.add_child(b)
        box.add_child(row);dialog.add_child(box);w.show_dialog(dialog);return False
    w.set_on_close(close);redraw();refresh_seeds();bounds=cloud.get_axis_aligned_bounding_box();scene.setup_camera(60,bounds,bounds.get_center());app.run()

if __name__=='__main__':main()
