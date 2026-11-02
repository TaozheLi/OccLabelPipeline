# Label semantics and provenance

IDs 0..17: `others`, `barrier`, `bicycle`, `bus`, `car`, `construction_vehicle`, `motorcycle`, `pedestrian`, `traffic_cone`, `trailer`, `truck`, `driveable_surface`, `other_flat`, `sidewalk`, `terrain`, `manmade`, `vegetation`, `free`.

PTv3 has 16 channels corresponding to IDs1..16; it does not predict free. No automatic image semantics are used.

```python
import numpy as np
with np.load('labels.npz') as d:
    semantics = d['semantics']
    valid = d['mask_lidar'].astype(bool) & ~d['mask_ego_excluded'].astype(bool)
    occupied = valid & (semantics != 17)
    known_free = valid & (semantics == 17)
    unknown = ~valid & ~d['mask_ego_excluded'].astype(bool)
```

Unknown and known free share numeric ID17 and must be distinguished by validity. Ego space is unknown/excluded, never forced free. `mask_lidar` means valid retained occupancy/free evidence, **including inferred completion**; it does not mean every cell was directly measured. See individual provenance masks (`mask_current_measured`, `mask_world_static`, `mask_inferred`, `mask_dynamic_temporal`, `mask_dynamic_bpa`, `mask_dynamic_interior`, `mask_ground_interpolated`, `mask_unknown_knn_assigned`, etc.) for origin.

This release intentionally returns `mask_camera=False` everywhere because no camera calibration is assumed. Do not use that unavailable mask for training supervision.

Semantic arrays are indexed X,Y,Z. Grid voxel centres equal `bounds[:3] + (index+0.5)*voxel_size`. World poses refer to the prepared, yaw-normalized raw-LiDAR frame. Raw Z is shifted by `lidar_height_m` for OCC, so transform back with the inverse shift before applying a raw-LiDAR world pose. Metadata stores `T_world_from_grid`.

Defaults: 0.1m surface/world cells, radius/confidence-gated static KNN, 26-connected static cleanup before dynamic merge, and <5-voxel cleanup at the corresponding world/OCC stage. Vehicle-interior samples are inside the convex hull with >=0.05m inward margin and detection-box clipping. Existing dynamic class votes take priority over interior-fill votes.

Only actual current-scan returns are used as free-ray endpoints; no rays are cast to temporal/BPA/interior samples. Occupied wins conflicts with ray-derived free. Stopped targets remain in the dynamic semantic branch. Source tags and confidence do not turn inferred geometry into measured evidence.

Counts summed across frames are voxel/point observations, not globally unique world points. Generated labels require review; agreement with these labels is an internal pseudo-GT metric.
