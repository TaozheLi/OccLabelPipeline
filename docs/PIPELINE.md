# Stage details

1. **Input normalization.** Rotate raw XYZ into vehicle-forward orientation; retain measured point count and source intensity. Export an internal binary PCD representation. Match optional images without using their pixels for labels.
2. **Poses.** KISS-ICP uses normalized measured LiDAR, 2..80m range and 0.5m odometry voxel setting. No pointwise-timing deskew in the generic adapter.
3. **Semantics.** PTv3 predicts each real sweep independently with inverse voxel restoration; outputs per-point Occ3D class and confidence. Thresholding controls later acceptance, not frame coordinates.
4. **Detection/tracking.** TransFusion-L uses up to ten pose-aligned sweeps within 0.5 seconds. MCTrack remains continuous across the capture; stopped/slow objects retain dynamic semantic identity.
5. **Temporal objects.** Independent 100-frame blocks accumulate static-speed tracks in world coordinates, or compensate moving tracks into the target object's frame. Source points do not cross block boundaries; tracking IDs do.
6. **Object surfaces.** BPA radii 0.15/0.25/0.4m; normal radius 0.4m; at least30 support points; reconstructed samples are restricted by box and surface-support gates. This stage does not fill whole boxes.
7. **Static world.** High-confidence semantic-static points are fused at 0.1m; track-supported nonground target points are excluded regardless of speed. Optional same-world manual edits apply only in the static branch; edits conflicting with positive dynamic evidence are protected against reintroduction.
8. **Static correction.** Small semantic components can be reassigned by all-class static KNN, dynamic IDs are protected, isolated components under5 cells become unknown, local 2.5D ground interpolation fills supported holes, and a single-pass static KNN assigns supported unknown points. Ground fill is local interpolation, not full drivable-polygon paving. No old-capture polygon is imported automatically.
9. **Vehicle interiors.** The observed+temporal+BPA object support forms a convex hull. Grid samples at 0.1m must be at least0.05m inside that hull and inside the box. Only car/truck/bus/trailer/construction_vehicle use interior fill. Existing dynamic semantics take priority over interior samples.
10. **Final merge.** Project the static world to each raw-LiDAR frame and apply grid Z shift. Static isolated cleanup occurs before all dynamic classes are merged with higher priority. No semantic KNN/cleanup runs afterward.
11. **Free/ego.** Cast only current measured rays. Occupancy wins conflicts. Exclude the configured ego box, save semantic and provenance masks, and assert no protected dynamic label was overwritten.

Defaults are the currently validated recipe. Several geometry and tracking thresholds remain fixed in the engine rather than exposed as arbitrary configuration options; review source/config together before changing them. Neither the pipeline nor its engineering verification guarantees manual-GT quality.
