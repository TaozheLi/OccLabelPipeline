# Verification scope

## Passed locally

- Eight CPU tests: measured-point preservation in the binary adapter, yaw rotation, config-relative paths, input/output separation, portable source compilation, dynamic priority over manual static labels, measured-return endpoint exclusion from free rays, and ego/free exclusion.
- A clean three-frame real-PCD integration run: adapter, KISS-ICP, PTv3 checkpoint loading/inference, TransFusion inference, continuous tracking, temporal accumulation, BPA, static world fusion, KNN/cleanup/ground fill, convex-hull interiors and final OCC voxelization.
- All three final labels had the default 200x200x16 shape; dynamic_overwritten_voxels was zero. Unknown/ego masking invariants were checked.
- A distributable wheel was built with setuptools.

```bash
python -m unittest discover -s tests -v
python -m compileall -q occ_label_pipeline
python -m pip wheel --no-deps --no-build-isolation . -w dist
```

The real scans and pretrained/fine-tuned weights used in that local integration are not distributed. The test used already installed external frameworks; it does not prove that a clean CUDA installation works on another machine. Full captures are substantially larger than the three-frame test. No independent segmentation/occupancy accuracy is claimed by these engineering checks.

The NPZ/CSV adapter uses explicit intensity-proxy mode and nearest epoch timestamps. The portable binary/rotation components have tests, but a full new-host NPZ-camera/GPU integration has not been established across sensor formats. Camera calibration and camera-mask inference are deliberately outside this release.

Run a small, complete capture before a long job. A tiny fragment with almost no static support or flat/degenerate object geometry may skip completion or be insufficient for reconstruction. Do not split a live moving scan at arbitrary point indices to fabricate input frames.
