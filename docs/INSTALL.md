# Environment and models

Use Python 3.10 (tested), Linux, a CUDA-capable NVIDIA GPU and sufficient RAM/disk for the capture. The code requires Python >=3.10,<3.12. Windows/macOS end-to-end GPU execution has not been tested.

## Tested runtime family

The original working runtime has PyTorch 2.5.1+cu124, NumPy 1.26.4, SciPy 1.15.3, Open3D 0.19.0, Numba 0.61.2 and KISS-ICP 1.3.0. These are runtime observations, not a portable binary image. Install an appropriate driver and the PyTorch CUDA wheel for your hardware using PyTorch's official instructions. Compiling CUDA extensions requires a compatible toolkit/compiler; `nvidia-smi` alone does not provide a compiler.

```bash
python -m venv .venv
source .venv/bin/activate
# Install the CUDA PyTorch wheel matching your machine first.
python -m pip install -r requirements.txt
python -m pip install -e .
mkdir -p third_party weights/ptv3 weights/transfusion
```

## External repositories

```bash
git clone https://github.com/Pointcept/Pointcept.git third_party/Pointcept
git clone https://github.com/open-mmlab/OpenPCDet.git third_party/OpenPCDet
git clone https://github.com/megvii-research/MCTrack.git third_party/MCTrack
```

The local integration was developed against these revisions:

- Pointcept: `ad653ee75857f57a44cc0ee7cea5fd0372908b4d`
- OpenPCDet: `233f849829b6ac19afb8af8837a0246890908755`
- MCTrack: `a648ee2763ea351a77ddb5232ca00d848d0e46c9`

Checkout the relevant revision before installing dependencies. Follow each repository's own installation instructions, including Pointcept's point operations, matching spconv and OpenPCDet's CUDA extensions. Install MCTrack's requirements and preserve its `tracker`, `kalmanfilter`, `utils` and `config` directories. The pipeline adds its checkout to the Python path; it is not a pip-distributed dependency in this project.

Do not blindly install incompatible Torch/spconv variants from different tutorials. No third-party repository, CUDA binary, pretrained checkpoint or model download credential is included here.

## Checkpoints

Place a compatible nuScenes PTv3 config in `weights/ptv3/config.py`, and its checkpoint in `weights/ptv3/model.pth`. It must produce the standard nuScenes 16 segmentation channels, not a custom reduced static-only head. The inference wrapper disables FlashAttention. A project-specific fine-tuned v2.2 checkpoint can be supplied by the maintainer separately; this archive does not distribute it.

Place the nuScenes TransFusion-L checkpoint in `weights/transfusion/model.pth`. Use the matching OpenPCDet `transfusion_lidar.yaml`. The adapter expects checkpoint `model_state` for detection and `state_dict` for segmentation, and loads model weights strictly.

```bash
occ-label validate --config configs/local.json
occ-label doctor --config configs/local.json
```

`validate` checks paths and configuration. `doctor` checks CUDA availability, key imports and detector extension imports; it is not a full model-forward validation. Run a short capture first. Model-config/checkpoint compatibility is verified during actual model construction.

## Local viewer

The viewer uses OpenCV desktop windows, requires a graphical display, and supports Space (play/pause), A/D (previous/next), a frame slider, and Esc/Q (exit). Six camera images must be present. No browser/server is used. Display target is 10 FPS; actual throughput depends on decoding/storage. It does not alter labels.
