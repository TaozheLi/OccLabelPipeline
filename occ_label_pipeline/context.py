"""Runtime paths shared by engine processes; raw data and output never alias."""
import os,json
from pathlib import Path
from .config import load_config
CFG,CONFIG_PATH=load_config(os.environ['OCC_PIPELINE_CONFIG'])
PROJECT_ROOT=Path(__file__).resolve().parents[1]
ENGINE_ROOT=Path(__file__).resolve().parent/'engine'
OUTPUT=Path(CFG['output_root']);DATA_ROOT=OUTPUT/'occ_pipeline_v3_0/input_frames'
TRACKER_ROOT=Path(CFG['tracking_root'])
p=OUTPUT/'occ_pipeline_v3_0/capture_manifest.json'
NFRAMES=int(json.loads(p.read_text())['frames']) if p.exists() else int(CFG.get('frame_count',0))
