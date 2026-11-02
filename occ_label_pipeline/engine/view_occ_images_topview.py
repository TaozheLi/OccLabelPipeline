"""Native synchronized six-camera and fixed OCC bird's-eye player."""
from occ_label_pipeline.context import CFG, CONFIG_PATH, PROJECT_ROOT, ENGINE_ROOT, OUTPUT as CAPTURE_ROOT, DATA_ROOT, TRACKER_ROOT, NFRAMES
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from visualize_sparse_occ import camera_montage, top_view

def render(data_root, occ_root, frame):
    root = occ_root / frame
    meta = json.loads((root / 'metadata.json').read_text())
    with np.load(root / 'labels.npz') as data:
        bev = top_view(data['semantics'], data['mask_lidar'], np.asarray(meta['bounds']), meta['voxel_size'], 40, 640, show_free=False)
    image = Image.new('RGB', (1920, 680), (18, 20, 24))
    image.paste(camera_montage(data_root / frame, (426, 320)), (0, 0))
    image.paste(bev, (1280, 0))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--occ-root', type=Path, required=True)
    p.add_argument('--fps', type=float, default=10)
    p.add_argument('--autoplay', action='store_true')
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    if a.fps <= 0:
        p.error('fps must be positive')
    frames = sorted((x.parent.name for x in a.occ_root.glob('*/labels.npz') if x.with_name('metadata.json').exists() and all(((a.data_root / x.parent.name / (c + '.png')).exists() for c in ('FL', 'F', 'FR', 'BL', 'B', 'BR')))))
    if not frames:
        p.error('No matching frames')
    if a.check:
        picture = render(a.data_root, a.occ_root, frames[0])
        print(f'Validated {len(frames)} frames; image {picture.shape}', flush=True)
        return
    title = 'Six cameras + OCC top view'
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1536, 544)
    state = {'seek': None, 'programmatic': False}

    def seek(value):
        if not state['programmatic']:
            state['seek'] = value
    cv2.createTrackbar('Frame', title, 0, max(1, len(frames) - 1), seek)
    index, playing, last, dirty = (0, a.autoplay, 0.0, True)
    jobs = {}
    pool = ThreadPoolExecutor(max_workers=3)
    try:
        while True:
            if state['seek'] is not None:
                index = min(state['seek'], len(frames) - 1)
                state['seek'] = None
                dirty = True
            if dirty:
                for j in range(index, min(index + 8, len(frames))):
                    if j not in jobs:
                        jobs[j] = pool.submit(render, a.data_root, a.occ_root, frames[j])
                for j in list(jobs):
                    if j < index - 2 or j > index + 10:
                        jobs.pop(j).cancel()
                picture = jobs[index].result().copy()
                cv2.putText(picture, f'{frames[index]}  |  {index + 1}/{len(frames)}  |  Target {a.fps:g} FPS  |  Space: Play/Pause   A/D: Previous/Next   Esc: Exit', (20, 666), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (225, 225, 225), 1, cv2.LINE_AA)
                cv2.imshow(title, picture)
                state['programmatic'] = True
                cv2.setTrackbarPos('Frame', title, index)
                state['programmatic'] = False
                dirty = False
                if last == 0:
                    last = time.monotonic()
            key = cv2.waitKey(1) & 255
            if key in (27, ord('q')) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == 32:
                playing = not playing
                last = time.monotonic()
            if key in (ord('a'), ord('d')):
                index = max(0, min(len(frames) - 1, index + (1 if key == ord('d') else -1)))
                dirty, playing = (True, False)
            if playing and time.monotonic() - last >= 1 / a.fps:
                last = time.monotonic()
                if index + 1 < len(frames):
                    index += 1
                    dirty = True
                else:
                    playing = False
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
