"""Local externally pretrained ball detection; no training or possession labels.

Resume requires identical inputs, settings, source and hashed output. CPU is
the reproduction mode. Raw detections are evidence, not possession predictions.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from games import GAMES, asset_paths, half_video_blocks
from starter.load_homography import load as load_homography
from scripts.research.video_ball.tracking import load_tracking, play_mask

SPORTS_BALL = 32
FIELDS = ('det_frame', 'conf', 'u', 'v', 'x', 'y')


def digest(path):
    value = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def pixel_to_pitch(matrix, u, v, pitch):
    matrix = np.asarray(matrix, float)
    if not np.isfinite(matrix).all():
        return np.nan, np.nan
    try:
        q = np.linalg.solve(matrix, np.array([u, v, 1.0]))
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    if not np.isfinite(q).all() or abs(q[2]) <= np.finfo(float).eps:
        return np.nan, np.nan
    return q[0] / q[2] * pitch[0], q[1] / q[2] * pitch[1]


def frame_list(args, n, ball=None):
    if n != GAMES['BAR-ATH']['n_video_frames']:
        raise ValueError(f'Wrong video frame count: {n}; require tactical BAR-ATH timeline')
    if args.frames == 'all':
        return np.arange(n)
    graded = np.flatnonzero(play_mask(n))
    if args.frames == 'graded':
        return graded
    if args.frames == 'missing':
        if ball is None or ball.shape != (n, 2):
            raise ValueError('Missing-frame selection requires aligned supplied tracking')
        return graded[~np.isfinite(ball[graded]).all(1)]
    if args.clips <= 0 or args.clip_length <= 0:
        raise ValueError('Clip count and length must be positive')
    eligible = play_mask(n)
    if args.frames == 'holdout':
        eligible &= (np.arange(n) // 15000) % 5 == 4
    width = args.clip_length
    if width > n:
        raise ValueError('No complete clips fit the selected windows')
    sums = np.r_[0, np.cumsum(eligible.astype(np.int64))]
    starts = np.flatnonzero(sums[width:] - sums[:-width] == width)
    if not len(starts):
        raise ValueError('No complete clips fit the selected windows')
    chosen = starts[np.linspace(0, len(starts) - 1, min(args.clips, len(starts))).astype(int)]
    return np.unique(np.concatenate([np.arange(s, s + width) for s in chosen]))


def save_checkpoint(out, selected, completed, rows, protocol):
    data = np.asarray(rows, float).reshape(-1, 6)
    part = out.with_suffix(out.suffix + '.partial')
    with part.open('wb') as stream:
        np.savez_compressed(stream, frames=selected, completed_frames=completed,
                            **{name: data[:, i].astype(np.int64) if i == 0 else data[:, i]
                               for i, name in enumerate(FIELDS)})
    part.replace(out)
    record = dict(protocol=protocol, output_sha256=digest(out),
                  completed_frames=len(completed), selected_frames=len(selected),
                  detections=len(rows), complete=len(completed) == len(selected),
                  training_performed=False, possession_labels_read=False)
    meta = out.with_suffix('.json')
    part = meta.with_suffix('.json.partial')
    part.write_text(json.dumps(record, indent=2), encoding='utf-8')
    part.replace(meta)


def resume_checkpoint(out, selected, protocol):
    meta = json.loads(out.with_suffix('.json').read_text(encoding='utf-8'))
    if meta['protocol'] != protocol or meta['output_sha256'] != digest(out):
        raise ValueError('Resume refused: inputs, settings, source or output changed')
    with np.load(out, allow_pickle=False) as data:
        completed = data['completed_frames']
        if (not np.array_equal(selected, data['frames']) or
                not np.array_equal(completed, selected[:len(completed)]) or
                len(completed) != meta['completed_frames']):
            raise ValueError('Resume refused: completed frames are not the selected prefix')
        columns = [data[field] for field in FIELDS]
        if len({len(column) for column in columns}) != 1:
            raise ValueError('Resume refused: inconsistent detection columns')
        if not np.isin(columns[0], completed).all():
            raise ValueError('Resume refused: detection on an unprocessed frame')
        rows = np.column_stack(columns).tolist()
    return completed.tolist(), rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path)
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--weights', type=Path, default=ROOT / 'yolo11m.pt')
    parser.add_argument('--imgsz', type=int, default=1920)
    parser.add_argument('--conf', type=float, default=0.05)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--frames', default='clips', choices=['clips', 'holdout', 'missing', 'graded', 'all'])
    parser.add_argument('--class-id', type=int, default=SPORTS_BALL)
    parser.add_argument('--clips', type=int, default=40)
    parser.add_argument('--clip-length', type=int, default=50)
    parser.add_argument('--checkpoint-every', type=int, default=1000)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.conf <= 1 or args.imgsz <= 0 or args.checkpoint_every <= 0:
        parser.error('Invalid inference settings')
    paths = asset_paths(data_dir=str(args.data))
    video = args.video or Path(paths['video'])
    for path in (video, args.weights, Path(paths['homography'])):
        if not path.is_file():
            parser.error(f'Required local input missing: {path}')
    if args.out.exists() and not args.resume:
        parser.error('Output exists; use --resume with the identical protocol or a new output')
    # This is a local-only pipeline. Keep library state in ignored output storage.
    os.environ['YOLO_CONFIG_DIR'] = str((args.out.parent / 'yolo-config').resolve())
    os.environ['YOLO_OFFLINE'] = 'true'
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO

    ultralytics.settings.update({'sync': False, 'clearml': False, 'comet': False,
                                 'dvc': False, 'mlflow': False, 'raytune': False,
                                 'tensorboard': False, 'wandb': False})

    model = YOLO(str(args.weights))
    if args.class_id not in model.names:
        parser.error('Requested class is absent from the supplied pretrained weights')
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        parser.error(f'Cannot open video: {video}')
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isclose(cap.get(cv2.CAP_PROP_FPS), 25):
            raise ValueError('Tactical video must be 25 fps')
        ball = load_tracking(args.data)[1] if args.frames == 'missing' else None
        selected = frame_list(args, n, ball)
        homographies, _, _ = load_homography(data_dir=str(args.data))
        if len(homographies) != n:
            raise ValueError('Homography/video frame counts differ')
        protocol = {
            'video_sha256': digest(video), 'weights_sha256': digest(args.weights),
            'homography_sha256': digest(paths['homography']),
            'tracking_sha256': digest(paths['atd_raw']) if ball is not None else None,
            'source_sha256': digest(__file__),
            'tracking_source_sha256': digest(Path(__file__).with_name('tracking.py')),
            'homography_source_sha256': digest(ROOT / 'starter/load_homography.py'),
            'games_source_sha256': digest(ROOT / 'games.py'),
            'selected_frames_sha256': hashlib.sha256(selected.astype('<i8').tobytes()).hexdigest(),
            'settings': {k: getattr(args, k) for k in ('imgsz', 'conf', 'device', 'frames', 'class_id', 'clips', 'clip_length')},
            'versions': dict(torch=torch.__version__, opencv=cv2.__version__,
                             ultralytics=ultralytics.__version__, numpy=np.__version__),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        completed, rows = resume_checkpoint(args.out, selected, protocol) if args.resume and args.out.exists() else ([], [])
        position, started = -1, time.monotonic()
        for f in selected[len(completed):]:
            if f != position + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, image = cap.read()
            if not ok:
                raise RuntimeError(f'Cannot decode tactical video frame {f}')
            position = f
            if int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) != f + 1:
                raise RuntimeError(f'Decoder position mismatch at frame {f}')
            result = model.predict(image, imgsz=args.imgsz, conf=args.conf,
                                   classes=[args.class_id], device=args.device,
                                   verbose=False, save=False)[0]
            for (x1, y1, x2, y2), confidence in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                u, v = (x1 + x2) / 2, (y1 + y2) / 2
                x, y = pixel_to_pitch(homographies[f], u, v, GAMES['BAR-ATH']['pitch'])
                rows.append((int(f), float(confidence), float(u), float(v), float(x), float(y)))
            completed.append(int(f))
            if len(completed) % args.checkpoint_every == 0:
                save_checkpoint(args.out, selected, completed, rows, protocol)
                print(f'{len(completed)}/{len(selected)} frames; {len(rows)} detections; {time.monotonic() - started:.0f}s', flush=True)
        save_checkpoint(args.out, selected, completed, rows, protocol)
        print(f'Complete: {len(completed)} frames, {len(rows)} detections; no possession score computed')
    finally:
        cap.release()


if __name__ == '__main__':
    main()
