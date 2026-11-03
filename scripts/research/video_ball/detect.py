"""Detect the ball in the tactical video with a COCO-pretrained YOLO and map it to pitch metres.

Writes every "sports ball" detection (frame, confidence, pixel centre, pitch x/y in metres) to an
.npz under data/research/. No labels are read; runs locally only.

  python scripts/research/video_ball/detect.py --clips 40 --clip-length 50 --out data/research/video_ball_pilot.npz
  python scripts/research/video_ball/detect.py --frames all --out data/research/video_ball_full.npz
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from ultralytics import YOLO

from games import GAMES, asset_paths, half_video_blocks

SPORTS_BALL = 32


def pixel_to_pitch(H, u, v, pitch):
    q = np.linalg.solve(H, np.array([u, v, 1.0]))
    return q[0] / q[2] * pitch[0], q[1] / q[2] * pitch[1]


def frame_list(args, n):
    graded = np.concatenate([np.arange(a, b + 1) for _, a, b in half_video_blocks("BAR-ATH")])
    if args.frames == "all":
        return np.arange(n)
    if args.frames == "graded":
        return graded
    if args.frames == "missing":
        ball = np.load("/tmp/bar_ath_tactical.npz")["ball"]
        return graded[~np.isfinite(ball[graded]).all(1)]
    if args.frames == "holdout":
        graded = graded[(graded // 15000) % 5 == 4]
    starts = graded[np.linspace(0, len(graded) - args.clip_length - 1, args.clips).astype(int)]
    return np.unique(np.concatenate([np.arange(s, s + args.clip_length) for s in starts]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None)
    ap.add_argument("--data", default="/tmp/mtah_data")
    ap.add_argument("--weights", default="yolo11m.pt")
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--frames", default="clips", choices=["clips", "holdout", "missing", "graded", "all"])
    ap.add_argument("--class-id", type=int, default=SPORTS_BALL, help="32 for COCO weights, 0 for the fine-tuned ball model")
    ap.add_argument("--clips", type=int, default=40)
    ap.add_argument("--clip-length", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    paths = asset_paths(data_dir=args.data)
    video = args.video or paths["video"]
    pitch = GAMES["BAR-ATH"]["pitch"]
    h = np.genfromtxt(paths["homography"], delimiter=",", skip_header=1)
    homographies = h[:, 1:10].reshape(-1, 3, 3)
    model = YOLO(args.weights)
    cap = cv2.VideoCapture(video)
    frames = frame_list(args, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    rows, position, started = [], -1, time.monotonic()
    for k, f in enumerate(frames):
        if f != position + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, image = cap.read()
        position = f
        if not ok:
            raise RuntimeError(f"cannot read frame {f}")
        result = model.predict(image, imgsz=args.imgsz, conf=args.conf, classes=[args.class_id],
                               device=args.device, verbose=False)[0]
        for (x1, y1, x2, y2), c in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            u, v = (x1 + x2) / 2, (y1 + y2) / 2
            px, py = pixel_to_pitch(homographies[f], u, v, pitch)
            rows.append((f, c, u, v, px, py))
        if k % 1000 == 0:
            print(k, len(frames), f, len(rows), round(time.monotonic() - started), "s", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    d = np.array(rows, float).reshape(-1, 6)
    np.savez(out, frames=frames, det_frame=d[:, 0].astype(int), conf=d[:, 1], u=d[:, 2], v=d[:, 3], x=d[:, 4], y=d[:, 5])
    print("done", len(frames), "frames", len(d), "detections", round(time.monotonic() - started), "s")


if __name__ == "__main__":
    main()
