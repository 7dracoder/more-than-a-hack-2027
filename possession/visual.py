"""Budgeted local visual motion measurements, not a trained contact recognizer."""

import numpy as np


def measure(video_path, queries, budget):
    try:
        import av
        import cv2
    except ImportError as error:
        raise RuntimeError("Visual queries require optional av and opencv-python-headless") from error
    results = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for query in queries[:budget]:
            start = max(0, query["frame"] - 10)
            container.seek(int((start / 25.) / stream.time_base), stream=stream, backward=True)
            frames = []
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                index = round(float(frame.pts * stream.time_base) * 25)
                if index < start:
                    continue
                if len(frames) == 0 or index >= frames[-1][0] + 5:
                    gray = cv2.cvtColor(frame.to_ndarray(format="bgr24"), cv2.COLOR_BGR2GRAY)
                    frames.append((index, cv2.resize(gray, (640, 360))))
                if len(frames) >= 3:
                    break
            residuals, matches = [], []
            for (_, first), (_, last) in zip(frames, frames[1:]):
                points = cv2.goodFeaturesToTrack(first, maxCorners=300, qualityLevel=.02, minDistance=8)
                if points is None or len(points) < 12:
                    continue
                tracked, status, _ = cv2.calcOpticalFlowPyrLK(first, last, points, None)
                if tracked is None:
                    continue
                a = points[status.ravel() == 1].reshape(-1, 2)
                b = tracked[status.ravel() == 1].reshape(-1, 2)
                if len(a) < 12:
                    continue
                matrix, _ = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.)
                if matrix is None:
                    continue
                predicted = np.column_stack([a, np.ones(len(a))]) @ matrix.T
                residual = np.linalg.norm(b - predicted, axis=1)
                residuals.extend(residual.tolist())
                matches.append(len(a))
            item = dict(query)
            item.update({"frames_decoded_for_measurement": [f[0] for f in frames],
                         "status": "measured" if residuals else "unresolved",
                         "motion_q90_pixels_at_640px": float(np.quantile(residuals, .9)) if residuals else None,
                         "mean_tracks": float(np.mean(matches)) if matches else 0.,
                         "interpretation": "Camera-compensated scene-motion cue only; cannot establish controlled contact or team"})
            results.append(item)
    return results
