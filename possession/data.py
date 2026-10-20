"""Streaming, team-order-invariant features. Never reads possession labels."""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from games import GAME, GAMES, asset_paths, half_video_blocks


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rolling_mean(values, width):
    """Centered finite-only mean; callers split halves before temporal inference."""
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(values)
    sums = np.r_[0., np.cumsum(np.where(ok, values, 0.))]
    counts = np.r_[0, np.cumsum(ok)]
    i = np.arange(len(values))
    lo, hi = np.maximum(0, i - width), np.minimum(len(values), i + width + 1)
    return np.divide(sums[hi] - sums[lo], counts[hi] - counts[lo],
                     out=np.full(len(values), np.nan), where=counts[hi] > counts[lo])


def slot_teams(path):
    root = ET.parse(path).getroot()
    teams = [t.get("id") for t in root.iter("Team")]
    if len(teams) != 2:
        raise ValueError("Expected exactly two teams")
    slots = [p.get("teamId") for p in root.iter("Player")]
    if any(t not in teams for t in slots):
        raise ValueError("Unknown track-slot team")
    return np.array([teams.index(t) for t in slots])


def frame_features(players, ball, teams, pitch):
    """Pure team-set reduction: a joint permutation of slots/teams changes nothing."""
    xy = players[:, :2] * pitch
    known = np.isfinite(xy).all(1)
    seen = bool(np.isfinite(ball[:2]).all())
    distances = np.linalg.norm(xy - ball[:2] * pitch, axis=1) if seen else np.full(len(xy), np.nan)
    nearest, counts = [], []
    for team in (0, 1):
        mask = known & (teams == team)
        counts.append(int(mask.sum()))
        nearest.append(float(np.min(distances[mask])) if seen and mask.any() else np.nan)
    speeds = players[known, 2]
    speeds = speeds[np.isfinite(speeds) & (speeds >= 0) & (speeds <= 15)]
    return (nearest, counts, float(np.median(speeds)) if len(speeds) else np.nan,
            float(np.mean(speeds < .5)) if len(speeds) else np.nan)


def tagging_features(path, n, fps):
    evidence = np.zeros(n + 1)
    coverage = np.zeros(n + 1)
    restarts = np.zeros(n + 1)
    unmapped = set()
    offensive = {"build up", "progression", "creating chances", "attacking transition",
                 "long ball", "ball in final third", "ball in the box", "players in the box"}
    defensive = {"defending in attacking third", "defending in middle third",
                 "defending in defensive third", "defensive transition"}
    for event in ET.parse(path).iter("instance"):
        start = max(0, min(n, round(float(event.findtext("start")) * fps)))
        stop = max(start, min(n, round(float(event.findtext("end")) * fps)))
        tags = {(x.findtext("group") or "").strip().lower():
                (x.findtext("text") or "").strip().lower() for x in event.findall("label")}
        code = (event.findtext("code") or "").strip().lower()
        if code == "set pieces":
            restarts[start] += 1
            restarts[stop] -= 1
        name = tags.get("team", "")
        sign = 1 if name == "barcelona" else -1 if name in ("atletic club", "athletic club") else 0
        if code in defensive:
            sign *= -1
        elif code not in offensive:
            if code != "set pieces":
                unmapped.add(code)
            continue
        # Set-piece team indicates restart ownership, not whether the ball is live.
        weight = .35 if code == "set pieces" else 1.
        if sign:
            evidence[start] += sign * weight
            evidence[stop] -= sign * weight
            coverage[start] += weight
            coverage[stop] -= weight
    support = np.cumsum(evidence[:-1])
    count = np.cumsum(coverage[:-1])
    return (support / np.maximum(1., count), np.cumsum(restarts[:-1]) > 0,
            {"unmapped_codes": sorted(unmapped)})


def extract(data_dir, cache_dir, rebuild=False):
    paths = asset_paths(data_dir=str(data_dir))
    names = ("atd_raw", "atd_meta", "smart_tagging", "homography")
    for key in names:
        if not Path(paths[key]).is_file():
            raise FileNotFoundError(paths[key])
    fingerprints = {key: sha256(paths[key]) for key in names}
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / "features-v1.npz"
    metadata = cache_dir / "features-v1.json"
    if not rebuild and cache.exists() and metadata.exists():
        old = json.loads(metadata.read_text())
        if old.get("input_sha256") == fingerprints:
            with np.load(cache) as z:
                return {key: z[key] for key in z.files}, old

    n, fps = GAMES[GAME]["n_video_frames"], GAMES[GAME]["fps"]
    pitch = np.asarray(GAMES[GAME]["pitch"])
    teams = slot_teams(paths["atd_meta"])
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    nearest = np.full((n, 2), np.nan, dtype=np.float32)
    counts = np.zeros((n, 2), dtype=np.int16)
    speed = np.full(n, np.nan, dtype=np.float32)
    stationary = np.full(n, np.nan, dtype=np.float32)
    previous_id = 0
    with open(paths["atd_raw"]) as handle:
        for line in handle:
            fields = line.strip().split(":")
            if len(fields) != 3:
                raise ValueError("Malformed ATD record")
            frame_id = int(fields[0])
            if frame_id != previous_id + 1 or frame_id > n:
                raise ValueError("ATD frame IDs must be exactly 1..n in file order")
            previous_id = frame_id
            players = np.fromstring(fields[1].replace(";", ","), sep=",")
            point = np.fromstring(fields[2], sep=",")
            if players.size != len(teams) * 3 or point.size < 2:
                raise ValueError("Unexpected ATD group dimensions")
            players = players.reshape(-1, 3)
            i = frame_id - 1
            ball[i] = point[:2]
            nearest[i], counts[i], speed[i], stationary[i] = frame_features(players, point, teams, pitch)
    if previous_id != n:
        raise ValueError("Incomplete ATD file")

    table = pd.read_csv(paths["homography"])
    if not np.array_equal(table["frame"].values, np.arange(n)):
        raise ValueError("Homography frame IDs do not match tactical video")
    matrices = table[["h%d%d" % (i, j) for i in range(3) for j in range(3)]].values.reshape(-1, 3, 3)
    valid = table["valid"].astype(bool).values & np.isfinite(matrices).all((1, 2))
    center = np.full((n, 2), np.nan)
    for start in range(0, n, 10000):
        ids = np.flatnonzero(valid[start:start + 10000]) + start
        points = np.linalg.inv(matrices[ids]) @ np.array([960., 540., 1.])
        center[ids] = points[:, :2] / points[:, 2, None]
    tag, restart, tag_meta = tagging_features(paths["smart_tagging"], n, fps)
    arrays = {"ball": ball, "nearest": nearest, "counts": counts, "median_speed": speed,
              "stationary": stationary, "camera_center": center.astype(np.float32),
              "camera_valid": valid, "tag": tag.astype(np.float32), "restart": restart}
    meta = {"input_sha256": fingerprints, "n_frames": n, "fps": fps, "track_slots": len(teams),
            "pitch_m": pitch.tolist(), "tagging": tag_meta, "feature_version": 1}
    np.savez_compressed(cache, **arrays)
    metadata.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return arrays, meta


def intervals(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def audit(arrays):
    seen = np.isfinite(arrays["ball"]).all(1)
    result = {}
    for half, start, stop in half_video_blocks(GAME):
        runs = intervals(~seen[start:stop + 1])
        lengths = np.array([b - a for a, b in runs]) / 25.
        result[str(half)] = {"observed_ball_fraction": float(seen[start:stop + 1].mean()),
                             "gaps": len(runs), "gap_seconds_quantiles":
                             np.quantile(lengths, [.5, .9, .99, 1]).tolist() if len(lengths) else [],
                             "mean_tracked_players": float(arrays["counts"][start:stop + 1].sum(1).mean())}
    return result
