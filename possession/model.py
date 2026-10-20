"""Interpretable emissions and per-half competing state histories."""

import numpy as np

from games import GAME, GAMES, half_video_blocks
from possession.data import intervals, rolling_mean
from possession.decode import infer, transition_costs

LETTERS = np.array(["H", "A", "D"])


def sigmoid(x):
    return 1. / (1. + np.exp(-np.clip(x, -30, 30)))


def evidence(arrays, start, stop, config, use_tags=True, use_camera=True):
    """Per-frame log scores. Only supplied observations/metadata enter this function."""
    nearest = arrays["nearest"][start:stop].astype(float)
    seen = np.isfinite(arrays["ball"][start:stop]).all(1)
    distance = np.nan_to_num(nearest, nan=100.)
    near = distance.min(1)
    reliability = np.exp(-np.maximum(0., near - 1.) / 4.) * seen
    margin = (distance[:, 1] - distance[:, 0]) / (distance.sum(1) + .5)
    geometry = np.clip(margin * 4., -3., 3.) * reliability
    # Missing observations contribute no invented team evidence.
    team = config["geometry_weight"] * geometry
    if use_tags:
        team += config["tag_weight"] * arrays["tag"][start:stop]
    home = sigmoid(team)

    speed = rolling_mean(arrays["median_speed"][start:stop], 12)
    stationary = rolling_mean(arrays["stationary"][start:stop], 12)
    speed_known = np.isfinite(speed)
    dead_logit = np.full(len(seen), -1.2)
    dead_logit += np.where(speed_known, 1.6 * np.clip(1. - speed, -1., 1.), 0.)
    dead_logit += np.nan_to_num(1.7 * (stationary - .5))
    ball = arrays["ball"][start:stop]
    outside = ((ball < -.015) | (ball > 1.015)).any(1) & seen
    dead_logit += outside * 1.5
    # Slow tracked ball is weak evidence only; it can also be controlled in play.
    displacement = np.linalg.norm(np.diff(ball, axis=0, prepend=ball[:1]) * np.array([105.3, 68.]), axis=1)
    contiguous = seen & np.r_[False, seen[:-1]]
    still = rolling_mean(np.where(contiguous, displacement * 25 < .5, np.nan), 12)
    dead_logit += np.nan_to_num(.6 * (still - .5))
    if use_tags:
        active_phase = np.abs(arrays["tag"][start:stop])
        dead_logit -= .6 * active_phase
        # A set-piece interval straddles the restart; never label it wholly dead.
        dead_logit += .2 * arrays["restart"][start:stop]
    camera_activity = np.zeros(len(seen))
    if use_camera:
        center = arrays["camera_center"][start:stop]
        velocity = np.linalg.norm(np.diff(center, axis=0, prepend=center[:1]) * np.array([105.3, 68.]), axis=1) * 25
        camera_activity = np.nan_to_num(np.clip(rolling_mean(velocity, 12) / 4., 0., 1.))
        # Small, bounded cue on missing-ball frames; no claim of independence.
        dead_logit -= config["camera_weight"] * camera_activity * (~seen)
    dead = np.clip(sigmoid(dead_logit), .02, .95)
    probabilities = np.column_stack([(1. - dead) * home, (1. - dead) * (1. - home), dead])
    return np.log(np.clip(probabilities, 1e-6, 1.)), {
        "geometry": geometry, "dead_probability": dead, "camera_activity": camera_activity}


def predict(arrays, config, use_tags=True, use_camera=True, switch_scale=1., samples=None, score_override=None):
    n = len(arrays["ball"])
    labels = np.full(n, "X", dtype="<U1")
    posterior = np.zeros((n, 3), dtype=np.float32)
    sample_count = config["history_samples"] if samples is None else samples
    histories = np.full((sample_count, n), 3, dtype=np.int8)
    diagnostics = {key: np.zeros(n, dtype=np.float32) for key in ("geometry", "dead_probability", "camera_activity")}
    for half, start, last in half_video_blocks(GAME):
        stop = last + 1
        score, parts = evidence(arrays, start, stop, config, use_tags, use_camera)
        if score_override is not None:
            score = np.asarray(score_override[start:stop], dtype=float)
        step = config["step"]
        starts = np.arange(0, len(score), step)
        lengths = np.minimum(step, len(score) - starts)
        coarse = np.add.reduceat(score, starts, axis=0) / lengths[:, None]
        transition = transition_costs(config["team_switch_cost"] * switch_scale,
                                      config["stoppage_switch_cost"] * switch_scale)
        path, marginal, alternatives = infer(coarse, transition, sample_count, config["seed"] + half)
        repeat = lambda values: np.repeat(values, step, axis=0)[:stop - start]
        labels[start:stop] = LETTERS[repeat(path)]
        posterior[start:stop] = repeat(marginal)
        for k in range(sample_count):
            histories[k, start:stop] = repeat(alternatives[k])
        for key in diagnostics:
            diagnostics[key][start:stop] = parts[key]
    return labels, posterior, histories, diagnostics


def choose_queries(posterior, disagreement, arrays, config, policy="boundary", blocks=None, decoded_labels=None):
    """Fixed-budget local video queries; never uses GT or evaluation errors."""
    entropy = -(posterior * np.log(np.clip(posterior, 1e-8, 1.))).sum(1) / np.log(3.)
    missing = ~np.isfinite(arrays["ball"]).all(1)
    labels = posterior.argmax(1) if decoded_labels is None else np.asarray(decoded_labels)
    if len(labels) != len(posterior):
        raise ValueError('Decoded labels and posterior must have the same length')
    boundary = np.r_[False, labels[1:] != labels[:-1]]
    vicinity = np.nan_to_num(rolling_mean(boundary.astype(float), 25)) * 50
    if policy == "boundary":
        score = entropy + .8 * disagreement + .4 * np.clip(vicinity, 0, 1) + .15 * missing
    elif policy == "entropy":
        score = entropy
    elif policy == "uniform":
        score = np.zeros(len(missing))
    elif policy == "random":
        score = np.random.default_rng(config["seed"]).random(len(missing))
    else:
        raise ValueError("Unknown query policy")
    budget = config["query_budget"]
    spacing = int(config["query_spacing_seconds"] * 25)
    picked = []
    blocks = half_video_blocks(GAME) if blocks is None else blocks
    for index, (half, start, stop) in enumerate(blocks):
        count = budget // len(blocks) + (1 if index < budget % len(blocks) else 0)
        candidates = np.arange(start + 25, stop - 25, config["step"])
        if policy == "uniform":
            order = np.linspace(0, len(candidates) - 1, max(1, count), dtype=int) if count else []
        else:
            order = np.argsort(-score[candidates], kind="stable")
        selected = []
        if count == 0:
            continue
        for j in order:
            frame = int(candidates[j])
            if all(abs(frame - other) >= spacing for other in selected):
                selected.append(frame)
            if len(selected) >= count:
                break
        for frame in selected:
            picked.append({"frame": frame, "half": half, "start_frame": max(start, frame - 25),
                           "stop_frame": min(stop + 1, frame + 26), "score": float(score[frame]),
                           "entropy": float(entropy[frame]), "source_disagreement": float(disagreement[frame]),
                           "ball_missing": bool(missing[frame]), "policy": policy,
                           "reason": "Model-relative ambiguity; not a known mistake or calibrated probability"})
    return sorted(picked, key=lambda item: (-item["score"], item["frame"]))


def boundary_receipts(labels, posterior, arrays, disagreement):
    rows = []
    for half, start, last in half_video_blocks(GAME):
        changes = np.flatnonzero(labels[start + 1:last + 1] != labels[start:last]) + start + 1
        for frame in changes:
            low, high = max(start, int(frame) - 50), min(last + 1, int(frame) + 51)
            # Band is local model ambiguity, not a statistically valid interval.
            ambiguous = np.flatnonzero(posterior[low:high].max(1) < .8)
            rows.append({"frame": int(frame), "half": half, "from": labels[frame - 1], "to": labels[frame],
                         "ambiguity_start": int(low + ambiguous[0]) if len(ambiguous) else int(frame),
                         "ambiguity_stop": int(low + ambiguous[-1] + 1) if len(ambiguous) else int(frame + 1),
                         "model_support": float(posterior[frame].max()),
                         "source_disagreement": float(disagreement[frame]),
                         "ball_observed": bool(np.isfinite(arrays["ball"][frame]).all()),
                         "tag_evidence_home_minus_away": float(arrays["tag"][frame])})
    return rows
