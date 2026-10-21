"""Private local synchronized video/pitch review; never edits predictions."""



import hashlib

import json

from fractions import Fraction

from pathlib import Path



import numpy as np



from games import GAME, half_video_blocks

from possession.data import slot_teams

from possession.review_ui import TEMPLATE





def build_review(data_dir, output, queries, labels, posterior, encode=True, budget=12,

                 experiment_id=None, title='BlindSpot review', notice='Model support is not calibrated confidence. Notes never change predictions.'):

    output = Path(output)

    clips = output / 'clips'

    clips.mkdir(parents=True, exist_ok=True)

    data_dir = Path(data_dir)

    chosen = []

    blocks = half_video_blocks(GAME)

    for rank, query in enumerate(queries[:budget]):

        _, first, last = next(block for block in blocks if block[1] <= query['frame'] <= block[2])

        a, b = max(first, query['frame'] - 100), min(last + 1, query['frame'] + 101)

        chosen.append(dict(query, rank=rank, clip_start=a, clip_stop=b, clip=f'clips/query_{rank:02d}.mp4', samples=[]))

    wanted = {frame for item in chosen for frame in range(item['clip_start'], item['clip_stop'], 5)}

    samples = {}

    with open(data_dir / 'tactical/BAR-ATH_tactical_atd.txt') as stream:

        for line in stream:

            raw_id, players, ball = line.strip().split(':')

            frame = int(raw_id) - 1

            if frame not in wanted:

                continue

            xy = np.fromstring(players.replace(';', ','), sep=',').reshape(-1, 3)[:, :2]

            point = np.fromstring(ball, sep=',')[:2]

            finite = lambda values: np.where(np.isfinite(values), np.round(values, 4), None).tolist()

            samples[frame] = {'frame': frame, 'xy': finite(xy), 'ball': finite(point),

                              'label': str(labels[frame]), 'p': np.round(posterior[frame], 3).tolist()}

    for item in chosen:

        item['samples'] = [samples[frame] for frame in range(item['clip_start'], item['clip_stop'], 5)]

        item['video_status'] = 'not_encoded'

        item['frame_labels'] = labels[item['clip_start']:item['clip_stop']].tolist()

        item['frame_support'] = np.round(posterior[item['clip_start']:item['clip_stop']], 6).tolist()

    if encode:

        import av

        video_path = data_dir / 'tactical/BAR-ATH_tactical.mp4'

        with av.open(str(video_path)) as source:

            stream = source.streams.video[0]

            for item in chosen:

                source.seek(int((item['clip_start'] / 25) / stream.time_base), stream=stream, backward=True)

                path = output / item['clip']

                temporary = path.with_suffix('.partial.mp4')

                decoded = []

                with av.open(str(temporary), 'w') as destination:

                    target = destination.add_stream('libx264', rate=25)

                    target.width, target.height, target.pix_fmt = 960, 540, 'yuv420p'

                    target.options = {'crf': '25', 'preset': 'veryfast'}

                    for frame in source.decode(stream):

                        if frame.pts is None:

                            continue

                        index = round(float(frame.pts * stream.time_base) * 25)

                        if index < item['clip_start']:

                            continue

                        if index >= item['clip_stop']:

                            break

                        converted = frame.reformat(width=960, height=540, format='yuv420p')

                        converted.pts = len(decoded)

                        converted.time_base = Fraction(1, 25)

                        for packet in target.encode(converted):

                            destination.mux(packet)

                        decoded.append(index)

                    for packet in target.encode():

                        destination.mux(packet)

                if decoded != list(range(item['clip_start'], item['clip_stop'])):

                    raise ValueError('Video timestamps are not the expected contiguous tactical frames')

                temporary.replace(path)

                with av.open(str(path)) as check:

                    observed = sum(1 for _ in check.decode(video=0))

                if observed != len(decoded):

                    raise ValueError('Encoded review clip frame count mismatch')

                item.update(video_status='verified_decoded', encoded_frames=observed)

    payload = {'queries': chosen, 'teams': slot_teams(data_dir / 'tactical/BAR-ATH_tactical_atd.xml').tolist(),

               'fps': 25, 'sample_step': 5, 'scope': 'Private local review. Display tracks are slots, not persistent player identities. Video alignment uses tactical PTS. Pitch snapshots are sampled at 5 Hz.'}


    payload.update(schema_version=3, experiment_id=experiment_id or hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), title=title, notice=notice)

    (output / 'review_data.json').write_text(json.dumps(payload, allow_nan=False, indent=2), encoding='utf-8')

    (output / 'index.html').write_text(TEMPLATE.replace('__DATA__', json.dumps(payload, allow_nan=False).replace('</', '<\\/')), encoding='utf-8')

    return {'queries': len(chosen), 'verified_video_clips': sum(q['video_status'] == 'verified_decoded' for q in chosen),

            'schema_version': 3, 'experiment_id': payload['experiment_id'], 'pitch_snapshots': len(samples), 'path': 'review/index.html', 'scope': payload['scope']}

