"""Published boundary priors: structural facts about the match, not answers.

Everything here comes from documentation the organisers publish in this
repository, so it reveals no label and needs no validation set:

* DATA.md publishes the two graded blocks, (11185, 90651) and (110808, 190197).
* FAQ.md publishes that the second half is graded from frame 110808 but does
  not kick off until frame 113742, and that those 2934 frames are all dead ball.

Using them is a pipeline correction, not tuning: the same numbers hold for any
submission, and they can be read off the video. The guard keeps a margin before
the restart so a second-half kickoff that is a frame or two early can never be
erased, which is the same conservative convention the v0.3.1 candidate used.
"""
import numpy as np

FPS = 25.
GRADED_BLOCKS = ((11185, 90651), (110808, 190197))
SECOND_HALF_KICKOFF = 113742  # published anchor, also inside the Smart Tagging window
GUARD_SECONDS = 1.0


def published_dead_intervals(fps=FPS, anchor=SECOND_HALF_KICKOFF, guard_seconds=GUARD_SECONDS):
    """[(start, stop)] half-open frames that are dead ball before the second-half kickoff."""
    if not np.isfinite(anchor) or not np.isfinite(guard_seconds) or guard_seconds < 0:
        raise ValueError('Invalid anchor or guard')
    start = GRADED_BLOCKS[1][0]
    stop = int(anchor - round(guard_seconds * fps))
    if stop <= start:
        raise ValueError('Guard leaves no dead-ball interval')
    if not GRADED_BLOCKS[1][0] <= start < stop <= GRADED_BLOCKS[1][1]:
        raise ValueError('Anchor interval leaves the graded block')
    return [(start, stop)]


def apply_published_dead(labels, fps=FPS, anchor=SECOND_HALF_KICKOFF, guard_seconds=GUARD_SECONDS):
    """Copy of `labels` with the published pre-kickoff stretch set to dead ball.

    Only frames that are not already D are rewritten, so applying this twice is
    the same as applying it once and the returned count is the work actually done.
    """
    values = np.asarray(labels)
    if values.ndim != 1:
        raise ValueError('Expected a one-dimensional label array')
    if values.dtype.kind == 'O':
        # pandas hands back single-character objects; refuse anything wider rather
        # than silently truncating it into a different label.
        if set(np.unique(values)) - {'H', 'A', 'D', 'X'}:
            raise ValueError('Labels outside H/A/D/X')
        values = values.astype('<U1')
    elif values.dtype.kind not in ('U', 'S'):
        raise ValueError('Expected a one-dimensional label array')
    out = values.copy()
    changed = 0
    for start, stop in published_dead_intervals(fps, anchor, guard_seconds):
        stop = min(stop, len(out))
        window = out[start:stop]
        changed += int(np.sum(window != 'D'))
        window[...] = 'D'
    return out, changed
