"""Keep historical public-label experiments out of the competition workflow."""

HISTORICAL_WARNING = (
    'Historical reproduction only: this workflow trains/tunes on BAR-ATH public labels. '
    'The October 23 organizer rules prohibit this for competition entries. '
    'Use scripts/run_local.py with external supervision for the current workflow. '
    'Pass --historical-reproduction only to audit past research, never to produce an entry.'
)


def require_historical_opt_in(enabled):
    if not enabled:
        raise ValueError(HISTORICAL_WARNING)

