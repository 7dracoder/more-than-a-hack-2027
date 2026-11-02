"""Parse tactical and broadcast BAR-ATH tracking into metre arrays (cached under /tmp, never committed)."""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from games import GAMES, asset_paths  # noqa: E402
from possession.data import slot_teams  # noqa: E402


def team_names(meta):
    return [t.findtext("Name") or t.get("name") or t.get("id") for t in ET.parse(meta).getroot().iter("Team")]


def load(source, data_dir):
    cache = Path(f"/tmp/bar_ath_{source}.npz")
    paths = asset_paths(data_dir=str(data_dir), source=source)
    teams = slot_teams(paths["atd_meta"])
    pitch = np.asarray(GAMES["BAR-ATH"]["pitch"], float)
    if cache.exists():
        z = np.load(cache)
        return z["players"], z["ball"], teams, pitch, z["valid"]
    lines = Path(paths["atd_raw"]).read_text().splitlines()
    players = np.full((len(lines), len(teams), 2), np.nan, np.float32)
    ball = np.full((len(lines), 2), np.nan, np.float32)
    for i, line in enumerate(lines):
        _, p, b = line.strip().split(":")
        players[i] = np.fromstring(p.replace(";", ","), sep=",").reshape(-1, 3)[:, :2]
        ball[i] = np.fromstring(b, sep=",")[:2]
    players *= pitch.astype(np.float32)
    ball *= pitch.astype(np.float32)
    h = np.genfromtxt(paths["homography"], delimiter=",", names=True)
    valid = h["valid"].astype(bool)
    np.savez(cache, players=players, ball=ball, valid=valid)
    return players, ball, teams, pitch, valid
