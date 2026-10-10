"""
Check the live college path (cfb_feed.live_raw → to_pbp) against the
published release, without ESPN.

Rebuilds ESPN-shaped summary JSON from published games (the release keeps
ESPN's own play fields: text, type, clock, start/end spots, scores, drive
metadata), runs it through sportsdataverse's processing exactly as a live game
is, and compares with the published game mapped by to_pbp:

- whole games: row counts and final scores, then, on plays matched by
  quarter, clock and text, play type, down/distance/spot, possession and
  score, and how far EPA and win probability are apart. They come from the
  same models but aren't identical: the release may be built with another
  sportsdataverse version, and the rebuilt JSON starts from the release's
  processed spots rather than ESPN's own;
- games cut off mid-way (an in-progress feed): the output must stop at the
  cut, with no later play, no later score and no END GAME row.

The spread comes from the release (odds_override), so nothing is fetched.

    python tools/validate_cfb_live.py 2026 [local_pbp.parquet] [n_games]
"""
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cfb_feed  # noqa: E402
from replay_core import PBP_COLS  # noqa: E402

PBP_URL = f"{cfb_feed.RELEASE}/espn_cfb_pbp/play_by_play_{{y}}.parquet"

# ESPN play fields as the release stores them, flattened. orig_play_type is
# ESPN's type.text before sportsdataverse relabels it.
PLAY_KEYS = [
    "id", "sequenceNumber", "text", "awayScore", "homeScore", "scoringPlay", "priority", "modified",
    "wallclock", "statYardage", "type.id", "orig_play_type", "type.abbreviation", "period.number",
    "clock.displayValue", "start.down", "start.distance", "start.yardLine", "start.yardsToEndzone",
    "start.team.id", "start.downDistanceText", "start.shortDownDistanceText", "start.possessionText",
    "end.down", "end.distance", "end.yardLine", "end.yardsToEndzone", "end.team.id",
    "end.downDistanceText", "end.shortDownDistanceText", "end.possessionText",
    "scoringType.name", "scoringType.displayName", "scoringType.abbreviation",
    "pointAfterAttempt.id", "pointAfterAttempt.text", "pointAfterAttempt.abbreviation",
    "pointAfterAttempt.value",
]
DRIVE_KEYS = [
    "drive.id", "drive.displayResult", "drive.isScore", "drive.team.shortDisplayName",
    "drive.team.displayName", "drive.team.name", "drive.team.abbreviation", "drive.yards",
    "drive.offensivePlays", "drive.result", "drive.description", "drive.shortDisplayResult",
    "drive.timeElapsed.displayValue", "drive.start.period.number", "drive.start.period.type",
    "drive.start.yardLine", "drive.start.clock.displayValue", "drive.start.text",
    "drive.end.period.number", "drive.end.period.type", "drive.end.yardLine",
    "drive.end.clock.displayValue",
]
GAME_KEYS = [
    "season", "seasonType", "week", "homeTeamId", "awayTeamId", "homeTeamName", "awayTeamName",
    "homeTeamMascot", "awayTeamMascot", "homeTeamAbbrev", "awayTeamAbbrev",
    "gameSpread", "homeFavorite", "overUnder", "gameSpreadAvailable", "status_type_completed",
]
COLS = ["game_id", "game_play_number"] + PLAY_KEYS + DRIVE_KEYS + GAME_KEYS


def _plain(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if math.isnan(v) else float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _nest(d: dict, key: str, v) -> None:
    *path, last = key.split(".")
    for p in path:
        d = d.setdefault(p, {})
    d[last] = v


def espn_summary(rows: pd.DataFrame, completed: bool) -> tuple[dict, dict]:
    """ESPN-shaped summary from one game's release rows, in published order.
    Returns (summary, odds_override)."""
    g = rows.iloc[0]
    drives, cur = [], None
    for _, r in rows.iterrows():
        play = {}
        for k in PLAY_KEYS:
            v = _plain(r[k])
            if v is not None:
                _nest(play, "type.text" if k == "orig_play_type" else k, v)
        if cur is None or cur["id"] != str(r["drive.id"]):
            cur = {"plays": []}
            for k in DRIVE_KEYS:
                v = _plain(r[k])
                if v is not None:
                    _nest(cur, k[len("drive."):], v)
            cur["id"] = str(r["drive.id"])
            drives.append(cur)
        cur["plays"].append(play)

    def team(side):
        return {"homeAway": side, "team": {
            "id": str(int(g[f"{side}TeamId"])), "name": g[f"{side}TeamMascot"],
            "location": g[f"{side}TeamName"], "abbreviation": g[f"{side}TeamAbbrev"]}}

    header = {"id": str(int(g["game_id"])), "week": int(g["week"]),
              "season": {"year": int(g["season"]), "type": int(g["seasonType"])},
              "competitions": [{"id": str(int(g["game_id"])), "playByPlaySource": "full",
                                "boxscoreSource": "full", "competitors": [team("home"), team("away")],
                                "status": {"type": {"completed": completed,
                                                    "state": "post" if completed else "in"}}}]}
    summary = json.loads(json.dumps({"header": header, "drives": {"previous": drives}}, default=_plain))
    odds = {"gameSpread": float(g["gameSpread"]), "overUnder": float(g["overUnder"]),
            "homeFavorite": bool(g["homeFavorite"]), "gameSpreadAvailable": bool(g["gameSpreadAvailable"])}
    return summary, odds


def _live(rows: pd.DataFrame, completed: bool) -> pd.DataFrame:
    summary, odds = espn_summary(rows, completed)
    raw = cfb_feed.live_raw(summary, rows["game_id"].iloc[0], odds=odds)
    return cfb_feed.to_pbp(raw, PBP_COLS)


def _matched(live: pd.DataFrame, pub: pd.DataFrame) -> pd.DataFrame:
    """Plays in both, matched on quarter, clock and text (nth occurrence)."""
    key = ["qtr", "time", "desc"]
    a = live.assign(_n=live.groupby(key, dropna=False).cumcount())
    b = pub.assign(_n=pub.groupby(key, dropna=False).cumcount())
    return a.merge(b, on=key + ["_n"], suffixes=("_l", "_p"))


def main(season: int, path: str | None = None, n_games: int = 40) -> None:
    warnings.filterwarnings("ignore")
    src = path or PBP_URL.format(y=season)
    rel = pd.read_parquet(src, columns=sorted(set(COLS) | set(cfb_feed.RAW_COLS)))
    ids = rel["game_id"].drop_duplicates().sample(min(n_games, rel["game_id"].nunique()), random_state=1)
    rng = np.random.default_rng(1)
    row_diff = []
    agree = {k: [] for k in ("play_type", "down", "ydstogo", "yardline_100", "posteam",
                             "total_home_score", "total_away_score")}
    gaps = {"epa": [], "home_wp": []}
    finals = []
    cut_ok, cut_fail = 0, []
    for gid in ids:
        rows = rel[rel["game_id"] == gid].sort_values("game_play_number")
        pub = cfb_feed.to_pbp(rows[cfb_feed.RAW_COLS], PBP_COLS)
        live = _live(rows, completed=bool(rows["status_type_completed"].iloc[0]))
        finals.append(live[["total_home_score", "total_away_score"]].iloc[-1].tolist()
                      == pub[["total_home_score", "total_away_score"]].iloc[-1].tolist())
        row_diff.append(len(live) - len(pub))
        m = _matched(live, pub)
        for k in agree:
            a, b = m[f"{k}_l"], m[f"{k}_p"]
            agree[k].append(((a == b) | (a.isna() & b.isna())).to_numpy())
        for k in gaps:
            gaps[k].append((m[f"{k}_l"] - m[f"{k}_p"]).abs().dropna().to_numpy())
        # An in-progress feed: everything up to a random play.
        cut = int(rng.integers(20, len(rows) - 5))
        part = _live(rows.iloc[:cut], completed=False)
        last_play = rows.iloc[cut - 1]
        leaked = (part["desc"].fillna("").str.upper().str.startswith("END GAME").any()
                  or part["total_home_score"].max() > last_play["homeScore"]
                  or part["total_away_score"].max() > last_play["awayScore"])
        if leaked:
            cut_fail.append(gid)
        else:
            cut_ok += 1
    n = len(ids)
    print(f"{season}: {n} games rebuilt as ESPN JSON and run through the live path")
    print(f"  rows vs published: {pd.Series(row_diff).value_counts().sort_index().to_dict()} (difference: games)")
    print(f"  same final score as published: {sum(finals)}/{n}")
    print("  on plays matched by quarter, clock and text:")
    for k, v in agree.items():
        print(f"    {k:18s} {np.concatenate(v).mean():8.2%} equal")
    for k, v in gaps.items():
        d = np.concatenate(v)
        print(f"    {k:18s} |live - published|: median {np.median(d):.3f}, 90th percentile {np.quantile(d, .9):.3f}")
    print(f"  games cut off mid-way: {cut_ok}/{n} stop at the cut with no later score or END GAME"
          + (f"; leaked: {cut_fail}" if cut_fail else ""))


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else None,
         int(sys.argv[3]) if len(sys.argv) > 3 else 40)
