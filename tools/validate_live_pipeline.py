"""
End-to-end check of the live path, minus the ESPN download.

For each published game, keep only what the live feed provides (clock,
down/distance, field position, teams, running score, gamebook text), rebuild
everything else with live_feed.add_derived_columns + nflfastR's models, and
compare with the official nflverse values.

    python tools/validate_live_pipeline.py 2025 [local.parquet]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import live_feed as lf  # noqa: E402
import nflfastr_models as nm  # noqa: E402

PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.parquet"
BASE = ["game_id", "season", "home_team", "away_team", "qtr", "time", "desc", "down",
        "ydstogo", "yardline_100", "posteam", "defteam", "drive", "total_home_score",
        "total_away_score", "roof", "location"]
CHECK = ["posteam_timeouts_remaining", "defteam_timeouts_remaining", "score_differential",
         "sp", "first_down", "third_down_converted", "third_down_failed", "fumble_lost",
         "ep", "epa", "home_wp"]


def main(season: int, path: str | None = None) -> None:
    pbp = pd.read_parquet(path or PBP_URL.format(y=season))
    models, fg = nm.load_models(), nm.load_fg_table()
    outs = []
    for _, g in pbp.groupby("game_id", sort=False):
        o = lf.add_derived_columns(g[BASE], g["season_type"].iloc[0])
        o = nm.add_ep_wp(o, models, fg)
        o.index = g.index
        outs.append(o)
    live = pd.concat(outs).loc[pbp.index]
    plays = pbp["posteam"].notna()
    print(f"{season}: {pbp['game_id'].nunique()} games, {plays.sum()} plays with an offense")
    for c in CHECK:
        a, b = live.loc[plays, c].astype(float), pbp.loc[plays, c].astype(float)
        both = a.notna() & b.notna()
        tol = 1e-3 if c in ("ep", "epa", "home_wp") else 0
        ok = ((a[both] - b[both]).abs() <= tol).sum() + (a.isna() & b.isna()).sum()
        print(f"  {c:28s} {ok / len(a):8.3%}")
    for c in ("epa", "home_wp"):
        err = (live.loc[plays, c] - pbp.loc[plays, c]).abs()
        print(f"  {c} abs error: median {err.median():.4f}  p95 {err.quantile(.95):.4f}")


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else None)
