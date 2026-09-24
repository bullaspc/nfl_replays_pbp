"""
Score live_feed.parse_play_text against nflverse's own columns.

Parses every `desc` of a published season and reports, per column, how often
the parsed value matches nflverse. Names are compared on rows where either
side has one; flags and yards on every play.

    python tools/validate_text_parser.py 2025 [local.parquet]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import live_feed as lf  # noqa: E402

PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.parquet"

NAME_COLS = [
    "passer_player_name", "rusher_player_name", "receiver_player_name",
    "solo_tackle_1_player_name", "assist_tackle_1_player_name", "assist_tackle_2_player_name",
    "sack_player_name", "half_sack_1_player_name", "half_sack_2_player_name",
    "qb_hit_1_player_name", "tackle_for_loss_1_player_name", "interception_player_name",
    "pass_defense_1_player_name", "forced_fumble_player_1_player_name",
]
FLAG_COLS = ["pass_attempt", "rush_attempt", "complete_pass", "sack", "interception",
             "touchdown", "pass_touchdown", "rush_touchdown", "penalty", "qb_kneel", "qb_spike",
             "safety", "qb_hit"]
VALUE_COLS = ["play_type", "yards_gained", "passing_yards", "rushing_yards", "receiving_yards",
              "field_goal_result", "extra_point_result", "two_point_conv_result",
              "penalty_team", "penalty_yards"]


def main(season: int, path: str | None = None, show: str | None = None) -> None:
    pbp = pd.read_parquet(path or PBP_URL.format(y=season))
    parsed = pd.DataFrame([lf.parse_play_text(d) for d in pbp["desc"]], index=pbp.index)
    print(f"{season}: {len(pbp)} plays")
    for col in NAME_COLS + VALUE_COLS + FLAG_COLS:
        a = parsed[col] if col in parsed else pd.Series(np.nan, index=pbp.index)
        b = pbp[col]
        if col in FLAG_COLS:
            a, b = a.fillna(0).astype(float), b.fillna(0).astype(float)
            rows = pd.Series(True, index=pbp.index)
        else:
            rows = a.notna() | b.notna()
            if col in ("yards_gained",):
                rows &= pbp["play_type"].isin(["pass", "run"])
        same = (a[rows] == b[rows]) | (a[rows].isna() & b[rows].isna())
        print(f"  {col:36s} {same.mean():8.3%}  of {rows.sum()}")
        if show == col:
            bad = pbp.loc[rows & ~((a == b) | (a.isna() & b.isna()))]
            print(pd.DataFrame({"nflverse": bad[col], "parsed": a[bad.index], "desc": bad["desc"]})
                  .head(25).to_string(max_colwidth=160))


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else None,
         sys.argv[3] if len(sys.argv) > 3 else None)
