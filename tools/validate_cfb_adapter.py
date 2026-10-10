"""
Check cfb_feed.to_pbp against the college release's own final scores and
ESPN's team box scores.

Maps a published season to the app's layout, then reports:
- games whose last row shows the final score, and rows showing more than the
  final score (a glitch row like that would spoil the result);
- per team-game totals from the mapped plays vs the team box score. College
  stats count sacks as runs, so box rushing = our runs + sacks, and passing
  yards aren't reduced by them.

    python tools/validate_cfb_adapter.py 2025 [local_pbp.parquet] [local_team_box.parquet]
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import cfb_feed  # noqa: E402
from replay_core import PBP_COLS  # noqa: E402

PBP_URL = f"{cfb_feed.RELEASE}/espn_cfb_pbp/play_by_play_{{y}}.parquet"
BOX_URL = f"{cfb_feed.RELEASE}/espn_cfb_team_box/team_box_{{y}}.parquet"


def _pair(s: pd.Series, sep: str) -> tuple[pd.Series, pd.Series]:
    parts = s.astype(str).str.split(sep, expand=True)
    return pd.to_numeric(parts[0], errors="coerce"), pd.to_numeric(parts[1], errors="coerce")


def main(season: int, pbp_path: str | None = None, box_path: str | None = None) -> None:
    src = pbp_path or PBP_URL.format(y=season)
    raw = pd.read_parquet(src, columns=cfb_feed.RAW_COLS)
    finals = pd.read_parquet(src, columns=["game_id", "homeFinalScore", "awayFinalScore"])
    finals = finals.drop_duplicates("game_id").assign(game_id=lambda d: d["game_id"].astype(str))
    pbp = cfb_feed.to_pbp(raw, PBP_COLS)
    print(f"{season}: {len(raw)} feed rows → {len(pbp)} rows, {pbp['game_id'].nunique()} games")
    print(f"  play_type unset on {pbp['play_type'].isna().mean():.1%} of rows")

    m = pbp.merge(finals, on="game_id")
    last = m.groupby("game_id").tail(1)
    ok = (last["total_home_score"] == last["homeFinalScore"]) & (last["total_away_score"] == last["awayFinalScore"])
    over = (m["total_home_score"] > m["homeFinalScore"]) | (m["total_away_score"] > m["awayFinalScore"])
    print(f"  last row shows the final score: {ok.mean():.2%} of {len(last)} games")
    print(f"  rows showing more than the final score: {int(over.sum())} in "
          f"{m.loc[over, 'game_id'].nunique()} games")

    box = pd.read_parquet(box_path or BOX_URL.format(y=season))
    box["game_id"] = box["game_id"].astype(str)
    box["cmp"], box["att"] = _pair(box["completionAttempts"], "/")
    box["conv3"], box["att3"] = _pair(box["thirdDownEff"], "-")
    box["pens"], box["pen_yds"] = _pair(box["totalPenaltiesYards"], "-")
    for c in ("rushingYards", "rushingAttempts", "netPassingYards", "interceptions", "fumblesLost"):
        box[c] = pd.to_numeric(box[c], errors="coerce")

    p = pbp[pbp["posteam"].notna()]
    sack = p["sack"] == 1
    sack_yds = p["yards_gained"].where(sack, 0).fillna(0)
    ours = pd.DataFrame({
        "game_id": p["game_id"], "team": p["posteam"],
        "cmp": p["complete_pass"], "att": p["pass_attempt"] - p["sack"],
        "rush_att": p["rush_attempt"] + p["sack"],
        "rush_yds": p["rushing_yards"].fillna(0) + sack_yds,
        "pass_yds": p["passing_yards"].fillna(0),
        "int": p["interception"], "fum": p["fumble_lost"],
        "conv3": p["third_down_converted"],
        "att3": p["third_down_converted"] + p["third_down_failed"],
    }).groupby(["game_id", "team"], as_index=False).sum()
    pen = (pbp[pbp["penalty"] == 1].groupby(["game_id", "penalty_team"])
           .agg(pens=("penalty", "size"), pen_yds=("penalty_yards", "sum"))
           .rename_axis(["game_id", "team"]).reset_index())
    ours = ours.merge(pen, on=["game_id", "team"], how="left").fillna({"pens": 0, "pen_yds": 0})
    j = ours.merge(box, left_on=["game_id", "team"], right_on=["game_id", "team_abbreviation"],
                   suffixes=("", "_box"))
    print(f"  team box comparison: {len(j)} team-games")
    checks = [
        ("completions", "cmp", "cmp_box", 0), ("pass attempts", "att", "att_box", 0),
        ("passing yds (±5)", "pass_yds", "netPassingYards", 5),
        ("rush attempts", "rush_att", "rushingAttempts", 0), ("rushing yds (±5)", "rush_yds", "rushingYards", 5),
        ("interceptions", "int", "interceptions", 0), ("fumbles lost", "fum", "fumblesLost", 0),
        ("3rd-down conversions", "conv3", "conv3_box", 0), ("3rd-down attempts", "att3", "att3_box", 0),
        ("penalties", "pens", "pens_box", 0), ("penalty yds", "pen_yds", "pen_yds_box", 0),
    ]
    for label, a, b, tol in checks:
        diff = j[a] - j[b]
        hit = (diff.abs() <= tol).mean()
        print(f"    {label:24s} {hit:8.1%}  median |diff| {diff.abs().median():.0f}")



if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else None,
         sys.argv[3] if len(sys.argv) > 3 else None)
