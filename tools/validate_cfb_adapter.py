"""
Check cfb_feed.to_pbp against the college release's own final scores and
ESPN's team box scores.

Maps a published season to the app's layout, then reports:
- games whose last row shows the final score, and rows showing more than the
  final score (a glitch row like that would spoil the result);
- per team-game totals from the mapped plays vs the team box score. College
  stats count sacks as runs, so box rushing = our runs + sacks, and passing
  yards aren't reduced by them;
- per player-game defensive stats from games whose stat-crew text names the
  defenders vs ESPN's player box. Players are matched on team and abbreviated
  name ('Dylan Lee' → 'D.Lee'). The box also counts special teams tackles and
  tackles on plays a penalty wiped out, which the app's leader table leaves out.

    python tools/validate_cfb_adapter.py 2025 [local_pbp.parquet] [local_team_box.parquet] [local_player_box.parquet]
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
PLAYER_BOX_URL = f"{cfb_feed.RELEASE}/espn_cfb_player_box/player_box_{{y}}.parquet"
# top_defenders() leaves these out.
_ST_TYPES = {"kickoff", "punt", "field_goal", "extra_point", "no_play"}


def _pair(s: pd.Series, sep: str) -> tuple[pd.Series, pd.Series]:
    parts = s.astype(str).str.split(sep, expand=True)
    return pd.to_numeric(parts[0], errors="coerce"), pd.to_numeric(parts[1], errors="coerce")


def _defender_counts(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per game, defense and player: the columns top_defenders() counts,
    tackles split into solo and assists."""
    d = pbp[pbp["defteam"].notna() & ~pbp["play_type"].isin(_ST_TYPES)]
    stats = {
        "solo": (["solo_tackle_1_player_name"], 1.0),
        "ast": (["assist_tackle_1_player_name", "assist_tackle_2_player_name"], 1.0),
        "sacks": (["sack_player_name"], 1.0), "half_sacks": (["half_sack_1_player_name", "half_sack_2_player_name"], 0.5),
        "tfl": (["tackle_for_loss_1_player_name"], 1.0), "half_tfl": (["half_tfl_1_player_name", "half_tfl_2_player_name"], 0.5),
        "hurries": ([f"qb_hurry_{i}_player_name" for i in (1, 2, 3)], 1.0),
        "pd": (["pass_defense_1_player_name", "pass_defense_2_player_name"], 1.0),
        "int": (["interception_player_name"], 1.0),
    }
    parts = []
    for stat, (cols, w) in stats.items():
        for c in cols:
            x = d.loc[d[c].notna(), ["game_id", "defteam", c]].rename(columns={c: "name"})
            parts.append(x.assign(stat=stat, v=w))
    long = pd.concat(parts, ignore_index=True)
    out = long.pivot_table(index=["game_id", "defteam", "name"], columns="stat", values="v",
                           aggfunc="sum", fill_value=0).reset_index()
    for c in stats:
        if c not in out.columns:
            out[c] = 0.0
    out["sacks"] += out.pop("half_sacks")
    out["tfl"] += out.pop("half_tfl")
    return out


def _player_box_check(raw: pd.DataFrame, pbp: pd.DataFrame, box: pd.DataFrame) -> None:
    named = pbp.groupby("game_id")["solo_tackle_1_player_name"].transform(lambda s: s.notna().any())
    ours = _defender_counts(pbp[named])
    print(f"  player box comparison: {ours['game_id'].nunique()} of {pbp['game_id'].nunique()} games "
          "have stat-crew text naming the defenders")
    if ours.empty:
        return
    teams = pd.concat([raw[["homeTeamId", "homeTeamAbbrev"]].set_axis(["team_id", "defteam"], axis=1),
                       raw[["awayTeamId", "awayTeamAbbrev"]].set_axis(["team_id", "defteam"], axis=1)])
    teams = teams.drop_duplicates("team_id").assign(team_id=lambda t: t["team_id"].astype(int))
    num = {"totalTackles": "box_total", "soloTackles": "box_solo", "sacks": "box_sacks",
           "tacklesForLoss": "box_tfl", "hurries": "box_hurries", "passesDefended": "box_pd"}
    db = box[box["category"] == "defensive"].rename(columns=num)
    ib = box[box["category"] == "interceptions"].assign(box_int=lambda b: pd.to_numeric(b["interceptions"], errors="coerce"))
    key = ["game_id", "team_id", "athlete_name"]
    pb = db[key + list(num.values())].merge(ib[key + ["box_int"]], on=key, how="outer")
    for c in list(num.values()) + ["box_int"]:
        pb[c] = pd.to_numeric(pb[c], errors="coerce").fillna(0)
    pb["game_id"] = pb["game_id"].astype(str)
    pb["team_id"] = pd.to_numeric(pb["team_id"], errors="coerce")
    pb = pb.merge(teams, on="team_id").assign(name=lambda b: b["athlete_name"].map(cfb_feed.crew_spelling))
    pb = pb[pb["game_id"].isin(set(ours["game_id"]))]
    pb = pb.groupby(["game_id", "defteam", "name"], as_index=False)[list(num.values()) + ["box_int"]].sum()
    j = ours.merge(pb, on=["game_id", "defteam", "name"], how="outer", indicator=True)
    print(f"    player-games: {(j['_merge'] == 'both').sum()} matched, {(j['_merge'] == 'left_only').sum()} "
          f"only ours, {(j['_merge'] == 'right_only').sum()} only in the box")
    j = j.fillna({c: 0 for c in j.columns if c not in ("game_id", "defteam", "name", "_merge")})
    checks = [("solo tackles", j["solo"], j["box_solo"]), ("total tackles", j["solo"] + j["ast"], j["box_total"]),
              ("sacks", j["sacks"], j["box_sacks"]), ("TFL", j["tfl"], j["box_tfl"]),
              ("QB hurries", j["hurries"], j["box_hurries"]), ("passes defensed", j["pd"], j["box_pd"]),
              ("interceptions", j["int"], j["box_int"])]
    for label, a, b in checks:
        diff = a - b
        print(f"    {label:24s} {(diff == 0).mean():8.1%} exact, {(diff.abs() <= 1).mean():6.1%} within 1"
              f"  (ours {a.sum():.0f}, box {b.sum():.0f})")


def main(season: int, pbp_path: str | None = None, box_path: str | None = None,
         player_box_path: str | None = None) -> None:
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

    _player_box_check(raw, pbp, pd.read_parquet(player_box_path or PLAYER_BOX_URL.format(y=season)))


if __name__ == "__main__":
    main(int(sys.argv[1]), *(sys.argv[2:5]))
