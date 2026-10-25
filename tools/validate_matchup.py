"""
Check the Matchup page's numbers against the games that followed.

For every regular-season week from `first_week` on, the season tables are
built from the weeks before it only (as the page builds them before a game),
raw and opponent-adjusted, and each offense's line of `matchup_table()`
against that week's opponent is compared with what the offense did in the
game: the projection (Proj.: league + `PROJ_SHRINK` × the offense's and
defense's distances from it), the offense's own season value alone, and the
league average. The slope of the game's distance from the league average on
the unshrunk projection's says how much of it carries over (what
`PROJ_SHRINK` is set from). Then the projected EPA/play
margin (each offense's Proj. less its opponent's) picks a winner, next to the
betting line's favorite. NFL only. `offense_games()` is the loop, shared with
tools/validate_injuries.py.

    python tools/validate_matchup.py 2025 [path] [first_week]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import replay_core as core  # noqa: E402

PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.parquet"
STATS = ["EPA/play", "Pass EPA/play", "Rush EPA/play", "Pass SR", "Rush SR", "1st Down %",
         "3rd Down %", "Sack %", "Stuff %", "Turnovers",
         "Lost Poss."]


def read_season(season: int, path: str | None = None) -> pd.DataFrame:
    """A season's regular-season pbp, with the betting line and result."""
    pbp = pd.read_parquet(path or PBP_URL.format(y=season),
                          columns=core.PBP_COLS + ["season_type", "spread_line", "result"])
    return pbp[pbp["season_type"] == "REG"].sort_values(["game_id", "play_id"])


def offense_games(pbp: pd.DataFrame, first_week: int = 4, kinds=("raw", "adjusted"),
                  stats: list[str] = STATS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every offense-game from `first_week` on with its `matchup_table()`
    built from the weeks before: one row per (game_id, week, team, opp, kind,
    stat) with the game's actual value, Proj., the offense's and defense's
    season values and the league's. Also the games (week, teams, betting
    line, result) by game_id."""
    games = pbp.groupby("game_id")[["week", "home_team", "away_team", "spread_line", "result"]].first()
    rows = []
    for week in range(first_week, int(pbp["week"].max()) + 1):
        before, now = pbp[pbp["week"] < week], pbp[pbp["week"] == week]
        if now.empty:
            continue
        tables = {}
        raw_off, raw_def = core.season_team_table(before), core.season_team_table(before, "defteam")
        if "raw" in kinds:
            tables["raw"] = (raw_off, raw_def)
        if "adjusted" in kinds:
            adj_off, adj_def, _, _ = core.opponent_adjust(raw_off, raw_def, core.season_games(before))
            # As leagues._season_tables() does, so these are the page's tables.
            tables["adjusted"] = (core.possible_values(adj_off), core.possible_values(adj_def))
        actual = core.season_team_table(now)  # one game each: per game = the game
        for gid, g in games[games["week"] == week].iterrows():
            for team, opp in ((g["away_team"], g["home_team"]), (g["home_team"], g["away_team"])):
                for kind, (off, def_) in tables.items():
                    if team not in off.index or opp not in def_.index or team not in actual.index:
                        continue
                    m = core.matchup_table(off, def_, team, opp, stats)
                    for stat in m.index:
                        rows.append({"game_id": gid, "week": week, "team": team, "opp": opp,
                                     "kind": kind, "stat": stat, "actual": actual.at[team, stat],
                                     "proj": m.at[stat, "Proj."], "offense": m.at[stat, "Offense"],
                                     "defense": m.at[stat, "Defense"], "league": m.at[stat, "League"]})
    return pd.DataFrame(rows), games


def winner_picks(rows: pd.DataFrame, games: pd.DataFrame, proj_col: str = "proj",
                 kinds=("raw", "adjusted")) -> pd.DataFrame:
    """Per game without a tie, whether each projected EPA/play margin (home
    Proj. less away, from `proj_col`) and the betting line's favorite picked
    the winner (1/0; NaN where there's no pick)."""
    epa = rows[rows["stat"] == "EPA/play"].set_index(["game_id", "kind", "team"])[proj_col]
    picks = []
    for gid, g in games.iterrows():
        if pd.isna(g["result"]) or g["result"] == 0:
            continue
        home_won = g["result"] > 0
        pick = {"line": g["spread_line"] > 0 if g["spread_line"] != 0 else np.nan}
        for kind in kinds:
            h, a = epa.get((gid, kind, g["home_team"])), epa.get((gid, kind, g["away_team"]))
            pick[kind] = (h > a) if h is not None and a is not None else np.nan
        if all(pd.isna(pick[k]) for k in kinds):
            continue  # before first_week
        picks.append({k: (np.nan if pd.isna(v) else float(v == home_won)) for k, v in pick.items()})
    return pd.DataFrame(picks)


def main(season: int, path: str | None = None, first_week: int = 4) -> None:
    rows, games = offense_games(read_season(season, path), first_week)
    df = rows.dropna(subset=["actual", "proj"])
    n = int(((df["kind"] == "raw") & (df["stat"] == "EPA/play")).sum())
    print(f"{season}, weeks {first_week}+: {n} offense-games\n")
    print("Mean absolute error against the game (lower is better), the correlation of Proj. with it, "
          "and how much of the unshrunk projection carries over (1 = all of it)")
    out = []
    for (kind, stat), d in df.groupby(["kind", "stat"], sort=False):
        x = (d["offense"] - d["league"]) + (d["defense"] - d["league"])  # unshrunk
        y = d["actual"] - d["league"]
        out.append({"stats": kind, "stat": stat, "n": len(d),
                    "Proj.": (d["proj"] - d["actual"]).abs().mean(),
                    "offense alone": (d["offense"] - d["actual"]).abs().mean(),
                    "league average": (d["league"] - d["actual"]).abs().mean(),
                    "r(Proj.)": d["proj"].corr(d["actual"]),
                    "r(offense)": d["offense"].corr(d["actual"]),
                    "carries over": (x * y).sum() / (x ** 2).sum()})
    print(pd.DataFrame(out).set_index(["stats", "stat"]).round(3).to_string())
    p = winner_picks(rows, games)
    print(f"\nWinner picked ({len(p)} games without a tie):")
    for col, label in (("raw", "projected EPA/play margin, raw"),
                       ("adjusted", "projected EPA/play margin, opponent-adjusted"),
                       ("line", "betting line's favorite")):
        s = p[col].dropna()
        print(f"  {label}: {s.mean():.1%} of {len(s)}")


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None,
         int(sys.argv[3]) if len(sys.argv) > 3 else 4)
