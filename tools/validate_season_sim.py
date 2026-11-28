"""
Check the Season simulation page against the seasons that followed.

For each season and each start week given, the season is simulated as the
page does from "Start of week k": the season tables over the weeks before k
only (opponent-adjusted), last season's ratings, the week's injury reports
and rosters, `season_sim.SIM_RUNS` runs. The game model is refitted without
the season being checked (tools/build_sim_model.py's rows, from
VALIDATE_CACHE), so no season is scored by a model that saw it; the feature
set and g0 stay models/season_sim.json's. Then, against what happened:

* final wins: how far the average was from them (and from two simpler
  guesses: each team's win rate so far over the games left, and .500 over
  them), and how often they landed in each confidence level's range, next
  to the share of the runs inside it (a range of whole wins holds a bit
  more than its level; calibrated, the two match);
* division titles, playoff spots, byes and the title: log loss and Brier
  score of the chances, next to the base rate, and the playoff chances'
  calibration (in the runs' 0–10% bin, how many made it);
* the draft: how often the actual top-five picks were given a top-five
  chance, and the Brier score of those chances.

    python tools/validate_season_sim.py [first=2016] [last=2025] [weeks=1,5,9,13] [pbp_dir]

Needs VALIDATE_CACHE with the build_sim_model.py rows (sim_<season>_games.pkl
and sim_<season>_returns.pkl) for the out-of-sample fits; without them the
page's model is used for every season (in sample).
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nflreadpy
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import season_sim as sim  # noqa: E402
from build_sim_model import FEATURE_SETS, _tables, fit_model, prior_ratings  # noqa: E402
from validate_injuries import _season_data  # noqa: E402
from validate_matchup import read_season  # noqa: E402


def held_out_model(season: int, seasons: list[int]) -> tuple[dict, bool]:
    """The page's model refitted without `season` (out of sample), or the
    page's own where the fit's rows aren't cached."""
    model = sim.load_model()
    cache = os.environ.get("VALIDATE_CACHE")
    if not cache:
        return model, False
    paths = {s: [Path(cache) / f"sim_{s}_{k}.pkl" for k in ("games", "returns")] for s in seasons if s != season}
    if not all(p.exists() for ps in paths.values() for p in ps):
        return model, False
    games = pd.concat([pd.read_pickle(ps[0]) for ps in paths.values()], ignore_index=True)
    returns = pd.concat([pd.read_pickle(ps[1]) for ps in paths.values()], ignore_index=True)
    games = games[games["margin"].notna()].reset_index(drop=True)
    names = list(model["features"])
    assert names in FEATURE_SETS.values()
    refit, _ = fit_model(games, returns, names, model["g0"])
    return refit, True


def check_season(season: int, weeks: list[int], seasons: list[int], pbp_dir: str | None) -> pd.DataFrame:
    """One row per (start week, team): the runs' numbers and what happened."""
    path = (lambda y: f"{pbp_dir}/play_by_play_{y}.parquet") if pbp_dir else (lambda y: None)
    model, held_out = held_out_model(season, seasons)
    pbp = read_season(season, path(season))
    prior = prior_ratings(read_season(season - 1, path(season - 1)))
    sched = nflreadpy.load_schedules([season]).to_pandas()
    teams = nflreadpy.load_teams().to_pandas()
    divisions = dict(zip(teams["team_abbr"], teams["team_division"]))
    injuries, snaps, roster = _season_data(season)
    final = sim.final_standings(sched, divisions).set_index("Team")
    final["W"] = [float(r.split("-")[0]) + 0.5 * (float(r.split("-")[2]) if r.count("-") == 2 else 0)
                  for r in final["Final"]]
    picks = {}
    rows = []
    for k in weeks:
        t = _tables(pbp[pbp["week"] < k])
        off, de = (t if t else (None, None))
        inp, _, _ = sim.prepare(season, sched, off, de, prior, divisions, model, snaps, injuries, roster, as_of=k)
        res = sim.simulate(inp)
        tt = sim.team_table(res, 0.9).set_index("Team")
        played = res.played_wins + res.played_losses + res.played_ties
        rate = np.where(played > 0, (res.played_wins + 0.5 * res.played_ties) / np.maximum(played, 1), 0.5)
        n_byes = 1 if res.seed.max() >= 7 else 2
        if not picks:  # the actual draft order: the runs of a finished season agree but for coin flips
            done = sim.simulate(sim.build_input(season, sim.regular_season(sched), divisions, pd.Series(dtype=float),
                                                pd.DataFrame(), model, pd.Series(dtype=float),
                                                sim.playoff_results(sched)), runs=1)
            picks = dict(zip(done.teams, done.pick[0]))
        for i, team in enumerate(res.teams):
            f = final.loc[team]
            row = {"season": season, "week": k, "team": team, "held_out": held_out,
                   "now": res.played_wins[i] + 0.5 * res.played_ties[i],
                   "pace": res.played_wins[i] + 0.5 * res.played_ties[i] + rate[i] * (res.games_total[i] - played[i]),
                   "even": res.played_wins[i] + 0.5 * res.played_ties[i] + 0.5 * (res.games_total[i] - played[i]),
                   "mean": tt.at[team, "W"], "actual": f["W"],
                   "p_div": tt.at[team, "Division %"], "div": float(1 <= f["Seed"] <= 4),
                   "p_po": tt.at[team, "Playoffs %"], "po": float(f["Seed"] > 0),
                   "p_bye": tt.at[team, "Bye %"], "bye": float(1 <= f["Seed"] <= n_byes),
                   "p_champ": tt.at[team, "Champion %"], "champ": float(f["Stage"] == 5),
                   "p_top5": tt.at[team, "Top 5 pick %"], "top5": float(picks[team] <= 5)}
            w = res.wins[:, i]
            for label, level in sim.CONFIDENCE_LEVELS.items():
                lo, hi = np.floor(np.quantile(w, (1 - level) / 2)), np.ceil(np.quantile(w, (1 + level) / 2))
                row[f"in {label}"] = float(lo <= f["W"] <= hi)
                row[f"runs {label}"] = float(((w >= lo) & (w <= hi)).mean())  # whole wins hold more than the level
                row[f"width {label}"] = hi - lo
            rows.append(row)
        print(f"{season} from week {k}: done{' (held out)' if held_out else ' (in sample)'}", flush=True)
    return pd.DataFrame(rows)


def _ll(p, y) -> float:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main(first: int = 2016, last: int = 2025, weeks: list[int] = (1, 5, 9, 13), pbp_dir: str | None = None):
    seasons = list(range(first, last + 1))
    with ProcessPoolExecutor(max_workers=min(4, len(seasons))) as pool:
        df = pd.concat(pool.map(check_season, seasons, [list(weeks)] * len(seasons),
                                [seasons] * len(seasons), [pbp_dir] * len(seasons)), ignore_index=True)
    held = "out of sample (the game model refitted without each season)" if df["held_out"].all() \
        else "IN SAMPLE (no cached rows: the page's model for every season)"
    print(f"\n{first}–{last}, {len(seasons)} seasons, {held}\n")
    print("Final wins: mean absolute error of the runs' average, of each team's win rate so far "
          "over the games left (pace) and of .500 over them; how often the actual total landed in "
          "each range (in), the share of the runs themselves inside it (runs: a range of whole wins "
          "holds more than its level, so this is what `in` should match), and the ranges' average "
          "width:")
    out = []
    for k, d in df.groupby("week"):
        out.append({"from week": k, "teams": len(d), "runs' average": (d["mean"] - d["actual"]).abs().mean(),
                    "pace": (d["pace"] - d["actual"]).abs().mean(), ".500": (d["even"] - d["actual"]).abs().mean(),
                    **{f"in {lab}": d[f"in {lab}"].mean() for lab in sim.CONFIDENCE_LEVELS},
                    **{f"runs {lab}": d[f"runs {lab}"].mean() for lab in sim.CONFIDENCE_LEVELS},
                    **{f"width {lab}": d[f"width {lab}"].mean() for lab in sim.CONFIDENCE_LEVELS}})
    print(pd.DataFrame(out).round(3).to_string(index=False))

    print("\nChances against what happened: log loss and Brier score of the runs' chances, and of "
          "the base rate (every team the same chance):")
    out = []
    for k, d in df.groupby("week"):
        for what in ("div", "po", "bye", "champ", "top5"):
            y, p = d[what].to_numpy(), d[f"p_{what}"].to_numpy()
            base = np.full(len(y), y.mean())
            out.append({"from week": k, "what": {"div": "division", "po": "playoffs", "bye": "bye",
                                                 "champ": "champion", "top5": "top-5 pick"}[what],
                        "log loss": _ll(p, y), "base log loss": _ll(base, y),
                        "Brier": float(np.mean((p - y) ** 2)), "base Brier": float(np.mean((base - y) ** 2))})
    print(pd.DataFrame(out).round(4).to_string(index=False))

    print("\nPlayoff chances' calibration (every start week): teams per bin of the runs' chance, "
          "the average chance and the share that made it:")
    bins = pd.cut(df["p_po"], [-0.001, 0.0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 0.9999, 1.0],
                  labels=["0", "0–5%", "5–20%", "20–40%", "40–60%", "60–80%", "80–95%", "95–<100%", "100%"])
    cal = df.groupby(bins, observed=True).agg(teams=("po", "size"), chance=("p_po", "mean"), made=("po", "mean"))
    print(cal.round(3).to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2016, int(sys.argv[2]) if len(sys.argv) > 2 else 2025,
         [int(w) for w in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1, 5, 9, 13],
         sys.argv[4] if len(sys.argv) > 4 else None)
