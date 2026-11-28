"""
Check the Season simulation's college view against the seasons that followed.

1. Title games: each conference's two title-game teams from the regular
   season's final results (cfp_sim's standings and tiebreakers, a few
   hundred runs since a tie past the rating steps is a coin flip) against
   the title games that were played.
2. Seasons: each season simulated from the start weeks given, as the page
   does ("Start of week k": the season tables over the weeks before k,
   last season's ratings), with the model refitted without the season
   (tools/build_cfb_sim_model.py's rows from VALIDATE_CACHE, the feature set
   and g0 kept), against what happened: final wins (regular season and title
   game) inside each confidence level's range, and the chances of the title
   game, the conference title, bowl eligibility, a bowl game, the playoff
   and the national title (log loss and Brier score, next to the base rate).

    python tools/validate_cfb_sim.py [first=2024] [last=2025] [weeks=5,9,13] [pbp_dir]
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cfb_feed  # noqa: E402
import cfp_sim  # noqa: E402
import season_sim as sim  # noqa: E402
from build_cfb_sim_model import fit_model, read_pbp, tables  # noqa: E402


def held_out_model(season: int) -> tuple[dict, bool]:
    model = cfp_sim.load_model()
    cache = os.environ.get("VALIDATE_CACHE")
    files = sorted(Path(cache).glob("cfb_*_games.pkl")) if cache else []
    keep = [f for f in files if f.name.split("_")[1] != str(season)]
    if not keep:
        return model, False
    games = pd.concat([pd.read_pickle(f) for f in keep], ignore_index=True)
    teams = pd.concat([pd.read_pickle(Path(str(f).replace("_games", "_teams"))) for f in keep], ignore_index=True)
    games = games[games["margin"].notna()].reset_index(drop=True)
    refit = fit_model(games, teams, list(model["features"]), model["g0"])
    refit["postseason"] = model.get("postseason", {})  # bowls' spread (scored here on who plays, not who wins)
    return refit, True


def actual(sched: pd.DataFrame, conf: pd.DataFrame, ids: dict[int, str]) -> pd.DataFrame:
    """What happened per FBS team: wins (regular season and title game),
    title game, conference champion, bowl eligible (one FCS win counted at
    most), a bowl game outside the playoff, playoff, national champion."""
    fbs = set(conf["team_id"])
    g = cfp_sim.schedule_games(sched, fbs)
    g = g[g["played"]]
    rows = {}
    for r in conf.itertuples(index=False):
        mine = g[(g["home_id"] == r.team_id) | (g["away_id"] == r.team_id)]
        won = ((mine["home_id"] == r.team_id) & (mine["home_margin"] > 0)) | \
              ((mine["away_id"] == r.team_id) & (mine["home_margin"] < 0))
        titles = mine[mine["title"] != ""]
        t_won = won[titles.index]
        opp = np.where(mine["home_id"] == r.team_id, mine["away_id"], mine["home_id"])
        fcs_wins = int((won & ~pd.Series(opp, index=mine.index).isin(fbs)).sum())
        rows[ids.get(r.team_id, r.abbr)] = {"wins": float(won.sum()), "title_game": float(len(titles) > 0),
                                            "champion": float(t_won.any()),
                                            "eligible": float(won.sum() - max(fcs_wins - 1, 0) >= cfp_sim.BOWL_WINS)}
    out = pd.DataFrame.from_dict(rows, orient="index")
    post = sched[sched["playoff_round_name"].notna()] if "playoff_round_name" in sched.columns else sched.iloc[0:0]
    field_ids = set(post["home_id"].dropna().astype(int)) | set(post["away_id"].dropna().astype(int))
    out["playoff"] = [float(any(ids.get(i) == t for i in field_ids)) for t in out.index]
    bowls, _ = cfp_sim.postseason_games(sched, ids, {t: i for i, t in enumerate(out.index)})
    in_bowl = {out.index[i] for i in set(bowls["home"]) | set(bowls["away"])}
    out["bowl_game"] = [float(t in in_bowl) for t in out.index]
    final = post[post["playoff_round_name"] == "National Championship"]
    champ = ""
    if not final.empty and final["home_points"].notna().all():
        f = final.iloc[0]
        champ = ids.get(int(f["home_id"] if f["home_points"] > f["away_points"] else f["away_id"]), "")
    out["national"] = [float(t == champ) for t in out.index]
    return out


def title_check(season: int, pbp_dir: str | None) -> list[str]:
    """Each conference's title game from the final standings vs the real one."""
    model = cfp_sim.load_model()
    pbp, sched, ids, fbs = read_pbp(season, pbp_dir)
    ppbp, _, pids, pfbs = read_pbp(season - 1, pbp_dir)
    pt = tables(ppbp, {pids[i] for i in pfbs if i in pids})
    conf = cfp_sim.read_conferences(season)
    g = cfp_sim.schedule_games(sched, fbs)
    week = int(g.loc[g["title"] != "", "week"].min())
    # The ratings at the end of the regular season: the last tiebreak step uses them.
    off, de = tables(pbp[pbp["week"] < week], {ids[i] for i in fbs if i in ids})
    inp, _ = cfp_sim.prepare(season, sched, conf, ids, off, de, pt[0]["EPA/play"] - pt[1]["EPA/play"],
                             model, as_of=week)
    res = cfp_sim.simulate(inp, runs=300)
    lines = []
    for c in sorted(set(res.conference)):
        real = g[(g["title"] != "") & g["home_id"].isin(conf.loc[conf["conference"] == c, "team_id"])]
        if real.empty:
            continue
        pair = {ids.get(int(real.iloc[0]["home_id"])), ids.get(int(real.iloc[0]["away_id"]))}
        idx = [i for i, cc in enumerate(res.conference) if cc == c]
        two = np.argsort(~res.title_game[:, idx], axis=1, kind="stable")[:, :2]
        share = np.mean([{res.teams[idx[i]] for i in row} == pair for row in two])
        lines.append(f"{season} {c}: real {sorted(pair)}, the standings give it in {share:.0%} of runs")
    return lines


def check_season(season: int, weeks: list[int], pbp_dir: str | None) -> pd.DataFrame:
    model, held_out = held_out_model(season)
    pbp, sched, ids, fbs = read_pbp(season, pbp_dir)
    ppbp, _, pids, pfbs = read_pbp(season - 1, pbp_dir)
    pt = tables(ppbp, {pids[i] for i in pfbs if i in pids})
    prior = pt[0]["EPA/play"] - pt[1]["EPA/play"]
    conf = cfp_sim.read_conferences(season)
    new = cfp_sim.new_to_fbs(conf, [cfp_sim.read_conferences(season - 1), cfp_sim.read_conferences(season - 2)])
    truth = actual(sched, conf, ids)
    fbs_abbr = {ids[i] for i in fbs if i in ids}
    rows = []
    for k in weeks:
        t = tables(pbp[pbp["week"] < k], fbs_abbr)
        off, de = t if t else (None, None)
        inp, _ = cfp_sim.prepare(season, sched, conf, ids, off, de, prior, model, as_of=k,
                                 bowl_slots=cfp_sim.bowl_slots(sched), transition=new)
        res = cfp_sim.simulate(inp)
        tt = cfp_sim.team_table(res, 0.9).set_index("Team")
        for i, team in enumerate(res.teams):
            if team not in truth.index:
                continue
            a = truth.loc[team]
            row = {"season": season, "week": k, "team": team, "held_out": held_out, "mean": tt.at[team, "W"],
                   "actual": a["wins"], "p_title_game": tt.at[team, "Title game %"], "title_game": a["title_game"],
                   "p_champion": tt.at[team, "Conf champ %"], "champion": a["champion"],
                   "p_eligible": tt.at[team, "Eligible %"], "eligible": a["eligible"],
                   "p_bowl_game": tt.at[team, "Bowl game %"], "bowl_game": a["bowl_game"],
                   "p_playoff": tt.at[team, "Playoff %"], "playoff": a["playoff"],
                   "p_national": tt.at[team, "Champion %"], "national": a["national"]}
            w = res.wins[:, i]
            for label, level in sim.CONFIDENCE_LEVELS.items():
                lo, hi = np.floor(np.quantile(w, (1 - level) / 2)), np.ceil(np.quantile(w, (1 + level) / 2))
                row[f"in {label}"] = float(lo <= a["wins"] <= hi)
                row[f"runs {label}"] = float(((w >= lo) & (w <= hi)).mean())
            rows.append(row)
        print(f"{season} from week {k}: done{' (held out)' if held_out else ' (in sample)'}", flush=True)
    return pd.DataFrame(rows)


def _ll(p, y) -> float:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main(first: int = 2024, last: int = 2025, weeks=(5, 9, 13), pbp_dir: str | None = None) -> None:
    seasons = list(range(first, last + 1))
    for s in seasons:
        for line in title_check(s, pbp_dir):
            print(line)
    with ProcessPoolExecutor(max_workers=min(4, len(seasons))) as pool:
        df = pd.concat(pool.map(check_season, seasons, [list(weeks)] * len(seasons),
                                [pbp_dir] * len(seasons)), ignore_index=True)
    print(f"\n{first}–{last}, {'out of sample' if df['held_out'].all() else 'IN SAMPLE'}\n")
    out = []
    for k, d in df.groupby("week"):
        out.append({"from week": k, "teams": len(d), "wins MAE": (d["mean"] - d["actual"]).abs().mean(),
                    **{f"in {lab}": d[f"in {lab}"].mean() for lab in sim.CONFIDENCE_LEVELS},
                    **{f"runs {lab}": d[f"runs {lab}"].mean() for lab in sim.CONFIDENCE_LEVELS}})
    print("Final wins (regular season and title game): mean absolute error, the share inside each range "
          "(in) and the runs' own share inside it (runs):")
    print(pd.DataFrame(out).round(3).to_string(index=False))
    out = []
    for k, d in df.groupby("week"):
        for what in ("title_game", "champion", "eligible", "bowl_game", "playoff", "national"):
            y, p = d[what].to_numpy(), d[f"p_{what}"].to_numpy()
            base = np.full(len(y), y.mean())
            out.append({"from week": k, "what": what, "log loss": _ll(p, y), "base": _ll(base, y),
                        "Brier": float(np.mean((p - y) ** 2)), "base Brier": float(np.mean((base - y) ** 2))})
    print("\nChances against what happened (base: every team the same chance):")
    print(pd.DataFrame(out).round(4).to_string(index=False))
    print("\nPlayoff chances' calibration (every start week):")
    bins = pd.cut(df["p_playoff"], [-0.001, 0.0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0])
    print(df.groupby(bins, observed=True).agg(teams=("playoff", "size"), chance=("p_playoff", "mean"),
                                               made=("playoff", "mean")).round(3).to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2024, int(sys.argv[2]) if len(sys.argv) > 2 else 2025,
         [int(w) for w in sys.argv[3].split(",")] if len(sys.argv) > 3 else [5, 9, 13],
         sys.argv[4] if len(sys.argv) > 4 else None)
