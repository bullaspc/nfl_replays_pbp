"""
Fit the Season simulation's college model (cfp_sim.py) and write it to
models/season_sim_cfb.json: the game model, the FCS model and the committee.

The game model is tools/build_sim_model.py's for college: for every season
and week k, the season tables over the regular-season weeks before k
(opponent-adjusted over every team in the plays, FCS opponents too, then the
FBS teams kept, as leagues._season_tables builds them), and every game from
week k on (conference title games included) projected from them. A team's
part of each feature is `PROJ_SHRINK` × (its offense's value less what its
defense allowed), so a game's feature is the Matchup page's Proj. of the
home offense less the away offense's. This season's part is weighted by the
team's games / (games + g0), last season's net EPA/play by the rest; the
feature set and g0 are picked by log loss a season at a time out of sample.
Then the rating error by games played (`tau`, from the covariance of a
team's residuals over its games ahead) and a game's own noise.

FCS: a game against an FCS team gives the FBS team's margin as intercept +
slope × its power rating, with its own spread.

The committee: each season's final CFP ranking (ESPN's, 2016–2025) against
every FBS team's résumé at the end of the regular season (title games in,
bowls out): wins, losses, power rating, schedule (its opponents' average
rating), a conference title, a Power conference (or Notre Dame). A pairwise
logistic model, P(i ranked above j) = σ(score_i − score_j), over every pair
of ranked teams and every ranked/unranked pair, the features standardized.
Checked a season at a time out of sample: how many of the committee's top
12 and top 4 the model's ranking had.

    python tools/build_cfb_sim_model.py [first=2016] [last=2025] [dir]

`dir` holds sportsdataverse's play_by_play_<season>.parquet (season first − 1
too, for its prior ratings); without it they're downloaded. Set
VALIDATE_CACHE to a folder to keep each season's rows there.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cfb_feed  # noqa: E402
import cfp_sim  # noqa: E402
import replay_core as core  # noqa: E402
import season_sim as sim  # noqa: E402
from build_sim_model import _scores  # noqa: E402

CANDIDATES = {"epa": ("EPA/play", 1), "pass_epa": ("Pass EPA/play", 1), "rush_epa": ("Rush EPA/play", 1),
              "pts": ("Pts/Drive", 1), "sr": ("ED SR", 1), "to": ("Turnovers", -1), "ypp": ("Yds/Play", 1)}
FEATURE_SETS = {"EPA/play": ["epa"], "EPA/play + Pts/Drive": ["epa", "pts"],
                "EPA/play + Pts/Drive + Turnovers": ["epa", "pts", "to"],
                "Pass/Rush EPA + Pts/Drive": ["pass_epa", "rush_epa", "pts"],
                "EPA/play + Pts/Drive + ED SR + Turnovers": ["epa", "pts", "sr", "to"]}
G0_GRID = [1, 2, 3, 4, 6, 8]
XRUSH = Path(__file__).resolve().parent.parent / "models" / "xrush_yards.csv"


def read_pbp(season: int, pbp_dir: str | None) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str], set[int]]:
    """A season's finished games with an FBS team, mapped (as
    leagues._cfb_season_pbp does), its schedule, ESPN id → abbreviation, and
    the FBS team ids."""
    raw = (cfb_feed.read_season_file(f"{pbp_dir}/play_by_play_{season}.parquet") if pbp_dir
           else cfb_feed.read_season(season))
    sched = cfb_feed.read_schedule(season)
    done = raw.groupby("game_id")["status_type_completed"].transform(lambda s: s.eq(True).all())
    raw = raw[done]
    fbs = (set(sched.loc[sched["home_division"] == "fbs", "home_id"].dropna().astype(int))
           | set(sched.loc[sched["away_division"] == "fbs", "away_id"].dropna().astype(int)))
    raw = raw[raw["homeTeamId"].isin(fbs) | raw["awayTeamId"].isin(fbs)]
    ids = cfb_feed.team_ids(raw, sched)
    xr = pd.read_csv(XRUSH)
    pbp = core.add_expected_rush(cfb_feed.to_pbp(raw, core.PBP_COLS), xr[xr["league"] == "CFB"].drop(columns="league"))
    post = set(raw.loc[raw["seasonType"] == 3, "game_id"].astype(str))
    pbp = pbp[~pbp["game_id"].astype(str).isin(post)]
    return pbp, sched, dict(zip(ids["id"], ids["abbr"])), fbs


def tables(before: pd.DataFrame, fbs_abbr: set[str]) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """The FBS teams' opponent-adjusted season tables from `before`, as
    leagues._build_season_tables makes them (every team in the adjustment)."""
    if before.empty:
        return None
    off, de = core.season_team_table(before), core.season_team_table(before, "defteam")
    off, de, _, _ = core.opponent_adjust(off, de, core.season_games(before))
    off, de = core.possible_values(off), core.possible_values(de)
    off, de = off[off.index.isin(fbs_abbr)], de[de.index.isin(fbs_abbr)]
    return core.season_qualified(off), core.season_qualified(de)


def team_x(t: tuple[pd.DataFrame, pd.DataFrame] | None) -> pd.DataFrame:
    """Each team's part of every candidate feature."""
    if t is None:
        return pd.DataFrame()
    off, de = t
    return pd.DataFrame({f: sign * core.PROJ_SHRINK * (off[stat] - de[stat].reindex(off.index))
                         for f, (stat, sign) in CANDIDATES.items()})


def collect(season: int, pbp_dir: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every (as-of week, game from that week on) of `season` with both
    teams' feature parts, games played, prior ratings and FBS flags, and the
    game; and the season's final résumé per FBS team (for the committee)."""
    cache = os.environ.get("VALIDATE_CACHE")
    paths = [Path(cache) / f"cfb_{season}_{k}.pkl" for k in ("games", "teams")] if cache else None
    if paths and all(p.exists() for p in paths):
        return pd.read_pickle(paths[0]), pd.read_pickle(paths[1])
    pbp, sched, ids, fbs = read_pbp(season, pbp_dir)
    ppbp, _, pids, pfbs = read_pbp(season - 1, pbp_dir)
    pt = tables(ppbp, {pids[i] for i in pfbs if i in pids})
    prior = pt[0]["EPA/play"] - pt[1]["EPA/play"]
    fbs_abbr = {ids[i] for i in fbs if i in ids}
    # A team new to FBS gets a low quantile of last season's ratings, as cfp_sim.power_ratings does.
    prior = pd.concat([prior, pd.Series(prior.quantile(cfp_sim.NEW_TEAM_PRIOR_Q),
                                        index=sorted(fbs_abbr - set(prior.index)))])
    games = cfp_sim.schedule_games(sched, fbs)
    title_rows = games[games["title"] != ""]
    games = games[games["played"]].copy()
    games["home"], games["away"] = games["home_id"].map(ids), games["away_id"].map(ids)
    games = games.dropna(subset=["home", "away"])
    games["home_fbs"], games["away_fbs"] = games["home_id"].isin(fbs), games["away_id"].isin(fbs)
    rows = []
    for k in range(1, int(games["week"].max()) + 1):
        ahead = games[games["week"] >= k]
        if ahead.empty:
            continue
        x = team_x(tables(pbp[pbp["week"] < k], fbs_abbr))
        done = games[games["week"] < k]
        played = pd.concat([done["home"], done["away"]]).value_counts()
        for g in ahead.itertuples(index=False):
            row = {"season": season, "k": k, "game_id": g.game_id, "week": g.week, "home": g.home,
                   "away": g.away, "home_fbs": g.home_fbs, "away_fbs": g.away_fbs, "neutral": g.neutral,
                   "title": g.title, "margin": g.home_margin,
                   "games_home": played.get(g.home, 0), "games_away": played.get(g.away, 0),
                   "prior_home": prior.get(g.home, 0.0), "prior_away": prior.get(g.away, 0.0)}
            for f in CANDIDATES:
                row[f"{f}_home"] = x[f].get(g.home, np.nan) if not x.empty else np.nan
                row[f"{f}_away"] = x[f].get(g.away, np.nan) if not x.empty else np.nan
            rows.append(row)
        print(f"{season} week {k}: {len(ahead)} games ahead", flush=True)
    # The final résumé: the regular season with its title games, before bowls.
    x = team_x(tables(pbp, fbs_abbr))
    w = pd.concat([games.loc[games["home_margin"] > 0, "home"], games.loc[games["home_margin"] < 0, "away"]]).value_counts()
    lo = pd.concat([games.loc[games["home_margin"] < 0, "home"], games.loc[games["home_margin"] > 0, "away"]]).value_counts()
    champs = {ids.get(h if m > 0 else a) for h, a, m in zip(title_rows["home_id"], title_rows["away_id"],
                                                           title_rows["home_margin"]) if pd.notna(m)}
    conf = cfp_sim.read_conferences(season)
    conf["abbr"] = conf["team_id"].map(ids).fillna(conf["abbr"])
    final = cfp_sim.final_cfp_ranking(season)
    team_rows = []
    for r in conf.itertuples(index=False):
        t = r.abbr
        mine = games[(games["home"] == t) | (games["away"] == t)]
        opps = [o for o in np.where(mine["home"] == t, mine["away"], mine["home"])]
        team_rows.append({"season": season, "team": t, "conference": r.conference,
                          "wins": w.get(t, 0), "losses": lo.get(t, 0), "games": len(mine),
                          "champion": float(t in champs), "opps": opps,
                          "prior": prior.get(t, 0.0), "rank": final.get(r.team_id, np.nan),
                          **{f: (x[f].get(t, np.nan) if not x.empty else np.nan) for f in CANDIDATES}})
    out = pd.DataFrame(rows), pd.DataFrame(team_rows)
    if paths:
        paths[0].parent.mkdir(parents=True, exist_ok=True)
        out[0].to_pickle(paths[0])
        out[1].to_pickle(paths[1])
    return out


def design(g: pd.DataFrame, names: list[str], g0: float) -> np.ndarray:
    """Home field, each feature's home-less-away (each team's part weighted by
    its own games / (games + g0)) and last season's (by the rest)."""
    wh = sim.data_weight(g["games_home"].to_numpy(), g0)
    wa = sim.data_weight(g["games_away"].to_numpy(), g0)
    cols = [1 - g["neutral"].astype(float).to_numpy()]
    cols += [wh * g[f"{f}_home"].fillna(0).to_numpy() - wa * g[f"{f}_away"].fillna(0).to_numpy() for f in names]
    cols.append((1 - wh) * g["prior_home"].fillna(0).to_numpy() - (1 - wa) * g["prior_away"].fillna(0).to_numpy())
    return np.column_stack(cols)


def powers(rows: pd.DataFrame, side: str, names: list[str], g0: float, b: np.ndarray) -> np.ndarray:
    """A side's power rating (points over an average FBS team) under the
    fitted coefficients `b` (home field first, last season's last)."""
    w = sim.data_weight(rows[f"games_{side}"].to_numpy(), g0)
    cur = sum(c * rows[f"{f}_{side}"].fillna(0).to_numpy() for f, c in zip(names, b[1:-1]))
    return w * cur + (1 - w) * b[-1] * rows[f"prior_{side}"].fillna(0).to_numpy()


def fit_committee(teams: pd.DataFrame, seasons: list[int], verbose: bool = True) -> dict:
    """The pairwise committee model on `teams` (with power, sos), and its
    out-of-sample check by season."""
    feats = cfp_sim.COMMITTEE_FEATURES
    X = teams[feats].to_numpy(float)
    mean, sd = X.mean(0), X.std(0)
    Z = (X - mean) / sd

    def pairs(mask):
        d, y = [], []
        for s in np.unique(teams.loc[mask, "season"]):
            m = mask & (teams["season"] == s).to_numpy()
            idx = np.flatnonzero(m)
            r = teams["rank"].to_numpy()[idx]
            for a_i, a in enumerate(idx):
                if np.isnan(r[a_i]):
                    continue
                for b_i, b in enumerate(idx):
                    if a == b or (not np.isnan(r[b_i]) and r[b_i] <= r[a_i]):
                        continue
                    d.append(Z[a] - Z[b]); y.append(1.0)
                    d.append(Z[b] - Z[a]); y.append(0.0)
        return np.array(d), np.array(y)

    def fit(mask):
        d, y = pairs(mask)
        return core._logit_fit(d, y, l2=1e-3)

    report = []
    for s in seasons:
        test = (teams["season"] == s).to_numpy()
        beta = fit(~test)
        t = teams[test].assign(score=Z[test] @ beta).sort_values("score", ascending=False)
        pos = pd.Series(np.arange(1, len(t) + 1), index=t["team"])
        top = t[t["rank"] <= 25]
        report.append({"season": s,
                       "top 4 of 4": len(set(t.head(4)["team"]) & set(top[top["rank"] <= 4]["team"])),
                       "top 12 of 12": len(set(t.head(12)["team"]) & set(top[top["rank"] <= 12]["team"])),
                       "top 25 of 25": len(set(t.head(25)["team"]) & set(top["team"])),
                       "rank corr": float(top["rank"].corr(top["team"].map(pos), method="spearman"))})
    rep = pd.DataFrame(report)
    if verbose:
        print("\nThe committee model, each season ranked by a model fitted on the others: how many of the "
              "committee's final top 4 / 12 / 25 its ranking had, and the rank correlation over the "
              "committee's 25:")
        print(rep.round(3).to_string(index=False))
        print(rep.drop(columns="season").mean().round(3).to_string())
    beta = fit(np.ones(len(teams), bool))
    if verbose:
        print("Coefficients (per SD):", dict(zip(feats, np.round(beta, 3))))
    return {"coef": {f: round(float(b), 4) for f, b in zip(feats, beta)},
            "mean": [round(float(m), 4) for m in mean], "sd": [round(float(v), 4) for v in sd],
            "check": {k: round(float(v), 3) for k, v in rep.drop(columns="season").mean().items()}}


def committee_rows(teams: pd.DataFrame, names: list[str], g0: float, b: np.ndarray, fcs_power: float) -> pd.DataFrame:
    """`teams` with power (the fitted ratings, from the final tables) and sos
    (the opponents' average power, an FCS opponent at `fcs_power`), power4."""
    t = teams.copy()
    w = sim.data_weight(t["games"].to_numpy(), g0)
    cur = sum(c * t[f].fillna(0).to_numpy() for f, c in zip(names, b[1:-1]))
    t["power"] = w * cur + (1 - w) * b[-1] * t["prior"].fillna(0).to_numpy()
    by = {(s, tm): p for s, tm, p in zip(t["season"], t["team"], t["power"])}
    t["sos"] = [np.mean([by.get((s, o), fcs_power) for o in opps]) if len(opps) else fcs_power
                for s, opps in zip(t["season"], t["opps"])]
    t["power4"] = [float(c in cfp_sim.POWER4 or (c == "Pac-12" and s < 2024) or tm == "ND")
                   for c, s, tm in zip(t["conference"], t["season"], t["team"])]
    return t


def fit_model(games: pd.DataFrame, teams: pd.DataFrame, names: list[str], g0: float,
              verbose: bool = False) -> dict:
    """The model (season_sim_cfb.json's fields but "fit") from `games` and
    `teams` rows."""
    from build_sim_model import fit_tau
    fbs = games[games["home_fbs"] & games["away_fbs"]].reset_index(drop=True)
    X, y = design(fbs, names, g0), fbs["margin"].to_numpy(float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    tau = fit_tau(fbs, resid, verbose)
    t_home = sim.team_sd({"tau": tau}, fbs["games_home"].to_numpy())
    t_away = sim.team_sd({"tau": tau}, fbs["games_away"].to_numpy())
    game_sd = float(np.sqrt(max(np.mean(resid ** 2 - t_home ** 2 - t_away ** 2), 1.0)))
    # FCS: the FBS team's margin on its power rating.
    one = games[games["home_fbs"] != games["away_fbs"]].reset_index(drop=True)
    home_is_fbs = one["home_fbs"].to_numpy()
    p_fbs = np.where(home_is_fbs, powers(one, "home", names, g0, b), powers(one, "away", names, g0, b))
    m_fbs = np.where(home_is_fbs, one["margin"], -one["margin"]).astype(float)
    adv = np.where(one["neutral"], 0.0, np.where(home_is_fbs, b[0], -b[0]))
    A = np.column_stack([np.ones(len(one)), p_fbs])
    c, *_ = np.linalg.lstsq(A, m_fbs - adv, rcond=None)
    fcs_res = m_fbs - adv - A @ c
    t_fbs = sim.team_sd({"tau": tau}, np.where(home_is_fbs, one["games_home"], one["games_away"]))
    fcs_sd = float(np.sqrt(max(np.mean(fcs_res ** 2 - t_fbs ** 2), 1.0)))
    model = {
        "features": {n: {"stat": CANDIDATES[n][0], "sign": CANDIDATES[n][1], "coef": round(float(v), 3)}
                     for n, v in zip(names, b[1:-1])},
        "prior": {"stat": "net EPA/play, last season", "coef": round(float(b[-1]), 3)},
        "g0": g0, "hfa": round(float(b[0]), 3), "game_sd": round(game_sd, 3), "tau": tau,
        "fcs": {"intercept": round(float(c[0]), 3), "slope": round(float(c[1]), 3), "sd": round(fcs_sd, 3)},
    }
    model["committee"] = fit_committee(committee_rows(teams, names, g0, b, -float(c[0])),
                                       sorted(teams["season"].unique()), verbose)
    if verbose:
        print(f"\nFCS games ({len(one)}): the FBS side by {c[0]:.1f} + {c[1]:.2f} × its power, SD {fcs_sd:.1f}")
    return model


def main(first: int = 2016, last: int = 2025, pbp_dir: str | None = None) -> None:
    seasons = list(range(first, last + 1))
    with ProcessPoolExecutor(max_workers=min(4, len(seasons))) as pool:
        parts = list(pool.map(collect, seasons, [pbp_dir] * len(seasons)))
    games = pd.concat([p[0] for p in parts], ignore_index=True)
    teams = pd.concat([p[1] for p in parts], ignore_index=True)
    games = games[games["margin"].notna()].reset_index(drop=True)
    fbs = games[games["home_fbs"] & games["away_fbs"]].reset_index(drop=True)
    print(f"\n{first}–{last}: {len(games)} (as-of week, game) rows, {fbs['game_id'].nunique()} FBS games")

    won = (fbs["margin"] > 0).astype(float).to_numpy()
    results = []
    for name, names in FEATURE_SETS.items():
        for g0 in G0_GRID:
            X, y = design(fbs, names, g0), fbs["margin"].to_numpy(float)
            pred = np.full(len(y), np.nan)
            for s in seasons:
                test = (fbs["season"] == s).to_numpy()
                b, *_ = np.linalg.lstsq(X[~test], y[~test], rcond=None)
                pred[test] = X[test] @ b
            p = ndtr(pred / np.std(y - pred))
            results.append({"features": name, "g0": g0, "RMSE": float(np.sqrt(np.mean((y - pred) ** 2))),
                            **_scores(p, won)})
    res = pd.DataFrame(results).sort_values("log loss")
    print("\nOut of sample, each season from the others (FBS games, every as-of week and game ahead):")
    print(res.head(10).round(4).to_string(index=False))
    name, g0 = res.iloc[0]["features"], float(res.iloc[0]["g0"])
    names = FEATURE_SETS[name]
    X, y = design(fbs, names, g0), fbs["margin"].to_numpy(float)
    no_prior = design(fbs, names, 0.0)
    pv = np.full(len(y), np.nan)
    for s in seasons:
        test = (fbs["season"] == s).to_numpy()
        bv, *_ = np.linalg.lstsq(no_prior[~test], y[~test], rcond=None)
        pv[test] = no_prior[test] @ bv
    print("... last season's ratings only before week 1:", _scores(ndtr(pv / np.std(y - pv)), won))

    model = fit_model(games, teams, names, g0, verbose=True)
    b = np.array([model["hfa"], *[v["coef"] for v in model["features"].values()], model["prior"]["coef"]])
    print(f"\nChosen: {name}, g0 = {g0:g}. Home field {model['hfa']:.2f}; "
          + ", ".join(f"{n} {v['coef']:.1f}" for n, v in model["features"].items())
          + f"; last season {model['prior']['coef']:.1f}; game noise {model['game_sd']:.2f}")

    # Out of sample with the chosen model and its spread.
    pred = np.full(len(y), np.nan)
    for s in seasons:
        test = (fbs["season"] == s).to_numpy()
        bs, *_ = np.linalg.lstsq(X[~test], y[~test], rcond=None)
        pred[test] = X[test] @ bs
    sd = np.sqrt(model["game_sd"] ** 2 + sim.team_sd(model, fbs["games_home"].to_numpy()) ** 2
                 + sim.team_sd(model, fbs["games_away"].to_numpy()) ** 2)
    p = ndtr(pred / sd)
    nxt = (fbs["week"] == fbs["k"]).to_numpy()
    print("\nWin probability out of sample: the week after the numbers, and every game ahead:")
    print(pd.DataFrame({"next week": _scores(p[nxt], won[nxt]), "every game ahead": _scores(p, won)}).T.round(4))
    dec = pd.qcut(p, 10)
    print(pd.DataFrame({"predicted": p, "won": won}).groupby(dec, observed=True).mean().round(3).to_string())

    model["fit"] = {"seasons": f"{first}-{last}", "games": int(fbs["game_id"].nunique()),
                    "next week": {k: round(v, 4) for k, v in _scores(p[nxt], won[nxt]).items()},
                    "every game ahead": {k: round(v, 4) for k, v in _scores(p, won).items()}}
    cfp_sim.MODEL_FILE.write_text(json.dumps(model, indent=1) + "\n")
    print(f"\nWrote {cfp_sim.MODEL_FILE}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2016, int(sys.argv[2]) if len(sys.argv) > 2 else 2025,
         sys.argv[3] if len(sys.argv) > 3 else None)
