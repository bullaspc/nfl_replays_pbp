"""
Fit the Season simulation page's game model (season_sim.py) and write it to
models/season_sim.json.

For every season and every week k, the season tables are built from the
regular-season weeks before k only (opponent-adjusted, as the page and the
Matchup page build them), and every game from week k on is projected from
them: for each candidate stat, each offense's Proj. against the other team's
defense (season_sim.proj_matrix, the Matchup page's formula), with each
team's lineup as the week's injury report and roster had it
(core.lineup_change: the next game's report, and how many of its games
ahead each game is), last season's net EPA/play, and the games each team
had played. Then:

* how long players stay gone: of the players a report listed Out or
  Doubtful (or a roster had on a reserve list, released, traded), the share
  of their snaps still missing 1, 2, … games later (`decay`), so a lineup
  fades toward full strength over the games ahead;
* the margin model: the home margin on the home field and the projected
  differences (Proj. of the home offense less the away offense's), this
  season's weighted by games / (games + g0) and last season's net rating by
  the rest, least squares; the feature set and g0 are picked by log loss a
  season at a time out of sample (each season predicted from the others);
* the spread around it: how far a team's results stay off its projection
  over its games ahead (the covariance of its residuals: its rating error,
  `tau`, by games played) and a game's own noise (`game_sd`), so a run of
  the simulation can draw a team's error once for all its games.

The win probabilities are then checked out of sample against the games (log
loss, Brier score, calibration by decile) next to the betting line's.

    python tools/build_sim_model.py [first=2016] [last=2025] [pbp_dir]

`pbp_dir` holds nflverse's play_by_play_<season>.parquet (season first − 1
too, for its prior ratings); without it they're downloaded. Set
VALIDATE_CACHE to a folder to keep each season's projected games there.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nflreadpy
import numpy as np
import pandas as pd
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import replay_core as core  # noqa: E402
import season_sim as sim  # noqa: E402
from validate_injuries import _season_data  # noqa: E402
from validate_matchup import read_season  # noqa: E402

# Candidate stats: (feature name, stat, sign: −1 where less is better for the offense).
CANDIDATES = {"epa": ("EPA/play", 1), "pass_epa": ("Pass EPA/play", 1), "rush_epa": ("Rush EPA/play", 1),
              "pts": ("Pts/Drive", 1), "sr": ("ED SR", 1), "to": ("Turnovers", -1),
              "ypp": ("Yds/Play", 1)}
FEATURE_SETS = {"EPA/play": ["epa"], "EPA/play + Pts/Drive": ["epa", "pts"],
                "EPA/play + Pts/Drive + Turnovers": ["epa", "pts", "to"],
                "Pass/Rush EPA + Pts/Drive": ["pass_epa", "rush_epa", "pts"],
                "EPA/play + Pts/Drive + ED SR + Turnovers": ["epa", "pts", "sr", "to"]}
G0_GRID = [0, 1, 2, 3, 4, 6, 8, 12]
HORIZON = 8  # games ahead the decay is measured for
GROUPS = list(core.LINEUP_GROUPS)


def _tables(before: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Opponent-adjusted, qualified season tables from `before`, as the page builds them."""
    if before.empty:
        return None
    off, de = core.season_team_table(before), core.season_team_table(before, "defteam")
    off, de, _, _ = core.opponent_adjust(off, de, core.season_games(before))
    off, de = core.possible_values(off), core.possible_values(de)
    return core.season_qualified(off), core.season_qualified(de)


def prior_ratings(prev: pd.DataFrame) -> pd.Series:
    """Last season's net EPA/play per team (offense less what the defense
    allowed, opponent-adjusted), under this season's abbreviations too."""
    t = _tables(prev)
    net = t[0]["EPA/play"] - t[1]["EPA/play"]
    for new, old in core.FRANCHISE_MOVES.items():
        if old in net.index and new not in net.index:
            net[new] = net[old]
    return net


def collect(season: int, pbp_dir: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every (as-of week, game from that week on) of `season`: the raw
    pieces of the features (both offenses' Proj. per candidate stat, each
    team's players gone by group from the report and the roster, games
    ahead, prior ratings, games played) and the game (week, teams, neutral,
    margin, betting line). Also each lineup_change row with whether the
    player played in each of the team's games after (for the decay)."""
    cache = os.environ.get("VALIDATE_CACHE")
    paths = [Path(cache) / f"sim_{season}_{kind}.pkl" for kind in ("games", "returns")] if cache else None
    if paths and all(p.exists() for p in paths):
        return pd.read_pickle(paths[0]), pd.read_pickle(paths[1])
    path = (lambda y: f"{pbp_dir}/play_by_play_{y}.parquet") if pbp_dir else (lambda y: None)
    pbp = read_season(season, path(season))
    prior = prior_ratings(read_season(season - 1, path(season - 1)))
    sched = nflreadpy.load_schedules([season]).to_pandas()
    reg = sim.regular_season(sched)
    reg = reg[reg["played"]]
    injuries, snaps, roster = _season_data(season)
    team_games = pd.concat([reg[["week", "game_id"]].assign(team=reg["home"]),
                            reg[["week", "game_id"]].assign(team=reg["away"])]).sort_values("week")
    rows, returns = [], []
    for k in range(1, int(reg["week"].max()) + 1):
        ahead = reg[reg["week"] >= k]
        if ahead.empty:
            continue
        before = pbp[pbp["week"] < k]
        t = _tables(before)
        proj = {f: sim.proj_matrix(t[0], t[1], stat) for f, (stat, _) in CANDIDATES.items()} if t else {}
        played = team_games[team_games["week"] < k].groupby("team").size()
        lost, order = {}, {}
        for team, tg in team_games[team_games["week"] >= k].groupby("team"):
            order[team] = list(tg["game_id"])
            before_ids = snaps.loc[(snaps["team"] == team) & (snaps["week"] < k), "game_id"].unique()
            if not len(before_ids):
                continue
            change = core.lineup_change(snaps, injuries, roster, team, int(tg["week"].iloc[0]), before_ids)
            lost[team] = sim.lineup_lost(change)
            # Did each player gone play in the team's games ahead (this one first)?
            for r in change[change["Change"] < 0].itertuples(index=False):
                kind = "report" if r.Why in sim.REPORT_WHY else "roster"
                col = f"{core.SNAP_PHASES[core.UNIT_PHASE[r.Unit]]}_snaps"
                mine = snaps[(snaps["team"] == team) & (snaps["player"] == r.Player)]
                out = [float(mine.loc[mine["game_id"] == gid, col].sum() <= 0) for gid in order[team][:HORIZON]]
                returns.append({"season": season, "k": k, "kind": kind, "weight": -r.Change,
                                **{f"g{j}": v for j, v in enumerate(out)}})
        for g in ahead.itertuples(index=False):
            row = {"season": season, "k": k, "game_id": g.game_id, "week": g.week, "home": g.home,
                   "away": g.away, "neutral": g.neutral,
                   "margin": float(sched.loc[sched["game_id"] == g.game_id, "result"].iloc[0]),
                   "spread": sched.loc[sched["game_id"] == g.game_id, "spread_line"].iloc[0],
                   "games_home": played.get(g.home, 0), "games_away": played.get(g.away, 0),
                   "prior_home": prior.get(g.home, np.nan), "prior_away": prior.get(g.away, np.nan),
                   "ahead_home": order.get(g.home, []).index(g.game_id),
                   "ahead_away": order.get(g.away, []).index(g.game_id)}
            for f, m in proj.items():
                ok = g.home in m.index and g.away in m.index
                row[f"{f}_home"] = m.at[g.home, g.away] if ok else np.nan
                row[f"{f}_away"] = m.at[g.away, g.home] if ok else np.nan
            for side, team in (("home", g.home), ("away", g.away)):
                rep, ros = lost.get(team, (pd.Series(0.0, index=GROUPS),) * 2)
                for grp in GROUPS:
                    row[f"rep_{grp}_{side}"] = rep[grp]
                    row[f"ros_{grp}_{side}"] = ros[grp]
            rows.append(row)
        print(f"{season} week {k}: {len(ahead)} games ahead", flush=True)
    games, returns = pd.DataFrame(rows), pd.DataFrame(returns)
    if paths:
        paths[0].parent.mkdir(parents=True, exist_ok=True)
        games.to_pickle(paths[0])
        returns.to_pickle(paths[1])
    return games, returns


def fit_decay(returns: pd.DataFrame) -> dict[str, list[float]]:
    """Per kind, the snap-weighted share of the players gone who still don't
    play j games later, relative to the first game (1.0), never rising."""
    out = {}
    for kind, d in returns.groupby("kind"):
        cols = [f"g{j}" for j in range(HORIZON) if f"g{j}" in d.columns]
        shares = []
        for c in cols:
            x = d.dropna(subset=[c])
            shares.append(float(np.average(x[c], weights=x["weight"])) if len(x) else np.nan)
        rel = np.minimum.accumulate(np.nan_to_num(np.array(shares) / shares[0], nan=0.0))
        out[kind] = [round(float(v), 3) for v in rel]
    return out


def features(games: pd.DataFrame, names: list[str], decay: dict) -> pd.DataFrame:
    """The model's features for `games` rows, as season_sim.game_features
    computes them on the page: each stat's Proj. home less away (sign
    flipped where less is better), the lineup moving the stats
    core.LINEUP_EFFECTS has, faded by `decay` over the games ahead."""
    out = pd.DataFrame(index=games.index)
    out["neutral"] = games["neutral"].astype(bool)
    out["games"] = (games["games_home"] + games["games_away"]) / 2
    lost = {}
    for side in ("home", "away"):
        rep = np.array([[sim.still_out(decay, "report", a)] for a in games[f"ahead_{side}"]])
        ros = np.array([[sim.still_out(decay, "roster", a)] for a in games[f"ahead_{side}"]])
        lost[side] = pd.DataFrame(games[[f"rep_{g}_{side}" for g in GROUPS]].to_numpy() * rep
                                  + games[[f"ros_{g}_{side}" for g in GROUPS]].to_numpy() * ros,
                                  columns=GROUPS, index=games.index)
    for f in names:
        stat, sign = CANDIDATES[f]
        sh = np.array([sim.lineup_shift(lost["home"].loc[i], lost["away"].loc[i], stat) for i in games.index]) \
            if stat in core.LINEUP_EFFECTS else 0.0
        sa = np.array([sim.lineup_shift(lost["away"].loc[i], lost["home"].loc[i], stat) for i in games.index]) \
            if stat in core.LINEUP_EFFECTS else 0.0
        out[f] = sign * ((games[f"{f}_home"] + sh) - (games[f"{f}_away"] + sa))
    out["prior"] = games["prior_home"] - games["prior_away"]
    return out


def design(feats: pd.DataFrame, names: list[str], g0: float) -> np.ndarray:
    w = sim.data_weight(feats["games"].to_numpy(), g0)
    cols = [1 - feats["neutral"].astype(float).to_numpy()]
    cols += [w * feats[f].fillna(0.0).to_numpy() for f in names]
    cols.append((1 - w) * feats["prior"].fillna(0.0).to_numpy())
    return np.column_stack(cols)


def fit_tau(games: pd.DataFrame, resid: np.ndarray, verbose: bool = True) -> dict:
    """A team's rating error by games played: the mean product of its
    residuals (its side's margin less the projection) over pairs of its
    games ahead from one as-of week, by games played, fitted as
    floor² + scale² / (games + g1)."""
    r = pd.concat([games[["season", "k", "home", "games_home"]].set_axis(["season", "k", "team", "g"], axis=1)
                   .assign(r=resid),
                   games[["season", "k", "away", "games_away"]].set_axis(["season", "k", "team", "g"], axis=1)
                   .assign(r=-resid)])
    grp = r.groupby(["season", "k", "team"]).agg(s=("r", "sum"), ss=("r", lambda x: (x ** 2).sum()),
                                                 n=("r", "size"), g=("g", "first"))
    grp = grp[grp["n"] >= 2]
    grp["cov"] = (grp["s"] ** 2 - grp["ss"]) / (grp["n"] * (grp["n"] - 1))
    pairs = grp["n"] * (grp["n"] - 1)
    by_g = grp.assign(w=pairs).groupby("g").apply(
        lambda d: pd.Series({"tau2": np.average(d["cov"], weights=d["w"]), "w": d["w"].sum()}),
        include_groups=False)
    best = None
    for g1 in (0.5, 1, 2, 3, 4, 6, 8):
        x = 1.0 / (by_g.index.to_numpy(float) + g1)
        A = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(A * np.sqrt(by_g["w"].to_numpy())[:, None],
                                   by_g["tau2"].to_numpy() * np.sqrt(by_g["w"].to_numpy()), rcond=None)
        coef = np.maximum(coef, 0.0)
        err = float((by_g["w"] * (A @ coef - by_g["tau2"]) ** 2).sum())
        if best is None or err < best[0]:
            best = (err, g1, coef)
    _, g1, (floor2, scale2) = best
    out = {"floor": round(float(np.sqrt(floor2)), 3), "scale": round(float(np.sqrt(scale2)), 3), "g1": g1}
    if not verbose:
        return out
    print("\nA team's rating error by games played (points; covariance of its residuals over its "
          "games ahead):")
    show = by_g.assign(tau=np.sqrt(by_g["tau2"].clip(lower=0)),
                       fitted=np.sqrt(floor2 + scale2 / (by_g.index.to_numpy(float) + g1)))
    print(show[["tau", "fitted"]].round(2).T.to_string())
    return out


def fit_model(games: pd.DataFrame, returns: pd.DataFrame, names: list[str], g0: float,
              verbose: bool = False) -> tuple[dict, dict]:
    """The model (season_sim.json's fields but "fit") from `games` and
    `returns` rows, with the feature set `names` and `g0`; also the pieces
    for the checks (features, design, coefficients, residuals, rating SDs)."""
    decay = fit_decay(returns)
    f = features(games, names, decay)
    X, y = design(f, names, g0), games["margin"].to_numpy()
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    tau = fit_tau(games, resid, verbose)
    t_home = sim.team_sd({"tau": tau}, games["games_home"].to_numpy())
    t_away = sim.team_sd({"tau": tau}, games["games_away"].to_numpy())
    game_sd = float(np.sqrt(max(np.mean(resid ** 2 - t_home ** 2 - t_away ** 2), 1.0)))
    model = {
        "features": {n: {"stat": CANDIDATES[n][0], "sign": CANDIDATES[n][1], "coef": round(float(c), 3)}
                     for n, c in zip(names, b[1:-1])},
        "prior": {"stat": "net EPA/play, last season", "coef": round(float(b[-1]), 3)},
        "g0": g0, "hfa": round(float(b[0]), 3), "game_sd": round(game_sd, 3), "tau": tau, "decay": decay,
    }
    return model, {"features": f, "X": X, "b": b, "t_home": t_home, "t_away": t_away}


def _scores(p: np.ndarray, won: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"log loss": float(-np.mean(won * np.log(p) + (1 - won) * np.log(1 - p))),
            "Brier": float(np.mean((p - won) ** 2)), "picked": float(np.mean((p > 0.5) == won))}


def main(first: int = 2016, last: int = 2025, pbp_dir: str | None = None) -> None:
    seasons = list(range(first, last + 1))
    with ProcessPoolExecutor(max_workers=min(4, len(seasons))) as pool:
        parts = list(pool.map(collect, seasons, [pbp_dir] * len(seasons)))
    games = pd.concat([p[0] for p in parts], ignore_index=True)
    returns = pd.concat([p[1] for p in parts], ignore_index=True)
    games = games[games["margin"].notna()].reset_index(drop=True)
    print(f"\n{first}–{last}: {len(games)} (as-of week, game) rows, {games['game_id'].nunique()} games")

    decay = fit_decay(returns)
    print("\nShare of the players gone still out, by games ahead (0 = the game they were listed for):")
    for kind, v in decay.items():
        print(f"  {kind}: {v}")

    # Pick the feature set and g0 a season at a time out of sample.
    won = (games["margin"] > 0).astype(float).to_numpy()
    decided = games["margin"].to_numpy() != 0
    feats = {name: features(games, names, decay) for name, names in FEATURE_SETS.items()}
    results = []
    for name, names in FEATURE_SETS.items():
        f = feats[name]
        for g0 in G0_GRID:
            X, y = design(f, names, g0), games["margin"].to_numpy()
            pred = np.full(len(y), np.nan)
            for s in seasons:
                test = (games["season"] == s).to_numpy()
                b, *_ = np.linalg.lstsq(X[~test], y[~test], rcond=None)
                pred[test] = X[test] @ b
            sd = float(np.std(y - pred))
            p = ndtr(pred / sd)
            results.append({"features": name, "g0": g0, "RMSE": float(np.sqrt(np.mean((y - pred) ** 2))),
                            **_scores(p[decided], won[decided])})
    res = pd.DataFrame(results).sort_values("log loss")
    print("\nOut of sample, each season from the others (every as-of week and game ahead):")
    print(res.head(12).round(4).to_string(index=False))
    best = res.iloc[0]
    name, g0 = best["features"], float(best["g0"])
    names = FEATURE_SETS[name]

    # The model on every season.
    model, parts = fit_model(games, returns, names, g0, verbose=True)
    f, X, b, t_home, t_away = (parts[k] for k in ("features", "X", "b", "t_home", "t_away"))
    y, game_sd = games["margin"].to_numpy(), model["game_sd"]
    print(f"\nChosen: {name}, g0 = {g0:g}. Home field {b[0]:.2f} points; per unit of projected "
          f"difference: " + ", ".join(f"{n} {c:.1f}" for n, c in zip(names, b[1:-1]))
          + f"; last season's net EPA/play {b[-1]:.1f}. A game's own noise {game_sd:.2f} points.")

    # What the lineups and last season's ratings add, out of sample.
    print("\nThe chosen model without its parts, out of sample (every as-of week and game ahead):")
    variants = {"as chosen": (decay, g0), "no lineups": ({"report": [0.0], "roster": [0.0]}, g0),
                "lineups never fade": ({"report": [1.0], "roster": [1.0]}, g0),
                "last season only before week 1": (decay, 0.0)}
    out = {}
    for label, (dec, g) in variants.items():
        Xv, yv = design(features(games, names, dec), names, g), games["margin"].to_numpy()
        pv = np.full(len(yv), np.nan)
        for s in seasons:
            test = (games["season"] == s).to_numpy()
            bv, *_ = np.linalg.lstsq(Xv[~test], yv[~test], rcond=None)
            pv[test] = Xv[test] @ bv
        out[label] = {"RMSE": float(np.sqrt(np.mean((yv - pv) ** 2))),
                      **_scores(ndtr(pv / np.std(yv - pv))[decided], won[decided])}
    print(pd.DataFrame(out).T.round(4).to_string())

    # Out of sample with the chosen model, and its spread: next to the betting line.
    pred = np.full(len(y), np.nan)
    for s in seasons:
        test = (games["season"] == s).to_numpy()
        bs, *_ = np.linalg.lstsq(X[~test], y[~test], rcond=None)
        pred[test] = X[test] @ bs
    sd_total = np.sqrt(game_sd ** 2 + t_home ** 2 + t_away ** 2)
    p = ndtr(pred / sd_total)
    games["p"], games["ahead"] = p, games[["ahead_home", "ahead_away"]].min(axis=1)
    next_game = (games["week"] == games["k"]).to_numpy() & decided & games["spread"].notna().to_numpy()
    line_p = ndtr(games["spread"].to_numpy(float) / 13.5)  # the line as a probability, a typical spread
    print("\nWin probability out of sample, games played the week after the numbers (with a betting line):")
    print(pd.DataFrame({"model": _scores(p[next_game], won[next_game]),
                        "betting line": _scores(line_p[next_game], won[next_game])}).round(4).T.to_string())
    print("... every game ahead, by weeks ahead:")
    wk = (games["week"] - games["k"]).clip(upper=8)
    print(pd.DataFrame({w: _scores(p[(wk == w).to_numpy() & decided], won[(wk == w).to_numpy() & decided])
                        for w in sorted(wk.unique())}).round(4).T.to_string())
    print("... by games played before (the as-of week's numbers):")
    gb = pd.cut(f["games"], [-1, 0, 2, 4, 8, 12, 17], labels=["0", "1-2", "3-4", "5-8", "9-12", "13+"])
    print(pd.DataFrame({str(c): _scores(p[(gb == c).to_numpy() & decided], won[(gb == c).to_numpy() & decided])
                        for c in gb.cat.categories}).round(4).T.to_string())
    print("Calibration (deciles of the home team's win probability):")
    dec = pd.qcut(p[decided], 10)
    cal = pd.DataFrame({"predicted": p[decided], "won": won[decided]}).groupby(dec, observed=True).mean()
    print(cal.round(3).to_string())

    model["fit"] = {
        "seasons": f"{first}-{last}", "rows": int(len(games)), "games": int(games["game_id"].nunique()),
        "next week": {k: round(v, 4) for k, v in _scores(p[next_game], won[next_game]).items()},
        "next week, betting line": {k: round(v, 4) for k, v in _scores(line_p[next_game], won[next_game]).items()},
        "every game ahead": {k: round(v, 4) for k, v in _scores(p[decided], won[decided]).items()}}
    sim.MODEL_FILE.write_text(json.dumps(model, indent=1) + "\n")
    print(f"\nWrote {sim.MODEL_FILE}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2016, int(sys.argv[2]) if len(sys.argv) > 2 else 2025,
         sys.argv[3] if len(sys.argv) > 3 else None)
