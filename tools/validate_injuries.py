"""
Check whether the Matchup page's projections should account for who's missing.

A team's season numbers were built by whoever played its earlier games, so
what matters for a game is how its lineup differs from that one
(replay_core.lineup_change): a starter out takes away the share of those
games' snaps he had (half, if he missed half of them), and a starter back
adds the share he plays when he does. Who's gone is what was known before
the game: Out or Doubtful on the week's injury report, or on a reserve
list, released or traded on the week's roster.

For every offense-game from `first_week` on in the seasons given, the
projection tools/validate_matchup.py checks (opponent-adjusted, as the page
shows it) is set against the game, and the miss (actual − Proj.) is fitted
on the change in five groups (replay_core.LINEUP_GROUPS): the offense's QB,
OL and skill players, and the defense's front seven and secondary. A
replacement's own numbers aren't needed: an effect is how far a team's
results moved, on average, with a group that far from the lineup behind
its numbers, whoever filled in.

The fit is tested out of sample, a season at a time: each season's
adjustments come from the effects fitted on the other seasons, and are
compared with the projection plus its average miss alone (so the
adjustment isn't credited with fixing a bias it didn't), for all five
groups and for the QB alone. Then the projected
EPA/play margin, with and without the adjustment, picks winners next to
the betting line's favorite.

    python tools/validate_injuries.py [first_season=2016] [last_season=2025] [first_week=4]

Set VALIDATE_CACHE to a folder to keep each season's offense-games there.
"""
import os
import sys
from pathlib import Path

import nflreadpy
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import replay_core as core  # noqa: E402
from validate_matchup import offense_games, read_season, winner_picks  # noqa: E402

TARGETS = ["EPA/play", "Pass EPA/play", "Rush EPA/play", "Pass SR", "Sack %"]
GROUPS = list(core.LINEUP_GROUPS)        # QB, OL, Skill, Front, DB
OWN, OPP = ["QB", "OL", "Skill"], ["Front", "DB"]
FEATURES = [f"own {g}" for g in OWN] + [f"opp {g}" for g in OPP]          # net changes
LOST = [f"{f} lost" for f in FEATURES]                                     # players gone only
BACK = [f"{f} back" for f in FEATURES]                                     # players back only
VARIANTS = {"net change": FEATURES, "losses only": LOST, "losses and returns apart": LOST + BACK,
            "QB losses only": ["own QB lost"]}


def _season_data(season: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The season's injury reports, regular-season snap counts (with gsis
    ids) and weekly rosters."""
    injuries = nflreadpy.load_injuries([season]).to_pandas()
    snaps = nflreadpy.load_snap_counts([season]).to_pandas()
    snaps = snaps[snaps["game_type"] == "REG"]
    players = nflreadpy.load_players().select(["gsis_id", "pfr_id"]).to_pandas()
    players = players.dropna().drop_duplicates("pfr_id").set_index("pfr_id")["gsis_id"]
    snaps = snaps.assign(gsis_id=snaps["pfr_player_id"].map(players))
    roster = nflreadpy.load_rosters_weekly([season]).select(["team", "week", "gsis_id", "status"]).to_pandas()
    return injuries, snaps, roster


def collect(season: int, first_week: int) -> pd.DataFrame:
    """One row per offense-game: season, game_id, week, team, opp, the
    game's teams, betting line and result, each TARGETS stat's actual and
    Proj., and the lineup changes (FEATURES)."""
    cache = os.environ.get("VALIDATE_CACHE")
    path = Path(cache) / f"lineups_{season}_{first_week}.pkl" if cache else None
    if path is not None and path.exists():
        return pd.read_pickle(path)
    rows, games = offense_games(read_season(season), first_week, kinds=("adjusted",), stats=TARGETS)
    wide = rows.pivot_table(index=["game_id", "week", "team", "opp"], columns="stat",
                            values=["actual", "proj"]).reset_index()
    wide.columns = [f"{a} {b}".strip() for a, b in wide.columns]
    wide = wide.merge(games[["home_team", "away_team", "spread_line", "result"]],
                      left_on="game_id", right_index=True)
    injuries, snaps, roster = _season_data(season)
    groups = {}
    for team, week in set(zip(wide["team"], wide["week"])) | set(zip(wide["opp"], wide["week"])):
        before = snaps.loc[(snaps["team"] == team) & (snaps["week"] < week), "game_id"].unique()
        change = core.lineup_change(snaps, injuries, roster, team, week, before) if len(before) else None
        groups[(team, week)] = {part: (core.lineup_groups(change, part) if change is not None
                                       else pd.Series(np.nan, index=GROUPS))
                                for part in ("net", "lost", "back")}
    for side, names, key in (("own", OWN, "team"), ("opp", OPP, "opp")):
        for g in names:
            for part, suffix in (("net", ""), ("lost", " lost"), ("back", " back")):
                wide[f"{side} {g}{suffix}"] = [groups[(t, w)][part][g] for t, w in zip(wide[key], wide["week"])]
    wide["season"] = season
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        wide.to_pickle(path)
    return wide


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least squares with an intercept (first): coefficients and standard errors."""
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    sigma2 = resid @ resid / max(len(y) - A.shape[1], 1)
    se = np.sqrt(np.diag(sigma2 * np.linalg.pinv(A.T @ A)))
    return beta, se


def _errors(actual: pd.Series, pred: pd.Series) -> dict:
    return {"MAE": (pred - actual).abs().mean(), "RMSE": float(np.sqrt(((pred - actual) ** 2).mean())),
            "r": pred.corr(actual)}


def main(first: int = 2016, last: int = 2025, first_week: int = 4) -> None:
    seasons = list(range(first, last + 1))
    df = pd.concat([collect(s, first_week) for s in seasons], ignore_index=True)
    covered = df[FEATURES].notna().all(axis=1)
    print(f"{first}-{last}, weeks {first_week}+: {len(df)} offense-games, {covered.mean():.1%} with "
          "snap counts for both teams' earlier games\n")
    df = df[covered]
    big = (df[FEATURES].abs() >= 0.5).any(axis=1)
    print("How far lineups were from the ones behind the season numbers (full-time players; "
          "− = fewer), net and the players gone and back apart:")
    print(df[FEATURES + LOST + BACK].describe(percentiles=[.05, .5, .95])
          .loc[["mean", "5%", "50%", "95%"]].round(2).T.to_string())
    print(f"Games with a group at least half a full-time player off: {big.mean():.0%}\n")

    adjusted = {}
    for stat in TARGETS:
        d = df.dropna(subset=[f"actual {stat}", f"proj {stat}"])
        y = (d[f"actual {stat}"] - d[f"proj {stat}"]).to_numpy()
        split = LOST + BACK
        beta, se = _ols(d[split].to_numpy(), y)
        print(f"== {stat}: how much the game's miss (actual − Proj.) moves per full-time player of "
              f"change (lost: per player gone; back: per player back), n = {len(d)}")
        print(pd.DataFrame({"per player gone (−) or back (+)": [(-b if c in LOST else b) for b, c in zip(beta[1:], split)],
                            "± 2 SE": 2 * se[1:]}, index=split).round(4).to_string())
        # Out of sample, a season at a time.
        bias_only = pd.Series(np.nan, index=d.index)
        with_lineup = {k: pd.Series(np.nan, index=d.index) for k in VARIANTS}
        for s in seasons:
            test, train = d["season"] == s, d["season"] != s
            if not test.any() or not train.any():
                continue
            yt = y[train.to_numpy()]
            bias_only[test] = d.loc[test, f"proj {stat}"] + yt.mean()
            for k, feats in VARIANTS.items():
                b, _ = _ols(d.loc[train, feats].to_numpy(), yt)
                with_lineup[k][test] = (d.loc[test, f"proj {stat}"] + b[0]
                                        + d.loc[test, feats].to_numpy() @ b[1:])
        a = d[f"actual {stat}"]
        res = pd.DataFrame({"Proj.": _errors(a, d[f"proj {stat}"]),
                            "Proj. + its average miss": _errors(a, bias_only),
                            **{f"Proj. + lineup ({k})": _errors(a, v) for k, v in with_lineup.items()}}).T
        sub = d["own QB"].abs() >= 0.5
        res_big = pd.DataFrame({"Proj. + its average miss": _errors(a[sub], bias_only[sub]),
                                **{f"Proj. + lineup ({k})": _errors(a[sub], v[sub])
                                   for k, v in with_lineup.items()}}).T
        print("Out of sample (each season from the others' effects), every game:")
        print(res.round(4).to_string())
        print(f"... and the {int(sub.sum())} games with the QB half a player or more off:")
        print(res_big.round(4).to_string(), "\n")
        adjusted[stat] = with_lineup

    # The two cases with the biggest effect, in plain numbers.
    epa = df.dropna(subset=["actual EPA/play", "proj EPA/play"])
    miss = epa["actual EPA/play"] - epa["proj EPA/play"]
    for label, mask in (("starting QB out (own QB ≤ −0.5)", epa["own QB"] <= -0.5),
                        ("starting QB back (own QB ≥ +0.5)", epa["own QB"] >= 0.5),
                        ("2+ OL players' worth out (own OL ≤ −2)", epa["own OL"] <= -2),
                        ("lineup about as usual (every group within ±0.25)",
                         (epa[FEATURES].abs() <= 0.25).all(axis=1))):
        print(f"{label}: {int(mask.sum())} games, EPA/play miss {miss[mask].mean():+.3f} "
              f"(± {2 * miss[mask].std() / max(np.sqrt(mask.sum()), 1):.3f})")

    # Winner picks: the projected EPA/play margin with and without the lineup.
    rows = epa.assign(kind="adjusted", stat="EPA/play", proj=epa["proj EPA/play"])
    rows = pd.concat([rows] + [rows.assign(kind=k, proj=v.reindex(epa.index))
                               for k, v in adjusted["EPA/play"].items()])
    games = df.groupby("game_id")[["week", "home_team", "away_team", "spread_line", "result"]].first()
    p = winner_picks(rows, games, kinds=("adjusted", *VARIANTS))
    both = p[["adjusted", *VARIANTS]].notna().all(axis=1)
    print(f"\nWinner picked ({int(both.sum())} games with every projection):")
    for col, label in (("adjusted", "projected EPA/play margin"),
                       *((k, f"projected EPA/play margin + lineup ({k})") for k in VARIANTS),
                       ("line", "betting line's favorite")):
        s = p.loc[both, col].dropna()
        print(f"  {label}: {s.mean():.1%} of {len(s)}")


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:4]]
    main(*args)
