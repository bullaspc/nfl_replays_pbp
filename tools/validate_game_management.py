"""
Check the Game management estimates against what happened next.

For every snap in a season of nflverse pbp, `next_snap()` is given the plays
up to the one before it and its estimate (offense, down, distance, spot,
clock, timeouts) is compared with that snap; the clock of a running-clock
snap with the flat RUNOFF_SECS and with runoff_secs() by the offense's margin.
Then every three-and-out late in a half (passes or runs on 1st to 3rd down,
then a punt, inside the last 6 minutes) is replayed with `three_and_out()`,
with every down a run and with models/pass_rate.csv's pass rates, and its
"ball back with" clock, for the timeouts the defense really used, compared
with the receiving team's first snap. Last, for each drive a trailing team ran until the clock ran out,
`plays_left()` from its first snap is compared with the snaps it got.

    python tools/validate_game_management.py 2025 [path] [n_games]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import replay_core as core  # noqa: E402

PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.parquet"
RATES = Path(__file__).resolve().parent.parent / "models" / "pass_rate.csv"
SNAPS = ["pass", "run", "punt", "field_goal", "qb_kneel", "qb_spike"]


def _half_secs(r) -> float:
    g = float(r["game_seconds_remaining"])
    return g - 1800 if r["qtr"] <= 2 else g


def main(season: int, path: str | None = None, n_games: int | None = None) -> None:
    pbp = pd.read_parquet(path or PBP_URL.format(y=season))
    # The published pbp has no xrush_yards (the app adds it on loading).
    pbp = pbp[[c for c in core.PBP_COLS if c in pbp.columns]]
    rates = pd.read_csv(RATES)
    rates = rates[rates["league"] == "NFL"].drop(columns="league").astype({k: float for k in core.PASS_KEYS})
    pbp = pbp[pbp["qtr"] <= 4].sort_values(["game_id", "play_id"])
    games = pbp["game_id"].unique()[:n_games]
    rows, tao, drill = [], [], []
    for gid in games:
        g = pbp[pbp["game_id"] == gid].reset_index(drop=True)
        snap_i = g.index[g["play_type"].isin(SNAPS) & g["down"].notna()]
        for j in snap_i:
            if j == 0:
                continue
            est = core.next_snap(g.iloc[:j], core.NFL_CLOCK)
            if est is None:
                continue
            r = g.iloc[j]
            rows.append({
                "posteam": est.posteam == r["posteam"],
                "down": est.down == r["down"], "ydstogo": est.ydstogo == r["ydstogo"],
                "spot_err": abs(est.yardline_100 - r["yardline_100"]),
                # The estimate is the clock at the last play's whistle; with
                # the clock running, the snap comes about RUNOFF_SECS later.
                "clock_err": (est.half_secs - core.RUNOFF_SECS * est.running - _half_secs(r)
                              if est.half_secs is not None and est.half_secs <= core.SCENARIO_SECS
                              else np.nan),
                # The same with the runoff by the offense's margin (runoff_secs()).
                "clock_err_margin": (est.half_secs - core.runoff_secs(est.score[est.posteam] - est.score[est.defteam])
                                     * est.running - _half_secs(r)
                                     if est.half_secs is not None and est.half_secs <= core.SCENARIO_SECS
                                     else np.nan),
                "running": est.running,
                "timeouts": (est.timeouts[r["home_team"]] == r["home_timeouts_remaining"])
                            and (est.timeouts[r["away_team"]] == r["away_timeouts_remaining"]),
            })
        # Three-and-outs late in a half: passes or runs on 1st to 3rd down,
        # then a punt. Each is replayed with every down a run and with the
        # league's pass rates.
        for _, d in g[g["drive"].notna()].groupby("drive"):
            sc = d[d["play_type"].isin(SNAPS)]
            if (len(sc) != 4 or not set(sc["play_type"].iloc[:3]) <= {"run", "pass"}
                    or sc["play_type"].iloc[3] != "punt"
                    or list(sc["down"]) != [1, 2, 3, 4] or sc["qb_kneel"].sum() > 0):
                continue
            first = sc.index[0]
            est = core.next_snap(g.iloc[:first], core.NFL_CLOCK)
            if est is None or est.half_secs is None or est.half_secs > core.SCENARIO_SECS or est.down != 1:
                continue
            after = g.iloc[sc.index[-1] + 1:]
            back = after[after["play_type"].isin(SNAPS) & after["down"].notna()]
            if back.empty or back.iloc[0]["qtr"] != sc.iloc[-1]["qtr"]:
                continue
            used = int(est.timeouts[est.defteam] - back.iloc[0][
                "home_timeouts_remaining" if est.defteam == g["home_team"].iloc[0] else "away_timeouts_remaining"])
            real = _half_secs(back.iloc[0])
            row = {"passes": int((sc["play_type"].iloc[:3] == "pass").sum()),
                   "margin": est.score[est.posteam] - est.score[est.defteam]}
            for name, table in (("runs", None), ("rates", rates)):
                tbl = core.three_and_out(est, core.NFL_CLOCK, table)
                pick = tbl.iloc[(tbl["Timeouts used"] - used).abs().argsort()[:1]]
                row[name] = float(pick["Ball back with"].iloc[0]) - real
            tao.append(row)
        # Drives a trailing team ran until the clock ran out: plays_left()
        # from its first snap vs the snaps it got.
        for _, d in g[g["drive"].notna()].groupby("drive"):
            sc = d[d["play_type"].isin(["pass", "run", "qb_spike"]) & d["down"].notna()]
            nxt = g.iloc[d.index[-1] + 1:d.index[-1] + 2]
            if (sc.empty or sc.iloc[0]["qtr"] not in (2, 4) or nxt.empty
                    or not str(nxt["desc"].iloc[0]).startswith("END QUARTER")
                    or d["sp"].sum() > 0 or d["interception"].sum() + d["fumble_lost"].sum() > 0):
                continue
            est = core.next_snap(g.iloc[:sc.index[0]], core.NFL_CLOCK)
            if est is None or est.half_secs is None or est.half_secs > core.SCENARIO_SECS:
                continue
            if est.score[est.posteam] >= est.score[est.defteam]:
                continue
            stops = core.clock_stops(est, est.posteam, est.half_secs, est.timeouts[est.posteam], core.NFL_CLOCK)
            drill.append(core.plays_left(est.half_secs, stops) - len(sc))
    df = pd.DataFrame(rows)
    print(f"{season}: {len(games)} games, {len(df)} snaps")
    for c in ("posteam", "down", "ydstogo", "timeouts"):
        print(f"  {c:9s} right {df[c].mean():7.2%}")
    print(f"  spot      within 0 yds {(df['spot_err'] == 0).mean():7.2%}  within 5 {(df['spot_err'] <= 5).mean():7.2%}")
    for running, d in df.dropna(subset=["clock_err"]).groupby("running"):
        ce = d["clock_err"].abs()
        print(f"  clock, last {core.SCENARIO_SECS // 60} min of a half, {'running' if running else 'stopped'}: "
              f"{len(ce)} snaps, |err| median {ce.median():.0f}s, within 10s {(ce <= 10).mean():7.2%}"
              + (f"; runoff by margin: median {d['clock_err_margin'].abs().median():.0f}s, within 10s "
                 f"{(d['clock_err_margin'].abs() <= 10).mean():7.2%}" if running else ""))
    t = pd.DataFrame(tao)
    if len(t):
        print(f"three-and-outs ending in a punt, last {core.SCENARIO_SECS // 60} min of a half: {len(t)}")
        groups = [("all", t), ("no passes", t[t["passes"] == 0]), ("1+ passes", t[t["passes"] > 0]),
                  ("offense ahead", t[t["margin"] > 0]), ("tied", t[t["margin"] == 0]),
                  ("offense behind", t[t["margin"] < 0])]
        for label, d in groups:
            if d.empty:
                continue
            print(f"  {label:15s} {len(d):4d}  " + "  ".join(
                f"{name}: |err| median {d[name].abs().median():3.0f}s within 15s {(d[name].abs() <= 15).mean():4.0%} "
                f"mean {d[name].mean():+4.0f}s" for name in ("runs", "rates")))
    if drill:
        e = np.abs(drill)
        print(f"trailing drives to 0:00: {len(e)}  plays |err| median {np.median(e):.0f}  "
              f"within 2 {(e <= 2).mean():.0%}  mean signed err {np.mean(drill):+.1f}")


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else None,
         int(sys.argv[3]) if len(sys.argv) > 3 else None)
