"""
Rebuild one league's rows of models/pass_rate.csv: how often offenses pass by
down, distance, score margin and the clock (replay_core.fit_pass_rate()),
which Game management's three-and-out scenarios and next-snap call read.

    python tools/build_pass_rate_table.py NFL 2019 2025 [dir]
    python tools/build_pass_rate_table.py CFB 2023 2025

`dir` holds local play_by_play_{season}.parquet files (else nflverse's are
downloaded). The other league's rows are kept. It prints the log loss of the
table's rates, in the fitted seasons and in each season left out of the fit in
turn, next to down × distance alone and, for the NFL, nflfastR's own xpass
model (which also reads field position, win probability and the spread).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import replay_core as core  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "models" / "pass_rate.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.parquet"
COLS = ["game_id", "season", "home_team", "posteam", "qtr", "game_seconds_remaining", "down", "ydstogo",
        "play_type", "qb_kneel", "qb_spike", "total_home_score", "total_away_score"]


def read(league: str, season: int, local: str | None) -> pd.DataFrame:
    if league == "NFL":
        src = Path(local) / f"play_by_play_{season}.parquet" if local else PBP_URL.format(y=season)
        pbp = pd.read_parquet(src, columns=[*COLS, "play_id", "xpass"]).sort_values(["game_id", "play_id"])
    else:
        import cfb_feed
        pbp = cfb_feed.to_pbp(cfb_feed.read_season(season), core.PBP_COLS).assign(season=season)
    calls = core.pass_calls(pbp.reset_index(drop=True))
    if "xpass" in pbp.columns:
        calls["xpass"] = pbp.reset_index(drop=True).loc[calls.index, "xpass"].to_numpy()
    return calls


def log_loss(y: pd.Series, p: pd.Series) -> float:
    p = p.clip(1e-4, 1 - 1e-4)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def scores(held: pd.DataFrame, table: pd.DataFrame, coarse: pd.DataFrame) -> str:
    p = held.merge(table, on=core.PASS_KEYS, how="left")["rate"]
    c = held.merge(coarse, on=["down", "dist_bin"], how="left")["c"]
    y = held["pass"].reset_index(drop=True)
    out = f"log loss {log_loss(y, p):.4f} (down × distance {log_loss(y, c):.4f}"
    if "xpass" in held.columns and held["xpass"].notna().all():
        out += f", nflfastR xpass {log_loss(y, held['xpass'].reset_index(drop=True)):.4f}"
    return out + ")"


def main(league: str, first: int, last: int, local: str | None = None) -> None:
    seasons = {y: read(league, y, local) for y in range(first, last + 1)}
    calls = pd.concat(seasons.values(), ignore_index=True)
    table = core.fit_pass_rate(calls)
    coarse = lambda d: d.groupby(["down", "dist_bin"])["pass"].mean().rename("c").reset_index()  # noqa: E731
    print(f"{league} {first}-{last}: {len(calls)} passes and runs, pass rate {calls['pass'].mean():.1%}, "
          f"{scores(calls, table, coarse(calls))}")
    for y, held in seasons.items():
        rest = pd.concat([d for s, d in seasons.items() if s != y], ignore_index=True)
        print(f"  {y} left out: {scores(held, core.fit_pass_rate(rest), coarse(rest))}")
    old = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["league"])
    out = pd.concat([old[old["league"] != league], table.assign(league=league)], ignore_index=True)
    out["rate"] = out["rate"].round(4)
    out[[*core.PASS_KEYS, "plays"]] = out[[*core.PASS_KEYS, "plays"]].astype(int)
    out[["league", *core.PASS_KEYS, "rate", "plays"]].sort_values(
        ["league", *core.PASS_KEYS]).to_csv(OUT, index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else None)
