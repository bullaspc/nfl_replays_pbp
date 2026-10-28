"""
Rebuild one league's rows of models/xrush_yards.csv: the expected rushing yards
of a designed run by down, distance and field position, which RYOE (rushing
yards over expected) is measured against (replay_core.fit_expected_rush()).

    python tools/build_xrush_table.py NFL 2019 2025
    python tools/build_xrush_table.py CFB 2022 2025

The other league's rows are kept. It prints how much of the variance in yards
per run the table explains, in the fitted seasons and, as a check that it isn't
overfit, in each season left out of the fit in turn.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import replay_core as core  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "models" / "xrush_yards.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.parquet"
COLS = ["season", "down", "ydstogo", "yardline_100", "rush_attempt", "qb_kneel", "desc", "rushing_yards"]


def read(league: str, season: int) -> pd.DataFrame:
    if league == "NFL":
        return pd.read_parquet(PBP_URL.format(y=season), columns=COLS)
    import cfb_feed
    pbp = cfb_feed.to_pbp(cfb_feed.read_season(season), core.PBP_COLS)
    return pbp.assign(season=season)[COLS]


def r2(pbp: pd.DataFrame, table: pd.DataFrame) -> float:
    """Share of the variance in a designed run's yards the table explains."""
    p = core.add_expected_rush(pbp, table)
    runs = p[p["xrush_yards"].notna()]
    resid = runs["rushing_yards"].fillna(0) - runs["xrush_yards"]
    return 1 - resid.var() / runs["rushing_yards"].fillna(0).var()


def main(league: str, first: int, last: int) -> None:
    seasons = {y: read(league, y) for y in range(first, last + 1)}
    pbp = pd.concat(seasons.values(), ignore_index=True)
    table = core.fit_expected_rush(pbp)
    print(f"{league} {first}-{last}: {int(table['runs'].sum())} designed runs, "
          f"{int((table['runs'] >= core.XRUSH_MIN_RUNS).sum())} of {len(table)} cells fitted on their own, "
          f"R² {r2(pbp, table):.3f}")
    for y, held in seasons.items():
        rest = pd.concat([d for s, d in seasons.items() if s != y], ignore_index=True)
        print(f"  {y} left out: R² {r2(held, core.fit_expected_rush(rest)):.3f}")
    old = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["league"])
    out = pd.concat([old[old["league"] != league], table.assign(league=league)], ignore_index=True)
    out["xyds"] = out["xyds"].round(3)
    out[[*core.XRUSH_KEYS, "runs"]] = out[[*core.XRUSH_KEYS, "runs"]].astype(int)
    out[["league", *core.XRUSH_KEYS, "xyds", "runs"]].sort_values(
        ["league", *core.XRUSH_KEYS]).to_csv(OUT, index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main(sys.argv[1].upper(), int(sys.argv[2]), int(sys.argv[3]))
