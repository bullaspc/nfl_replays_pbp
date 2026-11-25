"""
Rebuild models/fourth_down.csv (replay_core.fit_fourth_down()): how often a
3rd or 4th down converts by distance and how far a conversion goes, and where
a punt leaves the other team, which the 4th-down decisions read (NFL).

    python tools/build_fourth_down_table.py 2019 2025 [dir]

`dir` holds local play_by_play_{season}.parquet files (else nflverse's are
downloaded). It prints the table, and the Brier score of its conversion rates
on each season left out of the fit in turn, next to one rate for every 4th down.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import replay_core as core  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "models" / "fourth_down.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.parquet"
COLS = ["game_id", "play_id", "season", "posteam", "defteam", "qtr", "down", "ydstogo", "yardline_100",
        "play_type", "pass_attempt", "rush_attempt", "qb_kneel", "qb_spike", "first_down",
        "pass_touchdown", "rush_touchdown", "yards_gained"]


def read(season: int, local: str | None) -> pd.DataFrame:
    src = Path(local) / f"play_by_play_{season}.parquet" if local else PBP_URL.format(y=season)
    return pd.read_parquet(src, columns=COLS).sort_values(["game_id", "play_id"]).reset_index(drop=True)


def fourth_goes(pbp: pd.DataFrame) -> pd.DataFrame:
    s = pbp[(pbp["down"] == 4) & pbp["play_type"].isin(["pass", "run"]) & (pbp["qtr"] <= 4)]
    conv = (s["first_down"].fillna(0) == 1) | (s["pass_touchdown"] == 1) | (s["rush_touchdown"] == 1)
    return pd.DataFrame({"bin": core._fourth_bin(s["ydstogo"]), "conv": conv.astype(float)})


def main(first: int, last: int, local: str | None = None) -> None:
    seasons = {y: read(y, local) for y in range(first, last + 1)}
    table = core.fit_fourth_down(pd.concat(seasons.values(), ignore_index=True))
    pd.set_option("display.width", 160)
    print(table.to_string(index=False))
    for y in seasons:
        fit = core.fit_fourth_down(pd.concat([p for k, p in seasons.items() if k != y], ignore_index=True))
        held = fourth_goes(seasons[y])
        rate = fit[fit["kind"] == "go"].set_index("bin")["rate"].reindex(held["bin"]).to_numpy()
        brier = ((held["conv"] - rate) ** 2).mean()
        flat = ((held["conv"] - held["conv"].mean()) ** 2).mean()
        print(f"{y}: {len(held)} 4th-down tries, Brier {brier:.4f} (one rate {flat:.4f}), "
              f"converted {held['conv'].mean():.1%} vs table {rate.mean():.1%}")
    OUT.parent.mkdir(exist_ok=True)
    table.round(4).to_csv(OUT, index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]), int(a[1]), a[2] if len(a) > 2 else None)
