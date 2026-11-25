"""
Check the 4th-down decisions (replay_core.fourth_down_wp()) against what
followed: for every 4th down of a season, the model's win probability for the
choice the team made next to nflverse's win probability on the row after the
play (the offense's), by choice and by how far apart the options were. Also
prints how often teams went, how often the model says go, and the teams'
WP lost.

    python tools/validate_fourth_downs.py 2025 [path]

`path` is a local play_by_play_{season}.parquet (else nflverse's is downloaded).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import leagues  # noqa: E402
import replay_core as core  # noqa: E402

PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.parquet"


def main(season: int, path: str | None = None) -> None:
    pbp = pd.read_parquet(path or PBP_URL.format(y=season))
    pbp = pbp.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    sits = core.fourth_situations(pbp)
    wps = leagues.NFLLeague.fourth_down_wp(None, sits)
    if wps is None:
        sys.exit("no WP model or models/fourth_down.csv")
    calls = core.fourth_down_calls(sits, wps)

    # nflverse's WP on the next row of the same game, for the offense.
    nxt = pbp.groupby("game_id")["home_wp"].shift(-1)
    after = nxt.loc[calls["row"]].to_numpy()
    home = calls["home"].to_numpy() == 1
    calls["realized"] = np.where(home, after, 1 - after)
    calls = calls.dropna(subset=["realized", "chosen"])

    pd.set_option("display.width", 160)
    print(f"{season}: {len(calls)} 4th downs; teams went {(calls['choice'] == 'Go').mean():.1%}, "
          f"the model says go {(calls['Best'] == 'Go').mean():.1%}, agree {(calls['choice'] == calls['Best']).mean():.1%}")
    print(pd.crosstab(calls["choice"], calls["Best"], margins=True).to_string())
    by = calls.groupby("choice").agg(n=("chosen", "size"), predicted=("chosen", "mean"),
                                     realized=("realized", "mean"))
    by["gap"] = by["realized"] - by["predicted"]
    print("\nThe choice's WP, predicted vs nflverse's after the play:")
    print(by.round(4).to_string())
    calls["edge"] = pd.cut(calls["lost"] * 100, [-0.01, 0.5, 1, 2, 5, 100])
    g = calls.groupby("edge", observed=True).agg(n=("lost", "size"), lost=("lost", "mean"),
                                                 predicted=("chosen", "mean"), realized=("realized", "mean"))
    print("\nBy WP the choice gave up (points):")
    print(g.round(4).to_string())
    print("\nTeams, best to worst:")
    print(core.fourth_down_summary(calls).round(2).to_string())


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]), a[1] if len(a) > 1 else None)
