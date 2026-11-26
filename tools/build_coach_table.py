"""
Rebuild models/fourth_coaches.json (replay_core.fit_coach_model()): how often
NFL teams go for it, kick or punt on 4th down in a given spot, and each head
coach's lean, which the 4th-down coach profiles read.

    python tools/build_coach_table.py 2019 2025 [dir]

`dir` holds local play_by_play_{season}.parquet files (else nflverse's are
downloaded). It prints the log loss of the choice (go / field goal / punt) on
each season left out of the fit in turn: the league model alone and with each
coach's lean, for a few prior widths (the coach's lean is fit on the other
seasons, so a coach new that season gets the league's).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import leagues  # noqa: E402
import replay_core as core  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "models" / "fourth_coaches.json"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.parquet"


def graded(season: int, local: str | None) -> pd.DataFrame:
    src = Path(local) / f"play_by_play_{season}.parquet" if local else PBP_URL.format(y=season)
    pbp = pd.read_parquet(src).sort_values(["game_id", "play_id"]).reset_index(drop=True)
    sits = core.fourth_situations(pbp)
    return core.fourth_down_calls(sits, leagues.NFLLeague.fourth_down_wp(None, sits))


def log_loss(calls: pd.DataFrame, probs: pd.DataFrame) -> float:
    p = probs.to_numpy()[np.arange(len(calls)), calls["choice"].map({"Go": 0, "FG": 1, "Punt": 2}).to_numpy()]
    return float(-np.log(np.clip(p, 1e-4, 1)).mean())


def main(first: int, last: int, local: str | None = None) -> None:
    seasons = {y: graded(y, local) for y in range(first, last + 1)}
    for sd in (0.125, 0.25, 0.5):
        league_ll, coach_ll, n = 0.0, 0.0, 0
        for y, held in seasons.items():
            model = core.fit_coach_model(pd.concat([c for k, c in seasons.items() if k != y]), sd)
            held = held.dropna(subset=["Go", "choice"])
            league = sum(log_loss(g, core.coach_probs(g, model)) * len(g) for _, g in [(None, held)])
            coach = sum(log_loss(g, core.coach_probs(g, model, name)) * len(g)
                        for name, g in held.groupby("coach"))
            league_ll += league
            coach_ll += coach
            n += len(held)
        print(f"prior sd {sd}: log loss league {league_ll / n:.4f}, with coach leans {coach_ll / n:.4f}")
    model = core.fit_coach_model(pd.concat(seasons.values()))
    model["fit"] = {"seasons": [first, last], "downs": int(sum(len(c) for c in seasons.values()))}
    OUT.write_text(json.dumps(model, indent=1))
    pd.set_option("display.width", 200)
    print(core.coach_table(model).round(2).head(15).to_string())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]), int(a[1]), a[2] if len(a) > 2 else None)
