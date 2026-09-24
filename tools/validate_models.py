"""
Check nflfastr_models.py against nflfastR's own published numbers.

Runs the Python port over a season of nflverse pbp (using only its input
columns) and compares ep / epa / wp / home_wp with the published values.

    python tools/validate_models.py 2025
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nflfastr_models as nm  # noqa: E402

PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.parquet"


def main(season: int, path: str | None = None) -> None:
    pbp = pd.read_parquet(path or PBP_URL.format(y=season))
    published = pbp[["ep", "epa", "wp", "home_wp"]].copy()
    inputs = pbp.drop(columns=["ep", "epa", "wp", "home_wp", "away_wp", "def_wp",
                               "td_prob", "fg_prob", "safety_prob", "no_score_prob",
                               "opp_td_prob", "opp_fg_prob", "opp_safety_prob"])
    out = nm.add_ep_wp(inputs, nm.load_models(), nm.load_fg_table())
    out = out.loc[pbp.index]
    print(f"{season}: {pbp['game_id'].nunique()} games, {len(pbp)} rows")
    for col in ["ep", "epa", "wp", "home_wp"]:
        a, b = out[col].to_numpy(float), published[col].to_numpy(float)
        both = ~np.isnan(a) & ~np.isnan(b)
        na_mismatch = (np.isnan(a) != np.isnan(b)).sum()
        err = np.abs(a[both] - b[both])
        print(f"  {col:8s} within 1e-3: {(err < 1e-3).mean():7.3%}  "
              f"max err {err.max():.4f}  NA mismatches {na_mismatch}")


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else None)
