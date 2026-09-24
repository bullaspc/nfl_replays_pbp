"""
Rebuild models/fg_make_prob.csv: nflfastR's field-goal make probability by
yard line and roof type.

nflfastR's FG model is an mgcv GAM (`fastrmodels::fg_model`) that Python can't
evaluate. Its inputs are only yardline_100, era and roof, and nflfastR folds its
output into the published EP class probabilities of every FG attempt:

    no_score_prob = (1 - p_make) * P_missed(No_Score)

where P_missed is the EP model evaluated at the post-miss situation (see
`add_ep` in nflfastr_models.py). With the EP booster we recompute P_missed and
solve for p_make exactly. Seasons from 2018 on share one era (era4, which the
GAM groups with era3).

    python tools/build_fg_table.py 2018 2025
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nflfastr_models as nm  # noqa: E402

PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.parquet"
COLS = ["play_type", "no_score_prob", "yardline_100", "roof", "posteam", "home_team",
        "location", "half_seconds_remaining", "posteam_timeouts_remaining",
        "defteam_timeouts_remaining", "season"]


def recover(df: pd.DataFrame, ep_booster) -> pd.DataFrame:
    df = df[(df["play_type"] == "field_goal") & df["no_score_prob"].notna()].copy()
    for k, v in nm._era_cols(df["season"]).items():
        df[k] = v
    df["model_roof"] = nm._model_roof(df["roof"])
    for r in ("retractable", "dome", "outdoors"):
        df[r] = (df["model_roof"] == r).astype(float)
    df["home"] = (df["posteam"] == df["home_team"]).astype(float)  # _ep_preds handles neutral sites
    missed = df.copy()
    missed["half_seconds_remaining"] -= 5.065401
    missed["yardline_100"] = 100 - (missed["yardline_100"] + 8)
    missed["down"] = 1
    missed = nm._with_downs(missed)
    missed["ydstogo"] = 10.0
    ok = missed["half_seconds_remaining"] > 0
    p = 1 - df["no_score_prob"] / nm._ep_preds(missed, ep_booster)["No_Score"]
    return pd.DataFrame({"roof": df["model_roof"], "yardline_100": df["yardline_100"], "p": p})[ok]


def main(first: int, last: int) -> None:
    ep_booster = nm.load_models()["ep_model"]
    frames = [recover(pd.read_parquet(PBP_URL.format(y=y), columns=COLS), ep_booster)
              for y in range(first, last + 1)]
    t = pd.concat(frames)
    g = t.groupby(["roof", "yardline_100"])["p"]
    spread = (g.max() - g.min()).max()
    assert spread < 1e-6, f"make probability isn't a function of (roof, yard line): {spread}"
    out = g.first().reset_index().sort_values(["roof", "yardline_100"])
    out["yardline_100"] = out["yardline_100"].astype(int)
    out.to_csv(nm.FG_TABLE, index=False, float_format="%.10f")
    print(f"wrote {len(out)} rows to {nm.FG_TABLE}")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
