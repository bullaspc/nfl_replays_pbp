"""
Does knowing the teams predict 4th-down conversions better than the league
table (models/fourth_down.csv) alone? Each 4th-down try 2019–2025 (regular
season) gets the offense's and the defense's numbers from the season's earlier
weeks only (3rd/4th-down conversions over the table's expectation, short
yardage, EPA/play, success rate), shrunk by their plays; a logistic model on
the table's logit plus those is fit on the other seasons and scored on each
season left out.

    python tools/validate_team_conversion.py <dir>

`dir` holds play_by_play_2019..2025.parquet. Result (2026-11): no feature set
beat the table (log loss 0.6445 alone, 0.6447–0.6453 with team numbers), so
the 4th-down decisions use the league table.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import replay_core as core  # noqa: E402

D = sys.argv[1]
COLS = ["game_id","play_id","season","week","season_type","posteam","defteam","qtr","down","ydstogo",
        "yardline_100","play_type","first_down","pass_touchdown","rush_touchdown","epa","qb_kneel","qb_spike"]
pbp = pd.concat([pd.read_parquet(f"{D}/play_by_play_{y}.parquet", columns=COLS) for y in range(2019, 2026)])
pbp = pbp[pbp.season_type == "REG"]
table = pd.read_csv(Path(__file__).resolve().parent.parent / "models" / "fourth_down.csv")
rate = table[table.kind == "go"].set_index("bin")["rate"]
snap = pbp.play_type.isin(["pass", "run"]) & (pbp.qtr <= 4) & (pbp.qb_kneel != 1) & (pbp.qb_spike != 1)
s = pbp[snap].copy()
s["conv"] = ((s.first_down == 1) | (s.pass_touchdown == 1) | (s.rush_touchdown == 1)).astype(float)
s["exp"] = rate.reindex(core._fourth_bin(s.ydstogo)).to_numpy()
s["late"] = s.down.isin([3, 4])
s["short"] = s.ydstogo <= 3
# per team-week sums (offense and defense), then season-to-date before the week
def to_date(side):
    g = s.groupby(["season", side, "week"])
    w = pd.DataFrame({
        "n_late": g.late.sum(), "oe_late": g.apply(lambda d: (d.conv - d.exp)[d.late].sum(), include_groups=False),
        "n_short": g.apply(lambda d: (d.late & d.short).sum(), include_groups=False),
        "oe_short": g.apply(lambda d: (d.conv - d.exp)[d.late & d.short].sum(), include_groups=False),
        "n": g.size(), "epa": g.epa.sum(), "sr": g.apply(lambda d: (d.epa > 0).sum(), include_groups=False),
    }).reset_index().sort_values(["season", side, "week"])
    c = w.groupby(["season", side])[["n_late","oe_late","n_short","oe_short","n","epa","sr"]].transform(
        lambda x: x.cumsum().shift().fillna(0))
    c[["season", side, "week"]] = w[["season", side, "week"]]
    return c
STATS = ["n_late","oe_late","n_short","oe_short","n","epa","sr"]
off = to_date("posteam").rename(columns={c: c + "_o" for c in STATS})
dfn = to_date("defteam").rename(columns={c: c + "_d" for c in STATS})
f = s[(s.down == 4)][["season","week","posteam","defteam","ydstogo","conv","exp"]]
f = f.merge(off, on=["season", "posteam", "week"], how="left").merge(
    dfn, on=["season", "defteam", "week"], how="left").fillna(0)
K = {"late": 10, "short": 5, "n": 50}
def shr(oe, n, k): return oe / (n + k)
X = pd.DataFrame({
    "logit": np.log(f.exp / (1 - f.exp)),
    "off_late": shr(f.oe_late_o, f.n_late_o, K["late"]), "def_late": shr(f.oe_late_d, f.n_late_d, K["late"]),
    "off_short": np.where(f.ydstogo <= 3, shr(f.oe_short_o, f.n_short_o, K["short"]), 0),
    "def_short": np.where(f.ydstogo <= 3, shr(f.oe_short_d, f.n_short_d, K["short"]), 0),
    "off_epa": f.epa_o / (f.n_o + K["n"]), "def_epa": f.epa_d / (f.n_d + K["n"]),
    "off_sr": (f.sr_o - 0.45 * f.n_o) / (f.n_o + K["n"]), "def_sr": (f.sr_d - 0.45 * f.n_d) / (f.n_d + K["n"]),
})
y = f.conv.to_numpy(); season = f.season.to_numpy()
sets = {"league table": ["logit"], "+ team conv (late downs)": ["logit","off_late","def_late"],
        "+ short-yardage conv": ["logit","off_late","def_late","off_short","def_short"],
        "+ EPA/play to date": ["logit","off_epa","def_epa"], "+ SR to date": ["logit","off_sr","def_sr"],
        "all": list(X.columns)}
print(f"{len(f)} 4th-down tries 2019–2025")
for name, cols in sets.items():
    ll = br = 0.0
    for yr in np.unique(season):
        tr, te = season != yr, season == yr
        A = np.column_stack([np.ones(len(X)), X[cols].to_numpy()])
        if cols == ["logit"]:
            p = 1 / (1 + np.exp(-A[te, 1]))
        else:
            w = core._logit_fit(A[tr], y[tr], l2=1e-3)
            p = core._sigmoid(A[te] @ w)
        ll += -(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p)).sum(); br += ((y[te] - p) ** 2).sum()
    print(f"{name:28s} log loss {ll / len(y):.4f}  Brier {br / len(y):.4f}")
A = np.column_stack([np.ones(len(X)), X.to_numpy()]); w = core._logit_fit(A, y, l2=1e-3)
print(dict(zip(["const", *X.columns], np.round(w, 3))))
