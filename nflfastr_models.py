"""
nflfastR's expected points and win probability models, in Python.

nflfastR computes `ep`, `epa`, `wp` and `home_wp` with xgboost models that it
ships in the `fastrmodels` R package. Each `.rda` there is an xz-compressed R
serialization wrapping one raw vector: the booster in xgboost's own UBJ format.
We pull that vector out and load it with the Python xgboost package, so the
predictions are nflfastR's, not a re-fit.

The feature preparation below mirrors nflfastR's R/helper_add_ep_wp.R
(`make_model_mutations`, `prepare_wp_data`, `add_ep_variables`,
`add_wp_variables`) as of nflfastR 6.0.0.9000, whose EP values a score net of
the possession the other team gets afterwards (`get_kickoff_values`,
`score_values`; nflfastR #618/#625) and whose field-goal model is a lookup
table (R/helper_fg_model.R). nflverse's published pbp is built with it, for
every season. `models/fg_prob_table.csv` and `KICK_START` are copied from
nflfastR's R/sysdata.rda by tools/build_fg_table.py.

Not ported: the overtime win probability model nflfastR rebuilt
(R/helper_add_ot_wp.R). Overtime `wp` here is the older formula from the EP
class probabilities, so it differs from nflverse's in overtime.
"""

from __future__ import annotations

import lzma
import os
import struct
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

MODEL_URL = "https://raw.githubusercontent.com/nflverse/fastrmodels/master/data/{name}.rda"
MODEL_NAMES = ("ep_model", "wp_model")
FG_TABLE = Path(__file__).parent / "models" / "fg_prob_table.csv"
CACHE_DIR = Path(os.environ.get("NFLFASTR_MODEL_CACHE",
                                Path.home() / ".cache" / "nfl_replays_pbp"))

EP_FEATURES = [
    "half_seconds_remaining", "yardline_100", "home", "retractable", "dome",
    "outdoors", "ydstogo", "era0", "era1", "era2", "era3", "era4",
    "down1", "down2", "down3", "down4",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining",
]
WP_FEATURES = [
    "receive_2h_ko", "home", "half_seconds_remaining", "game_seconds_remaining",
    "Diff_Time_Ratio", "score_differential", "down", "ydstogo", "yardline_100",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining",
]
EP_CLASSES = ["Touchdown", "Opp_Touchdown", "Field_Goal", "Opp_Field_Goal",
              "Safety", "Opp_Safety", "No_Score"]
# nflfastR's two_point_make_prob: tries from the 2 since 2004.
TWO_POINT_PROB = 0.49
# The receiving team's typical first snap after a kickoff, as yardline_100, by
# kickoff-rule regime: seasons before 2011, 2011-15, 2016-23, 2024, 2025 on
# (nflfastR's wp_kickoff_start(): wp_vegas_model$kick_start by ot_kick_first).
KICK_REGIMES = (2011, 2016, 2024, 2025)
KICK_START = (72.0, 78.0, 75.0, 70.0, 69.0)
# Where the scoring team's drive starts after a safety free kick, and the EP
# per yard between that and a kickoff start (safety_free_kick_yardline,
# safety_free_kick_ep_per_yard).
SAFETY_FREE_KICK_YARDLINE = 61.0
SAFETY_FREE_KICK_EP_PER_YARD = 0.05
# Field goal kicking eras (fg_era(): recency buckets, not rule changes).
FG_ERAS = (2004, 2009, 2014, 2023)


# ---------- Model loading ----------
def booster_from_rda(blob: bytes) -> xgb.Booster:
    """Extract the raw UBJ booster from a fastrmodels `.rda` file."""
    data = lzma.decompress(blob)
    # R's XDR serialization: a RAWSXP header (type 24) then a 4-byte length.
    i = data.find(b"\x00\x00\x00\x18")
    if i < 0:
        raise ValueError("no raw vector found in .rda")
    n = struct.unpack(">i", data[i + 4:i + 8])[0]
    booster = xgb.Booster()
    booster.load_model(bytearray(data[i + 8:i + 8 + n]))
    return booster


def load_models(cache_dir: Path = CACHE_DIR) -> dict[str, xgb.Booster]:
    """Download (once) and load nflfastR's EP and WP boosters."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    models = {}
    for name in MODEL_NAMES:
        path = cache_dir / f"{name}.rda"
        if not path.exists():
            with urllib.request.urlopen(MODEL_URL.format(name=name), timeout=60) as r:
                blob = r.read()
            tmp = path.with_suffix(".part")
            tmp.write_bytes(blob)
            tmp.replace(path)
        models[name] = booster_from_rda(path.read_bytes())
    return models


def load_fg_table(path: Path = FG_TABLE) -> pd.DataFrame:
    """nflfastR's fg_prob_table: make probability by yardline_100 (1-99, a
    kick of yardline_100 + 18 yards), one column per era x roof."""
    return pd.read_csv(path).set_index("yardline_100").sort_index()


# ---------- Feature prep ----------
def _model_roof(roof: pd.Series) -> pd.Series:
    r = roof.astype("object")
    return r.where(~(r.isna() | r.isin(["open", "closed"])), "retractable")


def _era_cols(season: pd.Series) -> dict[str, pd.Series]:
    s = season.astype(float)
    return {
        "era0": (s <= 2001).astype(float),
        "era1": ((s > 2001) & (s <= 2005)).astype(float),
        "era2": ((s > 2005) & (s <= 2013)).astype(float),
        "era3": ((s > 2013) & (s <= 2017)).astype(float),
        "era4": (s > 2017).astype(float),
    }


def _with_downs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for d in range(1, 5):
        df[f"down{d}"] = (df["down"] == d).astype(float)
    return df


def _ep_preds(df: pd.DataFrame, booster: xgb.Booster) -> pd.DataFrame:
    X = df[EP_FEATURES].astype(float)
    if "location" in df.columns:
        # nflfastR's get_preds(): a neutral site has no home team, for EP only.
        X.loc[(df["location"] == "Neutral").to_numpy(), "home"] = 0.0
    X = X.to_numpy()
    p = booster.predict(xgb.DMatrix(X)).reshape(-1, 7)
    return pd.DataFrame(p, columns=EP_CLASSES, index=df.index)


def _wp_preds(df: pd.DataFrame, booster: xgb.Booster) -> np.ndarray:
    X = df[WP_FEATURES].astype(float).to_numpy()
    return booster.predict(xgb.DMatrix(X))


def _arr(x, n: int | None = None) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return np.full(n, float(a)) if a.ndim == 0 and n is not None else a


def fg_make_prob(season, roof, yardline_100, table: pd.DataFrame) -> np.ndarray:
    """nflfastR's fg_make_prob(): the probability that a field goal from
    yardline_100 is good, by kicking era and roof (anything but "outdoors" is
    indoors; a missing roof is outdoors, a missing season the latest era)."""
    yl = _arr(yardline_100)
    n = yl.size if yl.ndim else 1
    yl = _arr(yardline_100, n)
    s = _arr(season, n)
    era = np.searchsorted(FG_ERAS, s, side="right") + 1  # findInterval; NaN → the latest
    r = pd.Series(np.broadcast_to(np.asarray(roof, dtype=object), (n,)))
    outdoors = (r.isna() | (r == "outdoors")).to_numpy()
    cols = [f"era{e}_{'outdoors' if o else 'indoors'}" for e, o in zip(era, outdoors)]
    out = np.full(n, np.nan)
    ok = ~np.isnan(yl)
    # R rounds half to even, as np.rint does.
    row = np.clip(np.rint(yl[ok]), 1, 99).astype(int)
    mat = table.to_numpy(float)
    ci = table.columns.get_indexer(np.asarray(cols, dtype=object)[ok])
    out[ok] = mat[row - table.index[0], ci]
    return out


def kickoff_start(season) -> np.ndarray:
    """nflfastR's wp_kickoff_start(): yardline_100 of the receiving team's
    typical first snap after a kickoff in that season (NaN → current rules)."""
    s = _arr(season)
    regime = np.searchsorted(KICK_REGIMES, np.where(np.isnan(s), np.inf, s), side="right")
    return np.asarray(KICK_START)[regime]


def conversion_value(season) -> np.ndarray:
    """Expected net points of the try after a touchdown (kick or two), the
    average of every try 2004-2025 in each era (nflfastR's conversion_value)."""
    s = _arr(season)
    return np.select([s <= 2014, s <= 2022, s > 2022], [0.99, 0.94, 0.96], np.nan)


def _first_and_ten(d: pd.DataFrame, yardline_100, flip: bool) -> pd.DataFrame:
    """1st and 10 at yardline_100 for the possession team or, flipped, the
    other team (home field and timeouts from its side)."""
    d = d.copy()
    if flip:
        d["home"] = 1 - d["home"]
        d["posteam_timeouts_remaining"], d["defteam_timeouts_remaining"] = (
            d["defteam_timeouts_remaining"].to_numpy(), d["posteam_timeouts_remaining"].to_numpy())
    d["yardline_100"] = yardline_100
    for k in range(1, 5):
        d[f"down{k}"] = 1.0 if k == 1 else 0.0
    d["ydstogo"] = 10.0
    return d


def _kickoff_values(df: pd.DataFrame, booster: xgb.Booster, kickoff_preds: pd.DataFrame | None = None,
                    exact_w=False) -> tuple[np.ndarray, np.ndarray]:
    """nflfastR's get_kickoff_values(): V, the EP of the possession the other
    team gets with a kickoff after a score, and W, the scoring team's after a
    safety free kick, at the current clock. Both come from the EP model, whose
    score values depend on V and W themselves, so they solve a pair of linear
    equations, on probabilities averaged over the two teams' versions of the
    state. 0 where no kickoff can follow (the end of a half). `kickoff_preds`
    are the possession team's predictions at the kickoff start (all rows of
    df), if the caller has them; `exact_w` marks rows whose W comes from the
    model rather than the per-yard shortcut (a safety is likely or is the play)."""
    n = len(df)
    v, w = np.zeros(n), np.zeros(n)
    hsr = df["half_seconds_remaining"].to_numpy(float)
    can = np.flatnonzero(~np.isnan(hsr) & (hsr > 0))
    if not len(can):
        return v, w
    exact = np.broadcast_to(np.asarray(exact_w, dtype=bool), (n,))[can]
    cols = EP_FEATURES + (["location"] if "location" in df.columns else [])
    base = df.iloc[can][cols]
    season = df["season"].to_numpy(float)[can]
    ko_yl = kickoff_start(season)
    xp = conversion_value(season)

    k_own = (kickoff_preds.iloc[can] if kickoff_preds is not None
             else _ep_preds(_first_and_ten(base, ko_yl, flip=False), booster))
    k_opp = _ep_preds(_first_and_ten(base, ko_yl, flip=True), booster)

    def coefs(p_own: pd.DataFrame, p_opp: pd.DataFrame, xp: np.ndarray):
        # EP of the state = a + b V + d W.
        p = (p_own[EP_CLASSES].to_numpy() + p_opp[EP_CLASSES].to_numpy()) / 2
        td, fg, sf = p[:, 0] - p[:, 1], p[:, 2] - p[:, 3], p[:, 4] - p[:, 5]
        return td * (6 + xp) + 3 * fg + 2 * sf, -(td + fg), sf

    a, b, d = coefs(k_own, k_opp, xp)
    # W = V + g, so V = (a + d g) / (1 - b - d).
    g = SAFETY_FREE_KICK_EP_PER_YARD * (ko_yl - SAFETY_FREE_KICK_YARDLINE)
    vv = (a + d * g) / (1 - b - d)
    ww = vv + g
    late = np.flatnonzero(exact)
    if len(late):
        fk = base.iloc[late]
        fa, fb, fd = coefs(_ep_preds(_first_and_ten(fk, SAFETY_FREE_KICK_YARDLINE, flip=False), booster),
                           _ep_preds(_first_and_ten(fk, SAFETY_FREE_KICK_YARDLINE, flip=True), booster),
                           xp[late])
        ka, kb, kd = a[late], b[late], d[late]
        den = (1 - kb) * (1 - fd) - kd * fb
        vv[late] = (ka * (1 - fd) + kd * fa) / den
        ww[late] = ((1 - kb) * fa + fb * ka) / den
    v[can], w[can] = vv, ww
    return v, w


def _ot_rule(game_id: pd.Series, season: np.ndarray) -> np.ndarray:
    """Overtime rules (nflfastR's ep_ot_rule): SD sudden death, MSD a first
    possession field goal can be answered, BOTH each team gets a possession.
    The postseason is read from the week in the game id."""
    week = pd.to_numeric(game_id.astype(str).str[5:7], errors="coerce").to_numpy(float)
    post = week > np.where(season >= 2021, 18, 17)
    return np.select([post & (season >= 2022), ~post & (season >= 2025),
                      post & (season >= 2010), ~post & (season >= 2012)],
                     ["BOTH", "BOTH", "MSD", "MSD"], "SD")


def _ot_continues(rule, possession, diff, pts) -> np.ndarray:
    """Does a kickoff follow a score of `pts` (7 touchdown, 3 field goal,
    2 safety) in overtime possession 1, 2 or 3+ (nflfastR's ep_ot_continues)?"""
    return np.select(
        [np.isnan(possession) | np.isnan(diff), rule == "SD", possession >= 3,
         (possession == 1) & (pts == 2), (possession == 1) & (pts == 7) & (rule == "MSD"),
         possession == 1, diff + pts == 0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], 0.0)


def _ot_td(rule, possession, diff):
    """(kickoff follows, conversion follows, it's a sure two-point try) for an
    overtime touchdown (nflfastR's ep_ot_td)."""
    cont = _ot_continues(rule, possession, diff, 7)
    two = (rule == "BOTH") & (possession == 2) & (diff == -7)
    cont[two] = 0.0
    return cont, ((cont == 1) | two).astype(float), two


def _continue_factors(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """nflfastR's ep_continue_factors(): 1 everywhere in regulation; in
    overtime whether a kickoff follows each kind of score, own or the other
    team's (assumed on the next possession), whether a conversion follows a
    touchdown, and whether this row is the sure two-point try."""
    n = len(df)
    out = {k: np.ones(n) for k in ("c_td_own", "c_fg_own", "c_saf_own", "c_td_opp", "c_fg_opp",
                                   "c_saf_opp", "k_td_own", "k_td_opp")}
    for k in ("two_td_own", "two_td_opp", "two_try"):
        out[k] = np.zeros(n, dtype=bool)
    needed = ("qtr", "game_id", "drive", "score_differential", "season")
    if not all(c in df.columns for c in needed):
        return out
    ot = (pd.to_numeric(df["qtr"], errors="coerce") > 4).to_numpy()
    if not ot.any():
        return out
    o = df.loc[ot]
    drive = pd.to_numeric(o["drive"], errors="coerce")
    first = drive.groupby(o["game_id"]).transform("min")
    possession = np.minimum((drive - first + 1).to_numpy(float), 3)
    season = pd.to_numeric(o["season"], errors="coerce").to_numpy(float)
    rule = _ot_rule(o["game_id"], season)
    diff = pd.to_numeric(o["score_differential"], errors="coerce").to_numpy(float)
    nxt = np.minimum(possession + 1, 3)
    for side, (poss, dd) in (("own", (possession, diff)), ("opp", (nxt, -diff))):
        cont, conv, two = _ot_td(rule, poss, dd)
        out[f"c_td_{side}"][ot], out[f"k_td_{side}"][ot], out[f"two_td_{side}"][ot] = cont, conv, two
        out[f"c_fg_{side}"][ot] = _ot_continues(rule, poss, dd, 3)
        out[f"c_saf_{side}"][ot] = _ot_continues(rule, poss, dd, 2)
    out["two_try"][ot] = (rule == "BOTH") & (possession == 2) & (diff == -1)
    return out


def _score_values(v: np.ndarray, w: np.ndarray, cf: dict, season: np.ndarray) -> dict[str, np.ndarray]:
    """Each kind of score's value for the possession team, net of the
    possession the other team gets afterwards (nflfastR's score_values): a
    touchdown is 6 + its conversion - V, a field goal 3 - V, a safety 2 + W."""
    conv = conversion_value(season)
    conv_own = np.where(cf["two_td_own"], 2 * TWO_POINT_PROB, conv)
    conv_opp = np.where(cf["two_td_opp"], 2 * TWO_POINT_PROB, conv)
    return {
        "td_own": 6 + cf["k_td_own"] * conv_own - cf["c_td_own"] * v,
        "fg_own": 3 - cf["c_fg_own"] * v,
        "saf_own": 2 + cf["c_saf_own"] * w,
        "td_opp": -6 - (cf["k_td_opp"] * conv_opp - cf["c_td_opp"] * v),
        "fg_opp": -3 + cf["c_fg_opp"] * v,
        "saf_opp": -2 - cf["c_saf_opp"] * w,
    }


def _flag(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0)


def _desc(df: pd.DataFrame) -> pd.Series:
    return df["desc"].fillna("").astype(str)


# ---------- EP / EPA ----------
def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """A column as numbers, NaN where missing (so a comparison is False, as
    R's which() drops NA)."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _is(s: pd.Series, value) -> pd.Series:
    return (s == value).fillna(False).astype(bool)


def add_ep(pbp: pd.DataFrame, models: dict, fg_table: pd.DataFrame,
           in_progress: bool = False) -> pd.DataFrame:
    """Games' pbp in (rows in game order) → adds ep, epa and the EP class
    probabilities, as nflfastR's add_ep_variables().

    Required columns: game_id, season, home_team, away_team, posteam, defteam,
    qtr, down, ydstogo, yardline_100, half_seconds_remaining,
    posteam_timeouts_remaining, defteam_timeouts_remaining, play_type, desc,
    sp, roof. Optional: location, drive, score_differential, td_team,
    safety_team, safety, touchdown, field_goal_result, extra_point_result,
    two_point_conv_result, timeout, kickoff_attempt, extra_point_attempt,
    two_point_attempt, play_type_nfl, defensive_two_point_conv.

    `in_progress`: the feed continues past each game's last row (a live game),
    so that row isn't the end of the game. nflfastR values a score on a game's
    last play at just its points, with no try or kickoff to follow.
    """
    df = pbp.copy()
    booster = models["ep_model"]
    for k, v in _era_cols(df["season"]).items():
        df[k] = v
    df = _with_downs(df)
    df["home"] = (df["posteam"] == df["home_team"]).astype(float)
    df["model_roof"] = _model_roof(df["roof"])
    for r in ("retractable", "dome", "outdoors"):
        df[r] = (df["model_roof"] == r).astype(float)

    desc = _desc(df)
    play_type = df["play_type"]
    season = pd.to_numeric(df["season"], errors="coerce").to_numpy(float)
    base = _ep_preds(df, booster)

    # Field goal attempts: blend the make probability with the EP of a miss,
    # the other team's ball at the spot of the kick or its own 20, whichever
    # is farther from its end zone.
    fg_i = (play_type == "field_goal").to_numpy()
    make_fg = fg_make_prob(season, df["roof"], df["yardline_100"], fg_table)
    if fg_i.any():
        missed = df.loc[fg_i].copy()
        missed["half_seconds_remaining"] = missed["half_seconds_remaining"] - 5.065401
        missed = _first_and_ten(missed, np.minimum(100 - (missed["yardline_100"] + 8), 80), flip=True)
        mpred = _ep_preds(missed, booster)
        end_i = (missed["half_seconds_remaining"] <= 0).to_numpy()
        mpred.loc[end_i, :] = 0.0
        mpred.loc[end_i, "No_Score"] = 1.0
        mpred = mpred.mul(1 - make_fg[fg_i], axis=0)
        swap = {"Touchdown": "Opp_Touchdown", "Opp_Field_Goal": "Field_Goal",
                "Opp_Touchdown": "Touchdown", "Safety": "Opp_Safety",
                "Opp_Safety": "Safety", "No_Score": "No_Score"}
        base.loc[fg_i, "Field_Goal"] = make_fg[fg_i] + mpred["Opp_Field_Goal"].to_numpy()
        for to, frm in swap.items():
            base.loc[fg_i, to] = mpred[frm].to_numpy()

    # Kickoffs: EP of the receiving team's typical first snap under the
    # season's kickoff rules. Every row's prediction there is also the
    # possession team's kickoff state in _kickoff_values().
    kpred = _ep_preds(_first_and_ten(df, kickoff_start(season), flip=False), booster)
    ko_att = (_num(df, "kickoff_attempt") if "kickoff_attempt" in df.columns
              else (play_type == "kickoff").astype(float))
    ko_i = ((play_type == "kickoff") | _is(ko_att, 1)).to_numpy()
    base.loc[ko_i, :] = kpred.loc[ko_i, :].to_numpy()
    # The free kick after a safety (a kickoff within 3 rows after a safety of
    # the same game) starts at the free kick spot; a touchback on a kickoff
    # from midfield is spotted at the 20 since 2026.
    is_safety = _is(_num(df, "safety"), 1)
    gid = df["game_id"]
    after_safety = np.zeros(len(df), dtype=bool)
    for back in (1, 2, 3):
        after_safety |= (is_safety.shift(back, fill_value=False).astype(bool)
                         & (gid.shift(back) == gid)).to_numpy()
    for rows, yl in ((ko_i & after_safety, SAFETY_FREE_KICK_YARDLINE),
                     (ko_i & (season >= 2026) & (df["yardline_100"] == 50).to_numpy(), 80.0)):
        if rows.any():
            st = _first_and_ten(df.loc[rows], yl, flip=False)
            base.loc[rows, :] = _ep_preds(st, booster).to_numpy()

    # QB kneels on your own side of the field can't score.
    kneel_i = ((play_type == "qb_kneel") & (df["yardline_100"] > 50)).to_numpy()
    base.loc[kneel_i, :] = 0.0
    base.loc[kneel_i, "No_Score"] = 1.0

    # PATs.
    ptn = df["play_type_nfl"] if "play_type_nfl" in df.columns else pd.Series(np.nan, index=df.index)
    xp_i = (((play_type == "extra_point") | (ptn == "XP_KICK"))
            & (ptn.isna() | (ptn != "PAT2"))).to_numpy()
    if "two_point_attempt" in df.columns:
        two_att = _num(df, "two_point_attempt")
    elif "two_point_conv_result" in df.columns:
        two_att = df["two_point_conv_result"].notna().astype(float)
    else:
        two_att = pd.Series(0.0, index=df.index)
    xp_att = (_num(df, "extra_point_attempt") if "extra_point_attempt" in df.columns
              else (play_type == "extra_point").astype(float))
    two_i = _is(two_att, 1).to_numpy()
    xp_prob = np.where(xp_i, make_fg, 0.0)
    two_prob = np.where(two_i, TWO_POINT_PROB, 0.0)

    # A penalty row around a try or the kickoff after it has no EP
    # (nflfastR's st_penalty_i_1 / st_penalty_i_2).
    td = _num(df, "touchdown")
    tmo = _num(df, "timeout")
    lag, lead = (lambda s: s.shift(1)), (lambda s: s.shift(-1))
    down_na = df["down"].isna()
    st1 = ((_is(td, 0) & down_na
            & (_is(lag(td), 1) | _is(lag(ptn), "XP_KICK") | _is(lag(tmo), 1))
            & (_is(lead(two_att), 1) | _is(lead(xp_att), 1) | _is(lead(ptn), "XP_KICK")))
           | ((_is(lag(two_att), 1) | _is(lag(xp_att), 1)) & _is(lead(ko_att), 1) & ~is_safety))
    no_play_no_down = down_na & (play_type == "no_play")
    st_term = no_play_no_down & desc.str.contains(
        "Kick formation|Pass formation|kicks onside|Offside on Free Kick", regex=True)
    st_two = no_play_no_down & desc.str.contains("TWO-POINT CONVERSION", regex=False) & _is(lead(ko_att), 0)
    st2 = lead(df["down"]).isna() & (st_term | st_two)
    st_i = (st1 | st2).to_numpy()

    # Timeouts and admin rows get no EP.
    kw = [" pass ", " sacked ", " scramble ", " punts ", " up the middle ",
          " left end ", " left guard ", " left tackle ", " right end ",
          " right guard ", " right tackle "]
    no_kw = ~desc.apply(lambda s: any(k in s for k in kw))
    missing_i = (((_flag(df, "timeout") == 1) & (play_type == "no_play") & no_kw)
                 | play_type.isna()).to_numpy()
    base.loc[missing_i | xp_i | two_i | st_i, :] = 0.0

    # The value of each score, net of the possession that follows it.
    v, w = _kickoff_values(df, booster, kickoff_preds=kpred,
                           exact_w=((base["Safety"] + base["Opp_Safety"]) > 0.01).to_numpy())
    cf = _continue_factors(df)
    sv = _score_values(v, w, cf, season)
    # An actual safety's endpoint always uses the model's W.
    end_saf_own, end_saf_opp = sv["saf_own"].copy(), sv["saf_opp"].copy()
    saf_rows = np.flatnonzero(is_safety.to_numpy())
    if len(saf_rows):
        vs, ws = _kickoff_values(df.iloc[saf_rows], booster, exact_w=True)
        cfs = {k: a[saf_rows] for k, a in cf.items()}
        svs = _score_values(vs, ws, cfs, season[saf_rows])
        end_saf_own[saf_rows], end_saf_opp[saf_rows] = svs["saf_own"], svs["saf_opp"]

    p = base[EP_CLASSES].to_numpy()
    ep = (p[:, 0] * sv["td_own"] + p[:, 1] * sv["td_opp"] + p[:, 2] * sv["fg_own"]
          + p[:, 3] * sv["fg_opp"] + p[:, 4] * sv["saf_own"] + p[:, 5] * sv["saf_opp"]
          + xp_prob + 2 * two_prob)
    ep[missing_i | st_i] = np.nan
    df["ExpPts"] = ep
    for c, name in zip(EP_CLASSES, ["td_prob", "opp_td_prob", "fg_prob", "opp_fg_prob",
                                    "safety_prob", "opp_safety_prob", "no_score_prob"]):
        df[name] = base[c].to_numpy()
    for k in ("td_own", "td_opp", "fg_own", "fg_opp"):
        df[f"_sv_{k}"] = sv[k]
    df["_end_saf_own"], df["_end_saf_opp"] = end_saf_own, end_saf_opp
    df["_conv"] = np.where(cf["two_try"], 2 * TWO_POINT_PROB, conversion_value(season))

    # EPA, grouped by game (nflfastR fills ep and posteam upward first).
    parts = []
    for _, g in df.groupby("game_id", sort=False):
        g = g.copy()
        g["ep"] = g["ExpPts"].bfill()
        tmp_pos = g["posteam"].bfill()
        known = tmp_pos.notna().to_numpy()
        home_side = (tmp_pos == g["home_team"]).to_numpy()
        home_ep = pd.Series(np.where(known, np.where(home_side, g["ep"], -g["ep"]), np.nan), index=g.index)
        home_epa = home_ep.shift(-1) - home_ep
        epa = pd.Series(np.where(known, np.where(home_side, home_epa, -home_epa), np.nan),
                        index=g.index, dtype=float)

        gd = _desc(g)
        end_game = gd.str.lower().str.contains("(?:end of game)|(?:end game)")
        # A score on a game's last play has no try or kickoff after it.
        game_over = end_game.shift(-1, fill_value=not in_progress).astype(bool)
        go = game_over.to_numpy()

        td_team = g["td_team"] if "td_team" in g.columns else pd.Series(np.nan, index=g.index)
        has_td = td_team.notna()
        own_td = (td_team == g["posteam"]).to_numpy()
        td_val = np.where(own_td, np.where(go, 6.0, g["_sv_td_own"]), np.where(go, -6.0, g["_sv_td_opp"]))
        epa = epa.where(~has_td, td_val - g["ep"])

        def result(col, *vals):
            if col not in g.columns:
                return pd.Series(False, index=g.index)
            return g[col].isin(vals)
        fgm, xpg = result("field_goal_result", "made"), result("extra_point_result", "good")
        xpf = result("extra_point_result", "failed", "blocked", "aborted")
        tpg = result("two_point_conv_result", "success")
        tpf = result("two_point_conv_result", "failure")
        no_td = ~has_td
        conv = g["_conv"]
        epa = epa.where(~(no_td & fgm), np.where(go, 3.0, g["_sv_fg_own"]) - g["ep"])
        # A try is measured against the conversion value in the touchdown's.
        epa = epa.where(~(no_td & ~fgm & xpg), 1 - conv)
        epa = epa.where(~(no_td & ~fgm & ~xpg & tpg), 2 - conv)
        epa = epa.where(~(no_td & ~fgm & ~xpg & (xpf | tpf)), 0 - conv)
        if "defensive_two_point_conv" in g.columns:
            epa = epa.where(~(_flag(g, "defensive_two_point_conv") == 1), -2 - conv)
        if "safety_team" in g.columns:
            st = g["safety_team"]
            epa = epa.where(~(st.notna() & (st == g["defteam"])),
                            np.where(go, -2.0, g["_end_saf_opp"]) - g["ep"])
            epa = epa.where(~(st.notna() & (st == g["posteam"])),
                            np.where(go, 2.0, g["_end_saf_own"]) - g["ep"])

        nxt_q, nxt_d, nxt_end = g["qtr"].shift(-1), gd.shift(-1), end_game.shift(-1, fill_value=False)
        sp0 = pd.to_numeric(g["sp"], errors="coerce") == 0
        half_end = (((g["qtr"] == 2) & ((nxt_q == 3) | (nxt_d == "END QUARTER 2")))
                    | ((g["qtr"] == 4) & ((nxt_q == 5) | (nxt_d == "END QUARTER 4") | nxt_end)))
        epa = epa.where(~(half_end & sp0 & g["play_type"].notna()), 0 - g["ep"])
        epa = epa.where(~((g["qtr"] > 4) & nxt_end & sp0), 0 - g["ep"])
        blank = (gd == "END QUARTER 2") | end_game
        epa[blank] = np.nan
        g["epa"] = epa
        g.loc[blank, "ep"] = np.nan
        parts.append(g)
    drop = ["ExpPts", "_conv", "_end_saf_own", "_end_saf_opp"] + [f"_sv_{k}" for k in ("td_own", "td_opp",
                                                                                     "fg_own", "fg_opp")]
    return pd.concat(parts).drop(columns=drop)


# ---------- WP ----------
def add_wp(pbp: pd.DataFrame, models: dict, fg_table: pd.DataFrame) -> pd.DataFrame:
    """Adds wp, home_wp, away_wp (nflfastR's non-spread model, like nflverse's
    `home_wp`). Expects the output of `add_ep` plus score_differential,
    game_seconds_remaining, drive, total_home_score, total_away_score."""
    df = pbp.copy()
    wp_b, ep_b = models["wp_model"], models["ep_model"]
    desc = _desc(df)

    first_def = df.groupby("game_id")["defteam"].transform(
        lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
    df["receive_2h_ko"] = ((df["qtr"] <= 2) & (df["posteam"] == first_def)).astype(float)
    df["elapsed_share"] = (3600 - df["game_seconds_remaining"]) / 3600
    df["Diff_Time_Ratio"] = df["score_differential"] / np.exp(-4 * df["elapsed_share"])

    off_wp = np.full(len(df), np.nan)
    ot = (df["qtr"] > 4).to_numpy()
    if ot.any():
        od = df[ot].copy()
        drive_diff = od["drive"] - od["drive"].min()
        one_fg_game = (od["score_differential"] == -3) & (drive_diff == 1)
        # nflfastR sets `yrdline100` here (a typo), so the kickoff yard line
        # is left alone; keep that to match its numbers.
        od_ko = od.copy()
        od_ko["down"] = 1
        od_ko = _with_downs(od_ko)
        od_ko["ydstogo"] = 10.0
        kp = _ep_preds(od_ko, ep_b)
        win_back = kp["No_Score"] + kp["Opp_Field_Goal"] + kp["Opp_Safety"] + kp["Opp_Touchdown"]
        sudden = od["fg_prob"] + od["td_prob"] + od["safety_prob"]
        one_fg = od["td_prob"] + od["fg_prob"] * win_back
        use_one = (od["season"] >= 2012) & ((drive_diff == 0) | ((drive_diff == 1) & one_fg_game))
        off_wp[ot] = np.where(use_one, one_fg, sudden)

    reg = (df["qtr"] <= 4).to_numpy()
    if reg.any():
        off_wp[reg] = _wp_preds(df[reg], wp_b)
    off_wp[df["down"].isna().to_numpy()] = np.nan

    # PAT fix: WP after the try, from the kicking team's view. The try is a
    # field goal from the 15 since 2015 (the 2 before), and the other team then
    # has its typical first snap after the kickoff.
    season = pd.to_numeric(df["season"], errors="coerce").to_numpy(float)
    make_pat = fg_make_prob(season, df["roof"], np.where(season >= 2015, 15, 2), fg_table)
    ko_flag = _flag(df, "kickoff_attempt")
    if "kickoff_attempt" not in df.columns:
        ko_flag = (df["play_type"] == "kickoff").astype(float)
    xp_res = df["extra_point_result"] if "extra_point_result" in df else pd.Series(np.nan, index=df.index)
    tp_res = df["two_point_conv_result"] if "two_point_conv_result" in df else pd.Series(np.nan, index=df.index)
    pat_i = (((ko_flag == 0) & ~desc.str.contains("Onside Kick") & desc.str.contains("Kick formation")
              & df["down"].isna()) | desc.str.contains("extra point") | xp_res.notna()).to_numpy()
    two_i = (((ko_flag == 0) & ~desc.str.contains("Onside Kick") & desc.str.contains("Pass formation")
              & df["down"].isna()) | desc.str.contains("TWO-POINT CONVERSION ATTEMPT")
             | tp_res.notna()).to_numpy()
    pat_i &= ~two_i
    if pat_i.any() or two_i.any():
        pd_ = df.copy()
        pd_["posteam_timeouts_remaining"] = df["defteam_timeouts_remaining"]
        pd_["defteam_timeouts_remaining"] = df["posteam_timeouts_remaining"]
        pd_["score_differential"] = -df["score_differential"]
        pd_["down"] = 1.0
        pd_["ydstogo"] = 10.0
        pd_["receive_2h_ko"] = np.where(df["qtr"] <= 2, 1 - df["receive_2h_ko"], df["receive_2h_ko"])
        pd_["home"] = 1 - df["home"]
        pd_["yardline_100"] = kickoff_start(season)

        def _pat_wp(delta):
            t = pd_.copy()
            t["score_differential"] = t["score_differential"] - delta
            t["Diff_Time_Ratio"] = t["score_differential"] / np.exp(-4 * t["elapsed_share"])
            return _wp_preds(t, wp_b)
        p0, p1, p2 = _pat_wp(0), _pat_wp(1), _pat_wp(2)
        go1 = 1 - (make_pat * p1 + (1 - make_pat) * p0)
        go2 = 1 - (TWO_POINT_PROB * p2 + (1 - TWO_POINT_PROB) * p0)
        off_wp[two_i] = go2[two_i]
        off_wp[pat_i] = go1[pat_i]

    # Kickoffs: WP at the receiving team's typical first snap under the
    # season's kickoff rules.
    ko = df.copy()
    ko["yardline_100"] = kickoff_start(season)
    ko["down"] = 1.0
    ko["ydstogo"] = 10.0
    kpred = _wp_preds(ko, wp_b)
    ko_i = (((df["play_type"] == "kickoff") | (ko_flag == 1)) & (df["qtr"] <= 4)).to_numpy()
    off_wp[ko_i] = kpred[ko_i]

    df["wp"] = off_wp
    parts = []
    for _, g in df.groupby("game_id", sort=False):
        g = g.copy()
        # The team comes from the row the wp is filled from (nflfastR #606):
        # a non-play can carry a stale posteam.
        tmp_pos = g["posteam"].where(g["wp"].notna()).bfill()
        g["wp"] = g["wp"].bfill()
        known = tmp_pos.notna().to_numpy()
        g["home_wp"] = np.where(known, np.where(tmp_pos == g["home_team"], g["wp"], 1 - g["wp"]), np.nan)
        end_game = _desc(g).str.lower().str.contains("(?:end of game)|(?:end game)")
        hs, as_ = g["total_home_score"], g["total_away_score"]
        final = np.select([hs > as_, as_ > hs], [1.0, 0.0], 0.5)
        g["home_wp"] = np.where(end_game, final, g["home_wp"])
        g["away_wp"] = 1 - g["home_wp"]
        g.loc[end_game, "wp"] = np.nan
        parts.append(g)
    return pd.concat(parts)


def wp_for_states(states: pd.DataFrame, models: dict) -> np.ndarray:
    """nflfastR's (non-spread) win probability for the offense in made-up
    regulation states: posteam's side, clock, score margin, down, distance,
    spot and timeouts, under WP_FEATURES' names except Diff_Time_Ratio, which
    is derived here as add_wp() does. For "what if" states, not plays."""
    df = states.copy()
    elapsed_share = (3600 - df["game_seconds_remaining"]) / 3600
    df["Diff_Time_Ratio"] = df["score_differential"] / np.exp(-4 * elapsed_share)
    return _wp_preds(df, models["wp_model"])


def add_ep_wp(pbp: pd.DataFrame, models: dict, fg_table: pd.DataFrame,
              in_progress: bool = False) -> pd.DataFrame:
    """`in_progress`: a live game's feed, which continues past its last row
    (see add_ep)."""
    return add_wp(add_ep(pbp, models, fg_table, in_progress), models, fg_table)
