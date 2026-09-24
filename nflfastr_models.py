"""
nflfastR's expected points and win probability models, in Python.

nflfastR computes `ep`, `epa`, `wp` and `home_wp` with xgboost models that it
ships in the `fastrmodels` R package. Each `.rda` there is an xz-compressed R
serialization wrapping one raw vector: the booster in xgboost's own UBJ format.
We pull that vector out and load it with the Python xgboost package, so the
predictions are nflfastR's, not a re-fit.

The feature preparation below mirrors nflfastR's R/helper_add_ep_wp.R
(`make_model_mutations`, `prepare_wp_data`, `add_ep_variables`,
`add_wp_variables`). Differences, all confined to rare rows:
  * PAT/kickoff penalty rows that nflfastR blanks out (`st_penalty_i_*`) are
    treated like ordinary rows.
  * The field-goal make probability comes from a mgcv GAM that can't run in
    Python. It depends only on yard line, era and roof type, so
    `models/fg_make_prob.csv` holds its values, recovered exactly from
    published nflverse pbp by tools/build_fg_table.py.
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
FG_TABLE = Path(__file__).parent / "models" / "fg_make_prob.csv"
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
EP_VALUES = np.array([7, -7, 3, -3, 2, -2, 0], dtype=float)
TWO_POINT_PROB = 0.4735


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
    return pd.read_csv(path)


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


def fg_make_prob(yardline_100: pd.Series, model_roof: pd.Series, season: pd.Series,
                 table: pd.DataFrame) -> np.ndarray:
    """nflfastR's FG make probability, from the recovered GAM table.

    The table covers the modern era (the GAM's era factor is 3 for 2014+).
    Yard lines between recorded values are interpolated on the logit scale;
    beyond the recorded range the nearest edge's slope is extended.
    """
    out = np.full(len(yardline_100), np.nan)
    yl = yardline_100.to_numpy(dtype=float)
    roof = model_roof.to_numpy()
    for r, grp in table.groupby("roof"):
        grp = grp.sort_values("yardline_100")
        x = grp["yardline_100"].to_numpy(float)
        lg = np.log(grp["p"] / (1 - grp["p"])).to_numpy(float)
        m = (roof == r) & ~np.isnan(yl)
        if not m.any():
            continue
        v = np.interp(yl[m], x, lg)
        lo, hi = yl[m] < x[0], yl[m] > x[-1]
        v[lo] = lg[0] + (yl[m][lo] - x[0]) * (lg[1] - lg[0]) / (x[1] - x[0])
        v[hi] = lg[-1] + (yl[m][hi] - x[-1]) * (lg[-1] - lg[-2]) / (x[-1] - x[-2])
        out[m] = 1 / (1 + np.exp(-v))
    return out


def _flag(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0)


def _desc(df: pd.DataFrame) -> pd.Series:
    return df["desc"].fillna("").astype(str)


# ---------- EP / EPA ----------
def add_ep(pbp: pd.DataFrame, models: dict, fg_table: pd.DataFrame) -> pd.DataFrame:
    """One game's pbp in → adds ep, epa and the EP class probabilities.

    Required columns: game_id, season, home_team, away_team, posteam, defteam,
    qtr, down, ydstogo, yardline_100, half_seconds_remaining,
    posteam_timeouts_remaining, defteam_timeouts_remaining, play_type, desc,
    sp, roof. Optional: location, td_team, safety_team, field_goal_result,
    extra_point_result, two_point_conv_result, timeout, kickoff_attempt,
    defensive_two_point_conv.
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
    base = _ep_preds(df, booster)

    # Field goal attempts: blend the make probability with the EP of a miss.
    missed = df.copy()
    missed["half_seconds_remaining"] = missed["half_seconds_remaining"] - 5.065401
    missed["yardline_100"] = 100 - (missed["yardline_100"] + 8)
    for d in range(1, 5):
        missed[f"down{d}"] = 1.0 if d == 1 else 0.0
    missed["ydstogo"] = 10.0
    mpred = _ep_preds(missed, booster)
    end_i = (missed["half_seconds_remaining"] <= 0).to_numpy()
    mpred.loc[end_i, :] = 0.0
    mpred.loc[end_i, "No_Score"] = 1.0
    make_fg = fg_make_prob(df["yardline_100"], df["model_roof"], df["season"], fg_table)
    mpred = mpred.mul(1 - make_fg, axis=0)
    fg_i = (play_type == "field_goal").to_numpy()
    swap = {"Touchdown": "Opp_Touchdown", "Opp_Field_Goal": "Field_Goal",
            "Opp_Touchdown": "Touchdown", "Safety": "Opp_Safety",
            "Opp_Safety": "Safety", "No_Score": "No_Score"}
    base.loc[fg_i, "Field_Goal"] = make_fg[fg_i] + mpred.loc[fg_i, "Opp_Field_Goal"]
    for to, frm in swap.items():
        base.loc[fg_i, to] = mpred.loc[fg_i, frm]

    # Kickoffs: EP of a touchback from the receiving team's side.
    ko = df.copy()
    ko["yardline_100"] = np.where(ko["season"] < 2016, 80, 75)
    for d in range(1, 5):
        ko[f"down{d}"] = 1.0 if d == 1 else 0.0
    ko["ydstogo"] = 10.0
    kpred = _ep_preds(ko, booster)
    ko_i = ((play_type == "kickoff") | (_flag(df, "kickoff_attempt") == 1)).to_numpy()
    base.loc[ko_i, :] = kpred.loc[ko_i, :]

    # QB kneels on your own side of the field can't score.
    kneel_i = ((play_type == "qb_kneel") & (df["yardline_100"] > 50)).to_numpy()
    base.loc[kneel_i, :] = 0.0
    base.loc[kneel_i, "No_Score"] = 1.0

    # PATs.
    xp_prob = np.zeros(len(df))
    two_prob = np.zeros(len(df))
    xp_i = (play_type == "extra_point").to_numpy()
    if "play_type_nfl" in df.columns:
        ptn = df["play_type_nfl"]
        xp_i = (((play_type == "extra_point") | (ptn == "XP_KICK"))
                & (ptn.isna() | (ptn != "PAT2"))).to_numpy()
    if "two_point_attempt" in df.columns:
        two_i = (_flag(df, "two_point_attempt") == 1).to_numpy()
    else:
        two_i = df.get("two_point_conv_result", pd.Series(np.nan, index=df.index)).notna().to_numpy()
    xp_prob[xp_i] = make_fg[xp_i]
    two_prob[two_i] = TWO_POINT_PROB

    # Timeouts and admin rows get no EP.
    kw = [" pass ", " sacked ", " scramble ", " punts ", " up the middle ",
          " left end ", " left guard ", " left tackle ", " right end ",
          " right guard ", " right tackle "]
    no_kw = ~desc.apply(lambda s: any(k in s for k in kw))
    missing_i = (((_flag(df, "timeout") == 1) & (play_type == "no_play") & no_kw)
                 | play_type.isna()).to_numpy()
    zero_i = missing_i | xp_i | two_i
    base.loc[zero_i, :] = 0.0

    ep = base.to_numpy() @ EP_VALUES + xp_prob + 2 * two_prob
    ep[missing_i] = np.nan
    df["ExpPts"] = ep
    for c, name in zip(EP_CLASSES, ["td_prob", "opp_td_prob", "fg_prob", "opp_fg_prob",
                                    "safety_prob", "opp_safety_prob", "no_score_prob"]):
        df[name] = base[c].to_numpy()

    # EPA, grouped by game (nflfastR fills ep and posteam upward first).
    parts = []
    for _, g in df.groupby("game_id", sort=False):
        g = g.copy()
        g["ep"] = g["ExpPts"].bfill()
        tmp_pos = g["posteam"].bfill()
        home_ep = np.where(tmp_pos == g["home_team"], g["ep"], -g["ep"])
        home_epa = pd.Series(home_ep, index=g.index).shift(-1) - home_ep
        epa = np.where(tmp_pos == g["home_team"], home_epa, -home_epa)
        epa = pd.Series(epa, index=g.index, dtype=float)

        td_team = g["td_team"] if "td_team" in g.columns else pd.Series(np.nan, index=g.index)
        has_td = td_team.notna()
        epa = epa.where(~has_td, np.where(td_team == g["posteam"], 7 - g["ep"], -7 - g["ep"]))

        fgm = (g.get("field_goal_result") == "made") if "field_goal_result" in g else False
        xpg = (g.get("extra_point_result") == "good") if "extra_point_result" in g else False
        xpf = (g["extra_point_result"].isin(["failed", "blocked", "aborted"])
               if "extra_point_result" in g else False)
        tpg = (g["two_point_conv_result"] == "success") if "two_point_conv_result" in g else False
        tpf = (g["two_point_conv_result"] == "failure") if "two_point_conv_result" in g else False
        no_td = ~has_td
        epa = epa.where(~(no_td & fgm), 3 - g["ep"])
        epa = epa.where(~(no_td & ~fgm & xpg), 1 - g["ep"])
        epa = epa.where(~(no_td & ~fgm & ~xpg & tpg), 2 - g["ep"])
        epa = epa.where(~(no_td & ~fgm & ~xpg & (xpf | tpf)), 0 - g["ep"])
        if "defensive_two_point_conv" in g.columns:
            epa = epa.where(~(_flag(g, "defensive_two_point_conv") == 1), -2 - g["ep"])
        if "safety_team" in g.columns:
            st = g["safety_team"]
            epa = epa.where(~(st.notna() & (st == g["posteam"])), 2 - g["ep"])
            epa = epa.where(~(st.notna() & (st == g["defteam"])), -2 - g["ep"])

        gd = _desc(g)
        end_game = gd.str.lower().str.contains("(?:end of game)|(?:end game)")
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
    return pd.concat(parts).drop(columns=["ExpPts"])


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

    # PAT fix: WP after the try, from the kicking team's view.
    make_pat = fg_make_prob(pd.Series(np.where(df["season"] >= 2015, 15, 3), index=df.index),
                            df["model_roof"], df["season"], fg_table)
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
        pd_["yardline_100"] = 75.0

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

    # Kickoffs: WP at a touchback, receiving team's view.
    ko = df.copy()
    ko["yardline_100"] = np.where(ko["season"] < 2016, 80, 75)
    ko["down"] = 1.0
    ko["ydstogo"] = 10.0
    kpred = _wp_preds(ko, wp_b)
    ko_i = (((df["play_type"] == "kickoff") | (ko_flag == 1)) & (df["qtr"] <= 4)).to_numpy()
    off_wp[ko_i] = kpred[ko_i]

    df["wp"] = off_wp
    parts = []
    for _, g in df.groupby("game_id", sort=False):
        g = g.copy()
        g["wp"] = g["wp"].bfill()
        tmp_pos = g["posteam"].bfill()
        g["home_wp"] = np.where(tmp_pos == g["home_team"], g["wp"], 1 - g["wp"])
        end_game = _desc(g).str.lower().str.contains("(?:end of game)|(?:end game)")
        hs, as_ = g["total_home_score"], g["total_away_score"]
        final = np.select([hs > as_, as_ > hs], [1.0, 0.0], 0.5)
        g["home_wp"] = np.where(end_game, final, g["home_wp"])
        g["away_wp"] = 1 - g["home_wp"]
        g.loc[end_game, "wp"] = np.nan
        parts.append(g)
    return pd.concat(parts)


def add_ep_wp(pbp: pd.DataFrame, models: dict, fg_table: pd.DataFrame) -> pd.DataFrame:
    return add_wp(add_ep(pbp, models, fg_table), models, fg_table)
