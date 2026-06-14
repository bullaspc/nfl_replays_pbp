"""
NFL Tape-Delay Replay Boxscore
-------------------------------
A spoiler-free way to follow an NFL game on tape delay.

You tell the app:
  1. Which game you're watching
  2. When you started watching (i.e. your personal "kickoff")
The app reveals plays, score, and stats only up to your current viewing point.

Run with:
    pip install streamlit nfl_data_py pandas plotly
    streamlit run nfl_replay_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import nfl_data_py as nfl
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta, time

st.set_page_config(page_title="NFL Replay Boxscore", layout="wide", page_icon="🏈")

# ---------- Data loading ----------
@st.cache_data(ttl=3600)
def load_team_colors() -> dict[str, str]:
    """Map team abbreviation → primary hex color."""
    df = nfl.import_team_desc()
    return dict(zip(df["team_abbr"], df["team_color"]))


@st.cache_data(ttl=120)  # refresh every 2 minutes; pbp updates aren't instant anyway
def load_pbp(season: int) -> pd.DataFrame:
    """Load play-by-play for a given season."""
    cols = [
        "game_id", "season", "week", "game_date", "home_team", "away_team",
        "posteam", "defteam", "qtr", "time", "game_seconds_remaining",
        "play_id", "desc", "play_type", "down", "ydstogo", "yards_gained",
        "touchdown", "field_goal_result", "extra_point_result",
        "two_point_conv_result", "safety", "sp",
        "total_home_score", "total_away_score",
        "home_wp", "away_wp", "epa",
        "passer_player_name", "rusher_player_name", "receiver_player_name",
        "passing_yards", "rushing_yards", "receiving_yards",
        "pass_touchdown", "rush_touchdown",
        "interception", "fumble_lost", "sack", "qb_hit",
        "complete_pass", "pass_attempt", "rush_attempt",
        "first_down",
        "air_yards",
        "third_down_converted", "third_down_failed",
        "fourth_down_converted", "fourth_down_failed",
        "goal_to_go", "yardline_100", "drive",
        # defensive player columns
        "solo_tackle_1_player_name", "solo_tackle_2_player_name",
        "assist_tackle_1_player_name", "assist_tackle_2_player_name",
        "assist_tackle_3_player_name", "assist_tackle_4_player_name",
        "sack_player_name", "half_sack_1_player_name", "half_sack_2_player_name",
        "qb_hit_1_player_name", "qb_hit_2_player_name",
        "tackle_for_loss_1_player_name", "tackle_for_loss_2_player_name",
        "interception_player_name",
        "pass_defense_1_player_name", "pass_defense_2_player_name",
        "forced_fumble_player_1_player_name", "forced_fumble_player_2_player_name",
    ]
    df = nfl.import_pbp_data([season], columns=cols, downcast=True)
    return df


def list_games(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per game with date, teams, and game_id."""
    g = (
        pbp.groupby("game_id", as_index=False)
        .agg(week=("week", "first"),
             game_date=("game_date", "first"),
             home_team=("home_team", "first"),
             away_team=("away_team", "first"))
        .sort_values(["week", "game_date"])
    )
    g["label"] = g.apply(
        lambda r: f"Wk {int(r['week'])} — {r['away_team']} @ {r['home_team']} ({r['game_date']})",
        axis=1,
    )
    return g


# ---------- Replay logic ----------
def elapsed_game_seconds(viewing_minutes: float) -> float:
    """
    Convert viewing-elapsed minutes into elapsed *game* seconds.
    A real NFL broadcast is ~3h10m for 60 minutes of game clock.
    We treat viewing time as a linear stretch of game clock.
    """
    BROADCAST_MINUTES = 190.0   # ~3h10m for a full game
    GAME_SECONDS = 3600.0
    frac = min(max(viewing_minutes / BROADCAST_MINUTES, 0.0), 1.0)
    return frac * GAME_SECONDS


def filter_revealed(pbp_game: pd.DataFrame, elapsed_game_s: float) -> pd.DataFrame:
    """Keep only plays that have happened by 'elapsed_game_s' of game time."""
    # game_seconds_remaining counts DOWN from 3600 at kickoff to 0 at final whistle
    # Null-clock rows (timeouts, end-of-period admin plays) inherit the nearest
    # real clock value so they aren't accidentally always-revealed.
    clock = pbp_game["game_seconds_remaining"].ffill().bfill().fillna(3600)
    played_at = 3600 - clock
    return pbp_game[played_at <= elapsed_game_s].copy()


# ---------- Stat builders ----------
def boxscore(revealed: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    """Quarter-by-quarter score from revealed plays."""
    if revealed.empty:
        return pd.DataFrame({"Team": [away, home], "Q1": [0, 0], "Q2": [0, 0],
                             "Q3": [0, 0], "Q4": [0, 0], "OT": [0, 0], "Total": [0, 0]})

    # Use the latest play in each quarter to get cumulative score, then diff
    out = {"Team": [away, home]}
    last_h, last_a = 0, 0
    for q in [1, 2, 3, 4, 5]:
        in_q = revealed[revealed["qtr"] == q]
        if in_q.empty:
            h_pts, a_pts = 0, 0
        else:
            last = in_q.iloc[-1]
            cum_h = int(last["total_home_score"] or 0)
            cum_a = int(last["total_away_score"] or 0)
            h_pts = cum_h - last_h
            a_pts = cum_a - last_a
            last_h, last_a = cum_h, cum_a
        label = "OT" if q == 5 else f"Q{q}"
        out[label] = [a_pts, h_pts]
    out["Total"] = [last_a, last_h]
    return pd.DataFrame(out)


def _rate(num: float, den: float) -> float:
    return num / den if den > 0 else float("nan")


_LOWER_IS_BETTER = {"Turnovers"}


def _top_seconds(td: pd.DataFrame) -> float:
    """Time of possession in seconds, estimated from drive×quarter clock diffs."""
    top = 0.0
    for _, grp in td.groupby(["drive", "qtr"], dropna=True):
        gsr = grp["game_seconds_remaining"].dropna()
        if len(gsr) >= 2:
            top += max(0.0, float(gsr.max() - gsr.min()))
    return top


def _fmt_top(v) -> str:
    if pd.isna(v):
        return "—"
    v = int(v)
    return f"{v // 60}:{v % 60:02d}"


def _smap(styled, func, **kwargs):
    try:
        return styled.map(func, **kwargs)
    except AttributeError:
        return styled.applymap(func, **kwargs)


def _percentile_of(value: float, sorted_arr: np.ndarray) -> float:
    """0–100 percentile rank of value in a pre-sorted array."""
    if pd.isna(value) or len(sorted_arr) == 0:
        return float("nan")
    return float(np.searchsorted(sorted_arr, value, side="right") / len(sorted_arr) * 100)


def _percentile_color(pct: float) -> str:
    """CSS string for a red→white→green diverging color at the given percentile."""
    if pd.isna(pct):
        return ""
    # Red(220,53,69) → White(255,255,255) → Green(40,167,69)
    if pct <= 50:
        t = pct / 50.0
        r = int(220 + t * (255 - 220))
        g = int(53  + t * (255 - 53))
        b = int(69  + t * (255 - 69))
    else:
        t = (pct - 50) / 50.0
        r = int(255 + t * (40  - 255))
        g = int(255 + t * (167 - 255))
        b = int(255 + t * (69  - 255))
    lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    text = "#ffffff" if lum < 0.45 else "#212529"
    return f"background-color: #{r:02x}{g:02x}{b:02x}; color: {text}"


def team_stats(revealed: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    """Advanced offensive team stats, returned as a transposed comparison table.

    Rows are stat names; columns are [away, home] team abbreviations.
    """
    stats: dict[str, list] = {}

    for team in [away, home]:
        td = revealed[revealed["posteam"] == team]

        pass_mask = td["pass_attempt"].fillna(0) == 1
        rush_mask = td["rush_attempt"].fillna(0) == 1
        scrimmage_mask = pass_mask | rush_mask

        pass_plays = int(pass_mask.sum())
        rush_plays = int(rush_mask.sum())
        total_plays = int(scrimmage_mask.sum())

        pass_yds = int(td["passing_yards"].fillna(0).sum())
        rush_yds = int(td["rushing_yards"].fillna(0).sum())

        cmp = td.loc[pass_mask, "complete_pass"].fillna(0).sum()
        adot = td.loc[pass_mask, "air_yards"].mean()  # nan if no pass plays

        pass_epa = td.loc[pass_mask, "epa"].fillna(0).sum()
        rush_epa = td.loc[rush_mask, "epa"].fillna(0).sum()
        total_epa = td.loc[scrimmage_mask, "epa"].fillna(0).sum()

        pass_sr = (td.loc[pass_mask, "epa"].fillna(0) > 0).sum()
        rush_sr = (td.loc[rush_mask, "epa"].fillna(0) > 0).sum()

        first_downs = td.loc[scrimmage_mask, "first_down"].fillna(0).sum()

        third_conv = td["third_down_converted"].fillna(0).sum()
        third_fail = td["third_down_failed"].fillna(0).sum()

        rz_mask = scrimmage_mask & (td["yardline_100"].fillna(100) <= 20)
        rz_plays = int(rz_mask.sum())
        rz_td = td.loc[rz_mask, "touchdown"].fillna(0).sum()

        tos = int(td["interception"].fillna(0).sum() + td["fumble_lost"].fillna(0).sum())
        comp_rush = int(cmp) + rush_plays
        top = _top_seconds(td)

        col = [
            total_plays,
            pass_plays,
            rush_plays,
            comp_rush,
            pass_yds,
            rush_yds,
            pass_yds + rush_yds,
            _rate(cmp, pass_plays),
            adot if not pd.isna(adot) else float("nan"),
            _rate(pass_epa, pass_plays),
            _rate(rush_epa, rush_plays),
            _rate(total_epa, total_plays),
            _rate(pass_sr, pass_plays),
            _rate(rush_sr, rush_plays),
            _rate(first_downs, total_plays),
            _rate(third_conv, third_conv + third_fail),
            _rate(rz_td, rz_plays),
            tos,
            top,
        ]
        stats[team] = col

    index = [
        "Plays", "Pass Plays", "Rush Plays", "Rush+Comp",
        "Pass Yds", "Rush Yds", "Total Yds",
        "CMP%", "aDoT",
        "Pass EPA/play", "Rush EPA/play", "EPA/play",
        "Pass SR", "Rush SR",
        "1st Down %", "3rd Down %", "RZ TD%",
        "Turnovers", "TOP",
    ]
    return pd.DataFrame(stats, index=pd.Index(index, name="Stat"))


@st.cache_data(ttl=86400)
def load_stat_baselines(season: int) -> dict[str, np.ndarray]:
    """Per-stat distributions from the 3 seasons prior to `season`, sorted ascending."""
    prior = [s for s in [season - 3, season - 2, season - 1] if s >= 1999]
    if not prior:
        return {}
    cols = [
        "game_id", "posteam",
        "pass_attempt", "rush_attempt", "epa",
        "passing_yards", "rushing_yards",
        "complete_pass", "air_yards",
        "interception", "fumble_lost", "first_down",
        "third_down_converted", "third_down_failed",
        "yardline_100", "touchdown",
        "drive", "qtr", "game_seconds_remaining",
    ]
    try:
        raw = nfl.import_pbp_data(prior, columns=cols, downcast=True)
    except Exception:  # network error, missing season data, etc.
        return {}

    raw = raw[raw["posteam"].notna()]

    def _game_stats(td):
        pass_mask = td["pass_attempt"].fillna(0) == 1
        rush_mask = td["rush_attempt"].fillna(0) == 1
        sc_mask   = pass_mask | rush_mask
        pp = int(pass_mask.sum())
        rp = int(rush_mask.sum())
        tp = int(sc_mask.sum())
        pass_yds = int(td["passing_yards"].fillna(0).sum())
        rush_yds = int(td["rushing_yards"].fillna(0).sum())
        cmp      = td.loc[pass_mask, "complete_pass"].fillna(0).sum()
        adot     = td.loc[pass_mask, "air_yards"].mean()
        pass_epa = td.loc[pass_mask, "epa"].fillna(0).sum()
        rush_epa = td.loc[rush_mask, "epa"].fillna(0).sum()
        tot_epa  = td.loc[sc_mask,   "epa"].fillna(0).sum()
        pass_sr  = (td.loc[pass_mask, "epa"].fillna(0) > 0).sum()
        rush_sr  = (td.loc[rush_mask, "epa"].fillna(0) > 0).sum()
        fd       = td.loc[sc_mask, "first_down"].fillna(0).sum()
        tc       = td["third_down_converted"].fillna(0).sum()
        tf       = td["third_down_failed"].fillna(0).sum()
        rz_mask  = sc_mask & (td["yardline_100"].fillna(100) <= 20)
        rz_plays = int(rz_mask.sum())
        rz_td    = int(td.loc[rz_mask, "touchdown"].fillna(0).sum())
        tos      = int(td["interception"].fillna(0).sum() + td["fumble_lost"].fillna(0).sum())
        comp_rush = int(cmp) + rp
        top       = _top_seconds(td)
        return pd.Series({
            "Plays":        tp,
            "Pass Plays":   pp,
            "Rush Plays":   rp,
            "Rush+Comp":    comp_rush,
            "Pass Yds":     pass_yds,
            "Rush Yds":     rush_yds,
            "Total Yds":    pass_yds + rush_yds,
            "CMP%":         _rate(cmp, pp),
            "aDoT":         adot,
            "Pass EPA/play":_rate(pass_epa, pp),
            "Rush EPA/play":_rate(rush_epa, rp),
            "EPA/play":     _rate(tot_epa,  tp),
            "Pass SR":      _rate(pass_sr,  pp),
            "Rush SR":      _rate(rush_sr,  rp),
            "1st Down %":   _rate(fd, tp),
            "3rd Down %":   _rate(tc, tc + tf),
            "RZ TD%":       _rate(rz_td, rz_plays),
            "Turnovers":    tos,
            "TOP":          top,
        })

    try:
        per_game = raw.groupby(["game_id", "posteam"]).apply(
            _game_stats, include_groups=False
        )
    except TypeError:
        per_game = raw.groupby(["game_id", "posteam"]).apply(_game_stats)
    return {
        stat: np.sort(per_game[stat].dropna().values)
        for stat in per_game.columns
    }


_SR_SITUATIONS = [
    ("Early · Short",  (1, 2), (1,  3)),
    ("Late · Short",   (3, 4), (1,  3)),
    ("Early · Medium", (1, 2), (4,  6)),
    ("Late · Medium",  (3, 4), (4,  6)),
    ("Early · Long",   (1, 2), (7, 99)),
    ("Late · Long",    (3, 4), (7, 99)),
]
_SR_METRICS = ["Pass SR%", "Pass EPA/play", "Rush SR%", "Rush EPA/play"]


def situational_success_rate(revealed: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    """Success rate and EPA/play split by down group × distance × play type."""
    def _sr(plays) -> float:
        if plays.empty:
            return float("nan")
        return (plays["epa"].fillna(0) > 0).mean() * 100

    def _epa_per_play(plays) -> float:
        if plays.empty:
            return float("nan")
        return plays["epa"].fillna(0).mean()

    rows = []
    counts: dict[tuple, dict[str, int]] = {}
    for label, downs, (d_min, d_max) in _SR_SITUATIONS:
        sit_mask = (
            revealed["down"].isin(downs) &
            revealed["ydstogo"].fillna(0).between(d_min, d_max)
        )
        for team in [away, home]:
            tm = revealed[sit_mask & (revealed["posteam"] == team)]
            pass_plays = tm[tm["pass_attempt"].fillna(0) == 1]
            rush_plays = tm[tm["rush_attempt"].fillna(0) == 1]
            counts[(label, team)] = {"Pass": len(pass_plays), "Rush": len(rush_plays)}
            rows.append({
                "Situation": label,
                "Team": team,
                "Pass SR%": _sr(pass_plays),
                "Pass EPA/play": _epa_per_play(pass_plays),
                "Rush SR%": _sr(rush_plays),
                "Rush EPA/play": _epa_per_play(rush_plays),
            })

    df = pd.DataFrame(rows)
    metrics = ["Pass SR%", "Pass EPA/play", "Rush SR%", "Rush EPA/play"]
    pivot = df.pivot(index="Situation", columns="Team", values=metrics)
    pivot.columns = [f"{team} {stat}" for stat, team in pivot.columns]
    col_order = []
    for team in [away, home]:
        for m in metrics:
            col_order.append(f"{team} {m}")
    pivot = pivot[[c for c in col_order if c in pivot.columns]]
    pivot.index.name = "Situation"
    return pivot, counts


@st.cache_data(ttl=86400)
def load_situational_baselines(season: int) -> dict[str, dict[str, np.ndarray]]:
    """Per-situation metric distributions from the 3 seasons prior to `season`."""
    prior = [s for s in [season - 3, season - 2, season - 1] if s >= 1999]
    if not prior:
        return {}
    cols = ["game_id", "posteam", "pass_attempt", "rush_attempt", "epa", "down", "ydstogo"]
    try:
        raw = nfl.import_pbp_data(prior, columns=cols, downcast=True)
    except Exception:
        return {}

    raw = raw[raw["posteam"].notna()]

    def _sr(plays) -> float:
        if plays.empty:
            return float("nan")
        return (plays["epa"].fillna(0) > 0).mean() * 100

    def _epa_per_play(plays) -> float:
        if plays.empty:
            return float("nan")
        return plays["epa"].fillna(0).mean()

    result: dict[str, dict[str, np.ndarray]] = {}
    for label, downs, (d_min, d_max) in _SR_SITUATIONS:
        sit_mask = (
            raw["down"].isin(downs) &
            raw["ydstogo"].fillna(0).between(d_min, d_max)
        )
        sit = raw[sit_mask]

        rows = []
        for (game_id, team), grp in sit[sit["posteam"].notna()].groupby(["game_id", "posteam"]):
            pass_plays = grp[grp["pass_attempt"].fillna(0) == 1]
            rush_plays = grp[grp["rush_attempt"].fillna(0) == 1]
            rows.append({
                "Pass SR%":      _sr(pass_plays),
                "Pass EPA/play": _epa_per_play(pass_plays),
                "Rush SR%":      _sr(rush_plays),
                "Rush EPA/play": _epa_per_play(rush_plays),
            })

        if not rows:
            result[label] = {}
            continue
        per_game = pd.DataFrame(rows)
        result[label] = {
            m: np.sort(per_game[m].dropna().values) for m in _SR_METRICS
        }
    return result


def _style_sr_table(df: pd.DataFrame,
                    baselines: dict[str, dict[str, np.ndarray]] | None = None,
                    counts: dict | None = None) -> object:
    _b = baselines or {}
    _c = counts or {}

    def _color_cell(val, situation: str, metric: str) -> str:
        if pd.isna(val):
            return ""
        sit_bl = _b.get(situation, {})
        arr = sit_bl.get(metric)
        if arr is not None and len(arr) > 0:
            return _percentile_color(_percentile_of(float(val), arr))
        # fallback: fixed thresholds
        if metric == "Pass SR%" or metric == "Rush SR%":
            if val >= 55:
                return "background-color: #d4edda; color: #155724"
            if val <= 40:
                return "background-color: #f8d7da; color: #721c24"
            return ""
        if val > 0:
            return "background-color: #d4edda; color: #155724"
        if val < 0:
            return "background-color: #f8d7da; color: #721c24"
        return ""

    styled = df.style
    for col in df.columns:
        metric = next((m for m in _SR_METRICS if col.endswith(m)), None)
        if metric is None:
            continue
        team_part = col[: -len(metric)].strip()
        play_type = "Pass" if "Pass" in metric else "Rush"
        for sit in df.index:
            n = _c.get((sit, team_part), {}).get(play_type)
            if "SR%" in metric:
                fmt_fn = lambda v, c=n: ("—" if pd.isna(v) else (f"{v:.0f}% ({c})" if c is not None else f"{v:.0f}%"))
            else:
                fmt_fn = lambda v, c=n: ("—" if pd.isna(v) else (f"{v:+.2f} ({c})" if c is not None else f"{v:+.2f}"))
            styled = _smap(
                styled,
                lambda v, s=sit, m=metric: _color_cell(v, s, m),
                subset=pd.IndexSlice[sit, col],
            ).format(fmt_fn, subset=pd.IndexSlice[sit, col])
    return (
        styled
        .set_properties(**{"text-align": "center"})
        .set_table_styles(
            [{"selector": "th", "props": [("text-align", "center"), ("font-weight", "bold")]}]
        )
    )


_EPA_ROWS        = {"Pass EPA/play", "Rush EPA/play", "EPA/play"}
_RATE_ROWS       = {"CMP%", "Pass SR", "Rush SR", "1st Down %", "3rd Down %", "RZ TD%"}
_NO_COLOR_ROWS   = {"Pass Yds", "Rush Yds", "Total Yds", "Pass Plays", "Rush Plays", "Plays", "Rush+Comp", "Turnovers", "TOP"}
_PCT_FORMAT_ROWS = _RATE_ROWS
_EPA_FORMAT_ROWS = _EPA_ROWS


def style_stat_table(df: pd.DataFrame, away: str, home: str,
                     baselines: dict | None = None):
    """Return a pandas Styler with percentile-based diverging colors and EPA coloring."""
    if baselines is None:
        baselines = {}

    def _color_epa(val):
        if pd.isna(val):
            return ""
        if val > 0:
            return "background-color: #d4edda; color: #155724"
        if val < 0:
            return "background-color: #f8d7da; color: #721c24"
        return ""

    styled = df.style

    for row in df.index:
        if row == "TOP":
            styled = styled.format(_fmt_top, subset=pd.IndexSlice[row, :])
            continue
        if row in _PCT_FORMAT_ROWS:
            fmt_str = "{:.1%}"
        elif row in _EPA_FORMAT_ROWS:
            fmt_str = "{:+.2f}"
        elif row == "aDoT":
            fmt_str = "{:.1f}"
        else:
            fmt_str = "{:.0f}"
        styled = styled.format(fmt_str, subset=pd.IndexSlice[row, :], na_rep="—")

    for row in _EPA_ROWS:
        if row in df.index:
            styled = _smap(styled, _color_epa, subset=pd.IndexSlice[row, :])

    for row in df.index:
        if row in _EPA_ROWS or row in _NO_COLOR_ROWS:
            continue
        arr   = baselines.get(row, np.array([]))
        lower = row in _LOWER_IS_BETTER

        def _cell(val, _arr=arr, _lower=lower):
            if pd.isna(val):
                return ""
            pct = _percentile_of(float(val), _arr)
            if pd.isna(pct):
                return ""
            if _lower:
                pct = 100.0 - pct
            return _percentile_color(pct)

        styled = _smap(styled, _cell, subset=pd.IndexSlice[row, :])

    styled = styled.set_properties(**{"text-align": "center"})
    styled = styled.set_table_styles(
        [{"selector": "th", "props": [("text-align", "center"), ("font-weight", "bold")]}]
    )
    return styled


def top_players(revealed: pd.DataFrame, team: str, kind: str, n: int = 3) -> pd.DataFrame:
    """Leaders for a team so far."""
    td = revealed[revealed["posteam"] == team]
    if kind == "passing":
        pass_td = td[td["pass_attempt"] == 1].copy()
        if pass_td.empty:
            return pd.DataFrame()
        pass_td["_success"] = (pass_td["epa"].fillna(0) > 0).astype(int)
        sr = pass_td.groupby("passer_player_name")["_success"].mean().rename("SR%")
        g = pass_td.groupby("passer_player_name", as_index=False).agg(
            Att=("pass_attempt", "sum"), Yds=("passing_yards", "sum"),
            TD=("pass_touchdown", "sum"), INT=("interception", "sum"),
            aDOT=("air_yards", "mean"), Sacks=("sack", "sum"), Hits=("qb_hit", "sum"),
            _epa=("epa", "sum"), _plays=("epa", "count"))
        g = g.rename(columns={"passer_player_name": "Player"})
        g = g.join(sr, on="Player")
        g["aDOT"] = g["aDOT"].round(1)
        g["SR%"] = (g["SR%"] * 100).round(1)
    elif kind == "rushing":
        rush_td = td[td["rush_attempt"] == 1].copy()
        if rush_td.empty:
            return pd.DataFrame()
        rush_td["_success"] = (rush_td["epa"].fillna(0) > 0).astype(int)
        rush_td["_stuffed"] = (rush_td["yards_gained"].fillna(0) <= 0).astype(int)
        sr = rush_td.groupby("rusher_player_name")["_success"].mean().rename("SR%")
        g = rush_td.groupby("rusher_player_name", as_index=False).agg(
            Att=("rush_attempt", "sum"), Yds=("rushing_yards", "sum"),
            TD=("rush_touchdown", "sum"), Stuffed=("_stuffed", "sum"),
            _epa=("epa", "sum"), _plays=("epa", "count"))
        g = g.rename(columns={"rusher_player_name": "Player"})
        g = g.join(sr, on="Player")
        g["SR%"] = (g["SR%"] * 100).round(1)
    else:  # receiving
        recv_td = td[td["receiver_player_name"].notna() & (td["pass_attempt"].fillna(0) == 1)]
        if recv_td.empty:
            return pd.DataFrame()
        g = recv_td.groupby("receiver_player_name", as_index=False).agg(
            Tgt=("pass_attempt", "sum"),
            Yds=("receiving_yards", "sum"), TD=("pass_touchdown", "sum"),
            _epa=("epa", "sum"), _plays=("epa", "count"))
        g["Yds"] = g["Yds"].fillna(0)
        g = g.rename(columns={"receiver_player_name": "Player"})
    g = g.dropna(subset=["Player"])
    int_cols = [c for c in g.select_dtypes("number").columns if c not in ("_epa", "_plays", "aDOT", "EPA/play", "SR%")]
    g[int_cols] = g[int_cols].astype(int)
    g["EPA/play"] = (g["_epa"] / g["_plays"]).round(2)
    g = g.drop(columns=["_epa", "_plays"])
    return g.sort_values("Yds", ascending=False).head(n)


def top_defenders(revealed: pd.DataFrame, team: str, n: int = 5) -> pd.DataFrame:
    """Defensive leaders for a team: tackles, sacks, QB hits, TFLs, INTs, PDs, FFs."""
    _ST = {"kickoff", "punt", "field_goal", "extra_point"}
    td = revealed[
        (revealed["defteam"] == team) &
        (~revealed["play_type"].isin(_ST))
    ]
    if td.empty:
        return pd.DataFrame()

    def _count(cols: list[str], weight: float = 1.0) -> pd.Series:
        frames = []
        for col in cols:
            if col in td.columns:
                s = td[col].dropna()
                frames.append(s)
        if not frames:
            return pd.Series(dtype=float)
        stacked = pd.concat(frames)
        return (stacked.value_counts() * weight).rename("v")

    solo   = _count(["solo_tackle_1_player_name", "solo_tackle_2_player_name"], 1.0)
    assist = _count([
        "assist_tackle_1_player_name", "assist_tackle_2_player_name",
        "assist_tackle_3_player_name", "assist_tackle_4_player_name",
    ], 0.5)
    tackles = solo.add(assist, fill_value=0).rename("Tackles")

    sacks = _count(["sack_player_name"], 1.0).add(
        _count(["half_sack_1_player_name", "half_sack_2_player_name"], 0.5), fill_value=0
    ).rename("Sacks")

    qb_hits = _count(["qb_hit_1_player_name", "qb_hit_2_player_name"]).rename("QB Hits")
    tfls    = _count(["tackle_for_loss_1_player_name", "tackle_for_loss_2_player_name"]).rename("TFL")
    ints    = _count(["interception_player_name"]).rename("INT")
    pds     = _count(["pass_defense_1_player_name", "pass_defense_2_player_name"]).rename("PD")
    ffs     = _count(["forced_fumble_player_1_player_name", "forced_fumble_player_2_player_name"]).rename("FF")

    g = pd.concat([tackles, sacks, qb_hits, tfls, ints, pds, ffs], axis=1).fillna(0)
    g.index.name = "Player"
    g = g.reset_index()
    g = g[g["Player"].notna()]

    for col in ["Tackles", "QB Hits", "TFL", "INT", "PD", "FF"]:
        if col in g.columns:
            g[col] = g[col].round(1)
    if "Sacks" in g.columns:
        g["Sacks"] = g["Sacks"].round(1)

    # weight sacks/INTs heavily so impact players surface even with low tackle counts
    g["_sort"] = (
        g.get("Tackles", 0) + g.get("Sacks", 0) * 3
        + g.get("INT", 0) * 2 + g.get("TFL", 0) + g.get("QB Hits", 0) * 0.5
        + g.get("PD", 0) * 0.5 + g.get("FF", 0) * 1.5
    )
    return g.sort_values("_sort", ascending=False).drop(columns=["_sort"]).head(n)


def _drive_outcome(drive_plays: pd.DataFrame) -> str:
    """Determine how a drive ended from its plays."""
    if drive_plays["touchdown"].fillna(0).astype(bool).any():
        return "TD"
    fg = drive_plays[drive_plays["play_type"] == "field_goal"]
    if not fg.empty:
        r = fg["field_goal_result"].dropna()
        if not r.empty:
            v = r.iloc[-1]
            return "FG" if v == "made" else ("FG Blocked" if v == "blocked" else "FG Miss")
    if drive_plays["interception"].fillna(0).astype(bool).any():
        return "Interception"
    if drive_plays["fumble_lost"].fillna(0).astype(bool).any():
        return "Fumble"
    if (drive_plays["play_type"] == "punt").any():
        return "Punt"
    if drive_plays["fourth_down_failed"].fillna(0).astype(bool).any():
        return "Downs"
    if drive_plays["safety"].fillna(0).astype(bool).any():
        return "Safety"
    return "EOH/EOG"


def drive_chart(revealed: pd.DataFrame) -> pd.DataFrame:
    """One row per drive: team, plays, pass/run split, yards, first downs, outcome."""
    if revealed.empty or "drive" not in revealed.columns:
        return pd.DataFrame()
    rows = []
    for drive_num, grp in revealed.groupby("drive", sort=True):
        off = grp[grp["posteam"].notna()]
        if off.empty:
            continue
        posteam = off["posteam"].dropna().iloc[0]
        qtr_start = int(off["qtr"].dropna().iloc[0]) if off["qtr"].notna().any() else ""
        yl_series = off["yardline_100"].dropna()
        if not yl_series.empty:
            y = int(yl_series.iloc[0])
            start = f"OWN {100 - y}" if y > 50 else ("50" if y == 50 else f"OPP {y}")
        else:
            start = ""
        p_mask = off["pass_attempt"].fillna(0) == 1
        r_mask = off["rush_attempt"].fillna(0) == 1
        sc = off[p_mask | r_mask]
        rows.append({
            "Drive": int(drive_num),
            "Team": posteam,
            "Qtr": qtr_start,
            "Start": start,
            "Plays": len(sc),
            "Pass": int(p_mask.sum()),
            "Run": int(r_mask.sum()),
            "Yards": int(sc["yards_gained"].fillna(0).sum()),
            "1st Downs": int(sc["first_down"].fillna(0).sum()),
            "Outcome": _drive_outcome(grp),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Drive", ascending=False).reset_index(drop=True)


def _style_drive_chart(df: pd.DataFrame):
    def _outcome_color(val):
        if val == "TD":
            return "background-color: #d4edda; color: #155724; font-weight: bold"
        if val in ("Interception", "Fumble", "Downs", "Safety"):
            return "background-color: #f8d7da; color: #721c24"
        if val == "FG":
            return "background-color: #cce5ff; color: #004085"
        if val in ("FG Miss", "FG Blocked"):
            return "background-color: #fff3cd; color: #856404"
        if val == "Punt":
            return "background-color: #e2e3e5; color: #383d41"
        return ""
    styled = df.style
    if "Outcome" in df.columns:
        styled = _smap(styled, _outcome_color, subset=["Outcome"])
    return (
        styled
        .set_properties(**{"text-align": "center"})
        .set_table_styles(
            [{"selector": "th", "props": [("text-align", "center"), ("font-weight", "bold")]}]
        )
    )


def top_plays_wpa(revealed: pd.DataFrame, home: str, away: str, n: int = 25) -> pd.DataFrame:
    """Top n plays by absolute win probability added, computed from home_wp shifts."""
    if revealed.empty:
        return pd.DataFrame()
    df = revealed.copy().reset_index(drop=True)
    wp = df["home_wp"].copy()
    wpa = wp.shift(-1) - wp
    df["_wpa"] = wpa
    df["_abs_wpa"] = wpa.abs()
    plays = df[df["_abs_wpa"].notna() & (df["_abs_wpa"] > 0)].nlargest(n, "_abs_wpa").copy()
    if plays.empty:
        return pd.DataFrame()
    plays["Q"] = plays["qtr"].apply(lambda x: str(int(x)) if pd.notna(x) else "")
    plays["Score"] = plays.apply(
        lambda r: f"{away} {int(r['total_away_score'] or 0)}–{int(r['total_home_score'] or 0)} {home}", axis=1
    )
    plays["WPA"] = plays["_wpa"].round(3)
    plays["For"] = plays["_wpa"].apply(lambda x: home if x > 0 else away)
    return plays[["Q", "time", "posteam", "Score", "For", "desc", "WPA", "_abs_wpa"]].rename(
        columns={"time": "Clock", "posteam": "Off", "desc": "Description"}
    ).reset_index(drop=True)


# ---------- UI ----------
st.title("🏈 NFL Tape-Delay Replay")
st.caption("Spoiler-free boxscore that unlocks as your broadcast progresses.")

with st.sidebar:
    st.header("Setup")
    season = st.number_input("Season", min_value=1999, max_value=2026, value=2025, step=1)

    with st.spinner("Loading play-by-play..."):
        try:
            pbp = load_pbp(int(season))
        except Exception as e:
            st.error(f"Could not load pbp: {e}")
            st.stop()

    games = list_games(pbp)
    if games.empty:
        st.warning(f"No games found for the {int(season)} season yet.")
        st.stop()
    game_label = st.selectbox("Game", games["label"].tolist())
    game_id = games.loc[games["label"] == game_label, "game_id"].iloc[0]

    st.divider()
    st.subheader("Your viewing")
    mode = st.radio(
        "How do you want to set your position?",
        ["Jump to a specific game clock","I started the broadcast at...",
         "I'm X minutes into the broadcast"
         ],
    )

    if mode == "I started the broadcast at...":
        start_date = st.date_input("Start date", value=datetime.now().date())
        start_time = st.time_input("Start time (your local clock)", value=time(20, 0))
        start_dt = datetime.combine(start_date, start_time)
        viewing_minutes = max((datetime.now() - start_dt).total_seconds() / 60.0, 0.0)
        elapsed_s = elapsed_game_seconds(viewing_minutes)
        auto = st.checkbox("Auto-refresh", value=False,
                           help="Recalculates your position from the clock automatically.")
        if auto:
            refresh_interval_s = st.selectbox(
                "Refresh interval",
                options=[15, 30, 45, 60, 90, 120],
                index=2,
                format_func=lambda x: f"{x}s",
                key="refresh_interval_wall",
            )
            st_autorefresh(interval=refresh_interval_s * 1000, key="autorefresh")

    elif mode == "I'm X minutes into the broadcast":
        viewing_minutes = st.number_input("Minutes into broadcast",
                                          min_value=0.0, max_value=240.0,
                                          value=30.0, step=1.0)
        baseline_elapsed = elapsed_game_seconds(viewing_minutes)
        auto = st.checkbox("Auto-advance play by play", value=False)
        if auto:
            refresh_interval_s = st.selectbox(
                "Refresh interval",
                options=[15, 30, 45, 60, 90, 120],
                index=2,
                format_func=lambda x: f"{x}s",
                key="refresh_interval_min",
            )
            # If baseline changed, reset the advancing position to the new input.
            if st.session_state.get("_fix_baseline") != round(baseline_elapsed):
                st.session_state["_fix_elapsed"] = baseline_elapsed
                st.session_state["_fix_baseline"] = round(baseline_elapsed)
            elapsed_s = st.session_state.get("_fix_elapsed", baseline_elapsed)
            viewing_minutes = (elapsed_s / 3600.0) * 190.0
            st_autorefresh(interval=refresh_interval_s * 1000, key="autorefresh")
        else:
            elapsed_s = baseline_elapsed

    else:  # Jump to a specific game clock
        # Game clock counts DOWN within each quarter from 15:00 to 0:00
        qtr_pick = st.selectbox("Quarter", ["Q1", "Q2", "Q3", "Q4", "OT"], index=0)
        clock_str = st.text_input("Game clock remaining (MM:SS)", value="15:00",
                                  help="Time left on the in-quarter clock, e.g. 7:32")
        try:
            mm, ss = clock_str.strip().split(":")
            remaining_in_qtr = int(mm) * 60 + int(ss)
            assert 0 <= remaining_in_qtr <= 15 * 60
        except Exception:
            st.warning("Use MM:SS format, e.g. 7:32. Defaulting to 15:00.")
            remaining_in_qtr = 15 * 60

        qtr_idx = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "OT": 5}[qtr_pick]
        # Game seconds elapsed = full quarters completed * 900 + (900 - remaining)
        # OT in pbp is qtr=5; treat it as starting after Q4 ends.
        completed_qtrs = qtr_idx - 1
        baseline_elapsed = float(completed_qtrs * 900 + (900 - remaining_in_qtr))
        auto = st.checkbox("Auto-advance play by play", value=False)
        if auto:
            refresh_interval_s = st.selectbox(
                "Refresh interval",
                options=[15, 30, 45, 60, 90, 120],
                index=2,
                format_func=lambda x: f"{x}s",
                key="refresh_interval_clock",
            )
            _baseline_key = (qtr_pick, clock_str)
            if st.session_state.get("_fix_baseline") != _baseline_key:
                st.session_state["_fix_elapsed"] = baseline_elapsed
                st.session_state["_fix_baseline"] = _baseline_key
            elapsed_s = st.session_state.get("_fix_elapsed", baseline_elapsed)
            st_autorefresh(interval=refresh_interval_s * 1000, key="autorefresh")
        else:
            elapsed_s = baseline_elapsed
        # Derive an approximate viewing-minutes equivalent just for the caption
        viewing_minutes = (elapsed_s / 3600.0) * 190.0

    st.caption(f"⏱ Game time elapsed: **{elapsed_s/60:.1f} min** "
               f"(≈ {viewing_minutes:.1f} broadcast min)")

    st.divider()
    st.subheader("🙈 Spoiler shield")
    safety_margin = st.slider(
        "Stay this many seconds *behind* my entered position",
        min_value=0, max_value=120, value=15, step=5,
        help="Buffer against accidentally revealing the next play. "
             "15s is enough to absorb small clock drift.",
    )
    hide_wp = st.checkbox("Hide win probability chart", value=False,
                          help="The WP curve telegraphs upcoming swings.")
    hide_descriptions = st.checkbox("Hide play descriptions", value=False,
                                    help="Show yards/EPA only — descriptions can foreshadow what's about to happen on screen.")
    hide_leaders = st.checkbox("Hide player leaders", value=False,
                               help="A QB suddenly at 4 TDs hints something just happened.")
    blur_until_ready = st.checkbox("Blur everything until I click 'Reveal'", value=False)

# ---------- Compute revealed slice ----------
pbp_game = pbp[pbp["game_id"] == game_id].sort_values("play_id").reset_index(drop=True)
home = pbp_game["home_team"].iloc[0]
away = pbp_game["away_team"].iloc[0]

elapsed_s = max(float(elapsed_s) - float(safety_margin), 0.0)
revealed = filter_revealed(pbp_game, elapsed_s)

# Header summary (no future info)
qtr_now = int(revealed["qtr"].iloc[-1]) if not revealed.empty else 1
game_clock = "—"
if not revealed.empty and pd.notna(revealed["time"].iloc[-1]):
    game_clock = str(revealed["time"].iloc[-1])

if not revealed.empty:
    _last = revealed.iloc[-1]
    home_score = int(_last["total_home_score"] or 0)
    away_score = int(_last["total_away_score"] or 0)
else:
    home_score = away_score = 0

c1, c2, c3 = st.columns(3)
c1.metric("Quarter", f"Q{qtr_now}" if qtr_now <= 4 else "OT")
c2.metric("Game clock (last play)", game_clock)
c3.metric("Score", f"{away} {away_score}  —  {home_score} {home}")

# ---------- Reveal gate ----------
if blur_until_ready:
    if "revealed_ok" not in st.session_state:
        st.session_state.revealed_ok = False
    if not st.session_state.revealed_ok:
        st.warning("Content is hidden. Click below when you've caught up on the broadcast and are ready to see the current state.")
        if st.button("👁 Reveal current state"):
            st.session_state.revealed_ok = True
            st.rerun()
        st.stop()
    if st.button("🙈 Re-hide (next scrub)"):
        st.session_state.revealed_ok = False
        st.rerun()

st.divider()

# ---------- Shared plays helpers ----------
def _field_pos_label(r) -> str:
    yl = r["yardline_100"]
    if pd.isna(yl):
        return ""
    yl = int(yl)
    if yl > 50:
        return f"OWN {100 - yl}"
    elif yl == 50:
        return "50"
    else:
        return f"OPP {yl}"

def _play_type_label(r) -> str:
    down = r["down"]
    is_pass = r["pass_attempt"] == 1
    is_run = r["rush_attempt"] == 1
    icon = "🏈 " if is_pass else ("🏃 " if is_run else "")
    play_kind = f"{icon}Pass" if is_pass else (f"{icon}Run" if is_run else "")
    if pd.notna(down) and down in (3, 4):
        prefix = "3rd" if down == 3 else "4th"
        return f"{prefix} & {play_kind}" if play_kind else prefix
    return play_kind

def _down_distance(r) -> str:
    if pd.notna(r["down"]) and pd.notna(r["ydstogo"]):
        return f"{int(r['down'])} & {int(r['ydstogo'])}"
    return ""

def _success_emoji(r) -> str:
    if pd.notna(r["epa"]) and r["epa"] > 0:
        return "✅"
    return ""

def _build_plays_df(raw: pd.DataFrame, hide_desc: bool, reverse: bool = True) -> pd.DataFrame:
    raw = raw.copy()
    raw["Type"] = raw.apply(_play_type_label, axis=1)
    raw["D&D"] = raw.apply(_down_distance, axis=1)
    raw["Success?"] = raw.apply(_success_emoji, axis=1)
    raw["Q"] = raw["qtr"].apply(lambda x: str(int(x)) if pd.notna(x) else "")
    raw["Field"] = raw.apply(_field_pos_label, axis=1)
    is_special = raw["play_type"].isin(["field_goal", "extra_point"])
    raw["_rz"] = (raw["yardline_100"].fillna(100) <= 20) & ~is_special
    if hide_desc:
        cols_sel = ["Q", "time", "posteam", "Field", "_rz", "Type", "D&D", "Success?", "yards_gained", "epa"]
        df = raw[cols_sel].copy()
        df.columns = ["Q", "Clock", "Off", "Field", "_rz", "Type", "D&D", "Success?", "Yds", "EPA"]
    else:
        cols_sel = ["Q", "time", "posteam", "Field", "_rz", "Type", "D&D", "Success?", "desc", "yards_gained", "epa"]
        df = raw[cols_sel].copy()
        df.columns = ["Q", "Clock", "Off", "Field", "_rz", "Type", "D&D", "Success?", "Description", "Yds", "EPA"]
    df["Yds"] = pd.to_numeric(df["Yds"], errors="coerce").fillna(0).astype(int)
    df["EPA"] = pd.to_numeric(df["EPA"], errors="coerce").round(2)
    return df.iloc[::-1] if reverse else df

def _style_plays(row):
    t = row["Type"]
    if t.startswith("4th"):
        row_bg = "#f5c6cb"
    elif t.startswith("3rd"):
        row_bg = "#ffeeba"
    else:
        row_bg = "#ffffff"
    styles = []
    for col in row.index:
        if col == "_rz":
            styles.append("")
        elif col == "Field" and row.get("_rz", False):
            styles.append("background-color: #dc3545; color: #ffffff")
        else:
            styles.append(f"background-color: {row_bg}; color: #000000")
    return styles

_EPA_COL_CFG = {"EPA": st.column_config.NumberColumn(format="%.2f")}

# ---------- Boxscore ----------
st.subheader("Boxscore")
st.dataframe(boxscore(revealed, home, away), hide_index=True, use_container_width=True)

# ---------- Recent plays (with pagination) ----------
st.subheader("Recent plays")
st.markdown(
    '<span style="background:#f5c6cb;padding:2px 8px;border-radius:3px;margin-right:6px">4th down</span>'
    '<span style="background:#ffeeba;padding:2px 8px;border-radius:3px;margin-right:6px">3rd down</span>'
    '<span style="background:#dc3545;color:#fff;padding:2px 8px;border-radius:3px;margin-right:6px">Red zone</span>'
    '<span style="margin-right:6px">🏈 Pass &nbsp; 🏃 Run</span>'
    '<span style="margin-right:6px">✅ Positive EPA</span>',
    unsafe_allow_html=True,
)
if not revealed.empty:
    _PAGE_SIZE = 15
    _all_rev = revealed.iloc[::-1].copy()
    _total_plays = len(_all_rev)
    _total_pages = max(1, (_total_plays + _PAGE_SIZE - 1) // _PAGE_SIZE)
    if "recent_plays_page" not in st.session_state:
        st.session_state.recent_plays_page = 1
    _page = min(int(st.session_state.recent_plays_page), _total_pages)

    _slice = _all_rev.iloc[(_page - 1) * _PAGE_SIZE : _page * _PAGE_SIZE]
    recent_df = _build_plays_df(_slice, hide_descriptions, reverse=False)
    st.dataframe(
        recent_df.style.apply(_style_plays, axis=1),
        hide_index=True, use_container_width=True,
        column_config={**_EPA_COL_CFG, "_rz": None},
    )
    col_prev, col_info, col_next = st.columns([1, 4, 1])
    with col_prev:
        if st.button("◀ Prev", disabled=(_page <= 1), key="prev_plays"):
            st.session_state.recent_plays_page = _page - 1
            st.rerun()
    with col_info:
        st.caption(f"Page {_page} of {_total_pages}  ({_total_plays} plays total)")
    with col_next:
        if st.button("Next ▶", disabled=(_page >= _total_pages), key="next_plays"):
            st.session_state.recent_plays_page = _page + 1
            st.rerun()
else:
    st.caption("No plays revealed yet.")

# ---------- Current drive ----------
st.subheader("Current drive")
if not revealed.empty:
    # Anchor the current drive on the most recent scrimmage play so that after a
    # score → PAT → kickoff sequence the section still shows the offensive drive,
    # not a one-play kickoff "drive".
    _scrimmage_all = revealed[
        (revealed["pass_attempt"].fillna(0) == 1) |
        (revealed["rush_attempt"].fillna(0) == 1)
    ]
    if not _scrimmage_all.empty:
        _cur_drive = _scrimmage_all["drive"].dropna().iloc[-1] if "drive" in revealed.columns else None
    else:
        _cur_drive = revealed["drive"].dropna().iloc[-1] if "drive" in revealed.columns else None

    if _cur_drive is not None:
        _drive_raw = revealed[revealed["drive"] == _cur_drive].copy()

        # Drive summary stats (scrimmage plays only)
        _scrimmage = _drive_raw[
            (_drive_raw["pass_attempt"].fillna(0) == 1) |
            (_drive_raw["rush_attempt"].fillna(0) == 1)
        ]
        _drive_plays = len(_scrimmage)
        _drive_sr = (_scrimmage["epa"].fillna(0) > 0).mean() * 100 if _drive_plays > 0 else float("nan")

        _clocks = _scrimmage["game_seconds_remaining"].dropna()
        if len(_clocks) >= 2:
            _drive_top_s = int(_clocks.iloc[0] - _clocks.iloc[-1])
            _top_str = f"{_drive_top_s // 60}:{_drive_top_s % 60:02d}"
        else:
            _top_str = "—"

        _possession_team = _scrimmage["posteam"].dropna().iloc[-1] if not _scrimmage.empty else (
            _drive_raw["posteam"].dropna().iloc[-1] if not _drive_raw.empty else "—"
        )
        _sr_str = f"{_drive_sr:.0f}%" if not pd.isna(_drive_sr) else "—"
        st.caption(
            f"**{_possession_team}** · {_drive_plays} plays · "
            f"Success rate: {_sr_str} · Time of possession: {_top_str}"
        )

        drive_df = _build_plays_df(_drive_raw, hide_descriptions)
        st.dataframe(
            drive_df.style.apply(_style_plays, axis=1),
            hide_index=True, use_container_width=True,
            column_config={**_EPA_COL_CFG, "_rz": None},
        )
    else:
        st.caption("No drive data available.")
else:
    st.caption("No plays revealed yet.")

# ---------- Drive chart ----------
st.subheader("Drive chart")
if not revealed.empty:
    _dc = drive_chart(revealed)
    if not _dc.empty:
        st.dataframe(
            _style_drive_chart(_dc),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.caption("No drive data available.")
else:
    st.caption("No plays revealed yet.")

# ---------- Team stats ----------
st.subheader("Team stats")
stat_df = team_stats(revealed, home, away)
_baselines = load_stat_baselines(int(season))
st.dataframe(style_stat_table(stat_df, away, home, _baselines), use_container_width=True)
st.caption("Colors show percentile vs last 3 seasons · green = top of league · red = bottom")

# ---------- Situational success rates ----------
st.subheader("Situational success rates")
st.caption("Colors show percentile vs last 3 seasons · green = top of league · red = bottom")
sr_df, _sit_counts = situational_success_rate(revealed, home, away)
_sit_baselines = load_situational_baselines(int(season))
st.dataframe(_style_sr_table(sr_df, _sit_baselines, _sit_counts), use_container_width=True)

# ---------- Player leaders ----------
if not hide_leaders:
    st.subheader("Player leaders")
    col_a, col_h = st.columns(2)
    for col, team in [(col_a, away), (col_h, home)]:
        with col:
            st.markdown(f"**{team}**")
            _pass_df = top_players(revealed, team, "passing")
            st.caption("Passing")
            if not _pass_df.empty:
                st.dataframe(_pass_df, hide_index=True, use_container_width=True,
                             column_config={
                                 "EPA/play": st.column_config.NumberColumn(format="%.2f"),
                                 "SR%": st.column_config.NumberColumn(format="%.1f%%"),
                                 "aDOT": st.column_config.NumberColumn(format="%.1f"),
                             })
            else:
                st.caption("No data yet")
            _rush_df = top_players(revealed, team, "rushing",4)
            st.caption("Rushing")
            if not _rush_df.empty:
                st.dataframe(_rush_df, hide_index=True, use_container_width=True,
                             column_config={
                                 "EPA/play": st.column_config.NumberColumn(format="%.2f"),
                                 "SR%": st.column_config.NumberColumn(format="%.1f%%"),
                             })
            else:
                st.caption("No data yet")
            _recv_df = top_players(revealed, team, "receiving",8)
            st.caption("Receiving")
            if not _recv_df.empty:
                st.dataframe(_recv_df, hide_index=True, use_container_width=True)
            else:
                st.caption("No data yet")
            _def_df = top_defenders(revealed, team,10)
            st.caption("Defense")
            if not _def_df.empty:
                st.dataframe(_def_df, hide_index=True, use_container_width=True,
                             column_config={
                                 "Tackles": st.column_config.NumberColumn(format="%.1f"),
                                 "Sacks": st.column_config.NumberColumn(format="%.1f"),
                                 "QB Hits": st.column_config.NumberColumn(format="%.0f"),
                                 "TFL": st.column_config.NumberColumn(format="%.0f"),
                                 "INT": st.column_config.NumberColumn(format="%.0f"),
                                 "PD": st.column_config.NumberColumn(format="%.0f"),
                                 "FF": st.column_config.NumberColumn(format="%.0f"),
                             })
            else:
                st.caption("No data yet")

# ---------- Win probability chart ----------
if not hide_wp:
    st.subheader("Win probability")
    if not revealed.empty:
        wp_df = revealed[["game_seconds_remaining", "home_wp", "away_wp"]].dropna()
        wp_df = wp_df.assign(elapsed=(3600 - wp_df["game_seconds_remaining"]) / 60.0)
        wp_long = wp_df.melt(id_vars="elapsed", value_vars=["home_wp", "away_wp"],
                             var_name="team", value_name="wp")
        wp_long["team"] = wp_long["team"].map({"home_wp": home, "away_wp": away})
        _team_colors = load_team_colors()
        _color_map = {home: _team_colors.get(home, "#1f77b4"),
                      away: _team_colors.get(away, "#ff7f0e")}
        fig = px.line(wp_long, x="elapsed", y="wp", color="team",
                      color_discrete_map=_color_map,
                      labels={"elapsed": "Game minutes elapsed", "wp": "Win probability"})
        fig.update_yaxes(range=[0, 1])
        # Lock x-axis to elapsed-so-far. Otherwise Plotly auto-fits to the data
        # and the right edge silently moves forward as you scrub — and if the
        # game went to OT, an axis ending at 75+ minutes is itself a spoiler.
        x_cap = max(elapsed_s / 60.0, 1.0)
        fig.update_xaxes(range=[0, x_cap])
        st.plotly_chart(fig, use_container_width=True)

# ---------- Top plays by win probability added ----------
st.subheader("Top plays by win probability added")
if not revealed.empty:
    _top_wpa = top_plays_wpa(revealed, home, away)
    if not _top_wpa.empty:
        display_cols = ["Q", "Clock", "Off", "Score", "For", "WPA"]
        if not hide_descriptions:
            display_cols = ["Q", "Clock", "Off", "Score", "For", "Description", "WPA"]
        _top_display = _top_wpa[display_cols].copy()
        _top_display.index = range(1, len(_top_display) + 1)

        def _wpa_color(val):
            if pd.isna(val):
                return ""
            if val > 0.15:
                return "background-color: #d4edda; color: #155724"
            if val < -0.15:
                return "background-color: #f8d7da; color: #721c24"
            if val > 0:
                return "background-color: #e8f5e9; color: #155724"
            return "background-color: #fdecea; color: #721c24"

        styled_wpa = _smap(_top_display.style, _wpa_color, subset=["WPA"])
        styled_wpa = styled_wpa.format({"WPA": "{:+.3f}"})
        styled_wpa = styled_wpa.set_properties(**{"text-align": "center"}).set_table_styles(
            [{"selector": "th", "props": [("text-align", "center"), ("font-weight", "bold")]}]
        )
        st.dataframe(styled_wpa, hide_index=False, use_container_width=True)
        st.caption("WPA = change in home-team win probability · green = home benefits · red = away benefits")
    else:
        st.caption("No win probability data available.")
else:
    st.caption("No plays revealed yet.")

# ---------- Auto-advance to next play ----------
# For fixed-position modes, advance session_state to the next play's timestamp so
# the next st_autorefresh tick reveals exactly one more play.
if auto and mode != "I started the broadcast at...":
    _cur = float(st.session_state.get("_fix_elapsed", 0.0))
    _played_at = 3600 - pbp_game["game_seconds_remaining"].fillna(3600)
    _future = _played_at[_played_at > _cur + 0.5]
    if not _future.empty:
        st.session_state["_fix_elapsed"] = float(_future.min())

