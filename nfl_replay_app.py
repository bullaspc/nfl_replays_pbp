"""
NFL Tape-Delay Replay Boxscore
-------------------------------
A spoiler-free way to follow an NFL or college football game on tape delay.

You tell the app:
  1. Which game you're watching
  2. When you started watching (i.e. your personal "kickoff")
The app reveals plays, score, and stats only up to your current viewing point.

Run with:
    pip install streamlit nfl_data_py pandas plotly
    streamlit run nfl_replay_app.py
"""

import os
import time
from datetime import date, datetime

import streamlit as st
import pandas as pd
import numpy as np
import nfl_data_py as nfl
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh

import cfb_feed
import game_summary
import live_feed
import nflfastr_models

st.set_page_config(page_title="Football Replay Boxscore", layout="wide", page_icon="🏈")

NFL, CFB = "NFL", "College football"

# ---------- Data loading ----------
@st.cache_data(ttl=3600)
def load_team_colors() -> dict[str, str]:
    """Map team abbreviation → primary hex color."""
    try:
        df = nfl.import_team_desc()
    except Exception:  # network error / upstream file moved: fall back to defaults
        return {}
    return dict(zip(df["team_abbr"], df["team_color"]))


@st.cache_data(ttl=3600)
def load_team_logos() -> dict[str, str]:
    """Map team abbreviation → ESPN logo URL."""
    try:
        df = nfl.import_team_desc()
    except Exception:
        return {}
    col = "team_logo_espn" if "team_logo_espn" in df.columns else "team_logo_wikipedia"
    if col not in df.columns:
        return {}
    return {a: u for a, u in zip(df["team_abbr"], df[col]) if isinstance(u, str)}


@st.cache_data(ttl=3600)
def load_team_nicknames() -> dict[str, str]:
    """Map team abbreviation → nickname (e.g. KC → Chiefs)."""
    try:
        df = nfl.import_team_desc()
    except Exception:
        return {}
    if "team_nick" not in df.columns:
        return {}
    return {a: n for a, n in zip(df["team_abbr"], df["team_nick"]) if isinstance(n, str)}


def wp_crossings(revealed: pd.DataFrame, elapsed_min: pd.Series) -> list[float]:
    """Elapsed minutes at which the revealed home win probability crosses 50%.
    `elapsed_min` is each revealed play's place on the play timeline."""
    d = revealed[["home_wp"]].assign(t=elapsed_min).dropna()
    if len(d) < 2:
        return []
    t = d["t"].to_numpy()
    w = d["home_wp"].to_numpy() - 0.5
    out = []
    for i in range(1, len(w)):
        if w[i - 1] * w[i] < 0:
            out.append(float(t[i - 1] + (t[i] - t[i - 1]) * w[i - 1] / (w[i - 1] - w[i])))
        elif w[i] == 0 and w[i - 1] != 0:
            out.append(float(t[i]))
    return out


NFLVERSE_RELEASE = "https://github.com/nflverse/nflverse-data/releases/download"

# Every column the app reads. The live feed is built to the same layout.
PBP_COLS = [
    "game_id", "season", "week", "game_date", "home_team", "away_team",
    "posteam", "defteam", "qtr", "time", "game_seconds_remaining",
    "play_id", "desc", "play_type", "down", "ydstogo", "yards_gained",
    "touchdown", "field_goal_result", "extra_point_result",
    "two_point_conv_result", "safety", "sp",
    "total_home_score", "total_away_score",
    "home_wp", "away_wp", "epa",
    "passer_player_name", "rusher_player_name", "receiver_player_name",
    "passing_yards", "rushing_yards", "receiving_yards", "yards_after_catch",
    "pass_touchdown", "rush_touchdown",
    "interception", "fumble_lost", "sack", "qb_hit",
    "complete_pass", "pass_attempt", "rush_attempt", "qb_kneel", "qb_spike",
    "first_down",
    "air_yards",
    "third_down_converted", "third_down_failed",
    "fourth_down_converted", "fourth_down_failed",
    "goal_to_go", "yardline_100", "drive",
    "penalty", "penalty_team", "penalty_yards",
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


@st.cache_data(ttl=60)
def nflverse_stamp() -> str:
    """When nflverse last rebuilt its pbp files. Checked every minute, so a
    newly published game shows up right away instead of on a fixed timer."""
    try:
        r = requests.get(f"{NFLVERSE_RELEASE}/pbp/timestamp.json", timeout=10)
        r.raise_for_status()
        return str(r.json().get("last_updated", ""))
    except Exception:
        # Unknown: fall back to refreshing every 10 minutes.
        return f"unknown-{int(time.time() // 600)}"


@st.cache_data(max_entries=4)
def load_pbp(season: int, stamp: str) -> pd.DataFrame:
    """Load nflverse play-by-play for a season. `stamp` is only a cache key:
    the file is downloaded again when nflverse republishes it."""
    # nfl_data_py swallows download errors (e.g. a 404 because nflverse hasn't
    # published this season's file yet) and returns an empty, column-less frame.
    # Participation data is not used by this app and is not published for every
    # season, so it must not be requested (older nfl_data_py raised a 404 on it).
    df = nfl.import_pbp_data([season], columns=PBP_COLS, downcast=True,
                             include_participation=False)
    if df.empty or "game_id" not in df.columns:
        raise ValueError(
            f"No play-by-play data is available for the {season} season yet. "
            "nflverse publishes it once games have been played."
        )
    return df


@st.cache_data(ttl=3600)
def load_schedule(season: int) -> pd.DataFrame:
    """nflverse's schedule: ESPN event id, spread line, roof, kickoff time."""
    try:
        sched = pd.read_parquet(f"{NFLVERSE_RELEASE}/schedules/games.parquet")
    except Exception:
        try:
            sched = nfl.import_schedules([season])
        except Exception:
            return pd.DataFrame()
    return sched[sched["season"] == season].reset_index(drop=True)


def list_games(pbp: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """One row per game with date, teams, game_id and data source.

    source is "nflfastR" once nflverse has published the game's pbp, and
    "live" for a game that has kicked off (or is about to) but isn't
    published yet — those are read from ESPN's play feed.
    """
    published = set(pbp["game_id"].unique()) if not pbp.empty else set()
    if not sched.empty:
        g = sched.copy()
        g["game_date"] = g["gameday"]
        today = date.today().isoformat()
        g = g[(g["game_id"].isin(published)) | ((g["gameday"] <= today) & g["espn"].notna())]
        g["source"] = np.where(g["game_id"].isin(published), "nflfastR", "live")
        g = g.sort_values(["week", "gameday", "gametime"])
    elif not pbp.empty:
        g = (
            pbp.groupby("game_id", as_index=False)
            .agg(week=("week", "first"),
                 game_date=("game_date", "first"),
                 home_team=("home_team", "first"),
                 away_team=("away_team", "first"))
            .sort_values(["week", "game_date"])
        )
        g["source"] = "nflfastR"
    else:
        return pd.DataFrame(columns=["game_id", "label", "source"])
    g["label"] = g.apply(
        lambda r: f"{r['away_team']} @ {r['home_team']} ({r['game_date']})"
                  + (" 🔴 live" if r["source"] == "live" else ""),
        axis=1,
    )
    return g.reset_index(drop=True)


@st.cache_resource
def load_nflfastr_models():
    """nflfastR's EP/WP boosters + FG table, or the error that stopped them."""
    try:
        return nflfastr_models.load_models(), nflfastr_models.load_fg_table(), None
    except Exception as e:  # no network to fastrmodels, xgboost missing, ...
        return None, None, str(e)


@st.cache_data(ttl=20)
def load_live_game(sched_row: dict) -> tuple[pd.DataFrame, str, str, str | None]:
    """A not-yet-published game from ESPN's play feed, in nflfastR's layout,
    with EP/EPA/WP from nflfastR's models. Returns (pbp, espn state,
    fetched-at clock, model error). Cached 20s, so reruns don't hammer ESPN."""
    summary = live_feed.fetch_summary(int(sched_row["espn"]))
    models, fg_table, model_err = load_nflfastr_models()
    df = live_feed.build_live_pbp(summary, sched_row, models, fg_table, PBP_COLS)
    return df, live_feed.game_state(summary), datetime.now().strftime("%H:%M:%S"), model_err


# ---------- College football ----------
# Published games only, from sportsdataverse (see cfb_feed). Like nflverse, it
# is rebuilt about once a day; a game shows up the morning after it's played.
@st.cache_data(ttl=60)
def cfb_stamp() -> str:
    """When the college pbp was last rebuilt. Checked every minute."""
    return cfb_feed.stamp() or f"unknown-{int(time.time() // 600)}"


@st.cache_resource(max_entries=2)
def load_cfb_season(season: int, stamp: str) -> pd.DataFrame:
    """A season of college pbp, raw release columns. Shared, not copied, so
    callers must not modify it. `stamp` is only a cache key."""
    return cfb_feed.read_season(season)


@st.cache_data(ttl=3600)
def load_cfb_schedule(season: int) -> pd.DataFrame:
    return cfb_feed.read_schedule(season)


@st.cache_data(max_entries=4)
def list_cfb_games(season: int, stamp: str) -> pd.DataFrame:
    return cfb_feed.list_games(load_cfb_season(season, stamp), load_cfb_schedule(season))


@st.cache_data(max_entries=8)
def load_cfb_game(season: int, stamp: str, game_id: str) -> pd.DataFrame:
    raw = load_cfb_season(season, stamp)
    return cfb_feed.to_pbp(raw[raw["game_id"].astype(str) == game_id], PBP_COLS)


@st.cache_data(ttl=3600)
def load_cfb_team_meta(season: int, stamp: str) -> tuple[dict, dict, dict]:
    """(colors, logos, nicknames) keyed by team abbreviation."""
    return cfb_feed.team_meta(load_cfb_season(season, stamp), cfb_feed.read_team_info(season))


# ---------- Replay logic ----------
# A real NFL broadcast runs ~3h10m for 60 minutes of game clock, a college one
# ~3h20m. Used only to show an approximate broadcast position alongside the
# game clock.
BROADCAST_MINUTES = 190.0
CFB_BROADCAST_MINUTES = 200.0
GAME_SECONDS = 3600.0


def play_timeline(pbp_game: pd.DataFrame) -> pd.Series:
    """Elapsed game seconds at each play, indexed positionally like `pbp_game`.

    game_seconds_remaining counts DOWN from 3600 at kickoff to 0 at the end of
    regulation. Null-clock rows (timeouts, end-of-period admin plays) inherit
    the nearest real clock value so they aren't treated as kickoff-time.

    Overtime gets its own stretch after regulation: OT period n (qtr 4 + n)
    spans 3600 + 900·(n-1) to 3600 + 900·n, the same 15-minute slot the
    sidebar's quarter + clock inputs give it. In nflfastR an OT clock counts
    down inside its period (a 2024 OT game carries 600, 563, 523, ...), so
    taken as game seconds remaining every OT play would land at or before the
    end of Q4, and seeding the cursor at any OT clock would unlock the whole
    overtime. College OT is untimed: its plays have no clock, so each is
    placed cfb_feed.OT_PLAY_SECS after the one before it within its period.

    cummax keeps the series sorted (every lookup into it needs that) where a
    feed's clock steps backwards; it only ever moves a play later.
    """
    q = pbp_game["qtr"].ffill().bfill().fillna(1).to_numpy(float)
    gsr = pbp_game["game_seconds_remaining"]
    el = (3600 - gsr.ffill().bfill().fillna(3600)).to_numpy(float)
    ot = q >= 5
    if ot.any():
        o = pd.DataFrame({"q": q[ot], "c": gsr.to_numpy(float)[ot]})
        clock = o.groupby("q")["c"].transform(lambda s: s.ffill().bfill())
        nth = o.groupby("q").cumcount() + 1
        start = 3600 + (o["q"] - 5) * 900
        # Strictly after the period start, so the end of Q4 never unlocks
        # a playoff OT's opening kickoff (its clock reads a full 15:00).
        el[ot] = np.where(clock.notna(), np.maximum(start + 900 - clock, start + 1),
                          start + np.minimum(nth * cfb_feed.OT_PLAY_SECS, 899))
    return pd.Series(el).cummax()


def cursor_from_elapsed(timeline: pd.Series, elapsed_game_s: float) -> int:
    """Index of the last play at or before `elapsed_game_s`; -1 if none."""
    return int(np.searchsorted(timeline.values, float(elapsed_game_s), side="right")) - 1


def cursor_anchor(timeline: pd.Series, idx: int) -> tuple[float, int]:
    """A frame-independent handle on play `idx`: its elapsed game seconds and
    its rank among the plays sharing that second (1 = first)."""
    if idx < 0:
        return (-1.0, 0)
    e = float(timeline.iloc[idx])
    return (e, idx - int(np.searchsorted(timeline.values, e, side="left")) + 1)


def cursor_from_anchor(timeline: pd.Series, anchor: tuple[float, int]) -> int:
    """The play in another frame of the same game that `anchor` points to.

    Used when the frame under the cursor changes: the live feed gains plays,
    or the game switches from the live feed to official nflfastR pbp (whose
    indices differ — it has extra admin rows). Never lands past the anchor's
    second, and within that second never past the same rank, so a switch
    can't unlock anything."""
    e, rank = anchor
    if e < 0:
        return -1
    lo = int(np.searchsorted(timeline.values, e, side="left"))
    hi = int(np.searchsorted(timeline.values, e, side="right"))
    if hi > lo:
        return min(lo + rank - 1, hi - 1)
    return lo - 1


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
        # Every overtime period goes in the one OT column.
        in_q = revealed[revealed["qtr"] >= 5] if q == 5 else revealed[revealed["qtr"] == q]
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


_LOWER_IS_BETTER = {"Turnovers", "Penalties", "Penalty Yds"}


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

        # Penalties are charged against the team penalized (penalty_team),
        # which may differ from possession — so this is computed on the
        # full revealed slice, not the posteam-filtered `td`.
        pen_mask = revealed["penalty_team"] == team
        penalties = int(pen_mask.sum())
        penalty_yds = int(revealed.loc[pen_mask, "penalty_yards"].fillna(0).sum())

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
            penalties,
            penalty_yds,
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
        "Turnovers", "Penalties", "Penalty Yds", "TOP",
    ]
    return pd.DataFrame(stats, index=pd.Index(index, name="Stat"))


# What the percentile colors compare against, per league.
BASELINE_LABEL = {NFL: "last 3 seasons", CFB: "last season's FBS-vs-FBS games"}
# Every column the baselines read, so the college season is mapped only once.
_BASELINE_COLS = [
    "game_id", "posteam",
    "pass_attempt", "rush_attempt", "qb_kneel", "qb_spike", "epa",
    "passing_yards", "rushing_yards",
    "complete_pass", "air_yards",
    "interception", "fumble_lost", "first_down",
    "third_down_converted", "third_down_failed",
    "yardline_100", "touchdown",
    "drive", "qtr", "game_seconds_remaining",
    "penalty", "penalty_team", "penalty_yards",
    "down", "ydstogo",
]


@st.cache_data(ttl=86400, max_entries=1)
def _cfb_baseline_pbp(season: int) -> pd.DataFrame:
    """The college season before `season`, FBS-vs-FBS games only, mapped to
    the app's layout. Empty if it isn't published."""
    prior = season - 1
    if prior < cfb_feed.FIRST_SEASON:
        return pd.DataFrame(columns=_BASELINE_COLS)
    pbp = cfb_feed.to_pbp(cfb_feed.read_season(prior), PBP_COLS)
    fbs = cfb_feed.fbs_games(cfb_feed.read_schedule(prior))
    if fbs:
        pbp = pbp[pbp["game_id"].isin(fbs)]
    return pbp[_BASELINE_COLS].reset_index(drop=True)


def _baseline_pbp(league: str, season: int, cols: list[str]) -> pd.DataFrame | None:
    """Play-by-play the percentile baselines are built from: the 3 NFL seasons
    before `season`, or last college season's FBS-vs-FBS games (one college
    season has as many team-games as three NFL ones). None if unavailable."""
    try:
        if league == CFB:
            df = _cfb_baseline_pbp(season)
            return df[cols] if not df.empty else None
        prior = [s for s in [season - 3, season - 2, season - 1] if s >= 1999]
        if not prior:
            return None
        return nfl.import_pbp_data(prior, columns=cols, downcast=True,
                                   include_participation=False)
    except Exception:  # network error, missing season data, etc.
        return None


@st.cache_data(ttl=86400)
def load_stat_baselines(season: int, league: str = NFL) -> dict[str, np.ndarray]:
    """Per-stat distributions from the seasons before `season`
    (see `_baseline_pbp`), sorted ascending."""
    cols = [
        "game_id", "posteam",
        "pass_attempt", "rush_attempt", "qb_kneel", "qb_spike", "epa",
        "passing_yards", "rushing_yards",
        "complete_pass", "air_yards",
        "interception", "fumble_lost", "first_down",
        "third_down_converted", "third_down_failed",
        "yardline_100", "touchdown",
        "drive", "qtr", "game_seconds_remaining",
        "penalty", "penalty_team", "penalty_yards",
    ]
    raw_all = _baseline_pbp(league, season, cols)
    if raw_all is None:
        return {}

    raw = raw_all[raw_all["posteam"].notna()]

    def _game_stats(td):
        pass_mask = (td["pass_attempt"].fillna(0) == 1) & (td["qb_spike"].fillna(0) == 0)
        rush_mask = (td["rush_attempt"].fillna(0) == 1) & (td["qb_kneel"].fillna(0) == 0)
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

    # Penalties are charged against penalty_team, which can differ from
    # posteam (e.g. a defensive penalty), so they're aggregated from the
    # full unfiltered play set and joined in by (game_id, penalty_team).
    pen = raw_all[
        (raw_all["penalty"].fillna(0) == 1) & raw_all["penalty_team"].notna()
    ]
    pen_stats = pen.groupby(["game_id", "penalty_team"]).agg(
        **{
            "Penalties": ("penalty", "size"),
            "Penalty Yds": ("penalty_yards", lambda s: s.fillna(0).sum()),
        }
    )
    pen_stats.index = pen_stats.index.set_names(["game_id", "posteam"])
    per_game = per_game.join(pen_stats, how="left")
    per_game[["Penalties", "Penalty Yds"]] = per_game[["Penalties", "Penalty Yds"]].fillna(0)

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
            pass_plays = tm[(tm["pass_attempt"].fillna(0) == 1) & (tm["qb_spike"].fillna(0) == 0)]
            rush_plays = tm[(tm["rush_attempt"].fillna(0) == 1) & (tm["qb_kneel"].fillna(0) == 0)]
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
def load_situational_baselines(season: int, league: str = NFL) -> dict[str, dict[str, np.ndarray]]:
    """Per-situation metric distributions from the seasons before `season`
    (see `_baseline_pbp`)."""
    cols = ["game_id", "posteam", "pass_attempt", "rush_attempt", "qb_kneel", "qb_spike", "epa", "down", "ydstogo"]
    raw = _baseline_pbp(league, season, cols)
    if raw is None:
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
            pass_plays = grp[(grp["pass_attempt"].fillna(0) == 1) & (grp["qb_spike"].fillna(0) == 0)]
            rush_plays = grp[(grp["rush_attempt"].fillna(0) == 1) & (grp["qb_kneel"].fillna(0) == 0)]
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


def stat_percentiles(df: pd.DataFrame, baselines: dict) -> pd.DataFrame:
    """Percentile of each Team stats cell vs the baseline seasons, 100 = best
    (flipped for stats where fewer is better). NaN without a baseline."""
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for row in df.index:
        arr = baselines.get(row, np.array([]))
        for team in df.columns:
            pct = _percentile_of(float(df.at[row, team]), arr)
            out.at[row, team] = 100.0 - pct if row in _LOWER_IS_BETTER and pd.notna(pct) else pct
    return out


def top_players(revealed: pd.DataFrame, team: str, kind: str, n: int = 3,
                drop: frozenset = frozenset()) -> pd.DataFrame:
    """Leaders for a team so far. `drop` names columns the data source doesn't
    have (they'd only ever read 0)."""
    td = revealed[revealed["posteam"] == team]
    if kind == "passing":
        pass_td = td[(td["pass_attempt"] == 1) & (td["qb_spike"].fillna(0) == 0)].copy()
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
        rush_td = td[(td["rush_attempt"] == 1) & (td["qb_kneel"].fillna(0) == 0)].copy()
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
        recv_td = td[
            td["receiver_player_name"].notna()
            & (td["pass_attempt"].fillna(0) == 1)
            & (td["qb_spike"].fillna(0) == 0)
        ].copy()
        if recv_td.empty:
            return pd.DataFrame()
        recv_td["_exp"] = (
            (recv_td["complete_pass"].fillna(0) == 1)
            & (recv_td["receiving_yards"].fillna(0) >= 20)
        ).astype(int)
        g = recv_td.groupby("receiver_player_name", as_index=False).agg(
            Tgt=("pass_attempt", "sum"),
            Rec=("complete_pass", "sum"),
            Yds=("receiving_yards", "sum"),
            YAC=("yards_after_catch", "sum"),
            aDOT=("air_yards", "mean"),
            TD=("pass_touchdown", "sum"),
            Exp=("_exp", "sum"),
            _epa=("epa", "sum"), _plays=("epa", "count"))
        g["Yds"] = g["Yds"].fillna(0)
        g["YAC"] = g["YAC"].fillna(0)
        g["aDOT"] = g["aDOT"].round(1)
        g = g.rename(columns={"receiver_player_name": "Player"})
    g = g.dropna(subset=["Player"])
    int_cols = [c for c in g.select_dtypes("number").columns if c not in ("_epa", "_plays", "aDOT", "EPA/play", "SR%")]
    g[int_cols] = g[int_cols].astype(int)
    g["EPA/play"] = (g["_epa"] / g["_plays"]).round(2)
    g = g.drop(columns=["_epa", "_plays", *[c for c in drop if c in g.columns]])
    return g.sort_values("Yds", ascending=False).head(n)


def top_defenders(revealed: pd.DataFrame, team: str, n: int = 5,
                  drop: frozenset = frozenset()) -> pd.DataFrame:
    """Defensive leaders for a team: tackles, sacks, QB hits, TFLs, INTs, PDs, FFs.
    `drop` names columns the data source doesn't have."""
    _ST_TYPES = {"kickoff", "punt", "field_goal", "extra_point", "no_play"}
    td = revealed[
        (revealed["defteam"] == team)
        & (~revealed["play_type"].isin(_ST_TYPES))
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
    g = g.drop(columns=[c for c in drop if c in g.columns])
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
        _special = {"kickoff", "extra_point", "no_play"}
        yl_series = off[~off["play_type"].isin(_special)]["yardline_100"].dropna()
        if yl_series.empty:
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


_FIELD_PLAYS = {"pass", "run", "punt", "field_goal", "qb_kneel", "qb_spike", "no_play"}


def _yl_label(y: float) -> str:
    """yardline_100 → 'OWN 25' / '50' / 'OPP 3'."""
    y = int(round(y))
    return f"OWN {100 - y}" if y > 50 else ("50" if y == 50 else f"OPP {y}")


# Play categories that colour the drive bars: (legend label, colour), in legend order.
_PLAY_CATS = {
    "pass_early": ("Pass · 1st/2nd down", "#2a78d6"),
    "pass_late": ("Pass · 3rd/4th down", "#4a3aa7"),
    "run_early": ("Run · 1st/2nd down", "#e34948"),
    "run_late": ("Run · 3rd/4th down", "#a8327a"),
    "penalty": ("Penalty", "#eda100"),
}


def _play_category(r) -> str | None:
    """Drive-bar category of a play; None for plays that don't get a segment
    (punts, field goals, timeouts and other no_play rows without a penalty).
    Scrambles count as runs and sacks as passes, as in nflfastR's play_type."""
    pt = r["play_type"]
    if pt == "no_play":
        return "penalty" if r.get("penalty") == 1 else None
    late = pd.notna(r["down"]) and r["down"] >= 3
    if pt in ("pass", "qb_spike"):
        return "pass_late" if late else "pass_early"
    if pt in ("run", "qb_kneel"):
        return "run_late" if late else "run_early"
    return None


def drive_field_spots(revealed: pd.DataFrame) -> pd.DataFrame:
    """Start and end spot (yardline_100 of the offense) for every revealed drive.

    The end spot is where the ball was when the drive ended: the goal line for
    a TD, the line of scrimmage for a punt, FG or interception, and the spot
    after the last play otherwise (including the drive in progress)."""
    if revealed.empty or "drive" not in revealed.columns:
        return pd.DataFrame()
    # Time of possession runs until the next drive starts, since the final
    # punt, kick or scoring play takes clock too. That only holds within a
    # half: the clock resets at halftime and again in overtime.
    groups = list(revealed.groupby("drive", sort=True))
    nxt_start = {}
    for (d, grp), (_, nxt) in zip(groups, groups[1:]):
        a = grp[["game_seconds_remaining", "qtr"]].dropna()
        b = nxt[["game_seconds_remaining", "qtr"]].dropna()
        if a.empty or b.empty:
            continue
        qa, qb = int(a["qtr"].iloc[-1]), int(b["qtr"].iloc[0])
        if qa < 5 and qb < 5 and (qa <= 2) == (qb <= 2):
            nxt_start[d] = float(b["game_seconds_remaining"].iloc[0])
    rows = []
    for drive_num, grp in groups:
        gsr = grp["game_seconds_remaining"].dropna()
        top = None
        if not gsr.empty:
            top = max(0.0, float(gsr.max()) - nxt_start.get(drive_num, float(gsr.min())))
        off = grp[grp["posteam"].notna()]
        plays = off[
            off["play_type"].isin(_FIELD_PLAYS)
            & off["two_point_conv_result"].isna()
            & off["yardline_100"].notna()
        ]
        if plays.empty:
            continue  # e.g. only the kickoff is revealed so far
        posteam = plays["posteam"].iloc[0]
        start = float(plays["yardline_100"].iloc[0])
        last = plays.iloc[-1]
        los = float(last["yardline_100"])
        turnover = bool(
            plays["interception"].fillna(0).astype(bool).any()
            or plays["fumble_lost"].fillna(0).astype(bool).any()
        )
        if last["play_type"] in ("punt", "field_goal") or last.get("interception") == 1:
            end = los
        elif last.get("safety") == 1:
            end = 100.0
        elif last.get("touchdown") == 1 and not turnover:
            end = 0.0
        else:
            gained = 0.0 if pd.isna(last["yards_gained"]) else float(last["yards_gained"])
            if last["play_type"] == "no_play" and last.get("penalty") == 1:
                pen = 0.0 if pd.isna(last["penalty_yards"]) else float(last["penalty_yards"])
                gained = -pen if last["penalty_team"] == posteam else pen
            end = los - gained
        end = min(max(end, 0.0), 100.0)
        # One segment per play, from its line of scrimmage to the next play's
        # (the drive's end spot for the last one).
        nxt_spot = plays["yardline_100"].astype(float).tolist()[1:] + [end]
        segments = []
        for (_, p), to in zip(plays.iterrows(), nxt_spot):
            cat = _play_category(p)
            if cat is None:
                continue
            frm = float(p["yardline_100"])
            desc = str(p["desc"] or "")
            dd = _down_distance(p)
            segments.append({
                "cat": cat, "from": frm, "to": to,
                "hover": (f"<b>{_PLAY_CATS[cat][0]}</b>"
                          + f"<br>{dd + ' · ' if dd else ''}{_yl_label(frm)} · {frm - to:+.0f} yds"
                          + f"<br>{desc[:90] + '…' if len(desc) > 90 else desc}"),
            })
        sc = off[(off["pass_attempt"].fillna(0) == 1) | (off["rush_attempt"].fillna(0) == 1)]
        rows.append({
            "drive": int(drive_num),
            "team": posteam,
            "qtr": int(off["qtr"].dropna().iloc[0]) if off["qtr"].notna().any() else None,
            "start": start,
            "end": end,
            "plays": len(sc),
            "yards": int(sc["yards_gained"].fillna(0).sum()),
            "top": top,
            "outcome": _drive_outcome(grp),
            "segments": segments,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # The last revealed drive hasn't necessarily ended: label it as in progress
    # unless the last revealed row is itself an end-of-quarter/half/game marker.
    last_desc = str(revealed["desc"].iloc[-1] or "")
    if df["outcome"].iloc[-1] == "EOH/EOG" and not last_desc.upper().startswith("END "):
        df.loc[df.index[-1], "outcome"] = "In progress"
    return df


_FIELD_GREEN = "#2f7a3b"
_FIELD_GREEN_ALT = "#2a6f35"


def drive_field_figure(spots: pd.DataFrame, home: str, away: str,
                       colors: dict[str, str], logos: dict[str, str],
                       nicknames: dict[str, str]) -> go.Figure:
    """Football field with one arrow per drive, latest drive on top.

    x is measured from the home team's goal line: the home team defends the
    left end zone and drives left → right; the away team drives right → left."""
    n = len(spots)
    fig = go.Figure()

    # Field: alternating 5-yard stripes, yard lines, end zones in team colors.
    shapes = []
    for i, x0 in enumerate(range(0, 100, 5)):
        shapes.append(dict(type="rect", x0=x0, x1=x0 + 5, y0=0, y1=1, yref="paper",
                           fillcolor=_FIELD_GREEN if i % 2 == 0 else _FIELD_GREEN_ALT,
                           line_width=0, layer="below"))
    for team, x0, x1 in [(home, -10, 0), (away, 100, 110)]:
        shapes.append(dict(type="rect", x0=x0, x1=x1, y0=0, y1=1, yref="paper",
                           fillcolor=colors.get(team, "#555555"),
                           line=dict(color="white", width=2), layer="below"))
    for x in range(5, 100, 5):
        shapes.append(dict(type="line", x0=x, x1=x, y0=0, y1=1, yref="paper",
                           line=dict(color="rgba(255,255,255,%s)" % (0.7 if x % 10 == 0 else 0.3),
                                     width=2 if x == 50 else 1),
                           layer="below"))
    shapes.append(dict(type="rect", x0=-10, x1=110, y0=0, y1=1, yref="paper",
                       line=dict(color="white", width=2), layer="below"))

    # End zone branding: logo top and bottom, nickname written along the zone.
    images, annotations = [], []
    for team, xc, angle in [(home, -5, -90), (away, 105, 90)]:
        logo = logos.get(team)
        if logo:
            for yc in (0.86, 0.14):
                images.append(dict(source=logo, xref="x", yref="paper", x=xc, y=yc,
                                   sizex=8, sizey=0.2, xanchor="center", yanchor="middle",
                                   sizing="contain", layer="above"))
        annotations.append(dict(
            x=xc, y=0.5, xref="x", yref="paper", showarrow=False, textangle=angle,
            text=f"<b>{nicknames.get(team, team).upper()}</b>",
            font=dict(color="white", size=12), xanchor="center", yanchor="middle"))

    # Each drive is a bar of per-play segments coloured by play category, with
    # a tick at every snap so plays that gained nothing still show. Traces are
    # batched per category (None breaks the line between segments).
    seg_xy = {c: ([], []) for c in _PLAY_CATS}
    snaps = {c: ([], [], []) for c in _PLAY_CATS}
    ends = []
    for i, d in enumerate(spots.itertuples(index=False)):
        is_home = d.team == home
        to_x = (lambda y: 100 - y) if is_home else (lambda y: y)
        x0, x1 = to_x(d.start), to_x(d.end)
        # White underlay so the bar reads against the turf. It spans every
        # segment, since a loss can carry the ball behind the drive's start.
        span = [x0, x1] + [to_x(sg[k]) for sg in d.segments for k in ("from", "to")]
        fig.add_trace(go.Scatter(x=[min(span), max(span)], y=[i, i], mode="lines",
                                 line=dict(color="white", width=8),
                                 hoverinfo="skip", showlegend=False))
        for sgm in d.segments:
            xs, ys = seg_xy[sgm["cat"]]
            xs += [to_x(sgm["from"]), to_x(sgm["to"]), None]
            ys += [i, i, None]
            sx, sy, tx = snaps[sgm["cat"]]
            sx.append(to_x(sgm["from"]))
            sy.append(i)
            tx.append(sgm["hover"])
        ends.append((i, d, x0, x1, colors.get(d.team, "#1f77b4" if is_home else "#ff7f0e")))

    for cat, (label, color) in _PLAY_CATS.items():
        xs, ys = seg_xy[cat]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=label, legendgroup=cat,
                                 line=dict(color=color, width=5), hoverinfo="skip"))
    for cat, (label, color) in _PLAY_CATS.items():
        sx, sy, tx = snaps[cat]
        fig.add_trace(go.Scatter(
            x=sx, y=sy, mode="markers", name=label, legendgroup=cat, showlegend=False,
            marker=dict(symbol="line-ns", size=13, line=dict(color=color, width=3)),
            hovertext=tx, hovertemplate="%{hovertext}<extra></extra>"))

    for i, d, x0, x1, color in ends:
        hover = (f"<b>Drive {d.drive} · {d.team}</b>"
                 + (f" · Q{d.qtr}" if d.qtr else "")
                 + f"<br>{_yl_label(d.start)} → {_yl_label(d.end)}"
                 + f"<br>{d.plays} plays, {d.yards} yds, {_fmt_top(d.top)}"
                 + f"<br>{d.outcome}")
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[i, i], mode="markers",
            marker=dict(symbol=["circle", "triangle-right" if x1 >= x0 else "triangle-left"],
                        size=[9, 13], color=color, line=dict(color="white", width=1.5)),
            hovertemplate=hover + "<extra></extra>", showlegend=False))
        annotations.append(dict(
            x=1.0, y=i, xref="paper", yref="y", xanchor="left", showarrow=False,
            text=f" {d.outcome}", font=dict(size=11)))

    tick = list(range(10, 100, 10))
    fig.update_layout(
        shapes=shapes, images=images, annotations=annotations,
        height=max(330, 100 + 26 * n),
        margin=dict(l=10, r=85, t=30, b=40),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=0, yanchor="top",
                    itemclick="toggle", itemdoubleclick="toggleothers"),
        plot_bgcolor=_FIELD_GREEN,
        hoverlabel=dict(align="left"),
        xaxis=dict(range=[-10, 110], tickvals=tick,
                   ticktext=[str(50 - abs(50 - t)) for t in tick],
                   side="top", showgrid=False, zeroline=False, fixedrange=True),
        yaxis=dict(range=[-0.7, n - 0.3], tickvals=list(range(n)),
                   ticktext=[f"Q{d.qtr} {d.team}" if d.qtr else d.team
                             for d in spots.itertuples(index=False)],
                   showgrid=False, zeroline=False, fixedrange=True),
    )
    return fig


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


def scoring_timeline(revealed: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    """Every scoring play in revealed: TD, FG, Safety, XP, 2PT."""
    if revealed.empty:
        return pd.DataFrame()
    mask = (
        (revealed["touchdown"].fillna(0) == 1) |
        (revealed["field_goal_result"] == "made") |
        (revealed["safety"].fillna(0) == 1) |
        (revealed["extra_point_result"] == "good") |
        (revealed["two_point_conv_result"] == "success")
    )
    plays = revealed[mask].copy()
    if plays.empty:
        return pd.DataFrame()

    def _score_type(r) -> str:
        if r["safety"] == 1:
            return "Safety"
        if r["field_goal_result"] == "made":
            return "FG"
        if r["extra_point_result"] == "good":
            return "XP"
        if r["two_point_conv_result"] == "success":
            return "2PT"
        return "TD"

    plays["Type"] = plays.apply(_score_type, axis=1)
    plays["Team"] = plays.apply(
        lambda r: r["defteam"] if r["safety"] == 1 else r["posteam"], axis=1
    )
    plays["Score"] = plays.apply(
        lambda r: f"{away} {int(r['total_away_score'] or 0)} — {int(r['total_home_score'] or 0)} {home}",
        axis=1,
    )
    plays["Q"] = plays["qtr"].apply(lambda x: str(int(x)) if pd.notna(x) else "")
    return plays[["Q", "time", "Team", "Type", "Score", "desc"]].rename(
        columns={"time": "Clock", "desc": "Description"}
    ).reset_index(drop=True)


def _style_scoring_timeline(df: pd.DataFrame):
    def _type_color(val):
        if val == "TD":
            return "background-color: #d4edda; color: #155724; font-weight: bold"
        if val == "FG":
            return "background-color: #cce5ff; color: #004085"
        if val == "Safety":
            return "background-color: #f8d7da; color: #721c24"
        if val in ("XP", "2PT"):
            return "background-color: #e2e3e5; color: #383d41"
        return ""

    styled = df.style
    if "Type" in df.columns:
        styled = _smap(styled, _type_color, subset=["Type"])
    return (
        styled
        .set_properties(**{"text-align": "center"})
        .set_table_styles(
            [{"selector": "th", "props": [("text-align", "center"), ("font-weight", "bold")]}]
        )
    )


def explosive_plays(revealed: pd.DataFrame, min_pass_yds: int = 15, min_rush_yds: int = 10) -> pd.DataFrame:
    """Passing plays >= min_pass_yds yards or rushing plays >= min_rush_yds yards, sorted by yards desc."""
    if revealed.empty:
        return pd.DataFrame()
    pass_mask = (revealed["pass_attempt"].fillna(0) == 1) & (revealed["passing_yards"].fillna(0) >= min_pass_yds)
    rush_mask = (revealed["rush_attempt"].fillna(0) == 1) & (revealed["rushing_yards"].fillna(0) >= min_rush_yds)
    return revealed[pass_mask | rush_mask].sort_values("yards_gained", ascending=False)


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
    plays["D&D"] = plays.apply(_down_distance, axis=1)
    plays["WPA"] = plays["_wpa"].round(3)
    plays["For"] = plays["_wpa"].apply(lambda x: home if x > 0 else away)
    return plays[["Q", "time", "posteam", "D&D", "Score", "For", "desc", "WPA", "_abs_wpa"]].rename(
        columns={"time": "Clock", "posteam": "Off", "desc": "Description"}
    ).reset_index(drop=True)


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
    if t.startswith("3rd"):
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
_PLAYS_COL_CFG = {
    **_EPA_COL_CFG, "_rz": None,
    "Description": st.column_config.TextColumn(width="large"),
}

def _plays_row_height(hide_desc: bool) -> int | None:
    """Row height for play tables: st.dataframe only wraps cell text when
    row_height is above 4rem (64px), so give descriptions room for ~3 lines."""
    return None if hide_desc else 84


# ---------- Time bar ----------
def _time_bar_html(frac: float, fill: str, in_ot: bool) -> str:
    """Track showing position within the game. Nothing past `frac` is drawn:
    no scoring marks, no play density, no total play count — any of those would
    give away what the viewer hasn't watched yet."""
    pct = min(max(float(frac), 0.0), 1.0) * 100.0
    ticks = "".join(
        f'<div style="position:absolute;left:{q}%;top:0;bottom:0;width:1px;'
        f'background:rgba(128,128,128,0.55)"></div>'
        for q in (25, 50, 75)
    )
    ot_track = ot_label = ""
    if in_ot:
        ot_track = (f'<div style="flex:0 0 12%;height:12px;border-radius:6px;'
                    f'background:{fill};margin-left:4px"></div>')
        ot_label = '<div style="flex:0 0 12%;margin-left:4px;text-align:center">OT</div>'
    return f"""
<div style="display:flex;align-items:center;margin:0.2rem 0 0.2rem 0">
  <div style="flex:1 1 auto;position:relative;height:12px;border-radius:6px;
              background:rgba(128,128,128,0.22);overflow:hidden">
    <div style="position:absolute;left:0;top:0;bottom:0;width:{pct:.3f}%;
                background:{fill}"></div>
    {ticks}
  </div>{ot_track}
</div>
<div style="display:flex;font-size:0.72rem;opacity:0.6;margin-bottom:0.2rem">
  <div style="flex:1 1 auto;display:flex">
    <div style="flex:1">Q1</div><div style="flex:1">Q2</div>
    <div style="flex:1">Q3</div><div style="flex:1">Q4</div>
  </div>{ot_label}
</div>
"""


def keep_screen_awake(enabled: bool) -> None:
    """Hold (or release) a Screen Wake Lock so a phone doesn't dim or lock mid-game.

    components.html runs in a same-origin iframe, so the script requests the lock on
    the parent (top-level) document -- iframes need a permissions-policy grant that
    Streamlit doesn't give. The sentinel lives on the parent window so reruns reuse
    it, and it's re-acquired on visibilitychange because the browser drops the lock
    whenever the tab is hidden. Needs HTTPS (or localhost); iOS Safari 16.4+.
    """
    components.html(f"""
<script>
(function () {{
  const w = window.parent, nav = w.navigator, doc = w.document;
  const want = {str(enabled).lower()};
  w.__nflWakeWanted = want;
  if (!("wakeLock" in nav)) return;
  async function acquire() {{
    if (!w.__nflWakeWanted || doc.visibilityState !== "visible") return;
    if (w.__nflWakeLock && !w.__nflWakeLock.released) return;
    try {{ w.__nflWakeLock = await nav.wakeLock.request("screen"); }}
    catch (e) {{ console.warn("Wake lock not granted:", e); }}
  }}
  if (!w.__nflWakeListener) {{
    w.__nflWakeListener = true;
    doc.addEventListener("visibilitychange", acquire);
  }}
  if (want) acquire();
  else if (w.__nflWakeLock) {{ w.__nflWakeLock.release(); w.__nflWakeLock = null; }}
}})();
</script>
""", height=0)


def _setting(name: str) -> str | None:
    """A setting from .streamlit/secrets.toml, else the environment."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # no secrets.toml
        pass
    return os.environ.get(name) or None


# ---------- UI ----------
with st.sidebar:
    st.header("Setup")
    league = st.radio("League", [NFL, CFB], horizontal=True)
is_cfb = league == CFB

st.title("🏈 College Football Tape-Delay Replay" if is_cfb else "🏈 NFL Tape-Delay Replay")
st.caption("Spoiler-free boxscore that unlocks as your broadcast progresses.")

with st.sidebar:
    season = st.number_input("Season", min_value=cfb_feed.FIRST_SEASON if is_cfb else 1999,
                             max_value=2026, value=2026, step=1)

    if is_cfb:
        stamp = cfb_stamp()
        with st.spinner("Loading college play-by-play..."):
            try:
                games = list_cfb_games(int(season), stamp)
            except Exception as e:
                st.error(f"No college play-by-play for the {int(season)} season yet ({e}).")
                st.stop()
        if games.empty:
            st.warning(f"No college games published for the {int(season)} season yet.")
            st.stop()
        week_labels = list(dict.fromkeys(games["week_label"]))
        week = st.selectbox("Week", week_labels, index=len(week_labels) - 1)
        games = games[games["week_label"] == week]
        # 50+ games a Saturday: narrow by conference.
        confs = sorted((set(games["home_conf"].dropna()) | set(games["away_conf"].dropna())) - {""})
        conf = st.selectbox("Conference", ["All"] + confs)
        if conf != "All":
            games = games[(games["home_conf"] == conf) | (games["away_conf"] == conf)]
    else:
        stamp = nflverse_stamp()
        with st.spinner("Loading play-by-play..."):
            try:
                pbp = load_pbp(int(season), stamp)
            except Exception as e:
                # Nothing published for this season yet: live games can still load.
                pbp = pd.DataFrame(columns=PBP_COLS)
                pbp_error = str(e)
            else:
                pbp_error = None

        games = list_games(pbp, load_schedule(int(season)))
        if games.empty:
            if pbp_error:
                st.error(f"Could not load pbp: {pbp_error}")
            else:
                st.warning(f"No games found for the {int(season)} season yet.")
            st.stop()
        weeks = sorted(games["week"].dropna().astype(int).unique())
        week = st.selectbox("Week", weeks, index=len(weeks) - 1)
        games = games[games["week"] == week]
    game_label = st.selectbox("Game", games["label"].tolist())
    game_row = games.loc[games["label"] == game_label].iloc[0]
    game_id = game_row["game_id"]
    source = game_row["source"]

    if source == "published":
        # College: sportsdataverse's published pbp. No live feed yet.
        pbp_game = load_cfb_game(int(season), stamp, game_id)
        st.caption(f"📊 College play-by-play from ESPN, with sportsdataverse's EPA and "
                   f"win probability models · data as of {stamp}")
        st.caption("A game shows up here once sportsdataverse publishes it, usually "
                   "the morning after. Live college games aren't supported yet.")
        live_state = "post"
    elif source == "nflfastR":
        pbp_game = pbp[pbp["game_id"] == game_id].sort_values("play_id").reset_index(drop=True)
        st.caption(f"📊 Official nflfastR play-by-play · nflverse data as of {stamp}")
        live_state = "post"
    else:
        # Not published by nflverse yet: ESPN's live play feed, run through
        # nflfastR's own EP/WP models. Switches to the official data by
        # itself once nflverse publishes the game.
        _sched_row = {k: (None if pd.isna(v) else v) for k, v in game_row.items()
                      if k in ("game_id", "season", "game_type", "week", "gameday", "home_team",
                               "away_team", "espn", "spread_line", "roof", "location")}
        try:
            pbp_game, live_state, fetched_at, model_err = load_live_game(_sched_row)
        except Exception as e:
            st.error(f"Could not load the live play feed: {e}")
            st.stop()
        st.caption(f"🔴 Live · ESPN play feed + nflfastR models · updated {fetched_at}. "
                   "Switches to official nflfastR data once nflverse publishes it.")
        if model_err:
            st.warning(f"nflfastR's models couldn't load ({model_err}); "
                       "EPA and win probability are blank until they do.")
    if pbp_game.empty:
        st.info("This game hasn't started yet — no plays to show.")
        st.stop()
    home = pbp_game["home_team"].iloc[0]
    away = pbp_game["away_team"].iloc[0]
    timeline = play_timeline(pbp_game)
    last_idx = len(pbp_game) - 1

    if is_cfb:
        team_colors, team_logos, team_nicks = load_cfb_team_meta(int(season), stamp)
    else:
        team_colors, team_logos, team_nicks = load_team_colors(), load_team_logos(), load_team_nicknames()
    # Player-leader columns the data source has no data for.
    missing_stats = cfb_feed.MISSING_LEADER_STATS if is_cfb else frozenset()
    broadcast_minutes = CFB_BROADCAST_MINUTES if is_cfb else BROADCAST_MINUTES

    st.divider()
    st.subheader("Your viewing")

    # Watching the game as it airs: follow the feed instead of a typed clock.
    # Plays are revealed only once they've sat in the feed for `live_delay`
    # seconds, so a stream that runs behind ESPN never shows a play first.
    follow_live = False
    live_delay = 0
    if source == "live" and live_state != "post":
        follow_live = st.toggle(
            "🔴 Follow live", value=False,
            help="Keep up with the game as it airs. New plays appear on their own, "
                 "each held back by the delay below so the feed can't run ahead "
                 "of your TV or stream.")
        if follow_live:
            live_delay = st.slider(
                "Hold each new play back (seconds)", min_value=0, max_value=180,
                value=45, step=5,
                help="How far your broadcast runs behind ESPN's play feed. Cable is "
                     "usually close to live; streaming apps often run 30–90s behind.")

    if follow_live:
        st.caption("Following the live feed. Scrub back any time — you'll stay "
                   "where you are until you catch up.")
        qtr_pick, clock_str = "Q1", "15:00"
    else:
        st.caption("Set roughly where you are with the game clock, then sync exactly "
                   "with the time bar.")
        # The game clock counts DOWN within each quarter from 15:00 to 0:00.
        qtr_pick = st.selectbox("Quarter", ["Q1", "Q2", "Q3", "Q4", "OT"], index=0)
        if is_cfb and qtr_pick == "OT":
            st.caption("College overtime has no clock: you start at the end of "
                       "regulation. Step through it with ▶ Next play.")
            clock_str = "15:00"
        else:
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
    auto = False if follow_live else st.checkbox("Auto-advance play by play", value=False)
    if follow_live:
        # Fast enough that the delay above is honoured to within ~10s.
        st_autorefresh(interval=10_000, key="live_follow")
    elif auto:
        refresh_interval = st.selectbox("Refresh interval", [30, 45, 60, 90],
                                        format_func=lambda x: f"{x}s",
                                        key="refresh_interval_clock")
        st_autorefresh(interval=refresh_interval * 1000, key="autorefresh")
    elif source == "live" and live_state != "post":
        # Keep pulling new plays in. This only makes them available to the
        # ▶ buttons — it never moves the cursor or unlocks anything.
        st_autorefresh(interval=30_000, key="live_refresh")
    awake = st.checkbox("Keep screen awake", value=True,
                        help="Stops your phone from dimming or locking while this "
                             "page is open. Needs HTTPS and a recent browser "
                             "(iOS 16.4+, Android Chrome).")
    keep_screen_awake(awake)

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

# ---------- Play cursor ----------
# The cursor is the single source of truth for viewing position. A play index is
# exact where a game-seconds scalar is not: plays sharing one clock value (a
# penalty, a timeout, two snaps inside a second) become individually addressable,
# and overtime steps one play at a time. OT is the worst case for a clock filter,
# because its clock restarts inside the regulation range — every OT play looks
# *earlier* than the end of Q4, so a game-seconds threshold unlocks the entire
# overtime period at once. The clock inputs only seed the cursor.
#   _cursor_idx  the play you're currently on
#   _cursor_max  furthest play unlocked; this is the spoiler gate
#   _cursor_key  when the clock inputs change, re-seed both from the baseline
#   _cursor_frame / _cursor_anchor  when the frame under the cursor changes
#                (live plays arrive, or live → official switch), carry the
#                position over by game time instead of by index
baseline_cursor = cursor_from_elapsed(timeline, max(float(baseline_elapsed) - float(safety_margin), 0.0))
baseline_cursor = min(baseline_cursor, last_idx)

_seed_key = (game_id, qtr_pick, clock_str, safety_margin)
if st.session_state.get("_cursor_key") != _seed_key:
    st.session_state["_cursor_key"] = _seed_key
    st.session_state["_cursor_max"] = baseline_cursor
    st.session_state["_cursor_idx"] = baseline_cursor
elif (st.session_state.get("_cursor_frame") != (game_id, source, len(pbp_game))
        and "_cursor_anchor" in st.session_state):
    _a_idx, _a_max = st.session_state["_cursor_anchor"]
    st.session_state["_cursor_max"] = cursor_from_anchor(timeline, _a_max)
    st.session_state["_cursor_idx"] = min(cursor_from_anchor(timeline, _a_idx),
                                          st.session_state["_cursor_max"])
st.session_state["_cursor_frame"] = (game_id, source, len(pbp_game))

cursor_max = max(min(int(st.session_state["_cursor_max"]), last_idx), -1)
cursor_idx = max(min(int(st.session_state["_cursor_idx"]), cursor_max), -1)

# Follow live: unlock every play that has been in the feed for `live_delay`
# seconds. `_live_seen[game_id]` is the wall time each play position first
# showed up. Plays already there when you start following count as aired, all
# but the newest, which waits out the delay like any play arriving later.
_just_followed = follow_live and st.session_state.get("_live_follow_on") != game_id
st.session_state["_live_follow_on"] = game_id if follow_live else None
if follow_live:
    _now = time.time()
    _seen = st.session_state.setdefault("_live_seen", {})
    _ts = _seen.get(game_id)
    if _ts is None:
        _ts = [0.0] * last_idx + [_now]
    else:
        # ESPN can drop a play on revision; positions past the end are forgotten.
        _ts = _ts[: last_idx + 1] + [_now] * (last_idx + 1 - len(_ts))
    _seen[game_id] = _ts
    _aired = [i for i, t in enumerate(_ts) if t <= _now - live_delay]
    _live_target = _aired[-1] if _aired else -1
    if _live_target > cursor_max:
        # Carry the view along only if it was already at the edge, as auto-advance
        # does, so a replayed drive isn't yanked away mid-scrub.
        if cursor_idx >= cursor_max:
            cursor_idx = _live_target
        cursor_max = _live_target
    if _just_followed:
        cursor_idx = cursor_max
st.session_state["_cursor_max"] = cursor_max
st.session_state["_cursor_idx"] = cursor_idx
st.session_state["_cursor_anchor"] = (cursor_anchor(timeline, cursor_idx),
                                      cursor_anchor(timeline, cursor_max))

revealed = pbp_game.iloc[: cursor_idx + 1].copy()
# elapsed_s is still derived because the win probability chart caps its x-axis
# with it; nothing filters on it any more.
elapsed_s = float(timeline.iloc[cursor_idx]) if cursor_idx >= 0 else 0.0

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


def _logo_img(team: str) -> str:
    url = team_logos.get(team)
    return f"<img src='{url}' alt='{team}' style='height:clamp(36px,9vw,64px);width:auto'>" if url else ""


# One flex row that never wraps, so the scoreboard stays a single compact line
# on phones (st.columns would stack the logos and score vertically there).
st.markdown(
    "<div style='display:flex;align-items:center;justify-content:center;"
    "gap:clamp(8px,3vw,24px);margin:0.25rem 0 0.5rem'>"
    f"{_logo_img(away)}"
    f"<div style='font-size:clamp(1.3rem,6vw,2.4rem);font-weight:700;white-space:nowrap'>"
    f"{away} {away_score} — {home_score} {home}</div>"
    f"{_logo_img(home)}</div>",
    unsafe_allow_html=True)

st.markdown(
    f"<div style='text-align:center;font-size:1.1rem;opacity:0.8;margin-bottom:0.5rem'>"
    f"{game_summary.period_label(qtr_now)}"
    # College overtime has no clock to show.
    f"{'' if qtr_now >= 5 and game_clock == '—' else ' · ' + game_clock}</div>",
    unsafe_allow_html=True)

# ---------- Play-by-play time bar ----------
def _hex_or_none(v):
    """nfl_data_py leaves a few team_color cells blank/NaN."""
    return v if isinstance(v, str) and v.startswith("#") else None


_bar_fill = "#4c78a8"
_in_ot = False
if cursor_idx >= 0:
    _cur_row = pbp_game.iloc[cursor_idx]
    _bar_fill = (_hex_or_none(team_colors.get(_cur_row["posteam"]))
                 or _hex_or_none(team_colors.get(home))
                 or _bar_fill)
    _in_ot = pd.notna(_cur_row["qtr"]) and int(_cur_row["qtr"]) >= 5
st.markdown(_time_bar_html(elapsed_s / 3600.0, _bar_fill, _in_ot), unsafe_allow_html=True)


def _move_cursor(idx: int, unlock: bool = False) -> None:
    """Move the cursor and rerun. Only `unlock` raises the ceiling — advancing the
    reveal is always an explicit act, never a side effect of scrubbing."""
    idx = max(min(int(idx), last_idx), -1)
    st.session_state["_cursor_idx"] = idx
    if unlock:
        st.session_state["_cursor_max"] = max(int(st.session_state["_cursor_max"]), idx)
    st.session_state["_cursor_anchor"] = (
        cursor_anchor(timeline, idx),
        cursor_anchor(timeline, int(st.session_state["_cursor_max"])))
    st.rerun()


def _drive_end(cur: int) -> int:
    """Last play of the drive that the play after `cur` belongs to."""
    nxt = min(cur + 1, last_idx)
    drv = pbp_game["drive"]
    d = drv.iloc[nxt]
    if pd.isna(d):
        return nxt
    j = nxt
    while j < last_idx and drv.iloc[j + 1] == d:
        j += 1
    return j


# max_value is the spoiler gate: the slider physically cannot reach a play you
# haven't unlocked, so scrubbing is free but advancing needs a button.
if cursor_max >= 1:
    # 1-based so the slider's endpoints read the same as the caption's "play N".
    _pick = st.slider("Scrub within unlocked plays",
                      min_value=1, max_value=cursor_max + 1,
                      value=max(cursor_idx, 0) + 1,
                      help="Drag back to review a play you've already seen. To reveal "
                           "the next one, use ▶ Next play — the slider can't run ahead "
                           "of the broadcast.")
    if _pick - 1 != cursor_idx:
        _move_cursor(_pick - 1)

_b1, _b2, _b3, _b4, _b5 = st.columns(5)
if _b1.button("⏮ Start", key="bar_start", width='stretch', disabled=cursor_idx <= 0):
    _move_cursor(0)
if _b2.button("◀ Prev", key="bar_prev", width='stretch', disabled=cursor_idx <= 0):
    _move_cursor(cursor_idx - 1)
if _b3.button("▶ Next play", key="bar_next", width='stretch', disabled=cursor_idx >= last_idx):
    _move_cursor(cursor_idx + 1, unlock=True)
if _b4.button("▶▶ Next drive", key="bar_drive", width='stretch', disabled=cursor_idx >= last_idx):
    _move_cursor(_drive_end(cursor_idx), unlock=True)
if _b5.button("⏭ Catch up", key="bar_catchup", width='stretch', disabled=cursor_idx >= cursor_max):
    _move_cursor(cursor_max)

if cursor_idx >= 0:
    _r = pbp_game.iloc[cursor_idx]
    _q = int(_r["qtr"]) if pd.notna(_r["qtr"]) else 0
    _bits = [game_summary.period_label(_q) if _q >= 1 else "—"]
    if pd.notna(_r["time"]):
        _bits.append(str(_r["time"]))
    _dd, _fp = _down_distance(_r), _field_pos_label(_r)
    if _dd and _fp:
        _bits.append(f"{_dd} at {_fp}")
    elif _dd or _fp:
        _bits.append(_dd or _fp)
    # "of N unlocked", never "of N total" — the play count alone would leak
    # whether the game ran long.
    _bits.append(f"play {cursor_idx + 1} of {cursor_max + 1} unlocked")
    if _q <= 4:  # overtime's place on the timeline isn't real game minutes
        _bits.append(f"⏱ {elapsed_s / 60:.1f} game min "
                     f"(≈ {elapsed_s / GAME_SECONDS * broadcast_minutes:.0f} broadcast min)")
    st.caption(" · ".join(_bits))
else:
    st.caption("Kickoff — nothing revealed yet. Press **▶ Next play** to begin.")

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

# ---------- Boxscore ----------
st.subheader("Boxscore")
st.dataframe(boxscore(revealed, home, away), hide_index=True, width='stretch')

# ---------- AI game summary ----------
# An agent reads the same stats as the sections below, built from `revealed`
# only, and explains the score so far. It runs only when asked, and the result
# is kept until the next request.
st.subheader("🧠 Why the score is what it is")
_providers = game_summary.configured_providers(_setting)
_summary = st.session_state.get("_summary")
# Show a summary only for this game and only up to the unlocked edge. Moving the
# clock inputs back lowers that edge and hides it until you get there again.
if _summary and (_summary["game_id"] != game_id
                 or _summary["anchor"] > cursor_anchor(timeline, cursor_max)):
    _summary = None

if not _providers:
    st.caption("Add `MOONSHOT_API_KEY` (Kimi) or `ANTHROPIC_API_KEY` (Claude) to "
               "`.streamlit/secrets.toml` or the environment to get an AI summary of "
               "what you've watched.")
elif revealed.empty:
    st.caption("No plays revealed yet.")
else:
    _sum_btn, _sum_pick = st.columns([3, 2])
    _prov = next(iter(_providers))
    if len(_providers) > 1:
        _prov = _sum_pick.selectbox(
            "Model", list(_providers), label_visibility="collapsed",
            format_func=lambda k: f"{_providers[k].label} · {_providers[k].model}")
    if _sum_btn.button("Update summary" if _summary else "Explain the score so far",
                       help="Only the plays you've unlocked are sent to the model."):
        _season_baselines = load_stat_baselines(int(season), league)
        _stats = team_stats(revealed, home, away)
        _sit, _sit_n = situational_success_rate(revealed, home, away)
        _ctx = game_summary.GameContext(
            home=home, away=away, revealed=revealed,
            league="CFB" if is_cfb else "NFL",
            boxscore=boxscore(revealed, home, away),
            scoring=scoring_timeline(revealed, home, away),
            team_stats=_stats, team_pct=stat_percentiles(_stats, _season_baselines),
            situational=_sit, situational_counts=_sit_n,
            drives=drive_chart(revealed),
            top_wpa=top_plays_wpa(revealed, home, away),
            explosive=explosive_plays(revealed),
            leaders={
                **{(t, k): top_players(revealed, t, k, n, drop=missing_stats) for t in (away, home)
                   for k, n in (("passing", 3), ("rushing", 4), ("receiving", 8))},
                **{(t, "defense"): top_defenders(revealed, t, 10, drop=missing_stats)
                   for t in (away, home)},
            },
        )
        # No Streamlit calls inside summarize(), so an auto-refresh rerun
        # requested while it runs waits for it instead of cutting it off.
        with st.spinner("Reading the stats and key plays…"):
            try:
                _res = game_summary.summarize(_ctx, _providers[_prov])
            except game_summary.SummaryError as e:
                st.error(str(e))
            else:
                _summary = {
                    "game_id": game_id, "text": _res.text, "model": _res.model,
                    "anchor": cursor_anchor(timeline, cursor_idx),
                    "as_of": f"{game_summary.game_status(revealed)[0]}, play {cursor_idx + 1}",
                }
                st.session_state["_summary"] = _summary
    if _summary:
        with st.container(border=True):
            st.markdown(_summary["text"])
            _note = f"As of {_summary['as_of']} · {_summary['model']}"
            if _summary["anchor"] != cursor_anchor(timeline, cursor_idx):
                _note += " · you've moved since, press Update summary to catch it up"
            st.caption(_note)

# ---------- Scoring timeline ----------
st.subheader("Scoring timeline")
if not revealed.empty:
    _stl_df = scoring_timeline(revealed, home, away)
    if not _stl_df.empty:
        _stl_cols = ["Q", "Clock", "Team", "Type", "Score"]
        if not hide_descriptions:
            _stl_cols = ["Q", "Clock", "Team", "Type", "Score", "Description"]
        st.dataframe(
            _style_scoring_timeline(_stl_df[_stl_cols]),
            hide_index=True,
            width='stretch',
        )
    else:
        st.caption("No scores yet.")
else:
    st.caption("No plays revealed yet.")

# ---------- Recent plays (with pagination) ----------
st.subheader("Recent plays")
st.markdown(
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
        hide_index=True, width = 'stretch',
        column_config=_PLAYS_COL_CFG,
        row_height=_plays_row_height(hide_descriptions),
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
        _drive_raw = revealed[
            (revealed["drive"] == _cur_drive) & (revealed["play_type"] != "kickoff")
        ].copy()

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
            hide_index=True, width='stretch',
            column_config=_PLAYS_COL_CFG,
            row_height=_plays_row_height(hide_descriptions),
        )
    else:
        st.caption("No drive data available.")
else:
    st.caption("No plays revealed yet.")

# ---------- Explosive plays ----------
if not revealed.empty:
    _exp_df = explosive_plays(revealed)
    _exp_count = len(_exp_df)
    with st.expander(f"Explosive plays ({_exp_count})", expanded=False):
        if not _exp_df.empty:
            exp_display = _build_plays_df(_exp_df, hide_descriptions, reverse=False)
            st.dataframe(
                exp_display.style.apply(_style_plays, axis=1),
                hide_index=True,
                width='stretch',
                column_config=_PLAYS_COL_CFG,
                row_height=_plays_row_height(hide_descriptions),
            )
        else:
            st.caption("No explosive plays yet.")

# ---------- Drive chart ----------
st.subheader("Drive chart")
if not revealed.empty:
    _dc = drive_chart(revealed)
    _spots = drive_field_spots(revealed)
    if not _spots.empty:
        st.plotly_chart(
            drive_field_figure(_spots, home, away, team_colors, team_logos, team_nicks),
            width='stretch', config={"displayModeBar": False},
        )
        st.caption(f"{home} drives left → right · {away} drives right → left · "
                   "● start · ▶ end · one segment per play, ticks mark each snap · "
                   "latest drive on top")
    if not _dc.empty:
        with st.expander(f"Drive table ({len(_dc)})", expanded=False):
            st.dataframe(
                _style_drive_chart(_dc),
                hide_index=True,
                width='stretch',
            )
    else:
        st.caption("No drive data available.")
else:
    st.caption("No plays revealed yet.")

# ---------- Team stats ----------
st.subheader("Team stats")
stat_df = team_stats(revealed, home, away)
_baselines = load_stat_baselines(int(season), league)
st.dataframe(style_stat_table(stat_df, away, home, _baselines), width='stretch')
st.caption(f"Colors show percentile vs {BASELINE_LABEL[league]} · green = top · red = bottom")

# ---------- Situational success rates ----------
st.subheader("Situational success rates")
st.caption(f"Colors show percentile vs {BASELINE_LABEL[league]} · green = top · red = bottom")
sr_df, _sit_counts = situational_success_rate(revealed, home, away)
_sit_baselines = load_situational_baselines(int(season), league)
st.dataframe(_style_sr_table(sr_df, _sit_baselines, _sit_counts), width='stretch')

# ---------- Player leaders ----------
if not hide_leaders:
    st.subheader("Player leaders")
    col_a, col_h = st.columns(2)
    for col, team in [(col_a, away), (col_h, home)]:
        with col:
            st.markdown(f"**{team}**")
            _pass_df = top_players(revealed, team, "passing", drop=missing_stats)
            st.caption("Passing")
            if not _pass_df.empty:
                st.dataframe(_pass_df, hide_index=True, width='stretch',
                             column_config={
                                 "EPA/play": st.column_config.NumberColumn(format="%.2f"),
                                 "SR%": st.column_config.NumberColumn(format="%.1f%%"),
                                 "aDOT": st.column_config.NumberColumn(format="%.1f"),
                             })
            else:
                st.caption("No data yet")
            _rush_df = top_players(revealed, team, "rushing", 4, drop=missing_stats)
            st.caption("Rushing")
            if not _rush_df.empty:
                st.dataframe(_rush_df, hide_index=True, width='stretch',
                             column_config={
                                 "EPA/play": st.column_config.NumberColumn(format="%.2f"),
                                 "SR%": st.column_config.NumberColumn(format="%.1f%%"),
                             })
            else:
                st.caption("No data yet")
            _recv_df = top_players(revealed, team, "receiving", 8, drop=missing_stats)
            st.caption("Receiving")
            if not _recv_df.empty:
                st.dataframe(_recv_df, hide_index=True, width='stretch',
                             column_config={
                                 "aDOT": st.column_config.NumberColumn(format="%.1f"),
                                 "EPA/play": st.column_config.NumberColumn(format="%.2f"),
                             })
            else:
                st.caption("No data yet")
            _def_df = top_defenders(revealed, team, 10, drop=missing_stats)
            st.caption("Defense")
            if not _def_df.empty:
                st.dataframe(_def_df, hide_index=True, width='stretch',
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
# x is each play's place on the play timeline, so overtime plots after
# regulation (and untimed college OT gets room) instead of on top of Q4.
_el_min = pd.Series(timeline.iloc[: len(revealed)].to_numpy() / 60.0, index=revealed.index)
if not hide_wp:
    st.subheader("Momentum")
    if not revealed.empty:
        _mom_scrimmage = revealed[
            (revealed["pass_attempt"].fillna(0) == 1) |
            (revealed["rush_attempt"].fillna(0) == 1)
        ].copy()
        if not _mom_scrimmage.empty:
            _mom_scrimmage["_elapsed_min"] = _el_min.loc[_mom_scrimmage.index]
            _mom_color_map = {
                home: team_colors.get(home, "#1f77b4"),
                away: team_colors.get(away, "#ff7f0e"),
            }
            _mom_rows = []
            for _mom_team in [away, home]:
                _td = _mom_scrimmage[_mom_scrimmage["posteam"] == _mom_team].copy()
                if _td.empty:
                    continue
                _td["_rolling_epa"] = (
                    _td["epa"].fillna(0).rolling(5, center=True, min_periods=1).mean()
                )
                for _, _row in _td.iterrows():
                    _mom_rows.append({
                        "elapsed_min": _row["_elapsed_min"],
                        "team": _mom_team,
                        "rolling_epa": _row["_rolling_epa"],
                    })
            if _mom_rows:
                _mom_df = pd.DataFrame(_mom_rows)
                fig_mom = px.line(
                    _mom_df, x="elapsed_min", y="rolling_epa", color="team",
                    color_discrete_map=_mom_color_map,
                    labels={
                        "elapsed_min": "Game minutes elapsed",
                        "rolling_epa": "EPA/play (5-play rolling avg)",
                    },
                )
                fig_mom.update_yaxes(range=[-2, 2])
                _mom_x_cap = max(elapsed_s / 60.0, 1.0)
                fig_mom.update_xaxes(range=[0, _mom_x_cap])
                fig_mom.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                _mom_score_mask = (
                    (revealed["touchdown"].fillna(0) == 1) |
                    (revealed["field_goal_result"] == "made") |
                    (revealed["safety"].fillna(0) == 1)
                )
                _mom_scores = revealed[_mom_score_mask].copy()
                _mom_scores["_elapsed"] = _el_min.loc[_mom_scores.index]
                _mom_scores["_team"] = _mom_scores.apply(
                    lambda r: r["defteam"] if r["safety"] == 1 else r["posteam"], axis=1
                )
                _mom_scores["_type"] = _mom_scores.apply(
                    lambda r: "Safety" if r["safety"] == 1
                    else ("FG" if r["field_goal_result"] == "made" else "TD"),
                    axis=1,
                )
                for _, _se in _mom_scores.iterrows():
                    _se_color = team_colors.get(str(_se["_team"]), "#333333")
                    fig_mom.add_vline(
                        x=float(_se["_elapsed"]),
                        line_dash="dot",
                        line_color=_se_color,
                        opacity=0.7,
                        annotation_text=f"{_se['_team']} {_se['_type']}",
                        annotation_position="top",
                        annotation_font_size=10,
                    )
                fig_mom.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), legend_title_text="")
                st.plotly_chart(fig_mom, width='stretch')
        else:
            st.caption("Not enough plays for momentum chart.")

    st.subheader("Win probability")
    if not revealed.empty:
        wp_df = revealed[["home_wp", "away_wp"]].assign(elapsed=_el_min).dropna()
        wp_long = wp_df.melt(id_vars="elapsed", value_vars=["home_wp", "away_wp"],
                             var_name="team", value_name="wp")
        wp_long["team"] = wp_long["team"].map({"home_wp": home, "away_wp": away})
        _color_map = {home: team_colors.get(home, "#1f77b4"),
                      away: team_colors.get(away, "#ff7f0e")}
        fig = px.line(wp_long, x="elapsed", y="wp", color="team",
                      color_discrete_map=_color_map,
                      labels={"elapsed": "Game minutes elapsed", "wp": "Win probability"})
        fig.update_yaxes(range=[0, 1])
        # Lock x-axis to elapsed-so-far. Otherwise Plotly auto-fits to the data
        # and the right edge silently moves forward as you scrub — and if the
        # game went to OT, an axis ending at 75+ minutes is itself a spoiler.
        x_cap = max(elapsed_s / 60.0, 1.0)
        fig.update_xaxes(range=[0, x_cap])
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
        for _x in wp_crossings(revealed, _el_min):
            fig.add_vline(x=_x, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), legend_title_text="")
        st.plotly_chart(fig, width='stretch')

# ---------- Top plays by win probability added ----------
st.subheader("Top plays by win probability added")
if not revealed.empty:
    _top_wpa = top_plays_wpa(revealed, home, away)
    if not _top_wpa.empty:
        display_cols = ["Q", "Clock", "Off", "D&D", "Score", "For", "WPA"]
        if not hide_descriptions:
            display_cols = ["Q", "Clock", "Off", "D&D", "Score", "For", "Description", "WPA"]
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
        st.dataframe(styled_wpa, hide_index=False, width='stretch')
        st.caption("WPA = change in home-team win probability · green = home benefits · red = away benefits")
    else:
        st.caption("No win probability data available.")
else:
    st.caption("No plays revealed yet.")

# ---------- Auto-advance to next play ----------
# One play per refresh tick.
if auto and cursor_max < last_idx:
    _at_edge = cursor_idx >= cursor_max
    st.session_state["_cursor_max"] = cursor_max + 1
    if _at_edge:
        st.session_state["_cursor_idx"] = cursor_max + 1
    st.session_state["_cursor_anchor"] = (
        cursor_anchor(timeline, int(st.session_state["_cursor_idx"])),
        cursor_anchor(timeline, cursor_max + 1))

