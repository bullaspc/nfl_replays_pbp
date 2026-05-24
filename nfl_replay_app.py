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
import nfl_data_py as nfl
import plotly.express as px
from datetime import datetime, timedelta, time

st.set_page_config(page_title="NFL Replay Boxscore", layout="wide", page_icon="🏈")

# ---------- Data loading ----------
@st.cache_data(ttl=120)  # refresh every 2 minutes; pbp updates aren't instant anyway
def load_pbp(season: int) -> pd.DataFrame:
    """Load play-by-play for a given season."""
    cols = [
        "game_id", "season", "week", "game_date", "home_team", "away_team",
        "posteam", "defteam", "qtr", "time", "game_seconds_remaining",
        "play_id", "desc", "play_type", "yards_gained",
        "touchdown", "field_goal_result", "extra_point_result",
        "two_point_conv_result", "safety", "sp",
        "total_home_score", "total_away_score",
        "home_wp", "away_wp", "epa",
        "passer_player_name", "rusher_player_name", "receiver_player_name",
        "passing_yards", "rushing_yards", "receiving_yards",
        "pass_touchdown", "rush_touchdown",
        "interception", "fumble_lost",
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
    # So a play has happened iff (3600 - game_seconds_remaining) <= elapsed_game_s
    played_at = 3600 - pbp_game["game_seconds_remaining"].fillna(3600)
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


def team_stats(revealed: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    """Basic offensive team stats so far."""
    rows = []
    for team in [away, home]:
        td = revealed[revealed["posteam"] == team]
        passing = int(td["passing_yards"].fillna(0).sum())
        rushing = int(td["rushing_yards"].fillna(0).sum())
        plays = int(((td["play_type"].isin(["pass", "run"]))).sum())
        tos = int(td["interception"].fillna(0).sum() + td["fumble_lost"].fillna(0).sum())
        epa = round(float(td["epa"].fillna(0).sum()), 2)
        rows.append({"Team": team, "Plays": plays, "Pass Yds": passing,
                     "Rush Yds": rushing, "Total Yds": passing + rushing,
                     "Turnovers": tos, "Total EPA": epa})
    return pd.DataFrame(rows)


def top_players(revealed: pd.DataFrame, team: str, kind: str, n: int = 3) -> pd.DataFrame:
    """Leaders for a team so far."""
    td = revealed[revealed["posteam"] == team]
    if kind == "passing":
        g = td.groupby("passer_player_name", as_index=False).agg(
            Yds=("passing_yards", "sum"), TD=("pass_touchdown", "sum"),
            INT=("interception", "sum"))
        g = g.rename(columns={"passer_player_name": "Player"})
    elif kind == "rushing":
        g = td.groupby("rusher_player_name", as_index=False).agg(
            Yds=("rushing_yards", "sum"), TD=("rush_touchdown", "sum"))
        g = g.rename(columns={"rusher_player_name": "Player"})
    else:  # receiving
        g = td.groupby("receiver_player_name", as_index=False).agg(
            Yds=("receiving_yards", "sum"), TD=("pass_touchdown", "sum"))
        g = g.rename(columns={"receiver_player_name": "Player"})
    g = g.dropna(subset=["Player"])
    g[g.select_dtypes("number").columns] = g.select_dtypes("number").astype(int)
    return g.sort_values("Yds", ascending=False).head(n)


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
    game_label = st.selectbox("Game", games["label"].tolist())
    game_id = games.loc[games["label"] == game_label, "game_id"].iloc[0]

    st.divider()
    st.subheader("Your viewing")
    mode = st.radio(
        "How do you want to set your position?",
        ["I started the broadcast at...",
         "I'm X minutes into the broadcast",
         "Jump to a specific game clock"],
    )

    if mode == "I started the broadcast at...":
        start_date = st.date_input("Start date", value=datetime.now().date())
        start_time = st.time_input("Start time (your local clock)", value=time(20, 0))
        start_dt = datetime.combine(start_date, start_time)
        viewing_minutes = max((datetime.now() - start_dt).total_seconds() / 60.0, 0.0)
        elapsed_s = elapsed_game_seconds(viewing_minutes)

    elif mode == "I'm X minutes into the broadcast":
        viewing_minutes = st.number_input("Minutes into broadcast",
                                          min_value=0.0, max_value=240.0,
                                          value=30.0, step=1.0)
        elapsed_s = elapsed_game_seconds(viewing_minutes)

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
        elapsed_s = completed_qtrs * 900 + (900 - remaining_in_qtr)
        # Derive an approximate viewing-minutes equivalent just for the caption
        viewing_minutes = (elapsed_s / 3600.0) * 190.0

    st.caption(f"⏱ Game time elapsed: **{elapsed_s/60:.1f} min** "
               f"(≈ {viewing_minutes:.1f} broadcast min)")
    auto = st.checkbox("Auto-refresh every 30s", value=False)
    if auto:
        st.experimental_rerun if False else None  # placeholder; see note below

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
shown_plays = len(revealed)
qtr_now = int(revealed["qtr"].iloc[-1]) if not revealed.empty else 1
game_clock = "—"
if not revealed.empty and pd.notna(revealed["time"].iloc[-1]):
    game_clock = str(revealed["time"].iloc[-1])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Quarter", f"Q{qtr_now}" if qtr_now <= 4 else "OT")
c2.metric("Game clock (last play)", game_clock)
c3.metric("Plays revealed", f"{shown_plays}")  # no denominator — would leak OT/blowout
c4.metric("Game time elapsed", f"{elapsed_s/60:.1f} min")

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
st.dataframe(boxscore(revealed, home, away), hide_index=True, use_container_width=True)

# ---------- Team stats ----------
st.subheader("Team stats")
st.dataframe(team_stats(revealed, home, away), hide_index=True, use_container_width=True)

# ---------- Player leaders ----------
if not hide_leaders:
    st.subheader("Player leaders")
    col_a, col_h = st.columns(2)
    for col, team in [(col_a, away), (col_h, home)]:
        with col:
            st.markdown(f"**{team}**")
            st.caption("Passing"); st.dataframe(top_players(revealed, team, "passing"),
                                                hide_index=True, use_container_width=True)
            st.caption("Rushing"); st.dataframe(top_players(revealed, team, "rushing"),
                                                hide_index=True, use_container_width=True)
            st.caption("Receiving"); st.dataframe(top_players(revealed, team, "receiving"),
                                                  hide_index=True, use_container_width=True)

# ---------- Win probability chart ----------
if not hide_wp:
    st.subheader("Win probability")
    if not revealed.empty:
        wp_df = revealed[["game_seconds_remaining", "home_wp", "away_wp"]].dropna()
        wp_df = wp_df.assign(elapsed=(3600 - wp_df["game_seconds_remaining"]) / 60.0)
        wp_long = wp_df.melt(id_vars="elapsed", value_vars=["home_wp", "away_wp"],
                             var_name="team", value_name="wp")
        wp_long["team"] = wp_long["team"].map({"home_wp": home, "away_wp": away})
        fig = px.line(wp_long, x="elapsed", y="wp", color="team",
                      labels={"elapsed": "Game minutes elapsed", "wp": "Win probability"})
        fig.update_yaxes(range=[0, 1])
        # Lock x-axis to elapsed-so-far. Otherwise Plotly auto-fits to the data
        # and the right edge silently moves forward as you scrub — and if the
        # game went to OT, an axis ending at 75+ minutes is itself a spoiler.
        x_cap = max(elapsed_s / 60.0, 1.0)
        fig.update_xaxes(range=[0, x_cap])
        st.plotly_chart(fig, use_container_width=True)

# ---------- Recent plays ----------
st.subheader("Recent plays")
if hide_descriptions:
    recent = revealed.tail(15)[["qtr", "time", "posteam", "yards_gained", "epa"]]
    recent.columns = ["Q", "Clock", "Off", "Yds", "EPA"]
else:
    recent = revealed.tail(15)[["qtr", "time", "posteam", "desc", "yards_gained", "epa"]]
    recent.columns = ["Q", "Clock", "Off", "Description", "Yds", "EPA"]
st.dataframe(recent.iloc[::-1], hide_index=True, use_container_width=True)

# ---------- Auto-refresh ----------
if auto:
    import time as _t
    _t.sleep(30)
    st.rerun()
