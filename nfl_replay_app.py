"""
Football Tape-Delay Replay Boxscore
-----------------------------------
A spoiler-free way to follow an NFL or college football game on tape delay.

You tell the app:
  1. Which game you're watching
  2. When you started watching (i.e. your personal "kickoff")
The app reveals plays, score, and stats only up to your current viewing point.

This file is the page: sidebar, play cursor and sections. Where the games
come from is in leagues.py, the stats in replay_core.py, and table styling,
figures and the time bar in views.py.

Run with:
    pip install -r requirements.txt
    streamlit run nfl_replay_app.py
"""

import contextlib
import os
import time

import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

import game_summary
import leagues
import scoreboard
from replay_core import (
    EXPLOSIVE_RUN_YDS, GAME_SECONDS, LINE_LOWER_IS_BETTER, boxscore, cursor_anchor, cursor_from_anchor, cursor_from_elapsed,
    down_distance, drive_chart, drive_field_spots, explosive_plays, field_pos_label, filter_plays,
    PLAY_KINDS, PLAYER_MIN_PLAYS, TENDENCY_DOWNS, has_air_yards, line_play_stats, pass_tendencies, period_label, play_timeline,
    run_tendencies, scoring_timeline, situational_success_rate, stat_percentiles, team_stats, top_defenders, top_players, top_plays_wpa, wp_crossings,
)
from views import (
    build_plays_df, drive_field_figure, hex_or_none, keep_screen_awake,
    logo_img, pass_tendency_figure, plays_table, run_tendency_figure, smap, style_drive_chart,
    stat_table, style_defense_table, style_line_table, style_player_table, style_scoring_timeline,
    style_sr_table, style_stat_table, ticker, ticker_pick, time_bar_html,
)

st.set_page_config(page_title="Football Replay Boxscore", layout="wide", page_icon="🏈")

FULL_GAME = "Full game"


def _setting(name: str) -> str | None:
    """A setting from .streamlit/secrets.toml, else the environment."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # no secrets.toml
        pass
    return os.environ.get(name) or None


# ---------- Saved position ----------
# The selection and the play cursor are mirrored into the URL (_save_url()), so
# a phone that loses its session comes back where it was: a tab left in the
# background longer than the server keeps a disconnected session, or a tab the
# browser unloaded and reloads. The URL is read once per session and its values
# only seed the widget defaults, which must stay put for the session: an
# unkeyed widget whose default changes starts over.
if "_url" not in st.session_state:
    st.session_state["_url"] = st.query_params.to_dict()
    st.session_state["_url_restore"] = True
_url = st.session_state["_url"]


def _url_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        return min(max(int(_url[name]), lo), hi)
    except (KeyError, ValueError):
        return default


def _url_anchor(name: str) -> tuple[float, int]:
    """A cursor_anchor() saved as "elapsed:rank"; KeyError/ValueError if absent."""
    e, rank = _url[name].split(":")
    return (float(e), int(rank))


# ---------- UI ----------
with st.sidebar:
    st.header("Setup")
    _league_names = list(leagues.LEAGUES)
    league_name = st.radio("League", _league_names, horizontal=True,
                           index=_league_names.index(_url["league"])
                           if _url.get("league") in _league_names else 0)
    league = leagues.LEAGUES[league_name]

st.title(league.title)
st.caption("Spoiler-free boxscore that unlocks as your broadcast progresses.")

with st.sidebar:
    season = int(st.number_input("Season", min_value=league.first_season,
                                 max_value=2026, step=1,
                                 value=_url_int("season", 2026, league.first_season, 2026)))
    stamp = league.stamp()
    with st.spinner("Loading play-by-play..."):
        try:
            games = league.games(season, stamp)
        except leagues.GameLoadError as e:
            st.error(str(e))
            st.stop()
    if games.empty:
        st.warning(f"No games found for the {season} season yet.")
        st.stop()
    # A game clicked in the ticker (see _ticker() below) is selected the way
    # the Game box selects one: through `_game_id`, which the Week and Game
    # boxes default to. The conference filter goes back to All if it hides it.
    _pick = st.session_state.pop("_ticker_pick", None)
    if _pick is not None:
        _pick_row = league.ticker_game(games, _pick)
        if _pick_row is None:
            st.toast(f"{_pick.away.abbr} @ {_pick.home.abbr} isn't in the game list yet.")
        else:
            st.session_state["_game_id"] = _pick_row["game_id"]
            if league.group_label and st.session_state.get("_group", "All") not in (
                    "All", *_pick_row["groups"]):
                st.session_state["_group"] = "All"
    week_labels = list(dict.fromkeys(games["week_label"]))
    # Default to the week and game last on screen (from the URL on a new
    # session). A game's label changes when it goes from live to official,
    # which starts the Game box over, so it reselects the game by id.
    _last_game = st.session_state.get("_game_id", _url.get("game"))
    _last_row = games[games["game_id"] == _last_game]
    week = st.selectbox("Week", week_labels,
                        index=week_labels.index(_last_row["week_label"].iloc[0])
                        if not _last_row.empty else len(week_labels) - 1)
    games = games[games["week_label"] == week]
    if league.group_label:
        # College has 50+ games a Saturday: narrow by conference.
        groups = sorted({g for gs in games["groups"] for g in gs if isinstance(g, str) and g})
        group = st.selectbox(league.group_label, ["All"] + groups, key="_group")
        if group != "All":
            games = games[games["groups"].apply(lambda gs: group in gs)]
    _game_ids = games["game_id"].tolist()
    game_label = st.selectbox("Game", games["label"].tolist(),
                              index=_game_ids.index(_last_game)
                              if _last_game in _game_ids else 0)
    game_row = games.loc[games["label"] == game_label].iloc[0]
    game_id = game_row["game_id"]
    st.session_state["_game_id"] = game_id
    source = game_row["source"]

    try:
        loaded = league.load_game(game_row, season, stamp)
    except leagues.GameLoadError as e:
        st.error(str(e))
        st.stop()
    pbp_game, live_state = loaded.pbp, loaded.live_state
    for note in loaded.notes:
        st.caption(note)
    if loaded.warning:
        st.warning(loaded.warning)
    if pbp_game.empty:
        st.info("This game hasn't started yet — no plays to show.")
        st.stop()
    home = pbp_game["home_team"].iloc[0]
    away = pbp_game["away_team"].iloc[0]
    timeline = play_timeline(pbp_game)
    last_idx = len(pbp_game) - 1
    team_colors, team_logos, team_nicks = league.team_meta(season, stamp)

    st.divider()
    st.subheader("Your viewing")

    # Watching the game as it airs: follow the feed instead of a typed clock.
    # Plays are revealed only once they've sat in the feed for `live_delay`
    # seconds, so a stream that runs behind ESPN never shows a play first.
    follow_live = False
    live_delay = 0
    if source == "live" and live_state != "post":
        follow_live = st.toggle(
            "🔴 Follow live", value=_url.get("follow") == "1",
            help="Keep up with the game as it airs. New plays appear on their own, "
                 "each held back by the delay below so the feed can't run ahead "
                 "of your TV or stream.")
        if follow_live:
            live_delay = st.slider(
                "Hold each new play back (seconds)", min_value=0, max_value=180,
                value=_url_int("delay", 45, 0, 180), step=5,
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
        _qtrs = ["Q1", "Q2", "Q3", "Q4", "OT", FULL_GAME]
        qtr_pick = st.selectbox("Quarter", _qtrs,
                                index=_qtrs.index(_url["q"]) if _url.get("q") in _qtrs else 0,
                                help=f"{FULL_GAME} unlocks every play: for a game you've "
                                     "already watched.")
        if qtr_pick == FULL_GAME:
            st.caption("Every play is unlocked.")
            clock_str = "0:00"
        elif league.untimed_ot and qtr_pick == "OT":
            st.caption("Overtime has no clock here: you start at the end of "
                       "regulation. Step through it with ▶ Next play.")
            clock_str = "15:00"
        else:
            clock_str = st.text_input("Game clock remaining (MM:SS)",
                                      value=_url.get("clock", "15:00"),
                                      help="Time left on the in-quarter clock, e.g. 7:32")
    try:
        mm, ss = clock_str.strip().split(":")
        remaining_in_qtr = int(mm) * 60 + int(ss)
        assert 0 <= remaining_in_qtr <= 15 * 60
    except Exception:
        st.warning("Use MM:SS format, e.g. 7:32. Defaulting to 15:00.")
        remaining_in_qtr = 15 * 60

    full_game = qtr_pick == FULL_GAME
    qtr_idx = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "OT": 5, FULL_GAME: 5}[qtr_pick]
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
    collapse_plays = st.checkbox("Collapse play lists", value=False,
                                 help="Recent plays shows only the last play and the "
                                      "current drive's plays fold into an expander.")

    st.divider()
    st.subheader("🙈 Spoiler shield")
    safety_margin = st.slider(
        "Stay this many seconds *behind* my entered position",
        min_value=0, max_value=120, value=_url_int("margin", 15, 0, 120), step=5,
        help="Buffer against accidentally revealing the next play. "
             "15s is enough to absorb small clock drift.",
    )
    hide_wp = st.checkbox("Hide win probability chart", value=False,
                          help="The WP curve telegraphs upcoming swings.")
    hide_descriptions = st.checkbox("Hide play descriptions", value=False,
                                    help="Show yards/EPA only — descriptions can foreshadow what's about to happen on screen.")
    hide_leaders = st.checkbox("Hide player leaders", value=False,
                               help="A QB suddenly at 4 TDs hints something just happened.")
    hide_ticker = st.checkbox("Hide other games' scores", value=False,
                              help="The ticker of the league's other live games above the "
                                   "score. The game you're watching is never in it.")
    blur_until_ready = st.checkbox("Blur everything until I click 'Reveal'", value=False)

    # A card clicked in the ticker (see _ticker() below) comes in here, outside
    # the ticker's fragment, so the click reruns the whole page (a click inside
    # a fragment is lost if the page is running). The game is selected at the
    # top of the next run, before the Week and Game boxes are drawn.
    _clicked = ticker_pick()
    if _clicked:
        _picked = next((g for g in league.ticker(game_row) if g.event_id == _clicked), None)
        if _picked is not None and _picked.season and _picked.season != season:
            st.toast(f"{_picked.away.abbr} @ {_picked.home.abbr} is in the {_picked.season} "
                     f"season: set Season to {_picked.season} to open it.")
        elif _picked is not None:
            st.session_state["_ticker_pick"] = _picked
            st.rerun()

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
baseline_cursor = last_idx if full_game else min(baseline_cursor, last_idx)

# Full game re-seeds as live plays arrive, so it keeps showing all of them.
_seed_key = (game_id, qtr_pick, clock_str, safety_margin, last_idx if full_game else None)
# A new session for a game in the URL picks up its saved cursor. The anchors
# are by game time, so they hold however many live plays arrived meanwhile,
# or after the switch from live to official, as in the frame change below.
if st.session_state.pop("_url_restore", False) and _url.get("game") == game_id:
    try:
        _r_anchor = (_url_anchor("at"), _url_anchor("max"))
    except (KeyError, ValueError):
        _r_anchor = None
    if _r_anchor:
        st.session_state["_cursor_key"] = _seed_key
        st.session_state["_cursor_frame"] = None  # carry the anchors over below
        st.session_state["_cursor_anchor"] = _r_anchor
        if follow_live:
            # When the plays past the saved edge arrived isn't known: start
            # their delay now rather than count them as aired, and don't jump
            # to the edge as switching Follow live on does.
            _r_max = cursor_from_anchor(timeline, _r_anchor[1])
            st.session_state.setdefault("_live_seen", {})[game_id] = (
                [0.0] * (_r_max + 1) + [time.time()] * (last_idx - _r_max))
            st.session_state["_live_follow_on"] = game_id
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


def _save_url() -> None:
    """Write the selection and the cursor anchors to the URL (see "Saved
    position" at the top). Only when they change: each write reaches the
    browser."""
    params = {"league": league_name, "season": str(season), "game": str(game_id),
              "margin": str(safety_margin)}
    if follow_live:
        params |= {"follow": "1", "delay": str(live_delay)}
    else:
        params |= {"q": qtr_pick, "clock": clock_str}
    for name, (e, rank) in zip(("at", "max"), st.session_state["_cursor_anchor"]):
        params[name] = f"{e!r}:{rank}"
    if st.query_params.to_dict() != params:
        st.query_params.from_dict(params)


_save_url()

revealed = pbp_game.iloc[: cursor_idx + 1].copy()
# Player-leader columns this game's data can't fill (e.g. tackles in older
# college games), dropped rather than shown as zeros.
missing_stats = league.missing_stats(revealed)
# Line play's QB Hit % drops out where the data has no QB hits (college).
_line_drop = frozenset({"QB Hit %"}) if "QB Hits" in missing_stats else frozenset()
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


# ---------- Other games ticker ----------
# The league's other games from ESPN's scoreboard, ESPN-ticker style, while any
# of them is on: score, clock, who has the ball and where. The game being
# watched is never in it (its live score would spoil it). A fragment, so it
# keeps itself current on a replay too, where nothing else reruns the page,
# without rerunning the page; fast only while a game is on or about to start.
# Clicking a card opens that game (ticker_pick() in the sidebar).
_TICKER_SECS, _TICKER_IDLE_SECS = 30, 300
_ticker_every = None
if not hide_ticker:
    _ticker_every = (_TICKER_SECS if scoreboard.busy(league.ticker(game_row), pd.Timestamp.now(tz="UTC"))
                     else _TICKER_IDLE_SECS)


@st.fragment(run_every=_ticker_every)
def _ticker() -> None:
    others = league.ticker(game_row)
    if any(g.state == "in" for g in others):
        ticker(others)


if not hide_ticker:
    _ticker()

# One flex row that never wraps, so the scoreboard stays a single compact line
# on phones (st.columns would stack the logos and score vertically there).
st.markdown(
    "<div style='display:flex;align-items:center;justify-content:center;"
    "gap:clamp(8px,3vw,24px);margin:0.25rem 0 0.5rem'>"
    f"{logo_img(away, team_logos)}"
    f"<div style='font-size:clamp(1.3rem,6vw,2.4rem);font-weight:700;white-space:nowrap'>"
    f"{away} {away_score} — {home_score} {home}</div>"
    f"{logo_img(home, team_logos)}</div>",
    unsafe_allow_html=True)

st.markdown(
    f"<div style='text-align:center;font-size:1.1rem;opacity:0.8;margin-bottom:0.5rem'>"
    f"{period_label(qtr_now)}"
    # College overtime has no clock to show.
    f"{'' if qtr_now >= 5 and game_clock == '—' else ' · ' + game_clock}</div>",
    unsafe_allow_html=True)

# ---------- Play-by-play time bar ----------
_bar_fill = "#4c78a8"
_in_ot = False
if cursor_idx >= 0:
    _cur_row = pbp_game.iloc[cursor_idx]
    _bar_fill = (hex_or_none(team_colors.get(_cur_row["posteam"]))
                 or hex_or_none(team_colors.get(home))
                 or _bar_fill)
    _in_ot = pd.notna(_cur_row["qtr"]) and int(_cur_row["qtr"]) >= 5
st.markdown(time_bar_html(elapsed_s / 3600.0, _bar_fill, _in_ot), unsafe_allow_html=True)


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

# How far the ▶ buttons may go on a game still being played:
# - Follow live: only to what has aired (`cursor_max`). Past it are plays the
#   delay is holding back, and a tap would show them early.
# - Otherwise ▶▶ Next drive waits for the drive to end. The feed's last drive
#   is still going, so its "end" would be ESPN's newest play, maybe several
#   plays past the broadcast. ▶ Next play still steps one play at a time.
_step_max = cursor_max if follow_live else last_idx
_drive_to = min(_drive_end(cursor_idx), _step_max) if cursor_idx < _step_max else cursor_idx
_drive_open = (source == "live" and live_state != "post" and not follow_live
               and _drive_to == last_idx)

_b1, _b2, _b3, _b4, _b5 = st.columns(5)
if _b1.button("⏮ Start", key="bar_start", width='stretch', disabled=cursor_idx <= 0):
    _move_cursor(0)
if _b2.button("◀ Prev", key="bar_prev", width='stretch', disabled=cursor_idx <= 0):
    _move_cursor(cursor_idx - 1)
if _b3.button("▶ Next play", key="bar_next", width='stretch', disabled=cursor_idx >= _step_max):
    _move_cursor(cursor_idx + 1, unlock=True)
if _b4.button("▶▶ Next drive", key="bar_drive", width='stretch',
              disabled=cursor_idx >= _step_max or _drive_open,
              help="This drive is still being played: step through it with ▶ Next play."
                   if _drive_open else None):
    _move_cursor(_drive_to, unlock=True)
if _b5.button("⏭ Catch up", key="bar_catchup", width='stretch', disabled=cursor_idx >= cursor_max):
    _move_cursor(cursor_max)

if cursor_idx >= 0:
    _r = pbp_game.iloc[cursor_idx]
    _q = int(_r["qtr"]) if pd.notna(_r["qtr"]) else 0
    _bits = [period_label(_q) if _q >= 1 else "—"]
    if pd.notna(_r["time"]):
        _bits.append(str(_r["time"]))
    _dd, _fp = down_distance(_r), field_pos_label(_r)
    if _dd and _fp:
        _bits.append(f"{_dd} at {_fp}")
    elif _dd or _fp:
        _bits.append(_dd or _fp)
    # "of N unlocked", never "of N total" — the play count alone would leak
    # whether the game ran long.
    _bits.append(f"play {cursor_idx + 1} of {cursor_max + 1} unlocked")
    if _q <= 4:  # overtime's place on the timeline isn't real game minutes
        _bits.append(f"⏱ {elapsed_s / 60:.1f} game min "
                     f"(≈ {elapsed_s / GAME_SECONDS * league.broadcast_minutes:.0f} broadcast min)")
    st.caption(" · ".join(_bits))
else:
    st.caption("Kickoff — nothing revealed yet. "
               + ("Plays show up here as they air." if follow_live
                  else "Press **▶ Next play** to begin."))

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
stat_table(boxscore(revealed, home, away), hide_index=True)

# ---------- AI game summary ----------
# An agent reads the same stats as the sections below, built from `revealed`
# only, and explains the score so far. It runs only when asked, and the result
# is kept until the next request.
#
# A full rerun stops the running script and starts a new one at once
# (Streamlit's runner.fastReruns), and a live game reruns every 10-30 s while
# a summary takes a minute or more. So nothing here waits on one run:
# - the click is recorded by the button's on_click callback, which runs at the
#   start of the click's run, in `_summary_requested` (game id). The first run
#   to get this far with it set starts the job, so a click whose run is cut
#   short still counts;
# - the summary is written on a worker thread (game_summary.submit), kept in
#   `_summary_job` with where the viewer was;
# - the result is shown by a fragment that, while a job is out, reruns on its
#   own every few seconds to collect it. Fragment reruns never cancel a full
#   run (they queue behind it), so they can't starve the page the way polling
#   with full reruns can when runs are slow, and they work on replays, where
#   nothing else reruns the page.
# The button stays outside the fragment: a fragment click queued behind a slow
# full run is dropped when the next full rerun replaces it.
# State is written before what it replaces is cleared, so a run cut off in
# between leaves work to redo, never a lost click or result.
_SUMMARY_POLL_SECS = 3

st.subheader("🧠 Why the score is what it is")
_providers = game_summary.configured_providers(_setting)
_job = st.session_state.get("_summary_job")
# A request made on another game (the viewer switched since) or while a job
# is already out is dropped.
if st.session_state.get("_summary_requested") not in (None, game_id) or (
        _job is not None and "_summary_requested" in st.session_state):
    del st.session_state["_summary_requested"]
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
    _has_summary = (st.session_state.get("_summary") or {}).get("game_id") == game_id
    _sum_btn.button("Writing the summary…" if _job is not None and _job["game_id"] == game_id
                    else ("Update summary" if _has_summary else "Explain the score so far"),
                    disabled=_job is not None,
                    on_click=lambda gid=game_id: st.session_state.update(_summary_requested=gid),
                    help="Only the plays you've unlocked are sent to the model.")
    if st.session_state.get("_summary_requested") == game_id:
        _season_baselines = leagues.stat_baselines(league, season)
        _stats = team_stats(revealed, home, away)
        _sit, _sit_n = situational_success_rate(revealed, home, away)
        _line = line_play_stats(revealed, home, away, drop=_line_drop)
        _ctx = game_summary.GameContext(
            home=home, away=away, revealed=revealed,
            league=league.key,
            line_play=_line,
            line_pct=stat_percentiles(_line, leagues.line_baselines(league, season),
                                      LINE_LOWER_IS_BETTER),
            run_gaps=league.run_gaps,
            tendency_baselines=leagues.tendency_baselines(league, season),
            boxscore=boxscore(revealed, home, away),
            scoring=scoring_timeline(revealed, home, away),
            team_stats=_stats, team_pct=stat_percentiles(_stats, _season_baselines),
            situational=_sit, situational_counts=_sit_n,
            drives=drive_chart(revealed),
            top_wpa=top_plays_wpa(revealed, home, away),
            explosive=explosive_plays(revealed),
            leaders={
                **{(t, k): top_players(revealed, t, k, n, drop=missing_stats)
                   for t in (away, home)
                   for k, n in (("passing", 3), ("rushing", 4), ("receiving", 8))},
                **{(t, "defense"): top_defenders(revealed, t, 10, drop=missing_stats)
                   for t in (away, home)},
            },
        )
        _future, _progress = game_summary.submit(_ctx, _providers[_prov])
        st.session_state["_summary_job"] = {
            "future": _future, "progress": _progress,
            "game_id": game_id, "started": time.time(),
            "anchor": cursor_anchor(timeline, cursor_idx),
            "as_of": f"{game_summary.game_status(revealed)[0]}, play {cursor_idx + 1}",
        }
        st.session_state.pop("_summary_error", None)
        del st.session_state["_summary_requested"]
        st.rerun()  # redraw the button as busy, and start the fragment polling


@st.fragment(run_every=_SUMMARY_POLL_SECS if "_summary_job" in st.session_state else None)
def _summary_result() -> None:
    """The summary, or that it's being written, or why it failed."""
    job = st.session_state.get("_summary_job")
    if job and job["future"].done():
        try:
            res = job["future"].result()
        except Exception as e:  # SummaryError, or anything the worker didn't expect
            st.session_state["_summary_error"] = {
                "game_id": job["game_id"],
                "text": str(e) if isinstance(e, game_summary.SummaryError) else f"The summary failed: {e}"}
        else:
            st.session_state["_summary"] = {
                "game_id": job["game_id"], "text": res.text, "model": res.model,
                "anchor": job["anchor"], "as_of": job["as_of"]}
        del st.session_state["_summary_job"]
        st.rerun()  # a full run: the button frees up, and polling stops
    if job and job["game_id"] == game_id:
        # What the agent is doing right now, and the lookups it has done.
        p = job["progress"]
        with st.status(f"🧠 {p.status}… ({time.time() - job['started']:.0f}s)",
                       state="running", expanded=True):
            for step in p.steps:
                st.markdown(f"✓ {step}")
            st.caption("The summary shows up here when it's ready; you can keep watching meanwhile.")
    err = st.session_state.get("_summary_error")
    if err and err["game_id"] == game_id:
        st.error(err["text"])
    summary = st.session_state.get("_summary")
    # Show a summary only for this game and only up to the unlocked edge. Moving the
    # clock inputs back lowers that edge and hides it until you get there again.
    if summary and summary["game_id"] == game_id and summary["anchor"] <= cursor_anchor(timeline, cursor_max):
        with st.container(border=True):
            st.markdown(summary["text"])
            note = f"As of {summary['as_of']} · {summary['model']}"
            if summary["anchor"] != cursor_anchor(timeline, cursor_idx):
                note += " · you've moved since, press Update summary to catch it up"
            st.caption(note)


_summary_result()

# ---------- Scoring timeline ----------
st.subheader("Scoring timeline")
if not revealed.empty:
    _stl_df = scoring_timeline(revealed, home, away)
    if not _stl_df.empty:
        _stl_cols = ["Q", "Clock", "Team", "Type", "Score"]
        if not hide_descriptions:
            _stl_cols = ["Q", "Clock", "Team", "Type", "Score", "Description"]
        stat_table(style_scoring_timeline(_stl_df[_stl_cols]), hide_index=True)
    else:
        st.caption("No scores yet.")
else:
    st.caption("No plays revealed yet.")

# ---------- Recent plays (with pagination) ----------
st.subheader("Recent plays")
st.markdown(
    '<span style="background:#ffeeba;padding:2px 8px;border-radius:3px;margin-right:6px">3rd down</span>'
    '<span style="background:#ffd0a6;padding:2px 8px;border-radius:3px;margin-right:6px">4th-down attempt</span>'
    '<span style="background:#dc3545;color:#fff;padding:2px 8px;border-radius:3px;margin-right:6px">Red zone</span>'
    '<span style="margin-right:6px">🏈 Pass &nbsp; 🏃 Run</span>'
    '<span style="margin-right:6px">✅ Positive EPA</span>',
    unsafe_allow_html=True,
)
def _reset_recent_page():
    st.session_state.recent_plays_page = 1


_DOWN_NAMES = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
_f_down, _f_kind, _f_team = st.columns([4, 6, 3])
with _f_down:
    _sel_downs = st.pills("Down", list(_DOWN_NAMES), selection_mode="multi",
                          format_func=_DOWN_NAMES.get, key="plays_downs",
                          on_change=_reset_recent_page)
with _f_kind:
    _sel_kinds = st.pills("Play type", PLAY_KINDS, selection_mode="multi",
                          key="plays_kinds", on_change=_reset_recent_page)
with _f_team:
    # Keyed by game: the options are this game's teams.
    _sel_teams = st.pills("Offense", [away, home], selection_mode="multi",
                          key=f"plays_teams_{game_id}", on_change=_reset_recent_page)
_filtered = bool(_sel_downs or _sel_kinds or _sel_teams)
_matching = filter_plays(revealed, _sel_downs, _sel_kinds, _sel_teams)

if _matching.empty:
    st.caption("No plays match these filters yet." if _filtered and not revealed.empty
               else "No plays revealed yet.")
elif collapse_plays:
    plays_table(build_plays_df(_matching.iloc[[-1]], hide_descriptions, reverse=False))
else:
    _PAGE_SIZE = 15
    _all_rev = _matching.iloc[::-1].copy()
    _total_plays = len(_all_rev)
    _total_pages = max(1, (_total_plays + _PAGE_SIZE - 1) // _PAGE_SIZE)
    if "recent_plays_page" not in st.session_state:
        st.session_state.recent_plays_page = 1
    _page = min(int(st.session_state.recent_plays_page), _total_pages)

    _slice = _all_rev.iloc[(_page - 1) * _PAGE_SIZE : _page * _PAGE_SIZE]
    recent_df = build_plays_df(_slice, hide_descriptions, reverse=False)
    plays_table(recent_df)
    col_prev, col_info, col_next = st.columns([1, 4, 1])
    with col_prev:
        if st.button("◀ Prev", disabled=(_page <= 1), key="prev_plays"):
            st.session_state.recent_plays_page = _page - 1
            st.rerun()
    with col_info:
        _count = (f"{_total_plays} of {len(revealed)} plays" if _filtered
                  else f"{_total_plays} plays total")
        st.caption(f"Page {_page} of {_total_pages}  ({_count})")
    with col_next:
        if st.button("Next ▶", disabled=(_page >= _total_pages), key="next_plays"):
            st.session_state.recent_plays_page = _page + 1
            st.rerun()

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

        drive_df = build_plays_df(_drive_raw, hide_descriptions)
        with (st.expander(f"Drive plays ({len(drive_df)})", expanded=False)
              if collapse_plays else contextlib.nullcontext()):
            plays_table(drive_df)
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
            exp_display = build_plays_df(_exp_df, hide_descriptions, reverse=False)
            plays_table(exp_display)
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
            stat_table(style_drive_chart(_dc), hide_index=True)
    else:
        st.caption("No drive data available.")
else:
    st.caption("No plays revealed yet.")

# ---------- Team stats ----------
st.subheader("Team stats")
stat_df = team_stats(revealed, home, away)
_baselines = leagues.stat_baselines(league, season)
stat_table(style_stat_table(stat_df, away, home, _baselines))
st.caption(f"Colors show percentile vs {league.baseline_label} · green = top · red = bottom")

# ---------- Situational success rates ----------
st.subheader("Situational success rates")
st.caption(f"Colors show percentile vs {league.baseline_label} · green = top · red = bottom")
_sr_subtotals = st.toggle("Early/late subtotals", value=False,
                          help="Add rows totalling early downs (1st–2nd) and "
                               "late downs (3rd–4th) over every distance.")
sr_df, _sit_counts = situational_success_rate(revealed, home, away,
                                              subtotals=_sr_subtotals)
_sit_baselines = leagues.situational_baselines(league, season)
stat_table(style_sr_table(sr_df, _sit_baselines, _sit_counts))

# ---------- Line play ----------
st.subheader("Line play")
st.caption(f"Each team's offensive line; the other column is its defensive front's side. "
           f"Colors show percentile vs {league.baseline_label} · green = top · red = bottom")
line_df = line_play_stats(revealed, home, away, drop=_line_drop)
stat_table(style_line_table(line_df, leagues.line_baselines(league, season)))
st.caption(f"Dropbacks include sacks; QB hits include sacks. Designed runs leave out "
           f"kneels and scrambles: stuffed = no gain or a loss, explosive = {EXPLOSIVE_RUN_YDS}+ yards. "
           f"OL penalties: holding, false starts and ineligible man downfield on the "
           f"offense, per snap (pass, run or no-play).")

# ---------- Tendencies ----------
st.subheader("Tendencies")
_tend_downs = st.segmented_control("Downs", list(TENDENCY_DOWNS), default="All downs",
                                   key="_tend_downs", label_visibility="collapsed") or "All downs"
_tend_base = leagues.tendency_baselines(league, season).get(_tend_downs, {})
_tend_air = has_air_yards(revealed)
_tend_any = False
col_a, col_h = st.columns(2)
for col, team in [(col_a, away), (col_h, home)]:
    with col:
        st.markdown(f"**{team}**")
        _runs = run_tendencies(revealed, team, league.run_gaps, TENDENCY_DOWNS[_tend_downs])
        _passes = pass_tendencies(revealed, team, TENDENCY_DOWNS[_tend_downs], _tend_air)
        st.caption(f"Runs · {_runs['Plays'].sum()} with a direction")
        if _runs["Plays"].sum():
            st.plotly_chart(run_tendency_figure(_runs, _tend_base.get("runs")), width='stretch',
                            config={"displayModeBar": False})
        st.caption(f"Passes · {_passes['Plays'].sum()} with a direction")
        if _passes["Plays"].sum():
            st.plotly_chart(pass_tendency_figure(_passes, _tend_base.get("air" if _tend_air else "text")), width='stretch',
                            config={"displayModeBar": False})
        _tend_any = _tend_any or bool(_runs["Plays"].sum() or _passes["Plays"].sum())
if _tend_any:
    _gap_note = " (LE = left end … Mid … RE = right end)" if league.run_gaps else ""
    st.caption("From the offense's side: left is the quarterback's left. Runs: share of designed "
               f"runs (no kneels or scrambles), under each gap{_gap_note} its carries · yards per "
               "carry; the grey dash is the league's share. Passes: attempts · share, completions · "
               "EPA/play, by air yards past the line of scrimmage"
               + ("" if _tend_air else " (this feed has none, so by the play text's short/deep)")
               + "; sacks aren't thrown, so they're out. Plays whose text names no direction "
               "are left out. Colors are EPA/play (green above 0, red below). League = "
               f"{league.baseline_label}.")
elif not revealed.empty:
    st.caption("This game's play text gives no run or pass directions.")

# ---------- Player leaders ----------
if not hide_leaders:
    st.subheader("Player leaders")
    _player_baselines = leagues.player_baselines(league, season)
    st.caption(f"SR%, aDOT and EPA/play colors show percentile vs player games in the "
               f"{league.baseline_label} · green = top · red = bottom · only players with "
               f"{PLAYER_MIN_PLAYS['passing']}+ dropbacks, {PLAYER_MIN_PLAYS['rushing']}+ "
               f"carries or {PLAYER_MIN_PLAYS['receiving']}+ targets are colored")
    col_a, col_h = st.columns(2)
    for col, team in [(col_a, away), (col_h, home)]:
        with col:
            st.markdown(f"**{team}**")
            for _kind, _n in [("passing", 3), ("rushing", 4), ("receiving", 8)]:
                _lead_df = top_players(revealed, team, _kind, _n, drop=missing_stats)
                st.caption(_kind.capitalize())
                if not _lead_df.empty:
                    stat_table(style_player_table(_lead_df, _kind, _player_baselines),
                               hide_index=True)
                else:
                    st.caption("No data yet")
            _def_df = top_defenders(revealed, team, 10, drop=missing_stats)
            st.caption("Defense")
            if not _def_df.empty:
                stat_table(style_defense_table(_def_df), hide_index=True)
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

        styled_wpa = smap(_top_display.style, _wpa_color, subset=["WPA"])
        styled_wpa = styled_wpa.format({"WPA": "{:+.3f}"})
        styled_wpa = styled_wpa.set_properties(**{"text-align": "center"}).set_table_styles(
            [{"selector": "th", "props": [("text-align", "center"), ("font-weight", "bold")]}]
        )
        stat_table(styled_wpa)
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
    _save_url()

