"""
The Matchup page: two teams' season stats set against each other for a game
not played yet, and on request a short AI preview of it (Kimi by default).
The matchups set each team's offense against the other's defense; head to
head compares like with like, offense with offense and defense with defense.

The numbers come from finished games only, so there's nothing to spoil.
No data loading or stat logic of its own: leagues.py loads and caches,
replay_core computes, views styles and game_summary writes the preview.
"""

import os
import time
from datetime import date

import pandas as pd
import streamlit as st

import game_summary
import leagues
from replay_core import (EDGE_EVEN, MATCHUP_ROWS, PLAYER_MIN_PLAYS, PROJ_SHRINK, SEASON_MIN_PLAYS,
                         SITUATION_MIN_PLAYS, PlayFilter, head_to_head, matchup_table,
                         season_player_distributions, season_qualified, situational_matchup)
from views import (MATCHUP_TINTS, logo_img, stat_table, style_defense_table, style_head_to_head,
                   style_matchup_table, style_player_table, style_situational_matchup)

st.set_page_config(page_title="Matchup", layout="wide", page_icon="🆚")

SIDES = {"Offense": "posteam", "Defense (allowed)": "defteam"}
# Each team's leaders shown, and sent to the preview: (kind, how many).
LEADERS = [("passing", 2), ("rushing", 3), ("receiving", 4), ("defenders", 5)]
_TODAY = date.today()
# A season is named for the year it starts in; before September it's last year's.
LATEST_SEASON = _TODAY.year if _TODAY.month >= 9 else _TODAY.year - 1
_PREVIEW_POLL_SECS = 3


def _setting(name: str) -> str | None:
    """A setting from .streamlit/secrets.toml, else the environment (as the
    replay page reads its summary keys)."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # no secrets.toml
        pass
    return os.environ.get(name) or None


# ---------- Sidebar: the game ----------
with st.sidebar:
    st.header("Matchup")
    league = leagues.LEAGUES[st.radio("League", list(leagues.LEAGUES), horizontal=True,
                                      key="_mu_league")]
    season = int(st.number_input("Season", min_value=league.first_season,
                                 max_value=LATEST_SEASON, value=LATEST_SEASON, key="_mu_season"))

stamp = league.stamp()
try:
    with st.spinner(f"Loading the {season} season…"):
        upcoming = league.upcoming(season, stamp)
        pbp_all = league.season_pbp(season, stamp)
        ranked_teams = league.season_teams(season, stamp)
except leagues.GameLoadError as e:
    st.title(f"🆚 {league.name} Matchup")
    st.warning(f"{e} A matchup needs games played this season.")
    st.stop()

# Only games between two teams with season numbers (not an FCS opponent).
if not upcoming.empty:
    upcoming = upcoming[upcoming["home_team"].isin(ranked_teams.index)
                        & upcoming["away_team"].isin(ranked_teams.index)]
if upcoming.empty:
    st.title(f"🆚 {league.name} Matchup")
    st.info(f"No games left to play in the {season} season.")
    st.stop()

with st.sidebar:
    week = st.selectbox("Week", list(dict.fromkeys(upcoming["week_label"])),
                        key=f"_mu_week_{league.name}_{season}")
    games = upcoming[upcoming["week_label"] == week]
    group = "All"
    if league.group_label:
        # Every upcoming week's groups, so the options don't change with the week.
        groups = sorted({g for gs in upcoming["groups"] for g in gs if isinstance(g, str) and g})
        group = st.selectbox(league.group_label, ["All"] + groups, key=f"_mu_group_{league.name}")
        if group != "All":
            games = games[games["groups"].apply(lambda gs: group in gs)]
    if games.empty:
        st.caption(f"No {group} games in {week}.")
        st.stop()
    # By game id: a college label carries AP ranks, which a new poll changes.
    labels = dict(zip(games["game_id"], games["label"]))
    game_id = st.selectbox("Game", list(labels), format_func=labels.get,
                           key=f"_mu_game_{league.name}_{season}_{week}_{group}")
    game = games[games["game_id"] == game_id].iloc[0]

# ---------- Sidebar: the season numbers ----------
reg_weeks = sorted(int(w) for w in pbp_all.loc[~pbp_all["postseason"], "week"].dropna().unique())
has_post = bool(pbp_all["postseason"].any())
with st.sidebar:
    st.subheader("Season numbers")
    adjust = st.radio("Stats", list(leagues.SEASON_ADJUST), index=1, horizontal=True,
                      key="_mu_adjust",
                      help="Opponent-adjusted: each offense's numbers less the average strength "
                           "of the defenses it faced, and each defense's less that of the "
                           "offenses it faced, as on the Season stats page.")
    weeks = (reg_weeks[0], reg_weeks[-1]) if reg_weeks else (0, 0)
    if len(reg_weeks) > 1:
        weeks = st.select_slider("Regular-season weeks", options=reg_weeks, value=weeks,
                                 # A newly published week resets it to the full range.
                                 key=f"_mu_weeks_{league.name}_{season}_{reg_weeks[-1]}")
    garbage = None
    if st.checkbox("Exclude garbage time", key="_mu_garbage",
                   help="Drops plays where the offense's win probability before the snap is "
                        "below the cutoff or above 100% minus it."):
        garbage = st.slider("Win probability cutoff (%)", 1, 25, 10, key="_mu_garbage_wp") / 100
    st.caption(f"Data as of {stamp}")
play_filter = PlayFilter((), garbage, None)

with st.spinner("Computing season stats…"):
    # "Both": any postseason games played so far count too.
    tables = leagues.season_tables(league, season, stamp, weeks, "Both", adjust, play_filter)
off_table = season_qualified(tables["offense"], SEASON_MIN_PLAYS)
def_table = season_qualified(tables["defense"], SEASON_MIN_PLAYS)
away, home = game["away_team"], game["home_team"]
short = [t for t in (away, home) if t not in off_table.index or t not in def_table.index]
if short:
    st.title(f"🆚 {away} @ {home}")
    st.info(f"{' and '.join(short)} {'has' if len(short) == 1 else 'have'} fewer than "
            f"{SEASON_MIN_PLAYS} plays in this selection.")
    st.stop()

_, logos, _ = league.team_meta(season, stamp)
tints = dict(zip((away, home), MATCHUP_TINTS))
tint_note = f"orange when {away} has it, blue when {home} does"
# The college feed has no QB hits: its QB Hit % would only repeat Sack %.
rows = [r for r in MATCHUP_ROWS if not (r == "QB Hit %" and "QB Hits" in tables["missing"])]
matchups = {away: matchup_table(off_table, def_table, away, home, rows),
            home: matchup_table(off_table, def_table, home, away, rows)}
sel_plays = leagues.season_plays(league, season, stamp, weeks, "Both", play_filter)
# Ranked among the same teams as the main tables.
sit_off = tables["situational_team"][tables["situational_team"].index.isin(off_table.index)]
sit_def = tables["situational_defense"][tables["situational_defense"].index.isin(def_table.index)]
situational = {away: situational_matchup(sit_off, sit_def, away, home, sel_plays),
               home: situational_matchup(sit_off, sit_def, home, away, sel_plays)}

# What the numbers cover, for the captions and the preview.
n_teams = len(off_table)
week_note = (f"week {weeks[0]}" if weeks[0] == weeks[1] else f"weeks {weeks[0]}–{weeks[1]}")
sel_note = f"{season} {week_note}" + (" + postseason" if has_post else "")
if garbage is not None:
    sel_note += f" · no garbage time (WP {garbage:.0%}–{1 - garbage:.0%})"
sel_note += " · opponent-adjusted" if leagues.SEASON_ADJUST[adjust] else " · as played"

kick = pd.Timestamp(game["game_date"])
when = f"{week} · {kick:%a %b} {kick.day}" + (" · neutral site" if game.get("neutral") else "")
spread, total = game.get("home_spread"), game.get("total")
line = ""
if pd.notna(spread):
    line = ("Pick'em" if spread == 0 else
            f"{home if spread > 0 else away} favored by {abs(spread):g}")
if pd.notna(total):
    line += (", " if line else "") + f"total {total:g}"

# ---------- Header ----------
st.markdown(
    "<div style='display:flex;align-items:center;justify-content:center;"
    "gap:clamp(8px,3vw,24px);margin:0.25rem 0 0.5rem'>"
    f"{logo_img(away, logos)}"
    f"<div style='font-size:clamp(1.3rem,6vw,2.4rem);font-weight:700;white-space:nowrap'>"
    f"{away} @ {home}</div>"
    f"{logo_img(home, logos)}</div>",
    unsafe_allow_html=True)
st.markdown(f"<div style='text-align:center'>{when}" + (f" · {line}" if line else "") + "</div>",
            unsafe_allow_html=True)
st.caption(f"Season numbers: {sel_note} · places among the {n_teams} teams with "
           f"{SEASON_MIN_PLAYS}+ plays · counts and yards are per game.")

# ---------- AI preview ----------
# As the replay's summary: the click is recorded by on_click (so a run cut
# short still counts), the first run to get here with it set starts the job
# on a worker thread, and a fragment polls for the result. The page has no
# timer of its own, but any widget change reruns it while the model writes.
st.subheader("🧠 Matchup preview")
preview_key = (league.name, season, game["game_id"])
_providers = game_summary.configured_providers(_setting)
_job = st.session_state.get("_mu_job")
if st.session_state.get("_mu_requested") not in (None, preview_key) or (
        _job is not None and "_mu_requested" in st.session_state):
    del st.session_state["_mu_requested"]
if not _providers:
    st.caption("Add `MOONSHOT_API_KEY` (Kimi) or `ANTHROPIC_API_KEY` (Claude) to "
               "`.streamlit/secrets.toml` or the environment to get an AI preview of the matchup.")
else:
    _btn, _pick = st.columns([3, 2])
    _prov = next(iter(_providers))
    if len(_providers) > 1:
        _prov = _pick.selectbox(
            "Model", list(_providers), label_visibility="collapsed", key="_mu_provider",
            format_func=lambda k: f"{_providers[k].label} · {_providers[k].model}")
    _has = preview_key in st.session_state.get("_mu_previews", {})
    _btn.button("Writing the preview…" if _job is not None and _job["key"] == preview_key
                else ("Update preview" if _has else "Write a matchup preview"),
                disabled=_job is not None,
                on_click=lambda k=preview_key: st.session_state.update(_mu_requested=k),
                help="Sends this page's tables to the model; nothing else.")
    if st.session_state.get("_mu_requested") == preview_key:
        _leaders = {}
        for kind, n in LEADERS:
            df = tables[kind]
            for team in (away, home):
                if not df.empty:
                    _leaders[(team, "defense" if kind == "defenders" else kind)] = (
                        df[df["Team"] == team].drop(columns="Team").head(n))
        _ctx = game_summary.MatchupContext(
            league=league.key, away=away, home=home, when=when.replace(" · ", ", "), line=line,
            selection=sel_note.replace(" · ", ", "), n_teams=n_teams,
            matchups=matchups, situational=situational, leaders=_leaders)
        _future, _progress = game_summary.submit_preview(_ctx, _providers[_prov])
        st.session_state["_mu_job"] = {"future": _future, "progress": _progress,
                                       "key": preview_key, "started": time.time(),
                                       "selection": sel_note}
        st.session_state.pop("_mu_error", None)
        del st.session_state["_mu_requested"]
        st.rerun()  # redraw the button as busy, and start the fragment polling


@st.fragment(run_every=_PREVIEW_POLL_SECS if "_mu_job" in st.session_state else None)
def _preview_result() -> None:
    """The preview, or that it's being written, or why it failed."""
    job = st.session_state.get("_mu_job")
    if job and job["future"].done():
        try:
            res = job["future"].result()
        except Exception as e:  # SummaryError, or anything the worker didn't expect
            st.session_state["_mu_error"] = {
                "key": job["key"],
                "text": str(e) if isinstance(e, game_summary.SummaryError) else f"The preview failed: {e}"}
        else:
            st.session_state["_mu_previews"] = {
                **st.session_state.get("_mu_previews", {}),
                job["key"]: {"text": res.text, "model": res.model, "selection": job["selection"]}}
        del st.session_state["_mu_job"]
        st.rerun()  # a full run: the button frees up, and polling stops
    if job and job["key"] == preview_key:
        with st.status(f"🧠 {job['progress'].status}… ({time.time() - job['started']:.0f}s)",
                       state="running"):
            st.caption("The preview shows up here when it's ready.")
    err = st.session_state.get("_mu_error")
    if err and err["key"] == preview_key:
        st.error(err["text"])
    preview = st.session_state.get("_mu_previews", {}).get(preview_key)
    if preview:
        with st.container(border=True):
            st.markdown(preview["text"])
            note = f"{preview['model']} · {preview['selection']}"
            if preview["selection"] != sel_note:
                note += " · the numbers below have changed since: press Update preview to redo it"
            st.caption(note)


_preview_result()

# ---------- Matchups ----------
st.subheader("Matchups")
for col, (off, def_) in zip(st.columns(2), ((away, home), (home, away))):
    with col:
        st.markdown(f"**When {off} has the ball** · {off} offense vs {def_} defense")
        stat_table(style_matchup_table(matchups[off], off, def_, tints))
st.caption(
    "Each value carries its place among the teams (for a defense, allowing less is better and "
    "forcing more turnovers and sacks is better), colored green = best, red = worst. **Edge** is "
    f"who ranks better and by how many percentile points, tinted {tint_note} "
    f"(\"Even\": within {EDGE_EVEN}). **Proj.** is a guess for this game: the league average plus "
    f"{PROJ_SHRINK:.0%} of the offense's and the defense's distances from it (games have landed "
    "about that far back toward the average). Plays and aDoT are style, so they get no edge.")

# ---------- Head to head ----------
st.subheader("Head to head")
side_label = st.segmented_control("Side", list(SIDES), default="Offense", key="_mu_side",
                                  label_visibility="collapsed") or "Offense"
side = SIDES[side_label]
h2h = head_to_head(off_table if side == "posteam" else def_table, away, home, side, rows)
stat_table(style_head_to_head(h2h, away, home, tints))
st.caption(f"{side_label}: each team's value and place. **Δ** is {away} − {home} in the stat's "
           f"units (rates in points of percent), tinted for the team that ranks better: "
           f"{tint_note} (none within {EDGE_EVEN} percentile points).")

# ---------- Down and distance ----------
st.subheader("By down and distance")
for off, def_ in ((away, home), (home, away)):  # stacked: seven columns each
    st.markdown(f"**When {off} has the ball** · O = {off} offense, D = {def_} defense allowed")
    stat_table(style_situational_matchup(situational[off], off, def_, tints))
st.caption("Success rate (EPA > 0), the season's plays in brackets. Early downs are 1st–2nd, late 3rd–4th; "
           "short is 1–3 yards to go, medium 4–6, long 7+. Passes include sacks but not spikes, "
           f"runs leave out kneels. No edge (—) unless both sides have {SITUATION_MIN_PLAYS}+ "
           "plays there.")

# ---------- Key players ----------
st.subheader("Key players")
games_max = int(tables["offense"]["Games"].max())
player_min = {k: max(1, round(PLAYER_MIN_PLAYS[k] * games_max * tables["play_share"]))
              for k in PLAYER_MIN_PLAYS}
player_dists = season_player_distributions({k: tables[k] for k in PLAYER_MIN_PLAYS}, player_min)
for col, team in zip(st.columns(2), (away, home)):
    with col:
        st.markdown(f"**{team}**")
        for kind, n in LEADERS:
            df = tables[kind]
            df = df[df["Team"] == team].drop(columns="Team").head(n) if not df.empty else df
            st.caption("Defense" if kind == "defenders" else kind.capitalize())
            if df.empty:
                st.caption("No data")
            elif kind == "defenders":
                stat_table(style_defense_table(df), hide_index=True)
            else:
                stat_table(style_player_table(df, kind, player_dists, player_min[kind]),
                           hide_index=True)
st.caption("Season totals over the same games. SR%, aDOT and EPA/play are colored against the "
           "season's players with enough plays, as on the Season stats page.")

# ---------- How to read this ----------
st.divider()
st.subheader("How to read this")
st.markdown(f"""
**Data.** The Season stats page's numbers, over the finished games picked in the sidebar (and
any postseason games played so far); the game previewed hasn't been played. Teams with fewer than
{SEASON_MIN_PLAYS} plays in the selection aren't ranked. Opponent-adjusted (the default here) takes
each team's schedule out, as on the Season stats page.

**Matchups** set one team's offense against the other's defense. Every value is ranked among the
{n_teams} teams and turned into a percentile (100 = best). **Edge** is the offense's percentile less the
defense's: "{away} 30" means {away}'s unit ranks 30 percentile points better than the one it faces.
**Proj.** starts from the league average and adds how far the offense and the defense each sit from
it (the same additive view of offense and defense the opponent adjustment uses), kept at
{PROJ_SHRINK:.0%}: in NFL backtests (`tools/validate_matchup.py`: each regular-season week from week 4
on in 2024 and 2025, from the weeks before it), game results landed a quarter to two thirds of the
way from the average to the full sum. Shrunk, Proj. came closer to the game than the offense's own number or the league
average for EPA/play, pass EPA/play and 1st-down rate in both seasons, and about as close for the
rest. Turnovers, 3rd-down rate, sack rate and the run game carry over least from one game to the
next: an edge there says more about the season than about the game. The projected EPA/play margin
picked 61–67% of winners opponent-adjusted (62–67% as played); the betting favorite 63–74%. It
knows nothing of injuries or weather.

**Head to head** compares like with like. Δ is {away} − {home}; whether more is better depends on
the stat, so the tint, not the sign, says who's better.

**Down and distance** splits the success rates by down and distance. Small samples are noisy, so
a cell needs {SITUATION_MIN_PLAYS} plays on both sides for an edge.

**Matchup preview.** On request, the matchup tables, down and distance and key players on this
page are sent to the model (Kimi by default; Claude if that's the key set) with instructions to
use only those numbers. It isn't told about injuries or news, and it doesn't predict a score.
""")
