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
from replay_core import (EDGE_EVEN, LIKELY_OUT, LINEUP_EFFECTS, MATCHUP_ROWS, PLAYER_MIN_PLAYS,
                         PROJ_SHRINK, SEASON_MIN_PLAYS, SITUATION_MIN_PLAYS, PlayFilter,
                         availability_summary, head_to_head, injury_report, lineup_adjustment,
                         lineup_change, lineup_groups, matchup_table, season_player_distributions,
                         season_qualified, situational_matchup, status_by_name, team_snaps)
from views import (MATCHUP_TINTS, logo_img, stat_table, status_badge, style_availability_summary,
                   style_defense_table, style_head_to_head, style_injury_report, style_lineup_change,
                   style_lineup_groups, style_matchup_table, style_player_table,
                   style_situational_matchup)

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

# Who's hurt: each team's injury report for the game's week, with each listed
# player's share of his team's snaps this season, and the statuses by the
# names the season tables use (for the Key players badges).
avail = league.availability(season)
game_week = int(game["week"])
player_ids = leagues.season_player_ids(league, season, stamp)
reports, report_units, statuses = {}, {}, {}
for team in (away, home):
    reports[team] = injury_report(avail.injuries, team, game_week, team_snaps(avail.snaps, team),
                                  avail.starters)
    report_units[team] = availability_summary(reports[team])
    statuses[team] = status_by_name(reports[team], team, player_ids, avail.short_names)
injury_note = (f"the week {game_week} reports, as of {avail.as_of or 'the latest build'}"
               if not avail.injuries.empty else "")

# How each lineup differs from the one behind its season numbers (its games in
# this selection): the week's report and roster, so also players on a reserve
# list, who drop off the injury report, and starters coming back.
_games = leagues.season_plays(league, season, stamp, weeks, "Both")
lineups, lineup_net = {}, {}
for team in (away, home):
    _ids = _games.loc[(_games["home_team"] == team) | (_games["away_team"] == team), "game_id"].unique()
    lineups[team] = lineup_change(avail.snaps, avail.injuries, avail.roster, team, game_week, _ids)
    lineup_net[team] = lineup_groups(lineups[team])
# Proj. moved by the players gone (replay_core.LINEUP_EFFECTS), where there are
# snap counts and injury reports to tell who they are.
has_lineup = not avail.snaps.empty and not avail.injuries.empty
adjustments = ({away: lineup_adjustment(lineups[away], lineups[home]),
                home: lineup_adjustment(lineups[home], lineups[away])} if has_lineup else {})
matchups = {away: matchup_table(off_table, def_table, away, home, rows, adjustments.get(away)),
            home: matchup_table(off_table, def_table, home, away, rows, adjustments.get(home))}
# Before the final report there are no game statuses, so only roster moves count.
no_status = [t for t in (away, home) if has_lineup and not (reports[t]["Status"] != "").any()]
# What the injury report doesn't show: gone off the roster, or back.
LINEUP_SHOWN = 0.1  # full-time players: smaller changes count in the totals only


def _off_report(team: str) -> pd.DataFrame:
    ch = lineups[team]
    return ch[~ch["Why"].isin(LIKELY_OUT) & (ch["Change"].abs() >= LINEUP_SHOWN)]


def _with_status(df: pd.DataFrame, team: str, badge: bool) -> pd.DataFrame:
    """A leaders table with a Status column after Player, if any of its
    players has a game status."""
    status = [statuses[team].get(p, "") for p in df["Player"]] if "Player" in df.columns else []
    if not any(status):
        return df
    df = df.copy()
    df.insert(1, "Status", [status_badge(s) for s in status] if badge else status)
    return df

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
                    _leaders[(team, "defense" if kind == "defenders" else kind)] = _with_status(
                        df[df["Team"] == team].drop(columns="Team").head(n), team, badge=False)
        _shown = ["Player", "Pos", "Status", "Snaps", "Starter", "Practice", "Injury"]
        _ctx = game_summary.MatchupContext(
            league=league.key, away=away, home=home, when=when.replace(" · ", ", "), line=line,
            selection=sel_note.replace(" · ", ", "), n_teams=n_teams,
            matchups=matchups, situational=situational, leaders=_leaders,
            injuries={t: reports[t][_shown] for t in (away, home)}, injury_units=report_units,
            injury_note=injury_note,
            lineup_groups=lineup_net if not avail.snaps.empty else {},
            lineup_players={t: _off_report(t).drop(columns="Unit").round(2) for t in (away, home)})
        _future, _progress = game_summary.submit_preview(_ctx, _providers[_prov])
        st.session_state["_mu_job"] = {"future": _future, "progress": _progress,
                                       "key": preview_key, "started": time.time(),
                                       "selection": sel_note, "injuries_as_of": avail.as_of}
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
                job["key"]: {"text": res.text, "model": res.model, "selection": job["selection"],
                             "injuries_as_of": job["injuries_as_of"]}}
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
            elif preview.get("injuries_as_of", "") != avail.as_of:
                note += " · the injury report has been updated since: press Update preview to redo it"
            st.caption(note)


_preview_result()

# ---------- Availability ----------
st.subheader("Availability")
if not avail.source:
    st.caption(f"No injury reports or snap counts for {league.name} in this app's data.")
elif avail.injuries.empty:
    st.caption("The injury reports couldn't be loaded.")
else:
    for col, team in zip(st.columns(2), (away, home)):
        with col:
            st.markdown(f"**{team}** · week {game_week} injury report")
            if reports[team].empty:
                st.caption("Not filed yet: teams file the first one about three days before the game.")
            else:
                if not report_units[team].empty:
                    stat_table(style_availability_summary(report_units[team]), hide_index=True)
                stat_table(style_injury_report(reports[team]), hide_index=True)
            if not avail.snaps.empty:
                st.markdown(f"**{team}** · lineup vs the season numbers")
                stat_table(style_lineup_groups(lineup_net[team]), hide_index=True)
                if not _off_report(team).empty:
                    stat_table(style_lineup_change(_off_report(team)), hide_index=True)
    st.caption(
        "**Status** is the game status on the final report, the day before the game (Wednesday for "
        "a Thursday game): Out, Doubtful (rarely plays) or Questionable (may play); before it there's "
        "only practice participation. **Snaps** is the player's share of his team's snaps this "
        "season in the phase he plays most (O offense, D defense, ST special teams), **Starter** "
        "his slot on the latest depth chart. **Snaps out** adds up the out-or-doubtful players' "
        "shares of the unit's phase: 1.0 is one full-time player. **Lineup vs the season "
        "numbers** is how far each group is from the players behind the numbers above, in "
        "full-time players: one out now takes away his share of those games' snaps (**Was**; "
        "half, if he missed half of them), one back adds the share he plays when he does "
        "(**Now**) less what he had. Gone counts the report's Out and Doubtful and the roster's "
        "reserve lists, releases and trades (players on injured reserve aren't on the report); "
        f"the table lists what the report doesn't show, {LINEUP_SHOWN:.1f}+ players' worth. "
        f"From {avail.source}; reports as of {avail.as_of or 'the latest build'}.")

# ---------- Matchups ----------
st.subheader("Matchups")
_epa = LINEUP_EFFECTS["EPA/play"]
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
    "about that far back toward the average). Plays and aDoT are style, so they get no edge."
    + (f" **w/ lineup** moves Proj. ({', '.join(LINEUP_EFFECTS)}) by the players gone from both "
       "lineups (Availability, above), by what such losses have done to games since 2016: about "
       f"{abs(_epa['QB']):.2f} EPA/play less for each full-time QB gone, {abs(_epa['OL']):.2f} for "
       f"each lineman or skill player, and {_epa['Front']:.2f} more for each starter the defense "
       "is missing. Players back aren't counted: they've made no measurable difference."
       if has_lineup else "")
    + (f" {' and '.join(no_status)} {'has' if len(no_status) == 1 else 'have'} no game statuses "
       "on the report yet (they come on the final one, the day before the game), so w/ lineup "
       "counts only roster moves." if no_status else ""))

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
                stat_table(style_defense_table(_with_status(df, team, badge=True)), hide_index=True)
            else:
                stat_table(style_player_table(_with_status(df, team, badge=True), kind,
                                              player_dists, player_min[kind]),
                           hide_index=True)
st.caption("Season totals over the same games. SR%, aDOT and EPA/play are colored against the "
           "season's players with enough plays, as on the Season stats page. **Status** is the "
           "player's game status on this week's injury report, where he has one.")

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
knows nothing of weather, and of injuries only through **w/ lineup**.

**Head to head** compares like with like. Δ is {away} − {home}; whether more is better depends on
the stat, so the tint, not the sign, says who's better.

**Down and distance** splits the success rates by down and distance. Small samples are noisy, so
a cell needs {SITUATION_MIN_PLAYS} plays on both sides for an edge.

**Availability** is each team's injury report for the game's week, the NFL's own as nflverse
publishes it, with each listed player's share of his team's snaps this season (PFR's snap counts)
and his slot on the latest depth chart. The season numbers above include the games he played, so a
starter who is out takes some of his unit's strength with him. **Lineup vs the season numbers**
measures that against the players behind the numbers, not against a full-strength team: a starter
out since week 2 has been missing from most of them already, so he takes away little.

**w/ lineup** is what the players gone have done to games before. `tools/validate_injuries.py` set
every opponent-adjusted Proj. from week 4 on, 2016–2025 (4,194 offense-games), against the game, and
fitted the miss on how many full-time players each group was missing: the offense's QB, its line,
its skill players, the defense's front seven and secondary. It doesn't need the replacements' own
numbers (often there are none): an effect is how far results moved, on average, with a group that
short, whoever filled in. A starting QB gone has cost his offense about {abs(_epa['QB']):.2f}
EPA/play, each other full-time player about {abs(_epa['OL']):.2f}, and each full-time player a
defense is missing has given the offense about {_epa['Front']:.2f}. Tested a season at a time on
effects fitted on the other seasons, it came closer to the game than Proj. for EPA/play (RMSE
0.1902 → 0.1875; 0.2015 → 0.1888 in the 471 games with a QB change), pass EPA/play (0.3096 →
0.3061) and pass success rate (0.0962 → 0.0950), and the projected EPA/play margin picked 63.6% of
winners instead of 62.4% (the betting favorite: 67.8%). The run game and sack rate didn't move, so
they aren't adjusted.
Players coming back aren't counted either: on their own, they made no measurable difference (a
starting QB back: −0.02 ± 0.04 EPA/play against Proj.). The adjustment is small next to how far
single games stray from any projection.

**Matchup preview.** On request, the matchup tables, down and distance, key players and the injury
reports on this page are sent to the model (Kimi by default; Claude if that's the key set) with
instructions to use only those numbers and reports. It isn't told about news, and it doesn't
predict a score.
""")
