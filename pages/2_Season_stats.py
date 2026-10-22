"""
The Season stats page: the replay's team and player stats over a whole season
(or some of its weeks), for every season a league has published. Teams are
ranked against each other: counts are per game, and each cell is colored by
its percentile among the season's teams (FBS teams for college).

Unlike the replay this page shows results; there is no play cursor here.
No data loading or stat logic of its own: leagues.py loads and caches,
replay_core computes, views styles.
"""

from datetime import date

import pandas as pd
import streamlit as st

import leagues
from replay_core import (
    ADJUST_MAX_PASSES, EXPLOSIVE_RUN_YDS, LINE_ROWS, PLAYER_MIN_PLAYS, PLAYER_VOLUME_COL,
    OT_QTR, SEASON_MIN_PLAYS, SHRINK_GAMES, SR_SITUATIONS, STUFF_MAX_YDS, TENDENCY_DOWNS, VENUES, PlayFilter,
    has_air_yards, venue_plays,
    pass_tendencies, run_tendencies, season_lower_is_better, season_player_distributions,
    season_qualified, season_rank_label, season_ranks, situational_distributions_from,
    situational_table,
)
from views import (
    pass_tendency_figure, run_tendency_figure, stat_table, style_defense_table,
    style_player_table, style_season_table, style_sr_table, style_team_season,
)

st.set_page_config(page_title="Season Stats", layout="wide", page_icon="📊")

QUARTERS = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4", OT_QTR: "OT"}

# The team table's column groups (replay_core.SEASON_COLS is "All").
TEAM_VIEWS = {
    "Efficiency": ["Games", "EPA/play", "Pass EPA/play", "Rush EPA/play", "Pass SR", "Rush SR",
                   "CMP%", "aDoT", "1st Down %", "3rd Down %", "RZ TD%", "Turnovers"],
    "Volume": ["Games", "Plays", "Pass Plays", "Rush Plays", "Rush+Comp", "Pass Yds", "Rush Yds",
               "Total Yds", "Turnovers", "Penalties", "Penalty Yds", "TOP"],
    "Line play": ["Games", *LINE_ROWS],
    "All": None,
}
SIDES = {"Offense": "posteam", "Defense (allowed)": "defteam"}
PLAYER_KINDS = {"Passing": "passing", "Rushing": "rushing", "Receiving": "receiving",
                "Defense": "defenders"}
_TODAY = date.today()
# A season is named for the year it starts in; before September it's last year's.
LATEST_SEASON = _TODAY.year if _TODAY.month >= 9 else _TODAY.year - 1

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Season")
    league = leagues.LEAGUES[st.radio("League", list(leagues.LEAGUES), horizontal=True,
                                      key="_ss_league")]
    season = int(st.number_input("Season", min_value=league.first_season,
                                 max_value=LATEST_SEASON, value=LATEST_SEASON, key="_ss_season"))

st.title(f"📊 {league.name} Season Stats · {season}")
st.caption("The replay's team and player stats over a season. This page shows results: "
           "it isn't spoiler-safe.")

stamp = league.stamp()
try:
    with st.spinner(f"Loading the {season} season…"):
        pbp_all = league.season_pbp(season, stamp)
        ranked_teams = league.season_teams(season, stamp)
except leagues.GameLoadError as e:
    st.warning(str(e))
    st.stop()

reg_weeks = sorted(int(w) for w in pbp_all.loc[~pbp_all["postseason"], "week"].dropna().unique())
has_post = bool(pbp_all["postseason"].any())

with st.sidebar:
    parts = [p for p, (reg, post) in leagues.SEASON_PARTS.items()
             if (not reg or reg_weeks) and (not post or has_post)]
    part = st.radio("Games", parts, index=parts.index("Regular season") if "Regular season" in parts
                    else 0, horizontal=True, key="_ss_part")
    weeks = (reg_weeks[0], reg_weeks[-1]) if reg_weeks else (0, 0)
    if leagues.SEASON_PARTS[part][0] and len(reg_weeks) > 1:
        weeks = st.select_slider("Regular-season weeks", options=reg_weeks, value=weeks,
                                 # A newly published week resets it to the full range.
                                 key=f"_ss_weeks_{league.name}_{season}_{reg_weeks[-1]}")
    groups = sorted(g for g in ranked_teams.unique() if g)
    picked = st.multiselect(league.season_group_label, groups, placeholder="All",
                            key=f"_ss_groups_{league.name}")
    adjust = st.radio("Stats", list(leagues.SEASON_ADJUST), horizontal=True, key="_ss_adjust",
                      help="Opponent-adjusted: each offense's numbers less the average strength "
                           "of the defenses it faced, and each defense's less that of the "
                           "offenses it faced, iterated until they settle (like SRS). Counts of "
                           "plays, penalties, aDOT and defenders' stats stay as played.")
    st.subheader("Situation")
    venue = st.radio("Venue", list(VENUES), horizontal=True, key="_ss_venue",
                     help="Each team's own home or away games, for its offense and its defense.")
    quarters = st.multiselect("Quarters", list(QUARTERS), placeholder="All", key="_ss_quarters",
                              format_func=QUARTERS.get)
    garbage = None
    if st.checkbox("Exclude garbage time", key="_ss_garbage",
                   help="Drops plays where the offense's win probability before the snap is "
                        "below the cutoff or above 100% minus it."):
        garbage = st.slider("Win probability cutoff (%)", 1, 25, 10, key="_ss_garbage_wp") / 100
    clutch = None
    if st.checkbox("Clutch time only", key="_ss_clutch",
                   help="The last minutes of the 4th quarter and all of overtime, with the "
                        "score within the margin before the snap."):
        clutch = (st.slider("Minutes left in the 4th quarter", 1, 15, 15, key="_ss_clutch_min"),
                  st.slider("Score within (points)", 0, 16, 8, key="_ss_clutch_pts"))
    play_filter = PlayFilter(tuple(sorted(quarters)), garbage, clutch)
    min_team_plays = int(st.number_input(
        "Min plays per team", min_value=0, value=SEASON_MIN_PLAYS, step=5, key="_ss_min_team_plays",
        help="Teams with fewer plays (offense) or plays faced (defense) in the selection aren't "
             "shown or ranked: their rates would be mostly chance."))
    st.caption(f"Data as of {stamp}")

shown_teams = sorted(ranked_teams.index[ranked_teams.isin(picked)] if picked else ranked_teams.index)

with st.spinner("Computing season stats…"):
    tables = leagues.season_tables(league, season, stamp, weeks, part, adjust, play_filter, venue)
if tables["offense"].empty:
    st.info("No plays in this selection.")
    st.stop()

# Teams with too few plays in the selection are left out of the tables and the ranks.
team_tables = {"offense": season_qualified(tables["offense"], min_team_plays),
               "defense": season_qualified(tables["defense"], min_team_plays)}
if team_tables["offense"].empty:
    st.info(f"No team has {min_team_plays}+ plays in this selection.")
    st.stop()
sit_dists = situational_distributions_from(
    tables["situational_team"][tables["situational_team"].index.isin(team_tables["offense"].index)])
games_max = int(tables["offense"]["Games"].max())
sel_note = (f"weeks {weeks[0]}–{weeks[1]}" if leagues.SEASON_PARTS[part][0] else "")
sel_note = {"Regular season": sel_note, "Postseason": "postseason",
            "Both": f"{sel_note} + postseason"}[part]
filter_notes = [
    " + ".join(QUARTERS[q] for q in sorted(quarters)) if quarters else "",
    f"no garbage time (WP {garbage:.0%}–{1 - garbage:.0%})" if garbage is not None else "",
    f"clutch (last {clutch[0]} min of Q4 + OT, within {clutch[1]})" if clutch else "",
    f"{venue.lower()} games" if VENUES[venue] else "",
]
sel_note = " · ".join(n for n in [sel_note, *filter_notes] if n)
adjusted = leagues.SEASON_ADJUST[adjust]
if adjusted:
    sel_note += " · opponent-adjusted"
# Player minimums scale with the share of plays the filters keep.
play_share = tables["play_share"]



def _default_min(kind: str) -> int:
    """The default player minimum: the per-game one × games, scaled by the
    share of plays the filters keep."""
    return max(1, round(PLAYER_MIN_PLAYS[kind] * games_max * play_share))


tab_teams, tab_team, tab_players = st.tabs(["Teams", "Team", "Players"])

# ---------- Teams ----------
with tab_teams:
    c1, c2, c3 = st.columns([2, 3, 2])
    side_label = c1.segmented_control("Side", list(SIDES), default="Offense",
                                      key="_ss_side") or "Offense"
    view = c2.segmented_control("Stats", list(TEAM_VIEWS), default="Efficiency",
                                key="_ss_view") or "Efficiency"
    side = SIDES[side_label]
    full = team_tables["offense" if side == "posteam" else "defense"]
    lower = season_lower_is_better(side)
    ranks = season_ranks(full, lower)
    cols = [c for c in (TEAM_VIEWS[view] or full.columns) if c in full.columns]
    sortable = [c for c in cols if c != "Games"]
    default_sort = "EPA/play" if "EPA/play" in sortable else sortable[0]
    sort_by = c3.selectbox("Sort by", sortable, index=sortable.index(default_sort),
                           key=f"_ss_sort_{view}")
    # Best first: by rank, so "lower is better" stats sort the right way.
    order = ranks.loc[full.index, sort_by].sort_values(ascending=False, kind="stable").index
    table = full.loc[[t for t in order if t in shown_teams], cols]
    stat_table(style_season_table(table, ranks))
    st.caption(
        f"{side_label} · {sel_note} · counts and yards are per game. Colors rank each team "
        f"among the {len(full)} {'FBS ' if league.season_group_label == 'Conference' else ''}teams "
        f"with {min_team_plays}+ plays "
        "(green = best, red = worst"
        + ("; for a defense, less allowed is better and more turnovers and sacks forced is"
           if side == "defteam" else "") + "). Plays, dropbacks, runs and snaps aren't "
        "colored: volume isn't good or bad. Sorted best first."
        + (" Adjusted for the " + ("offenses" if side == "defteam" else "defenses")
           + " faced; play counts and penalties are as played." if adjusted else ""))

# ---------- Team ----------
with tab_team:
    team = st.selectbox("Team", shown_teams, key=f"_ss_team_{league.name}")
    pbp = venue_plays(leagues.season_plays(league, season, stamp, weeks, part, play_filter),
                      VENUES[venue])
    st.subheader(f"{team} · {ranked_teams.get(team, '')}".rstrip(" ·"))
    col_o, col_d = st.columns(2)
    for col, side_label in [(col_o, "Offense"), (col_d, "Defense (allowed)")]:
        side = SIDES[side_label]
        full = team_tables["offense" if side == "posteam" else "defense"]
        lower = season_lower_is_better(side)
        with col:
            st.markdown(f"**{side_label}**")
            if team not in full.index:
                st.caption(f"Fewer than {min_team_plays} plays in this selection: not ranked.")
                continue
            stat_table(style_team_season(full.loc[team], season_ranks(full, lower).loc[team],
                                         season_rank_label(full, team, lower)))
    st.caption(f"{sel_note} · per game where it's a count · rank among the teams with {min_team_plays}+ plays.")

    st.markdown("**Situational success rates**")
    sr, counts = situational_table(pbp, [team, None], subtotals=True)
    if team in tables["situational_team"].index:  # raw, or adjusted for the defenses faced
        own = tables["situational_team"].loc[team]
        for (label, metric), v in own.items():
            if label in sr.index and f"{team} {metric}" in sr.columns:
                sr.at[label, f"{team} {metric}"] = v
    stat_table(style_sr_table(sr, sit_dists, counts))
    st.caption("Success rate and EPA/play by down and distance (pass plays include sacks; "
               "no spikes or kneels), plays in brackets. Colors rank the season against every "
               "team's season: green = top, red = bottom."
               + (f" {team}'s numbers are adjusted for the defenses faced; the league's are as "
                  "played." if adjusted else ""))

    st.markdown("**Tendencies**")
    tend_downs = st.segmented_control("Downs", list(TENDENCY_DOWNS), default="All downs",
                                      key="_ss_tend_downs", label_visibility="collapsed") or "All downs"
    tend_base = leagues.season_tendencies(league, season, stamp, weeks, part,
                                         play_filter).get(tend_downs, {})
    air = has_air_yards(pbp)
    runs = run_tendencies(pbp, team, league.run_gaps, TENDENCY_DOWNS[tend_downs])
    passes = pass_tendencies(pbp, team, TENDENCY_DOWNS[tend_downs], air)
    col_r, col_p = st.columns(2)
    with col_r:
        st.caption(f"Runs · {runs['Plays'].sum()} with a direction")
        if runs["Plays"].sum():
            st.plotly_chart(run_tendency_figure(runs, tend_base.get("runs")), width="stretch",
                            config={"displayModeBar": False})
    with col_p:
        st.caption(f"Passes · {passes['Plays'].sum()} with a direction")
        if passes["Plays"].sum():
            st.plotly_chart(pass_tendency_figure(passes, tend_base.get("air" if air else "text")),
                            width="stretch", config={"displayModeBar": False})
    st.caption("From the offense's side, as on the replay; the grey dash and hover compare "
               "with the whole league over the same games. Colors are EPA/play.")

    st.markdown("**Player leaders**")
    player_min = {k: _default_min(k) for k in PLAYER_MIN_PLAYS}
    player_dists = season_player_distributions({k: tables[k] for k in PLAYER_MIN_PLAYS}, player_min)
    col_a, col_b = st.columns(2)
    for col, kinds in [(col_a, [("passing", 3), ("rushing", 5)]),
                       (col_b, [("receiving", 8), ("defenders", 10)])]:
        with col:
            for kind, n in kinds:
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

# ---------- Players ----------
def _leaderboard(kind: str, c2, c3) -> None:
    """The Players tab's `kind` leaderboard, with its minimum and sort controls."""
    df = tables[kind]
    if df.empty:
        st.info("No players in this selection.")
        return
    df = df[df["Team"].isin(shown_teams)]
    if kind == "defenders":
        min_plays = None
    else:
        vol = PLAYER_VOLUME_COL[kind]
        min_plays = int(c2.number_input(
            f"Min {'dropbacks' if kind == 'passing' else 'carries' if kind == 'rushing' else 'targets'}",
            min_value=0, value=_default_min(kind), step=5,
            key=f"_ss_min_{kind}_{_default_min(kind)}"))
        df = df[df[vol] >= min_plays]
    numeric = [c for c in df.columns if c not in ("Player", "Team")]
    sort_by = c3.selectbox("Sort by", numeric,
                           index=numeric.index("Yds" if "Yds" in numeric else numeric[0]),
                           key=f"_ss_psort_{kind}")
    top_n = st.slider("Show", 10, 200, 50, step=10, key="_ss_topn")
    df = df.sort_values(sort_by, ascending=False, kind="stable").head(top_n).reset_index(drop=True)
    df.index = df.index + 1
    if kind == "defenders":
        stat_table(style_defense_table(df))
        if "Tackles" in df.columns and league.season_group_label == "Conference":
            st.caption("College tackles, TFLs and hurries come from the play text's stat-crew "
                       "notes, which only some games have, so season totals undercount.")
    else:
        dists = season_player_distributions({kind: tables[kind]}, {kind: min_plays})
        stat_table(style_player_table(df, kind, dists, min_plays))
        st.caption(f"Season totals ({sel_note}); dropbacks include sacks. SR%, aDOT and "
                   f"EPA/play colors rank each player among those with {min_plays}+ "
                   f"{PLAYER_VOLUME_COL[kind].lower()} · green = top · red = bottom. A traded "
                   "player has a row per team."
                   + (" SR% and EPA/play are adjusted for the defenses each player's plays "
                      "came against; counts and aDOT are as played." if adjusted else ""))


with tab_players:
    c1, c2, c3 = st.columns([3, 2, 2])
    kind_label = c1.segmented_control("Leaderboard", list(PLAYER_KINDS), default="Passing",
                                      key="_ss_kind") or "Passing"
    kind = PLAYER_KINDS[kind_label]
    _leaderboard(kind, c2, c3)

# ---------- Methodology ----------
_bands = {lbl.split(" · ")[1]: dist for lbl, _, dist in SR_SITUATIONS}
_band = lambda k: (f"{_bands[k][0]}+" if _bands[k][1] >= 99 else f"{_bands[k][0]}–{_bands[k][1]}")
_college = league.season_group_label == "Conference"
st.divider()
st.subheader("Methodology")
st.markdown(f"""
**Data.** {"College play-by-play from ESPN as published by sportsdataverse, with its college "
           "EPA model; games count when an FBS team played in them, and only FBS teams are "
           "ranked (FCS opponents still count as opponents)" if _college else
           "nflverse's nflfastR play-by-play, with nflfastR's EPA model"}. Only finished, published
games are used; the sidebar picks the regular-season weeks and/or the postseason.

**Definitions** are the replay's. Plays are pass attempts (sacks included) plus rush attempts
(scrambles and kneels included); a play is a *success* when its EPA is above 0, and SR is the share
of successes. 1st Down % is first downs per play, 3rd Down % third-down conversions per attempt,
RZ TD% touchdowns per play snapped at or inside the opponent's 20. Turnovers are interceptions plus lost
fumbles. Time of possession is read from the game clock within each drive. Line play counts downs
only: dropbacks (no spikes), Sack % and QB Hit % (hits include sacks) per dropback, designed runs
(no kneels or scrambles), Stuff % (≤ {STUFF_MAX_YDS} yds) and Explosive Run % (≥ {EXPLOSIVE_RUN_YDS} yds)
per designed run, and OL Penalty % (the offense's holding, false start and ineligible-downfield
flags) per snap.

**Per game and colors.** Counts and yards are divided by the games each team played. Every
colored cell is the team's percentile within its column among the ranked teams: green is best,
red worst. Where lower is better (turnovers, penalties, sacks, stuffs …) the scale is flipped,
and for a defense it's flipped throughout: allowing less is better, forcing more turnovers and
sacks is better. Plays, dropbacks, runs and snaps aren't colored: volume isn't good or bad.

**Situational success.** Early downs are 1st–2nd, late downs 3rd–4th; distance is short
({_band("Short")} yds), medium ({_band("Medium")}) or long ({_band("Long")}). Passes include sacks
but not spikes, runs leave out kneels. Colors compare the team's season with every ranked team's.

**Tendencies.** Run and pass direction come from the play text (from the offense's side; {"college "
"text names the side of a run, not the gap" if _college else "runs by gap"}), pass depth from air
yards. The grey dash and hover are the whole league over the same games.

**Players.** Season totals for the selection; a traded player has a row per team. Rates (SR%,
aDOT, EPA/play) are colored only for players above the minimum, by default the per-game
minimum ({PLAYER_MIN_PLAYS["passing"]} dropbacks, {PLAYER_MIN_PLAYS["rushing"]} carries,
{PLAYER_MIN_PLAYS["receiving"]} targets) times the most games a team played.{" College tackles, "
"TFLs and hurries need the stat-crew play text, which only some games have, so they undercount."
if _college else ""}

**Opponent adjustment** (Stats: Opponent-adjusted). Each offense's season value is reduced by
the average *strength* of the defenses it faced, and each defense's by that of the offenses it
faced; a strength is how far a team's adjusted value sits from the league average. Because each
team's strength depends on its opponents' adjusted values, the two are recomputed in turn until
they stop moving (at most {ADJUST_MAX_PASSES} passes), as the Simple Rating System does for
points. A team's strength is shrunk toward average by games ÷ (games + {SHRINK_GAMES}), so a team
seen in only a game or two can't swing its opponents' numbers, and the league average is kept
where it was. Play counts, a team's own penalties, aDOT and defenders' stats stay as played.
A player's SR% and EPA/play are adjusted by the strength of the defenses his plays came against,
weighted by how many plays he had against each. The adjustment adds and subtracts, so a team or
player with only a few plays in a cell can land past what the stat can be (a success rate below
0% or above 100%, fewer than 0 turnovers a game): such a value is shown at the nearest one it can
take.

**Filters** (sidebar, Situation). *Home / Away* keeps each team's own home or away games, for its
offense and its defense (neutral-site games count for the team listed at home); with
Opponent-adjusted on, opponent strengths still come from all games and only the team's venue games
are adjusted, by the opponents it met there. *Quarters* keeps plays in the picked quarters (OT is
every overtime period). *Exclude garbage time* drops plays where the offense's win probability
before the snap is below the cutoff or above 100% minus it (default 10%–90%). *Clutch time* keeps
the last minutes of the 4th quarter (default all 15) and overtime when the score before the snap is
within the margin (default 8, one score). Filters act on plays, so Games counts the games with a
play left in, per-game numbers and time of possession cover only the plays kept, and the default
player minimums shrink with the share of plays kept. *Min plays per team* (default {SEASON_MIN_PLAYS})
leaves out, and doesn't rank, a team with fewer plays (or plays faced, for a defense) in the
selection, so a narrow filter isn't topped by a team's handful of snaps.

**Limits.** No home-field term in the adjustment. The adjustment is
additive, not a regression, and like any season stat it is noisy on small samples: early weeks,
the postseason, or a team's few plays in one situation.
""")
