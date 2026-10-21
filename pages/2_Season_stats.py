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
    LINE_ROWS, PLAYER_MIN_PLAYS, PLAYER_VOLUME_COL, TENDENCY_DOWNS, has_air_yards,
    pass_tendencies, run_tendencies, season_lower_is_better, season_player_distributions,
    season_rank_label, season_ranks, situational_table,
)
from views import (
    pass_tendency_figure, run_tendency_figure, stat_table, style_defense_table,
    style_player_table, style_season_table, style_sr_table, style_team_season,
)

st.set_page_config(page_title="Season Stats", layout="wide", page_icon="📊")

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
                                 key=f"_ss_weeks_{league.name}_{season}")
    groups = sorted(g for g in ranked_teams.unique() if g)
    picked = st.multiselect(league.season_group_label, groups, placeholder="All",
                            key=f"_ss_groups_{league.name}")
    adjust = st.radio("Stats", list(leagues.SEASON_ADJUST), horizontal=True, key="_ss_adjust",
                      help="Opponent-adjusted: each offense's numbers less the average strength "
                           "of the defenses it faced, and each defense's less that of the "
                           "offenses it faced, iterated until they settle (like SRS). Counts of "
                           "plays, penalties, aDOT and defenders' stats stay as played.")
    st.caption(f"Data as of {stamp}")

shown_teams = sorted(ranked_teams.index[ranked_teams.isin(picked)] if picked else ranked_teams.index)

with st.spinner("Computing season stats…"):
    tables = leagues.season_tables(league, season, stamp, weeks, part, adjust)
if tables["offense"].empty:
    st.info("No games in this selection.")
    st.stop()

n_teams = len(tables["offense"])
games_max = int(tables["offense"]["Games"].max())
sel_note = (f"weeks {weeks[0]}–{weeks[1]}" if leagues.SEASON_PARTS[part][0] else "")
sel_note = {"Regular season": sel_note, "Postseason": "postseason",
            "Both": f"{sel_note} + postseason"}[part]
adjusted = leagues.SEASON_ADJUST[adjust]
if adjusted:
    sel_note += " · opponent-adjusted"

tab_teams, tab_team, tab_players = st.tabs(["Teams", "Team", "Players"])

# ---------- Teams ----------
with tab_teams:
    c1, c2, c3 = st.columns([2, 3, 2])
    side_label = c1.segmented_control("Side", list(SIDES), default="Offense",
                                      key="_ss_side") or "Offense"
    view = c2.segmented_control("Stats", list(TEAM_VIEWS), default="Efficiency",
                                key="_ss_view") or "Efficiency"
    side = SIDES[side_label]
    full = tables["offense" if side == "posteam" else "defense"]
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
        f"among the {n_teams} {'FBS ' if league.season_group_label == 'Conference' else ''}teams "
        "(green = best, red = worst"
        + ("; for a defense, less allowed is better and more turnovers and sacks forced is"
           if side == "defteam" else "") + "). Plays, dropbacks, runs and snaps aren't "
        "colored: volume isn't good or bad. Sorted best first."
        + (" Adjusted for the " + ("offenses" if side == "defteam" else "defenses")
           + " faced; play counts and penalties are as played." if adjusted else ""))

# ---------- Team ----------
with tab_team:
    team = st.selectbox("Team", shown_teams, key=f"_ss_team_{league.name}")
    pbp = leagues.season_plays(league, season, stamp, weeks, part)
    st.subheader(f"{team} · {ranked_teams.get(team, '')}".rstrip(" ·"))
    col_o, col_d = st.columns(2)
    for col, side_label in [(col_o, "Offense"), (col_d, "Defense (allowed)")]:
        side = SIDES[side_label]
        full = tables["offense" if side == "posteam" else "defense"]
        if team not in full.index:
            continue
        lower = season_lower_is_better(side)
        with col:
            st.markdown(f"**{side_label}**")
            stat_table(style_team_season(full.loc[team], season_ranks(full, lower).loc[team],
                                         season_rank_label(full, team, lower)))
    st.caption(f"{sel_note} · per game where it's a count · rank among {n_teams} teams.")

    st.markdown("**Situational success rates**")
    sr, counts = situational_table(pbp, [team, None], subtotals=True)
    if team in tables["situational_team"].index:  # raw, or adjusted for the defenses faced
        own = tables["situational_team"].loc[team]
        for (label, metric), v in own.items():
            if label in sr.index and f"{team} {metric}" in sr.columns:
                sr.at[label, f"{team} {metric}"] = v
    stat_table(style_sr_table(sr, tables["situational"], counts))
    st.caption("Success rate and EPA/play by down and distance (pass plays include sacks; "
               "no spikes or kneels), plays in brackets. Colors rank the season against every "
               "team's season: green = top, red = bottom."
               + (f" {team}'s numbers are adjusted for the defenses faced; the league's are as "
                  "played." if adjusted else ""))

    st.markdown("**Tendencies**")
    tend_downs = st.segmented_control("Downs", list(TENDENCY_DOWNS), default="All downs",
                                      key="_ss_tend_downs", label_visibility="collapsed") or "All downs"
    tend_base = leagues.season_tendencies(league, season, stamp, weeks, part).get(tend_downs, {})
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
    player_min = {k: PLAYER_MIN_PLAYS[k] * games_max for k in PLAYER_MIN_PLAYS}
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
with tab_players:
    c1, c2, c3 = st.columns([3, 2, 2])
    kind_label = c1.segmented_control("Leaderboard", list(PLAYER_KINDS), default="Passing",
                                      key="_ss_kind") or "Passing"
    kind = PLAYER_KINDS[kind_label]
    df = tables[kind]
    if df.empty:
        st.info("No players in this selection.")
        st.stop()
    df = df[df["Team"].isin(shown_teams)]
    if kind == "defenders":
        min_plays = None
    else:
        vol = PLAYER_VOLUME_COL[kind]
        min_plays = int(c2.number_input(
            f"Min {'dropbacks' if kind == 'passing' else 'carries' if kind == 'rushing' else 'targets'}",
            min_value=0, value=PLAYER_MIN_PLAYS[kind] * games_max, step=5, key=f"_ss_min_{kind}_{games_max}"))
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
