"""
The Season simulation page's college football view (`render()`): the rest
of an FBS season played out 10,000 times by cfp_sim.py, through the
conference title games and the 12-team College Football Playoff, 2024 on.

UI and caching only, like rosters.py: cfp_sim computes, views styles,
leagues loads the season. Outside `leagues._CODE_FILES`, so a change here
doesn't make the season store rebuild.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import cfp_sim
import leagues
import season_sim as sim
from replay_core import SEASON_MIN_PLAYS, season_qualified
from views import bracket_html, prob_text, section_nav, stat_table, style_seed_odds, style_sim_table

COLLEGE = leagues.COLLEGE
PROBS = ["Title game", "Conf. title", "Bowl", "Playoff", "Bye", "Quarterfinal", "Semifinal", "Final",
         "Champion", "Top 25", "Win %", "If win", "If loss", "Share", "Wins it", "Playoff if so", "If not"]


@st.cache_data(ttl=1800, show_spinner=False)
def _schedule(season: int) -> pd.DataFrame:
    return leagues.load_cfb_schedule(season)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _conferences(season: int) -> pd.DataFrame:
    return cfp_sim.read_conferences(season)


@st.cache_data(ttl=1800, show_spinner=False)
def _ranking(season: int, as_of: int | None) -> tuple[pd.Series, str]:
    """The committee's ranking out now (or before `as_of`'s games)."""
    try:
        return cfp_sim.read_cfp_ranking(season) if as_of is None else cfp_sim.ranking_before(season, as_of)
    except Exception:
        return pd.Series(dtype=int), ""


def _reg_weeks(sched: pd.DataFrame) -> list[int]:
    s = sched[sched["season_type"].astype(str).str.startswith("reg") & sched["fbs_participant"].fillna(False)]
    return sorted(int(w) for w in s["week"].dropna().unique())


def _prior(season: int) -> pd.Series | None:
    """`season`'s net EPA/play per FBS team (offense less what the defense
    allowed, opponent-adjusted), the regular season with its title games."""
    sched = _schedule(season)
    if sched.empty:
        return None
    done = sched[sched["completed"].eq(True) & sched["season_type"].astype(str).str.startswith("reg")]
    if done.empty:
        return None
    weeks = (int(done["week"].min()), int(done["week"].max()))
    try:
        t = leagues.season_tables(COLLEGE, season, COLLEGE.stamp(), weeks, "Regular season", "Opponent-adjusted")
    except leagues.GameLoadError:
        return None
    return t["offense"]["EPA/play"] - t["defense"]["EPA/play"]


@st.cache_data(max_entries=4, show_spinner=False, ttl=6 * 3600)
def _run(season: int, stamp: str, as_of: int | None):
    """The college simulation for a season from a week (None: now), cached by
    the data's stamp: (CollegeInput, games left, CollegeResult, the weeks
    the numbers cover)."""
    model = cfp_sim.load_model()
    sched = _schedule(season)
    ids = leagues.load_cfb_team_ids(season, stamp)
    pbp = COLLEGE.season_pbp(season, stamp)
    played = sorted(int(w) for w in pbp.loc[~pbp["postseason"], "week"].dropna().unique())
    weeks = [w for w in played if as_of is None or w < as_of]
    off = def_ = None
    if weeks:
        tables = leagues.season_tables(COLLEGE, season, stamp, (weeks[0], weeks[-1]), "Regular season",
                                       "Opponent-adjusted")
        off = season_qualified(tables["offense"], SEASON_MIN_PLAYS)
        def_ = season_qualified(tables["defense"], SEASON_MIN_PLAYS)
    inp, left = cfp_sim.prepare(season, sched, _conferences(season), dict(zip(ids["id"], ids["abbr"])),
                                off, def_, _prior(season - 1), model, as_of)
    return inp, left, cfp_sim.simulate(inp), (weeks[0], weeks[-1]) if weeks else None


def _actual(season: int, ids: dict[int, str]) -> pd.DataFrame | None:
    """What happened, once the season's playoff is decided: wins, title game,
    conference title, the playoff field and seed, the national champion."""
    sched = _schedule(season)
    if "playoff_round_name" not in sched.columns:
        return None
    final = sched[sched["playoff_round_name"] == "National Championship"]
    if final.empty or final["home_points"].isna().any():
        return None
    conf = _conferences(season)
    g = cfp_sim.schedule_games(sched, set(conf["team_id"]))
    g = g[g["played"]]
    post = sched[sched["playoff_round_name"].notna()]
    seeds = {}
    for h, a, hs, as_ in zip(post["home_id"], post["away_id"], post["playoff_home_seed"], post["playoff_away_seed"]):
        if pd.notna(h) and pd.notna(hs):
            seeds[ids.get(int(h))] = int(hs)
        if pd.notna(a) and pd.notna(as_):
            seeds[ids.get(int(a))] = int(as_)
    f = final.iloc[0]
    champ = ids.get(int(f["home_id"] if f["home_points"] > f["away_points"] else f["away_id"]))
    rows = []
    for r in conf.itertuples(index=False):
        t = ids.get(r.team_id, r.abbr)
        mine = g[(g["home_id"] == r.team_id) | (g["away_id"] == r.team_id)]
        won = ((mine["home_id"] == r.team_id) & (mine["home_margin"] > 0)) | \
              ((mine["away_id"] == r.team_id) & (mine["home_margin"] < 0))
        titles = mine["title"] != ""
        rows.append({"Team": t, "Final": f"{int(won.sum())}-{int((~won).sum())}",
                     "Conf. title": bool((won & titles).any()), "Seed": seeds.get(t, 0), "Champion": t == champ})
    return pd.DataFrame(rows).set_index("Team")


def render(latest_season: int) -> None:
    """The college page under the page's League switch."""
    with st.sidebar:
        season = int(st.number_input("Season", min_value=cfp_sim.FIRST_SEASON, max_value=latest_season,
                                     value=latest_season, key="_sim_cfb_season",
                                     help="From 2024, the 12-team playoff."))
    model = cfp_sim.load_model()
    if model is None:
        st.title("🎲 Season simulation")
        st.error("The college model (models/season_sim_cfb.json) is missing: run "
                 "tools/build_cfb_sim_model.py.")
        st.stop()
    sched = _schedule(season)
    if sched.empty:
        st.title(f"🎲 {season} college football simulation")
        st.warning("The college schedule couldn't be read just now; try again in a minute.")
        st.stop()
    weeks = _reg_weeks(sched)
    # Through the title games' week: from there on, only the playoff is left.
    g = cfp_sim.schedule_games(sched, set(_conferences(season)["team_id"]))
    title_week = g.loc[g["title"] != "", "week"]
    last_week = int(title_week.min()) if len(title_week) else weeks[-1]
    final = (sched[sched["playoff_round_name"].eq("National Championship")]
             if "playoff_round_name" in sched.columns else sched.iloc[0:0])
    unfinished = final.empty or final["home_points"].isna().any()
    options = ([None] if unfinished else []) + [w for w in weeks if w <= last_week]
    with st.sidebar:
        default = 0 if options[0] is None else options.index(max(w for w in options if w <= 10))
        as_of = st.selectbox("Simulate from", options, index=default,
                             format_func=lambda w: "Now" if w is None else f"Start of week {w}",
                             key=f"_sim_cfb_from_{season}",
                             help="Now: every game played so far counts. Start of week k: only the "
                                  "games before week k, as they were known then.")
        level_label = st.segmented_control("Confidence level", list(sim.CONFIDENCE_LEVELS), default="90%",
                                           key="_sim_cfb_level",
                                           help="The share of the runs each range of wins holds, and the "
                                                "line between likely and on the bubble.") or "90%"
    level = sim.CONFIDENCE_LEVELS[level_label]

    stamp = COLLEGE.stamp()
    try:
        with st.spinner(f"Playing out the {season} season {sim.SIM_RUNS:,} times…"):
            inp, left, res, span = _run(season, stamp, as_of)
    except leagues.GameLoadError as e:
        st.title(f"🎲 {season} college football simulation")
        st.warning(str(e))
        st.stop()
    names = COLLEGE.team_names(season, stamp)
    _, logos, _ = COLLEGE.team_meta(season, stamp)
    ids_df = leagues.load_cfb_team_ids(season, stamp)
    ids = dict(zip(ids_df["id"], ids_df["abbr"]))
    ranking, rank_label = _ranking(season, as_of)
    cfp_now = {ids.get(i): r for i, r in ranking.items()}
    actual = _actual(season, ids) if as_of is not None else None

    def nm(t: str) -> str:
        return names.get(t, t)

    tt = cfp_sim.team_table(res, level)
    tt["CFP now"] = tt["Team"].map(cfp_now)

    # ---------- Header ----------
    st.title(f"🎲 {season} college football simulation")
    n_left = len(res.left)
    st.caption(f"{sim.SIM_RUNS:,} runs of the {n_left} regular-season games left with an FBS team, the "
               "conference title games and the College Football Playoff" if n_left else
               f"{sim.SIM_RUNS:,} runs of the title games and the playoff")
    st.caption((f"From the season numbers through week {span[1]}, opponent-adjusted, blended with last "
                "season's" if span else "From last season's numbers alone (no games yet)")
               + f". The playoff: {cfp_sim.format_note(season)}. Ranges hold at least {level_label} of the "
               f"runs; a probability's own margin of error is at most ±{sim.mc_margin(0.5, res.runs, level):.1%}.")
    section_nav([("Playoff race", "race"), ("Bracket", "bracket"), ("Conferences", "conferences"),
                 ("Team", "team"), ("Games", "games"), ("How it works", "how")])

    # ---------- Playoff race ----------
    st.subheader("Playoff race", anchor="race")
    race = tt[tt["Playoff %"] >= 0.005].sort_values(["Playoff %", "Champion %"], ascending=False)
    view = pd.DataFrame({
        "Conf": race["Conference"].to_numpy(), "Now": race["Now"].to_numpy(),
        "CFP now": [("" if pd.isna(r) else f"{int(r)}") for r in race["CFP now"]],
        "Proj.": [f"{w:.1f}–{l:.1f}" for w, l in zip(race["W"], race["L"])],
        f"Wins ({level_label})": [f"{lo:.0f}–{hi:.0f}" if lo != hi else f"{lo:.0f}"
                                  for lo, hi in zip(race["W low"], race["W high"])],
        "Conf. title": race["Conf champ %"].to_numpy(), "Playoff": race["Playoff %"].to_numpy(),
        "Call": [sim.call(p, level) for p in race["Playoff %"]], "Bye": race["Bye %"].to_numpy(),
        "Semifinal": race["Semifinal %"].to_numpy(), "Final": race["Final %"].to_numpy(),
        "Champion": race["Champion %"].to_numpy(),
    })
    view.index = pd.Index([nm(t) for t in race["Team"]], name="Team")
    if actual is not None:
        a = actual.reindex(race["Team"])
        view.insert(view.columns.get_loc("Proj.") + 1, "Actual", a["Final"].to_numpy())
        view.insert(view.columns.get_loc("Call") + 1, "Made it",
                    np.where(a["Seed"].fillna(0).to_numpy() > 0, "✓", ""))
    stat_table(style_sim_table(view, PROBS))
    st.caption(
        f"Every team in the field in at least 0.5% of the runs. **CFP now** is the committee's {rank_label} "
        "ranking" + (" (none out yet)" if not cfp_now else "") + f"; **Proj.** the average record over the "
        f"runs (regular season and title game), **Wins** the range holding at least {level_label} of them. "
        "The percentages are the share of runs a team wins its conference, makes the field, gets a bye "
        "(seeds 1–4), reaches the semifinals and the final, and wins it all. **Call** as for the NFL: "
        f"likely at {level_label}+, unlikely at {1 - level:.0%} or less."
        + (" **Actual** (the record with the title game) and **Made it**: what happened." if actual is not None
           else ""))

    # ---------- Bracket ----------
    st.subheader("Projected bracket", anchor="bracket")
    field_ = cfp_sim.projected_field(res, season)
    seeds_t = cfp_sim.seed_table(res, 0.0)
    rows = [(k + 1, t, nm(t), float(seeds_t.at[t, k + 1])) for k, t in enumerate(field_)]
    first = [(h, a, cfp_sim.game_prob(inp, field_[h - 1], field_[a - 1], home=True))
             for h, a in cfp_sim._FIRST_ROUND]
    st.html(bracket_html(rows, first, [1, 2, 3, 4], logos))
    st.caption("The most likely field under the rules: the likeliest champion of each conference whose "
               "title brings a bid" + (" and the Group of 6 team likeliest in" if cfp_sim.cfp_format(season).power4_in
                                       else "") + ", then the teams likeliest to make it, seeded by the "
               "assignment with the most runs behind it, each team's share of runs at that seed by its name. "
               "Seeds 1–4 have "
               "byes; 5–8 host 12–9 on campus (each game's chances on the right), and the winners go on to "
               "the quarterfinals against 4, 3, 2 and 1 at bowl sites.")
    with st.expander("Seed odds", expanded=False, key="_sim_cfb_exp_seeds"):
        t = cfp_sim.seed_table(res)
        t.index = [nm(x) for x in t.index]
        t.index.name = "Team"
        t.columns = [str(c) for c in t.columns]
        stat_table(style_seed_odds(t))
        st.caption("The share of runs each team ends at each seed, or out.")

    # ---------- Conferences ----------
    st.subheader("Conferences", anchor="conferences")
    confs = sorted(set(res.conference), key=lambda c: (c not in cfp_sim.POWER4, c == "Independent", c))
    conf = st.segmented_control("Conference", confs, default=confs[0], key=f"_sim_cfb_conf_{season}",
                                label_visibility="collapsed") or confs[0]
    c = tt[tt["Conference"] == conf].sort_values(["Conf W", "W"], ascending=False)
    cv = pd.DataFrame({
        "Now": c["Now"].to_numpy(), "Conf": c["Conf now"].to_numpy(),
        "Proj.": [f"{w:.1f}–{l:.1f}" for w, l in zip(c["W"], c["L"])],
        "Conf proj.": [f"{w:.1f}–{l:.1f}" for w, l in zip(c["Conf W"], c["Conf L"])],
        f"Wins ({level_label})": [f"{lo:.0f}–{hi:.0f}" if lo != hi else f"{lo:.0f}"
                                  for lo, hi in zip(c["W low"], c["W high"])],
        "Title game": c["Title game %"].to_numpy(), "Conf. title": c["Conf champ %"].to_numpy(),
        "Bowl": c["Bowl %"].to_numpy(), "Playoff": c["Playoff %"].to_numpy(),
        "Champion": c["Champion %"].to_numpy(),
    })
    cv.index = pd.Index([nm(t) for t in c["Team"]], name="Team")
    stat_table(style_sim_table(cv, PROBS))
    tm = cfp_sim.title_matchups(res, conf, 3)
    if not tm.empty:
        st.markdown(f"**Likeliest {conf} title games**")
        stat_table(style_sim_table(pd.DataFrame({
            "Game": [f"{nm(a)} vs {nm(b)}" for a, b in zip(tm["Team"], tm["Opponent"])],
            "Share": tm["Share"], "Wins it": tm["Wins"]}), PROBS), hide_index=True)
    st.caption("By projected conference wins. **Title game** is the share of runs a team plays in the "
               "conference's title game (the top two by conference record; in the Sun Belt the two division "
               "winners), **Conf. title** wins it, **Bowl** reaches six wins. Ties in the standings go to "
               "head-to-head (when each pair played), the record against common conference opponents, the "
               "conference opponents' combined record, then the team rating (the conferences use computer "
               "rankings last). **Wins it**: how often the first team named wins that game.")

    # ---------- Team ----------
    st.subheader("Team", anchor="team")
    by_name = sorted(res.teams, key=nm)
    pick = race["Team"].iloc[min(5, len(race) - 1)] if len(race) else by_name[0]
    team = st.selectbox("Team", by_name, format_func=nm, index=by_name.index(pick), key=f"_sim_cfb_team_{season}",
                        label_visibility="collapsed")
    row = tt.set_index("Team").loc[team]
    with st.container(key="row_sim_cfb_cards"):
        c1 = st.columns(2, wrap=False)
        c1[0].metric("Now", row["Now"], help=f"Conference {row['Conf now']}")
        c1[1].metric("Projected", f"{row['W']:.1f}–{row['L']:.1f}",
                     help=f"Wins in {level_label} of the runs: {row['W low']:.0f}–{row['W high']:.0f}")
        c2 = st.columns(2, wrap=False)
        c2[0].metric("Playoff", prob_text(row["Playoff %"]).replace("⁠", ""),
                     help=f"Conference title {prob_text(row['Conf champ %'])} · bye {prob_text(row['Bye %'])}")
        c2[1].metric("National title", prob_text(row["Champion %"]).replace("⁠", ""),
                     help=f"Avg committee rank {row['Avg rank']:.1f}")
    dist = cfp_sim.wins_distribution(res, team)
    fig = go.Figure(go.Bar(x=dist.index, y=dist.to_numpy(), marker_color="#1a9e8f",
                           text=[prob_text(p).replace("⁠", "") for p in dist.to_numpy()],
                           textposition="outside", hovertemplate="%{x:g} wins: %{y:.1%}<extra></extra>"))
    fig.update_layout(height=240, margin=dict(l=10, r=10, t=30, b=30), dragmode=False,
                      title=dict(text=f"{nm(team)}: wins (regular season and title game)", font=dict(size=14)),
                      yaxis=dict(visible=False, fixedrange=True), xaxis=dict(dtick=1, fixedrange=True))
    st.plotly_chart(fig, config={"displayModeBar": False, "scrollZoom": False}, width="stretch")
    lev = cfp_sim.leverage(res, team)
    mine = (lev[lev["own"]].merge(left[["game_id", "p_home", "mu"]], on="game_id", how="left")
            .sort_values("week").reset_index(drop=True))
    if not mine.empty:
        home = mine["home_team"] == team
        stat_table(style_sim_table(pd.DataFrame({
            "Week": mine["week"],
            "Opp": [f"vs {nm(a)}" if h else f"@ {nm(hm)}" for h, a, hm in zip(home, mine["away_team"], mine["home_team"])],
            "Win %": np.where(home, mine["p_home"], 1 - mine["p_home"]),
            "Proj. margin": [f"{m:+.1f}" for m in np.where(home, mine["mu"], -mine["mu"])],
            "If win": np.where(home, mine["Home wins"], mine["Away wins"]),
            "If loss": np.where(home, mine["Away wins"], mine["Home wins"]),
        }), PROBS), hide_index=True)
        st.caption(f"{nm(team)}'s regular-season games left, and its playoff chances in the runs where it "
                   "wins or loses each.")
    others = lev[~lev["own"]].head(5)
    if not others.empty and others["Swing"].iloc[0] >= 0.01:
        root = [{"Week": r.week, "Game": f"{nm(r.away_team)} @ {nm(r.home_team)}",
                 "Root for": nm(r.home_team if r._4 >= r._5 else r.away_team),
                 "Playoff if so": max(r._4, r._5), "If not": min(r._4, r._5)}
                for r in others.itertuples(index=False)]
        st.markdown(f"**Other games that matter most to {nm(team)}**")
        stat_table(style_sim_table(pd.DataFrame(root), PROBS), hide_index=True)

    # ---------- Games left ----------
    st.subheader("Games left", anchor="games")
    if left.empty:
        st.caption("No regular-season games left.")
    else:
        weeks_left = sorted(left["week"].unique())
        wk = st.selectbox("Week", weeks_left, format_func=lambda w: f"Week {w}",
                          key=f"_sim_cfb_week_{season}_{as_of}", label_visibility="collapsed")
        g = left[left["week"] == wk].sort_values("p_home", key=lambda p: abs(p - 0.5))  # closest first
        fav = np.where(g["mu"] >= 0, g["home_team"], g["away_team"])
        stat_table(style_sim_table(pd.DataFrame({
            "Game": [f"{nm(a)} @ {nm(h)}" + (" (neutral)" if n else "")
                     for a, h, n in zip(g["away_team"], g["home_team"], g["neutral"])],
            "Proj.": [f"{nm(t)} by {abs(m):.1f}" for t, m in zip(fav, g["mu"])],
            "Win %": np.maximum(g["p_home"], 1 - g["p_home"]),
        }), PROBS), hide_index=True)
        st.caption("Games with an FBS team, closest first: the model's expected margin and the favorite's "
                   "chance. An FCS opponent is rated as the FCS average.")

    # ---------- How it works ----------
    st.subheader("How it works", anchor="how")
    fit = model.get("fit", {})
    com = model.get("committee", {})
    chk = com.get("check", {})
    with st.expander("The model, the committee and the confidence levels", expanded=False,
                     key="_sim_cfb_exp_howto"):
        st.markdown(f"""
**Each game.** Each team's power rating, in points against an average FBS team: this season's
opponent-adjusted {', '.join(f"{v['stat']} (× {v['coef']:.1f})" for v in model['features'].values())}
(the Matchup page's numbers: half of the offense's value less what the defense allowed), weighted
games / (games + {model['g0']:g}), and last season's net EPA/play (× {model['prior']['coef']:.1f}) for the
rest; the home team gets {model['hfa']:.1f} points. Against an FCS team the FBS side is expected to win by
{model['fcs']['intercept']:.1f} + {model['fcs']['slope']:.2f} × its rating. Fitted on every week of
{fit.get('seasons', '2016–2025')} ({fit.get('games', 0):,} FBS games, each projected from the weeks before it).
Out of sample, games the week after the numbers: log loss {fit.get('next week', {}).get('log loss', float('nan')):.3f},
winner picked {fit.get('next week', {}).get('picked', float('nan')):.1%}. College has no injury reports in
this app's data, so lineups aren't counted.

**The runs.** Each run draws every team's rating error once for all its games
({sim.team_sd(model, np.array([0]))[0]:.1f} points before a game, {sim.team_sd(model, np.array([12]))[0]:.1f}
after twelve) and each game's noise ({model['game_sd']:.1f} points). Then the conference standings and
title games, the committee's ranking, the field and the bracket.

**The committee.** Nobody can replay the committee, so its final ranking is a model: a score from
the record (wins and losses, title game included), the power rating, the schedule (the opponents'
average rating), a conference title and a Power conference (or Notre Dame), fitted on the committee's
final rankings of {fit.get('seasons', '2016–2025')} (which team it put above which), plus a draw for what the
score doesn't see. Ranking each season from a model fitted on the others, it had on average
{chk.get('top 4 of 4', float('nan')):.1f} of the committee's top 4, {chk.get('top 12 of 12', float('nan')):.1f} of its
top 12 and {chk.get('top 25 of 25', float('nan')):.1f} of its top 25. The current ranking is shown for
reference; the runs don't start from it.

**The field.** {cfp_sim.format_note(season)}. The first round is played at the better seed's
home, the rest at neutral sites; games already played keep their result.

**Title games.** From the final regular-season results, the standings put 16 of 2024–25's 18 title games
together in every run (2025's five-way ACC tie included); the two misses were three- and four-way ties in
the American and the Mountain West, which those conferences settled by computer rankings.

**Confidence.** Every percentage is a share of the {sim.SIM_RUNS:,} runs: at {level_label}, its own margin of
error is at most ±{sim.mc_margin(0.5, res.runs, level):.1%}. The ranges hold at least {level_label} of the runs.
`tools/validate_cfb_sim.py` simulated 2024 and 2025 from weeks 5, 9 and 13 with models fitted without
them: final wins landed in the 90% ranges 95–100% of the time, as often as the runs put them there, and
the playoff chances tracked what happened (teams given 40–60% made it 48% of the time; two seasons are a
small sample). From week 9 the projected wins missed by 0.9 on average. Bowl games other than the
playoff aren't played out.
""")
