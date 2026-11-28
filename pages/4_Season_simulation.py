"""
The Season simulation page: the rest of a season played out 10,000 times,
for each team's final record and its chances of each title, the projected
bracket and (NFL) the draft order, each with the range the runs give at the
confidence level picked. NFL: season_sim.py, divisions, wild cards and the
draft. College football (League switch): cfp_sim.py, conference title games
and the 12-team College Football Playoff, 2024 on.

Each game left gets an expected margin from the Matchup page's numbers
(opponent-adjusted Proj., moved by the players gone on the injury reports
and rosters, and early on blended with last season's ratings), with the
spread the fit (tools/build_sim_model.py) found. "Simulate from" can also
start an earlier week of any season since 2016, with only what was known
then, next to what happened.

No data loading or stat logic of its own: leagues.py loads and caches,
season_sim computes, views styles.
"""
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import fresh  # reload modules a hot update left stale, before importing from them
fresh.reload_stale()
import cfp_sim
import leagues
import season_sim as sim
from replay_core import FRANCHISE_MOVES, SEASON_MIN_PLAYS, season_qualified
from views import (bracket_html, mobile_css, prob_text, section_nav, stat_table, style_seed_odds,
                   style_sim_table)

st.set_page_config(page_title="Season simulation", layout="wide", page_icon="🎲")
mobile_css()  # the section links' sticky row

FIRST_SEASON = 2016   # NFL: snap counts, injury reports and the model's fit start here
_TODAY = date.today()
LATEST_SEASON = _TODAY.year if _TODAY.month >= 9 else _TODAY.year - 1
NOW = "Now"
PROBS = ["Division", "Playoffs", "Bye", "Top seed", "Divisional", "Conf. game", "Super Bowl",
         "Champion", "#1 %", "Top 3 %", "Top 5 %", "Top 10 %", "W", "L", "If win", "If loss",
         "Win %", "Home wins", "Away wins", "Share", "AFC wins",
         "Title game", "Conf. title", "Bowl", "Playoff", "Quarterfinal", "Semifinal", "Final", "Top 25"]


def _week_label(as_of: int | None) -> str:
    return NOW if as_of is None else f"Start of week {as_of}"


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Season simulation")
    league = leagues.LEAGUES[st.radio("League", list(leagues.LEAGUES), horizontal=True, key="_sim_league")]

if league is leagues.COLLEGE:
    import college_sim_view  # the college page: its own sidebar controls and sections
    college_sim_view.render(LATEST_SEASON)
    st.stop()

with st.sidebar:
    season = int(st.number_input("Season", min_value=FIRST_SEASON, max_value=LATEST_SEASON,
                                 value=LATEST_SEASON, key="_sim_season"))

model = sim.load_model()
if model is None:
    st.title("🎲 Season simulation")
    st.error("The game model (models/season_sim.json) is missing: run tools/build_sim_model.py.")
    st.stop()
@st.cache_data(ttl=600, show_spinner=False)
def _schedule(season: int) -> pd.DataFrame:
    """nflverse's schedule with ESPN's finals on the games it doesn't have a
    result for yet (it's rebuilt a few times a day; ESPN has them in minutes)."""
    return sim.nfl_with_finals(leagues.load_schedule(season))


sched = _schedule(season)
if sched.empty:
    st.title(f"🎲 {season} NFL season simulation")
    st.warning("nflverse's schedule couldn't be read just now; try again in a minute.")
    st.stop()
reg_weeks = sorted(int(w) for w in sched.loc[sched["game_type"] == "REG", "week"].unique())
games_left = bool(sched.loc[sched["game_type"] == "REG", "result"].isna().any())
post_left = bool(sched.loc[sched["game_type"] != "REG", "result"].isna().any()) or not (
    sched["game_type"] == "SB").any()
options = ([None] if games_left or post_left else []) + reg_weeks
with st.sidebar:
    # The current season starts from now; a finished one from mid-season, to
    # set the runs against what happened.
    default = 0 if options[0] is None else options.index(reg_weeks[len(reg_weeks) // 2])
    as_of = st.selectbox("Simulate from", options, index=default, format_func=_week_label,
                         key=f"_sim_from_{season}",
                         help="Now: every game played so far counts. Start of week k: only the "
                              "games before week k count, with that week's injury reports, as "
                              "they were known then.")
    level_label = st.segmented_control("Confidence level", list(sim.CONFIDENCE_LEVELS), default="90%",
                                       key="_sim_level",
                                       help="The share of the runs each range (wins, draft pick) "
                                            "holds, and the line between likely and on the bubble.") or "90%"
    level = sim.CONFIDENCE_LEVELS[level_label]

stamp = league.stamp()
avail_stamp = leagues.nflverse_stamp("injuries")


def _prior(season: int) -> pd.Series | None:
    """Last season's net EPA/play per team (offense less what the defense
    allowed, opponent-adjusted), under this season's abbreviations too."""
    prev = leagues.load_schedule(season)
    if prev.empty:
        return None
    weeks = sorted(int(w) for w in prev.loc[prev["game_type"] == "REG", "week"].unique())
    try:
        t = leagues.season_tables(league, season, league.stamp(), (weeks[0], weeks[-1]),
                                  "Regular season", "Opponent-adjusted")
    except leagues.GameLoadError:
        return None
    net = t["offense"]["EPA/play"] - t["defense"]["EPA/play"]
    for new, old in FRANCHISE_MOVES.items():
        if old in net.index and new not in net.index:
            net[new] = net[old]
    return net


@st.cache_data(max_entries=4, show_spinner=False, ttl=6 * 3600)
def _run(season: int, stamp: str, avail_stamp: str, as_of: int | None, results: str):
    """The simulation for a season from a week (None: now), cached by the
    data's stamps and the games finished so far (`results`, so a new final
    redoes it): (SimInput, games left, lineup changes, SimResult, the weeks
    the numbers cover)."""
    pbp = league.season_pbp(season, stamp)
    played_weeks = sorted(int(w) for w in pbp.loc[~pbp["postseason"], "week"].dropna().unique())
    weeks = [w for w in played_weeks if as_of is None or w < as_of]
    off = def_ = None
    if weeks:
        tables = leagues.season_tables(league, season, stamp, (weeks[0], weeks[-1]), "Regular season",
                                       "Opponent-adjusted")
        off = season_qualified(tables["offense"], SEASON_MIN_PLAYS)
        def_ = season_qualified(tables["defense"], SEASON_MIN_PLAYS)
    avail = league.availability(season)
    inp, left, changes = sim.prepare(season, _schedule(season), off, def_, _prior(season - 1),
                                     leagues.load_team_divisions(), model, avail.snaps, avail.injuries,
                                     avail.roster, as_of)
    return inp, left, changes, sim.simulate(inp), (weeks[0], weeks[-1]) if weeks else None


try:
    with st.spinner(f"Playing out the {season} season {sim.SIM_RUNS:,} times…"):
        inp, left, changes, res, span = _run(season, stamp, avail_stamp, as_of,
                                             sim.results_key(sched.loc[sched["result"].notna(), "game_id"]))
except leagues.GameLoadError as e:
    st.title(f"🎲 {season} NFL season simulation")
    st.warning(str(e))
    st.stop()

names = league.team_names(season, stamp)
_, logos, _ = league.team_meta(season, stamp)


def nm(team: str) -> str:
    return names.get(team, team)


teams_df = sim.team_table(res, level)
final = sim.final_standings(sched, leagues.load_team_divisions()) if as_of is not None else None
avail = league.availability(season)

# ---------- Header ----------
st.title(f"🎲 {season} NFL season simulation")
n_left = len(res.left)
weeks_left = (f"weeks {res.left['week'].min()}–{res.left['week'].max()}" if n_left else "")
numbers = (f"the season numbers through week {span[1]}" if span else "last season's numbers alone "
           "(no games played yet)")
st.caption(
    f"{sim.SIM_RUNS:,} runs of the {n_left} regular-season games left ({weeks_left}) and the playoffs"
    if n_left else f"{sim.SIM_RUNS:,} runs of the playoffs")
st.caption(f"From {numbers}, opponent-adjusted"
           + (", blended with last season's" if span else "")
           + (f"; lineups from the injury reports and rosters as of {avail.as_of}"
              if as_of is None and avail.as_of else
              "; lineups from that week's injury reports and rosters" if not avail.injuries.empty else "")
           + f". Ranges hold {level_label} of the runs; a probability's own margin of error is at most "
           f"±{sim.mc_margin(0.5, res.runs, level):.1%} at that level.")

section_nav([("Standings", "standings"), ("Playoffs", "playoffs"), ("Draft order", "draft"),
             ("Team", "team"), ("Games", "games"), ("How it works", "how")])

# ---------- Standings ----------
st.subheader("Projected standings", anchor="standings")
order = st.segmented_control("Order", ["By division", "By playoff odds"], default="By division",
                             key="_sim_order", label_visibility="collapsed") or "By division"


def _standings(conf: str) -> pd.DataFrame:
    d = teams_df[teams_df["Conference"] == conf].copy()
    d = (d.sort_values(["Division", "W"], ascending=[True, False]) if order == "By division"
         else d.sort_values(["Playoffs %", "W"], ascending=False))
    d = d.reset_index(drop=True)
    out = pd.DataFrame({
        "Div": d["Division"].str.split(" ").str[1], "Now": d["Now"],
        "Proj.": [f"{w:.1f}–{l:.1f}" for w, l in zip(d["W"], d["L"])],
        f"Wins ({level_label})": [f"{lo:.0f}–{hi:.0f}" if lo != hi else f"{lo:.0f}"
                                  for lo, hi in zip(d["W low"], d["W high"])],
        "Division": d["Division %"], "Playoffs": d["Playoffs %"],
        "Call": [sim.call(p, level) for p in d["Playoffs %"]],
        ("Bye" if res.seed.max() == 6 else "Top seed"): d["Bye %"],
        "Conf. game": d["Conf. game %"], "Super Bowl": d["Super Bowl %"], "Champion": d["Champion %"],
    })
    out.index = pd.Index([nm(t) for t in d["Team"]], name="Team")
    if final is not None:
        f = final.set_index("Team").reindex(d["Team"])
        out.insert(out.columns.get_loc("Proj.") + 1, "Final", f["Final"].to_numpy())
        out.insert(out.columns.get_loc("Call") + 1, "Made it", np.where(f["Seed"].to_numpy() > 0, "✓", ""))
    return out


for tab, conf in zip(st.tabs(["AFC", "NFC"]), ("AFC", "NFC")):
    with tab:
        stat_table(style_sim_table(_standings(conf), PROBS))
st.caption(
    f"**Proj.** is the average record over the runs, **Wins** the range holding at least {level_label} "
    "of them (whole wins, so usually a little more). "
    f"The percentages are the share of runs a team wins its division, makes the playoffs, gets "
    f"{'a bye (seeds 1–2)' if res.seed.max() == 6 else 'the top seed (the only bye)'}, reaches the "
    "conference championship and the Super Bowl, and wins it. **Call**: likely at "
    f"{level_label}+ of the runs, unlikely at {1 - level:.0%} or less, in or out in every run "
    "(the runs' word, not a mathematical clinch)."
    + (" **Final** and **Made it**: what happened." if final is not None else ""))

# ---------- Playoffs ----------
st.subheader("Playoff picture", anchor="playoffs")
n_seeds = int(res.seed.max())
byes = list(range(1, 9 - n_seeds))
pairs = [(2, 7), (3, 6), (4, 5)] if n_seeds == 7 else [(3, 6), (4, 5)]
for col, conf in zip(st.columns(2), ("AFC", "NFC")):
    with col:
        seeds_t = sim.seed_table(res, conf)
        bracket = sim.projected_seeds(res, conf)
        rows = [(k + 1, t, nm(t), float(seeds_t.at[t, k + 1])) for k, t in enumerate(bracket)]
        games = [(h, a, sim.game_prob(inp, bracket[h - 1], bracket[a - 1])) for h, a in pairs]
        st.markdown(f"**{conf}** · projected first round")
        st.html(bracket_html(rows, games, byes, logos))
st.caption(
    "The most likely bracket: each division's likeliest winner and the likeliest wild cards, seeded "
    "by the assignment with the most runs behind it. Each "
    "team's share of runs at that seed is next to its name, and each wild-card game's winning chances "
    "on the right (the better seed at home). Who ends up where moves a lot in the weeks left: the seed "
    "odds below have every team's chances.")
with st.expander("Seed odds", expanded=False, key="_sim_exp_seeds"):
    for conf in ("AFC", "NFC"):
        t = sim.seed_table(res, conf)
        t = t[t.drop(columns="Out").sum(axis=1) > 0]
        t.index = [nm(x) for x in t.index]
        t.index.name = conf
        t.columns = [str(c) for c in t.columns]  # seeds and "Out": one type for st.table
        stat_table(style_seed_odds(t))
    st.caption("The share of runs each team ends at each seed (1–4: division winners, by record), or "
               "out. Teams in no run's playoffs are left out.")
sb = sim.sb_matchups(res, 5)
st.markdown("**Likeliest Super Bowls**")
stat_table(style_sim_table(pd.DataFrame({
    "AFC": [nm(t) for t in sb["AFC"]], "NFC": [nm(t) for t in sb["NFC"]], "Share": sb["share"],
    "AFC wins": sb["afc_wins"]}), PROBS), hide_index=True)
st.caption("The pairings the runs reach most, the share of runs each is the Super Bowl, and how often "
           "the AFC team wins it then (a neutral site).")

# ---------- Draft order ----------
st.subheader("Draft order", anchor="draft")
dr = sim.draft_table(res, level)
draft_view = pd.DataFrame({
    "Pick": np.arange(1, len(dr) + 1), "Team": [nm(t) for t in dr["Team"]], "Now": dr["Now"],
    "Proj. W": dr["W"], "Avg pick": dr["Avg pick"],
    f"Range ({level_label})": [f"{lo:.0f}–{hi:.0f}" if lo != hi else f"{lo:.0f}"
                               for lo, hi in zip(dr["Pick low"], dr["Pick high"])],
    "Likeliest": dr["Likeliest"], "#1 %": dr["#1 %"], "Top 3 %": dr["Top 3 %"],
    "Top 5 %": dr["Top 5 %"], "Top 10 %": dr["Top 10 %"],
})
if final is not None and final["Pick"].notna().all():
    draft_view.insert(draft_view.columns.get_loc("Likeliest") + 1, "Actual",
                      dr["Team"].map(final.set_index("Team")["Pick"]).astype(int).to_numpy())
stat_table(style_sim_table(draft_view, PROBS, {"Proj. W": 1, "Avg pick": 1}), hide_index=True,
           pin="Team")
st.caption(
    f"The {season + 1} draft's first round, by average pick over the runs: the pick each team's own "
    "record earns (traded picks aren't followed). Teams out of the playoffs pick first, worst record "
    "first, then the playoff teams by the round they go out in; a tie goes to the team with the "
    "weaker schedule (strength of schedule), then a coin flip (the league's division and "
    "conference steps before the coin are left out). **Range** holds "
    f"{level_label} of the runs, **Likeliest** is the most common pick"
    + (", **Actual** the pick the final record earned." if final is not None and final["Pick"].notna().all()
       else "."))

# ---------- Team ----------
st.subheader("Team", anchor="team")
by_name = sorted(res.teams, key=nm)
team = st.selectbox("Team", by_name, format_func=nm, key=f"_sim_team_{season}",
                    index=by_name.index(teams_df.sort_values("Playoffs %").iloc[len(teams_df) // 2]["Team"]),
                    label_visibility="collapsed")
row = teams_df.set_index("Team").loc[team]
# Two cards a row, kept side by side on a phone (views' row_* CSS).
with st.container(key="row_sim_cards"):
    c = st.columns(2, wrap=False)
    c[0].metric("Now", row["Now"])
    c[1].metric("Projected", f"{row['W']:.1f}–{row['L']:.1f}",
                help=f"Wins in {level_label} of the runs: {row['W low']:.0f}–{row['W high']:.0f}")
    c = st.columns(2, wrap=False)
    c[0].metric("Playoffs", prob_text(row["Playoffs %"]).replace("\u2060", ""),
                help=f"Division {prob_text(row['Division %'])} · Super Bowl {prob_text(row['Super Bowl %'])}")
    c[1].metric("Avg draft pick", f"{row['Avg pick']:.1f}",
                help=f"Top 5: {prob_text(row['Top 5 pick %'])}")

dist = sim.wins_distribution(res, team)
fig = go.Figure(go.Bar(x=dist.index, y=dist.to_numpy(), marker_color="#1a9e8f",
                       text=[prob_text(p) for p in dist.to_numpy()], textposition="outside",
                       hovertemplate="%{x:g} wins: %{y:.1%}<extra></extra>"))
fig.update_layout(height=240, margin=dict(l=10, r=10, t=30, b=30), dragmode=False,
                  title=dict(text=f"{nm(team)}: final regular-season wins", font=dict(size=14)),
                  yaxis=dict(visible=False, fixedrange=True), xaxis=dict(dtick=1, fixedrange=True))
st.plotly_chart(fig, config={"displayModeBar": False, "scrollZoom": False}, width="stretch")

lev = sim.leverage(res, team, "playoffs")
mine = (lev[lev["own"]].merge(left[["game_id", "p_home", "mu"]], on="game_id", how="left")
        .sort_values("week").reset_index(drop=True))
if not mine.empty:
    home = mine["home"] == team
    own = pd.DataFrame({
        "Week": mine["week"],
        "Opp": [f"vs {nm(a)}" if h else f"@ {nm(hm)}" for h, a, hm in zip(home, mine["away"], mine["home"])],
        "Win %": np.where(home, mine["p_home"], 1 - mine["p_home"]),
        "Proj. margin": [f"{m:+.1f}" for m in np.where(home, mine["mu"], -mine["mu"])],
        "If win": np.where(home, mine["Home wins"], mine["Away wins"]),
        "If loss": np.where(home, mine["Away wins"], mine["Home wins"]),
    })
    st.markdown(f"**{nm(team)}** · games left, and the playoff chances if it wins or loses each")
    stat_table(style_sim_table(own, PROBS), hide_index=True)
others = lev[~lev["own"]].head(5)
if not others.empty and others["Swing"].iloc[0] >= 0.01:
    root = []
    for r in others.itertuples(index=False):
        pick_home = r._4 >= r._5  # Home wins ≥ Away wins
        root.append({"Week": r.week, "Game": f"{nm(r.away)} @ {nm(r.home)}",
                     "Root for": nm(r.home if pick_home else r.away),
                     "Playoffs if so": max(r._4, r._5), "If not": min(r._4, r._5)})
    st.markdown(f"**Other games that matter most to {nm(team)}**")
    stat_table(style_sim_table(pd.DataFrame(root), PROBS + ["Playoffs if so", "If not"]), hide_index=True)
ch = changes.get(team)
if ch is not None and not ch.empty:
    with st.expander(f"Players gone counted ({len(ch)})", expanded=False, key="_sim_exp_lineup"):
        shown = ch.assign(Fades=np.where(ch["Why"].isin(sim.REPORT_WHY), "by game", "slowly"))
        stat_table(style_sim_table(shown[["Player", "Pos", "Why", "Was", "Change", "Fades"]],
                                   [], {"Was": 2, "Change": 2}), hide_index=True)
        st.caption("Each player's share of the snaps behind the season numbers (Was) that's gone. "
                   "Out and Doubtful players come back game by game at the rate they have since "
                   "2016; reserve lists, releases and trades fade far slower. A team that played "
                   "since its last report keeps it, a game older, until the next one is filed.")
st.caption("If win / If loss: the team's playoff chances in the runs where it wins that game and "
           "in those where it loses it. Root for: the other games whose result moves those "
           "chances most.")

# ---------- Games ----------
st.subheader("Games left", anchor="games")
if left.empty:
    st.caption("No regular-season games left.")
else:
    weeks = sorted(left["week"].unique())
    wk = st.selectbox("Week", weeks, format_func=lambda w: f"Week {w}", key=f"_sim_week_{season}_{as_of}",
                      label_visibility="collapsed")
    g = left[left["week"] == wk].merge(sched[["game_id", "spread_line"]], on="game_id", how="left")
    fav = np.where(g["mu"] >= 0, g["home"], g["away"])
    view = pd.DataFrame({
        "Game": [f"{nm(a)} @ {nm(h)}" + (" (neutral)" if n else "")
                 for a, h, n in zip(g["away"], g["home"], g["neutral"])],
        "Proj.": [f"{nm(t)} by {abs(m):.1f}" for t, m in zip(fav, g["mu"])],
        "Win %": np.maximum(g["p_home"], 1 - g["p_home"]),
        "Line": [("" if pd.isna(s) else "Pick'em" if s == 0 else
                  f"{nm(h) if s > 0 else nm(a)} by {abs(s):g}")
                 for s, h, a in zip(g["spread_line"], g["home"], g["away"])],
    })
    stat_table(style_sim_table(view, PROBS), hide_index=True)
    st.caption("**Proj.** is the model's expected margin and **Win %** the favorite's chance; **Line** "
               "the betting line where nflverse has one (usually the next week or two), for comparison: "
               "the simulation doesn't use it.")

# ---------- How it works ----------
st.subheader("How it works", anchor="how")
fit = model.get("fit", {})
feats = model["features"]
nw, nl = fit.get("next week", {}), fit.get("next week, betting line", {})
with st.expander("The model, its fit and the confidence levels", expanded=False, key="_sim_exp_howto"):
    st.markdown(f"""
**Each game.** The Matchup page's projection for both offenses: the league average plus
half of the offense's and the defense's distances from it, over the season's games so far,
opponent-adjusted. The difference between the two (home less away) of
{', '.join(f"{s['stat']} (× {s['coef']:.1f} points)" for s in feats.values())}
gives the expected margin, plus {model['hfa']:.1f} points of home field (none at a neutral
site). Early in a season it leans on last season's net EPA/play (× {model['prior']['coef']:.1f}):
this season's numbers weigh games / (games + {model['g0']:g}), so after four games they carry
{4 / (4 + model['g0']):.0%} of it and after ten {10 / (10 + model['g0']):.0%}. Those weights were fitted
on every week of {fit.get('seasons', '2016–2025')} ({fit.get('games', 0):,} games, each projected from
every week before it) and picked by how well they did on seasons left out of the fit.

**Who's playing.** The projection moves by the players gone from both lineups, as on the
Matchup page (about {abs(sim.core.LINEUP_EFFECTS['EPA/play']['QB']):.2f} EPA/play less for a starting
QB out): the next game's injury report (Out, Doubtful) and the week's roster (reserve lists,
releases, trades), measured against the players behind the season numbers. Players on the report
come back over the games after it at the rate they have since 2016
({', '.join(f'{v:.0%}' for v in model['decay'].get('report', [])[:4])} still gone 0–3 games
later); roster moves fade far slower ({', '.join(f'{v:.0%}' for v in model['decay'].get('roster', [])[:4])}).
Out of sample the lineups made the margins a little better (log loss 0.655 → 0.653, winners picked
61.6% → 62.0% over every week and game ahead) and last season's ratings a lot (0.663 without them).

**The runs.** Each run draws every team's rating error once, for all its games (how far its
true strength can be from its numbers: {sim.team_sd(model, np.array([0]))[0]:.1f} points before a
game, {sim.team_sd(model, np.array([8]))[0]:.1f} after eight, {sim.team_sd(model, np.array([16]))[0]:.1f}
after sixteen), and every game's own noise ({model['game_sd']:.1f} points),
so a team that's better than its numbers in a run is better all season, as real seasons go.
Simulated games don't end tied. Then the standings: division winners and wild cards by the NFL's
tiebreakers (head-to-head, division record, common games, conference record, strength of victory
and of schedule; the points steps are left out, so a tie past those is a coin flip; on 2016–2025's
final results it gives every season's real seeds), the playoffs game by game with the better seed
at home and the Super Bowl at a neutral site, and the draft order.

**How good is it.** The win probabilities, out of sample, for the games of the week after the
numbers: log loss {nw.get('log loss', float('nan')):.3f}, Brier {nw.get('Brier', float('nan')):.3f},
winner picked {nw.get('picked', float('nan')):.1%}; the betting line on the same games:
{nl.get('log loss', float('nan')):.3f}, {nl.get('Brier', float('nan')):.3f},
{nl.get('picked', float('nan')):.1%}. The line knows more (news, weather, rest) and it isn't used here.

**Confidence.** Every percentage is a share of the {sim.SIM_RUNS:,} runs, so it carries its own
margin of error: at {level_label}, at most ±{sim.mc_margin(0.5, res.runs, level):.1%} (a 50% chance),
±{sim.mc_margin(0.1, res.runs, level):.1%} for a 10% one. The ranges (wins, draft pick) hold at least
{level_label} of the runs. Whether the runs' spread is right was checked on 2016–2025
(`tools/validate_season_sim.py`: each season simulated from weeks 1, 5, 9 and 13 with a game model
fitted without it). Of the teams given a 20–40% playoff chance, 30% made it; 40–60%: 51%; 60–80%: 72%;
80–95%: 87%. Final win totals landed in the 90% ranges 92–96% of the time, about as often as the runs
themselves put them there (whole-win ranges hold a bit more than the level), and in the 80% ranges
83–91%. From week 9 the projected wins missed by 1.5 on average (1.7 for each team's pace so far),
from week 13 by 1.0. The numbers know nothing of weather, rest or a coaching change, and a team's
numbers don't move within a run as its season goes.
""")
