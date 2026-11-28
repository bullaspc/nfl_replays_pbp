"""
The Season simulation page's college football view: the rest of an FBS
season played out `season_sim.SIM_RUNS` times, through the conference title
games and the College Football Playoff.

Each game left gets an expected margin from each team's power rating, in
points against an average FBS team: this season's opponent-adjusted EPA/play
and points per drive (the Matchup page's numbers, as `season_sim` uses them
for the NFL), blended early on with last season's, plus the home field. A
game against an FCS team takes the FBS team's rating against the FCS
average. The weights, the home field and the spreads are fitted by
tools/build_cfb_sim_model.py on 2016–2025 (`MODEL_FILE`).

A run draws each team's rating error once and each game's noise, then:
the conference standings and each conference's title game (the top two by
conference record, division winners in the Sun Belt), the committee's
final ranking, the 12-team field and its bracket. The committee is a model
(`committee_scores`): a score from the record, the power rating, the
schedule, a conference title and a Power 4 schedule, fitted on the
committee's final rankings of 2016–2025, plus a draw for what the model
doesn't see, so the same record can land a few places apart from run to run.

Bowls: once the matchups are announced (Selection Day), the schedule's
games are played out; before that (and simulating from an earlier week), each
run fills the season's bowl slots with the bowl-eligible teams outside the
playoff (six wins, one FCS win at most; 5–7 teams fill in when short), each
paired with the next team in the committee's order from another conference.
A bowl's margin is the model's `postseason.bowl.shrink` of the rating
difference with its own spread: bowls, with players opting out, have gone
by about half of it since 2016. Once the playoff field is announced, its seeds
are the schedule's, not the committee model's.

The field follows each season's rules (`cfp_format`): 2024, the five
highest-ranked conference champions in and the top four champions seeded
1–4; 2025, the same five champions and seeds straight by ranking; from 2026,
the four Power 4 champions and the highest-ranked Group of 6 team, seeds
straight by ranking. Seeds 1–4 get byes, 5–8 host 9–12, and the bracket is
fixed after that.

No Streamlit and not in `leagues._CODE_FILES`, like season_sim.py.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.special import ndtr

import replay_core as core
import season_sim as sim

MODEL_FILE = Path(__file__).parent / "models" / "season_sim_cfb.json"
FIRST_SEASON = 2024  # the 12-team playoff; earlier seasons had divisions and four teams
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; nfl_replays_pbp)"}
STANDINGS_URL = "https://site.api.espn.com/apis/v2/sports/football/college-football/standings"
RANKINGS_URL = ("https://sports.core.api.espn.com/v2/sports/football/leagues/college-football/"
                "seasons/{season}/types/2/weeks/{week}/rankings/21")  # 21: the CFP committee's
LATEST_RANKINGS_URL = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/rankings"
# ESPN's conference abbreviations → the names the page shows.
CONF_NAMES = {"sec": "SEC", "big10": "Big Ten", "big12": "Big 12", "acc": "ACC", "American": "American",
              "usa": "Conference USA", "midam": "MAC", "mwest": "Mountain West", "pac12": "Pac-12",
              "belt": "Sun Belt", "ind": "Independent"}
POWER4 = {"SEC", "Big Ten", "Big 12", "ACC"}
GROUP6 = {"American", "Conference USA", "MAC", "Mountain West", "Pac-12", "Sun Belt"}
# Title games at a neutral site; the others are hosted by the better-placed team.
NEUTRAL_TITLE = POWER4 | {"MAC"}
TITLE_MIN_TEAMS = 6   # a conference this big plays a title game (the two-team Pac-12 of 2024–25 didn't)
BOWL_WINS = 6
FCS = "FCS"
NEW_TEAM_PRIOR_Q = 0.1  # a team new to FBS gets this quantile of last season's ratings
# How far a team got in the playoff: out, then lost in each round, then won it.
STAGES = ["Out", "First round", "Quarterfinal", "Semifinal", "Final", "Champion"]
_TITLE_GAME = re.compile(r"\bChampionship\b")


def load_model(path: Path = MODEL_FILE) -> dict | None:
    """The fitted college model (tools/build_cfb_sim_model.py), or None."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


@dataclass(frozen=True)
class CFPFormat:
    teams: int              # in the field
    champions_in: int       # the highest-ranked conference champions in (2024–25), 0 from 2026
    power4_in: bool         # the four Power 4 champions in, plus the top Group of 6 team (2026 on)
    champion_byes: bool     # seeds 1–4 to the four highest-ranked champions (2024)


def cfp_format(season: int) -> CFPFormat:
    """The playoff's selection rules for `season` (2024 on)."""
    if season <= 2024:
        return CFPFormat(12, 5, False, True)
    if season == 2025:
        return CFPFormat(12, 5, False, False)
    return CFPFormat(12, 0, True, False)


def format_note(season: int) -> str:
    f = cfp_format(season)
    if f.power4_in:
        return ("12 teams: the SEC, Big Ten, Big 12 and ACC champions and the highest-ranked Group of 6 "
                "team in, seven at-large; seeds straight by the final ranking, 1–4 get byes")
    if f.champion_byes:
        return ("12 teams: the five highest-ranked conference champions in, seven at-large; the four "
                "highest-ranked champions seeded 1–4 with byes, the rest by ranking")
    return ("12 teams: the five highest-ranked conference champions in, seven at-large; seeds "
            "straight by the final ranking, 1–4 get byes")


# ---------- ESPN: conferences and the committee's rankings ----------
def _get_json(url: str, params: dict | None = None, timeout: float = 20) -> dict:
    r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def read_conferences(season: int) -> pd.DataFrame:
    """Each FBS team's conference (and division, where the conference plays
    in divisions) for `season`, from ESPN's standings: team_id, abbr,
    conference, division (None without)."""
    d = _get_json(STANDINGS_URL, {"season": season, "group": 80})
    rows = []
    for c in d.get("children", []):
        conf = CONF_NAMES.get(c.get("abbreviation"), c.get("abbreviation") or c.get("name"))
        groups = c.get("children") or [c]
        for g in groups:
            div = g.get("name") if g is not c else None
            if div:
                div = div.split(" - ")[-1]
            for e in g.get("standings", {}).get("entries", []):
                t = e.get("team", {})
                rows.append({"team_id": int(t["id"]), "abbr": t.get("abbreviation"),
                             "conference": conf, "division": div})
    return pd.DataFrame(rows, columns=["team_id", "abbr", "conference", "division"])


def _ranks_from(entries) -> pd.Series:
    out = {}
    for x in entries:
        team = x.get("team", {})
        tid = team.get("id")
        if tid is None and "$ref" in team:
            m = re.search(r"/teams/(\d+)", team["$ref"])
            tid = m.group(1) if m else None
        if tid is not None:
            out[int(tid)] = int(x["current"])
    return pd.Series(out, dtype=int)


def read_cfp_ranking(season: int, week: int | None = None) -> tuple[pd.Series, str]:
    """The committee's ranking (ESPN team id → rank 1–25) and its label
    ("Week 13"): of `week`, else the latest one out. Empty before the first."""
    if week is None:
        d = _get_json(LATEST_RANKINGS_URL)
        for r in d.get("rankings", []):
            if r.get("type") == "cfp" and str(r.get("season", {}).get("year", season)) == str(season):
                return _ranks_from(r.get("ranks", [])), r.get("headline", "").rsplit(" - ", 1)[-1]
        return pd.Series(dtype=int), ""
    try:
        d = _get_json(RANKINGS_URL.format(season=season, week=week))
    except requests.HTTPError:
        return pd.Series(dtype=int), ""
    return _ranks_from(d.get("ranks", [])), f"Week {week}"


def final_cfp_ranking(season: int) -> pd.Series:
    """The committee's last ranking of `season` (the one that picks the
    field): ESPN team id → rank."""
    for week in range(18, 9, -1):
        ranks, _ = read_cfp_ranking(season, week)
        if not ranks.empty:
            return ranks
    return pd.Series(dtype=int)


def ranking_before(season: int, week: int) -> tuple[pd.Series, str]:
    """The committee's latest ranking out before `week`'s games."""
    for w in range(week - 1, 9, -1):
        ranks, label = read_cfp_ranking(season, w)
        if not ranks.empty:
            return ranks, label
    return pd.Series(dtype=int), ""


# ---------- The schedule ----------
def schedule_games(sched: pd.DataFrame, fbs_ids: set[int], as_of: int | None = None) -> pd.DataFrame:
    """The regular season's games with an FBS team (and both teams known),
    title games apart: game_id, week, home_id, away_id (ESPN ids), neutral,
    conference (a conference game counts in the standings), title (the
    conference whose title game it is, "" for others), played, home_margin.
    With `as_of`, games from that week on count as not played yet."""
    s = sched[sched["season_type"].astype(str).str.startswith("reg")]
    s = s[s["home_id"].notna() & s["away_id"].notna()]
    s = s[s["home_id"].astype(int).isin(fbs_ids) | s["away_id"].astype(int).isin(fbs_ids)]
    notes = s["notes"].fillna("").astype(str)
    title = np.where(notes.str.contains(_TITLE_GAME) & ~notes.str.contains("Division|FCS"),
                     s["home_conference"].fillna(""), "")
    played = s["completed"].eq(True) & s["home_points"].notna()
    if as_of is not None:
        played &= s["week"] < as_of
    return pd.DataFrame({
        "game_id": s["game_id"].astype(str).to_numpy(), "week": s["week"].astype(int).to_numpy(),
        "home_id": s["home_id"].astype(int).to_numpy(), "away_id": s["away_id"].astype(int).to_numpy(),
        "neutral": s["neutral_site"].eq(True).to_numpy(), "conference": s["conference_game"].eq(True).to_numpy(),
        "title": title, "played": played.to_numpy(),
        "home_margin": np.where(played, s["home_points"] - s["away_points"], np.nan),
    }).sort_values(["week", "game_id"]).reset_index(drop=True)


def playoff_results(sched: pd.DataFrame, ids: dict[int, str], as_of: int | None = None) -> dict:
    """The playoff games played (none when simulating from a regular-season
    week): the two teams → the winner."""
    if as_of is not None:
        return {}
    s = sched[sched["playoff_round_name"].notna() & sched["completed"].eq(True)
              & sched["home_points"].notna()] if "playoff_round_name" in sched.columns else sched.iloc[0:0]
    out = {}
    for h, a, hp, ap in zip(s["home_id"], s["away_id"], s["home_points"], s["away_points"]):
        if pd.notna(h) and pd.notna(a) and int(h) in ids and int(a) in ids and hp != ap:
            out[frozenset((ids[int(h)], ids[int(a)]))] = ids[int(h)] if hp > ap else ids[int(a)]
    return out


def postseason_games(sched: pd.DataFrame, ids: dict[int, str], idx: dict[str, int],
                     as_of: int | None = None) -> tuple[pd.DataFrame, dict[int, int]]:
    """Once Selection Day has set them (and only when simulating from now):
    the bowls outside the playoff with both teams FBS (game_id, bowl, date,
    home, away as team indices, neutral, played, home_margin), and the
    playoff's seeds (team index → seed) when the schedule lists all twelve."""
    cols = ["game_id", "bowl", "date", "kickoff", "home", "away", "neutral", "played", "home_margin"]
    if as_of is not None or sched.empty or "season_type" not in sched.columns:
        return pd.DataFrame(columns=cols), {}
    post = sched[sched["season_type"].astype(str).str.startswith("post")
                 & sched["home_id"].notna() & sched["away_id"].notna()]
    notes = post["notes"].fillna("").astype(str)
    cfp = (post["playoff_round_name"].notna() if "playoff_round_name" in post.columns
           else pd.Series(False, index=post.index)) | notes.str.contains("College Football Playoff")
    seeds = {}
    for side in ("home", "away"):
        col = f"playoff_{side}_seed"
        if col in post.columns:
            for tid, sd in zip(post.loc[cfp, f"{side}_id"], post.loc[cfp, col]):
                t = idx.get(ids.get(int(tid)))
                if t is not None and pd.notna(sd):
                    seeds[t] = int(sd)
    if sorted(seeds.values()) != list(range(1, 13)):
        seeds = {}
    b = post[~cfp]
    home = b["home_id"].astype(int).map(ids).map(idx)
    away = b["away_id"].astype(int).map(ids).map(idx)
    keep = home.notna() & away.notna()
    b = b[keep]
    played = b["completed"].eq(True) & b["home_points"].notna()
    bowls = pd.DataFrame({
        "game_id": b["game_id"].astype(str).to_numpy(), "bowl": b["notes"].fillna("Bowl").to_numpy(),
        "date": [("" if pd.isna(d) else f"{d:%a %b} {d.day}")
                 for d in pd.to_datetime(b["start_date"], utc=True, errors="coerce").dt.tz_convert("America/New_York")],
        "kickoff": pd.to_datetime(b["start_date"], utc=True, errors="coerce").to_numpy(),
        "home": home[keep].astype(int).to_numpy(), "away": away[keep].astype(int).to_numpy(),
        "neutral": b["neutral_site"].eq(True).to_numpy(), "played": played.to_numpy(),
        "home_margin": np.where(played, b["home_points"] - b["away_points"], np.nan),
    }, columns=cols)
    return bowls.sort_values(["kickoff", "game_id"]).reset_index(drop=True), seeds


def bowl_slots(sched: pd.DataFrame, previous: pd.DataFrame | None = None) -> int:
    """The season's bowl games outside the playoff: the schedule's, else (not
    announced yet) last season's."""
    for s in (sched, previous):
        if s is None or s.empty or "season_type" not in s.columns:
            continue
        post = s[s["season_type"].astype(str).str.startswith("post") & s["fbs_participant"].fillna(False)]
        notes = post["notes"].fillna("").astype(str)
        cfp = (post["playoff_round_name"].notna() if "playoff_round_name" in post.columns
               else pd.Series(False, index=post.index)) | notes.str.contains("College Football Playoff")
        n = int((~cfp).sum())
        if n:
            return n
    return 0


def new_to_fbs(conferences: pd.DataFrame, earlier: list[pd.DataFrame]) -> set[int]:
    """ESPN ids of the teams moving up from FCS (`read_conferences` of the
    season and of the two before it): FBS now but not in each of those two
    seasons. The NCAA gives them a bowl only when there aren't enough
    eligible teams."""
    before = [set(e["team_id"]) for e in earlier if e is not None and not e.empty]
    if not before:
        return set()
    always = set.intersection(*before)
    return set(conferences.loc[~conferences["team_id"].isin(always), "team_id"].astype(int))


def bowl_model(model: dict) -> tuple[float, float]:
    """A bowl's (shrink, spread): how much of the rating difference its
    margin follows, and its own noise (tools/build_cfb_sim_model.py's
    postseason check); the regular season's without it."""
    b = model.get("postseason", {}).get("bowl", {})
    return float(b.get("shrink", 1.0)), float(b.get("sd", model["game_sd"]))


# ---------- Ratings ----------
def team_features(off: pd.DataFrame, def_: pd.DataFrame, model: dict) -> pd.DataFrame:
    """Each team's part of the model's features (`model["features"]`):
    `PROJ_SHRINK` × (its offense's value less what its defense allowed),
    sign flipped where less is better. A game's difference in Proj. (the
    Matchup page's, `season_sim.proj_matrix`) is the home team's less the
    away team's."""
    out = pd.DataFrame(index=off.index)
    for f, spec in model["features"].items():
        stat, sign = spec["stat"], spec.get("sign", 1)
        if stat in off.columns and stat in def_.columns:
            out[f] = sign * core.PROJ_SHRINK * (off[stat].astype(float) - def_[stat].reindex(off.index).astype(float))
    return out


def power_ratings(feats: pd.DataFrame | None, prior: pd.Series | None, games: pd.Series,
                  teams: list[str], model: dict) -> pd.Series:
    """Each team's power rating: its expected margin (points) against an
    average FBS team at a neutral site, from this season's numbers weighted by
    games / (games + g0) and last season's net EPA/play by the rest."""
    g = games.reindex(teams).fillna(0).to_numpy(float)
    w = sim.data_weight(g, model["g0"])
    cur = np.zeros(len(teams))
    if feats is not None and not feats.empty:
        f = feats.reindex(teams)
        cur = sum(model["features"][c]["coef"] * f[c].fillna(0.0).to_numpy() for c in model["features"]
                  if c in f.columns)
    # A team new to FBS (no rating last season) starts near the bottom.
    p = (prior.reindex(teams).fillna(prior.quantile(NEW_TEAM_PRIOR_Q)).to_numpy()
         if prior is not None and len(prior) else np.zeros(len(teams)))
    return pd.Series(w * cur + (1 - w) * model["prior"]["coef"] * p, index=teams)


# ---------- The committee ----------
COMMITTEE_FEATURES = ["wins", "losses", "power", "sos", "champion", "power4"]


def committee_features(wins, losses, power, sos, champion, power4) -> np.ndarray:
    """Stacked (…, len(COMMITTEE_FEATURES)) feature array, in that order."""
    return np.stack([np.asarray(x, float) for x in (wins, losses, power, sos, champion, power4)], axis=-1)


def committee_scores(x: np.ndarray, model: dict) -> np.ndarray:
    """The committee model's score (higher ranks higher) for features `x`
    (`committee_features`), before its draw."""
    c = model["committee"]
    beta = np.array([c["coef"][f] for f in COMMITTEE_FEATURES])
    mu, sd = np.array(c["mean"]), np.array(c["sd"])
    return ((x - mu) / sd) @ beta


# ---------- The simulation ----------
@dataclass
class CollegeInput:
    season: int
    teams: list[str]              # FBS teams
    conference: list[str]         # per team
    division: list[str | None]    # per team (Sun Belt), else None
    power: np.ndarray             # per team, points over an average FBS team
    team_sd: np.ndarray           # per team, its rating error's SD
    games: pd.DataFrame           # schedule_games() with home/away team indices (-1: FCS) and mu
    hfa: float
    game_sd: float
    fcs: dict                     # the FCS model: intercept, slope, sd
    model: dict
    fixed_title: dict = field(default_factory=dict)      # conference → (team a, team b) already set
    title_results: dict = field(default_factory=dict)    # conference → winner index (played)
    playoff_results: dict = field(default_factory=dict)  # frozenset({i, j}) → winner index
    bowl_slots: int = 0                                   # bowl games outside the playoff
    bowls: pd.DataFrame | None = None                     # the bowls announced (`postseason_games`), else None
    fixed_seeds: dict = field(default_factory=dict)       # team index → seed, once the field is announced
    transition: set = field(default_factory=set)          # indices of teams up from FCS in the last two seasons


@dataclass
class CollegeResult:
    teams: list[str]
    conference: list[str]
    runs: int
    wins: np.ndarray        # runs × T: wins (regular season and title game)
    losses: np.ndarray
    conf_wins: np.ndarray   # runs × T: conference wins
    title_game: np.ndarray  # runs × T bool
    champion: np.ndarray    # runs × T bool: won its conference
    rank: np.ndarray        # runs × T: the committee's final rank (1 = first)
    seed: np.ndarray        # runs × T: playoff seed, 0 = out
    stage: np.ndarray       # runs × T: index into STAGES
    outcomes: np.ndarray    # runs × games left: True where the home team won
    left: pd.DataFrame      # the games left (team names), in `outcomes`' order
    played_wins: np.ndarray
    played_losses: np.ndarray
    played_conf: np.ndarray  # conference record so far: wins
    played_conf_losses: np.ndarray
    conf_games: np.ndarray   # conference games in all
    eligible: np.ndarray     # runs × T bool: six wins (one FCS win counted at most)
    bowl_game: np.ndarray    # runs × T bool: plays a bowl outside the playoff
    bowl_win: np.ndarray     # runs × T bool: and wins it
    final_wins: np.ndarray   # runs × T: wins with the bowl or playoff games
    final_losses: np.ndarray
    bowls: pd.DataFrame | None = None  # the bowls announced, with each one's chances


def prepare(season: int, sched: pd.DataFrame, conferences: pd.DataFrame, ids: dict[int, str],
            off: pd.DataFrame | None, def_: pd.DataFrame | None, prior: pd.Series | None, model: dict,
            as_of: int | None = None, bowl_slots: int = 0,
            transition: set[int] = frozenset()) -> tuple[CollegeInput, pd.DataFrame]:
    """The simulation's input: the FBS teams (`conferences`, ESPN's), their
    power ratings from the season tables (`off` / `def_`, the FBS teams',
    opponent-adjusted; None before any game) and last season's (`prior`),
    and the schedule (`ids`: ESPN id → the play-by-play's abbreviation),
    with `bowl_slots` bowl games to fill before the matchups are out
    (`transition`: ESPN ids of teams still moving up from FCS, `new_to_fbs`). Also the
    games left with their expected margins and win chances."""
    conf = conferences.assign(abbr=conferences["team_id"].map(ids).fillna(conferences["abbr"]))
    teams = list(conf["abbr"])
    idx = {t: i for i, t in enumerate(teams)}
    fbs_ids = set(conf["team_id"])
    games = schedule_games(sched, fbs_ids, as_of)
    games["home"] = games["home_id"].map(ids).map(idx).fillna(-1).astype(int)
    games["away"] = games["away_id"].map(ids).map(idx).fillna(-1).astype(int)
    games = games[(games["home"] >= 0) | (games["away"] >= 0)].reset_index(drop=True)
    # Title games are simulated from the standings; one already set or played is kept.
    titles = games[games["title"] != ""]
    games = games[games["title"] == ""].reset_index(drop=True)
    played = games[games["played"]]
    counts = (pd.concat([played.loc[played["home"] >= 0, "home"], played.loc[played["away"] >= 0, "away"]])
              .map(lambda i: teams[i]).value_counts())
    feats = team_features(off, def_, model) if off is not None and def_ is not None else None
    power = power_ratings(feats, prior, counts, teams, model)
    tau = sim.team_sd(model, counts.reindex(teams).fillna(0).to_numpy())
    p = power.to_numpy()
    fcs = model["fcs"]
    hfa = model["hfa"]

    def mu_of(h, a, neutral):
        home_adv = 0.0 if neutral else hfa
        if h >= 0 and a >= 0:
            return p[h] - p[a] + home_adv
        if h >= 0:
            return fcs["intercept"] + fcs["slope"] * p[h] + home_adv
        return -(fcs["intercept"] + fcs["slope"] * p[a]) + home_adv

    games["mu"] = [mu_of(h, a, n) for h, a, n in zip(games["home"], games["away"], games["neutral"])]
    # A title game the schedule has set: its teams (only once known: from an
    # earlier week, they weren't) and, once played, its winner.
    fixed, results = {}, {}
    for r in titles.itertuples(index=False):
        if r.home < 0 or r.away < 0 or (as_of is not None and r.week >= as_of):
            continue
        c = conf["conference"].iloc[r.home]
        fixed[c] = (r.home, r.away)
        if r.played:
            results[c] = r.home if r.home_margin > 0 else r.away
    bowls, seeds = postseason_games(sched, ids, idx, as_of)
    if not bowls.empty:  # each announced bowl's chances, as the runs play it
        shrink, sd_b = bowl_model(model)
        mu = shrink * (p[bowls["home"]] - p[bowls["away"]] + np.where(bowls["neutral"], 0.0, hfa))
        bowls["mu"] = mu
        bowls["p_home"] = ndtr(mu / np.sqrt(sd_b ** 2 + shrink ** 2 * (tau[bowls["home"]] ** 2
                                                                         + tau[bowls["away"]] ** 2)))
        bowls["home_team"] = [teams[i] for i in bowls["home"]]
        bowls["away_team"] = [teams[i] for i in bowls["away"]]
    inp = CollegeInput(season, teams, list(conf["conference"]), list(conf["division"].where(conf["division"].notna(), None)),
                       p, tau, games, hfa, model["game_sd"], fcs, model, fixed, results,
                       {frozenset(idx[t] for t in k): idx[w] for k, w in playoff_results(sched, ids, as_of).items()
                        if all(t in idx for t in k)},
                       bowl_slots if bowls.empty else len(bowls), None if bowls.empty else bowls, seeds,
                       {idx[ids[i]] for i in transition if ids.get(i) in idx})
    left = games[~games["played"]].copy()
    # A game's spread: its own noise (an FCS game's is wider) and both FBS teams' rating errors.
    fcs_game = (left["home"] < 0) | (left["away"] < 0)
    t_h = np.where(left["home"] >= 0, tau[left["home"].clip(lower=0)], 0.0)
    t_a = np.where(left["away"] >= 0, tau[left["away"].clip(lower=0)], 0.0)
    sd = np.sqrt(np.where(fcs_game, fcs["sd"], model["game_sd"]) ** 2 + t_h ** 2 + t_a ** 2)
    left["p_home"] = ndtr(left["mu"] / sd)
    left["home_team"] = [teams[i] if i >= 0 else FCS for i in left["home"]]
    left["away_team"] = [teams[i] if i >= 0 else FCS for i in left["away"]]
    return inp, left.reset_index(drop=True)


_EPS = 1e-9


def _conf_rank(group: list[int], run: dict) -> int:
    """The team that wins a tie in its conference's standings (same
    conference record): head-to-head among the tied (when each pair played),
    record against the conference opponents they all played, the combined
    conference record of their conference opponents, then the team rating
    (the computer rankings the conferences use last), then a coin flip; the
    rest start over when a step leaves some behind, as the conferences'
    rules do."""
    group = list(group)
    while len(group) > 1:
        for step in (_h2h_cfb, _common_cfb, _opp_conf_pct, _rating):
            vals = step(group, run)
            if vals is None:
                continue
            top = max(vals)
            kept = [t for t, v in zip(group, vals) if v >= top - _EPS]
            if len(kept) < len(group):
                group = kept
                break
        else:
            return min(group, key=lambda t: run["coin"][t])
    return group[0]


def _record(run: dict, team: int, opps) -> tuple[float, int]:
    c = n = 0
    for o in opps:
        for k, home in run["conf_games"][team].get(o, ()):
            w = run["credit"][k]
            c += w if home else 1.0 - w
            n += 1
    return c, n


def _h2h_cfb(group, run):
    vals = []
    for t in group:
        c, n = _record(run, t, [o for o in group if o != t])
        if n < len(group) - 1:  # not everyone played everyone
            return None
        vals.append(c / n)
    return vals


def _common_cfb(group, run):
    common = set.intersection(*(set(run["conf_games"][t]) for t in group)) - set(group)
    if not common:
        return None
    vals = []
    for t in group:
        c, n = _record(run, t, common)
        vals.append(c / n if n else 0.0)
    return vals


def _opp_conf_pct(group, run):
    return [np.mean([run["conf_pct"][o] for o in run["conf_games"][t]]) if run["conf_games"][t] else 0.0
            for t in group]


def _rating(group, run):
    return [run["strength"][t] for t in group]


def _order_conf(teams: list[int], run: dict, need: int) -> list[int]:
    """The conference's (or division's) top `need` teams: by conference
    record, ties by `_conf_rank`."""
    by = sorted(teams, key=lambda t: -run["conf_pct"][t])
    out, i = [], 0
    while i < len(by) and len(out) < need:
        j = i
        while j + 1 < len(by) and abs(run["conf_pct"][by[j + 1]] - run["conf_pct"][by[i]]) < _EPS:
            j += 1
        group = by[i:j + 1]
        while group and len(out) < need:
            b = group[0] if len(group) == 1 else _conf_rank(group, run)
            out.append(b)
            group.remove(b)
        i = j + 1
    return out


def simulate(inp: CollegeInput, runs: int = sim.SIM_RUNS, seed: int = sim.SIM_SEED) -> CollegeResult:
    """The season played out `runs` times from `inp`."""
    rng = np.random.default_rng(seed)
    T = len(inp.teams)
    g = inp.games
    h, a = g["home"].to_numpy(), g["away"].to_numpy()
    u = rng.normal(0.0, 1.0, (runs, T)) * inp.team_sd[None, :]
    strength = inp.power[None, :] + u   # each run's true team strengths
    left = ~g["played"].to_numpy()
    # The home team's credit (1 win, 0 loss) per game and run.
    credit = np.tile(np.where(g["played"], (g["home_margin"] > 0).astype(float), 0.0), (runs, 1))
    if left.any():
        hl, al = h[left], a[left]
        uh = np.where(hl >= 0, u[:, hl.clip(min=0)], 0.0)
        ua = np.where(al >= 0, u[:, al.clip(min=0)], 0.0)
        fcs_game = (hl < 0) | (al < 0)
        sd = np.where(fcs_game, inp.fcs["sd"], inp.game_sd)
        m = g["mu"].to_numpy()[left][None, :] + uh - ua + rng.normal(0.0, 1.0, (runs, left.sum())) * sd[None, :]
        outcomes = m > 0
        credit[:, left] = outcomes
    else:
        outcomes = np.zeros((runs, 0), dtype=bool)

    # Records: one-hot of each FBS side.
    G = len(g)
    H = np.zeros((G, T), np.float32)
    A = np.zeros((G, T), np.float32)
    H[np.flatnonzero(h >= 0), h[h >= 0]] = 1
    A[np.flatnonzero(a >= 0), a[a >= 0]] = 1
    cr = credit.astype(np.float32)
    wins = cr @ H + (1 - cr) @ A
    losses = (1 - cr) @ H + cr @ A
    conf_mask = g["conference"].to_numpy() & (h >= 0) & (a >= 0)
    conf_w = cr[:, conf_mask] @ H[conf_mask] + (1 - cr[:, conf_mask]) @ A[conf_mask]
    conf_n = H[conf_mask].sum(0) + A[conf_mask].sum(0)
    conf_pct = conf_w / np.maximum(conf_n, 1)

    # Each team's conference games, for the tiebreakers.
    conf_games: list[dict[int, list[tuple[int, bool]]]] = [{} for _ in range(T)]
    for k in np.flatnonzero(conf_mask):
        conf_games[h[k]].setdefault(a[k], []).append((k, True))
        conf_games[a[k]].setdefault(h[k], []).append((k, False))
    by_conf: dict[str, dict[str | None, list[int]]] = {}
    for t, (c, d) in enumerate(zip(inp.conference, inp.division)):
        by_conf.setdefault(c, {}).setdefault(d, []).append(t)
    titled = {c: groups for c, groups in by_conf.items()
              if c != "Independent" and sum(len(v) for v in groups.values()) >= TITLE_MIN_TEAMS}

    coin = rng.random((runs, T))
    title_game = np.zeros((runs, T), bool)
    champion = np.zeros((runs, T), bool)
    conf_pct_l, coin_l, str_l = conf_pct.tolist(), coin.tolist(), strength.tolist()
    pairs = {c: np.zeros((runs, 2), int) for c in titled}
    for s in range(runs):
        run = {"conf_pct": conf_pct_l[s], "coin": coin_l[s], "strength": str_l[s], "credit": credit[s],
               "conf_games": conf_games}
        for c, groups in titled.items():
            if c in inp.fixed_title:
                pairs[c][s] = inp.fixed_title[c]
            elif len(groups) > 1:  # division winners, the better one first
                heads = [_order_conf(members, run, 1)[0] for members in groups.values()]
                pairs[c][s] = _order_conf(heads, run, 2)
            else:
                pairs[c][s] = _order_conf(next(iter(groups.values())), run, 2)
    rows = np.arange(runs)
    for c, pr in pairs.items():
        t1, t2 = pr[:, 0], pr[:, 1]
        title_game[rows, t1] = True
        title_game[rows, t2] = True
        if c in inp.title_results:
            win = np.full(runs, inp.title_results[c])
        else:
            home_adv = 0.0 if c in NEUTRAL_TITLE else inp.hfa
            m = (strength[rows, t1] - strength[rows, t2] + home_adv
                 + rng.normal(0.0, inp.game_sd, runs))
            win = np.where(m > 0, t1, t2)
        lose = np.where(win == t1, t2, t1)
        wins[rows, win] += 1
        losses[rows, lose] += 1
        champion[rows, win] = True

    # The committee's final ranking: its score from each run's résumé, plus its draw.
    opp_str = np.zeros((runs, T))
    n_opp = np.zeros(T)
    fcs_power = inp.fcs["intercept"] * -1.0  # an FCS team's rating: what an average FBS team beats it by, negated
    for k in range(G):
        hk, ak = h[k], a[k]
        if hk >= 0:
            opp_str[:, hk] += inp.power[ak] if ak >= 0 else fcs_power
            n_opp[hk] += 1
        if ak >= 0:
            opp_str[:, ak] += inp.power[hk] if hk >= 0 else fcs_power
            n_opp[ak] += 1
    sos = opp_str / np.maximum(n_opp, 1)
    power4 = np.array([c in POWER4 or t == "ND" for c, t in zip(inp.conference, inp.teams)], float)
    x = committee_features(wins, losses, strength, sos, champion, np.broadcast_to(power4, (runs, T)))
    score = committee_scores(x, inp.model) + rng.gumbel(0.0, 1.0, (runs, T))
    order = np.argsort(-score, axis=1)
    rank = np.empty((runs, T), np.int16)
    rank[rows[:, None], order] = np.arange(1, T + 1, dtype=np.int16)[None, :]

    if inp.fixed_seeds:  # the field Selection Day announced
        seed = np.zeros((runs, T), np.int8)
        for t, k in inp.fixed_seeds.items():
            seed[:, t] = k
    else:
        seed = _select_field(inp, rank, champion)
    stage = _play_bracket(inp, seed, strength, rng)

    # Bowls: six wins, counting one win over an FCS team at most.
    fh, fa = (h >= 0) & (a < 0), (a >= 0) & (h < 0)
    fcs_wins = cr[:, fh] @ H[fh] + (1 - cr[:, fa]) @ A[fa]
    counted = wins - np.maximum(fcs_wins - 1, 0)
    eligible = counted >= BOWL_WINS
    bowl_game, bowl_win = _play_bowls(inp, rank, seed, counted, strength, rng)
    in_field = seed > 0
    cfp_w = np.where(in_field, stage - np.where(seed <= 4, 2, 1), 0).clip(min=0)
    cfp_l = (in_field & (stage < 5)).astype(np.float32)
    final_wins = wins + bowl_win + cfp_w
    final_losses = losses + (bowl_game & ~bowl_win) + cfp_l
    played = g[g["played"]]

    def tally(frame, mask=None):
        w = np.zeros(T); l = np.zeros(T)
        f = frame if mask is None else frame[mask]
        for hh, aa, mg in zip(f["home"], f["away"], f["home_margin"]):
            if hh >= 0:
                (w if mg > 0 else l)[hh] += 1
            if aa >= 0:
                (w if mg < 0 else l)[aa] += 1
        return w, l

    pw, pl = tally(played)
    cw, cl = tally(played, played["conference"] & (played["home"] >= 0) & (played["away"] >= 0))
    for c, winner in inp.title_results.items():  # title games played count in the record now
        pair = inp.fixed_title[c]
        pw[winner] += 1
        pl[pair[1] if pair[0] == winner else pair[0]] += 1
    left_games = g[left].assign(home_team=[inp.teams[i] if i >= 0 else FCS for i in h[left]],
                                away_team=[inp.teams[i] if i >= 0 else FCS for i in a[left]])
    return CollegeResult(inp.teams, inp.conference, runs, wins, losses, conf_w, title_game, champion,
                         rank, seed, stage, outcomes, left_games.reset_index(drop=True),
                         pw, pl, cw, cl, conf_n, eligible, bowl_game, bowl_win, final_wins, final_losses,
                         inp.bowls)


def _play_bowls(inp: CollegeInput, rank: np.ndarray, seed: np.ndarray, counted: np.ndarray,
                strength: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """Each run's bowls outside the playoff (runs × T: plays one, wins it):
    the ones announced, or `inp.bowl_slots` games filled by the eligible
    teams outside the field in the committee's order (the lowest left out
    when there are too many; when too few, eligible teams still moving up
    from FCS, then 5–7 teams, as the NCAA fills them), each paired with the
    next team from another conference."""
    runs, T = rank.shape
    rows = np.arange(runs)
    shrink, sd = bowl_model(inp.model)
    bowl_game = np.zeros((runs, T), bool)
    bowl_win = np.zeros((runs, T), bool)
    if inp.bowls is not None and not inp.bowls.empty:
        for r in inp.bowls.itertuples(index=False):
            h, a = r.home, r.away
            if r.played:
                home_won = np.full(runs, r.home_margin > 0)
            else:
                adv = 0.0 if r.neutral else inp.hfa
                m = shrink * (strength[:, h] - strength[:, a] + adv) + rng.normal(0.0, sd, runs)
                home_won = m > 0
            bowl_game[:, h] = bowl_game[:, a] = True
            bowl_win[:, h], bowl_win[:, a] = home_won, ~home_won
        return bowl_game, bowl_win
    slots = 2 * inp.bowl_slots
    if not slots:
        return bowl_game, bowl_win
    conf = inp.conference
    run_i, t1, t2 = [], [], []
    order = np.argsort(rank, axis=1)
    out = seed > 0
    elig = counted >= BOWL_WINS
    five = counted == BOWL_WINS - 1
    new = inp.transition
    for s in range(runs):
        o = order[s]
        pool = [t for t in o if elig[s, t] and not out[s, t] and t not in new]
        if len(pool) < slots:
            pool += [t for t in o if elig[s, t] and not out[s, t] and t in new][:slots - len(pool)]
        if len(pool) < slots:
            pool += [t for t in o if five[s, t] and not out[s, t] and t not in new][:slots - len(pool)]
        pool = pool[:slots]
        pool = pool[:len(pool) - len(pool) % 2]  # pairs only
        while len(pool) >= 2:
            t = pool.pop(0)
            j = next((i for i, x in enumerate(pool) if conf[x] != conf[t]), 0)
            run_i.append(s); t1.append(t); t2.append(pool.pop(j))
    run_i, t1, t2 = np.array(run_i, int), np.array(t1, int), np.array(t2, int)
    m = shrink * (strength[run_i, t1] - strength[run_i, t2]) + rng.normal(0.0, sd, len(run_i))
    bowl_game[run_i, t1] = bowl_game[run_i, t2] = True
    bowl_win[run_i, t1], bowl_win[run_i, t2] = m > 0, m <= 0
    return bowl_game, bowl_win


def _select_field(inp: CollegeInput, rank: np.ndarray, champion: np.ndarray) -> np.ndarray:
    """Each run's playoff seeds (runs × T, 0 = out) under the season's
    rules (`cfp_format`)."""
    fmt = cfp_format(inp.season)
    runs, T = rank.shape
    conf = np.array(inp.conference)
    seeds = np.zeros((runs, T), np.int8)
    p4 = np.isin(conf, list(POWER4))
    g6 = np.isin(conf, list(GROUP6))
    for s in range(runs):
        r = rank[s]
        order = np.argsort(r)  # best first
        champs = [t for t in order if champion[s, t]]
        if fmt.power4_in:
            auto = [t for t in champs if p4[t]]
            g6_top = next((t for t in order if g6[t]), None)
            auto += [g6_top] if g6_top is not None else []
        else:
            auto = champs[:fmt.champions_in]
        auto_set = set(auto)
        at_large = [t for t in order if t not in auto_set][:fmt.teams - len(auto)]
        field_ = sorted(auto_set | set(at_large), key=lambda t: r[t])
        if fmt.champion_byes:
            top = [t for t in field_ if champion[s, t]][:4]
            field_ = top + [t for t in field_ if t not in top]
        for k, t in enumerate(field_):
            seeds[s, t] = k + 1
    return seeds


# The fixed bracket: first round (higher seed hosts), then each quarterfinal's top seed.
_FIRST_ROUND = [(5, 12), (6, 11), (7, 10), (8, 9)]
_QUARTERS = {1: (8, 9), 2: (7, 10), 3: (6, 11), 4: (5, 12)}
_SEMIS = [(1, 4), (2, 3)]  # quarterfinal brackets that meet


def _play_bracket(inp: CollegeInput, seed: np.ndarray, strength: np.ndarray, rng) -> np.ndarray:
    """Each run's playoff (stage per team, `STAGES`): the first round at the
    better seed's home, the rest at neutral sites; games already played keep
    their result."""
    runs, T = seed.shape
    rows = np.arange(runs)
    by_seed = np.full((runs, 13), -1)
    for k in range(1, 13):
        hit = seed == k
        by_seed[hit.any(1), k] = np.argmax(hit, axis=1)[hit.any(1)]
    stage = np.where(seed > 0, 1, 0).astype(np.int8)

    def play(t1, t2, home: bool, rnd: int):
        m = strength[rows, t1] - strength[rows, t2] + (inp.hfa if home else 0.0) \
            + rng.normal(0.0, inp.game_sd, runs)
        win = np.where(m > 0, t1, t2)
        for pair, w in inp.playoff_results.items():
            i, j = tuple(pair)
            hit = ((t1 == i) & (t2 == j)) | ((t1 == j) & (t2 == i))
            win[hit] = w
        lose = np.where(win == t1, t2, t1)
        stage[rows, lose] = rnd
        stage[rows, win] = rnd + 1
        return win

    first = {hi: play(by_seed[:, hi], by_seed[:, lo], True, 1) for hi, lo in _FIRST_ROUND}
    stage[rows[:, None], by_seed[:, 1:5]] = 2  # the byes start in the quarterfinals
    quarter = {}
    for top, (x, y) in _QUARTERS.items():
        other = first[min(x, y)]
        quarter[top] = play(by_seed[:, top], other, False, 2)
    semis = [play(quarter[i], quarter[j], False, 3) for i, j in _SEMIS]
    play(semis[0], semis[1], False, 4)
    return stage


# ---------- What the runs say ----------
def record(w, l) -> str:
    return f"{int(w)}-{int(l)}"


def team_table(res: CollegeResult, level: float) -> pd.DataFrame:
    """One row per FBS team: record and conference record now, projected
    wins and the range holding `level` of the runs, and the share of runs it
    plays in and wins its title game, is bowl eligible, makes the playoff,
    gets a bye, reaches each round and wins it; its average committee rank."""
    lo_q, hi_q = (1 - level) / 2, (1 + level) / 2
    w = res.wins
    return pd.DataFrame({
        "Team": res.teams, "Conference": res.conference,
        "Now": [record(a, b) for a, b in zip(res.played_wins, res.played_losses)],
        "Conf now": [record(a, b) for a, b in zip(res.played_conf, res.played_conf_losses)],
        "W": w.mean(0), "L": res.losses.mean(0),
        "Conf W": res.conf_wins.mean(0), "Conf L": res.conf_games - res.conf_wins.mean(0),
        "W low": np.floor(np.quantile(w, lo_q, axis=0)), "W high": np.ceil(np.quantile(w, hi_q, axis=0)),
        "Title game %": res.title_game.mean(0), "Conf champ %": res.champion.mean(0),
        "Bowl %": (w >= BOWL_WINS).mean(0), "Playoff %": (res.seed > 0).mean(0),
        "Bye %": ((res.seed >= 1) & (res.seed <= 4)).mean(0),
        "Quarterfinal %": (res.stage >= 2).mean(0), "Semifinal %": (res.stage >= 3).mean(0),
        "Final %": (res.stage >= 4).mean(0), "Champion %": (res.stage == 5).mean(0),
        "Avg rank": res.rank.mean(0), "Top 25 %": (res.rank <= 25).mean(0),
        "Eligible %": res.eligible.mean(0), "Bowl game %": res.bowl_game.mean(0),
        "Bowl win %": res.bowl_win.mean(0), "Postseason %": (res.bowl_game | (res.seed > 0)).mean(0),
        "Final W": res.final_wins.mean(0), "Final L": res.final_losses.mean(0),
    })


def seed_table(res: CollegeResult, min_share: float = 0.005) -> pd.DataFrame:
    """The share of runs each team ends up at each seed (1–12) or out, for
    teams in the field in at least `min_share` of the runs, by average seed."""
    s = res.seed
    keep = (s > 0).mean(0) >= min_share
    df = pd.DataFrame({k: (s[:, keep] == k).mean(0) for k in range(1, 13)},
                      index=[t for t, k in zip(res.teams, keep) if k])
    df["Out"] = (s[:, keep] == 0).mean(0)
    order = np.where(s[:, keep] > 0, s[:, keep], 14).mean(0)
    return df.iloc[np.argsort(order)]


def projected_field(res: CollegeResult, season: int) -> list[str]:
    """The most likely bracket under the season's rules (`cfp_format`): each
    conference's likeliest champion where a title brings a bid (the Power 4
    from 2026, else the five champions likeliest in the field), the Group of
    6 team likeliest in the field (2026 on), then the teams likeliest in the
    field, seeded by the assignment with the most runs behind it."""
    fmt = cfp_format(season)
    t = seed_table(res, 0.0)
    p_in = 1 - t["Out"]
    champ = pd.Series(res.champion.mean(0), index=res.teams)
    conf = dict(zip(res.teams, res.conference))
    likeliest = {c: champ[[x for x in res.teams if conf[x] == c]].idxmax()
                 for c in set(res.conference) if champ[[x for x in res.teams if conf[x] == c]].max() > 0}
    if fmt.power4_in:
        auto = [likeliest[c] for c in sorted(POWER4) if c in likeliest]
        g6 = p_in[[x for x in p_in.index if conf[x] in GROUP6]]
        auto += [g6.idxmax()] if len(g6) else []
    else:
        auto = sorted(likeliest.values(), key=lambda x: -p_in.get(x, 0.0))[:fmt.champions_in]
    rest = list(p_in.drop(auto).sort_values(ascending=False, kind="stable").index[:fmt.teams - len(auto)])
    field_ = auto + rest
    order = sim._assign(t.loc[field_, list(range(1, fmt.teams + 1))].to_numpy())
    return [field_[i] for i in order]


def game_prob(inp: CollegeInput, t1: str, t2: str, home: bool) -> float:
    """The chance `t1` beats `t2` (at home, or at a neutral site)."""
    i, j = inp.teams.index(t1), inp.teams.index(t2)
    mu = inp.power[i] - inp.power[j] + (inp.hfa if home else 0.0)
    return float(ndtr(mu / np.sqrt(inp.game_sd ** 2 + inp.team_sd[i] ** 2 + inp.team_sd[j] ** 2)))


def wins_distribution(res: CollegeResult, team: str) -> pd.Series:
    w = res.wins[:, res.teams.index(team)]
    return pd.Series(w).value_counts(normalize=True).sort_index()


def leverage(res: CollegeResult, team: str) -> pd.DataFrame:
    """For every game left, `team`'s playoff chances in the runs where the
    home team won and where it lost, and the swing; its own games first."""
    t = res.teams.index(team)
    y = (res.seed[:, t] > 0).astype(float)
    o = res.outcomes
    n_home = o.sum(0)
    if_home = np.where(n_home > 0, (o * y[:, None]).sum(0) / np.maximum(n_home, 1), np.nan)
    if_away = np.where(res.runs - n_home > 0, ((~o) * y[:, None]).sum(0) / np.maximum(res.runs - n_home, 1),
                       np.nan)
    df = res.left[["game_id", "week", "home_team", "away_team"]].copy()
    df["Home wins"], df["Away wins"] = if_home, if_away
    df["Swing"] = np.abs(if_home - if_away)
    df["own"] = (df["home_team"] == team) | (df["away_team"] == team)
    return df.sort_values(["own", "Swing"], ascending=[False, False]).reset_index(drop=True)


def title_matchups(res: CollegeResult, conference: str, n: int = 3) -> pd.DataFrame:
    """The conference's likeliest title games: the two teams, the share of
    runs, and how often the first wins it then."""
    idx = [i for i, c in enumerate(res.conference) if c == conference]
    tg = res.title_game[:, idx]
    if not tg.any():
        return pd.DataFrame(columns=["Team", "Opponent", "Share", "Wins"])
    two = np.argsort(~tg, axis=1, kind="stable")[:, :2]
    a, b = np.array(idx)[two[:, 0]], np.array(idx)[two[:, 1]]
    first = np.minimum(a, b)
    second = np.maximum(a, b)
    won = res.champion[np.arange(res.runs), first]
    df = pd.DataFrame({"Team": [res.teams[i] for i in first], "Opponent": [res.teams[i] for i in second],
                       "won": won})
    out = df.groupby(["Team", "Opponent"]).agg(Share=("won", "size"), Wins=("won", "mean")).reset_index()
    out["Share"] /= res.runs
    return out.sort_values("Share", ascending=False).head(n).reset_index(drop=True)
