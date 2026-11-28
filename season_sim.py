"""
The Season simulation page's model: the rest of an NFL season played out
`SIM_RUNS` times, for each team's final record, the playoff bracket and the
draft order.

Each game not played yet gets an expected margin from the Matchup page's
numbers: every offense's Proj. against the other team's defense
(replay_core.matchup_table's formula, `proj_matrix`), opponent-adjusted,
moved by the players gone from both lineups (replay_core.LINEUP_EFFECTS)
and, while a season is young, blended with last season's ratings. How much
each projected difference is worth in points, the home field, and how far
games stray from it are fitted by tools/build_sim_model.py on 2016–2025 and
read from `MODEL_FILE`.

A run draws each team's rating error once (the numbers behind a team can be
off for all of its games: `tau`) and each game's own noise, so a run where a
team is better than its numbers is better all season, as real seasons go.
Then the NFL's standings: division winners and wild cards by the league's
tiebreaking steps (`_best`), the playoffs game by game with the better seed
at home, and the draft order from the records and the playoff exits.

No Streamlit and not in `leagues._CODE_FILES`, like records.py: a change
here doesn't make the season store rebuild.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import requests
from scipy.special import ndtr

import replay_core as core
import scoreboard
from live_feed import ESPN_HEADERS

SIM_RUNS = 10_000
SIM_SEED = 2026  # fixed, so the same inputs give the same numbers on every rerun
MODEL_FILE = Path(__file__).parent / "models" / "season_sim.json"
# The page's confidence levels: the share of runs a range covers.
CONFIDENCE_LEVELS = {"80%": 0.80, "90%": 0.90, "95%": 0.95}
# How far a team got: missed the playoffs, then lost in each round, then won it all.
STAGES = ["Missed", "Wild Card", "Divisional", "Conference", "Super Bowl", "Champion"]
STAGE_ROUND = {"WC": 1, "DIV": 2, "CON": 3, "SB": 4}  # nflverse's game_type → round
# The injury report's statuses that fade as players return; roster moves
# (reserve lists, releases, trades) fade by their own rate.
REPORT_WHY = core.LIKELY_OUT


def load_model(path: Path = MODEL_FILE) -> dict | None:
    """The fitted game model (tools/build_sim_model.py), or None without it."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def playoff_seeds(season: int) -> int:
    """Playoff teams per conference: 7 since 2020, 6 before."""
    return 7 if season >= 2020 else 6


# ---------- One game's expected margin ----------
def proj_matrix(off: pd.DataFrame, def_: pd.DataFrame, stat: str) -> pd.DataFrame:
    """Proj. of `stat` for every offense (rows, `off`'s teams) against every
    defense (columns, `def_`'s): the league average plus `PROJ_SHRINK` of
    the offense's and the defense's distances from it, as
    core.matchup_table computes it for one pair."""
    league = (off[stat].mean() + def_[stat].mean()) / 2
    o = off[stat].astype(float).to_numpy() - league
    d = def_[stat].astype(float).to_numpy() - league
    return pd.DataFrame(league + core.PROJ_SHRINK * (o[:, None] + d[None, :]),
                        index=off.index, columns=def_.index)


def lineup_lost(change: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """A core.lineup_change() as the full-time players gone per
    core.LINEUP_GROUPS (negative), (from the injury report, from the
    roster): Out or Doubtful players come back sooner than ones on a reserve
    list, released or traded."""
    report = change[change["Why"].isin(REPORT_WHY)]
    roster = change[~change["Why"].isin(REPORT_WHY | {"Back"})]
    return core.lineup_groups(report, "lost"), core.lineup_groups(roster, "lost")


def still_out(decay: dict, kind: str, games_ahead: int) -> float:
    """The share of players gone (`kind` "report" or "roster") still gone
    `games_ahead` of the team's games after the one they were listed for
    (0: that game, where it's 1). The model's last value holds after it."""
    rates = decay.get(kind) or [1.0]
    return float(rates[min(max(games_ahead, 0), len(rates) - 1)])


def lineup_shift(off_lost: pd.Series, def_lost: pd.Series, stat: str) -> float:
    """How far the players gone move an offense's Proj. of `stat`: its own
    QB, line and skill players (`off_lost`), the defense's front and
    secondary (`def_lost`), core.lineup_adjustment's effects."""
    eff = core.LINEUP_EFFECTS.get(stat)
    if not eff:
        return 0.0
    total = 0.0
    for g, units in core.LINEUP_GROUPS.items():
        lost = off_lost if core.UNIT_PHASE[units[0]] == "Off" else def_lost
        total += eff[g] * -float(lost.get(g, 0.0))
    return total


def data_weight(games: np.ndarray, g0: float) -> np.ndarray:
    """How much of a margin comes from this season's numbers after `games`
    games (the rest from last season's): games / (games + g0)."""
    games = np.asarray(games, dtype=float)
    return games / (games + g0) if g0 > 0 else (games > 0).astype(float)


def team_sd(model: dict, games: np.ndarray) -> np.ndarray:
    """The SD (points) of a team's rating error after `games` games: how far
    its true strength can be from its numbers, for every game it plays."""
    t = model["tau"]
    return np.sqrt(t["floor"] ** 2 + t["scale"] ** 2 / (np.asarray(games, float) + t["g1"]))


def margins(feats: pd.DataFrame, model: dict) -> np.ndarray:
    """Expected home margins (points) for rows of `game_features()`: the home
    field where not neutral, plus this season's projected differences
    (weighted by `data_weight`) and last season's (the rest)."""
    w = data_weight(feats["games"].to_numpy(), model["g0"])
    cur = sum(model["features"][f]["coef"] * feats[f].fillna(0.0).to_numpy()
              for f in model["features"])
    prior = model["prior"]["coef"] * feats["prior"].fillna(0.0).to_numpy()
    return model["hfa"] * (1 - feats["neutral"].astype(float).to_numpy()) + w * cur + (1 - w) * prior


def game_features(games: pd.DataFrame, proj: dict[str, pd.DataFrame], lost: dict, prior: pd.Series,
                  model: dict, played: pd.Series) -> pd.DataFrame:
    """One row per game of `games` (home, away, neutral, and `ahead_home` /
    `ahead_away`: how many of each team's games after the one its injury
    report was for), with the model's features: each `model["features"]`
    stat's Proj. for the home offense less the away offense's (sign flipped
    where less is better), the players gone (`lost`, by team: `lineup_lost()`,
    faded by `still_out()`) moving the stats core.LINEUP_EFFECTS has;
    `prior`, last season's net rating difference; `games`, the two teams'
    games played so far on average (`played`, by team)."""
    out = pd.DataFrame(index=games.index)
    out["neutral"] = games["neutral"].astype(bool).to_numpy()
    out["games"] = [(played.get(h, 0) + played.get(a, 0)) / 2
                    for h, a in zip(games["home"], games["away"])]
    decay = model.get("decay", {})

    def gone(team, ahead):
        rep, ros = lost.get(team, (None, None))
        if rep is None:
            return pd.Series(0.0, index=list(core.LINEUP_GROUPS))
        return rep * still_out(decay, "report", ahead) + ros * still_out(decay, "roster", ahead)

    shifts = {}
    for i, r in games.iterrows():
        gh, ga = gone(r["home"], r.get("ahead_home", 0)), gone(r["away"], r.get("ahead_away", 0))
        shifts[i] = (gh, ga)
    for f, spec in model["features"].items():
        stat, sign = spec["stat"], spec.get("sign", 1)
        m = proj.get(stat)
        vals = []
        for i, r in games.iterrows():
            h, a = r["home"], r["away"]
            if m is None or h not in m.index or a not in m.index:
                vals.append(np.nan)
                continue
            gh, ga = shifts[i]
            ph = m.at[h, a] + lineup_shift(gh, ga, stat)
            pa = m.at[a, h] + lineup_shift(ga, gh, stat)
            vals.append(sign * (ph - pa))
        out[f] = vals
    out["prior"] = [prior.get(h, np.nan) - prior.get(a, np.nan) if prior is not None else np.nan
                    for h, a in zip(games["home"], games["away"])]
    return out


# ---------- Results ESPN has before the schedule does ----------
# nflverse's and sportsdataverse's schedules are rebuilt a few times a day, so
# a game that ended an hour (or a morning) ago can still read as not played.
# ESPN's scoreboard has the final within minutes: for each day with a game
# that has kicked off and isn't final in the schedule (up to FINALS_DAYS
# back), its finals are read and put on the schedule's rows.
FINALS_DAYS = 10


def espn_finals(league_key: str, day: str, timeout: float = 10) -> pd.DataFrame:
    """ESPN's finished games on `day` (YYYYMMDD, US Eastern), for a league key
    ("NFL" / "CFB"): event_id, home_score, away_score. Empty if it can't be read."""
    url, params = scoreboard.SCOREBOARDS[league_key]
    try:
        r = requests.get(url, params={**params, "dates": day}, headers=ESPN_HEADERS, timeout=timeout)
        r.raise_for_status()
        events = r.json().get("events", [])
    except Exception:
        return pd.DataFrame(columns=["event_id", "home_score", "away_score"])
    rows = []
    for e in events:
        if not e.get("status", {}).get("type", {}).get("completed"):
            continue
        comp = (e.get("competitions") or [{}])[0]
        side = {c.get("homeAway"): c for c in comp.get("competitors", [])}
        try:
            rows.append({"event_id": str(e["id"]), "home_score": float(side["home"]["score"]),
                         "away_score": float(side["away"]["score"])})
        except (KeyError, TypeError, ValueError):
            continue
    return pd.DataFrame(rows, columns=["event_id", "home_score", "away_score"])


def _finals(league_key: str, days: list[str]) -> pd.DataFrame:
    frames = [f for f in (espn_finals(league_key, d) for d in days) if not f.empty]
    return (pd.concat(frames, ignore_index=True).drop_duplicates("event_id").set_index("event_id")
            if frames else pd.DataFrame(columns=["home_score", "away_score"]))


def pending_days(kickoff: pd.Series, done: pd.Series, now: pd.Timestamp | None = None) -> list[str]:
    """The US Eastern days (YYYYMMDD) of games that have kicked off (UTC
    `kickoff`) but aren't `done`, the last `FINALS_DAYS` only."""
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    k = pd.to_datetime(kickoff, utc=True, errors="coerce")
    m = ~done.fillna(False).astype(bool) & (k <= now) & (k >= now - pd.Timedelta(days=FINALS_DAYS))
    return sorted(set(k[m].dt.tz_convert("America/New_York").dt.strftime("%Y%m%d")))


def nfl_with_finals(sched: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """nflverse's schedule with ESPN's finals on the games it doesn't have
    a result for yet (matched by ESPN event id)."""
    if sched.empty or "espn" not in sched.columns:
        return sched
    kick = pd.to_datetime(sched["gameday"].astype(str) + " " + sched["gametime"].fillna("00:00").astype(str),
                          errors="coerce").dt.tz_localize("America/New_York", ambiguous="NaT",
                                                          nonexistent="NaT").dt.tz_convert("UTC")
    days = pending_days(kick, sched["result"].notna(), now)
    if not days:
        return sched
    finals = _finals("NFL", days)
    if finals.empty:
        return sched
    out = sched.copy()
    hit = out["result"].isna() & out["espn"].astype(str).isin(finals.index)
    ev = out.loc[hit, "espn"].astype(str)
    out.loc[hit, "home_score"] = finals.loc[ev, "home_score"].to_numpy()
    out.loc[hit, "away_score"] = finals.loc[ev, "away_score"].to_numpy()
    out.loc[hit, "result"] = out.loc[hit, "home_score"] - out.loc[hit, "away_score"]
    return out


def cfb_with_finals(sched: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """sportsdataverse's college schedule with ESPN's finals on the games
    it doesn't have as completed yet (its game_id is ESPN's event id)."""
    if sched.empty or "start_date" not in sched.columns:
        return sched
    days = pending_days(sched["start_date"], sched["completed"], now)
    if not days:
        return sched
    finals = _finals("CFB", days)
    if finals.empty:
        return sched
    out = sched.copy()
    hit = ~out["completed"].fillna(False).astype(bool) & out["game_id"].astype(str).isin(finals.index)
    ev = out.loc[hit, "game_id"].astype(str)
    out.loc[hit, "home_points"] = finals.loc[ev, "home_score"].to_numpy()
    out.loc[hit, "away_points"] = finals.loc[ev, "away_score"].to_numpy()
    out.loc[hit, "completed"] = True
    return out


def results_key(finished_ids) -> str:
    """A short key for the set of finished games, so a cached run is
    redone when a result comes in."""
    import hashlib
    return hashlib.sha1(",".join(sorted(map(str, finished_ids))).encode()).hexdigest()[:12]


# ---------- The schedule ----------
def regular_season(sched: pd.DataFrame, as_of: int | None = None) -> pd.DataFrame:
    """nflverse's schedule as the simulation's regular season: game_id, week,
    home, away (abbreviations), neutral, played and credit (the home team's
    win: 1, ½ for a tie, 0; NaN until played). With `as_of`, games from that
    week on count as not played yet, whatever their result."""
    g = sched[sched["game_type"] == "REG"]
    played = g["result"].notna()
    if as_of is not None:
        played &= g["week"] < as_of
    credit = np.where(g["result"] > 0, 1.0, np.where(g["result"] < 0, 0.0, 0.5))
    return pd.DataFrame({
        "game_id": g["game_id"].to_numpy(), "week": g["week"].astype(int).to_numpy(),
        "home": g["home_team"].to_numpy(), "away": g["away_team"].to_numpy(),
        "neutral": g["location"].eq("Neutral").to_numpy(), "played": played.to_numpy(),
        "credit": np.where(played, credit, np.nan),
    }).sort_values(["week", "game_id"]).reset_index(drop=True)


def playoff_results(sched: pd.DataFrame, as_of: int | None = None) -> dict[frozenset, str]:
    """The postseason games played (before week `as_of`, if set): the two
    teams → the winner."""
    g = sched[(sched["game_type"] != "REG") & sched["result"].notna() & (sched["result"] != 0)]
    if as_of is not None:
        g = g[g["week"] < as_of]
    return {frozenset((h, a)): (h if r > 0 else a)
            for h, a, r in zip(g["home_team"], g["away_team"], g["result"])}


def build_input(season: int, reg: pd.DataFrame, divisions: dict[str, str], mu: pd.Series,
                pair: pd.DataFrame, model: dict, games_played: pd.Series,
                post: dict[frozenset, str] | None = None) -> SimInput:
    """A `SimInput` from `regular_season()` rows, each team's division, the
    expected home margin of each game not played yet (`mu`, by game_id), the
    neutral-site margins between any two teams (`pair`) and the model's home
    field and spreads; `games_played` (by team) sets each team's rating SD."""
    teams = sorted(set(reg["home"]) | set(reg["away"]))
    idx = {t: i for i, t in enumerate(teams)}
    g = reg.assign(home=reg["home"].map(idx), away=reg["away"].map(idx),
                   mu=reg["game_id"].map(mu).astype(float))
    p = pair.reindex(index=teams, columns=teams).fillna(0.0).to_numpy(float)
    tau = team_sd(model, games_played.reindex(teams).fillna(0).to_numpy())
    return SimInput(season, teams, [divisions[t] for t in teams], g, p, model["hfa"],
                    model["game_sd"], tau,
                    playoff_seeds(season),
                    {frozenset(idx[t] for t in k): idx[w] for k, w in (post or {}).items()
                     if all(t in idx for t in k)})


# ---------- The simulation ----------
@dataclass
class SimInput:
    """Everything a run needs, by team index."""
    season: int
    teams: list[str]
    divisions: list[str]            # "AFC West", per team
    games: pd.DataFrame             # regular season: home, away (indices), week, played, credit (home's: 1/0.5/0), mu, neutral
    pair: np.ndarray                # T×T: expected margin of the row team over the column team at a neutral site (playoffs)
    hfa: float                      # home field, points
    game_sd: float                  # a game's own noise, points
    team_sd: np.ndarray             # per team: its rating error's SD, points
    seeds: int                      # playoff teams per conference
    playoff_results: dict = field(default_factory=dict)  # frozenset({i, j}) -> winner index


@dataclass
class SimResult:
    teams: list[str]
    divisions: list[str]
    runs: int
    wins: np.ndarray        # runs × T: final wins (a tie counts half)
    seed: np.ndarray        # runs × T: playoff seed, 0 = missed
    stage: np.ndarray       # runs × T: index into STAGES
    pick: np.ndarray        # runs × T: draft pick earned by the team's own record
    outcomes: np.ndarray    # runs × games left: True where the home team won
    left: pd.DataFrame      # the games left (SimInput.games rows, team names), in `outcomes`' order
    games_total: np.ndarray  # per team: regular-season games
    played_wins: np.ndarray  # per team: wins so far
    played_losses: np.ndarray
    played_ties: np.ndarray


class _Run:
    """One run's standings, for the tiebreakers: lists by team index."""
    __slots__ = ("pct", "div", "conf", "sov", "sos", "coin", "credit", "div_rank")

    def __init__(self, pct, div, conf, sov, sos, coin, credit):
        self.pct, self.div, self.conf, self.sov, self.sos, self.coin = pct, div, conf, sov, sos, coin
        self.credit = credit  # numpy row: each game's credit to the home team
        self.div_rank: dict[int, int] = {}


class _Schedule:
    """The regular season's static facts the tiebreakers look up."""

    def __init__(self, inp: SimInput):
        g = inp.games
        self.home, self.away = g["home"].to_numpy(), g["away"].to_numpy()
        self.div_of = inp.divisions
        T = len(inp.teams)
        self.opp_games: list[dict[int, list[tuple[int, bool]]]] = [{} for _ in range(T)]
        for k, (h, a) in enumerate(zip(self.home, self.away)):
            self.opp_games[h].setdefault(a, []).append((k, True))
            self.opp_games[a].setdefault(h, []).append((k, False))
        self._common: dict = {}

    def record_vs(self, run: _Run, team: int, opps) -> tuple[float, int]:
        """`team`'s credit and games against the teams in `opps`."""
        c, n = 0.0, 0
        for o in opps:
            for k, home in self.opp_games[team].get(o, ()):
                c += run.credit[k] if home else 1.0 - run.credit[k]
                n += 1
        return c, n

    def common_opps(self, group: tuple[int, ...]) -> frozenset:
        key = frozenset(group)
        if key not in self._common:
            sets = [set(self.opp_games[t]) for t in group]
            self._common[key] = frozenset(set.intersection(*sets) - key)
        return self._common[key]


_EPS = 1e-9


def _h2h(group, run, sched, kind):
    """Head-to-head: in a division, each team's record in the games among
    the group; in a wild-card two-team tie, the record between them if they
    played; with three or more, only a sweep counts (1 for the team that
    beat every other, −1 for one that lost to every other)."""
    if kind == "div" or len(group) == 2:
        vals = []
        for t in group:
            c, n = sched.record_vs(run, t, [o for o in group if o != t])
            if n == 0:
                return None
            vals.append(c / n)
        return vals
    vals = []
    for t in group:
        beat = lost = True
        for o in group:
            if o == t:
                continue
            c, n = sched.record_vs(run, t, [o])
            if n == 0:
                beat = lost = False
                break
            beat &= c == n
            lost &= c == 0
        vals.append(1.0 if beat else -1.0 if lost else 0.0)
    return vals if any(vals) else None


def _common(group, run, sched, min_games):
    opps = sched.common_opps(tuple(group))
    vals = []
    for t in group:
        c, n = sched.record_vs(run, t, opps)
        if n < max(min_games, 1):
            return None
        vals.append(c / n)
    return vals


_DIV_STEPS = (
    lambda g, r, s: _h2h(g, r, s, "div"),
    lambda g, r, s: [r.div[t] for t in g],
    lambda g, r, s: _common(g, r, s, 1),
    lambda g, r, s: [r.conf[t] for t in g],
    lambda g, r, s: [r.sov[t] for t in g],
    lambda g, r, s: [r.sos[t] for t in g],
)
_WC_STEPS = (
    lambda g, r, s: _h2h(g, r, s, "wc"),
    lambda g, r, s: [r.conf[t] for t in g],
    lambda g, r, s: _common(g, r, s, 4),
    lambda g, r, s: [r.sov[t] for t in g],
    lambda g, r, s: [r.sos[t] for t in g],
)


def _best(group: list[int], kind: str, run: _Run, sched: _Schedule) -> int:
    """The team that wins a tie among `group` (same record): the NFL's
    division steps (`kind` "div": head-to-head, division record, common
    games, conference record, strength of victory, strength of schedule) or
    its wild-card ones ("wc": first only the best-placed team of each
    division stays in, then head-to-head, conference record, common games
    (four or more), strength of victory and of schedule). When a step leaves
    some teams behind, the rest start over from the first step, as the
    league's procedure does; the points-based steps are left out, so a tie
    through strength of schedule is a coin flip."""
    group = list(group)
    while True:
        if kind == "wc" and len(group) > 1:
            heads = {}
            for t in group:
                d = sched.div_of[t]
                if d not in heads or run.div_rank[t] < run.div_rank[heads[d]]:
                    heads[d] = t
            group = list(heads.values())
        if len(group) == 1:
            return group[0]
        for step in (_DIV_STEPS if kind == "div" else _WC_STEPS):
            vals = step(group, run, sched)
            if vals is None:
                continue
            top = max(vals)
            kept = [t for t, v in zip(group, vals) if v >= top - _EPS]
            if len(kept) < len(group):
                group = kept
                break
        else:
            return min(group, key=lambda t: run.coin[t])


def _order(teams: list[int], kind: str, run: _Run, sched: _Schedule, need: int | None = None) -> list[int]:
    """`teams` best first: by winning percentage, ties by `_best`; only the
    first `need` places are worked out (the rest follow by percentage)."""
    by = sorted(teams, key=lambda t: -run.pct[t])
    out: list[int] = []
    i = 0
    while i < len(by):
        j = i
        while j + 1 < len(by) and abs(run.pct[by[j + 1]] - run.pct[by[i]]) < _EPS:
            j += 1
        group = by[i:j + 1]
        if need is not None and len(out) >= need:
            out.extend(group)
        else:
            while group:
                b = group[0] if len(group) == 1 else _best(group, kind, run, sched)
                out.append(b)
                group.remove(b)
        i = j + 1
    return out


def standings(credit: np.ndarray, inp: SimInput, coin: np.ndarray) -> dict[str, np.ndarray]:
    """Each run's standings from `credit` (runs × games: the home team's
    win, ½ for a tie, 0 for a loss): wins, winning percentage, division and
    conference percentages, strength of victory and of schedule (the
    combined percentage of the teams beaten, and of every opponent), and the
    playoff seeds (runs × T, 0 = out)."""
    g, T = inp.games, len(inp.teams)
    H = np.zeros((len(g), T), dtype=np.float32)
    A = np.zeros((len(g), T), dtype=np.float32)
    H[np.arange(len(g)), g["home"].to_numpy()] = 1
    A[np.arange(len(g)), g["away"].to_numpy()] = 1
    conf = np.array([d.split(" ")[0] for d in inp.divisions])
    hc, ac = conf[g["home"].to_numpy()], conf[g["away"].to_numpy()]
    is_div = np.array([inp.divisions[h] == inp.divisions[a] for h, a in zip(g["home"], g["away"])])
    is_conf = hc == ac
    credit = credit.astype(np.float32)
    loss = 1.0 - credit
    wins = credit @ H + loss @ A
    n = H.sum(0) + A.sum(0)
    pct = wins / np.maximum(n, 1)

    def part(mask):
        w = credit[:, mask] @ H[mask] + loss[:, mask] @ A[mask]
        return w / np.maximum(H[mask].sum(0) + A[mask].sum(0), 1)

    div_pct, conf_pct = part(is_div), part(is_conf)
    hi, ai = g["home"].to_numpy(), g["away"].to_numpy()
    opp_home, opp_away = pct[:, ai], pct[:, hi]  # the home team's opponent, the away team's
    sos = (opp_home @ H + opp_away @ A) / np.maximum(n, 1)
    sov = ((credit * opp_home) @ H + (loss * opp_away) @ A) / np.maximum(wins, 1e-9)

    sched = _Schedule(inp)
    by_div: dict[str, list[int]] = {}
    for t, d in enumerate(inp.divisions):
        by_div.setdefault(d, []).append(t)
    confs = sorted(set(conf))
    seeds = np.zeros((len(credit), T), dtype=np.int8)
    pct_l, div_l, conf_l = pct.tolist(), div_pct.tolist(), conf_pct.tolist()
    sov_l, sos_l, coin_l = sov.tolist(), sos.tolist(), coin.tolist()
    n_wc = inp.seeds - 4
    for s in range(len(credit)):
        run = _Run(pct_l[s], div_l[s], conf_l[s], sov_l[s], sos_l[s], coin_l[s], credit[s])
        winners = {c: [] for c in confs}
        for d, members in by_div.items():
            order = _order(members, "div", run, sched)
            for rank, t in enumerate(order):
                run.div_rank[t] = rank
            winners[d.split(" ")[0]].append(order[0])
        for c in confs:
            top = _order(winners[c], "wc", run, sched)
            pool = [t for t in range(T) if conf[t] == c and t not in top]
            wc = _order(pool, "wc", run, sched, need=n_wc)[:n_wc]
            for k, t in enumerate(top + wc):
                seeds[s, t] = k + 1
    return {"wins": wins, "pct": pct, "sos": sos, "seed": seeds, "conf": conf}


def _play(home: np.ndarray, away: np.ndarray, inp: SimInput, u: np.ndarray, rng,
          neutral: bool, rows: np.ndarray) -> np.ndarray:
    """Winners (team indices) of one game per run in `rows`, `home` hosting
    `away` (no home field if `neutral`); a game already played (in
    `inp.playoff_results`) keeps its result."""
    mu = inp.pair[home, away] + (0.0 if neutral else inp.hfa)
    m = mu + u[rows, home] - u[rows, away] + rng.normal(0.0, inp.game_sd, len(rows))
    win = np.where(m > 0, home, away)
    for pair, winner in inp.playoff_results.items():
        i, j = tuple(pair)
        hit = ((home == i) & (away == j)) | ((home == j) & (away == i))
        win[hit] = winner
    return win


def simulate(inp: SimInput, runs: int = SIM_RUNS, seed: int = SIM_SEED) -> SimResult:
    """The season played out `runs` times from `inp`: the games left, the
    standings and seeds, the playoffs and the draft order."""
    rng = np.random.default_rng(seed)
    T = len(inp.teams)
    g = inp.games
    left = g[~g["played"]]
    u = rng.normal(0.0, 1.0, (runs, T)) * inp.team_sd[None, :]
    credit = np.tile(g["credit"].fillna(0.0).to_numpy(np.float32), (runs, 1))
    if len(left):
        h, a = left["home"].to_numpy(), left["away"].to_numpy()
        m = (left["mu"].to_numpy()[None, :] + u[:, h] - u[:, a]
             + rng.normal(0.0, inp.game_sd, (runs, len(left))))
        outcomes = m > 0  # simulated games don't end tied (about 1 in 300 NFL games does)
        credit[:, np.flatnonzero(~g["played"].to_numpy())] = outcomes
    else:
        outcomes = np.zeros((runs, 0), dtype=bool)
    coin = rng.random((runs, T))
    st = standings(credit, inp, coin)
    seed_of = st["seed"]

    # The playoffs: byes for the top seeds, then each round the best seed
    # left hosts the worst; the Super Bowl at a neutral site.
    stage = np.where(seed_of > 0, 1, 0).astype(np.int8)
    rows_all = np.arange(runs)
    champs = {}
    for c in sorted(set(st["conf"])):
        # team index per seed (runs × seeds)
        by_seed = np.full((runs, inp.seeds), -1)
        for t in np.flatnonzero(st["conf"] == c):
            s_ = seed_of[:, t]
            hit = s_ > 0
            by_seed[hit, s_[hit] - 1] = t
        alive = by_seed.copy()   # seeds still in, in seed order (-1 once out)
        byes = 8 - inp.seeds     # 1 with seven teams, 2 with six
        rnd = 1
        while True:
            # the teams still in, best seed first
            order = np.sort(np.where(alive >= 0, np.arange(inp.seeds)[None, :], 99), axis=1)
            k = int((order[0] < 99).sum())
            if k <= 1:
                break
            playing = order[:, byes:k] if rnd == 1 else order[:, :k]
            npl = playing.shape[1]
            for i in range(npl // 2):
                hs, as_ = playing[:, i], playing[:, npl - 1 - i]
                home = alive[rows_all, hs]
                away = alive[rows_all, as_]
                win = _play(home, away, inp, u, rng, False, rows_all)
                lose_seed = np.where(win == home, as_, hs)
                stage[rows_all, alive[rows_all, lose_seed]] = rnd  # out in this round
                stage[rows_all, win] = rnd + 1                     # on to the next
                alive[rows_all, lose_seed] = -1
            rnd += 1
            byes = 0
        champs[c] = alive.max(axis=1)
    c1, c2 = sorted(champs)
    sb = _play(champs[c1], champs[c2], inp, u, rng, True, rows_all)
    loser = np.where(sb == champs[c1], champs[c2], champs[c1])
    stage[rows_all, loser] = 4
    stage[rows_all, sb] = 5

    # The draft: teams out of the playoffs first, worst record first, then
    # the playoff teams by the round they went out in; ties by the weaker
    # schedule (lower strength of schedule picks first), then a coin flip.
    order = np.lexsort((coin, st["sos"], st["pct"], stage), axis=1)
    pick = np.empty((runs, T), dtype=np.int8)
    pick[rows_all[:, None], order] = np.arange(1, T + 1, dtype=np.int8)[None, :]

    played = g[g["played"]]
    pw = np.zeros(T); pl = np.zeros(T); pt = np.zeros(T)
    for h, a, c in zip(played["home"], played["away"], played["credit"]):
        if c == 0.5:
            pt[h] += 1; pt[a] += 1
        elif c > 0.5:
            pw[h] += 1; pl[a] += 1
        else:
            pw[a] += 1; pl[h] += 1
    n_games = np.bincount(g["home"], minlength=T) + np.bincount(g["away"], minlength=T)
    return SimResult(inp.teams, inp.divisions, runs, st["wins"].astype(np.float32), seed_of,
                     stage, pick, outcomes,
                     left.assign(home=[inp.teams[i] for i in left["home"]],
                                 away=[inp.teams[i] for i in left["away"]]).reset_index(drop=True),
                     n_games, pw, pl, pt)


# ---------- From the season's data to a SimInput ----------
# A report older than this many of the team's games isn't used: its players
# have mostly played again (the decay) and a newer one is due.
REPORT_MAX_AGE = 2


def team_lineups(reg: pd.DataFrame, snaps: pd.DataFrame, injuries: pd.DataFrame,
                 roster: pd.DataFrame) -> tuple[dict, dict, dict]:
    """Each team's players gone for its next game, by `lineup_lost()` (report,
    roster), from the week of that game: its injury report and roster, or,
    before that report is filed (a team that played on Thursday, or after
    its bye), its last report, `offset` of its games back. Also each team's
    core.lineup_change() rows counted, for the page. Teams without snap counts
    or games left are left out."""
    lost, offset, changes = {}, {}, {}
    if snaps is None or snaps.empty:
        return lost, offset, changes
    played, left = reg[reg["played"]], reg[~reg["played"]]
    for team in sorted(set(reg["home"]) | set(reg["away"])):
        mine = (left["home"] == team) | (left["away"] == team)
        ids = played.loc[(played["home"] == team) | (played["away"] == team), "game_id"].unique()
        if not mine.any() or not len(ids):
            continue
        nxt = int(left.loc[mine, "week"].min())
        change = core.lineup_change(snaps, injuries, roster, team, nxt, ids)
        report, back = change[change["Why"].isin(REPORT_WHY)], 0
        inj = injuries[injuries["team"] == team] if not injuries.empty else injuries
        if not inj.empty and not (inj["week"] == nxt).any() and (inj["week"] < nxt).any():
            week = int(inj.loc[inj["week"] < nxt, "week"].max())
            back = int(((played["home"] == team) | (played["away"] == team))[played["week"] >= week].sum())
            if 0 < back <= REPORT_MAX_AGE:
                old = core.lineup_change(snaps, injuries, roster, team, week, ids)
                # The roster's word is newer: a player since moved to a reserve list counts there.
                report = old[old["Why"].isin(REPORT_WHY) & ~old["Player"].isin(change["Player"])]
            else:
                back = 0
        _, ros = lineup_lost(change)
        lost[team] = (core.lineup_groups(report, "lost"), ros)
        offset[team] = back
        gone = change[~change["Why"].isin(REPORT_WHY | {"Back"})]
        changes[team] = (pd.concat([report.assign(Ahead=back), gone.assign(Ahead=0)], ignore_index=True)
                         .sort_values("Change", kind="stable").reset_index(drop=True))
    return lost, offset, changes


def prepare(season: int, sched: pd.DataFrame, off: pd.DataFrame | None, def_: pd.DataFrame | None,
            prior: pd.Series | None, divisions: dict[str, str], model: dict,
            snaps: pd.DataFrame | None = None, injuries: pd.DataFrame | None = None,
            roster: pd.DataFrame | None = None, as_of: int | None = None) -> tuple[SimInput, pd.DataFrame, dict]:
    """The simulation's input from the season's schedule (nflverse's), its
    opponent-adjusted season tables over the games played (`off` /
    `def_`, None before any), last season's net ratings (`prior`), the
    divisions and the availability data: the SimInput, the games left with
    their expected margins and win probabilities, and each team's lineup
    changes counted (`team_lineups`)."""
    reg = regular_season(sched, as_of)
    stats = {spec["stat"] for spec in model["features"].values()}
    proj = ({s: proj_matrix(off, def_, s) for s in stats if s in off.columns and s in def_.columns}
            if off is not None and def_ is not None and not off.empty else {})
    lost, offset, changes = team_lineups(
        reg, snaps if snaps is not None else pd.DataFrame(),
        injuries if injuries is not None else pd.DataFrame(),
        roster if roster is not None else pd.DataFrame())
    played = reg[reg["played"]]
    counts = pd.concat([played["home"], played["away"]]).value_counts()
    left = reg[~reg["played"]].copy()
    left["ahead_home"], left["ahead_away"] = 0, 0
    # Each game's place among each team's games left, plus the report's age.
    for team in set(left["home"]) | set(left["away"]):
        mine = left.index[(left["home"] == team) | (left["away"] == team)]
        for n, i in enumerate(mine):
            side = "home" if left.at[i, "home"] == team else "away"
            left.at[i, f"ahead_{side}"] = offset.get(team, 0) + n
    feats = game_features(left, proj, lost, prior, model, counts)
    left["mu"] = margins(feats, model)
    tau = dict(zip(sorted(set(reg["home"]) | set(reg["away"])),
                   team_sd(model, counts.reindex(sorted(set(reg["home"]) | set(reg["away"]))).fillna(0).to_numpy())))
    sd = np.sqrt(model["game_sd"] ** 2 + left["home"].map(tau) ** 2 + left["away"].map(tau) ** 2)
    left["p_home"] = ndtr(left["mu"] / sd)
    for f in model["features"]:
        left[f] = feats[f]
    # The playoffs: any two teams at a neutral site, with only the players a
    # roster move took away still gone (the report's are back by then).
    teams = sorted(set(reg["home"]) | set(reg["away"]))
    pairs = pd.DataFrame([(h, a) for h in teams for a in teams if h != a], columns=["home", "away"])
    pairs["neutral"], pairs["ahead_home"], pairs["ahead_away"] = True, 99, 99
    pf = game_features(pairs, proj, lost, prior, model, counts)
    pairs["mu"] = margins(pf, model)
    pair = pairs.pivot(index="home", columns="away", values="mu")
    inp = build_input(season, reg, divisions, left.set_index("game_id")["mu"], pair, model,
                      counts, playoff_results(sched, as_of))
    return inp, left.reset_index(drop=True), changes


# ---------- What the runs say ----------
def z_score(level: float) -> float:
    """The normal quantile for a two-sided `level` (1.645 for 90%)."""
    return float(NormalDist().inv_cdf((1 + level) / 2))


def mc_margin(p, runs: int, level: float):
    """The simulation's own margin of error on a probability `p` from `runs`
    runs at `level` (±1 point at 95% for p = 0.5 over 10,000 runs)."""
    p = np.asarray(p, float)
    return z_score(level) * np.sqrt(p * (1 - p) / runs)


def record_now(res: SimResult) -> list[str]:
    return [f"{int(w)}-{int(l)}" + (f"-{int(t)}" if t else "")
            for w, l, t in zip(res.played_wins, res.played_losses, res.played_ties)]


def team_table(res: SimResult, level: float) -> pd.DataFrame:
    """One row per team: its record now, projected wins (mean) and the range
    holding `level` of the runs, and the share of runs in which it wins the
    division, makes the playoffs, gets the bye (the top seed; the top two
    with six seeds), reaches each round and wins it all; its average draft
    pick and the share of runs it picks in the top five."""
    lo_q, hi_q = (1 - level) / 2, (1 + level) / 2
    w = res.wins
    seed, stage = res.seed, res.stage
    byes = 1 if seed.max() >= 7 else 2
    df = pd.DataFrame({
        "Team": res.teams, "Division": res.divisions, "Now": record_now(res),
        "W": w.mean(0), "L": res.games_total - w.mean(0),
        "W low": np.floor(np.quantile(w, lo_q, axis=0)), "W high": np.ceil(np.quantile(w, hi_q, axis=0)),
        "Division %": ((seed >= 1) & (seed <= 4)).mean(0), "Playoffs %": (seed > 0).mean(0),
        "Bye %": ((seed >= 1) & (seed <= byes)).mean(0), "Top seed %": (seed == 1).mean(0),
        "Divisional %": (stage >= 2).mean(0), "Conf. game %": (stage >= 3).mean(0),
        "Super Bowl %": (stage >= 4).mean(0), "Champion %": (stage == 5).mean(0),
        "Avg pick": res.pick.mean(0), "Top 5 pick %": (res.pick <= 5).mean(0),
    })
    df["Conference"] = df["Division"].str.split(" ").str[0]
    return df


def seed_table(res: SimResult, conference: str) -> pd.DataFrame:
    """The share of runs each of the conference's teams ends up at each seed
    (columns 1 … n) or out ("Out"), teams by their average seed."""
    n = int(res.seed.max())
    idx = [i for i, d in enumerate(res.divisions) if d.startswith(conference)]
    s = res.seed[:, idx]
    df = pd.DataFrame({k: (s == k).mean(0) for k in range(1, n + 1)}, index=[res.teams[i] for i in idx])
    df["Out"] = (s == 0).mean(0)
    order = np.where(s > 0, s, n + 2).mean(0)
    return df.iloc[np.argsort(order)]


def _assign(probs: np.ndarray) -> list[int]:
    """For each column (a seed), the row (a team) that gives the most runs
    summed over columns, each row at most once."""
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(-probs)
        return [int(r) for _, r in sorted(zip(cols, rows))]
    except ImportError:  # greedy: the likeliest (team, seed) first
        by_col, used = {}, set()
        for r, c in sorted(np.ndindex(probs.shape), key=lambda rc: -probs[rc]):
            if c not in by_col and r not in used:
                by_col[c] = r
                used.add(r)
        return [by_col[c] for c in range(probs.shape[1])]


def projected_seeds(res: SimResult, conference: str) -> list[str]:
    """The conference's most likely bracket, one team per seed: each
    division's likeliest winner, seeded 1–4 by the assignment with the most
    runs behind it, then the teams likeliest to be wild cards (of the rest),
    seeded the same way."""
    t = seed_table(res, conference)
    n = t.shape[1] - 1
    div_of = dict(zip(res.teams, res.divisions))
    div_p = t[[1, 2, 3, 4]].sum(axis=1)
    winners = [div_p[[x for x in t.index if div_of[x] == d]].idxmax()
               for d in sorted({div_of[x] for x in t.index})]
    wc_p = t[list(range(5, n + 1))].sum(axis=1).drop(winners)
    wild = list(wc_p.sort_values(ascending=False, kind="stable").index[:n - 4])
    top = _assign(t.loc[winners, [1, 2, 3, 4]].to_numpy())
    rest = _assign(t.loc[wild, list(range(5, n + 1))].to_numpy())
    return [winners[i] for i in top] + [wild[i] for i in rest]


def game_prob(inp: SimInput, home: str, away: str, neutral: bool = False) -> float:
    """The chance `home` beats `away` at home (or at a neutral site) in a
    playoff game: the pair's expected margin over the game's and both teams'
    spreads."""
    i, j = inp.teams.index(home), inp.teams.index(away)
    mu = inp.pair[i, j] + (0.0 if neutral else inp.hfa)
    return float(ndtr(mu / np.sqrt(inp.game_sd ** 2 + inp.team_sd[i] ** 2 + inp.team_sd[j] ** 2)))


def draft_table(res: SimResult, level: float) -> pd.DataFrame:
    """One row per team, by its average draft pick: the range holding `level`
    of the runs, the most common pick, and the share of runs it picks first,
    in the top three, five and ten."""
    lo_q, hi_q = (1 - level) / 2, (1 + level) / 2
    p = res.pick.astype(int)
    mode = [int(np.bincount(p[:, i]).argmax()) for i in range(p.shape[1])]
    df = pd.DataFrame({
        "Team": res.teams, "Now": record_now(res), "W": res.wins.mean(0),
        "Avg pick": p.mean(0), "Pick low": np.quantile(p, lo_q, axis=0),
        "Pick high": np.quantile(p, hi_q, axis=0), "Likeliest": mode,
        "#1 %": (p == 1).mean(0), "Top 3 %": (p <= 3).mean(0), "Top 5 %": (p <= 5).mean(0),
        "Top 10 %": (p <= 10).mean(0), "Playoffs %": (res.seed > 0).mean(0),
    })
    return df.sort_values("Avg pick").reset_index(drop=True)


def wins_distribution(res: SimResult, team: str) -> pd.Series:
    """The share of runs `team` ends the regular season with each number of wins."""
    w = res.wins[:, res.teams.index(team)]
    return pd.Series(w).value_counts(normalize=True).sort_index()


def leverage(res: SimResult, team: str, what: str = "playoffs") -> pd.DataFrame:
    """For every game left, `team`'s chance of `what` ("playoffs",
    "division", "bye") in the runs where the home team won and in those where
    it lost, and the swing between them; the team's own games first, then
    the rest by swing."""
    t = res.teams.index(team)
    s = res.seed[:, t]
    byes = 1 if res.seed.max() >= 7 else 2
    y = {"playoffs": s > 0, "division": (s >= 1) & (s <= 4), "bye": (s >= 1) & (s <= byes)}[what]
    y = y.astype(float)
    o = res.outcomes
    n_home = o.sum(0)
    if_home = np.where(n_home > 0, (o * y[:, None]).sum(0) / np.maximum(n_home, 1), np.nan)
    if_away = np.where(res.runs - n_home > 0, ((~o) * y[:, None]).sum(0) / np.maximum(res.runs - n_home, 1),
                       np.nan)
    df = res.left[["game_id", "week", "home", "away"]].copy()
    df["Home wins"], df["Away wins"] = if_home, if_away
    df["Swing"] = np.abs(if_home - if_away)
    df["own"] = (df["home"] == team) | (df["away"] == team)
    return df.sort_values(["own", "Swing"], ascending=[False, False]).reset_index(drop=True)


def call(p: float, level: float) -> str:
    """A playoff chance in words at a confidence `level`: in or out in every
    run, likely (at least `level` of the runs), unlikely (at most 1 − level)
    or on the bubble."""
    if p >= 1:
        return "✅ In"
    if p <= 0:
        return "❌ Out"
    return "Likely" if p >= level else "Unlikely" if p <= 1 - level else "Bubble"


def sb_matchups(res: SimResult, n: int = 5) -> pd.DataFrame:
    """The `n` Super Bowl pairings the runs reach most (AFC team first), with
    their share of runs and the AFC team's share of those it wins."""
    two = np.argsort(res.stage < 4, axis=1, kind="stable")[:, :2]  # the two teams with stage ≥ 4
    conf = np.array([d.split(" ")[0] for d in res.divisions])
    afc_first = conf[two[:, 0]] == "AFC"
    afc = np.where(afc_first, two[:, 0], two[:, 1])
    nfc = np.where(afc_first, two[:, 1], two[:, 0])
    won = res.stage[np.arange(res.runs), afc] == 5
    df = pd.DataFrame({"AFC": [res.teams[i] for i in afc], "NFC": [res.teams[i] for i in nfc], "won": won})
    out = df.groupby(["AFC", "NFC"]).agg(share=("won", "size"), afc_wins=("won", "mean")).reset_index()
    out["share"] /= res.runs
    return out.sort_values("share", ascending=False).head(n).reset_index(drop=True)


def final_standings(sched: pd.DataFrame, divisions: dict[str, str]) -> pd.DataFrame | None:
    """What happened, once the regular season is over: each team's final
    record and seed (0: missed), and its STAGES index and draft pick once the
    postseason is over too (NaN before; a tie for a pick past strength of
    schedule is a coin flip here). None while regular-season games are left."""
    reg = regular_season(sched)
    if not reg["played"].all():
        return None
    model = {"hfa": 0.0, "game_sd": 1.0, "tau": {"floor": 0.0, "scale": 0.0, "g1": 1.0}}
    inp = build_input(int(sched["season"].iloc[0]), reg, divisions, pd.Series(dtype=float),
                      pd.DataFrame(), model, pd.Series(dtype=float), playoff_results(sched))
    res = simulate(inp, runs=1)
    done = sched["game_type"].eq("SB").any() and sched.loc[sched["game_type"] == "SB", "result"].notna().all()
    return pd.DataFrame({"Team": res.teams, "Final": record_now(res), "Seed": res.seed[0],
                         "Stage": res.stage[0] if done else np.nan,
                         "Pick": res.pick[0] if done else np.nan})
