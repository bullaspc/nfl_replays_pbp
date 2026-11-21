"""Team records for the replay's scoreboard line: each team's wins and losses
going into the game and, in college, its conference record, from the league's
schedule, under its name.

A separate module on purpose, like rosters.py: it isn't in
`leagues._CODE_FILES`, so a change here doesn't make the season store rebuild
(nothing in it feeds a stat).

Spoiler-free: a record counts only the finished games that kicked off before
this one, never this game's result (the schedule has it) or a later game's.
Once the revealed plays reach the final whistle (core.game_over), the page
passes this game's margin from its own revealed score, and the record
includes it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import leagues

# One row per game: kickoff (comparable within one league's frame), the two
# teams as the play-by-play abbreviates them, final scores (NaN until the game
# is final), each team's conference, whether the game counts in the record
# and whether it's a conference game.
RESULT_COLS = ["game_id", "kickoff", "home", "away", "home_score", "away_score",
               "home_conf", "away_conf", "counts", "conference"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=RESULT_COLS)


def nfl_results(sched: pd.DataFrame) -> pd.DataFrame:
    """nflverse's schedule as RESULT_COLS, kickoff from its gameday and
    gametime (US Eastern). Only regular-season games count, as the standings
    have it, so a playoff game shows (and keeps) the regular-season record.
    No conference records."""
    need = {"game_id", "gameday", "home_team", "away_team", "home_score", "away_score"}
    if sched.empty or not need <= set(sched.columns):
        return _empty()
    time = (sched["gametime"] if "gametime" in sched.columns
            else pd.Series("00:00", index=sched.index)).fillna("00:00").astype(str)
    return pd.DataFrame({
        "game_id": sched["game_id"].astype(str),
        "kickoff": pd.to_datetime(sched["gameday"].astype(str) + " " + time, errors="coerce"),
        "home": sched["home_team"], "away": sched["away_team"],
        "home_score": pd.to_numeric(sched["home_score"], errors="coerce"),
        "away_score": pd.to_numeric(sched["away_score"], errors="coerce"),
        "home_conf": None, "away_conf": None,
        "counts": (sched["game_type"].eq("REG") if "game_type" in sched.columns
                   else pd.Series(True, index=sched.index)),
        "conference": False,
    }).reset_index(drop=True)


def cfb_results(sched: pd.DataFrame, ids: pd.DataFrame) -> pd.DataFrame:
    """The college schedule (cfb_feed.read_schedule) as RESULT_COLS, teams as
    the play-by-play's abbreviations through `ids` (cfb_feed.team_ids, by
    ESPN team id: the schedule leaves most abbreviations blank). A team `ids`
    doesn't know (a lower-division opponent) keeps its id, "#2345", so its
    games still count for the other team. Every game counts (FCS opponents,
    conference title games, bowls), as the NCAA's records do; the conference
    record counts the games the schedule marks `conference_game`, which leaves
    out the conference title games, as the standings do."""
    need = {"game_id", "start_date", "home_id", "away_id", "home_points", "away_points"}
    if sched.empty or not need <= set(sched.columns):
        return _empty()
    abbr = dict(zip(ids["id"].astype(int), ids["abbr"])) if not ids.empty else {}

    def team(side: str) -> list:
        return [abbr.get(int(i), f"#{int(i)}") if pd.notna(i) else None
                for i in pd.to_numeric(sched[f"{side}_id"], errors="coerce")]

    done = (sched["completed"].eq(True) if "completed" in sched.columns
            else pd.Series(True, index=sched.index))

    def conf(side: str):
        col = f"{side}_conference"
        return sched[col] if col in sched.columns else None

    return pd.DataFrame({
        "game_id": sched["game_id"].astype(str),
        "kickoff": pd.to_datetime(sched["start_date"], utc=True, errors="coerce"),
        "home": team("home"), "away": team("away"),
        "home_score": pd.to_numeric(sched["home_points"], errors="coerce").where(done),
        "away_score": pd.to_numeric(sched["away_points"], errors="coerce").where(done),
        "home_conf": conf("home"), "away_conf": conf("away"),
        "counts": True,
        "conference": (sched["conference_game"].eq(True) if "conference_game" in sched.columns
                       else False),
    }).reset_index(drop=True)


@st.cache_data(ttl=3600, max_entries=4, show_spinner=False)
def _game_results(league_key: str, season: int, stamp: str) -> pd.DataFrame:
    """Raises when there's nothing to show, so an unreadable schedule isn't
    cached for the hour."""
    if league_key == "NFL":
        out = nfl_results(leagues.load_schedule(season))
    elif league_key == "CFB":
        out = cfb_results(leagues.load_cfb_schedule(season),
                          leagues.load_cfb_team_ids(season, stamp))
    else:
        out = _empty()
    if out.empty:
        raise LookupError(f"no {league_key} {season} schedule")
    return out


def game_results(league_key: str, season: int, stamp: str) -> pd.DataFrame:
    """The season's games as RESULT_COLS for `league_key` (League.key: "NFL",
    "CFB"); empty if the schedule can't be read. `stamp` (the play-by-play's)
    keys the cache."""
    try:
        return _game_results(league_key, int(season), stamp)
    except Exception:
        return _empty()


def _wlt(margins: np.ndarray) -> tuple[int, int, int]:
    return int((margins > 0).sum()), int((margins < 0).sum()), int((margins == 0).sum())


def _independent(conf) -> bool:
    """No conference record: no conference in the data, or an independent."""
    return not isinstance(conf, str) or not conf.strip() or "independent" in conf.lower()


def _sides(own: pd.Series, away: str, home: str) -> dict[str, str]:
    """The game's teams (as its play-by-play names them) → their side of its
    own `results` row, "home" or "away": by name, else by position where that
    side isn't the other team's (a school whose abbreviation changed, JXST
    / JVST)."""
    sides = {t: s for t in (away, home) for s in ("home", "away") if own[s] == t}
    for t, s in ((away, "away"), (home, "home")):
        if t not in sides and s not in sides.values():
            sides[t] = s
    return sides


def team_records(results: pd.DataFrame, game_id, away: str, home: str,
                 final: dict[str, float] | None = None) -> dict[str, dict]:
    """The `away` and `home` teams' records going into game `game_id`:
    {"overall": (wins, losses, ties), "conference": (w, l, t), or None for an
    independent or a league without conference records, "conf_name": the
    schedule's conference, or None likewise}, over the finished
    games of `results` (RESULT_COLS) that kicked off before it and count.
    Never this game's result or a later game's.

    `final` (team → this game's margin, from the revealed final score) adds
    the game itself, where it counts. Empty if the game isn't in `results`."""
    if results.empty:
        return {}
    own = results[results["game_id"] == str(game_id)]
    if own.empty or pd.isna(own["kickoff"].iloc[0]):
        return {}
    own = own.iloc[0]
    prior = results[(results["kickoff"] < own["kickoff"]) & results["counts"].astype(bool)
                    & results["home_score"].notna() & results["away_score"].notna()]
    out = {}
    for t, side in _sides(own, away, home).items():
        key = own[side]  # the team as `results` names it
        h, a = prior[prior["home"] == key], prior[prior["away"] == key]
        margin = np.concatenate([(h["home_score"] - h["away_score"]).to_numpy(float),
                                 (a["away_score"] - a["home_score"]).to_numpy(float)])
        in_conf = np.concatenate([h["conference"].to_numpy(bool), a["conference"].to_numpy(bool)])
        if final is not None and t in final and bool(own["counts"]):
            margin = np.append(margin, float(final[t]))
            in_conf = np.append(in_conf, bool(own["conference"]))
        conf = own[f"{side}_conf"]
        out[t] = {"overall": _wlt(margin),
                  "conference": None if _independent(conf) else _wlt(margin[in_conf]),
                  "conf_name": None if _independent(conf) else conf}
    return out


def record_label(wlt: tuple[int, int, int] | None) -> str:
    """(8, 2, 0) → "8-2"; a tie shows, "8-2-1"; None → "—"."""
    if wlt is None:
        return "—"
    w, l, t = wlt
    return f"{w}-{l}" + (f"-{t}" if t else "")


# Long conference names, shortened under a team's name so the scoreboard line
# stays narrow on a phone; the rest go as the schedule writes them.
CONF_SHORT = {"American Athletic": "American", "Mid-American": "MAC", "Mountain West": "MWC",
              "Conference USA": "CUSA", "Coastal Athletic": "CAA"}


def record_line(rec: dict) -> str:
    """A team's record as the line under its name in the scoreboard: "8-2",
    or in college "8-2, 5-1 Big Ten" (an independent has no conference part).
    Non-breaking hyphens and spaces, so on a phone it can only wrap after the
    comma."""
    line = record_label(rec["overall"])
    if rec.get("conference") is not None:
        name = rec.get("conf_name") or "Conf"
        line += f", {record_label(rec['conference'])}\u00a0{CONF_SHORT.get(name, name).replace(' ', chr(0xa0))}"
    return line.replace("-", "\u2011")


def record_text(rec: dict) -> str:
    """A team's record as plain text for the AI summary and booth: "8-2", or
    in college "8-2, 5-1 in American Athletic games" (the full conference
    name)."""
    text = record_label(rec["overall"])
    if rec.get("conference") is not None:
        text += f", {record_label(rec['conference'])} in {rec.get('conf_name') or 'conference'} games"
    return text


def summary_records(records: dict[str, dict]) -> dict[str, str]:
    """team_records() as team → record_text(), for game_summary.GameContext."""
    return {t: record_text(r) for t, r in records.items()}


def record_note(results: pd.DataFrame, game_id, final: bool) -> str:
    """What the records count: the records' tooltip, and how the AI summary's
    snapshot introduces them. `final`: the game's end is revealed."""
    own = results[results["game_id"] == str(game_id)]
    counts = own.empty or bool(own["counts"].iloc[0])
    return ("Regular-season records" if not counts else
            "Records including this game" if final else "Records going into the game")


def final_margins(away: str, home: str, away_score: int, home_score: int) -> dict[str, float]:
    """Team → its margin in the final score, for team_records' `final`."""
    return {away: away_score - home_score, home: home_score - away_score}
