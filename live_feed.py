"""
Live play-by-play for games nflverse hasn't published yet.

ESPN's public game summary carries each play's NFL gamebook text plus the
clock, down/distance, field position and score. This module turns that into
the nflfastR column layout the app reads, so the rest of the app can't tell
the difference. EP/EPA/WP are then added by nflfastR's own models
(nflfastr_models.py). Once nflverse publishes the official pbp, the app
switches to that.

Almost everything nflfastR itself derives from the gamebook text (play type,
players, yards, results, defenders) is parsed here from the same text with
`parse_play_text`, which is validated against published nflverse pbp by
tools/validate_text_parser.py.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import requests

SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"

# ESPN → nflverse team codes (only the ones that differ).
ESPN_TEAM_FIX = {"WSH": "WAS", "LAR": "LA", "JAC": "JAX"}

# How the gamebook names a team in challenge text ("Green Bay challenged ...").
TEAM_PLACES = {
    "Arizona": "ARI", "Atlanta": "ATL", "Baltimore": "BAL", "Buffalo": "BUF",
    "Carolina": "CAR", "Chicago": "CHI", "Cincinnati": "CIN", "Cleveland": "CLE",
    "Dallas": "DAL", "Denver": "DEN", "Detroit": "DET", "Green Bay": "GB",
    "Houston": "HOU", "Indianapolis": "IND", "Jacksonville": "JAX", "Kansas City": "KC",
    "Los Angeles Rams": "LA", "Los Angeles Chargers": "LAC", "Las Vegas": "LV",
    "Miami": "MIA", "Minnesota": "MIN", "New England": "NE", "New Orleans": "NO",
    "New York Giants": "NYG", "New York Jets": "NYJ", "Philadelphia": "PHI",
    "Pittsburgh": "PIT", "Seattle": "SEA", "San Francisco": "SF", "Tampa Bay": "TB",
    "Tennessee": "TEN", "Washington": "WAS",
}
_CHALLENGE_TO = re.compile(
    "(" + "|".join(sorted(map(re.escape, TEAM_PLACES), key=len, reverse=True))
    + r") challenged.*?\(Timeout #\d+\.\)")

# Defensive fouls that give the offense a first down whatever the yardage.
AUTO_FIRST_DOWN_FOULS = {
    "Defensive Holding", "Defensive Pass Interference", "Illegal Contact",
    "Roughing the Passer", "Unnecessary Roughness", "Face Mask", "Horse Collar Tackle",
    "Illegal Use of Hands", "Unsportsmanlike Conduct", "Taunting", "Roughing the Kicker",
    "Lowering the Head to Initiate Contact",
}

# ---------- Gamebook text parsing ----------
# A player as the gamebook writes them: "J.Hurts", "Ja.Horn", "A.St. Brown",
# "L.Van Ness", "A.Al-Shaair", "D.Smith-Njigba". Jersey numbers ("1-J.Hurts")
# are stripped before matching.
_PARTICLE = r"(?:Van|Vander|St\.|De|Da|Di|Du|La|Le|El|Mac|Ta|Te|Ya|Von|Des)"
NAME = (rf"[A-Z][A-Za-z]{{0,3}}\.(?:[A-Z][A-Za-z]{{0,2}}\.)?\s?"
        rf"(?:{_PARTICLE}\s)?[A-Z][A-Za-z'\-]*(?:\s(?:Jr\.|Sr\.|II|III|IV))?")
_NAME_LIST = rf"{NAME}(?:\s*[;,]\s*{NAME})*"

_JERSEY = re.compile(r"(?<![\w-])\d{1,2}-(?=[A-Z][A-Za-z]{0,3}\.)")
_TEAM_JERSEY = re.compile(r"\b([A-Z]{2,3})-\d{1,2}-")
_LEAD = re.compile(r"^\s*(?:\([^)]*\)\s*)+")          # "(12:48) (Shotgun) "
_YARDS = re.compile(r"for (-?\d+) yards?|for (no gain)")
_RUN_DIR = re.compile(r"\b(?:left|right) (?:end|tackle|guard)\b|\bup the middle\b|\brushes\b")
_PREAMBLE = re.compile(rf"^(?:(?:{NAME}(?:,? and {NAME})*|{NAME}(?:, {NAME})+,? and {NAME}) "
                       rf"reported in as eligible\.\s*|Direct snap to {NAME}\.\s*)+")
_PAREN_NAMES = re.compile(rf"\(({_NAME_LIST})\)")
_BRACKET_NAMES = re.compile(rf"\[({_NAME_LIST})\]")


def clean_desc(desc: str) -> str:
    """Drop jersey numbers so both nflverse's and ESPN's text parse the same."""
    s = _TEAM_JERSEY.sub(r"\1-", desc)
    return _JERSEY.sub("", s)


def _names(group: str) -> list[str]:
    return [n.strip() for n in re.split(r"\s*[;,]\s*", group) if n.strip()]


def _first(pattern: str, s: str) -> str | None:
    m = re.search(pattern, s)
    return m.group(1) if m else None


def parse_play_text(desc: str) -> dict:
    """nflfastR-style fields from one play's gamebook text.

    Returns play_type plus the pass/rush/result/player columns the app uses.
    Everything not recognized is left out of the dict (→ NaN).
    """
    out: dict = {}
    if not isinstance(desc, str) or not desc.strip():
        return out
    raw = clean_desc(desc)
    body = _LEAD.sub("", raw)
    # After a reversed review the gamebook restates the play as ruled.
    if "the play was REVERSED." in body:
        body = _LEAD.sub("", body.rsplit("the play was REVERSED.", 1)[1])
    # Formation notes that precede the snap aren't part of the play.
    body = _PREAMBLE.sub("", body)
    body = _LEAD.sub("", body)
    upper = body.upper()

    # --- play type (nflfastR's values) ---
    no_play = bool(re.search(r"\bNo Play\b", body, re.I)) or (
        bool(re.search(r"PENALTY on", body, re.I))
        and not re.search(r" pass | sacked | scrambles | punts | kicks |field goal|extra point|kneels|spiked", body)
        and not _RUN_DIR.search(body))
    if re.match(r"^(END (QUARTER|GAME)|END OF|GAME$|Two-Minute Warning)", body.strip(), re.I):
        play_type = None
    elif re.match(r"Timeout #\d", body):
        play_type = "no_play"
    elif no_play:
        play_type = "no_play"
    elif re.search(r" kicks (?:onside )?(?:-?\d+ yards? )?from ", body) or " kicks onside" in body:
        play_type = "kickoff"
    elif " punts " in body or re.search(r"\bpunt is BLOCKED", body):
        play_type = "punt"
    elif "field goal" in body:
        play_type = "field_goal"
    elif "extra point" in body:
        play_type = "extra_point"
    elif " kneels" in body:
        play_type = "qb_kneel"
    elif " spiked" in body:
        play_type = "qb_spike"
    elif re.search(r" pass |sacked|INTERCEPTED", body):
        play_type = "pass"
    elif " scrambles " in body or _RUN_DIR.search(body) or re.search(r"\bAborted\b", body):
        play_type = "run"
    else:
        play_type = None
    if "TWO-POINT CONVERSION ATTEMPT" in upper:
        # nflfastR keeps the run/pass type on successful-snap 2-pt tries.
        play_type = "pass" if " pass " in body else ("run" if _RUN_DIR.search(body) or "rushes" in body else play_type)
    out["play_type"] = play_type

    two_pt = "TWO-POINT CONVERSION ATTEMPT" in upper
    if two_pt:
        out["two_point_conv_result"] = "success" if "ATTEMPT SUCCEEDS" in upper else "failure"

    # --- yardage ---
    m = _YARDS.search(body)
    yards = 0.0 if (m and m.group(2)) else (float(m.group(1)) if m else np.nan)

    is_pass = play_type == "pass" and not two_pt
    is_sack = is_pass and " sacked" in body
    is_int = is_pass and "INTERCEPTED" in body
    is_inc = is_pass and "incomplete" in body
    is_run = play_type == "run" and not two_pt
    scramble = is_run and " scrambles " in body

    if is_pass or is_run or play_type in ("qb_kneel", "qb_spike"):
        out["yards_gained"] = 0.0 if (is_inc or is_int or np.isnan(yards)) else yards
    out["pass_attempt"] = float(is_pass or (two_pt and " pass " in body))
    out["rush_attempt"] = float(is_run or play_type == "qb_kneel")
    out["sack"] = float(is_sack)
    out["interception"] = float(is_int)
    out["qb_kneel"] = float(play_type == "qb_kneel")
    out["qb_spike"] = float(play_type == "qb_spike")
    out["complete_pass"] = float(is_pass and not (is_sack or is_int or is_inc))

    # --- offensive players ---
    if is_pass or play_type == "qb_spike":
        out["passer_player_name"] = _first(rf"^({NAME}) (?:pass|sacked|spiked)", body)
    if is_pass and not is_sack:
        out["receiver_player_name"] = (_first(rf"intended for ({NAME})", body)
                                       or _first(rf" pass (?:incomplete )?(?:short|deep) (?:left|middle|right) to ({NAME})", body)
                                       or _first(rf" pass (?:incomplete )?to ({NAME})", body))
    if is_run or play_type == "qb_kneel":
        out["rusher_player_name"] = _first(rf"^({NAME}) (?:scrambles|kneels|left|right|up|rushes|Aborted)", body)
    if out.get("complete_pass") == 1 and not np.isnan(yards):
        out["passing_yards"] = yards
        out["receiving_yards"] = yards
    if (is_run or play_type == "qb_kneel") and not np.isnan(yards):
        out["rushing_yards"] = yards

    # --- results ---
    td = "TOUCHDOWN" in upper and "TOUCHDOWN NULLIFIED" not in upper and play_type != "no_play" and not two_pt
    out["touchdown"] = float(td)
    out["pass_touchdown"] = float(td and is_pass and not is_int and not is_sack
                                  and "FUMBLES" not in upper)
    out["rush_touchdown"] = float(td and is_run and "FUMBLES" not in upper)
    out["safety"] = float("SAFETY" in upper and play_type != "no_play")
    if play_type == "field_goal":
        out["field_goal_result"] = ("made" if re.search(r"field goal is GOOD", body)
                                    else "blocked" if "BLOCKED" in upper else "missed")
    if play_type == "extra_point":
        out["extra_point_result"] = ("good" if re.search(r"extra point is GOOD", body)
                                     else "blocked" if "BLOCKED" in upper
                                     else "aborted" if "Aborted" in body else "failed")
    out["fumble"] = float("FUMBLES" in upper)
    out["_recovered_by"] = _first(r"RECOVERED by ([A-Z]{2,3})-", body)
    out["penalty"] = float(bool(re.search(r"PENALTY on", body, re.I)))
    # nflfastR credits the first penalty that wasn't declined.
    for seg in re.split(r"(?=PENALTY on )", body, flags=re.I)[1:]:
        pm = re.match(r"PENALTY on ([A-Z]{2,3})(?:-|,)", seg, re.I)
        if not pm or re.search(r"declined|offsetting", seg.split(" - ")[0], re.I):
            continue
        out["penalty_team"] = pm.group(1).upper()
        foul = re.match(rf"PENALTY on [A-Z]{{2,3}}(?:-{NAME})?, ([^,]+),", seg, re.I)
        if foul:
            out["penalty_type"] = foul.group(1).strip()
        yds = re.search(r", (\d+) yards?, enforced", seg)
        if yds:
            out["penalty_yards"] = float(yds.group(1))
        break

    # --- defenders ---
    # The first player group in parentheses after the action is the tackle
    # (solo if one name; two names split by ';' or ',' are assists); on an
    # incomplete pass it's the defender who broke it up; on a sack, the sacker.
    action = re.split(r"\. (?=[A-Z])|FUMBLES|INTERCEPTED by|PENALTY on|Penalty on", body)[0]
    split_sack = re.search(rf"sack split by ({NAME}) and ({NAME})", body)
    grp = _PAREN_NAMES.search(action)
    tacklers = _names(grp.group(1)) if grp else []
    tackle_sep = "," if grp and "," in grp.group(1) else ";"
    if is_inc:
        for i, n in enumerate(tacklers[:2], 1):
            out[f"pass_defense_{i}_player_name"] = n
        tacklers = []
    if is_sack:
        if split_sack:
            out["half_sack_1_player_name"] = split_sack.group(1)
            out["half_sack_2_player_name"] = split_sack.group(2)
        elif tacklers:
            out["sack_player_name"] = tacklers[0]
    if is_int:
        out["interception_player_name"] = _first(rf"INTERCEPTED by ({NAME})", body)
        # A tipped pick credits the tipper; otherwise the interceptor.
        out["pass_defense_1_player_name"] = (_first(rf"INTERCEPTED by {NAME} \(({NAME})\)", body)
                                             or out["interception_player_name"])
        # Tackle on the return, after the INT.
        after = body.split("INTERCEPTED by", 1)[1]
        after = after.split(". ", 1)[1] if ". " in after else ""
        g2 = _PAREN_NAMES.search(re.split(r"FUMBLES|PENALTY on", after)[0])
        tacklers = _names(g2.group(1)) if g2 else []
        tackle_sep = "," if g2 and "," in g2.group(1) else ";"
    if play_type in ("kickoff", "punt"):
        tacklers = []
        ret = re.split(r"FUMBLES|PENALTY on", body)[0]
        for g2 in _PAREN_NAMES.finditer(ret):
            tacklers = _names(g2.group(1))
            tackle_sep = "," if "," in g2.group(1) else ";"
    if play_type == "no_play":
        tacklers = []
    if len(tacklers) == 1:
        out["solo_tackle_1_player_name"] = tacklers[0]
    elif tacklers and tackle_sep == ",":
        # "(A, B)": A made the tackle, B assisted.
        out["tackle_with_assist_1_player_name"] = tacklers[0]
        for i, n in enumerate(tacklers[1:4], 1):
            out[f"assist_tackle_{i}_player_name"] = n
    elif tacklers:
        # "(A; B)": a shared tackle, both credited with an assist.
        for i, n in enumerate(tacklers[:4], 1):
            out[f"assist_tackle_{i}_player_name"] = n
    if (is_run or (is_pass and out.get("complete_pass") == 1) or is_sack) and yards < 0 and tacklers:
        for i, n in enumerate(tacklers[:2], 1):
            out[f"tackle_for_loss_{i}_player_name"] = n
    ff = re.search(rf"FUMBLES \(({NAME})\)", body)
    if ff:
        out["forced_fumble_player_1_player_name"] = ff.group(1)
    hits = _BRACKET_NAMES.search(body)
    hitters = _names(hits.group(1)) if hits and is_pass else []
    if is_sack and not hitters:
        # A sack counts as a QB hit for the sacker(s).
        hitters = ([split_sack.group(1), split_sack.group(2)] if split_sack
                   else tacklers[:1])
    for i, n in enumerate(hitters[:2], 1):
        out[f"qb_hit_{i}_player_name"] = n
    out["qb_hit"] = float(bool(hitters))
    return out


# ---------- Game-level derivations (shared by the ESPN path and its tests) ----------
def _clock_seconds(t) -> float:
    if not isinstance(t, str) or ":" not in t:
        return np.nan
    mm, ss = t.split(":", 1)
    try:
        return int(mm or 0) * 60 + int(ss)
    except ValueError:
        return np.nan


def add_derived_columns(base: pd.DataFrame, season_type: str = "REG") -> pd.DataFrame:
    """From one game's play rows (clock, down/distance, field position, teams,
    running score and gamebook text) derive every other column the app and
    nflfastR's models use, the way nflfastR does.

    `base` needs: game_id, season, home_team, away_team, qtr, time, desc, down,
    ydstogo, yardline_100, posteam, defteam, drive, total_home_score,
    total_away_score, roof, location. Rows must be in game order.
    """
    df = base.reset_index(drop=True).copy()
    parsed = pd.DataFrame([parse_play_text(d) for d in df["desc"]], index=df.index)
    for c in parsed.columns:
        df[c] = parsed[c]
    home, away = df["home_team"].iloc[0], df["away_team"].iloc[0]

    # Clock.
    secs = df["time"].map(_clock_seconds)
    q = df["qtr"]
    df["game_seconds_remaining"] = np.where(q <= 4, (4 - q) * 900 + secs, secs)
    df["half_seconds_remaining"] = np.where(q.isin([1, 3]), 900 + secs, secs)
    df["quarter_seconds_remaining"] = secs

    # Score before the play, from the offense's side.
    pre_h = df["total_home_score"].shift(1).fillna(0)
    pre_a = df["total_away_score"].shift(1).fillna(0)
    pos_home = df["posteam"] == home
    df["posteam_score"] = np.where(df["posteam"].isna(), np.nan, np.where(pos_home, pre_h, pre_a))
    df["defteam_score"] = np.where(df["posteam"].isna(), np.nan, np.where(pos_home, pre_a, pre_h))
    df["score_differential"] = df["posteam_score"] - df["defteam_score"]
    d_h = df["total_home_score"] - pre_h
    d_a = df["total_away_score"] - pre_a
    df["sp"] = ((d_h != 0) | (d_a != 0)).astype(float)
    scorer = np.where(d_h > 0, home, np.where(d_a > 0, away, None))
    df["td_team"] = np.where(df["touchdown"] == 1, scorer, None)
    df["safety_team"] = np.where((df["safety"] == 1) & ((d_h == 2) | (d_a == 2)), scorer, None)
    df.loc[df["td_team"].isna() & (df["touchdown"] == 1), "td_team"] = np.nan

    # Timeouts left for each side, reset every half (and in overtime).
    ot_tos = 3 if season_type == "POST" else 2
    h_to = a_to = 3
    cur_q = None
    home_left, away_left = [], []
    for qq, d in zip(df["qtr"], df["desc"].fillna("")):
        if qq != cur_q and qq in (1, 3, 5):
            h_to = a_to = ot_tos if qq == 5 else 3
        cur_q = qq
        # Timeouts also hide inside challenge/review text, so scan it all.
        charged = [ESPN_TEAM_FIX.get(t, t) for t in re.findall(r"Timeout #\d+ by ([A-Z]{2,3})\b", d)]
        charged += [TEAM_PLACES[t] for t in _CHALLENGE_TO.findall(d)]
        for team in charged:
            if team == home:
                h_to = max(h_to - 1, 0)
            elif team == away:
                a_to = max(a_to - 1, 0)
        home_left.append(h_to)
        away_left.append(a_to)
    df["home_timeouts_remaining"] = home_left
    df["away_timeouts_remaining"] = away_left
    df["posteam_timeouts_remaining"] = np.where(
        df["posteam"].isna(), np.nan, np.where(pos_home, df["home_timeouts_remaining"], df["away_timeouts_remaining"]))
    df["defteam_timeouts_remaining"] = np.where(
        df["posteam"].isna(), np.nan, np.where(pos_home, df["away_timeouts_remaining"], df["home_timeouts_remaining"]))
    df["timeout"] = df["desc"].fillna("").str.match(r"\s*Timeout #\d").astype(float)
    df["kickoff_attempt"] = (df["play_type"] == "kickoff").astype(float)

    # Conversions, turnovers.
    scrimmage = df["play_type"].isin(["pass", "run"])
    by_play = scrimmage & ((df["yards_gained"] >= df["ydstogo"]) | (df["touchdown"] == 1))
    ptype = df["penalty_type"] if "penalty_type" in df else pd.Series(np.nan, index=df.index)
    by_penalty = (df["down"].notna() & df["penalty_team"].notna() & (df["penalty_team"] == df["defteam"])
                  & ((df["penalty_yards"] >= df["ydstogo"]) | ptype.isin(AUTO_FIRST_DOWN_FOULS)))
    df["first_down"] = (by_play | by_penalty).astype(float)
    for d, name in ((3, "third"), (4, "fourth")):
        on_down = scrimmage & (df["down"] == d)
        df[f"{name}_down_converted"] = (on_down & (df["first_down"] == 1)).astype(float)
        df[f"{name}_down_failed"] = (on_down & (df["first_down"] == 0)).astype(float)
    rec = df.pop("_recovered_by").map(lambda t: ESPN_TEAM_FIX.get(t, t) if isinstance(t, str) else t)
    df["fumble_lost"] = ((df["fumble"] == 1) & rec.notna() & (rec == df["defteam"])).astype(float)
    df["goal_to_go"] = ((df["yardline_100"] <= df["ydstogo"]) & df["down"].notna()).astype(float)
    df["season_type"] = season_type
    return df


# ---------- ESPN feed ----------
def fetch_summary(event_id, timeout: float = 15) -> dict:
    """ESPN's game summary JSON (drives, plays, header)."""
    r = requests.get(SUMMARY_URL, params={"event": str(event_id)}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def game_state(summary: dict) -> str:
    """'pre', 'in' or 'post' from the summary header."""
    try:
        return summary["header"]["competitions"][0]["status"]["type"]["state"]
    except (KeyError, IndexError, TypeError):
        return "in"


def _team_lookup(summary: dict) -> dict[str, str]:
    """ESPN team id / name variants → nflverse abbreviation."""
    out: dict[str, str] = {}
    try:
        comps = summary["header"]["competitions"][0]["competitors"]
    except (KeyError, IndexError, TypeError):
        comps = []
    for c in comps:
        t = c.get("team", {})
        abbr = ESPN_TEAM_FIX.get(t.get("abbreviation", ""), t.get("abbreviation", ""))
        for key in (t.get("id"), t.get("abbreviation"), t.get("location"),
                    t.get("displayName"), t.get("shortDisplayName"), t.get("name")):
            if key:
                out[str(key)] = abbr
                out[str(key).upper()] = abbr
    return out


def _mmss(s) -> str | None:
    """'(:21)' / '0:21' / '12:05' → nflverse's 'MM:SS'."""
    if not isinstance(s, str):
        return None
    m = re.match(r"\s*\(?(\d{0,2}):(\d{2})\)?", s)
    if not m:
        return None
    return f"{int(m.group(1) or 0):02d}:{m.group(2)}"


_END_TYPES = re.compile(r"end (?:of )?(?:period|quarter|half|regulation|game)", re.I)
_PAT_SPLIT = re.compile(r"(?<=TOUCHDOWN)(?:[^.]*\.)?\s+(?=(?:\(Kick formation\)\s*|\(Pass formation\)\s*)?"
                        rf"(?:{NAME} extra point is|TWO-POINT CONVERSION ATTEMPT))")


def _espn_rows(summary: dict) -> list[dict]:
    """Every play in game order, deduplicated by ESPN play id."""
    drives = summary.get("drives") or {}
    seq = list(drives.get("previous") or [])
    if drives.get("current"):
        seq.append(drives["current"])
    rows, seen = [], set()
    for d_i, drive in enumerate(seq, 1):
        dteam = (drive.get("team") or {})
        for p in drive.get("plays") or []:
            pid = p.get("id") or f"{d_i}-{p.get('sequenceNumber')}"
            if pid in seen:
                continue
            seen.add(pid)
            rows.append({"_drive_i": d_i, "_drive_team": dteam.get("abbreviation") or dteam.get("id"),
                         "_p": p})
    return rows


def espn_to_base(summary: dict, sched: dict) -> pd.DataFrame:
    """ESPN summary → one row per play with the columns `add_derived_columns`
    starts from, in nflfastR's conventions (receiving team on kickoffs, a
    separate row for the try after a touchdown, 'END QUARTER n' rows)."""
    teams = _team_lookup(summary)
    home, away = sched["home_team"], sched["away_team"]

    def norm(t):
        if t is None:
            return None
        t = str(t)
        return teams.get(t) or teams.get(t.upper()) or ESPN_TEAM_FIX.get(t, t)

    out = []
    pre_h = pre_a = 0.0
    to_count = {home: 0, away: 0}
    last_q = None
    for r in _espn_rows(summary):
        p = r["_p"]
        q = (p.get("period") or {}).get("number")
        if q != last_q and q in (1, 3, 5):
            to_count = {home: 0, away: 0}
        last_q = q
        ptype = ((p.get("type") or {}).get("text") or "")
        text = (p.get("text") or "").strip()
        start = p.get("start") or {}
        start_team = norm((start.get("team") or {}).get("id")) or norm(r["_drive_team"])
        clock = _mmss(text) or _mmss((p.get("clock") or {}).get("displayValue"))
        post_h = float(p.get("homeScore", pre_h) or 0)
        post_a = float(p.get("awayScore", pre_a) or 0)

        row = {"qtr": q, "time": clock, "desc": text, "drive": r["_drive_i"],
               "down": np.nan, "ydstogo": np.nan, "yardline_100": np.nan,
               "posteam": start_team, "total_home_score": post_h, "total_away_score": post_a}

        # Admin rows, written the way nflfastR's rules look for them.
        if _END_TYPES.search(ptype) or re.match(r"END (QUARTER|GAME)", text, re.I):
            game_over = bool(re.search(r"game|regulation", ptype, re.I)) or "GAME" in text.upper()
            row["desc"] = "END GAME" if game_over else f"END QUARTER {q}"
            row["posteam"] = None
        elif re.search(r"two.minute", ptype + text, re.I):
            row["desc"] = "Two-Minute Warning"
            row["posteam"] = None
        elif re.search(r"timeout", ptype, re.I) or re.match(r"Timeout", text, re.I):
            team = None
            m = re.search(r"Timeout #\d+ by ([A-Z]{2,3})", text)
            if m:
                team = norm(m.group(1))
            else:
                for key, abbr in teams.items():
                    if len(key) > 3 and key.upper() in text.upper():
                        team = abbr
                        break
            if team in to_count and "official" not in (ptype + text).lower():
                to_count[team] += 1
                row["desc"] = f"Timeout #{to_count[team]} by {team} at {clock or ''}."
            row["posteam"] = None
        else:
            down = start.get("down")
            if down and 1 <= int(down) <= 4:
                row["down"] = float(down)
                row["ydstogo"] = float(start.get("distance") or np.nan)
            ytg = start.get("yardsToEndzone")
            if ytg is not None and 0 < float(ytg) < 100:
                row["yardline_100"] = float(ytg)
            if re.search(r" kicks ", text):
                # nflfastR: the receiving team has the ball on a kickoff.
                m = re.search(r" from ([A-Z]{2,3}) (\d+)", text)
                kicker = norm(m.group(1)) if m else None
                if kicker in (home, away):
                    row["posteam"] = away if kicker == home else home
                    row["yardline_100"] = float(m.group(2))
                row["down"] = row["ydstogo"] = np.nan

        # A touchdown row that also carries the try becomes two rows.
        parts = _PAT_SPLIT.split(row["desc"], maxsplit=1) if "TOUCHDOWN" in row["desc"] else [row["desc"]]
        if len(parts) == 2:
            td_row = dict(row, desc=parts[0].strip())
            try_row = dict(row, desc=parts[1].strip(), down=np.nan, ydstogo=np.nan,
                           yardline_100=2.0 if "TWO-POINT" in parts[1] else 15.0)
            if post_h - pre_h >= 6:
                td_row["total_home_score"] = pre_h + 6
            if post_a - pre_a >= 6:
                td_row["total_away_score"] = pre_a + 6
            out += [td_row, try_row]
        else:
            if "extra point" in row["desc"] or "TWO-POINT" in row["desc"]:
                row["yardline_100"] = 2.0 if "TWO-POINT" in row["desc"] else 15.0
                row["down"] = row["ydstogo"] = np.nan
            out.append(row)
        pre_h, pre_a = post_h, post_a

    df = pd.DataFrame(out)
    if df.empty:
        return df
    df["defteam"] = np.where(df["posteam"] == home, away, np.where(df["posteam"] == away, home, None))
    df.loc[df["posteam"].isna(), "defteam"] = None
    df["game_id"] = sched["game_id"]
    df["season"] = int(sched["season"])
    df["week"] = sched.get("week")
    df["game_date"] = sched.get("gameday")
    df["home_team"], df["away_team"] = home, away
    roof = sched.get("roof")
    df["roof"] = roof if isinstance(roof, str) and roof else np.nan
    df["location"] = sched.get("location") or "Home"
    df["spread_line"] = sched.get("spread_line")
    return df


def build_live_pbp(summary: dict, sched: dict, models: dict | None, fg_table: pd.DataFrame | None,
                   columns: list[str]) -> pd.DataFrame:
    """ESPN summary → nflfastR-layout pbp for one game, with nflfastR's EP/WP
    when `models` is given. Every name in `columns` is present (NaN if unknown)."""
    base = espn_to_base(summary, sched)
    if base.empty:
        return pd.DataFrame(columns=columns)
    game_type = sched.get("game_type") or "REG"
    df = add_derived_columns(base, "REG" if game_type == "REG" else "POST")
    if models is not None:
        import nflfastr_models as nm
        df = nm.add_ep_wp(df, models, fg_table)
    df["play_id"] = np.arange(1, len(df) + 1, dtype=float)
    for c in columns:
        if c not in df.columns:
            df[c] = np.nan
    return df.reset_index(drop=True)
