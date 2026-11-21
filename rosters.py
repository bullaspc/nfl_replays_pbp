"""College rosters for the Season stats Team tab's collapsed Roster section.

CFBD's season rosters through cfbfastR-data (the same file `cfb_feed.read_rosters`
reads for the leaders' class year), with the columns that one drops: jersey,
position, height, weight and hometown. A separate module on purpose: it isn't
in `leagues._CODE_FILES`, so a change here doesn't make the season store
rebuild (nothing in it feeds a stat).
"""
from __future__ import annotations

import io

import pandas as pd
import streamlit as st

import cfb_feed
from replay_core import CLASS_YEARS

ROSTER_FULL_COLS = ["first_name", "last_name", "team", "jersey", "position", "year",
                    "height", "weight", "home_city", "home_state", "home_country"]

# CFBD position → the group the Roster's filter and sort use.
POSITION_GROUPS = {
    "QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
    "OL": "OL", "OT": "OL", "OG": "OL", "C": "OL", "G": "OL", "T": "OL",
    "DL": "DL", "DE": "DL", "DT": "DL", "NT": "DL", "EDGE": "DL",
    "LB": "LB", "ILB": "LB", "OLB": "LB",
    "DB": "DB", "CB": "DB", "S": "DB", "FS": "DB", "SS": "DB", "NB": "DB",
    "K": "ST", "P": "ST", "PK": "ST", "LS": "ST", "KR": "ST", "PR": "ST",
}
GROUP_ORDER = ["QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB", "ST", "Other"]
ROSTER_COLUMNS = ["#", "Player", "Pos", "Yr", "Ht", "Wt", "Hometown"]


def read_roster(season: int, names: dict[str, str]) -> pd.DataFrame:
    """The season's full rosters (`ROSTER_FULL_COLS`), team as the play-by-play's
    abbreviation through `names` (abbreviation → school); schools it doesn't
    know are dropped. Empty if it can't be read."""
    try:
        r = pd.read_parquet(io.BytesIO(cfb_feed._get(cfb_feed.ROSTER_URL.format(int(season)))),
                            columns=ROSTER_FULL_COLS)
    except Exception:
        return pd.DataFrame(columns=ROSTER_FULL_COLS)
    school_abbr = {school: abbr for abbr, school in names.items()}
    return r.assign(team=r["team"].map(school_abbr)).dropna(subset=["team"]).reset_index(drop=True)


@st.cache_data(ttl=86400, max_entries=2, show_spinner=False)
def load_roster(season: int, names_key: tuple, _names: dict[str, str]) -> pd.DataFrame:
    """`read_roster`, cached a day per season. `names_key` (the data stamp)
    keys the cache; `_names` isn't hashed."""
    return read_roster(season, _names)


def _height(inches) -> str:
    if pd.isna(inches) or inches <= 0:
        return ""
    inches = int(round(inches))
    return f"{inches // 12}'{inches % 12}\""


def _hometown(city, state, country) -> str:
    parts = [str(city).strip() if isinstance(city, str) else "",
             (str(state).strip() if isinstance(state, str) and state else
              (str(country).strip() if isinstance(country, str) and country not in ("USA", "") else ""))]
    return ", ".join(p for p in parts if p)


def roster_table(roster: pd.DataFrame, team: str, group: str | None = None) -> pd.DataFrame:
    """One team's roster as `ROSTER_COLUMNS`, by position group then position
    then jersey. `group` (a `GROUP_ORDER` entry) keeps one group. Empty where
    the roster has no such team."""
    r = roster[roster["team"] == team] if not roster.empty else roster
    if r.empty:
        return pd.DataFrame(columns=ROSTER_COLUMNS)
    pos = r["position"].fillna("").astype(str)
    grp = pos.map(POSITION_GROUPS).fillna("Other")
    out = pd.DataFrame({
        "#": r["jersey"].map(lambda v: "" if pd.isna(v) else str(int(v))),
        "Player": (r["first_name"].fillna("") + " " + r["last_name"].fillna("")).str.strip(),
        "Pos": pos,
        "Yr": r["year"].map(CLASS_YEARS).fillna(""),
        "Ht": r["height"].map(_height),
        "Wt": r["weight"].map(lambda v: "" if pd.isna(v) or v <= 0 else str(int(v))),
        "Hometown": [_hometown(c, s, k) for c, s, k in
                     zip(r["home_city"], r["home_state"], r["home_country"])],
        "_group": grp,
        "_jersey": r["jersey"].fillna(999),
    })
    if group:
        out = out[out["_group"] == group]
    out["_order"] = out["_group"].map({g: i for i, g in enumerate(GROUP_ORDER)})
    out = out.sort_values(["_order", "Pos", "_jersey", "Player"], kind="stable")
    return out[ROSTER_COLUMNS].reset_index(drop=True)


def roster_groups(roster: pd.DataFrame, team: str) -> list[str]:
    """The `GROUP_ORDER` groups the team's roster has players in."""
    r = roster[roster["team"] == team] if not roster.empty else roster
    if r.empty:
        return []
    have = set(r["position"].fillna("").astype(str).map(POSITION_GROUPS).fillna("Other"))
    return [g for g in GROUP_ORDER if g in have]
