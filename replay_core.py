"""
League-neutral replay logic: the play timeline and cursor, and every stat the
app shows, built from play-by-play in nflfastR's column layout (`PBP_COLS`).

Both leagues' data is mapped to that layout (see leagues.py), so nothing here
knows which league it's looking at. No Streamlit: everything is a plain
function of DataFrames, importable from tools and tests.
"""

import re
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd

# ---------- Layout ----------
# Every column the app reads. The live feed is built to the same layout.
PBP_COLS = [
    "game_id", "season", "week", "game_date", "home_team", "away_team",
    "posteam", "defteam", "qtr", "time", "game_seconds_remaining",
    "play_id", "desc", "play_type", "down", "ydstogo", "yards_gained",
    "touchdown", "field_goal_result", "extra_point_result",
    "two_point_conv_result", "safety", "sp",
    "total_home_score", "total_away_score",
    "home_wp", "away_wp", "epa",
    "passer_player_name", "rusher_player_name", "receiver_player_name",
    "passing_yards", "rushing_yards", "receiving_yards", "yards_after_catch",
    "pass_touchdown", "rush_touchdown",
    "interception", "fumble_lost", "sack", "qb_hit",
    "complete_pass", "pass_attempt", "rush_attempt", "qb_kneel", "qb_spike",
    "first_down",
    "air_yards",
    "run_location", "run_gap", "pass_location", "pass_length",
    "third_down_converted", "third_down_failed",
    "fourth_down_converted", "fourth_down_failed",
    "goal_to_go", "yardline_100", "drive",
    "home_timeouts_remaining", "away_timeouts_remaining",
    "penalty", "penalty_team", "penalty_yards",
    # defensive player columns
    "solo_tackle_1_player_name", "solo_tackle_2_player_name",
    "assist_tackle_1_player_name", "assist_tackle_2_player_name",
    "assist_tackle_3_player_name", "assist_tackle_4_player_name",
    "sack_player_name", "half_sack_1_player_name", "half_sack_2_player_name",
    "qb_hit_1_player_name", "qb_hit_2_player_name",
    "tackle_for_loss_1_player_name", "tackle_for_loss_2_player_name",
    "interception_player_name",
    "pass_defense_1_player_name", "pass_defense_2_player_name",
    "forced_fumble_player_1_player_name", "forced_fumble_player_2_player_name",
]

GAME_SECONDS = 3600.0
# College overtime is untimed: play_timeline() spaces its plays this far apart.
UNTIMED_OT_PLAY_SECS = 30


# ---------- Play timeline and cursor ----------
def play_timeline(pbp_game: pd.DataFrame) -> pd.Series:
    """Elapsed game seconds at each play, indexed positionally like `pbp_game`.

    game_seconds_remaining counts DOWN from 3600 at kickoff to 0 at the end of
    regulation. Null-clock rows (timeouts, end-of-period admin plays) inherit
    the nearest real clock value so they aren't treated as kickoff-time.

    Overtime gets its own stretch after regulation: OT period n (qtr 4 + n)
    spans 3600 + 900·(n-1) to 3600 + 900·n, the same 15-minute slot the
    sidebar's quarter + clock inputs give it. In nflfastR an OT clock counts
    down inside its period (a 2024 OT game carries 600, 563, 523, ...), so
    taken as game seconds remaining every OT play would land at or before the
    end of Q4, and seeding the cursor at any OT clock would unlock the whole
    overtime. College OT is untimed: its plays have no clock, so each is
    placed UNTIMED_OT_PLAY_SECS after the one before it within its period.

    cummax keeps the series sorted (every lookup into it needs that) where a
    feed's clock steps backwards; it only ever moves a play later.
    """
    q = pbp_game["qtr"].ffill().bfill().fillna(1).to_numpy(float)
    gsr = pbp_game["game_seconds_remaining"]
    el = (3600 - gsr.ffill().bfill().fillna(3600)).to_numpy(float)
    ot = q >= 5
    if ot.any():
        o = pd.DataFrame({"q": q[ot], "c": gsr.to_numpy(float)[ot]})
        clock = o.groupby("q")["c"].transform(lambda s: s.ffill().bfill())
        nth = o.groupby("q").cumcount() + 1
        start = 3600 + (o["q"] - 5) * 900
        # Strictly after the period start, so the end of Q4 never unlocks
        # a playoff OT's opening kickoff (its clock reads a full 15:00).
        el[ot] = np.where(clock.notna(), np.maximum(start + 900 - clock, start + 1),
                          start + np.minimum(nth * UNTIMED_OT_PLAY_SECS, 899))
    return pd.Series(el).cummax()


def cursor_from_elapsed(timeline: pd.Series, elapsed_game_s: float) -> int:
    """Index of the last play at or before `elapsed_game_s`; -1 if none."""
    return int(np.searchsorted(timeline.values, float(elapsed_game_s), side="right")) - 1

def cursor_anchor(timeline: pd.Series, idx: int) -> tuple[float, int]:
    """A frame-independent handle on play `idx`: its elapsed game seconds and
    its rank among the plays sharing that second (1 = first)."""
    if idx < 0:
        return (-1.0, 0)
    e = float(timeline.iloc[idx])
    return (e, idx - int(np.searchsorted(timeline.values, e, side="left")) + 1)

def cursor_from_anchor(timeline: pd.Series, anchor: tuple[float, int]) -> int:
    """The play in another frame of the same game that `anchor` points to.

    Used when the frame under the cursor changes: the live feed gains plays,
    or the game switches from the live feed to official nflfastR pbp (whose
    indices differ — it has extra admin rows). Never lands past the anchor's
    second, and within that second never past the same rank, so a switch
    can't unlock anything."""
    e, rank = anchor
    if e < 0:
        return -1
    lo = int(np.searchsorted(timeline.values, e, side="left"))
    hi = int(np.searchsorted(timeline.values, e, side="right"))
    if hi > lo:
        return min(lo + rank - 1, hi - 1)
    return lo - 1


def period_label(q) -> str:
    """Q1-Q4, then OT, 2OT, 3OT, ..."""
    q = int(q)
    return f"Q{q}" if q <= 4 else ("OT" if q == 5 else f"{q - 4}OT")


# ---------- Stat builders ----------
def boxscore(revealed: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    """Quarter-by-quarter score from revealed plays."""
    if revealed.empty:
        return pd.DataFrame({"Team": [away, home], "Q1": [0, 0], "Q2": [0, 0],
                             "Q3": [0, 0], "Q4": [0, 0], "OT": [0, 0], "Total": [0, 0]})

    # Use the latest play in each quarter to get cumulative score, then diff
    out = {"Team": [away, home]}
    last_h, last_a = 0, 0
    for q in [1, 2, 3, 4, 5]:
        # Every overtime period goes in the one OT column.
        in_q = revealed[revealed["qtr"] >= 5] if q == 5 else revealed[revealed["qtr"] == q]
        if in_q.empty:
            h_pts, a_pts = 0, 0
        else:
            last = in_q.iloc[-1]
            cum_h = int(last["total_home_score"] or 0)
            cum_a = int(last["total_away_score"] or 0)
            h_pts = cum_h - last_h
            a_pts = cum_a - last_a
            last_h, last_a = cum_h, cum_a
        label = "OT" if q == 5 else f"Q{q}"
        out[label] = [a_pts, h_pts]
    out["Total"] = [last_a, last_h]
    return pd.DataFrame(out)

def _rate(num: float, den: float) -> float:
    return num / den if den > 0 else float("nan")

LOWER_IS_BETTER = {"Turnovers", "Penalties", "Penalty Yds"}

def _top_seconds(td: pd.DataFrame) -> float:
    """Time of possession in seconds, estimated from drive×quarter clock diffs."""
    top = 0.0
    # Drive numbers restart every game, so a season's plays group by game too.
    keys = ["game_id", "drive", "qtr"] if "game_id" in td.columns else ["drive", "qtr"]
    for _, grp in td.groupby(keys, dropna=True):
        gsr = grp["game_seconds_remaining"].dropna()
        if len(gsr) >= 2:
            top += max(0.0, float(gsr.max() - gsr.min()))
    return top

def fmt_top(v) -> str:
    if pd.isna(v):
        return "—"
    v = int(v)
    return f"{v // 60}:{v % 60:02d}"

def percentile_of(value: float, sorted_arr: np.ndarray) -> float:
    """0–100 percentile rank of value in a pre-sorted array."""
    if pd.isna(value) or len(sorted_arr) == 0:
        return float("nan")
    return float(np.searchsorted(sorted_arr, value, side="right") / len(sorted_arr) * 100)

TEAM_STAT_ROWS = [
    "Plays", "Pass Plays", "Rush Plays", "Rush+Comp",
    "Pass Yds", "Rush Yds", "Total Yds",
    "CMP%", "aDoT",
    "Pass EPA/play", "Rush EPA/play", "EPA/play",
    "Pass SR", "Rush SR",
    "1st Down %", "3rd Down %", "RZ TD%",
    "Turnovers", "Penalties", "Penalty Yds", "TOP",
]


def _team_col(plays: pd.DataFrame, team: str, side: str = "posteam") -> list:
    """One team's Team stats (`TEAM_STAT_ROWS` order) over `plays`: its
    offense with side="posteam", or what its defense allowed with
    side="defteam". Penalties are always the team's own."""
    td = plays[plays[side] == team]

    pass_mask = td["pass_attempt"].fillna(0) == 1
    rush_mask = td["rush_attempt"].fillna(0) == 1
    scrimmage_mask = pass_mask | rush_mask

    pass_plays = int(pass_mask.sum())
    rush_plays = int(rush_mask.sum())
    total_plays = int(scrimmage_mask.sum())

    pass_yds = int(td["passing_yards"].fillna(0).sum())
    rush_yds = int(td["rushing_yards"].fillna(0).sum())

    cmp = td.loc[pass_mask, "complete_pass"].fillna(0).sum()
    adot = td.loc[pass_mask, "air_yards"].mean()  # nan if no pass plays

    pass_epa = td.loc[pass_mask, "epa"].fillna(0).sum()
    rush_epa = td.loc[rush_mask, "epa"].fillna(0).sum()
    total_epa = td.loc[scrimmage_mask, "epa"].fillna(0).sum()

    pass_sr = (td.loc[pass_mask, "epa"].fillna(0) > 0).sum()
    rush_sr = (td.loc[rush_mask, "epa"].fillna(0) > 0).sum()

    first_downs = td.loc[scrimmage_mask, "first_down"].fillna(0).sum()

    third_conv = td["third_down_converted"].fillna(0).sum()
    third_fail = td["third_down_failed"].fillna(0).sum()

    rz_mask = scrimmage_mask & (td["yardline_100"].fillna(100) <= 20)
    rz_plays = int(rz_mask.sum())
    rz_td = td.loc[rz_mask, "touchdown"].fillna(0).sum()

    tos = int(td["interception"].fillna(0).sum() + td["fumble_lost"].fillna(0).sum())
    comp_rush = int(cmp) + rush_plays
    top = _top_seconds(td)

    # Penalties are charged against the team penalized (penalty_team),
    # which may differ from possession — so this is computed on the
    # full play set, not the posteam-filtered `td`.
    pen_mask = plays["penalty_team"] == team
    penalties = int(pen_mask.sum())
    penalty_yds = int(plays.loc[pen_mask, "penalty_yards"].fillna(0).sum())

    return [
        total_plays,
        pass_plays,
        rush_plays,
        comp_rush,
        pass_yds,
        rush_yds,
        pass_yds + rush_yds,
        _rate(cmp, pass_plays),
        adot if not pd.isna(adot) else float("nan"),
        _rate(pass_epa, pass_plays),
        _rate(rush_epa, rush_plays),
        _rate(total_epa, total_plays),
        _rate(pass_sr, pass_plays),
        _rate(rush_sr, rush_plays),
        _rate(first_downs, total_plays),
        _rate(third_conv, third_conv + third_fail),
        _rate(rz_td, rz_plays),
        tos,
        penalties,
        penalty_yds,
        top,
    ]


def team_stats(revealed: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    """Advanced offensive team stats, returned as a transposed comparison table.

    Rows are stat names; columns are [away, home] team abbreviations.
    """
    stats = {team: _team_col(revealed, team) for team in [away, home]}
    return pd.DataFrame(stats, index=pd.Index(TEAM_STAT_ROWS, name="Stat"))

def stat_percentiles(df: pd.DataFrame, baselines: dict,
                     lower_is_better: set = LOWER_IS_BETTER) -> pd.DataFrame:
    """Percentile of each Team stats cell vs the baseline seasons, 100 = best
    (flipped for the stats in `lower_is_better`). NaN without a baseline."""
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for row in df.index:
        arr = baselines.get(row, np.array([]))
        for team in df.columns:
            pct = percentile_of(float(df.at[row, team]), arr)
            out.at[row, team] = 100.0 - pct if row in lower_is_better and pd.notna(pct) else pct
    return out


# ---------- Percentile baselines ----------
# Every column the baselines read, so the college season is mapped only once.
BASELINE_COLS = [
    "game_id", "posteam",
    "pass_attempt", "rush_attempt", "qb_kneel", "qb_spike", "epa",
    "passing_yards", "rushing_yards",
    "complete_pass", "air_yards",
    "interception", "fumble_lost", "first_down",
    "third_down_converted", "third_down_failed",
    "yardline_100", "touchdown",
    "drive", "qtr", "game_seconds_remaining",
    "penalty", "penalty_team", "penalty_yards",
    "down", "ydstogo",
    "desc", "play_type", "yards_gained", "sack", "qb_hit",
    "passer_player_name", "rusher_player_name", "receiver_player_name",
    "run_location", "run_gap", "pass_location", "pass_length",
]


def stat_distributions(raw_all: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per-stat distributions over every team-game in `raw_all` (plays with
    `BASELINE_COLS`), sorted ascending: what Team stats percentiles compare
    against."""
    raw = raw_all[raw_all["posteam"].notna()]

    def _game_stats(td):
        pass_mask = (td["pass_attempt"].fillna(0) == 1) & (td["qb_spike"].fillna(0) == 0)
        rush_mask = (td["rush_attempt"].fillna(0) == 1) & (td["qb_kneel"].fillna(0) == 0)
        sc_mask   = pass_mask | rush_mask
        pp = int(pass_mask.sum())
        rp = int(rush_mask.sum())
        tp = int(sc_mask.sum())
        pass_yds = int(td["passing_yards"].fillna(0).sum())
        rush_yds = int(td["rushing_yards"].fillna(0).sum())
        cmp      = td.loc[pass_mask, "complete_pass"].fillna(0).sum()
        adot     = td.loc[pass_mask, "air_yards"].mean()
        pass_epa = td.loc[pass_mask, "epa"].fillna(0).sum()
        rush_epa = td.loc[rush_mask, "epa"].fillna(0).sum()
        tot_epa  = td.loc[sc_mask,   "epa"].fillna(0).sum()
        pass_sr  = (td.loc[pass_mask, "epa"].fillna(0) > 0).sum()
        rush_sr  = (td.loc[rush_mask, "epa"].fillna(0) > 0).sum()
        fd       = td.loc[sc_mask, "first_down"].fillna(0).sum()
        tc       = td["third_down_converted"].fillna(0).sum()
        tf       = td["third_down_failed"].fillna(0).sum()
        rz_mask  = sc_mask & (td["yardline_100"].fillna(100) <= 20)
        rz_plays = int(rz_mask.sum())
        rz_td    = int(td.loc[rz_mask, "touchdown"].fillna(0).sum())
        tos      = int(td["interception"].fillna(0).sum() + td["fumble_lost"].fillna(0).sum())
        comp_rush = int(cmp) + rp
        top       = _top_seconds(td)
        return pd.Series({
            "Plays":        tp,
            "Pass Plays":   pp,
            "Rush Plays":   rp,
            "Rush+Comp":    comp_rush,
            "Pass Yds":     pass_yds,
            "Rush Yds":     rush_yds,
            "Total Yds":    pass_yds + rush_yds,
            "CMP%":         _rate(cmp, pp),
            "aDoT":         adot,
            "Pass EPA/play":_rate(pass_epa, pp),
            "Rush EPA/play":_rate(rush_epa, rp),
            "EPA/play":     _rate(tot_epa,  tp),
            "Pass SR":      _rate(pass_sr,  pp),
            "Rush SR":      _rate(rush_sr,  rp),
            "1st Down %":   _rate(fd, tp),
            "3rd Down %":   _rate(tc, tc + tf),
            "RZ TD%":       _rate(rz_td, rz_plays),
            "Turnovers":    tos,
            "TOP":          top,
        })

    try:
        per_game = raw.groupby(["game_id", "posteam"]).apply(
            _game_stats, include_groups=False
        )
    except TypeError:
        per_game = raw.groupby(["game_id", "posteam"]).apply(_game_stats)

    # Penalties are charged against penalty_team, which can differ from
    # posteam (e.g. a defensive penalty), so they're aggregated from the
    # full unfiltered play set and joined in by (game_id, penalty_team).
    pen = raw_all[
        (raw_all["penalty"].fillna(0) == 1) & raw_all["penalty_team"].notna()
    ]
    pen_stats = pen.groupby(["game_id", "penalty_team"]).agg(
        **{
            "Penalties": ("penalty", "size"),
            "Penalty Yds": ("penalty_yards", lambda s: s.fillna(0).sum()),
        }
    )
    pen_stats.index = pen_stats.index.set_names(["game_id", "posteam"])
    per_game = per_game.join(pen_stats, how="left")
    per_game[["Penalties", "Penalty Yds"]] = per_game[["Penalties", "Penalty Yds"]].fillna(0)

    return {
        stat: np.sort(per_game[stat].dropna().values)
        for stat in per_game.columns
    }


# ---------- Line play ----------
LINE_ROWS = ["Dropbacks", "Sack %", "QB Hit %", "Designed Runs", "Stuff %",
             "Explosive Run %", "Snaps", "OL Penalty %"]
LINE_COUNT_ROWS = {"Dropbacks", "Designed Runs", "Snaps"}
LINE_LOWER_IS_BETTER = {"Sack %", "QB Hit %", "Stuff %", "OL Penalty %"}
STUFF_MAX_YDS = 0       # a designed run for no gain or a loss
EXPLOSIVE_RUN_YDS = 10  # as explosive_plays()
# The penalties that are on the offensive line, as both leagues' text names
# them: "Offensive Holding" / college "Holding" (the flag must be the
# offense's, so defensive holding never counts), "False Start", and
# "Ineligible Downfield Pass" / college "Ineligible Downfield on Pass".
OL_PENALTY = r"(?i)(?<!defensive )holding|false start|ineligible downfield|illegal man downfield"


def _designed_runs(td: pd.DataFrame) -> pd.Series:
    """Runs on a down, leaving out kneels and scrambles."""
    scramble = td["desc"].fillna("").str.contains("scrambles", case=False, regex=False)
    return (td["down"].notna() & (td["rush_attempt"].fillna(0) == 1)
            & (td["qb_kneel"].fillna(0) == 0) & ~scramble)


def _line_stats(td: pd.DataFrame) -> dict[str, float]:
    """Line play numbers for offensive plays `td` (one offense's, or what one
    defense faced). Downs only, so two-point tries don't count. Dropbacks are pass attempts
    plus sacks (nflfastR's pass_attempt); QB hits include sacks; designed
    runs leave out kneels and scrambles; snaps are pass, run and no-play
    rows, and the penalty rate counts the offense's own holding, false start
    and ineligible-downfield flags on them (`OL_PENALTY`)."""
    on_down = td["down"].notna()
    drop = on_down & (td["pass_attempt"].fillna(0) == 1) & (td["qb_spike"].fillna(0) == 0)
    sacks = drop & (td["sack"].fillna(0) == 1)
    hits = drop & ((td["qb_hit"].fillna(0) == 1) | sacks)
    desc = td["desc"].fillna("")
    run = _designed_runs(td)
    yds = td["yards_gained"]
    snaps = on_down & td["play_type"].isin(["pass", "run", "no_play"])
    flags = (snaps & (td["penalty"].fillna(0) == 1) & (td["penalty_team"] == td["posteam"])
             & desc.str.contains(OL_PENALTY))
    n_drop, n_run, n_snaps = int(drop.sum()), int(run.sum()), int(snaps.sum())
    return {
        "Dropbacks":       n_drop,
        "Sack %":          _rate(sacks.sum(), n_drop),
        "QB Hit %":        _rate(hits.sum(), n_drop),
        "Designed Runs":   n_run,
        "Stuff %":         _rate((run & (yds <= STUFF_MAX_YDS)).sum(), n_run),
        "Explosive Run %": _rate((run & (yds >= EXPLOSIVE_RUN_YDS)).sum(), n_run),
        "Snaps":           n_snaps,
        "OL Penalty %":    _rate(flags.sum(), n_snaps),
    }


def line_play_stats(revealed: pd.DataFrame, home: str, away: str,
                    drop: frozenset = frozenset()) -> pd.DataFrame:
    """Each offense's line play (`_line_stats`) as a transposed table: rows
    are `LINE_ROWS` (less any in `drop`), columns [away, home]. The other
    team's column is the same numbers from its defensive front's side."""
    df = pd.DataFrame(
        {team: _line_stats(revealed[revealed["posteam"] == team]) for team in [away, home]},
        index=pd.Index(LINE_ROWS, name="Stat"),
    )
    return df.drop(index=[r for r in df.index if r in drop])


def line_play_distributions(raw: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per-stat line play distributions over every team-game in `raw`."""
    raw = raw[raw["posteam"].notna()]
    per_game = pd.DataFrame(
        [_line_stats(grp) for _, grp in raw.groupby(["game_id", "posteam"])]
    )
    if per_game.empty:
        return {}
    return {stat: np.sort(per_game[stat].dropna().values) for stat in per_game.columns}


# ---------- Tendencies ----------
# Where each offense runs and throws, from its own side (its left is the
# quarterback's left), as the play text says: NFL runs name a gap ('left
# tackle', 'up the middle'), passes a depth and side ('pass short right').
# College stat-crew text names a side for runs but no gap. Pass depth is
# measured in air yards (`PASS_BANDS`) where the data has them; live NFL games
# don't, so they fall back on the text's short/deep (`PASS_DEPTHS`).
RUN_GAPS = {("left", "end"): "LE", ("left", "tackle"): "LT", ("left", "guard"): "LG",
            ("middle", None): "Mid",
            ("right", "guard"): "RG", ("right", "tackle"): "RT", ("right", "end"): "RE"}
RUN_SIDES = {"left": "Left", "middle": "Middle", "right": "Right"}
PASS_DEPTHS = {"deep": "Deep", "short": "Short"}
# Air-yard bands, deepest first: (label, lowest, highest + 1).
PASS_BANDS = [("Deep 20+", 20, np.inf), ("Inter 10–19", 10, 20), ("Short 0–9", 0, 10),
              ("Behind LOS", -np.inf, 0)]
PASS_SIDES = RUN_SIDES
TENDENCY_DOWNS = {"All downs": (1, 4), "Early downs": (1, 2), "Late downs": (3, 4)}


def _tendency(plays: pd.DataFrame, labels: pd.Series, order: list) -> pd.DataFrame:
    """Plays, Share, Yds/Play, SR% and EPA/play per label, rows in `order`
    (a label with no plays has Plays 0 and the rest NaN)."""
    plays = plays.assign(_label=labels, _epa=plays["epa"].fillna(0)).dropna(subset=["_label"])
    g = plays.groupby("_label")
    n = g.size().reindex(order, fill_value=0)
    out = pd.DataFrame({
        "Plays": n,
        "Share": n / n.sum() if n.sum() else np.nan,
        "Yds/Play": g["yards_gained"].mean(),
        "SR%": g["_epa"].apply(lambda e: (e > 0).mean()),
        "EPA/play": g["_epa"].mean(),
    }, index=pd.Index(order))
    out["Plays"] = out["Plays"].astype(int)
    return out


def run_tendencies(pbp: pd.DataFrame, team: str | None = None, gaps: bool = True,
                   downs: tuple[int, int] = (1, 4)) -> pd.DataFrame:
    """`team`'s designed runs (every team's if None) on `downs`, by gap
    (`RUN_GAPS`) or, without `gaps`, by side (`RUN_SIDES`). Runs whose text
    names no direction are left out."""
    td = pbp if team is None else pbp[pbp["posteam"] == team]
    td = td[_designed_runs(td) & td["down"].between(*downs)]
    loc = td["run_location"]
    if gaps:
        gap = td["run_gap"].where(loc != "middle")
        labels = pd.Series([RUN_GAPS.get((l, None if pd.isna(g) else g)) for l, g in zip(loc, gap)],
                           index=td.index, dtype=object)
        return _tendency(td, labels, list(RUN_GAPS.values()))
    return _tendency(td, loc.map(RUN_SIDES), list(RUN_SIDES.values()))


def has_air_yards(pbp: pd.DataFrame) -> bool:
    """Whether `pbp`'s passes carry air yards: False only once passes have
    been thrown and none has them (a live NFL game)."""
    thrown = pbp[(pbp["pass_attempt"].fillna(0) == 1) & (pbp["sack"].fillna(0) == 0)
                 & pbp["pass_location"].notna()]
    return thrown.empty or bool(thrown["air_yards"].notna().any())


def pass_tendencies(pbp: pd.DataFrame, team: str | None = None,
                    downs: tuple[int, int] = (1, 4), air: bool = True) -> pd.DataFrame:
    """`team`'s passes thrown (no sacks or spikes; every team's if None) on
    `downs`, by depth × side: rows ("Deep 20+", "Left") … ("Behind LOS",
    "Right") by air yards (`PASS_BANDS`), or without `air` ("Deep", "Left") …
    ("Short", "Right") by the text's depth, with Cmp% and aDOT added. Passes
    whose text names no direction (college interceptions, for one) are left
    out, and by air yards so are those without any."""
    td = pbp if team is None else pbp[pbp["posteam"] == team]
    td = td[td["down"].between(*downs) & (td["pass_attempt"].fillna(0) == 1)
            & (td["sack"].fillna(0) == 0) & (td["qb_spike"].fillna(0) == 0)]
    side = td["pass_location"].map(PASS_SIDES)
    if air:
        depths = [b[0] for b in PASS_BANDS]
        depth = pd.Series(np.nan, index=td.index, dtype=object)
        for label, lo, hi in PASS_BANDS:
            depth = depth.mask((td["air_yards"] >= lo) & (td["air_yards"] < hi), label)
    else:
        depths = list(PASS_DEPTHS.values())
        depth = td["pass_length"].map(PASS_DEPTHS)
    labels = depth + " " + side
    order = [(d, s) for d in depths for s in PASS_SIDES.values()]
    out = _tendency(td, labels, [f"{d} {s}" for d, s in order])
    g = td.assign(_label=labels).dropna(subset=["_label"]).groupby("_label")
    out["Cmp%"] = g["complete_pass"].mean()
    out["aDOT"] = g["air_yards"].mean()
    out.index = pd.MultiIndex.from_tuples(order, names=["Depth", "Side"])
    return out


def tendency_baselines(raw: pd.DataFrame, gaps: bool) -> dict[str, dict[str, pd.DataFrame]]:
    """League-wide tendencies over every play in `raw`, per `TENDENCY_DOWNS`
    option: what each team's split is compared against. Each holds "runs",
    and passes both by air yards ("air") and by the text's depth ("text")."""
    return {k: {"runs": run_tendencies(raw, gaps=gaps, downs=d),
                "air": pass_tendencies(raw, downs=d),
                "text": pass_tendencies(raw, downs=d, air=False)}
            for k, d in TENDENCY_DOWNS.items()}


SR_SITUATIONS = [
    ("Early · Short",  (1, 2), (1,  3)),
    ("Late · Short",   (3, 4), (1,  3)),
    ("Early · Medium", (1, 2), (4,  6)),
    ("Late · Medium",  (3, 4), (4,  6)),
    ("Early · Long",   (1, 2), (7, 99)),
    ("Late · Long",    (3, 4), (7, 99)),
]
# Subtotal rows: every distance, early downs vs late downs.
SR_SUBTOTALS = [
    ("Early · Total",  (1, 2), (0, 99)),
    ("Late · Total",   (3, 4), (0, 99)),
]
SR_METRICS = ["Pass SR%", "Pass EPA/play", "Rush SR%", "Rush EPA/play"]


def _situation_row(plays: pd.DataFrame) -> dict[str, float]:
    """`SR_METRICS` for one team's plays in one situation: success rate (EPA
    > 0) and EPA/play on passes without spikes and runs without kneels."""
    pass_plays = plays[(plays["pass_attempt"].fillna(0) == 1) & (plays["qb_spike"].fillna(0) == 0)]
    rush_plays = plays[(plays["rush_attempt"].fillna(0) == 1) & (plays["qb_kneel"].fillna(0) == 0)]

    def _sr(p) -> float:
        return float("nan") if p.empty else (p["epa"].fillna(0) > 0).mean() * 100

    def _epa_per_play(p) -> float:
        return float("nan") if p.empty else p["epa"].fillna(0).mean()

    return {
        "Pass SR%": _sr(pass_plays),
        "Pass EPA/play": _epa_per_play(pass_plays),
        "Rush SR%": _sr(rush_plays),
        "Rush EPA/play": _epa_per_play(rush_plays),
        "_pass": len(pass_plays),
        "_rush": len(rush_plays),
    }


def _situation_mask(plays: pd.DataFrame, downs, d_min, d_max) -> pd.Series:
    return plays["down"].isin(downs) & plays["ydstogo"].fillna(0).between(d_min, d_max)


def situational_success_rate(revealed: pd.DataFrame, home: str, away: str,
                             subtotals: bool = False) -> pd.DataFrame:
    """Success rate and EPA/play split by down group × distance × play type.
    With `subtotals`, the early-down and late-down totals (`SR_SUBTOTALS`)
    follow the situation rows."""
    return situational_table(revealed, [away, home], subtotals)


def situational_table(plays: pd.DataFrame, teams: list, subtotals: bool = False,
                      side: str = "posteam"):
    """`situational_success_rate` for any columns: `teams` are values of
    `side` (posteam, or defteam for what a defense allowed), and a team of
    None takes every play: the league over a season, labelled "League"."""
    situations = SR_SITUATIONS + (SR_SUBTOTALS if subtotals else [])
    rows = []
    counts: dict[tuple, dict[str, int]] = {}
    for label, downs, (d_min, d_max) in situations:
        sit_mask = _situation_mask(plays, downs, d_min, d_max)
        for team in teams:
            tm = plays[sit_mask & ((plays[side] == team) if team is not None
                                   else plays["posteam"].notna())]
            r = _situation_row(tm)
            name = "League" if team is None else team
            counts[(label, name)] = {"Pass": r.pop("_pass"), "Rush": r.pop("_rush")}
            rows.append({"Situation": label, "Team": name, **r})

    df = pd.DataFrame(rows)
    metrics = SR_METRICS
    pivot = df.pivot(index="Situation", columns="Team", values=metrics)
    pivot.columns = [f"{team} {stat}" for stat, team in pivot.columns]
    col_order = []
    for team in teams:
        for m in metrics:
            col_order.append(f"{'League' if team is None else team} {m}")
    pivot = pivot[[c for c in col_order if c in pivot.columns]]
    # pivot sorts the rows; keep the situations alphabetical, subtotals last.
    subtotal_labels = [lbl for lbl, _, _ in SR_SUBTOTALS]
    pivot = pivot.reindex(
        sorted(i for i in pivot.index if i not in subtotal_labels)
        + [lbl for lbl in subtotal_labels if lbl in pivot.index]
    )
    pivot.index.name = "Situation"
    return pivot, counts


def situational_distributions(raw: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """Per-situation metric distributions over every team-game in `raw`."""
    raw = raw[raw["posteam"].notna()]
    result: dict[str, dict[str, np.ndarray]] = {}
    for label, downs, (d_min, d_max) in SR_SITUATIONS + SR_SUBTOTALS:
        sit = raw[_situation_mask(raw, downs, d_min, d_max)]
        rows = [_situation_row(grp) for _, grp in sit.groupby(["game_id", "posteam"])]
        if not rows:
            result[label] = {}
            continue
        per_game = pd.DataFrame(rows)
        result[label] = {
            m: np.sort(per_game[m].dropna().values) for m in SR_METRICS
        }
    return result


# ---------- Player leaders ----------
PLAYER_NAME_COL = {"passing": "passer_player_name", "rushing": "rusher_player_name",
                   "receiving": "receiver_player_name"}
# The per-play averages and ratios in each leaders table, colored by
# percentile; the rest are counts.
PLAYER_RATE_COLS = {"passing": ["SR%", "aDOT", "EPA/play"],
                    "rushing": ["SR%", "EPA/play"],
                    "receiving": ["aDOT", "EPA/play"]}
# The volume column (one per play) and the minimum a player-game needs to go
# in the baseline and to be colored: a rate over a couple of plays says little.
PLAYER_VOLUME_COL = {"passing": "Att", "rushing": "Att", "receiving": "Tgt"}
PLAYER_MIN_PLAYS = {"passing": 10, "rushing": 5, "receiving": 3}


def _player_plays(td: pd.DataFrame, kind: str) -> pd.DataFrame:
    """The plays a `kind` leaders table credits to a player: dropbacks (sacks
    included) without spikes, carries without kneels, or targets."""
    no_spike = td["qb_spike"].fillna(0) == 0
    if kind == "passing":
        return td[(td["pass_attempt"] == 1) & no_spike]
    if kind == "rushing":
        return td[(td["rush_attempt"] == 1) & (td["qb_kneel"].fillna(0) == 0)]
    return td[td["receiver_player_name"].notna() & (td["pass_attempt"].fillna(0) == 1) & no_spike]


def top_players(revealed: pd.DataFrame, team: str, kind: str, n: int = 3,
                drop: frozenset = frozenset()) -> pd.DataFrame:
    """Leaders for a team so far. `drop` names columns the data source doesn't
    have (they'd only ever read 0)."""
    td = revealed[revealed["posteam"] == team]
    if kind == "passing":
        pass_td = _player_plays(td, kind).copy()
        if pass_td.empty:
            return pd.DataFrame()
        pass_td["_success"] = (pass_td["epa"].fillna(0) > 0).astype(int)
        sr = pass_td.groupby("passer_player_name")["_success"].mean().rename("SR%")
        g = pass_td.groupby("passer_player_name", as_index=False).agg(
            Att=("pass_attempt", "sum"), Yds=("passing_yards", "sum"),
            TD=("pass_touchdown", "sum"), INT=("interception", "sum"),
            aDOT=("air_yards", "mean"), Sacks=("sack", "sum"), Hits=("qb_hit", "sum"),
            _epa=("epa", "sum"), _plays=("epa", "count"))
        g = g.rename(columns={"passer_player_name": "Player"})
        g = g.join(sr, on="Player")
        g["aDOT"] = g["aDOT"].round(1)
        g["SR%"] = (g["SR%"] * 100).round(1)
    elif kind == "rushing":
        rush_td = _player_plays(td, kind).copy()
        if rush_td.empty:
            return pd.DataFrame()
        rush_td["_success"] = (rush_td["epa"].fillna(0) > 0).astype(int)
        rush_td["_stuffed"] = (rush_td["yards_gained"].fillna(0) <= 0).astype(int)
        sr = rush_td.groupby("rusher_player_name")["_success"].mean().rename("SR%")
        g = rush_td.groupby("rusher_player_name", as_index=False).agg(
            Att=("rush_attempt", "sum"), Yds=("rushing_yards", "sum"),
            TD=("rush_touchdown", "sum"), Stuffed=("_stuffed", "sum"),
            _epa=("epa", "sum"), _plays=("epa", "count"))
        g = g.rename(columns={"rusher_player_name": "Player"})
        g = g.join(sr, on="Player")
        g["SR%"] = (g["SR%"] * 100).round(1)
    else:  # receiving
        recv_td = _player_plays(td, kind).copy()
        if recv_td.empty:
            return pd.DataFrame()
        recv_td["_exp"] = (
            (recv_td["complete_pass"].fillna(0) == 1)
            & (recv_td["receiving_yards"].fillna(0) >= 20)
        ).astype(int)
        g = recv_td.groupby("receiver_player_name", as_index=False).agg(
            Tgt=("pass_attempt", "sum"),
            Rec=("complete_pass", "sum"),
            Yds=("receiving_yards", "sum"),
            YAC=("yards_after_catch", "sum"),
            aDOT=("air_yards", "mean"),
            TD=("pass_touchdown", "sum"),
            Exp=("_exp", "sum"),
            _epa=("epa", "sum"), _plays=("epa", "count"))
        g["Yds"] = g["Yds"].fillna(0)
        g["YAC"] = g["YAC"].fillna(0)
        g["aDOT"] = g["aDOT"].round(1)
        g = g.rename(columns={"receiver_player_name": "Player"})
    g = g.dropna(subset=["Player"])
    int_cols = [c for c in g.select_dtypes("number").columns if c not in ("_epa", "_plays", "aDOT", "EPA/play", "SR%")]
    g[int_cols] = g[int_cols].astype(int)
    g["EPA/play"] = (g["_epa"] / g["_plays"]).round(2)
    g = g.drop(columns=["_epa", "_plays", *[c for c in drop if c in g.columns]])
    return g.sort_values("Yds", ascending=False).head(n)


def player_distributions(raw: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """Per-kind distributions of the leaders tables' rates (`PLAYER_RATE_COLS`)
    over every player-game in `raw` with at least `PLAYER_MIN_PLAYS` plays,
    computed as `top_players` does, sorted ascending."""
    raw = raw[raw["posteam"].notna()]
    out = {}
    for kind, name in PLAYER_NAME_COL.items():
        plays = _player_plays(raw, kind)
        plays = plays.assign(_success=(plays["epa"].fillna(0) > 0) * 100.0)
        g = plays.groupby(["game_id", name]).agg(
            _n=("epa", "size"), **{"SR%": ("_success", "mean"),
                                   "aDOT": ("air_yards", "mean"),
                                   "EPA/play": ("epa", "mean")})
        g = g[g["_n"] >= PLAYER_MIN_PLAYS[kind]]
        out[kind] = {stat: np.sort(g[stat].dropna().values) for stat in PLAYER_RATE_COLS[kind]}
    return out

def top_defenders(revealed: pd.DataFrame, team: str, n: int = 5,
                  drop: frozenset = frozenset()) -> pd.DataFrame:
    """Defensive leaders for a team: tackles, sacks, QB hits, hurries, TFLs,
    INTs, PDs, FFs. `drop` names columns the data source doesn't have (nflfastR
    has no hurries; the college feed has no QB hits)."""
    _ST_TYPES = {"kickoff", "punt", "field_goal", "extra_point", "no_play"}
    td = revealed[
        (revealed["defteam"] == team)
        & (~revealed["play_type"].isin(_ST_TYPES))
    ]
    if td.empty:
        return pd.DataFrame()

    def _count(cols: list[str], weight: float = 1.0) -> pd.Series:
        frames = []
        for col in cols:
            if col in td.columns:
                s = td[col].dropna()
                frames.append(s)
        if not frames:
            return pd.Series(dtype=float)
        stacked = pd.concat(frames)
        return (stacked.value_counts() * weight).rename("v")

    # Total tackles as the NFL and NCAA count them: solo + assists, each
    # assist a full tackle for every player in on it.
    solo   = _count(["solo_tackle_1_player_name", "solo_tackle_2_player_name"], 1.0)
    assist = _count([
        "assist_tackle_1_player_name", "assist_tackle_2_player_name",
        "assist_tackle_3_player_name", "assist_tackle_4_player_name",
    ], 1.0)
    tackles = solo.add(assist, fill_value=0).rename("Tackles")

    sacks = _count(["sack_player_name"], 1.0).add(
        _count(["half_sack_1_player_name", "half_sack_2_player_name"], 0.5), fill_value=0
    ).rename("Sacks")

    qb_hits = _count(["qb_hit_1_player_name", "qb_hit_2_player_name"]).rename("QB Hits")
    hurries = _count(["qb_hurry_1_player_name", "qb_hurry_2_player_name",
                      "qb_hurry_3_player_name"]).rename("Hurries")
    tfls    = _count(["tackle_for_loss_1_player_name", "tackle_for_loss_2_player_name"]).add(
        _count(["half_tfl_1_player_name", "half_tfl_2_player_name"], 0.5), fill_value=0
    ).rename("TFL")
    ints    = _count(["interception_player_name"]).rename("INT")
    pds     = _count(["pass_defense_1_player_name", "pass_defense_2_player_name"]).rename("PD")
    ffs     = _count(["forced_fumble_player_1_player_name", "forced_fumble_player_2_player_name"]).rename("FF")

    g = pd.concat([tackles, sacks, qb_hits, hurries, tfls, ints, pds, ffs], axis=1).fillna(0)
    g.index.name = "Player"
    g = g.reset_index()
    g = g[g["Player"].notna()]

    for col in ["Tackles", "QB Hits", "Hurries", "TFL", "INT", "PD", "FF"]:
        if col in g.columns:
            g[col] = g[col].round(1)
    if "Sacks" in g.columns:
        g["Sacks"] = g["Sacks"].round(1)

    # weight sacks/INTs heavily so impact players surface even with low tackle counts
    g["_sort"] = (
        g.get("Tackles", 0) + g.get("Sacks", 0) * 3
        + g.get("INT", 0) * 2 + g.get("TFL", 0) + g.get("QB Hits", 0) * 0.5
        + g.get("Hurries", 0) * 0.5
        + g.get("PD", 0) * 0.5 + g.get("FF", 0) * 1.5
    )
    g = g.drop(columns=[c for c in drop if c in g.columns])
    return g.sort_values("_sort", ascending=False).drop(columns=["_sort"]).head(n)


# ---------- Season filters ----------
# Situations the Season stats page can narrow its plays to. Garbage time is
# judged by the offense's win probability before the snap; clutch time is the
# 4th quarter's last minutes or overtime with the score close before the snap.
class PlayFilter(NamedTuple):
    quarters: tuple[int, ...] = ()       # 1–4, 5 = every overtime period; () = all
    garbage_wp: float | None = None      # drop plays with offense WP < x or > 1 − x
    clutch: tuple[int, int] | None = None  # (minutes left in Q4, within points)


NO_FILTER = PlayFilter()
OT_QTR = 5
# Each team's own home or away games.
VENUES = {"All": None, "Home": "home_team", "Away": "away_team"}


def pre_play_margin(pbp: pd.DataFrame) -> pd.Series:
    """Home score minus away score before each play: the previous row's
    totals within its game (the totals are after the play). `pbp` must hold
    whole games, in order."""
    margin = (pbp["total_home_score"] - pbp["total_away_score"]).astype(float)
    margin = margin.groupby(pbp["game_id"]).ffill()
    return margin.groupby(pbp["game_id"]).shift().fillna(0.0)


def offense_wp(pbp: pd.DataFrame) -> pd.Series:
    """The offense's win probability before the snap."""
    return pbp["home_wp"].where(pbp["posteam"] == pbp["home_team"], 1 - pbp["home_wp"])


def filter_plays_by(pbp: pd.DataFrame, f: PlayFilter) -> pd.DataFrame:
    """`pbp`'s plays in the situations `f` names. Clutch needs the
    `_margin` column (`pre_play_margin`, taken before filtering)."""
    keep = pd.Series(True, index=pbp.index)
    if f.quarters:
        q = pbp["qtr"].clip(upper=OT_QTR)
        keep &= q.isin(f.quarters)
    if f.garbage_wp is not None:
        wp = offense_wp(pbp)
        keep &= wp.isna() | wp.between(f.garbage_wp, 1 - f.garbage_wp)
    if f.clutch is not None:
        minutes, points = f.clutch
        late = ((pbp["qtr"] == 4) & (pbp["game_seconds_remaining"] <= minutes * 60)) | (pbp["qtr"] >= OT_QTR)
        keep &= late & (pbp["_margin"].abs() <= points)
    return pbp[keep]


def venue_plays(pbp: pd.DataFrame, venue: str | None, side: str = "posteam") -> pd.DataFrame:
    """Plays where the team on `side` is at home (venue "home_team") or away
    ("away_team"); all of them for None."""
    return pbp if venue is None else pbp[pbp[side] == pbp[venue]]


# ---------- Season ----------
# The Season stats page: the same stats over many games. Counts are per game
# (teams play different numbers of games), and every cell is ranked against
# the other teams (or qualified players) in the same selection.
SEASON_PER_GAME = {"Plays", "Pass Plays", "Rush Plays", "Rush+Comp", "Pass Yds", "Rush Yds",
                   "Total Yds", "Turnovers", "Penalties", "Penalty Yds", "TOP", *LINE_COUNT_ROWS}
# Volumes: neither good nor bad, so never colored.
SEASON_VOLUME = {"Games", "Plays", "Pass Plays", "Rush Plays", "Dropbacks", "Designed Runs", "Snaps"}
SEASON_COLS = ["Games", *TEAM_STAT_ROWS, *LINE_ROWS]
# A defense's own penalties aren't "allowed"; they're in the offense table.
SEASON_DEFENSE_SKIP = {"Penalties", "Penalty Yds"}


def season_lower_is_better(side: str = "posteam") -> set:
    """The season table's columns where lower is better: the Team stats and
    Line play ones for an offense, every other one for a defense (what it
    allowed, where forcing turnovers and sacks is good)."""
    offense = LOWER_IS_BETTER | LINE_LOWER_IS_BETTER
    if side == "posteam":
        return offense
    return set(SEASON_COLS) - offense - SEASON_VOLUME


def season_team_table(pbp: pd.DataFrame, side: str = "posteam",
                      venue: str | None = None) -> pd.DataFrame:
    """One row per team in `pbp` (a season's plays, or part of one): Games,
    then Team stats and Line play (`SEASON_COLS`) for its offense, or with
    side="defteam" what its defense allowed. Counts (`SEASON_PER_GAME`) are
    per game. `venue` ("home_team" / "away_team", `VENUES`) keeps each team's
    own home or away games."""
    teams = sorted(pbp[side].dropna().unique())
    rows = {}
    for team in teams:
        plays = pbp if venue is None else pbp[pbp[venue] == team]
        mine = plays[plays[side] == team]
        if mine.empty:
            continue
        stats = dict(zip(TEAM_STAT_ROWS, _team_col(plays, team, side)))
        stats.update(_line_stats(mine))
        n = int(mine["game_id"].nunique())
        rows[team] = {"Games": n, **{k: (v / n if k in SEASON_PER_GAME and n else v)
                                     for k, v in stats.items()}}
    df = pd.DataFrame.from_dict(rows, orient="index", columns=SEASON_COLS)
    df.index.name = "Team"
    if side != "posteam":
        df = df.drop(columns=[c for c in SEASON_DEFENSE_SKIP if c in df.columns])
    return df


def season_ranks(df: pd.DataFrame, lower_is_better: set = frozenset()) -> pd.DataFrame:
    """0–100 percentile of every cell within its column, 100 = best (the
    lowest in `lower_is_better`); NaN where the value is."""
    pct = df.apply(pd.to_numeric, errors="coerce").rank(pct=True, method="average") * 100
    for col in df.columns:
        if col in lower_is_better:
            pct[col] = 100.0 - pct[col] + 100.0 / max(df[col].notna().sum(), 1)
    return pct


def season_rank_label(df: pd.DataFrame, team: str, lower_is_better: set = frozenset()) -> pd.Series:
    """`team`'s place in each column of `df`, 1 = best: "5th of 32"."""
    out = {}
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        if col in SEASON_VOLUME or pd.isna(s.get(team)):
            out[col] = ""
            continue
        r = s.rank(ascending=col in lower_is_better, method="min")
        out[col] = f"{_ordinal(int(r[team]))} of {int(s.notna().sum())}"
    return pd.Series(out)


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


SEASON_TABLE_ALL = 10**6  # top_players / top_defenders: every player


def season_players(pbp: pd.DataFrame, kind: str, drop: frozenset = frozenset(),
                   teams=None) -> pd.DataFrame:
    """Every player's `top_players` line over `pbp`, one row per player and
    team (a traded player has one per team), with Team and the Games he
    had a play in. `teams` limits it to those offenses."""
    teams = sorted(pbp["posteam"].dropna().unique()) if teams is None else teams
    name = PLAYER_NAME_COL[kind]
    plays = _player_plays(pbp, kind)
    games = plays.groupby(["posteam", name])["game_id"].nunique()
    frames = []
    for team in teams:
        g = top_players(pbp, team, kind, SEASON_TABLE_ALL, drop=drop)
        if g.empty:
            continue
        g.insert(1, "Team", team)
        g.insert(2, "G", [int(games.get((team, p), 0)) for p in g["Player"]])
        frames.append(g)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("Yds", ascending=False, ignore_index=True)


def season_defenders(pbp: pd.DataFrame, drop: frozenset = frozenset(), teams=None) -> pd.DataFrame:
    """Every defender's `top_defenders` line over `pbp`, with Team."""
    teams = sorted(pbp["defteam"].dropna().unique()) if teams is None else teams
    frames = []
    for team in teams:
        g = top_defenders(pbp, team, SEASON_TABLE_ALL, drop=drop)
        if not g.empty:
            g.insert(1, "Team", team)
            frames.append(g)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(out.columns[2], ascending=False, ignore_index=True)


def season_situational(pbp: pd.DataFrame, side: str = "posteam",
                       venue: str | None = None) -> pd.DataFrame:
    """Every team's situational success over `pbp`: teams as rows (offenses,
    or with side="defteam" what each defense allowed), columns (situation,
    metric) over `SR_SITUATIONS` + `SR_SUBTOTALS` × `SR_METRICS`. `venue`
    keeps each team's home or away games (`venue_plays`)."""
    pbp = venue_plays(pbp, venue, side)
    pbp = pbp[pbp[side].notna() & pbp["posteam"].notna()]
    cols = {}
    for label, downs, (d_min, d_max) in SR_SITUATIONS + SR_SUBTOTALS:
        sit = pbp[_situation_mask(pbp, downs, d_min, d_max)]
        rows = {team: _situation_row(grp) for team, grp in sit.groupby(side)}
        for m in SR_METRICS:
            cols[(label, m)] = pd.Series({t: r[m] for t, r in rows.items()}, dtype=float)
    df = pd.DataFrame(cols)
    df.index.name = "Team"
    return df


def situational_distributions_from(table: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """A `season_situational()` table as the {situation: {metric: sorted
    values}} baselines `style_sr_table` colors by."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for (label, m), col in table.items():
        out.setdefault(label, {})[m] = np.sort(col.dropna().to_numpy(dtype=float))
    return out


# ---------- Opponent adjustment ----------
# A season's numbers adjusted for the opponents faced, as SRS does for
# points: an offense's value minus the average strength of the defenses it
# played, a defense's minus that of the offenses it played, iterated until
# both settle. A strength is a team's adjusted value minus the league mean,
# shrunk toward 0 by games / (games + SHRINK_GAMES) so a team seen only once
# or twice (a college team's FCS opponent) can't swing its opponents' numbers.
SHRINK_GAMES = 2
ADJUST_MAX_PASSES = 100
ADJUST_TOL = 1e-9
# Volumes, a team's own penalties and the depth a passer chooses aren't
# down to the opponent.
SEASON_ADJUST_SKIP = SEASON_VOLUME | {"Penalties", "Penalty Yds", "aDoT"}
# The team column whose defensive strength adjusts each player rate, and the
# scale between them (player SR% is in percent, team SR a fraction).
PLAYER_ADJUST = {"passing": {"SR%": ("Pass SR", 100.0), "EPA/play": ("Pass EPA/play", 1.0)},
                 "rushing": {"SR%": ("Rush SR", 100.0), "EPA/play": ("Rush EPA/play", 1.0)},
                 "receiving": {"EPA/play": ("Pass EPA/play", 1.0)}}


def season_games(pbp: pd.DataFrame) -> pd.DataFrame:
    """Each game's two teams: one row per team per game (team, opp, and
    whether the team was at home)."""
    g = pbp.groupby("game_id")[["home_team", "away_team"]].first().reset_index()
    return pd.concat([
        g.rename(columns={"home_team": "team", "away_team": "opp"}).assign(home=True),
        g.rename(columns={"away_team": "team", "home_team": "opp"}).assign(home=False),
    ], ignore_index=True)[["game_id", "team", "opp", "home"]]


def venue_games(games: pd.DataFrame, venue: str | None) -> pd.DataFrame:
    """`season_games` rows for each team's home (venue "home_team") or away
    ("away_team") games; all of them for None."""
    return games if venue is None else games[games["home"] == (venue == "home_team")]


def faced(strength: pd.DataFrame, games: pd.DataFrame, teams) -> pd.DataFrame:
    """Each team's average opponent strength over its `games`."""
    s = strength.reindex(games["opp"]).set_axis(games["team"].to_numpy())
    return s.groupby(level=0).mean().reindex(teams).fillna(0.0)


def apply_strength(raw: pd.DataFrame, strength: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """`raw` (teams as rows) less the average strength of the opponents each
    team met in `games`, in the columns `strength` has: an `opponent_adjust`
    on a subset of games (a venue) with strengths from all of them. The
    subset's mean is kept."""
    cols = [c for c in strength.columns if c in raw.columns]
    out = raw.copy()
    adj = raw[cols].astype(float) - faced(strength[cols], games, raw.index)
    out[cols] = adj + (raw[cols].astype(float).mean() - adj.mean())
    return out


def opponent_adjust(off: pd.DataFrame, def_: pd.DataFrame, games: pd.DataFrame,
                    skip: set = SEASON_ADJUST_SKIP):
    """Opponent-adjusted copies of a season's offense table `off` (what each
    team produced) and defense table `def_` (what each allowed), teams as
    rows, over the schedule `games` (`season_games`). Columns in both tables
    and not in `skip` are adjusted; the rest are left as they are. Returns
    (off_adj, def_adj, off_strength, def_strength); a strength is how far a
    team pushes its opponents' numbers, already shrunk."""
    cols = [c for c in off.columns if c in def_.columns and c not in skip
            and pd.api.types.is_numeric_dtype(off[c])]
    o, d = off[cols].astype(float), def_[cols].astype(float)
    m_off, m_def = o.mean(), d.mean()
    n = games.groupby("team").size()

    def _weight(idx) -> pd.Series:
        g = n.reindex(idx).fillna(0)
        return g / (g + SHRINK_GAMES)

    w_off, w_def = _weight(o.index), _weight(d.index)

    o_adj, d_adj = o.copy(), d.copy()
    for _ in range(ADJUST_MAX_PASSES):
        o_str = (o_adj - m_off).mul(w_off, axis=0)
        d_str = (d_adj - m_def).mul(w_def, axis=0)
        o_new = o - faced(d_str, games, o.index)
        d_new = d - faced(o_str, games, d.index)
        delta = max(float(np.nanmax(np.abs((o_new - o_adj).to_numpy()), initial=0)),
                    float(np.nanmax(np.abs((d_new - d_adj).to_numpy()), initial=0)))
        o_adj, d_adj = o_new, d_new
        if delta < ADJUST_TOL:
            break
    # Keep the league's mean where it was: adjusting moves teams, not the league.
    o_adj += m_off - o_adj.mean()
    d_adj += m_def - d_adj.mean()
    off_out, def_out = off.copy(), def_.copy()
    off_out[cols], def_out[cols] = o_adj, d_adj
    return (off_out, def_out, (o_adj - m_off).mul(w_off, axis=0),
            (d_adj - m_def).mul(w_def, axis=0))


def adjust_players(df: pd.DataFrame, pbp: pd.DataFrame, kind: str,
                   def_strength: pd.DataFrame) -> pd.DataFrame:
    """A `season_players()` table with its rates (`PLAYER_ADJUST`) less the
    average strength of the defenses each player's plays came against,
    weighted by his plays against each."""
    if df.empty:
        return df
    name = PLAYER_NAME_COL[kind]
    plays = _player_plays(pbp, kind)
    per_def = plays.groupby(["posteam", name, "defteam"]).size().rename("n").reset_index()
    out = df.copy()
    key = pd.MultiIndex.from_arrays([out["Team"], out["Player"]])
    for col, (team_col, scale) in PLAYER_ADJUST[kind].items():
        if col not in out.columns or team_col not in def_strength.columns:
            continue
        s = per_def.assign(_s=per_def["defteam"].map(def_strength[team_col]).fillna(0.0) * scale)
        faced = (s["_s"] * s["n"]).groupby([s["posteam"], s[name]]).sum() / \
            s.groupby(["posteam", name])["n"].sum()
        out[col] = (out[col] - faced.reindex(key).fillna(0.0).to_numpy()).round(
            1 if col == "SR%" else 2)
    return out


def season_player_distributions(players: dict[str, pd.DataFrame],
                                min_plays: dict[str, int]) -> dict[str, dict[str, np.ndarray]]:
    """The leaders' rates (`PLAYER_RATE_COLS`) over the season tables'
    players with at least `min_plays[kind]` plays: `player_distributions()`'s
    shape, for `style_player_table`."""
    out = {}
    for kind, df in players.items():
        if df.empty:
            out[kind] = {}
            continue
        q = df[df[PLAYER_VOLUME_COL[kind]] >= min_plays[kind]]
        out[kind] = {c: np.sort(q[c].dropna().to_numpy(dtype=float))
                     for c in PLAYER_RATE_COLS[kind] if c in q.columns}
    return out


# ---------- Drives ----------
def _drive_outcome(drive_plays: pd.DataFrame) -> str:
    """Determine how a drive ended from its plays."""
    if drive_plays["touchdown"].fillna(0).astype(bool).any():
        return "TD"
    fg = drive_plays[drive_plays["play_type"] == "field_goal"]
    if not fg.empty:
        r = fg["field_goal_result"].dropna()
        if not r.empty:
            v = r.iloc[-1]
            return "FG" if v == "made" else ("FG Blocked" if v == "blocked" else "FG Miss")
    if drive_plays["interception"].fillna(0).astype(bool).any():
        return "Interception"
    if drive_plays["fumble_lost"].fillna(0).astype(bool).any():
        return "Fumble"
    if (drive_plays["play_type"] == "punt").any():
        return "Punt"
    if drive_plays["fourth_down_failed"].fillna(0).astype(bool).any():
        return "Downs"
    if drive_plays["safety"].fillna(0).astype(bool).any():
        return "Safety"
    return "EOH/EOG"

def drive_chart(revealed: pd.DataFrame) -> pd.DataFrame:
    """One row per drive: team, plays, pass/run split, yards, first downs, outcome."""
    if revealed.empty or "drive" not in revealed.columns:
        return pd.DataFrame()
    rows = []
    for drive_num, grp in revealed.groupby("drive", sort=True):
        off = grp[grp["posteam"].notna()]
        if off.empty:
            continue
        posteam = off["posteam"].dropna().iloc[0]
        qtr_start = int(off["qtr"].dropna().iloc[0]) if off["qtr"].notna().any() else ""
        _special = {"kickoff", "extra_point", "no_play"}
        yl_series = off[~off["play_type"].isin(_special)]["yardline_100"].dropna()
        if yl_series.empty:
            yl_series = off["yardline_100"].dropna()
        if not yl_series.empty:
            y = int(yl_series.iloc[0])
            start = f"OWN {100 - y}" if y > 50 else ("50" if y == 50 else f"OPP {y}")
        else:
            start = ""
        p_mask = off["pass_attempt"].fillna(0) == 1
        r_mask = off["rush_attempt"].fillna(0) == 1
        sc = off[p_mask | r_mask]
        rows.append({
            "Drive": int(drive_num),
            "Team": posteam,
            "Qtr": qtr_start,
            "Start": start,
            "Plays": len(sc),
            "Pass": int(p_mask.sum()),
            "Run": int(r_mask.sum()),
            "Yards": int(sc["yards_gained"].fillna(0).sum()),
            "1st Downs": int(sc["first_down"].fillna(0).sum()),
            "Outcome": _drive_outcome(grp),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Drive", ascending=False).reset_index(drop=True)

_FIELD_PLAYS = {"pass", "run", "punt", "field_goal", "qb_kneel", "qb_spike", "no_play"}

def yl_label(y: float) -> str:
    """yardline_100 → 'OWN 25' / '50' / 'OPP 3'."""
    y = int(round(y))
    return f"OWN {100 - y}" if y > 50 else ("50" if y == 50 else f"OPP {y}")

# Play categories that colour the drive bars: (legend label, colour), in legend order.
PLAY_CATS = {
    "pass_early": ("Pass · 1st/2nd down", "#2a78d6"),
    "pass_late": ("Pass · 3rd/4th down", "#4a3aa7"),
    "run_early": ("Run · 1st/2nd down", "#e34948"),
    "run_late": ("Run · 3rd/4th down", "#a8327a"),
    "penalty": ("Penalty", "#eda100"),
}

def _play_category(r) -> str | None:
    """Drive-bar category of a play; None for plays that don't get a segment
    (punts, field goals, timeouts and other no_play rows without a penalty).
    Scrambles count as runs and sacks as passes, as in nflfastR's play_type."""
    pt = r["play_type"]
    if pt == "no_play":
        return "penalty" if r.get("penalty") == 1 else None
    late = pd.notna(r["down"]) and r["down"] >= 3
    if pt in ("pass", "qb_spike"):
        return "pass_late" if late else "pass_early"
    if pt in ("run", "qb_kneel"):
        return "run_late" if late else "run_early"
    return None

def drive_field_spots(revealed: pd.DataFrame) -> pd.DataFrame:
    """Start and end spot (yardline_100 of the offense) for every revealed drive.

    The end spot is where the ball was when the drive ended: the goal line for
    a TD, the line of scrimmage for a punt, FG or interception, and the spot
    after the last play otherwise (including the drive in progress)."""
    if revealed.empty or "drive" not in revealed.columns:
        return pd.DataFrame()
    # Time of possession runs until the next drive starts, since the final
    # punt, kick or scoring play takes clock too. That only holds within a
    # half: the clock resets at halftime and again in overtime.
    groups = list(revealed.groupby("drive", sort=True))
    nxt_start = {}
    for (d, grp), (_, nxt) in zip(groups, groups[1:]):
        a = grp[["game_seconds_remaining", "qtr"]].dropna()
        b = nxt[["game_seconds_remaining", "qtr"]].dropna()
        if a.empty or b.empty:
            continue
        qa, qb = int(a["qtr"].iloc[-1]), int(b["qtr"].iloc[0])
        if qa < 5 and qb < 5 and (qa <= 2) == (qb <= 2):
            nxt_start[d] = float(b["game_seconds_remaining"].iloc[0])
    rows = []
    for drive_num, grp in groups:
        gsr = grp["game_seconds_remaining"].dropna()
        top = None
        if not gsr.empty:
            top = max(0.0, float(gsr.max()) - nxt_start.get(drive_num, float(gsr.min())))
        off = grp[grp["posteam"].notna()]
        plays = off[
            off["play_type"].isin(_FIELD_PLAYS)
            & off["two_point_conv_result"].isna()
            & off["yardline_100"].notna()
        ]
        if plays.empty:
            continue  # e.g. only the kickoff is revealed so far
        posteam = plays["posteam"].iloc[0]
        start = float(plays["yardline_100"].iloc[0])
        last = plays.iloc[-1]
        los = float(last["yardline_100"])
        turnover = bool(
            plays["interception"].fillna(0).astype(bool).any()
            or plays["fumble_lost"].fillna(0).astype(bool).any()
        )
        if last["play_type"] in ("punt", "field_goal") or last.get("interception") == 1:
            end = los
        elif last.get("safety") == 1:
            end = 100.0
        elif last.get("touchdown") == 1 and not turnover:
            end = 0.0
        else:
            gained = 0.0 if pd.isna(last["yards_gained"]) else float(last["yards_gained"])
            if last["play_type"] == "no_play" and last.get("penalty") == 1:
                pen = 0.0 if pd.isna(last["penalty_yards"]) else float(last["penalty_yards"])
                gained = -pen if last["penalty_team"] == posteam else pen
            end = los - gained
        end = min(max(end, 0.0), 100.0)
        # One segment per play, from its line of scrimmage to the next play's
        # (the drive's end spot for the last one).
        nxt_spot = plays["yardline_100"].astype(float).tolist()[1:] + [end]
        segments = []
        for (_, p), to in zip(plays.iterrows(), nxt_spot):
            cat = _play_category(p)
            if cat is None:
                continue
            frm = float(p["yardline_100"])
            desc = str(p["desc"] or "")
            dd = down_distance(p)
            segments.append({
                "cat": cat, "from": frm, "to": to,
                "hover": (f"<b>{PLAY_CATS[cat][0]}</b>"
                          + f"<br>{dd + ' · ' if dd else ''}{yl_label(frm)} · {frm - to:+.0f} yds"
                          + f"<br>{desc[:90] + '…' if len(desc) > 90 else desc}"),
            })
        sc = off[(off["pass_attempt"].fillna(0) == 1) | (off["rush_attempt"].fillna(0) == 1)]
        rows.append({
            "drive": int(drive_num),
            "team": posteam,
            "qtr": int(off["qtr"].dropna().iloc[0]) if off["qtr"].notna().any() else None,
            "start": start,
            "end": end,
            "plays": len(sc),
            "yards": int(sc["yards_gained"].fillna(0).sum()),
            "top": top,
            "outcome": _drive_outcome(grp),
            "segments": segments,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # The last revealed drive hasn't necessarily ended: label it as in progress
    # unless the last revealed row is itself an end-of-quarter/half/game marker.
    last_desc = str(revealed["desc"].iloc[-1] or "")
    if df["outcome"].iloc[-1] == "EOH/EOG" and not last_desc.upper().startswith("END "):
        df.loc[df.index[-1], "outcome"] = "In progress"
    return df


# ---------- Scoring and key plays ----------
def scoring_timeline(revealed: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    """Every scoring play in revealed: TD, FG, Safety, XP, 2PT."""
    if revealed.empty:
        return pd.DataFrame()
    mask = (
        (revealed["touchdown"].fillna(0) == 1) |
        (revealed["field_goal_result"] == "made") |
        (revealed["safety"].fillna(0) == 1) |
        (revealed["extra_point_result"] == "good") |
        (revealed["two_point_conv_result"] == "success")
    )
    plays = revealed[mask].copy()
    if plays.empty:
        return pd.DataFrame()

    def _score_type(r) -> str:
        if r["safety"] == 1:
            return "Safety"
        if r["field_goal_result"] == "made":
            return "FG"
        if r["extra_point_result"] == "good":
            return "XP"
        if r["two_point_conv_result"] == "success":
            return "2PT"
        return "TD"

    plays["Type"] = plays.apply(_score_type, axis=1)
    plays["Team"] = plays.apply(
        lambda r: r["defteam"] if r["safety"] == 1 else r["posteam"], axis=1
    )
    plays["Score"] = plays.apply(
        lambda r: f"{away} {int(r['total_away_score'] or 0)} — {int(r['total_home_score'] or 0)} {home}",
        axis=1,
    )
    plays["Q"] = plays["qtr"].apply(lambda x: str(int(x)) if pd.notna(x) else "")
    return plays[["Q", "time", "Team", "Type", "Score", "desc"]].rename(
        columns={"time": "Clock", "desc": "Description"}
    ).reset_index(drop=True)

PLAY_KINDS = ["Pass", "Run", "Special teams", "Penalty"]


def play_kind(r) -> str | None:
    """Filter category of a play, from nflfastR's play_type: scrambles count as
    runs and sacks as passes, as on the drive chart; Penalty is an accepted
    penalty with no play. None for timeouts and period markers."""
    pt = r["play_type"]
    if pt in ("pass", "qb_spike"):
        return "Pass"
    if pt in ("run", "qb_kneel"):
        return "Run"
    if pt in ("punt", "field_goal", "extra_point", "kickoff"):
        return "Special teams"
    if pt == "no_play" and r.get("penalty") == 1:
        return "Penalty"
    return None


def filter_plays(plays: pd.DataFrame, downs=(), kinds=(), teams=()) -> pd.DataFrame:
    """The rows of `plays` on any of `downs` (1-4), of any of `kinds`
    (PLAY_KINDS) and run by any of `teams` (posteam). An empty selection
    keeps everything for that filter."""
    mask = pd.Series(True, index=plays.index)
    if downs:
        mask &= plays["down"].isin(list(downs))
    if kinds:
        mask &= plays.apply(play_kind, axis=1).isin(list(kinds))
    if teams:
        mask &= plays["posteam"].isin(list(teams))
    return plays[mask]


def explosive_plays(revealed: pd.DataFrame, min_pass_yds: int = 15, min_rush_yds: int = 10) -> pd.DataFrame:
    """Passing plays >= min_pass_yds yards or rushing plays >= min_rush_yds yards, sorted by yards desc."""
    if revealed.empty:
        return pd.DataFrame()
    pass_mask = (revealed["pass_attempt"].fillna(0) == 1) & (revealed["passing_yards"].fillna(0) >= min_pass_yds)
    rush_mask = (revealed["rush_attempt"].fillna(0) == 1) & (revealed["rushing_yards"].fillna(0) >= min_rush_yds)
    return revealed[pass_mask | rush_mask].sort_values("yards_gained", ascending=False)

def top_plays_wpa(revealed: pd.DataFrame, home: str, away: str, n: int = 25) -> pd.DataFrame:
    """Top n plays by absolute win probability added, computed from home_wp shifts."""
    if revealed.empty:
        return pd.DataFrame()
    df = revealed.copy().reset_index(drop=True)
    wp = df["home_wp"].copy()
    wpa = wp.shift(-1) - wp
    df["_wpa"] = wpa
    df["_abs_wpa"] = wpa.abs()
    plays = df[df["_abs_wpa"].notna() & (df["_abs_wpa"] > 0)].nlargest(n, "_abs_wpa").copy()
    if plays.empty:
        return pd.DataFrame()
    plays["Q"] = plays["qtr"].apply(lambda x: str(int(x)) if pd.notna(x) else "")
    plays["Score"] = plays.apply(
        lambda r: f"{away} {int(r['total_away_score'] or 0)}–{int(r['total_home_score'] or 0)} {home}", axis=1
    )
    plays["D&D"] = plays.apply(down_distance, axis=1)
    plays["WPA"] = plays["_wpa"].round(3)
    plays["For"] = plays["_wpa"].apply(lambda x: home if x > 0 else away)
    return plays[["Q", "time", "posteam", "D&D", "Score", "For", "desc", "WPA", "_abs_wpa"]].rename(
        columns={"time": "Clock", "posteam": "Off", "desc": "Description"}
    ).reset_index(drop=True)

def wp_crossings(revealed: pd.DataFrame, elapsed_min: pd.Series) -> list[float]:
    """Elapsed minutes at which the revealed home win probability crosses 50%.
    `elapsed_min` is each revealed play's place on the play timeline."""
    d = revealed[["home_wp"]].assign(t=elapsed_min).dropna()
    if len(d) < 2:
        return []
    t = d["t"].to_numpy()
    w = d["home_wp"].to_numpy() - 0.5
    out = []
    for i in range(1, len(w)):
        if w[i - 1] * w[i] < 0:
            out.append(float(t[i - 1] + (t[i] - t[i - 1]) * w[i - 1] / (w[i - 1] - w[i])))
        elif w[i] == 0 and w[i - 1] != 0:
            out.append(float(t[i]))
    return out


# ---------- Game management ----------
# Where the game stands for the clock: timeouts left, the next snap, and what
# the end of a half could look like from here: if the offense goes
# three-and-out, how much time the other team gets the ball back with; whether
# the leader can kneel it out. All of it is estimated from the plays revealed
# so far, never from the play after them, which isn't revealed.

# Clock runoff, from 2024 NFL plays: a leading offense's run in the 4th
# quarter to its next snap takes 41s (median), kneel to kneel 37s, a punt's
# snap to the next snap 9s.
RUN_SECS = 5        # snap to whistle, a run in bounds
KNEEL_SECS = 2
PUNT_SECS = 9
FG_SECS = 11        # the kick and the kickoff after it
KICKOFF_SECS = 6
RUNOFF_SECS = 37    # whistle to the next snap with the clock running
TWO_MINUTES = 120
SCENARIO_SECS = 360  # the scenarios show in the last 6 minutes of each half
FG_RANGE = 37       # yardline_100 a three-and-out kicks a field goal from (a 54-yarder)
NET_PUNT = 42
TIMEOUTS_PER_HALF = 3

_TIMEOUT = re.compile(r"\s*Timeout\b", re.I)
_HALF_END = re.compile(r"\s*END (QUARTER 2|HALF)\b|\s*End of (the )?(2nd Quarter|1st Half|Half)\b", re.I)
_GAME_END = re.compile(r"\s*END (QUARTER 4|GAME|OVERTIME)\b|\s*End of (4th Quarter|Game|Regulation|Overtime)\b", re.I)
_OOB = re.compile(r"\b(ran|pushed|run|sacked|forced) ob\b|out of bounds", re.I)
# The spot a kick or turnover ends at: "... to the ARI 30", "pushed ob at BUF 45".
_SPOT = re.compile(r"\b(?:to|at) (?:the )?([A-Z][A-Za-z&]{1,5}) ?(\d{1,2})\b")


@dataclass(frozen=True)
class ClockRules:
    """The league's clock rules the game-management estimates follow."""
    two_minute_from: int            # first season the clock stops at 2:00 of each half
    oob_stop_secs: tuple[int, int]  # out of bounds stops the clock inside this much of the 1st / 2nd half
    first_down_stop_secs: int       # a first down stops the clock inside this much of a half
    touchbacks: tuple[tuple[int, int], ...]  # (from season, yardline_100) of a kickoff touchback
    missed_fg_back: int             # a missed field goal goes back to the kick's spot, this far behind the line

    def two_minute(self, season) -> bool:
        return pd.notna(season) and int(season) >= self.two_minute_from

    def touchback(self, season) -> int:
        s = int(season) if pd.notna(season) else 9999
        return [yl for start, yl in self.touchbacks if s >= start][-1]


# NFL: out of bounds stops the clock inside 2:00 of the 1st half and 5:00 of
# the 2nd; kickoff touchbacks to the 25, then the 30 (2024), the 35 (2025).
NFL_CLOCK = ClockRules(0, (120, 300), 0, ((0, 75), (2024, 70), (2025, 65)), 7)
# College: out of bounds always stops it, first downs inside 2:00 of a half
# (2023 on), the two-minute timeout came in 2024. Missed kicks go back to the
# line of scrimmage.
CFB_CLOCK = ClockRules(2024, (1800, 1800), 120, ((0, 75),), 0)


@dataclass
class Snap:
    """The next snap, estimated from the plays revealed so far."""
    posteam: str
    defteam: str
    qtr: int
    half_secs: float | None     # clock left in the half (in OT, the period); None if untimed
    running: bool               # the clock is running into this snap
    down: int
    ydstogo: float
    yardline_100: float
    timeouts: dict              # team → timeouts left (None if unknown)
    score: dict                 # team → points
    kickoff: bool               # the offense first receives a kickoff (after a score)
    season: float
    home: str
    receives_2h: str | None     # who receives the 2nd-half kickoff


def timeouts_left(revealed: pd.DataFrame, home: str, away: str) -> dict:
    """Each team's timeouts left after the last revealed play: the feed's
    count on the last row that has one, 3 each at the start of a half. None
    for a team when it isn't known (no rows yet, or overtime about to start)."""
    if revealed.empty:
        return {home: TIMEOUTS_PER_HALF, away: TIMEOUTS_PER_HALF}
    last = str(revealed["desc"].iloc[-1] or "")
    if _HALF_END.match(last):
        return {home: TIMEOUTS_PER_HALF, away: TIMEOUTS_PER_HALF}
    if _GAME_END.match(last):
        return {home: None, away: None}
    t = revealed[["home_timeouts_remaining", "away_timeouts_remaining"]].dropna()
    if t.empty:
        return {home: TIMEOUTS_PER_HALF, away: TIMEOUTS_PER_HALF}
    h, a = t.iloc[-1]
    return {home: int(h), away: int(a)}


def _end_spot(desc: str, team: str, other: str) -> float | None:
    """Where `desc`'s play ended, as yardline_100 for `team`, from its last
    "to/at TEAM NN" spot; None if the text names none."""
    spots = _SPOT.findall(desc or "")
    if not spots:
        return None
    side, n = spots[-1][0].upper(), int(spots[-1][1])
    if n == 50:
        return 50.0
    if side == team.upper():
        return float(100 - n)
    if side == other.upper():
        return float(n)
    return None


def next_snap(revealed: pd.DataFrame, rules: ClockRules) -> Snap | None:
    """The next snap's offense, down, distance, spot and clock, estimated from
    the last revealed play (its result, a timeout or a quarter break after it).
    After a score it is the receiving team's first snap after the kickoff, at
    a touchback. None before any play, once the half or game is over, or when
    the last play leaves it unclear."""
    if revealed.empty:
        return None
    desc = revealed["desc"].fillna("").astype(str)
    if _HALF_END.match(desc.iloc[-1]) or _GAME_END.match(desc.iloc[-1]):
        return None
    snaps = revealed["play_type"].notna() & revealed["posteam"].notna() & ~desc.str.match(_TIMEOUT)
    if not snaps.any():
        return None
    pos = int(np.flatnonzero(snaps.to_numpy())[-1])
    s = revealed.iloc[pos]
    home, away = s["home_team"], s["away_team"]
    off, dfn = s["posteam"], s["defteam"]
    if pd.isna(dfn):
        dfn = away if off == home else home
    season = s["season"]
    scores = revealed[["total_home_score", "total_away_score"]].ffill().fillna(0)
    score = {home: int(scores["total_home_score"].iloc[-1]), away: int(scores["total_away_score"].iloc[-1])}
    before = scores.iloc[pos - 1] if pos > 0 else pd.Series({"total_home_score": 0, "total_away_score": 0})
    gained = {home: score[home] - before["total_home_score"], away: score[away] - before["total_away_score"]}
    first_def = revealed["defteam"].dropna()
    receives_2h = first_def.iloc[0] if not first_def.empty else None

    ptype = s["play_type"]
    yl = float(s["yardline_100"]) if pd.notna(s["yardline_100"]) else 75.0
    desc_s = desc.iloc[pos]
    flag = lambda c: pd.notna(s.get(c)) and float(s.get(c)) == 1  # noqa: E731
    secs = {"run": RUN_SECS, "pass": RUN_SECS, "qb_kneel": KNEEL_SECS, "qb_spike": 1,
            "punt": PUNT_SECS, "field_goal": 5, "kickoff": KICKOFF_SECS}.get(ptype, 0)
    gsr = s["game_seconds_remaining"]
    if pd.isna(gsr):
        half = None   # untimed college overtime
    else:
        g = float(gsr) - secs
        half = g - 1800 if s["qtr"] <= 2 else g
        if half <= 0:
            return None
    two_min = rules.two_minute(season) and s["qtr"] in (2, 4)
    stopped_after = pos < len(revealed) - 1   # a timeout or a quarter break since

    def change(team, other, yardline, extra=0.0, kickoff=False):
        h = None if half is None else max(half - extra, 1.0)
        return team, other, 1, 10.0, yardline, h, False, kickoff

    tb = float(rules.touchback(season))
    scorer = home if gained[home] > 0 else (away if gained[away] > 0 else None)
    try_row = ptype == "extra_point" or pd.notna(s.get("two_point_conv_result"))
    if scorer is not None and (flag("touchdown") or try_row or flag("safety")
                               or s.get("field_goal_result") == "made"):
        # The scorer kicks off (after a safety, the team scored on does).
        recv = scorer if flag("safety") else (away if scorer == home else home)
        res = change(recv, home if recv == away else away, tb, KICKOFF_SECS, kickoff=True)
    elif flag("touchdown") or try_row:
        return None   # the score isn't in yet
    elif ptype == "kickoff":
        recv, kick = (dfn, off) if flag("fumble_lost") else (off, dfn)
        spot = _end_spot(desc_s, recv, kick)
        res = change(recv, kick, spot if spot is not None else tb)
    elif ptype == "punt":
        recv, kick = (off, dfn) if flag("fumble_lost") else (dfn, off)
        spot = _end_spot(desc_s, recv, kick)
        if spot is None:
            spot = 80.0 if yl - NET_PUNT < 20 else 100 - (yl - NET_PUNT)
        res = change(recv, kick, spot)
    elif flag("interception") or flag("fumble_lost"):
        spot = _end_spot(desc_s, dfn, off)
        res = change(dfn, off, spot if spot is not None else 100 - yl)
    elif ptype == "field_goal":
        res = change(dfn, off, 100 - max(yl + rules.missed_fg_back, 20))
    else:
        if pd.isna(s["down"]) or pd.isna(s["ydstogo"]):
            return None
        down, togo = int(s["down"]), float(s["ydstogo"])
        if ptype == "no_play":
            pen = float(s["penalty_yards"]) if pd.notna(s.get("penalty_yards")) else 0.0
            yards = pen if s.get("penalty_team") == dfn else (-pen if s.get("penalty_team") == off else 0.0)
            new_down = 1 if flag("first_down") else down
        else:
            yards = float(s["yards_gained"]) if pd.notna(s["yards_gained"]) else 0.0
            new_down = 1 if flag("first_down") else down + 1
        new_yl = min(max(yl - yards, 1.0), 99.0)
        if new_down > 4:   # turned over on downs
            res = change(dfn, off, 100 - new_yl)
        else:
            new_togo = min(10.0, new_yl) if new_down == 1 else max(togo - yards, 1.0)
            incomplete = (ptype == "pass" and not flag("complete_pass") and not flag("sack")
                          and not flag("interception"))
            oob = bool(_OOB.search(desc_s)) and half is not None and \
                half <= rules.oob_stop_secs[0 if s["qtr"] <= 2 else 1]
            first = new_down == 1 and half is not None and half <= rules.first_down_stop_secs
            at_warning = two_min and half is not None and half + secs > TWO_MINUTES >= half
            running = not (incomplete or ptype in ("qb_spike", "no_play") or oob or first
                           or flag("penalty") or at_warning or stopped_after or half is None)
            res = (off, dfn, new_down, new_togo, new_yl, half, running, False)
    team, other, down, togo, yardline, h, running, kickoff = res
    qtr = int(s["qtr"])
    if h is not None and qtr <= 4:
        qtr = (1 if h > 900 else 2) if qtr <= 2 else (3 if h > 900 else 4)
    return Snap(team, other, qtr, h, running and not stopped_after, int(down), float(togo),
                float(yardline), timeouts_left(revealed, home, away), score, kickoff,
                season, home, receives_2h)


def _burn(secs_left: float, running: bool, stop_tos: int, plays: list[float],
          two_min: bool) -> tuple[float, int]:
    """Clock left after the offense snaps `plays` (each play's seconds) and
    lets the clock run between them, while the other side spends up to
    `stop_tos` timeouts stopping it, and the two-minute warning stops it once
    if `two_min`. Returns (seconds left after the last play, timeouts spent)."""
    h, used = secs_left, 0
    for p in plays:
        if running:
            if used < stop_tos:
                used += 1
            else:
                nxt = h - RUNOFF_SECS
                h = TWO_MINUTES if two_min and h > TWO_MINUTES >= nxt else nxt
        if h <= 0:
            return 0.0, used
        start, h = h, h - p
        running = not (two_min and start > TWO_MINUTES >= h)
        if h <= 0:
            return 0.0, used
    return h, used


def three_and_out(snap: Snap, rules: ClockRules) -> pd.DataFrame:
    """If the offense runs on each down left before 4th and then kicks (a
    punt, or a field goal in range), when the other team gets the ball back,
    one row per number of timeouts it spends stopping the clock, most first.
    Columns: Timeouts used, Ball back with (seconds), Timeouts left, Plays
    (about how many it can run then, `plays_left()`); plus the
    other team's state then for a win probability model (`wp_state`)."""
    if snap.half_secs is None or snap.down > 4:
        return pd.DataFrame()
    fg = snap.yardline_100 <= FG_RANGE
    runs = max(4 - snap.down, 0)
    two_min = rules.two_minute(snap.season) and snap.qtr in (2, 4)
    tos = snap.timeouts.get(snap.defteam) or 0
    rows, seen = [], set()
    for k in range(tos, -1, -1):
        left, used = _burn(snap.half_secs, snap.running, k, [RUN_SECS] * runs + [FG_SECS if fg else PUNT_SECS],
                           two_min)
        if (left, used) in seen:
            continue
        seen.add((left, used))
        if left <= 0:   # the clock runs out first: no kick
            yl, lead = 75.0, 0
        elif fg:
            yl, lead = float(rules.touchback(snap.season)), 3
        else:
            yl = 80.0 if snap.yardline_100 - NET_PUNT < 20 else 100 - (snap.yardline_100 - NET_PUNT)
            lead = 0
        rows.append({"Timeouts used": used, "Ball back with": left, "Timeouts left": tos - used,
                     "Plays": plays_left(left, clock_stops(snap, snap.defteam, left, tos - used, rules)),
                     **wp_state(snap, snap.defteam, left, tos - used,
                                snap.timeouts.get(snap.posteam), yl, -lead)})
    return pd.DataFrame(rows)


# A trailing offense's pace late in a half, from 2024 NFL plays (trailing by
# 1-16, last 5:00 of the 4th or 2:00 of the 2nd): 44% of its plays stop the
# clock (incomplete, out of bounds, spike, flag) and take ~6s snap to snap; a
# play with the clock running takes 24s, or ~7s when a timeout stops it.
HURRY_PLAY_SECS = 6
HURRY_RUNOFF_SECS = 18
HURRY_STOP_SHARE = 0.44


def plays_left(half_secs: float, stops: int) -> int:
    """About how many snaps a hurrying offense gets in `half_secs` with
    `stops` clock stoppages of its own (timeouts, the two-minute warning):
    each saves the runoff after one play the clock would run on. Never more
    than if every play stopped the clock."""
    if half_secs is None or half_secs <= 0:
        return 0
    per_play = HURRY_PLAY_SECS + (1 - HURRY_STOP_SHARE) * HURRY_RUNOFF_SECS
    n = min((half_secs + stops * HURRY_RUNOFF_SECS) / per_play, half_secs / HURRY_PLAY_SECS)
    return max(int(n), 1)


def clock_stops(snap: Snap, team: str, half_secs: float, tos: int | None, rules: ClockRules) -> int:
    """`team`'s clock stoppages with `half_secs` left: its timeouts, plus the
    two-minute warning while it's still to come."""
    warn = rules.two_minute(snap.season) and snap.qtr in (2, 4) and half_secs > TWO_MINUTES
    return (tos or 0) + int(warn)


def first_downs_to_kneel(snap: Snap, rules: ClockRules, max_first_downs: int = 3) -> int | None:
    """How many more first downs the offense needs before it can kneel out
    the half: 0 if it can kneel now, with the defense spending all its
    timeouts. None if it would need more than `max_first_downs`."""
    if snap.half_secs is None:
        return None
    two_min = rules.two_minute(snap.season) and snap.qtr in (2, 4)
    tos = snap.timeouts.get(snap.defteam) or 0
    for n in range(max_first_downs + 1):
        # Runs to convert each first down (on 3rd down), then kneels on 1st to 4th.
        plays = [KNEEL_SECS] * (5 - snap.down) if n == 0 else \
            [RUN_SECS] * (max(4 - snap.down, 1) + 3 * (n - 1)) + [KNEEL_SECS] * 4
        if _burn(snap.half_secs, snap.running, tos, plays, two_min)[0] <= 0:
            return n
    return None


def scores_needed(behind: int) -> str:
    """What it takes to tie or win, `behind` points down."""
    if behind <= 0:
        return ""
    if behind <= 2:
        return "a field goal wins it"
    if behind == 3:
        return "a field goal ties, a touchdown wins"
    if behind <= 6:
        return "needs a touchdown"
    if behind == 7:
        return "a touchdown ties, with a two-point try to win"
    if behind == 8:
        return "a touchdown and a two-point try tie it"
    n = int(np.ceil(behind / 8))
    return f"needs {n} scores"


def wp_state(snap: Snap, team: str, half_secs: float, tos: int | None, opp_tos: int | None,
             yardline_100: float, diff_change: int = 0) -> dict:
    """`team` with the ball 1st & 10 at `yardline_100` with `half_secs` left,
    as the columns nflfastR's win probability model reads (prefixed `_`, so a
    table can carry them hidden). `diff_change` adds to its score margin."""
    other = next(t for t in snap.score if t != team)
    first_half = snap.qtr <= 2
    game = half_secs + (1800 if first_half else 0) if snap.qtr <= 4 else np.nan
    return {"_posteam": team, "_home": float(team == snap.home), "_qtr": snap.qtr,
            "_half_seconds_remaining": half_secs, "_game_seconds_remaining": game,
            "_score_differential": snap.score[team] - snap.score[other] + diff_change,
            "_down": 1.0, "_ydstogo": 10.0, "_yardline_100": yardline_100,
            "_posteam_timeouts_remaining": tos, "_defteam_timeouts_remaining": opp_tos,
            "_receive_2h_ko": float(first_half and team == snap.receives_2h)}


def snap_wp_state(snap: Snap) -> dict:
    """The next snap itself, as `wp_state()` columns."""
    st = wp_state(snap, snap.posteam, snap.half_secs if snap.half_secs is not None else np.nan,
                  snap.timeouts.get(snap.posteam), snap.timeouts.get(snap.defteam), snap.yardline_100)
    st.update({"_down": float(snap.down), "_ydstogo": snap.ydstogo})
    return st


def fmt_clock(secs: float | None) -> str:
    if secs is None or pd.isna(secs):
        return "—"
    s = int(round(max(secs, 0)))
    return f"{s // 60}:{s % 60:02d}"


# ---------- Play labels ----------
def field_pos_label(r) -> str:
    yl = r["yardline_100"]
    if pd.isna(yl):
        return ""
    yl = int(yl)
    if yl > 50:
        return f"OWN {100 - yl}"
    elif yl == 50:
        return "50"
    else:
        return f"OPP {yl}"

def play_type_label(r) -> str:
    down = r["down"]
    is_pass = r["pass_attempt"] == 1
    is_run = r["rush_attempt"] == 1
    icon = "🏈 " if is_pass else ("🏃 " if is_run else "")
    play_kind = f"{icon}Pass" if is_pass else (f"{icon}Run" if is_run else "")
    if pd.notna(down) and down in (3, 4):
        prefix = "3rd" if down == 3 else "4th"
        return f"{prefix} & {play_kind}" if play_kind else prefix
    return play_kind

def down_distance(r) -> str:
    if pd.notna(r["down"]) and pd.notna(r["ydstogo"]):
        return f"{int(r['down'])} & {int(r['ydstogo'])}"
    return ""

def success_emoji(r) -> str:
    if pd.notna(r["epa"]) and r["epa"] > 0:
        return "✅"
    return ""
