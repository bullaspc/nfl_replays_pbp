"""
League-neutral replay logic: the play timeline and cursor, and every stat the
app shows, built from play-by-play in nflfastR's column layout (`PBP_COLS`).

Both leagues' data is mapped to that layout (see leagues.py), so nothing here
knows which league it's looking at. No Streamlit: everything is a plain
function of DataFrames, importable from tools and tests.
"""

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
    "third_down_converted", "third_down_failed",
    "fourth_down_converted", "fourth_down_failed",
    "goal_to_go", "yardline_100", "drive",
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
    for _, grp in td.groupby(["drive", "qtr"], dropna=True):
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

def team_stats(revealed: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    """Advanced offensive team stats, returned as a transposed comparison table.

    Rows are stat names; columns are [away, home] team abbreviations.
    """
    stats: dict[str, list] = {}

    for team in [away, home]:
        td = revealed[revealed["posteam"] == team]

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
        # full revealed slice, not the posteam-filtered `td`.
        pen_mask = revealed["penalty_team"] == team
        penalties = int(pen_mask.sum())
        penalty_yds = int(revealed.loc[pen_mask, "penalty_yards"].fillna(0).sum())

        col = [
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
        stats[team] = col

    index = [
        "Plays", "Pass Plays", "Rush Plays", "Rush+Comp",
        "Pass Yds", "Rush Yds", "Total Yds",
        "CMP%", "aDoT",
        "Pass EPA/play", "Rush EPA/play", "EPA/play",
        "Pass SR", "Rush SR",
        "1st Down %", "3rd Down %", "RZ TD%",
        "Turnovers", "Penalties", "Penalty Yds", "TOP",
    ]
    return pd.DataFrame(stats, index=pd.Index(index, name="Stat"))

def stat_percentiles(df: pd.DataFrame, baselines: dict) -> pd.DataFrame:
    """Percentile of each Team stats cell vs the baseline seasons, 100 = best
    (flipped for stats where fewer is better). NaN without a baseline."""
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for row in df.index:
        arr = baselines.get(row, np.array([]))
        for team in df.columns:
            pct = percentile_of(float(df.at[row, team]), arr)
            out.at[row, team] = 100.0 - pct if row in LOWER_IS_BETTER and pd.notna(pct) else pct
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


SR_SITUATIONS = [
    ("Early · Short",  (1, 2), (1,  3)),
    ("Late · Short",   (3, 4), (1,  3)),
    ("Early · Medium", (1, 2), (4,  6)),
    ("Late · Medium",  (3, 4), (4,  6)),
    ("Early · Long",   (1, 2), (7, 99)),
    ("Late · Long",    (3, 4), (7, 99)),
]
SR_METRICS = ["Pass SR%", "Pass EPA/play", "Rush SR%", "Rush EPA/play"]


def situational_success_rate(revealed: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    """Success rate and EPA/play split by down group × distance × play type."""
    def _sr(plays) -> float:
        if plays.empty:
            return float("nan")
        return (plays["epa"].fillna(0) > 0).mean() * 100

    def _epa_per_play(plays) -> float:
        if plays.empty:
            return float("nan")
        return plays["epa"].fillna(0).mean()

    rows = []
    counts: dict[tuple, dict[str, int]] = {}
    for label, downs, (d_min, d_max) in SR_SITUATIONS:
        sit_mask = (
            revealed["down"].isin(downs) &
            revealed["ydstogo"].fillna(0).between(d_min, d_max)
        )
        for team in [away, home]:
            tm = revealed[sit_mask & (revealed["posteam"] == team)]
            pass_plays = tm[(tm["pass_attempt"].fillna(0) == 1) & (tm["qb_spike"].fillna(0) == 0)]
            rush_plays = tm[(tm["rush_attempt"].fillna(0) == 1) & (tm["qb_kneel"].fillna(0) == 0)]
            counts[(label, team)] = {"Pass": len(pass_plays), "Rush": len(rush_plays)}
            rows.append({
                "Situation": label,
                "Team": team,
                "Pass SR%": _sr(pass_plays),
                "Pass EPA/play": _epa_per_play(pass_plays),
                "Rush SR%": _sr(rush_plays),
                "Rush EPA/play": _epa_per_play(rush_plays),
            })

    df = pd.DataFrame(rows)
    metrics = ["Pass SR%", "Pass EPA/play", "Rush SR%", "Rush EPA/play"]
    pivot = df.pivot(index="Situation", columns="Team", values=metrics)
    pivot.columns = [f"{team} {stat}" for stat, team in pivot.columns]
    col_order = []
    for team in [away, home]:
        for m in metrics:
            col_order.append(f"{team} {m}")
    pivot = pivot[[c for c in col_order if c in pivot.columns]]
    pivot.index.name = "Situation"
    return pivot, counts


def situational_distributions(raw: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """Per-situation metric distributions over every team-game in `raw`."""
    raw = raw[raw["posteam"].notna()]

    def _sr(plays) -> float:
        if plays.empty:
            return float("nan")
        return (plays["epa"].fillna(0) > 0).mean() * 100

    def _epa_per_play(plays) -> float:
        if plays.empty:
            return float("nan")
        return plays["epa"].fillna(0).mean()

    result: dict[str, dict[str, np.ndarray]] = {}
    for label, downs, (d_min, d_max) in SR_SITUATIONS:
        sit_mask = (
            raw["down"].isin(downs) &
            raw["ydstogo"].fillna(0).between(d_min, d_max)
        )
        sit = raw[sit_mask]

        rows = []
        for (game_id, team), grp in sit[sit["posteam"].notna()].groupby(["game_id", "posteam"]):
            pass_plays = grp[(grp["pass_attempt"].fillna(0) == 1) & (grp["qb_spike"].fillna(0) == 0)]
            rush_plays = grp[(grp["rush_attempt"].fillna(0) == 1) & (grp["qb_kneel"].fillna(0) == 0)]
            rows.append({
                "Pass SR%":      _sr(pass_plays),
                "Pass EPA/play": _epa_per_play(pass_plays),
                "Rush SR%":      _sr(rush_plays),
                "Rush EPA/play": _epa_per_play(rush_plays),
            })

        if not rows:
            result[label] = {}
            continue
        per_game = pd.DataFrame(rows)
        result[label] = {
            m: np.sort(per_game[m].dropna().values) for m in SR_METRICS
        }
    return result


# ---------- Player leaders ----------
def top_players(revealed: pd.DataFrame, team: str, kind: str, n: int = 3,
                drop: frozenset = frozenset()) -> pd.DataFrame:
    """Leaders for a team so far. `drop` names columns the data source doesn't
    have (they'd only ever read 0)."""
    td = revealed[revealed["posteam"] == team]
    if kind == "passing":
        pass_td = td[(td["pass_attempt"] == 1) & (td["qb_spike"].fillna(0) == 0)].copy()
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
        rush_td = td[(td["rush_attempt"] == 1) & (td["qb_kneel"].fillna(0) == 0)].copy()
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
        recv_td = td[
            td["receiver_player_name"].notna()
            & (td["pass_attempt"].fillna(0) == 1)
            & (td["qb_spike"].fillna(0) == 0)
        ].copy()
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

def top_defenders(revealed: pd.DataFrame, team: str, n: int = 5,
                  drop: frozenset = frozenset()) -> pd.DataFrame:
    """Defensive leaders for a team: tackles, sacks, QB hits, TFLs, INTs, PDs, FFs.
    `drop` names columns the data source doesn't have."""
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

    solo   = _count(["solo_tackle_1_player_name", "solo_tackle_2_player_name"], 1.0)
    assist = _count([
        "assist_tackle_1_player_name", "assist_tackle_2_player_name",
        "assist_tackle_3_player_name", "assist_tackle_4_player_name",
    ], 0.5)
    tackles = solo.add(assist, fill_value=0).rename("Tackles")

    sacks = _count(["sack_player_name"], 1.0).add(
        _count(["half_sack_1_player_name", "half_sack_2_player_name"], 0.5), fill_value=0
    ).rename("Sacks")

    qb_hits = _count(["qb_hit_1_player_name", "qb_hit_2_player_name"]).rename("QB Hits")
    tfls    = _count(["tackle_for_loss_1_player_name", "tackle_for_loss_2_player_name"]).rename("TFL")
    ints    = _count(["interception_player_name"]).rename("INT")
    pds     = _count(["pass_defense_1_player_name", "pass_defense_2_player_name"]).rename("PD")
    ffs     = _count(["forced_fumble_player_1_player_name", "forced_fumble_player_2_player_name"]).rename("FF")

    g = pd.concat([tackles, sacks, qb_hits, tfls, ints, pds, ffs], axis=1).fillna(0)
    g.index.name = "Player"
    g = g.reset_index()
    g = g[g["Player"].notna()]

    for col in ["Tackles", "QB Hits", "TFL", "INT", "PD", "FF"]:
        if col in g.columns:
            g[col] = g[col].round(1)
    if "Sacks" in g.columns:
        g["Sacks"] = g["Sacks"].round(1)

    # weight sacks/INTs heavily so impact players surface even with low tackle counts
    g["_sort"] = (
        g.get("Tackles", 0) + g.get("Sacks", 0) * 3
        + g.get("INT", 0) * 2 + g.get("TFL", 0) + g.get("QB Hits", 0) * 0.5
        + g.get("PD", 0) * 0.5 + g.get("FF", 0) * 1.5
    )
    g = g.drop(columns=[c for c in drop if c in g.columns])
    return g.sort_values("_sort", ascending=False).drop(columns=["_sort"]).head(n)


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
