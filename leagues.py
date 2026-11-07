"""
Where each league's games come from, behind one interface.

A `League` lists a season's games, loads one as play-by-play in the app's
layout (`replay_core.PBP_COLS`), and supplies team colors/logos/nicknames and
the play-by-play its percentile baselines are built from. The page picks one
with the sidebar's League radio and otherwise never branches on the league.

- NFL: nflverse's published nflfastR pbp, read with nflreadpy (nflverse's
  successor to nfl_data_py, which can't run on pandas 2), and for games it
  hasn't published yet ESPN's live feed (live_feed.py) with nflfastR's models
  (nflfastr_models.py).
- College football: sportsdataverse's published pbp (cfb_feed.py), and for
  games it hasn't published yet ESPN's live feed run through sportsdataverse's
  own processing (cfb_feed.live_raw).

The loaders here are cached with Streamlit; the modules they call aren't.
"""

import hashlib
import json
import os
import pickle
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import nflreadpy
import numpy as np
import pandas as pd
import requests
import streamlit as st
from nflreadpy.config import update_config

import cfb_feed
import game_summary
import live_feed
import nflfastr_models
import replay_core as core
import season_store
import scoreboard


class GameLoadError(Exception):
    """Nothing could be loaded; the message says why, for the sidebar."""


@dataclass
class LoadedGame:
    pbp: pd.DataFrame           # PBP_COLS layout, one row per play, in game order
    live_state: str             # ESPN's "pre" / "in" / "post"
    notes: list[str]            # sidebar captions about where the data comes from
    warning: str | None = None  # something the viewer should know (e.g. no models)
    # FTN has charted the game, and `pbp` carries it (core.with_charting); a
    # fact about the data, not the game, so the replay can say so up front
    charted: bool = False


@dataclass
class Availability:
    """Who's hurt and how much each player plays (replay_core's Availability
    section); empty frames where a league has none."""
    injuries: pd.DataFrame      # weekly injury reports, one row per player and week
    snaps: pd.DataFrame         # snap counts per player and game, with gsis_id
    starters: pd.Series         # gsis id → slot on the team's latest depth chart
    short_names: pd.Series      # gsis id → short name ("T.Edwards")
    roster: pd.DataFrame = field(default_factory=pd.DataFrame)  # weekly rosters: team, week, gsis_id, status
    source: str = ""            # where they come from, for captions; "" when the league has none
    as_of: str = ""             # when the injury reports were last rebuilt


NO_AVAILABILITY = Availability(pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=object),
                               pd.Series(dtype=object))


@dataclass(frozen=True)
class League:
    """What the page needs from a league. Subclasses fill in the methods."""
    name: str                   # the sidebar's League option
    key: str                    # game_summary's league key
    title: str
    first_season: int
    broadcast_minutes: float    # a typical broadcast, for the "≈ broadcast min" readout
    baseline_label: str         # what the percentile colors compare against
    group_label: str | None = None  # a sidebar filter on games["groups"], e.g. conference
    untimed_ot: bool = False    # OT has no clock: the sidebar seeds OT at its start
    run_gaps: bool = False      # the play text names a run's gap, not just its side
    espn_id_col: str = "game_id"  # the games() column with ESPN's event id
    clock: core.ClockRules = core.NFL_CLOCK  # what Game management's clock estimates follow
    season_group_label: str = "Division"  # the Season stats page's team filter
    # The day of `season` + 1 from which the season is over and won't change
    # (month, day): its Season stats are then kept on disk (`season_final`).
    season_over: tuple[int, int] = (3, 1)

    def stamp(self) -> str:
        """When the published data was last rebuilt; a cache key."""
        raise NotImplementedError

    def games(self, season: int, stamp: str) -> pd.DataFrame:
        """The season's games in display order: game_id, label, source,
        week_label (and groups, a tuple per game, if group_label is set).
        Raises GameLoadError if nothing could be loaded."""
        raise NotImplementedError

    def load_game(self, row: pd.Series, season: int, stamp: str) -> LoadedGame:
        """One game, `row` from games(). Raises GameLoadError."""
        raise NotImplementedError

    def team_meta(self, season: int, stamp: str) -> tuple[dict, dict, dict]:
        """(colors, logos, nicknames) keyed by team abbreviation."""
        raise NotImplementedError

    def team_names(self, season: int, stamp: str) -> dict[str, str]:
        """Team abbreviation → the name text outside tables uses ("Chiefs",
        "Texas Tech"); a team missing from it keeps its abbreviation."""
        raise NotImplementedError

    def baseline_pbp(self, season: int) -> pd.DataFrame | None:
        """Play-by-play (core.BASELINE_COLS) the percentile baselines for
        `season` are built from; None if unavailable."""
        raise NotImplementedError

    def missing_stats(self, pbp: pd.DataFrame) -> frozenset:
        """Player-leader columns (top_players/top_defenders `drop=`) the data
        behind `pbp` doesn't have."""
        raise NotImplementedError

    def scenario_wp(self, states: pd.DataFrame) -> np.ndarray | None:
        """The offense's win probability in made-up states (core.wp_state()
        columns, without their `_`), for Game management's what-ifs; None
        if the league has no model to run on them."""
        return None

    def season_pbp(self, season: int, stamp: str) -> pd.DataFrame:
        """Every finished game of `season` published so far (PBP_COLS), plus
        `postseason` (bool), for the Season stats page. Raises
        GameLoadError if the season isn't published. A finished season
        (`season_final`) is read from the disk cache after the first time."""
        if self.season_final(season):
            return _final_season_pbp(self.name, season)
        return self._season_pbp(season, stamp)

    def _season_pbp(self, season: int, stamp: str) -> pd.DataFrame:
        """`season_pbp` from the published data."""
        raise NotImplementedError

    def season_final(self, season: int) -> bool:
        """Whether `season` is over and its published data won't change: past
        `season_over` of the next year (the NFL's Super Bowl in February,
        college's title game in January, with a margin for the last rebuilds)."""
        month, day = self.season_over
        return date.today() >= date(season + 1, month, day)

    def season_teams(self, season: int, stamp: str) -> pd.Series:
        """The teams the Season stats page ranks, abbreviation → group
        (`season_group_label`): NFL divisions, FBS conferences."""
        raise NotImplementedError

    def upcoming(self, season: int, stamp: str) -> pd.DataFrame:
        """The season's games not played yet, for the Matchup page, in week
        and kickoff order: game_id, week, week_label, game_date, home_team,
        away_team, label, neutral, home_spread (the betting line, home team
        favored by; NaN where the data has none), total (NaN likewise) and,
        if group_label is set, groups."""
        raise NotImplementedError

    def availability(self, season: int) -> Availability:
        """The season's injury reports, snap counts and starters; empty
        (`NO_AVAILABILITY`) where the league has none."""
        return NO_AVAILABILITY

    def nextgen(self, season: int) -> dict[str, pd.DataFrame]:
        """The season's Next Gen Stats by kind ("passing", "receiving"), one
        row per player and week (week 0: the regular season), as nflverse
        publishes them; empty where the league has none (college)."""
        return {}

    def charting(self, season: int) -> pd.DataFrame:
        """FTN's charting of the season's plays, one row per play keyed by
        nflverse_game_id / nflverse_play_id (core.with_charting); empty
        where the league has none (college, NFL before
        `CHART_FIRST_SEASON`)."""
        return pd.DataFrame()

    def class_years(self, season: int, stamp: str) -> dict[tuple[str, str], str]:
        """(team, core.player_key(name)) → class year ("Fr" … "Sr") for the
        leaders tables' Yr column (core.with_class_year); empty where the
        league has none (the NFL)."""
        return {}

    def draft_board(self, season: int, stamp: str) -> dict[tuple[str, str], dict]:
        """(team, core.player_key(name)) → the leaders tables' Draft / Mock
        cells (core.draft_keys, core.with_draft) for the players on the
        next draft's boards; empty where the league has none (the NFL)."""
        return {}

    def ticker(self, row: pd.Series) -> list[scoreboard.TickerGame]:
        """The league's other games on ESPN's scoreboard around now
        (scoreboard.ticker), never `row`'s: its live score would spoil it."""
        return scoreboard.ticker(load_scoreboard(self.key), row.get(self.espn_id_col),
                                 (row["home_team"], row["away_team"]), pd.Timestamp.now(tz="UTC"))


# ---------- NFL ----------
NFLVERSE_RELEASE = "https://github.com/nflverse/nflverse-data/releases/download"

# The app decides when to download again (nflverse's timestamp.json, below):
# nflreadpy's own day-long memory cache would keep serving a season after
# nflverse republishes it. A season of pbp can take longer than its 30s default.
update_config(cache_mode="off", timeout=120)


def nflverse_teams() -> pd.DataFrame:
    """nflverse's team table: abbreviation, colors, logos, nickname."""
    return nflreadpy.load_teams().to_pandas()


def nflverse_pbp(seasons: list[int], columns: list[str]) -> pd.DataFrame:
    """nflverse play-by-play for `seasons`, `columns` only, with float64 columns
    stored as float32 to save memory, as nfl_data_py did, and the players of
    a pass or run a penalty wiped out named (`core.name_nullified()`). Raises
    if a season isn't published."""
    extra = list(core.NFLVERSE_NAME_COLS)
    read = [c for c in columns if c != "xrush_yards"]  # the app's own, added below
    df = nflreadpy.load_pbp(seasons).select(read + extra).to_pandas()
    df = core.name_nullified(df).drop(columns=extra)
    floats = df.select_dtypes("float64").columns
    df[floats] = df[floats].astype("float32")
    return with_xrush(df, "NFL") if "xrush_yards" in columns else df


XRUSH_TABLE = Path(__file__).parent / "models" / "xrush_yards.csv"


@st.cache_resource
def xrush_table(league_key: str) -> pd.DataFrame | None:
    """The league's expected rushing yards table (tools/build_xrush_table.py),
    or None if it isn't there: RYOE is then blank."""
    try:
        t = pd.read_csv(XRUSH_TABLE)
    except OSError:
        return None
    t = t[t["league"] == league_key].drop(columns="league")
    return None if t.empty else t


PASS_RATE_TABLE = Path(__file__).parent / "models" / "pass_rate.csv"


@st.cache_resource
def pass_rate_table(league_key: str) -> pd.DataFrame | None:
    """How often the league's offenses pass by down, distance, score margin and
    clock (tools/build_pass_rate_table.py), or None if it isn't there: Game
    management's scenarios then run on every down."""
    try:
        t = pd.read_csv(PASS_RATE_TABLE)
    except OSError:
        return None
    t = t[t["league"] == league_key].drop(columns="league")
    return None if t.empty else t.astype({k: float for k in core.PASS_KEYS})


def with_xrush(pbp: pd.DataFrame, league_key: str) -> pd.DataFrame:
    """`pbp` with each designed run's expected yards (`xrush_yards`)."""
    return core.add_expected_rush(pbp, xrush_table(league_key))


@st.cache_data(ttl=3600)
def load_team_colors() -> dict[str, str]:
    """Map team abbreviation → primary hex color."""
    try:
        df = nflverse_teams()
    except Exception:  # network error / upstream file moved: fall back to defaults
        return {}
    return dict(zip(df["team_abbr"], df["team_color"]))


@st.cache_data(ttl=3600)
def load_team_logos() -> dict[str, str]:
    """Map team abbreviation → ESPN logo URL."""
    try:
        df = nflverse_teams()
    except Exception:
        return {}
    col = "team_logo_espn" if "team_logo_espn" in df.columns else "team_logo_wikipedia"
    if col not in df.columns:
        return {}
    return {a: u for a, u in zip(df["team_abbr"], df[col]) if isinstance(u, str)}


@st.cache_data(ttl=3600)
def load_team_nicknames() -> dict[str, str]:
    """Map team abbreviation → nickname (e.g. KC → Chiefs)."""
    try:
        df = nflverse_teams()
    except Exception:
        return {}
    if "team_nick" not in df.columns:
        return {}
    return {a: n for a, n in zip(df["team_abbr"], df["team_nick"]) if isinstance(n, str)}


@st.cache_data(ttl=60)
def nflverse_stamp(release: str = "pbp") -> str:
    """When nflverse last rebuilt a release's files (pbp, injuries, …).
    Checked every minute, so a newly published game shows up right away
    instead of on a fixed timer."""
    try:
        r = requests.get(f"{NFLVERSE_RELEASE}/{release}/timestamp.json", timeout=10)
        r.raise_for_status()
        return str(r.json().get("last_updated", ""))
    except Exception:
        # Unknown: fall back to refreshing every 10 minutes.
        return f"unknown-{int(time.time() // 600)}"


@st.cache_data(max_entries=4)
def load_pbp(season: int, stamp: str) -> pd.DataFrame:
    """Load nflverse play-by-play for a season. `stamp` is only a cache key:
    the file is downloaded again when nflverse republishes it."""
    try:
        df = nflverse_pbp([season], core.PBP_COLS)
    except Exception as e:
        raise ValueError(
            f"No play-by-play data is available for the {season} season yet. "
            "nflverse publishes it once games have been played."
        ) from e
    if df.empty:
        raise ValueError(f"No play-by-play data is available for the {season} season yet.")
    return df


@st.cache_data(ttl=3600)
def load_schedule(season: int) -> pd.DataFrame:
    """nflverse's schedule: ESPN event id, spread line, roof, kickoff time."""
    try:
        sched = nflreadpy.load_schedules([season]).to_pandas()
    except Exception:
        return pd.DataFrame()
    return sched[sched["season"] == season].reset_index(drop=True)


def list_games(pbp: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """One row per game with date, teams, game_id and data source.

    source is "nflfastR" once nflverse has published the game's pbp, and
    "live" for a game that has kicked off (or is about to) but isn't
    published yet — those are read from ESPN's play feed.
    """
    published = set(pbp["game_id"].unique()) if not pbp.empty else set()
    if not sched.empty:
        g = sched.copy()
        g["game_date"] = g["gameday"]
        today = date.today().isoformat()
        g = g[(g["game_id"].isin(published)) | ((g["gameday"] <= today) & g["espn"].notna())]
        g["source"] = np.where(g["game_id"].isin(published), "nflfastR", "live")
        g = g.sort_values(["week", "gameday", "gametime"])
    elif not pbp.empty:
        g = (
            pbp.groupby("game_id", as_index=False)
            .agg(week=("week", "first"),
                 game_date=("game_date", "first"),
                 home_team=("home_team", "first"),
                 away_team=("away_team", "first"))
            .sort_values(["week", "game_date"])
        )
        g["source"] = "nflfastR"
    else:
        return pd.DataFrame(columns=["game_id", "label", "source", "week_label"])
    g["week_label"] = "Week " + g["week"].astype(int).astype(str)
    g["label"] = g.apply(
        lambda r: f"{r['away_team']} @ {r['home_team']} ({r['game_date']})"
                  + (" 🔴 live" if r["source"] == "live" else ""),
        axis=1,
    )
    return g.reset_index(drop=True)


def upcoming_games(sched: pd.DataFrame, published: set) -> pd.DataFrame:
    """The schedule's games without a result that aren't in the published
    pbp, with `League.upcoming`'s columns. nflverse's spread_line is how
    many points the home team is favored by."""
    if sched.empty:
        return pd.DataFrame(columns=["game_id", "week_label", "home_team", "away_team", "label"])
    g = sched[sched["result"].isna() & ~sched["game_id"].isin(published)]
    g = g.sort_values(["week", "gameday", "gametime"]).reset_index(drop=True)
    day = pd.to_datetime(g["gameday"]).dt.strftime("%a")
    return g.assign(
        week_label="Week " + g["week"].astype(int).astype(str), game_date=g["gameday"],
        neutral=g["location"].eq("Neutral"), home_spread=g["spread_line"], total=g["total_line"],
        label=[f"{a} @ {h} ({d} {gd})" for a, h, d, gd in zip(g["away_team"], g["home_team"], day,
                                                                 g["gameday"])])


@st.cache_resource
def load_nflfastr_models():
    """nflfastR's EP/WP boosters + FG table, or the error that stopped them."""
    try:
        return nflfastr_models.load_models(), nflfastr_models.load_fg_table(), None
    except Exception as e:  # no network to fastrmodels, xgboost missing, ...
        return None, None, str(e)


@st.cache_data(ttl=20)
def load_live_game(sched_row: dict) -> tuple[pd.DataFrame, str, str, str | None]:
    """A not-yet-published game from ESPN's play feed, in nflfastR's layout,
    with EP/EPA/WP from nflfastR's models. Returns (pbp, espn state,
    fetched-at clock, model error). Cached 20s, so reruns don't hammer ESPN."""
    summary = live_feed.fetch_summary(int(sched_row["espn"]))
    models, fg_table, model_err = load_nflfastr_models()
    df = with_xrush(live_feed.build_live_pbp(summary, sched_row, models, fg_table, core.PBP_COLS), "NFL")
    return df, live_feed.game_state(summary), datetime.now().strftime("%H:%M:%S"), model_err


@st.cache_data(max_entries=4)
def _nfl_games(season: int, stamp: str, today: str) -> tuple[pd.DataFrame, str | None]:
    """The game list and, if the season's pbp didn't load, why. `today` is a
    cache key: live games are listed from their kickoff date."""
    try:
        pbp, err = load_pbp(season, stamp), None
    except Exception as e:
        # Nothing published for this season yet: live games can still load.
        pbp, err = pd.DataFrame(columns=core.PBP_COLS), str(e)
    return list_games(pbp, load_schedule(season)), err


@st.cache_data(max_entries=8)
def _nfl_game(season: int, stamp: str, game_id: str) -> pd.DataFrame:
    pbp = load_pbp(season, stamp)
    return pbp[pbp["game_id"] == game_id].sort_values("play_id").reset_index(drop=True)


@st.cache_data(ttl=3600)
def load_team_divisions() -> dict[str, str]:
    """Map team abbreviation → division (e.g. KC → AFC West)."""
    try:
        df = nflverse_teams()
    except Exception:
        return {}
    return {a: d for a, d in zip(df["team_abbr"], df["team_division"]) if isinstance(d, str)}


@st.cache_data(max_entries=2)
def _nfl_season_pbp(season: int, stamp: str) -> pd.DataFrame:
    pbp = load_pbp(season, stamp)
    sched = load_schedule(season)
    if not sched.empty and "game_type" in sched.columns:
        post = set(sched.loc[sched["game_type"] != "REG", "game_id"])
        return pbp.assign(postseason=pbp["game_id"].isin(post))
    # No schedule: the regular season ran 16 weeks to 2020, 17 games since.
    return pbp.assign(postseason=pbp["week"] > (17 if season <= 2020 else 18))


@st.cache_data(ttl=86400, max_entries=1)
def _nfl_baseline_pbp(season: int) -> pd.DataFrame | None:
    """The 3 NFL seasons before `season`."""
    prior = [s for s in [season - 3, season - 2, season - 1] if s >= 1999]
    if not prior:
        return None
    df = nflverse_pbp(prior, core.BASELINE_COLS)
    return None if df.empty else df


# Injury reports, snap counts and depth charts (replay_core's Availability).
# Each nflverse release has its own timestamp.json, so each file is downloaded
# again only when it's rebuilt; one that can't be read is empty (and cached as
# such until its stamp changes), and the pages say there's no report.
@st.cache_data(ttl=86400)
def _players() -> pd.DataFrame:
    """nflverse's players: gsis id, PFR id and short name ("D.Maye")."""
    try:
        return nflreadpy.load_players().select(["gsis_id", "pfr_id", "short_name"]).to_pandas()
    except Exception:
        return pd.DataFrame(columns=["gsis_id", "pfr_id", "short_name"])


@st.cache_data(max_entries=2)
def load_injuries(season: int, stamp: str) -> pd.DataFrame:
    """The NFL's weekly injury reports for `season`, as nflverse publishes them."""
    try:
        return nflreadpy.load_injuries([season]).to_pandas()
    except Exception:
        return pd.DataFrame()


@st.cache_data(max_entries=2)
def load_snap_counts(season: int, stamp: str) -> pd.DataFrame:
    """PFR's snap counts per player and game, with gsis_id (through nflverse's
    players table; NaN for a player it doesn't link)."""
    try:
        snaps = nflreadpy.load_snap_counts([season]).to_pandas()
    except Exception:
        return pd.DataFrame()
    p = _players().dropna(subset=["pfr_id", "gsis_id"]).drop_duplicates("pfr_id")
    return snaps.assign(gsis_id=snaps["pfr_player_id"].map(p.set_index("pfr_id")["gsis_id"]))


@st.cache_data(max_entries=4)
def load_nextgen(season: int, kind: str, stamp: str) -> pd.DataFrame:
    """NGS's `kind` stats ("passing", "rushing", "receiving") for `season`,
    one row per player and week, from 2016; empty if nflverse has none."""
    try:
        return nflreadpy.load_nextgen_stats([season], kind).to_pandas()
    except Exception:
        return pd.DataFrame()


CHART_FIRST_SEASON = 2022  # FTN's charting starts in 2022


@st.cache_data(max_entries=5)  # the season shown plus the replay baseline's three
def load_charting(season: int, stamp: str) -> pd.DataFrame:
    """FTN's charting of `season` (nflverse's ftn_charting release, weekly
    during the season), one row per play; empty if nflverse has none."""
    try:
        return nflreadpy.load_ftn_charting([season]).to_pandas()
    except Exception:
        return pd.DataFrame()


@st.cache_data(max_entries=2)
def load_roster_status(season: int, stamp: str) -> pd.DataFrame:
    """Each team's weekly roster: team, week, gsis_id and status (ACT,
    RES for a reserve list, CUT, …), for who's gone that the injury report
    doesn't list (replay_core.lineup_change)."""
    try:
        return nflreadpy.load_rosters_weekly([season]).select(["team", "week", "gsis_id", "status"]).to_pandas()
    except Exception:
        return pd.DataFrame(columns=["team", "week", "gsis_id", "status"])


@st.cache_data(max_entries=2)
def load_starters(season: int, stamp: str) -> pd.Series:
    """gsis id → slot ("LT", "MLB", "RCB") for each team's first-team players
    on its latest depth chart. Since 2025 nflverse keeps ESPN's daily charts
    (dt, pos_abb, pos_rank); before, one per week (week, depth_team,
    depth_position)."""
    try:
        dc = nflreadpy.load_depth_charts([season]).to_pandas()
    except Exception:
        return pd.Series(dtype=object)
    if {"dt", "pos_rank", "pos_abb"} <= set(dc.columns):
        first = dc[(dc["dt"] == dc.groupby("team")["dt"].transform("max")) & (dc["pos_rank"] == 1)]
        slot = first["pos_abb"]
    elif {"week", "depth_team", "depth_position"} <= set(dc.columns):
        team = "club_code" if "club_code" in dc.columns else "team"
        first = dc[(dc["week"] == dc.groupby(team)["week"].transform("max"))
                   & (dc["depth_team"].astype(str) == "1")]
        slot = first["depth_position"]
    else:
        return pd.Series(dtype=object)
    return pd.Series(slot.to_numpy(), index=first["gsis_id"].to_numpy(), dtype=object).loc[
        lambda s: ~s.index.duplicated() & pd.notna(s.index)]


class NFLLeague(League):
    def stamp(self) -> str:
        return nflverse_stamp()

    def games(self, season: int, stamp: str) -> pd.DataFrame:
        g, err = _nfl_games(season, stamp, date.today().isoformat())
        if g.empty and err:
            raise GameLoadError(f"Could not load pbp: {err}")
        return g

    def load_game(self, row: pd.Series, season: int, stamp: str) -> LoadedGame:
        if row["source"] == "nflfastR":
            pbp = _nfl_game(season, stamp, row["game_id"])
            # FTN's charting, once it has charted the game (a day or two after it).
            ftn = self.charting(season)
            ftn = (ftn[ftn["nflverse_game_id"] == row["game_id"]]
                   if not ftn.empty and "nflverse_game_id" in ftn.columns else ftn.iloc[0:0])
            return LoadedGame(core.with_charting(pbp, ftn) if not ftn.empty else pbp, "post",
                              [f"📊 Official nflfastR play-by-play · nflverse data as of {stamp}"],
                              charted=not ftn.empty)
        # Not published by nflverse yet: ESPN's live play feed, run through
        # nflfastR's own EP/WP models. Switches to the official data by
        # itself once nflverse publishes the game.
        sched_row = {k: (None if pd.isna(v) else v) for k, v in row.items()
                     if k in ("game_id", "season", "game_type", "week", "gameday", "home_team",
                              "away_team", "espn", "spread_line", "roof", "location")}
        try:
            pbp, state, fetched_at, model_err = load_live_game(sched_row)
        except Exception as e:
            raise GameLoadError(f"Could not load the live play feed: {e}") from e
        return LoadedGame(
            pbp, state,
            [f"🔴 Live · ESPN play feed + nflfastR models · updated {fetched_at}. "
             "Switches to official nflfastR data once nflverse publishes it."],
            warning=(f"nflfastR's models couldn't load ({model_err}); "
                     "EPA and win probability are blank until they do.") if model_err else None)

    def team_meta(self, season: int, stamp: str) -> tuple[dict, dict, dict]:
        return load_team_colors(), load_team_logos(), load_team_nicknames()

    def team_names(self, season: int, stamp: str) -> dict[str, str]:
        return load_team_nicknames()

    def baseline_pbp(self, season: int) -> pd.DataFrame | None:
        return _nfl_baseline_pbp(season)

    def missing_stats(self, pbp: pd.DataFrame) -> frozenset:
        return frozenset({"Hurries"})  # nflfastR has QB hits instead

    def _season_pbp(self, season: int, stamp: str) -> pd.DataFrame:
        try:
            return _nfl_season_pbp(season, stamp)
        except ValueError as e:
            raise GameLoadError(str(e)) from e

    def season_teams(self, season: int, stamp: str) -> pd.Series:
        divisions = load_team_divisions()
        teams = sorted(self.season_pbp(season, stamp)["posteam"].dropna().unique())
        return pd.Series({t: divisions.get(t, "") for t in teams}, dtype=object)

    def upcoming(self, season: int, stamp: str) -> pd.DataFrame:
        try:
            published = set(load_pbp(season, stamp)["game_id"])
        except ValueError:  # nothing published yet: every game is still to come
            published = set()
        return upcoming_games(load_schedule(season), published)

    def availability(self, season: int) -> Availability:
        injuries_stamp = nflverse_stamp("injuries")
        p = _players().dropna(subset=["gsis_id", "short_name"]).drop_duplicates("gsis_id")
        return Availability(
            injuries=load_injuries(season, injuries_stamp),
            snaps=load_snap_counts(season, nflverse_stamp("snap_counts")),
            starters=load_starters(season, nflverse_stamp("depth_charts")),
            short_names=p.set_index("gsis_id")["short_name"],
            roster=load_roster_status(season, nflverse_stamp("weekly_rosters")),
            source="the NFL's injury reports, PFR snap counts and ESPN depth charts, via nflverse",
            as_of="" if injuries_stamp.startswith("unknown") else injuries_stamp)

    def nextgen(self, season: int) -> dict[str, pd.DataFrame]:
        if season < 2016:  # NGS starts in 2016
            return {}
        stamp = nflverse_stamp("nextgen_stats")
        return {kind: load_nextgen(season, kind, stamp) for kind in core.NGS_COLS}

    def charting(self, season: int) -> pd.DataFrame:
        if season < CHART_FIRST_SEASON:
            return pd.DataFrame()
        return load_charting(season, nflverse_stamp("ftn_charting"))

    def scenario_wp(self, states: pd.DataFrame) -> np.ndarray | None:
        models, _, _ = load_nflfastr_models()
        if models is None or states.empty:
            return None
        return nflfastr_models.wp_for_states(states, models)


# ---------- College football ----------
# Published games only, from sportsdataverse (see cfb_feed). Like nflverse, it
# is rebuilt about once a day; a game shows up the morning after it's played.
# Published games only, from sportsdataverse (see cfb_feed). Like nflverse, it
# is rebuilt about once a day; a game shows up the morning after it's played.
@st.cache_data(ttl=60)
def cfb_stamp() -> str:
    """When the college pbp was last rebuilt. Checked every minute."""
    return cfb_feed.stamp() or f"unknown-{int(time.time() // 600)}"


@st.cache_resource(max_entries=2)
def load_cfb_season(season: int, stamp: str) -> pd.DataFrame:
    """A season of college pbp, raw release columns. Shared, not copied, so
    callers must not modify it. `stamp` is only a cache key."""
    return cfb_feed.read_season(season)


@st.cache_data(ttl=3600)
def load_cfb_schedule(season: int) -> pd.DataFrame:
    return cfb_feed.read_schedule(season)


def _cfb_season_or_empty(season: int, stamp: str) -> tuple[pd.DataFrame, str | None]:
    """The published season, or an empty one (and why) if it isn't out yet:
    its live games can still be listed from the schedule."""
    try:
        return load_cfb_season(season, stamp), None
    except Exception as e:
        return pd.DataFrame(columns=cfb_feed.RAW_COLS), str(e)


@st.cache_data(max_entries=4)
def list_cfb_games(season: int, stamp: str, today: str) -> tuple[pd.DataFrame, str | None]:
    """The game list and, if the season's pbp didn't load, why. `today` (US
    Eastern) is a cache key: live games are listed from their kickoff date."""
    raw, err = _cfb_season_or_empty(season, stamp)
    return cfb_feed.list_games(raw, load_cfb_schedule(season), today), err


@st.cache_data(max_entries=2)
def list_cfb_upcoming(season: int, stamp: str, today: str) -> pd.DataFrame:
    """Games not played yet (cfb_feed.upcoming_games). `today` (US Eastern)
    is a cache key."""
    raw, _ = _cfb_season_or_empty(season, stamp)
    return cfb_feed.upcoming_games(raw, load_cfb_schedule(season), today)


@st.cache_data(max_entries=8)
def load_cfb_game(season: int, stamp: str, game_id: str) -> pd.DataFrame:
    raw = load_cfb_season(season, stamp)
    return with_xrush(cfb_feed.to_pbp(raw[raw["game_id"].astype(str) == game_id], core.PBP_COLS), "CFB")


@st.cache_data(ttl=20)
def load_cfb_live_game(game_id: str) -> tuple[pd.DataFrame, str, str]:
    """A not-yet-published college game from ESPN's feed, through
    sportsdataverse's processing, in the app's layout. Returns (pbp, ESPN
    state, fetched-at clock). Cached 20s, so reruns don't hammer ESPN."""
    summary = cfb_feed.fetch_summary(game_id)
    pbp = with_xrush(cfb_feed.to_pbp(cfb_feed.live_raw(summary, game_id), core.PBP_COLS), "CFB")
    return pbp, cfb_feed.game_state(summary), datetime.now().strftime("%H:%M:%S")


@st.cache_data(ttl=3600)
def load_cfb_team_meta(season: int, stamp: str) -> tuple[dict, dict, dict]:
    """(colors, logos, nicknames) keyed by team abbreviation."""
    return cfb_feed.team_meta(_cfb_season_or_empty(season, stamp)[0],
                              cfb_feed.read_team_info(season), load_cfb_schedule(season))


@st.cache_data(ttl=3600)
def load_cfb_team_names(season: int, stamp: str) -> dict[str, str]:
    """Abbreviation → school name."""
    return cfb_feed.team_names(_cfb_season_or_empty(season, stamp)[0],
                               cfb_feed.read_team_info(season), load_cfb_schedule(season))


@st.cache_data(ttl=86400, max_entries=2, show_spinner=False)
def load_cfb_class_years(season: int, stamp: str) -> dict[tuple[str, str], str]:
    """The season's rosters as core.class_year_keys(); empty if unavailable."""
    return core.class_year_keys(cfb_feed.read_rosters(season, load_cfb_team_names(season, stamp)))


@st.cache_data(ttl=3600, show_spinner=False)
def load_cfb_team_ids(season: int, stamp: str) -> pd.DataFrame:
    """(abbr, id): ESPN's team id → the abbreviation the pbp uses."""
    return cfb_feed.team_ids(_cfb_season_or_empty(season, stamp)[0], load_cfb_schedule(season),
                             cfb_feed.read_team_info(season))


@st.cache_data(ttl=3600, max_entries=2, show_spinner=False)
def load_cfb_draft_board(season: int, stamp: str) -> dict[tuple[str, str], str]:
    """The draft after `season` (the 2026 season's players go in the 2027
    draft): its latest snapshot in models/draft_boards.csv (ESPN's big board,
    Tankathon's board and mock draft; tools/build_draft_board.py adds one a
    week), as core.draft_keys(). No network calls; empty if the archive has
    no snapshot of that draft."""
    board = cfb_feed.read_draft_archive(int(season) + 1)
    if board.empty:
        return {}
    board = cfb_feed.board_teams(board, load_cfb_team_ids(season, stamp),
                                 load_cfb_team_names(season, stamp))
    return core.draft_keys(board, load_cfb_class_years(season, stamp))


@st.cache_data(max_entries=2)
def _cfb_season_pbp(season: int, stamp: str) -> pd.DataFrame:
    """The season's finished games with an FBS team in them, mapped to the
    app's layout, with `postseason`."""
    raw = load_cfb_season(season, stamp)
    done = raw.groupby("game_id")["status_type_completed"].transform(lambda s: s.eq(True).all())
    raw = raw[done]
    fbs = set(_cfb_season_teams(season, stamp).index)
    raw = raw[raw["homeTeamAbbrev"].isin(fbs) | raw["awayTeamAbbrev"].isin(fbs)]
    post = set(raw.loc[raw["seasonType"] == 3, "game_id"].astype(str))
    pbp = with_xrush(cfb_feed.to_pbp(raw, core.PBP_COLS), "CFB")
    return pbp.assign(postseason=pbp["game_id"].astype(str).isin(post))


@st.cache_data(max_entries=2)
def _cfb_season_teams(season: int, stamp: str) -> pd.Series:
    """FBS teams that have played a published game, abbreviation → conference."""
    games, _ = list_cfb_games(season, stamp, "")
    sched = load_cfb_schedule(season)
    if games.empty:
        return pd.Series(dtype=object)
    games = games[games["source"] == "published"]
    division = {}
    if not sched.empty and {"home_division", "away_division"} <= set(sched.columns):
        s = sched.assign(game_id=sched["game_id"].astype(str)).set_index("game_id")
        division = {"home": games["game_id"].map(s["home_division"]),
                    "away": games["game_id"].map(s["away_division"])}
    out = {}
    for side in ("home", "away"):
        fbs = division[side].eq("fbs") if division else pd.Series(True, index=games.index)
        for team, conf in zip(games.loc[fbs, f"{side}_team"], games.loc[fbs, f"{side}_conf"]):
            if isinstance(team, str) and team:
                out[team] = conf if isinstance(conf, str) else ""
    return pd.Series(out, dtype=object).sort_index()


@st.cache_data(ttl=86400, max_entries=1)
def _cfb_baseline_pbp(season: int) -> pd.DataFrame:
    """The college season before `season`, FBS-vs-FBS games only, mapped to
    the app's layout. Empty if it isn't published."""
    prior = season - 1
    if prior < cfb_feed.FIRST_SEASON:
        return pd.DataFrame(columns=core.BASELINE_COLS)
    pbp = with_xrush(cfb_feed.to_pbp(cfb_feed.read_season(prior), core.PBP_COLS), "CFB")
    fbs = cfb_feed.fbs_games(cfb_feed.read_schedule(prior))
    if fbs:
        pbp = pbp[pbp["game_id"].isin(fbs)]
    return pbp[core.BASELINE_COLS].reset_index(drop=True)


class CollegeLeague(League):
    def stamp(self) -> str:
        return cfb_stamp()

    def games(self, season: int, stamp: str) -> pd.DataFrame:
        today = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
        g, err = list_cfb_games(season, stamp, today)
        if g.empty and err:
            raise GameLoadError(f"No college play-by-play for the {season} season yet ({err}).")
        return g.assign(groups=list(zip(g["home_conf"], g["away_conf"])))

    def load_game(self, row: pd.Series, season: int, stamp: str) -> LoadedGame:
        if row["source"] == "published":
            return LoadedGame(load_cfb_game(season, stamp, row["game_id"]), "post", [
                f"📊 College play-by-play from ESPN, with sportsdataverse's EPA and "
                f"win probability models · data as of {stamp}"])
        # Not published yet: ESPN's live feed through sportsdataverse's own
        # processing. Switches to the published data by itself once it's out.
        try:
            pbp, state, fetched_at = load_cfb_live_game(row["game_id"])
        except ImportError as e:
            raise GameLoadError("Live college games need the sportsdataverse package "
                                "(pip install -r requirements.txt).") from e
        except Exception as e:
            raise GameLoadError(f"Could not load the live play feed: {e}") from e
        return LoadedGame(pbp, state, [
            f"🔴 Live · ESPN play feed + sportsdataverse's college models · updated {fetched_at}. "
            "Switches to the published data once sportsdataverse publishes it."])

    def team_meta(self, season: int, stamp: str) -> tuple[dict, dict, dict]:
        return load_cfb_team_meta(season, stamp)

    def team_names(self, season: int, stamp: str) -> dict[str, str]:
        return load_cfb_team_names(season, stamp)

    def baseline_pbp(self, season: int) -> pd.DataFrame | None:
        df = _cfb_baseline_pbp(season)
        return None if df.empty else df

    def missing_stats(self, pbp: pd.DataFrame) -> frozenset:
        return cfb_feed.missing_leader_stats(pbp)

    def class_years(self, season: int, stamp: str) -> dict[tuple[str, str], str]:
        return load_cfb_class_years(season, stamp)

    def draft_board(self, season: int, stamp: str) -> dict[tuple[str, str], dict]:
        return load_cfb_draft_board(season, stamp)

    def _season_pbp(self, season: int, stamp: str) -> pd.DataFrame:
        try:
            return _cfb_season_pbp(season, stamp)
        except Exception as e:
            raise GameLoadError(f"No college play-by-play for the {season} season yet ({e}).") from e

    def season_teams(self, season: int, stamp: str) -> pd.Series:
        return _cfb_season_teams(season, stamp)

    def upcoming(self, season: int, stamp: str) -> pd.DataFrame:
        today = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
        g = list_cfb_upcoming(season, stamp, today)
        if g.empty:
            return g
        # The college schedule has no betting lines.
        return g.assign(groups=list(zip(g["home_conf"], g["away_conf"])), home_spread=np.nan,
                        total=np.nan)


NFL = NFLLeague(
    name="NFL", key="NFL", title="🏈 NFL Tape-Delay Replay", first_season=1999,
    # A real NFL broadcast runs ~3h10m for 60 minutes of game clock.
    broadcast_minutes=190.0, baseline_label="last 3 seasons", run_gaps=True,
    espn_id_col="espn")
COLLEGE = CollegeLeague(
    name="College football", key="CFB", title="🏈 College Football Tape-Delay Replay",
    first_season=cfb_feed.FIRST_SEASON,
    # ~3h20m. One college season has about as many team-games as three NFL ones.
    broadcast_minutes=200.0, baseline_label="last season's FBS-vs-FBS games",
    group_label="Conference", untimed_ot=True, clock=core.CFB_CLOCK,
    season_group_label="Conference", season_over=(2, 1))
LEAGUES = {lg.name: lg for lg in (NFL, COLLEGE)}


# ---------- Other games ----------
@st.cache_data(ttl=20)
def load_scoreboard(league_key: str) -> list[scoreboard.TickerGame]:
    """The league's games on ESPN's scoreboard (scoreboard.games), or none
    if it can't be read: the ticker then just doesn't show. Cached 20s like
    the live play feeds, and shared by every viewer."""
    try:
        return scoreboard.games(league_key)
    except Exception:
        return []


# ---------- Percentile baselines ----------
# Failures are cached too (as {}), so a missing season or a dropped network
# doesn't make every auto-refresh rerun retry the download.
@st.cache_data(ttl=86400)
def _stat_baselines(league: str, season: int) -> dict[str, np.ndarray]:
    try:
        raw = LEAGUES[league].baseline_pbp(season)
    except Exception:  # network error, missing season data, etc.
        return {}
    return {} if raw is None else core.stat_distributions(raw)


@st.cache_data(ttl=86400)
def _situational_baselines(league: str, season: int) -> dict[str, dict[str, np.ndarray]]:
    try:
        raw = LEAGUES[league].baseline_pbp(season)
    except Exception:
        return {}
    return {} if raw is None else core.situational_distributions(raw)


@st.cache_data(ttl=86400)
def _series_baselines(league: str, season: int) -> dict[str, dict]:
    try:
        raw = LEAGUES[league].baseline_pbp(season)
    except Exception:
        return {}
    return {} if raw is None else {"series": core.series_distributions(raw),
                                   "splits": core.play_split_distributions(raw)}


@st.cache_data(ttl=86400)
def _line_baselines(league: str, season: int) -> dict[str, np.ndarray]:
    try:
        raw = LEAGUES[league].baseline_pbp(season)
    except Exception:
        return {}
    return {} if raw is None else core.line_play_distributions(raw)


@st.cache_data(ttl=86400)
def _player_baselines(league: str, season: int) -> dict[str, dict[str, np.ndarray]]:
    try:
        raw = LEAGUES[league].baseline_pbp(season)
    except Exception:
        return {}
    return {} if raw is None else core.player_distributions(raw)


@st.cache_data(ttl=86400)
def _tendency_baselines(league: str, season: int) -> dict[str, dict[str, pd.DataFrame]]:
    try:
        raw = LEAGUES[league].baseline_pbp(season)
    except Exception:
        return {}
    return {} if raw is None else core.tendency_baselines(raw, LEAGUES[league].run_gaps)


def tendency_baselines(league: League, season: int) -> dict[str, dict[str, pd.DataFrame]]:
    """League-wide tendencies for `season`'s baseline, by
    `core.TENDENCY_DOWNS` option (see `core.tendency_baselines`)."""
    return _tendency_baselines(league.name, season)


def player_baselines(league: League, season: int) -> dict[str, dict[str, np.ndarray]]:
    """Player leaders percentile baselines for `season`, by table kind and stat."""
    return _player_baselines(league.name, season)


@st.cache_data(ttl=86400)
def _chart_baselines(league: str, season: int) -> dict[str, np.ndarray]:
    lg = LEAGUES[league]
    try:
        raw = lg.baseline_pbp(season)
    except Exception:
        return {}
    if raw is None or raw.empty or "play_id" not in raw.columns:
        return {}
    # FTN's charting of the baseline seasons it has (2022 on), from game ids' seasons.
    seasons = sorted({int(g[:4]) for g in raw["game_id"].astype(str).unique() if g[:4].isdigit()})
    ftn = [c for c in (lg.charting(s) for s in seasons) if not c.empty]
    raw = core.with_charting(raw, pd.concat(ftn, ignore_index=True)) if ftn else raw
    return core.chart_distributions(raw)


def chart_baselines(league: League, season: int) -> dict[str, np.ndarray]:
    """The replay's Charting baselines for `season`: each number over the
    baseline seasons' team-games (FTN's where it has charted them), by stat."""
    return _chart_baselines(league.name, season)


def line_baselines(league: League, season: int) -> dict[str, np.ndarray]:
    """Line play percentile baselines for `season`, by stat."""
    return _line_baselines(league.name, season)


def series_baselines(league: League, season: int) -> dict[str, dict]:
    """Series success ("series": SCR by `core.SERIES_ROWS` row) and play-type
    split ("splits": by `core.PLAY_SPLITS` row and rate) baselines for
    `season`, over team-games."""
    return _series_baselines(league.name, season)


def stat_baselines(league: League, season: int) -> dict[str, np.ndarray]:
    """Team stats percentile baselines for `season`, by stat."""
    return _stat_baselines(league.name, season)


def situational_baselines(league: League, season: int) -> dict[str, dict[str, np.ndarray]]:
    """Situational success-rate baselines for `season`, by situation and metric."""
    return _situational_baselines(league.name, season)


# ---------- Disk cache (finished seasons) ----------
# A finished season's published data doesn't change, so its play-by-play and
# Season stats tables are computed once and kept on disk: a restart, or the
# server sleeping, doesn't mean downloading and computing them again. Files
# are keyed by what they were built from plus a hash of the code that builds
# them, so a change to the stat logic starts over instead of serving old
# numbers. The oldest-used files go once the folder passes its size cap.
SEASON_CACHE_DIR = Path(os.environ.get("SEASON_CACHE_DIR",
                                       nflfastr_models.CACHE_DIR / "seasons"))
SEASON_CACHE_MAX_BYTES = int(float(os.environ.get("SEASON_CACHE_MAX_GB", "2")) * 1e9)
# Bump to drop every saved season (e.g. after nflverse republishes old seasons).
SEASON_CACHE_EPOCH = 1
_CODE_FILES = ("replay_core.py", "leagues.py", "cfb_feed.py", "live_feed.py",
               "nflfastr_models.py", "season_store.py")


def _code_version() -> str:
    """A hash of the modules the cached data is built with."""
    h = hashlib.sha256(str(SEASON_CACHE_EPOCH).encode())
    here = Path(__file__).resolve().parent
    for name in _CODE_FILES:
        try:
            h.update((here / name).read_bytes())
        except OSError:
            h.update(name.encode())
    return h.hexdigest()[:12]


_CODE_VERSION = _code_version()


def _disk_path(kind: str, key: tuple) -> Path:
    digest = hashlib.sha256(repr((key, _CODE_VERSION)).encode()).hexdigest()[:24]
    return SEASON_CACHE_DIR / f"{kind}_{key[0].replace(' ', '_')}_{key[1]}_{digest}.pkl"


def disk_cached(kind: str, key: tuple, build):
    """`build()`'s result, saved under (`kind`, `key`, the code's hash) the
    first time and read back after that. `key` starts (league name, season).
    A file that can't be read is built again; a cache that can't be written
    is skipped, so the page works the same, only slower."""
    path = _disk_path(kind, key)
    try:
        with path.open("rb") as f:
            value = pickle.load(f)
        os.utime(path)  # most recently used: pruned last
        return value
    except FileNotFoundError:
        pass
    except Exception:  # truncated or from another pandas: build it again
        path.unlink(missing_ok=True)
    value = build()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        with tmp.open("wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)  # whole files only, for other sessions reading it
        _prune_disk_cache()
    except OSError:
        pass
    return value


def _prune_disk_cache() -> None:
    """Delete the least recently used files past `SEASON_CACHE_MAX_BYTES`
    (files from an older version of the code are never read again, so they
    go first)."""
    stats = []
    for p in SEASON_CACHE_DIR.glob("*.pkl"):
        try:
            stats.append((p.stat().st_mtime, p.stat().st_size, p))
        except OSError:
            continue
    total = sum(size for _, size, _ in stats)
    for _, size, p in sorted(stats):
        if total <= SEASON_CACHE_MAX_BYTES:
            break
        p.unlink(missing_ok=True)
        total -= size


# ---------- Season store (precomputed finished seasons) ----------
# A fresh deploy has an empty disk, so finished seasons' common selections are
# also published precomputed (season_store.py; tools/build_season_store.py and
# .github/workflows/season_store.yml build them) as release assets of this
# public repo. A file built by other code than the app's is ignored.
SEASON_STORE_URL = os.environ.get(
    "SEASON_STORE_URL",
    "https://github.com/bullaspc/nfl_replays_pbp/releases/download/season-store")
# The leagues and first seasons the workflow publishes.
SEASON_STORE_FIRST = {"NFL": 2016}


def _store_get(name: str) -> bytes | None:
    try:
        r = requests.get(f"{SEASON_STORE_URL}/{name}", timeout=20)
        return r.content if r.status_code == 200 else None
    except requests.RequestException:
        return None


@st.cache_data(ttl=3600, max_entries=40, show_spinner=False)
def _store_manifest(league_name: str, season: int) -> dict | None:
    """The season's store manifest (reg_weeks, code), if it was built by
    this code; checked again hourly, so files the workflow publishes after
    a deploy are picked up."""
    if season < SEASON_STORE_FIRST.get(league_name, 10_000):
        return None
    data = _store_get(season_store.asset_name(league_name, season, "manifest"))
    try:
        manifest = json.loads(data) if data else None
    except ValueError:
        return None
    return manifest if manifest and manifest.get("code") == _CODE_VERSION else None


def _stored_tables(league_name: str, season: int, weeks: tuple[int, int], part: str,
                   play_filter: core.PlayFilter, venue: str | None, adjusted: bool) -> dict | None:
    """The selection's tables from the season store, or None if it doesn't
    keep this selection (or can't be reached)."""
    manifest = _store_manifest(league_name, season)
    name = manifest and season_store.lookup(manifest, weeks, part, play_filter, venue, adjusted)
    if not name:
        return None
    data = _store_get(season_store.asset_name(league_name, season, name))
    try:
        return season_store.unpack_tables(data, _CODE_VERSION) if data else None
    except Exception:
        return None


@st.cache_data(max_entries=4, show_spinner=False)
def _final_season_pbp(league_name: str, season: int) -> pd.DataFrame:
    """A finished season's `League.season_pbp`, from the disk cache."""
    league = LEAGUES[league_name]
    return disk_cached("pbp", (league_name, season),
                       lambda: league._season_pbp(season, league.stamp()))


# ---------- Season stats ----------
# Which games the Season stats page counts: (regular season, postseason).
SEASON_PARTS = {"Regular season": (True, False), "Postseason": (False, True),
                "Both": (True, True)}


def season_plays(league: League, season: int, stamp: str, weeks: tuple[int, int], part: str,
                 play_filter: core.PlayFilter = core.NO_FILTER) -> pd.DataFrame:
    """The season's plays in regular-season `weeks` (inclusive) and/or the
    postseason, as `SEASON_PARTS[part]` says, in the situations
    `play_filter` names (`core.filter_plays_by`)."""
    pbp = league.season_pbp(season, stamp)
    reg, post = SEASON_PARTS[part]
    keep = pbp["postseason"] & post
    if reg:
        keep |= ~pbp["postseason"] & pbp["week"].between(*weeks)
    pbp = pbp[keep]
    if play_filter == core.NO_FILTER:
        return pbp
    # The score before each play needs whole games, so it's taken first.
    return core.filter_plays_by(pbp.assign(_margin=core.pre_play_margin(pbp)), play_filter)


# How the Season stats page's numbers are taken: as played, or adjusted for
# the opponents faced (core.opponent_adjust).
SEASON_ADJUST = {"Raw": False, "Opponent-adjusted": True}


def _scrimmage(pbp: pd.DataFrame) -> int:
    return int(((pbp["pass_attempt"] == 1) | (pbp["rush_attempt"] == 1)).sum())


@st.cache_data(max_entries=6, show_spinner=False)
def _season_tables(league_name: str, season: int, stamp: str, weeks: tuple[int, int],
                   part: str, play_filter: core.PlayFilter, venue: str | None,
                   adjusted: bool, chart_stamp: str = "") -> dict:
    return _build_season_tables(league_name, season, stamp, weeks, part, play_filter, venue,
                                adjusted)


@st.cache_data(max_entries=6, show_spinner=False)
def _final_season_tables(league_name: str, season: int, weeks: tuple[int, int], part: str,
                         play_filter: core.PlayFilter, venue: str | None,
                         adjusted: bool) -> dict:
    """`_season_tables` for a finished season, from the disk cache, else the
    season store, else computed (and saved to disk); no stamps, since the
    data won't change."""
    league = LEAGUES[league_name]

    def build() -> dict:
        stored = _stored_tables(league_name, season, weeks, part, play_filter, venue, adjusted)
        if stored is not None:
            return stored
        return _build_season_tables(league_name, season, league.stamp(), weeks, part,
                                    play_filter, venue, adjusted)

    return disk_cached(
        "tables", (league_name, season, weeks, part, tuple(play_filter), venue, adjusted), build)


def _build_season_tables(league_name: str, season: int, stamp: str, weeks: tuple[int, int],
                         part: str, play_filter: core.PlayFilter, venue: str | None,
                         adjusted: bool) -> dict:
    league = LEAGUES[league_name]
    # FTN's charting adds core.CHART_ROWS to the team tables (chart_stamp keys the cache).
    pbp = core.with_charting(season_plays(league, season, stamp, weeks, part, play_filter),
                             league.charting(season))
    every = _scrimmage(season_plays(league, season, stamp, weeks, part))
    ranked = list(league.season_teams(season, stamp).index)
    missing = league.missing_stats(pbp)

    def _rows(df: pd.DataFrame) -> pd.DataFrame:
        return df[df.index.isin(ranked)] if ranked and not df.empty else df

    # Every team plays a part in the adjustment, then only the ranked are kept.
    # Opponent strengths always come from all of a team's games; a venue only
    # picks which of its own games are shown (and adjusted).
    offense, defense = core.season_team_table(pbp), core.season_team_table(pbp, "defteam")
    situational, sit_defense = core.season_situational(pbp), core.season_situational(pbp, "defteam")
    if venue:
        v_offense = core.season_team_table(pbp, venue=venue)
        v_defense = core.season_team_table(pbp, "defteam", venue)
        v_situational = core.season_situational(pbp, venue=venue)
        v_sit_defense = core.season_situational(pbp, "defteam", venue)
    team_plays = core.venue_plays(pbp, venue)
    players = {kind: core.season_players(team_plays, kind, missing, teams=ranked or None)
               for kind in core.PLAYER_NAME_COL}
    if adjusted:
        games = core.season_games(pbp)
        offense, defense, off_strength, def_strength = core.opponent_adjust(offense, defense, games)
        situational, sit_defense, sit_off, sit_def = core.opponent_adjust(
            situational, sit_defense, games, set())
        if venue:
            v_games = core.venue_games(games, venue)
            v_offense = core.apply_strength(v_offense, def_strength, v_games)
            v_defense = core.apply_strength(v_defense, off_strength, v_games)
            v_situational = core.apply_strength(v_situational, sit_def, v_games)
            v_sit_defense = core.apply_strength(v_sit_defense, sit_off, v_games)
        players = {kind: core.adjust_players(df, team_plays, kind, def_strength)
                   for kind, df in players.items()}
    if venue:
        offense, defense = v_offense, v_defense
        situational, sit_defense = v_situational, v_sit_defense
    if adjusted:  # additive, so a few plays can land past a stat's range
        offense, defense = core.possible_values(offense), core.possible_values(defense)
        situational, sit_defense = core.possible_values(situational), core.possible_values(sit_defense)
        players = {kind: core.possible_values(df) for kind, df in players.items()}
    situational = _rows(situational)
    return {
        "offense": _rows(offense),
        "defense": _rows(defense),
        **players,
        "defenders": core.season_defenders(core.venue_plays(pbp, venue, "defteam"), missing,
                                           teams=ranked or None),
        # Kickers and punters, as played (no opponent adjustment).
        "kicking": core.season_kickers(team_plays, teams=ranked or None),
        "punting": core.season_punters(team_plays, teams=ranked or None),
        "situational_team": situational,
        "situational_defense": _rows(sit_defense),
        "situational": core.situational_distributions_from(situational),
        "missing": missing,
        "play_share": _scrimmage(pbp) / every if every else 1.0,
        "charted": core.charted_games(pbp),
    }


def season_tables(league: League, season: int, stamp: str, weeks: tuple[int, int], part: str,
                  adjust: str = "Raw", play_filter: core.PlayFilter = core.NO_FILTER,
                  venue: str = "All") -> dict:
    """The Season stats page's tables for the selection, cached: "offense" /
    "defense" (`core.season_team_table`, the ranked teams only), the
    "passing" / "rushing" / "receiving" / "defenders" leaderboards, each
    ranked offense's situational success ("situational_team",
    `core.season_situational`) and their distributions ("situational"),
    what each ranked defense allowed by situation ("situational_defense"), the
    leader columns the data lacks ("missing") and the share of the
    selection's scrimmage plays `play_filter` keeps ("play_share"). `venue`
    (`core.VENUES`) keeps each team's home or away games. With `adjust`
    "Opponent-adjusted" (`SEASON_ADJUST`) the team tables, situational
    success and player rates are adjusted for the opponents faced
    (`core.opponent_adjust` / `core.adjust_players`), opponent strengths
    from all games (`core.apply_strength` for a venue); counts, penalties
    and defenders aren't. Where the league has FTN's charting
    (`League.charting`), the team tables add `core.CHART_ROWS`, and
    "charted" is (games charted, games) in the selection."""
    if league.season_final(season):  # kept on disk: loads in well under a second
        return _final_season_tables(league.name, season, tuple(int(w) for w in weeks), part,
                                    play_filter, core.VENUES[venue], SEASON_ADJUST[adjust])
    chart_stamp = (nflverse_stamp("ftn_charting")
                   if isinstance(league, NFLLeague) and season >= CHART_FIRST_SEASON else "")
    return _season_tables(league.name, season, stamp, tuple(weeks), part, play_filter,
                          core.VENUES[venue], SEASON_ADJUST[adjust], chart_stamp)


@st.cache_data(max_entries=6, show_spinner=False)
def _season_nextgen(league_name: str, season: int, stamp: str, ngs_stamp: str,
                    weeks: tuple[int, int], part: str, venue: str | None) -> dict:
    league = LEAGUES[league_name]
    ngs = league.nextgen(season)
    if not ngs:
        return {}
    pbp_all = league.season_pbp(season, stamp)
    pbp = season_plays(league, season, stamp, weeks, part)  # whole games: no situation filter
    reg_weeks = pbp_all.loc[~pbp_all["postseason"], "week"].dropna()
    post_weeks = pbp_all.loc[pbp_all["postseason"], "week"].dropna().unique()
    out = {}
    for kind, df in ngs.items():
        # Week 0 is NGS's own regular season: used when that's the selection
        # and NGS has the same weeks published as the play-by-play.
        ngs_reg = df.loc[(df["season_type"] == "REG") & (df["week"] > 0), "week"] if not df.empty else []
        whole = (SEASON_PARTS[part] == (True, False) and venue is None and len(reg_weeks)
                 and tuple(weeks) == (int(reg_weeks.min()), int(reg_weeks.max()))
                 and len(ngs_reg) and int(max(ngs_reg)) == int(reg_weeks.max()))
        out[kind] = core.ngs_season(df, kind, core.team_games(pbp, venue), post_weeks, bool(whole))
        out[kind].attrs["whole_season"] = bool(whole)  # for the page's caption
    return out


def season_nextgen(league: League, season: int, stamp: str, weeks: tuple[int, int], part: str,
                   venue: str = "All") -> dict[str, pd.DataFrame]:
    """Next Gen Stats per (gsis_id, team) for the Season stats leaderboards
    (`core.ngs_season`, `core.NGS_COLS`) over the selection's weeks, games
    and venue; whole games, so the situation filters don't apply. Each
    table's attrs["whole_season"] says whether it's NGS's own season line.
    Empty where the league has none."""
    ngs_stamp = nflverse_stamp("nextgen_stats") if isinstance(league, NFLLeague) else ""
    return _season_nextgen(league.name, season, stamp, ngs_stamp, tuple(weeks), part,
                           core.VENUES[venue])


@st.cache_data(max_entries=4, show_spinner=False)
def _season_chart_games(league_name: str, season: int, stamp: str, chart_stamp: str,
                        weeks: tuple[int, int], part: str, play_filter: core.PlayFilter) -> dict:
    league = LEAGUES[league_name]
    pbp = core.with_charting(season_plays(league, season, stamp, weeks, part, play_filter),
                             league.charting(season))
    return {side: core.chart_by_game(pbp, side) for side in ("posteam", "defteam")}


def season_chart_games(league: League, season: int, stamp: str, weeks: tuple[int, int], part: str,
                       play_filter: core.PlayFilter = core.NO_FILTER) -> dict[str, pd.DataFrame]:
    """Each team-game's charting counts over the selection (`core.chart_by_game`,
    "posteam" for offenses, "defteam" for what each defense faced), for the
    Season stats Team tab's Charting by game: the play text's formation, plus
    FTN's charting where the league has it (`League.charting`)."""
    chart_stamp = (nflverse_stamp("ftn_charting")
                   if isinstance(league, NFLLeague) and season >= CHART_FIRST_SEASON else "")
    return _season_chart_games(league.name, season, stamp, chart_stamp, tuple(weeks), part,
                               play_filter)


@st.cache_data(max_entries=2, show_spinner=False)
def _season_player_ids(league_name: str, season: int, stamp: str) -> pd.DataFrame:
    return core.player_ids(LEAGUES[league_name].season_pbp(season, stamp))


@st.cache_data(max_entries=6, show_spinner=False)
def _final_season_player_ids(league_name: str, season: int) -> pd.DataFrame:
    def build() -> pd.DataFrame:
        if _store_manifest(league_name, season):
            data = _store_get(season_store.asset_name(league_name, season, "ids"))
            try:
                ids = season_store.unpack_frame(data, _CODE_VERSION) if data else None
            except Exception:
                ids = None
            if ids is not None:
                return ids
        league = LEAGUES[league_name]
        return core.player_ids(league.season_pbp(season, league.stamp()))
    return disk_cached("ids", (league_name, season), build)


def season_player_ids(league: League, season: int, stamp: str) -> pd.DataFrame:
    """(team, name, gsis_id) for the season's passers, rushers and receivers
    (`core.player_ids`), to find a season table's players on an injury
    report. Empty without ids (college). A finished season's comes from the
    disk cache or the season store, without its play-by-play."""
    if league.season_final(season):
        return _final_season_player_ids(league.name, season)
    return _season_player_ids(league.name, season, stamp)


@st.cache_data(max_entries=6, show_spinner=False)
def _season_tendencies(league_name: str, season: int, stamp: str, weeks: tuple[int, int],
                       part: str, play_filter: core.PlayFilter) -> dict:
    league = LEAGUES[league_name]
    return core.tendency_baselines(season_plays(league, season, stamp, weeks, part, play_filter),
                                   league.run_gaps)


def season_tendencies(league: League, season: int, stamp: str, weeks: tuple[int, int],
                      part: str, play_filter: core.PlayFilter = core.NO_FILTER) -> dict:
    """League-wide tendencies over the selection (`core.tendency_baselines`),
    what a team's split is compared against on the Season stats page."""
    return _season_tendencies(league.name, season, stamp, tuple(weeks), part, play_filter)


# ---------- Pre-game preview ----------
@dataclass
class Pregame:
    """What a game's spoiler-free preview is built from: `context` for
    game_summary.preview() and the `matchups` it shows (offense team ->
    `core.matchup_table`), or why there are none (`reason`)."""
    context: game_summary.MatchupContext | None
    matchups: dict
    reason: str = ""


def is_postseason(game: pd.Series) -> bool:
    """Whether a `League.games()` row is a postseason game (the NFL's
    `game_type`, college's `season_type` 3)."""
    return game.get("game_type", "REG") != "REG" or game.get("season_type") == 3


def betting_line(home: str, away: str, spread, total) -> str:
    """"Bears favored by 3, total 50.5" from the home team's spread and the total
    (either may be missing); "" without either."""
    line = ""
    if pd.notna(spread):
        line = "Pick'em" if spread == 0 else f"{home if spread > 0 else away} favored by {abs(spread):g}"
    if pd.notna(total):
        line += (", " if line else "") + f"total {total:g}"
    return line


def pregame(league: League, season: int, stamp: str, game: pd.Series) -> Pregame:
    """The Matchup page's preview of `game` (a `League.games()` row), made
    safe to read while watching it: the season numbers, snap counts and
    lineups cover only the regular-season weeks before the game's (every
    one, for a postseason game), opponent-adjusted, so neither the game nor
    anything after it is in them. The injury report and the betting line are
    the ones the game was played under."""
    away, home = game["away_team"], game["home_team"]
    names = league.team_names(season, stamp)
    week = int(game["week"])
    played = league.season_pbp(season, stamp)
    played = played[~played["postseason"]]
    weeks = sorted(int(w) for w in played["week"].dropna().unique()
                   if is_postseason(game) or w < week)
    if not weeks:
        return Pregame(None, {}, "No games were played before this one this season.")
    span = (weeks[0], weeks[-1])
    tables = season_tables(league, season, stamp, span, "Regular season", "Opponent-adjusted")
    off_table = core.season_qualified(tables["offense"], core.SEASON_MIN_PLAYS)
    def_table = core.season_qualified(tables["defense"], core.SEASON_MIN_PLAYS)
    short = [t for t in (away, home) if t not in off_table.index or t not in def_table.index]
    if short:
        return Pregame(None, {}, f"{' and '.join(short)} had fewer than {core.SEASON_MIN_PLAYS} "
                                 "plays before this game.")
    rows = core.matchup_stats(off_table, tables["missing"])
    plays = season_plays(league, season, stamp, span, "Regular season")
    sit_off = tables["situational_team"][tables["situational_team"].index.isin(off_table.index)]
    sit_def = tables["situational_defense"][tables["situational_defense"].index.isin(def_table.index)]
    situational = {away: core.situational_matchup(sit_off, sit_def, away, home, plays),
                   home: core.situational_matchup(sit_off, sit_def, home, away, plays)}

    avail = league.availability(season)
    player_ids = season_player_ids(league, season, stamp)
    reports, report_units, statuses, lineups, lineup_net = {}, {}, {}, {}, {}
    for team in (away, home):
        ids = plays.loc[(plays["home_team"] == team) | (plays["away_team"] == team),
                        "game_id"].unique()
        reports[team] = core.injury_report(avail.injuries, team, week,
                                           core.team_snaps(avail.snaps, team, ids), avail.starters)
        report_units[team] = core.availability_summary(reports[team])
        statuses[team] = core.status_by_name(reports[team], team, player_ids, avail.short_names)
        lineups[team] = core.lineup_change(avail.snaps, avail.injuries, avail.roster, team, week, ids)
        lineup_net[team] = core.lineup_groups(lineups[team])
    has_lineup = not avail.snaps.empty and not avail.injuries.empty
    adjustments = ({away: core.lineup_adjustment(lineups[away], lineups[home]),
                    home: core.lineup_adjustment(lineups[home], lineups[away])} if has_lineup else {})
    matchups = {away: core.matchup_table(off_table, def_table, away, home, rows, adjustments.get(away)),
                home: core.matchup_table(off_table, def_table, home, away, rows, adjustments.get(home))}
    # FTN's charting (NFL 2022 on), as the Matchup page's Charting table.
    chart_rows = core.matchup_chart_stats(off_table)
    charting = ({away: core.matchup_table(off_table, def_table, away, home, chart_rows),
                 home: core.matchup_table(off_table, def_table, home, away, chart_rows)}
                if chart_rows else {})
    n_charted, n_games = tables["charted"]
    charting_note = ("Under C % and No-huddle % from the play text"
                     + ("; the rest FTN Data's charting via nflverse, over the plays charted"
                        + (f" ({n_charted} of {n_games} games)" if n_charted < n_games else "")
                        if "Motion %" in chart_rows else "")
                     if charting else "")

    years = league.class_years(season, stamp)
    draft = league.draft_board(season, stamp)

    def _with_status(df: pd.DataFrame, team: str) -> pd.DataFrame:
        df = core.with_class_year(df, years, team)
        status = [statuses[team].get(p, "") for p in df["Player"]] if "Player" in df.columns else []
        if any(status):
            df = df.copy().assign(Status=status)[["Player", "Status",
                                                  *[c for c in df.columns if c != "Player"]]]
        return core.with_draft(df, draft, team)

    leaders = {}
    for kind, n in core.MATCHUP_LEADERS:
        df = tables[kind]
        for team in (away, home):
            if not df.empty:
                leaders[(team, "defense" if kind == "defenders" else kind)] = _with_status(
                    df[df["Team"] == team].drop(columns="Team").head(n), team)
    shown = ["Player", "Pos", "Status", "Snaps", "Starter", "Practice", "Injury"]

    def _off_report(team: str) -> pd.DataFrame:
        ch = lineups[team]
        return ch[~ch["Why"].isin(core.LIKELY_OUT) & (ch["Change"].abs() >= core.LINEUP_SHOWN)]

    kick = pd.Timestamp(game["game_date"])
    when = f"{game['week_label']}, {kick:%a %b} {kick.day}" + (
        ", neutral site" if game.get("location") == "Neutral" or game.get("neutral") is True else "")
    week_note = f"week {span[0]}" if span[0] == span[1] else f"weeks {span[0]}–{span[1]}"
    context = game_summary.MatchupContext(
        league=league.key, away=away, home=home, when=when,
        names={t: names.get(t, t) for t in (away, home)},
        line=betting_line(names.get(home, home), names.get(away, away),
                          game.get("spread_line"), game.get("total_line")),
        selection=f"{season} {week_note}, the games before this one, opponent-adjusted",
        n_teams=len(off_table), matchups=matchups, situational=situational, leaders=leaders,
        injuries={t: reports[t][shown] for t in (away, home)}, injury_units=report_units,
        injury_note=f"the week {week} reports" if not avail.injuries.empty else "",
        lineup_groups=lineup_net if not avail.snaps.empty else {},
        lineup_players={t: _off_report(t).drop(columns="Unit").round(2) for t in (away, home)},
        charting=charting, charting_note=charting_note)
    return Pregame(context, matchups)
