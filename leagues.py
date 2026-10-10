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

import time
from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

import nflreadpy
import numpy as np
import pandas as pd
import requests
import streamlit as st
from nflreadpy.config import update_config

import cfb_feed
import live_feed
import nflfastr_models
import replay_core as core


class GameLoadError(Exception):
    """Nothing could be loaded; the message says why, for the sidebar."""


@dataclass
class LoadedGame:
    pbp: pd.DataFrame           # PBP_COLS layout, one row per play, in game order
    live_state: str             # ESPN's "pre" / "in" / "post"
    notes: list[str]            # sidebar captions about where the data comes from
    warning: str | None = None  # something the viewer should know (e.g. no models)


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
    missing_stats: frozenset = frozenset()  # Player-leader columns the data doesn't have

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

    def baseline_pbp(self, season: int) -> pd.DataFrame | None:
        """Play-by-play (core.BASELINE_COLS) the percentile baselines for
        `season` are built from; None if unavailable."""
        raise NotImplementedError


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
    stored as float32 to save memory, as nfl_data_py did. Raises if a season
    isn't published."""
    df = nflreadpy.load_pbp(seasons).select(columns).to_pandas()
    floats = df.select_dtypes("float64").columns
    df[floats] = df[floats].astype("float32")
    return df


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
def nflverse_stamp() -> str:
    """When nflverse last rebuilt its pbp files. Checked every minute, so a
    newly published game shows up right away instead of on a fixed timer."""
    try:
        r = requests.get(f"{NFLVERSE_RELEASE}/pbp/timestamp.json", timeout=10)
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
    df = live_feed.build_live_pbp(summary, sched_row, models, fg_table, core.PBP_COLS)
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


@st.cache_data(ttl=86400, max_entries=1)
def _nfl_baseline_pbp(season: int) -> pd.DataFrame | None:
    """The 3 NFL seasons before `season`."""
    prior = [s for s in [season - 3, season - 2, season - 1] if s >= 1999]
    if not prior:
        return None
    df = nflverse_pbp(prior, core.BASELINE_COLS)
    return None if df.empty else df


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
            return LoadedGame(_nfl_game(season, stamp, row["game_id"]), "post",
                              [f"📊 Official nflfastR play-by-play · nflverse data as of {stamp}"])
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

    def baseline_pbp(self, season: int) -> pd.DataFrame | None:
        return _nfl_baseline_pbp(season)


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


@st.cache_data(max_entries=8)
def load_cfb_game(season: int, stamp: str, game_id: str) -> pd.DataFrame:
    raw = load_cfb_season(season, stamp)
    return cfb_feed.to_pbp(raw[raw["game_id"].astype(str) == game_id], core.PBP_COLS)


@st.cache_data(ttl=20)
def load_cfb_live_game(game_id: str) -> tuple[pd.DataFrame, str, str]:
    """A not-yet-published college game from ESPN's feed, through
    sportsdataverse's processing, in the app's layout. Returns (pbp, ESPN
    state, fetched-at clock). Cached 20s, so reruns don't hammer ESPN."""
    summary = cfb_feed.fetch_summary(game_id)
    pbp = cfb_feed.to_pbp(cfb_feed.live_raw(summary, game_id), core.PBP_COLS)
    return pbp, cfb_feed.game_state(summary), datetime.now().strftime("%H:%M:%S")


@st.cache_data(ttl=3600)
def load_cfb_team_meta(season: int, stamp: str) -> tuple[dict, dict, dict]:
    """(colors, logos, nicknames) keyed by team abbreviation."""
    return cfb_feed.team_meta(_cfb_season_or_empty(season, stamp)[0],
                              cfb_feed.read_team_info(season), load_cfb_schedule(season))


@st.cache_data(ttl=86400, max_entries=1)
def _cfb_baseline_pbp(season: int) -> pd.DataFrame:
    """The college season before `season`, FBS-vs-FBS games only, mapped to
    the app's layout. Empty if it isn't published."""
    prior = season - 1
    if prior < cfb_feed.FIRST_SEASON:
        return pd.DataFrame(columns=core.BASELINE_COLS)
    pbp = cfb_feed.to_pbp(cfb_feed.read_season(prior), core.PBP_COLS)
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

    def baseline_pbp(self, season: int) -> pd.DataFrame | None:
        df = _cfb_baseline_pbp(season)
        return None if df.empty else df


NFL = NFLLeague(
    name="NFL", key="NFL", title="🏈 NFL Tape-Delay Replay", first_season=1999,
    # A real NFL broadcast runs ~3h10m for 60 minutes of game clock.
    broadcast_minutes=190.0, baseline_label="last 3 seasons")
COLLEGE = CollegeLeague(
    name="College football", key="CFB", title="🏈 College Football Tape-Delay Replay",
    first_season=cfb_feed.FIRST_SEASON,
    # ~3h20m. One college season has about as many team-games as three NFL ones.
    broadcast_minutes=200.0, baseline_label="last season's FBS-vs-FBS games",
    group_label="Conference", untimed_ot=True, missing_stats=cfb_feed.MISSING_LEADER_STATS)
LEAGUES = {lg.name: lg for lg in (NFL, COLLEGE)}


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


def stat_baselines(league: League, season: int) -> dict[str, np.ndarray]:
    """Team stats percentile baselines for `season`, by stat."""
    return _stat_baselines(league.name, season)


def situational_baselines(league: League, season: int) -> dict[str, dict[str, np.ndarray]]:
    """Situational success-rate baselines for `season`, by situation and metric."""
    return _situational_baselines(league.name, season)
