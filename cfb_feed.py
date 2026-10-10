"""
College football play-by-play in the app's nflfastR column layout.

Source: sportsdataverse's `espn_cfb_pbp` release, ESPN's college play feed run
through sportsdataverse-py's processing (its own EP/EPA and WP models, trained
on college games). It is rebuilt about once a day, like nflverse's NFL pbp.
Schedules (kickoff time, conferences, AP rank) and team info (colors, logos,
mascots) come from the same release repo.

The release carries columns that look past the play they sit on: final scores,
how the drive ended, the next play's values (`lead_*`), win probability after
the play. Reading them would leak what hasn't been watched yet, so only the
columns in `RAW_COLS` are ever read, and none of those look ahead.

`to_pbp()` maps the raw rows to `PBP_COLS` the way `live_feed` builds the NFL
layout from ESPN: the receiving team has the ball on kickoffs, the try after a
touchdown is its own row, END QUARTER / END GAME rows close each period, and
drives are numbered from changes of possession.

Live games: `fetch_summary()` reads ESPN's college summary for a game and
`live_raw()` runs it through sportsdataverse-py's own processing, the pipeline
that builds the release. Its rows have the release's columns, so `to_pbp()`
maps a live game exactly like a published one, and a game moving from live to
published keeps the same layout. No Streamlit calls here.
"""

import io
import re

import numpy as np
import pandas as pd
import requests

RELEASE = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/summary"
FIRST_SEASON = 2004
ESPN_LOGO = "https://a.espncdn.com/i/teamlogos/ncaa/500/{}.png"

# Every column read from the release. Each describes the play it's on (or the
# state before it), never anything later. homeScore/awayScore are the score
# after the play; home_wp_after/away_wp_after are only used for the try row
# inserted right after a touchdown, as that row's pre-snap win probability.
RAW_COLS = [
    "season", "week", "seasonType", "game_id", "game_play_number", "wallclock",
    "homeTeamId", "awayTeamId", "homeTeamAbbrev", "awayTeamAbbrev", "pos_team_id",
    "period", "clock.displayValue", "start.adj_TimeSecsRem",
    "type.text", "text", "start.down", "start.distance", "start.yardsToEndzone",
    "goal_to_go", "statYardage", "homeScore", "awayScore",
    "EPA", "home_wp_before", "away_wp_before", "home_wp_after", "away_wp_after",
    "pass", "rush", "completion", "sack", "int", "fumble_lost",
    "touchdown", "pass_td", "rush_td", "safety",
    "kickoff_play", "punt", "fg_attempt", "field_goal_result", "kneel_down",
    "penalty_flag", "penalty_declined", "penalty_offset", "penalty_no_play",
    "penalized_team", "yds_penalty", "first_down_created", "firstD_by_penalty",
    "yds_rushed", "yds_receiving", "air_yards", "yards_after_catch",
    "passer_player_name", "rusher_player_name", "receiver_player_name",
    "sack_player_name", "sack_player_name2", "interception_player_name",
    "pass_breakup_player_name", "fumble_forced_player_name",
    "pointAfterAttempt.text", "status_type_completed",
]

# The college feed has no tackles, QB hits or tackles for loss: these
# Player-leader columns would only ever show zeros, so the app drops them.
MISSING_LEADER_STATS = frozenset({"Tackles", "QB Hits", "TFL", "Hits"})

# A clock reading this far after both neighbours is a feed glitch (a stray
# 0:00 mid-quarter, say), not a play that happened later.
_SPIKE_SECS = 60


def _get(url: str, timeout: float = 60) -> bytes:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


def stamp() -> str:
    """When the college pbp release was last rebuilt."""
    try:
        r = requests.get(f"{RELEASE}/espn_cfb_pbp/timestamp.json", timeout=10)
        r.raise_for_status()
        return str(r.json().get("last_updated", ""))
    except Exception:
        return ""


def read_season(season: int) -> pd.DataFrame:
    """One season of published college pbp, `RAW_COLS` only. Raises if the
    season isn't published (yet)."""
    blob = _get(f"{RELEASE}/espn_cfb_pbp/play_by_play_{int(season)}.parquet", timeout=120)
    return pd.read_parquet(io.BytesIO(blob), columns=RAW_COLS)


def read_schedule(season: int) -> pd.DataFrame:
    """The season's schedule. It has final scores too: never show those."""
    try:
        return pd.read_parquet(io.BytesIO(_get(f"{RELEASE}/cfb_schedules/cfb_schedules_{int(season)}.parquet")))
    except Exception:
        return pd.DataFrame()


def read_team_info(season: int) -> pd.DataFrame:
    for s in (int(season), int(season) - 1):
        try:
            return pd.read_parquet(io.BytesIO(_get(f"{RELEASE}/cfb_team_info/cfb_team_info_{s}.parquet")))
        except Exception:
            continue
    return pd.DataFrame()


def _utc(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, utc=True, errors="coerce")


def _local_date(ts: pd.Series) -> pd.Series:
    """UTC timestamps → the calendar date in US Eastern time."""
    return _utc(ts).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")


def week_label(season_type, week) -> str:
    return "Postseason" if season_type == 3 else f"Week {int(week)}"


def team_ids(raw: pd.DataFrame, sched: pd.DataFrame, info: pd.DataFrame | None = None) -> pd.DataFrame:
    """(abbr, id) for every team: ESPN's abbreviations from the published plays,
    then from the schedule (it leaves them blank for games not played yet),
    then team info's as a last resort."""
    pairs = [
        raw[["homeTeamAbbrev", "homeTeamId"]].set_axis(["abbr", "id"], axis=1),
        raw[["awayTeamAbbrev", "awayTeamId"]].set_axis(["abbr", "id"], axis=1),
    ]
    if not sched.empty and {"home_abbreviation", "home_id"} <= set(sched.columns):
        pairs += [sched[[f"{s}_abbreviation", f"{s}_id"]].set_axis(["abbr", "id"], axis=1)
                  for s in ("home", "away")]
    if info is not None and not info.empty and {"abbreviation", "team_id"} <= set(info.columns):
        pairs.append(info[["abbreviation", "team_id"]].set_axis(["abbr", "id"], axis=1))
    ids = pd.concat(pairs).dropna()
    ids["id"] = ids["id"].astype(int)
    return ids.drop_duplicates("id").drop_duplicates("abbr").reset_index(drop=True)


def list_games(raw: pd.DataFrame, sched: pd.DataFrame, today: str | None = None) -> pd.DataFrame:
    """One row per game: game_id, week label, date, teams, label, conferences
    and source. Ranks are the AP rank going into the game.

    source is "published" for a game in the release and "live" for one that
    kicks off by `today` (US Eastern, YYYY-MM-DD) with an FBS team in it but
    isn't published yet: those are read from ESPN's feed."""
    g = (raw.assign(kickoff=_utc(raw["wallclock"]))
         .groupby("game_id", as_index=False)
         .agg(season_type=("seasonType", "first"), week=("week", "first"),
              home_team=("homeTeamAbbrev", "first"), away_team=("awayTeamAbbrev", "first"),
              kickoff=("kickoff", "min")))
    g["home_conf"] = g["away_conf"] = ""
    g["home_rank"] = g["away_rank"] = np.nan
    if not sched.empty and "game_id" in sched.columns:
        s = sched.set_index("game_id")
        if "start_date" in s.columns:
            k = _utc(g["game_id"].map(s["start_date"]))
            g["kickoff"] = k.where(k.notna(), g["kickoff"])
        for col, src in (("home_conf", "home_conference"),
                         ("away_conf", "away_conference"), ("home_rank", "home_rank"),
                         ("away_rank", "away_rank")):
            if src in s.columns:
                v = g["game_id"].map(s[src])
                g[col] = v.where(v.notna(), g[col])
    g["source"] = "published"
    g["game_id"] = g["game_id"].astype(str)
    if today and not sched.empty and "start_date" in sched.columns:
        g = pd.concat([g, _live_games(sched, set(g["game_id"]), team_ids(raw, sched), today)],
                      ignore_index=True)
    g["game_date"] = _local_date(g["kickoff"])
    g["week_label"] = [week_label(t, w) for t, w in zip(g["season_type"], g["week"])]
    g["week_order"] = np.where(g["season_type"] == 3, 100, g["week"])

    def team(abbr, rank):
        return f"#{int(rank)} {abbr}" if pd.notna(rank) and rank <= 25 else abbr

    g["label"] = [f"{team(a, ar)} @ {team(h, hr)} ({d})" + (" 🔴 live" if src == "live" else "")
                  for a, ar, h, hr, d, src in zip(g["away_team"], g["away_rank"], g["home_team"],
                                                  g["home_rank"], g["game_date"], g["source"])]
    return g.sort_values(["week_order", "kickoff", "game_id"]).reset_index(drop=True)


def _live_games(sched: pd.DataFrame, published: set[str], ids: pd.DataFrame, today: str) -> pd.DataFrame:
    """Schedule rows for games that kick off by `today`, have an FBS team (the
    release covers those) and aren't published yet."""
    s = sched.assign(game_id=sched["game_id"].astype(str), kickoff=_utc(sched["start_date"]))
    fbs = s["fbs_participant"].fillna(False).astype(bool) if "fbs_participant" in s.columns else True
    s = s[fbs & ~s["game_id"].isin(published) & (_local_date(s["kickoff"]).fillna("9999") <= today)]
    abbr = dict(zip(ids["id"], ids["abbr"]))

    def name(side):
        known = s[f"{side}_id"].map(lambda i: abbr.get(int(i)) if pd.notna(i) else None)
        return s[f"{side}_abbreviation"].fillna(known).fillna(s[f"{side}_team"])

    return pd.DataFrame({
        "game_id": s["game_id"],
        "season_type": np.where(s["season_type"].astype(str).str.startswith("post"), 3, 2),
        "week": s["week"], "home_team": name("home"), "away_team": name("away"),
        "kickoff": s["kickoff"],
        "home_conf": s.get("home_conference", ""), "away_conf": s.get("away_conference", ""),
        "home_rank": s.get("home_rank", np.nan), "away_rank": s.get("away_rank", np.nan),
        "source": "live",
    })


def team_meta(raw: pd.DataFrame, info: pd.DataFrame,
              sched: pd.DataFrame | None = None) -> tuple[dict, dict, dict]:
    """(colors, logos, nicknames) keyed by the abbreviation the pbp uses."""
    ids = team_ids(raw, sched if sched is not None else pd.DataFrame(), info)
    by_id = info.set_index("team_id") if not info.empty and "team_id" in info.columns else pd.DataFrame()
    colors, logos, nicks = {}, {}, {}
    for abbr, tid in zip(ids["abbr"], ids["id"]):
        r = by_id.loc[tid] if tid in by_id.index else None
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        color = r.get("color") if r is not None else None
        if isinstance(color, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            colors[abbr] = color
        logo = r.get("logo") if r is not None else None
        logos[abbr] = logo if isinstance(logo, str) and logo.startswith("http") else ESPN_LOGO.format(int(tid))
        nick = r.get("mascot") if r is not None else None
        if isinstance(nick, str) and nick:
            nicks[abbr] = nick
    return colors, logos, nicks


# ---------- live games ----------
# What sportsdataverse's own fetch (CFBPlayProcess.espn_cfb_pbp) puts in place
# of a key ESPN's summary leaves out, e.g. "drives" before kickoff.
_SUMMARY_DICTS = ("boxscore", "format", "gameInfo", "drives", "predictor", "header", "standings")
_SUMMARY_LISTS = ("leaders", "broadcasts", "pickcenter", "againstTheSpread", "odds",
                  "winprobability", "scoringPlays", "videos", "injuries", "gameNotes")


def fetch_summary(event_id, timeout: float = 15) -> dict:
    """ESPN's public college summary for one game: header, drives and plays."""
    r = requests.get(SUMMARY_URL, params={"event": str(event_id)}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def game_state(summary: dict) -> str:
    """ESPN's "pre" / "in" / "post"."""
    comp = ((summary.get("header") or {}).get("competitions") or [{}])[0]
    return ((comp.get("status") or {}).get("type") or {}).get("state") or "pre"


def live_raw(summary: dict, game_id, odds: dict | None = None) -> pd.DataFrame:
    """ESPN's summary for one game → rows with the release's `RAW_COLS`.

    Runs sportsdataverse-py's processing, the same that builds the release:
    EP/EPA and win probability from its college models. Skips its player-id
    lookups (names come from the play text) and its 4th-down and two-point
    models, so nothing but the spread needs the network: when the summary has
    no pickcenter, it asks ESPN's odds endpoint. `odds` (gameSpread,
    overUnder, homeFavorite, gameSpreadAvailable) skips that too."""
    from sportsdataverse.cfb import CFBPlayProcess  # heavy (polars, models): live games only

    payload = {"timeouts": {}}
    payload.update({k: summary.get(k) or {} for k in _SUMMARY_DICTS})
    payload.update({k: summary.get(k) or [] for k in _SUMMARY_LISTS})
    proc = CFBPlayProcess(gameId=int(game_id), join_participants=False, odds_override=odds)
    proc.json = payload
    out = proc.run_processing_pipeline(fourth_down_probs=False, two_pt_probs=False)
    plays = pd.DataFrame(out.get("plays") or [])
    if plays.empty:
        return pd.DataFrame(columns=RAW_COLS)
    if "EPA" not in plays.columns:
        # sportsdataverse skips a feed it judges corrupt (e.g. a finished game
        # with under 50 plays) and hands back the plays unprocessed.
        raise ValueError("sportsdataverse couldn't process this game's ESPN feed")
    # The release keeps the possession team's id as pos_team_id (and its name
    # in pos_team); the pipeline's own output has the id in pos_team.
    if "pos_team_id" not in plays.columns:
        plays["pos_team_id"] = plays["pos_team"]
    for c in RAW_COLS:
        if c not in plays.columns:
            plays[c] = np.nan
    return plays[RAW_COLS]


def fbs_games(sched: pd.DataFrame) -> set[str]:
    """Game ids of FBS-vs-FBS games, for the percentile baselines."""
    if sched.empty or not {"home_division", "away_division"} <= set(sched.columns):
        return set()
    m = (sched["home_division"] == "fbs") & (sched["away_division"] == "fbs")
    return set(sched.loc[m, "game_id"].astype(str))


# ---------- raw rows → nflfastR layout ----------
_TD_TRY = re.compile(r"^(.*?\bTOUCHDOWN\b(?:,? clock \d{1,2}:\d{2})?(?:,? 1ST DOWN)?\.?)\s+(\S.*)$", re.S)
_PAREN_TRY = re.compile(r"^(.*\S)\s*\(([^()]*)\)\s*$", re.S)


def _split_try(text: str, fallback: str) -> tuple[str, str]:
    """A touchdown row's text → (touchdown part, try part). ESPN puts the try
    on the touchdown's row, in two styles: '... TOUCHDOWN, clock 05:07 #39
    C.Salas kick attempt good (...)' and '... for a TD (Conor Talty KICK)'."""
    m = _TD_TRY.match(text) or _PAREN_TRY.match(text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return text, fallback


def _b(s: pd.Series) -> pd.Series:
    return s.fillna(False).astype(bool)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def to_pbp(raw: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Raw release rows (one or many games) → the app's pbp layout. Every name
    in `columns` is present (NaN if the college feed doesn't have it)."""
    if raw.empty:
        return pd.DataFrame(columns=columns)
    df = _relocate_late_rows(raw.sort_values(["game_id", "game_play_number"], kind="stable"))
    gid = df["game_id"]
    first_row = gid != gid.shift()
    home, away = df["homeTeamAbbrev"], df["awayTeamAbbrev"]
    ttype = df["type.text"].fillna("")
    text = df["text"].fillna("").str.strip()

    # Any row still listed after a later period keeps the period in progress,
    # so boxscore quarters and drive clocks stay in order.
    period = df["period"].astype(float)
    qtr = period.groupby(gid).cummax()

    timeout = ttype.eq("Timeout")
    no_play = timeout | ttype.eq("Penalty") | _b(df["penalty_no_play"])
    two_pt = ttype.str.startswith("Two Point")
    is_pass, is_rush = _b(df["pass"]), _b(df["rush"])
    kneel = _b(df["kneel_down"])
    spike = is_pass & text.str.contains(r"\bspike", case=False)
    play_type = pd.Series(np.select(
        [no_play, _b(df["kickoff_play"]), _b(df["punt"]), _b(df["fg_attempt"]),
         two_pt & ttype.str.contains("Pass"), two_pt, kneel, spike, is_pass, is_rush],
        ["no_play", "kickoff", "punt", "field_goal", "pass", "run",
         "qb_kneel", "qb_spike", "pass", "run"], default=""), index=df.index).replace("", None)

    out = pd.DataFrame(index=df.index)
    out["game_id"] = gid.astype(str)
    out["season"] = df["season"]
    out["week"] = df["week"]
    out["game_date"] = _local_date(_utc(df["wallclock"]).groupby(gid).transform("min"))
    out["home_team"], out["away_team"] = home, away
    pos = np.where(df["pos_team_id"] == df["homeTeamId"], home,
                   np.where(df["pos_team_id"] == df["awayTeamId"], away, None))
    out["posteam"] = pd.Series(pos, index=df.index).where(~timeout, None)
    out["defteam"] = np.where(out["posteam"] == home, away, np.where(out["posteam"] == away, home, None))
    out["qtr"] = qtr
    reg = qtr <= 4
    out["time"] = df["clock.displayValue"].where(reg & (period == qtr))

    # Clock: game seconds remaining in regulation. Overtime has none (untimed),
    # nor does a row listed in a later period than its own.
    gsr = _num(df["start.adj_TimeSecsRem"]).where(reg & (period == qtr))
    el = 3600 - gsr
    spike_clock = ((el - el.groupby(gid).shift(1) > _SPIKE_SECS)
                   & (el - el.groupby(gid).shift(-1) > _SPIKE_SECS))
    out["game_seconds_remaining"] = gsr.mask(spike_clock)

    out["desc"] = text
    out["play_type"] = play_type
    down = _num(df["start.down"])
    valid_down = down.between(1, 4) & ~play_type.isin(["kickoff"]) & ~two_pt
    out["down"] = down.where(valid_down)
    out["ydstogo"] = _num(df["start.distance"]).where(valid_down)
    ytg = _num(df["start.yardsToEndzone"])
    # ESPN measures a kickoff from the kicking team; nflfastR from the receiver.
    ytg = ytg.where(play_type != "kickoff", 100 - ytg)
    out["yardline_100"] = ytg.where(ytg.between(1, 99))
    out["goal_to_go"] = (_b(df["goal_to_go"]) & valid_down).astype(float)

    scrimmage = play_type.isin(["pass", "run"]) & ~two_pt
    pass_att = (play_type == "pass") & ~two_pt
    rush_att = play_type.isin(["run", "qb_kneel"]) & ~two_pt
    complete = pass_att & _b(df["completion"])
    out["yards_gained"] = _num(df["statYardage"]).where(play_type != "no_play", 0.0)
    out["pass_attempt"] = pass_att.astype(float)
    out["rush_attempt"] = rush_att.astype(float)
    out["complete_pass"] = complete.astype(float)
    out["sack"] = (pass_att & _b(df["sack"])).astype(float)
    out["interception"] = (pass_att & _b(df["int"])).astype(float)
    out["qb_kneel"] = (play_type == "qb_kneel").astype(float)
    out["qb_spike"] = (play_type == "qb_spike").astype(float)
    out["qb_hit"] = np.nan
    rec_yds = _num(df["yds_receiving"]).fillna(_num(df["statYardage"]))
    out["passing_yards"] = rec_yds.where(complete)
    out["receiving_yards"] = rec_yds.where(complete)
    out["rushing_yards"] = _num(df["yds_rushed"]).fillna(_num(df["statYardage"])).where(rush_att)
    real_pass = pass_att & ~out["sack"].astype(bool)
    out["air_yards"] = _num(df["air_yards"]).where(real_pass)
    out["yards_after_catch"] = _num(df["yards_after_catch"]).where(complete)

    live = ~no_play
    out["touchdown"] = (_b(df["touchdown"]) & live).astype(float)
    out["pass_touchdown"] = (_b(df["pass_td"]) & live).astype(float)
    out["rush_touchdown"] = (_b(df["rush_td"]) & live).astype(float)
    out["safety"] = _b(df["safety"]).astype(float)
    out["fumble_lost"] = (_b(df["fumble_lost"]) & live).astype(float)
    out["field_goal_result"] = df["field_goal_result"].where(play_type == "field_goal")
    out["extra_point_result"] = None
    out["two_point_conv_result"] = None

    first_down = (_b(df["first_down_created"]) | _b(df["firstD_by_penalty"])).astype(float)
    out["first_down"] = first_down
    converted = (first_down == 1) | (out["touchdown"] == 1)
    for d, name in ((3, "third"), (4, "fourth")):
        on_down = scrimmage & (out["down"] == d)
        out[f"{name}_down_converted"] = (on_down & converted).astype(float)
        out[f"{name}_down_failed"] = (on_down & ~converted).astype(float)

    penalty = _b(df["penalty_flag"]) & ~_b(df["penalty_declined"]) & ~_b(df["penalty_offset"])
    pen_team = np.where(df["penalized_team"] == df["homeTeamId"], home,
                        np.where(df["penalized_team"] == df["awayTeamId"], away, None))
    out["penalty"] = penalty.astype(float)
    out["penalty_team"] = pd.Series(pen_team, index=df.index).where(penalty, None)
    out["penalty_yards"] = _num(df["yds_penalty"]).abs().where(penalty)

    for c in ("passer_player_name", "rusher_player_name", "receiver_player_name"):
        out[c] = df[c]
    two_sackers = df["sack_player_name2"].notna()
    sacked = out["sack"] == 1
    out["sack_player_name"] = df["sack_player_name"].where(sacked & ~two_sackers)
    out["half_sack_1_player_name"] = df["sack_player_name"].where(sacked & two_sackers)
    out["half_sack_2_player_name"] = df["sack_player_name2"].where(sacked & two_sackers)
    out["interception_player_name"] = df["interception_player_name"].where(out["interception"] == 1)
    out["pass_defense_1_player_name"] = df["pass_breakup_player_name"].where(live)
    out["forced_fumble_player_1_player_name"] = df["fumble_forced_player_name"].where(live)

    out["total_home_score"], out["total_away_score"] = _clean_scores(df)
    out["home_wp"] = _num(df["home_wp_before"])
    out["away_wp"] = _num(df["away_wp_before"])
    out["epa"] = _num(df["EPA"])

    # Standalone two-point tries (the alternating tries from the third OT on).
    pre_h = out["total_home_score"].groupby(gid).shift(1).where(~first_row, 0.0)
    pre_a = out["total_away_score"].groupby(gid).shift(1).where(~first_row, 0.0)
    gained = (out["total_home_score"] - pre_h) + (out["total_away_score"] - pre_a)
    out.loc[two_pt, "two_point_conv_result"] = np.where(gained[two_pt] >= 2, "success", "failure")
    out["sp"] = (gained.fillna(0) != 0).astype(float)

    out["_ord"] = np.arange(len(out), dtype=float)
    out["_try"] = False
    out["_end"] = False
    # END rows first: they copy the score after the period's last play, which
    # _try_rows() then rewrites to the touchdown's 6 points on touchdown rows.
    ends = _end_rows(out, df)
    tries = _try_rows(out, df, pre_h, pre_a)
    full = pd.concat([r for r in (out, tries, ends) if not r.empty], ignore_index=True)
    full = full.sort_values(["game_id", "_ord"], kind="stable").reset_index(drop=True)
    full["drive"] = _drives(full)
    full["play_id"] = full.groupby("game_id").cumcount().astype(float) + 1
    full = full.drop(columns=["_ord", "_try", "_end"])
    for c in columns:
        if c not in full.columns:
            full[c] = np.nan
    return full


def _clean_scores(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """The score after each play, with the feed's glitches taken out.

    Scores never go down, but ESPN's college feed has rows that say otherwise:
    a stale score on a late-listed play, a negative score, and, worst for a
    spoiler-free app, a row in the 2nd quarter carrying the game's final
    score. Per game, the longest chain of rows whose scores never go down is
    kept (runs of equal scores weighted by length); every other row is a
    glitch and gets the last good score before it."""
    h = _num(df["homeScore"]).to_numpy()
    a = _num(df["awayScore"]).to_numpy()
    out_h, out_a = np.zeros(len(df)), np.zeros(len(df))
    for idx in df.groupby("game_id", sort=False).indices.values():
        hv, av = h[idx], a[idx]
        change = np.r_[True, (hv[1:] != hv[:-1]) | (av[1:] != av[:-1])]
        starts = np.flatnonzero(change)
        width = np.diff(np.r_[starts, len(idx)])
        rh, ra = hv[starts], av[starts]
        ok = ~np.isnan(rh) & ~np.isnan(ra) & (rh >= 0) & (ra >= 0)
        best = np.full(len(starts), -1.0)
        prev = np.full(len(starts), -1)
        for j in np.flatnonzero(ok):
            fits = np.flatnonzero((best[:j] >= 0) & (rh[:j] <= rh[j]) & (ra[:j] <= ra[j]))
            # Ties go to the later run: the feed corrects itself as it goes.
            k = fits[len(fits) - 1 - np.argmax(best[fits][::-1])] if len(fits) else -1
            best[j] = width[j] + (best[k] if k >= 0 else 0)
            prev[j] = k
        keep = np.zeros(len(starts), bool)
        j = len(best) - 1 - int(np.argmax(best[::-1])) if (best >= 0).any() else -1
        while j >= 0:
            keep[j] = True
            j = prev[j]
        good_h, good_a = np.where(keep, rh, np.nan), np.where(keep, ra, np.nan)
        run_h = pd.Series(np.repeat(good_h, width)).ffill().fillna(0).to_numpy()
        run_a = pd.Series(np.repeat(good_a, width)).ffill().fillna(0).to_numpy()
        out_h[idx], out_a[idx] = run_h, run_a
    return pd.Series(out_h, index=df.index), pd.Series(out_a, index=df.index)


def _relocate_late_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Move rows the feed lists after a later period back into their own.

    ESPN sometimes appends plays it fixed up late, e.g. a run of 3rd-quarter
    plays after the final kneel-down. Left there, they'd show up after the
    game ended, carrying a stale score. Each goes back into its period, after
    the last play there whose clock shows at least as much time left."""
    df = df.reset_index(drop=True)
    gid, per = df["game_id"].to_numpy(), df["period"].to_numpy(float)
    late = per < df.groupby("game_id")["period"].cummax().to_numpy(float)
    if not late.any():
        return df
    secs = pd.to_numeric(df["start.adj_TimeSecsRem"], errors="coerce").to_numpy(float)
    order = np.arange(len(df), dtype=float)
    for i in np.flatnonzero(late):
        home = np.flatnonzero((gid == gid[i]) & (per == per[i]) & ~late)
        if len(home) == 0:
            continue
        before = home[secs[home] >= secs[i]] if not np.isnan(secs[i]) else home
        anchor = order[before[-1]] if len(before) else order[home[0]] - 1
        # Between the anchor and the next row; ties keep their feed order.
        order[i] = anchor + 0.5 + i * 1e-7
    return df.iloc[np.argsort(order, kind="stable")].reset_index(drop=True)


def _try_rows(out: pd.DataFrame, df: pd.DataFrame, pre_h: pd.Series, pre_a: pd.Series) -> pd.DataFrame:
    """Split each touchdown row in two, as nflfastR has them: the touchdown
    (worth 6) and the try. Edits the touchdown rows of `out` in place and
    returns the new try rows."""
    post_h, post_a = out["total_home_score"], out["total_away_score"]
    dh, da = post_h - pre_h, post_a - pre_a
    home_scored, away_scored = dh >= 6, da >= 6
    pat = df["pointAfterAttempt.text"]
    pts = np.where(home_scored, dh, da) - 6
    split = ((out["touchdown"] == 1) & (home_scored ^ away_scored)
             & pd.Series(pts, index=out.index).isin([0, 1, 2]) & (pat.notna() | (pts > 0)))
    if not split.any():
        return pd.DataFrame()
    idx = out.index[split]
    pts = pd.Series(pts, index=out.index)[idx]
    pat = pat[idx].fillna("Not Available")
    is_two = pat.str.startswith("Two Point") | ((pat == "Not Available") & (pts == 2))

    tries = out.loc[idx].copy()
    td_desc, try_desc = zip(*[_split_try(t, p) for t, p in zip(out.loc[idx, "desc"], pat)])
    out.loc[idx, "desc"] = list(td_desc)
    out.loc[idx, "total_home_score"] = np.where(home_scored[idx], pre_h[idx] + 6, post_h[idx])
    out.loc[idx, "total_away_score"] = np.where(away_scored[idx], pre_a[idx] + 6, post_a[idx])

    scorer = np.where(home_scored[idx], out.loc[idx, "home_team"], out.loc[idx, "away_team"])
    other = np.where(home_scored[idx], out.loc[idx, "away_team"], out.loc[idx, "home_team"])
    tries["desc"] = list(try_desc)
    tries["posteam"], tries["defteam"] = scorer, other
    tries["play_type"] = np.where(is_two, np.where(pat.str.contains("Pass"), "pass", "run"), "extra_point")
    tries["extra_point_result"] = np.where(
        is_two, None, np.where(pts == 1, "good", np.where(pat.str.contains("Blocked"), "blocked", "failed")))
    tries["two_point_conv_result"] = np.where(is_two, np.where(pts == 2, "success", "failure"), None)
    tries["yardline_100"] = 3.0
    tries["sp"] = (pts > 0).astype(float).values
    for c in ("down", "ydstogo", "yards_gained", "epa", "air_yards", "yards_after_catch",
              "passing_yards", "receiving_yards", "rushing_yards", "field_goal_result",
              "penalty_team", "penalty_yards"):
        tries[c] = np.nan
    for c in ("goal_to_go", "pass_attempt", "rush_attempt", "complete_pass", "sack", "interception",
              "qb_kneel", "qb_spike", "touchdown", "pass_touchdown", "rush_touchdown", "safety",
              "fumble_lost", "first_down", "third_down_converted", "third_down_failed",
              "fourth_down_converted", "fourth_down_failed", "penalty"):
        tries[c] = 0.0
    for c in out.columns:
        if c.endswith("_player_name"):
            tries[c] = None
    # The try's pre-snap win probability is the touchdown's after-play one.
    tries["home_wp"] = _num(df.loc[idx, "home_wp_after"]).values
    tries["away_wp"] = _num(df.loc[idx, "away_wp_after"]).values
    tries["_ord"] = out.loc[idx, "_ord"] + 0.5
    tries["_try"] = True
    return tries


def _end_rows(out: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """'END QUARTER n' after each period and 'END GAME' after a finished game,
    as nflfastR has them. The app reads them to close the drive in progress."""
    gid = out["game_id"]
    last_in_game = gid != gid.shift(-1)
    period_ends = (out["qtr"] != out["qtr"].shift(-1)) & ~last_in_game
    done = _b(df["status_type_completed"]) & last_in_game
    idx = out.index[period_ends | done]
    if len(idx) == 0:
        return pd.DataFrame()
    ends = out.loc[idx, ["game_id", "season", "week", "game_date", "home_team", "away_team",
                         "qtr", "total_home_score", "total_away_score", "home_wp", "away_wp",
                         "_ord"]].copy()
    q = ends["qtr"]
    ends["desc"] = np.where(done[idx], "END GAME",
                            np.where(q <= 4, "END QUARTER " + q.astype(int).astype(str),
                                     "END OVERTIME " + (q - 4).astype(int).astype(str)))
    ends["time"] = np.where(q <= 4, "0:00", None)
    ends["game_seconds_remaining"] = np.where(q <= 4, (4 - q) * 900, np.nan)
    ends["posteam"] = ends["defteam"] = ends["play_type"] = None
    ends["sp"] = 0.0
    ends["_ord"] += 0.75
    ends["_end"] = True
    ends["_try"] = False
    return ends


def _drives(df: pd.DataFrame) -> pd.Series:
    """Drive numbers from changes of possession, as nflfastR's fixed_drive
    counts them: a new drive whenever posteam changes, with every kickoff, and
    at the start of the second half and of each overtime period (an OT period
    can open with the team that had the ball last). Tries and admin rows join
    the drive in progress. A re-kick or an onside recovery stays in the
    kickoff's drive, as in live_feed."""
    gid = df["game_id"]
    has = df["posteam"].notna() & ~df["_try"] & ~df["_end"]
    p = df.loc[has, ["game_id", "posteam", "qtr", "play_type"]]
    same_game = p["game_id"] == p["game_id"].shift()
    kick = p["play_type"] == "kickoff"
    after_kick = kick.shift(fill_value=False) & same_game
    q_change = (p["qtr"] != p["qtr"].shift()) & ((p["qtr"] == 3) | (p["qtr"] >= 5))
    new = (~same_game) | (((p["posteam"] != p["posteam"].shift()) | kick) & ~after_kick) | (q_change & same_game)
    drive = pd.Series(np.nan, index=df.index)
    drive[has] = new.astype(int).groupby(p["game_id"]).cumsum().astype(float)
    return drive.groupby(gid).transform(lambda s: s.ffill().bfill())
