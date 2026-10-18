"""
The league's other games, ticker style, from ESPN's public scoreboard.

ESPN's scoreboard lists a league's games for the current week with the score,
the clock and, for a game in progress, who has the ball and where (its
`situation`: "2nd & 7", "CAR 35", red zone, last play). `games()` reads it
into one `TickerGame` per game; `ticker()` keeps the ones around now and
leaves out the game being watched, whose live score would spoil it, and
`find()` matches a ticker game to its row in the app's game list.

No Streamlit here: leagues.load_scoreboard() caches it and views.ticker()
draws it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
import requests

from live_feed import ESPN_TEAM_FIX

_BASE = "https://site.api.espn.com/apis/site/v2/sports/football"
# League key → (scoreboard URL, query). College asks for every FBS game
# (group 80); ESPN's default is only the Top 25's.
SCOREBOARDS = {
    "NFL": (f"{_BASE}/nfl/scoreboard", {}),
    "CFB": (f"{_BASE}/college-football/scoreboard", {"groups": "80", "limit": "400"}),
}
# ESPN → the app's team codes: nflverse's for the NFL; college uses ESPN's own.
TEAM_FIX = {"NFL": ESPN_TEAM_FIX, "CFB": {}}

# Finals and upcoming games are listed only this close to now, so Thursday's
# final or next Monday's kickoff don't crowd out the games on today.
WINDOW = pd.Timedelta(hours=12)

# The spot ESPN prints: "CAR 35", "50", or the end of "2nd & 7 at CAR 35".
_SPOT = re.compile(r"(?:^|\bat )([A-Za-z&\-]{1,6})?\s*(\d{1,2})\s*$")
_ORDER = {"in": 0, "post": 1, "pre": 2}


@dataclass(frozen=True)
class TickerTeam:
    abbr: str
    score: str              # "" before kickoff
    logo: str | None
    color: str | None       # "#rrggbb", readable on a light or dark page
    rank: int | None        # college poll rank, 1-25
    has_ball: bool
    winner: bool


@dataclass(frozen=True)
class TickerGame:
    event_id: str
    season: int | None      # ESPN's season year, as nflverse and sportsdataverse number it
    state: str              # ESPN's "pre" / "in" / "post"
    status: str             # ESPN's own: "7:12 - 2nd", "Halftime", "Final/OT", kickoff time
    kickoff: str            # ESPN's event date, ISO UTC
    away: TickerTeam
    home: TickerTeam
    down_distance: str | None = None  # "2nd & 7"
    spot: str | None = None           # "CAR 35"
    to_goal: int | None = None        # yards the offense has to go, 1-99
    red_zone: bool = False
    last_play: str | None = None
    network: str | None = None


def fetch(league: str, timeout: float = 5) -> dict:
    """ESPN's scoreboard JSON for a league key ("NFL" / "CFB"). A short
    timeout: the page waits on it before drawing the score."""
    url, params = SCOREBOARDS[league]
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _luminance(hex6: str) -> float:
    r, g, b = (int(hex6[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.299 * r + 0.587 * g + 0.114 * b


def _color(team: dict) -> str | None:
    """The team's color, or its alternate where the color is near black or
    white and would vanish on a dark or light page."""
    ok = [c for c in (team.get("color"), team.get("alternateColor"))
          if isinstance(c, str) and re.fullmatch(r"[0-9a-fA-F]{6}", c)]
    if not ok:
        return None
    mid = [c for c in ok if 0.08 < _luminance(c) < 0.92]
    return "#" + (mid or ok)[0]


def _to_goal(sit: dict, offense: str, defense: str) -> tuple[str | None, int | None]:
    """(spot as printed, yards to the goal line) from ESPN's printed spot,
    which names the side of the field, as live_feed._spot() reads it."""
    for key in ("possessionText", "downDistanceText"):
        m = _SPOT.search(str(sit.get(key) or ""))
        if not m or not 1 <= int(m[2]) <= 50 or not (m[1] or int(m[2]) == 50):
            continue
        side, yd = (m[1] or "").upper(), int(m[2])
        if yd == 50:
            return "50", 50
        if side == offense.upper():
            return f"{m[1]} {yd}", 100 - yd
        if side == defense.upper():
            return f"{m[1]} {yd}", yd
    return None, None


def parse(data: dict, team_fix: dict | None = None) -> list[TickerGame]:
    """ESPN scoreboard JSON → one TickerGame per game: in progress first,
    then final, then upcoming, each by kickoff."""
    fix = team_fix or {}
    default_season = (data.get("season") or {}).get("year")
    out = []
    for ev in data.get("events") or []:
        comp = (ev.get("competitions") or [{}])[0]
        sides = {c.get("homeAway"): c for c in comp.get("competitors") or []}
        if not {"home", "away"} <= set(sides):
            continue
        stype = ((comp.get("status") or ev.get("status") or {}).get("type") or {})
        state = stype.get("state") or "pre"
        sit = (comp.get("situation") or {}) if state == "in" else {}
        if stype.get("name") == "STATUS_HALFTIME":
            sit = {}  # whatever it says is about the second-half kickoff
        poss = str(sit.get("possession") or "")
        teams = {s: (sides[s].get("team") or {}) for s in ("home", "away")}
        raw = {s: str(t.get("abbreviation") or t.get("shortDisplayName") or "?") for s, t in teams.items()}

        def team(s: str) -> TickerTeam:
            c, t = sides[s], teams[s]
            rank = (c.get("curatedRank") or {}).get("current")
            return TickerTeam(
                abbr=fix.get(raw[s], raw[s]),
                score="" if state == "pre" else str(c.get("score") or "0"),
                logo=t.get("logo") or None, color=_color(t),
                rank=int(rank) if isinstance(rank, (int, float)) and 1 <= rank <= 25 else None,
                has_ball=bool(poss) and str(t.get("id") or c.get("id")) == poss,
                winner=bool(c.get("winner")))

        home, away = team("home"), team("away")
        spot = to_goal = dd = None
        offense = "home" if home.has_ball else "away" if away.has_ball else None
        if offense:
            defense = "away" if offense == "home" else "home"
            spot, to_goal = _to_goal(sit, raw[offense], raw[defense])
            if spot and spot != "50":
                side, yd = spot.rsplit(" ", 1)
                spot = f"{fix.get(side, side)} {yd}"
            down = sit.get("down")
            if not isinstance(down, (int, float)) or 1 <= down <= 4:
                dd = sit.get("shortDownDistanceText") or None
        names = ((comp.get("broadcasts") or [{}])[0] or {}).get("names") or []
        out.append(TickerGame(
            event_id=str(ev.get("id") or comp.get("id") or ""),
            season=_int((ev.get("season") or {}).get("year") or default_season),
            state=state,
            status=str(stype.get("shortDetail") or stype.get("detail") or stype.get("description") or ""),
            kickoff=str(ev.get("date") or comp.get("date") or ""),
            away=away, home=home, down_distance=dd, spot=spot, to_goal=to_goal,
            red_zone=bool(sit.get("isRedZone")) or (to_goal is not None and to_goal <= 20),
            last_play=(sit.get("lastPlay") or {}).get("text") or None,
            network=names[0] if names else comp.get("broadcast") or None))
    return sorted(out, key=lambda g: (_ORDER.get(g.state, 3), g.kickoff))


def games(league: str) -> list[TickerGame]:
    """The league's games on ESPN's scoreboard now (see parse)."""
    return parse(fetch(league), TEAM_FIX[league])


def _int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _event_id(v) -> str | None:
    """An ESPN event id as a string; the NFL schedule may hold it as a number."""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    try:
        return str(int(float(v)))
    except (TypeError, ValueError):
        return str(v)


def _kickoff(g: TickerGame) -> pd.Timestamp:
    return pd.to_datetime(g.kickoff, utc=True, errors="coerce")


def ticker(games: list[TickerGame], event_id, teams, now: pd.Timestamp) -> list[TickerGame]:
    """`games` for the ticker: every game in progress, finals and upcoming
    games within WINDOW of `now`, and never the game being watched, matched
    by its ESPN event id or by its two teams."""
    eid, pair = _event_id(event_id), set(teams)
    return [g for g in games
            if g.event_id != eid and {g.home.abbr, g.away.abbr} != pair
            and (g.state == "in" or abs(_kickoff(g) - now) <= WINDOW)]


def find(games: pd.DataFrame, id_col: str, game: TickerGame) -> pd.Series | None:
    """`game`'s row in a League.games() table: by ESPN event id (`id_col`),
    else by its home and away teams on a date within a day of its kickoff
    (two teams can meet twice in a season)."""
    if id_col in games.columns:
        hit = games[games[id_col].map(_event_id) == game.event_id]
        if not hit.empty:
            return hit.iloc[-1]
    kickoff = _kickoff(game)
    if pd.isna(kickoff) or "game_date" not in games.columns:
        return None
    days = (pd.to_datetime(games["game_date"], errors="coerce")
            - kickoff.tz_localize(None).normalize()).abs()
    hit = games[(games["home_team"] == game.home.abbr) & (games["away_team"] == game.away.abbr)
                & (days <= pd.Timedelta(days=1))]
    return hit.iloc[-1] if not hit.empty else None


def busy(games: list[TickerGame], now: pd.Timestamp,
         soon: pd.Timedelta = pd.Timedelta(minutes=15)) -> bool:
    """Whether a game is on, or due to kick off within `soon` (or a little
    past due): the ticker updates often only then."""
    return any(g.state == "in" or (g.state == "pre" and -WINDOW <= _kickoff(g) - now <= soon)
               for g in games)
