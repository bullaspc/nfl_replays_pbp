"""
AI game summary: an agent that explains why the score is what it is.

The agent sees exactly what the dashboard shows: the revealed plays and the
stats the app builds from them. It knows nothing past the viewer's cursor. It
starts from a snapshot (scoreboard, offense and defense tables) and calls tools
to look at key plays, drives, splits and players before it writes.

The Matchup page's preview of a game not played yet (preview()) is one call
with the page's season tables and no tools.

The model is called through the Anthropic Messages API, so any compatible
endpoint works. The default is Kimi through Moonshot's /anthropic route; set
SUMMARY_PROVIDER=claude to use Claude. This module doesn't import Streamlit;
the app passes everything in.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import anthropic
import pandas as pd

from replay_core import (LINE_COUNT_ROWS, TENDENCY_DOWNS, edge_label, has_air_yards, nullified_snaps,
                          pass_tendencies, period_label, run_tendencies)

# ---------- Providers ----------
# name → (label, default model, default base URL, key settings, key sent as Bearer).
# Each one can be overridden with <NAME>_MODEL, <NAME>_BASE_URL and <NAME>_EFFORT.
_PROVIDER_DEFAULTS = {
    "kimi": ("Kimi (Moonshot)", "kimi-k3", "https://api.moonshot.ai/anthropic",
             ("MOONSHOT_API_KEY", "KIMI_API_KEY"), True),
    "claude": ("Claude (Anthropic)", "claude-sonnet-5-5", "https://api.anthropic.com",
               ("ANTHROPIC_API_KEY",), False),
}
# Claude models whose Claude API requests take the server-side refusal fallback.
_FALLBACK_MODELS = ("claude-sonnet-5-5", "claude-opus-5", "claude-fable-5")


@dataclass(frozen=True)
class Provider:
    name: str
    label: str
    model: str
    base_url: str
    api_key: str
    bearer: bool        # Moonshot wants Authorization: Bearer, Anthropic wants x-api-key
    effort: str         # Claude only; other endpoints don't get the parameter

    @property
    def is_claude(self) -> bool:
        return self.model.startswith("claude-")


def configured_providers(get: Callable[[str], str | None]) -> dict[str, Provider]:
    """Every provider that has an API key, with the default one first.
    `get` reads a setting (st.secrets, then the environment). The default is
    SUMMARY_PROVIDER, or kimi if that isn't set."""
    out = {}
    for name, (label, model, base_url, key_names, bearer) in _PROVIDER_DEFAULTS.items():
        key = next((get(k) for k in key_names if get(k)), None)
        if not key:
            continue
        up = name.upper()
        out[name] = Provider(name, label, get(f"{up}_MODEL") or model,
                             get(f"{up}_BASE_URL") or base_url, key, bearer,
                             get(f"{up}_EFFORT") or "medium")
    first = (get("SUMMARY_PROVIDER") or "kimi").lower()
    return dict(sorted(out.items(), key=lambda kv: kv[0] != first))


def _client(p: Provider) -> anthropic.Anthropic:
    # An explicit credential stops the SDK reading ANTHROPIC_API_KEY from the
    # environment, so an Anthropic key is never sent to another endpoint.
    cred = {"auth_token": p.api_key} if p.bearer else {"api_key": p.api_key}
    return anthropic.Anthropic(base_url=p.base_url, timeout=180.0, max_retries=2, **cred)


def _create(client: anthropic.Anthropic, p: Provider, **kw):
    """One Messages call. Compatible endpoints get the plain request. Claude
    also gets effort, prompt caching and, on the Claude API, the server-side
    refusal fallback."""
    if not p.is_claude:
        return client.messages.create(model=p.model, **kw)
    kw.update(output_config={"effort": p.effort}, cache_control={"type": "ephemeral"})
    if p.model.startswith(_FALLBACK_MODELS) and "api.anthropic.com" in p.base_url:
        kw.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        return client.beta.messages.create(model=p.model, **kw)
    return client.messages.create(model=p.model, **kw)


def _call(client: anthropic.Anthropic, p: Provider, **kw):
    """_create(), with the API's errors as SummaryErrors saying what went wrong."""
    try:
        resp = _create(client, p, **kw)
    except anthropic.AuthenticationError:
        raise SummaryError(f"{p.label} rejected the API key.") from None
    except anthropic.NotFoundError:
        raise SummaryError(f"{p.base_url} has no model {p.model!r}. "
                           f"Set {p.name.upper()}_MODEL.") from None
    except anthropic.RateLimitError:
        raise SummaryError(f"{p.label} is rate-limiting requests. "
                           "Try again in a minute.") from None
    except anthropic.APIStatusError as e:
        raise SummaryError(f"{p.label} returned an error "
                           f"({e.status_code}): {e.message}") from None
    except anthropic.APIConnectionError:
        raise SummaryError(f"Couldn't reach {p.base_url}.") from None
    if resp.stop_reason == "refusal":
        raise SummaryError("The model declined to write this summary.")
    return resp


def _final_text(resp) -> str:
    """A reply's text, marked if it hit the length limit."""
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    if not text:
        raise SummaryError(f"The model returned no text (stop reason: {resp.stop_reason}).")
    if resp.stop_reason == "max_tokens":
        text += "\n\n*(Cut off at the length limit.)*"
    return text


# ---------- What the agent can see ----------
@dataclass
class GameContext:
    """Everything the agent may look at. Every frame is built from `revealed`,
    the plays up to the viewer's cursor, never from the full game."""
    home: str
    away: str
    revealed: pd.DataFrame
    boxscore: pd.DataFrame
    scoring: pd.DataFrame
    team_stats: pd.DataFrame        # the Team stats table: stat × [away, home]
    team_pct: pd.DataFrame          # same shape: percentile vs the last 3 seasons, 100 = best
    situational: pd.DataFrame       # the Situational success rates table
    situational_counts: dict        # (situation, team) → {"Pass": n, "Rush": n}
    drives: pd.DataFrame            # the drive table, latest drive first
    top_wpa: pd.DataFrame           # Top plays by win probability added
    explosive: pd.DataFrame         # Explosive plays (rows of `revealed`)
    leaders: dict[tuple[str, str], pd.DataFrame] = field(default_factory=dict)
    # (team, "passing" | "rushing" | "receiving" | "defense") → the Player leaders table
    league: str = "NFL"             # "NFL" or "CFB"
    line_play: pd.DataFrame | None = None  # the Line play table: stat × [away, home]
    line_pct: pd.DataFrame | None = None   # same shape: percentile, 100 = best for the line
    run_gaps: bool = True           # the play text names a run's gap (NFL), not just its side
    # TENDENCY_DOWNS option → league-wide tendencies (replay_core.tendency_baselines)
    tendency_baselines: dict = field(default_factory=dict)


# What changes between the leagues: who's talking, whose models, and what the
# percentiles compare against.
_LEAGUE = {
    "NFL": {
        "analyst": "an NFL analyst",
        "model": "nflfastR model",
        "baseline": "every team-game of the last 3 seasons",
        "defense": "tackles, sacks, QB hits, TFLs, INTs, passes defensed and forced fumbles",
        "notes": "",
    },
    "CFB": {
        "analyst": "a college football analyst",
        "model": "sportsdataverse's college model",
        "baseline": "every FBS-vs-FBS team-game of last season",
        "defense": ("tackles, sacks, QB hurries, TFLs, INTs, passes defensed and forced fumbles (the "
                    "college feed has no QB hits, and older games' feeds have no tackles, TFLs or hurries)"),
        "notes": (" College overtime is untimed: each team gets a possession from the "
                  "opponent's 25, and from the third overtime on, teams trade two-point tries."),
    },
}


def _csv(df: pd.DataFrame | None, index: bool = False, float_format: str = "%.3g") -> str:
    """Compact table text for the model."""
    if df is None or df.empty:
        return "(none)"
    return df.to_csv(index=index, float_format=float_format).strip()


def game_status(revealed: pd.DataFrame) -> tuple[str, bool]:
    """('Q3 7:32', False), or ('Final', True) once the END GAME row is revealed."""
    last = revealed.iloc[-1]
    if str(last["desc"] or "").upper().startswith("END GAME"):
        q = last["qtr"]
        return ("Final" if pd.isna(q) or q <= 4 else f"Final/{period_label(q)}"), True
    q = int(last["qtr"]) if pd.notna(last["qtr"]) else 1
    clock = f" {last['time']}" if pd.notna(last["time"]) else ""
    return f"{period_label(q)}{clock}", False


_PCT_STATS = {"CMP%", "Pass SR", "Rush SR", "1st Down %", "3rd Down %", "RZ TD%"}
_EPA_STATS = {"Pass EPA/play", "Rush EPA/play", "EPA/play"}


def _fmt_stat(stat: str, v) -> str:
    if pd.isna(v):
        return "-"
    if stat in _PCT_STATS:
        return f"{v:.0%}"
    if stat in _EPA_STATS:
        return f"{v:+.2f}"
    if stat == "aDoT":
        return f"{v:.1f}"
    if stat == "TOP":
        return f"{int(v) // 60}:{int(v) % 60:02d}"
    return f"{v:.0f}"


def offense_table(ctx: GameContext) -> pd.DataFrame:
    """The Team stats table with a percentile column per team."""
    rows = []
    for stat in ctx.team_stats.index:
        row = {"Stat": stat}
        for team in (ctx.away, ctx.home):
            row[team] = _fmt_stat(stat, ctx.team_stats.at[stat, team])
            pct = ctx.team_pct.at[stat, team] if stat in ctx.team_pct.index else float("nan")
            row[f"{team} pct"] = "" if pd.isna(pct) else f"{pct:.0f}"
        rows.append(row)
    return pd.DataFrame(rows)


def line_table(ctx: GameContext) -> pd.DataFrame:
    """The Line play table with a percentile column per team (none on the
    play counts, which are only context)."""
    rows = []
    for stat in ctx.line_play.index:
        row = {"Stat": stat}
        for team in (ctx.away, ctx.home):
            v = ctx.line_play.at[stat, team]
            row[team] = "-" if pd.isna(v) else (f"{v:.0f}" if stat in LINE_COUNT_ROWS else f"{v:.1%}")
            pct = (float("nan") if stat in LINE_COUNT_ROWS or ctx.line_pct is None
                   or stat not in ctx.line_pct.index else ctx.line_pct.at[stat, team])
            row[f"{team} pct"] = "" if pd.isna(pct) else f"{pct:.0f}"
        rows.append(row)
    return pd.DataFrame(rows)


_TEND_DOWNS = {"all": "All downs", "early": "Early downs", "late": "Late downs"}


def tendency_table(ctx: GameContext, downs: str = "all") -> pd.DataFrame:
    """The Tendencies section as one table: a row per run gap (or side) and
    pass depth × side, each team's plays, share, yds/play, success rate and
    EPA/play there, and the league's share and EPA/play."""
    label = _TEND_DOWNS[downs]
    rng = TENDENCY_DOWNS[label]
    base = ctx.tendency_baselines.get(label, {})
    air = has_air_yards(ctx.revealed)
    parts = []
    for kind, tables, base in (
            ("Run", {t: run_tendencies(ctx.revealed, t, ctx.run_gaps, rng) for t in (ctx.away, ctx.home)},
             base.get("runs")),
            ("Pass", {t: pass_tendencies(ctx.revealed, t, rng, air) for t in (ctx.away, ctx.home)},
             base.get("air" if air else "text"))):
        for key in next(iter(tables.values())).index:
            row = {"Play": kind, "Where": " ".join(key) if isinstance(key, tuple) else key}
            for team, df in tables.items():
                r = df.loc[key]
                n = int(r["Plays"])
                row[f"{team} plays"] = n
                row[f"{team} share"] = f"{r['Share']:.0%}" if n else "-"
                row[f"{team} yds/play"] = f"{r['Yds/Play']:.1f}" if n else "-"
                row[f"{team} SR"] = f"{r['SR%']:.0%}" if n else "-"
                row[f"{team} EPA/play"] = f"{r['EPA/play']:+.2f}" if n else "-"
            known = base is not None and key in base.index
            row["league share"] = f"{base.at[key, 'Share']:.0%}" if known else "-"
            row["league EPA/play"] = f"{base.at[key, 'EPA/play']:+.2f}" if known else "-"
            parts.append(row)
    return pd.DataFrame(parts)


def _tendency_note(ctx: GameContext) -> str:
    runs = ("by gap: LE/LT/LG = left end/tackle/guard, Mid, RG/RT/RE" if ctx.run_gaps
            else "by side (the college text names no gap)")
    depth = ("by air yards past the line of scrimmage (Deep 20+, Inter 10-19, Short 0-9, Behind LOS)"
             if has_air_yards(ctx.revealed) else "by the play text's short/deep (this feed has no air yards)")
    ints = "" if ctx.run_gaps else " Interceptions name no direction in the college text, so they're missing."
    return (f"From the offense's side (left = the quarterback's left). Designed runs (no kneels or "
            f"scrambles) {runs}; passes thrown (no sacks) {depth} × side. Shares are of the plays "
            f"whose text names a direction; plays without one are left out.{ints} League = "
            f"{_LEAGUE[ctx.league]['baseline']}, all plays together.")


def defense_table(ctx: GameContext) -> pd.DataFrame:
    """What each defense allowed (its opponent's offense, with the same play
    masks as the Team stats table) plus the defense's own pressure, stops and
    takeaways."""
    r = ctx.revealed
    last = r.iloc[-1]
    pts = {ctx.home: int(last["total_home_score"] or 0), ctx.away: int(last["total_away_score"] or 0)}
    cols = {}
    for team, opp in ((ctx.away, ctx.home), (ctx.home, ctx.away)):
        d = r[r["defteam"] == team]
        pm = d["pass_attempt"].fillna(0) == 1
        rm = d["rush_attempt"].fillna(0) == 1
        sc = d[pm | rm]
        # EPA and success rate count the snaps a penalty wiped out, as Team stats do.
        null_pass, null_rush = nullified_snaps(d)
        epa = d.loc[pm | rm | null_pass | null_rush, "epa"].fillna(0)
        conv = int(d["third_down_converted"].fillna(0).sum())
        fail = int(d["third_down_failed"].fillna(0).sum())
        rz = sc[sc["yardline_100"].fillna(100) <= 20]
        expl = int(((d["passing_yards"].fillna(0) >= 15) & pm).sum()
                   + ((d["rushing_yards"].fillna(0) >= 10) & rm).sum())
        stuffs = int((d.loc[rm, "yards_gained"].fillna(0) <= 0).sum())
        pds = int(d[["pass_defense_1_player_name", "pass_defense_2_player_name"]].notna().sum().sum())
        pen = r[(r["penalty_team"] == team) & (r["defteam"] == team)]
        cols[team] = {
            "Points allowed (all phases)": str(pts[opp]),
            "Plays faced": str(len(sc)),
            "Yds/play allowed": "-" if sc.empty else f"{sc['yards_gained'].fillna(0).mean():.1f}",
            "EPA/play allowed": "-" if sc.empty else f"{epa.mean():+.2f}",
            "Success rate allowed": "-" if sc.empty else f"{(epa > 0).mean():.0%}",
            "Pass EPA/play allowed": "-" if not pm.any() else f"{d.loc[pm | null_pass, 'epa'].fillna(0).mean():+.2f}",
            "Rush EPA/play allowed": "-" if not rm.any() else f"{d.loc[rm | null_rush, 'epa'].fillna(0).mean():+.2f}",
            "Explosive plays allowed (pass 15+, run 10+)": str(expl),
            "3rd-down stops": f"{fail} of {conv + fail}",
            "Red-zone plays faced / TDs allowed": f"{len(rz)} / {int(rz['touchdown'].fillna(0).sum())}",
            "Sacks": f"{d['sack'].fillna(0).sum():.0f}",
            # The college feed has no QB hits.
            **({} if ctx.league == "CFB" else {"QB hits": f"{d['qb_hit'].fillna(0).sum():.0f}"}),
            "Runs stuffed (0 or fewer yds)": str(stuffs),
            "Passes defensed": str(pds),
            "Takeaways": f"{d['interception'].fillna(0).sum() + d['fumble_lost'].fillna(0).sum():.0f}",
            "Penalties on defense (yds)": f"{len(pen)} ({pen['penalty_yards'].fillna(0).sum():.0f})",
        }
    out = pd.DataFrame(cols)
    out.index.name = "Stat"
    return out.reset_index()


def snapshot(ctx: GameContext) -> str:
    """The dashboard as text: the agent's starting point."""
    status, final = game_status(ctx.revealed)
    last = ctx.revealed.iloc[-1]
    hs, as_ = int(last["total_home_score"] or 0), int(last["total_away_score"] or 0)
    where = (f"{status}: the viewer has watched the whole game." if final else
             f"{status}, as far as the viewer has watched. You can't see anything after it.")
    wp = ctx.revealed["home_wp"].dropna()
    wp_line = (f"\nHome ({ctx.home}) win probability now: {wp.iloc[-1]:.0%} ({_LEAGUE[ctx.league]['model']})"
               if not final and not wp.empty else "")
    line = "" if ctx.line_play is None or ctx.line_play.empty else f"""

Line play: each offense's line, so a team's column is also what the other team's defensive front did against it. Dropbacks include sacks and QB hits include sacks; designed runs leave out kneels and scrambles, stuffed = no gain or a loss, explosive = 10+ yards; OL penalties are holding, false starts and ineligible man downfield on the offense, per snap. "pct" is against the same baseline, 100 = best for the offensive line (for Sack %, QB Hit %, Stuff % and OL Penalty %, lower is better).
{_csv(line_table(ctx))}"""
    tend = tendency_table(ctx)
    tend = "" if tend.empty or not tend.filter(like=" plays").to_numpy().any() else f"""

Tendencies, all downs: where each offense ran and threw. {_tendency_note(ctx)} get_tendencies has early and late downs.
{_csv(tend)}"""
    return f"""Game: {ctx.away} (away) at {ctx.home} (home)
Viewer's position: {where}
Score: {ctx.away} {as_}, {ctx.home} {hs}{wp_line}

Score by quarter:
{_csv(ctx.boxscore)}

Scoring plays:
{_csv(ctx.scoring)}

Offense: the dashboard's Team stats. "pct" is the percentile against {_LEAGUE[ctx.league]['baseline']} (100 = best; for Turnovers, Lost Poss. and Penalties, fewer is better; Lost Poss. is turnovers plus turnovers on downs and missed or blocked field goals).
{_csv(offense_table(ctx))}

Defense: what each defense allowed, and its own pressure, stops and takeaways.
{_csv(defense_table(ctx))}{line}{tend}"""


# ---------- Tools ----------
def _plays(ctx: GameContext) -> pd.DataFrame:
    """Revealed plays in a readable layout. home_wpa is the change in the home
    team's win probability; the last revealed play has none, because that
    would need the next play."""
    r = ctx.revealed
    dd = [f"{int(d)}&{int(t)}" if pd.notna(d) and pd.notna(t) else ""
          for d, t in zip(r["down"], r["ydstogo"])]
    q = r["qtr"].map(lambda x: "" if pd.isna(x) else ("OT" if x >= 5 else f"Q{int(x)}"))
    return pd.DataFrame({
        "drive": r["drive"], "qtr": q, "clock": r["time"], "off": r["posteam"],
        "down_dist": dd, "yds_to_goal": r["yardline_100"], "yds": r["yards_gained"],
        "epa": r["epa"].round(2), "home_wpa": (r["home_wp"].shift(-1) - r["home_wp"]).round(3),
        "desc": r["desc"].fillna("").str.slice(0, 180),
    }, index=r.index)


_KEY_PLAY_KINDS = {
    "win_probability": "biggest win-probability swings, the dashboard's Top plays by WPA "
                       "(team = the team the swing favored)",
    "explosive": "passes of 15+ and runs of 10+ yards (the dashboard's Explosive plays)",
    "turnovers": "interceptions and lost fumbles",
    "sacks": "sacks",
    "fourth_downs": "4th downs the offense went for",
    "red_zone": "snaps inside the opponent's 20",
    "penalties": "accepted penalties (team = penalized team)",
}


def _key_plays(ctx: GameContext, kind: str, team: str, limit: int) -> str:
    if kind == "win_probability":
        df = ctx.top_wpa.drop(columns=["_abs_wpa"], errors="ignore")
        if team != "both":
            df = df[df["For"] == team]
        return _csv(df.head(limit))
    p, r = _plays(ctx), ctx.revealed
    scrimmage = (r["pass_attempt"].fillna(0) == 1) | (r["rush_attempt"].fillna(0) == 1)
    if kind == "explosive":
        p = p.loc[p.index.intersection(ctx.explosive.index)].sort_values("yds", ascending=False)
    elif kind == "turnovers":
        p = p[(r["interception"].fillna(0) == 1) | (r["fumble_lost"].fillna(0) == 1)]
    elif kind == "sacks":
        p = p[r["sack"].fillna(0) == 1]
    elif kind == "fourth_downs":
        p = p[(r["down"] == 4) & scrimmage]
    elif kind == "red_zone":
        p = p[(r["yardline_100"].fillna(100) <= 20) & scrimmage]
    elif kind == "penalties":
        p = p[r["penalty"].fillna(0) == 1].assign(
            penalized=r["penalty_team"], pen_yds=r["penalty_yards"])
        if team != "both":
            p = p[p["penalized"] == team]
        return _csv(p.head(limit))
    else:
        return f"Unknown kind {kind!r}."
    if team != "both":
        p = p[p["off"] == team]
    return _csv(p.head(limit))


def _drives(ctx: GameContext, team: str) -> str:
    df = ctx.drives.copy()
    if df.empty:
        return "(none)"
    # The newest drive may still be going: label it so, unless an END row closed it.
    last_desc = str(ctx.revealed["desc"].iloc[-1] or "").upper()
    if df.at[0, "Outcome"] == "EOH/EOG" and not last_desc.startswith("END "):
        df.at[0, "Outcome"] = "In progress"
    if team != "both":
        df = df[df["Team"] == team]
    return _csv(df.sort_values("Drive"))


def _drive_plays(ctx: GameContext, drive: int) -> str:
    p = _plays(ctx)
    return _csv(p[p["drive"] == drive].drop(columns=["drive"]))


_SPLITS = ("quarter", "down", "field_zone", "play_type")


def _splits(ctx: GameContext, team: str, by: str) -> str:
    r = ctx.revealed
    pm = r["pass_attempt"].fillna(0) == 1
    rm = r["rush_attempt"].fillna(0) == 1
    d = r[(r["posteam"] == team) & (pm | rm)].copy()
    if d.empty:
        return "(no plays)"
    d["is_pass"] = pm[d.index].astype(int)
    keys = {
        "quarter": d["qtr"].map(lambda q: "OT" if q >= 5 else f"Q{int(q)}"),
        "down": d["down"].map(lambda x: "" if pd.isna(x) else f"{int(x)}"),
        "field_zone": pd.cut(d["yardline_100"], [0, 20, 50, 80, 100],
                             labels=["red zone (opp 1-20)", "opp 21-50", "own 21-49",
                                     "own 1-20"]).astype(str),
        "play_type": d["is_pass"].map({1: "pass", 0: "run"}),
    }
    g = d.groupby(keys[by], sort=False).agg(
        plays=("epa", "size"), yds_per_play=("yards_gained", "mean"),
        epa_per_play=("epa", lambda s: s.fillna(0).mean()),
        success_rate=("epa", lambda s: (s.fillna(0) > 0).mean()),
        pass_share=("is_pass", "mean"), first_downs=("first_down", "sum"))
    g.index.name = by
    return _csv(g.reset_index())


def _situational(ctx: GameContext) -> str:
    """The Situational success rates table, each cell with its play count."""
    out = pd.DataFrame(index=ctx.situational.index)
    for col in ctx.situational.columns:
        team, metric = col.split(" ", 1)
        kind = "Pass" if metric.startswith("Pass") else "Rush"
        cells = []
        for sit, v in ctx.situational[col].items():
            n = ctx.situational_counts.get((sit, team), {}).get(kind, 0)
            cells.append("-" if pd.isna(v) else
                         (f"{v:.0f}% ({n})" if "SR" in metric else f"{v:+.2f} ({n})"))
        out[col] = cells
    return _csv(out, index=True)


def _tendencies(ctx: GameContext, downs: str) -> str:
    df = tendency_table(ctx, downs if downs in _TEND_DOWNS else "all")
    if df.empty or not df.filter(like=" plays").to_numpy().any():
        return "(no plays with a direction)"
    return _csv(df)


def _leaders(ctx: GameContext, team: str, unit: str) -> str:
    return _csv(ctx.leaders.get((team, unit)))


def _tools(ctx: GameContext, strict: bool) -> list[dict]:
    team = {"type": "string", "enum": [ctx.away, ctx.home, "both"]}
    specs = [
        ("get_key_plays",
         "Revealed plays of one kind, with quarter, clock, down & distance, yards, "
         "EPA, home-team WPA and the play text. Kinds: "
         + "; ".join(f"{k}: {v}" for k, v in _KEY_PLAY_KINDS.items())
         + ". Otherwise team means the team with the ball.",
         {"kind": {"type": "string", "enum": list(_KEY_PLAY_KINDS)}, "team": team,
          "limit": {"type": "integer", "description": "Most rows to return (1-25)."}},
         ["kind", "team", "limit"]),
        ("get_drives",
         "The drive table: start spot, plays, pass/run split, yards, first downs and "
         "how each drive ended.",
         {"team": team}, ["team"]),
        ("get_drive_plays", "Every play of one drive, in order.",
         {"drive": {"type": "integer", "description": "Drive number from get_drives."}},
         ["drive"]),
        ("get_splits",
         "One team's offense split by quarter, down, field zone or play type: plays, "
         "yards/play, EPA/play, success rate, pass share, first downs. Shows where "
         "an offense worked or stalled, and how the defense facing it held up.",
         {"team": {"type": "string", "enum": [ctx.away, ctx.home]},
          "by": {"type": "string", "enum": list(_SPLITS)}},
         ["team", "by"]),
        ("get_situational_success",
         "The dashboard's Situational success rates: pass/run success rate and "
         "EPA/play by early or late down and short, medium or long distance, for "
         "both teams. Each cell has its play count in brackets.",
         {}, []),
        ("get_tendencies",
         "The dashboard's Tendencies on early downs (1st-2nd), late downs (3rd-4th) or "
         "all: each offense's run share, yds/play, success rate and EPA/play by gap or "
         "side, and pass share and results by depth × side, next to the league's. "
         + _tendency_note(ctx),
         {"downs": {"type": "string", "enum": list(_TEND_DOWNS)}}, ["downs"]),
        ("get_player_leaders",
         "The dashboard's Player leaders for one team and unit. Defense lists "
         + _LEAGUE[ctx.league]["defense"] + ".",
         {"team": {"type": "string", "enum": [ctx.away, ctx.home]},
          "unit": {"type": "string", "enum": ["passing", "rushing", "receiving", "defense"]}},
         ["team", "unit"]),
    ]
    tools = []
    for name, desc, props, required in specs:
        t = {"name": name, "description": desc,
             "input_schema": {"type": "object", "properties": props,
                              "required": required, "additionalProperties": False}}
        if strict:
            t["strict"] = True
        tools.append(t)
    return tools


def _run_tool(ctx: GameContext, name: str, args: dict) -> str:
    if name == "get_key_plays":
        limit = max(1, min(int(args.get("limit") or 10), 25))
        return _key_plays(ctx, args["kind"], args.get("team") or "both", limit)
    if name == "get_drives":
        return _drives(ctx, args.get("team") or "both")
    if name == "get_drive_plays":
        return _drive_plays(ctx, int(args["drive"]))
    if name == "get_splits":
        return _splits(ctx, args["team"], args["by"])
    if name == "get_situational_success":
        return _situational(ctx)
    if name == "get_tendencies":
        return _tendencies(ctx, args.get("downs") or "all")
    if name == "get_player_leaders":
        return _leaders(ctx, args["team"], args["unit"])
    return f"Unknown tool {name!r}."


# ---------- The agent ----------
SYSTEM_PROMPT = """You are {analyst} inside a spoiler-free replay app. The fan is watching this game on tape delay. Explain why the score is what it is at their current position: the factors, key plays and events behind it, covering offense and defense for both teams.

Ground rules:
- Use only the numbers in the dashboard snapshot and in tool results. Don't invent stats, players or plays, and don't bring in outside knowledge of these teams, their season or this game.
- Your data stops at the viewer's current play. If the game isn't over, describe where it stands. Never predict, hint at or speculate about what happens next or how it ends.
- Support each claim with numbers from the data: EPA/play, success rate, 3rd-down rate, red-zone results, turnovers, sacks and pressure, line play, where each offense ran and threw (and how it worked against the league's split), explosive plays, field position, WPA. Use the percentiles to say whether a number is actually good or bad. Name the players who drove it, from the player leaders and play text.
- Look at the key plays (turnovers, biggest win-probability swings, explosive plays, 4th downs, red-zone trips) with the tools before writing about them. Make several tool calls in one turn when you need several things. Skip tools that won't change the story.

Terms: EPA is expected points added by a play ({model}). A successful play has positive EPA; EPA/play and success rate, the team's and the players', also count a pass or run a penalty wiped out (its EPA is the penalty's). WPA is the change in the home team's win probability on a play. A defense's numbers are what it allowed.{notes}

Write the summary in Markdown, about 350 to 500 words, with this structure:
**Headline**: one or two sentences with the score and the main reason for it.
### What's deciding it
(Use "What decided it" when the game is final.) Three or four bullets, the biggest factor first, each backed by numbers.
### <away team>
- **Offense:** what worked or didn't, and why.
- **Defense:** how it held up against the other offense.
### <home team>
The same two bullets.
### Key plays
Three to five bullets: quarter and clock, what happened, and why it mattered (points or WPA).

Write only the summary, without a preamble or notes about the data or tools."""

MAX_TURNS = 8


def system_prompt(league: str) -> str:
    t = _LEAGUE[league]
    return SYSTEM_PROMPT.format(analyst=t["analyst"], model=t["model"], notes=t["notes"])


@dataclass
class Summary:
    text: str
    model: str
    tools_used: list[str]


class SummaryError(Exception):
    pass


@dataclass
class Progress:
    """What the agent is doing, for the page to show while it works. The
    worker thread writes it and the page reads it: single attribute writes
    and list appends, so no lock is needed."""
    status: str = "Reading the dashboard"
    calls: int = 0                                   # model calls made so far
    steps: list[str] = field(default_factory=list)   # lookups done, in order


_KIND_LABELS = {
    "win_probability": "biggest win-probability swings", "explosive": "explosive plays",
    "turnovers": "turnovers", "sacks": "sacks", "fourth_downs": "4th-down tries",
    "red_zone": "red-zone snaps", "penalties": "penalties",
}


def describe_lookup(name: str, args: dict) -> str:
    """A tool call as a short progress line, e.g. "Looking at ALA turnovers"."""
    team = args.get("team")
    whose = "" if team in (None, "both") else f"{team} "
    if name == "get_key_plays":
        return f"Looking at {whose}{_KIND_LABELS.get(args.get('kind'), 'key plays')}"
    if name == "get_drives":
        return f"Going through {whose}drives"
    if name == "get_drive_plays":
        return f"Replaying drive {args.get('drive')}"
    if name == "get_splits":
        return f"Splitting {team}'s offense by {str(args.get('by', '')).replace('_', ' ')}"
    if name == "get_situational_success":
        return "Checking success rates by down and distance"
    if name == "get_tendencies":
        downs = args.get("downs")
        return "Checking where the offenses run and throw" + (
            f" on {downs} downs" if downs in ("early", "late") else "")
    if name == "get_player_leaders":
        return f"Checking {team}'s {args.get('unit')} leaders"
    return f"Looking up {name}"


# Summaries are written on these threads, not on Streamlit's script thread. A
# rerun (a live game's auto-refresh every 10-30 s, or any click) stops the
# running script and starts a new one straight away (runner.fastReruns), which
# would drop a summary still being written. Module-level, so the pool outlives
# reruns; shared by every session.
_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="game-summary")


def submit(ctx: GameContext, provider: Provider) -> tuple[Future, Progress]:
    """Start summarize() on a worker thread. The Future's result is the
    Summary, or it raises the SummaryError; Progress says what it's doing."""
    progress = Progress()
    return _POOL.submit(summarize, ctx, provider, progress), progress


def summarize(ctx: GameContext, provider: Provider, progress: Progress | None = None) -> Summary:
    """Run the agent to its final answer, reporting each step to `progress`.
    Makes no Streamlit calls, so it can run on a worker thread (submit())."""
    progress = progress or Progress()
    client = _client(provider)
    tools = _tools(ctx, strict=provider.is_claude)
    messages: list[dict] = [{"role": "user", "content": (
        snapshot(ctx) + f"\n\nExplain the score so far: what is driving it, on offense and "
        f"defense, for {ctx.away} and {ctx.home}. Away team first.")}]
    used: list[str] = []
    for turn in range(MAX_TURNS):
        progress.calls = turn + 1
        progress.status = ("Reading the dashboard" if turn == 0
                           else "Writing the summary" if turn == MAX_TURNS - 1
                           else "Thinking it over")
        resp = _call(client, provider, max_tokens=16000, system=system_prompt(ctx.league),
                     tools=tools, messages=messages)
        calls = [b for b in resp.content if b.type == "tool_use"]
        if resp.stop_reason != "tool_use" or not calls:
            return Summary(_final_text(resp), resp.model, used)
        # Send the whole assistant turn back unchanged: thinking blocks included.
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for c in calls:
            used.append(c.name)
            step = describe_lookup(c.name, dict(c.input or {}))
            progress.status = step
            try:
                out, err = _run_tool(ctx, c.name, dict(c.input or {})), False
            except KeyError as e:  # bad arguments: tell the model, let it retry
                out, err = f"Missing argument {e}.", True
            except Exception as e:
                out, err = f"Error: {e}", True
            results.append({"type": "tool_result", "tool_use_id": c.id,
                            "content": out, "is_error": err})
            progress.steps.append(step)
        if turn == MAX_TURNS - 2:
            results.append({"type": "text", "text": "That's all the lookups available. "
                            "Write the summary now from what you have."})
        messages.append({"role": "user", "content": results})
    raise SummaryError("The model kept looking things up and never wrote the summary.")



# ---------- Matchup preview ----------
# A short preview of a game not played yet, from the Matchup page's season
# tables: one model call, no tools. It has no spoiler gate to keep; it only
# sees finished games.
@dataclass
class MatchupContext:
    """What the preview may look at: the Matchup page's tables."""
    league: str                     # "NFL" or "CFB"
    away: str
    home: str
    when: str                       # "Week 7, Sun 2026-10-25", plus "neutral site"
    line: str                       # "CHI favored by 3, total 50.5"; "" without one
    selection: str                  # which games and how the season numbers are taken
    n_teams: int                    # how many teams the places are among
    # offense team → replay_core.matchup_table / situational_matchup against
    # the other team's defense
    matchups: dict[str, pd.DataFrame]
    situational: dict[str, pd.DataFrame]
    # (team, "passing" | "rushing" | "receiving" | "defense") → season leaders,
    # with a Status column for players on the injury report
    leaders: dict[tuple[str, str], pd.DataFrame] = field(default_factory=dict)
    # team → its injury report for the game's week (replay_core.injury_report,
    # shown columns) and that report by unit (availability_summary); empty
    # when there's none
    injuries: dict[str, pd.DataFrame] = field(default_factory=dict)
    injury_units: dict[str, pd.DataFrame] = field(default_factory=dict)
    injury_note: str = ""           # which report and when; "" without reports
    # team → replay_core.lineup_groups (how far each group is from the players
    # behind its season numbers) and the players behind it that the injury
    # report doesn't show (off the roster, or back); empty without snap counts
    lineup_groups: dict[str, pd.Series] = field(default_factory=dict)
    lineup_players: dict[str, pd.DataFrame] = field(default_factory=dict)


def _fmt_season(stat: str, v) -> str:
    if pd.isna(v):
        return "-"
    if stat in _EPA_STATS:
        return f"{v:+.2f}"
    if stat in _PCT_STATS or stat.endswith("%"):
        return f"{v:.1%}"
    return f"{v:.1f}"  # aDoT and the per-game counts


def matchup_rows(df: pd.DataFrame, offense: str, defense: str) -> pd.DataFrame:
    """A replay_core.matchup_table as text, the way the page shows it."""
    return pd.DataFrame([{
        "Stat": stat,
        f"{offense} offense": _fmt_season(stat, r["Offense"]), f"{offense} place": r["Off rank"],
        f"{defense} defense allowed": _fmt_season(stat, r["Defense"]),
        f"{defense} place": r["Def rank"],
        "edge": edge_label(r["Edge"], offense, defense),
        "proj": _fmt_season(stat, r["Proj."]),
        **({"proj w/ lineup": _fmt_season(stat, r["w/ lineup"])} if "w/ lineup" in df else {}),
        "league": _fmt_season(stat, r["League"]),
    } for stat, r in df.iterrows()])


def situational_rows(df: pd.DataFrame, offense: str, defense: str) -> pd.DataFrame:
    """A replay_core.situational_matchup as text: success rates with their
    plays in brackets, and the edges."""
    rows = []
    for sit, r in df.iterrows():
        row = {"Situation": sit}
        for m, kind in (("Pass SR%", "pass"), ("Rush SR%", "run")):
            for side, who in (("O", f"{offense} {kind} SR"), ("D", f"{defense} {kind} SR allowed")):
                v, n = r[f"{m} {side}"], r[f"_{m} {side} n"]
                row[who] = "-" if pd.isna(v) else f"{v:.0f}% ({int(n)})"
            row[f"{kind} edge"] = edge_label(r[f"{m} edge"], offense, defense)
        rows.append(row)
    return pd.DataFrame(rows)


def matchup_snapshot(ctx: MatchupContext) -> str:
    """The Matchup page as text."""
    parts = [f"Game: {ctx.away} (away) at {ctx.home} (home), {ctx.when}"]
    if ctx.line:
        parts.append(f"Betting line: {ctx.line}")
    parts.append(f"Season numbers: {ctx.selection}. Places are among the {ctx.n_teams} teams; "
                 "in the matchup tables, counts and yards are per game.")
    for off, def_ in ((ctx.away, ctx.home), (ctx.home, ctx.away)):
        parts.append(f"When {off} has the ball: {off}'s offense against what {def_}'s defense "
                     f"allowed.\n{_csv(matchup_rows(ctx.matchups[off], off, def_))}")
    for off, def_ in ((ctx.away, ctx.home), (ctx.home, ctx.away)):
        parts.append(f"Down and distance when {off} has the ball: success rate (the "
                     f"season's plays in brackets, not per game). Early = 1st-2nd down, late = 3rd-4th; short 1-3 yds to go, "
                     f"medium 4-6, long 7+. No edge where either side has too few plays.\n"
                     f"{_csv(situational_rows(ctx.situational[off], off, def_))}")
    for team in (ctx.away, ctx.home):
        for unit in ("passing", "rushing", "receiving", "defense"):
            df = ctx.leaders.get((team, unit))
            if df is not None and not df.empty:
                parts.append(f"{team} {unit} leaders, season totals:\n"
                             f"{_csv(df, float_format='%.4g')}")
    if ctx.injury_note:
        parts.append(f"Injury reports: {ctx.injury_note}. Status is the game status on the final "
                     "report before the game: Out, Doubtful (rarely plays) or Questionable (may play); "
                     "blank means no game status (yet, or he's expected to play). Practice is the "
                     "latest day's participation. Snaps is the player's share of his team's snaps "
                     "this season in the phase he plays most (O offense, D defense, ST special "
                     "teams); Starter is his slot on the latest depth chart. In the by-unit table, "
                     "Snaps out adds up the out-or-doubtful players' shares of the unit's phase "
                     "(1.0 = one full-time player).")
        for team in (ctx.away, ctx.home):
            rep = ctx.injuries.get(team)
            if rep is None or rep.empty:
                parts.append(f"{team} injury report: none filed yet.")
                continue
            parts.append(f"{team} injury report by unit:\n{_csv(ctx.injury_units.get(team))}\n"
                         f"{team} injury report:\n{_csv(rep)}")
    if ctx.lineup_groups:
        parts.append("Lineup vs the season numbers: how far each group (QB; OL; Skill = RB, WR, TE; "
                     "Front = DL and LB; DB) is from the players behind the season numbers, in "
                     "full-time players (-1.0 = one fewer). A player out now takes away his share of "
                     "those games' snaps (Was; half if he missed half of them); one back adds the share "
                     "he plays when he does (Now) less Was. It counts the injury report's Out and "
                     "Doubtful and players on a reserve list, released or traded: players on injured "
                     "reserve aren't on the injury report.")
        for team in (ctx.away, ctx.home):
            g = ctx.lineup_groups.get(team)
            if g is None:
                continue
            players = ctx.lineup_players.get(team)
            listed = ("" if players is None or players.empty else
                      f"\nNot on the injury report:\n{_csv(players)}")
            parts.append(f"{team} lineup vs its season numbers: "
                         + ", ".join(f"{k} {v:+.1f}" for k, v in g.items()) + listed)
    return "\n\n".join(parts)


MATCHUP_PROMPT = """You are {analyst} writing a short preview of an upcoming game from the two teams' season stats. The fan sees the same tables next to your preview; your job is to say where the game will be decided.

Ground rules:
- Use only the numbers given. Don't bring in outside knowledge of these teams, players, injuries or news: it may be out of date, and the tables are the season so far.
- Each team's offense meets the other's defense. Pick the two to four unit-against-unit matchups that matter most, with the biggest edges first: EPA/play and success rate, then turnovers, pass protection and pressure, run blocking (stuffs, explosive runs), 3rd downs and the red zone. Give the numbers and places behind each. Say where a strength meets a strength.
- Don't predict a final score. Say which team the numbers favor and how clearly; if there's a betting line, say whether the numbers agree with it (the line usually knows more than season stats do).
- Season numbers over a few games, and down-and-distance cells with few plays, are noisy. Say so when it matters.
- If injury reports are given, weigh them: a starter who is Out or Doubtful takes his unit's season numbers with him, so say which matchup it changes and how much he plays (his snap share), e.g. two starting offensive linemen out against a strong pass rush. A Questionable player may play. The lineup lines count the same way, and add starters on injured reserve (not on the report) and starters back. Where "proj w/ lineup" is given, it already counts the players gone from both lineups: use it instead of proj for those stats, and say when it moves a matchup. Players back aren't in it: in backtests they made no measurable difference, so don't promise a lift from one. Mention only players the reports and lineup lines list, and if there are none, don't bring up anyone's health.

Terms: EPA is expected points added by a play ({model}); success rate is the share of plays with positive EPA; both count a pass or run a penalty wiped out (its EPA is the penalty's). Each place is a team's rank among the season's teams (1st = best; for a defense, allowing less is better and forcing more turnovers and sacks is better). Edge "{{team}} N" means that team's unit ranks N percentile points better than the unit it faces; "Even" is within 5. proj is a guess for this game: the league average plus half of the offense's and the defense's distances from it; proj w/ lineup (EPA/play, pass EPA/play and pass SR only) moves it by what players gone have done to games before, about -0.10 EPA/play for a starting QB out, -0.02 for each other full-time player the offense is missing and +0.02 for each one the defense is missing, and came a little closer to the games in backtests. From game to game, EPA/play, passing and 1st-down rate carry over best; turnovers and lost possessions, 3rd-down rate, sacks and the run game carry over least, so don't hang the preview on an edge there alone. Plays are pass attempts (sacks included) and runs. 1st Down % is first downs per play, 3rd Down % conversions per 3rd-down try, and RZ TD% touchdowns per play snapped inside the opponent's 20 (per play, not per trip). Turnovers are per game; Lost Poss. (per game) adds turnovers on downs and missed or blocked field goals. Sack % and QB Hit % are per dropback (hits include sacks); Stuff % (no gain or a loss) and Explosive Run % (10+ yards) are per designed run.

Write Markdown, about 150 to 220 words:
**Headline**: one sentence: who the numbers favor and why.
Three or four bullets: the matchups that will decide it, the biggest first, each with numbers.
**Watch**: one sentence on the matchup to watch.

Write only the preview, without a preamble or notes about the data."""


def matchup_prompt(league: str) -> str:
    t = _LEAGUE[league]
    return MATCHUP_PROMPT.format(analyst=t["analyst"], model=t["model"])


def preview(ctx: MatchupContext, provider: Provider, progress: Progress | None = None) -> Summary:
    """Write the matchup preview: one call, no tools. Makes no Streamlit
    calls, so it can run on a worker thread (submit_preview())."""
    progress = progress or Progress()
    progress.status, progress.calls = "Reading the matchup", 1
    resp = _call(_client(provider), provider, max_tokens=16000, system=matchup_prompt(ctx.league),
                 messages=[{"role": "user", "content": (
                     matchup_snapshot(ctx) + f"\n\nPreview {ctx.away} at {ctx.home}.")}])
    progress.status = "Done"
    return Summary(_final_text(resp), resp.model, [])


def submit_preview(ctx: MatchupContext, provider: Provider) -> tuple[Future, Progress]:
    """Start preview() on the summaries' worker threads, as submit() does."""
    progress = Progress(status="Reading the matchup")
    return _POOL.submit(preview, ctx, provider, progress), progress
