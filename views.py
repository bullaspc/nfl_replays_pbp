"""
Presentation helpers for the app: table styling, the drive field figure, the
play tables, the time bar, the other-games ticker and the screen wake lock.
They turn replay_core's DataFrames into what Streamlit draws; no data loading
here.
"""

from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from replay_core import (EDGE_EVEN, LINE_COUNT_ROWS, LINE_LOWER_IS_BETTER, LOWER_IS_BETTER, PLAY_CATS,
                         PLAYER_MIN_PLAYS, PLAYER_RATE_COLS, PLAYER_VOLUME_COL, SEASON_PER_GAME, SEASON_VOLUME,
                         SR_METRICS, SR_SUBTOTALS, down_distance, edge_label,
                         field_pos_label, fmt_top, percentile_of, play_type_label, success_emoji, yl_label)
from scoreboard import TickerGame, TickerTeam


# ---------- Stat tables ----------
def smap(styled, func, **kwargs):
    try:
        return styled.map(func, **kwargs)
    except AttributeError:
        return styled.applymap(func, **kwargs)

def percentile_color(pct: float) -> str:
    """CSS string for a red→white→green diverging color at the given percentile."""
    if pd.isna(pct):
        return ""
    # Red(220,53,69) → White(255,255,255) → Green(40,167,69)
    if pct <= 50:
        t = pct / 50.0
        r = int(220 + t * (255 - 220))
        g = int(53  + t * (255 - 53))
        b = int(69  + t * (255 - 69))
    else:
        t = (pct - 50) / 50.0
        r = int(255 + t * (40  - 255))
        g = int(255 + t * (167 - 255))
        b = int(255 + t * (69  - 255))
    lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    text = "#ffffff" if lum < 0.45 else "#212529"
    return f"background-color: #{r:02x}{g:02x}{b:02x}; color: {text}"

def style_sr_table(df: pd.DataFrame,
                    baselines: dict[str, dict[str, np.ndarray]] | None = None,
                    counts: dict | None = None) -> object:
    _b = baselines or {}
    _c = counts or {}

    def _color_cell(val, situation: str, metric: str) -> str:
        if pd.isna(val):
            return ""
        sit_bl = _b.get(situation, {})
        arr = sit_bl.get(metric)
        if arr is not None and len(arr) > 0:
            return percentile_color(percentile_of(float(val), arr))
        # fallback: fixed thresholds
        if metric == "Pass SR%" or metric == "Rush SR%":
            if val >= 55:
                return "background-color: #d4edda; color: #155724"
            if val <= 40:
                return "background-color: #f8d7da; color: #721c24"
            return ""
        if val > 0:
            return "background-color: #d4edda; color: #155724"
        if val < 0:
            return "background-color: #f8d7da; color: #721c24"
        return ""

    styled = df.style
    for col in df.columns:
        metric = next((m for m in SR_METRICS if col.endswith(m)), None)
        if metric is None:
            continue
        team_part = col[: -len(metric)].strip()
        play_type = "Pass" if "Pass" in metric else "Rush"
        for sit in df.index:
            n = _c.get((sit, team_part), {}).get(play_type)
            if "SR%" in metric:
                fmt_fn = lambda v, c=n: ("—" if pd.isna(v) else (f"{v:.0f}% ({c})" if c is not None else f"{v:.0f}%"))
            else:
                fmt_fn = lambda v, c=n: ("—" if pd.isna(v) else (f"{v:+.2f} ({c})" if c is not None else f"{v:+.2f}"))
            styled = smap(
                styled,
                lambda v, s=sit, m=metric: _color_cell(v, s, m),
                subset=pd.IndexSlice[sit, col],
            ).format(fmt_fn, subset=pd.IndexSlice[sit, col])
    subtotals = [lbl for lbl, _, _ in SR_SUBTOTALS if lbl in df.index]
    if subtotals:
        styled = styled.set_properties(subset=pd.IndexSlice[subtotals, :],
                                       **{"font-weight": "bold"})
    return (
        styled
        .set_properties(**{"text-align": "center"})
        .set_table_styles(
            [{"selector": "th", "props": [("text-align", "center"), ("font-weight", "bold")]}]
        )
    )


_EPA_ROWS        = {"Pass EPA/play", "Rush EPA/play", "EPA/play"}
_RATE_ROWS       = {"CMP%", "Pass SR", "Rush SR", "1st Down %", "3rd Down %", "RZ TD%"}
_NO_COLOR_ROWS   = {"Pass Yds", "Rush Yds", "Total Yds", "Pass Plays", "Rush Plays", "Plays", "Rush+Comp", "Turnovers", "TOP"}
_PCT_FORMAT_ROWS = _RATE_ROWS
_EPA_FORMAT_ROWS = _EPA_ROWS


def style_stat_table(df: pd.DataFrame, away: str, home: str,
                     baselines: dict | None = None):
    """Return a pandas Styler with percentile-based diverging colors and EPA coloring."""
    if baselines is None:
        baselines = {}

    def _color_epa(val):
        if pd.isna(val):
            return ""
        if val > 0:
            return "background-color: #d4edda; color: #155724"
        if val < 0:
            return "background-color: #f8d7da; color: #721c24"
        return ""

    styled = df.style

    for row in df.index:
        if row == "TOP":
            styled = styled.format(fmt_top, subset=pd.IndexSlice[row, :])
            continue
        if row in _PCT_FORMAT_ROWS:
            fmt_str = "{:.1%}"
        elif row in _EPA_FORMAT_ROWS:
            fmt_str = "{:+.2f}"
        elif row == "aDoT":
            fmt_str = "{:.1f}"
        else:
            fmt_str = "{:.0f}"
        styled = styled.format(fmt_str, subset=pd.IndexSlice[row, :], na_rep="—")

    for row in _EPA_ROWS:
        if row in df.index:
            styled = smap(styled, _color_epa, subset=pd.IndexSlice[row, :])

    for row in df.index:
        if row in _EPA_ROWS or row in _NO_COLOR_ROWS:
            continue
        arr   = baselines.get(row, np.array([]))
        lower = row in LOWER_IS_BETTER

        def _cell(val, _arr=arr, _lower=lower):
            if pd.isna(val):
                return ""
            pct = percentile_of(float(val), _arr)
            if pd.isna(pct):
                return ""
            if _lower:
                pct = 100.0 - pct
            return percentile_color(pct)

        styled = smap(styled, _cell, subset=pd.IndexSlice[row, :])

    styled = styled.set_properties(**{"text-align": "center"})
    styled = styled.set_table_styles(
        [{"selector": "th", "props": [("text-align", "center"), ("font-weight", "bold")]}]
    )
    return styled


def style_line_table(df: pd.DataFrame, baselines: dict | None = None):
    """Line play table: rates colored by percentile vs `baselines` (flipped
    where lower is better), counts plain."""
    baselines = baselines or {}
    styled = df.style
    for row in df.index:
        if row in LINE_COUNT_ROWS:
            styled = styled.format("{:.0f}", subset=pd.IndexSlice[row, :], na_rep="—")
            continue
        styled = styled.format("{:.1%}", subset=pd.IndexSlice[row, :], na_rep="—")

        def _cell(val, _arr=baselines.get(row, np.array([])), _lower=row in LINE_LOWER_IS_BETTER):
            pct = percentile_of(float(val), _arr) if pd.notna(val) else float("nan")
            if pd.isna(pct):
                return ""
            return percentile_color(100.0 - pct if _lower else pct)

        styled = smap(styled, _cell, subset=pd.IndexSlice[row, :])
    return (
        styled
        .set_properties(**{"text-align": "center"})
        .set_table_styles(
            [{"selector": "th", "props": [("text-align", "center"), ("font-weight", "bold")]}]
        )
    )


_PLAYER_FORMATS = {"SR%": "{:.1f}%", "aDOT": "{:.1f}", "EPA/play": "{:.2f}"}
def style_defense_table(df: pd.DataFrame):
    """A defensive leaders table: counts, shown as 1 or 0.5 (half sacks and
    half TFLs) rather than 1.0."""
    return df.style.format("{:g}", subset=df.select_dtypes("number").columns)


def style_player_table(df: pd.DataFrame, kind: str, baselines: dict | None = None,
                       min_plays: int | None = None):
    """A passing/rushing/receiving leaders table: its rates (`PLAYER_RATE_COLS`)
    colored by percentile vs `baselines[kind]`; players with fewer than
    `min_plays` (default `PLAYER_MIN_PLAYS`, the baseline's) stay plain."""
    arrs = (baselines or {}).get(kind, {})
    min_plays = PLAYER_MIN_PLAYS[kind] if min_plays is None else min_plays
    qualified = df[PLAYER_VOLUME_COL[kind]] >= min_plays

    def _colors(col):
        arr = arrs.get(col.name, np.array([]))
        return [percentile_color(percentile_of(float(v), arr)) if q else ""
                for v, q in zip(col, qualified)]

    styled = df.style.format({c: f for c, f in _PLAYER_FORMATS.items() if c in df.columns},
                             na_rep="—")
    rate_cols = [c for c in PLAYER_RATE_COLS[kind] if c in df.columns]
    return styled.apply(_colors, subset=rate_cols) if rate_cols else styled


def style_season_table(df: pd.DataFrame, ranks: pd.DataFrame):
    """The Season stats team table (teams as rows, replay_core.SEASON_COLS
    as columns): formatted as Team stats and Line play are, every cell
    colored by its rank among the table's teams (`ranks`, 0–100, 100 =
    best) except the volumes (`SEASON_VOLUME`)."""
    fmt = {col: _season_format(col) for col in df.columns}
    styled = df.style.format(fmt, na_rep="—")
    colored = [c for c in df.columns if c not in SEASON_VOLUME and c in ranks.columns]
    if colored:
        styled = styled.apply(lambda col: [percentile_color(p) for p in ranks.loc[col.index, col.name]],
                              subset=colored)
    return (
        styled
        .set_properties(**{"text-align": "center"})
        .set_table_styles(
            [{"selector": "th", "props": [("text-align", "center"), ("font-weight", "bold")]}]
        )
    )


def style_team_season(values: pd.Series, ranks: pd.Series, places: pd.Series):
    """One team's column of the season table as a card: rows are the stats,
    Value formatted and colored as `style_season_table` does, and Rank its
    place among the teams ("5th of 32")."""
    shown = pd.DataFrame({"Value": [_season_format(c)(v) for c, v in values.items()],
                          "Rank": places.reindex(values.index).fillna("")},
                         index=pd.Index(values.index, name="Stat"))
    colors = ["" if c in SEASON_VOLUME else percentile_color(ranks.get(c, np.nan))
              for c in values.index]
    return (
        shown.style
        .apply(lambda col: colors, subset=["Value"])
        .set_properties(**{"text-align": "center"})
        .set_table_styles(
            [{"selector": "th", "props": [("text-align", "center"), ("font-weight", "bold")]}]
        )
    )


def _season_format(col: str):
    """A formatter for one season table column, as `style_season_table`'s."""
    if col == "TOP":
        return fmt_top
    if col in _RATE_ROWS or (col.endswith("%") and col not in LINE_COUNT_ROWS):
        f = "{:.1%}"
    elif col in _EPA_ROWS:
        f = "{:+.2f}"
    elif col == "aDoT" or col in SEASON_PER_GAME:
        f = "{:.1f}"
    else:
        f = "{:.0f}"
    return lambda v: "—" if pd.isna(v) else f.format(v)


# ---------- Matchup ----------
# The Matchup page's tables are drawn as text: each value carries its place
# ("+0.04 (10th)"), colored by percentile like the season tables, and the
# Δ / Edge cell is tinted in the hue of the team with the edge. One hue per
# side of the game, the replay charts' fallbacks: they show on both themes
# and never look alike, as two teams' own colors can (two navies).
MATCHUP_TINTS = ("#ff7f0e", "#1f77b4")  # away, home


def _tint(color: str | None, edge: float) -> str:
    """A background in `color`, stronger with the `edge` (percentile
    points), light enough for either theme's text."""
    h = hex_or_none(color)
    if h is None or len(h) != 7 or pd.isna(edge) or abs(edge) < EDGE_EVEN:
        return ""
    r, g, b = (int(h[i:i + 2], 16) for i in (1, 3, 5))
    return f"background-color: rgba({r}, {g}, {b}, {0.15 + 0.45 * min(abs(edge) / 100, 1.0):.2f})"


def _ranked(stat: str, v, place: str) -> str:
    s = _season_format(stat)(v)
    return f"{s} ({place})" if place and s != "—" else s


def _delta_format(stat: str, v) -> str:
    """A difference in `stat`'s units, signed: rates in points of percent."""
    if pd.isna(v):
        return "—"
    if stat in _RATE_ROWS or (stat.endswith("%") and stat not in LINE_COUNT_ROWS):
        s = f"{v:+.1%}"
    else:
        s = f"{v:+.2f}" if stat in _EPA_ROWS else f"{v:+.1f}"
    return s.lstrip("+-") if float(s.rstrip("%")) == 0 else s  # no "-0.0"


def _centered(styled):
    return (
        styled
        .set_properties(**{"text-align": "center"})
        .set_table_styles(
            [{"selector": "th", "props": [("text-align", "center"), ("font-weight", "bold")]}]
        )
    )


def style_head_to_head(df: pd.DataFrame, away: str, home: str, tints: dict):
    """A `replay_core.head_to_head` table: each team's value and place,
    colored by its percentile among the season's teams, and Δ (away − home)
    tinted in the hue (`tints`, team → color) of the team that ranks
    better (none for style stats or a gap under `EDGE_EVEN`)."""
    shown = pd.DataFrame({
        away: [_ranked(s, v, p) for s, v, p in zip(df.index, df["Away"], df["Away rank"])],
        home: [_ranked(s, v, p) for s, v, p in zip(df.index, df["Home"], df["Home rank"])],
        "Δ": [_delta_format(s, v) for s, v in zip(df.index, df["Δ"])],
    }, index=df.index)
    gap = df["_away_pct"] - df["_home_pct"]

    def _colors(col):
        if col.name == "Δ":
            return [_tint(tints.get(away if g > 0 else home), g) for g in gap]
        return [percentile_color(p) for p in df["_away_pct" if col.name == away else "_home_pct"]]

    return _centered(shown.style.apply(_colors))


def style_matchup_table(df: pd.DataFrame, offense: str, defense: str, tints: dict):
    """A `replay_core.matchup_table`: the offense's value and what the
    defense allowed, each with its place and colored by its percentile,
    Edge ("KC 25": who ranks better, by how many percentile points) tinted
    in that team's hue (`tints`), and Proj. and League as plain values."""
    off_col, def_col = f"{offense} offense", f"{defense} defense"
    shown = pd.DataFrame({
        off_col: [_ranked(s, v, p) for s, v, p in zip(df.index, df["Offense"], df["Off rank"])],
        def_col: [_ranked(s, v, p) for s, v, p in zip(df.index, df["Defense"], df["Def rank"])],
        "Edge": [edge_label(e, offense, defense) for e in df["Edge"]],
        "Proj.": [_season_format(s)(v) for s, v in zip(df.index, df["Proj."])],
        "League": [_season_format(s)(v) for s, v in zip(df.index, df["League"])],
    }, index=df.index)

    def _colors(col):
        if col.name == off_col:
            return [percentile_color(p) for p in df["_off_pct"]]
        if col.name == def_col:
            return [percentile_color(p) for p in df["_def_pct"]]
        if col.name == "Edge":
            return [_tint(tints.get(offense if e > 0 else defense), e) for e in df["Edge"]]
        return [""] * len(col)

    return _centered(shown.style.apply(_colors))


def style_situational_matchup(df: pd.DataFrame, offense: str, defense: str, tints: dict,
                              metrics: tuple[str, ...] = ("Pass SR%", "Rush SR%")):
    """A `replay_core.situational_matchup`: per metric the offense's value
    ("O") and what the defense allowed ("D"), each with its plays in
    brackets and colored by percentile, then the edge as in
    `style_matchup_table` ("—": too few plays for one)."""
    shown, colors_by_col = {}, {}
    for m in metrics:
        kind = "Run" if m.startswith("Rush") else "Pass"
        for side, team in (("O", offense), ("D", defense)):
            col = f"{kind} {side}"
            shown[col] = ["—" if pd.isna(v) else f"{v:.0f}% ({int(n)})"
                          for v, n in zip(df[f"{m} {side}"], df[f"_{m} {side} n"])]
            colors_by_col[col] = [percentile_color(p) for p in df[f"_{m} {side} pct"]]
        edges = df[f"{m} edge"]
        shown[f"{kind} edge"] = [edge_label(e, offense, defense) or "—" for e in edges]
        colors_by_col[f"{kind} edge"] = [_tint(tints.get(offense if e > 0 else defense), e)
                                         for e in edges]
    shown = pd.DataFrame(shown, index=df.index)
    styled = shown.style.apply(lambda col: colors_by_col[col.name])
    subtotals = [lbl for lbl, _, _ in SR_SUBTOTALS if lbl in shown.index]
    if subtotals:
        styled = styled.set_properties(subset=pd.IndexSlice[subtotals, :], **{"font-weight": "bold"})
    return _centered(styled)


# ---------- Drives ----------
_FIELD_GREEN = "#2f7a3b"
_FIELD_GREEN_ALT = "#2a6f35"


# ---------- Tendencies ----------
# EPA/play on the percentile tables' red → white → green, white at 0.
_EPA_SCALE = [[0.0, "#dc3545"], [0.5, "#ffffff"], [1.0, "#28a745"]]
_EPA_RANGE = 0.8  # ± EPA/play at the scale's ends


def _vs_league(base: pd.DataFrame | None, key) -> tuple[float, float]:
    """(league share, league EPA/play) for `key`, NaN without a baseline."""
    if base is None or key not in base.index:
        return float("nan"), float("nan")
    return base.at[key, "Share"], base.at[key, "EPA/play"]


def run_tendency_figure(runs: pd.DataFrame, base: pd.DataFrame | None = None) -> go.Figure:
    """One bar per gap (or side), left to right as the offense sees it: height
    is the share of runs, color EPA/play, the dash the league's share. Each
    label carries the runs there and their yards per carry."""
    league = [_vs_league(base, k) for k in runs.index]
    share = runs["Share"].fillna(0) * 100
    fig = go.Figure(go.Bar(
        x=list(runs.index), y=share,
        marker=dict(color=runs["EPA/play"].fillna(0), colorscale=_EPA_SCALE,
                    cmin=-_EPA_RANGE, cmax=_EPA_RANGE,
                    line=dict(color="rgba(128,128,128,0.6)", width=1)),
        customdata=[[r["Plays"], r["Yds/Play"], r["SR%"], r["EPA/play"], ls * 100, le]
                    for (_, r), (ls, le) in zip(runs.iterrows(), league)],
        hovertemplate=("<b>%{x}</b><br>%{customdata[0]} runs · %{y:.0f}% of runs "
                       "(league %{customdata[4]:.0f}%)<br>%{customdata[1]:.1f} yds/carry · "
                       "SR %{customdata[2]:.0%}<br>EPA/play %{customdata[3]:+.2f} "
                       "(league %{customdata[5]:+.2f})<extra></extra>"),
    ))
    for i, (ls, _) in enumerate(league):
        if pd.notna(ls):
            fig.add_shape(type="line", x0=i - 0.45, x1=i + 0.45, y0=ls * 100, y1=ls * 100,
                          line=dict(width=2, color="rgba(128,128,128,0.9)"))
    top = max([*share, *[ls * 100 for ls, _ in league if pd.notna(ls)], 10])
    fig.update_layout(
        height=240, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, bargap=0.15,
        xaxis=dict(fixedrange=True, tickangle=0, tickvals=list(runs.index),
                   ticktext=[f"{k}<br><b>{n}</b>" + (f" · {y:.1f}" if n else "")
                             for k, n, y in zip(runs.index, runs["Plays"], runs["Yds/Play"])]),
        yaxis=dict(range=[0, top * 1.15], ticksuffix="%", fixedrange=True, showgrid=False),
    )
    return fig


def pass_tendency_figure(passes: pd.DataFrame, base: pd.DataFrame | None = None) -> go.Figure:
    """Depth × side grid as the quarterback sees it, deepest row on top: each
    cell's color is EPA/play, its text attempts, share and completions."""
    depths = list(dict.fromkeys(passes.index.get_level_values(0)))
    sides = list(dict.fromkeys(passes.index.get_level_values(1)))
    z, text, custom = [], [], []
    for d in depths:
        zr, tr, cr = [], [], []
        for sd in sides:
            r = passes.loc[(d, sd)]
            ls, le = _vs_league(base, (d, sd))
            n = int(r["Plays"])
            zr.append(r["EPA/play"] if n else None)
            tr.append(f"<b>{n}</b> · {r['Share']:.0%}<br>{r['Cmp%'] * n:.0f}/{n} · {r['EPA/play']:+.2f}"
                      if n else "—")
            cr.append([n, r["Share"] * 100 if n else 0, r["Cmp%"], r["Yds/Play"], r["SR%"],
                       r["EPA/play"], f" · aDOT {r['aDOT']:.1f}" if pd.notna(r["aDOT"]) else "", ls * 100, le])
        z.append(zr), text.append(tr), custom.append(cr)
    fig = go.Figure(go.Heatmap(
        z=z, x=sides, y=depths, text=text, texttemplate="%{text}", customdata=custom,
        colorscale=_EPA_SCALE, zmin=-_EPA_RANGE, zmax=_EPA_RANGE, showscale=False,
        xgap=3, ygap=3, hoverongaps=False,
        hovertemplate=("<b>%{y} %{x}</b><br>%{customdata[0]} att · %{customdata[1]:.0f}% of "
                       "passes (league %{customdata[7]:.0f}%)<br>Cmp %{customdata[2]:.0%} · "
                       "%{customdata[3]:.1f} yds/att%{customdata[6]}<br>"
                       "SR %{customdata[4]:.0%} · EPA/play %{customdata[5]:+.2f} "
                       "(league %{customdata[8]:+.2f})<extra></extra>"),
    ))
    fig.update_layout(
        height=60 + 70 * len(depths), margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(fixedrange=True, side="bottom", showgrid=False),
        yaxis=dict(fixedrange=True, autorange="reversed", showgrid=False),
    )
    return fig


def drive_field_figure(spots: pd.DataFrame, home: str, away: str,
                       colors: dict[str, str], logos: dict[str, str],
                       nicknames: dict[str, str]) -> go.Figure:
    """Football field with one arrow per drive, latest drive on top.

    x is measured from the home team's goal line: the home team defends the
    left end zone and drives left → right; the away team drives right → left."""
    n = len(spots)
    fig = go.Figure()

    # Field: alternating 5-yard stripes, yard lines, end zones in team colors.
    shapes = []
    for i, x0 in enumerate(range(0, 100, 5)):
        shapes.append(dict(type="rect", x0=x0, x1=x0 + 5, y0=0, y1=1, yref="paper",
                           fillcolor=_FIELD_GREEN if i % 2 == 0 else _FIELD_GREEN_ALT,
                           line_width=0, layer="below"))
    for team, x0, x1 in [(home, -10, 0), (away, 100, 110)]:
        shapes.append(dict(type="rect", x0=x0, x1=x1, y0=0, y1=1, yref="paper",
                           fillcolor=colors.get(team, "#555555"),
                           line=dict(color="white", width=2), layer="below"))
    for x in range(5, 100, 5):
        shapes.append(dict(type="line", x0=x, x1=x, y0=0, y1=1, yref="paper",
                           line=dict(color="rgba(255,255,255,%s)" % (0.7 if x % 10 == 0 else 0.3),
                                     width=2 if x == 50 else 1),
                           layer="below"))
    shapes.append(dict(type="rect", x0=-10, x1=110, y0=0, y1=1, yref="paper",
                       line=dict(color="white", width=2), layer="below"))

    # End zone branding: logo top and bottom, nickname written along the zone.
    images, annotations = [], []
    for team, xc, angle in [(home, -5, -90), (away, 105, 90)]:
        logo = logos.get(team)
        if logo:
            for yc in (0.86, 0.14):
                images.append(dict(source=logo, xref="x", yref="paper", x=xc, y=yc,
                                   sizex=8, sizey=0.2, xanchor="center", yanchor="middle",
                                   sizing="contain", layer="above"))
        annotations.append(dict(
            x=xc, y=0.5, xref="x", yref="paper", showarrow=False, textangle=angle,
            text=f"<b>{nicknames.get(team, team).upper()}</b>",
            font=dict(color="white", size=12), xanchor="center", yanchor="middle"))

    # Each drive is a bar of per-play segments coloured by play category, with
    # a tick at every snap so plays that gained nothing still show. Traces are
    # batched per category (None breaks the line between segments).
    seg_xy = {c: ([], []) for c in PLAY_CATS}
    snaps = {c: ([], [], []) for c in PLAY_CATS}
    ends = []
    for i, d in enumerate(spots.itertuples(index=False)):
        is_home = d.team == home
        to_x = (lambda y: 100 - y) if is_home else (lambda y: y)
        x0, x1 = to_x(d.start), to_x(d.end)
        # White underlay so the bar reads against the turf. It spans every
        # segment, since a loss can carry the ball behind the drive's start.
        span = [x0, x1] + [to_x(sg[k]) for sg in d.segments for k in ("from", "to")]
        fig.add_trace(go.Scatter(x=[min(span), max(span)], y=[i, i], mode="lines",
                                 line=dict(color="white", width=8),
                                 hoverinfo="skip", showlegend=False))
        for sgm in d.segments:
            xs, ys = seg_xy[sgm["cat"]]
            xs += [to_x(sgm["from"]), to_x(sgm["to"]), None]
            ys += [i, i, None]
            sx, sy, tx = snaps[sgm["cat"]]
            sx.append(to_x(sgm["from"]))
            sy.append(i)
            tx.append(sgm["hover"])
        ends.append((i, d, x0, x1, colors.get(d.team, "#1f77b4" if is_home else "#ff7f0e")))

    for cat, (label, color) in PLAY_CATS.items():
        xs, ys = seg_xy[cat]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=label, legendgroup=cat,
                                 line=dict(color=color, width=5), hoverinfo="skip"))
    for cat, (label, color) in PLAY_CATS.items():
        sx, sy, tx = snaps[cat]
        fig.add_trace(go.Scatter(
            x=sx, y=sy, mode="markers", name=label, legendgroup=cat, showlegend=False,
            marker=dict(symbol="line-ns", size=13, line=dict(color=color, width=3)),
            hovertext=tx, hovertemplate="%{hovertext}<extra></extra>"))

    for i, d, x0, x1, color in ends:
        hover = (f"<b>Drive {d.drive} · {d.team}</b>"
                 + (f" · Q{d.qtr}" if d.qtr else "")
                 + f"<br>{yl_label(d.start)} → {yl_label(d.end)}"
                 + f"<br>{d.plays} plays, {d.yards} yds, {fmt_top(d.top)}"
                 + f"<br>{d.outcome}")
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[i, i], mode="markers",
            marker=dict(symbol=["circle", "triangle-right" if x1 >= x0 else "triangle-left"],
                        size=[9, 13], color=color, line=dict(color="white", width=1.5)),
            hovertemplate=hover + "<extra></extra>", showlegend=False))
        annotations.append(dict(
            x=1.0, y=i, xref="paper", yref="y", xanchor="left", showarrow=False,
            text=f" {d.outcome}", font=dict(size=11)))

    tick = list(range(10, 100, 10))
    fig.update_layout(
        shapes=shapes, images=images, annotations=annotations,
        height=max(330, 100 + 26 * n),
        margin=dict(l=10, r=85, t=30, b=40),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=0, yanchor="top",
                    itemclick="toggle", itemdoubleclick="toggleothers"),
        plot_bgcolor=_FIELD_GREEN,
        hoverlabel=dict(align="left"),
        xaxis=dict(range=[-10, 110], tickvals=tick,
                   ticktext=[str(50 - abs(50 - t)) for t in tick],
                   side="top", showgrid=False, zeroline=False, fixedrange=True),
        yaxis=dict(range=[-0.7, n - 0.3], tickvals=list(range(n)),
                   ticktext=[f"Q{d.qtr} {d.team}" if d.qtr else d.team
                             for d in spots.itertuples(index=False)],
                   showgrid=False, zeroline=False, fixedrange=True),
    )
    return fig

def style_drive_chart(df: pd.DataFrame):
    def _outcome_color(val):
        if val == "TD":
            return "background-color: #d4edda; color: #155724; font-weight: bold"
        if val in ("Interception", "Fumble", "Downs", "Safety"):
            return "background-color: #f8d7da; color: #721c24"
        if val == "FG":
            return "background-color: #cce5ff; color: #004085"
        if val in ("FG Miss", "FG Blocked"):
            return "background-color: #fff3cd; color: #856404"
        if val == "Punt":
            return "background-color: #e2e3e5; color: #383d41"
        return ""
    styled = df.style
    if "Outcome" in df.columns:
        styled = smap(styled, _outcome_color, subset=["Outcome"])
    return (
        styled
        .set_properties(**{"text-align": "center"})
        .set_table_styles(
            [{"selector": "th", "props": [("text-align", "center"), ("font-weight", "bold")]}]
        )
    )

def style_scoring_timeline(df: pd.DataFrame):
    def _type_color(val):
        if val == "TD":
            return "background-color: #d4edda; color: #155724; font-weight: bold"
        if val == "FG":
            return "background-color: #cce5ff; color: #004085"
        if val == "Safety":
            return "background-color: #f8d7da; color: #721c24"
        if val in ("XP", "2PT"):
            return "background-color: #e2e3e5; color: #383d41"
        return ""

    styled = df.style
    if "Type" in df.columns:
        styled = smap(styled, _type_color, subset=["Type"])
    return (
        styled
        .set_properties(**{"text-align": "center"})
        .set_table_styles(
            [{"selector": "th", "props": [("text-align", "center"), ("font-weight", "bold")]}]
        )
    )


# ---------- Play tables ----------
def build_plays_df(raw: pd.DataFrame, hide_desc: bool, reverse: bool = True) -> pd.DataFrame:
    raw = raw.copy()
    raw["Type"] = raw.apply(play_type_label, axis=1)
    raw["D&D"] = raw.apply(down_distance, axis=1)
    raw["Success?"] = raw.apply(success_emoji, axis=1)
    raw["Q"] = raw["qtr"].apply(lambda x: str(int(x)) if pd.notna(x) else "")
    raw["Field"] = raw.apply(field_pos_label, axis=1)
    # On an empty frame apply() gives float columns, which .str rejects.
    labels = ["Type", "D&D", "Success?", "Field"]
    raw[labels] = raw[labels].astype(str)
    is_special = raw["play_type"].isin(["field_goal", "extra_point"])
    raw["_rz"] = (raw["yardline_100"].fillna(100) <= 20) & ~is_special
    # 4th-down attempts: the offense went for it (fakes included, as nflfastR
    # types them pass/run), not a punt, field goal or penalty-only snap.
    raw["_go4"] = (raw["down"] == 4) & raw["play_type"].isin(["pass", "run"])
    if hide_desc:
        cols_sel = ["Q", "time", "posteam", "Field", "_rz", "_go4", "Type", "D&D", "Success?", "yards_gained", "epa"]
        df = raw[cols_sel].copy()
        df.columns = ["Q", "Clock", "Off", "Field", "_rz", "_go4", "Type", "D&D", "Success?", "Yds", "EPA"]
    else:
        cols_sel = ["Q", "time", "posteam", "Field", "_rz", "_go4", "Type", "D&D", "Success?", "desc", "yards_gained", "epa"]
        df = raw[cols_sel].copy()
        df.columns = ["Q", "Clock", "Off", "Field", "_rz", "_go4", "Type", "D&D", "Success?", "Description", "Yds", "EPA"]
    df["Yds"] = pd.to_numeric(df["Yds"], errors="coerce").fillna(0).astype(int)
    df["EPA"] = pd.to_numeric(df["EPA"], errors="coerce").round(2)
    return df.iloc[::-1] if reverse else df

_PLAY_FLAGS = ["_rz", "_go4"]


def style_plays(df: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Cell styles for a play table: the row's colour from its down (4th-down
    attempt, then 3rd down), and the Field cell red inside the red zone.
    `flags` holds build_plays_df()'s _rz/_go4 columns, kept out of `df`
    because st.table shows every column it is given."""
    third = df["Type"].str.startswith("3rd")
    row_bg = pd.Series(
        np.select([flags["_go4"].astype(bool), third], ["#ffd0a6", "#ffeeba"], "#ffffff"),
        index=df.index)
    styles = pd.DataFrame(
        {c: "background-color: " + row_bg + "; color: #000000" for c in df.columns},
        index=df.index)
    if "Field" in styles.columns:
        styles.loc[flags["_rz"].astype(bool), "Field"] = (
            "background-color: #dc3545; color: #ffffff")
    return styles


def _nowrap(v) -> str:
    """v with no line break opportunities: non-breaking spaces and hyphens, and
    word joiners around each dash ("ARI 20–10 NO")."""
    if pd.isna(v):
        return ""
    s = str(v).replace(" ", "\u00a0").replace("-", "\u2011")
    return s.replace("–", "\u2060–\u2060").replace("—", "\u2060—\u2060")


# The column that stays put while a wide table scrolls sideways on a phone.
PINNED_COL = "Player"
# Streamlit's page background, which the pinned cells need to hide the
# columns scrolling under them. Streamlit doesn't expose it to CSS.
_PAGE_BG = {"light": "#ffffff", "dark": "#0e1117"}


def _pinned_props() -> list[tuple[str, str]]:
    """CSS that pins a column (header and cells) to the left edge, with an
    inset line as its border: a collapsed table border stays behind."""
    return [("position", "sticky"), ("left", "0"), ("z-index", "1"),
            ("background-color", _PAGE_BG.get(st.context.theme.type, _PAGE_BG["light"])),
            ("box-shadow", "inset -1px 0 0 rgba(128, 128, 128, 0.45)")]


def stat_table(table, hide_index: bool = False) -> None:
    """A table (DataFrame or Styler) as a static st.table: on a phone it
    scrolls with the page instead of catching swipes, its cells are tighter
    than st.dataframe's, and its columns can't be sorted or dragged. A
    Description column wraps, and a `PINNED_COL` column stays at the left
    edge while the table scrolls sideways. st.table drops white-space styles,
    so the other text columns ("ARI 3 — 0 NO", "OWN 26", "A.Davis-Gaither")
    are displayed with non-breaking spaces and hyphens to stay on one line.
    Only the display changes, so styles keyed on the values still match."""
    styled = table.style if isinstance(table, pd.DataFrame) else table
    df = styled.data
    text = [c for c in df.columns if c != "Description" and df[c].notna().any()
            and df[c].dropna().map(lambda v: isinstance(v, str)).all()]
    if text:
        styled = styled.format(_nowrap, subset=text)
    if "Description" in df.columns:
        styled = styled.set_properties(
            subset=["Description"], **{"min-width": "260px", "text-align": "left"})
    if PINNED_COL in df.columns:
        styled = styled.set_table_styles(
            {PINNED_COL: [{"selector": "", "props": _pinned_props()}]}, overwrite=False)
    st.table(styled, hide_index=hide_index)


def plays_table(df: pd.DataFrame) -> None:
    """A play table from build_plays_df(), as a static st.table: its columns
    can't be dragged out of order or sorted out of game order, and
    descriptions wrap to whatever height they need."""
    df = df.reset_index(drop=True)
    flags, shown = df[_PLAY_FLAGS], df.drop(columns=_PLAY_FLAGS)
    stat_table(
        shown.style.apply(style_plays, axis=None, flags=flags)
        .format({"EPA": "{:.2f}"}, na_rep=""),
        hide_index=True,
    )


# ---------- Header and time bar ----------
def hex_or_none(v):
    """A "#rrggbb" color, or None for blank/NaN cells."""
    return v if isinstance(v, str) and v.startswith("#") else None


def timeout_pips(n: int | None, total: int = 3) -> str:
    """Timeouts left as dots, filled for each one left: ●●○."""
    if n is None:
        return "—"
    return "●" * n + "○" * max(total - n, 0)


def logo_img(team: str, logos: dict[str, str]) -> str:
    url = logos.get(team)
    return f"<img src='{url}' alt='{team}' style='height:clamp(36px,9vw,64px);width:auto'>" if url else ""


def time_bar_html(frac: float, fill: str, in_ot: bool) -> str:
    """Track showing position within the game. Nothing past `frac` is drawn:
    no scoring marks, no play density, no total play count — any of those would
    give away what the viewer hasn't watched yet."""
    pct = min(max(float(frac), 0.0), 1.0) * 100.0
    ticks = "".join(
        f'<div style="position:absolute;left:{q}%;top:0;bottom:0;width:1px;'
        f'background:rgba(128,128,128,0.55)"></div>'
        for q in (25, 50, 75)
    )
    ot_track = ot_label = ""
    if in_ot:
        ot_track = (f'<div style="flex:0 0 12%;height:12px;border-radius:6px;'
                    f'background:{fill};margin-left:4px"></div>')
        ot_label = '<div style="flex:0 0 12%;margin-left:4px;text-align:center">OT</div>'
    return f"""
<div style="display:flex;align-items:center;margin:0.2rem 0 0.2rem 0">
  <div style="flex:1 1 auto;position:relative;height:12px;border-radius:6px;
              background:rgba(128,128,128,0.22);overflow:hidden">
    <div style="position:absolute;left:0;top:0;bottom:0;width:{pct:.3f}%;
                background:{fill}"></div>
    {ticks}
  </div>{ot_track}
</div>
<div style="display:flex;font-size:0.72rem;opacity:0.6;margin-bottom:0.2rem">
  <div style="flex:1 1 auto;display:flex">
    <div style="flex:1">Q1</div><div style="flex:1">Q2</div>
    <div style="flex:1">Q3</div><div style="flex:1">Q4</div>
  </div>{ot_label}
</div>
"""

# ---------- Other games ticker ----------
_LIVE_RED = "#d62728"


def _ticker_team(t: TickerTeam, dim: bool) -> str:
    logo = (f'<img src="{escape(t.logo)}" alt="" style="height:16px;width:16px;object-fit:contain">'
            if t.logo else '<span style="width:16px"></span>')
    rank = f'<span style="font-size:0.65rem;opacity:0.7">{t.rank}</span>' if t.rank else ""
    ball = '<span style="font-size:0.62rem" title="Has the ball">🏈</span>' if t.has_ball else ""
    return (f'<div style="display:flex;align-items:center;gap:4px;{"opacity:0.55;" if dim else ""}">'
            f'{logo}{rank}<span style="font-weight:600">{escape(t.abbr)}</span>{ball}'
            f'<span style="margin-left:auto;font-weight:700;font-variant-numeric:tabular-nums">'
            f'{escape(t.score)}</span></div>')


def _ticker_card(g: TickerGame) -> str:
    live, final = g.state == "in", g.state == "post"
    head = (f'<div style="display:flex;gap:6px;font-size:0.68rem;white-space:nowrap">'
            f'<span style="{f"color:{_LIVE_RED};font-weight:700" if live else "opacity:0.7"}">'
            f'{"● " if live else ""}{escape(g.status)}</span>'
            f'<span style="margin-left:auto;opacity:0.6">{escape(g.network or "")}</span></div>')
    # A final dims the loser.
    teams = _ticker_team(g.away, final and g.home.winner) + _ticker_team(g.home, final and g.away.winner)
    sit = ""
    if live and (g.down_distance or g.spot):
        text = " · ".join(x for x in (g.down_distance, g.spot and f"at {g.spot}") if x)
        sit = (f'<div style="font-size:0.68rem;white-space:nowrap;'
               f'{f"color:{_LIVE_RED};font-weight:600" if g.red_zone else "opacity:0.85"}">'
               f'{escape(text)}</div>')
    if live and g.to_goal is not None:
        # The offense's drive so far: its own goal line on the left, the one
        # it's going for on the right, the red zone tinted, a tick at midfield.
        offense = g.home if g.home.has_ball else g.away
        sit += (f'<div style="position:relative;height:4px;margin-top:3px;border-radius:2px;'
                f'background:rgba(128,128,128,0.25)">'
                f'<div style="position:absolute;right:0;top:0;bottom:0;width:20%;'
                f'background:rgba(214,39,40,0.3);border-radius:0 2px 2px 0"></div>'
                f'<div style="position:absolute;left:0;top:0;bottom:0;width:{100 - g.to_goal}%;'
                f'background:{offense.color or "#4c78a8"};border-radius:2px"></div>'
                f'<div style="position:absolute;left:50%;top:-1px;bottom:-1px;width:1px;'
                f'background:rgba(128,128,128,0.7)"></div></div>')
    title = f' title="{escape(g.last_play)}"' if live and g.last_play else ""
    border = _LIVE_RED if live and g.red_zone else "rgba(128,128,128,0.35)"
    return (f'<div{title} style="flex:0 0 auto;width:8.5rem;border:1px solid {border};'
            f'border-radius:6px;padding:3px 7px 4px;font-size:0.78rem;line-height:1.35">'
            f'{head}{teams}{sit}</div>')


def ticker_html(games: list[TickerGame]) -> str:
    """The league's other games as a row of ESPN-style cards that scrolls
    sideways: clock, score, who has the ball (🏈), down and distance, and a
    bar of the offense's drive toward the goal. Hovering a live card shows its
    last play. One line, no blank lines, so st.markdown keeps it raw HTML."""
    cards = "".join(_ticker_card(g) for g in games)
    return (f'<div style="display:flex;gap:6px;overflow-x:auto;padding:2px 0 6px;'
            f'margin-bottom:0.25rem;-webkit-overflow-scrolling:touch;scrollbar-width:thin">'
            f'{cards}</div>')


def keep_screen_awake(enabled: bool) -> None:
    """Hold (or release) a Screen Wake Lock so a phone doesn't dim or lock mid-game.

    st.iframe runs the HTML in a same-origin iframe, so the script requests the lock on
    the parent (top-level) document -- iframes need a permissions-policy grant that
    Streamlit doesn't give. The sentinel lives on the parent window so reruns reuse
    it, and it's re-acquired on visibilitychange because the browser drops the lock
    whenever the tab is hidden. Needs HTTPS (or localhost); iOS Safari 16.4+.
    """
    st.iframe(f"""
<script>
(function () {{
  const w = window.parent, nav = w.navigator, doc = w.document;
  const want = {str(enabled).lower()};
  w.__nflWakeWanted = want;
  if (!("wakeLock" in nav)) return;
  async function acquire() {{
    if (!w.__nflWakeWanted || doc.visibilityState !== "visible") return;
    if (w.__nflWakeLock && !w.__nflWakeLock.released) return;
    try {{ w.__nflWakeLock = await nav.wakeLock.request("screen"); }}
    catch (e) {{ console.warn("Wake lock not granted:", e); }}
  }}
  if (!w.__nflWakeListener) {{
    w.__nflWakeListener = true;
    doc.addEventListener("visibilitychange", acquire);
  }}
  if (want) acquire();
  else if (w.__nflWakeLock) {{ w.__nflWakeLock.release(); w.__nflWakeLock = null; }}
}})();
</script>
""", height="content")
