"""
Presentation helpers for the app: table styling, the drive field figure, the
play tables, the time bar and the screen wake lock. They turn replay_core's
DataFrames into what Streamlit draws; no data loading here.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from replay_core import (LOWER_IS_BETTER, PLAY_CATS, SR_METRICS, down_distance, field_pos_label,
                         fmt_top, percentile_of, play_type_label, success_emoji, yl_label)


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


# ---------- Drives ----------
_FIELD_GREEN = "#2f7a3b"
_FIELD_GREEN_ALT = "#2a6f35"


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
    is_special = raw["play_type"].isin(["field_goal", "extra_point"])
    raw["_rz"] = (raw["yardline_100"].fillna(100) <= 20) & ~is_special
    if hide_desc:
        cols_sel = ["Q", "time", "posteam", "Field", "_rz", "Type", "D&D", "Success?", "yards_gained", "epa"]
        df = raw[cols_sel].copy()
        df.columns = ["Q", "Clock", "Off", "Field", "_rz", "Type", "D&D", "Success?", "Yds", "EPA"]
    else:
        cols_sel = ["Q", "time", "posteam", "Field", "_rz", "Type", "D&D", "Success?", "desc", "yards_gained", "epa"]
        df = raw[cols_sel].copy()
        df.columns = ["Q", "Clock", "Off", "Field", "_rz", "Type", "D&D", "Success?", "Description", "Yds", "EPA"]
    df["Yds"] = pd.to_numeric(df["Yds"], errors="coerce").fillna(0).astype(int)
    df["EPA"] = pd.to_numeric(df["EPA"], errors="coerce").round(2)
    return df.iloc[::-1] if reverse else df

def style_plays(row):
    t = row["Type"]
    if t.startswith("3rd"):
        row_bg = "#ffeeba"
    else:
        row_bg = "#ffffff"
    styles = []
    for col in row.index:
        if col == "_rz":
            styles.append("")
        elif col == "Field" and row.get("_rz", False):
            styles.append("background-color: #dc3545; color: #ffffff")
        else:
            styles.append(f"background-color: {row_bg}; color: #000000")
    return styles


EPA_COL_CFG = {"EPA": st.column_config.NumberColumn(format="%.2f")}
PLAYS_COL_CFG = {
    **EPA_COL_CFG, "_rz": None,
    "Description": st.column_config.TextColumn(width="large"),
}


def plays_row_height(hide_desc: bool) -> int | None:
    """Row height for play tables: st.dataframe only wraps cell text when
    row_height is above 4rem (64px), so give descriptions room for ~3 lines."""
    return None if hide_desc else 84


# ---------- Header and time bar ----------
def hex_or_none(v):
    """A "#rrggbb" color, or None for blank/NaN cells."""
    return v if isinstance(v, str) and v.startswith("#") else None


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

def keep_screen_awake(enabled: bool) -> None:
    """Hold (or release) a Screen Wake Lock so a phone doesn't dim or lock mid-game.

    components.html runs in a same-origin iframe, so the script requests the lock on
    the parent (top-level) document -- iframes need a permissions-policy grant that
    Streamlit doesn't give. The sentinel lives on the parent window so reruns reuse
    it, and it's re-acquired on visibilitychange because the browser drops the lock
    whenever the tab is hidden. Needs HTTPS (or localhost); iOS Safari 16.4+.
    """
    components.html(f"""
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
""", height=0)
