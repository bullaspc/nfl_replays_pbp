# Percentile-Based Team Stats Table Coloring — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Color each cell in the team stats table using a red→white→green diverging palette based on where that value falls in the distribution of per-game team stats from the 3 prior NFL seasons.

**Architecture:** A new `load_stat_baselines(season)` cached function loads 3 seasons of PBP, groups by (game_id, posteam), and computes per-game distributions for each stat. `style_stat_table` gains a `baselines` parameter; for each non-EPA cell it computes a percentile rank and maps it to an interpolated RGB color. EPA rows keep their existing positive/negative coloring.

**Tech Stack:** Python 3.11, pandas, numpy, nfl_data_py, Streamlit

---

## File Map

| File | Change |
|---|---|
| `nfl_replay_app.py:16–21` | Add `import numpy as np` |
| `nfl_replay_app.py:126–127` | Add `_LOWER_IS_BETTER`, `_percentile_of`, `_percentile_color` after `_rate` |
| `nfl_replay_app.py:203` | Add `load_stat_baselines(season)` after `team_stats` |
| `nfl_replay_app.py:294–346` | Update `_EPA_ROWS` constants; rewrite `style_stat_table` to use percentile coloring |
| `nfl_replay_app.py:720–724` | Add `baselines` load, pass to `style_stat_table`, update stat table caption |

---

## Task 1: Add numpy import and low-level helpers

**Files:**
- Modify: `nfl_replay_app.py:16–21` (imports)
- Modify: `nfl_replay_app.py:126–128` (after `_rate`)

- [ ] **Step 1: Add numpy import**

In `nfl_replay_app.py`, change the import block from:

```python
import streamlit as st
import pandas as pd
import nfl_data_py as nfl
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta, time
```

to:

```python
import streamlit as st
import pandas as pd
import numpy as np
import nfl_data_py as nfl
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta, time
```

- [ ] **Step 2: Add helpers and constant after `_rate`**

After line 127 (`return num / den if den > 0 else float("nan")`), insert:

```python
_LOWER_IS_BETTER = {"Turnovers"}


def _percentile_of(value: float, sorted_arr: np.ndarray) -> float:
    """0–100 percentile rank of value in a pre-sorted array."""
    if pd.isna(value) or len(sorted_arr) == 0:
        return float("nan")
    return float(np.searchsorted(sorted_arr, value, side="right") / len(sorted_arr) * 100)


def _percentile_color(pct: float) -> str:
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
```

- [ ] **Step 3: Quick sanity check in terminal**

```bash
cd /home/locker/Documents/test_python/nfl_play_by_play
python - <<'EOF'
import numpy as np, pandas as pd
# paste helpers here to verify
def _percentile_of(value, sorted_arr):
    if pd.isna(value) or len(sorted_arr) == 0:
        return float("nan")
    return float(np.searchsorted(sorted_arr, value, side="right") / len(sorted_arr) * 100)

def _percentile_color(pct):
    if pd.isna(pct): return ""
    if pct <= 50:
        t = pct / 50.0
        r = int(220 + t*(255-220)); g = int(53+t*(255-53)); b = int(69+t*(255-69))
    else:
        t = (pct-50)/50.0
        r = int(255+t*(40-255)); g = int(255+t*(167-255)); b = int(255+t*(69-255))
    lum = (0.299*r + 0.587*g + 0.114*b)/255
    text = "#ffffff" if lum < 0.45 else "#212529"
    return f"background-color: #{r:02x}{g:02x}{b:02x}; color: {text}"

arr = np.sort(np.array([0.3,0.4,0.5,0.6,0.7]))
print(_percentile_of(0.5, arr))   # ~60.0
print(_percentile_of(0.3, arr))   # ~20.0
print(_percentile_color(0))       # deep red bg, white text
print(_percentile_color(50))      # white bg, dark text
print(_percentile_color(100))     # green bg, white text
EOF
```

Expected output (approximately):
```
60.0
20.0
background-color: #dc3545; color: #ffffff
background-color: #ffffff; color: #212529
background-color: #28a745; color: #ffffff
```

- [ ] **Step 4: Commit**

```bash
git add nfl_replay_app.py
git commit -m "feat: add numpy import and percentile color helpers"
```

---

## Task 2: Add `load_stat_baselines(season)` function

**Files:**
- Modify: `nfl_replay_app.py` — insert after `team_stats` (currently ends at line 202)

- [ ] **Step 1: Insert `load_stat_baselines` after `team_stats`**

After the `return pd.DataFrame(stats, index=pd.Index(index, name="Stat"))` line that closes `team_stats`, insert:

```python
@st.cache_data(ttl=86400)
def load_stat_baselines(season: int) -> dict[str, np.ndarray]:
    """Per-stat distributions from the 3 seasons prior to `season`, sorted ascending."""
    prior = [s for s in [season - 3, season - 2, season - 1] if s >= 1999]
    if not prior:
        return {}
    cols = [
        "game_id", "posteam",
        "pass_attempt", "rush_attempt", "epa",
        "passing_yards", "rushing_yards",
        "complete_pass", "air_yards",
        "interception", "fumble_lost", "first_down",
        "third_down_converted", "third_down_failed",
        "yardline_100", "touchdown",
    ]
    try:
        raw = nfl.import_pbp_data(prior, columns=cols, downcast=True)
    except Exception:
        return {}

    raw = raw[raw["posteam"].notna()]

    def _game_stats(td):
        pass_mask = td["pass_attempt"].fillna(0) == 1
        rush_mask = td["rush_attempt"].fillna(0) == 1
        sc_mask   = pass_mask | rush_mask
        pp = int(pass_mask.sum())
        rp = int(rush_mask.sum())
        tp = int(sc_mask.sum())
        pass_yds = td["passing_yards"].fillna(0).sum()
        rush_yds = td["rushing_yards"].fillna(0).sum()
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
        rz_td    = td.loc[rz_mask, "touchdown"].fillna(0).sum()
        tos      = int(td["interception"].fillna(0).sum() + td["fumble_lost"].fillna(0).sum())
        return pd.Series({
            "Plays":        tp,
            "Pass Plays":   pp,
            "Rush Plays":   rp,
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
        })

    per_game = raw.groupby(["game_id", "posteam"]).apply(
        _game_stats, include_groups=False
    )
    return {
        stat: np.sort(per_game[stat].dropna().values)
        for stat in per_game.columns
    }
```

- [ ] **Step 2: Verify it parses without errors**

```bash
cd /home/locker/Documents/test_python/nfl_play_by_play
python -c "
import ast, sys
with open('nfl_replay_app.py') as f:
    src = f.read()
ast.parse(src)
print('OK — no syntax errors')
"
```

Expected: `OK — no syntax errors`

- [ ] **Step 3: Commit**

```bash
git add nfl_replay_app.py
git commit -m "feat: add load_stat_baselines for per-game historical distributions"
```

---

## Task 3: Rewrite `style_stat_table` to use percentile coloring

**Files:**
- Modify: `nfl_replay_app.py:294–346`

- [ ] **Step 1: Replace `style_stat_table` and surrounding constants**

Replace the entire block from `# EPA/play columns` through the closing `return styled` of `style_stat_table` (currently lines 294–346) with:

```python
_EPA_ROWS     = {"Pass EPA/play", "Rush EPA/play", "EPA/play"}
_RATE_ROWS    = {"CMP%", "Pass SR", "Rush SR", "1st Down %", "3rd Down %", "RZ TD%"}
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
            styled = styled.map(_color_epa, subset=pd.IndexSlice[row, :])

    for row in df.index:
        if row in _EPA_ROWS:
            continue
        arr   = baselines.get(row, np.array([]))
        lower = row in _LOWER_IS_BETTER

        def _cell(val, _arr=arr, _lower=lower):
            if pd.isna(val):
                return ""
            pct = _percentile_of(float(val), _arr)
            if pd.isna(pct):
                return ""
            if _lower:
                pct = 100.0 - pct
            return _percentile_color(pct)

        styled = styled.map(_cell, subset=pd.IndexSlice[row, :])

    styled = styled.set_properties(**{"text-align": "right"})
    styled = styled.set_table_styles(
        [{"selector": "th", "props": [("text-align", "center"), ("font-weight", "bold")]}]
    )
    return styled
```

- [ ] **Step 2: Check for syntax errors**

```bash
cd /home/locker/Documents/test_python/nfl_play_by_play
python -c "
import ast
with open('nfl_replay_app.py') as f:
    src = f.read()
ast.parse(src)
print('OK — no syntax errors')
"
```

Expected: `OK — no syntax errors`

- [ ] **Step 3: Commit**

```bash
git add nfl_replay_app.py
git commit -m "feat: rewrite style_stat_table with percentile-diverging palette"
```

---

## Task 4: Wire baselines into the call site

**Files:**
- Modify: `nfl_replay_app.py:720–724`

- [ ] **Step 1: Update team stats section**

Find the team stats block (currently):

```python
# ---------- Team stats ----------
st.subheader("Team stats")
stat_df = team_stats(revealed, home, away)
st.dataframe(style_stat_table(stat_df, away, home), use_container_width=True)
```

Replace with:

```python
# ---------- Team stats ----------
st.subheader("Team stats")
stat_df = team_stats(revealed, home, away)
_baselines = load_stat_baselines(season)
st.dataframe(style_stat_table(stat_df, away, home, _baselines), use_container_width=True)
st.caption("Colors show percentile vs last 3 seasons · green = top of league · red = bottom")
```

- [ ] **Step 2: Verify syntax**

```bash
cd /home/locker/Documents/test_python/nfl_play_by_play
python -c "
import ast
with open('nfl_replay_app.py') as f:
    src = f.read()
ast.parse(src)
print('OK — no syntax errors')
"
```

Expected: `OK — no syntax errors`

- [ ] **Step 3: Start the app and visually verify**

```bash
streamlit run nfl_replay_app.py --server.enableCORS false --server.enableXsrfProtection false
```

Check:
1. App starts without errors (watch the terminal for tracebacks)
2. Select a game from a past season (e.g. 2024)
3. Advance the clock so some plays are revealed
4. Team stats table shows colored cells — red at low values, white near average, green at top values
5. EPA rows (`Pass EPA/play`, `Rush EPA/play`, `EPA/play`) still show green/red based on sign, not percentile
6. `Turnovers` row: a team with 0 turnovers should be green (best in league), many turnovers should be red
7. Caption "Colors show percentile vs last 3 seasons" appears below the table
8. If you select a 2025 game, baselines load from 2022–2024; if you select 2024, from 2021–2023

- [ ] **Step 4: Commit**

```bash
git add nfl_replay_app.py
git commit -m "feat: wire percentile baselines into team stats table"
```
