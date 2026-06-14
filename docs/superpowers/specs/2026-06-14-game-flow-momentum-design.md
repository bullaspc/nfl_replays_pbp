# Game Flow & Momentum — Design Spec

**Date:** 2026-06-14  
**Cluster:** Game Flow & Momentum (1 of 3 planned depth clusters)  
**File:** `nfl_replay_app.py` (single-file Streamlit app)

---

## Overview

Add three new sections to the NFL Tape-Delay Replay app that deepen game-flow analysis:

1. **Scoring Timeline** — a table of every scoring play, placed after the boxscore
2. **Explosive Plays** — a collapsible table of chunk plays, placed after the current drive section
3. **Momentum Chart** — a rolling EPA chart annotated with score events, placed above the win probability chart

All three sections operate only on `revealed` (the spoiler-safe filtered slice) and respect existing spoiler-shield toggles.

---

## Section 1: Scoring Timeline

### Placement
After the boxscore, before recent plays.

### Function signature
```python
def scoring_timeline(revealed: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
```

### Logic
Scan `revealed` for scoring rows using these conditions (any one true = scoring play):
- `touchdown == 1`
- `field_goal_result == 'made'`
- `safety == 1`
- `extra_point_result == 'good'`
- `two_point_conv_result == 'success'`

For each matching row, derive:
- **Type** label: `"TD"`, `"FG"`, `"Safety"`, `"XP"`, or `"2PT"`
- **Team**: `posteam` (for safeties, the scoring team is `defteam` — handle this)
- **Score**: formatted as `"{away} {away_score} — {home_score} {home}"` using `total_away_score` / `total_home_score` from that row

### Output columns
`Q | Clock | Team | Type | Score | Description`

`Description` is omitted when `hide_descriptions` is `True`.

### Styling
Reuse the color language already established in `_style_drive_chart`:
- TD rows: green (`#d4edda` / `#155724`)
- FG rows: blue (`#cce5ff` / `#004085`)
- Safety rows: red (`#f8d7da` / `#721c24`)
- XP / 2PT rows: neutral gray (`#e2e3e5` / `#383d41`)

Apply color to the `Type` column only (not the whole row), consistent with how `_style_drive_chart` colors `Outcome`.

---

## Section 2: Explosive Plays

### Placement
After the current drive section, inside an `st.expander` (collapsed by default). Expander label includes a live count: `"Explosive plays (N)"`.

### Function signature
```python
def explosive_plays(revealed: pd.DataFrame, min_pass_yds: int = 15, min_rush_yds: int = 10) -> pd.DataFrame:
```

### Logic
Filter `revealed` for:
- Pass plays (`pass_attempt == 1`) where `passing_yards >= min_pass_yds`
- Rush plays (`rush_attempt == 1`) where `rushing_yards >= min_rush_yds`

Sort by `yards_gained` descending.

### Display
Pass through `_build_plays_df` (existing helper) with `reverse=False`, then style with `_style_plays` (existing helper). This gives identical column layout, row colors (3rd/4th down), and red-zone field highlighting to the recent plays table.

Columns: `Q | Clock | Off | Field | Type | D&D | Success? | Yds | EPA` (plus `Description` when not hidden).

If no explosive plays exist yet, show `st.caption("No explosive plays yet.")` inside the expander.

---

## Section 3: Momentum Chart

### Placement
Immediately above the existing win probability chart. Both charts are gated by the same `hide_wp` sidebar checkbox — when checked, both disappear.

### Logic
From `revealed`, take all scrimmage plays (pass or rush attempts) in chronological order. For each offensive team, compute a **5-play centered rolling mean of EPA** using `pandas.Series.rolling(5, center=True, min_periods=1).mean()`.

Build a long-format DataFrame with columns `[elapsed_min, team, rolling_epa]` where `elapsed_min = (3600 - game_seconds_remaining) / 60`.

### Chart
Plotly line chart:
- One line per team, colored by `load_team_colors()` (same color map used in the WP chart)
- X-axis: `"Game minutes elapsed"`, range `[0, elapsed_s / 60]` (spoiler-safe cap)
- Y-axis: `"EPA/play (5-play rolling avg)"`, range approximately `[-2, 2]`
- Horizontal zero line at `y=0` (dashed, gray) to make positive/negative momentum immediately readable

### Score annotations
For each row in `scoring_timeline(revealed, home, away)`:
- Add a vertical dashed line at the play's `elapsed_min`
- Color the line by the scoring team's team color
- Label it with a short string: `"{team} {type}"` (e.g. `"KC TD"`, `"BUF FG"`)

Use `fig.add_vline` with `annotation_text` for each score event.

---

## Spoiler Safety

All three sections operate exclusively on `revealed` — the slice already filtered by `elapsed_s - safety_margin`. No additional spoiler guards needed beyond what `filter_revealed` already enforces.

The momentum chart and win probability chart share the same x-axis cap (`elapsed_s / 60`) and the same `hide_wp` toggle, so neither chart shape can reveal future game state.

---

## Data columns required

All required columns are already fetched in `load_pbp()`:
- `touchdown`, `field_goal_result`, `safety`, `extra_point_result`, `two_point_conv_result` — for scoring timeline
- `passing_yards`, `rushing_yards`, `pass_attempt`, `rush_attempt` — for explosive plays
- `epa`, `game_seconds_remaining`, `posteam` — for momentum chart

No changes to `load_pbp()` column list needed.

---

## Page layout after changes

```
[Header metrics]
[Boxscore]
[Scoring Timeline]          ← NEW
[Recent plays]
[Current drive]
[Explosive Plays expander]  ← NEW
[Drive chart]
[Team stats]
[Situational success rates]
[Player leaders]
[Momentum Chart]            ← NEW
[Win probability chart]
[Top plays by WPA]
```

---

## Out of scope for this cluster

- Passing game depth (pressure rate, time to throw, clean-pocket splits) — planned as cluster 2
- Special teams (FG tracker, punt stats, kick returns) — planned as cluster 3
- Thresholds for explosive plays are hardcoded (15 pass / 10 rush); no sidebar controls
