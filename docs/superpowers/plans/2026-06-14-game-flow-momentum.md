# Game Flow & Momentum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three new game-analysis sections to `nfl_replay_app.py`: a Scoring Timeline table, an Explosive Plays expander, and a rolling-EPA Momentum chart with score annotations.

**Architecture:** All changes are in the single-file app. New pure functions are added in the stat-builders block (after `_style_drive_chart`, before `top_plays_wpa`). New UI sections are inserted at specific positions in the vertical page layout. Everything operates on `revealed` (the spoiler-safe slice) — no changes to data loading or filtering logic.

**Tech Stack:** Python 3.11, Streamlit, pandas, plotly.express (already imported)

---

## File

- Modify: `nfl_replay_app.py`

Insertion points (all in one file):
- **Functions** → after line 773 (blank line after `_style_drive_chart`), before `def top_plays_wpa`
- **Scoring Timeline UI** → after line 1026 (after boxscore `st.dataframe`), before `# ---------- Recent plays`
- **Explosive Plays UI** → after line 1119 (end of current drive section), before `# ---------- Drive chart`
- **Momentum Chart** → after `if not hide_wp:` on line 1201, before `st.subheader("Win probability")`

---

## Task 1: Add `scoring_timeline()` and `_style_scoring_timeline()`

**Files:**
- Modify: `nfl_replay_app.py` (after line 773, before `def top_plays_wpa`)

- [ ] **Step 1: Insert `scoring_timeline()` after the blank line following `_style_drive_chart` (line 773)**

Find this exact string in `nfl_replay_app.py`:
```python


def top_plays_wpa(revealed: pd.DataFrame, home: str, away: str, n: int = 25) -> pd.DataFrame:
```

Replace with:
```python


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


def _style_scoring_timeline(df: pd.DataFrame):
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
        styled = _smap(styled, _type_color, subset=["Type"])
    return (
        styled
        .set_properties(**{"text-align": "center"})
        .set_table_styles(
            [{"selector": "th", "props": [("text-align", "center"), ("font-weight", "bold")]}]
        )
    )


def top_plays_wpa(revealed: pd.DataFrame, home: str, away: str, n: int = 25) -> pd.DataFrame:
```

- [ ] **Step 2: Verify the app starts cleanly**

Run: `streamlit run nfl_replay_app.py --server.enableCORS false --server.enableXsrfProtection false`

Expected: no import or syntax errors in terminal output.

- [ ] **Step 3: Commit**

```bash
git add nfl_replay_app.py
git commit -m "feat: add scoring_timeline() and _style_scoring_timeline()"
```

---

## Task 2: Add Scoring Timeline UI section

**Files:**
- Modify: `nfl_replay_app.py` (after line 1026, before `# ---------- Recent plays`)

- [ ] **Step 1: Insert the Scoring Timeline UI section**

Find this exact string in `nfl_replay_app.py`:
```python
st.dataframe(boxscore(revealed, home, away), hide_index=True, use_container_width=True)

# ---------- Recent plays (with pagination) ----------
```

Replace with:
```python
st.dataframe(boxscore(revealed, home, away), hide_index=True, use_container_width=True)

# ---------- Scoring timeline ----------
st.subheader("Scoring timeline")
if not revealed.empty:
    _stl_df = scoring_timeline(revealed, home, away)
    if not _stl_df.empty:
        _stl_cols = ["Q", "Clock", "Team", "Type", "Score"]
        if not hide_descriptions:
            _stl_cols = ["Q", "Clock", "Team", "Type", "Score", "Description"]
        st.dataframe(
            _style_scoring_timeline(_stl_df[_stl_cols]),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.caption("No scores yet.")
else:
    st.caption("No plays revealed yet.")

# ---------- Recent plays (with pagination) ----------
```

- [ ] **Step 2: Open the app and verify**

Load any game that has at least one score. Confirm:
- "Scoring timeline" section appears between Boxscore and Recent plays
- Rows show Q, Clock, Team, Type, Score columns
- TD rows are green, FG rows are blue
- With `hide_descriptions` checked in sidebar, Description column disappears
- With game at 0 elapsed seconds, section shows "No scores yet."

- [ ] **Step 3: Commit**

```bash
git add nfl_replay_app.py
git commit -m "feat: add Scoring Timeline UI section after boxscore"
```

---

## Task 3: Add `explosive_plays()` function

**Files:**
- Modify: `nfl_replay_app.py` (add after `_style_scoring_timeline`, before `def top_plays_wpa`)

- [ ] **Step 1: Insert `explosive_plays()` between `_style_scoring_timeline` and `top_plays_wpa`**

Find this exact string in `nfl_replay_app.py`:
```python

def top_plays_wpa(revealed: pd.DataFrame, home: str, away: str, n: int = 25) -> pd.DataFrame:
    """Top n plays by absolute win probability added, computed from home_wp shifts."""
```

Replace with:
```python

def explosive_plays(revealed: pd.DataFrame, min_pass_yds: int = 15, min_rush_yds: int = 10) -> pd.DataFrame:
    """Passing plays >= min_pass_yds yards or rushing plays >= min_rush_yds yards, sorted by yards desc."""
    if revealed.empty:
        return pd.DataFrame()
    pass_mask = (revealed["pass_attempt"].fillna(0) == 1) & (revealed["passing_yards"].fillna(0) >= min_pass_yds)
    rush_mask = (revealed["rush_attempt"].fillna(0) == 1) & (revealed["rushing_yards"].fillna(0) >= min_rush_yds)
    return revealed[pass_mask | rush_mask].sort_values("yards_gained", ascending=False)


def top_plays_wpa(revealed: pd.DataFrame, home: str, away: str, n: int = 25) -> pd.DataFrame:
    """Top n plays by absolute win probability added, computed from home_wp shifts."""
```

- [ ] **Step 2: Verify the app starts cleanly**

Run: `streamlit run nfl_replay_app.py --server.enableCORS false --server.enableXsrfProtection false`

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add nfl_replay_app.py
git commit -m "feat: add explosive_plays() function"
```

---

## Task 4: Add Explosive Plays UI section

**Files:**
- Modify: `nfl_replay_app.py` (after line 1119, before `# ---------- Drive chart`)

- [ ] **Step 1: Insert the Explosive Plays expander**

Find this exact string in `nfl_replay_app.py`:
```python
else:
    st.caption("No plays revealed yet.")

# ---------- Drive chart ----------
```

Replace with:
```python
else:
    st.caption("No plays revealed yet.")

# ---------- Explosive plays ----------
if not revealed.empty:
    _exp_df = explosive_plays(revealed)
    _exp_count = len(_exp_df)
    with st.expander(f"Explosive plays ({_exp_count})", expanded=False):
        if not _exp_df.empty:
            exp_display = _build_plays_df(_exp_df, hide_descriptions, reverse=False)
            st.dataframe(
                exp_display.style.apply(_style_plays, axis=1),
                hide_index=True,
                use_container_width=True,
                column_config={**_EPA_COL_CFG, "_rz": None},
            )
        else:
            st.caption("No explosive plays yet.")

# ---------- Drive chart ----------
```

- [ ] **Step 2: Open the app and verify**

Load any game with several plays revealed. Confirm:
- "Explosive plays (N)" expander appears between Current drive and Drive chart
- Count in the label is correct (manually count 15+ yd pass or 10+ yd rush plays in Recent plays)
- Expanding shows the table with the same style as Recent plays (3rd/4th down coloring, red zone red field)
- Red zone field position shows red when explosive play started in the red zone
- With `hide_descriptions` checked, Description column is absent

- [ ] **Step 3: Commit**

```bash
git add nfl_replay_app.py
git commit -m "feat: add Explosive Plays expander after current drive"
```

---

## Task 5: Add Momentum Chart

**Files:**
- Modify: `nfl_replay_app.py` (inside the `if not hide_wp:` block at line 1201, before `st.subheader("Win probability")`)

- [ ] **Step 1: Insert the Momentum chart block**

Find this exact string in `nfl_replay_app.py`:
```python
# ---------- Win probability chart ----------
if not hide_wp:
    st.subheader("Win probability")
```

Replace with:
```python
# ---------- Win probability chart ----------
if not hide_wp:
    st.subheader("Momentum")
    if not revealed.empty:
        _mom_scrimmage = revealed[
            (revealed["pass_attempt"].fillna(0) == 1) |
            (revealed["rush_attempt"].fillna(0) == 1)
        ].copy()
        if not _mom_scrimmage.empty:
            _mom_scrimmage["_elapsed_min"] = (
                (3600 - _mom_scrimmage["game_seconds_remaining"].fillna(3600)) / 60.0
            )
            _mom_team_colors = load_team_colors()
            _mom_color_map = {
                home: _mom_team_colors.get(home, "#1f77b4"),
                away: _mom_team_colors.get(away, "#ff7f0e"),
            }
            _mom_rows = []
            for _mom_team in [away, home]:
                _td = _mom_scrimmage[_mom_scrimmage["posteam"] == _mom_team].copy()
                if _td.empty:
                    continue
                _td["_rolling_epa"] = (
                    _td["epa"].fillna(0).rolling(5, center=True, min_periods=1).mean()
                )
                for _, _row in _td.iterrows():
                    _mom_rows.append({
                        "elapsed_min": _row["_elapsed_min"],
                        "team": _mom_team,
                        "rolling_epa": _row["_rolling_epa"],
                    })
            if _mom_rows:
                _mom_df = pd.DataFrame(_mom_rows)
                fig_mom = px.line(
                    _mom_df, x="elapsed_min", y="rolling_epa", color="team",
                    color_discrete_map=_mom_color_map,
                    labels={
                        "elapsed_min": "Game minutes elapsed",
                        "rolling_epa": "EPA/play (5-play rolling avg)",
                    },
                )
                fig_mom.update_yaxes(range=[-2, 2])
                _mom_x_cap = max(elapsed_s / 60.0, 1.0)
                fig_mom.update_xaxes(range=[0, _mom_x_cap])
                fig_mom.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                # Score event annotations (TD, FG, Safety only — skip XP/2PT)
                _mom_score_mask = (
                    (revealed["touchdown"].fillna(0) == 1) |
                    (revealed["field_goal_result"] == "made") |
                    (revealed["safety"].fillna(0) == 1)
                )
                _mom_scores = revealed[_mom_score_mask].copy()
                _mom_scores["_elapsed"] = (
                    (3600 - _mom_scores["game_seconds_remaining"].fillna(3600)) / 60.0
                )
                _mom_scores["_team"] = _mom_scores.apply(
                    lambda r: r["defteam"] if r["safety"] == 1 else r["posteam"], axis=1
                )
                _mom_scores["_type"] = _mom_scores.apply(
                    lambda r: "Safety" if r["safety"] == 1
                    else ("FG" if r["field_goal_result"] == "made" else "TD"),
                    axis=1,
                )
                for _, _se in _mom_scores.iterrows():
                    _se_color = _mom_team_colors.get(str(_se["_team"]), "#333333")
                    fig_mom.add_vline(
                        x=float(_se["_elapsed"]),
                        line_dash="dot",
                        line_color=_se_color,
                        opacity=0.7,
                        annotation_text=f"{_se['_team']} {_se['_type']}",
                        annotation_position="top",
                        annotation_font_size=10,
                    )
                st.plotly_chart(fig_mom, use_container_width=True)
        else:
            st.caption("Not enough plays for momentum chart.")

    st.subheader("Win probability")
```

- [ ] **Step 2: Open the app and verify**

Load a game with at least a quarter of play revealed. Confirm:
- "Momentum" section appears above "Win probability"
- Two lines (one per team) in team colors
- Zero line is visible as a gray dashed horizontal
- Score events appear as dotted vertical lines labeled e.g. "KC TD", "BUF FG"
- X-axis ends at current elapsed time (not beyond — no spoilers)
- Checking "Hide win probability chart" in the sidebar hides both the momentum chart AND the WP chart

- [ ] **Step 3: Commit**

```bash
git add nfl_replay_app.py
git commit -m "feat: add rolling-EPA Momentum chart with score annotations"
```

---

## Self-Review Notes

- `scoring_timeline()` handles safety scoring correctly: `defteam` is used as `Team` when `safety == 1`.
- Momentum chart score annotations recompute the score mask directly from `revealed` rather than calling `scoring_timeline()` to avoid column mismatch — `game_seconds_remaining` isn't exposed in the scoring timeline output.
- `explosive_plays()` uses `passing_yards` (not `yards_gained`) to filter pass plays, matching the intent of "15+ yard pass play" regardless of penalty yardage on the play.
- All three sections respect `hide_descriptions` (scoring timeline and explosive plays) and `hide_wp` (momentum chart).
- No new columns added to `load_pbp()` — all required columns (`touchdown`, `field_goal_result`, `safety`, `extra_point_result`, `two_point_conv_result`, `passing_yards`, `rushing_yards`, `epa`, `game_seconds_remaining`, `defteam`) were already loaded.
