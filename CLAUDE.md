# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app (default port 8501)
streamlit run nfl_replay_app.py

# Run with CORS/XSRF disabled (matches devcontainer config)
streamlit run nfl_replay_app.py --server.enableCORS false --server.enableXsrfProtection false
```

There are no tests or linting configurations in this project.

## Architecture

This is a single-file Streamlit application (`nfl_replay_app.py`). The entire app — data loading, logic, and UI — lives in that one file. Python 3.11.

**Data flow:**
1. `load_pbp(season)` — fetches NFL play-by-play via `nfl_data_py`, cached for 2 minutes with `@st.cache_data`
2. `list_games(pbp)` — extracts a game selector list from the full season PBP
3. `play_timeline(pbp_game)` — elapsed game seconds per play (`3600 - game_seconds_remaining`, `ffill/bfill` for null-clock rows, `cummax` to keep it monotonic)
4. The sidebar's quarter + MM:SS inputs produce `baseline_elapsed`; `cursor_from_elapsed()` turns that into a **play cursor**
5. `revealed = pbp_game.iloc[: cursor_idx + 1]` drives all displayed sections

**Key invariant — spoiler safety:** Every data-driven section must only use `revealed` (the sliced DataFrame), never `pbp_game` directly. The `safety_margin` slider subtracts seconds from `baseline_elapsed` before it is converted to a cursor. The win probability chart x-axis is capped at `elapsed_s / 60` to prevent the chart shape itself from being a spoiler.

**Position model — the play cursor:** viewing position is a *play index*, not a game-seconds scalar. Three session-state keys:

- `_cursor_idx` — the play currently being viewed
- `_cursor_max` — furthest play unlocked; the time bar's slider `max_value`, and therefore the spoiler gate
- `_cursor_key` — `(game_id, qtr_pick, clock_str, safety_margin)`; a change re-seeds both from the clock baseline

An index is used because a clock value cannot address plays individually: plays sharing one `game_seconds_remaining` (penalties, timeouts, two snaps in a second) collapse together. Overtime is worse — its clock restarts *inside* the regulation range (a 2024 OT game carries values like 442, 403, 363), so every OT play looks earlier than the end of Q4 and a game-seconds threshold reveals the whole OT period at once. This is also why `play_timeline()` applies `cummax`: the raw `3600 - clock` series runs backwards at the start of OT. `elapsed_s` is still derived from the cursor, but only for display and the WP x-axis cap — nothing filters on it.

**Viewing position (sidebar):** a single control — quarter + MM:SS game clock, converted to elapsed game seconds (`completed_qtrs * 900 + (900 - remaining)`; OT is `qtr=5`, treated as starting after Q4). Its only job is to seed the cursor; the time bar does the exact syncing from there. `safety_margin` is subtracted from the baseline before the cursor lookup, clamped at 0.

**Time bar (main panel, between the header metrics and the reveal gate):** an HTML track (`_time_bar_html()`, quarter ticks, fill coloured by possession team, OT segment only once the cursor is in OT) plus an `st.slider` capped at `_cursor_max` and a row of step buttons. The slider deliberately has **no `key=`** so it re-seeds from `_cursor_idx` each rerun and buttons/auto-advance can drive it without a widget-state exception. Only `▶ Next play` and `▶▶ Next drive` raise `_cursor_max`; scrubbing and `⏭ Catch up` never do. Nothing past the cursor is drawn — no scoring marks, no play density, and the caption says "of N *unlocked*", never a total, since the play count alone leaks whether the game ran long.

**Auto-advance:** When `auto=True`, `st_autorefresh` fires and the block at the bottom of the file bumps `_cursor_max` by one, carrying `_cursor_idx` with it only if it was already at the edge — so an auto-advancing replay doesn't yank you forward while you're scrubbing back through a drive.

**UI sections (top to bottom):** header metrics → boxscore → recent plays (paginated, 15/page) → current drive → team stats → player leaders → win probability chart.

**Stat tables:**
- `boxscore()` — quarter-by-quarter score via cumulative score diffs
- `team_stats()` — advanced EPA-based stats, returned as a transposed DataFrame (stat names as index, team abbrs as columns); styled by `style_stat_table()`
- `top_players()` — per-team passing/rushing/receiving leaders sorted by yards
