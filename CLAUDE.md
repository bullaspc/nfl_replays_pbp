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
3. Sidebar inputs determine `elapsed_s`: how many game-seconds the user has "watched"
4. `filter_revealed(pbp_game, elapsed_s)` — filters plays by `game_seconds_remaining` (counts down from 3600); uses `ffill/bfill` to handle null-clock rows (timeouts, admin plays)
5. The revealed slice drives all displayed sections

**Key invariant — spoiler safety:** Every data-driven section must only use `revealed` (the filtered DataFrame), never `pbp_game` directly. The `safety_margin` slider subtracts additional seconds from `elapsed_s` before filtering. The win probability chart x-axis is capped at `elapsed_s / 60` to prevent the chart shape itself from being a spoiler.

**Viewing position modes (sidebar):**
- *Started at* — computes elapsed wall-clock minutes since a chosen datetime, converts via `elapsed_game_seconds()` (190 broadcast min → 3600 game seconds linear mapping)
- *X minutes in* — direct broadcast-minute input
- *Jump to game clock* — quarter + MM:SS input, converts to game-seconds elapsed

**Auto-advance:** When `auto=True` in the non-wall-clock modes, `st_autorefresh` fires every 30s and `st.session_state["_fix_elapsed"]` advances to the next play's timestamp (bottom of file).

**UI sections (top to bottom):** header metrics → boxscore → recent plays (paginated, 15/page) → current drive → team stats → player leaders → win probability chart.

**Stat tables:**
- `boxscore()` — quarter-by-quarter score via cumulative score diffs
- `team_stats()` — advanced EPA-based stats, returned as a transposed DataFrame (stat names as index, team abbrs as columns); styled by `style_stat_table()`
- `top_players()` — per-team passing/rushing/receiving leaders sorted by yards
