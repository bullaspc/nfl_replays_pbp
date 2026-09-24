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

There is no test suite or linter. Three scripts check the live path against published nflverse data (each takes a season and an optional local parquet path):

```bash
python tools/validate_models.py 2025         # nflfastr_models.py vs published ep/epa/wp
python tools/validate_text_parser.py 2025    # gamebook-text parser vs nflverse columns
python tools/validate_live_pipeline.py 2025  # live derivations + models vs official
python tools/validate_espn_adapter.py 2025   # ESPN-shaped JSON round trip
python tools/build_fg_table.py 2018 2025     # rebuild models/fg_make_prob.csv
```

## Architecture

The Streamlit app is `nfl_replay_app.py` (data loading, logic and UI). Two helper modules feed it games that nflverse hasn't published yet. Python 3.11.

- `live_feed.py`: ESPN summary JSON → the same nflfastR column layout (`PBP_COLS`). `parse_play_text()` reads the NFL gamebook text the way nflfastR does: play type, players, yards, results, tacklers, sacks, INTs, pass defenses, QB hits and forced fumbles. `add_derived_columns()` derives the rest: clock seconds, pre-play score differential, timeouts left (challenge timeouts included), first downs, `td_team`.
- `nflfastr_models.py`: nflfastR's own EP and WP xgboost models. They are extracted from `nflverse/fastrmodels` `.rda` files, downloaded once to `~/.cache/nfl_replays_pbp`. The module also ports nflfastR's EP/EPA/WP feature prep. The field-goal GAM can't run in Python, so its output is read from `models/fg_make_prob.csv`, which `tools/build_fg_table.py` recovers exactly from published pbp.

**Data sources:** nflverse pbp isn't live; it's rebuilt about once a day after games end.
- `nflverse_stamp()` reads `pbp/timestamp.json` every 60s. `load_pbp(season, stamp)` downloads again only when that stamp changes.
- `load_schedule()` reads the nflverse schedule, which has the ESPN event id, spread and roof.
- `list_games()` gives each game a `source`:
  - `"nflfastR"`: the game is in the published pbp.
  - `"live"`: kickoff has passed, it isn't published yet, and it has an ESPN id. These show 🔴 in the label.
- `load_live_game()` (cached 20s) builds a live game from ESPN plus nflfastR's models.
- The game moves to the official data by itself once nflverse publishes it.
- If the models can't load, EPA/WP stay NaN and the sidebar shows a warning.

**Data flow:**
1. `load_pbp` / `load_live_game` → `pbp_game` for the selected game
2. `play_timeline(pbp_game)` — elapsed game seconds per play (`3600 - game_seconds_remaining`, `ffill/bfill` for null-clock rows, `cummax` to keep it monotonic)
3. The sidebar's quarter + MM:SS inputs produce `baseline_elapsed`; `cursor_from_elapsed()` turns that into a **play cursor**
4. `revealed = pbp_game.iloc[: cursor_idx + 1]` drives all displayed sections

**Key invariant — spoiler safety:** Every data-driven section must only use `revealed` (the sliced DataFrame), never `pbp_game` directly. The `safety_margin` slider subtracts seconds from `baseline_elapsed` before it is converted to a cursor. The win probability chart x-axis is capped at `elapsed_s / 60` to prevent the chart shape itself from being a spoiler.

**Position model — the play cursor:** viewing position is a *play index*, not a game-seconds scalar. Three session-state keys:

- `_cursor_idx` — the play currently being viewed
- `_cursor_max` — furthest play unlocked; the time bar's slider `max_value`, and therefore the spoiler gate
- `_cursor_key` — `(game_id, qtr_pick, clock_str, safety_margin)`; a change re-seeds both from the clock baseline
- `_cursor_frame` / `_cursor_anchor` — `(game_id, source, len(pbp_game))` and a `(elapsed, rank-within-that-second)` anchor for both cursors. When the frame changes (live plays arrive, or live → official, whose indices differ), `cursor_from_anchor()` carries the position over by game time. It never lands on a later second, or on a later rank within the same second, so a switch can't unlock anything.

An index is used because a clock value cannot address plays individually: plays sharing one `game_seconds_remaining` (penalties, timeouts, two snaps in a second) collapse together. Overtime is worse — its clock restarts *inside* the regulation range (a 2024 OT game carries values like 442, 403, 363), so every OT play looks earlier than the end of Q4 and a game-seconds threshold reveals the whole OT period at once. This is also why `play_timeline()` applies `cummax`: the raw `3600 - clock` series runs backwards at the start of OT. `elapsed_s` is still derived from the cursor, but only for display and the WP x-axis cap — nothing filters on it.

**Viewing position (sidebar):** a single control — quarter + MM:SS game clock, converted to elapsed game seconds (`completed_qtrs * 900 + (900 - remaining)`; OT is `qtr=5`, treated as starting after Q4). Its only job is to seed the cursor; the time bar does the exact syncing from there. `safety_margin` is subtracted from the baseline before the cursor lookup, clamped at 0.

**Time bar (main panel, between the header metrics and the reveal gate):** an HTML track (`_time_bar_html()`, quarter ticks, fill coloured by possession team, OT segment only once the cursor is in OT) plus an `st.slider` capped at `_cursor_max` and a row of step buttons. The slider deliberately has **no `key=`** so it re-seeds from `_cursor_idx` each rerun and buttons/auto-advance can drive it without a widget-state exception. Only `▶ Next play` and `▶▶ Next drive` raise `_cursor_max`; scrubbing and `⏭ Catch up` never do. Nothing past the cursor is drawn — no scoring marks, no play density, and the caption says "of N *unlocked*", never a total, since the play count alone leaks whether the game ran long.

**Live refresh:** a live game that isn't over reruns every 30s (`live_refresh`) to pull in new plays. That only makes them reachable by the ▶ buttons; it never moves the cursor.

**Auto-advance:** When `auto=True`, `st_autorefresh` fires and the block at the bottom of the file bumps `_cursor_max` by one, carrying `_cursor_idx` with it only if it was already at the edge — so an auto-advancing replay doesn't yank you forward while you're scrubbing back through a drive.

**UI sections (top to bottom):** header metrics → boxscore → recent plays (paginated, 15/page) → current drive → team stats → player leaders → win probability chart.

**Stat tables:**
- `boxscore()` — quarter-by-quarter score via cumulative score diffs
- `team_stats()` — advanced EPA-based stats, returned as a transposed DataFrame (stat names as index, team abbrs as columns); styled by `style_stat_table()`
- `top_players()` — per-team passing/rushing/receiving leaders sorted by yards
