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

There is no test suite or linter. These scripts check the data paths against published data (each takes a season and an optional local parquet path):

```bash
python tools/validate_models.py 2025         # nflfastr_models.py vs published ep/epa/wp
python tools/validate_text_parser.py 2025    # gamebook-text parser vs nflverse columns
python tools/validate_live_pipeline.py 2025  # live derivations + models vs official
python tools/validate_espn_adapter.py 2025   # ESPN-shaped JSON round trip
python tools/build_fg_table.py 2018 2025     # rebuild models/fg_make_prob.csv
python tools/validate_cfb_adapter.py 2026     # cfb_feed.to_pbp vs college final scores, ESPN team box + player box (defense)
python tools/validate_cfb_live.py 2026 [path] [n]  # live college path on ESPN JSON rebuilt from published games
```

Dependencies: pandas 2.x (`<3`). NFL data comes through `nflreadpy`, nflverse's successor to `nfl_data_py` (whose last release pins pandas and numpy < 2). `sportsdataverse` processes live college games and needs pandas 2.

## Architecture

Python 3.11. The app covers the NFL and college football. A sidebar **League** radio picks a `League` object, and nothing else in the page branches on the league: both leagues produce play-by-play in the same nflfastR column layout (`PBP_COLS`).

- `nfl_replay_app.py`: the Streamlit page and the only Streamlit script: sidebar, play cursor state and the sections, top to bottom. No data loading or stat logic of its own. The file name is kept so deployments don't need reconfiguring.
- `leagues.py`: where games come from. NFL data is read with `nflreadpy` (`nflverse_pbp()` downcasts float64 to float32 as `nfl_data_py` did; `nflverse_teams()`; `load_schedule()`). Its own cache is turned off (`update_config(cache_mode="off")`) so `nflverse_stamp()` alone decides when a season is downloaded again. `League` (frozen dataclass) carries the per-league settings (`title`, `first_season`, `broadcast_minutes`, `baseline_label`, `group_label` for the conference filter, `untimed_ot`, `key` for game_summary) and the methods the page calls: `stamp()`, `games(season, stamp)` (game_id, label, source, week_label, and `groups` when `group_label` is set), `load_game(row, season, stamp)` → `LoadedGame` (pbp, live_state, sidebar notes, warning), `team_meta()`, `baseline_pbp()` and `missing_stats(revealed)` (Player-leader columns the game's data can't fill: Hurries for the NFL; QB Hits for college, plus Tackles/TFL/Hurries when its text names no tacklers). Both raise `GameLoadError` with a message for the sidebar. `NFLLeague` and `CollegeLeague` implement them; `LEAGUES` maps the radio's option to each. All Streamlit caching lives here, including `stat_baselines()` / `situational_baselines()`.
- `replay_core.py`: the league-neutral logic, no Streamlit: `PBP_COLS`, `play_timeline()` and the cursor functions, `period_label()`, every stat builder (boxscore, team stats, situational success, leaders, drives, scoring, WPA) and the percentile math (`stat_distributions()`, `situational_distributions()` over `BASELINE_COLS`). Tools import from it.
- `views.py`: presentation: Styler functions for the tables, `drive_field_figure()`, the play tables (`build_plays_df()`), `time_bar_html()`, `logo_img()` and `keep_screen_awake()`.

- `live_feed.py`: ESPN summary JSON → the same nflfastR column layout (`PBP_COLS`). `parse_play_text()` reads the NFL gamebook text the way nflfastR does: play type, players, yards, results, tacklers, sacks, INTs, pass defenses, QB hits and forced fumbles. Drive numbers come from possession changes (`_possession_drives()`, nflfastR's fixed_drive rules), not ESPN's drive grouping, which lags after turnovers mid-game. `add_derived_columns()` derives the rest: clock seconds, pre-play score differential, timeouts left (challenge timeouts included), first downs, `td_team`.
- `game_summary.py`: the AI game summary agent. The app builds a `GameContext` from its own stat functions run on `revealed`: boxscore, scoring timeline, team stats with `stat_percentiles()`, situational success, drive chart, top WPA plays, explosive plays and player leaders. `summarize()` sends a text snapshot (scoreboard, offense table with percentiles, a derived defense-allowed table) and runs a manual tool loop of up to `MAX_TURNS`. The tools are key plays by kind, drives, one drive's plays, offensive splits, situational success and player leaders. It calls any Anthropic-compatible Messages endpoint. `configured_providers()` lists the ones that have a key: Kimi (Moonshot `/anthropic`, Bearer auth, default) and Claude. Only Claude models get `output_config.effort`, `strict` tools, prompt caching and the `fallbacks: "default"` beta. It makes no Streamlit calls, which lets the page run it on a worker thread: `submit()` hands it to a module-level `ThreadPoolExecutor` and returns a Future plus a `Progress` (current `status`, model `calls`, `steps` done) that `summarize()` updates as it goes; `describe_lookup()` turns each tool call into a line like "Looking at ALA turnovers".
- `cfb_feed.py`: college football from sportsdataverse's `espn_cfb_pbp` release (ESPN's feed with sportsdataverse-py's college EP/WP models, rebuilt about daily, 2004 onward), plus its schedules and team info. Only `RAW_COLS` are read: the release also has final scores, drive results, `lead_*` and after-play columns, which would spoil. `to_pbp()` maps to `PBP_COLS` the way `live_feed` builds the NFL layout: receiving team on kickoffs, the try split off the touchdown row (it's on the same row in ESPN's college feed), END QUARTER/END OVERTIME/END GAME rows, drives from possession changes. It also repairs the feed: rows listed after a later period go back into their own (`_relocate_late_rows`), scores keep the longest never-decreasing chain (`_clean_scores`; some rows carry a stale score or even the final score mid-game), and isolated clock spikes are dropped. Overtime rows get no clock (college OT is untimed). Defenders: most games since late 2025 have NCAA stat-crew text (`(08:47) Shotgun #7 A.Powell rush ... to the NDSU09 (#51 G.Sell; #9 K.Ford Jr)`), and `_crew_columns()` reads tacklers (the parentheses after the ball carrier's spot, before any turnover, return, flag or review note), QB hurries, pass breakups, interceptors and fumble forcers from it, into nflfastR's tackle/sack/TFL/pass-defense/forced-fumble columns plus `half_tfl_*` and `qb_hurry_1..3_player_name`. College credit: one tackler is solo, two are an assist each; a shared sack or TFL is a half each; the sacker also gets the tackle. On those rows the text's names replace the feed's (and a game's other rows get `crew_spelling()`), so each player has one spelling. Older text names no tacklers; `missing_leader_stats()` then drops Tackles/TFL/Hurries, and the feed has no QB hits at all. Live games: `fetch_summary()` reads ESPN's college summary and `live_raw()` runs it through sportsdataverse-py's `CFBPlayProcess` (the pipeline that builds the release, so its rows have the release's columns and `to_pbp()` maps them unchanged; the one rename is `pos_team` → `pos_team_id`). It runs with `join_participants=False` and no 4th-down/two-point models, so only the spread can hit the network (ESPN's odds endpoint, when the summary has no pickcenter). `list_games()` adds schedule games that kick off by today, have an FBS team and aren't published as `source="live"`; `team_ids()` fills the schedule's blank abbreviations for upcoming games from published plays, then team info. No Streamlit calls.
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
- College: `cfb_stamp()` reads the release's `timestamp.json` every 60s; `load_cfb_season()` (`cache_resource`, shared, don't mutate) holds the raw season, `list_cfb_games(season, stamp, today)` lists published games plus today's live ones with week label, AP ranks and conferences (the sidebar filters by conference). A season not published yet still lists its live games. `source == "published"` → `load_cfb_game()` maps one game; `source == "live"` → `load_cfb_live_game()` (cached 20s) runs ESPN's feed through sportsdataverse. A live game moves to the published data by itself once sportsdataverse publishes it, and Follow live / live refresh work as for the NFL. Team colors/logos/nicknames come from `load_cfb_team_meta()`.
- The page reads `team_colors`/`team_logos`/`team_nicks` from `league.team_meta()`, set once after loading.

**Data flow:**
1. `league.load_game()` (`load_pbp`, `load_live_game`, `load_cfb_game` or `load_cfb_live_game`) → `pbp_game` for the selected game
2. `play_timeline(pbp_game)` — elapsed game seconds per play (`3600 - game_seconds_remaining`, `ffill/bfill` for null-clock rows, `cummax` to keep it monotonic)
3. The sidebar's quarter + MM:SS inputs produce `baseline_elapsed`; `cursor_from_elapsed()` turns that into a **play cursor**
4. `revealed = pbp_game.iloc[: cursor_idx + 1]` drives all displayed sections

**Key invariant — spoiler safety:** Every data-driven section must only use `revealed` (the sliced DataFrame), never `pbp_game` directly. The `safety_margin` slider subtracts seconds from `baseline_elapsed` before it is converted to a cursor. The win probability and momentum charts plot each play at its `play_timeline()` position (`_el_min`), so OT lands after regulation, and the x-axis is capped at `elapsed_s / 60` to prevent the chart shape itself from being a spoiler.

**Position model — the play cursor:** viewing position is a *play index*, not a game-seconds scalar. Three session-state keys:

- `_cursor_idx` — the play currently being viewed
- `_cursor_max` — furthest play unlocked; the time bar's slider `max_value`, and therefore the spoiler gate
- `_cursor_key` — `(game_id, qtr_pick, clock_str, safety_margin)`; a change re-seeds both from the clock baseline
- `_cursor_frame` / `_cursor_anchor` — `(game_id, source, len(pbp_game))` and a `(elapsed, rank-within-that-second)` anchor for both cursors. When the frame changes (live plays arrive, or live → official, whose indices differ), `cursor_from_anchor()` carries the position over by game time. It never lands on a later second, or on a later rank within the same second, so a switch can't unlock anything.

An index is used because a clock value cannot address plays individually: plays sharing one `game_seconds_remaining` (penalties, timeouts, two snaps in a second) collapse together. Overtime is worse: its clock restarts *inside* the regulation range (a 2024 OT game carries values like 600, 563, 523), so taken as game seconds every OT play looks earlier than the end of Q4. `play_timeline()` therefore gives OT period n its own slot after regulation, `3600 + 900·(n-1)` to `3600 + 900·n` (the same slot the sidebar's OT + clock input maps to), strictly after the period start. Untimed college OT plays have no clock and are spaced `UNTIMED_OT_PLAY_SECS` apart within their slot. `cummax` keeps the series sorted where a feed's clock steps backwards; it only ever moves a play later. `elapsed_s` is still derived from the cursor, but only for display and the WP x-axis cap — nothing filters on it.

**Viewing position (sidebar):** a single control — quarter + MM:SS game clock, converted to elapsed game seconds (`completed_qtrs * 900 + (900 - remaining)`; OT is `qtr=5`, treated as starting after Q4). Its only job is to seed the cursor; the time bar does the exact syncing from there. `safety_margin` is subtracted from the baseline before the cursor lookup, clamped at 0. For college OT (`league.untimed_ot`) the clock input is hidden: OT seeds at the end of regulation and you step through it. **Full game** (`FULL_GAME`) seeds the cursor at the last play, unlocking everything, for games already watched; its seed key includes `last_idx`, so on a live game it re-seeds as plays arrive.

**Time bar (main panel, between the header metrics and the reveal gate):** an HTML track (`_time_bar_html()`, quarter ticks, fill coloured by possession team, OT segment only once the cursor is in OT) plus an `st.slider` capped at `_cursor_max` and a row of step buttons. The slider deliberately has **no `key=`** so it re-seeds from `_cursor_idx` each rerun and buttons/auto-advance can drive it without a widget-state exception. Only `▶ Next play` and `▶▶ Next drive` raise `_cursor_max`; scrubbing and `⏭ Catch up` never do. Nothing past the cursor is drawn — no scoring marks, no play density, and the caption says "of N *unlocked*", never a total, since the play count alone leaks whether the game ran long.

**Live refresh:** a live game that isn't over reruns every 30s (`live_refresh`) to pull in new plays. That only makes them reachable by the ▶ buttons; it never moves the cursor.

**Follow live:** for a live game that isn't over, a sidebar `🔴 Follow live` toggle replaces the quarter/clock inputs and auto-advance. `_live_seen[game_id]` records the wall time each play position first appeared in the feed; a play is unlocked only once it has been there `live_delay` seconds (slider, default 45s), so a stream running behind ESPN can't be spoiled. Plays already in the feed when you switch it on count as aired, except the newest. It reruns every 10s and, like auto-advance, carries `_cursor_idx` along only when it was at the edge.

**Auto-advance:** When `auto=True`, `st_autorefresh` fires and the block at the bottom of the file bumps `_cursor_max` by one, carrying `_cursor_idx` with it only if it was already at the edge — so an auto-advancing replay doesn't yank you forward while you're scrubbing back through a drive.

**Keep screen awake:** a sidebar checkbox (default on) calls `keep_screen_awake()`, which injects a `components.html` script that requests a Screen Wake Lock on `window.parent` (the component iframe itself lacks the permissions-policy grant) and re-acquires it on `visibilitychange`. Requires HTTPS or localhost.

**AI summary league:** `GameContext.league` (`"NFL"`/`"CFB"`) picks the analyst, model name, percentile baseline and OT note in `system_prompt()` (`_LEAGUE`); the NFL prompt text is unchanged. `replay_core.period_label()` gives Q1–Q4, OT, 2OT, … for the page and the summary.

**AI summary and reruns:** a full rerun stops the running script and starts a new one at once (Streamlit's default `runner.fastReruns`), and a live game reruns every 10–30s while a summary takes a minute or more. So nothing in the summary flow waits on one run:
- the button's `on_click` callback records the request in `_summary_requested` (game id) at the start of the click's run; the first run to reach the section with it set builds the `GameContext` and starts the job, so a click whose run is cut short still counts;
- the job (`game_summary.submit()`) runs on a worker thread; `_summary_job` holds its Future, its `Progress`, the viewer's `cursor_anchor` and when it started. While it runs, the fragment shows an `st.status` box with what the agent is doing ("Reading the dashboard", "Thinking it over", or the lookup in progress), the elapsed time, and a checklist of the lookups done;
- `_summary_result()` is an `st.fragment` that, while a job is out, reruns on its own every `_SUMMARY_POLL_SECS` to collect it into `_summary` (or `_summary_error`), then calls `st.rerun()` once to free the button and stop polling. Fragment reruns queue behind full runs instead of cancelling them, so they can't starve slow pages the way full-rerun polling did, and they work on replays where nothing else reruns. The button stays outside the fragment: a fragment click queued behind a slow full run is dropped when the next full rerun replaces it.
- state is written before what it replaces is cleared, so a run cut off in between only leaves work to redo.

**AI summary spoiler gate:** `st.session_state["_summary"]` holds the latest summary with the `cursor_anchor` it was written at. It is shown only for the same game and while that anchor is ≤ the anchor of `_cursor_max`, so re-seeding the clock earlier hides it. Keys come from `_setting()`, which reads `st.secrets` and then the environment.

**UI sections (top to bottom):** header metrics → boxscore → AI summary (on demand) → recent plays (paginated, 15/page) → current drive → team stats → player leaders → win probability chart.

**Stat tables:**
- `boxscore()` — quarter-by-quarter score via cumulative score diffs; every OT period goes in the one OT column
- `team_stats()` — advanced EPA-based stats, returned as a transposed DataFrame (stat names as index, team abbrs as columns); styled by `style_stat_table()`. Percentiles come from `leagues.stat_baselines(league, season)` / `leagues.situational_baselines(league, season)`: the 3 prior NFL seasons, or last college season's FBS-vs-FBS games (`_cfb_baseline_pbp`); `league.baseline_label` names them in the captions
- `top_players()` / `top_defenders()` — per-team passing/rushing/receiving leaders sorted by yards, and defensive leaders; `drop=missing_stats` (`league.missing_stats(revealed)`) removes columns this game's data doesn't have
- `drive_chart()` — one row per drive; above it, `drive_field_figure()` draws each drive from `drive_field_spots()` as an arrow on a field (home drives left → right from its own end zone on the left, away right → left; end zones carry team colors, logo and nickname). Each arrow is split into one segment per play (line of scrimmage to the next play's spot), coloured by `_play_category()` — pass or run on 1st/2nd down, pass or run on 3rd/4th down, or accepted penalty (`no_play` with `penalty == 1`); scrambles count as runs and sacks as passes, following nflfastR's `play_type`. A tick marks every snap so no-gain plays still show, and hovering a tick shows that play. The start/end markers keep the team color. Hover shows each drive's time of possession, measured to the next drive's start within the same half, as nflfastR does. The field graph is always shown; the table sits in an expander, collapsed by default. The last revealed drive is labelled "In progress" unless the last revealed row is an END marker.
