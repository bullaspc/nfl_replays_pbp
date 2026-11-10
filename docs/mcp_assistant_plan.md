# Plan: a data assistant over MCP, with Kimi as the LLM

Goal: ask questions in plain language about the project's data ("Which defenses
blitz most on 3rd and long in 2025?", "How did DET's EPA/play change vs 2024?",
"Who's Out for KC this week?") and get answers computed with the project's own
stat code, not the LLM's memory.

Two pieces:

1. **`mcp_server.py`**: an MCP server (Python `mcp` SDK, `FastMCP`) whose tools
   wrap the functions in `leagues.py` / `replay_core.py`. Any MCP client can use it
   (Claude Desktop, Claude Code, Cursor...), but it holds no LLM of its own.
2. **`assistant.py`**: the assistant itself, an MCP *client* whose LLM is always
   Kimi (Moonshot's Anthropic-compatible endpoint, the one `game_summary` already
   uses). It lists the server's tools, hands them to Kimi as tool definitions and
   runs the tool loop. A CLI first, then an **Ask the data** Streamlit page.

```
 user ──▶ assistant.py (Kimi tool loop) ──MCP stdio──▶ mcp_server.py ──▶ leagues.py / replay_core.py
                     ▲                                                     │  (nflverse, ESPN, sportsdataverse,
                     └────────────── tool results (small tables as text) ◀─┘   season store, disk cache)
```

## 1. The MCP server (`mcp_server.py`)

No Streamlit at the top level, like `replay_core.py`. `leagues.py`'s
`st.cache_data` / `cache_resource` work outside `streamlit run` (in-memory, with
a "no runtime" warning we silence with `logging`), and the finished-season disk
cache and season store keep it fast. If that proves noisy, step 0 is to move the
loaders' bodies into undecorated functions that `leagues.py` caches and the
server calls directly.

Every tool returns a **small** table as text (`to_markdown()` / CSV, capped at
`MAX_ROWS` = 50 rows, with a "truncated" note) plus the selection it used, so
Kimi can quote it. Arguments are validated (league in `LEAGUES`, season ≥
`league.first_season`, team in `season_teams()`), and errors (`GameLoadError`,
`SeasonNotPublished`, `DownloadFailed`) come back as a tool error message, not a
crash.

### Tools (first cut)

| Tool | Wraps | Arguments |
|---|---|---|
| `list_seasons` | `LEAGUES`, `first_season`, `season_final()` | league |
| `list_teams` | `season_teams()`, `team_names()` | league, season |
| `list_games` | `league.games()` | league, season, week?, team? |
| `upcoming_games` | `league.upcoming()` | league, season, week? |
| `team_season` | `season_tables()` → `season_team_table()`, `season_ranks()`, `season_places()` | league, season, team?, side (offense/defense), view (`TEAM_VIEWS`), stats (Raw/adjusted), weeks, part (`SEASON_PARTS`), venue, filters |
| `team_rankings` | same, sorted by one column | league, season, stat, side, top_n, ... |
| `situational` | `season_situational()` / `situational_matchup()` | league, season, team, side, ... |
| `player_leaders` | `season_players()` / `season_defenders()` / `season_kickers()` / `season_punters()` (+ `with_nextgen()` for NFL) | league, season, kind, team?, min_plays?, sort_by, top_n |
| `team_schedule` | `team_schedule()` + `game_epa()` | league, season, team |
| `matchup` | `matchup_table()` both ways, `head_to_head()` | league, season, away, home, weeks? |
| `availability` | `injury_report()`, `availability_summary()`, `team_snaps()`, `lineup_change()` | season, team, week? (NFL only) |
| `charting` | `team_chart_summary()`, `team_chart_games()` | season, team (NFL 2022+) |
| `compare_seasons` | `team_season_compare()` / `player_season_compare()` | league, team or players, seasons (≤ `COMPARE_MAX`), mode (`COMPARE_MODES`) |
| `game_stats` | `league.load_game()` + `team_stats()`, `boxscore()`, `drive_chart()`, `top_plays_wpa()`, `top_players()` | league, season, game_id, section |
| `glossary` | `PLAYER_GLOSSARY`, `NGS_GLOSSARY`, the prompt's "Terms" text | term? |
| `query_sql` (step 3) | DuckDB over a season's pbp | league, season, sql |

`query_sql` covers the questions no fixed tool does: the season's pbp (the
`PBP_COLS` layout) registered as a DuckDB view `pbp`, read-only, `SELECT` only
(parsed with `sqlglot` or DuckDB's `json_serialize_sql`, anything else
refused), a row cap and a timeout. That also gives you a SQL way in, BigQuery-
style, without the LLM writing pandas.

Resources: `nfl://columns` (the `PBP_COLS` list with one-line meanings) and
`nfl://methodology` (the Season stats page's Methodology text), so the model
can read definitions instead of guessing.

### Spoilers

The replay page's spoiler rule (only `revealed`) is about the cursor; the
assistant has no cursor, so it answers about finished data only. `game_stats`
on a live game (`source == "live"`) refuses by default, with an
`include_live=True` argument the user must ask for explicitly. The system
prompt says so.

## 2. The assistant (`assistant.py`), always Kimi

- **Provider:** reuse `game_summary.configured_providers()` but take
  `providers["kimi"]` only; with no `MOONSHOT_API_KEY` / `KIMI_API_KEY` it stops
  with a clear message, never falls back to Claude. Model `KIMI_MODEL` (default
  `kimi-k3`), base URL `KIMI_BASE_URL` (default
  `https://api.moonshot.ai/anthropic`), Bearer auth, through
  `game_summary._client()` / `_call()` so errors are the same `SummaryError`s.
  No Claude-only options (`effort`, `strict`, caching, fallbacks), which
  `Provider.is_claude` already gates.
- **MCP client:** `mcp.ClientSession` over `stdio_client` launching
  `python mcp_server.py`. `list_tools()` → Anthropic-format tools
  (`name`, `description`, `input_schema` straight from the MCP schema).
- **Loop:** like `game_summary._agent()`: send messages + tools, run every
  `tool_use` block through `session.call_tool()`, return `tool_result`s, up to
  `ASSIST_MAX_TURNS` (8) calls; a bad-argument error goes back to the model to
  retry. Keeps the conversation history for follow-ups.
- **System prompt** (`ASSIST_PROMPT`): answer only from tool results, name the
  season/selection used, say "opponent-adjusted" or "raw", give numbers with
  their place ("3rd of 32"), say when data is missing (college has no QB hits;
  FTN charting NFL 2022+; NGS 2016+), the same "Terms" definitions as
  `SYSTEM_PROMPT`, and Italian or English to match the question.
- **Front ends:**
  - CLI: `python assistant.py "domanda..."` or an interactive prompt.
  - `pages/4_Ask_the_data.py`: `st.chat_input` + `st.chat_message`, the job on
    `game_summary`'s worker pool with a `Progress` and a polling fragment, as the
    summary does (a reply takes 10–60 s and the page can rerun meanwhile);
    `describe_lookup()`-style lines ("Looking at 2025 defenses, blitz %").
    Session keys `_ask_`. The MCP server runs as a subprocess per job (or one
    kept in `st.cache_resource` if start-up is slow).

## 3. Steps

1. **Server skeleton + 4 tools** (`list_teams`, `team_season`, `player_leaders`,
   `matchup`), tested with `mcp dev mcp_server.py` (the MCP inspector) and from
   Claude Code (`claude mcp add nfl -- python mcp_server.py`) to check the tool
   descriptions.
2. **`assistant.py` CLI with Kimi**, end to end on ~20 test questions written in
   `tools/assistant_questions.txt` (answers checked by hand against the Season
   stats page).
3. Remaining tools, then `query_sql` with DuckDB.
4. **Ask the data** page.
5. `tools/validate_assistant.py`: runs the question list, checks every number in
   each answer appears in a tool result (no invented stats) and logs tool calls
   and latency.
6. CLAUDE.md: the server, the assistant and the new page in Architecture.

New dependencies: `mcp>=1.2`, `duckdb`, `tabulate` (for `to_markdown`),
optionally `sqlglot`. Settings: `MOONSHOT_API_KEY` (already used),
`KIMI_MODEL`, `KIMI_BASE_URL`, `ASSIST_MAX_TURNS`.

## Open questions

- Should the MCP server also be deployed (Streamlit Cloud can't host it; a small
  container with SSE/streamable HTTP transport could), or is local stdio enough?
- College too from day one, or NFL first? College seasons take about a minute
  the first time, so NFL first is quicker to iterate on.
