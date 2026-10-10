# NFL Tape-Delay Replay Boxscore

A spoiler-free Streamlit app for following NFL and college football games on tape delay. You tell the app when you started watching; it reveals only the plays, score, and stats up to your current viewing position — no accidental spoilers.

## Features

- **Spoiler-safe live stats** — score, boxscore, team stats, and player leaders are filtered to your exact viewing position
- **Three viewing modes** — wall-clock start time, broadcast minutes elapsed, or jump to a specific game clock
- **Auto-advance** — automatically progresses to the next play every 30 seconds so you can set it and forget it
- **Win probability chart** — x-axis capped at your current position so the chart shape itself can't spoil the ending
- **Per-game play-by-play** — paginated recent plays list with current drive summary
- **Player leaders** — passing, rushing, receiving, and defensive stats per team

## Quick Start

```bash
pip install -r requirements.txt
streamlit run nfl_replay_app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

## Usage

1. **Select a season and game** from the sidebar
2. **Choose your viewing mode:**
   - *Started at* — pick the real-world date/time you pressed play; the app computes elapsed game time automatically
   - *X minutes in* — enter how many broadcast minutes you've watched
   - *Jump to game clock* — pick a quarter and MM:SS timestamp directly
3. Use the **safety margin** slider to subtract extra seconds if you're worried about accidental spoilers
4. Enable **Auto-advance** to let the app tick forward in real time

## Requirements

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `nfl_data_py` | NFL play-by-play data |
| `pandas` | Data manipulation |
| `numpy` | Numeric helpers |
| `plotly` | Win probability chart |
| `streamlit-autorefresh` | Auto-advance timer |
| `requests` | nflverse timestamp + ESPN live feed |
| `xgboost` | nflfastR's EP/WP models for live games |
| `anthropic` | AI game summary (Kimi or Claude through the Messages API) |

## Architecture

Single-file app: [nfl_replay_app.py](nfl_replay_app.py)

**Data flow:**
1. `load_pbp(season)` — fetches play-by-play via `nfl_data_py`, cached for 2 minutes
2. `list_games(pbp)` — builds the game selector from the season data
3. Sidebar inputs compute `elapsed_s` (game-seconds watched so far)
4. `filter_revealed(pbp_game, elapsed_s)` — keeps only plays up to that point; `ffill/bfill` handles null-clock rows (timeouts, admin plays)
5. Every displayed section reads from `revealed` only — never from the full game data

**Broadcast-to-game-seconds mapping:** 190 broadcast minutes maps linearly to 3600 game seconds.

## Data Source

Play-by-play comes from [nflverse](https://nflverse.com/), meaning nflfastR's play-by-play, loaded through [nfl_data_py](https://github.com/nflverse/nfl_data_py). nflverse rebuilds it about once a day after games finish. The app checks nflverse's `timestamp.json` every minute and reloads as soon as a new build is out.

## College football

Pick **College football** under **League** in the sidebar. Everything works the same way as for the NFL, with a few differences:

- **Data:** [sportsdataverse](https://github.com/sportsdataverse/sportsdataverse-data)'s published college play-by-play (2004 onward). It is ESPN's play feed with sportsdataverse's own college EPA and win probability models. It's rebuilt about once a day, so a game shows up the morning after it's played. **Live college games aren't supported yet.**
- **Game picker:** week, then an optional conference filter (there are 50+ games on a Saturday). AP ranks going into the game are shown.
- **Overtime** is untimed in college, so picking OT starts you at the end of regulation and you step through it with ▶ Next play. Every overtime period adds into one OT column in the boxscore.
- **Percentile colors** compare against last season's FBS-vs-FBS games.
- **No tackles, QB hits or tackles for loss:** the college feed doesn't have them, so those columns are left out of the player leaders.
- ESPN's college feed has glitches: plays listed after a later quarter, stale scores, and occasionally a mid-game row carrying the final score. The app repairs these (see `cfb_feed.py`). On 2026 data, 99.75% of games end on the official final score (2025: 99.2%), and team totals match ESPN's box score with a median difference of 0. `tools/validate_cfb_adapter.py` reports this for any season.
- In under 1% of games the feed adds points that never happened, such as a phantom field goal late in the 4th quarter. The app can't catch those without looking at the final score, and that would be a spoiler.

## AI game summary

**🧠 Why the score is what it is** explains the score at your viewing position. It covers the factors and key plays behind it, with offense and defense for both teams. An agent reads the dashboard's own stats: team stats with percentiles, what each defense allowed, drives, situational success rates, player leaders and the plays with the biggest win-probability swings. It looks up what it needs with tools, then writes.

It only sees the plays you've unlocked. It runs only when you press the button. A summary stays hidden if you move back before the point it was written at.

It calls the model through the Anthropic Messages API, so any compatible endpoint works. Add a key to `.streamlit/secrets.toml` or the environment:

```toml
MOONSHOT_API_KEY = "..."      # Kimi, through Moonshot's Anthropic-compatible endpoint (default)
ANTHROPIC_API_KEY = "..."     # Claude
# Optional
SUMMARY_PROVIDER = "claude"   # which one comes first when both keys are set (default: kimi)
KIMI_MODEL = "kimi-k3"        # default; KIMI_BASE_URL defaults to https://api.moonshot.ai/anthropic
CLAUDE_MODEL = "claude-sonnet-5-5"
CLAUDE_EFFORT = "medium"      # Claude only
```

With both keys set, a selector picks the model per summary. Claude requests also get prompt caching and, on Claude Sonnet 5.5 and newer, the server-side refusal fallback (`fallbacks: "default"`).

## Live games

Games that have kicked off but aren't in nflverse yet show up in the game list with 🔴 live:

- **Plays** come from ESPN's public play-by-play feed, refreshed every 30 s. The NFL gamebook text is parsed the way nflfastR parses it, into play type, players, yards, results and defensive credits (tackles, sacks, INTs, pass breakups, QB hits, forced fumbles).
- **EP, EPA and win probability** come from nflfastR's own trained models, taken from [`nflverse/fastrmodels`](https://github.com/nflverse/fastrmodels) and run in Python with xgboost. No R is needed.
- **Switch to official data:** once nflverse publishes the game, the app moves to the official nflfastR pbp and keeps your viewing position.
- New plays arriving never reveal anything. They only become reachable with **▶ Next play** or auto-advance.

**Accuracy against published 2025 nflverse data:**

- **Model port:** reproduces nflfastR's `ep`/`epa` on 99.9% of plays and `wp` on 99.99%.
- **Live pipeline:** starting only from what the live feed provides (clock, down and distance, field position, score, play text), it matches official EPA on 99.8% of plays and `home_wp` on 99.6%.
- **Player credits:** 97–100% per column.

See `tools/validate_*.py`.
