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
   - *Jump to game clock* — pick a quarter and MM:SS timestamp directly, or **Full game** to unlock every play of a game you've already watched
3. Use the **safety margin** slider to subtract extra seconds if you're worried about accidental spoilers
4. Enable **Auto-advance** to let the app tick forward in real time

## Requirements

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `nflreadpy` | NFL play-by-play, schedules and team colors from nflverse (the successor to `nfl_data_py`) |
| `pandas`, `pyarrow` | Data manipulation, reading sportsdataverse parquet files |
| `numpy` | Numeric helpers |
| `plotly` | Win probability chart |
| `streamlit-autorefresh` | Auto-advance timer |
| `requests` | nflverse and sportsdataverse downloads, ESPN live feeds |
| `sportsdataverse` | Processes live college games (ESPN's feed with sportsdataverse's college EPA and win probability models) |
| `xgboost` | nflfastR's EP/WP models for live games |
| `anthropic` | AI game summary (Kimi or Claude through the Messages API) |

## Architecture

| File | What it does |
|---|---|
| [nfl_replay_app.py](nfl_replay_app.py) | The Streamlit page: sidebar, play cursor, sections |
| [leagues.py](leagues.py) | Where each league's games come from, behind one `League` interface |
| [replay_core.py](replay_core.py) | League-neutral logic: play timeline, cursor, every stat (no Streamlit) |
| [views.py](views.py) | Table styling, the drive field figure, the time bar |
| [live_feed.py](live_feed.py), [nflfastr_models.py](nflfastr_models.py) | Live NFL games from ESPN with nflfastR's models |
| [cfb_feed.py](cfb_feed.py) | College play-by-play in the same layout as the NFL's |
| [game_summary.py](game_summary.py) | The AI game summary agent |

**Data flow:**
1. The league picked in the sidebar loads the selected game in nflfastR's column layout
2. `play_timeline()` gives each play its elapsed game seconds
3. Your quarter and clock (or **Full game**) set a play cursor; the time bar and ▶ buttons move it
4. Every displayed section reads from `revealed`, the plays up to the cursor, never from the full game data

**Broadcast-to-game-seconds mapping:** 190 broadcast minutes maps linearly to 3600 game seconds.

## Data Source

Play-by-play comes from [nflverse](https://nflverse.com/), meaning nflfastR's play-by-play, loaded through [nflreadpy](https://github.com/nflverse/nflreadpy). nflverse rebuilds it about once a day after games finish. The app checks nflverse's `timestamp.json` every minute and reloads as soon as a new build is out.

## College football

Pick **College football** under **League** in the sidebar. Everything works the same way as for the NFL, with a few differences:

- **Data:** [sportsdataverse](https://github.com/sportsdataverse/sportsdataverse-data)'s published college play-by-play (2004 onward). It is ESPN's play feed with sportsdataverse's own college EPA and win probability models, rebuilt about once a day.
- **Live games:** games with an FBS team that have kicked off today but aren't published yet show up with 🔴 live. So do games sportsdataverse published while they were still being played (its data is rebuilt during game days too, so it can hold the first quarter of a game in progress); they switch to the published data once a rebuild has them finished. They're read from ESPN's play feed and run through [sportsdataverse-py](https://github.com/sportsdataverse/sportsdataverse-py)'s own processing, the same that builds the published data, so EPA and win probability come from the same college models. Follow live and the play-by-play buttons work as for the NFL, and the game switches to the published data by itself once it's out. Player names come from the play text (e.g. "J.Smith") until then: the passer, rusher and receiver are read from it directly, which on counted plays matches the published names for 99% of passes and 96–97% of runs and catches (most of the rest differ only by a "Jr" or "III"). `tools/validate_cfb_live.py` runs the live path on published games rebuilt as ESPN feeds. On 2026 games it gets down, distance and possession right on every matched play, play types on 99.5%, and every final score. Its win probability is within 0.005 of the published value on a typical play (EPA within 0.02). A game cut off mid-way never shows a later play or score.
- **Game picker:** week, then an optional conference filter (there are 50+ games on a Saturday). AP ranks going into the game are shown.
- **Overtime** is untimed in college, so picking OT starts you at the end of regulation and you step through it with ▶ Next play. Every overtime period adds into one OT column in the boxscore.
- **Percentile colors** compare against last season's FBS-vs-FBS games.
- **Defensive leaders:** since late 2025, ESPN's college feed mostly uses the NCAA stat-crew play text ("(08:47) Shotgun #7 A.Powell rush middle for 2 yards gain to the NDSU09 (#51 G.Sell)"), which names the tacklers, QB hurries, pass breakups and fumble forcers. For those games (almost all 2026 games) the defense table has tackles, sacks, QB hurries, tackles for loss, INTs, pass breakups and forced fumbles, credited the college way: a shared tackle is an assist for each player, and a shared sack or TFL is a half for each. In both leagues Tackles is the official total, solo tackles plus assists, each assist counting as a full tackle. Older games' text names no tacklers, so their table shows only sacks, INTs, pass breakups and forced fumbles. College stats have no QB hits. Defenders are named as the play text writes them ("G.Sell"). Checked against ESPN's player box for 2026 games: sacks, pass breakups, QB hurries and INTs match exactly for 99% of players, TFLs 97%. Tackles are exact for 79% and within 1 for 95%, because ESPN's box also counts special teams tackles, which the table leaves out (as it does for the NFL).
- ESPN's college feed has glitches: plays listed after a later quarter, stale scores, and occasionally a mid-game row carrying the final score. The app repairs these (see `cfb_feed.py`). On 2026 data, 99.75% of games end on the official final score (2025: 99.2%), and team totals match ESPN's box score with a median difference of 0. `tools/validate_cfb_adapter.py` reports this for any season, plus the defensive stats against ESPN's player box.
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

## Live NFL games

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
