# NFL Tape-Delay Replay Boxscore

A spoiler-free Streamlit app for following NFL games on tape delay. You tell the app when you started watching; it reveals only the plays, score, and stats up to your current viewing position — no accidental spoilers.

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

Play-by-play data comes from [nfl_data_py](https://github.com/nflverse/nfl_data_py), which pulls from the [nflverse](https://nflverse.com/) data infrastructure. Data refreshes every 2 minutes during live games.
