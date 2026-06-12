# Percentile-Based Team Stats Table Coloring

**Date:** 2026-06-12
**Status:** Approved

## Goal

Replace the fixed-threshold color logic in the team stats table with a 3-color diverging palette driven by where each value sits in the distribution of full-game team stats across the last 3 NFL seasons.

## Baseline Data

### Loading

New function `load_stat_baselines(season: int) -> dict[str, np.ndarray]`, cached with `ttl=86400` (24 h — historical seasons never change).

- Loads PBP for `[season-3, season-2, season-1]` using `nfl.import_pbp_data` with only the columns required for stat computation (subset of the existing `load_pbp` column list).
- Groups by `(game_id, posteam)` to produce one row per team-game.
- Computes the same stats as `team_stats()` for each group.
- Returns `{stat_name: np.ndarray}` where each array contains all per-team-game values across the 3 seasons, sorted ascending.

### Columns needed for baseline

```
game_id, posteam, pass_attempt, rush_attempt, epa,
complete_pass, air_yards, interception, fumble_lost,
first_down, third_down_converted, third_down_failed,
yardline_100, touchdown
```

### Stats computed per team-game

Same 16 stats produced by `team_stats()`:
Plays, Pass Plays, Rush Plays, Pass Yds, Rush Yds, Total Yds,
CMP%, aDoT, Pass EPA/play, Rush EPA/play, EPA/play,
Pass SR, Rush SR, 1st Down %, 3rd Down %, RZ TD%, Turnovers.

## Percentile Lookup

```python
def _percentile_of(value, sorted_arr) -> float:
    """Returns 0–100 percentile rank of value in a pre-sorted array."""
    if np.isnan(value) or len(sorted_arr) == 0:
        return float("nan")
    return np.searchsorted(sorted_arr, value, side="right") / len(sorted_arr) * 100
```

**Lower-is-better stats:** Only `"Turnovers"`. Its percentile is flipped: `pct = 100 - pct`.

EPA rows (`Pass EPA/play`, `Rush EPA/play`, `EPA/play`) keep their existing positive/negative coloring — percentile coloring is not applied to them since the zero crossing is already the meaningful threshold.

## Color Palette

3-color diverging, linearly interpolated in RGB space:

| Percentile | Background | Meaning |
|---|---|---|
| 0th | `#dc3545` (red) | Bottom of league |
| 50th | `#ffffff` (white) | League average |
| 100th | `#28a745` (green) | Top of league |

**Interpolation:**
- pct 0–50: interpolate between red and white (`t = pct / 50`)
- pct 50–100: interpolate between white and green (`t = (pct - 50) / 50`)

**Text color:** Computed from background luminance. If luminance < 0.45, use white `#ffffff`; otherwise dark `#212529`. This ensures readable text across the full gradient.

## Implementation Changes

### `style_stat_table(df, away, home, baselines)`

- Gains a `baselines: dict[str, np.ndarray]` parameter.
- For each row in the stats index (excluding EPA rows), applies the percentile color per cell.
- Existing format logic (%, EPA, integer) is unchanged.
- Existing EPA coloring (`_color_epa`) is unchanged.
- `_color_sr` fixed-threshold function is removed (superseded by percentile coloring).

### Call site (`nfl_replay_app.py` main body)

```python
baselines = load_stat_baselines(season)
st.dataframe(style_stat_table(stat_df, away, home, baselines), use_container_width=True)
```

### Helper `_percentile_color(pct, lower_is_better=False) -> str`

Returns a CSS string `"background-color: #rrggbb; color: #rrggbb"`.

## Error Handling

- If baseline data fails to load (network error, season out of range), `load_stat_baselines` returns an empty dict `{}`. `style_stat_table` falls back to no coloring for affected rows (returns `""`).
- NaN stat values are formatted as `"—"` (existing behavior) with no background color.

## Performance

- Baseline loading hits nfl_data_py once per (season, app restart) cycle, then is served from Streamlit's cache.
- Per-render cost is O(rows × 2 teams × log N) for binary search — negligible.
