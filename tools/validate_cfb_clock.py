"""
Check cfb_feed's stale-clock repair (`stale_clocks`, `repair_raw_clock`) on
the college release's published seasons.

ESPN's college feed sometimes carries one clock over a whole drive. For each
season, and each STALE_CLOCK_SNAPS threshold given:

1. As published: how many plays and games the repair changes, and how many of
   those plays have a leading stat-crew text clock ("(08:47) ...") that agrees
   with the feed. Those are suspect changes (often the text's own clock is
   stuck too: six snaps all at "(09:32)").
2. Drive simulation: games the repair leaves alone, with every play of a drive
   but its last given the drive's first clock; the repaired clocks against the
   real ones.
3. Text simulation: the same games with stat-crew text, with every leading
   text clock stripped, so each play carries the last clock its text gives
   (", clock 11:32" on scores and timeouts): how ESPN's feed goes stale.

    python tools/validate_cfb_clock.py [first=2025] [last=2026] [thresholds=1,2,3]
"""
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cfb_feed  # noqa: E402

LEAD = re.compile(r"^\((\d{1,2}):(\d{2})\)")
IN_TEXT = re.compile(r"\bclock (\d{1,2}):(\d{2})")


def _secs(m) -> float:
    return float(int(m.group(1)) * 60 + int(m.group(2))) if m else np.nan


def _clock(df: pd.DataFrame) -> pd.Series:
    return df["clock.displayValue"].map(cfb_feed._clock_secs)


def read_season(season: int) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"cfb_{season}.parquet"
        cfb_feed.download_season(season, path)
        raw = pd.read_parquet(path, columns=cfb_feed.RAW_COLS + ["drive.id"])
    return raw.sort_values(["game_id", "game_play_number"], kind="stable").reset_index(drop=True)


def repaired(df: pd.DataFrame, threshold: int) -> pd.Series:
    cfb_feed.STALE_CLOCK_SNAPS = threshold
    return _clock(cfb_feed.repair_raw_clock(df))


def summary(truth: pd.Series, before: pd.Series, after: pd.Series) -> str:
    m = before.ne(truth) & truth.notna() & before.notna()
    e1, e2 = (before - truth).abs()[m], (after - truth).abs()[m]
    return (f"{m.sum()} plays off by {e1.mean():.0f} s (median {e1.median():.0f}) -> repaired {e2.mean():.0f} s "
            f"(median {e2.median():.0f}); within a minute {(e1 <= 60).mean():.0%} -> {(e2 <= 60).mean():.0%}")


def drive_stale(g: pd.DataFrame) -> pd.DataFrame:
    key = [g["game_id"], g["drive.id"]]
    first = g.groupby(key)["clock.displayValue"].transform("first")
    last = g.groupby(key).cumcount(ascending=False) == 0
    same_period = g["period"].eq(g.groupby(key)["period"].transform("first"))
    out = g.copy()
    out.loc[~last & same_period & (g["period"] <= 4), "clock.displayValue"] = first
    return out


def text_stale(g: pd.DataFrame) -> pd.DataFrame:
    text = g["text"].fillna("")
    crew = text.str.match(LEAD).groupby(g["game_id"]).transform("mean") > 0.5
    g, text = g[crew], text[crew]
    known = text.map(lambda t: _secs(IN_TEXT.search(t)))
    known = known.where(~g["type.text"].fillna("").str.contains("End"), 0.0)
    period_start = g["period"].ne(g["period"].shift()) | g["game_id"].ne(g["game_id"].shift())
    known = known.where(~(period_start & known.isna()), 900.0)
    out = g.copy()
    sim = known.groupby([g["game_id"], g["period"]]).ffill()
    out["clock.displayValue"] = sim.map(lambda v: cfb_feed._clock_text(v) if pd.notna(v) else None)
    return out


def main(first: int = 2025, last: int = 2026, thresholds=(1, 2, 3)) -> None:
    for season in range(first, last + 1):
        raw = read_season(season)
        clock = _clock(raw)
        lead = raw["text"].fillna("").map(lambda t: _secs(LEAD.match(t)))
        print(f"\n{season}: {raw['game_id'].nunique()} games, {len(raw)} plays")
        changed_any = None
        for k in thresholds:
            rc = repaired(raw, k)
            ch = rc.ne(clock) & clock.notna()
            games = ch.groupby(raw["game_id"]).sum()
            changed_any = games if changed_any is None else changed_any.add(games)
            agreed = ch & lead.eq(clock)
            print(f"  as published, {k} snaps: {ch.sum()} plays ({ch.mean():.1%}) in {(games > 0).sum()} games "
                  f"changed, {agreed.sum()} of them where the text's own clock agreed with the feed "
                  f"(by {(rc - clock)[agreed].abs().median():.0f} s, median)")
        clean = raw[raw["game_id"].isin(changed_any[changed_any == 0].index)]
        for name, make in (("drive simulation", drive_stale), ("text simulation", text_stale)):
            stale = make(clean)
            before = _clock(stale)
            stale["start.adj_TimeSecsRem"] = before + (4 - stale["period"]) * 900
            truth = _clock(clean).loc[stale.index]
            for k in thresholds:
                print(f"  {name} ({stale['game_id'].nunique()} games), {k} snaps: "
                      f"{summary(truth, before, repaired(stale, k))}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2025, int(sys.argv[2]) if len(sys.argv) > 2 else 2026,
         [int(t) for t in sys.argv[3].split(",")] if len(sys.argv) > 3 else (1, 2, 3))
