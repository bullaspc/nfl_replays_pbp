"""
Add today's snapshot of the NFL draft boards to models/draft_boards.csv, the
archive the app's draft badges are read from (leagues.load_cfb_draft_board()),
so the app itself makes no calls to ESPN or Tankathon.

    python tools/build_draft_board.py          # the next draft
    python tools/build_draft_board.py 2027     # a given draft

It reads ESPN's big board (its top cfb_feed.DRAFT_BOARD_MAX, one request per
player) and Tankathon's big board and mock draft (two pages, only while the
site is on that draft) and appends them with draft_year and as_of (today, UTC).
A rerun the same day replaces that day's rows; earlier snapshots are kept, so
the file is also a week-by-week record of the boards. GitHub Actions runs it
weekly (.github/workflows/draft_boards.yml) and commits the file.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cfb_feed  # noqa: E402

OUT = cfb_feed.DRAFT_ARCHIVE


def _rows(*frames: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """The frames' rows in one frame (concat warns over all-empty columns)."""
    return pd.DataFrame([r for f in frames for r in f.to_dict("records")], columns=columns)


def next_draft_year(today: datetime) -> int:
    """The draft Tankathon is on, else the next April's."""
    try:
        page = cfb_feed._get(cfb_feed.TANKATHON_BOARD_URL, timeout=20).decode("utf-8", "replace")
        year = cfb_feed.tankathon_year(page)
        if year:
            return year
    except Exception:
        pass
    return today.year + 1 if today.month >= 5 else today.year


def main() -> int:
    today = datetime.now(timezone.utc)
    draft_year = int(sys.argv[1]) if len(sys.argv) > 1 else next_draft_year(today)
    as_of = today.strftime("%Y-%m-%d")
    espn = cfb_feed.read_espn_draft_board(draft_year)
    tank = cfb_feed.read_tankathon(draft_year)
    print(f"{draft_year} draft, {as_of}: ESPN {len(espn)} players, Tankathon {len(tank)} "
          f"({tank['mock_pick'].notna().sum()} in its mock)")
    snap = _rows(espn, tank, columns=cfb_feed.DRAFT_COLS)
    if snap.empty:
        print("Nothing read: the archive is left as it was.")
        return 1
    snap.insert(0, "as_of", as_of)
    snap.insert(0, "draft_year", draft_year)
    if OUT.exists():
        old = pd.read_csv(OUT, dtype={"school": str, "mock_team": str})
        old = old[~((old["draft_year"] == draft_year) & (old["as_of"] == as_of))]
        snap = _rows(old, snap, columns=cfb_feed.ARCHIVE_COLS)
    snap = snap[cfb_feed.ARCHIVE_COLS].sort_values(
        ["draft_year", "as_of", "source", "rank"], kind="stable")
    for col in ("team_id", "rank", "mock_pick"):
        snap[col] = pd.to_numeric(snap[col], errors="coerce").astype("Int64")
    snap.to_csv(OUT, index=False)
    n = snap.groupby("draft_year")["as_of"].nunique()
    print(f"Wrote {OUT.relative_to(OUT.parent.parent)}: "
          + ", ".join(f"{y} draft {k} snapshot{'s' if k > 1 else ''}" for y, k in n.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
