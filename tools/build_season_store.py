"""
Build the season store's files for finished seasons (season_store.py): each
stored selection's Season stats tables, the player ids and a manifest, into
a folder that .github/workflows/season_store.yml uploads to the
`season-store` release.

    python tools/build_season_store.py 2016 2025 [out_dir] [league]
    python tools/build_season_store.py baselines 2026 [out_dir] [league]

League defaults to NFL, out_dir to ./season_store_out. A season that isn't over
yet (`League.season_final`) is skipped. About 40 selections a season, each
built from scratch like the app builds it (~5–10 s each).

`baselines` builds the replay's percentile baselines for one season instead
(`leagues.compute_baselines`, from the seasons before it; ~40 s, ~0.5 MB):
the replay page then never loads those seasons' play-by-play.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import leagues  # noqa: E402
import replay_core as core  # noqa: E402
import season_store  # noqa: E402


def build(league: leagues.League, season: int, out: Path) -> None:
    stamp = league.stamp()
    code = leagues._CODE_VERSION
    pbp = league.season_pbp(season, stamp)
    reg_weeks = sorted(int(w) for w in pbp.loc[~pbp["postseason"], "week"].dropna().unique())
    has_post = bool(pbp["postseason"].any())
    built = []
    for part, weeks, adjusted in season_store.stored_selections(reg_weeks):
        if part != "Regular season" and not has_post:
            continue
        t = time.time()
        # The app's own builder, as `_final_season_tables` runs it.
        tables = leagues._build_season_tables(league.name, season, stamp, weeks, part,
                                              core.NO_FILTER, None, adjusted)
        name = season_store.selection_name(part, weeks, adjusted)
        (out / season_store.asset_name(league.name, season, name)).write_bytes(
            season_store.pack_tables(tables, code))
        built.append(name)
        print(f"{league.name} {season} {name}: {time.time() - t:.1f}s", flush=True)
    (out / season_store.asset_name(league.name, season, "ids")).write_bytes(
        season_store.pack_frame(core.player_ids(pbp), code))
    # Written last: the app only looks for the selections once it sees this.
    (out / season_store.asset_name(league.name, season, "manifest")).write_text(json.dumps(
        {"format": season_store.FORMAT, "code": code, "reg_weeks": reg_weeks,
         "selections": built}))


def build_baselines(league: leagues.League, season: int, out: Path) -> None:
    t = time.time()
    tree = leagues.compute_baselines(league.name, season)
    if not any(tree.values()):
        raise SystemExit(f"{league.name} {season}: no baseline data")
    (out / season_store.asset_name(league.name, season, "baselines")).write_bytes(
        season_store.pack_tree(tree, leagues._CODE_VERSION))
    print(f"{league.name} {season} baselines: {time.time() - t:.1f}s", flush=True)


def main() -> None:
    if sys.argv[1] == "baselines":
        season = int(sys.argv[2])
        out = Path(sys.argv[3] if len(sys.argv) > 3 else "season_store_out")
        league = leagues.LEAGUES[sys.argv[4] if len(sys.argv) > 4 else "NFL"]
        out.mkdir(parents=True, exist_ok=True)
        build_baselines(league, season, out)
        return
    first, last = int(sys.argv[1]), int(sys.argv[2])
    out = Path(sys.argv[3] if len(sys.argv) > 3 else "season_store_out")
    league = leagues.LEAGUES[sys.argv[4] if len(sys.argv) > 4 else "NFL"]
    out.mkdir(parents=True, exist_ok=True)
    for season in range(first, last + 1):
        if not league.season_final(season):
            print(f"{league.name} {season}: not over yet, skipped")
            continue
        build(league, season, out)


if __name__ == "__main__":
    main()
