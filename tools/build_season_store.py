"""
Build the season store's files for finished seasons (season_store.py): each
stored selection's Season stats tables, the player ids and a manifest, into
a folder that .github/workflows/season_store.yml uploads to the
`season-store` release.

    python tools/build_season_store.py 2016 2025 [out_dir] [league]
    python tools/build_season_store.py baselines 2015 2026 [out_dir] [league]

League defaults to NFL, out_dir to ./season_store_out. A season that isn't over
yet (`League.season_final`) is skipped. About 40 selections a season, each
built from scratch like the app builds it (~5–10 s each).

`baselines` builds the replay's percentile baselines for each season instead
(`leagues.compute_baselines`, from the seasons before it; ~40 s, ~0.5 MB
each): the replay page then never loads those seasons' play-by-play. A season
without baseline data is skipped with a warning.

`current` builds the current season's files (season_store.current_*), for
.github/workflows/current_season_store.yml:

    python tools/build_season_store.py current plan [out_dir] [league] [--force]
    python tools/build_season_store.py current build <chunk> <chunks> <manifest> [out_dir] [league]

`plan` writes the manifest and the play-by-play (tagged with the data stamp)
unless the store already has this data built by this code, and sets the
workflow outputs `build` and `chunks`; each `build` job then computes every
`chunks`-th selection of that manifest. A job fails if the data's stamp
changes under it, so nothing built from mixed data is published.
"""
import json
import os
import sys
import time
from datetime import date
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
        print(f"::warning::{league.name} {season}: no baseline data, skipped", flush=True)
        return
    (out / season_store.asset_name(league.name, season, "baselines")).write_bytes(
        season_store.pack_tree(tree, leagues._CODE_VERSION))
    print(f"{league.name} {season} baselines: {time.time() - t:.1f}s", flush=True)


# ---------- The current season ----------
CHUNKS = 4  # parallel build jobs: a college season's ~22 selections take ~40 s each


def _output(key: str, value: str) -> None:
    """A GitHub Actions step output (printed when run by hand)."""
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as f:
            f.write(f"{key}={value}\n")
    print(f"{key}={value}", flush=True)


def _fresh_stamp(league: leagues.League) -> str:
    """The league's data stamp, read now (its 60 s cache cleared)."""
    leagues.cfb_stamp.clear()
    leagues.nflverse_stamp.clear()
    return league.stamp()


def current_season(league: leagues.League) -> int | None:
    """The season under way (college and NFL seasons start in August or
    September and end in the new year), or None once it's over."""
    today = date.today()
    season = today.year if today.month >= 8 else today.year - 1
    return None if league.season_final(season) else season


def plan_current(league: leagues.League, out: Path, force: bool) -> None:
    season = current_season(league)
    stamp = _fresh_stamp(league)
    if season is None or not stamp or stamp.startswith("unknown"):
        print(f"{league.name}: no season under way or no data stamp", flush=True)
        _output("build", "false")
        return
    code = leagues._CODE_VERSION
    data = leagues._store_get(season_store.asset_name(league.name, season,
                                                      season_store.current("manifest")))
    published = json.loads(data) if data else {}
    if not force and published.get("code") == code and published.get("stamp") == stamp:
        print(f"{league.name} {season}: up to date (data {stamp})", flush=True)
        _output("build", "false")
        return
    t = time.time()
    pbp = league.season_pbp(season, stamp)
    teams = league.season_teams(season, stamp)
    if _fresh_stamp(league) != stamp:
        sys.exit(f"{league.name} {season}: the data changed while loading; the next run builds it")
    reg_weeks = sorted(int(w) for w in pbp.loc[~pbp["postseason"], "week"].dropna().unique())
    has_post = bool(pbp["postseason"].any())
    if not reg_weeks:
        print(f"{league.name} {season}: no regular-season games yet", flush=True)
        _output("build", "false")
        return
    names = [season_store.selection_name(*sel) for sel in
             season_store.current_selections(reg_weeks, has_post)]
    (out / season_store.asset_name(league.name, season, season_store.current("pbp"))).write_bytes(
        season_store.pack_frame(pbp, code, stamp))
    (out / season_store.asset_name(league.name, season, season_store.current("manifest"))).write_text(
        json.dumps({"format": season_store.FORMAT, "code": code, "stamp": stamp,
                    "season": season, "reg_weeks": reg_weeks, "has_post": has_post,
                    "teams": {str(k): v for k, v in teams.items()}, "selections": names}))
    print(f"{league.name} {season}: {len(names)} selections to build from data {stamp} "
          f"({time.time() - t:.1f}s)", flush=True)
    _output("build", "true")
    _output("chunks", json.dumps(list(range(min(CHUNKS, len(names))))))


def build_current(league: leagues.League, chunk: int, chunks: int, manifest_path: Path,
                  out: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    season, stamp, code = manifest["season"], manifest["stamp"], leagues._CODE_VERSION
    if manifest["code"] != code:
        sys.exit("the manifest was built by other code")
    if _fresh_stamp(league) != stamp:
        sys.exit(f"{league.name} {season}: the data changed since the plan; the next run builds it")
    by_name = {season_store.selection_name(*sel): sel for sel in
               season_store.current_selections(manifest["reg_weeks"], manifest["has_post"])}
    # The longest (most weeks) first, dealt out in turn so the jobs take about as long.
    names = sorted(manifest["selections"], key=lambda n: -by_name[n][1][1])[chunk::chunks]
    for name in names:
        part, weeks, adjusted = by_name[name]
        t = time.time()
        tables = leagues._build_season_tables(league.name, season, stamp, weeks, part,
                                              core.NO_FILTER, None, adjusted)
        (out / season_store.asset_name(league.name, season, season_store.current(name))).write_bytes(
            season_store.pack_tables(tables, code, stamp))
        print(f"{league.name} {season} {name}: {time.time() - t:.1f}s", flush=True)
    if _fresh_stamp(league) != stamp:
        sys.exit(f"{league.name} {season}: the data changed during the build; the next run builds it")


def main() -> None:
    if sys.argv[1] == "current":
        # The builder computes: it must not read the store it's rebuilding.
        leagues.CURRENT_STORE_LEAGUES = set()
        args = [a for a in sys.argv[2:] if a != "--force"]
        if args[0] == "plan":
            out = Path(args[1] if len(args) > 1 else "season_store_out")
            league = leagues.LEAGUES[args[2] if len(args) > 2 else "College football"]
            out.mkdir(parents=True, exist_ok=True)
            plan_current(league, out, "--force" in sys.argv)
        else:
            chunk, chunks, manifest = int(args[1]), int(args[2]), Path(args[3])
            out = Path(args[4] if len(args) > 4 else "season_store_out")
            league = leagues.LEAGUES[args[5] if len(args) > 5 else "College football"]
            out.mkdir(parents=True, exist_ok=True)
            build_current(league, chunk, chunks, manifest, out)
        return
    if sys.argv[1] == "baselines":
        first, last = int(sys.argv[2]), int(sys.argv[3])
        out = Path(sys.argv[4] if len(sys.argv) > 4 else "season_store_out")
        league = leagues.LEAGUES[sys.argv[5] if len(sys.argv) > 5 else "NFL"]
        out.mkdir(parents=True, exist_ok=True)
        for season in range(first, last + 1):
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
