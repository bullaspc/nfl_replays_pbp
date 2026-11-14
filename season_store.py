"""
The season store: finished seasons' Season stats tables, precomputed and
published as GitHub Release assets, so a fresh deploy (Streamlit Community
Cloud wipes its disk on every reboot and redeploy) reads a ~270 KB file
instead of downloading a season's play-by-play and computing for ~20s.

    tools/build_season_store.py builds the files;
    .github/workflows/season_store.yml publishes them to the release;
    leagues.season_tables() / season_player_ids() read them.

One zip per (league, season, selection), plus a manifest and the player ids
per season. The stored selections are the ones the Season stats page and its
comparisons ask for most: no situation filter, all venues, Raw and
Opponent-adjusted, over regular-season weeks 1–k for every k, the whole
season with the postseason, and the postseason alone. Anything else is
computed as before. Every file carries the hash of the code that built it
(`leagues._CODE_VERSION`), and the app only uses files from its own code.

The current season of a league in `leagues.CURRENT_STORE_LEAGUES` (college
football) is stored too, as "current_*" files: its mapped play-by-play, its
teams (in the manifest) and the regular-season selections, which take ~40 s
each to compute for a college season. That data changes with every rebuild of
the league's release, so these files also carry the data stamp they were
built from, and the app uses them only while it matches the live stamp.
.github/workflows/current_season_store.yml rebuilds them when it changes.

Plain data only: parquet tables and JSON, no pickles from the network.
No Streamlit calls.
"""
import io
import json
import zipfile

import pandas as pd

import replay_core as core

FORMAT = 1
# What a selection's zip holds: its DataFrames by `_season_tables()` key,
# and the scalars beside them in meta.json.
TABLE_KEYS = ("offense", "defense", "passing", "rushing", "receiving", "defenders",
              "kicking", "punting", "situational_team", "situational_defense")
PARTS = {"Regular season": "reg", "Both": "both", "Postseason": "post"}


def selection_name(part: str, weeks: tuple[int, int] | None, adjusted: bool) -> str:
    """A stored selection's name: "reg_1-9_adj", "both_raw", "post_adj"."""
    tag = "adj" if adjusted else "raw"
    if part == "Regular season":
        return f"reg_{weeks[0]}-{weeks[1]}_{tag}"
    return f"{PARTS[part]}_{tag}"


def stored_selections(reg_weeks: list[int]) -> list[tuple[str, tuple[int, int], bool]]:
    """(part, weeks, adjusted) for every selection the store keeps."""
    first, last = reg_weeks[0], reg_weeks[-1]
    out = []
    for adjusted in (False, True):
        out += [("Regular season", (first, k), adjusted) for k in reg_weeks]
        out += [("Both", (first, last), adjusted), ("Postseason", (first, last), adjusted)]
    return out


def lookup(manifest: dict, weeks: tuple[int, int], part: str, play_filter, venue,
           adjusted: bool) -> str | None:
    """The stored selection that gives the same tables as this one, or None:
    no situation filter or venue, and weeks from the season's first (the
    last clipped to the season's own: weeks 1–18 of a 17-week season are
    all of it)."""
    if play_filter != core.NO_FILTER or venue or part not in PARTS:
        return None
    reg = manifest.get("reg_weeks") or []
    if not reg:
        return None
    if part == "Postseason":
        return selection_name(part, None, adjusted)
    lo, hi = int(weeks[0]), min(int(weeks[1]), reg[-1])
    if lo > reg[0] or hi < lo:  # starts later than week 1: not stored
        return None
    if part == "Both":
        if hi != reg[-1]:
            return None
        if manifest.get("has_post", True):
            return selection_name(part, None, adjusted)
        part = "Regular season"  # no postseason games yet: the same plays
    hi = max(w for w in reg if w <= hi) if any(w <= hi for w in reg) else None
    return selection_name(part, (reg[0], hi), adjusted) if hi is not None else None


def current_selections(reg_weeks: list[int], has_post: bool) -> list[tuple[str, tuple[int, int], bool]]:
    """(part, weeks, adjusted) the current season's store keeps: regular-season
    weeks 1–k for every k, Raw and Opponent-adjusted (the Season stats page's
    defaults, the Matchup page's and the replay preview's weeks before a
    game), plus Both and Postseason once there are postseason games (before,
    `lookup` answers Both with the whole regular season)."""
    return [sel for sel in stored_selections(reg_weeks) if has_post or sel[0] == "Regular season"]


def current(name: str) -> str:
    """The current season's file for a selection or "manifest" / "pbp"."""
    return f"current_{name}"


def asset_name(league: str, season: int, what: str) -> str:
    """"NFL_2023_reg_1-9_adj.zip", "NFL_2023_ids.zip", "NFL_2023_manifest.json",
    "College_football_2026_current_manifest.json"."""
    ext = "json" if what.endswith("manifest") else "zip"
    return f"{league.replace(' ', '_')}_{season}_{what}.{ext}"


def _parquet(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf)
    return buf.getvalue()


def pack_tables(tables: dict, code: str, stamp: str | None = None) -> bytes:
    """A `_season_tables()` dict as a zip: one parquet per table plus
    meta.json (the scalars, the code hash and, for the current season, the
    data stamp)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:  # parquet is compressed already
        for key in TABLE_KEYS:
            z.writestr(f"{key}.parquet", _parquet(tables[key]))
        z.writestr("meta.json", json.dumps({
            "format": FORMAT, "code": code, "stamp": stamp, "missing": sorted(tables["missing"]),
            "play_share": float(tables["play_share"]),
            "charted": [int(x) for x in tables["charted"]]}))
    return buf.getvalue()


def _current(meta: dict, code: str, stamp: str | None) -> bool:
    """Whether a file's meta.json says it was built by `code` (and, when a
    stamp is asked for, from that data)."""
    return (meta.get("format") == FORMAT and meta.get("code") == code
            and (stamp is None or meta.get("stamp") == stamp))


def unpack_tables(data: bytes, code: str, stamp: str | None = None) -> dict | None:
    """`pack_tables()`'s zip back as the `_season_tables()` dict, or None if
    it was built by other code (or another format, or from other data than
    `stamp`)."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        meta = json.loads(z.read("meta.json"))
        if not _current(meta, code, stamp):
            return None
        tables = {key: pd.read_parquet(io.BytesIO(z.read(f"{key}.parquet"))) for key in TABLE_KEYS}
    tables["situational"] = core.situational_distributions_from(tables["situational_team"])
    tables["missing"] = frozenset(meta["missing"])
    tables["play_share"] = meta["play_share"]
    tables["charted"] = tuple(meta["charted"])
    return tables


def pack_frame(df: pd.DataFrame, code: str, stamp: str | None = None) -> bytes:
    """One DataFrame (the season's player ids, the current season's
    play-by-play) with the code hash and the data stamp."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        z.writestr("data.parquet", _parquet(df))
        z.writestr("meta.json", json.dumps({"format": FORMAT, "code": code, "stamp": stamp}))
    return buf.getvalue()


def unpack_frame(data: bytes, code: str, stamp: str | None = None) -> pd.DataFrame | None:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        meta = json.loads(z.read("meta.json"))
        if not _current(meta, code, stamp):
            return None
        return pd.read_parquet(io.BytesIO(z.read("data.parquet")))


# ---------- Percentile baselines ----------
# The replay's percentile colors come from a league's earlier seasons
# (`leagues.compute_baselines`): nested dicts of arrays and a few small
# DataFrames. Building them means loading those seasons' play-by-play, which
# for college peaked over a gigabyte, so they're published here too.
import numpy as np  # noqa: E402


def pack_tree(tree, code: str) -> bytes:
    """Nested dicts (string keys) of numpy arrays, DataFrames and plain
    values as a zip: tree.json for the shape, a .npy or .parquet per leaf.
    Raises TypeError on anything else (object arrays included), so a build
    fails rather than publishing something the app can't read back."""
    files = {}

    def enc(o):
        if isinstance(o, dict):
            if not all(isinstance(k, str) for k in o):
                raise TypeError("only string keys can be stored")
            return {"t": "dict", "v": {k: enc(v) for k, v in o.items()}}
        if isinstance(o, np.ndarray):
            name = f"{len(files)}.npy"
            buf = io.BytesIO()
            np.save(buf, o, allow_pickle=False)
            files[name] = buf.getvalue()
            return {"t": "array", "f": name}
        if isinstance(o, pd.DataFrame):
            name = f"{len(files)}.parquet"
            files[name] = _parquet(o)
            return {"t": "frame", "f": name}
        if isinstance(o, (np.integer, np.floating, np.bool_)):
            return {"t": "value", "v": o.item()}
        if o is None or isinstance(o, (bool, int, float, str)):
            return {"t": "value", "v": o}
        raise TypeError(f"can't store a {type(o).__name__}")

    shape = enc(tree)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
        z.writestr("tree.json", json.dumps({"format": FORMAT, "code": code, "tree": shape}))
    return buf.getvalue()


def unpack_tree(data: bytes, code: str):
    """`pack_tree()`'s zip back as the nested dicts, or None if it was built
    by other code (or another format)."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        meta = json.loads(z.read("tree.json"))
        if meta.get("format") != FORMAT or meta.get("code") != code:
            return None

        def dec(node):
            t = node["t"]
            if t == "dict":
                return {k: dec(v) for k, v in node["v"].items()}
            if t == "array":
                return np.load(io.BytesIO(z.read(node["f"])), allow_pickle=False)
            if t == "frame":
                return pd.read_parquet(io.BytesIO(z.read(node["f"])))
            return node["v"]

        return dec(meta["tree"])
