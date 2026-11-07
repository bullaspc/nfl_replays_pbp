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
        return selection_name(part, None, adjusted) if hi == reg[-1] else None
    hi = max(w for w in reg if w <= hi) if any(w <= hi for w in reg) else None
    return selection_name(part, (reg[0], hi), adjusted) if hi is not None else None


def asset_name(league: str, season: int, what: str) -> str:
    """"NFL_2023_reg_1-9_adj.zip", "NFL_2023_ids.zip", "NFL_2023_manifest.json"."""
    ext = "json" if what == "manifest" else "zip"
    return f"{league.replace(' ', '_')}_{season}_{what}.{ext}"


def _parquet(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf)
    return buf.getvalue()


def pack_tables(tables: dict, code: str) -> bytes:
    """A `_season_tables()` dict as a zip: one parquet per table plus
    meta.json (the scalars and the code hash)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:  # parquet is compressed already
        for key in TABLE_KEYS:
            z.writestr(f"{key}.parquet", _parquet(tables[key]))
        z.writestr("meta.json", json.dumps({
            "format": FORMAT, "code": code, "missing": sorted(tables["missing"]),
            "play_share": float(tables["play_share"]),
            "charted": [int(x) for x in tables["charted"]]}))
    return buf.getvalue()


def unpack_tables(data: bytes, code: str) -> dict | None:
    """`pack_tables()`'s zip back as the `_season_tables()` dict, or None if
    it was built by other code (or another format)."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        meta = json.loads(z.read("meta.json"))
        if meta.get("format") != FORMAT or meta.get("code") != code:
            return None
        tables = {key: pd.read_parquet(io.BytesIO(z.read(f"{key}.parquet"))) for key in TABLE_KEYS}
    tables["situational"] = core.situational_distributions_from(tables["situational_team"])
    tables["missing"] = frozenset(meta["missing"])
    tables["play_share"] = meta["play_share"]
    tables["charted"] = tuple(meta["charted"])
    return tables


def pack_frame(df: pd.DataFrame, code: str) -> bytes:
    """One DataFrame (the season's player ids) with the code hash."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        z.writestr("data.parquet", _parquet(df))
        z.writestr("meta.json", json.dumps({"format": FORMAT, "code": code}))
    return buf.getvalue()


def unpack_frame(data: bytes, code: str) -> pd.DataFrame | None:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        meta = json.loads(z.read("meta.json"))
        if meta.get("format") != FORMAT or meta.get("code") != code:
            return None
        return pd.read_parquet(io.BytesIO(z.read("data.parquet")))
