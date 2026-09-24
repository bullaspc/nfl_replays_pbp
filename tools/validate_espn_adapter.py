"""
Round-trip check of live_feed.espn_to_base / build_live_pbp.

Builds ESPN-shaped summary JSON from published nflverse games (gamebook text
without jersey numbers, start down/distance/yardsToEndzone, team ids, running
score, kickoffs on the kicking team, and, on every other game, the try folded
into the touchdown text), runs it through the live path and compares with the
official values. The JSON layout follows ESPN's public NFL summary endpoint.

    python tools/validate_espn_adapter.py 2025 [local.parquet]
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import live_feed as lf  # noqa: E402
import nflfastr_models as nm  # noqa: E402

PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.parquet"
TO_ESPN = {v: k for k, v in lf.ESPN_TEAM_FIX.items() if k != "JAC"}


def fake_summary(g: pd.DataFrame, fold_pat: bool) -> dict:
    home, away = g["home_team"].iloc[0], g["away_team"].iloc[0]
    ids = {home: "1", away: "2"}
    plays, i = [], 0
    rows = g.to_dict("records")
    while i < len(rows):
        r = rows[i]
        text = re.sub(r"(?<![\w-])\d{1,2}-(?=[A-Z])", "", str(r["desc"]))
        text = re.sub(r"\b([A-Z]{2,3})-\d{1,2}-", r"\1-", text)
        ptype = "Play"
        if text == "GAME":
            i += 1
            continue
        if re.match(r"END QUARTER", text):
            ptype = "End Period"
        elif text == "END GAME":
            ptype = "End of Game"
        elif text.startswith("Timeout #"):
            ptype = "Timeout"
        elif "Two-Minute Warning" in text:
            ptype = "Two-minute warning"
        team = r["posteam"]
        if r["play_type"] == "kickoff":
            team = r["defteam"]  # ESPN: the kicking team starts with the ball
        post_h, post_a = r["total_home_score"], r["total_away_score"]
        nxt = rows[i + 1] if i + 1 < len(rows) else None
        if (fold_pat and r["touchdown"] == 1 and nxt is not None
                and (nxt["play_type"] == "extra_point" or pd.notna(nxt["two_point_conv_result"]))):
            text += " " + re.sub(r"(?<![\w-])\d{1,2}-(?=[A-Z])", "", str(nxt["desc"]))
            post_h, post_a = nxt["total_home_score"], nxt["total_away_score"]
            i += 1
        plays.append({
            "id": str(len(plays)), "sequenceNumber": str(len(plays)),
            "type": {"text": ptype}, "text": text,
            "period": {"number": int(r["qtr"])},
            "clock": {"displayValue": str(r["time"] or "")},
            "homeScore": int(post_h), "awayScore": int(post_a),
            "start": {"down": int(r["down"]) if pd.notna(r["down"]) else 0,
                      "distance": int(r["ydstogo"]) if pd.notna(r["ydstogo"]) else 0,
                      "yardsToEndzone": int(r["yardline_100"]) if pd.notna(r["yardline_100"]) else None,
                      "team": {"id": ids.get(team)} if team in ids else {}},
        })
        i += 1
    comps = [{"homeAway": "home", "team": {"id": "1", "abbreviation": TO_ESPN.get(home, home)}},
             {"homeAway": "away", "team": {"id": "2", "abbreviation": TO_ESPN.get(away, away)}}]
    # One ESPN "drive" per nflverse drive number (admin rows ride along).
    drives, cur, cur_d = [], [], None
    for p, r in zip(plays, [x for x in rows if x["desc"] != "GAME"]):
        d = r["drive"]
        if cur and pd.notna(d) and d != cur_d:
            drives.append({"plays": cur})
            cur = []
        cur.append(p)
        cur_d = d if pd.notna(d) else cur_d
    drives.append({"plays": cur})
    return {"header": {"competitions": [{"competitors": comps, "status": {"type": {"state": "post"}}}]},
            "drives": {"previous": drives[:-1], "current": drives[-1]}}


def main(season: int, path: str | None = None, n_games: int = 60) -> None:
    pbp = pd.read_parquet(path or PBP_URL.format(y=season))
    models, fg = nm.load_models(), nm.load_fg_table()
    game_ids = pbp["game_id"].drop_duplicates().iloc[:n_games]
    res = []
    for k, gid in enumerate(game_ids):
        g = pbp[pbp["game_id"] == gid]
        sched = {"game_id": gid, "season": season, "home_team": g["home_team"].iloc[0],
                 "away_team": g["away_team"].iloc[0], "roof": g["roof"].iloc[0],
                 "location": g["location"].iloc[0], "game_type": g["season_type"].iloc[0],
                 "spread_line": g["spread_line"].iloc[0]}
        live = lf.build_live_pbp(fake_summary(g, fold_pat=k % 2 == 1), sched, models, fg, [])
        # Pair plays by quarter + gamebook text with the official rows.
        off = g[g["posteam"].notna() & g["play_type"].isin(["pass", "run", "punt", "field_goal"])].copy()
        off["key"] = off["desc"].map(lf.clean_desc)
        lv = live.copy()
        lv["key"] = lv["desc"].map(lf.clean_desc)
        for x in (off, lv):
            x["k2"] = x.groupby(["qtr", "key"]).cumcount()
        m = off.merge(lv, on=["qtr", "key", "k2"], suffixes=("_off", "_live"))
        res.append(pd.DataFrame({
            "game": gid, "folded": k % 2 == 1, "paired": len(m) / len(off),
            "epa_off": m["epa_off"], "epa_live": m["epa_live"],
            "wp_off": m["home_wp_off"], "wp_live": m["home_wp_live"],
            "score_off": m["total_home_score_off"], "score_live": m["total_home_score_live"]}))
    r = pd.concat(res)
    print(f"{season}: {len(game_ids)} games, {len(r)} scrimmage/kick plays paired by text")
    print(f"  official plays found in the live frame: {r.groupby('game')['paired'].first().mean():.2%}")
    for folded, x in r.groupby("folded"):
        print(f"  try {'folded into TD text' if folded else 'as its own row'}:")
        print(f"    epa within 1e-3:     {((x.epa_off - x.epa_live).abs() < 1e-3).mean():.3%}")
        print(f"    home_wp within 1e-3: {((x.wp_off - x.wp_live).abs() < 1e-3).mean():.3%}")
        print(f"    home score equal:    {(x.score_off == x.score_live).mean():.3%}")


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else None)
