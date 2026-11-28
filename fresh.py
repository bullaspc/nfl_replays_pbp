"""Reload the app's own modules that are older than their files.

Streamlit Community Cloud updates a running app in place when main moves: it
pulls the code and reruns the pages, but a module already imported can stay
in memory as it was. A page that then imports a name the new module defines
fails ("cannot import name 'draft_note' from 'views'") until the app is
rebooted. Each page calls `reload_stale()` before its other imports: a module
whose file changed since it was compiled (the .pyc header records the
source's mtime and size) is reloaded, dependencies first.
"""
from __future__ import annotations

import importlib
import os
import struct
import sys

# The app's modules, each after the ones it imports.
ORDER = ["replay_core", "season_sim", "cfp_sim", "live_feed", "cfb_feed", "scoreboard", "nflfastr_models",
         "season_store", "leagues", "records", "game_summary", "rosters", "views", "college_sim_view"]
# (module, mtime, size) already reloaded, so a .pyc that can't be rewritten
# doesn't reload the module on every run.
_reloaded: set[tuple[str, int, int]] = set()


def _source_stamp(mod) -> tuple[int, int] | None:
    """(mtime, size) of the module's file now; None without one."""
    try:
        st = os.stat(mod.__file__)
    except (AttributeError, TypeError, OSError):
        return None
    return int(st.st_mtime) & 0xFFFFFFFF, st.st_size & 0xFFFFFFFF


def _compiled_stamp(mod) -> tuple[int, int] | None:
    """(mtime, size) of the source the module's .pyc was compiled from;
    None without a timestamp-based .pyc."""
    try:
        with open(mod.__cached__, "rb") as f:
            head = f.read(16)
    except (AttributeError, TypeError, OSError):
        return None
    if len(head) < 16 or struct.unpack("<I", head[4:8])[0]:  # hash-based pyc
        return None
    return struct.unpack("<II", head[8:16])


def reload_stale() -> list[str]:
    """Reload every `ORDER` module in memory whose file changed since it was
    compiled; the names reloaded."""
    done = []
    for name in ORDER:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        now, then = _source_stamp(mod), _compiled_stamp(mod)
        if now is None or then is None or now == then or (name, *now) in _reloaded:
            continue
        try:
            importlib.reload(mod)
        except Exception:
            continue  # the page's own import then shows the real error
        _reloaded.add((name, *now))
        done.append(name)
    return done
