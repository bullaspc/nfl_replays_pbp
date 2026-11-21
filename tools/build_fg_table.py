"""
Rebuild models/fg_prob_table.csv from nflfastR's own data, and check the
kickoff starts in nflfastr_models.py.

nflfastR's field goal model (R/helper_fg_model.R) is a lookup table,
`fg_prob_table` in the package's R/sysdata.rda: the make probability for
yardline_100 1-99 (a kick of yardline_100 + 18 yards), one column per kicking
era (`fg_era()`) and roof (outdoors or indoors). The same file holds
`wp_vegas_model$kick_start`, the receiving team's typical first snap after a
kickoff in each kickoff-rule regime, which the EP and WP models use; this
fails if it no longer matches nflfastr_models.KICK_START.

R's .rda is a bzip2/gzip/xz-compressed XDR serialization; `read_rda()` below
reads the object types sysdata.rda uses (vectors, lists, symbols, raw).

    python tools/build_fg_table.py [path/to/sysdata.rda]
"""
import bz2
import gzip
import lzma
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nflfastr_models as nm  # noqa: E402

SYSDATA_URL = "https://raw.githubusercontent.com/nflverse/nflfastR/master/R/sysdata.rda"


class _Reader:
    """R's XDR serialization (format 2 or 3), enough for data objects."""

    def __init__(self, data: bytes):
        self.d, self.i, self.refs = data, 0, []

    def int(self) -> int:
        v = struct.unpack_from(">i", self.d, self.i)[0]
        self.i += 4
        return v

    def length(self) -> int:
        n = self.int()
        if n == -1:
            hi, lo = self.int(), self.int()
            n = (hi << 32) + lo
        return n

    def take(self, n: int) -> bytes:
        b = self.d[self.i:self.i + n]
        self.i += n
        return b

    def obj(self):
        flags = self.int()
        typ, has_attr, has_tag = flags & 0xFF, flags & (1 << 9), flags & (1 << 10)
        if typ == 254:  # NILVALUE
            return None
        if typ == 255:  # REFSXP
            idx = flags >> 8 or self.int()
            return self.refs[idx - 1]
        if typ in (241, 242, 253):  # base / empty / global environment
            return None
        if typ == 1:  # SYMSXP
            name = self.obj()
            self.refs.append(name)
            return name
        if typ in (2, 6, 239, 240):  # pairlist, language
            attr = self.obj() if has_attr else None
            tag = self.obj() if has_tag else None
            car = self.obj()
            cdr = self.obj()
            node = [(tag, car)] + (cdr if isinstance(cdr, list) else [])
            return _Pairlist(node, attr)
        if typ == 9:  # CHARSXP
            n = self.int()
            return None if n == -1 else self.take(n).decode("utf-8", "replace")
        if typ in (10, 13):  # logical, integer
            n = self.length()
            val = np.frombuffer(self.take(4 * n), dtype=">i4").astype(np.int64)
        elif typ == 14:  # double
            n = self.length()
            val = np.frombuffer(self.take(8 * n), dtype=">f8").astype(float)
        elif typ == 16:  # character
            val = [self.obj() for _ in range(self.length())]
        elif typ in (19, 20):  # list, expression
            val = [self.obj() for _ in range(self.length())]
        elif typ == 24:  # raw
            val = self.take(self.length())
        else:
            raise NotImplementedError(f"R object type {typ} at byte {self.i}")
        attr = self.obj() if has_attr else None
        return _RObj(val, attr)


class _RObj:
    def __init__(self, value, attr):
        self.value, self.attr = value, attr

    def attribute(self, name: str):
        return self.attr.get(name) if isinstance(self.attr, _Pairlist) else None

    def __getitem__(self, name: str):
        names = self.attribute("names").value
        return self.value[names.index(name)]


class _Pairlist(list):
    def __init__(self, items, attr=None):
        super().__init__(items)
        self.attr = attr

    def get(self, name: str):
        return next((v for k, v in self if k == name), None)


def read_rda(blob: bytes) -> dict:
    """An .rda file's objects by name."""
    for opener in (bz2.decompress, gzip.decompress, lzma.decompress):
        try:
            blob = opener(blob)
            break
        except (OSError, ValueError, lzma.LZMAError):
            continue
    if blob[:5] not in (b"RDX2\n", b"RDX3\n") or blob[5:7] != b"X\n":
        raise ValueError("not an XDR .rda file")
    r = _Reader(blob[7:])
    version = r.int()
    r.int(), r.int()  # writer and minimal reader versions
    if version == 3:
        r.take(r.int())  # native encoding
    return dict(r.obj())


def main(path: str | None = None) -> None:
    if path:
        blob = Path(path).read_bytes()
    else:
        with urllib.request.urlopen(SYSDATA_URL, timeout=60) as resp:
            blob = resp.read()
    objs = read_rda(blob)

    fg = objs["fg_prob_table"]
    n_rows, n_cols = fg.attribute("dim").value
    cols = fg.attribute("dimnames").value[1].value
    mat = fg.value.reshape(n_cols, n_rows).T  # R stores matrices by column
    assert n_rows == 99, n_rows
    out = pd.DataFrame(mat, columns=cols)
    out.insert(0, "yardline_100", np.arange(1, n_rows + 1))
    expected = [f"era{e}_{r}" for e in range(1, len(nm.FG_ERAS) + 2) for r in ("indoors", "outdoors")]
    assert sorted(cols) == sorted(expected), f"FG table columns changed: {cols}"
    out.to_csv(nm.FG_TABLE, index=False, float_format="%.10g")
    print(f"wrote {n_rows} yard lines x {n_cols} era/roof columns to {nm.FG_TABLE}")

    kick_start = tuple(float(x) for x in objs["wp_vegas_model"]["kick_start"].value)
    if kick_start != tuple(nm.KICK_START):
        sys.exit(f"nflfastR's kick_start is now {kick_start}; update nflfastr_models.KICK_START "
                 f"(and KICK_REGIMES from ot_kick_first in R/helper_add_ot_wp.R)")
    print(f"kick_start matches nflfastr_models.KICK_START: {kick_start}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
