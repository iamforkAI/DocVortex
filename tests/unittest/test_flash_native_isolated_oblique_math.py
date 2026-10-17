"""方向准入保护真实同行孤立数学字，不能将图表的仿斜体表头作为斜向水印删除。"""

import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import pickle

import pytest

from docvortex.analyzers.native.pdf.native_text import (
    _build_native_line_items_from_records,
    _filter_native_direction_chars,
)


FIXTURE = Path(__file__).parent / "fixtures/layout_fallback/isolated_oblique_math_source116.pickle.gz"


@pytest.mark.parametrize("key", ("d-a48ceec828ddab68:19", "d-a48ceec828ddab68:20", "d-ed2e1025e10cf2e6:20"))
def test_actual_table_header_and_definition_math_glyphs_are_retained(key):
    """两页12个x表头及首格λ、释义λ均保原身份，实际原页其余原字也完整保留。"""
    payload = FIXTURE.read_bytes()
    metadata = json.loads(FIXTURE.with_name("isolated_oblique_math_source116.metadata.json").read_text())
    assert sha256(payload).hexdigest() == metadata["fixture_sha256"]
    row = pickle.loads(gzip.decompress(payload))[key]
    chars = row["before_chars"]
    originals = {c["char_idx"]: c for c in chars}
    lines = _build_native_line_items_from_records(row["records"], row["page_size"])
    filtered = _filter_native_direction_chars(chars, page_rotation=row["rotation"])
    assert [c["char_idx"] for c in filtered] == [c["char_idx"] for c in chars]
    assert all(c is originals[c["char_idx"]] for c in filtered)
    headers = {
        "d-a48ceec828ddab68:19": (145, 353, 517, 672, 876, 1021),
        "d-a48ceec828ddab68:20": (104, 231, 412, 563, 706, 890),
        "d-ed2e1025e10cf2e6:20": (21, 393),
    }[key]
    expected = "λ" if key.startswith("d-ed") else "x"
    assert all(originals[i]["char"] == expected for i in headers)
    assert all(math.degrees(originals[i]["rotation"]) == pytest.approx(9.982, abs=0.01) for i in headers)
    assert all(i in {c["char_idx"] for line in lines for c in line.chars} for i in headers)
