"""完整原件数学局部旋转不能被斜水印入口误删，原字索引与候选角色必须保持。"""

from hashlib import sha256
import math
from pathlib import Path
import pickle

import pytest

from docvortex.analyzers.native.pdf import native_text
from _native_direction_fixtures import load_direction_pickle


FIXTURE = Path(__file__).parent / "fixtures/batch2_auxiliary/v18_actual_math_local_rotation_input.pickle.gz"
FIXTURE_SHA = "7e3a30a6f7f1f5de2410d09b25c68f68374f12efb4a27a533321f128292da5a7"

# 全页实际records中的数学候选，固定原字身份；空格不充当数学墨迹证明。
FORMULA_MEMBERS = (
    (1284, 1285, 1286, 1287, 1288, 1290),
    (1296, 1297),
    (1300,),
    (1303, 1304, 1305),
    (1314, 1315),
    (1318, 1319, 1320),
    (1327, 1328, 1329, 1330, 1331, 1333),
    (1339, 1340),
    (1343,),
    (1346, 1347, 1348),
    (1357, 1358),
    (1361, 1362, 1363),
    (1399, 1400, 1401, 1402, 1403, 1405),
    (1414, 1415, 1416, 1417, 1418),
    (1427, 1428, 1429, 1430, 1431),
)


def _saved_actual_input():
    """校验完整实捕fixture后恢复共享原字和旧视觉行，不打开或提取原PDF。"""
    raw = FIXTURE.read_bytes()
    assert sha256(raw).hexdigest() == FIXTURE_SHA
    result = load_direction_pickle(raw)
    assert len(result["raw_chars"]) == 2186
    assert len(result["records"]) == 120
    assert len(result["lines"]) == 119
    assert result["rotation"] == 0
    assert {char["char_idx"] for char in result["raw_chars"]} == set(range(2186))
    return result


def _native_members(lines):
    """只核真实原字身份，文字相等、框重叠或清除候选标志都不能替代成员保留。"""
    return {char["char_idx"] for line in lines for char in line.chars}


def test_actual_math_rotation_fixture_is_radians_with_mixed_size_standard_anchors():
    """保存事实明确10度数学斜体与小字号标准脚标同原流相邻，不能将0.174当度数。"""
    saved = _saved_actual_input()
    raw = {char["char_idx"]: char for char in saved["raw_chars"]}
    assert math.degrees(raw[1284]["rotation"]) == pytest.approx(9.981829997999, abs=0.01)
    assert raw[1284]["writing_angle"] == -raw[1284]["rotation"]
    assert raw[1285]["rotation"] == 0
    assert raw[1284]["font"]["size"] > raw[1285]["font"]["size"]
    assert raw[1284]["origin"][0] < raw[1285]["origin"][0]


@pytest.mark.parametrize("members", FORMULA_MEMBERS)
def test_actual_full_math_records_builder_keeps_original_formula_members(members):
    """完整120records走真实builder，既有数学左项/脚标不可因formula_only标志无条件丢弃。"""
    saved = _saved_actual_input()
    lines = native_text._build_native_line_items_from_records(saved["records"], saved["page_size"])
    assert set(members) <= _native_members(lines)


@pytest.mark.parametrize("members", FORMULA_MEMBERS)
def test_actual_full_math_direction_filter_keeps_original_formula_members(members):
    """隔离真实方向filter核全119行和2186原字，候选角色合法可留但不能被直接剪掉。"""
    saved = _saved_actual_input()
    raw = saved["raw_chars"]
    # Bbox采用对象身份相等；序列化完整序列可以核验数值/顺序/共享关系未被改写。
    before = pickle.dumps(raw, protocol=pickle.HIGHEST_PROTOCOL)
    lines = saved["lines"]
    chars = native_text._filter_native_direction_chars(raw, page_rotation=saved["rotation"])
    assert set(members) <= _native_members(lines)
    assert set(members) <= {char["char_idx"] for char in chars}
    originals = {char["char_idx"]: char for char in raw}
    assert all(originals[char["char_idx"]] is char for char in chars)
    assert pickle.dumps(raw, protocol=pickle.HIGHEST_PROTOCOL) == before


def test_actual_full_math_builder_and_filter_keep_all_old_math_members_together():
    """真实两个入口串联后全部旧候选字仍可交给公式恢复，不先删除任何左项/分子分母。"""
    saved = _saved_actual_input()
    built = native_text._build_native_line_items_from_records(saved["records"], saved["page_size"])
    lines = built
    chars = native_text._filter_native_direction_chars(saved["raw_chars"], page_rotation=saved["rotation"])
    expected = {index for group in FORMULA_MEMBERS for index in group}
    assert expected <= _native_members(lines)
    assert expected <= {char["char_idx"] for char in chars}
