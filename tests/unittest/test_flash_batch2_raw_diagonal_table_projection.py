"""完整实源验证斜水印不进入表格原字投影，默认公共输出保留真实表头和版本记录。"""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import pickle

from bs4 import BeautifulSoup
import pytest

from docvortex.analyzers.native.pdf.native_text import _filter_native_direction_chars
from _native_direction_fixtures import load_direction_pickle
from test_flash_pdf_char_geometry import _read_pdf_fixture


FIXTURE = Path(__file__).parent / "fixtures/batch2_auxiliary/v17_actual_raw_diagonal_table_sources.pickle.gz"
FIXTURE_SHA256 = "0784cc44d35300ede304a0fc2478e6bfea3222fb6b6ea560ffcd7d1a43eadafa"
PDF_SHA256 = "87d98f6f4e42152c38b0a1f8dcfefa3e5ac163cc086a6f37169cabadf77bd107"


def _saved_case(page: int) -> dict:
    """读取完整旧Source及同原件实际新原字集合，不伪造字号、字体、角色或包络。"""
    payload = FIXTURE.read_bytes()
    assert sha256(payload).hexdigest() == FIXTURE_SHA256
    case = load_direction_pickle(payload)[page]
    metadata = json.loads(FIXTURE.with_name(FIXTURE.name.replace(".pickle.gz", ".metadata.json")).read_text())
    page_metadata = next(item for item in metadata["rotation_metadata"]["record"]["pages"] if item["physical_page"] == page)
    assert list(case["source"].page_size) == page_metadata["page_size"]
    case["page_rotation"] = page_metadata["rotation"]
    return case


@pytest.mark.parametrize("page", [2, 21])
def test_actual_raw_projection_removes_only_standalone_diagonal_watermark(page: int) -> None:
    """两页各96个真315°斜字退出raw投影，42/54原行和全部标准表字仍保持原对象/原索引。"""
    case = _saved_case(page)
    source = case["source"]
    before = pickle.dumps((source.lines, source.chars), protocol=5)
    originals = {char["char_idx"]: char for char in source.chars}
    removed = [originals[index] for index in case["expected_removed_char_indices"]]
    assert len(removed) == 96
    assert "".join(char["char"] for char in removed) == "通用测试文档" * 16
    assert all(math.degrees(char["rotation"]) % 360 == pytest.approx(315, abs=0.001) for char in removed)
    # 水印字原本就没有进入正常视觉行；旧错误通道是全raw字进入表格投影，不能反向认领为正文。
    line_members = {char["char_idx"] for line in source.lines for char in line.chars}
    assert line_members.isdisjoint(case["expected_removed_char_indices"])
    for offset in range(0, len(removed), 6):
        first, last = removed[offset], removed[offset + 5]
        dx = last["origin"][0] - first["origin"][0]
        dy = last["origin"][1] - first["origin"][1]
        assert dx > 0 and dy < 0
        assert math.degrees(math.atan2(dy, dx)) == pytest.approx(-45, abs=0.001)

    lines = source.lines
    chars = _filter_native_direction_chars(source.chars, page_rotation=case["page_rotation"])
    assert len(lines) == case["expected_line_count"]
    assert len(lines) == len(source.lines)
    assert all(line is original for line, original in zip(lines, source.lines, strict=True))
    assert [char["char_idx"] for char in chars] == case["expected_retained_char_indices"]
    assert all(char is originals[char["char_idx"]] for char in chars)
    assert line_members <= {char["char_idx"] for char in chars}
    # 标准原文同样包含“用/文/档”等字，按原件几何及索引保护，不能以字面水印词清掉真正原文。
    overlap = {char["char"] for char in chars} & {char["char"] for char in removed}
    assert {"用", "文", "档"} <= overlap
    standard = [
        char["char_idx"]
        for char in source.chars
        if min(abs((math.degrees(char["rotation"]) - angle + 180) % 360 - 180) for angle in (0, 90, 180, 270)) <= 0.1
    ]
    assert set(standard) <= {char["char_idx"] for char in chars}
    assert pickle.dumps((source.lines, source.chars), protocol=5) == before


@pytest.fixture(scope="module")
def _default_public_mixed_pages() -> list:
    """完整24页公共parse核对源事实，不借历史检测辅助器改变公式或表格结果。"""
    from docvortex import parse

    source = Path(__file__).parents[2] / "tests/unittest/pdfs/flash_layout/mixed_text_layout_sample.pdf.xor"
    payload = _read_pdf_fixture(source)
    assert sha256(payload).hexdigest() == PDF_SHA256
    result = parse(payload, parse_mode="txt", keep_model_json=True)
    assert result.model_json is not None
    assert len(result.model_json.pages) == 24
    return result.model_json.pages


def _table_rows(page: list, bbox: list) -> list[list[str]]:
    """完整表父框定位后读取原生HTML单元格；只归一显示空白，不省略版本字符或污染字。"""
    tables = [block for block in page if block["type"] == "table" and list(block["bbox"]) == bbox]
    assert len(tables) == 1
    soup = BeautifulSoup(tables[0]["content"], "html.parser")
    return [
        ["".join(cell.get_text().split()) for cell in row.find_all(["td", "th"], recursive=False)]
        for row in soup.find_all("tr")
    ]


def test_default_public_version_history_table_has_source_headers_and_clean_records(_default_public_mixed_pages) -> None:
    """第2页四列表的真实表头、日期和版本逐项完整，不能让斜水印再成为上下标或单元正文。"""
    rows = _table_rows(_default_public_mixed_pages[1], [0.071, 0.11, 0.932, 0.4])
    assert rows == [
        ["版本号", "日期", "制/修订人", "内容描述"],
        ["1.0", "2023.04.10", "AWA2090", "初始版本"],
        ["2.0", "2024.02.20", "AWA1382", "1)第1章新增使用范围；2)第3章新增安装Docker镜像环境；3)新增第4章常见问题。"],
        ["2.1", "2024.04.11", "AWA1382", "增加MR536、T536平台说明。"],
        ["2.2", "2024.11.09", "AWA1382", "1)增加A733平台说明；2)第3章调整Docker方式安装为第一节，推荐使用。"],
        ["2.3", "2025.04.03", "AWA1382", "增加T736平台说明。"],
        ["2.4", "2025.09.25", "AWA1382", "第2.1节更新工具下载方式。"],
        ["3.0", "2025.10.16", "AWA1382", "全部章节根据modelzoo的使用更新说明。"],
    ]


def test_default_public_driver_table_preserves_every_native_version_field(_default_public_mixed_pages) -> None:
    """第21页每个版本值保真，不能将315°水印插入十六进制、软件版本及工具记录。"""
    rows = _table_rows(_default_public_mixed_pages[20], [0.071, 0.334, 0.929, 0.62])
    assert rows == [
        ["driverversion", "softwareversion", "NetworkVersion", "ToolVersion"],
        ["0x00020003", "2.0.3.0", "0x20000", "Acuity6.30.5+IDE5.10.1及之前"],
        ["0x00010d00", "1.13.0.0", "0x1001e", "Acuity6.21.1+IDE5.8.2及之前"],
        ["0x00010c05", "1.12.5.0", "0x1001d", "Acuity6.20.3+IDE5.8.1及之前"],
        ["0x00010a00", "1.10.0.0", "0x10016", "Acuity6.12.0+IDE5.7.2及之前"],
        ["0x00010800", "1.8.0.0", "0x10015", "Acuity6.9.3+IDE5.7.1及之前"],
        ["0x00010604", "1.6.4.0", "0x1000f", "Acuity6.0.17+IDE5.5.0及之前"],
    ]
