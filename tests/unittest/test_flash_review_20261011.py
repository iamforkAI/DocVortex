"""用原页字符和路径证据验证十月十一日 Flash 容器、成员及阅读边界。"""

from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
import gzip
import json
from pathlib import Path
import re

from bs4 import BeautifulSoup
import pytest

from docvortex.document.pdf._document import PDFPathInfo
from docvortex.analyzers.native.pdf import pipeline
from docvortex.analyzers.native.pdf.models import _LineItem, _AxisLine, _PageSource, _PreparedPage
from docvortex.analyzers.native.pdf.graphics import _detect_strong_graphic_bboxes, _build_raster_image_blocks
from docvortex.analyzers.native.pdf.geometry import _bbox_overlap_in_smaller
from docvortex.analyzers.native.pdf.table_detection import _detect_table_candidates
from docvortex.analyzers.native.pdf.table_materialization import _materialize_table_blocks
from docvortex.analyzers.native.pdf.text_assembly.continuity import mark_document_reference_regions, group_reference_lines
from docvortex.analyzers.native.pdf.auxiliary_text import _classify_repeated_page_marginals


@lru_cache(maxsize=1)
def _fixtures():
    """加载带原件SHA256的全文原语，避免CI依赖用户下载目录。"""
    with gzip.open(Path(__file__).parents[1] / "fixtures/flash_review_20261011_sources.json.gz", "rt") as stream:
        return {item["name"]: item for item in json.load(stream)}


def _source(name, page):
    """重建独立可变页面；字体签名、路径和字框恢复为原协议的元组。"""
    data = deepcopy(_fixtures()[name]["pages"][page - 1])
    char_map = {char["char_idx"]: char for char in data["chars"]}
    lines = []
    for item in data["lines"]:
        if "char_indices" in item:
            item["chars"] = [char_map[index] for index in item.pop("char_indices")]
        for key in ("bbox", "source_bbox", "ink_bbox", "font_signature"):
            if item.get(key) is not None:
                item[key] = tuple(item[key])
        item["native_math_words"] = frozenset(item["native_math_words"])
        item["inline_math_regions"] = [tuple(box) for box in item["inline_math_regions"]]
        lines.append(_LineItem(**item))
    paths = [
        PDFPathInfo(
            **{
                **p,
                "bbox": tuple(p["bbox"]),
                "rectangle_bboxes": tuple(tuple(b) for b in p["rectangle_bboxes"]),
                "fill_rgba": tuple(p["fill_rgba"]) if p["fill_rgba"] else None,
            }
        )
        for p in data["path_infos"]
    ]
    return _PageSource(
        tuple(data["page_size"]),
        lines,
        data["chars"],
        [_AxisLine(tuple(item["bbox"]), item["width"], item["orientation"]) for item in data["drawing_lines"]],
        [tuple(box) for box in data["image_bboxes"]],
        [tuple(box) for box in data["signature_bboxes"]],
        [tuple(box) for box in data["form_bboxes"]],
        paths,
        data["page_index"],
    )


def _prepared(source):
    """构造只用于辅助分类断言的页面，保留完整原行而不提前认领。"""
    return _PreparedPage(source.page_size, source.lines, [], source.drawing_lines, [])


@pytest.mark.parametrize(
    "page,region", [(19, (65, 174, 549, 581)), (20, (64, 164, 550, 755)), (21, (65, 79, 376, 412)), (23, (65, 174, 548, 460))]
)
def test_native_table_fill_is_not_a_bar_chart(page, region):
    """多行数据表与页眉底纹不是柱图，不能用图形核心阻止表格认领。"""
    boxes = _detect_strong_graphic_bboxes(_source("guide", page))
    assert not any(_bbox_overlap_in_smaller(box, region) >= 0.6 for box in boxes)


@pytest.mark.parametrize(
    "page,anchors",
    [
        (4, ["简称", "定义", "A股"]),
        (11, ["注册制", "2014年"]),
        (37, ["具体条款", "市值及财务指标"]),
        (44, ["主板", "创业板、科创板", "北交所"]),
        (45, ["8", "财务", "其他"]),
    ],
)
def test_table_candidates_retain_edge_members(page, anchors):
    """原页表头、末行和跨行左列必须进入同一候选，而不是散落成正文。"""
    source = _source("guide", page)
    candidates = _detect_table_candidates(source)
    for anchor in anchors:
        members = [line.source_index for line in source.lines if anchor in line.text and line.bbox[1] > 70]
        assert members, anchor
        assert any(set(members) & set(candidate.line_indices) for candidate in candidates), (page, anchor)
    if page == 37:
        assert len(candidates) == 1


@pytest.mark.parametrize(
    "page,expected",
    [
        (44, ["招股文件", "发行人关于本次发行的申请及授权文件"]),
        (45, ["发行人律师关于本次发行的文件"]),
        (58, ["证券监管机构", "保荐人"]),
    ],
)
def test_table_cells_follow_physical_lines(page, expected):
    """短标签和长单元格按原页行序读取，不能穿插字列或产生假上下标。"""
    source = _source("guide", page)
    blocks, _, _ = _materialize_table_blocks(source, _detect_table_candidates(source))
    texts = [
        re.sub(r"\s+", "", cell.get_text())
        for block in blocks
        for cell in BeautifulSoup(block["content"], "html.parser").find_all("td")
    ]
    for value in expected:
        assert value in texts
    if page == 44:
        assert all(value in texts for value in ["主板", "创业板、科创板", "北交所"])


@pytest.mark.parametrize("page", [5, 27, 61])
def test_decorative_background_does_not_own_native_prose(page):
    """装饰底图前的多行原生目录/正文保留文字身份，不进入图片内容。"""
    source = _source("guide", page)
    blocks, claimed = _build_raster_image_blocks(source, [], set())
    body = {line.source_index for line in source.lines if 100 < line.bbox[1] < 760 and len(line.text) > 12}
    assert body and not body & claimed
    assert not any(len(block["content"]) > 100 for block in blocks)


def test_fullwidth_numbered_reference_region_preserves_all_entries():
    """字间空白的参考标题与全角编号建立连续参考区，44条文献边界仍唯一。"""
    pages = [_prepared(_source("landslide", page)) for page in range(1, 14)]
    mark_document_reference_regions(pages)
    starts = []
    for page in pages:
        group_reference_lines(page.remaining_lines, page)
        starts.extend(line for line in page.remaining_lines if line.reference_start)
    assert len(starts) == 44
    assert all(page.numbered_references for page in pages[9:12])
    assert not pages[12].reference_regions


def test_caption_start_is_not_a_repeated_footer():
    """相邻底部图题的递增编号不得成为页脚，真正重复版权文字继续识别。"""
    pages = [_prepared(_source("landslide", page)) for page in [4, 5]]
    for page in pages:
        pipeline._mark_native_caption_starts(page.remaining_lines)
    _classify_repeated_page_marginals(pages)
    captions = [line for page in pages for line in page.remaining_lines if re.match(r"^Fig[.]\s*[34]", line.text)]
    assert len(captions) == 2 and all(line.semantic_type != "footer" for line in captions)


@lru_cache(maxsize=1)
def _landslide_layout():
    """保留全文排版上下文验证最终组装，测试期待始终取自原页而非候选输出。"""
    pages = [pipeline._prepare_page_source(_source("landslide", page), page_index=page - 1) for page in range(1, 14)]
    profiles = pipeline._classify_document_text(pages)
    mark_document_reference_regions(pages)
    return [
        pipeline._finalize_prepared_page(
            page,
            index,
            canonical_body_profile=profiles.canonical_body,
            document_body_profile=profiles.body,
            document_title_profile=profiles.title,
        )
        for index, page in enumerate(pages)
    ]


def _text(block):
    """读取模型文字内容以断言语义成员，不把私有检测证据混入公共验收。"""
    content = block["content"]
    return (
        content
        if isinstance(content, str)
        else "".join(item.get("content", "") for item in content if isinstance(item.get("content"), str))
    )


@pytest.mark.parametrize(
    "page,anchor", [(1, "术（interferometric"), (2, "sistent scatterer"), (2, "涉测量（small"), (3, "括 2011")]
)
def test_mixed_script_regular_pitch_does_not_become_a_title(page, anchor):
    """混排正文的字体度量变化不能打断原页连续段落。"""
    blocks = [
        block for block in _landslide_layout()[page - 1] if re.sub(r"\s+", "", anchor) in re.sub(r"\s+", "", _text(block))
    ]
    assert blocks and all(block["type"] == "text" for block in blocks)


def test_inline_fraction_retains_prose_relationship():
    """解释段中的原生分式保留分子分母关系，显示公式仍独立输出。"""
    blocks = _landslide_layout()[3]
    assert any("4π/λ" in re.sub(r"\s+", "", _text(block)) and block["type"] == "text" for block in blocks)
    assert not any(block["type"] == "paragraph_title" and _text(block).strip() == "4π" for block in blocks)
    assert sum(block["type"] == "equation" for block in blocks) >= 2


def test_wrapped_section_and_translated_title_are_complete():
    """编号小节及译文题名按原页完整行组输出，续行不遗留为正文。"""
    titles = [_text(block) for block in _landslide_layout()[2] if block["type"] == "paragraph_title"]
    assert any(text.startswith("1.1") and "数据" in text for text in titles)
    assert any(text.startswith("1.2") and "数据" in text for text in titles)
    titles = [_text(block) for block in _landslide_layout()[12] if block["type"] == "doc_title"]
    assert len(titles) == 1 and "Early Detection" in titles[0] and "InSAR Technologies" in titles[0]


@pytest.mark.parametrize("page,first,last", [(1, "2018", "县波罗乡"), (5, "针对ALOS", "相干性"), (6, "从ALOS", "H28")])
def test_mixed_script_paragraph_lead_is_not_a_separate_paragraph(page, first, last):
    """日期和英文缩写首行的字体度量不同，仍须与同缩进后的正文组成一个物理自然段。"""
    matches = [block for block in _landslide_layout()[page - 1] if re.sub(r"\s+", "", _text(block)).startswith(first)]
    assert len(matches) == 1 and last in re.sub(r"\s+", "", _text(matches[0]))


@pytest.mark.parametrize(
    "page,region",
    [
        (9, (101, 186, 460, 529)),
        (24, (94, 228, 519, 412)),
        (34, (84, 98, 532, 253)),
        (41, (65, 252, 550, 778)),
        (47, (66, 314, 548, 600)),
        (50, (66, 208, 548, 504)),
        (53, (228, 132, 407, 715)),
    ],
)
def test_complete_native_diagram_paths_have_one_container(page, region):
    """矢量组件外框覆盖原页完整图体，不留下被截去的圆弧、箭头、图标和时间轴。"""
    boxes = _detect_strong_graphic_bboxes(_source("guide", page))
    assert any(
        _bbox_overlap_in_smaller(box, region) >= 0.98
        and (box[2] - box[0]) >= 0.98 * (region[2] - region[0])
        and (box[3] - box[1]) >= 0.98 * (region[3] - region[1])
        for box in boxes
    )


@lru_cache(maxsize=1)
def _guide_layout():
    """在全文上下文检查容器成员、独立自然段和表注父子身份。"""
    pages = [pipeline._prepare_page_source(_source("guide", page), page_index=page - 1) for page in range(1, 62)]
    profiles = pipeline._classify_document_text(pages)
    return [
        pipeline._finalize_prepared_page(
            page,
            index,
            canonical_body_profile=profiles.canonical_body,
            document_body_profile=profiles.body,
            document_title_profile=profiles.title,
        )
        for index, page in enumerate(pages)
    ]


@pytest.mark.parametrize(
    "page,heading", [(9, "1.主板"), (34, "（一）创业板板块定位"), (47, "1.受理和预先披露"), (50, "1.受理与补正")]
)
def test_neighboring_heading_is_not_part_of_diagram(page, heading):
    """外框完整与成员正确分别验收，图后小标题不能随图一起被裁入。"""
    compact = re.sub(r"\s+", "", heading)
    blocks = _guide_layout()[page - 1]
    assert any(block["type"] == "paragraph_title" and compact in re.sub(r"\s+", "", _text(block)) for block in blocks)
    assert not any(block["type"] == "image" and compact in re.sub(r"\s+", "", _text(block)) for block in blocks)


def test_three_adjacent_tables_do_not_consume_the_next_caption():
    """两行小表仍是独立表格，第二表题名不属于第一表的尾部。"""
    blocks = _guide_layout()[56]
    tables = [block for block in blocks if block["type"] == "table"]
    assert len(tables) == 3
    assert "深交所上市费" not in _text(tables[0])
    for market in ("上交所", "深交所", "北交所"):
        assert any(block["type"] == "caption" and market + "上市费" in _text(block) for block in blocks)


@pytest.mark.parametrize("page,prose", [(14, "2011年至今"), (55, "首先以2022年")])
def test_chart_source_note_does_not_consume_body_paragraph(page, prose):
    """图下小字号来源与后续正文的段界、类型及成员分别保留。"""
    blocks = _guide_layout()[page - 1]
    assert any(block["type"] == "text" and prose in re.sub(r"\s+", "", _text(block)) for block in blocks)
    notes = [block for block in blocks if block["type"] == "footnote" and "资料来源" in _text(block)]
    assert notes and all(prose not in re.sub(r"\s+", "", _text(block)) for block in notes)


def test_icon_timeline_is_complete_and_separate_from_prose():
    """八个矢量图标及列下文字进入一个时间轴，周围正文保持独立段落。"""
    blocks = _guide_layout()[6]
    images = [block for block in blocks if block["type"] == "image"]
    assert any(
        "统一监管" in re.sub(r"\s+", "", _text(block)) and "正式开市" in re.sub(r"\s+", "", _text(block)) for block in images
    )
    assert not any(block["type"] == "equation" for block in blocks)
    assert any(block["type"] == "text" and _text(block).startswith("截至目前") for block in blocks)


def _transform_source(source, scale, offset):
    """独立改变位置、字号及字体标识，验证规则只消费相对排版而不依赖文档身份。"""
    dx, dy = offset

    def box(values):
        """对四坐标字框和路径框应用一致的仿射变换。"""
        return tuple(value * scale + (dx if index % 2 == 0 else dy) for index, value in enumerate(values))

    source.page_size = (source.page_size[0] * scale + 2 * dx, source.page_size[1] * scale + 2 * dy)
    for char in source.chars:
        for key in ("bbox", "loose_bbox", "tight_bbox"):
            if char.get(key):
                char[key] = box(char[key])
        if char.get("origin"):
            char["origin"] = (char["origin"][0] * scale + dx, char["origin"][1] * scale + dy)
        char["font"]["name"] = "TEST+Unrelated" + char["font"]["name"].split("+")[-1]
    for line in source.lines:
        line.bbox = box(line.bbox)
        line.em_height *= scale
        line.effective_height *= scale
        if line.font_signature:
            line.font_signature = ("TEST+Unrelated" + line.font_signature[0].split("+")[-1], line.font_signature[1])
    source.path_infos = [
        replace(path, bbox=box(path.bbox), rectangle_bboxes=tuple(box(b) for b in path.rectangle_bboxes))
        for path in source.path_infos
    ]
    source.drawing_lines = [replace(rule, bbox=box(rule.bbox), width=rule.width * scale) for rule in source.drawing_lines]
    source.image_bboxes = [box(b) for b in source.image_bboxes]
    return source


@pytest.mark.parametrize("scale,offset", [(0.7, (23, 15)), (1.5, (80, 40))])
def test_tiled_grid_recovery_generalizes_with_position_and_scale(scale, offset):
    """同一物理单元格结构移动缩放后仍完整；移除列线则不能借底纹猜表格。"""
    from docvortex.analyzers.native.pdf.table_filled_grid import _detect_tiled_fill_table_candidates

    source = _transform_source(_source("guide", 37), scale, offset)
    candidates = _detect_tiled_fill_table_candidates(source, [])
    assert len(candidates) == 1
    expected = {line.source_index for line in source.lines if "具体条款" in line.text or "市值及财务指标" in line.text}
    assert expected <= candidates[0].line_indices
    source.drawing_lines = [rule for rule in source.drawing_lines if rule.orientation != "vertical"]
    assert not _detect_tiled_fill_table_candidates(source, [])


def test_native_photo_with_few_labels_is_preserved():
    """只有少量图中标签时保留大照片，不能以页面覆盖率代替装饰底图证据。"""
    source = _source("guide", 27)
    source.lines = source.lines[8:11]
    blocks, claimed = _build_raster_image_blocks(source, [], set())
    assert blocks and claimed


def test_reference_style_keeps_page_number_continuation_and_editor_note_separate():
    """方括号文献的卷页续行不是点号条目，编辑说明和装饰线不属于最后一条参考。"""
    values = [
        ("【参 考 文 献】", 70, 40),
        ("［1］ Alpha author. A different paper", 100, 40),
        ("20. Citation continuation after a wrapped page range", 115, 55),
        ("［2］ Beta author. Another different paper", 145, 40),
        ("［3］ Gamma author. A third different paper", 175, 40),
        ("［责任编辑：某编辑］", 210, 40),
        ("____________", 230, 40),
    ]
    lines = [
        _LineItem(text, (left, top, left + 190, top + 10), 0, index, effective_height=10, em_height=10)
        for index, (text, top, left) in enumerate(values)
    ]
    page = _PreparedPage((400, 600), lines, [], [], [])
    mark_document_reference_regions([page])
    group_reference_lines(lines, page)
    assert lines[0].semantic_type == "paragraph_title"
    assert sum(line.reference_start is True for line in lines) == 3
    assert lines[2].paragraph_group == lines[1].paragraph_group
    assert lines[-2].paragraph_group is None and lines[-1].paragraph_group is None


def test_author_year_reference_keeps_separate_native_accent_in_its_row():
    """作者年份文献中的独立重音字形保留行内归属，不套用编号文献的装饰线过滤。"""
    values = [
        ("First author. 2020. Paper", 40, 100),
        ("Wrapped continuation", 50, 115),
        ("´", 110, 115),
        ("Another continuation", 50, 130),
    ]
    lines = [
        _LineItem(text, (left, top, left + (3 if text == "´" else 150), top + 10), 0, index, effective_height=10, em_height=10)
        for index, (text, left, top) in enumerate(values)
    ]
    page = _PreparedPage((400, 600), lines, [], [], [])
    page.reference_regions = [(30, 90, 220, 160)]
    group_reference_lines(lines, page)
    assert lines[2].paragraph_group is not None
    assert lines[2].paragraph_group == lines[1].paragraph_group


@pytest.mark.parametrize("page,tail", [(16, "倍。"), (17, "20%。")])
def test_short_native_prose_tail_stays_in_its_paragraph(page, tail):
    """异常字框页的短数字或单字末行延续同字体正常行距的正文，不能变成孤立标题。"""
    blocks = [block for block in _guide_layout()[page - 1] if _text(block).endswith(tail)]
    assert blocks and all(block["type"] == "text" and len(_text(block)) > 60 for block in blocks)


def test_two_row_native_prose_keeps_its_regular_pitch_continuation():
    """两行正文同样提供满行、字体与行距证据，第二行不因短于首行而变成标题。"""
    blocks = [block for block in _guide_layout()[28] if "2019年" in _text(block)]
    assert any(block["type"] == "text" and "上市条件也有所不同" in _text(block) for block in blocks)


def test_continued_native_table_keeps_all_unfilled_column_headers():
    """续表列名的实际基线允许小幅字体偏移，但必须完整进入表格首行。"""
    tables = [block for block in _guide_layout()[44] if block["type"] == "table"]
    assert len(tables) == 1
    assert all(value in _text(tables[0]) for value in ("主板", "创业板", "北交所"))
