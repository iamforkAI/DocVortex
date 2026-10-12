"""连字展开空格的真实 PDF、几何反例和只读性回归。"""

from copy import deepcopy
from pathlib import Path
import math
from typing import Any

import pytest

from docvortex.analyzers.pdf import join_tight_text
from docvortex.document.pdf import PDFDocument

SAMPLE = Path(__file__).parent / "fixtures/pdf_text_5635_5636/source.pdf"


def _ligature_chars(text: str = "fi") -> list[dict[str, Any]]:
    """构造同一字形展开、字形内部空格和紧邻下一字母的真实记录。"""
    values = []
    for index, value in enumerate(f"{text} n"):
        box = (10.0, 10.0, 15.0, 20.0)
        origin = (10.0, 20.0)
        if value == " ":
            box, origin = (12.0, 19.99, 15.0, 20.0), (12.0, 20.0)
        elif index == len(text) + 1:
            box, origin = (15.3, 10.0, 20.0, 20.0), (15.2, 20.0)
        values.append(
            {
                "char": value,
                "char_idx": index,
                "bbox": box,
                "tight_bbox": box,
                "origin": origin,
                "writing_angle": 0.0,
                "text_object_id": 1,
                "font": {"name": "Fixture", "size": 1.0, "flags": 0},
                "is_generated": False,
            }
        )
    return values


@pytest.mark.parametrize("text", ["fi", "fl", "ff", "ffi", "ffl", "ft", "st"])
def test_overlapping_ligature_space_is_suppressed(text: str) -> None:
    """同一连字的展开记录允许移除真实编码但没有视觉词界的空格。"""
    chars = _ligature_chars(text)
    original = deepcopy(chars)
    assert join_tight_text(chars) == f"{text}n"
    assert chars == original


@pytest.mark.parametrize(
    "case",
    ["word_gap", "separate_glyphs", "different_object", "different_baseline", "missing_origin", "source_gap", "not_ligature"],
)
def test_real_or_unproven_ligature_spaces_survive(case: str) -> None:
    """真实词界、不同字形、缺失来源或几何证据不能被当作连字内部空格删除。"""
    chars = _ligature_chars("ab" if case == "not_ligature" else "fi")
    if case == "word_gap":
        chars[-1].update(tight_bbox=(18.0, 10.0, 23.0, 20.0), origin=(18.0, 20.0))
    elif case == "separate_glyphs":
        chars[0].update(tight_bbox=(5.0, 10.0, 9.0, 20.0), origin=(5.0, 20.0))
    elif case == "different_object":
        chars[1]["text_object_id"] = 2
    elif case == "different_baseline":
        chars[-1]["origin"] = (15.2, 30.0)
    elif case == "missing_origin":
        chars[0].pop("origin")
    elif case == "source_gap":
        chars[0]["char_idx"] = -2
    assert " " in join_tight_text(chars)


def test_real_pdf_ligature_spaces_and_source_records() -> None:
    """三页原件的五个 financial 均恢复，PDF 字符记录仍保留原始空格。"""
    with PDFDocument(SAMPLE.read_bytes()) as document:
        pages = [document[index].get_chars_with_geometry() for index in range(document.page_count)]
    source = "".join(char["char"] for page in pages for char in page.chars)
    before = [repr(page.chars) for page in pages]
    text = "".join(join_tight_text(page.chars, tight_bboxes=page.tight_bboxes, origins=page.origins) for page in pages)
    assert source.count("fi nancial") == 5
    assert text.count("financial") == 5
    assert "fi nancial" not in text
    assert [repr(page.chars) for page in pages] == before


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_rotated_ligature_and_geometry_sidecars(angle: int) -> None:
    """四个标准方向及外置几何采用相同的连字判定，缺失 tight 框时不猜测。"""
    from docvortex.analyzers.pdf import is_ligature_expansion_space
    from docvortex.document.pdf.text.spacing import _local_box

    chars = _ligature_chars()
    boxes, origins = {}, {}
    for char in chars:
        index = char["char_idx"]
        boxes[index] = _local_box(char.pop("tight_bbox"), (-angle) % 360)
        point = char.pop("origin")
        origins[index] = _local_box((*point, *point), (-angle) % 360)[:2]
        char["writing_angle"] = math.radians(angle)
    assert is_ligature_expansion_space(chars, 2, tight_bboxes=boxes, origins=origins)
    assert not is_ligature_expansion_space(chars, 2, origins=origins)


@pytest.mark.parametrize("kind", ["text", "code", "equation"])
def test_shared_materialization_keeps_code_and_formula_spaces(kind: str) -> None:
    """共享几何证据仅修改自然语言，不将代码或公式中的真实空格当作词界损坏。"""
    from docvortex.analyzers.native.pdf.inline.spacing import apply_spacing_lines, prepare_spacing_lines
    from docvortex.analyzers.native.pdf.models import _LineItem

    chars = _ligature_chars()
    line = _LineItem(text="fi n", bbox=(10.0, 10.0, 20.0, 20.0), angle=0, source_index=0, chars=chars)
    blocks = [{"type": kind, "bbox": (0.0, 0.0, 50.0, 50.0), "content": "fi n"}]
    evidence = prepare_spacing_lines([line])
    apply_spacing_lines(blocks, evidence, (100.0, 100.0))
    apply_spacing_lines(blocks, evidence, (100.0, 100.0))
    assert blocks[0]["content"] == ("fin" if kind == "text" else "fi n")


def test_real_flash_native_document_preserves_first_letter() -> None:
    """DocVortex 自己的原生分析与后处理同时恢复连字词语并保留原有首字和断词。"""
    import json
    from _native_test_utils import analyze_native_test_document

    middle, model = analyze_native_test_document(SAMPLE.read_bytes(), file_suffix="pdf")
    for document in [middle, model]:
        text = json.dumps(document.to_dict(), ensure_ascii=False)
        assert "This book is a practical" in text
        assert text.count("financial") == 5
        assert "fi nancial" not in text and "mo mentum" not in text
