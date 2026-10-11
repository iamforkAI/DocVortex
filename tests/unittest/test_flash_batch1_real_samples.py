"""第一批修复的真实完整原件与公开入口回归，不依赖外接样本目录。"""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re
import sys

import numpy as np
import pytest
from pypdf import PdfReader

from docvortex import parse
from docvortex.document.pdf import PDFDocument
from docvortex.visualization import render_layout_pdf
from test_pdf_visualization import _page, _render_pixels

FIXTURES = Path(__file__).parent / "pdfs" / "flash_layout"


def _visible(value):
    """递归读取公共语义块的内容，不以布局框存在替代正文恢复。"""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_visible(item) for item in value)
    if isinstance(value, dict):
        return _visible(value.get("content", value.get("blocks", [])))
    return ""


def _with_unreadable_info(source: bytes) -> bytes:
    """仅增量追加不可读 Info 对象，保留真实深结构原件的目录、页面及原字节。"""
    reader = PdfReader(BytesIO(source))
    root = reader.trailer.raw_get("/Root")
    info_id = int(reader.trailer["/Size"])
    previous_xref = int(re.findall(rb"startxref\s+(\d+)", source)[-1])
    object_offset = len(source) + 1
    extra_object = f"\n{info_id} 0 obj\noo /Producer (unreadable) >>\nendobj\n".encode("ascii")
    xref_offset = len(source) + len(extra_object)
    extra_trailer = (
        f"xref\n{info_id} 1\n{object_offset:010d} 00000 n \n"
        f"trailer\n<< /Size {info_id + 1} /Root {root.idnum} {root.generation} R "
        f"/Info {info_id} 0 R /Prev {previous_xref} >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode("ascii")
    return source + extra_object + extra_trailer


def test_negative_scale_report_restores_every_physical_page_and_heading(tmp_path):
    """完整解析三十一页负字号原件，每页保留正文覆盖及加粗标题的真实锚点。"""
    source = FIXTURES / "negative_font_scale_report.pdf"
    assert sha256(source.read_bytes()).hexdigest() == "3b9a72ce8ac7fd0a6a39cf5dbbc8bc4b69532470cb0b83283b4d11aed09e46c5"
    result = parse(source, parse_mode="txt", keep_model_json=True)
    assert result.model_json is not None and len(result.model_json.pages) == 31
    pages = result.to_dict()["pages"]
    assert [page["page_idx"] for page in pages] == list(range(31))
    texts = [_visible(page) for page in pages]
    with PDFDocument(str(source)) as pdf:
        for index, text in enumerate(texts):
            source_count = sum(sum(char.isalnum() for char in item["char"]) for item in pdf.get_page_chars(index))
            assert sum(char.isalnum() for char in text) >= 0.9 * source_count, f"物理页 {index + 1} 正文覆盖不足"
    assert "迅速制定一揽子工作方案" in texts[1]
    assert "第一，在抗击疫情、服务发展中展现" in texts[1]
    assert "境外单位疫情防控工作" in texts[30]
    result.export(tmp_path / "report.html", output_format="html")
    assert "第一，在抗击疫情" in (tmp_path / "report.html").read_text()


def test_real_signature_widgets_keep_visible_appearance_when_no_boxes_are_added():
    """真实签章原件的字段目录和两页可见像素保留，PDFium 必须启用表单环境。"""
    source = (FIXTURES / "signature_widget_appearance.pdf").read_bytes()
    assert sha256(source).hexdigest() == "f470a5b5c7eda318f5d58aa6d8e5e1476b03f09a63967441819445816bc9e300"
    output = render_layout_pdf(source, [])
    original, copied = PdfReader(BytesIO(source)), PdfReader(BytesIO(output))
    assert copied.get_fields().keys() == original.get_fields().keys()
    assert any(field["/FT"] == "/Sig" for field in original.get_fields().values())
    with PDFDocument(source) as before, PDFDocument(output) as after:
        assert before.page_count == after.page_count == 2
        for index in range(2):
            first = before.render_page(index, scale=1).pil_image
            second = after.render_page(index, scale=1).pil_image
            try:
                assert np.array_equal(np.asarray(first), np.asarray(second))
            finally:
                first.close()
                second.close()


def test_real_outside_crop_keeps_all_pages_and_aligns_overlay():
    """整本真实异常页框原件不截页，首面标框按 PDFium 的有效显示区域定位。"""
    source = (FIXTURES / "outside_crop_ship_design.pdf").read_bytes()
    assert sha256(source).hexdigest() == "a086bf1a5cbad91654cd3c79d71ea70f9f77c9e39cde8a1089eb1e2ebded29bf"
    output = render_layout_pdf(source, [_page()])
    assert len(PdfReader(BytesIO(output)).pages) == 265
    pixels = _render_pixels(output)
    red, green, blue = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    mask = (red > 100) & (red < 220) & (green < 70) & (blue > 40) & (blue < 160)
    mask &= np.any(pixels != _render_pixels(source), axis=2)
    height, width = mask.shape
    mask[: round(height * 0.2) - 2] = False
    y, x = np.where(mask)
    np.testing.assert_allclose(
        [x.min(), y.min(), x.max(), y.max()], [width * 0.1, height * 0.2, width * 0.6, height * 0.4], atol=2
    )


def test_real_bad_info_keeps_five_pages_and_visible_content():
    """真实坏元数据 PDF 仍能画框，全原件无标框复制保留所有页面可见像素。"""
    source = (FIXTURES / "unreadable_info_metadata.pdf").read_bytes()
    assert sha256(source).hexdigest() == "022b681f6fed9cbcd46f9d263502b8ba85d27dea25be4332d522572731d1f6ba"
    with pytest.warns(RuntimeWarning, match="unreadable PDF /Info"):
        output = render_layout_pdf(source, [])
    with PDFDocument(source) as before, PDFDocument(output) as after:
        assert before.page_count == after.page_count == 5
        for index in range(5):
            first = before.render_page(index, scale=1).pil_image
            second = after.render_page(index, scale=1).pil_image
            try:
                assert np.array_equal(np.asarray(first), np.asarray(second))
            finally:
                first.close()
                second.close()


def test_real_deep_structure_catalog_keeps_all_pages_and_visible_content():
    """真实深结构树原件通过完整加载复制，保持全部页、目录和原生可见内容，不改递归上限。"""
    source = (FIXTURES / "deep_structure_catalog.pdf").read_bytes()
    assert sha256(source).hexdigest() == "6e0eccd9bbcbf0a91715a212477e133151e13bdeacf6f15df70461d9211a2042"
    limit = sys.getrecursionlimit()
    with pytest.warns(RuntimeWarning, match="recursive layout clone"):
        output = render_layout_pdf(source, [])
    assert sys.getrecursionlimit() == limit
    original, copied = PdfReader(BytesIO(source)), PdfReader(BytesIO(output))
    assert original.root_object.keys() == copied.root_object.keys()
    assert {"/StructTreeRoot", "/AcroForm", "/OCProperties"} <= copied.root_object.keys()
    assert original.get_fields().keys() == copied.get_fields().keys()
    assert len(original.pages) == len(copied.pages) == 328
    for first, second in zip(original.pages, copied.pages):
        assert tuple(first.mediabox) == tuple(second.mediabox)
        assert tuple(first.cropbox) == tuple(second.cropbox)
        assert first.rotation == second.rotation
    with PDFDocument(source) as before, PDFDocument(output) as after:
        for index in (0, 163, 327):
            first = before.render_page(index, scale=1).pil_image
            second = after.render_page(index, scale=1).pil_image
            try:
                assert np.array_equal(np.asarray(first), np.asarray(second))
            finally:
                first.close()
                second.close()


def test_deep_catalog_with_unreadable_info_keeps_catalog_pages_and_visible_content():
    """坏 Info 与深目录完整加载组合仍只省略元数据，不重读坏对象或丢失页面表单。"""
    path = FIXTURES / "deep_structure_catalog.pdf"
    source = path.read_bytes()
    source_sha256 = "6e0eccd9bbcbf0a91715a212477e133151e13bdeacf6f15df70461d9211a2042"
    assert sha256(source).hexdigest() == source_sha256
    broken = _with_unreadable_info(source)
    original = PdfReader(BytesIO(broken))
    assert len(original.pages) == 328
    limit = sys.getrecursionlimit()
    with pytest.warns(RuntimeWarning) as caught:
        output = render_layout_pdf(broken, [])
    messages = [str(item.message) for item in caught]
    assert any("unreadable PDF /Info" in message for message in messages)
    assert any("recursive layout clone" in message for message in messages)
    assert sys.getrecursionlimit() == limit
    copied = PdfReader(BytesIO(output))
    assert "/Info" not in copied.trailer
    assert copied.root_object.keys() == original.root_object.keys()
    assert {"/StructTreeRoot", "/AcroForm", "/OCProperties"} <= copied.root_object.keys()
    assert copied.get_fields().keys() == original.get_fields().keys()
    assert len(copied.pages) == len(original.pages) == 328
    for first, second in zip(original.pages, copied.pages):
        assert tuple(first.mediabox) == tuple(second.mediabox)
        assert tuple(first.cropbox) == tuple(second.cropbox)
        assert first.rotation == second.rotation
    with PDFDocument(broken) as before, PDFDocument(output) as after:
        for index in (0, 163, 327):
            first = before.render_page(index, scale=1).pil_image
            second = after.render_page(index, scale=1).pil_image
            try:
                assert np.array_equal(np.asarray(first), np.asarray(second))
            finally:
                first.close()
                second.close()
    assert sha256(path.read_bytes()).hexdigest() == source_sha256
