"""验证 OCR 移除后的公开错误、离线边界和 PDF 资源生命周期。"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pytest
from click.testing import CliRunner
from reportlab.pdfgen.canvas import Canvas

import docvortex
from docvortex import api
from docvortex.analyzers.native import PdfModel
from docvortex.cli import main
from docvortex.document.pdf import PDFDocument
from docvortex.errors import DocumentError, InvalidRequestError

OCR_MESSAGE = (
    "PDF OCR is not supported by DocVortex. Please download and install MinerU to process this document: "
    "https://github.com/opendatalab/MinerU"
)


def _pdf_bytes(pages: int = 1) -> bytes:
    """生成有足够原生文字的 PDF，支持抽页和真实文本解析回归。"""
    output = BytesIO()
    canvas = Canvas(output)
    for page in range(pages):
        for row in range(30):
            canvas.drawString(30, 800 - row * 20, f"Page {page + 1}, native document line {row}.")
        canvas.showPage()
    canvas.save()
    return output.getvalue()


@pytest.fixture
def offline_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """隔离模型缓存目录并阻止网络连接，暴露拒绝路径中的隐式副作用。"""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", Mock(return_value=home))
    monkeypatch.setattr("socket.socket.connect", Mock(side_effect=AssertionError("Network forbidden")))
    monkeypatch.setattr("socket.create_connection", Mock(side_effect=AssertionError("Network forbidden")))
    return home


@pytest.mark.parametrize("entrypoint", ["analyze", "parse", "convert"])
@pytest.mark.parametrize("mode", ["auto", "ocr"])
def test_public_api_rejects_ocr_before_analysis(
    entrypoint: str, mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offline_home: Path
) -> None:
    """三个公开入口统一拒绝 OCR，且拒绝发生在几何提取、原生分析和写出之前。"""
    classify = Mock(return_value="ocr") if mode == "auto" else Mock(side_effect=AssertionError("Classification forbidden"))
    monkeypatch.setattr(PDFDocument, "classify", classify)
    monkeypatch.setattr("docvortex.document.pdf.layout.extract_layout_geometry", Mock(side_effect=AssertionError("Geometry")))
    monkeypatch.setattr(PdfModel, "predict", Mock(side_effect=AssertionError("Native analysis")))
    output = tmp_path / "output" / "result.md"
    arguments = (_pdf_bytes(), output) if entrypoint == "convert" else (_pdf_bytes(),)
    with pytest.raises(DocumentError) as captured:
        getattr(docvortex, entrypoint)(*arguments, file_suffix="pdf", parse_mode=mode)
    assert captured.value.code == "pdf_ocr_unsupported"
    assert str(captured.value) == OCR_MESSAGE
    assert classify.call_count == (mode == "auto")
    assert not output.parent.exists()
    assert not list(offline_home.iterdir())


@pytest.mark.parametrize("mode", ["auto", "ocr"])
@pytest.mark.parametrize("existing_output", [False, True])
def test_cli_rejects_ocr_without_writing(
    mode: str, existing_output: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offline_home: Path
) -> None:
    """CLI 非零退出并显示完整指引，覆盖模式也不能破坏已有文件。"""
    monkeypatch.setattr(PDFDocument, "classify", Mock(return_value="ocr"))
    source = tmp_path / "source.pdf"
    source.write_bytes(_pdf_bytes())
    output = tmp_path / "output" / "result.md"
    if existing_output:
        output.parent.mkdir()
        output.write_text("existing result", encoding="utf-8")
    result = CliRunner().invoke(main, ["convert", str(source), "-o", str(output), "--parse-mode", mode, "--overwrite"])
    assert result.exit_code == 1
    assert OCR_MESSAGE in result.output
    assert "Traceback" not in result.output
    if existing_output:
        assert output.read_text(encoding="utf-8") == "existing result"
        assert list(output.parent.iterdir()) == [output]
    else:
        assert not output.parent.exists()
    assert not list(offline_home.iterdir())


@pytest.mark.parametrize("mode", ["auto", "ocr"])
@pytest.mark.parametrize("caller_owned", [False, True])
@pytest.mark.parametrize("page_range", ["", "2"])
def test_ocr_rejection_preserves_document_ownership(
    mode: str, caller_owned: bool, page_range: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """异常释放引擎和抽页副本，调用方原文仍可用，分类只接收选定页面。"""
    prepared_sources = []
    original_prepare = api.prepare_source

    def prepare(*args, **kwargs):
        """保留真实输入准备结果，以检查异常后的文档关闭状态。"""
        prepared = original_prepare(*args, **kwargs)
        prepared_sources.append(prepared)
        return prepared

    def classify(document: PDFDocument) -> str:
        """检查自动分类看到抽页后的文档，并判定为需要 OCR。"""
        assert mode == "auto"
        assert document.page_count == (1 if page_range else 3)
        return "ocr"

    monkeypatch.setattr(api, "prepare_source", prepare)
    monkeypatch.setattr(PDFDocument, "classify", classify)
    payload = _pdf_bytes(3)
    with PDFDocument(payload) as original:
        with pytest.raises(DocumentError, match="MinerU"):
            docvortex.analyze(original if caller_owned else payload, file_suffix="pdf", page_range=page_range, parse_mode=mode)
        prepared = prepared_sources[0]
        if page_range:
            assert prepared.page_index_map == [1]
        assert prepared.owns_document == (not caller_owned or bool(page_range))
        assert (prepared.document._pdf_doc_opened is None) == prepared.owns_document
        assert original.page_count == 3
        assert "Page 1" in "".join(char["char"] for char in original.get_page_chars(0))


def test_forced_txt_and_low_level_native_model_skip_classification(monkeypatch: pytest.MonkeyPatch, offline_home: Path) -> None:
    """显式 txt 和底层原生入口均不分类、不下载，继续产出真实文字。"""
    monkeypatch.setattr(PDFDocument, "classify", Mock(side_effect=AssertionError("Classification forbidden")))
    payload = _pdf_bytes()
    result = docvortex.parse(payload, file_suffix="pdf", parse_mode="txt")
    assert "native document line" in result.middle_json.to_json()
    with PDFDocument(payload) as document:
        assert PdfModel().predict(document)
    assert not list(offline_home.iterdir())


def test_auto_native_selection_preserves_page_map(offline_home: Path) -> None:
    """真实原生 PDF 的自动分类和抽页映射继续贯穿模型与最终结果。"""
    result = docvortex.parse(_pdf_bytes(3), file_suffix="pdf", page_range="2", keep_model_json=True)
    assert result.model_json.page_index_map == [1]
    assert [page.page_idx for page in result.middle_json.pages] == [1]
    assert "Page 2" in result.middle_json.to_json()
    assert "Page 1" not in result.middle_json.to_json()
    assert not list(offline_home.iterdir())


def test_real_scan_remains_classifiable_but_cannot_be_parsed(offline_home: Path) -> None:
    """扫描样本仍可独立分类和读取元数据，默认解析返回明确的能力错误。"""
    source = Path(__file__).resolve().parents[1] / "demo/pdfs/small_ocr.pdf"
    result = CliRunner().invoke(main, ["classify", str(source)])
    assert result.exit_code == 0
    assert result.stdout.strip() == "ocr"
    assert docvortex.extract_metadata(source).metadata.file_suffix == "pdf"
    with pytest.raises(DocumentError) as captured:
        docvortex.parse(source)
    assert captured.value.code == "pdf_ocr_unsupported"
    assert str(captured.value) == OCR_MESSAGE
    assert not list(offline_home.iterdir())


@pytest.mark.parametrize("mode", ["txt", "ocr", "invalid"])
def test_non_pdf_explicit_modes_still_invalid(mode: str) -> None:
    """非 PDF 继续使用既有参数错误，默认模式仍正常解析。"""
    with pytest.raises(InvalidRequestError) as captured:
        docvortex.analyze(b"<p>Hello</p>", file_suffix="html", parse_mode=mode)
    assert captured.value.code == "parse_mode_invalid"
    assert captured.value.param == "parse_mode"
    assert docvortex.analyze(b"<p>Hello</p>", file_suffix="html").model_json.pages


def test_invalid_pdf_mode_still_invalid() -> None:
    """非法 PDF 模式保留参数错误，不能被误报为 OCR 能力错误。"""
    with pytest.raises(InvalidRequestError) as captured:
        docvortex.analyze(_pdf_bytes(), file_suffix="pdf", parse_mode="invalid")
    assert captured.value.code == "parse_mode_invalid"
    assert captured.value.param == "parse_mode"
