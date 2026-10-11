"""核验语料路径覆盖、内容去重与禁止覆盖既有清单的契约。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tools import flash_corpus_inventory as inventory


class _Document:
    """使用固定样本隔离 PDF 引擎，验证清单行为而非分类器内部规则。"""

    def __init__(self, data: bytes) -> None:
        """从测试字节确定页数与分类，并保留一个打开失败样本。"""
        if data == b"broken":
            raise ValueError("broken PDF")
        self.page_count = 150 if data == b"long" else 3
        self.kind = "ocr" if data == b"scan" else "txt"

    def __enter__(self):
        """返回当前测试文档，匹配公开上下文管理器入口。"""
        return self

    def __exit__(self, *args):
        """允许测试文档按正常生命周期退出。"""
        return None

    def classify(self) -> str:
        """返回样本预设的原生文本或扫描分类。"""
        return self.kind


def test_manifest_preserves_aliases_and_every_exclusion(tmp_path: Path, monkeypatch) -> None:
    """核验大小写扩展名、重复内容、伴生文件和失败路径均有唯一归属。"""
    files = {
        "short.pdf": b"short",
        "nested/copy.PDF": b"short",
        "long.PDF": b"long",
        "scan.pdf": b"scan",
        "broken.pdf": b"broken",
        "._short.pdf": b"sidecar",
        "ignored.txt": b"ignored",
    }
    for name, data in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    monkeypatch.setattr(inventory, "PDFDocument", _Document)
    monkeypatch.setattr(inventory, "source_fingerprint", lambda root: "frozen")
    monkeypatch.setattr(inventory, "capture_environment", lambda root: {"source_sha256": "frozen"})

    manifest = inventory.build_manifest(tmp_path)
    counts = manifest["counts"]
    assert counts["pdf_named_paths"] == 6
    assert counts["real_pdf_paths"] == 5
    assert counts["unique_pdf_documents"] == 4
    assert counts["duplicate_paths"] == 1
    assert counts["txt_documents"] == 2
    assert counts["txt_paths"] == 3
    assert counts["short_pages"] == 3
    assert counts["long_pages"] == 150
    assert counts["visual_pages"] == 103
    assert counts["ocr_documents"] == 1
    assert counts["failed_documents"] == 1
    short = next(item for item in manifest["documents"] if item["page_count"] == 3)
    digest = hashlib.sha256(b"short").hexdigest()
    assert short["name"] == f"d-{digest[:16]}"
    assert short["sha256"] == digest
    assert short["aliases"] == [str(tmp_path / "nested/copy.PDF"), str(tmp_path / "short.pdf")]
    assert {item["reason"] for item in manifest["excluded"]} == {"apple_double", "ocr", "pdf_error"}
    broken = next(item for item in manifest["excluded"] if item["reason"] == "pdf_error")
    assert broken["error"] == "ValueError: broken PDF"


def test_existing_manifest_is_not_overwritten_or_reclassified(tmp_path: Path, monkeypatch) -> None:
    """拒绝覆盖发生在分类之前，既有审阅产物保持原字节。"""
    output = tmp_path / "manifest.json"
    output.write_bytes(b"existing evidence")

    def unexpected_build(source_root):
        """在禁止覆盖场景中捕获不应启动的语料遍历。"""
        pytest.fail("existing manifest must prevent classification")

    monkeypatch.setattr(inventory, "build_manifest", unexpected_build)
    with pytest.raises(FileExistsError):
        inventory.write_manifest(tmp_path / "missing-source", output)
    assert output.read_bytes() == b"existing evidence"


def test_source_change_does_not_publish_manifest(tmp_path: Path, monkeypatch) -> None:
    """统计期间源码身份变化时中止，避免发布混合版本证据。"""
    source = tmp_path / "source"
    source.mkdir()
    (source / "short.pdf").write_bytes(b"short")
    monkeypatch.setattr(inventory, "PDFDocument", _Document)
    monkeypatch.setattr(inventory, "source_fingerprint", lambda root: "before")
    monkeypatch.setattr(inventory, "capture_environment", lambda root: {"source_sha256": "after"})
    output = tmp_path / "output/manifest.json"
    with pytest.raises(RuntimeError, match="源码变化"):
        inventory.write_manifest(source, output)
    assert not output.exists()
