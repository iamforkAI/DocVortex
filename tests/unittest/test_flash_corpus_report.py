"""验证未审页不会被汇总为通过，且视觉问题必须有实际证据记录。"""

import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "flash_corpus_report", Path(__file__).parents[2] / "tools/flash_corpus_report.py"
)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def _write(path, value):
    """构造独立临时审阅记录，避免引用真实语料或既有报告。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _run(tmp_path):
    """建立两页已渲染但尚未看图的文档，并提供独立证据文件。"""
    _write(tmp_path / "manifest.json", {"documents": [
        {"name": "d-one", "path": "/input/a.pdf", "sha256": "a" * 64, "page_count": 2}
    ]})
    folder = tmp_path / "documents/d-one"
    _write(folder / "status.json", {"stages": {key: "succeeded" for key in ("parse", "export", "overlay", "render")}})
    _write(folder / "page-index.json", {"review_pages": [1, 2]})
    for name in ("source-0001.png", "layout-0001.png"):
        (folder / name).write_bytes(b"test evidence")
    return tmp_path


def _vote():
    """返回单页人工裁决，与渲染状态分离。"""
    return {"reviewer": "test-reviewer", "pages": [{"document_id": "d-one", "page": 1,
            "status": "no_obvious_issue", "evidence": ["documents/d-one/source-0001.png",
                                                        "documents/d-one/layout-0001.png"]}]}


def test_render_success_does_not_imply_review_success(tmp_path):
    """已生成图像的两页仍需真实视觉裁决，单页裁决不得放大覆盖量。"""
    root = _run(tmp_path)
    initial = REPORT.aggregate(root)
    assert initial["reviewed_selected_pages"] == 0
    assert initial["selected_page_status"] == {"pending": 2}
    _write(root / "reviews/one.json", _vote())
    actual = REPORT.aggregate(root)
    assert actual["reviewed_selected_pages"] == 1
    assert actual["selected_page_status"] == {"no_obvious_issue": 1, "pending": 1}
    assert (root / "index.html").read_text(encoding="utf-8") == (root / "overview.html").read_text(encoding="utf-8")
    assert "实际已审 1 / 2" in (root / "index.html").read_text(encoding="utf-8")


def test_conflicting_votes_remain_ambiguous(tmp_path):
    """两个 agent 结论冲突时，不得按最新文件自动宣布无问题。"""
    root = _run(tmp_path)
    first = _vote()
    second = _vote()
    second["reviewer"] = "second-reviewer"
    second["pages"][0]["status"] = "issues"
    _write(root / "reviews/one.json", first)
    _write(root / "reviews/two.json", second)
    summary = REPORT.aggregate(root)
    assert summary["selected_page_status"]["ambiguous"] == 1


def test_review_without_both_images_is_rejected(tmp_path):
    """禁止仅列页号或缺失原页图像就计入已审覆盖。"""
    root = _run(tmp_path)
    data = _vote()
    data["pages"][0]["evidence"] = ["documents/d-one/layout-0001.png"]
    _write(root / "reviews/one.json", data)
    with pytest.raises(ValueError, match="Source and layout"):
        REPORT.aggregate(root)


def test_invalid_sampling_count_is_rejected(tmp_path):
    """缺选一页不能把较短文档冒充全页审阅计划。"""
    root = _run(tmp_path)
    _write(root / "documents/d-one/page-index.json", {"review_pages": [1]})
    with pytest.raises(ValueError, match="selected page count"):
        REPORT.aggregate(root)


def test_finding_canonicalizes_original_indices_and_keeps_source_identity(tmp_path):
    """兼容独立记录字段并从真实块索引补类型，不能把原编号误作数组下标。"""
    root = _run(tmp_path)
    _write(root / "documents/d-one/page-index.json", {"review_pages": [1, 2], "pages": [
        {"page": 1, "blocks": [{"index": 17, "type": "text", "tree_path": "blocks.0", "text_start": "真实源块锚点"}]}]})
    data = _vote()
    data["pages"][0]["status"] = "issues"
    data["findings"] = [{"id": "one", "document_id": "d-one", "page": 1, "category": "text",
                         "status": "confirmed", "priority": "P1", "indices": [17], "bbox": [0, 0, 1, 1],
                         "observed": "真实问题", "expected": "保留原文", "evidence": data["pages"][0]["evidence"]}]
    _write(root / "reviews/one.json", data)
    summary = REPORT.aggregate(root)
    finding = json.loads((root / "findings.json").read_text(encoding="utf-8"))[0]
    assert finding["block_indices"] == [17]
    assert finding["block_refs"][0]["type"] == "text"
    assert finding["severity"] == "P1"
    assert finding["source_sha256"] == "a" * 64
    assert finding["anchors"] == {"block-17": "真实源块锚点"}
    assert finding["block_reference_status"] == "matched"
    assert "REPAIR_PLAN.md" in (root / "overview.html").read_text(encoding="utf-8")
    assert summary["confirmed_issue_documents"] == summary["confirmed_issue_pages"] == 1


def test_retry_history_backend_and_gallery_remain_auditable(tmp_path):
    """重试成功后仍公开首次失败、实际后端和逐页裁决，重复发布不丢导航。"""
    root = _run(tmp_path)
    folder = root / "documents/d-one"
    state = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    state["backend"] = {"backend": "rust", "protocol": 34, "extension_sha256": "b" * 64}
    _write(folder / "status.json", state)
    _write(folder / "history/attempt-0001/status.json", {
        "status": "failed", "errors": ["native cleanup aborted"], "stages": {"parse": "succeeded"}})
    (folder / "gallery.html").write_text(
        '<p><a href="../../overview.html">总览</a> 未审</p>'
        '<section id="page-1"><h2>页 1</h2></section>'
        '<section id="page-2"><h2>页 2</h2></section>', encoding="utf-8")
    _write(root / "reviews/one.json", _vote())
    for _ in range(2):
        summary = REPORT.aggregate(root)
    assert summary["parsed_pages"] == 2
    assert summary["actual_backends"][0]["documents"] == 1
    assert summary["previous_failed_attempts"][0]["errors"] == ["native cleanup aborted"]
    assert summary["visual_review_complete"] is False
    from bs4 import BeautifulSoup

    soup = BeautifulSoup((folder / "gallery.html").read_text(encoding="utf-8"), "html.parser")
    assert soup.find("a")["href"] == "../../overview.html"
    assert soup.find("section", id="page-1")["data-verdict"] == "no_obvious_issue"
    assert soup.find("section", id="page-2")["data-verdict"] == "pending"
    assert len(soup.find_all("p", attrs={"data-review-note": True})) == 2
    published = json.loads((folder / "review.json").read_text(encoding="utf-8"))
    assert published["pages"][0]["review"]["votes"][0]["reviewer"] == "test-reviewer"


def test_resolved_native_boundary_is_not_an_unresolved_finding(tmp_path):
    """复核为转曲能力边界后保留初始观察，但不再计为待确认布局缺陷。"""
    root = _run(tmp_path)
    data = _vote()
    data["pages"][0]["status"] = "out_of_scope"
    data["findings"] = [{"id": "boundary", "document_id": "d-one", "page": 1,
                         "category": "table", "status": "ambiguous", "resolution_status": "out_of_scope",
                         "initial_status": "ambiguous", "bbox": [0, 0, 1, 1],
                         "observed": "表格文字为纯转曲路径", "expected": "不要求原生路径执行OCR",
                         "evidence": data["pages"][0]["evidence"]}]
    _write(root / "reviews/boundary.json", data)
    summary = REPORT.aggregate(root)
    finding = json.loads((root / "findings.json").read_text(encoding="utf-8"))[0]
    assert summary["ambiguous_findings"] == 0
    assert summary["out_of_scope_findings"] == 1
    assert finding["status"] == "out_of_scope"
    assert finding["initial_status"] == "ambiguous"


def test_neighbor_review_is_visible_in_gallery_without_changing_sampling(tmp_path):
    """补审邻页独立加入图库和记录，但不能改变一百页抽样基线。"""
    root = _run(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["documents"][0]["page_count"] = 101
    _write(root / "manifest.json", manifest)
    folder = root / "documents/d-one"
    _write(folder / "page-index.json", {"review_pages": list(range(1, 101))})
    (folder / "gallery.html").write_text('<title>ID</title><h1>ID</h1><p><a href="document.html">HTML</a></p>', encoding="utf-8")
    for stem in ("source", "layout"):
        (folder / f"{stem}-0101.png").write_bytes(b"extra evidence")
    data = _vote()
    data["pages"][0].update(page=101, evidence=["documents/d-one/source-0101.png", "documents/d-one/layout-0101.png"])
    _write(root / "reviews/adjacent.json", data)
    summary = REPORT.aggregate(root)
    assert summary["extra_reviewed_pages"] == 1
    assert summary["planned_visual_pages"] == 100
    assert summary["reviewed_selected_pages"] == 0
    from bs4 import BeautifulSoup

    soup = BeautifulSoup((folder / "gallery.html").read_text(encoding="utf-8"), "html.parser")
    assert soup.find("h1").text == "a.pdf"
    extra = soup.find("section", id="page-101")
    assert extra["data-verdict"] == "no_obvious_issue"
    assert {image["src"] for image in extra.find_all("img")} == {"source-0101.png", "layout-0101.png"}
