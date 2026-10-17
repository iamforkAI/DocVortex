"""验证 Flash 批次审阅的覆盖边界、物理页身份及原生进程失败隔离。"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import review_flash_corpus as corpus


def test_import_preserves_package_search_path() -> None:
    """导入审阅工具不能让后续 spawn 渲染进程绕过已安装的原生 wheel。"""
    subprocess.run(
        [sys.executable, "-c", "import sys; before=list(sys.path); import tools.review_flash_corpus; assert sys.path == before"],
        cwd=Path(__file__).parents[2],
        check=True,
    )


@pytest.mark.parametrize("count", [1, 100, 101, 1017])
def test_selection_has_exact_deterministic_physical_coverage(count):
    """边界页数和千页文档都保持准确的预算、首尾覆盖和稳定选择。"""
    risks = [{"page": page, "risk": 0} for page in range(1, count + 1)]
    risks[count // 2]["risk"] = 1000
    selected = corpus.select_review_pages(count, risks)
    assert selected == corpus.select_review_pages(count, list(reversed(risks)))
    assert len(selected) == min(100, count)
    assert selected == sorted(set(selected))
    assert all(1 <= physical <= count for physical in selected)
    assert risks[count // 2]["page"] in selected
    if count > 100:
        assert set(range(1, 6)) | set(range(count - 4, count + 1)) <= set(selected)


def test_duplicate_and_invalid_risk_records_do_not_expand_page_budget():
    """重复风险记录和越界页不能造成重复审阅、超出预算或虚构物理页。"""
    risks = [{"page": 42, "risk": 10}] * 30 + [{"page": 0, "risk": 999}, {"page": 5000, "risk": 999}]
    selected = corpus.select_review_pages(1017, risks)
    assert len(selected) == 100
    assert 42 in selected
    assert 0 not in selected and 5000 not in selected


class _SourcePages:
    """提供独立于解析输出顺序的物理页元信息。"""

    page_count = 3

    def page_char_count(self, index):
        """返回按源物理页编号绑定的字符数。"""
        return [18, 0, 30][index]

    def page_rotation(self, index):
        """返回按源物理页编号绑定的旋转。"""
        return [0, 90, 0][index]


def test_page_index_preserves_physical_pages_nested_paths_and_original_indices():
    """输出顺序变化、跳号和父子重复号不能改写证据页号或造成假重叠。"""
    box = [0.1, 0.1, 0.7, 0.5]
    child = {"type": "text", "index": 7, "bbox": box, "content": [{"type": "text", "content": "实际正文"}]}
    parent = {"type": "list", "index": 7, "bbox": box, "content": [child]}
    middle = {
        "pages": [
            {"page_idx": 2, "blocks": [child]},
            {"page_idx": 0, "blocks": [parent]},
            {"page_idx": 1, "blocks": [child]},
        ]
    }
    original = deepcopy(middle)
    result = corpus.build_page_index({"name": "fixture", "sha256": "a" * 64}, middle, _SourcePages(), "b" * 64)
    assert middle == original
    assert [(page["page"], page["page_idx"], page["native_char_count"], page["rotation"]) for page in result["pages"]] == [
        (1, 0, 18, 0),
        (2, 1, 0, 90),
        (3, 2, 30, 0),
    ]
    blocks = result["pages"][0]["blocks"]
    assert [block["index"] for block in blocks] == [7, 7]
    assert [block["tree_path"] for block in blocks] == ["blocks.0", "blocks.0.content.0"]
    assert blocks[1]["parent"] == {"tree_path": "blocks.0", "type": "list", "index": 7}
    assert blocks[1]["text_start"] == "实际正文"
    assert result["pages"][0]["risk_evidence"]["overlap_pairs"] == 0
    assert result["pages"][1]["risk_evidence"]["mixed_zero_native_chars"] is True


@pytest.mark.parametrize("indices", [[0, 0, 2], [0, 1, 4]])
def test_full_parse_index_rejects_duplicate_or_missing_physical_pages(indices):
    """完整文档审阅不能把重复页或缺页当成有效覆盖。"""
    middle = {"pages": [{"page_idx": index, "blocks": []} for index in indices]}
    with pytest.raises(ValueError, match="page_idx"):
        corpus.build_page_index({"name": "fixture", "sha256": "a" * 64}, middle, _SourcePages(), "b" * 64)


def test_empty_public_page_without_blocks_is_preserved():
    """公共协议省略空 blocks 时仍保留物理页，不能中止含空白页的完整审阅。"""
    middle = {"pages": [{"page_idx": 0, "blocks": []}, {"page_idx": 1}, {"page_idx": 2, "blocks": []}]}
    result = corpus.build_page_index({"name": "fixture", "sha256": "a" * 64}, middle, _SourcePages(), "b" * 64)
    assert result["review_pages"] == [1, 2, 3]
    assert result["pages"][1]["blocks"] == []
    assert result["pages"][1]["native_char_count"] == 0


def test_replay_keeps_frozen_review_pages_instead_of_resampling():
    """修复后的风险排序变化不能替换已冻结的对照审阅页。"""
    middle = {"pages": [{"page_idx": index, "blocks": []} for index in range(3)]}
    document = {"name": "fixture", "sha256": "a" * 64, "review_pages": [1, 3]}
    result = corpus.build_page_index(document, middle, _SourcePages(), "b" * 64)
    assert result["review_pages"] == [1, 3]
    assert result["selection"] == "frozen"
    assert [page["selected"] for page in result["pages"]] == [True, False, True]


@pytest.mark.parametrize("selected", [[], [3, 1], [1, 1], [0], [4], [True], "all"])
def test_replay_rejects_invalid_frozen_review_pages(selected):
    """固定审阅页不能重复、乱序、越界或使用布尔值代替物理页号。"""
    middle = {"pages": [{"page_idx": index, "blocks": []} for index in range(3)]}
    document = {"name": "fixture", "sha256": "a" * 64, "review_pages": selected}
    with pytest.raises(ValueError, match="固定审阅页"):
        corpus.build_page_index(document, middle, _SourcePages(), "b" * 64)


def _manifest(tmp_path, names):
    """创建只供调度测试使用的源字节和可核验清单，不替代 PDF 解析验收。"""
    documents = []
    for name in names:
        source = tmp_path / f"{name}.pdf"
        source.write_bytes(name.encode())
        documents.append(
            {
                "name": name,
                "path": str(source),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "aliases": [str(source)],
                "page_count": 1,
            }
        )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"documents": documents}))
    return path


def test_native_worker_exit_is_saved_and_does_not_stop_later_documents(tmp_path, monkeypatch):
    """用真实独立进程异常退出验证失败隔离，后续文档仍完成且保持视觉待审。"""
    manifest = _manifest(tmp_path, ["crash", "success", "later"])
    monkeypatch.setattr(corpus, "fingerprint", lambda: "b" * 64)
    monkeypatch.setattr(corpus, "_environment", lambda: {})

    def command(manifest, output, name, code):
        """失败进程模拟原生终止，成功进程只提交调度状态。"""
        script = (
            "import json,os,sys;from pathlib import Path;"
            "folder=Path(sys.argv[1]);name=sys.argv[2];"
            "os._exit(73) if name=='crash' else None;"
            "state=json.loads((folder/'status.json').read_text(encoding='utf-8'));"
            "state.update(status='succeeded',visual_review='pending',review_pages=[1]);"
            "state['stages']={key:'succeeded' for key in state['stages']};"
            "(folder/'status.json').write_text(json.dumps(state))"
        )
        return [sys.executable, "-c", script, str(output / "documents" / name), name]

    monkeypatch.setattr(corpus, "_worker_command", command)
    output = tmp_path / "run"
    report = corpus.run_batch(manifest, output, workers=1, timeout=10)
    states = {document["document_id"]: document for document in report["documents"]}
    assert states["crash"]["status"] == "failed"
    assert states["crash"]["returncode"] == 73
    assert states["success"]["status"] == states["later"]["status"] == "succeeded"
    assert states["success"]["visual_review"] == "pending"
    saved = json.loads((output / "documents/crash/status.json").read_text(encoding='utf-8'))
    assert saved["stages"]["parse"] == "failed"
    assert saved["errors"][0]["type"] == "WorkerProcessError"
    assert json.loads((output / "summary.json").read_text(encoding='utf-8'))["counts"] == {"failed": 1, "succeeded": 2}
    assert "visual_review_completed" not in json.dumps(report)


def test_timeout_records_active_stage_and_reaps_worker(tmp_path, monkeypatch):
    """超时会终止独立进程并保留当时阶段，而不是把整轮保持在 running。"""
    manifest = _manifest(tmp_path, ["timeout"])
    document = json.loads(manifest.read_text(encoding='utf-8'))["documents"][0]

    def command(manifest, output, name, code):
        """在导出阶段暂停独立进程，测试父进程超时回收。"""
        script = (
            "import json,time,sys;from pathlib import Path;folder=Path(sys.argv[1]);"
            "state=json.loads((folder/'status.json').read_text(encoding='utf-8'));state['stages']['export']='running';"
            "(folder/'status.json').write_text(json.dumps(state));time.sleep(30)"
        )
        return [sys.executable, "-c", script, str(output / "documents" / name)]

    monkeypatch.setattr(corpus, "_worker_command", command)
    # Windows CI 启动解释器可能超过半秒；超时仍远短于 worker 的实际阻塞时间。
    state = corpus._run_one(manifest, tmp_path / "run", document, "b" * 64, False, 5)
    assert state["status"] == "failed"
    assert state["stages"]["export"] == "failed"
    assert "超时" in state["errors"][0]["message"]


def test_resume_rejects_source_or_code_drift_without_overwriting_evidence(tmp_path):
    """恢复只能针对同一原件和源码，拒绝时必须保留已有输出。"""
    manifest = _manifest(tmp_path, ["sample"])
    document = json.loads(manifest.read_text(encoding='utf-8'))["documents"][0]
    folder = tmp_path / "run/documents/sample"
    folder.mkdir(parents=True)
    state = corpus._new_status(document, "b" * 64)
    corpus._save_status(folder, state)
    before = (folder / "status.json").read_bytes()
    with pytest.raises(ValueError, match="身份"):
        corpus._resumable(folder, document, "c" * 64)
    assert (folder / "status.json").read_bytes() == before
    Path(document["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="改变"):
        corpus._resumable(folder, document, "b" * 64)
    assert (folder / "status.json").read_bytes() == before


def test_retry_archives_failed_artifacts_instead_of_overwriting(tmp_path):
    """失败重试把旧 PNG 和阶段记录留在独立历史目录。"""
    folder = tmp_path / "document"
    folder.mkdir()
    (folder / "source-0001.png").write_bytes(b"old image")
    (folder / "status.json").write_text('{"status":"failed"}')
    corpus._archive_attempt(folder)
    assert (folder / "history/attempt-0001/source-0001.png").read_bytes() == b"old image"
    assert not (folder / "source-0001.png").exists()
    (folder / "status.json").write_text('{"status":"failed"}')
    corpus._archive_attempt(folder)
    assert (folder / "history/attempt-0002/status.json").is_file()


def test_manifest_source_identity_is_checked_before_starting_any_worker(tmp_path, monkeypatch):
    """清单生成后解析源码发生变化时，应在任何文档开始前拒绝新旧混用。"""
    manifest = _manifest(tmp_path, ["sample"])
    data = json.loads(manifest.read_text(encoding='utf-8'))
    data["environment"] = {"source_sha256": "a" * 64}
    manifest.write_text(json.dumps(data))
    monkeypatch.setattr(corpus, "fingerprint", lambda: "b" * 64)
    with pytest.raises(ValueError, match="清单生成"):
        corpus.run_batch(manifest, tmp_path / "run")
    assert not (tmp_path / "run/documents").exists()
