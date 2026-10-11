"""汇总真实逐页视觉裁决，保持未审页面和运行失败可见。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import html
import json
from pathlib import Path


VERDICTS = {"no_obvious_issue", "issues", "ambiguous", "out_of_scope"}
CATEGORIES = {
    "table": "表格范围、碎片与成员归属",
    "membership": "图表注释与成员归属",
    "graphics": "图形、图片及注释归属",
    "formula": "公式范围、碎片与编号",
    "text": "文本段界、碎片及成员完整性",
    "title": "标题与正文分类",
    "index": "目录条目识别与页码",
    "order": "阅读顺序与跨页连续性",
    "auxiliary": "页眉页脚、脚注及水印",
    "visualization": "标框可视化自身问题",
    "other": "代码与特殊文本分类",
}
VERDICT_LABELS = {"no_obvious_issue": "无明显问题", "issues": "确认问题", "ambiguous": "待确认",
                  "out_of_scope": "原生能力边界", "pending": "未审", "not_selected": "未选页"}


def read_json(path: Path) -> dict:
    """读取本轮结构化证据，不将 PDF 或审阅文字当作可执行指令。"""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict | list) -> None:
    """原子替换可重新生成的汇总，不修改独立审阅记录或历史输出。"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def evidence_path(root: Path, value: str) -> Path:
    """限定证据到本轮运行目录，并验证引用的图像已经存在。"""
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Invalid evidence path: {value}")
    return path


def collect_reviews(root: Path, documents: dict) -> tuple[dict, list]:
    """合并独立 agent 记录；有分歧时保留待确认，只有复核者可明确裁决。"""
    votes = defaultdict(list)
    findings = []
    ids = set()
    page_indexes = {}
    for path in sorted((root / "reviews").glob("**/*.json")):
        data = read_json(path)
        reviewer = data["reviewer"]
        for vote in data.get("pages", []):
            doc = documents[vote["document_id"]]
            number = vote["page"]
            if vote["status"] not in VERDICTS or not isinstance(number, int) or not 1 <= number <= doc["page_count"]:
                raise ValueError(f"Invalid page verdict in {path}")
            paths = vote.get("evidence", [])
            required = {f"documents/{doc['name']}/source-{number:04d}.png",
                        f"documents/{doc['name']}/layout-{number:04d}.png"}
            if not required <= set(paths):
                raise ValueError(f"Source and layout evidence required: {path}, {number}")
            for reference in paths:
                evidence_path(root, reference)
            votes[(doc["name"], number)].append({**vote, "reviewer": reviewer, "record": str(path.relative_to(root))})
        for finding in data.get("findings", []):
            doc = documents[finding["document_id"]]
            if finding["id"] in ids:
                raise ValueError(f"Duplicate finding id: {finding['id']}")
            ids.add(finding["id"])
            if finding.get("category") not in CATEGORIES or finding.get("status") not in {"confirmed", "ambiguous"}:
                raise ValueError(f"Invalid finding category or status: {path}")
            bbox = finding["bbox"]
            if len(bbox) != 4 or not 0 <= bbox[0] < bbox[2] <= 1 or not 0 <= bbox[1] < bbox[3] <= 1:
                raise ValueError(f"Invalid normalized finding bbox: {path}")
            if not 1 <= finding["page"] <= doc["page_count"]:
                raise ValueError(f"Invalid finding page: {path}")
            required = {f"documents/{doc['name']}/source-{finding['page']:04d}.png",
                        f"documents/{doc['name']}/layout-{finding['page']:04d}.png"}
            if not required <= set(finding["evidence"]):
                raise ValueError(f"Source and layout finding evidence required: {path}")
            for reference in finding["evidence"]:
                evidence_path(root, reference)
            if doc["name"] not in page_indexes:
                index_path = root / "documents" / doc["name"] / "page-index.json"
                page_indexes[doc["name"]] = read_json(index_path).get("pages", []) if index_path.is_file() else []
            page_info = next((page for page in page_indexes[doc["name"]] if page["page"] == finding["page"]), {})
            indices = finding.get("block_indices", finding.get("indices", []))
            block_refs = [block for block in page_info.get("blocks", []) if block["index"] in indices]
            anchors = finding.get("anchors") or {
                f"block-{block['index']}": block["text_start"]
                for block in block_refs if block.get("text_start")
            }
            if not anchors:
                anchors = {"unavailable_reason": "漏块或图像区域没有可用文字锚点，定位依据为原页归一化区域。"}
            matched = {block["index"] for block in block_refs}
            reference_status = "matched" if indices and set(indices) <= matched else "partial" if matched else "not_specified_or_not_emitted"
            state_path = root / "documents" / doc["name"] / "status.json"
            state = read_json(state_path) if state_path.is_file() else {}
            findings.append({**finding, "block_indices": indices, "block_refs": block_refs,
                             "status": "out_of_scope" if finding.get("resolution_status") == "out_of_scope" else finding["status"],
                             "block_reference_status": reference_status, "anchors": anchors,
                             "anchors_source": "reviewer" if finding.get("anchors") else "page-index-or-explicit-unavailable",
                             "severity": finding.get("severity", finding.get("priority", "unrated")),
                             "source_sha256": doc["sha256"], "code_sha256": state.get("code_sha256"),
                             "source_path": doc["path"],
                             "reviewer": reviewer, "record": str(path.relative_to(root))})
    merged = {}
    for key, rows in votes.items():
        decisive = [row for row in rows if row.get("adjudication") is True]
        statuses = {row["status"] for row in (decisive or rows)}
        merged[key] = {"status": statuses.pop() if len(statuses) == 1 else "ambiguous", "votes": rows}
    for finding in findings:
        key = finding["document_id"], finding["page"]
        if key not in merged:
            raise ValueError(f"Finding without an actual page review: {finding['id']}")
        if finding["status"] == "confirmed" and merged[key]["status"] == "no_obvious_issue":
            raise ValueError(f"Confirmed finding contradicts page verdict: {finding['id']}")
    return merged, findings


def aggregate(root: Path) -> dict:
    """逐文档对照选页清单，分别计算解析覆盖和真正看图的视觉覆盖。"""
    manifest = read_json(root / "manifest.json")
    documents = {doc["name"]: doc for doc in manifest["documents"]}
    reviews, findings = collect_reviews(root, documents)
    coverage = []
    failures = []
    previous_failures = []
    parsed_pages = 0
    actual_backends = Counter()
    stages = defaultdict(Counter)
    planned = 0
    for doc in documents.values():
        folder = root / "documents" / doc["name"]
        state = read_json(folder / "status.json") if (folder / "status.json").is_file() else {"stages": {}}
        if state.get("stages", {}).get("parse") == "succeeded":
            parsed_pages += doc["page_count"]
        backend = state.get("backend", {})
        if backend:
            actual_backends[(backend.get("backend"), backend.get("protocol"), backend.get("extension_sha256"))] += 1
        for previous in sorted((folder / "history").glob("*/status.json")):
            attempt = read_json(previous)
            if attempt.get("status") == "failed" or attempt.get("errors"):
                previous_failures.append({"document_id": doc["name"], "status": attempt.get("status"),
                                          "errors": attempt.get("errors", []), "stages": attempt.get("stages", {}),
                                          "evidence": str(previous.relative_to(root))})
        for stage in ("parse", "export", "overlay", "render"):
            stages[stage][state.get("stages", {}).get(stage, "pending")] += 1
        if any(value == "failed" for value in state.get("stages", {}).values()) or state.get("errors"):
            failures.append({"document_id": doc["name"], "source_path": doc["path"],
                             "stages": state.get("stages", {}), "errors": state.get("errors", []),
                             "evidence": f"documents/{doc['name']}/status.json"})
        index = read_json(folder / "page-index.json") if (folder / "page-index.json").is_file() else {}
        selected = index.get("review_pages", state.get("review_pages", []))
        planned += min(doc["page_count"], 100)
        if selected and len(selected) != min(doc["page_count"], 100):
            raise ValueError(f"Incorrect selected page count: {doc['name']}")
        for number in range(1, doc["page_count"] + 1):
            review = reviews.get((doc["name"], number))
            coverage.append({"document_id": doc["name"], "page": number, "selected": number in selected,
                             "status": review["status"] if review else "pending" if number in selected else "not_selected",
                             "review": review})
    counts = Counter(row["status"] for row in coverage if row["selected"])
    summary = {"documents": len(documents), "source_pages": sum(doc["page_count"] for doc in documents.values()),
               "parsed_pages": parsed_pages, "corpus_counts": manifest.get("counts", {}),
               "previous_failed_attempts": previous_failures,
               "actual_backends": [{"backend": key[0], "protocol": key[1], "extension_sha256": key[2],
                                    "documents": count} for key, count in actual_backends.items()],
               "planned_visual_pages": planned, "selected_pages_ready": sum(row["selected"] for row in coverage),
               "reviewed_selected_pages": sum(counts[key] for key in VERDICTS), "selected_page_status": dict(counts),
               "unreviewed_planned_pages": planned - sum(counts[key] for key in VERDICTS),
               "extra_reviewed_pages": sum(not row["selected"] and row["review"] is not None for row in coverage),
               "stages": {key: dict(value) for key, value in stages.items()},
               "failures": failures,
               "confirmed_issue_documents": len({finding["document_id"] for finding in findings if finding["status"] == "confirmed"}),
               "confirmed_issue_pages": len({(finding["document_id"], finding["page"]) for finding in findings if finding["status"] == "confirmed"}),
               "confirmed_findings": sum(finding["status"] == "confirmed" for finding in findings),
               "ambiguous_findings": sum(finding["status"] == "ambiguous" for finding in findings),
               "out_of_scope_findings": sum(finding["status"] == "out_of_scope" for finding in findings)}
    summary["visual_review_complete"] = summary["unreviewed_planned_pages"] == 0
    write_json(root / "coverage.json", {"summary": summary, "pages": coverage})
    write_json(root / "findings.json", findings)
    write_json(root / "review-summary.json", summary)
    publish_document_reviews(root, documents, coverage, findings)
    render_overview(root, summary, documents, coverage, findings)
    render_report(root, summary, documents, findings)
    return summary


def publish_document_reviews(root: Path, documents: dict, coverage: list, findings: list) -> None:
    """发布各文档的合并裁决并同步静态图库，不覆盖独立审阅原始证据。"""
    from bs4 import BeautifulSoup

    by_document = defaultdict(list)
    finding_ids = defaultdict(list)
    for page in coverage:
        if page["selected"] or page["review"] is not None:
            by_document[page["document_id"]].append(page)
    for finding in findings:
        finding_ids[(finding["document_id"], finding["page"])].append(finding["id"])
    for document_id, rows in by_document.items():
        folder = root / "documents" / document_id
        published = [{**row, "finding_ids": finding_ids[(document_id, row["page"])]} for row in rows]
        write_json(folder / "review.json", {"document_id": document_id, "pages": published})
        gallery = folder / "gallery.html"
        if not gallery.is_file():
            continue
        soup = BeautifulSoup(gallery.read_text(encoding="utf-8"), "html.parser")
        document_name = Path(documents[document_id]["path"]).name
        if soup.find("h1") is not None:
            soup.find("h1").string = document_name
        if soup.find("title") is not None:
            soup.find("title").string = document_name
        style = soup.find("style")
        if style is not None and "/* audit-responsive */" not in style.text:
            style.string = style.text + "/* audit-responsive */h1{font-size:24px;overflow-wrap:anywhere}details{overflow:auto}td{overflow-wrap:anywhere}"
        for row in rows:
            section = soup.find("section", id=f"page-{row['page']}")
            if section is None:
                section = soup.new_tag("section", id=f"page-{row['page']}")
                section.append(soup.new_tag("h2"))
                pair = soup.new_tag("div", attrs={"class": "pair"})
                for stem, label in (("source", "原页"), ("layout", "布局")):
                    figure = soup.new_tag("figure")
                    caption = soup.new_tag("figcaption")
                    caption.string = label
                    figure.append(caption)
                    source = f"{stem}-{row['page']:04d}.png"
                    link = soup.new_tag("a", href=source)
                    link.append(soup.new_tag("img", src=source, loading="lazy"))
                    figure.append(link)
                    pair.append(figure)
                section.append(pair)
                later = next((item for item in soup.find_all("section")
                              if int(item.get("id", "page-0").removeprefix("page-")) > row["page"]), None)
                later.insert_before(section) if later is not None else soup.append(section)
            section["data-verdict"] = row["status"]
            heading = section.find("h2")
            if heading is not None:
                heading.string = f"物理页 {row['page']} · {VERDICT_LABELS[row['status']]}"
            note = section.find("p", attrs={"data-review-note": True})
            if note is None:
                note = soup.new_tag("p")
                note["data-review-note"] = "true"
                heading.insert_after(note) if heading is not None else section.insert(0, note)
            votes = (row["review"] or {}).get("votes", [])
            note.string = "；".join(vote.get("note", "") for vote in votes) or VERDICT_LABELS[row["status"]]
        paragraph = soup.find("p")
        reviewed = sum(row["status"] in VERDICTS for row in rows if row["selected"])
        extra = sum(row["status"] in VERDICTS for row in rows if not row["selected"])
        if paragraph is not None:
            links = list(paragraph.find_all("a", recursive=False))
            if not any(link.get("href") == "../../overview.html" for link in links):
                overview = soup.new_tag("a", href="../../overview.html")
                overview.string = "总览"
                links.insert(0, overview)
            paragraph.clear()
            for link in links:
                paragraph.append(link)
                paragraph.append(" · ")
            paragraph.append(f"实际已审 {reviewed} / {min(documents[document_id]['page_count'], 100)} 计划页；补审 {extra} 页")
        gallery.write_text(str(soup), encoding="utf-8")


def render_overview(root: Path, summary: dict, documents: dict, coverage: list, findings: list) -> None:
    """用原文件名和真实审阅量建立总览，避免哈希目录掩盖来源和未审状态。"""
    by_document = defaultdict(list)
    issue_counts = Counter(finding["document_id"] for finding in findings if finding["status"] == "confirmed")
    for page in coverage:
        by_document[page["document_id"]].append(page)
    rows = []
    source_root = Path(read_json(root / "manifest.json").get("source_root", "/"))
    for doc in sorted(documents.values(), key=lambda item: item["path"]):
        selected = [page for page in by_document[doc["name"]] if page["selected"]]
        reviewed = sum(page["status"] in VERDICTS for page in selected)
        folder = root / "documents" / doc["name"]
        name = html.escape(str(Path(doc["path"]).relative_to(source_root)))
        gallery = f"documents/{doc['name']}/gallery.html"
        label = f'<a href="{gallery}">{name}</a>' if (folder / "gallery.html").is_file() else name
        stage = read_json(folder / "status.json").get("status", "pending") if (folder / "status.json").is_file() else "pending"
        rows.append(f'<tr><td>{label}</td><td>{doc["page_count"]}</td><td>{stage}</td>'
                    f'<td>{reviewed} / {min(doc["page_count"], 100)}</td><td>{issue_counts[doc["name"]]}</td>'
                    f'<td>{len(doc.get("aliases", []))}</td></tr>')
    content = (
        '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<title>Flash PDF 视觉审阅</title><style>body{font:14px system-ui;margin:24px;color:#222}'
        'h1{font-size:24px}a{color:#126f9b}table{border-collapse:collapse;width:100%;min-width:1040px;table-layout:fixed}'
        'td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}td:first-child{overflow-wrap:anywhere}'
        'th:first-child{width:58%}td:not(:first-child),th:not(:first-child){white-space:nowrap}'
        'th{background:#f1f4f5;position:sticky;top:0}.scroll{overflow:auto}nav{margin:16px 0}</style>'
        f'<h1>Flash PDF 视觉审阅</h1><p>{summary["documents"]} 份 txt 文档，'
        f'{summary["source_pages"]} 页完整解析目标；实际已审 {summary["reviewed_selected_pages"]} / '
        f'{summary["planned_visual_pages"]} 计划页，确认 {summary["confirmed_findings"]} 条问题。</p>'
        '<nav><a href="REPORT.md">分析报告</a> · <a href="REPAIR_PLAN.md">详细修复计划</a> · '
        '<a href="findings.html">问题证据</a> · '
        '<a href="coverage.json">逐页记录</a> · <a href="manifest.json">来源清单</a></nav>'
        '<div class="scroll"><table><thead><tr><th>源文件</th><th>总页数</th><th>产物状态</th>'
        '<th>已审 / 计划</th><th>问题数</th><th>路径数</th></tr></thead><tbody>' + "".join(rows) +
        '</tbody></table></div>')
    for filename in ("overview.html", "index.html"):
        (root / filename).write_text(content, encoding="utf-8")


def render_report(root: Path, summary: dict, documents: dict, findings: list) -> None:
    """按已确认问题生成类别报告，尚无视觉证据时明确显示未完成。"""
    diagnoses = read_json(root / "diagnoses.json") if (root / "diagnoses.json").is_file() else {}
    lines = ["# Flash PDF 样本视觉审阅报告", "", "## 覆盖与运行状态", "",
             f"- 去重 txt 文档：{summary['documents']} 份，源页面 {summary['source_pages']} 页。",
             f"- 实际完整解析：{summary['parsed_pages']} 页；原生后端身份见 review-summary.json / actual_backends。",
             f"- 计划视觉审阅：{summary['planned_visual_pages']} 页；实际已审计划页：{summary['reviewed_selected_pages']} 页。",
             f"- 补充审阅：{summary['extra_reviewed_pages']} 页；确认问题 {summary['confirmed_findings']} 条；"
             f"待确认 {summary['ambiguous_findings']} 条；已裁为能力边界 {summary['out_of_scope_findings']} 条。",
             f"- 确认问题涉及 {summary['confirmed_issue_documents']} 份文档、{summary['confirmed_issue_pages']} 个已审页。",
             "- 逐页状态：" + "、".join(f"{key} {value}" for key, value in summary["selected_page_status"].items()) + "。",
             "- 100 页以内全审，超过 100 页抽样 100 页；抽样结论不能代表未审页面。", ""]
    for stage, counts in summary["stages"].items():
        lines.append(f"- {stage}：" + "、".join(f"{key} {value}" for key, value in counts.items()))
    if summary["failures"]:
        lines += ["", "### 运行异常", ""]
        for failure in summary["failures"]:
            reason = json.dumps(failure["errors"], ensure_ascii=False).replace("\n", " ")
            lines.append(f"- [{Path(failure['source_path']).name}]({failure['evidence']})：{reason}")
    if summary["previous_failed_attempts"]:
        lines += ["", "### 历史失败与重试", "", "以下首次异常已保留，成功重试不代表首次运行没有异常。", ""]
        for attempt in summary["previous_failed_attempts"]:
            reason = json.dumps(attempt["errors"], ensure_ascii=False).replace("\n", " ")
            lines.append(f"- [{attempt['document_id']}]({attempt['evidence']})：{reason}")
    manifest = read_json(root / "manifest.json")
    lines += ["", "### 筛选边界", "", "- 递归分类与 SHA256 去重统计：" +
              json.dumps(manifest.get("counts", {}), ensure_ascii=False) + "。",
              "- 排除 AppleDouble、OCR 类文档和无法打开的文件；全部原路径及明确排除原因在 manifest.json。",
              "- 零原生字符计数不等同缺陷；图片内容不要求 Flash 无 OCR 恢复文字。",
              "- 风险分数仅用于抽样和分流，不是视觉结论；历史问题不计作本轮发现。"]
    lines += ["", "- 本轮为明显版面问题视觉审阅，不是逐字正确率或全部表格单元格准确率测评。",
              "- 问题类别统计基于已落盘的具体问题记录；同页可有多类，复核记录也可能描述同一缺陷。",
              "- 已审问题页的逐页裁决与代表问题清单分别保留，不将记录条数当成独立缺陷率。",
              "- [规则首错证据、真实断言与反例](REPAIR_PLAN.md)；[运行验收](ACCEPTANCE.json)。"]
    lines += ["", "## 问题类别与修复计划", ""]
    groups = defaultdict(list)
    for finding in findings:
        if finding["status"] == "confirmed":
            groups[finding["category"]].append(finding)
    if not groups:
        lines += ["尚无已确认问题记录；这不表示语料无缺陷。", ""]
    for category, rows in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        rows.sort(key=lambda row: (row["severity"], row["document_id"], row["page"], row["id"]))
        doc_count = len({row["document_id"] for row in rows})
        page_count = len({(row["document_id"], row["page"]) for row in rows})
        lines += [f"### {CATEGORIES[category]}", "", f"{len(rows)} 条记录，涉及 {doc_count} 份文档、{page_count} 个已审页。", ""]
        diagnosis = diagnoses.get(category, {})
        for field, label in (("priority", "优先级"), ("cause", "归因与确认程度"), ("entrypoints", "修改入口"),
                             ("plan", "修复方案"), ("tests", "正反例与回归验收")):
            lines.append(f"- {label}：{diagnosis.get(field, '待代表案例溯源确认。')}")
        lines += ["", "| 案例 | 级别 | 文档与物理页 | 观察 |", "|---|---|---|---|"]
        for row in rows[:8]:
            doc = documents[row["document_id"]]
            evidence = row["evidence"][-1]
            lines.append(f"| [{row['id']}]({evidence}) | {row['severity']} | {Path(doc['path']).name} / {row['page']} | "
                         f"{row['observed'].replace('|', '/').replace(chr(10), ' ')} |")
        lines.append("")
    lines += ["## 证据", "", "- [详细修复计划](REPAIR_PLAN.md)", "- [并排图库](overview.html)", "- [逐页覆盖记录](coverage.json)",
              "- [完整问题清单](findings.json)", "- [源文件与别名清单](manifest.json)",
              "- [图库桌面与手机验收](BROWSER_QA.json)", "",
              "本轮没有修改 Flash 解析规则或公共 API。失败、未审和待确认记录均保留。", ""]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    rows = []
    for finding in findings:
        doc = documents[finding["document_id"]]
        images = "".join(f'<a href="{html.escape(value)}"><img loading="lazy" src="{html.escape(value)}"></a>'
                         for value in finding["evidence"][:2])
        label = {"confirmed": "确认问题", "ambiguous": "待确认", "out_of_scope": "原生能力边界"}[finding["status"]]
        rows.append(f'<section><h2>{html.escape(finding["id"])} / {html.escape(Path(doc["path"]).name)} / '
                    f'{finding["page"]}</h2><p>{html.escape(finding["observed"])}</p>'
                    f'<p>{label} · {html.escape(CATEGORIES[finding["category"]])} · {finding["severity"]}</p>'
                    f'<p>期望：{html.escape(finding["expected"])}</p><div>{images}</div></section>')
    (root / "findings.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Flash 问题证据</title><style>'
        'body{font:15px system-ui;margin:24px}section{border-top:1px solid #ccc;padding:16px 0}'
        'h2{font-size:17px}div{display:flex;gap:12px}div a{width:49%}img{width:100%}</style>'
        '<h1>Flash 问题证据</h1>' + "".join(rows), encoding="utf-8")


def main() -> None:
    """从本轮独立证据记录重建报告，可在解析或审阅过程中重复运行。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
