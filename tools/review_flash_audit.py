"""完整重放 Flash 原件并保存源码身份、公共资产和逐页布局证据。"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import sys

from bs4 import BeautifulSoup
from loguru import logger

import docvortex
from docvortex import parse
from docvortex._compute_backend import backend_info
from docvortex.document.pdf import PDFDocument

ROOT = Path(docvortex.__file__).resolve().parents[2]


def fingerprint() -> str:
    """按相对路径与文件字节冻结实际 Python、Rust 源码的组合身份。"""
    value = hashlib.sha256()
    for path in sorted([*(ROOT / "src").rglob("*.py"), *(ROOT / "rust").rglob("*.rs")]):
        value.update(str(path.relative_to(ROOT)).encode())
        value.update(path.read_bytes())
    return value.hexdigest()


def replay(manifest: Path, output: Path, before: Path | None, images: bool) -> None:
    """保留全文上下文并校验公共资产；变化页只进入待审队列，不自动改金标。"""
    output.mkdir(parents=True, exist_ok=False)
    logger.remove()
    logger.add(output / "parse.log", level="INFO")
    code = fingerprint()
    report = {"code_sha256": code, "python": sys.executable, "documents": []}
    sections = []
    for info in json.loads(manifest.read_text())["documents"]:
        source = Path(info["path"])
        assert hashlib.sha256(source.read_bytes()).hexdigest() == info["sha256"]
        folder = output / info["name"]
        folder.mkdir()
        result = parse(source, parse_mode="txt", keep_model_json=True)
        assert len(result.middle_json.pages) == info["page_count"]
        result.export(folder / "document.html", output_format="html")
        result.save_bundle(folder / "bundle")
        middle = result.to_dict()
        model = result.model_json.pages
        (folder / "middle.json").write_text(json.dumps(middle, ensure_ascii=False))
        (folder / "model.json").write_text(json.dumps(model, ensure_ascii=False))
        soup = BeautifulSoup((folder / "document.html").read_text(), "html.parser")
        missing = [
            item["src"]
            for item in soup.find_all("img", src=True)
            if not item["src"].startswith(("data:", "http:", "https:")) and not (folder / item["src"]).is_file()
        ]
        assert not missing
        previous = json.loads((before / info["name"] / "middle.json").read_text()) if before else middle
        changed = [i + 1 for i, (a, b) in enumerate(zip(previous["pages"], middle["pages"], strict=True)) if a != b]
        with PDFDocument(source.read_bytes()) as pdf:
            pdf.draw_layout_bbox(result.middle_json.pages, str(folder / "layout.pdf"))
            if images:
                with PDFDocument(str(folder / "layout.pdf")) as layout:
                    for i in range(pdf.page_count):
                        pdf.render_page(i, scale=1.7).pil_image.save(folder / f"source-{i + 1:03d}.png")
                        layout.render_page(i, scale=1.7).pil_image.save(folder / f"layout-{i + 1:03d}.png")
                        old = f'<img src="../{before.name}/{info["name"]}/layout-{i + 1:03d}.png">' if before else ""
                        sections.append(
                            f"<section><h3>{html.escape(info['name'])} / {i + 1}</h3><div>"
                            f'<img src="{info["name"]}/source-{i + 1:03d}.png">{old}'
                            f'<img src="{info["name"]}/layout-{i + 1:03d}.png"></div></section>'
                        )
        item = {**info, "changed_pages": changed, "assets": len(result.assets), "missing_assets": missing}
        report["documents"].append(item)
        report["backend"] = backend_info()
        (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(info["name"], "changed", changed, "assets", len(result.assets), flush=True)
    assert fingerprint() == code, "重放期间源码改变，输出不能作为本轮版本证据"
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;background:#eee}'
        "section{margin:24px}div{display:flex;gap:8px}img{width:32%;object-fit:contain;align-self:start}</style>"
        "<h1>原页 / 修改前 / 当前 layout</h1>" + "".join(sections)
    )


def main() -> None:
    """用真实文件入口启动渲染进程，避免 macOS 从标准输入启动 spawn 失败。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--images", action="store_true")
    args = parser.parse_args()
    replay(args.manifest, args.output, args.before, args.images)


if __name__ == "__main__":
    main()
