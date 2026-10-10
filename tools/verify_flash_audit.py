"""核验 Flash 复核资料的后端一致性、叶子归属、裁图和导出包回读。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from docvortex import load_bundle
from replay_flash_merged import canonical
from review_flash_audit import fingerprint


def validate_document(folder: Path) -> dict:
    """允许没有内容块的真实空白页，其余页检查叶子编号、资产像素及无损包回读。"""
    middle = json.loads((folder / "middle.json").read_text())
    images = equations = 0
    for page in middle["pages"]:
        pending = list(page.get("blocks", []))
        leaves = []
        while pending:
            block = pending.pop()
            children = (
                [item for item in block.get("content", []) if isinstance(item, dict) and "bbox" in item]
                if isinstance(block.get("content"), list)
                else []
            )
            if children:
                pending.extend(children)
            else:
                leaves.append(block)
        indices = [block["index"] for block in leaves]
        assert len(indices) == len(set(indices)), (folder, page["page_idx"], "叶子重复")
        for block in leaves:
            path = block.get("image_path")
            if not path:
                continue
            with Image.open(folder / "bundle" / path) as original:
                original.load()
                assert min(original.size) > 0
                images += 1
                if block["type"] == "equation":
                    equations += 1
                    with original.convert("L") as gray:
                        assert gray.getextrema()[0] < 250, (folder, path, "公式裁图为空")
    assert canonical(load_bundle(folder / "bundle").to_dict()) == canonical(middle), folder
    return {"decoded_images": images, "formula_crops": equations, "leaf_indices_unique": True, "bundle_equal": True}


def asset_hashes(folder: Path) -> dict[str, str]:
    """按包内相对路径逐字节比较二进制资产，避免 JSON 相等掩盖裁图差异。"""
    return {
        str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in folder.rglob("*")
        if path.is_file() and path.suffix != ".json"
    }


def verify(root: Path, candidate: str, python: str, rust: str, corpus: str) -> dict:
    """只接受同一实际源码指纹的完整文档输出，完整核验两份原件及维护语料。"""
    summaries = {name: json.loads((root / name / "summary.json").read_text()) for name in (candidate, python, rust, corpus)}
    code = fingerprint()
    assert {value["code_sha256"] for value in summaries.values()} == {code}
    report = {"code_sha256": code, "backends": {}, "documents": [], "public_artifacts": {}}
    for info in summaries[candidate]["documents"]:
        models = {}
        assets = {}
        for name in (candidate, python, rust):
            folder = root / name / info["name"]
            report["backends"][name] = summaries[name]["backend"]
            models[name] = {key: canonical(json.loads((folder / key).read_text())) for key in ("model.json", "middle.json")}
            assets[name] = asset_hashes(folder / "bundle")
        assert models[candidate] == models[python] == models[rust], info["name"]
        assert assets[candidate] == assets[python] == assets[rust], info["name"]
        report["documents"].append({"name": info["name"], "backend_equal": True, "assets_sha256_equal": True})
    for name in (candidate, corpus):
        records = [
            {"name": info["name"], **validate_document(root / name / info["name"])} for info in summaries[name]["documents"]
        ]
        report["public_artifacts"][name] = {
            "documents": len(records),
            "pages": sum(info["page_count"] for info in summaries[name]["documents"]),
            "images": sum(info["decoded_images"] for info in records),
            "formula_crops": sum(info["formula_crops"] for info in records),
            "results": records,
        }
    assert fingerprint() == code
    (root / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    """接收本轮独立输出目录，并在断言全部完成后生成机器可读验收记录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--rust", required=True)
    parser.add_argument("--corpus", required=True)
    args = parser.parse_args()
    report = verify(args.root, args.candidate, args.python, args.rust, args.corpus)
    print(
        {
            name: {key: value for key, value in info.items() if key != "results"}
            for name, info in report["public_artifacts"].items()
        }
    )


if __name__ == "__main__":
    main()
