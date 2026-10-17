"""在独立安装环境核对 Magika 发行包及平台对应的唯一 ORT 发行包。"""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
import platform
import sys


def verify_runtime_dependencies() -> None:
    """检查实际安装身份和模块来源，防止同名模块掩盖新旧发行包共存。"""
    expected = (
        "onnxruntime-directml"
        if sys.platform == "win32" and platform.machine().lower() in {"amd64", "x86_64"}
        else "onnxruntime"
    )
    installed = {distribution.metadata["Name"].lower().replace("_", "-") for distribution in metadata.distributions()}
    assert "magika" not in installed, "Upstream magika must not be installed alongside mineru-magika"
    runtimes = {name for name in installed if name.startswith("onnxruntime")}
    assert runtimes == {expected}, f"Expected only {expected}, found {sorted(runtimes)}"
    magika_distribution = metadata.distribution("mineru-magika")
    runtime_distribution = metadata.distribution(expected)

    import magika
    import onnxruntime

    assert magika.__version__ == magika_distribution.version
    assert Path(magika.__file__).resolve() == Path(magika_distribution.locate_file("magika/__init__.py")).resolve()
    assert Path(onnxruntime.__file__).resolve() == Path(runtime_distribution.locate_file("onnxruntime/__init__.py")).resolve()
    assert "CPUExecutionProvider" in onnxruntime.get_available_providers()
    print(f"Installed mineru-magika {magika_distribution.version} and {expected} {runtime_distribution.version} verified")


if __name__ == "__main__":
    verify_runtime_dependencies()
