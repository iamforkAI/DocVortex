"""验证安装产物中的运行时依赖路由，不依赖运行测试机器的平台。"""

from __future__ import annotations

from importlib import metadata

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
import pytest


@pytest.mark.parametrize("python_version", ("3.10", "3.11", "3.12", "3.13", "3.14"))
@pytest.mark.parametrize(
    ("sys_platform", "machine", "expected"),
    (
        ("win32", "AMD64", "onnxruntime-directml"),
        ("win32", "amd64", "onnxruntime-directml"),
        ("win32", "x86_64", "onnxruntime-directml"),
        ("win32", "ARM64", "onnxruntime"),
        ("win32", "arm64", "onnxruntime"),
        ("win32", "aarch64", "onnxruntime"),
        ("win32", "x86", "onnxruntime"),
        ("win32", "i686", "onnxruntime"),
        ("win32", "", "onnxruntime"),
        ("darwin", "x86_64", "onnxruntime"),
        ("darwin", "arm64", "onnxruntime"),
        ("linux", "x86_64", "onnxruntime"),
        ("linux", "aarch64", "onnxruntime"),
    ),
)
def test_installed_metadata_selects_one_runtime(sys_platform: str, machine: str, expected: str, python_version: str) -> None:
    """以完整平台与 Python 矩阵检查实际发行元数据，保证两个运行时条件互斥且无遗漏。"""
    environment = {
        **default_environment(),
        "sys_platform": sys_platform,
        "platform_machine": machine,
        "python_version": python_version,
        "python_full_version": f"{python_version}.0",
    }
    requirements = [Requirement(value) for value in metadata.requires("docvortex") or ()]
    selected = [
        requirement
        for requirement in requirements
        if canonicalize_name(requirement.name).startswith("onnxruntime")
        and (requirement.marker is None or requirement.marker.evaluate(environment))
    ]
    assert len(selected) == 1, selected
    assert canonicalize_name(selected[0].name) == expected
    minimum = "1.23" if expected == "onnxruntime-directml" else "1.20.1"
    unsupported = "1.22.1" if expected == "onnxruntime-directml" else "1.20.0"
    assert minimum in selected[0].specifier
    assert unsupported not in selected[0].specifier
    assert "2" not in selected[0].specifier


def test_installed_metadata_requires_runtime_free_magika_fork() -> None:
    """确保宿主依赖新发行包，新包自身也不能再次引入任一种 ORT。"""
    requirements = [Requirement(value) for value in metadata.requires("docvortex") or ()]
    assert all(canonicalize_name(requirement.name) != "magika" for requirement in requirements)
    forks = [requirement for requirement in requirements if canonicalize_name(requirement.name) == "mineru-magika"]
    assert len(forks) == 1 and forks[0].marker is None
    assert "1.0.3" in forks[0].specifier
    assert "1.0.2" not in forks[0].specifier
    assert "1.1.0" not in forks[0].specifier
    fork_dependencies = [Requirement(value) for value in metadata.requires("mineru-magika") or ()]
    assert all(not canonicalize_name(requirement.name).startswith("onnxruntime") for requirement in fork_dependencies)
