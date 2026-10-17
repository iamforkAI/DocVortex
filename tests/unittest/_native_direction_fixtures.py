"""只读恢复方向回归的完整原字样本，兼容已撤销的无关排版字段。"""

from dataclasses import fields
import gzip
from hashlib import sha256
from io import BytesIO
import json
import pickle

from docvortex.analyzers.native.pdf import models
from docvortex.analyzers.native.pdf.inline import types as inline_types
from docvortex.document.pdf._document import PDFPathInfo
from docvortex.document.pdf.text._contracts import Bbox as CharacterBbox


_REMOVED_FIELDS = {
    models._LineItem: {"uniform_uppercase_style"},
    models._PageSource: {"previous_sparse_table"},
    models._PreparedPage: {"sparse_table_context"},
}


def _current_fields(cls, values):
    """仅忽略明确撤销的非方向字段，未知字段报错，字符与几何对象不复制。"""
    known = {item.name for item in fields(cls)}
    unknown = values.keys() - known
    assert unknown <= _REMOVED_FIELDS.get(cls, set()), (cls.__name__, unknown)
    return {key: value for key, value in values.items() if key in known}


class _LegacyState:
    """通过受限状态恢复保留 pickle 的共享字符关系，不扩展生产数据模型。"""

    __slots__ = ()
    _current_class = None

    def __setstate__(self, state):
        """旧 slots 状态仅过滤已撤销的字段，其余值及对象身份逐项恢复。"""
        assert isinstance(state, tuple) and state[0] is None and isinstance(state[1], dict)
        for key, value in _current_fields(self._current_class, state[1]).items():
            setattr(self, key, value)


class _LegacyLine(_LegacyState, models._LineItem):
    """兼容旧行记录中的标题样式缓存。"""

    __slots__ = ()
    _current_class = models._LineItem


class _LegacySource(_LegacyState, models._PageSource):
    """兼容旧页面记录中的少线续表提示。"""

    __slots__ = ()
    _current_class = models._PageSource


class _LegacyPrepared(_LegacyState, models._PreparedPage):
    """兼容旧准备页面中的少线续表缓存。"""

    __slots__ = ()
    _current_class = models._PreparedPage


class _DirectionUnpickler(pickle.Unpickler):
    """只替换三种已知旧状态类，其余样本对象采用原有类型。"""

    def find_class(self, module, name):
        """将已知模型的旧状态解码到测试兼容类，避免为样本保留生产冗余字段。"""
        replacements = {"_LineItem": _LegacyLine, "_PageSource": _LegacySource, "_PreparedPage": _LegacyPrepared}
        if module == models.__name__ and name in replacements:
            return replacements[name]
        return super().find_class(module, name)


def load_direction_pickle(payload):
    """解码已由调用方校验过的原始 gzip 字节，保持原文件及共享引用不变。"""
    return _DirectionUnpickler(BytesIO(gzip.decompress(payload))).load()


def actual_direction_case(name, fixture, fixture_sha):
    """恢复整页真实 Source，不导入段落、标题或公式修复的测试代码。"""
    raw = fixture.read_bytes()
    assert sha256(raw).hexdigest() == fixture_sha
    record = next(case for case in json.loads(gzip.decompress(raw))["cases"] if case["name"] == name)
    pool, dictionary_pool = {}, {}

    def restore(value):
        """按显式运输标记恢复原字、行和已知几何类，保留所有共享关系。"""
        if isinstance(value, list):
            return [restore(item) for item in value]
        if not isinstance(value, dict):
            return value
        kind = value.get("$type")
        if kind == "dictionary":
            index = value["index"]
            if index not in dictionary_pool:
                dictionary_pool[index] = {}
                dictionary_pool[index].update({restore(key): restore(item) for key, item in record["dictionary_pool"][index]})
            return dictionary_pool[index]
        if kind == "line":
            index = value["index"]
            if index not in pool:
                pool[index] = models._LineItem(**_current_fields(models._LineItem, restore(record["line_pool"][index])))
            return pool[index]
        if kind in {"tuple", "set", "frozenset"}:
            return {"tuple": tuple, "set": set, "frozenset": frozenset}[kind](restore(value["items"]))
        if kind == "native_character_box":
            return CharacterBbox(value["bbox"])
        if kind == "dataclass":
            allowed = {
                cls.__name__: cls
                for cls in (
                    models._AxisLine,
                    models._DocumentBodyProfile,
                    models._DocumentTitleProfile,
                    models._TitleStylePrototype,
                    models._PreparedPage,
                    models._LocalAxisLine,
                    models._PageSource,
                    models._DrawingComponentSummary,
                )
            }
            registry = {
                models.__name__: allowed,
                inline_types.__name__: {
                    "PDFTextScriptLine": inline_types.PDFTextScriptLine,
                    "PDFTextScriptRange": inline_types.PDFTextScriptRange,
                },
                PDFPathInfo.__module__: {"PDFPathInfo": PDFPathInfo},
            }
            cls = registry[value["module"]][value["name"]]
            return cls(**_current_fields(cls, restore(value["fields"])))
        if kind == "mapping":
            return {restore(key): restore(item) for key, item in value["items"]}
        return {key: restore(item) for key, item in value.items()}

    return restore(record["call"])[0][0]
