"""入参检查：H、w、档距、高差、测量值、取样点。

按需求，这些检查与双曲数学分开实现。
"""
from __future__ import annotations

import math

from .errors import InvalidRequest

_NAME_MAX_LEN = 64


def _as_finite_number(value, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidRequest(f"{field} 必须是数，收到的是 {value!r}")
    f = float(value)
    if not math.isfinite(f):
        raise InvalidRequest(f"{field} 必须是有限数，不能是 NaN 或无穷")
    return f


def require_positive(value, field: str) -> float:
    """H、w、档距、弧垂都必须严格为正。"""
    f = _as_finite_number(value, field)
    if f <= 0.0:
        raise InvalidRequest(f"{field} 必须为正数，收到 {f:g}")
    return f


def require_nonnegative(value, field: str) -> float:
    f = _as_finite_number(value, field)
    if f < 0.0:
        raise InvalidRequest(f"{field} 不能为负，收到 {f:g}")
    return f


def require_fraction(value, field: str) -> float:
    """(0, 1) 开区间内的比例量，如联合标定的残差相对容差。"""
    f = _as_finite_number(value, field)
    if not 0.0 < f < 1.0:
        raise InvalidRequest(f"{field} 必须在 (0, 1) 开区间内，收到 {f:g}")
    return f


def require_height_difference(value, field: str = "两端高差") -> float:
    """高差允许带符号、允许为 0；只要求有限。"""
    return _as_finite_number(value, field)


def require_measure_position(x, L: float) -> float:
    """测量点水平位置必须落在档内，且不能正好压在支座上（那里弧垂恒为 0）。"""
    xp = _as_finite_number(x, "测量点水平位置")
    if xp <= 0.0 or xp >= L:
        raise InvalidRequest(f"测量点水平位置必须在档距内部 (0, {L:g})，收到 {xp:g}")
    return xp


def require_samples(samples, L: float) -> list[float]:
    """正算取样点：必须全部落在 [0, L] 上，落到档外当场拒绝。"""
    if not isinstance(samples, list):
        raise InvalidRequest("samples 必须是水平坐标的列表")
    if len(samples) == 0:
        raise InvalidRequest("samples 不能为空")
    out: list[float] = []
    for raw in samples:
        x = _as_finite_number(raw, "samples 中的取样点")
        if x < 0.0 or x > L:
            raise InvalidRequest(f"取样点 {x:g} 落到档距 [0, {L:g}] 外面，拒绝正算")
        out.append(x)
    return out


def require_span_name(name) -> str:
    if not isinstance(name, str) or not name:
        raise InvalidRequest("档名必须是非空字符串")
    if len(name) > _NAME_MAX_LEN:
        raise InvalidRequest(f"档名长度不能超过 {_NAME_MAX_LEN} 个字符")
    if name in (".", "..") or "/" in name or "\\" in name or name.startswith("."):
        raise InvalidRequest("档名不能含路径分隔符或以点开头")
    return name


def require_sample_count(value) -> int:
    n = _as_finite_number(value, "sample_count")
    if n != int(n) or int(n) < 2 or int(n) > 10000:
        raise InvalidRequest("sample_count 必须是 [2, 10000] 内的整数")
    return int(n)
