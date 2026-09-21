"""服务编排：把正算 / 反演结果序列化成对外 JSON。

正算与反演必须走同一套双曲关系：反演结果里带“正算再反演”的闭合误差，
它由已知 H 正算一次、再把算出的弧垂交回反演得到。
"""
from __future__ import annotations

from . import catenary as cat
from . import invert
from . import validation
from .catenary import CatenarySolution, SpanGeometry

_DEFAULT_SAMPLE_COUNT = 21


def uniform_samples(L: float, n: int) -> list[float]:
    n = validation.require_sample_count(n)
    if n == 1:
        return [0.0]
    return [L * i / (n - 1) for i in range(n)]


def resolve_samples(L: float, body: dict) -> list[float]:
    if "samples" in body and body["samples"] is not None:
        return validation.require_samples(body["samples"], L)
    n = body.get("sample_count", _DEFAULT_SAMPLE_COUNT)
    return uniform_samples(L, n)


def _point_dict(p: cat.CablePoint) -> dict:
    return {"x": p.x, "y": p.y, "chord": p.chord, "sag": p.sag}


def serialize_solution(sol: CatenarySolution, xs: list[float], geometry_source: str) -> dict:
    g = sol.geometry
    return {
        "H": sol.H,
        "c": sol.c,
        "lowest_point": {"x": sol.x0, "y": sol.y0},
        "cable_length": sol.length,
        "supports": {"left": {"x": 0.0, "y": 0.0}, "right": {"x": g.span, "y": g.height_difference}},
        "geometry": {
            "source": geometry_source,
            **g.to_dict(),
        },
        "curve": [_point_dict(p) for p in sol.points(xs)],
    }


def _add_measurement(result: dict, sol: CatenarySolution, sag_measured: float, x: float) -> None:
    computed = sol.sag(x)
    result["measurement"] = {
        "x": x,
        "measured_sag": sag_measured,
        "computed_sag": computed,
        "residual": computed - sag_measured,
    }


def calibrate(
    geometry: SpanGeometry,
    measured_sag: float,
    x: float,
    xs: list[float],
    *,
    geometry_source: str = "inline",
) -> dict:
    """反演标定：量弧垂 → 解 H → 铺整条曲线 → 附闭合误差。"""
    sol = invert.calibrate(geometry, measured_sag, x)
    result = serialize_solution(sol, xs, geometry_source)
    _add_measurement(result, sol, float(measured_sag), sol.point(x).x)

    # 正算再反演：拿解出的 H 正算测点弧垂，重新送回反演。
    closure = invert.forward_inverse_closure(geometry, sol.H, x)
    result["forward_inverse_closure"] = closure
    return result


def forward(
    geometry: SpanGeometry,
    H: float,
    xs: list[float],
    *,
    geometry_source: str = "inline",
) -> dict:
    """显式给定 H 的正算；H 非正或高差够不着在这里被拒。"""
    sol = cat.solve_with_H(geometry, H)
    result = serialize_solution(sol, xs, geometry_source)
    # 正算接口同样附闭合证据：正算出跨中弧垂再反演回来。
    closure = invert.forward_inverse_closure(geometry, sol.H, geometry.span / 2.0)
    result["forward_inverse_closure"] = closure
    return result
