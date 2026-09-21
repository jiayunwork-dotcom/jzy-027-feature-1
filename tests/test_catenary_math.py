"""锁住双曲几何本身的行为。

覆盖需求点名的七条验收：
1. 等高跨中弧垂对上闭式 c(cosh(L/2c)−1)
2. 正算再反演 H 在钉死容差内闭合
3. 高差加大，最低点偏向较低一端
4. 大垂度时抛物近似 wL²/(8H) 系统偏短
5. H 非正被拒
6. 档距非正被拒
7. 弧垂与高差矛盾时标定失败，而不是硬画一条曲线
"""
from __future__ import annotations

import math

import pytest

from app import catenary as cat
from app import invert
from app.catenary import SpanGeometry
from app.errors import (
    InvalidRequest,
    SagHeightConflict,
    SupportsUnreachable,
)


def geom(span, h, w=10.0):
    return SpanGeometry.create(span, h, w)


# ---------------------------------------------------------------------------
# 1. 等高跨中弧垂对上闭式，且最低点必须落在跨中
# ---------------------------------------------------------------------------
def test_level_midpoint_sag_matches_closed_form():
    L, c = 400.0, 250.0
    g = geom(L, 0.0)
    sol = cat.solve_with_c(g, c)

    assert sol.x0 == pytest.approx(L / 2.0, abs=1e-9)
    expected = c * (math.cosh(L / (2.0 * c)) - 1.0)
    assert sol.sag(L / 2.0) == pytest.approx(expected, rel=1e-13)
    # 直接用闭式函数再对一次
    assert cat.level_span_midpoint_sag(c, L) == pytest.approx(expected, rel=1e-15)
    # 两端边界严格闭合、端点弧垂为 0
    assert sol.y(0.0) == pytest.approx(0.0, abs=1e-9)
    assert sol.y(L) == pytest.approx(0.0, abs=1e-9)
    assert sol.sag(0.0) == pytest.approx(0.0, abs=1e-12)
    assert sol.sag(L) == pytest.approx(0.0, abs=1e-12)


def test_level_solution_is_symmetric():
    L, c = 300.0, 120.0
    g = geom(L, 0.0)
    sol = cat.solve_with_c(g, c)
    for x in (0.0, 30.0, 77.7, L / 2.0):
        assert sol.y(x) == pytest.approx(sol.y(L - x), rel=1e-12, abs=1e-12)


# ---------------------------------------------------------------------------
# 2. 正算再反演 H 闭合（等高、不等高、跨中、偏心测点都要对）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "L,h,w,H,x",
    [
        (400.0, 0.0, 10.0, 5000.0, 200.0),
        (300.0, 40.0, 8.0, 3000.0, 150.0),
        (300.0, -60.0, 12.0, 4000.0, 150.0),
        (250.0, 30.0, 10.0, 1800.0, 60.0),   # 偏心测量点
        (250.0, 30.0, 10.0, 900.0, 200.0),   # 较大垂度
    ],
)
def test_forward_then_inverse_H_closes(L, h, w, H, x):
    g = geom(L, h, w)
    f = cat.solve_with_H(g, H).sag(x)
    recovered = invert.calibrate(g, f, x)
    rel = abs(recovered.H - H) / H
    assert rel <= invert.CLOSURE_RTOL, (rel, recovered.H, H)

    closure = invert.forward_inverse_closure(g, H, x)
    assert closure["passed"] is True
    assert closure["relative_error"] <= invert.CLOSURE_RTOL
    # 反演解出的曲线必须精确复述测量弧垂
    assert recovered.sag(x) == pytest.approx(f, rel=1e-10)


# ---------------------------------------------------------------------------
# 3. 高差加大，最低点偏向较低一端
# ---------------------------------------------------------------------------
def test_lowest_point_shifts_toward_lower_support():
    L = 300.0
    h_level = 0.0
    h_small = 20.0
    h_large = 80.0
    c = 500.0

    x_level = cat.solve_with_c(geom(L, h_level), c).x0
    x_small = cat.solve_with_c(geom(L, h_small), c).x0
    x_large = cat.solve_with_c(geom(L, h_large), c).x0

    # h>0：右支座高，低端在左，最低点应在跨中左侧，且高差越大越靠左
    assert x_level == pytest.approx(L / 2.0, abs=1e-9)
    assert 0.0 < x_small < L / 2.0
    assert 0.0 < x_large < x_small

    # h<0：左支座高，最低点偏到跨中右侧
    x_neg = cat.solve_with_c(geom(L, -h_large), c).x0
    assert L / 2.0 < x_neg < L

    # 不等高档绝不能再把最低点报在跨中
    assert x_small != pytest.approx(L / 2.0, abs=1e-6)


def test_calibrated_lowest_point_inside_span_and_boundaries_hold():
    g = geom(300.0, 80.0)
    sol = invert.calibrate(g, measured_sag=40.0, x=150.0)
    assert 0.0 < sol.x0 < 300.0
    assert sol.y(0.0) == pytest.approx(0.0, abs=1e-7)
    assert sol.y(300.0) == pytest.approx(80.0, rel=1e-9, abs=1e-7)
    assert sol.sag(150.0) == pytest.approx(40.0, rel=1e-9)


# ---------------------------------------------------------------------------
# 4. 大垂度：抛物近似 wL²/(8H) 系统偏短
# ---------------------------------------------------------------------------
def test_parabolic_approx_is_too_short_at_large_sag():
    L, c = 100.0, 40.0
    g = geom(L, 0.0, w=10.0)
    sol = cat.solve_with_c(g, c)
    true_sag = sol.sag(L / 2.0)
    parabolic = g.weight_per_length * L**2 / (8.0 * sol.H)

    # 大垂度档（弧垂/档距 ≈ 0.35）：抛物值明显偏短，差距一成以上
    assert true_sag / L > 0.3
    assert parabolic < true_sag
    assert (true_sag - parabolic) / true_sag > 0.10

    # 小垂度极限下二者才接近——但实现依然走双曲
    c2 = 10_000.0
    true_small = cat.level_span_midpoint_sag(c2, L)
    para_small = L**2 / (8.0 * c2)
    assert true_small == pytest.approx(para_small, rel=1e-4)


# ---------------------------------------------------------------------------
# 5. H 非正被拒
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_H", [0.0, -1.0, -1e9])
def test_nonpositive_H_rejected(bad_H):
    g = geom(200.0, 0.0)
    with pytest.raises(InvalidRequest):
        cat.solve_with_H(g, bad_H)
    # 闭合计算入口同样拒
    with pytest.raises(InvalidRequest):
        invert.forward_inverse_closure(g, bad_H)


# ---------------------------------------------------------------------------
# 6. 档距非正被拒（w、高差带非法值一并锁住）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_span", [0.0, -300.0])
def test_nonpositive_span_rejected(bad_span):
    with pytest.raises(InvalidRequest):
        SpanGeometry.create(bad_span, 0.0, 10.0)


@pytest.mark.parametrize("bad_w", [0.0, -1.0])
def test_nonpositive_weight_rejected(bad_w):
    with pytest.raises(InvalidRequest):
        SpanGeometry.create(200.0, 0.0, bad_w)


# ---------------------------------------------------------------------------
# 7. 弧垂与高差矛盾时失败，而不是硬画
# ---------------------------------------------------------------------------
def test_sag_conflicting_with_height_difference_fails():
    L, h = 300.0, 80.0
    g = geom(L, h)
    f_min = cat.boundary_sag(L, h, 0.5)
    assert f_min > 0.0

    # 低于可行下限的“测量”必须失败，并说明矛盾原因与下限
    with pytest.raises(SagHeightConflict) as exc:
        invert.calibrate(g, measured_sag=f_min * 0.5, x=L / 2.0)
    assert exc.value.details["reason"] == "sag_height_difference_conflict"
    assert exc.value.details["minimum_feasible_sag"] == pytest.approx(f_min, rel=1e-9)

    # 等于零/负的弧垂属于非法输入（400 类），不是几何矛盾
    with pytest.raises(InvalidRequest):
        invert.calibrate(g, measured_sag=0.0, x=L / 2.0)


def test_forward_unreachable_supports_raises():
    # H 太大（索太绷）：|δ| > cosh(ℓ)−1，最低点跑到档外，够不着
    g = geom(300.0, 80.0, w=10.0)
    assert cat.is_reachable(300.0, 80.0, 40.0) is True
    assert cat.is_reachable(300.0, 80.0, 5000.0) is False
    with pytest.raises(SupportsUnreachable) as exc:
        cat.solve_with_H(g, H=50_000.0)
    assert exc.value.details["reason"] == "supports_unreachable"
    assert exc.value.details["max_feasible_H"] > 0.0

    # 而反演在同一几何上给一个合理大弧垂能成功，且最低点贴在档内
    sol = invert.calibrate(g, measured_sag=60.0, x=150.0)
    assert 0.0 < sol.x0 < 300.0


def test_measurement_point_outside_span_rejected():
    g = geom(300.0, 10.0)
    with pytest.raises(InvalidRequest):
        invert.calibrate(g, measured_sag=5.0, x=301.0)
    with pytest.raises(InvalidRequest):
        invert.calibrate(g, measured_sag=5.0, x=-1.0)
    with pytest.raises(InvalidRequest):
        invert.calibrate(g, measured_sag=5.0, x=0.0)  # 压在支座上


def test_calibration_exactly_at_boundary_sag_returns_critical_solution():
    # 弧垂恰好等于可行下限时给临界解（最低点贴低端支座），不是报矛盾
    L, h, xi = 300.0, 80.0, 0.5
    g = geom(L, h)
    f_star = cat.boundary_sag(L, h, xi)
    sol = invert.calibrate(g, f_star, L * xi)
    # 最低点落在较低（左）支座
    assert sol.x0 == pytest.approx(0.0, abs=1e-7)
    assert sol.sag(L * xi) == pytest.approx(f_star, rel=1e-9)


def test_large_sag_forward_inverse_still_closes():
    # 大垂度档（弧垂/档距 > 0.3）：绝不能靠抛物近似蒙混，闭合照样成立
    L, h, w = 100.0, 5.0, 10.0
    g = geom(L, h, w)
    H = 400.0  # c = 40，跨中弧垂约 35m
    closure = invert.forward_inverse_closure(g, H, L / 2.0)
    assert closure["passed"] is True
    assert closure["forward_sag"] / L > 0.3


def test_solution_serialization_contains_all_fields():
    g = geom(300.0, 40.0, 10.0)
    sol = invert.calibrate(g, measured_sag=12.0, x=150.0)
    assert sol.length > 300.0
    assert 0.0 < sol.x0 < 150.0  # 右高，最低点偏左
    assert sol.H == pytest.approx(10.0 * sol.c, rel=1e-15)


def test_cable_length_formula_level_span():
    L, c = 200.0, 80.0
    g = geom(L, 0.0)
    sol = cat.solve_with_c(g, c)
    expected = 2.0 * c * math.sinh(L / (2.0 * c))
    assert sol.length == pytest.approx(expected, rel=1e-13)
    assert sol.length > L  # 索长恒大于弦长
