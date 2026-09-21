"""耐张段联合标定：同一耐张段内多档共用一个水平张力 H。

与单档标定的关系
----------------
单档标定是“一档一测量点解一个未知量”，方程与未知量一一对应，直接二分。
耐张段标定是一个未知量（公共 H）同时满足多档各自的残差方程，方程数与
未知量不再一一对应，求解策略重新设计如下——但所有逐档计算（逐档反演、
逐档正算、逐档可行域）都直接调用 invert / catenary 模块的同一套双曲
数学，绝不另起炉灶，也绝不引入抛物近似或线性化。

存在性与唯一性的判据
--------------------
固定档几何与测点，测点弧垂 f_i(H) 关于 H 严格递减（c 越小索越松弧垂越大）。
给每个测量档 i 定容差 tol_i = rtol·f_i^meas，则“该档残差落入容差”等价于
公共 H 落在区间

    J_i = [H_i^lo, H_i^hi]，其中 f_i(H_i^lo) = f_i^meas + tol_i，
                                f_i(H_i^hi) = f_i^meas − tol_i。

区间端点由单档反演 invert.calibrate 解出（与单档标定同一套底层数学）；
若容差下沿跌破该档可行弧垂下限，则 H_i^hi 顶到该档物理上限 H_max,i
（最低点贴较低支座的临界张力）。未测量档不约束测量拟合，只贡献物理
可行约束 K_j = (0, H_max,j]。

- 存在性：公共解存在 ⟺ Fm = ∩ J_i 非空（测量相容），且解出的 H 不超过
  段内每一档的 H_max。∩ J_i 为空时，由区间端点可直接指认冲突双方：
  要求 H ≥ max H_i^lo 的档与要求 H ≤ min H_i^hi 的档对不上。
- 唯一性（点估计的取法）：取规范化残差 ρ_i(H) = (f_i(H) − f_i^meas)/tol_i
  的极小极大解，即最小化 R(H) = max_i |ρ_i(H)|。每个 ρ_i 严格递减，故
  D(H) = max_i ρ_i(H) + min_i ρ_i(H) 严格递减，其根唯一，即 R 的唯一
  最小化子；且只要 Fm 非空，该根必落在 Fm 内（R ≤ 1 ⟺ 存在公共解）。
- 退化：只有一个测量档时 D(H) = 2ρ_1(H)，根就是该档独立反演的 H_1，
  与单档标定结果逐位一致——这是同一套数学自然退化，不是特判分支。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from . import catenary as cat
from . import invert
from .catenary import CatenarySolution, SpanGeometry
from .errors import (
    SagHeightConflict,
    SectionMeasurementBelowMinimum,
    SectionMeasurementsInconsistent,
    SectionSpanInfeasible,
)

# 联合标定的正算→反演闭合验收尺，与单档同一把。
CLOSURE_RTOL: float = invert.CLOSURE_RTOL

# 残差容差的默认相对值：tol_i = DEFAULT_RESIDUAL_RTOL × 该档实测弧垂。
DEFAULT_RESIDUAL_RTOL: float = 1e-3

# D(H) 二分迭代次数（ bracket 宽度压到 2^-160，远低于双精度分辨率的余量）。
_BISECT_ITERS = 160
# 区间判空 / 可行性扫描的相对宽容：只吃浮点噪声，绝不吃物理矛盾。
_FEAS_RTOL = 1e-9


@dataclass(frozen=True)
class MeasuredSpan:
    """耐张段内提交了实测弧垂的一档。"""
    name: str
    geometry: SpanGeometry
    measured_sag: float
    measurement_x: float


@dataclass(frozen=True)
class MeasurementFit:
    """一个测量档在公共 H 下的拟合诊断（全部可复算）。"""
    name: str
    measured_sag: float
    measurement_x: float
    tolerance: float                    # tol_i = rtol · f^meas
    implied_H: float                    # 该档单独反演得到的 H_i
    compatible_H_range: tuple[float, float]  # J_i = [H_i^lo, H_i^hi]
    computed_sag: float                 # 公共 H 下该测点的理论弧垂
    residual: float                     # computed − measured
    normalized_residual: float          # residual / tolerance
    within_tolerance: bool


@dataclass(frozen=True)
class SectionSolution:
    """联合求解结果：公共 H、逐测量档诊断、测量相容区间。"""
    H: float
    fits: tuple[MeasurementFit, ...]
    compatible_H_range: tuple[float, float]  # Fm = ∩ J_i
    max_abs_normalized_residual: float


@dataclass(frozen=True)
class JointCalibration:
    """耐张段标定完整结果：公共 H + 段内每一档的整条曲线 + 闭合证据。"""
    H: float
    solution: SectionSolution
    curves: tuple[tuple[str, CatenarySolution], ...]  # 按段内成员顺序
    closure: dict


def max_feasible_H(geometry: SpanGeometry) -> float:
    """该档物理可行（最低点不跑出档外）的水平张力上限；等高时无界。"""
    ell_star = cat.boundary_ell(geometry.span, geometry.height_difference)
    if not math.isfinite(ell_star):
        return math.inf
    return geometry.weight_per_length * geometry.span / ell_star


def _sag_at_H(geometry: SpanGeometry, H: float, x: float) -> float:
    """公共张力 H 下某档测点的理论弧垂；与 CatenarySolution.sag 同一求值路径。"""
    c = H / geometry.weight_per_length
    return c * cat.dimensionless_sag(geometry.span, geometry.height_difference,
                                     c, x / geometry.span)


def _implied_H_and_range(m: MeasuredSpan, rtol: float) -> tuple[float, float, float, float]:
    """返回 (H_i, H_i^lo, H_i^hi, tol_i)：该测量档单独反演的 H 与容差相容区间。

    三次求解都走 invert.calibrate——和单档标定逐位同一套反演数学。
    """
    g = m.geometry
    tol = rtol * m.measured_sag
    try:
        H_i = invert.calibrate(g, m.measured_sag, m.measurement_x).H
    except SagHeightConflict as exc:
        raise SectionMeasurementBelowMinimum(
            f"档 {m.name!r} 的测量弧垂 {m.measured_sag:g} 低于该档可行下限："
            f"在档距 {g.span:g}、高差 {g.height_difference:g} 下，测点 "
            f"x={m.measurement_x:g} 上任何可行悬链的弧垂都不可能小于 "
            f"{exc.details.get('minimum_feasible_sag', 0.0):.6g}，耐张段不可标定",
            span=m.name,
            measured_sag=m.measured_sag,
            minimum_feasible_sag=exc.details.get("minimum_feasible_sag"),
        ) from exc

    # 容差上沿（弧垂更大 ⇒ H 更小）：f^meas + tol > f^meas ≥ f_min，必有解。
    H_lo = invert.calibrate(g, m.measured_sag + tol, m.measurement_x).H

    # 容差下沿（弧垂更小 ⇒ H 更大）：跌破该档可行下限时，H_i^hi 顶到 H_max,i。
    xi = m.measurement_x / g.span
    f_min = cat.boundary_sag(g.span, g.height_difference, xi)
    if m.measured_sag - tol <= f_min:
        H_hi = max_feasible_H(g)
    else:
        H_hi = invert.calibrate(g, m.measured_sag - tol, m.measurement_x).H
    return H_i, H_lo, H_hi, tol


def solve_common_H(measured: list[MeasuredSpan],
                   members: list[tuple[str, SpanGeometry]],
                   rtol: float) -> SectionSolution:
    """联合求解公共水平张力。

    measured：提交了实测弧垂的档（至少一档）；members：段内全部成员档
    （含未测量档，只参与物理可行性约束）。失败时抛出
    SectionMeasurementsInconsistent / SectionSpanInfeasible /
    SectionMeasurementBelowMinimum，全部点名到档。
    """
    per_span = [_implied_H_and_range(m, rtol) for m in measured]
    implied = [p[0] for p in per_span]

    # ---- 存在性判据一：测量相容区间 Fm = ∩ J_i 必须非空 -------------------
    Fm_lo = max(p[1] for p in per_span)
    Fm_hi = min(p[2] for p in per_span)
    if Fm_lo > Fm_hi * (1.0 + _FEAS_RTOL):
        # 指认冲突双方：把 H 抬得最高的档 vs 把 H 压得最低的档。
        i_lo = max(range(len(per_span)), key=lambda i: per_span[i][1])
        i_hi = min(range(len(per_span)), key=lambda i: per_span[i][2])
        detail = [
            {
                "span": m.name,
                "measured_sag": m.measured_sag,
                "measurement_x": m.measurement_x,
                "implied_H": p[0],
                "tolerance": p[3],
                "compatible_H_range": [p[1], p[2]],
            }
            for m, p in zip(measured, per_span)
        ]
        raise SectionMeasurementsInconsistent(
            f"耐张段测量互相矛盾：档 {measured[i_lo].name!r} 要求公共水平张力 "
            f"H ≥ {per_span[i_lo][1]:.6g}，而档 {measured[i_hi].name!r} 要求 "
            f"H ≤ {per_span[i_hi][2]:.6g}，两者的相容区间不相交；不存在让全部 "
            f"测量档残差都落入容差（rtol={rtol:g}）的公共水平张力，拒绝折中返回",
            conflicting_spans=[measured[i_lo].name, measured[i_hi].name],
            measurements_detail=detail,
        )
    # 浮点噪声级的“恰好相切”：夹成同一值，避免后续 bracket 倒置。
    if Fm_lo > Fm_hi:
        Fm_lo = Fm_hi = 0.5 * (Fm_lo + Fm_hi)

    # ---- 点估计：规范化残差的极小极大解（D(H) 的唯一根） ------------------
    # 根必落在 [min H_i, max H_i] 内：H ≤ min H_i 时所有 ρ_i ≥ 0 故 D ≥ 0；
    # H ≥ max H_i 时所有 ρ_i ≤ 0 故 D ≤ 0。再与 Fm 求交保证残差全部在容差内。
    lo = max(min(implied), Fm_lo)
    hi = min(max(implied), Fm_hi)

    def normalized_residuals(H: float) -> list[float]:
        return [
            (_sag_at_H(m.geometry, H, m.measurement_x) - m.measured_sag) / p[3]
            for m, p in zip(measured, per_span)
        ]

    def D(H: float) -> float:
        vals = normalized_residuals(H)
        return max(vals) + min(vals)

    if hi <= lo:
        # 单测量档（或各档反演值完全一致）时 bracket 退化：H* 就是 H_1 本身，
        # 与 invert.calibrate 的输出逐位相同。
        H_star = lo
    elif D(lo) <= 0.0:
        H_star = lo
    elif D(hi) >= 0.0:
        H_star = hi
    else:
        a, b = lo, hi
        for _ in range(_BISECT_ITERS):
            mid = 0.5 * (a + b)
            if D(mid) >= 0.0:
                a = mid
            else:
                b = mid
        H_star = 0.5 * (a + b)

    # ---- 存在性判据二：段内每一档在公共 H 下都必须物理可行 -----------------
    caps = [(name, max_feasible_H(g)) for name, g in members]
    violated = [
        {"span": name, "max_feasible_H": cap}
        for name, cap in caps
        if H_star > cap and not math.isclose(H_star, cap, rel_tol=_FEAS_RTOL)
    ]
    if violated:
        worst = min(violated, key=lambda v: v["max_feasible_H"])
        raise SectionSpanInfeasible(
            f"测量数据要求公共水平张力 H ≈ {H_star:.6g}，但档 {worst['span']!r} "
            f"在该张力下最低点将落到档外：该档可行上限 H_max = "
            f"{worst['max_feasible_H']:.6g}。不能返回只在部分档上成立的公共解",
            span=worst["span"],
            required_H=H_star,
            max_feasible_H=worst["max_feasible_H"],
            infeasible_spans=violated,
            compatible_H_range=[Fm_lo, Fm_hi],
        )
    # 浮点噪声级的越界：贴回上限（临界态本身可行，最低点贴较低支座）。
    H_cap = min(cap for _, cap in caps)
    H_star = min(H_star, H_cap)

    # ---- 汇总逐档拟合诊断 ---------------------------------------------------
    fits = []
    for m, p in zip(measured, per_span):
        computed = _sag_at_H(m.geometry, H_star, m.measurement_x)
        residual = computed - m.measured_sag
        fits.append(MeasurementFit(
            name=m.name,
            measured_sag=m.measured_sag,
            measurement_x=m.measurement_x,
            tolerance=p[3],
            implied_H=p[0],
            compatible_H_range=(p[1], p[2]),
            computed_sag=computed,
            residual=residual,
            normalized_residual=residual / p[3],
            within_tolerance=abs(residual) <= p[3] * (1.0 + _FEAS_RTOL),
        ))
    return SectionSolution(
        H=H_star,
        fits=tuple(fits),
        compatible_H_range=(Fm_lo, Fm_hi),
        max_abs_normalized_residual=max(abs(f.normalized_residual) for f in fits),
    )


def _joint_closure(measured: list[MeasuredSpan],
                   members: list[tuple[str, SpanGeometry]],
                   H: float, rtol: float) -> dict:
    """联合闭合验收：用解出的公共 H 对各测量档正算弧垂，再当测量值送回
    同一个联合求解器，必须回到同一个 H（容差与单档同一把 CLOSURE_RTOL）。"""
    forward_sags = {m.name: _sag_at_H(m.geometry, H, m.measurement_x) for m in measured}
    re_measured = [
        MeasuredSpan(name=m.name, geometry=m.geometry,
                     measured_sag=forward_sags[m.name], measurement_x=m.measurement_x)
        for m in measured
    ]
    re_solved = solve_common_H(re_measured, members, rtol)
    rel = abs(re_solved.H - H) / H
    return {
        "given_H": H,
        "recovered_H": re_solved.H,
        "relative_error": rel,
        "rtol": CLOSURE_RTOL,
        "passed": rel <= CLOSURE_RTOL,
        "forward_sags": forward_sags,
    }


def calibrate_section(measured: list[MeasuredSpan],
                      members: list[tuple[str, SpanGeometry]],
                      rtol: float) -> JointCalibration:
    """耐张段联合标定入口：解公共 H，并对段内每一档（含未测量档）铺好曲线。"""
    solution = solve_common_H(measured, members, rtol)
    curves = tuple(
        (name, cat.solve_with_H(g, solution.H)) for name, g in members
    )
    closure = _joint_closure(measured, members, solution.H, rtol)
    return JointCalibration(H=solution.H, solution=solution, curves=curves, closure=closure)
