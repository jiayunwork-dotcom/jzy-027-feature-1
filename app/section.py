"""耐张段联合标定：同一耐张段内多档共用一个水平张力 H 的联合反演。

物理模型
--------
同一耐张段内，导线连续跨过若干基悬垂杆塔，悬垂线夹不约束水平方向，
因此整段导线共享同一个水平张力 H；每档的几何（档距 L_i、带符号高差
h_i、单位长度重量 w_i）各自独立，形状参数 c_i = H / w_i 随档而异。
每档的曲线仍由 catenary 模块的双曲关系严格给出——本模块不引入任何
抛物近似或线性化，弧垂与张力之间保持严格的双曲解析形式。

问题结构（与单档标定的本质区别）
--------------------------------
未知量只有一个：公共水平张力 H。每个提交了实测弧垂的档贡献一个残差
方程

    r_i(H) = sag_i(H; x_i) − f_i = 0,

其中 sag_i 是该档在 H 下、测点 x_i 处按双曲关系正算的弧垂，f_i 为实测
值。测量档数 k ≥ 1：k = 1 时退化为单档反演；k > 1 时是超定系统，方程
个数与未知量不再一一对应，不能逐档各自反演再取平均（平均出的 H 在
任何一档上都不精确闭合）。因此先回答存在性与唯一性，再设计求解。

存在性（能不能标定）
--------------------
固定档 i 与测点，r_i(H) 关于 H 连续且严格递减：H → 0+ 时索松弛、
r_i → +∞；H 增大到该档物理上限 H_max_i（最低点贴到较低支座的临界张
力，等高档为 +∞）时 r_i 取最小值 f_min_i − f_i。于是单档的容差约束
|r_i(H)| ≤ tol_i（tol_i = rtol·f_i）的解集是一个闭区间
[H_i_lo, H_i_hi]：

    H_i_lo 由 sag_i = f_i + tol_i 反演得到（H 再小弧垂就偏大超差），
    H_i_hi 由 sag_i = f_i − tol_i 反演得到；若 f_i − tol_i 已低于该档
    可行弧垂下限 f_min_i，则上界就是物理上限 H_max_i 本身。

所有测量档的容差区间，再与段内每一档（含未测量档）的物理上限约束
H ≤ H_max_j 求交，得到联合可行域 [H_lo, H_hi]——区间的交仍是区间，
所以公共解存在当且仅当 H_lo ≤ H_hi。交集为空时，最大下界与最小上界
各自来自哪一档是确定的，矛盾双方可以明确指出：

- 若最小上界是某档的物理上限：该档在候选公共张力下物理不可行
  （SectionSpanInfeasible，指出是哪一档、允许到多大 H）；
- 否则是测量档之间的实测数据互相矛盾（JointCalibrationConflict，
  指出是哪两档对不上），绝不强行返回折中值。

唯一性（报哪一个值）
--------------------
容差 tol_i > 0 时可行域一般是一个区间而非单点，满足条件的 H 不唯一。
本服务报告其中唯一确定的极小极大（Chebyshev）解：令归一化残差
ρ_i(H) = r_i(H)/tol_i，定义

    B(H) = max_i ρ_i(H) + min_i ρ_i(H)。

严格递减函数取 max/min 仍严格递减，故 B 连续且严格递减；又
B(H_lo) ≥ 0 ≥ B(H_hi)，所以 B 在可行区间内有唯一零点 H*。该点使
max_i |ρ_i| 取到最小值（最大与最小归一化残差在 H* 处等幅反号），且
必然落在可行区间内——“极小极大值 ≤ 1”恰好等价于问题可标定。

两个退化/极限情形保证这条路径与既有单档路径的一致：

- 只有一档测量时 B = 2ρ_1，其零点就是该档的独立反演解——与
  invert.calibrate 解的是同一个方程、同一个双曲模型，不是另一条
  可能给出不同答案的代码路径；
- 多档测量完全一致（存在公共精确解）时，所有 ρ_i 在该精确解处同时
  为零，B 的零点就是它——联合反演把公共 H 精确找回来，而不是平均
  出一个在任何一档上都不闭合的折中值。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from . import catenary as cat
from . import invert
from . import validation
from .catenary import CatenarySolution, SpanGeometry
from .errors import (
    InvalidRequest,
    JointCalibrationConflict,
    SagHeightConflict,
    SectionSpanInfeasible,
)

# 联合标定残差的默认相对容差（相对各档实测弧垂）。
# 与单档闭合验收同一把尺：1e-9。调用方可按需在 (0, 1) 内放宽。
DEFAULT_RESIDUAL_RTOL: float = 1e-9

_BISECT_ITERS = 100


@dataclass(frozen=True)
class MemberMeasurement:
    """一档提交的实测：测点水平位置 x 与该处实测弧垂。"""

    x: float
    sag: float


@dataclass(frozen=True)
class SectionMember:
    """耐张段成员档：登记名 + 已登记几何 + 可选实测（未挂测量仪的档为 None）。"""

    name: str
    geometry: SpanGeometry
    measurement: MemberMeasurement | None = None


@dataclass(frozen=True)
class _Band:
    """一个测量档的容差约束对应的公共 H 可行区间 [lo, hi]。

    hi_is_physical_ceiling 为 True 表示上界已被该档物理上限钳制
    （f − tol 低于可行弧垂下限），交集为空时据此区分失败类型。
    """

    lo: float
    hi: float
    hi_is_physical_ceiling: bool


@dataclass(frozen=True)
class JointCalibration:
    """联合标定结果：公共 H、可行公共张力区间、每档在公共 H 下的曲线解。

    solutions 与 members 按下标一一对应；每档的解都满足该档物理可行
    （最低点落在档内），测量档的解在测点上复述实测弧垂到容差内。
    """

    H: float
    residual_rtol: float
    H_interval: tuple[float, float]
    members: tuple[SectionMember, ...]
    solutions: tuple[CatenarySolution, ...]


def max_feasible_H(geometry: SpanGeometry) -> float:
    """该档的物理上限水平张力：最低点贴较低支座的临界张力；等高档为 +∞。"""
    ell_star = cat.boundary_ell(geometry.span, geometry.height_difference)
    if not math.isfinite(ell_star):
        return math.inf
    return geometry.weight_per_length * geometry.span / ell_star


def _normalized_residual(member: SectionMember, H: float, rtol: float) -> float:
    """归一化残差 ρ_i(H) = (sag_i(H; x_i) − f_i) / (rtol·f_i)，严格双曲正算。"""
    geom = member.geometry
    meas = member.measurement
    c = H / geom.weight_per_length
    sag = c * cat.dimensionless_sag(geom.span, geom.height_difference, c, meas.x / geom.span)
    return (sag - meas.sag) / (rtol * meas.sag)


def _tolerance_band(member: SectionMember, rtol: float, ceiling: float) -> _Band:
    """单档容差约束 |r_i| ≤ tol_i 对应的公共 H 可行区间。

    测量弧垂与该档自身几何矛盾（低于可行下限）时抛 SagHeightConflict——
    与单档独立反演同一把尺、同一个错误码。
    """
    geom = member.geometry
    meas = member.measurement
    L, h = geom.span, geom.height_difference
    xi = meas.x / L
    f = meas.sag
    tol = rtol * f

    f_min = cat.boundary_sag(L, h, xi)
    if f_min > 0.0:
        gap = f_min - f
        if gap > 1e-9 * max(f_min, f):
            raise SagHeightConflict(
                f"档 {member.name}：测量弧垂 {f:g} 与该档两端高差 {h:g} 互相矛盾："
                f"测点 x={meas.x:g} 上任何可行悬链的弧垂都不可能小于 {f_min:.6g}；"
                f"该下限对应最低点正好贴在较低支座的临界状态",
                minimum_sag=f_min,
                given_sag=f,
            )

    # 下界：sag = f + tol 的反演（与单档反演走同一个 invert.calibrate）。
    lo = invert.calibrate(geom, f + tol, meas.x).H
    # 上界：sag = f − tol 的反演；f − tol 低于可行下限时上界即物理上限。
    target_hi = f - tol
    if target_hi <= f_min:
        return _Band(lo=lo, hi=ceiling, hi_is_physical_ceiling=True)
    return _Band(lo=lo, hi=invert.calibrate(geom, target_hi, meas.x).H,
                 hi_is_physical_ceiling=False)


def _solve_common_H(measured: list[SectionMember], rtol: float,
                    H_lo: float, H_hi: float) -> float:
    """B(H) = max ρ_i + min ρ_i 在 [H_lo, H_hi] 上的唯一零点（对数二分）。

    B 连续严格递减且 B(H_lo) ≥ 0 ≥ B(H_hi)，零点唯一；该点即极小极大
    （Chebyshev）解：max_i |ρ_i| 在可行区间上的最小值点。
    """
    def B(H: float) -> float:
        rhos = [_normalized_residual(m, H, rtol) for m in measured]
        return max(rhos) + min(rhos)

    if not H_lo < H_hi:
        return H_lo  # 区间退化为点（容差带恰好相切）
    if B(H_lo) <= 0.0:
        return H_lo
    if B(H_hi) >= 0.0:
        return H_hi
    t_lo, t_hi = math.log(H_lo), math.log(H_hi)
    for _ in range(_BISECT_ITERS):
        t_mid = 0.5 * (t_lo + t_hi)
        if B(math.exp(t_mid)) >= 0.0:
            t_lo = t_mid
        else:
            t_hi = t_mid
    return math.exp(0.5 * (t_lo + t_hi))


def _forward_at_H(member: SectionMember, H: float, ceiling: float) -> CatenarySolution:
    """公共 H 下铺该档完整曲线。

    H 已由区间交保证不超过该档物理上限；贴边界时临界档距的浮点舍入
    可能让可达性判断差几个 ulp，向内夹回档内。真正越界（防御性，正常
    不会走到）抛 SectionSpanInfeasible 并指名该档。
    """
    geom = member.geometry
    c = H / geom.weight_per_length
    for _ in range(8):
        if cat.is_reachable(geom.span, geom.height_difference, c):
            return cat.solve_with_c(geom, c)
        c = math.nextafter(c, 0.0)
    raise SectionSpanInfeasible(
        f"档 {member.name} 在公共水平张力 {H:.6g} 下不可行：最低点落到档外，"
        f"该档允许的最大水平张力为 {ceiling:.6g}",
        span=member.name,
        max_feasible_H=ceiling,
        given_H=H,
    )


def calibrate_joint(members, residual_rtol: float = DEFAULT_RESIDUAL_RTOL) -> JointCalibration:
    """耐张段联合标定入口：成员档（部分带实测）→ 公共 H → 每档曲线解。

    失败语义：
    - SagHeightConflict：某测量档的实测与该档自身几何矛盾（同单档）；
    - SectionSpanInfeasible：候选公共 H 超出某档物理可行范围，指名该档；
    - JointCalibrationConflict：测量档之间互相矛盾，指名对不上的双方。
    """
    rtol = validation.require_fraction(residual_rtol, "残差相对容差 residual_rtol")
    members = tuple(members)
    if not members:
        raise InvalidRequest("耐张段至少需要一个成员档")
    names = [m.name for m in members]
    if len(set(names)) != len(names):
        raise InvalidRequest("耐张段内档名重复")
    measured = [m for m in members if m.measurement is not None]
    if not measured:
        raise InvalidRequest("耐张段联合标定至少需要一档提交实测弧垂")

    # 1) 段内每一档（含未测量档）的物理上限 H_max。
    ceilings = {m.name: max_feasible_H(m.geometry) for m in members}

    # 2) 每个测量档的容差区间（含自身几何矛盾检查）。
    bands = {m.name: _tolerance_band(m, rtol, ceilings[m.name]) for m in measured}

    # 3) 联合可行域 = 各测量档容差区间 ∩ 各档物理上限。区间的交仍是区间。
    lo_name = max(bands, key=lambda n: bands[n].lo)
    H_lo = bands[lo_name].lo
    hi_name, H_hi, hi_physical = None, math.inf, False
    for m in members:
        band = bands.get(m.name)
        if band is not None:
            cand, physical = band.hi, band.hi_is_physical_ceiling
        else:
            cand, physical = ceilings[m.name], True
        if cand < H_hi:
            hi_name, H_hi, hi_physical = m.name, cand, physical

    if H_lo > H_hi:
        by_name = {m.name: m for m in members}
        if hi_physical:
            bad = by_name[hi_name].geometry
            raise SectionSpanInfeasible(
                f"档 {hi_name} 在候选公共水平张力下不可行：该档（档距 {bad.span:g}、"
                f"高差 {bad.height_difference:g}）要求 H ≤ {H_hi:.6g}，否则最低点落到档外；"
                f"但档 {lo_name} 的实测要求公共 H ≥ {H_lo:.6g}",
                span=hi_name,
                max_feasible_H=H_hi,
                required_min_H=H_lo,
                required_min_by=lo_name,
            )
        raise JointCalibrationConflict(
            f"耐张段测量互相矛盾：档 {lo_name} 要求公共水平张力 H ≥ {H_lo:.6g}，"
            f"而档 {hi_name} 只允许 H ≤ {H_hi:.6g}；在相对容差 {rtol:g} 下不存在"
            f"同时满足所有测量档的公共水平张力",
            required_min_H=H_lo,
            required_min_by=lo_name,
            allowed_max_H=H_hi,
            allowed_max_by=hi_name,
            measured_H_intervals={
                n: {"min_H": b.lo, "max_H": b.hi} for n, b in bands.items()
            },
        )

    # 4) 可行区间内的极小极大解：B(H) = max ρ + min ρ 的唯一零点。
    H = _solve_common_H(measured, rtol, H_lo, H_hi)

    # 5) 公共 H 对段内每一档（含未测量档）正算完整曲线。
    solutions = tuple(_forward_at_H(m, H, ceilings[m.name]) for m in members)
    return JointCalibration(
        H=H,
        residual_rtol=rtol,
        H_interval=(H_lo, H_hi),
        members=members,
        solutions=solutions,
    )
