"""由实测弧垂反演水平张力 H。

反演与正算走同一个无因次函数 ``dimensionless_sag``（见 catenary 模块），
这是验收主尺子：正算出弧垂再反演必须回到同一个 H。

单调性
------
固定几何 (L, h) 与测点 ξ，弧垂 f(c) 关于 c 严格递减：
c → ∞（索绷直）时 f → 0；c → 0（索松弛）时 f → ∞。
但工程可行域要求最低点落在档内，即 c ≤ c*（对应 ℓ ≥ ℓ*），
所以测点上的弧垂有下限 f_min：测量弧垂小于它即“弧垂与高差矛盾”。
"""
from __future__ import annotations

import math

from . import catenary as cat
from . import validation
from .errors import SagHeightConflict
from .catenary import CatenarySolution, SpanGeometry

# 正算→反演闭合验收的钉死相对容差（H 维度）。
CLOSURE_RTOL: float = 1e-9

_BISECT_ITERS = 100
# ℓ 搜索上界（cosh 等函数远在此前已溢出，f(c) 必然已大于任何有限弧垂）。
_ELL_MAX = 1.0e250


def _sag(ell: float, L: float, h: float, xi: float) -> float:
    c = L / ell
    return c * cat.dimensionless_sag(L, h, c, xi)


def calibrate(geometry: SpanGeometry, measured_sag: float, x: float) -> CatenarySolution:
    """反演入口：给一档几何 + 测点实测弧垂，解出 H 并铺好整条曲线。"""
    L, h = geometry.span, geometry.height_difference
    f_p = validation.require_positive(measured_sag, "测量弧垂")
    xp = validation.require_measure_position(x, L)
    xi = xp / L

    # 可行域边界：最低点贴在较低支座时测点弧垂，f(c*) = f_min。
    ell_star = cat.boundary_ell(L, h)
    f_min = 0.0
    if math.isfinite(ell_star):
        f_min = _sag(ell_star, L, h, xi)

    # 测量弧垂小于可行下限：没有任何可行 H 能同时满足边界与弧垂。
    if f_min > 0.0:
        gap = f_min - f_p
        tol_gap = 1e-9 * max(f_min, f_p)
        if gap > tol_gap:
            raise SagHeightConflict(
                f"测量弧垂 {f_p:g} 与两端高差 {h:g} 互相矛盾：在该几何下，"
                f"测点 x={xp:g} 上任何可行悬链的弧垂都不可能小于 {f_min:.6g}；"
                f"该下限对应最低点正好贴在较低支座的临界状态",
                minimum_sag=f_min,
                given_sag=f_p,
            )
        if gap >= 0.0:
            # 恰在临界上（浮点误差以内）：直接给最低点贴低端支座的临界解。
            return cat.solve_with_c(geometry, L / ell_star)

    # F(ℓ) = f − f_p，关于 ℓ 严格递增（ℓ 大 ⇒ c 小 ⇒ 索松 ⇒ 弧垂大）。
    # 可行 ℓ ∈ [ℓ*, ∞)，故 F(ℓ*) = f_min − f_p ≤ 0 是天然的负号端；
    # 正号端向 ℓ 增大方向（松弛方向）找。
    def F(ell: float) -> float:
        try:
            v = _sag(ell, L, h, xi) - f_p
        except OverflowError:
            return math.inf
        # f_p 已校验为有限正数；+inf 表示弧垂已足够大，nan 也按“够大”处理。
        return math.inf if not math.isfinite(v) else v

    # 抛物极限（c 大、弧垂小）的初猜：f ≈ w x(L-x)/(2H) ⇒ ℓ ≈ sqrt(2f/(L·ξ(1−ξ)))
    xi_term = max(xi * (1.0 - xi), 1e-30)
    ell_guess = math.sqrt(2.0 * f_p / (L * xi_term))

    ell_star_cap = min(ell_star, _ELL_MAX) if math.isfinite(ell_star) else _ELL_MAX
    if math.isfinite(ell_star):
        ell_lo = ell_star_cap          # F(ell_lo) 已保证 ≤ 0
    else:
        ell_lo = min(ell_guess, _ELL_MAX)
        while ell_lo > 1e-300 and F(ell_lo) > 0.0:
            ell_lo *= 0.5              # 等高时向绷紧端退到 F<0

    ell_hi = max(ell_lo, ell_guess)
    guard = 0
    while F(ell_hi) <= 0.0 and ell_hi < _ELL_MAX:
        ell_hi = min(ell_hi * 2.0, _ELL_MAX)
        guard += 1
        if guard > 2000:
            break

    for _ in range(_BISECT_ITERS):
        mid = 0.5 * (ell_lo + ell_hi)
        if F(mid) <= 0.0:
            ell_lo = mid
        else:
            ell_hi = mid

    ell = 0.5 * (ell_lo + ell_hi)
    c = L / ell
    return cat.solve_with_c(geometry, c)


def forward_inverse_closure(geometry: SpanGeometry, H: float, x: float | None = None) -> dict:
    """同一套几何上，已知 H 正算得弧垂，再把该弧垂当测量值反演回来。

    返回 {x, forward_sag, recovered_H, relative_error}；
    relative_error 必须落在钉死的 CLOSURE_RTOL 内。
    """
    sol_fwd = cat.solve_with_H(geometry, H)  # H 校验 + 可达性检查都在这里
    L = geometry.span
    if x is None:
        x = L / 2.0
    validation.require_measure_position(x, L)

    f = sol_fwd.sag(x)
    sol_inv = calibrate(geometry, f, x)
    rel = abs(sol_inv.H - H) / abs(H)
    return {
        "x": x,
        "forward_sag": f,
        "recovered_H": sol_inv.H,
        "given_H": H,
        "relative_error": rel,
        "rtol": CLOSURE_RTOL,
        "passed": rel <= CLOSURE_RTOL,
    }
