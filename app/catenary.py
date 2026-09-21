"""双曲悬链正算（本服务唯一的曲线数学，禁止使用抛物近似实现）。

坐标系
------
左支座取在 ``x = 0, y = 0``，右支座在 ``x = L, y = h``，其中
``L`` 为档距、``h`` 为带符号的两端高差（右支座高取正）。

柔索形状参数 ``c = H / w``（H 为水平张力，w 为单位长度重量）。
最低点（顶点）在 ``x0 = a·c``，曲线为

    y(x) = c · ( cosh((x - x0)/c) - cosh(a) )

两端无因次水平坐标（从最低点起算）分别为 ``-a`` 与 ``b``：

    a + b = L/c = ℓ,
    cosh b - cosh a = h/c = δ,
    a = ℓ/2 - s,  b = ℓ/2 + s,  s = asinh( δ / (2 sinh(ℓ/2)) )

索长 ``S = c(sinh b - sinh(-a)) = c(sinh a + sinh b)``。
弧垂一律按“弦线 − 索”定义，等高跨中即 c(cosh(L/2c) − 1)。

可行域（最低点必须落在档内）
----------------------------
``a > 0`` 且 ``b > 0``，等价于 ``|δ| ≤ 2 sinh²(ℓ/2)``；
等号对应最低点正好贴在较低支座上。给定 H 正算越界即“够不着”。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from . import validation
from .errors import SupportsUnreachable

# math.cosh 的浮点溢出界，超过就走指数分支，避免 OverflowError / 无穷相减。
_EXP_LIMIT = 700.0


@dataclass(frozen=True)
class SpanGeometry:
    """一档的跨距几何：档距 L、带符号高差 h、单位长度重量 w。"""

    span: float
    height_difference: float
    weight_per_length: float

    @classmethod
    def create(cls, span: float, height_difference: float, weight_per_length: float) -> "SpanGeometry":
        L = validation.require_positive(span, "档距")
        h = validation.require_height_difference(height_difference)
        w = validation.require_positive(weight_per_length, "单位长度重量")
        return cls(L, h, w)

    def to_dict(self) -> dict:
        return {
            "span": self.span,
            "height_difference": self.height_difference,
            "weight_per_length": self.weight_per_length,
        }


@dataclass(frozen=True)
class CablePoint:
    x: float
    y: float       # 索相对左支座的高度
    chord: float   # 弦线在该 x 处的高度
    sag: float     # 相对弦的弧垂（弦高 − 索高，恒非负）


@dataclass(frozen=True)
class CatenarySolution:
    geometry: SpanGeometry
    c: float
    s: float       # 顶点偏移无因次量 s = asinh(h/(2c sinh(ℓ/2)))
    a: float       # 左支座相对最低点的无因次距离（半档距 − s）
    b: float       # 右支座相对最低点的无因次距离（半档距 + s）
    x0: float      # 最低点水平位置
    y0: float      # 最低点相对左支座的高度
    length: float  # 全档索长

    @property
    def H(self) -> float:
        return self.geometry.weight_per_length * self.c

    @property
    def ell(self) -> float:
        return self.geometry.span / self.c

    # ---- 单点正算（一律走乘积恒等式，端点严格闭合） ----------------------
    def chord_height(self, x: float) -> float:
        L, h = self.geometry.span, self.geometry.height_difference
        return h * x / L

    def sag(self, x: float) -> float:
        L = self.geometry.span
        return self.c * dimensionless_sag(L, self.geometry.height_difference, self.c, x / L, self.s)

    def y(self, x: float) -> float:
        return self.chord_height(x) - self.sag(x)

    def point(self, x: float) -> CablePoint:
        return CablePoint(x=x, y=self.y(x), chord=self.chord_height(x), sag=self.sag(x))

    def points(self, xs) -> list[CablePoint]:
        return [self.point(float(x)) for x in xs]


# ---------------------------------------------------------------------------
# 无因次端点参数
# ---------------------------------------------------------------------------
def endpoint_dimensionless(L: float, h: float, c: float) -> tuple[float, float, float, float, float]:
    """返回 (ℓ, δ, s, a, b)。

    由 cosh b − cosh a = δ 与 b−a = ℓ，和差化积得
        sinh s = h / (2c·sinh(ℓ/2))，
    a = ℓ/2 − s（左支座到最低点的无因次距），b = ℓ/2 + s。
    s 直接用量纲值 h、c 计算，避免先算 δ=h/c 再除大数造成精度损失。
    """
    ell = L / c
    delta = h / c
    half = ell / 2.0
    s = _vertex_shift_physical(h, c, half)
    return ell, delta, s, half - s, half + s


def is_reachable(L: float, h: float, c: float) -> bool:
    """给定 c 时索能否在最低点落档内的情况下同时够到两个支座。"""
    ell = L / c
    delta_abs = abs(h) / c
    try:
        limit = 2.0 * math.sinh(ell / 2.0) ** 2
    except OverflowError:
        return True  # ℓ 极大，允许高差远超任何有限值
    return delta_abs <= limit


def boundary_ell(L: float, h: float) -> float:
    """最低点恰好贴在较低支座时的临界无因次档距 ℓ*；等高时无界（+inf）。

    临界条件 |h|/c = cosh(L/c) − 1。令 u = L/c、k = |h|/L，解
    cosh u − 1 = k·u 的非零正根。
    """
    if h == 0.0:
        return math.inf
    k = abs(h) / L

    def f(u: float) -> float:
        return math.cosh(u) - 1.0 - k * u

    hi = 1.0
    guard = 0
    while f(hi) <= 0.0:
        hi *= 2.0
        guard += 1
        if guard > 1000:
            raise OverflowError("临界档距搜索越界")
    lo = 1e-12  # f 在 0+ 附近 ≈ -k·u < 0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if f(mid) <= 0.0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ---------------------------------------------------------------------------
# 由 c 正算整条悬链
# ---------------------------------------------------------------------------
def solve_with_c(geometry: SpanGeometry, c: float) -> CatenarySolution:
    """校验后按形状参数 c 铺整条曲线；最低点落到档外时抛 SupportsUnreachable。"""
    c = validation.require_positive(c, "形状参数 c")
    L, h, w = geometry.span, geometry.height_difference, geometry.weight_per_length

    if not is_reachable(L, h, c):
        ell_star = boundary_ell(L, h)
        c_star = L / ell_star if math.isfinite(ell_star) else math.inf
        raise SupportsUnreachable(
            f"高差 {h:g} 对当前水平张力太大：最低点落到档外，"
            f"索无法在该张力下同时够到两个支座（H 必须不超过 {w * c_star:.6g}）",
            H=w * c,
            H_max=w * c_star,
        )

    ell, delta, s, a, b = endpoint_dimensionless(L, h, c)
    # 数值上恰好压在临界上时，把落到档外的端点夹回支座点。
    if a < 0.0:
        a = 0.0
    if b < 0.0:
        b = 0.0

    x0 = a * c
    # 最低点高度 = −c(cosh a − 1) = −2c sinh²(a/2)，无减法形式。
    try:
        y0 = -2.0 * c * math.sinh(a / 2.0) ** 2
    except OverflowError:
        y0 = -math.inf
    length = c * _sinh_sum_stable(a, b)  # c(sinh a + sinh b)
    return CatenarySolution(geometry=geometry, c=c, s=s, a=a, b=b, x0=x0, y0=y0, length=length)


def solve_with_H(geometry: SpanGeometry, H: float) -> CatenarySolution:
    """按显式给定的水平张力 H 正算。"""
    H = validation.require_positive(H, "水平张力 H")
    c = H / geometry.weight_per_length
    return solve_with_c(geometry, c)


def level_span_midpoint_sag(c: float, L: float) -> float:
    """等高档跨中弧垂闭式：c·(cosh(L/(2c)) − 1)。"""
    return c * (math.cosh(L / (2.0 * c)) - 1.0)


# ---------------------------------------------------------------------------
# 无因次弧垂（反演与正算共用，保证同一套双曲关系）
# ---------------------------------------------------------------------------
def dimensionless_sag(L: float, h: float, c: float, xi: float,
                      s: float | None = None) -> float:
    """返回无因次弧垂：测点弧垂 / c（ξ = x/L）。

    用和差化积恒等式求值，全程没有“两个相近大数相减”：

        弧垂 / c = δξ + cosh a − cosh u
                 = δξ − 2·sinh(ξℓ/2)·sinh(s − (1−ξ)ℓ/2)

    ξ=0 时第一因子为 0，左端弧垂严格为 0；ξ=1 时是纯乘积，
    右端弧垂也严格为 0——小高差的信息全部由 s 直接携带。
    """
    ell = L / c
    if s is None:
        s = _vertex_shift_physical(h, c, ell / 2.0)
    p = xi * ell / 2.0
    q = s - (1.0 - xi) * ell / 2.0
    return (h / c) * xi - 2.0 * _sinh_product(p, q)


def sag_at(ell: float, L: float, h: float, xi: float) -> float:
    """有量纲弧垂 f = c·g = (L/ℓ)·dimensionless_sag(...)。"""
    c = L / ell
    return c * dimensionless_sag(L, h, c, xi)


def boundary_sag(L: float, h: float, xi: float) -> float:
    """最低点贴在较低支座（可行域边界）时，测点上的弧垂，即可行弧垂的下限。"""
    ell_star = boundary_ell(L, h)
    if not math.isfinite(ell_star):
        return 0.0  # 等高时下限就是 0（张力无穷大的极限）
    return sag_at(ell_star, L, h, xi)


# ---------------------------------------------------------------------------
# 大参数安全双曲工具
# ---------------------------------------------------------------------------
def _vertex_shift_physical(h: float, c: float, half: float) -> float:
    """s = asinh(h/(2c·sinh(half)))，直接用量纲值，避免先算 h/c 丢精度。

    half 大到 sinh 溢出时走对数空间：
        2 sinh half ≈ e^half，故比值 ≈ (h/c)·e^(-half)，
    这个小量决定大垂度档两端边界的抵消精度，不能当 0 丢掉。
    """
    if half < 100.0:
        try:
            return math.asinh(h / (2.0 * c * math.sinh(half)))
        except OverflowError:
            pass
    if h == 0.0:
        return 0.0
    try:
        log_ratio = math.log(abs(h)) - math.log(c) - half
        if log_ratio < -1.0:
            return math.copysign(math.exp(log_ratio), h)
        return math.copysign(math.asinh(math.exp(log_ratio)), h)
    except OverflowError:
        return math.copysign(half, h)


def _sinh_product(p: float, q: float) -> float:
    """数值稳定地求 sinh(p)·sinh(q)。

    - 小参数直接相乘；
    - 大参数（sinh 会溢出）走指数分解 sinh x = (e^x/2)(1 − e^{-2x})，
      让大数只出现在最后一次 exp 里，避免中途溢出后把小量信息吞掉。
    """
    ap, aq = abs(p), abs(q)
    if max(ap, aq) <= 500.0:
        return math.sinh(p) * math.sinh(q)

    sign = math.copysign(1.0, p) * math.copysign(1.0, q)
    hi, lo = (ap, aq) if ap >= aq else (aq, ap)
    tail_hi = -math.expm1(-2.0 * hi)  # 1 − e^{-2hi}
    try:
        if lo <= 500.0:
            # 大边指数化，小边保留 sinh(lo)
            return sign * 0.5 * math.exp(hi) * tail_hi * math.sinh(lo)
        # 两边都大
        tail_lo = -math.expm1(-2.0 * lo)
        return sign * 0.25 * math.exp(hi + lo) * tail_hi * tail_lo
    except OverflowError:
        return sign * math.inf


def _sinh_sum_stable(a: float, b: float) -> float:
    """sinh a + sinh b（a、b 非负），用 2 sinh((a+b)/2) cosh((a−b)/2)。

    大参数时前者是和、后者接近 1，不会把“两个大数相加”的低位磨光，
    溢出时返回 inf。
    """
    a = max(a, 0.0)
    b = max(b, 0.0)
    if max(a, b) <= _EXP_LIMIT:
        return 2.0 * math.sinh((a + b) / 2.0) * math.cosh(abs(a - b) / 2.0)
    try:
        return 2.0 * math.sinh((a + b) / 2.0) * math.cosh(abs(a - b) / 2.0)
    except OverflowError:
        return math.inf
