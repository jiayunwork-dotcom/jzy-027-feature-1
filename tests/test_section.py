"""耐张段联合标定的验收测试。

覆盖需求点名的五条验收：
1. 单测量档退化到独立反演结果，与现有单档反演在数值上一致；
2. 多测量档合成数据（同一 H 正算出弧垂当测量）必须找回公共 H，
   且各档残差落在容差内；未测量档随结果给出完整曲线；
3. 某档在候选公共张力下不可行时，明确定位到具体哪一档、什么量值；
4. 测量数据互相矛盾时判定为不可标定，绝不返回折中值；
5. 新增能力不影响现有单档标定与正算接口的既有行为。
"""
from __future__ import annotations

import pytest

from app import catenary as cat
from app import invert
from app import section
from app.catenary import SpanGeometry
from app.errors import (
    SectionMeasurementBelowMinimum,
    SectionMeasurementsInconsistent,
    SectionSpanInfeasible,
)

# ---------------------------------------------------------------------------
# 公共道具：一个三档耐张段，几何互不相同（档距/高差/w 都不同）
# ---------------------------------------------------------------------------
GEOMS = {
    "S1": (300.0, 0.0, 10.0),
    "S2": (250.0, 30.0, 9.0),
    "S3": (400.0, -60.0, 12.0),
}
H_TRUE = 2500.0


def _geom(name):
    L, h, w = GEOMS[name]
    return SpanGeometry.create(L, h, w)


def _members(names=("S1", "S2", "S3")):
    return [(n, _geom(n)) for n in names]


def _measured(name, H, xi=0.5):
    """用已知 H 正算出测点弧垂，当作实测输入（合成数据）。"""
    g = _geom(name)
    x = g.span * xi
    f = cat.solve_with_H(g, H).sag(x)
    return section.MeasuredSpan(name=name, geometry=g, measured_sag=f, measurement_x=x)


def _register(client, name):
    L, h, w = GEOMS[name]
    r = client.post("/spans", json={
        "name": name, "span": L, "height_difference": h, "weight_per_length": w})
    assert r.status_code == 201, r.get_json()


def _register_all(client):
    for n in GEOMS:
        _register(client, n)


# ---------------------------------------------------------------------------
# 1. 单测量档退化：与现有单档反演数值一致，且共用同一套底层数学
# ---------------------------------------------------------------------------
def test_single_measurement_degenerates_bit_exact_to_single_span_inversion():
    m = _measured("S2", H_TRUE, xi=0.4)
    joint = section.calibrate_section([m], _members(), rtol=1e-3)

    solo = invert.calibrate(m.geometry, m.measured_sag, m.measurement_x)
    # 同一套底层数学自然退化：逐位一致，不是“约等于”
    assert joint.H == solo.H
    # 未测量档也按公共 H 铺了曲线
    assert {name for name, _ in joint.curves} == {"S1", "S2", "S3"}
    for name, sol in joint.curves:
        assert sol.H == pytest.approx(joint.H, rel=1e-15)
        assert 0.0 <= sol.x0 <= sol.geometry.span


def test_single_measurement_http_matches_existing_calibrate(client):
    _register_all(client)
    m = _measured("S1", H_TRUE, xi=0.5)

    r_joint = client.post("/calibrate-section", json={
        "spans": ["S1", "S2", "S3"],
        "measurements": [{"span": "S1", "measured_sag": m.measured_sag,
                          "measurement_x": m.measurement_x}],
    })
    assert r_joint.status_code == 200, r_joint.get_json()
    r_solo = client.post("/calibrate", json={
        "span_name": "S1", "measured_sag": m.measured_sag,
        "measurement_x": m.measurement_x})
    assert r_solo.status_code == 200, r_solo.get_json()

    assert r_joint.get_json()["H"] == pytest.approx(r_solo.get_json()["H"], rel=1e-12)


# ---------------------------------------------------------------------------
# 2. 多测量档合成数据：找回公共 H，残差全在容差内，未测量档曲线完整
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rtol", [1e-3, 1e-6])
def test_joint_calibration_recovers_common_H(rtol):
    measured = [_measured("S1", H_TRUE, xi=0.4), _measured("S3", H_TRUE, xi=0.6)]
    joint = section.calibrate_section(measured, _members(), rtol=rtol)

    assert abs(joint.H - H_TRUE) / H_TRUE <= 1e-9
    # 各测量档残差都落在容差内（合成数据下应接近 0）
    for fit in joint.solution.fits:
        assert fit.within_tolerance
        assert abs(fit.normalized_residual) <= 1.0
        assert abs(fit.residual) <= fit.tolerance * 1e-6
    # 联合闭合验收：正算再联合反演必须回到同一个 H
    assert joint.closure["passed"] is True
    assert joint.closure["relative_error"] <= section.CLOSURE_RTOL


def test_unmeasured_span_gets_full_forward_curve_at_common_H():
    measured = [_measured("S1", H_TRUE), _measured("S2", H_TRUE)]
    joint = section.calibrate_section(measured, _members(), rtol=1e-3)
    curves = dict(joint.curves)

    # S3 未提交测量，但必须按公共 H 正算出完整曲线
    sol = curves["S3"]
    direct = cat.solve_with_H(_geom("S3"), joint.H)
    assert sol.x0 == pytest.approx(direct.x0, rel=1e-15)
    assert sol.length == pytest.approx(direct.length, rel=1e-15)
    for xi in (0.13, 0.5, 0.87):
        x = sol.geometry.span * xi
        assert sol.sag(x) == pytest.approx(direct.sag(x), rel=1e-15)
        assert sol.y(x) == pytest.approx(direct.y(x), rel=1e-15)
    assert 0.0 < sol.x0 < sol.geometry.span
    assert sol.length > sol.geometry.span


def test_joint_http_response_structure_and_per_span_curves(client):
    _register_all(client)
    m1, m3 = _measured("S1", H_TRUE, xi=0.4), _measured("S3", H_TRUE, xi=0.6)
    r = client.post("/calibrate-section", json={
        "spans": ["S1", "S2", "S3"],
        "measurements": [
            {"span": "S1", "measured_sag": m1.measured_sag, "measurement_x": m1.measurement_x},
            {"span": "S3", "measured_sag": m3.measured_sag, "measurement_x": m3.measurement_x},
        ],
        "sample_count": 5,
    })
    assert r.status_code == 200, r.get_json()
    d = r.get_json()

    assert d["H"] == pytest.approx(H_TRUE, rel=1e-9)
    assert d["tension_section"]["measured_spans"] == ["S1", "S3"]
    assert d["tension_section"]["member_count"] == 3
    assert d["max_normalized_residual"] <= 1.0
    assert d["forward_inverse_closure"]["passed"] is True

    by_name = {s["name"]: s for s in d["spans"]}
    assert set(by_name) == {"S1", "S2", "S3"}
    # 测量档带残差诊断，未测量档不带但曲线完整
    assert by_name["S1"]["measured"] is True
    assert by_name["S2"]["measured"] is False
    assert "measurement" not in by_name["S2"]
    for name, entry in by_name.items():
        assert len(entry["curve"]) == 5
        assert entry["H"] == pytest.approx(d["H"], rel=1e-15)
        L = GEOMS[name][0]
        assert 0.0 <= entry["lowest_point"]["x"] <= L
        assert entry["cable_length"] > L
    m = by_name["S1"]["measurement"]
    assert m["within_tolerance"] is True
    assert abs(m["normalized_residual"]) <= 1.0
    assert m["computed_sag"] == pytest.approx(m["measured_sag"], rel=1e-9)

    # 未测量档的曲线与直接用公共 H 走 /forward 一致
    r_fwd = client.post("/forward", json={"span_name": "S2", "H": d["H"],
                                          "sample_count": 5})
    fwd_pts = {p["x"]: p for p in r_fwd.get_json()["curve"]}
    for p in by_name["S2"]["curve"]:
        assert p["sag"] == pytest.approx(fwd_pts[p["x"]]["sag"], rel=1e-12)
        assert p["y"] == pytest.approx(fwd_pts[p["x"]]["y"], rel=1e-12)


def test_tolerance_semantics_are_exact_and_recomputable():
    """扰动在容差内→成功且残差可复算；扰动出容差→判矛盾。"""
    m1 = _measured("S1", H_TRUE)
    base = _measured("S2", H_TRUE)
    rtol = 1e-3

    # S2 的弧垂人为加 0.5‰（在 1‰ 容差内）：应成功，且残差可复算
    perturbed = section.MeasuredSpan("S2", base.geometry,
                                     base.measured_sag * (1.0 + 5e-4),
                                     base.measurement_x)
    joint = section.calibrate_section([m1, perturbed], _members(), rtol=rtol)
    fit = {f.name: f for f in joint.solution.fits}["S2"]
    assert fit.tolerance == pytest.approx(rtol * perturbed.measured_sag, rel=1e-15)
    assert fit.normalized_residual == pytest.approx(fit.residual / fit.tolerance, rel=1e-12)
    assert abs(fit.normalized_residual) <= 1.0

    # 加 5‰（远超容差）：两个测量档对不上，必须判矛盾
    contradicted = section.MeasuredSpan("S2", base.geometry,
                                        base.measured_sag * (1.0 + 5e-3),
                                        base.measurement_x)
    with pytest.raises(SectionMeasurementsInconsistent):
        section.calibrate_section([m1, contradicted], _members(), rtol=rtol)


# ---------------------------------------------------------------------------
# 3. 候选公共张力下某档不可行：点名到档、给出量值
# ---------------------------------------------------------------------------
def test_infeasible_member_span_is_identified():
    steep = SpanGeometry.create(300.0, 80.0, 10.0)  # 大高差档，H_max 有限
    members = [("S1", _geom("S1")), ("SX", steep)]
    H_demanded = 8000.0
    assert H_demanded > section.max_feasible_H(steep)

    m = _measured("S1", H_demanded)
    with pytest.raises(SectionSpanInfeasible) as exc:
        section.calibrate_section([m], members, rtol=1e-3)
    d = exc.value.details
    assert d["reason"] == "span_infeasible_at_tension"
    assert d["span"] == "SX"                      # 明确是哪一档
    assert d["required_H"] == pytest.approx(H_demanded, rel=1e-9)
    assert d["max_feasible_H"] == pytest.approx(section.max_feasible_H(steep), rel=1e-12)
    assert d["max_feasible_H"] < d["required_H"]  # 量值关系一目了然


def test_infeasible_member_span_http_422(client):
    _register(client, "S1")
    r = client.post("/spans", json={"name": "SX", "span": 300.0,
                                    "height_difference": 80.0, "weight_per_length": 10.0})
    assert r.status_code == 201
    m = _measured("S1", 8000.0)
    r = client.post("/calibrate-section", json={
        "spans": ["S1", "SX"],
        "measurements": [{"span": "S1", "measured_sag": m.measured_sag,
                          "measurement_x": m.measurement_x}],
    })
    assert r.status_code == 422
    body = r.get_json()
    assert body["error"] == "section_span_infeasible"
    assert body["span"] == "SX"
    assert body["max_feasible_H"] < body["required_H"]
    assert "H" not in body  # 不可行时不返回任何“公共解”


def test_feasible_borderline_tension_still_succeeds():
    """贴着可行上限以内时必须正常出解（判误报与漏报一样都是错）。"""
    steep = SpanGeometry.create(300.0, 80.0, 10.0)
    members = [("S1", _geom("S1")), ("SX", steep)]
    H_ok = section.max_feasible_H(steep) * 0.9
    m = _measured("S1", H_ok)
    joint = section.calibrate_section([m], members, rtol=1e-3)
    assert joint.H == pytest.approx(H_ok, rel=1e-9)
    assert joint.closure["passed"] is True


# ---------------------------------------------------------------------------
# 4. 测量互相矛盾：判不可标定，绝不给折中值
# ---------------------------------------------------------------------------
def test_contradictory_measurements_are_rejected_not_averaged():
    m1 = _measured("S1", 2500.0)
    m2 = _measured("S2", 3200.0)  # 与 S1 矛盾的“实测”
    with pytest.raises(SectionMeasurementsInconsistent) as exc:
        section.calibrate_section([m1, m2], _members(), rtol=1e-3)
    d = exc.value.details
    assert d["reason"] == "measurements_inconsistent"
    # 说清是哪些档对不上，并给出各档独立反演值供核查
    assert set(d["conflicting_spans"]) == {"S1", "S2"}
    implied = {m["span"]: m["implied_H"] for m in d["measurements"]}
    assert implied["S1"] == pytest.approx(2500.0, rel=1e-9)
    assert implied["S2"] == pytest.approx(3200.0, rel=1e-9)


def test_contradictory_measurements_http_422_returns_no_compromise(client):
    _register_all(client)
    m1, m2 = _measured("S1", 2500.0), _measured("S2", 3200.0)
    r = client.post("/calibrate-section", json={
        "spans": ["S1", "S2", "S3"],
        "measurements": [
            {"span": "S1", "measured_sag": m1.measured_sag, "measurement_x": m1.measurement_x},
            {"span": "S2", "measured_sag": m2.measured_sag, "measurement_x": m2.measurement_x},
        ],
    })
    assert r.status_code == 422
    body = r.get_json()
    assert body["error"] == "section_measurements_inconsistent"
    assert set(body["conflicting_spans"]) == {"S1", "S2"}
    assert "H" not in body  # 绝不返回折中的公共 H


def test_measurement_below_span_minimum_is_identified():
    bad = section.MeasuredSpan("S3", _geom("S3"), measured_sag=1.0,
                               measurement_x=200.0)
    f_min = cat.boundary_sag(400.0, -60.0, 0.5)
    assert 1.0 < f_min
    with pytest.raises(SectionMeasurementBelowMinimum) as exc:
        section.calibrate_section([bad], _members(), rtol=1e-3)
    assert exc.value.details["span"] == "S3"
    assert exc.value.details["minimum_feasible_sag"] == pytest.approx(f_min, rel=1e-9)


# ---------------------------------------------------------------------------
# 5. 与现有单档能力共存：旧接口行为不变，新接口按需选用
# ---------------------------------------------------------------------------
def test_existing_endpoints_unaffected_by_section_capability(client):
    _register(client, "S1")
    # 单档标定照旧
    r = client.post("/calibrate", json={"span_name": "S1", "measured_sag": 44.564143523500285,
                                        "measurement_x": 150.0})
    assert r.status_code == 200
    assert r.get_json()["forward_inverse_closure"]["passed"] is True
    # 正算照旧
    r2 = client.post("/forward", json={"span_name": "S1", "H": 2500.0})
    assert r2.status_code == 200
    assert r2.get_json()["H"] == pytest.approx(2500.0)
    # 旧错误语义照旧：弧垂与高差矛盾仍是单档错误码
    r3 = client.post("/calibrate", json={"span": 300.0, "height_difference": 80.0,
                                         "weight_per_length": 10.0,
                                         "measured_sag": 5.0, "measurement_x": 150.0})
    assert r3.status_code == 422
    assert r3.get_json()["error"] == "sag_height_difference_conflict"


# ---------------------------------------------------------------------------
# 入参校验
# ---------------------------------------------------------------------------
def test_section_request_validation(client):
    _register_all(client)
    m = _measured("S1", H_TRUE)
    good_meas = [{"span": "S1", "measured_sag": m.measured_sag,
                  "measurement_x": m.measurement_x}]

    # 未知档名 → 404
    r = client.post("/calibrate-section", json={
        "spans": ["S1", "NOPE"], "measurements": good_meas})
    assert r.status_code == 404
    assert r.get_json()["error"] == "span_not_found"

    # 测量指向非成员档 → 400
    r = client.post("/calibrate-section", json={
        "spans": ["S1"], "measurements": [
            {"span": "S2", "measured_sag": 1.0, "measurement_x": 100.0}]})
    assert r.status_code == 400
    assert "不在本耐张段" in r.get_json()["message"]

    # 同一档重复测量 → 400
    r = client.post("/calibrate-section", json={
        "spans": ["S1"], "measurements": good_meas + good_meas})
    assert r.status_code == 400
    assert "多条测量" in r.get_json()["message"]

    # 空测量 / 空成员 / 重复成员 → 400
    for bad in ({"spans": ["S1"], "measurements": []},
                {"spans": [], "measurements": good_meas},
                {"spans": ["S1", "S1"], "measurements": good_meas}):
        r = client.post("/calibrate-section", json=bad)
        assert r.status_code == 400, bad

    # 弧垂非正 / 测点越界 / 容差非正 → 400
    r = client.post("/calibrate-section", json={
        "spans": ["S1"], "measurements": [
            {"span": "S1", "measured_sag": 0.0, "measurement_x": 150.0}]})
    assert r.status_code == 400
    r = client.post("/calibrate-section", json={
        "spans": ["S1"], "measurements": [
            {"span": "S1", "measured_sag": 10.0, "measurement_x": 300.0}]})
    assert r.status_code == 400
    r = client.post("/calibrate-section", json={
        "spans": ["S1"], "measurements": good_meas, "residual_rtol": -1e-3})
    assert r.status_code == 400


def test_section_custom_samples_per_span(client):
    _register_all(client)
    m = _measured("S1", H_TRUE)
    r = client.post("/calibrate-section", json={
        "spans": ["S1", "S2"],
        "measurements": [{"span": "S1", "measured_sag": m.measured_sag,
                          "measurement_x": m.measurement_x}],
        "samples": {"S2": [0.0, 125.0, 250.0]},
        "sample_count": 3,
    })
    assert r.status_code == 200, r.get_json()
    by_name = {s["name"]: s for s in r.get_json()["spans"]}
    assert [p["x"] for p in by_name["S2"]["curve"]] == [0.0, 125.0, 250.0]
    assert [p["x"] for p in by_name["S1"]["curve"]] == [0.0, 150.0, 300.0]

    # samples 点名非成员档 → 400
    r = client.post("/calibrate-section", json={
        "spans": ["S1"], "measurements": [
            {"span": "S1", "measured_sag": m.measured_sag, "measurement_x": m.measurement_x}],
        "samples": {"S2": [1.0]}})
    assert r.status_code == 400


def test_joint_math_stays_hyperbolic_at_large_sag():
    """大垂度段上联合反演同样严格走双曲：抛物值系统偏短，联合解仍闭合。"""
    gA = SpanGeometry.create(100.0, 5.0, 10.0)   # 弧垂/档距 > 0.3 的大垂度档
    gB = SpanGeometry.create(120.0, -8.0, 10.0)
    members = [("A", gA), ("B", gB)]
    H = 400.0
    mA = section.MeasuredSpan("A", gA, cat.solve_with_H(gA, H).sag(50.0), 50.0)
    mB = section.MeasuredSpan("B", gB, cat.solve_with_H(gB, H).sag(60.0), 60.0)
    assert mA.measured_sag / gA.span > 0.3

    joint = section.calibrate_section([mA, mB], members, rtol=1e-6)
    assert abs(joint.H - H) / H <= 1e-9
    assert joint.closure["passed"] is True
    # 抛物近似在这种垂度下明显偏短，联合解绝不是线性化凑出来的
    parabolic = gA.weight_per_length * gA.span**2 / (8.0 * joint.H)
    assert parabolic < mA.measured_sag * 0.9
