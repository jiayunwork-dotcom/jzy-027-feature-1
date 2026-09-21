"""耐张段联合标定：公共水平张力的联合反演。

锁住的验收行为：
1. 单测量档退化为该档独立反演，与现有单档反演数值一致（同一套底层数学）；
2. 多测量档合成数据（同一 H 正算弧垂当实测）能精确找回公共 H，残差全在容差内；
3. 未测量档用公共 H 正算出完整曲线（最低点、索长、取样点高度与弧垂）随结果返回；
4. 某档在候选公共张力下物理不可行时，明确指出是哪一档、允许到多大量值；
5. 测量互相矛盾时判定不可标定并指名对不上的档，绝不返回折中值；
6. 容差可调：放宽后轻度不一致可标定，且解是极小极大中心（残差等幅反号）；
7. 新增能力不影响现有单档标定与正算接口的既有行为。
"""
from __future__ import annotations

import pytest

from app import catenary as cat
from app import invert, section
from app.catenary import SpanGeometry
from app.errors import (
    InvalidRequest,
    JointCalibrationConflict,
    SagHeightConflict,
    SectionSpanInfeasible,
)


def _geom(span, h, w=10.0):
    return SpanGeometry.create(span, h, w)


def _member(name, span, h, w=10.0, meas=None):
    return section.SectionMember(name=name, geometry=_geom(span, h, w), measurement=meas)


def _measured_member(name, span, h, w, H_true, x):
    """用已知公共 H 正算弧垂当实测，构造合成测量档。"""
    g = _geom(span, h, w)
    f = cat.solve_with_H(g, H_true).sag(x)
    return section.SectionMember(
        name=name, geometry=g, measurement=section.MemberMeasurement(x=x, sag=f)
    )


def _create_span(client, name, span, h, w=10.0):
    return client.post("/spans", json={
        "name": name, "span": span, "height_difference": h, "weight_per_length": w})


def _forward_sag(client, name, H, x):
    r = client.post("/forward", json={"span_name": name, "H": H, "samples": [x]})
    assert r.status_code == 200, r.get_data(as_text=True)
    return r.get_json()["curve"][0]["sag"]


# ---------------------------------------------------------------------------
# 1. 单测量档退化为独立反演，与现有单档反演数值一致
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("L,h,w,H,x", [
    (300.0, 35.0, 10.0, 5000.0, 120.0),
    (400.0, 0.0, 10.0, 2600.0, 200.0),
    (250.0, -60.0, 12.0, 4000.0, 200.0),
])
def test_single_measured_span_degenerates_to_single_inversion(L, h, w, H, x):
    g = _geom(L, h, w)
    f = cat.solve_with_H(g, H).sag(x)
    meas = section.MemberMeasurement(x=x, sag=f)

    # 段内还有其他只登记几何的档：退化结论不应受它们影响
    joint = section.calibrate_joint([
        section.SectionMember("M", g, meas),
        _member("U1", 280.0, -20.0, 9.0),
        _member("U2", 320.0, 0.0, 11.0),
    ])
    single = invert.calibrate(g, f, x)

    assert joint.H == pytest.approx(single.H, rel=1e-9)
    assert joint.H == pytest.approx(H, rel=1e-9)
    # 测量档曲线在测点上复述实测弧垂
    assert joint.solutions[0].sag(x) == pytest.approx(f, rel=1e-9)


def test_single_member_section_is_plain_inversion():
    g = _geom(300.0, 35.0, 10.0)
    f = cat.solve_with_H(g, 5000.0).sag(120.0)
    joint = section.calibrate_joint([
        section.SectionMember("only", g, section.MemberMeasurement(120.0, f))])
    assert joint.H == pytest.approx(invert.calibrate(g, f, 120.0).H, rel=1e-9)
    assert len(joint.solutions) == 1


# ---------------------------------------------------------------------------
# 2. 多测量档合成数据：精确找回公共 H，各档残差都在容差内
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("H_true", [900.0, 5200.0, 12000.0])
def test_joint_recovers_common_H_from_consistent_measurements(H_true):
    members = [
        _measured_member("A", 300.0, 35.0, 10.0, H_true, 120.0),
        _member("B", 250.0, -15.0, 9.0),                       # 只登记几何
        _measured_member("C", 400.0, 0.0, 12.0, H_true, 200.0),
        _measured_member("D", 280.0, -12.0, 9.0, H_true, 140.0),
    ]
    joint = section.calibrate_joint(members)

    assert joint.H == pytest.approx(H_true, rel=1e-9)
    for m, sol in zip(members, joint.solutions):
        assert sol.H == pytest.approx(joint.H, rel=1e-12)
        if m.measurement is not None:
            f = m.measurement.sag
            residual = sol.sag(m.measurement.x) - f
            assert abs(residual) <= joint.residual_rtol * f


def test_joint_recovery_is_exact_not_averaged():
    """合成数据下各档残差必须压到远小于容差——不是任何一档都不闭合的折中值。"""
    H_true = 5000.0
    members = [
        _measured_member("A", 300.0, 10.0, 10.0, H_true, 150.0),
        _measured_member("B", 280.0, -12.0, 9.0, H_true, 140.0),
        _measured_member("C", 350.0, 25.0, 11.0, H_true, 300.0),
    ]
    joint = section.calibrate_joint(members)
    for m, sol in zip(members, joint.solutions):
        rel = abs(sol.sag(m.measurement.x) - m.measurement.sag) / m.measurement.sag
        assert rel <= 1e-9  # 默认容差 1e-9 内精确闭合


# ---------------------------------------------------------------------------
# 3. 未测量档：用公共 H 正算的完整曲线随联合结果返回
# ---------------------------------------------------------------------------
def test_unmeasured_spans_get_forward_curves_at_common_H():
    H_true = 4200.0
    members = [
        _measured_member("A", 300.0, 35.0, 10.0, H_true, 120.0),
        _member("B", 250.0, -15.0, 9.0),
        _member("C", 320.0, 0.0, 11.0),
    ]
    joint = section.calibrate_joint(members)

    for m, sol in zip(members, joint.solutions):
        L = m.geometry.span
        # 每档都是完整物理解：最低点在档内、索长严格大于档距、端点闭合
        assert 0.0 <= sol.x0 <= L
        assert sol.length > L
        assert sol.y(0.0) == pytest.approx(0.0, abs=1e-9)
        assert sol.y(L) == pytest.approx(m.geometry.height_difference, rel=1e-9, abs=1e-9)
        # 与独立正算逐点一致（任意取样点的高度与弧垂）
        ref = cat.solve_with_H(m.geometry, joint.H)
        for x in (0.0, L / 3.0, L / 2.0, L):
            assert sol.sag(x) == pytest.approx(ref.sag(x), rel=1e-12, abs=1e-12)
            assert sol.y(x) == pytest.approx(ref.y(x), rel=1e-12, abs=1e-12)


# ---------------------------------------------------------------------------
# 4. 候选公共 H 下某档物理不可行：指名哪一档、允许到多大
# ---------------------------------------------------------------------------
def test_infeasible_span_is_identified_with_value():
    # S1 等高，实测对应 H≈50000（绷得很紧）；S2 高差 80m，物理上限约 5.8e3
    members = [
        _measured_member("S1", 300.0, 0.0, 10.0, 50000.0, 150.0),
        _member("S2", 300.0, 80.0, 10.0),
    ]
    ceiling_s2 = section.max_feasible_H(_geom(300.0, 80.0, 10.0))
    assert 0.0 < ceiling_s2 < 50000.0  # 构造前提：S2 撑不到候选张力

    with pytest.raises(SectionSpanInfeasible) as exc:
        section.calibrate_joint(members)
    d = exc.value.details
    assert d["reason"] == "section_span_infeasible"
    assert d["span"] == "S2"                       # 明确指出是哪一档
    assert d["max_feasible_H"] == pytest.approx(ceiling_s2, rel=1e-12)
    assert d["required_min_H"] > d["max_feasible_H"]  # 量值对比可复算
    assert d["required_min_by"] == "S1"


def test_measured_span_self_conflict_uses_single_span_error():
    # 单档自身“弧垂比可行下限还小”的矛盾，与单档反演同一个错误码
    g = _geom(300.0, 80.0, 10.0)
    f_min = cat.boundary_sag(300.0, 80.0, 0.5)
    members = [section.SectionMember("W", g, section.MemberMeasurement(150.0, f_min * 0.5))]
    with pytest.raises(SagHeightConflict) as exc:
        section.calibrate_joint(members)
    assert exc.value.details["reason"] == "sag_height_difference_conflict"
    assert exc.value.details["minimum_feasible_sag"] == pytest.approx(f_min, rel=1e-9)


# ---------------------------------------------------------------------------
# 5. 测量互相矛盾：判定不可标定，绝不返回折中值
# ---------------------------------------------------------------------------
def test_contradictory_measurements_are_uncalibratable():
    # 两档各自对应截然不同的张力（5000 与 9000），不存在公共解
    members = [
        _measured_member("A", 300.0, 10.0, 10.0, 5000.0, 150.0),
        _measured_member("B", 280.0, -12.0, 9.0, 9000.0, 140.0),
    ]
    with pytest.raises(JointCalibrationConflict) as exc:
        section.calibrate_joint(members)
    d = exc.value.details
    assert d["reason"] == "joint_calibration_conflict"
    assert d["required_min_H"] > d["allowed_max_H"]
    # 指名对不上的双方：B 要求大张力，A 只允许小张力
    assert {d["required_min_by"], d["allowed_max_by"]} == {"A", "B"}
    intervals = d["measured_span_H_intervals"]
    assert intervals["A"]["max_H"] < intervals["B"]["min_H"]


# ---------------------------------------------------------------------------
# 6. 容差可调：放宽后轻度不一致可标定，解是极小极大中心
# ---------------------------------------------------------------------------
def test_loosened_tolerance_accepts_and_centers_minimax():
    # 两档实测分别对应 H=5000 与 H=5001（相对偏差 2e-4）
    members = [
        _measured_member("A", 300.0, 10.0, 10.0, 5000.0, 150.0),
        _measured_member("B", 280.0, -12.0, 9.0, 5001.0, 140.0),
    ]
    # 默认容差 1e-9 下不可标定
    with pytest.raises(JointCalibrationConflict):
        section.calibrate_joint(members)

    rtol = 1e-3
    joint = section.calibrate_joint(members, residual_rtol=rtol)
    assert 5000.0 < joint.H < 5001.0
    rhos = []
    for m, sol in zip(members, joint.solutions):
        f = m.measurement.sag
        rho = (sol.sag(m.measurement.x) - f) / (rtol * f)
        assert abs(rho) <= 1.0 + 1e-12  # 残差都落在容差内
        rhos.append(rho)
    # 极小极大（Chebyshev）性质：两个归一化残差等幅反号
    assert rhos[0] == pytest.approx(-rhos[1], abs=1e-9)


def test_residual_rtol_must_be_a_fraction():
    members = [_measured_member("A", 300.0, 10.0, 10.0, 5000.0, 150.0)]
    for bad in (0.0, -1e-3, 1.0, 2.5, float("nan")):
        with pytest.raises(InvalidRequest):
            section.calibrate_joint(members, residual_rtol=bad)


def test_section_requires_at_least_one_measurement():
    with pytest.raises(InvalidRequest):
        section.calibrate_joint([_member("A", 300.0, 0.0), _member("B", 200.0, 5.0)])
    with pytest.raises(InvalidRequest):
        section.calibrate_joint([])


# ---------------------------------------------------------------------------
# 7. HTTP 全流程
# ---------------------------------------------------------------------------
def test_http_joint_calibrate_recovers_common_H(client):
    _create_span(client, "N1", 300.0, 35.0, 10.0)
    _create_span(client, "N2", 250.0, -15.0, 9.0)
    _create_span(client, "N3", 400.0, 0.0, 12.0)
    H_true = 5200.0
    sag1 = _forward_sag(client, "N1", H_true, 120.0)
    sag3 = _forward_sag(client, "N3", H_true, 200.0)

    r = client.post("/calibrate_section", json={
        "spans": [
            {"name": "N1", "measured_sag": sag1, "measurement_x": 120.0},
            {"name": "N2"},  # 只登记几何，不挂测量仪
            {"name": "N3", "measured_sag": sag3, "measurement_x": 200.0},
        ],
        "sample_count": 11,
    })
    assert r.status_code == 200, r.get_data(as_text=True)
    data = r.get_json()

    assert data["H"] == pytest.approx(H_true, rel=1e-9)
    assert data["residual_rtol"] == pytest.approx(1e-9)
    sol_meta = data["solution"]
    assert sol_meta["span_count"] == 3
    assert sol_meta["measured_span_count"] == 2
    assert sol_meta["feasible_H_interval"][0] <= data["H"] <= sol_meta["feasible_H_interval"][1]
    assert sol_meta["max_relative_residual"] <= data["residual_rtol"]
    assert sol_meta["all_residuals_within_tolerance"] is True

    blocks = {b["name"]: b for b in data["spans"]}
    assert [b["name"] for b in data["spans"]] == ["N1", "N2", "N3"]  # 保持成员顺序
    for name, L in (("N1", 300.0), ("N2", 250.0), ("N3", 400.0)):
        b = blocks[name]
        assert b["H"] == pytest.approx(data["H"], rel=1e-12)
        assert b["c"] == pytest.approx(data["H"] / b["geometry"]["weight_per_length"], rel=1e-12)
        assert 0.0 <= b["lowest_point"]["x"] <= L
        assert b["cable_length"] > L
        assert len(b["curve"]) == 11
        assert b["curve"][0]["x"] == 0.0 and b["curve"][-1]["x"] == L
        assert b["geometry"]["source"] == f"span:{name}"

    # 测量档：残差与容差逐档可复算
    for name, sag, x in (("N1", sag1, 120.0), ("N3", sag3, 200.0)):
        b = blocks[name]
        assert b["measured"] is True
        m = b["measurement"]
        assert m["x"] == x and m["measured_sag"] == sag
        assert m["computed_sag"] == pytest.approx(sag, rel=1e-9)
        assert abs(m["residual"]) <= m["tolerance"]
        assert m["within_tolerance"] is True

    # 未测量档：曲线与公共 H 的独立正算逐点一致
    b2 = blocks["N2"]
    assert b2["measured"] is False
    assert "measurement" not in b2
    fwd = client.post("/forward", json={"span_name": "N2", "H": data["H"],
                                        "sample_count": 11}).get_json()
    for p_joint, p_fwd in zip(b2["curve"], fwd["curve"]):
        assert p_joint["x"] == pytest.approx(p_fwd["x"], abs=1e-12)
        assert p_joint["y"] == pytest.approx(p_fwd["y"], rel=1e-9, abs=1e-9)
        assert p_joint["sag"] == pytest.approx(p_fwd["sag"], rel=1e-9, abs=1e-9)


def test_http_single_measured_member_matches_single_calibrate(client):
    """退化一致性（HTTP 层）：联合路径与现有单档接口给出同一个 H。"""
    _create_span(client, "D1", 300.0, 35.0, 10.0)
    _create_span(client, "D2", 280.0, -20.0, 9.0)
    sag = _forward_sag(client, "D1", 5000.0, 120.0)

    r_joint = client.post("/calibrate_section", json={"spans": [
        {"name": "D1", "measured_sag": sag, "measurement_x": 120.0},
        {"name": "D2"},
    ]})
    assert r_joint.status_code == 200, r_joint.get_data(as_text=True)

    r_single = client.post("/calibrate", json={
        "span_name": "D1", "measured_sag": sag, "measurement_x": 120.0})
    assert r_single.status_code == 200
    assert r_joint.get_json()["H"] == pytest.approx(r_single.get_json()["H"], rel=1e-9)


def test_http_infeasible_span_reports_span_and_value(client):
    _create_span(client, "L1", 300.0, 0.0, 10.0)
    _create_span(client, "L2", 300.0, 80.0, 10.0)
    sag = _forward_sag(client, "L1", 50000.0, 150.0)

    r = client.post("/calibrate_section", json={"spans": [
        {"name": "L1", "measured_sag": sag, "measurement_x": 150.0},
        {"name": "L2"},
    ]})
    assert r.status_code == 422
    body = r.get_json()
    assert body["error"] == "section_span_infeasible"
    assert body["reason"] == "section_span_infeasible"
    assert body["span"] == "L2"
    assert body["max_feasible_H"] > 0.0
    assert body["required_min_H"] > body["max_feasible_H"]
    assert body["required_min_by"] == "L1"


def test_http_contradictory_measurements_422_no_compromise(client):
    _create_span(client, "M1", 300.0, 10.0, 10.0)
    _create_span(client, "M2", 280.0, -12.0, 9.0)
    sag1 = _forward_sag(client, "M1", 5000.0, 150.0)
    sag2 = _forward_sag(client, "M2", 5400.0, 140.0)

    r = client.post("/calibrate_section", json={"spans": [
        {"name": "M1", "measured_sag": sag1, "measurement_x": 150.0},
        {"name": "M2", "measured_sag": sag2, "measurement_x": 140.0},
    ]})
    assert r.status_code == 422
    body = r.get_json()
    assert body["error"] == "joint_calibration_conflict"
    assert body["required_min_H"] > body["allowed_max_H"]
    assert {body["required_min_by"], body["allowed_max_by"]} == {"M1", "M2"}

    # 放宽容差后同一组数据可标定（容差语义可调）
    r2 = client.post("/calibrate_section", json={
        "spans": [
            {"name": "M1", "measured_sag": sag1, "measurement_x": 150.0},
            {"name": "M2", "measured_sag": sag2, "measurement_x": 140.0},
        ],
        "residual_rtol": 0.05,
    })
    assert r2.status_code == 200, r2.get_data(as_text=True)
    data2 = r2.get_json()
    assert data2["residual_rtol"] == pytest.approx(0.05)
    assert 5000.0 < data2["H"] < 5400.0
    assert data2["solution"]["all_residuals_within_tolerance"] is True


def test_http_section_member_self_conflict_422(client):
    _create_span(client, "W1", 300.0, 80.0, 10.0)
    r = client.post("/calibrate_section", json={"spans": [
        {"name": "W1", "measured_sag": 1.0, "measurement_x": 150.0}]})
    assert r.status_code == 422
    assert r.get_json()["error"] == "sag_height_difference_conflict"


def test_http_section_request_validation(client):
    _create_span(client, "V1", 300.0, 10.0, 10.0)
    sag = _forward_sag(client, "V1", 5000.0, 150.0)

    # 空成员列表 / 非列表
    assert client.post("/calibrate_section", json={"spans": []}).status_code == 400
    assert client.post("/calibrate_section", json={"spans": "V1"}).status_code == 400
    # 未登记档名 404
    r = client.post("/calibrate_section", json={"spans": [
        {"name": "GHOST", "measured_sag": sag, "measurement_x": 150.0}]})
    assert r.status_code == 404
    assert r.get_json()["error"] == "span_not_found"
    # 测量字段只给一半
    assert client.post("/calibrate_section", json={"spans": [
        {"name": "V1", "measured_sag": sag}]}).status_code == 400
    assert client.post("/calibrate_section", json={"spans": [
        {"name": "V1", "measurement_x": 150.0}]}).status_code == 400
    # 整段没有任何测量
    r = client.post("/calibrate_section", json={"spans": [{"name": "V1"}]})
    assert r.status_code == 400
    assert "实测" in r.get_json()["message"]
    # 档名重复
    assert client.post("/calibrate_section", json={"spans": [
        {"name": "V1", "measured_sag": sag, "measurement_x": 150.0},
        {"name": "V1"}]}).status_code == 400
    # 测量点越界 / 弧垂非正
    assert client.post("/calibrate_section", json={"spans": [
        {"name": "V1", "measured_sag": sag, "measurement_x": 300.0}]}).status_code == 400
    assert client.post("/calibrate_section", json={"spans": [
        {"name": "V1", "measured_sag": -1.0, "measurement_x": 150.0}]}).status_code == 400
    # 非法容差
    for bad in (0.0, 1.0, -0.5):
        assert client.post("/calibrate_section", json={
            "spans": [{"name": "V1", "measured_sag": sag, "measurement_x": 150.0}],
            "residual_rtol": bad}).status_code == 400


def test_http_section_per_member_samples(client):
    _create_span(client, "P1", 300.0, 0.0, 10.0)
    _create_span(client, "P2", 200.0, 0.0, 10.0)
    sag = _forward_sag(client, "P1", 5000.0, 150.0)
    r = client.post("/calibrate_section", json={
        "spans": [
            {"name": "P1", "measured_sag": sag, "measurement_x": 150.0,
             "samples": [0.0, 150.0, 300.0]},
            {"name": "P2"},
        ],
        "sample_count": 5,
    })
    assert r.status_code == 200, r.get_data(as_text=True)
    blocks = {b["name"]: b for b in r.get_json()["spans"]}
    assert [p["x"] for p in blocks["P1"]["curve"]] == [0.0, 150.0, 300.0]
    assert len(blocks["P2"]["curve"]) == 5


# ---------------------------------------------------------------------------
# 8. 现有单档标定与正算接口行为不变
# ---------------------------------------------------------------------------
def test_existing_single_span_endpoints_unchanged(client):
    r = client.post("/calibrate", json={
        "span": 300.0, "height_difference": 35.0, "weight_per_length": 10.0,
        "measured_sag": 18.5, "measurement_x": 120.0})
    assert r.status_code == 200, r.get_data(as_text=True)
    assert set(r.get_json()) == {
        "H", "c", "lowest_point", "cable_length", "supports", "geometry",
        "measurement", "curve", "forward_inverse_closure",
    }

    r2 = client.post("/forward", json={
        "span": 300.0, "height_difference": 35.0, "weight_per_length": 10.0,
        "H": 5000.0})
    assert r2.status_code == 200, r2.get_data(as_text=True)
    assert set(r2.get_json()) == {
        "H", "c", "lowest_point", "cable_length", "supports", "geometry",
        "curve", "forward_inverse_closure",
    }
    assert r2.get_json()["forward_inverse_closure"]["passed"] is True
