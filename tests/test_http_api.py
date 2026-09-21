"""HTTP 行为测试：登记 / 列档 / 标定 / 正算 / 错误原因。"""
from __future__ import annotations

import math

import pytest


def _create_span(client, name="A1", span=300.0, h=0.0, w=10.0, **extra):
    body = {"name": name, "span": span, "height_difference": h,
            "weight_per_length": w, **extra}
    return client.post("/spans", json=body)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json() == {"status": "ok"}


def test_register_and_list_spans_shows_full_geometry(client):
    assert _create_span(client, "A1", 300.0, 40.0, 12.5).status_code == 201
    r = client.get("/spans")
    spans = r.get_json()["spans"]
    assert len(spans) == 1
    assert spans[0] == {
        "name": "A1",
        "span": 300.0,
        "height_difference": 40.0,
        "weight_per_length": 12.5,
    }

    r2 = client.get("/spans/A1")
    assert r2.status_code == 200
    assert r2.get_json()["height_difference"] == 40.0


def test_duplicate_span_conflict(client):
    _create_span(client, "A1")
    r = _create_span(client, "A1")
    assert r.status_code == 409
    assert r.get_json()["error"] == "span_conflict"
    # overwrite 可更新
    r2 = _create_span(client, "A1", h=33.0, overwrite=True)
    assert r2.status_code == 201
    assert client.get("/spans/A1").get_json()["height_difference"] == 33.0


def test_unknown_span_name_is_rejected_with_reason(client):
    r = client.post("/calibrate", json={
        "span_name": "NOPE", "measured_sag": 5.0, "measurement_x": 150.0})
    assert r.status_code == 404
    assert r.get_json()["error"] == "span_not_found"

    r2 = client.post("/forward", json={"span_name": "NOPE", "H": 1000.0})
    assert r2.status_code == 404


def test_inline_calibration_roundtrip_closure(client):
    body = {"span": 400.0, "height_difference": 0.0, "weight_per_length": 10.0,
            "measured_sag": 21.0, "measurement_x": 200.0}
    r = client.post("/calibrate", json=body)
    assert r.status_code == 200, r.get_data(as_text=True)
    data = r.get_json()

    assert data["H"] > 0.0
    assert data["c"] == pytest.approx(data["H"] / 10.0)
    assert data["lowest_point"]["x"] == pytest.approx(200.0, abs=1e-8)
    assert data["measurement"]["computed_sag"] == pytest.approx(21.0, rel=1e-9)
    # 21 个默认取样点，覆盖两端，全部在档内
    xs = [p["x"] for p in data["curve"]]
    assert len(xs) == 21
    assert xs[0] == 0.0 and xs[-1] == 400.0
    # 正算再反演闭合
    closure = data["forward_inverse_closure"]
    assert closure["passed"] is True
    assert closure["relative_error"] <= closure["rtol"]


def test_named_span_forward_then_calibrate_same_H(client):
    """同一套几何：已知 H 正算得弧垂，再当测量值送回去，H 必须对上。"""
    _create_span(client, "B7", 250.0, 30.0, 9.0)
    H0 = 2600.0
    r = client.post("/forward", json={"span_name": "B7", "H": H0,
                                      "samples": [0.0, 62.5, 125.0, 250.0]})
    assert r.status_code == 200, r.get_data(as_text=True)
    fwd = r.get_json()
    sag_mid = next(p["sag"] for p in fwd["curve"] if p["x"] == 125.0)
    assert fwd["cable_length"] > 250.0
    assert fwd["lowest_point"]["x"] < 125.0  # 右高左低，最低点偏左

    r2 = client.post("/calibrate", json={"span_name": "B7",
                                         "measured_sag": sag_mid,
                                         "measurement_x": 125.0})
    assert r2.status_code == 200, r2.get_data(as_text=True)
    cal = r2.get_json()
    assert cal["H"] == pytest.approx(H0, rel=1e-8)
    assert cal["geometry"]["source"] == "span:B7"
    assert cal["measurement"]["residual"] == pytest.approx(0.0, abs=1e-7)


def test_forward_nonpositive_H_rejected(client):
    r = client.post("/forward", json={"span": 200.0, "height_difference": 0.0,
                                      "weight_per_length": 10.0, "H": 0.0})
    assert r.status_code == 400
    assert r.get_json()["error"] == "invalid_request"
    assert "水平张力" in r.get_json()["message"]


def test_nonpositive_span_rejected_http(client):
    r = _create_span(client, "X", span=-10.0)
    assert r.status_code == 400
    assert "档距" in r.get_json()["message"]


def test_nonpositive_sag_rejected_http(client):
    r = client.post("/calibrate", json={"span": 300.0, "height_difference": 10.0,
                                        "weight_per_length": 10.0,
                                        "measured_sag": 0.0, "measurement_x": 150.0})
    assert r.status_code == 400
    assert "测量弧垂" in r.get_json()["message"]


def test_sag_height_conflict_returns_422_with_reason(client):
    r = client.post("/calibrate", json={"span": 300.0, "height_difference": 80.0,
                                        "weight_per_length": 10.0,
                                        "measured_sag": 5.0, "measurement_x": 150.0})
    assert r.status_code == 422
    body = r.get_json()
    assert body["error"] == "sag_height_difference_conflict"
    assert body["reason"] == "sag_height_difference_conflict"
    assert body["minimum_feasible_sag"] > 5.0


def test_unreachable_supports_returns_422(client):
    r = client.post("/forward", json={"span": 300.0, "height_difference": 80.0,
                                      "weight_per_length": 10.0, "H": 50000.0})
    assert r.status_code == 422
    body = r.get_json()
    assert body["error"] == "supports_unreachable"
    assert body["reason"] == "supports_unreachable"
    assert body["max_feasible_H"] > 0.0


def test_sample_outside_span_rejected(client):
    r = client.post("/forward", json={"span": 300.0, "height_difference": 0.0,
                                      "weight_per_length": 10.0, "H": 3000.0,
                                      "samples": [0.0, 301.0]})
    assert r.status_code == 400
    assert "档距" in r.get_json()["message"]


def test_custom_samples_and_count(client):
    _create_span(client, "C3", 100.0, 0.0, 10.0)
    r = client.post("/forward", json={"span_name": "C3", "H": 5000.0,
                                      "sample_count": 3})
    pts = r.get_json()["curve"]
    assert [p["x"] for p in pts] == [0.0, 50.0, 100.0]


def test_delete_span(client):
    _create_span(client, "D9")
    assert client.delete("/spans/D9").status_code == 204
    assert client.get("/spans/D9").status_code == 404
    assert client.delete("/spans/D9").status_code == 404
