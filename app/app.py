"""Flask HTTP 入口：几何档登记 + 标定（反演）+ 正算 + 耐张段联合标定。"""
from __future__ import annotations

import os

from flask import Flask, jsonify, request

from . import section, service, validation
from .catenary import SpanGeometry
from .errors import CalibrationError, InvalidRequest, SpanConflict, SpanNotFound
from .storage import SpanStore


def _json_body() -> dict:
    if not request.is_json:
        raise InvalidRequest("请求体必须是 application/json")
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise InvalidRequest("JSON 请求体必须是对象")
    return body


def _geometry_from_body(body: dict) -> SpanGeometry:
    missing = [k for k in ("span", "height_difference", "weight_per_length") if k not in body]
    if missing:
        raise InvalidRequest(f"缺少几何参数：{', '.join(missing)}")
    return SpanGeometry.create(body["span"], body["height_difference"], body["weight_per_length"])


def _resolve_geometry(body: dict, store: SpanStore) -> tuple[SpanGeometry, str]:
    """点名某档时只再补测量弧垂与位置；否则从内联几何参数构造。"""
    name = body.get("span_name")
    if name is not None:
        _registered, geometry = store.get(name)
        return geometry, f"span:{name}"
    return _geometry_from_body(body), "inline"


def create_app(data_dir: str | None = None) -> Flask:
    app = Flask(__name__)
    store = SpanStore(data_dir or os.environ.get("CATENARY_DATA_DIR", "/data"))

    @app.errorhandler(CalibrationError)
    def _handle_calibration_error(err: CalibrationError):
        body, status = err.to_response()
        return jsonify(body), status

    @app.errorhandler(404)
    def _handle_404(_err):
        return jsonify({"error": "not_found", "message": "路径不存在"}), 404

    @app.errorhandler(405)
    def _handle_405(_err):
        return jsonify({"error": "method_not_allowed", "message": "方法不允许"}), 405

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    # ---- 具名几何档 -----------------------------------------------------
    @app.post("/spans")
    def create_span():
        body = _json_body()
        name = body.get("name")
        if name is None:
            raise InvalidRequest("缺少档名 name")
        geometry = _geometry_from_body(body)
        overwrite = bool(body.get("overwrite", False))
        try:
            store.save(name, geometry, overwrite=overwrite)
        except SpanConflict:
            raise
        return jsonify({"name": name, **geometry.to_dict()}), 201

    @app.get("/spans")
    def list_spans():
        return jsonify({"spans": store.list()})

    @app.get("/spans/<name>")
    def get_span(name: str):
        stored, geometry = store.get(name)
        return jsonify({"name": stored, **geometry.to_dict()})

    @app.delete("/spans/<name>")
    def delete_span(name: str):
        store.delete(name)
        return ("", 204)

    # ---- 标定：量弧垂反演 H，再铺整条曲线 -------------------------------
    @app.post("/calibrate")
    def calibrate_route():
        body = _json_body()
        for field in ("measured_sag", "measurement_x"):
            if field not in body:
                raise InvalidRequest(f"标定缺少字段：{field}")
        geometry, source = _resolve_geometry(body, store)
        xs = service.resolve_samples(geometry.span, body)
        result = service.calibrate(
            geometry,
            body["measured_sag"],
            body["measurement_x"],
            xs,
            geometry_source=source,
        )
        return jsonify(result)

    # ---- 正算：给定 H 铺整条曲线 ----------------------------------------
    @app.post("/forward")
    def forward_route():
        body = _json_body()
        if "H" not in body:
            raise InvalidRequest("正算缺少字段：H")
        geometry, source = _resolve_geometry(body, store)
        xs = service.resolve_samples(geometry.span, body)
        result = service.forward(geometry, body["H"], xs, geometry_source=source)
        return jsonify(result)

    # ---- 耐张段联合标定：多档共享公共 H 的联合反演 -----------------------
    @app.post("/calibrate_section")
    def calibrate_section_route():
        body = _json_body()
        raw_members = body.get("spans")
        if not isinstance(raw_members, list) or not raw_members:
            raise InvalidRequest("耐张段联合标定需要非空的 spans 成员列表")

        members: list[section.SectionMember] = []
        xs_per_member: list[list[float]] = []
        seen: set[str] = set()
        for entry in raw_members:
            if not isinstance(entry, dict):
                raise InvalidRequest("spans 的每个成员必须是对象")
            name = entry.get("name")
            if name is None:
                raise InvalidRequest("耐张段成员缺少档名 name")
            if name in seen:
                raise InvalidRequest(f"耐张段内档名重复：{name!r}")
            seen.add(name)
            _stored, geometry = store.get(name)  # 档名非法 400、未登记 404

            has_sag = entry.get("measured_sag") is not None
            has_x = entry.get("measurement_x") is not None
            if has_sag != has_x:
                raise InvalidRequest(f"档 {name}：measured_sag 与 measurement_x 必须同时提供")
            measurement = None
            if has_sag:
                sag = validation.require_positive(entry["measured_sag"], f"档 {name} 的测量弧垂")
                x = validation.require_measure_position(entry["measurement_x"], geometry.span)
                measurement = section.MemberMeasurement(x=x, sag=sag)
            members.append(section.SectionMember(name=name, geometry=geometry, measurement=measurement))

            # 取样：成员自己的 samples/sample_count 优先，否则用全局 sample_count。
            merged = dict(entry)
            if "sample_count" not in merged and "sample_count" in body:
                merged["sample_count"] = body["sample_count"]
            xs_per_member.append(service.resolve_samples(geometry.span, merged))

        result = service.calibrate_section(
            members,
            xs_per_member,
            body.get("residual_rtol", section.DEFAULT_RESIDUAL_RTOL),
        )
        return jsonify(result)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
