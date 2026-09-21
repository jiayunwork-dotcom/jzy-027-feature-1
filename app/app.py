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


def _resolve_section_samples(body: dict,
                             members: list[tuple[str, SpanGeometry]]) -> dict[str, list[float]]:
    """耐张段逐档取样：默认按 sample_count 均匀取样；samples 以档名为键给显式点。"""
    explicit = body.get("samples")
    if explicit is not None and not isinstance(explicit, dict):
        raise InvalidRequest("耐张段的 samples 必须是以档名为键、取样点列表为值的对象")
    explicit = explicit or {}
    member_names = {name for name, _ in members}
    for key in explicit:
        if key not in member_names:
            raise InvalidRequest(f"samples 指向的档 {key!r} 不在本耐张段成员内")
    n = body.get("sample_count", service.DEFAULT_SAMPLE_COUNT)
    return {
        name: (validation.require_samples(explicit[name], g.span)
               if name in explicit else service.uniform_samples(g.span, n))
        for name, g in members
    }


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

    # ---- 耐张段联合标定：多档共解一个公共 H ------------------------------
    @app.post("/calibrate-section")
    def calibrate_section_route():
        body = _json_body()
        names = validation.require_span_name_list(body.get("spans"))
        members = [(name, store.get(name)[1]) for name in names]
        geometries = dict(members)

        entries = validation.require_measurement_entries(body.get("measurements"))
        seen: set[str] = set()
        measured: list[section.MeasuredSpan] = []
        for entry in entries:
            name = validation.require_span_name(entry["span"])
            if name not in geometries:
                raise InvalidRequest(
                    f"测量条目指向的档 {name!r} 不在本耐张段成员内")
            if name in seen:
                raise InvalidRequest(f"档 {name!r} 提交了多条测量；每档至多一条")
            seen.add(name)
            g = geometries[name]
            sag = validation.require_positive(entry["measured_sag"], "测量弧垂")
            x = validation.require_measure_position(entry["measurement_x"], g.span)
            measured.append(section.MeasuredSpan(
                name=name, geometry=g, measured_sag=sag, measurement_x=x))

        rtol = validation.require_residual_rtol(
            body.get("residual_rtol", section.DEFAULT_RESIDUAL_RTOL))
        xs_by_name = _resolve_section_samples(body, members)
        result = service.calibrate_section(members, measured, rtol, xs_by_name)
        return jsonify(result)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
