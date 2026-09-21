"""标定服务统一异常体系。

所有对调用方可见的错误都以 JSON 返回：
    {"error": <机器可读错误码>, "message": <中文说明>, ...}
"""
from __future__ import annotations


class CalibrationError(Exception):
    """业务错误基类。"""

    status_code: int = 400
    error_code: str = "bad_request"

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_response(self) -> tuple[dict, int]:
        body = {"error": self.error_code, "message": self.message}
        body.update(self.details)
        return body, self.status_code


class InvalidRequest(CalibrationError):
    """入参本身不合法（非正、非数、越界等），状态码 400。"""

    status_code = 400
    error_code = "invalid_request"


class SpanNotFound(CalibrationError):
    """点名的具名档不存在，状态码 404。"""

    status_code = 404
    error_code = "span_not_found"


class SpanConflict(CalibrationError):
    """具名档重复登记，状态码 409。"""

    status_code = 409
    error_code = "span_conflict"


class CalibrationFailed(CalibrationError):
    """几何/测量本身合法，但数学上给不出一条满足条件的悬链，状态码 422。"""

    status_code = 422
    error_code = "calibration_failed"


class SupportsUnreachable(CalibrationFailed):
    """高差大到在给定水平张力下索够不着两个支座（最低点落到档外）。"""

    error_code = "supports_unreachable"

    def __init__(self, message: str, *, H: float | None = None, H_max: float | None = None) -> None:
        details: dict = {"reason": "supports_unreachable"}
        if H is not None:
            details["given_H"] = H
        if H_max is not None:
            details["max_feasible_H"] = H_max
        super().__init__(message, details=details)


class SagHeightConflict(CalibrationFailed):
    """测量弧垂与高差互相矛盾：可行悬链在该点的弧垂不可能这么小。"""

    error_code = "sag_height_difference_conflict"

    def __init__(self, message: str, *, minimum_sag: float | None = None, given_sag: float | None = None) -> None:
        details: dict = {"reason": "sag_height_difference_conflict"}
        if minimum_sag is not None:
            details["minimum_feasible_sag"] = minimum_sag
        if given_sag is not None:
            details["given_sag"] = given_sag
        super().__init__(message, details=details)


class SectionMeasurementBelowMinimum(CalibrationFailed):
    """耐张段成员档的测量弧垂低于该档自身的可行下限，单档都立不住，段更无从谈起。"""

    error_code = "section_sag_below_minimum"

    def __init__(self, message: str, *, span: str, measured_sag: float,
                 minimum_feasible_sag: float | None = None) -> None:
        details: dict = {
            "reason": "sag_below_minimum",
            "span": span,
            "measured_sag": measured_sag,
        }
        if minimum_feasible_sag is not None:
            details["minimum_feasible_sag"] = minimum_feasible_sag
        super().__init__(message, details=details)


class SectionMeasurementsInconsistent(CalibrationFailed):
    """耐张段内多个测量档的实测数据互相矛盾：不存在让全部残差落入容差的公共 H。"""

    error_code = "section_measurements_inconsistent"

    def __init__(self, message: str, *, conflicting_spans: list[str],
                 measurements_detail: list[dict]) -> None:
        details: dict = {
            "reason": "measurements_inconsistent",
            "conflicting_spans": conflicting_spans,
            "measurements": measurements_detail,
        }
        super().__init__(message, details=details)


class SectionSpanInfeasible(CalibrationFailed):
    """测量要求的公共水平张力下，段内某档的最低点将落到档外（超出该档可行上限）。"""

    error_code = "section_span_infeasible"

    def __init__(self, message: str, *, span: str, required_H: float,
                 max_feasible_H: float, infeasible_spans: list[dict],
                 compatible_H_range: list) -> None:
        details: dict = {
            "reason": "span_infeasible_at_tension",
            "span": span,
            "required_H": required_H,
            "max_feasible_H": max_feasible_H,
            "infeasible_spans": infeasible_spans,
            "compatible_H_range": compatible_H_range,
        }
        super().__init__(message, details=details)
