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
