"""Typed scientific failure modes for the research framework."""

from __future__ import annotations


class FrameworkError(Exception):
    """Base framework error."""

    code = "FRAMEWORK_ERROR"

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


class DataSourceError(FrameworkError):
    code = "DATA_SOURCE_ERROR"


class IndicatorUnavailableError(FrameworkError):
    code = "INDICATOR_UNAVAILABLE"


class InsufficientSampleError(FrameworkError):
    code = "INSUFFICIENT_SAMPLE"


class ModelNotApplicableError(FrameworkError):
    code = "MODEL_NOT_APPLICABLE"


class ValidationError(FrameworkError):
    code = "VALIDATION_ERROR"


class PipelineStageError(FrameworkError):
    code = "PIPELINE_STAGE_ERROR"

    def __init__(self, stage: str, message: str, *, cause: Exception | None = None):
        details = {"stage": stage}
        if cause is not None:
            details["cause"] = repr(cause)
            if isinstance(cause, FrameworkError):
                details["cause_code"] = cause.code
        super().__init__(message, details=details)
        self.stage = stage
        self.cause = cause