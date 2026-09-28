"""Typed error codes."""

from common.errors import (
    DataSourceError,
    IndicatorUnavailableError,
    InsufficientSampleError,
    ModelNotApplicableError,
)


def test_error_codes():
    assert DataSourceError("x").code == "DATA_SOURCE_ERROR"
    assert IndicatorUnavailableError("x").code == "INDICATOR_UNAVAILABLE"
    assert InsufficientSampleError("x").code == "INSUFFICIENT_SAMPLE"
    assert ModelNotApplicableError("x").code == "MODEL_NOT_APPLICABLE"