"""External data-source adapters for the digital maturity framework."""

from data_sources.base import STANDARD_COLUMNS, DataSourceAdapter
from data_sources.registry import get_adapter, list_adapters

__all__ = [
    "STANDARD_COLUMNS",
    "DataSourceAdapter",
    "get_adapter",
    "list_adapters",
]
