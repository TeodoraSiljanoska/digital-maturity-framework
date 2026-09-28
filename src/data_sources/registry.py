"""Factory for data-source adapters."""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

from common.errors import DataSourceError
from data_sources.base import DataSourceAdapter
from data_sources.eurostat import EurostatAdapter
from data_sources.gii import GIIAdapter
from data_sources.itu import ITUAdapter
from data_sources.ncsi import NCSIAdapter
from data_sources.oecd import OECDAdapter
from data_sources.oxford import OxfordAdapter
from data_sources.un import UNAdapter
from data_sources.undp import UNDPAdapter
from data_sources.world_bank import WorldBankAdapter


ADAPTER_REGISTRY: Dict[str, Type[DataSourceAdapter]] = {
    "world_bank": WorldBankAdapter,
    "un": UNAdapter,
    "ncsi": NCSIAdapter,
    "gii": GIIAdapter,
    "oxford": OxfordAdapter,
    "undp": UNDPAdapter,
    "oecd": OECDAdapter,
    "eurostat": EurostatAdapter,
    "itu": ITUAdapter,
}

# Map sources.yaml keys -> adapter names when adapter field omitted.
SOURCE_ALIASES: Dict[str, str] = {
    "world_bank": "world_bank",
    "un_egdi": "un",
    "un": "un",
    "ncsi": "ncsi",
    "gii": "gii",
    "oxford_insights": "oxford",
    "oxford": "oxford",
    "undp": "undp",
    "oecd": "oecd",
    "eurostat": "eurostat",
    "itu": "itu",
}


def get_adapter(
    source_name: str,
    config: Optional[Dict[str, Any]] = None,
    project_root: Optional[Any] = None,
) -> DataSourceAdapter:
    """Return a configured adapter instance for a source name or adapter key."""
    cfg = dict(config or {})
    key = str(cfg.get("adapter") or source_name).strip().lower()
    key = SOURCE_ALIASES.get(key, key)
    cls = ADAPTER_REGISTRY.get(key)
    if cls is None:
        raise DataSourceError(
            f"Unknown data source adapter: {source_name}",
            details={"resolved": key, "available": sorted(ADAPTER_REGISTRY.keys())},
        )
    return cls(config=cfg, project_root=project_root)


def list_adapters() -> Dict[str, str]:
    return {name: cls.__name__ for name, cls in ADAPTER_REGISTRY.items()}
