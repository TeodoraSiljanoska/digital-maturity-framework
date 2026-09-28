"""Oxford Insights Government AI Readiness Index snapshot adapter."""

from __future__ import annotations

from typing import Any, Dict, Optional

from data_sources.base import SnapshotCsvAdapter


class OxfordAdapter(SnapshotCsvAdapter):
    name = "oxford"
    value_columns = {"AI_READY": "AI_READY"}

    def __init__(self, config: Optional[Dict[str, Any]] = None, project_root: Optional[Any] = None):
        super().__init__(config=config, project_root=project_root)
        if "snapshot_file" not in self.config:
            self.config["snapshot_file"] = "data/raw/oxford/ai_readiness_panel.csv"
