"""National Cyber Security Index (NCSI) snapshot adapter."""

from __future__ import annotations

from typing import Any, Dict, Optional

from data_sources.base import SnapshotCsvAdapter


class NCSIAdapter(SnapshotCsvAdapter):
    name = "ncsi"
    value_columns = {"NCSI_SCORE": "NCSI_SCORE"}

    def __init__(self, config: Optional[Dict[str, Any]] = None, project_root: Optional[Any] = None):
        super().__init__(config=config, project_root=project_root)
        if "snapshot_file" not in self.config:
            self.config["snapshot_file"] = "data/raw/ncsi/ncsi_panel.csv"
