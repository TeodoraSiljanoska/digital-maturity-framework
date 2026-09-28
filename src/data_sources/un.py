"""UN EGDI / OSI curated snapshot adapter."""

from __future__ import annotations

from typing import Any, Dict, Optional

from data_sources.base import SnapshotCsvAdapter


class UNAdapter(SnapshotCsvAdapter):
    """Load EGDI/OSI from data/raw/un_egdi/egdi_panel.csv."""

    name = "un"
    value_columns = {
        "EGDI": "EGDI",
        "OSI": "OSI",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None, project_root: Optional[Any] = None):
        super().__init__(config=config, project_root=project_root)
        if "snapshot_file" not in self.config:
            self.config["snapshot_file"] = "data/raw/un_egdi/egdi_panel.csv"

    def update_snapshot(self, df) -> Any:
        """Write/update the EGDI/OSI snapshot CSV."""
        return self.write_snapshot(df)
