"""Load and expose all YAML configuration for the research pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
import yaml


CONFIG_FILES = {
    "research": "research.yaml",
    "countries": "countries.yaml",
    "indicators": "indicators.yaml",
    "sources": "sources.yaml",
    "preprocessing": "preprocessing.yaml",
    "imputation": "imputation.yaml",
    "index": "index.yaml",
    "econometrics": "econometrics.yaml",
    "machine_learning": "machine_learning.yaml",
    "xai": "xai.yaml",
    "convergence": "convergence.yaml",
    "visualization": "visualization.yaml",
    "dashboard": "dashboard.yaml",
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


@dataclass
class PipelineConfig:
    """Container for all pipeline YAML configs."""

    project_root: Path
    research: Dict[str, Any] = field(default_factory=dict)
    countries: Dict[str, Any] = field(default_factory=dict)
    indicators: Dict[str, Any] = field(default_factory=dict)
    sources: Dict[str, Any] = field(default_factory=dict)
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    imputation: Dict[str, Any] = field(default_factory=dict)
    index: Dict[str, Any] = field(default_factory=dict)
    econometrics: Dict[str, Any] = field(default_factory=dict)
    machine_learning: Dict[str, Any] = field(default_factory=dict)
    xai: Dict[str, Any] = field(default_factory=dict)
    convergence: Dict[str, Any] = field(default_factory=dict)
    visualization: Dict[str, Any] = field(default_factory=dict)
    dashboard: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, project_root: Path) -> "PipelineConfig":
        project_root = Path(project_root).resolve()
        config_dir = project_root / "config"
        if not config_dir.is_dir():
            raise FileNotFoundError(f"Config directory not found: {config_dir}")

        loaded: Dict[str, Dict[str, Any]] = {}
        for attr, filename in CONFIG_FILES.items():
            loaded[attr] = _load_yaml(config_dir / filename)

        return cls(project_root=project_root, **loaded)

    def country_table(
        self, as_dataframe: bool = True
    ) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """Flatten country groups into rows with iso3/iso2/name/group fields."""
        rows: List[Dict[str, Any]] = []
        groups = self.countries.get("groups", {}) or {}
        for group_id, group in groups.items():
            if not isinstance(group, dict):
                continue
            group_name = group.get("name", group_id)
            for country in group.get("countries", []) or []:
                if not isinstance(country, dict):
                    continue
                rows.append(
                    {
                        "iso3": country.get("iso3"),
                        "iso2": country.get("iso2"),
                        "name": country.get("name"),
                        "group_id": group_id,
                        "group_name": group_name,
                    }
                )
        if as_dataframe:
            return pd.DataFrame(rows)
        return rows

    def iso3_list(self) -> List[str]:
        table = self.country_table(as_dataframe=True)
        if table.empty:
            return []
        return [str(x) for x in table["iso3"].tolist() if pd.notna(x)]

    def year_range(self) -> tuple[int, int]:
        panel = self.research.get("panel", {}) or {}
        start = int(panel.get("start_year", 2012))
        end = int(panel.get("end_year", 2023))
        return start, end

    def random_seed(self) -> int:
        return int(self.research.get("random_seed", 42))

    def path(self, key: str) -> Path:
        """Resolve a named path from research.paths relative to project_root."""
        paths = self.research.get("paths", {}) or {}
        if key not in paths:
            raise KeyError(f"Unknown path key: {key}")
        return self.project_root / paths[key]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "research": self.research,
            "countries": self.countries,
            "indicators": self.indicators,
            "sources": self.sources,
            "preprocessing": self.preprocessing,
            "imputation": self.imputation,
            "index": self.index,
            "econometrics": self.econometrics,
            "machine_learning": self.machine_learning,
            "xai": self.xai,
            "convergence": self.convergence,
            "visualization": self.visualization,
            "dashboard": self.dashboard,
        }
