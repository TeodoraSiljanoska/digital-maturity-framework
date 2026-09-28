"""Record and query data lineage edges for auditability."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from common.io import ensure_dir, read_json, write_json


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class LineageNode:
    node_id: str
    node_type: str
    parent_ids: List[str] = field(default_factory=list)
    transform: Optional[str] = None
    path: Optional[str] = None
    timestamp: str = field(default_factory=_utcnow_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LineageNode":
        return cls(
            node_id=str(data["node_id"]),
            node_type=str(data.get("node_type", "unknown")),
            parent_ids=list(data.get("parent_ids") or []),
            transform=data.get("transform"),
            path=data.get("path"),
            timestamp=str(data.get("timestamp") or _utcnow_iso()),
        )


class LineageTracker:
    """Persist lineage edges to outputs/audit/lineage.json."""

    def __init__(self, project_root: Path | str):
        self.project_root = Path(project_root).resolve()
        self.path = self.project_root / "outputs" / "audit" / "lineage.json"
        self._nodes: Dict[str, LineageNode] = {}
        if self.path.exists():
            self.load()

    def record(
        self,
        node_id: str,
        node_type: str,
        parent_ids: Optional[Sequence[str]] = None,
        transform: Optional[str] = None,
        path: Optional[str | Path] = None,
        timestamp: Optional[str] = None,
    ) -> LineageNode:
        node = LineageNode(
            node_id=node_id,
            node_type=node_type,
            parent_ids=[str(p) for p in (parent_ids or [])],
            transform=transform,
            path=str(path) if path is not None else None,
            timestamp=timestamp or _utcnow_iso(),
        )
        self._nodes[node_id] = node
        self.save()
        return node

    def get(self, node_id: str) -> Optional[LineageNode]:
        return self._nodes.get(node_id)

    def trace(self, node_id: str) -> List[Dict[str, Any]]:
        """
        Return the ancestry chain for ``node_id`` (node first, then parents BFS).

        Raises KeyError if the node is unknown.
        """
        if node_id not in self._nodes:
            raise KeyError(f"Unknown lineage node_id: {node_id}")

        ordered: List[Dict[str, Any]] = []
        seen = set()
        queue: List[str] = [node_id]
        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            node = self._nodes.get(current)
            if node is None:
                ordered.append(
                    {
                        "node_id": current,
                        "node_type": "missing",
                        "parent_ids": [],
                        "transform": None,
                        "path": None,
                        "timestamp": None,
                    }
                )
                continue
            ordered.append(node.to_dict())
            for parent in node.parent_ids:
                if parent not in seen:
                    queue.append(parent)
        return ordered

    def edges(self) -> List[Dict[str, Any]]:
        return [n.to_dict() for n in self._nodes.values()]

    def save(self) -> Path:
        ensure_dir(self.path.parent)
        payload = {
            "updated_at": _utcnow_iso(),
            "nodes": [n.to_dict() for n in self._nodes.values()],
        }
        return write_json(self.path, payload)

    def load(self) -> None:
        if not self.path.exists():
            return
        data = read_json(self.path)
        nodes = data.get("nodes", data if isinstance(data, list) else [])
        self._nodes = {}
        for item in nodes or []:
            if isinstance(item, dict) and "node_id" in item:
                node = LineageNode.from_dict(item)
                self._nodes[node.node_id] = node
