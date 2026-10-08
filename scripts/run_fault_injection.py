"""Plants ten known faults in copies of the knowledge graph and records which
of them the OWL reasoner (HermiT) and the SHACL engine detect.

    python scripts/run_knowledge_layer.py   # builds outputs/knowledge_layer/ontology/dmi-kg.ttl
    python scripts/run_fault_injection.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
from rdflib import Graph  # noqa: E402

from knowledge_layer import validate  # noqa: E402


def main() -> None:
    g = Graph().parse(ROOT / "outputs/knowledge_layer/ontology/dmi-kg.ttl", format="turtle")
    with tempfile.TemporaryDirectory() as tmp:
        rows = validate.fault_injection(g, ROOT / "ontology/shapes/dmi-shapes.ttl", Path(tmp))
    df = pd.DataFrame(rows)
    out = ROOT / "outputs/knowledge_layer/fault_injection.csv"
    df.to_csv(out, index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
