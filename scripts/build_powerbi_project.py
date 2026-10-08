"""Writes the Power BI project (star schema CSV + TMDL semantic model + PBIR
report) into outputs/knowledge_layer/powerbi/.

    python scripts/run_knowledge_layer.py      # produces the cells and results
    python scripts/build_powerbi_project.py [--active-page monitoring|provenance|context|models]

Open outputs/knowledge_layer/powerbi/DMI_VDA.pbip in Power BI Desktop and
select Refresh to load the CSV files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from knowledge_layer import powerbi  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--active-page", default="monitoring")
    ap.add_argument("--theme", default=None, help="optional base theme JSON to embed")
    args = ap.parse_args()
    cells = pd.read_csv(ROOT / "outputs/knowledge_layer/ontology/cells.csv")
    info = powerbi.write_project(ROOT, cells, Path(args.theme) if args.theme else None, args.active_page)
    (ROOT / "outputs/knowledge_layer/powerbi/project_summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
