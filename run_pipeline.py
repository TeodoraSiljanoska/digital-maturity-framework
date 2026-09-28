#!/usr/bin/env python3
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from pipeline.orchestrator import PipelineOrchestrator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_stage", default=None)
    parser.add_argument("--to", dest="to_stage", default=None)
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()
    orch = PipelineOrchestrator(ROOT)
    orch.run(from_stage=args.from_stage, to_stage=args.to_stage, skip_completed=args.skip_completed)

if __name__ == "__main__":
    main()
