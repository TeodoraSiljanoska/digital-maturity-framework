"""Digital Maturity Framework ontology: TBox on disk, ABox from pipeline artifacts."""

from __future__ import annotations

from ontology.export import NS, run
from ontology.query import answer_all, answer_cq

__all__ = ["NS", "run", "answer_cq", "answer_all"]
