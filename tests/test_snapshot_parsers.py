"""Parsers of scripts/build_snapshots_v3.py accept only values that pass their built-in checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_snapshots_v3.py"
spec = importlib.util.spec_from_file_location("build_snapshots_v3", SCRIPT)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_un_history_row_is_read_in_published_order():
    text = (
        "E-Government Development Index 2024 2022 2020 2018 2016 2014 2012 "
        "Maldives (Rank) 94 104 105 97 117 94 95 "
        "Maldives (Value) 0.67453 0.58850 0.57400 0.56150 0.43299 0.48129 0.49936 "
    )
    history = builder._un_history(text, "E-Government Development Index")
    assert history[2024] == 0.67453
    assert history[2012] == 0.49936


def test_ncsi_placeholder_before_first_ranking_is_not_a_publication():
    html = (
        'var ncsi_country_timeline =[{"date":"2023-09-28","value":null,"hidden":true},'
        '{"date":"2023-11-15","value":"0.00"},{"date":"2025-03-15","value":"71.67"}];'
        'var ncsi_country_ranking_timeline = [{"date":"2018-12-28","value":null,"hidden":true},'
        '{"date":"2025-03-15","value":24}];'
    )
    assert builder._ncsi_timeline(html)[0] == {"date": "2023-11-15", "value": 0.0}
    assert builder._first_ranked(html) == "2025-03-15"


def test_gii_score_requires_the_official_rank():
    text = "\nSerbia\n35.46\n55\nUM\n\nSerbia\n99.99\n3\nHI\n"
    ranks = {("SRB", 2018): 55}
    assert builder._gii_from_report(2018, text, ranks) == {"SRB": 35.46}


def test_oxford_row_requires_total_equal_to_pillar_mean():
    good = "\n71\nNorth Macedonia\n46.11\n50.66\n32.05\n55.62\n"
    bad = "\n71\nNorth Macedonia\n60.00\n50.66\n32.05\n55.62\n"
    assert builder._oxford_pillar_rows(good, with_rank=True)["MKD"]["total"] == 46.11
    assert "MKD" not in builder._oxford_pillar_rows(bad, with_rank=True)
