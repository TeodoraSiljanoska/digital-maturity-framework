"""Auto-generate Markdown reports from registries and result artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from common.io import ensure_dir, read_df, read_json, write_json
from common.logging_utils import get_logger
from registries.result_registry import ResultRegistry

logger = get_logger("dmf.reporting.generate")


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return read_json(path)
    except Exception:
        return None


def _safe_read_df(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return read_df(path)
    except Exception:
        return None


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "—"
    try:
        if pd.isna(v):
            return "—"
    except Exception:
        pass
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _df_markdown(df: pd.DataFrame, max_rows: int = 40) -> str:
    """Render a DataFrame as GitHub-flavored markdown without requiring tabulate."""
    show = df.head(max_rows).copy()
    cols = [str(c) for c in show.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in show.iterrows():
        cells = [_fmt(row[c]) if not isinstance(row[c], str) else str(row[c]) for c in show.columns]
        # escape pipes
        cells = [c.replace("|", "\\|") for c in cells]
        lines.append("| " + " | ".join(cells) + " |")
    if len(df) > max_rows:
        lines.append("")
        lines.append(f"_… {len(df) - max_rows} more rows omitted_")
    return "\n".join(lines)


def _build_results_summary(root: Path) -> str:
    lines: List[str] = [
        "# Results summary",
        "",
        f"_Generated: {datetime.now(timezone.utc).isoformat()}_",
        "",
        "This report is generated from pipeline artifacts under `results/` and `outputs/`.",
        "",
    ]

    # Panel basics
    panel_path = root / "data" / "processed" / "analysis_panel.parquet"
    if panel_path.exists():
        panel = read_df(panel_path)
        lines += [
            "## Analysis panel",
            "",
            f"- Rows: **{len(panel)}**",
            f"- Countries: **{panel['country_iso3'].nunique() if 'country_iso3' in panel.columns else '—'}**",
            f"- Years: **{int(panel['year'].min())}–{int(panel['year'].max())}**"
            if "year" in panel.columns
            else "- Years: —",
            f"- Mean DMI: **{_fmt(panel['DMI'].mean(), 2)}**"
            if "DMI" in panel.columns
            else "",
            "",
        ]

    # Descriptive
    desc = _safe_read_df(root / "results" / "descriptive" / "dmi_moments_by_group.csv")
    if desc is not None and not desc.empty:
        lines += ["## DMI by group", "", _df_markdown(desc), ""]

    # Econometrics
    eco = _safe_read_json(root / "results" / "econometrics" / "econometrics_summary.json")
    coef = _safe_read_df(root / "results" / "econometrics" / "coefficients.csv")
    if eco:
        lines += [
            "## Econometrics",
            "",
            f"- Formula: `{eco.get('formula', '—')}`",
            f"- N observations: **{eco.get('nobs', '—')}**",
            f"- Preferred model (Hausman): **{(eco.get('hausman') or {}).get('prefer', '—')}**",
            "",
        ]
    if coef is not None and not coef.empty:
        fe = coef[coef["model"].astype(str) == "FE"] if "model" in coef.columns else coef
        show = fe.head(20)
        lines += ["### FE coefficients (excerpt)", "", _df_markdown(show), ""]

    # ML
    ml = _safe_read_df(root / "results" / "machine_learning" / "model_comparison.csv")
    best = _safe_read_json(root / "outputs" / "models" / "best_model.json")
    if best:
        lines += [
            "## Machine learning",
            "",
            f"- Best model: **{best.get('best_model')}**",
            f"- Test RMSE: **{_fmt(best.get('rmse'), 4)}**",
            f"- Model id: `{best.get('model_id')}`",
            "",
        ]
    if ml is not None and not ml.empty:
        cols = [c for c in ("model_type", "status", "rmse", "mae", "r2") if c in ml.columns]
        lines += ["### Model comparison", "", _df_markdown(ml[cols]), ""]

    # Comparative
    comp = _safe_read_df(root / "results" / "comparative" / "group_comparison.csv")
    if comp is not None and not comp.empty:
        lines += ["## Comparative group metrics", "", _df_markdown(comp), ""]

    # Convergence
    conv_summary = None
    for p in (
        root / "results" / "convergence" / "convergence_summary.json",
        root / "outputs" / "dashboard" / "framework_status.json",
    ):
        data = _safe_read_json(p)
        if data and ("sigma_start" in data or "convergence_summary" in data):
            conv_summary = data.get("convergence_summary", data)
            break
    if conv_summary:
        lines += [
            "## Convergence",
            "",
            f"- Source: `{conv_summary.get('source', '—')}`",
            f"- Sigma start → end: **{_fmt(conv_summary.get('sigma_start'))} → {_fmt(conv_summary.get('sigma_end'))}**",
            f"- Converging: **{conv_summary.get('converging', '—')}**",
            "",
        ]

    # Registry counts
    registry = ResultRegistry(root)
    lines += [
        "## Result registry",
        "",
        f"- Registered results: **{len(registry.list())}**",
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _build_discussion_evidence(root: Path) -> str:
    lines: List[str] = [
        "# Discussion evidence (hypothesis-linked)",
        "",
        f"_Generated: {datetime.now(timezone.utc).isoformat()}_",
        "",
        "Evidence statements below are derived from `results/hypotheses/hypothesis_evaluation.json`.",
        "",
    ]
    hyp = _safe_read_json(root / "results" / "hypotheses" / "hypothesis_evaluation.json")
    if not hyp:
        hyp = _safe_read_json(root / "outputs" / "hypothesis_testing" / "hypothesis_evaluation.json")
    if not hyp:
        lines += [
            "> Hypothesis evaluation not found. Run `hypotheses.testing.run` first.",
            "",
        ]
        return "\n".join(lines)

    summary = hyp.get("summary") or {}
    lines += ["## Status overview", ""]
    for hid, status in summary.items():
        lines.append(f"- **{hid}**: {status}")
    lines.append("")

    bodies = hyp.get("hypotheses") or {}
    for hid in ["H1", "H1.1", "H1.2", "H1.3", "H1.4", "H1.5", "H1.6", "H1.7", "H1.8"]:
        body = bodies.get(hid) or {}
        lines += [f"## {hid}", ""]
        if body.get("text"):
            lines.append(f"**Hypothesis.** {body['text']}")
            lines.append("")
        lines.append(f"**Evaluation status:** `{body.get('status', '—')}`")
        lines.append("")

        if hid in {"H1.1", "H1.2", "H1.3", "H1.4", "H1.7"}:
            details = body.get("details") or []
            if details:
                lines.append("Coefficient evidence:")
                lines.append("")
                for d in details:
                    if not d.get("found"):
                        lines.append(f"- `{d.get('variable')}`: not found in FE results")
                        continue
                    lines.append(
                        f"- `{d.get('term')}`: coef={_fmt(d.get('coefficient'))}, "
                        f"p={_fmt(d.get('pvalue'))}, "
                        f"sign_ok={d.get('sign_matches')}, significant={d.get('significant')}"
                    )
                lines.append("")
        elif hid == "H1.5":
            lines.append(
                f"- ANOVA p={_fmt((body.get('anova') or {}).get('pvalue'))}; "
                f"Kruskal–Wallis p={_fmt((body.get('kruskal_wallis') or {}).get('pvalue'))}"
            )
            if body.get("group_means"):
                lines.append(f"- Group means: `{body['group_means']}`")
            lines.append("")
        elif hid == "H1.6":
            ml = body.get("best_ml") or {}
            base = body.get("best_baseline") or {}
            lines.append(
                f"- Best ML: `{ml.get('model_type')}` RMSE={_fmt(ml.get('rmse'))}"
            )
            lines.append(
                f"- Best baseline: `{base.get('model_type')}` RMSE={_fmt(base.get('rmse'))}"
            )
            lines.append(f"- RMSE improvement: {_fmt(body.get('rmse_improvement'))}")
            lines.append("")
        elif hid == "H1.8":
            lines.append(f"- XAI artifacts exist: **{body.get('xai_artifacts_exist')}**")
            lines.append(f"- Top features: `{body.get('top_features')}`")
            lines.append(
                f"- Overlap with hypothesized drivers: `{body.get('overlap_with_hypothesized_drivers')}`"
            )
            lines.append("")
        elif hid == "H1":
            lines.append(f"- Mean support score: **{_fmt(body.get('mean_support_score'), 3)}**")
            lines.append(f"- Breakdown: `{body.get('breakdown')}`")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _build_tables_index(root: Path) -> str:
    lines: List[str] = [
        "# Tables index",
        "",
        f"_Generated: {datetime.now(timezone.utc).isoformat()}_",
        "",
        "Indexed from `results/registry.json` and known `outputs/tables/` artifacts.",
        "",
        "| Result ID | Category | Format | Path |",
        "|---|---|---|---|",
    ]
    registry = ResultRegistry(root)
    for entry in sorted(registry.list(), key=lambda e: (e.get("category") or "", e.get("result_id") or "")):
        lines.append(
            f"| `{entry.get('result_id')}` | {entry.get('category')} | {entry.get('format')} | "
            f"`{entry.get('artifact_path')}` |"
        )

    tables_dir = root / "outputs" / "tables"
    if tables_dir.is_dir():
        lines += ["", "## outputs/tables", ""]
        for f in sorted(tables_dir.glob("*")):
            if f.is_file():
                lines.append(f"- `{f.relative_to(root)}`")

    figures_dir = root / "outputs" / "figures"
    if figures_dir.is_dir():
        figs = [f for f in figures_dir.glob("*") if f.is_file()]
        if figs:
            lines += ["", "## outputs/figures", ""]
            for f in sorted(figs):
                lines.append(f"- `{f.relative_to(root)}`")

    lines.append("")
    return "\n".join(lines)


def run(project_root: Path | str, config: Any = None) -> Dict[str, Any]:
    """
    Generate Markdown reports in ``outputs/reports/`` from real results only.
    """
    root = Path(project_root).resolve()
    out_dir = ensure_dir(root / "outputs" / "reports")

    reports = {
        "results_summary.md": _build_results_summary(root),
        "discussion_evidence.md": _build_discussion_evidence(root),
        "tables_index.md": _build_tables_index(root),
    }
    written = []
    for name, content in reports.items():
        path = out_dir / name
        path.write_text(content, encoding="utf-8")
        written.append(str(path.relative_to(root)))

    index = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reports": written,
    }
    write_json(out_dir / "reports_manifest.json", index)
    logger.info("Wrote reports: %s", written)
    return index
