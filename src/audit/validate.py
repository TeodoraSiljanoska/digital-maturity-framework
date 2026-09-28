"""Framework validation checklist across data → framework layers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.errors import FrameworkError
from common.io import ensure_dir, read_json, write_json
from common.logging_utils import get_logger

logger = get_logger("dmf.audit")


Criterion = Dict[str, Any]


def _exists(root: Path, rel: str) -> bool:
    return (root / rel).exists()


def _nonempty_dir(root: Path, rel: str, pattern: str = "*") -> bool:
    d = root / rel
    if not d.is_dir():
        return False
    return any(d.glob(pattern))


def _check(
    criterion_id: str,
    layer: str,
    description: str,
    passed: bool,
    *,
    critical: bool = False,
    detail: Optional[str] = None,
    artifact: Optional[str] = None,
) -> Criterion:
    return {
        "id": criterion_id,
        "layer": layer,
        "description": description,
        "status": "pass" if passed else "fail",
        "critical": critical,
        "detail": detail,
        "artifact": artifact,
    }


def _definition_of_complete_checks(root: Path) -> List[Criterion]:
    """
    Criteria aligned with the research pipeline Definition of Complete,
    scored pass/fail based on whether expected artifacts exist.
    """
    checks: List[Criterion] = []

    # --- Data layer ---
    checks.append(
        _check(
            "DATA_RAW",
            "data",
            "Integrated raw panel exists",
            _exists(root, "data/raw/integrated_raw.parquet"),
            critical=True,
            artifact="data/raw/integrated_raw.parquet",
        )
    )
    checks.append(
        _check(
            "DATA_VALIDATED",
            "data",
            "Validation report produced",
            _exists(root, "outputs/audit/validation_report.json"),
            artifact="outputs/audit/validation_report.json",
        )
    )
    checks.append(
        _check(
            "DATA_INTERIM",
            "data",
            "Validated interim parquet exists",
            _exists(root, "data/interim/validated_raw.parquet")
            or _exists(root, "data/interim/panel_clean.parquet"),
            artifact="data/interim/",
        )
    )

    # --- Processing layer ---
    checks.append(
        _check(
            "PROC_WIDE",
            "processing",
            "Processed wide panel exists",
            _exists(root, "data/processed/panel_wide.parquet"),
            artifact="data/processed/panel_wide.parquet",
        )
    )
    checks.append(
        _check(
            "PROC_FEATURES",
            "processing",
            "Feature panel exists",
            _exists(root, "data/processed/panel_features.parquet"),
            artifact="data/processed/panel_features.parquet",
        )
    )

    # --- Index layer ---
    dmi_ok = _exists(root, "data/processed/dmi_panel.parquet") or _exists(
        root, "outputs/data/dmi_panel.parquet"
    )
    analysis_ok = _exists(root, "data/processed/analysis_panel.parquet")
    checks.append(
        _check(
            "INDEX_DMI_PANEL",
            "index",
            "DMI panel artifact exists",
            dmi_ok,
            critical=True,
            artifact="data/processed/dmi_panel.parquet|outputs/data/dmi_panel.parquet",
        )
    )
    checks.append(
        _check(
            "INDEX_ANALYSIS_PANEL",
            "index",
            "Analysis panel ready for modeling",
            analysis_ok,
            critical=True,
            artifact="data/processed/analysis_panel.parquet",
        )
    )
    checks.append(
        _check(
            "INDEX_RELIABILITY",
            "index",
            "DMI reliability / sensitivity artifacts",
            _exists(root, "outputs/tables/dmi_reliability.json")
            or _exists(root, "outputs/tables/dmi_sensitivity.csv"),
            artifact="outputs/tables/dmi_*",
        )
    )

    # --- Descriptive / statistics ---
    checks.append(
        _check(
            "STATS_DESCRIPTIVE",
            "statistics",
            "Descriptive statistics results present",
            _exists(root, "results/descriptive/descriptive_summary.json")
            or _exists(root, "results/descriptive/correlation_matrix.csv"),
            artifact="results/descriptive/",
        )
    )

    # --- Econometric layer ---
    checks.append(
        _check(
            "ECO_COEFFICIENTS",
            "econometric",
            "Econometric coefficients table",
            _exists(root, "results/econometrics/coefficients.csv"),
            artifact="results/econometrics/coefficients.csv",
        )
    )
    checks.append(
        _check(
            "ECO_DIAGNOSTICS",
            "econometric",
            "Econometric diagnostics / Hausman",
            _exists(root, "results/econometrics/diagnostics.json")
            or _exists(root, "results/econometrics/hausman_test.json"),
            artifact="results/econometrics/",
        )
    )

    # --- ML layer ---
    ml_cmp = _exists(root, "results/machine_learning/model_comparison.csv")
    checks.append(
        _check(
            "ML_MODEL_COMPARISON",
            "ml",
            "ML model comparison CSV",
            ml_cmp,
            critical=True,
            artifact="results/machine_learning/model_comparison.csv",
        )
    )
    checks.append(
        _check(
            "ML_BEST_MODEL",
            "ml",
            "Best model selection recorded",
            _exists(root, "outputs/models/best_model.json"),
            artifact="outputs/models/best_model.json",
        )
    )
    checks.append(
        _check(
            "ML_PREDICTIONS",
            "ml",
            "Prediction artifacts exported",
            _nonempty_dir(root, "outputs/predictions", "*.csv"),
            artifact="outputs/predictions/",
        )
    )

    # --- XAI layer ---
    checks.append(
        _check(
            "XAI_ARTIFACTS",
            "xai",
            "Explainability artifacts under results/xai or outputs/xai",
            _nonempty_dir(root, "results/xai") or _nonempty_dir(root, "outputs/xai"),
            artifact="results/xai|outputs/xai",
        )
    )

    # --- Convergence ---
    checks.append(
        _check(
            "CONV_ARTIFACTS",
            "convergence",
            "Convergence results available (or sigma series derived)",
            _nonempty_dir(root, "results/convergence")
            or _exists(root, "outputs/dashboard/convergence.parquet")
            or _exists(root, "outputs/figures/sigma_series.csv")
            or _exists(root, "outputs/figures/convergence_sigma.png"),
            artifact="results/convergence|outputs/figures/convergence_sigma.png",
        )
    )

    # --- Visualization / reporting / framework ---
    checks.append(
        _check(
            "VIZ_FIGURES",
            "visualization",
            "Figures written to outputs/figures",
            _nonempty_dir(root, "outputs/figures", "*.png")
            or _nonempty_dir(root, "outputs/figures", "*.html"),
            artifact="outputs/figures/",
        )
    )
    checks.append(
        _check(
            "HYP_EVALUATION",
            "framework",
            "Hypothesis evaluation JSON",
            _exists(root, "results/hypotheses/hypothesis_evaluation.json"),
            artifact="results/hypotheses/hypothesis_evaluation.json",
        )
    )
    checks.append(
        _check(
            "COMPARATIVE",
            "framework",
            "Comparative group analysis",
            _exists(root, "results/comparative/group_comparison.csv"),
            artifact="results/comparative/",
        )
    )
    checks.append(
        _check(
            "DASHBOARD_EXPORT",
            "framework",
            "Dashboard data package exported",
            _exists(root, "outputs/dashboard/panel.parquet")
            or _exists(root, "outputs/dashboard/manifest.json"),
            artifact="outputs/dashboard/",
        )
    )
    checks.append(
        _check(
            "FRAMEWORK_STATUS",
            "framework",
            "Intelligent framework status JSON",
            _exists(root, "outputs/dashboard/framework_status.json"),
            artifact="outputs/dashboard/framework_status.json",
        )
    )
    checks.append(
        _check(
            "REPORTS",
            "reporting",
            "Markdown reports generated",
            _exists(root, "outputs/reports/results_summary.md")
            and _exists(root, "outputs/reports/discussion_evidence.md"),
            artifact="outputs/reports/",
        )
    )
    checks.append(
        _check(
            "REPRODUCIBILITY",
            "audit",
            "Reproducibility metadata captured",
            _exists(root, "outputs/audit/reproducibility.json"),
            artifact="outputs/audit/reproducibility.json",
        )
    )
    checks.append(
        _check(
            "CONFIG_SNAPSHOT",
            "audit",
            "Config snapshot stored",
            _nonempty_dir(root, "outputs/audit/config_snapshot", "*.yaml"),
            artifact="outputs/audit/config_snapshot/",
        )
    )
    return checks


def build_audit(project_root: Path | str, config: Any = None) -> Dict[str, Any]:
    root = Path(project_root).resolve()
    criteria = _definition_of_complete_checks(root)
    n_pass = sum(1 for c in criteria if c["status"] == "pass")
    n_fail = sum(1 for c in criteria if c["status"] == "fail")
    critical_fails = [c for c in criteria if c["status"] == "fail" and c.get("critical")]

    by_layer: Dict[str, Dict[str, int]] = {}
    for c in criteria:
        layer = c["layer"]
        by_layer.setdefault(layer, {"pass": 0, "fail": 0})
        by_layer[layer][c["status"]] += 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": Path(root).name,
        "definition_of_complete": {
            "n_criteria": len(criteria),
            "n_pass": n_pass,
            "n_fail": n_fail,
            "pass_rate": n_pass / len(criteria) if criteria else 0.0,
            "critical_failures": [c["id"] for c in critical_fails],
            "complete": n_fail == 0,
            "critical_ok": len(critical_fails) == 0,
        },
        "by_layer": by_layer,
        "criteria": criteria,
    }


def _audit_markdown(audit: Dict[str, Any]) -> str:
    doc = audit.get("definition_of_complete") or {}
    lines = [
        "# Final framework audit",
        "",
        f"_Generated: {audit.get('generated_at')}_",
        "",
        "## Definition of Complete",
        "",
        f"- Criteria: **{doc.get('n_pass')}** pass / **{doc.get('n_fail')}** fail "
        f"(of {doc.get('n_criteria')})",
        f"- Pass rate: **{doc.get('pass_rate', 0):.1%}**",
        f"- Critical OK: **{doc.get('critical_ok')}**",
        f"- Fully complete: **{doc.get('complete')}**",
        "",
        "## Criteria",
        "",
        "| ID | Layer | Status | Critical | Description | Artifact |",
        "|---|---|---|---|---|---|",
    ]
    for c in audit.get("criteria") or []:
        lines.append(
            f"| `{c['id']}` | {c['layer']} | **{c['status']}** | "
            f"{'yes' if c.get('critical') else 'no'} | {c['description']} | "
            f"`{c.get('artifact') or ''}` |"
        )
    lines.append("")
    if doc.get("critical_failures"):
        lines += [
            "## Critical failures",
            "",
            *[f"- `{x}`" for x in doc["critical_failures"]],
            "",
        ]
    return "\n".join(lines)


def run(project_root: Path | str, config: Any = None) -> Dict[str, Any]:
    """
    Write ``outputs/audit/final_audit.json`` and ``final_audit.md``.

    Returns the audit dict. Raises ``FrameworkError`` only if critical artifacts
    are missing (raw data, DMI/analysis panel, model comparison).
    """
    root = Path(project_root).resolve()
    audit = build_audit(root, config)

    out_dir = ensure_dir(root / "outputs" / "audit")
    json_path = out_dir / "final_audit.json"
    md_path = out_dir / "final_audit.md"
    write_json(json_path, audit)
    md_path.write_text(_audit_markdown(audit), encoding="utf-8")
    logger.info(
        "Final audit: %s pass / %s fail (critical_ok=%s)",
        audit["definition_of_complete"]["n_pass"],
        audit["definition_of_complete"]["n_fail"],
        audit["definition_of_complete"]["critical_ok"],
    )

    critical_fails = audit["definition_of_complete"].get("critical_failures") or []
    # Map criterion ids to human messages for FrameworkError
    if critical_fails:
        # Only raise for the three truly critical artifact classes named in the brief
        raise_ids = {"DATA_RAW", "INDEX_DMI_PANEL", "INDEX_ANALYSIS_PANEL", "ML_MODEL_COMPARISON"}
        # DMI panel OR analysis panel: if either critical related to panel is ok, soften
        # Spec: raise only if critical artifacts missing (raw data, DMI panel, model comparison)
        hard = []
        if "DATA_RAW" in critical_fails:
            hard.append("raw data (data/raw/integrated_raw.parquet)")
        if "INDEX_DMI_PANEL" in critical_fails and "INDEX_ANALYSIS_PANEL" in critical_fails:
            hard.append("DMI/analysis panel")
        elif "INDEX_DMI_PANEL" in critical_fails and not _exists(
            root, "data/processed/analysis_panel.parquet"
        ):
            hard.append("DMI panel")
        elif "INDEX_ANALYSIS_PANEL" in critical_fails and not (
            _exists(root, "data/processed/dmi_panel.parquet")
            or _exists(root, "outputs/data/dmi_panel.parquet")
        ):
            hard.append("analysis panel")
        if "ML_MODEL_COMPARISON" in critical_fails:
            hard.append("model comparison (results/machine_learning/model_comparison.csv)")

        if hard:
            raise FrameworkError(
                "Critical framework artifacts missing: " + "; ".join(hard),
                details={
                    "critical_failures": critical_fails,
                    "missing": hard,
                    "audit_path": str(json_path.relative_to(root)),
                },
            )

    return audit
