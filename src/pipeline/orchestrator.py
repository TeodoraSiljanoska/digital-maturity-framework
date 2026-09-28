"""End-to-end pipeline orchestrator with lazy stage wiring."""

from __future__ import annotations

import importlib
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from common.errors import FrameworkError, PipelineStageError
from common.io import ensure_dir, write_json
from common.logging_utils import get_logger, setup_logging
from common.seeds import set_global_seed
from pipeline.pipeline_config import PipelineConfig
from pipeline.pipeline_state import STAGE_ORDER, PipelineStage, PipelineState

# Prefer stdlib; fall back for older environments
try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover
    import importlib_metadata  # type: ignore


# Map each pipeline stage to its orchestrator method name
STAGE_HANDLERS: Dict[PipelineStage, str] = {
    PipelineStage.RAW_DATA_ACQUIRED: "acquire_data",
    PipelineStage.DATA_VALIDATED: "validate_data",
    PipelineStage.DATA_PROCESSED: "process_data",
    PipelineStage.INDEX_CREATED: "create_index",
    PipelineStage.STATISTICS_COMPLETED: "run_statistics",
    PipelineStage.ECONOMETRICS_COMPLETED: "run_econometrics",
    PipelineStage.ML_COMPLETED: "run_ml",
    PipelineStage.XAI_COMPLETED: "run_xai",
    PipelineStage.CONVERGENCE_COMPLETED: "run_convergence",
    PipelineStage.VISUALIZATION_COMPLETED: "run_visualization",
    PipelineStage.REPORTING_COMPLETED: "run_reporting",
    PipelineStage.FRAMEWORK_VALIDATED: "run_audit",
}


class PipelineOrchestrator:
    """Coordinate research stages, state, and reproducibility artifacts."""

    STAGE_ORDER = STAGE_ORDER

    def __init__(self, project_root: Path | str):
        self.project_root = Path(project_root).resolve()
        self.config = PipelineConfig.load(self.project_root)
        self.state = PipelineState.load(self.project_root)

        log_dir = self.project_root / "outputs" / "audit"
        ensure_dir(log_dir)
        self.logger = setup_logging(log_dir)
        self.logger = get_logger("dmf.pipeline")

        seed = self.config.random_seed()
        set_global_seed(seed)

        # Stage callables registered for later wiring / overrides
        self._stage_callables: Dict[PipelineStage, Callable[[], Any]] = {}
        self._register_default_stages()

        self._capture_reproducibility()
        self._snapshot_config()

    def _register_default_stages(self) -> None:
        for stage, method_name in STAGE_HANDLERS.items():
            self._stage_callables[stage] = getattr(self, method_name)

    def register_stage(
        self, stage: PipelineStage | str, fn: Callable[[], Any]
    ) -> None:
        """Override or wire a stage callable."""
        self._stage_callables[PipelineStage(stage)] = fn

    # ------------------------------------------------------------------
    # Reproducibility / audit helpers
    # ------------------------------------------------------------------

    def _audit_dir(self) -> Path:
        return ensure_dir(self.project_root / "outputs" / "audit")

    def _capture_reproducibility(self) -> Path:
        packages_of_interest = [
            "pandas",
            "numpy",
            "pyarrow",
            "PyYAML",
            "scikit-learn",
            "xgboost",
            "lightgbm",
            "catboost",
            "shap",
            "lime",
            "statsmodels",
            "linearmodels",
            "matplotlib",
            "plotly",
            "scipy",
            "pydantic",
            "joblib",
        ]
        versions: Dict[str, Optional[str]] = {}
        for name in packages_of_interest:
            try:
                versions[name] = importlib_metadata.version(name)
            except importlib_metadata.PackageNotFoundError:
                versions[name] = None

        # Also include any already-imported packages that match interest names
        for dist in importlib_metadata.distributions():
            dist_name = dist.metadata.get("Name") or dist.metadata.get("name")
            if dist_name and dist_name.lower() in {p.lower() for p in packages_of_interest}:
                versions[dist_name] = dist.version

        payload = {
            "python_version": sys.version,
            "python_version_info": list(sys.version_info[:3]),
            "platform": sys.platform,
            # Names only: absolute paths would expose the local directory layout.
            "executable": Path(sys.executable).name,
            "package_versions": versions,
            "seed": self.config.random_seed(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "project_root": Path(self.project_root).name,
        }
        path = self._audit_dir() / "reproducibility.json"
        write_json(path, payload)
        self.logger.info("Wrote reproducibility metadata to %s", path)
        return path

    def _snapshot_config(self) -> Path:
        src = self.project_root / "config"
        dest = ensure_dir(self._audit_dir() / "config_snapshot")
        if src.is_dir():
            for item in src.iterdir():
                if item.is_file() and item.suffix.lower() in {".yaml", ".yml"}:
                    shutil.copy2(item, dest / item.name)
        self.logger.info("Copied config snapshot to %s", dest)
        return dest

    def _lazy_call(
        self,
        module_name: str,
        attr: str = "run",
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Import ``module_name`` and call ``attr`` with project_root/config."""
        module = importlib.import_module(module_name)
        fn = getattr(module, attr)
        return fn(self.project_root, self.config, *args, **kwargs)

    def _call_optional(
        self,
        specs: Sequence[tuple[str, str]],
        *,
        required: bool = False,
    ) -> List[Any]:
        """
        Try (module, attr) specs in order; skip missing modules unless required.

        Used while downstream packages are still being wired.
        """
        results: List[Any] = []
        last_error: Optional[Exception] = None
        for module_name, attr in specs:
            try:
                results.append(self._lazy_call(module_name, attr))
            except ModuleNotFoundError as exc:
                last_error = exc
                self.logger.warning(
                    "Stage module not available yet: %s (%s)", module_name, exc
                )
            except AttributeError as exc:
                last_error = exc
                self.logger.warning(
                    "Stage entrypoint missing: %s.%s (%s)", module_name, attr, exc
                )
        if required and not results and last_error is not None:
            raise last_error
        return results

    # ------------------------------------------------------------------
    # Stage methods (lazy imports into domain packages)
    # ------------------------------------------------------------------

    def _require_first(self, specs: Sequence[tuple[str, str]]) -> Any:
        """Call the first available (module, attr); raise if none succeed."""
        last_error: Optional[Exception] = None
        for module_name, attr in specs:
            try:
                return self._lazy_call(module_name, attr)
            except ModuleNotFoundError as exc:
                last_error = exc
                self.logger.warning("Module unavailable: %s (%s)", module_name, exc)
            except AttributeError as exc:
                last_error = exc
                self.logger.warning("Entrypoint missing: %s.%s (%s)", module_name, attr, exc)
        raise ModuleNotFoundError(
            f"No stage implementation found among {[m for m, _ in specs]}"
        ) from last_error

    def acquire_data(self) -> Any:
        return self._require_first(
            [
                ("ingestion.acquire", "run"),
                ("ingestion", "run"),
            ],
        )

    def validate_data(self) -> Any:
        return self._require_first(
            [
                ("validation.validate", "run"),
                ("validation", "run"),
            ],
        )

    def process_data(self) -> Any:
        """Preprocess then feature-engineer."""
        pre = self._require_first(
            [
                ("preprocessing.pipeline", "run"),
                ("preprocessing", "run"),
            ],
        )
        feats = self._require_first(
            [
                ("feature_engineering.pipeline", "run"),
                ("feature_engineering", "run"),
            ],
        )
        return {"preprocessing": pre, "feature_engineering": feats}

    def create_index(self) -> Any:
        return self._require_first(
            [
                ("index.build", "run"),
                ("index", "run"),
            ],
        )

    def run_statistics(self) -> Any:
        return self._require_first(
            [
                ("statistics.descriptive", "run"),
                ("statistics", "run"),
            ],
        )

    def run_econometrics(self) -> Any:
        return self._require_first(
            [
                ("econometrics.models", "run"),
                ("econometrics", "run"),
            ],
        )

    def run_ml(self) -> Any:
        return self._require_first(
            [
                ("machine_learning.train", "run"),
                ("machine_learning", "run"),
            ],
        )

    def run_xai(self) -> Any:
        return self._require_first(
            [
                ("explainability.explain", "run"),
                ("explainability", "run"),
            ],
        )

    def run_convergence(self) -> Any:
        return self._require_first(
            [
                ("convergence.analysis", "run"),
                ("convergence", "run"),
            ],
        )

    def run_visualization(self) -> Any:
        return self._require_first(
            [
                ("visualization.plots", "run"),
                ("visualization", "run"),
            ],
        )

    def run_reporting(self) -> Any:
        """Hypotheses, comparative analysis, dashboard export, framework outputs."""
        parts = {
            # Runs first: hypothesis verdicts consume its MI-pooled and
            # leakage-safe evidence when the panel contains imputed cells.
            "imputation_robustness": self._require_first(
                [("imputation.robustness", "run")]
            ),
            "hypotheses": self._require_first(
                [("hypotheses.testing", "run"), ("hypotheses", "run")]
            ),
            "comparative": self._require_first(
                [("reporting.comparative", "run")]
            ),
            "ontology": self._require_first(
                [("ontology.export", "run"), ("ontology", "run")]
            ),
            "powerbi": self._require_first(
                [("dashboard.powerbi_export", "run")]
            ),
            "dashboard": self._require_first(
                [("dashboard.export", "run"), ("dashboard", "run")]
            ),
            "framework": self._require_first(
                [("framework.outputs", "run"), ("framework", "run")]
            ),
            "reporting": self._require_first(
                [("reporting.generate", "run"), ("reporting", "run")]
            ),
        }
        return parts

    def run_audit(self) -> Any:
        """Final framework validation / audit stage."""
        self._capture_reproducibility()
        self._snapshot_config()
        return self._require_first(
            [
                ("audit.validate", "run"),
                ("audit", "run"),
            ],
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _resolve_stage_slice(
        self,
        from_stage: Optional[PipelineStage | str],
        to_stage: Optional[PipelineStage | str],
    ) -> List[PipelineStage]:
        stages = list(STAGE_ORDER)
        start_idx = 0
        end_idx = len(stages) - 1
        if from_stage is not None:
            start_idx = stages.index(PipelineStage(from_stage))
        if to_stage is not None:
            end_idx = stages.index(PipelineStage(to_stage))
        if start_idx > end_idx:
            raise ValueError(
                f"from_stage ({from_stage}) must precede to_stage ({to_stage})"
            )
        return stages[start_idx : end_idx + 1]

    def run(
        self,
        from_stage: Optional[PipelineStage | str] = None,
        to_stage: Optional[PipelineStage | str] = None,
        *,
        skip_completed: bool = False,
    ) -> PipelineState:
        """
        Execute pipeline stages in order.

        Prior outputs are preserved; failures are recorded without deleting
        artifacts from earlier successful stages.
        """
        if from_stage is not None:
            self.state.reset_from(from_stage)

        selected = self._resolve_stage_slice(from_stage, to_stage)
        self.logger.info(
            "Starting pipeline: %s -> %s",
            selected[0].value,
            selected[-1].value,
        )

        for stage in selected:
            if skip_completed and self.state.is_completed(stage):
                self.logger.info("Skipping completed stage: %s", stage.value)
                continue

            method_name = STAGE_HANDLERS[stage]
            fn = self._stage_callables.get(stage) or getattr(self, method_name)
            self.logger.info("=== START %s (%s) ===", stage.value, method_name)
            try:
                fn()
            except PipelineStageError as exc:
                self.logger.exception("Stage failed: %s", stage.value)
                self.state.mark_failed(stage, exc, details=exc.details)
                raise
            except FrameworkError as exc:
                wrapped = PipelineStageError(
                    stage.value, exc.message, cause=exc
                )
                self.logger.exception("Stage failed: %s", stage.value)
                self.state.mark_failed(stage, wrapped, details=wrapped.details)
                raise wrapped from exc
            except Exception as exc:
                wrapped = PipelineStageError(
                    stage.value,
                    f"Unhandled error in stage {stage.value}: {exc}",
                    cause=exc,
                )
                self.logger.exception("Stage failed: %s", stage.value)
                self.state.mark_failed(stage, wrapped, details=wrapped.details)
                raise wrapped from exc

            self.state.mark_completed(stage)
            self.logger.info("=== END %s ===", stage.value)

        self.logger.info("Pipeline finished successfully")
        return self.state
