# Research result versions

The study is reported in two analytical arms plus one preserved historical snapshot.
All three run the same pipeline; they differ only in how missing official cells are
handled and in whether the measurement corrections of August 2026 are applied.

| Version | Path | Missing-data policy | Corrections applied | Role |
|---|---|---|---|---|
| **v1 (preserved, do not edit)** | `versions/v1_official_unbalanced/` | Official gaps retained | No — predates them | Historical provenance only |
| **v1b (robustness arm)** | `versions/v1b_official_corrected/` | Official gaps retained | Yes | Credibility check: what survives with nothing reconstructed |
| **v2 (inferential arm, active `outputs/` / `results/`)** | project root, snapshot in `versions/v2_mice_imputed/` | **MICE** completes the panel; observed values never overwritten | Yes | Primary results and inference |

Read `v1b` and `v2` together. `v1` is retained so the earlier figures remain
traceable, but where it disagrees with `v1b` the corrected numbers supersede it.

## Method choice for v2

**MICE (Multiple Imputation by Chained Equations)** via `sklearn.impute.IterativeImputer`
with `RandomForestRegressor` (tree-based MICE).

Rationale from recent literature (2024–2026):

- MICE remains the dominant multivariate imputation family in empirical research (e.g. ACM SIGMOD/PACMMOD 2024 in-database MICE).
- 2024 comparative work shows tree-based MICE / RF chained imputation is competitive with classical MICE-PMM for inference.
- In software-engineering datasets, **KNN imputation** is also widely used; MICE was selected here because digitalisation indicators are multivariate and interdependent (better suited to chained equations than local neighbor averaging alone).

Imputed cells are **statistical reconstructions**, not new official statistics. See
`outputs/audit/imputation_report.json` and `data/processed/imputation_mask.parquet`.

The masked-cell benchmark quantifies how good those reconstructions are: hiding 15% of
observed cells and re-imputing gives a normalised RMSE of 0.224, against 0.406 for a
country-mean fill and 1.042 for a global-median fill.

## v2 revision 2 (2026-08-11)

The second version was re-checked end to end. Five changes were made; each one is
config-driven and leaves an artifact trail.

### Data integrity

1. **X7 scale break fixed.** Oxford Insights published Government AI Readiness on a
   0–10 scale through the 2019 edition and on a 0–100 scale from 2020. The panel
   spliced both without rescaling, creating a spurious ~57-point jump in 2020 for
   every country. `config/preprocessing.yaml → scale_harmonization` now rescales
   pre-2020 values by 10.
2. **Carry-forward reach corrected.** The gap-limited fill ran twice (grouped by
   `group_id`, then by country), which doubled the effective `max_gap_years` and let
   a single biennial observation cover five years. It now runs once per country.
3. **Cell provenance recorded.** `data/processed/cell_provenance.parquet` labels every
   indicator cell `official` / `carried_forward` / `mice_imputed`, so authenticity is
   quantified rather than asserted: 65.8% official, 25.2% carried forward, 9.0% imputed.

### Inference validity

4. **Multiple imputation with Rubin pooling.** The single completed panel understates
   standard errors. Ten predictive-mean-matching draws are now estimated separately and
   pooled (`results/imputation/mi_pooled_fe.csv`), reporting the fraction of missing
   information per coefficient.
5. **Leakage-safe predictive evaluation.** The point panel is imputed using all years,
   so hold-out ML metrics computed on it look ahead. A parallel horse-race fits the
   imputer on training years only (`results/imputation/leakage_safe_ml.csv`).

Additionally: VIF is computed with an intercept (previous values were inflated by
orders of magnitude), and a negative Hausman statistic is reported as **inconclusive**
with model choice decided by a Mundlak auxiliary regression instead of being read as
evidence for random effects.

## v1b: the official-data-only arm (2026-08-11)

Reproduce with:

```bash
python scripts/run_v1b_track.py                       # pipeline + snapshot + VERSION.txt
python scripts/build_results_discussion_v1b_docx.py   # chapter with v1b-vs-v2 comparison
```

The script builds a throwaway workspace (`.v1b_run/`), copies `config/` and the source
data into it, disables the multivariate stage there, and runs the standard orchestrator
against that root — so the active `outputs/` and `results/` trees are never touched. The
run is deterministic: repeating it reproduces the chapter text exactly.

### What v1b establishes

| Question | Answer |
|---|---|
| Is the DMI an artifact of imputation? | No. The two index series correlate at r = 0.996 (Spearman 0.992), mean absolute difference 0.98 index points. |
| Can the econometrics be run on official data alone? | No. Listwise deletion cuts the estimation sample from 156 rows to 57 for 12 regressors plus 12 country effects; VIF reaches 70, EGDI flips sign, GDP per capita collapses to p = 0.99, and group-level models cannot be fitted. |
| Do machine learning models beat classical ones? | No. Ridge (RMSE 2.023) and OLS (2.228) beat every non-linear learner (best XGBoost 2.737) — matching the leakage-safe v2 result and contradicting only the leaky point-panel comparison. |
| Does convergence hold? | Unconditionally yes (β = −0.677, p < 0.001). Conditionally it is significant here (p = 0.049) but not in v2 (p = 0.077), so it should be reported as suggestive. |

Five of the nine hypothesis verdicts agree across the two arms; four differ. Two of the
agreements (H1.1, H1.7) share a verdict label on different evidence — v1b reaches
`partially_supported` with **no** variable significant, v2 with one of two — so they rest
on the completed panel alone.

### Consequence for the hypotheses

`results/hypotheses/hypothesis_evaluation.json → robustness_summary` records each verdict
under the point panel, the MI-pooled specification and complete cases. Reading that
together with v1b:

- **H1.6 (ML beats classical models) — report as not supported.** Two independent designs
  reject it; only the leaky configuration supports it.
- **H1.2 (e-government) — report as suggestive.** Supported on the point panel, weakened
  by MI pooling (p = 0.079), untestable in v1b.
- **H1.1 / H1.7 — report as resting on the completed panel**, with broadband and
  cybersecurity carrying the effects and the innovation index contributing nothing.
- **H1.3, H1.5, H1.8 — durable.** Identical verdicts on identical evidence in both arms.

> Note: the frozen v1 artifacts predate these corrections and still contain the X7
> scale break and the doubled carry-forward reach. Use v1b for any like-for-like
> comparison against v2.
