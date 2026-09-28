# Four-part methodology (mentor comments 2–3, 7–8)

**COST Action (locked from the public record).** HiTEc —
*Text, functional and other high-dimensional data in econometrics: New models,
methods, applications* — **CA21163**. Mentor Savoska is listed on WG 1 (Data)
at https://www.hitecaction.org/data.php. MoU:
https://www.cost.eu/actions/CA21163

HiTEc does **not** publish a numbered list titled “four obtaining methods”.
The MoU names four **data kinds** used together in econometric work, plus an
emphasis on sensors:

| HiTEc kind (MoU) | Used in this study? |
|---|---|
| Traditional official series | **yes** — World Bank API and curated published indices |
| Text data (newspapers, articles, opinions) | no |
| Functional data (curves, continuously recorded functions) | no |
| Perceptions / imprecise data (polls, non-precise values) | no as primary collection |
| Sensor data (MoU emphasis, not a fifth “obtaining method”) | no |

Retrieval procedures for the traditional series (ontology `CollectionMethod`):
`official_api`, `curated_snapshot` (used); `primary_survey`, `sensor_automated`
(unused; they sit next to HiTEc perceptions and sensor kinds).

## Four methodology parts

| Part | Macedonian framing (mentor) | This study |
|---|---|---|
| **M1 Collection framework** | рамка за собирање на податоците | Panel design: 12 countries × 2012–2025, ISO-3 join key, indicator catalog with project IRIs (not a registered PURL), source adapters |
| **M2 Collection methods** | методи на добивање | HiTEc traditional series via API + curated snapshot. Text, functional, perception and sensor kinds unused |
| **M3 Preparation for VDA** | подготовка на податоците за визуелна податочна анализа | Validate → scale-harmonise X7 → gap-limited carry → optional MICE → lag features → DMI pillars |
| **M4 VDA presentation** | приказ со VDA | Process flowcharts, Streamlit (country / indicator / phase / provenance + year slider); Power BI star-schema CSV + Power Query (Windows Desktop; no `.pbix` on macOS) |

## Model-to-phase map (mentor comment 3)

ML models are **not** collection methods. They belong to later process phases.

| Method | Process phase | Purpose |
|---|---|---|
| World Bank API, snapshot loaders | `Phase_Acquire` | Obtain official cells |
| Validation, scale fix, carry-forward, MICE | `Phase_Process` | Make the panel VDA-ready |
| DMI weighted pillars | `Phase_Index` | Composite dependent variable Y |
| Descriptive stats, VIF, correlation | `Phase_Analyse` | Structure of the panel |
| FE / RE / Mundlak / convergence | `Phase_Analyse` | Within-country inference |
| OLS, Ridge | `Phase_Predict` | Linear baselines for H1.6 |
| Random Forest, XGBoost, LightGBM, CatBoost, SVR, MLP | `Phase_Predict` | Non-linear prediction of DMI |
| SHAP, LIME, PDP, native importance | `Phase_Explain` | Result explainability |
| Streamlit, Power BI, process diagrams | `Phase_Visualize` | Interactive VDA |

Dependent variable: **DMI (Y)**. Controls: **C1** (log GDP per capita), **C2** (Education Index). Independents: **X1–X10**.
