# Semantic mapping σ (framework explainability)

Domain-neutral OWL identifiers get digital-maturity readings here. This table is
analogous to the F&A semantic mapping in Savoska & Loshkovska (2026, §4.6):
identifiers stay domain-neutral; the domain reading is documented outside the
TBox. It is not an `owl:imports` axiom. It makes the **framework** explainable:
why an indicator sits in a pillar, which phase uses it, and how the cell
was obtained.

| Individual | σ reading (digital maturity) |
|---|---|
| `Ind_X1` | Mobile access density; saturated in this panel; infrastructure pillar; obtained by World Bank API |
| `Ind_X2` | Fixed high-capacity connectivity; primary infrastructure driver of DMI |
| `Ind_X3` | E-government development (UN EGDI); biennial snapshot; e-government pillar |
| `Ind_X4` | Skills proxy (Internet users); skills pillar; API |
| `Ind_X5` | National cybersecurity capacity; sparse snapshot; trust-innovation pillar |
| `Ind_X6` | Innovation capacity (GII); trust-innovation; snapshot |
| `Ind_X7` | Government AI readiness; scale-harmonised 2019/2020; snapshot |
| `Ind_X8` | R&D intensity; digital-economy pillar; API |
| `Ind_X9` | ICT service-export intensity (ICT-sector proxy); API |
| `Ind_X10` | Online public services (UN OSI); biennial snapshot |
| `Ind_C1` | Economic development control (log GDP per capita PPP); not in DMI |
| `Ind_C2` | Human-capital control (UNDP Education Index); not in DMI |
| `Ind_DMI` | Dependent variable Y; equal-weight five-pillar composite, scale 0–100 |
| `Phase_Acquire` | Obtain official cells via API or snapshot; never fabricate |
| `Phase_Process` | Scale fix, ≤2-year carry, optional MICE; record provenance |
| `Phase_Index` | Min–max normalise; aggregate pillars; emit DMI |
| `Phase_Analyse` | Correlation, VIF with intercept, FE/RE, Mundlak, convergence |
| `Phase_Predict` | Temporal hold-out 2024–2025; ensembles vs linear baselines |
| `Phase_Explain` | SHAP, LIME, PDP as Explanation individuals |
| `Phase_Visualize` | Streamlit + Power BI star schema + process flowcharts |
| `Method_official_api` | Secondary official statistics, live API; HiTEc CA21163 kind: traditional |
| `Method_curated_snapshot` | Secondary documentary source, versioned CSV; HiTEc kind: traditional |
| `Method_primary_survey` | Unused; would map to HiTEc perceptions/imprecise data |
| `Method_sensor_automated` | Unused; HiTEc sensor kind is in the MoU, not in this panel |
| `Kind_traditional` | Only HiTEc data kind realised in this study |
| `Kind_text` / `Kind_functional` / `Kind_perceptions` / `Kind_sensor` | Declared, unused |
| `CTX_overview_dashboard` | Monitor DMI by country / group with year slider |
| `CTX_hypothesis_confirmation` | Confirm H1.x with FE + SHAP overlap |
| `CTX_prediction_xai` | Predict DMI and explain binding indicators |

VTE-style rationale (Visibility / Interpretability / Insight) for VDA choices:

- **Visibility:** country–year DMI trajectories and choropleth of latest DMI.
- **Interpretability:** pillar decomposition and provenance shares (official / carried / imputed).
- **Insight:** SHAP ranking vs FE signs; two contrasting scenarios (North Macedonia vs Germany, 2025). The scenario rows are raw indicator values of globally important SHAP features, not local SHAP.
