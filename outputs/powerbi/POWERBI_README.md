# Power BI Desktop recipe (Windows)

Streamlit (`streamlit run src/dashboard/app.py`) is the VDA that runs on this
machine. Power BI Desktop is a Windows application; this folder does not contain
a `.pbix` because it cannot be generated or opened on macOS.

Related practice in HiTEc CA21163 (repository, Objective B1): Siljanoska Taskovska,
Savoska and Jolevski, *Integrating Python into Power BI for analyzing and predicting
digital development: Case study – Balkan countries*
(https://www.hitecaction.org/repository.php).

## Load the star schema

1. Open Power BI Desktop (Windows).
2. Home → Transform data → New parameter `FolderPath` (text) = this folder.
3. Home → Get data → Blank query → Advanced Editor → paste `LoadStarSchema.pq`.
4. Close and apply.

Relationships:
- fact_indicator_value[indicator_id] → dim_indicator[indicator_id]
- fact_indicator_value[country_iso3] → dim_country[country_iso3]
- fact_indicator_value[phase_id] → dim_phase[phase_id]
- fact_indicator_value[source_id] → dim_source[source_id]

Slicers: year (slider), country_name, group_name, indicator_id, provenance, phase_name.
Visuals: line chart of DMI by year; matrix indicator × country; donut of provenance.
Do not type values by hand. Refresh from this folder after each pipeline run.
