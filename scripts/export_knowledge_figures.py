"""Figures of the knowledge-layer evaluation (300 dpi PNG, English labels).

    python scripts/export_knowledge_figures.py

Reads only artifacts written by run_knowledge_layer.py, run_fault_injection.py
and build_powerbi_project.py, and the Protégé / Power BI screenshots stored next
to them; writes outputs/knowledge_layer/figures/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KL = ROOT / "outputs" / "knowledge_layer"
OUT = KL / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#e34948"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
GROUPS = {"balkan": ("Balkan", BLUE), "developed_eu": ("Developed European", ORANGE),
          "developed_non_eu": ("Developed non-European", AQUA), "developing": ("Developing", VIOLET)}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white",
})


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ Figure 1
def architecture():
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")
    cols = [
        ("Sources", ["World Bank|WDI API|(X1, X2, X4, X8,|X9, C1, Z1–Z6)", "Publisher|releases|(UN DESA, eGA,|WIPO, Oxford|Insights, UNDP)"]),
        ("Acquisition &|validation", ["Adapters,|frozen vintages,|SHA-256", "Data contract,|edition families,|break diagnostic"]),
        ("Processing &|index", ["Carry ≤ 2 years,|MICE; published|cells kept", "Cell provenance,|reconstruction|scope", "DMI: five|pillars, min–max,|weights 0.2"]),
        ("Analysis &|learning", ["Panel FE,|MI pooling,|Driscoll–Kraay", "Rolling-origin|forecasts vs.|persistence", "SHAP, LIME,|PDP"]),
        ("Knowledge|layer", ["OWL 2 DL graph|(QB, PROV-O,|SKOS, DCAT)", "HermiT, SHACL,|QB integrity|constraints", "SPARQL CQs,|JSON-LD|metadata"]),
        ("Visual|analytics", ["Power BI|project|(TMDL + PBIR)", "Streamlit +|Plotly", "Views defined|once as|dmi:AnalyticView"]),
    ]
    w, gap, x0 = 13.4, 3.4, 0.6
    fills = ["#eaf1fb", "#eaf1fb", "#fdeee7", "#fdeee7", "#e6f6ef", "#ece9f7"]
    edges = [BLUE, BLUE, ORANGE, ORANGE, AQUA, VIOLET]
    top, bottom = 51.0, 10.5
    for i, (title, boxes) in enumerate(cols):
        x = x0 + i * (w + gap)
        ax.text(x + w / 2, 57.5, title.replace("|", "\n"), ha="center", va="center", fontsize=7.2, fontweight="bold", color=INK)
        n = len(boxes)
        h = (top - bottom - (n - 1) * 1.6) / n
        for j, b in enumerate(boxes):
            y = top - (j + 1) * h - j * 1.6
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.0",
                                        fc=fills[i], ec=edges[i], lw=0.9))
            ax.text(x + w / 2, y + h / 2, b.replace("|", "\n"), ha="center", va="center", fontsize=5.7, color=INK, linespacing=1.12)
        if i < len(cols) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.4, (top + bottom) / 2), (x + w + gap - 0.4, (top + bottom) / 2),
                                         arrowstyle="-|>", mutation_scale=9, color=INK2, lw=1.0))
    span = 6 * w + 5 * gap
    ax.add_patch(FancyBboxPatch((x0, 1.0), span, 6.2, boxstyle="round,pad=0.2,rounding_size=1.0",
                                fc="#f4f3f0", ec=MUTED, lw=0.7))
    ax.text(x0 + span / 2, 4.1,
            ("Carried end to end for every cell: indicator × country × year key, edition, vintage,|"
             "retrieval channel, file checksum, provenance state and reconstruction scope").replace("|", "\n"),
            ha="center", va="center", fontsize=6.0, color=INK2, linespacing=1.2)
    save(fig, "fig1_architecture.png")


# ------------------------------------------------------------------ Figure 2
def provenance_raster():
    cells = pd.read_csv(KL / "ontology/cells.csv")
    zp = pd.read_csv(KL / "context/context_provenance.csv")
    gaps = zp.melt(id_vars=["country_iso3", "year"], var_name="indicator_id", value_name="provenance")
    gaps = gaps[gaps.provenance == "missing"]
    allc = pd.concat([cells[["indicator_id", "country_iso3", "year", "provenance"]], gaps], ignore_index=True)
    allc = allc[allc.indicator_id != "DMI"]
    order = [f"X{i}" for i in range(1, 11)] + ["C1", "C2"] + [f"Z{i}" for i in range(1, 7)]
    countries = ["MKD", "SRB", "SVN", "DEU", "FRA", "GBR", "AUS", "CAN", "USA", "ARG", "BRA", "IND"]
    code = {"official": 0, "carried_forward": 1, "mice_imputed": 2, "missing": 3}
    years = list(range(2012, 2026))
    mat = np.full((len(order) * len(countries), len(years)), np.nan)
    lookup = allc.set_index(["indicator_id", "country_iso3", "year"])["provenance"]
    for i, ind in enumerate(order):
        for j, c in enumerate(countries):
            for k, y in enumerate(years):
                v = lookup.get((ind, c, y))
                mat[i * len(countries) + j, k] = code.get(v, 3)
    cmap = ListedColormap([BLUE, "#9fdcc3", ORANGE, "#d9d8d4"])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 7.6), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.55})
    half = 9
    for p, ax in enumerate(axes):
        sub = order[p * half:(p + 1) * half]
        m = mat[p * half * 12:(p + 1) * half * 12]
        ax.imshow(m, aspect="auto", cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest")
        ax.grid(False)
        ax.set_xticks(range(len(years))); ax.set_xticklabels([str(y)[2:] for y in years], fontsize=6)
        ax.set_yticks([i * 12 + 5.5 for i in range(len(sub))]); ax.set_yticklabels(sub, fontsize=7.5, fontweight="bold")
        for i in range(1, len(sub)):
            ax.axhline(i * 12 - 0.5, color="white", lw=1.6)
        ax.set_yticks([i * 12 + j for i in range(len(sub)) for j in range(12)], minor=True)
        ax.tick_params(axis="y", which="minor", length=0)
        ax.set_xlabel("Year (20··)")
        for spine in ax.spines.values():
            spine.set_visible(False)
        sec = ax.secondary_yaxis("right")
        sec.set_yticks([i * 12 + j for i in range(len(sub)) for j in range(12)])
        sec.set_yticklabels(countries * len(sub), fontsize=3.6, color=MUTED)
        sec.tick_params(length=0)
        for s in sec.spines.values():
            s.set_visible(False)
    handles = [Patch(fc=BLUE, label="Published"), Patch(fc="#9fdcc3", label="Carried (≤ 2 years)"),
               Patch(fc=ORANGE, label="Reconstructed (MICE)"), Patch(fc="#d9d8d4", label="Missing")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.935), fontsize=7)
    save(fig, "fig2_provenance_raster.png")


# ------------------------------------------------------------------ Figure 3
def ontology_diagram():
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 70.5); ax.axis("off")

    def node(x, y, text, sub=None, color=BLUE, w=17, h=6.0):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.2,rounding_size=1.0",
                                    fc="white", ec=color, lw=1.1))
        three = bool(sub) and sub.count("\n") >= 2
        ax.text(x, y + (3.1 if three else 0.9 if sub else 0), text, ha="center", va="center", fontsize=6.6, fontweight="bold", color=INK)
        if sub:
            ax.text(x, y - (1.0 if three else 1.5), sub, ha="center", va="center", fontsize=5.2, color=INK2, style="italic")

    def edge(a, b, label, rad=0.0, lx=0, ly=0):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, color=INK2, lw=0.8,
                                     connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
        mx, my = (a[0] + b[0]) / 2 + lx, (a[1] + b[1]) / 2 + ly
        ax.text(mx, my, label, fontsize=5.2, color=INK2, ha="center", va="center",
                bbox=dict(fc="white", ec="none", pad=0.4))

    node(50, 38, "dmi:ScoreEntry", "⊑ qb:Observation, prov:Entity", ORANGE, w=22)
    node(17, 56, "dmi:Indicator", "Core · Control · Composite · Context", BLUE, w=24)
    node(50, 61, "dmi:TimeSeries", "nOfficial, officialShare, longestGap", BLUE, w=25)
    node(84, 56, "dmi:Country", "memberOf → dmi:CountryGroup", BLUE, w=22)
    node(17, 38, "dmi:Pillar", "⊑ skos:Concept", BLUE, w=17)
    node(84, 38, "dmi:ProvenanceState", "Official · Carried · Reconstructed ·\nConstructed · Missing", BLUE, w=24, h=7.5)
    node(17, 20, "dmi:DataSource", "⊑ dcat:Dataset, prov:Entity", AQUA, w=22)
    node(17, 6, "dmi:Publisher", "⊑ prov:Organization", AQUA, w=18)
    node(50, 18, "dmi:ProcessPhase", "⊑ prov:Activity", AQUA, w=20)
    node(84, 21, "dmi:ModelOutput", "Estimate · ForecastScore · Explanation ·\nDispersionReading; estimateFor → Indicator", VIOLET, w=30, h=7.5)
    node(85, 6, "dmi:AnalyticView", "showsIndicator · servesContext ·\nsupportsTask · slicerField", VIOLET, w=28, h=7.5)
    node(47.5, 5.5, "Defined classes (classified by HermiT)", "OfficialEntry · ReconstructedEntry · BackcastEntry ·\nContextEntry · ComponentEntry · BalkanEntry ·\nCompositeComponent · UnobservedSeries · SparseSeries",
         YELLOW, w=37, h=10.0)

    edge((41, 41), (24, 53), "entryIndicator", lx=-1)
    edge((50, 41), (50, 58), "inSeries")
    edge((59, 41), (77, 53), "entryCountry", lx=1)
    edge((17, 53), (17, 41), "belongsToPillar /\nrelatedPillar", lx=-6)
    edge((61, 38), (72, 38), "hasProvenance", ly=2.4)
    edge((41, 36), (26, 22), "entrySource\n⊑ prov:wasDerivedFrom", lx=-3)
    edge((50, 35), (50, 21), "entryPhase\n⊑ prov:wasGeneratedBy", lx=-0.5, ly=2)
    edge((17, 17), (17, 9), "publishedBy", lx=-5)
    edge((69, 21), (60, 18.5), "assignedToPhase", ly=2.2)
    ax.text(50, 68.5, "property chains: entryGroup = entryCountry ∘ memberOf;   entryPillar = entryIndicator ∘ belongsToPillar",
            ha="center", fontsize=5.6, color=INK2, style="italic")
    save(fig, "fig3_ontology.png")


# ------------------------------------------------------------------ Figure 4
def protege():
    # The ontology metrics shown by Protégé are reported in a table; the figure shows the inferred view.
    shots = ROOT / "outputs/knowledge_layer/ontology/screenshots"
    b = Image.open(shots / "protege_composite_component_inferred.png").crop((0, 105, 1246, 905))
    fig, ax = plt.subplots(figsize=(7.4, 7.4 * b.height / b.width))
    ax.imshow(b); ax.axis("off")
    save(fig, "fig4_protege.png")


# ------------------------------------------------------------------ Figure 5
def rolling_origin():
    sc = pd.read_csv(KL / "timeseries/rolling_origin_scores.csv")
    pr = pd.read_csv(KL / "timeseries/rolling_origin_predictions.csv")
    names = {"persistence": "Persistence", "ridge_arx": "Ridge + lagged DMI", "ols": "OLS", "ridge": "Ridge",
             "catboost_arx": "CatBoost + lagged DMI", "catboost": "CatBoost", "svr": "SVR", "xgboost": "XGBoost",
             "random_forest": "Random forest"}
    order = list(names)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.45})
    ax = axes[0]
    for t, (col, off, lab) in {"DMI_mice": (BLUE, -0.17, "Target: DMI (reconstructed panel)"),
                               "DMI_official_only": (ORANGE, 0.17, "Target: DMI (published cells only)")}.items():
        g = sc[sc.target == t].set_index("model").loc[order]
        y = np.arange(len(order)) + off
        ax.errorbar(g.delta_rmse_vs_persistence, y, xerr=[g.delta_rmse_vs_persistence - g.delta_ci_low, g.delta_ci_high - g.delta_rmse_vs_persistence],
                    fmt="o", ms=4, color=col, ecolor=col, elinewidth=1.2, capsize=0, label=lab)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([names[m] for m in order], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE minus persistence RMSE (DMI points)\n95% country-cluster bootstrap interval")
    ax.legend(loc="upper right", fontsize=6.0)
    ax.set_title("(a) Accuracy relative to persistence, 96 forecasts", fontsize=8, loc="left")
    ax = axes[1]
    g = pr[pr.target == "DMI_official_only"]
    for m, col, lab in (("persistence", INK2, "Persistence"), ("ridge_arx", BLUE, "Ridge + lagged DMI"),
                        ("catboost", ORANGE, "CatBoost"), ("random_forest", VIOLET, "Random forest")):
        r = g.assign(e=(g[f"pred_{m}"] - g.y) ** 2).groupby("origin")["e"].mean() ** 0.5
        ax.plot(r.index + 1, r.values, marker="o", ms=4, lw=1.6, color=col, label=lab)
    ax.set_xlabel("Forecast year (origin + 1)"); ax.set_ylabel("RMSE (DMI points)")
    ax.set_xticks(range(2018, 2026)); ax.set_xticklabels([str(y) for y in range(2018, 2026)], rotation=45, fontsize=6.5)
    ax.legend(fontsize=6.0, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    ax.set_title("(b) RMSE by forecast year, published-cells target", fontsize=8, loc="left")
    save(fig, "fig5_rolling_origin.png")


# ------------------------------------------------------------------ Figure 6
def context():
    a = pd.read_csv(KL / "association/context_association.csv")
    lab = {"Z1": "Z1 Secure Internet servers", "Z2": "Z2 ICT goods exports", "Z3": "Z3 High-tech exports",
           "Z4": "Z4 Tertiary enrollment", "Z5": "Z5 Researchers in R&D", "Z6": "Z6 Statistical performance (SPI)"}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.5})
    ax = axes[0]
    y = np.arange(len(a))
    ax.errorbar(a.beta_per_sd, y - 0.15, xerr=1.96 * a.se_driscoll_kraay, fmt="o", ms=4, color=BLUE, elinewidth=1.2, label="Driscoll–Kraay SE")
    ax.errorbar(a.beta_per_sd, y + 0.15, xerr=1.96 * a.se_cluster_country, fmt="s", ms=3.5, color=ORANGE, elinewidth=1.2, label="Country-clustered SE")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([f"{lab[z]}  (n={n})" for z, n in zip(a.indicator_id, a.n_obs)], fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlabel("DMI points per SD of the lagged indicator\n(two-way FE, 95% interval)")
    ax.legend(fontsize=6.0, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    ax.set_title("(a) Within association, published values only", fontsize=8, loc="left")
    ax = axes[1]
    d = pd.read_csv(KL / "association/dmi_plus.csv")
    for gid, (gl, col) in GROUPS.items():
        g = d[d.group_id == gid]
        ax.scatter(g.DMI, g.DMI_plus, s=10, color=col, label=gl, edgecolor="white", linewidth=0.4)
    lim = [10, 90]
    ax.plot(lim, lim, color=MUTED, lw=0.8, ls="--")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("DMI"); ax.set_ylabel("DMI+ (with Z1–Z6 in their related pillars)")
    ax.legend(fontsize=6.2, loc="upper left")
    ax.set_title("(b) Index extension, 168 country–years", fontsize=8, loc="left")
    save(fig, "fig6_context.png")


# --------------------------------------------------------------- Figures 7–8
def powerbi():
    shots = KL / "powerbi/screenshots"
    for name, files in (("fig7_powerbi_monitoring_provenance.png", ("page1_monitoring_balkan.png", "page2_provenance.png")),
                        ("fig8_powerbi_context_models.png", ("page3_context_Z2.png", "page4_models.png"))):
        ims = [Image.open(shots / f) for f in files]
        fig, axes = plt.subplots(2, 1, figsize=(7.4, 8.6), gridspec_kw={"hspace": 0.06})
        for ax, im, lab in zip(axes, ims, ("(a)", "(b)")):
            ax.imshow(im); ax.axis("off"); ax.set_title(lab, fontsize=8, loc="left")
        save(fig, name)


if __name__ == "__main__":
    architecture()
    provenance_raster()
    ontology_diagram()
    protege()
    rolling_origin()
    context()
    powerbi()
