"""Generate process flowcharts (PNG + Mermaid) from pipeline definitions — not hand-drawn slides."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from common.io import ensure_dir, write_json
from common.logging_utils import get_logger
from pipeline.pipeline_state import STAGE_ORDER

logger = get_logger("dmf.process_diagrams")

Box = Tuple[str, str]  # (id, label)

STAGE_IO = [
    ("RAW_DATA_ACQUIRED", "Acquire", "data/raw/integrated_raw.parquet"),
    ("DATA_VALIDATED", "Validate", "outputs/audit/validation_report.json"),
    ("DATA_PROCESSED", "Process", "data/processed/panel_wide.parquet"),
    ("INDEX_CREATED", "Index", "data/processed/dmi_panel.parquet"),
    ("STATISTICS_COMPLETED", "Descriptives", "results/descriptive/"),
    ("ECONOMETRICS_COMPLETED", "Econometrics", "results/econometrics/"),
    ("ML_COMPLETED", "Predict", "results/machine_learning/"),
    ("XAI_COMPLETED", "Explain", "results/xai/"),
    ("CONVERGENCE_COMPLETED", "Converge", "results/convergence/"),
    ("VISUALIZATION_COMPLETED", "Visualize", "outputs/figures/"),
    ("REPORTING_COMPLETED", "Report", "outputs/reports/"),
    ("FRAMEWORK_VALIDATED", "Audit", "outputs/audit/final_audit.json"),
]


def _draw_chain(
    boxes: Sequence[Box],
    out_path: Path,
    title: str,
    *,
    dpi: int = 150,
    note: str = "",
    wrap_at: int = 6,
) -> Path:
    n = len(boxes)
    cols = max(1, min(int(wrap_at), n))
    nrows = (n + cols - 1) // cols
    fig_w = max(12.0, cols * 2.2)
    fig_h = max(4.4, nrows * 2.2 + (0.8 if note else 0.4))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_xlim(0, cols)
    ax.set_ylim(-0.2 if note else 0.0, nrows)
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=10)
    cy: List[float] = []
    cx: List[float] = []
    for i, (_bid, label) in enumerate(boxes):
        r = i // cols
        c = i % cols
        y0 = (nrows - 1 - r) + 0.28
        x0 = c + 0.08
        rect = mpatches.FancyBboxPatch(
            (x0, y0),
            0.84,
            0.52,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            facecolor="#E8EEF7",
            edgecolor="#1F4E79",
            linewidth=1.6,
        )
        ax.add_patch(rect)
        ax.text(c + 0.5, y0 + 0.26, label, ha="center", va="center", fontsize=11)
        cx.append(c + 0.5)
        cy.append(y0 + 0.26)
    for i in range(n - 1):
        same_row = (i // cols) == ((i + 1) // cols)
        if same_row:
            ax.annotate(
                "",
                xy=(cx[i + 1] - 0.42, cy[i + 1]),
                xytext=(cx[i] + 0.42, cy[i]),
                arrowprops=dict(arrowstyle="->", color="#1F4E79", lw=2.5),
            )
        else:
            ax.annotate(
                "",
                xy=(cx[i + 1], cy[i + 1] + 0.28),
                xytext=(cx[i], cy[i] - 0.28),
                arrowprops=dict(arrowstyle="->", color="#1F4E79", lw=2.5),
            )
    if note:
        ax.text(cols / 2, -0.08, note, ha="center", va="center", fontsize=10, style="italic")
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _draw_split(
    title: str,
    used: Sequence[Box],
    unused: Sequence[Box],
    out_path: Path,
    *,
    dpi: int = 150,
) -> Path:
    n_rows = max(len(used), len(unused), 1)
    fig, ax = plt.subplots(figsize=(11, 1.6 + n_rows * 0.85), dpi=dpi)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, n_rows + 0.9)
    ax.axis("off")
    ax.set_title(title, fontsize=13)
    top = n_rows + 0.55
    ax.text(1, top, "Used in this study", ha="center", fontsize=12, fontweight="bold")
    ax.text(3, top, "Not used (documented)", ha="center", fontsize=12, fontweight="bold")
    for i, (_bid, label) in enumerate(used):
        y = n_rows - 0.15 - i * 0.75
        rect = mpatches.FancyBboxPatch(
            (0.25, y), 1.5, 0.55, boxstyle="round,pad=0.03", facecolor="#D9EAD3", edgecolor="#274E13"
        )
        ax.add_patch(rect)
        ax.text(1.0, y + 0.28, label, ha="center", va="center", fontsize=11)
    for i, (_bid, label) in enumerate(unused):
        y = n_rows - 0.15 - i * 0.75
        rect = mpatches.FancyBboxPatch(
            (2.25, y), 1.5, 0.55, boxstyle="round,pad=0.03", facecolor="#F4CCCC", edgecolor="#990000"
        )
        ax.add_patch(rect)
        ax.text(3.0, y + 0.28, label, ha="center", va="center", fontsize=11)
    ax.text(
        2.0,
        0.18,
        "Two columns, no flow between them. CollectionMethod individuals in ontology/dmi-framework.ttl",
        ha="center",
        fontsize=10,
        style="italic",
    )
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _draw_imputation(out_path: Path, dpi: int = 150) -> Path:
    boxes = [
        ("o", "Official cell"),
        ("g", "Gap ≤ 2 years?\ncarry forward/back"),
        ("m", "Still missing?\nMICE + PMM"),
        ("r", "Rubin pool m=10\n(v2 inference)"),
        ("b", "v1b arm:\nno reconstruction"),
    ]
    fig, ax = plt.subplots(figsize=(12, 3.8), dpi=dpi)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Missing-data workflow (v2 vs v1b)", fontsize=12)
    colors = ["#D9EAD3", "#FFF2CC", "#FCE5CD", "#D0E0E3", "#EAD1DC"]
    for i, ((_bid, label), color) in enumerate(zip(boxes, colors)):
        rect = mpatches.FancyBboxPatch(
            (i + 0.08, 0.32), 0.84, 0.45, boxstyle="round,pad=0.03", facecolor=color, edgecolor="#333"
        )
        ax.add_patch(rect)
        ax.text(i + 0.5, 0.54, label, ha="center", va="center", fontsize=8)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(i + 1.05, 0.54), xytext=(i + 0.94, 0.54),
                        arrowprops=dict(arrowstyle="->", color="#333"))
    ax.text(0.5, 0.12, "Official values are never overwritten. Reconstructed cells are labelled mice_imputed.",
            ha="center", fontsize=8, style="italic")
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _draw_correlation_vif_process(out_path: Path, dpi: int = 300) -> Path:
    """Detailed MK flowchart for Figure 7: sources -> processed panel -> design matrix -> {Pearson, VIF}.

    Mirrors statistics/descriptive.py exactly: correlation is pairwise-complete
    (DataFrame.corr), VIF is listwise (dropna over all predictors) on a design
    matrix with an explicit constant column. Figure inches are chosen so that the
    9.5 pt box text still prints near 7.5 pt at the 6.3 in width used in the DOCX.
    """
    BLUE_F, BLUE_E = "#E8EEF7", "#1F4E79"
    AMB_F, AMB_E = "#FFF2CC", "#BF9000"
    GRN_F, GRN_E = "#D9EAD3", "#274E13"
    ORG_F, ORG_E = "#FCE5CD", "#B45F06"
    ART_F, ART_E = "#FFFFFF", "#666666"

    fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=dpi)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_title(
        "Како се добиваат корелациските коефициенти и VIF",
        fontsize=12.5, fontweight="bold", pad=10,
    )

    def box(x, y, w, h, text, fc, ec, fs=9.5, dashed=False, min_fs=6.0):
        """Draw a box and shrink the label until it fits inside it."""
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.35,rounding_size=0.8",
            facecolor=fc, edgecolor=ec, linewidth=1.3, zorder=2,
            linestyle="--" if dashed else "-",
        ))
        t = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                    fontsize=fs, zorder=3, linespacing=1.4)
        renderer = fig.canvas.get_renderer()
        p0 = ax.transData.transform((x, y))
        p1 = ax.transData.transform((x + w, y + h))
        avail_w, avail_h = (p1[0] - p0[0]) * 0.92, (p1[1] - p0[1]) * 0.92
        cur = fs
        while cur > min_fs:
            bb = t.get_window_extent(renderer=renderer)
            if bb.width <= avail_w and bb.height <= avail_h:
                break
            cur -= 0.25
            t.set_fontsize(cur)
        return (x, y, w, h)

    B = lambda b: (b[0] + b[2] / 2, b[1])
    T = lambda b: (b[0] + b[2] / 2, b[1] + b[3])
    R = lambda b: (b[0] + b[2], b[1] + b[3] / 2)
    L = lambda b: (b[0], b[1] + b[3] / 2)

    def arrow(p, q, color=BLUE_E):
        ax.annotate("", xy=q, xytext=p, zorder=1, arrowprops=dict(
            arrowstyle="-|>", color=color, lw=1.7, shrinkA=1, shrinkB=1,
            mutation_scale=15))

    def polyarrow(pts, color=BLUE_E):
        for i in range(len(pts) - 2):
            ax.plot([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]],
                    color=color, lw=1.7, zorder=1, solid_capstyle="round")
        arrow(pts[-2], pts[-1], color)

    def band(y, text):
        ax.text(1.5, y, text, fontsize=10.5, fontweight="bold", color="#333333", va="center")

    # ---- Band 1: data processing -------------------------------------------
    band(93.0, "1. ОБРАБОТКА НА ПОДАТОЦИТЕ — секоја ќелија добива ознака за потеклото")
    b1_labels = [
        "Официјални извори\nWDI API + снимки",
        "Валидација и\nхармонизација на X7\n(×10 до 2019)",
        "Пренос на најблиска\nофицијална вредност\n(≤ 2 години)",
        "MICE за останатите\nпразнини",
        "Панел со provenance\ndmi_panel.parquet",
    ]
    b1 = []
    for i, lab in enumerate(b1_labels):
        fc, ec = (ART_F, ART_E) if i == len(b1_labels) - 1 else (BLUE_F, BLUE_E)
        b1.append(box(2 + i * 19.7, 79, 17.2, 11, lab, fc, ec, fs=9.5))
    for i in range(len(b1) - 1):
        arrow(R(b1[i]), L(b1[i + 1]))

    # ---- Band 2: design matrix ---------------------------------------------
    band(74.0, "2. КОНСТРУКЦИЈА НА ДИЗАЈН-МАТРИЦАТА")
    b2a = box(16, 61, 30, 10,
              "Задоцнување t−1 на\nX1–X10, C1, C2\n(DMI останува на t)",
              AMB_F, AMB_E)
    b2b = box(54, 61, 34, 10,
              "Дизајн-матрица\n[ DMI | X*_lag1 | C*_lag1 ]\nединствен влез за двете",
              AMB_F, AMB_E)
    polyarrow([B(b1[-1]), (90.8, 75.0), (31, 75.0), (31, 71)])
    arrow(R(b2a), L(b2b), AMB_E)

    # ---- Band 3: the two branches ------------------------------------------
    band(55.5, "3. ДВЕ ПРЕСМЕТКИ НА ИСТАТА МАТРИЦА, НО НА РАЗЛИЧЕН ПОТПРИМЕРОК")
    polyarrow([B(b2b), (71, 51.5), (9, 51.5), (9, 44)], GRN_E)
    polyarrow([B(b2b), (71, 51.5), (91, 51.5), (91, 44)], ORG_E)
    ax.text(24, 47.0, "ГРАНКА А — КОРЕЛАЦИЈА", ha="center", fontsize=10,
            fontweight="bold", color=GRN_E)
    ax.text(75, 47.0, "ГРАНКА Б — МУЛТИКОЛИНЕАРНОСТ (VIF)", ha="center", fontsize=10,
            fontweight="bold", color=ORG_E)

    la = box(3, 35, 42, 9, "Pearson r по парови целосни набљудувања:\n"
             "за секој пар се користат сите земја–години\nво кои двете променливи имаат вредност",
             GRN_F, GRN_E)
    lb = box(3, 23, 42, 7, "correlation_matrix.csv", ART_F, ART_E, dashed=True)
    lc = box(3, 10, 42, 9, "Топлинска карта на задоцнетите\nпредиктори и DMI (Слика 7а)",
             GRN_F, GRN_E)
    arrow(B(la), T(lb), GRN_E)
    arrow(B(lb), T(lc), GRN_E)

    ra = box(54, 35, 43, 9, "Листовно отфрлање: редот со која било\n"
             "празнина се отфрла за сите предиктори\nодеднаш → само целосни случаи",
             ORG_F, ORG_E)
    rb = box(54, 25, 43, 7.5, "Додавање константна колона од единици\n"
             "(без неа VIF излегува вештачки висок)", ORG_F, ORG_E)
    rc = box(54, 13.5, 43, 9, "Помошна регресија: секој предиктор\nврз сите останати → R²\n"
             "VIF = 1 / (1 − R²)", ORG_F, ORG_E)
    rd = box(54, 3.5, 43, 7, "vif_predictors.csv", ART_F, ART_E, dashed=True)
    arrow(B(ra), T(rb), ORG_E)
    arrow(B(rb), T(rc), ORG_E)
    arrow(B(rc), T(rd), ORG_E)

    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def _local_name(term: Any) -> str:
    s = str(term)
    if "#" in s:
        return s.rsplit("#", 1)[-1]
    return s.rsplit("/", 1)[-1]


def _layer_classes_from_ttl(ttl_path: Path) -> Dict[str, List[str]]:
    """Read Ctax/Cdec/Ceval subclasses from the TBox (IRI local names). Empty if missing."""
    from rdflib import OWL, RDF, RDFS, Graph

    layers: Dict[str, List[str]] = {"Ctax": [], "Cdec": [], "Ceval": []}
    if not ttl_path.exists():
        return layers
    g = Graph()
    g.parse(ttl_path, format="turtle")
    for cls in g.subjects(RDF.type, OWL.Class):
        local = _local_name(cls)
        for parent in g.objects(cls, RDFS.subClassOf):
            key = _local_name(parent)
            if key in layers and local not in ("Ctax", "Cdec", "Ceval"):
                layers[key].append(local)
    for key in layers:
        layers[key] = sorted(set(layers[key]))
    return layers


def _tbox_object_properties(ttl_path: Path) -> List[Tuple[str, str, str]]:
    """Object properties as (local name, domain local, range local). Empty if missing."""
    from rdflib import OWL, RDF, RDFS, Graph

    rows: List[Tuple[str, str, str]] = []
    if not ttl_path.exists():
        return rows
    g = Graph()
    g.parse(ttl_path, format="turtle")
    for prop in g.subjects(RDF.type, OWL.ObjectProperty):
        domains = [_local_name(d) for d in g.objects(prop, RDFS.domain)] or [""]
        ranges = [_local_name(r) for r in g.objects(prop, RDFS.range)] or [""]
        name = _local_name(prop)
        for domain in domains:
            for rng in ranges:
                rows.append((name, domain, rng))
    return rows


# Backbone drawn on Слика 4а. Layout coordinates stay in code; membership is from the TBox.
_FIG_CLASSES: Dict[str, List[str]] = {
    "Ctax": ["Indicator", "Country", "ProcessPhase", "DataSource", "CollectionMethod"],
    "Cdec": ["ScoreEntry", "Explanation", "ModelOutput"],
    "Ceval": ["EvaluationRun", "CompetencyQuestion", "ValidationQuery"],
}
_FIG_PROPS: List[Tuple[str, str, str]] = [
    ("entryIndicator", "ScoreEntry", "Indicator"),
    ("entryCountry", "ScoreEntry", "Country"),
    ("entryPhase", "ScoreEntry", "ProcessPhase"),
    ("entrySource", "ScoreEntry", "DataSource"),
    ("obtainedBy", "Indicator", "CollectionMethod"),
    ("explainsOutput", "Explanation", "ModelOutput"),
    ("aboutIndicator", "Explanation", "Indicator"),
    ("evaluates", "EvaluationRun", "CompetencyQuestion"),
    ("usesValidationQuery", "CompetencyQuestion", "ValidationQuery"),
    ("evaluatedWith", "EvaluationRun", "ValidationQuery"),
]


def _draw_ontology_layers(ttl_path: Path, out_path: Path, dpi: int = 150) -> Path:
    """Class boxes plus TBox object-property arrows. Not a HermiT inference diagram.

    Classes and arrows are read from the TBox with rdflib. Box coordinates are in this
    function (not an automatic layout). Recommendation is a TBox class with zero ABox
    individuals and is not drawn.
    """
    layers = _layer_classes_from_ttl(ttl_path)
    props = set(_tbox_object_properties(ttl_path))
    parsed = ttl_path.exists()
    if not parsed:
        logger.warning("TBox missing at %s; drawing declared backbone anyway", ttl_path)
    else:
        missing_cls = [
            f"{layer}.{name}"
            for layer, names in _FIG_CLASSES.items()
            for name in names
            if name not in layers.get(layer, [])
        ]
        missing_prop = [f"{n}:{d}->{r}" for n, d, r in _FIG_PROPS if (n, d, r) not in props]
        if missing_cls or missing_prop:
            raise ValueError(
                "Слика 4а TBox parse incomplete: "
                f"classes={missing_cls or 'ok'} properties={missing_prop or 'ok'}"
            )
    fig, ax = plt.subplots(figsize=(14.8, 9.2), dpi=dpi)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9.2)
    ax.axis("off")
    ax.set_title(
        "Ontology: classes and relations from ontology/dmi-framework.ttl "
        "(rdflib; box layout in code; reasoner none)"
        if parsed
        else "Ontology: declared backbone (TBox file missing; layout in code; reasoner none)",
        fontsize=12,
        pad=8,
    )
    boxes: Dict[str, Tuple[float, float, float, float]] = {}

    def add(key: str, x: float, y: float, w: float, h: float, text: str, fc: str, ec: str) -> None:
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.04",
                facecolor=fc,
                edgecolor=ec,
                linewidth=1.4,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)
        boxes[key] = (x, y, w, h)

    def port(key: str, side: str) -> Tuple[float, float]:
        x, y, w, h = boxes[key]
        if side == "top":
            return (x + w / 2, y + h)
        if side == "bottom":
            return (x + w / 2, y)
        if side == "left":
            return (x, y + h / 2)
        if side == "right":
            return (x + w, y + h / 2)
        if side == "top-left":
            return (x + w * 0.25, y + h)
        if side == "top-right":
            return (x + w * 0.75, y + h)
        if side == "bottom-left":
            return (x + w * 0.25, y)
        if side == "bottom-right":
            return (x + w * 0.75, y)
        raise ValueError(side)

    def arrow(start: Tuple[float, float], end: Tuple[float, float], label: str, label_xy: Tuple[float, float]) -> None:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.25, shrinkA=1, shrinkB=1),
        )
        ax.text(label_xy[0], label_xy[1], label, ha="center", va="center", fontsize=6.8, color="#222222")

    def polyarrow(pts: List[Tuple[float, float]], label: str, label_xy: Tuple[float, float]) -> None:
        ax.plot(
            [p[0] for p in pts[:-1]],
            [p[1] for p in pts[:-1]],
            color="#333333",
            lw=1.25,
            solid_capstyle="round",
        )
        ax.annotate(
            "",
            xy=pts[-1],
            xytext=pts[-2],
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.25, shrinkA=0, shrinkB=1),
        )
        ax.text(label_xy[0], label_xy[1], label, ha="center", va="center", fontsize=6.8, color="#222222")

    ax.text(0.2, 8.85, "Ctax — what is what", fontsize=11, fontweight="bold", color="#1F4E79")
    ax.text(0.2, 4.42, "Cdec — reusable cell", fontsize=11, fontweight="bold", color="#274E13")
    ax.text(0.2, 1.92, "Ceval — did we answer", fontsize=11, fontweight="bold", color="#B45F06")

    ctax, cdec, ceval = "#E8EEF7", "#D9EAD3", "#FFF2CD"
    e_ctax, e_cdec, e_ceval = "#1F4E79", "#274E13", "#B45F06"
    add("Ind", 0.30, 7.45, 2.30, 0.68, "Indicator", ctax, e_ctax)
    add("Ctry", 3.20, 7.45, 2.15, 0.68, "Country", ctax, e_ctax)
    add("Ph", 5.55, 7.45, 2.15, 0.68, "ProcessPhase", ctax, e_ctax)
    add("Src", 7.95, 7.45, 2.20, 0.68, "DataSource", ctax, e_ctax)
    add("Meth", 0.30, 6.15, 2.00, 0.68, "CollectionMethod", ctax, e_ctax)

    add("SE", 3.35, 3.15, 2.5, 0.78, "ScoreEntry", cdec, e_cdec)
    add("Expl", 6.35, 3.05, 2.3, 0.95, "Explanation\naboutIndicator → Indicator", cdec, e_cdec)
    add("MO", 9.15, 3.15, 2.4, 0.78, "ModelOutput", cdec, e_cdec)

    add("Eval", 0.7, 0.48, 2.5, 0.75, "EvaluationRun", ceval, e_ceval)
    add("CQ", 4.55, 0.48, 2.7, 0.75, "CompetencyQuestion", ceval, e_ceval)
    add("VQ", 8.5, 0.48, 2.7, 0.75, "ValidationQuery", ceval, e_ceval)

    # Orthogonal bus from ScoreEntry: stem up, then verticals that miss CollectionMethod.
    bus_y = 5.55
    se_top = port("SE", "top")
    ind_in = (boxes["Ind"][0] + boxes["Ind"][2], boxes["Ind"][1])
    ctry_in = port("Ctry", "bottom")
    ph_in = port("Ph", "bottom")
    src_in = port("Src", "bottom")
    ax.plot([se_top[0], se_top[0]], [se_top[1], bus_y], color="#333333", lw=1.25)
    ax.plot([ind_in[0], src_in[0]], [bus_y, bus_y], color="#333333", lw=1.25)
    arrow((ind_in[0], bus_y), ind_in, "entryIndicator", (ind_in[0], bus_y - 0.22))
    arrow((ctry_in[0], bus_y), ctry_in, "entryCountry", (ctry_in[0], bus_y - 0.22))
    arrow((ph_in[0], bus_y), ph_in, "entryPhase", (ph_in[0], bus_y - 0.22))
    arrow((src_in[0], bus_y), src_in, "entrySource", (src_in[0], bus_y - 0.22))
    arrow(port("Ind", "bottom"), port("Meth", "top"), "obtainedBy", (1.42, 6.92))
    arrow(port("Expl", "right"), port("MO", "left"), "explainsOutput", (8.55, 3.68))
    arrow(port("Eval", "right"), port("CQ", "left"), "evaluates", (3.95, 0.95))
    arrow(port("CQ", "right"), port("VQ", "left"), "usesValidationQuery", (7.85, 0.95))
    eval_top = port("Eval", "top")
    vq_top = port("VQ", "top")
    rail = 1.58
    polyarrow(
        [eval_top, (eval_top[0], rail), (vq_top[0], rail), vq_top],
        "evaluatedWith",
        (5.85, 1.72),
    )

    ax.text(
        6.0,
        0.12,
        "Classes and arrows parsed from the TBox (rdflib). Box positions are in code. "
        "Recommendation is unused (zero individuals). Not HermiT.",
        ha="center",
        fontsize=9,
        style="italic",
    )
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _mermaid(name: str, body: str, md_dir: Path) -> Path:
    path = md_dir / f"{name}.md"
    path.write_text(f"# {name}\n\n```mermaid\n{body.strip()}\n```\n", encoding="utf-8")
    return path


def run(project_root: Path | str, config: Any = None) -> Dict[str, Any]:
    root = Path(project_root).resolve()
    fig_dir = ensure_dir(root / "outputs" / "figures" / "processes")
    md_dir = ensure_dir(root / "docs" / "processes")
    produced: List[str] = []

    p1 = _draw_chain(
        [
            ("p1", "P1 Scope"),
            ("p2", "P2 Competency Qs"),
            ("p3", "P3 Knowledge sources"),
            ("p4", "P4 Conceptual model"),
            ("p5", "P5 OWL TBox"),
            ("p6", "P6 Decision layer"),
            ("p7", "P7 ABox population"),
            ("p8", "P8 CQ validation"),
        ],
        fig_dir / "construction_p1_p8.png",
        "Ontology construction (Savoska & Loshkovska 2026 analogue)",
        note="Design-science process. Artifact: ontology/dmi-framework.ttl + outputs/ontology/",
    )
    produced.append(str(p1.relative_to(root)))

    assess_boxes = [(s, io[1]) for s, io in zip([x.value for x in STAGE_ORDER], STAGE_IO)]
    p2 = _draw_chain(
        assess_boxes,
        fig_dir / "assessment_pipeline.png",
        "DigitalMaturityAssessmentProcess — 12 PipelineStage boxes (code); TBox ProcessPhase = 8",
        note="I/O contracts: " + " → ".join(io[2] for io in STAGE_IO[:4]) + " …",
    )
    produced.append(str(p2.relative_to(root)))

    p3 = _draw_split(
        "CollectionMethod individuals (used vs unused)",
        [
            ("api", "Official API\n(HiTEc traditional)"),
            ("snap", "Curated snapshot\n(HiTEc traditional)"),
            ("cons", "Constructed composite\n(DMI, 168 cells)"),
        ],
        [
            ("surv", "Perceptions / survey\n(HiTEc unused here)"),
            ("sens", "Sensor / functional\n(HiTEc unused here)"),
        ],
        fig_dir / "acquisition_methods.png",
    )
    produced.append(str(p3.relative_to(root)))

    p4 = _draw_imputation(fig_dir / "missing_data_workflow.png")
    produced.append(str(p4.relative_to(root)))

    p5 = _draw_chain(
        [
            ("c", "Collect\nAPI + snapshots"),
            ("pr", "Prepare\nvalidate, carry, MICE"),
            ("a", "Analyse\nFE/RE, VIF, corr"),
            ("pd", "Predict\nRF, XGB, LGBM,\nCatBoost, SVR, MLP"),
            ("x", "Explain\nSHAP, LIME, PDP"),
            ("v", "VDA\nStreamlit + Power BI"),
        ],
        fig_dir / "model_to_phase.png",
        "Methods mapped to process phases (ML is not collection)",
        note="OLS/Ridge = linear baselines in Predict. Ensembles do not collect data.",
    )
    produced.append(str(p5.relative_to(root)))

    p6 = _draw_correlation_vif_process(fig_dir / "correlation_vif_process.png")
    produced.append(str(p6.relative_to(root)))

    p7 = _draw_ontology_layers(
        root / "ontology" / "dmi-framework.ttl",
        fig_dir / "ontology_layers.png",
    )
    produced.append(str(p7.relative_to(root)))

    mermaid_specs = {
        "construction_p1_p8": """
flowchart LR
  P1[P1 Scope] --> P2[P2 Competency questions]
  P2 --> P3[P3 Knowledge sources]
  P3 --> P4[P4 Conceptual model]
  P4 --> P5[P5 OWL TBox]
  P5 --> P6[P6 Decision layer]
  P6 --> P7[P7 ABox population]
  P7 --> P8[P8 CQ validation]
""",
        "assessment_pipeline": """
flowchart LR
  A[Acquire\ndata/raw] --> V[Validate\naudit] --> P[Process\npanel_wide] --> I[Index\ndmi_panel]
  I --> S[Descriptives] --> E[Econometrics] --> M[Predict]
  M --> X[Explain] --> C[Converge] --> Viz[Visualize]
  Viz --> R[Report] --> Aud[Audit]
""",
        "acquisition_methods": """
flowchart TB
  subgraph used [Used]
    API[Official statistical API]
    SNAP[Curated published snapshot]
    CONS[Constructed composite DMI]
  end
  subgraph unused [Not used — documented only, no flow from used]
    SURV[Primary field survey]
    SENS[Sensor / automated capture]
  end
""",
        "ontology_layers": """
flowchart TB
  subgraph Ctax [Ctax]
    Indicator
    Country
    ProcessPhase
    DataSource
    CollectionMethod
  end
  subgraph Cdec [Cdec]
    ScoreEntry
    Explanation
    ModelOutput
  end
  subgraph Ceval [Ceval]
    EvaluationRun
    CompetencyQuestion
    ValidationQuery
  end
  ScoreEntry -->|entryIndicator| Indicator
  ScoreEntry -->|entryCountry| Country
  ScoreEntry -->|entryPhase| ProcessPhase
  ScoreEntry -->|entrySource| DataSource
  Indicator -->|obtainedBy| CollectionMethod
  Explanation -->|aboutIndicator| Indicator
  Explanation -->|explainsOutput| ModelOutput
  EvaluationRun -->|evaluates| CompetencyQuestion
  CompetencyQuestion -->|usesValidationQuery| ValidationQuery
  EvaluationRun -->|evaluatedWith| ValidationQuery
""",
        "missing_data_workflow": """
flowchart LR
  O[Official cell] --> G{Gap <= 2 years?}
  G -->|yes| C[Carry forward/back]
  G -->|no| M[MICE + PMM]
  C --> R[v2 Rubin pool m=10]
  M --> R
  O --> B[v1b: no reconstruction]
""",
        "model_to_phase": """
flowchart LR
  Collect[Collect: API + snapshots] --> Prepare[Prepare]
  Prepare --> Analyse[Analyse: FE/RE VIF]
  Analyse --> Predict[Predict: CatBoost RF XGB LGBM SVR MLP]
  Predict --> Explain[Explain: SHAP LIME PDP]
  Explain --> VDA[VDA: Streamlit Power BI]
""",
        "correlation_vif_process": """
flowchart LR
  Panel[Lagged analysis panel] --> Corr[Pearson correlation]
  Corr --> VIF[VIF with intercept]
  VIF --> Fig[Heatmap with caption]
""",
    }
    for name, body in mermaid_specs.items():
        produced.append(str(_mermaid(name, body, md_dir).relative_to(root)))

    captions = {
        "construction_p1_p8.png": (
            "Eight-phase ontology construction. P1–P8 follow the design-science pipeline "
            "in Savoska & Loshkovska (2026, Fig. 1), applied here to digital-maturity indicators."
        ),
        "assessment_pipeline.png": (
            "Twelve PipelineStage boxes from src/pipeline/pipeline_state.py, each with an I/O "
            "artifact. The TBox class ProcessPhase has eight occurrent phases (Acquire…Visualize); "
            "the extra four stages are code (descriptives, econometrics, converge, audit), not OWL individuals."
        ),
        "acquisition_methods.png": (
            "Five CollectionMethod individuals. Green = used: official API and curated snapshot "
            "(HiTEc traditional official series) plus constructed composite (DMI, 168 cells). "
            "Red = unused (primary survey, sensors). COST Action CA21163 names data kinds in the "
            "MoU; it does not publish a numbered list of four obtaining methods, and this panel "
            "does not invent one."
        ),
        "missing_data_workflow.png": (
            "Missing-data workflow. Official cells are preserved. Gaps of at most two years "
            "are carried; remaining holes are MICE-imputed on v2 and left empty on v1b."
        ),
        "model_to_phase.png": (
            "Every method is assigned to one process phase. Ensemble learners belong to "
            "Predict, not Collect."
        ),
        "correlation_vif_process.png": (
            "Correlation matrix is Pearson r on the lagged design. VIF is computed after "
            "adding an intercept column. Read VIF>10 as collinearity, not as a DMI score."
        ),
        "ontology_layers.png": (
            "Ctax / Cdec / Ceval classes and TBox arrows parsed from ontology/dmi-framework.ttl "
            "(rdflib). Box positions are in process_diagrams.py, not an automatic layout. "
            "aboutIndicator is labelled on Explanation. Not a HermiT run."
        ),
    }
    write_json(fig_dir / "captions.json", captions)
    write_json(
        fig_dir / "manifest.json",
        {"artifacts": produced, "n_stages": len(STAGE_ORDER)},
    )
    logger.info("Wrote %d process-diagram artifacts", len(produced))
    return {"artifacts": produced}
