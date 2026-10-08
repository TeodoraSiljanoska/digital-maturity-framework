"""Association of the context indicators Z1–Z6 with the DMI and the effect of
adding them to the index.

Z values enter only where published (no carried or reconstructed cells), one
year before the DMI. Two specifications are reported per indicator:
two-way fixed effects (country and year) on the standardised lagged value with
Driscoll–Kraay standard errors, and the pooled correlation of first
differences. The extended index DMI+ adds each Z to the pillar it relates to,
with the same min–max scaling and equal pillar weights as the DMI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from scipy import stats

from common.io import ensure_dir, write_df, write_json
from knowledge_layer.context import cell_panel, load_config, load_vintage

PILLARS = {
    "infrastructure": ["X1", "X2"],
    "e_government": ["X3"],
    "skills": ["X4"],
    "trust_innovation": ["X5", "X6", "X7"],
    "digital_economy": ["X8", "X9", "X10"],
}


def fe_association(dmi: pd.DataFrame, zoff: pd.DataFrame, zids) -> pd.DataFrame:
    rows = []
    for z in zids:
        lag = zoff[["country_iso3", "year", z]].copy()
        lag["year"] = lag["year"] + 1
        df = dmi.merge(lag, on=["country_iso3", "year"], how="inner").dropna(subset=[z, "DMI"])
        sd = df[z].std(ddof=1)
        df["z_std"] = (df[z] - df[z].mean()) / sd
        panel = df.set_index(["country_iso3", "year"])
        mod = PanelOLS(panel["DMI"], panel[["z_std"]], entity_effects=True, time_effects=True)
        dk = mod.fit(cov_type="kernel")
        cl = mod.fit(cov_type="clustered", cluster_entity=True)
        # first differences, consecutive published years only
        s = zoff[["country_iso3", "year", z]].sort_values(["country_iso3", "year"])
        s["dZ"] = s.groupby("country_iso3")[z].diff()
        s.loc[s.groupby("country_iso3")["year"].diff() != 1, "dZ"] = np.nan
        s["year"] = s["year"] + 1
        d = dmi.sort_values(["country_iso3", "year"]).copy()
        d["dDMI"] = d.groupby("country_iso3")["DMI"].diff()
        fd = d.merge(s[["country_iso3", "year", "dZ"]], on=["country_iso3", "year"]).dropna(subset=["dZ", "dDMI"])
        r, p = stats.pearsonr(fd["dZ"], fd["dDMI"]) if len(fd) > 3 else (np.nan, np.nan)
        rows.append(
            {
                "indicator_id": z,
                "n_obs": int(len(df)),
                "n_countries": int(df["country_iso3"].nunique()),
                "beta_per_sd": float(dk.params["z_std"]),
                "se_driscoll_kraay": float(dk.std_errors["z_std"]),
                "p_driscoll_kraay": float(dk.pvalues["z_std"]),
                "se_cluster_country": float(cl.std_errors["z_std"]),
                "p_cluster_country": float(cl.pvalues["z_std"]),
                "within_r2": float(dk.rsquared_within),
                "fd_n": int(len(fd)),
                "fd_pearson_r": float(r),
                "fd_p": float(p),
            }
        )
    return pd.DataFrame(rows)


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) * 100.0


def extended_index(analysis: pd.DataFrame, zfilled: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    df = analysis[["country_iso3", "year", "group_id", "DMI"] + [f"{x}_norm" for p in PILLARS.values() for x in p]].merge(
        zfilled, on=["country_iso3", "year"], how="left"
    )
    pillar_members = {p: [f"{x}_norm" for x in xs] for p, xs in PILLARS.items()}
    for z, spec in cfg["indicators"].items():
        df[f"{z}_norm"] = _minmax(df[z])
        pillar_members[spec["related_pillar"]].append(f"{z}_norm")
    pill = pd.DataFrame({p: df[cols].mean(axis=1, skipna=True) for p, cols in pillar_members.items()})
    df["DMI_plus"] = pill.mean(axis=1, skipna=True)
    for p in pill.columns:
        df[f"pillar_plus_{p}"] = pill[p]
    return df


def compare_indices(ext: pd.DataFrame) -> Dict[str, Any]:
    r = stats.pearsonr(ext["DMI"], ext["DMI_plus"])[0]
    per_year = []
    for y, g in ext.groupby("year"):
        rk = g["DMI"].rank(ascending=False)
        rk2 = g["DMI_plus"].rank(ascending=False)
        rho = stats.spearmanr(g["DMI"], g["DMI_plus"])[0]
        per_year.append(
            {
                "year": int(y),
                "spearman_rho": float(rho),
                "n_rank_changes": int((rk != rk2).sum()),
                "max_abs_rank_shift": int((rk - rk2).abs().max()),
            }
        )
    py = pd.DataFrame(per_year)
    grp = ext.groupby("group_id")[["DMI", "DMI_plus"]].mean()
    return {
        "pearson_r": float(r),
        "mean_abs_diff": float((ext["DMI"] - ext["DMI_plus"]).abs().mean()),
        "max_abs_diff": float((ext["DMI"] - ext["DMI_plus"]).abs().max()),
        "min_spearman_by_year": float(py["spearman_rho"].min()),
        "years_with_rank_change": int((py["n_rank_changes"] > 0).sum()),
        "max_abs_rank_shift": int(py["max_abs_rank_shift"].max()),
        "group_means": grp.round(3).to_dict(orient="index"),
        "per_year": per_year,
    }


def run(project_root: Path | str) -> Dict[str, Any]:
    root = Path(project_root).resolve()
    out = ensure_dir(root / "outputs" / "knowledge_layer" / "association")
    cfg = load_config(root)
    long = load_vintage(root, cfg)
    zfilled, zprov = cell_panel(long, cfg)
    zids = list(cfg["indicators"].keys())
    zoff = zfilled.copy()
    for z in zids:
        zoff[z] = zfilled[z].where(zprov[z] == "official")
    analysis = pd.read_parquet(root / "data/processed/analysis_panel.parquet")
    dmi = analysis[["country_iso3", "year", "DMI"]]
    fe = fe_association(dmi, zoff, zids)
    write_df(fe, out / "context_association.csv")
    ext = extended_index(analysis, zfilled, cfg)
    write_df(ext[["country_iso3", "year", "group_id", "DMI", "DMI_plus"] + [c for c in ext.columns if c.startswith("pillar_plus_")]], out / "dmi_plus.csv")
    cmp = compare_indices(ext)
    write_json(out / "dmi_plus_comparison.json", cmp)
    return {"association": fe.to_dict(orient="records"), "dmi_plus": {k: v for k, v in cmp.items() if k != "per_year"}}
