"""
Independent reproduction of the paper's core results (Groh et al., Nature Medicine 2024)

What it does:
  Reads a table where "each row = one physician's diagnosis of one image", filters the sample as the paper does, and outputs
    Table 1  overall accuracy by physician type (top-1 / top-3)
    Table 2  accuracy gap between dark and light skin (OLS + participant-clustered robust SEs)
    Table 3  skin-tone gap per disease (BCD, PCP; two-sided t-tests)
    Table 4  accuracy before/after AI (DLS) assistance (including the Wizard-of-Oz enhanced AI)
    Table 5  regression (after Extended Data Table 3: specialty × skin tone × AI interactions)
    Table 6  effect of button order (interface) on physicians' choices
    Table 7  skin-tone differences in biopsy / referral recommendations
    Table 8  moderators for primary care physicians (patient mix, adequacy of training)
  Every table puts the value reported in the paper alongside, to show "reproduced vs paper" in a presentation.
  Figures are saved in the --out folder (PNG).

Three data sources (all use this one script):
  1) the authors' public data: Experimental Data.csv(.zip) from ResearchBox #1802
     → first run inspect_data.py to create the column mapping mapping.json, then:
       python analysis/key_results.py --data "Experimental Data.csv.zip" --map mapping.json
  2) simulated data (for rehearsal): python analysis/key_results.py --data data/synthetic_experiment.csv
  3) data collected by the classroom demo: python analysis/key_results.py --data demo_app/data/responses.csv --min-trials 1

Important: the authors' own Analysis.ipynb is the "official reproduction"; this script is an independent re-implementation,
with regression specifications inferred from the paper's text, so some coefficients may be defined slightly differently.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.diseases import MAIN8, in_differential, matches  # noqa: E402
from stats_utils import diff_test, fmt_p, mean_ci, ols_cluster, ttest  # noqa: E402

TYPES = ["BCD", "Resident", "PCP", "Other", "Student"]   # Student = non-clinician participants in the classroom demo

# ---------------------------------------------------------------- Values reported in the paper (for comparison)
PAPER = {
    "top3_all": {"BCD": 0.38, "Resident": 0.36, "PCP": 0.19, "Other": 0.18},
    "top3_main": {"BCD": 0.37, "Resident": 0.35, "PCP": 0.17, "Other": 0.16},
    "top1_all": {"BCD": 0.27, "Resident": 0.24, "PCP": 0.14, "Other": 0.13},
    "top1_main": {"BCD": 0.27, "Resident": 0.24, "PCP": 0.13, "Other": 0.12},
    "gap_top1": {"All": (-0.04, "P<0.001"), "BCD": (-0.05, "P=0.011"), "Resident": (-0.05, "P=0.114"),
                 "PCP": (-0.03, "P=0.006"), "Other": (-0.05, "P=0.012")},
    "gap_top3": {"All": (-0.04, "P=0.001"), "BCD": (-0.03, "P=0.117"), "Resident": (-0.05, "P=0.113"),
                 "PCP": (-0.04, "P=0.008"), "Other": (-0.04, "P=0.092")},
    "dls_control": {"BCD": {"top1": (0.27, 0.36), "top34": (0.37, 0.60)},
                    "PCP": {"top1": (0.13, 0.22), "top34": (0.17, 0.47)}},
    "regression_top1": {"BCD": 0.13, "Dark": -0.03, "DLS": 0.08, "DLS:TreatDLS": 0.08,
                        "DLS:Dark": -0.05, "DLS:BCD:Dark": 0.04},
    "regression_top4": {"DLS:Dark": -0.05, "DLS:BCD:Dark": 0.08},
    "interface": {"include": +0.09, "top": -0.09},
    "biopsy": {"BCD": 0.28, "PCP": 0.07}, "refer_pcp": 0.28, "refer_pcp_dark_gap": 0.044,
}

CANONICAL = ["participant_id", "physician_type", "trial", "image_id", "disease", "fst", "attention_check",
             "dx1", "dx2", "dx3", "top1_correct", "top3_correct", "dls_condition", "interface",
             "dls_suggestion", "dls_correct", "choice", "top1_after", "top4_after", "biopsy", "refer_derm",
             "confidence", "patient_mix", "sufficient_training", "country", "years"]


# ---------------------------------------------------------------- Loading and normalisation
def load(path: str, mapping: dict | None) -> pd.DataFrame:
    from inspect_data import read_any                 # handles .csv and .csv.zip (skips __MACOSX automatically)
    df = read_any(path)
    if mapping:
        cols = {v: k for k, v in mapping.get("columns", {}).items() if v}
        df = df.rename(columns=cols)
        for col, vmap in mapping.get("values", {}).items():
            if col in df:
                df[col] = df[col].map(lambda x, m=vmap: next((k for k, vs in m.items() if str(x) in map(str, vs)), x))
    return normalise(df)


def _norm_type(x) -> str:
    s = str(x).lower()
    if "student" in s or "non-clinician" in s or "not a clinician" in s:
        return "Student"
    if s in ("bcd",) or ("board" in s and "derm" in s) or s == "dermatologist":
        return "BCD"
    if "resident" in s:
        return "Resident"
    if s in ("pcp",) or any(w in s for w in ("primary", "family", "general practi", "gp", "internal medicine")):
        return "PCP"
    return "Other"


def _norm_cond(x) -> str:
    s = str(x).lower()
    if s in ("treatment", "enhanced", "treat", "1", "true", "84", "0.84", "woz", "wizard"):
        return "treatment"
    return "control"


def _to01(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    m = {"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}
    return s.map(lambda v: m.get(str(v).strip().lower(), np.nan) if not isinstance(v, (int, float)) or pd.isna(v)
                 else int(v)).astype(float)


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "physician_type" in df:
        df["physician_type"] = df["physician_type"].map(_norm_type)
    for c in ("dls_condition", "interface"):
        if c in df:
            df[c] = df[c].map(_norm_cond)
    for c in ("attention_check", "top1_correct", "top3_correct", "dls_correct", "top1_after", "top4_after",
              "biopsy", "refer_derm"):
        if c in df:
            df[c] = _to01(df[c])
    if "fst" in df:
        df["fst"] = pd.to_numeric(df["fst"], errors="coerce")
        df["dark"] = (df["fst"] >= 5).astype(int)
    # Without ready-made grade columns, grade by text (the authors reviewed every answer by hand; this is an approximation)
    if "top1_correct" not in df and {"dx1", "disease"} <= set(df):
        df["top1_correct"] = [int(matches(a, d)) for a, d in zip(df.dx1.fillna(""), df.disease)]
    if "top3_correct" not in df and {"dx1", "dx2", "dx3", "disease"} <= set(df):
        df["top3_correct"] = [int(in_differential([a, b, c], d)) for a, b, c, d in
                              zip(df.dx1.fillna(""), df.dx2.fillna(""), df.dx3.fillna(""), df.disease)]
    if "dls_correct" not in df and {"dls_suggestion", "disease"} <= set(df):
        df["dls_correct"] = [int(matches(s, d)) for s, d in zip(df.dls_suggestion.fillna(""), df.disease)]
    if "main8" not in df and "disease" in df:
        lut = {u: next((k for k in MAIN8 if matches(str(u), k)), "Other") for u in df["disease"].dropna().unique()}
        df["disease8"] = df["disease"].map(lut).fillna("Other")
        df["main8"] = (df["disease8"] != "Other").astype(int)
    if "choice" in df:
        df["choice"] = df["choice"].astype(str).str.lower().map(
            lambda s: "top" if "top" in s else "include" if "includ" in s or "different" in s
            else "keep" if "keep" in s or "ignore" in s else "same" if s in ("same", "match", "matched", "nan", "")
            else s)
    return df


def select_sample(df: pd.DataFrame, min_trials: int = 10, first_n: int = 10, attention_filter: bool = True) -> pd.DataFrame:
    """The paper's sample rule: participants who passed the attention check and completed at least 10 images; use only their first 10 diagnoses and drop the attention-check image."""
    d = df.sort_values(["participant_id", "trial"]) if "trial" in df else df
    n_by = d.groupby("participant_id").size()
    keep = set(n_by[n_by >= min_trials].index)
    if attention_filter and "attention_check" in d and d["attention_check"].sum() > 0:
        att = d[d.attention_check == 1].groupby("participant_id")["top3_correct"].max()
        keep &= set(att[att == 1].index) | (keep - set(att.index))
    d = d[d.participant_id.isin(keep)]
    if "trial" in d:
        d = d.groupby("participant_id", group_keys=False).head(first_n)
    if "attention_check" in d:
        d = d[d.attention_check != 1]
    return d.reset_index(drop=True)


# ---------------------------------------------------------------- Analysis
def pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


def pp(x) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:+.1f} pp"


def t1_overall(d):
    rows = []
    for t in TYPES:
        s = d[d.physician_type == t]
        if s.empty:
            continue
        m = s[s.main8 == 1]
        rows.append({"Physician type": t, "Participants": s.participant_id.nunique(), "Diagnoses": len(s),
                     "top-1 (all)": pct(s.top1_correct.mean()), "Paper": pct(PAPER["top1_all"].get(t)),
                     "top-3 (all)": pct(s.top3_correct.mean()), "Paper ": pct(PAPER["top3_all"].get(t)),
                     "top-1 (8 main)": pct(m.top1_correct.mean()), "Paper  ": pct(PAPER["top1_main"].get(t)),
                     "top-3 (8 main)": pct(m.top3_correct.mean()), "Paper   ": pct(PAPER["top3_main"].get(t))})
    return pd.DataFrame(rows)


def t2_skin(d):
    rows = []
    for t in ["All"] + TYPES:
        s = d if t == "All" else d[d.physician_type == t]
        if s.participant_id.nunique() < 2:
            continue
        r = {"Physician type": t}
        for y, key in (("top1_correct", "gap_top1"), ("top3_correct", "gap_top3")):
            res = diff_test(s, y, "dark", "participant_id")
            lab = "top-1" if y == "top1_correct" else "top-3"
            r[f"{lab} light"] = pct(res["mean0"])
            r[f"{lab} dark"] = pct(res["mean1"])
            r[f"{lab} gap"] = f"{pp(res['diff'])} ({fmt_p(res['p'])})"
            g, p = PAPER[key].get(t, (None, ""))
            r[f"{lab} paper"] = f"{100 * g:+.0f} pp ({p})" if g is not None else "—"
        rows.append(r)
    return pd.DataFrame(rows)


def t3_disease(d, ptype):
    s = d[(d.physician_type == ptype) & (d.main8 == 1)]
    rows = []
    for dis in list(MAIN8):
        x = s[s.disease8 == dis]
        if x.empty:
            continue
        r = {"Disease": dis, "n": len(x)}
        for y in ("top1_correct", "top3_correct"):
            lab = "top-1" if y == "top1_correct" else "top-3"
            a, b = x.loc[x.dark == 0, y], x.loc[x.dark == 1, y]
            r[f"{lab} light/dark"] = f"{pct(a.mean())} / {pct(b.mean())}"
            r[f"{lab} P"] = fmt_p(ttest(a, b))
        rows.append(r)
    return pd.DataFrame(rows)


def t4_dls(d):
    s = d[(d.main8 == 1) & d.physician_type.isin(["BCD", "PCP"])]
    rows = []
    for t in ("BCD", "PCP"):
        for cond in ("control", "treatment"):
            x = s[(s.physician_type == t) & (s.dls_condition == cond)]
            if x.empty:
                continue
            long1 = pd.concat([x.assign(y=x.top1_correct, after=0), x.assign(y=x.top1_after, after=1)])
            long4 = pd.concat([x.assign(y=x.top3_correct, after=0), x.assign(y=x.top4_after, after=1)])
            p1 = diff_test(long1, "y", "after", "participant_id")["p"]
            p4 = diff_test(long4, "y", "after", "participant_id")["p"]
            ref = PAPER["dls_control"][t] if cond == "control" else None
            rows.append({
                "Physician": t, "AI version": "control AI (≈47%)" if cond == "control" else "enhanced AI (≈84%)",
                "top-1 before→after": f"{pct(x.top1_correct.mean())} → {pct(x.top1_after.mean())} ({fmt_p(p1)})",
                "relative gain": f"{100 * (x.top1_after.mean() / x.top1_correct.mean() - 1):+.0f}%" if x.top1_correct.mean() > 0 else "—",
                "top-3 → top-4": f"{pct(x.top3_correct.mean())} → {pct(x.top4_after.mean())} ({fmt_p(p4)})",
                "Paper (control AI)": (f"top-1 {pct(ref['top1'][0])}→{pct(ref['top1'][1])}; "
                                  f"top-3→4 {pct(ref['top34'][0])}→{pct(ref['top34'][1])}") if ref else "Paper: about +8 pp more",
            })
    return pd.DataFrame(rows)


def t4b_errors(d):
    """Effect of wrong AI suggestions (paper: top-1 drops only 1.2 pp; a correct top diagnosis is overridden <2% of the time; when all three of their own are wrong, 10%/14% make it their top diagnosis)"""
    s = d[(d.main8 == 1) & d.physician_type.isin(["BCD", "PCP"])]
    rows = []
    for t in ("BCD", "PCP"):
        x = s[(s.physician_type == t) & (s.dls_correct == 0)]
        if x.empty:
            continue
        a = x[x.top1_correct == 1]
        b = x[x.top3_correct == 0]
        rows.append({"Physician": t, "Diagnoses with a wrong AI suggestion": len(x),
                     "top-1 change": pp(x.top1_after.mean() - x.top1_correct.mean()),
                     "Correct top diagnosis replaced by the wrong suggestion": pct((a.choice == "top").mean()) if len(a) else "—",
                     "All three own answers wrong → wrong suggestion made top diagnosis": pct((b.choice == "top").mean()) if len(b) else "—",
                     "Paper": "−1.2 pp; <2%; " + ("10%" if t == "BCD" else "14%")})
    return pd.DataFrame(rows)


def t5_regression(d):
    s = d[(d.main8 == 1) & d.physician_type.isin(["BCD", "PCP"])].copy()
    if s.physician_type.nunique() < 2:
        return None, None
    s["BCD"] = (s.physician_type == "BCD").astype(int)
    s["TreatDLS"] = (s.dls_condition == "treatment").astype(int)
    terms = ["BCD", "Dark", "BCD:Dark", "DLS", "DLS:TreatDLS", "DLS:Dark", "DLS:BCD", "DLS:BCD:Dark"]
    out = {}
    for name, before, after in (("top-1", "top1_correct", "top1_after"), ("top-4", "top3_correct", "top4_after")):
        long = pd.concat([s.assign(y=s[before], DLS=0), s.assign(y=s[after], DLS=1)])
        long["Dark"] = long["dark"]
        out[name] = ols_cluster(long, "y", terms, "participant_id")
    tab = pd.DataFrame({
        "Variable": ["Intercept"] + terms,
        "Meaning": ["Baseline: PCP, light skin, no AI", "How much higher BCDs are than PCPs", "Dark-skin gap (PCPs)", "Difference between the BCD and PCP skin-tone gaps",
                 "Gain from the control AI (PCP, light skin)", "Extra gain from the enhanced AI", "Change in the PCP skin-tone gap with AI", "Extra AI gain for BCDs",
                 "Extra change in the BCD skin-tone gap with AI (three-way interaction)"],
    })
    for name, res in out.items():
        tab[f"{name} coef"] = [pp(res.loc[v, "coef"]) if v != "Intercept" else pct(res.loc[v, "coef"]) for v in tab["Variable"]]
        tab[f"{name} P"] = [fmt_p(res.loc[v, "p"]) for v in tab["Variable"]]
    ref1, ref4 = PAPER["regression_top1"], PAPER["regression_top4"]
    tab["Paper (top-1 / top-4)"] = [
        " / ".join(x for x in (f"{100 * ref1[v]:+.0f}" if v in ref1 else "", f"{100 * ref4[v]:+.0f}" if v in ref4 else "") if x)
        for v in tab["Variable"]]
    return tab, out


def t6_interface(d):
    s = d[d.choice.isin(["top", "include", "keep"])].copy()
    if s.empty or s.interface.nunique() < 2:
        return None
    s["TreatUI"] = (s.interface == "treatment").astype(int)
    rows = []
    for c in ("top", "include", "keep"):
        s["y"] = (s.choice == c).astype(int)
        r = diff_test(s, "y", "TreatUI", "participant_id")
        rows.append({"Option": {"top": "Update my top prediction", "include": "Update my differential to include",
                              "keep": "Keep my differential"}[c],
                     "Control interface ('update top' first)": pct(r["mean0"]), "Treatment interface ('keep' first)": pct(r["mean1"]),
                     "Difference": f"{pp(r['diff'])} ({fmt_p(r['p'])})",
                     "Paper": {"top": "−9 pp", "include": "+9 pp", "keep": "no significant difference"}[c]})
    for t, ref in (("BCD", "−12 pp"), ("PCP", "−7 pp")):
        x = s[(s.physician_type == t) & (s.main8 == 1)]
        if x.interface.nunique() == 2:
            r = diff_test(x, "top1_after", "TreatUI", "participant_id")
            rows.append({"Option": f"{t} top-1 after seeing the suggestion", "Control interface ('update top' first)": pct(r["mean0"]),
                         "Treatment interface ('keep' first)": pct(r["mean1"]), "Difference": f"{pp(r['diff'])} ({fmt_p(r['p'])})",
                         "Paper": ref})
    return pd.DataFrame(rows)


def t7_referral(d):
    rows = []
    for t in ("BCD", "PCP"):
        x = d[d.physician_type == t]
        if x.empty:
            continue
        r = diff_test(x, "biopsy", "dark", "participant_id")
        rows.append({"Measure": f"{t} recommends biopsy", "Overall": pct(x.biopsy.mean()), "Light": pct(r["mean0"]),
                     "Dark": pct(r["mean1"]), "Difference": f"{pp(r['diff'])} ({fmt_p(r['p'])})",
                     "Paper": f"overall {pct(PAPER['biopsy'][t])}"})
    x = d[d.physician_type == "PCP"].dropna(subset=["refer_derm"])
    if not x.empty:
        r = diff_test(x, "refer_derm", "dark", "participant_id")
        rows.append({"Measure": "PCP referral to dermatology", "Overall": pct(x.refer_derm.mean()), "Light": pct(r["mean0"]),
                     "Dark": pct(r["mean1"]), "Difference": f"{pp(r['diff'])} ({fmt_p(r['p'])})",
                     "Paper": "overall 28%; dark +4.4 pp (P=0.012)"})
    x = d[(d.physician_type == "BCD") & (d.main8 == 1)]
    for dis, ref in (("CTCL", "less on dark (P=0.001)"), ("Atopic dermatitis", "more on dark (P=0.008)"),
                     ("Pityriasis rosea", "more on dark (P=0.015)"), ("Pityriasis rubra pilaris", "less on dark (P=0.033)")):
        y = x[x.disease8 == dis]
        if y.empty:
            continue
        a, b = y.loc[y.dark == 0, "biopsy"], y.loc[y.dark == 1, "biopsy"]
        rows.append({"Measure": f"BCD biopsy: {dis}", "Overall": pct(y.biopsy.mean()), "Light": pct(a.mean()),
                     "Dark": pct(b.mean()), "Difference": f"{pp(b.mean() - a.mean())} ({fmt_p(ttest(a, b))})", "Paper": ref})
    return pd.DataFrame(rows)


def t8_moderators(d):
    x = d[d.physician_type == "PCP"].copy()
    rows = []
    if "patient_mix" in x and x.patient_mix.notna().any():
        # matches both the original survey options ("Mostly white patients", "All white patients")
        # and the older short forms ("mostly white", "all white")
        x["white_mix"] = x.patient_mix.astype(str).str.strip().str.lower().str.match(r"^(mostly|all) white").astype(int)
        for g, lab in ((1, "Mostly/all white patients"), (0, "Other")):
            r = diff_test(x[x.white_mix == g], "top3_correct", "dark", "participant_id")
            rows.append({"PCP subgroup": lab, "top-3 dark−light": f"{pp(r['diff'])} ({fmt_p(r['p'])})",
                         "Paper": "−7 pp (P=0.009)" if g == 1 else ""})
    if "sufficient_training" in x and x.sufficient_training.notna().any():
        x["trained"] = x.sufficient_training.astype(str).str.lower().str.startswith("y").astype(int)
        x["Dark"] = x["dark"]
        res = ols_cluster(x, "top3_correct", ["Dark", "trained", "Dark:trained"], "participant_id")
        rows.append({"PCP subgroup": "Self-rated training adequate vs not (interaction Dark×Trained)",
                     "top-3 dark−light": f"{pp(res.loc['Dark:trained', 'coef'])} ({fmt_p(res.loc['Dark:trained', 'p'])})",
                     "Paper": "+5 pp (P=0.079)"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Plotting (colours follow the dataviz guidelines: blue = light skin, orange = dark skin)
C_LIGHT, C_DARK, C_ALT = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")


def _bars(ax, groups, series, width=0.34):
    """series: [(label, color, [(mean, halfCI), ...]), ...]"""
    x = np.arange(len(groups))
    n = len(series)
    for i, (lab, col, vals) in enumerate(series):
        means = [v[0] for v in vals]
        errs = [v[1] if not np.isnan(v[1]) else 0 for v in vals]
        pos = x + (i - (n - 1) / 2) * (width + 0.02)
        ax.bar(pos, means, width=width, color=col, label=lab, zorder=2)
        ax.errorbar(pos, means, yerr=errs, fmt="none", ecolor=INK2, elinewidth=1, capsize=0, zorder=3)
        for px, m, e in zip(pos, means, errs):
            if not np.isnan(m):
                ax.text(px, m + e + 0.012, f"{m:.0%}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(x, groups)


def make_figures(d, out: Path, watermark: str | None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE})
    files = []

    def finish(fig, name):
        if watermark:
            fig.text(0.01, -0.02, "⚠ " + watermark, ha="left", va="top", fontsize=10, color=INK, weight="bold")
        p = out / name
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        files.append(p)

    # Figure 1: physician type × skin tone (after the paper's Fig. 3a,b)
    m = d[d.main8 == 1]
    types = [t for t in TYPES if t in set(m.physician_type)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (y, lab) in zip(axes, (("top1_correct", "Top-1 accuracy"), ("top3_correct", "Top-3 accuracy"))):
        ser = [(name, col, [mean_ci(m[(m.physician_type == t) & (m.dark == dk)][y]) for t in types])
               for name, col, dk in (("FST 1–4 (light)", C_LIGHT, 0), ("FST 5–6 (dark)", C_DARK, 1))]
        _bars(ax, types, ser)
        _style(ax, lab, "")
        ax.set_ylim(0, max(0.6, ax.get_ylim()[1]))
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Diagnostic accuracy by physician type and skin tone · 8 main diseases",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    finish(fig, "fig1_accuracy_by_skin_tone.png")

    # Figure 2: before/after AI assistance × skin tone (after the paper's Fig. 6)
    s = m[m.physician_type.isin(["BCD", "PCP"])]
    groups = [(t, c) for t in ("BCD", "PCP") for c in ("control", "treatment") if
              not s[(s.physician_type == t) & (s.dls_condition == c)].empty]
    if groups:
        labels = [f"{t}\n{'control DLS' if c == 'control' else 'enhanced DLS'}" for t, c in groups]
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey="row")
        spec = [("top1_correct", "Top-1 · without DLS"), ("top1_after", "Top-1 · with DLS"),
                ("top3_correct", "Top-3 · without DLS"), ("top4_after", "Top-4 · with DLS")]
        for ax, (y, title) in zip(axes.flat, spec):
            ser = [(name, col, [mean_ci(s[(s.physician_type == t) & (s.dls_condition == c) & (s.dark == dk)][y])
                                for t, c in groups])
                   for name, col, dk in (("FST 1–4 (light)", C_LIGHT, 0), ("FST 5–6 (dark)", C_DARK, 1))]
            _bars(ax, labels, ser, width=0.32)
            _style(ax, title, "")
            ax.set_ylim(0, 1)
        axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
        fig.tight_layout()
        finish(fig, "fig2_dls_effect_by_skin_tone.png")

    # Figure 3: effect of button order (interface) on choices
    c = d[d.choice.isin(["top", "include", "keep"])]
    if not c.empty and c.interface.nunique() == 2:
        opts = ["top", "include", "keep"]
        names = ["Update top\nprediction", "Update differential\nto include", "Keep my\ndifferential"]
        ser = []
        for lab, col, ui in (("Control UI ('Update top' first)", C_LIGHT, "control"),
                             ("Treatment UI ('Keep' first)", C_ALT, "treatment")):
            x = c[c.interface == ui]
            ser.append((lab, col, [mean_ci((x.choice == o).astype(float)) for o in opts]))
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        _bars(ax, names, ser)
        _style(ax, "How button order changes physicians' choices", "share of decisions")
        ax.set_ylim(0, 0.8)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        finish(fig, "fig3_interface_choices.png")
    return files


# ---------------------------------------------------------------- Main
def to_md(df: pd.DataFrame | None) -> str:
    if df is None or len(df) == 0:
        return "_(not enough data; skipped)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c).strip() for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]).replace("\n", " ") for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Independent reproduction of the core results of Groh et al. 2024")
    ap.add_argument("--data", required=True, help="data CSV (or a zip containing one CSV)")
    ap.add_argument("--map", help="column-mapping JSON (generated by inspect_data.py, then checked by hand)")
    ap.add_argument("--out", default=str(ROOT / "outputs"), help="output folder")
    ap.add_argument("--min-trials", type=int, default=10, help="minimum diagnoses per participant for inclusion (paper = 10; use 1 for class data)")
    ap.add_argument("--first-n", type=int, default=10, help="use only each participant's first N diagnoses (paper = 10)")
    ap.add_argument("--keep-failed-attention", action="store_true",
                    help="do not filter on the attention check (recommended for class data: non-clinician students mostly won't recognise the acne image)")
    ap.add_argument("--simulated", action="store_true", help="label the output as SIMULATED DATA")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()

    mapping = json.loads(Path(a.map).read_text(encoding="utf-8")) if a.map else None
    raw = load(a.data, mapping)
    missing = [c for c in ("participant_id", "physician_type", "fst", "top1_correct", "top3_correct") if c not in raw]
    if missing:
        sys.exit(f"Missing required fields {missing}. First run analysis/inspect_data.py to create and check a column mapping, then pass it with --map.")
    simulated = a.simulated or "synthetic" in Path(a.data).name.lower()
    tag = "SIMULATED DATA – not the paper's results" if simulated else None

    d = select_sample(raw, a.min_trials, a.first_n, attention_filter=not a.keep_failed_attention)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def safe(fn, *args):
        """With a small sample (e.g. only a few people in class) some tests cannot be computed; skip them instead of failing."""
        try:
            return fn(*args)
        except (ValueError, ZeroDivisionError, np.linalg.LinAlgError, KeyError) as e:
            print(f"  [skipped {fn.__name__}: {e}]")
            return None

    sections = [("Table 1 Overall diagnostic accuracy", safe(t1_overall, d)),
                ("Table 2 Dark (FST 5–6) vs light (FST 1–4) skin: accuracy gap (OLS, participant-clustered robust SEs)", safe(t2_skin, d)),
                ("Table 3a BCD: skin-tone gap per disease (two-sided t-tests)", safe(t3_disease, d, "BCD")),
                ("Table 3b PCP: skin-tone gap per disease (two-sided t-tests)", safe(t3_disease, d, "PCP"))]
    has_dls = {"top1_after", "top4_after", "dls_condition"} <= set(d)
    if has_dls:
        reg = safe(t5_regression, d)
        sections += [("Table 4 Accuracy before/after AI (DLS) assistance (8 main conditions)", safe(t4_dls, d)),
                     ("Table 4b When the AI gives a wrong suggestion", safe(t4b_errors, d)),
                     ("Table 5 Regression: specialty × skin tone × AI (after Extended Data Table 3; reference group = PCP, light skin, no AI)",
                      reg[0] if reg else None)]
    if {"choice", "interface"} <= set(d):
        sections.append(("Table 6 Effect of button order (interface) on choices (only diagnoses where the suggestion page was shown)", safe(t6_interface, d)))
    if "biopsy" in d:
        sections.append(("Table 7 Biopsy / referral recommendations", safe(t7_referral, d)))
    sections.append(("Table 8 Moderators for primary care physicians", safe(t8_moderators, d)))

    header = [f"# Reproduction results: Groh et al., Nature Medicine 2024\n",
              f"- Data file: `{Path(a.data).name}`",
              f"- Included: {d.participant_id.nunique()} participants, {len(d)} diagnoses"
              f" (rule: ≥{a.min_trials} diagnoses, passing the attention check {'not ' if a.keep_failed_attention else ''}required, "
              f"first {a.first_n} used, attention-check image dropped)",
              f"- Share of dark-skin images: {pct(d.dark.mean())} (paper 32%)"]
    if simulated:
        header.insert(1, "> ⚠️ **Simulated data**: the numbers below are randomly generated, for rehearsal only, and are not the paper's results.\n")
    md = "\n".join(header) + "\n\n"
    print("\n".join(header).replace("> ", ""))
    with pd.option_context("display.max_columns", None, "display.width", 250, "display.max_colwidth", 60):
        for title, tab in sections:
            print(f"\n=== {title} ===")
            print(tab.to_string(index=False) if tab is not None and len(tab) else "(not enough data; skipped)")
            md += f"## {title}\n\n{to_md(tab)}\n"
            if tab is not None and len(tab):
                tab.to_csv(out / ("table" + title.split()[1] + ".csv"), index=False, encoding="utf-8-sig")

    if not a.no_figures and len(d):
        files = make_figures(d, out, tag)
        md += "## Figures\n\n" + "\n".join(f"![{f.stem}]({f.name})" for f in files) + "\n"
        print("\nFigures saved: " + ", ".join(f.name for f in files))
    (out / "results_summary.md").write_text(md, encoding="utf-8")
    print(f"\nSummary report: {out / 'results_summary.md'}")


if __name__ == "__main__":
    main()
