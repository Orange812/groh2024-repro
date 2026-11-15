"""
独立复现论文的核心结果（Groh et al., Nature Medicine 2024）

它做什么：
  读取一份「每行 = 一位医生对一张图片的一次诊断」的数据表，按论文口径筛样本，然后输出
    表1  各类医生的总体准确率（top-1 / top-3）
    表2  深色 vs 浅色皮肤的准确率差距（OLS + 参与者聚类稳健标准误）
    表3  各疾病的肤色差距（BCD、PCP；双侧 t 检验）
    表4  AI（DLS）辅助前后的准确率（含 Wizard-of-Oz 增强版 AI）
    表5  回归（仿 Extended Data Table 3：专长 × 肤色 × AI 交互项）
    表6  按钮顺序（界面）对医生选择的影响
    表7  活检 / 转诊建议的肤色差异
    表8  全科医生的调节因素（接诊人群、培训是否充分）
  每张表都把「论文报告值」放在旁边，方便在演讲里展示"复现值 vs 论文值"。
  图片保存在 --out 目录（PNG，英文标注，避免中文字体问题）。

三种数据来源（都用同一个脚本）：
  1) 作者公开数据：ResearchBox #1802 的 Experimental Data.csv(.zip)
     → 先运行 inspect_data.py 生成列名映射 mapping.json，再：
       python analysis/key_results.py --data "Experimental Data.csv.zip" --map mapping.json
  2) 模拟数据（彩排用）：python analysis/key_results.py --data data/synthetic_experiment.csv
  3) 课堂 demo 收集的数据：python analysis/key_results.py --data demo_app/data/responses.csv --min-trials 1

重要说明：作者自己的 Analysis.ipynb 才是"官方复现"；本脚本是独立重写（re-implementation），
回归设定按论文文字描述推断，个别系数的定义可能与原文略有不同。
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

TYPES = ["BCD", "Resident", "PCP", "Other", "Student"]   # Student = 课堂 demo 里的非临床参与者

# ---------------------------------------------------------------- 论文报告值（用于对照）
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


# ---------------------------------------------------------------- 读取与标准化
def load(path: str, mapping: dict | None) -> pd.DataFrame:
    from inspect_data import read_any                 # 支持 .csv 和 .csv.zip（自动跳过 __MACOSX）
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
    # 没有现成的判分列时，按文本判分（论文作者是人工逐条审核的，这里是近似）
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
    """论文口径：通过注意力检查且至少完成 10 张的参与者，只用他们的前 10 个诊断，并去掉注意力检查图。"""
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


# ---------------------------------------------------------------- 分析
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
        rows.append({"医生类型": t, "人数": s.participant_id.nunique(), "诊断数": len(s),
                     "top-1(全部)": pct(s.top1_correct.mean()), "论文": pct(PAPER["top1_all"].get(t)),
                     "top-3(全部)": pct(s.top3_correct.mean()), "论文 ": pct(PAPER["top3_all"].get(t)),
                     "top-1(8病)": pct(m.top1_correct.mean()), "论文  ": pct(PAPER["top1_main"].get(t)),
                     "top-3(8病)": pct(m.top3_correct.mean()), "论文   ": pct(PAPER["top3_main"].get(t))})
    return pd.DataFrame(rows)


def t2_skin(d):
    rows = []
    for t in ["All"] + TYPES:
        s = d if t == "All" else d[d.physician_type == t]
        if s.participant_id.nunique() < 2:
            continue
        r = {"医生类型": t}
        for y, key in (("top1_correct", "gap_top1"), ("top3_correct", "gap_top3")):
            res = diff_test(s, y, "dark", "participant_id")
            lab = "top-1" if y == "top1_correct" else "top-3"
            r[f"{lab} 浅色"] = pct(res["mean0"])
            r[f"{lab} 深色"] = pct(res["mean1"])
            r[f"{lab} 差距"] = f"{pp(res['diff'])} ({fmt_p(res['p'])})"
            g, p = PAPER[key].get(t, (None, ""))
            r[f"{lab} 论文"] = f"{100 * g:+.0f} pp ({p})" if g is not None else "—"
        rows.append(r)
    return pd.DataFrame(rows)


def t3_disease(d, ptype):
    s = d[(d.physician_type == ptype) & (d.main8 == 1)]
    rows = []
    for dis in list(MAIN8):
        x = s[s.disease8 == dis]
        if x.empty:
            continue
        r = {"疾病": dis, "n": len(x)}
        for y in ("top1_correct", "top3_correct"):
            lab = "top-1" if y == "top1_correct" else "top-3"
            a, b = x.loc[x.dark == 0, y], x.loc[x.dark == 1, y]
            r[f"{lab} 浅/深"] = f"{pct(a.mean())} / {pct(b.mean())}"
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
                "医生": t, "AI 版本": "对照 AI (≈47%)" if cond == "control" else "增强 AI (≈84%)",
                "top-1 前→后": f"{pct(x.top1_correct.mean())} → {pct(x.top1_after.mean())} ({fmt_p(p1)})",
                "相对提升": f"{100 * (x.top1_after.mean() / x.top1_correct.mean() - 1):+.0f}%" if x.top1_correct.mean() > 0 else "—",
                "top-3 → top-4": f"{pct(x.top3_correct.mean())} → {pct(x.top4_after.mean())} ({fmt_p(p4)})",
                "论文(对照 AI)": (f"top-1 {pct(ref['top1'][0])}→{pct(ref['top1'][1])}；"
                                  f"top-3→4 {pct(ref['top34'][0])}→{pct(ref['top34'][1])}") if ref else "论文：再多约 +8 pp",
            })
    return pd.DataFrame(rows)


def t4b_errors(d):
    """AI 给错建议时的影响（论文：top-1 只降 1.2 pp；覆盖正确首要诊断 <2%；三个都错时 10%/14% 采纳为首要诊断）"""
    s = d[(d.main8 == 1) & d.physician_type.isin(["BCD", "PCP"])]
    rows = []
    for t in ("BCD", "PCP"):
        x = s[(s.physician_type == t) & (s.dls_correct == 0)]
        if x.empty:
            continue
        a = x[x.top1_correct == 1]
        b = x[x.top3_correct == 0]
        rows.append({"医生": t, "AI 错误建议的诊断数": len(x),
                     "top-1 变化": pp(x.top1_after.mean() - x.top1_correct.mean()),
                     "原本首要诊断正确却被错误建议替换": pct((a.choice == "top").mean()) if len(a) else "—",
                     "自己三个都错时把错误建议设为首要诊断": pct((b.choice == "top").mean()) if len(b) else "—",
                     "论文": "−1.2 pp；<2%；" + ("10%" if t == "BCD" else "14%")})
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
        "变量": ["Intercept"] + terms,
        "含义": ["PCP·浅色·无AI 的基线", "BCD 比 PCP 高多少", "深色皮肤（PCP）的差距", "BCD 的肤色差距与 PCP 的差别",
                 "对照 AI 的提升（PCP·浅色）", "增强 AI 额外提升", "AI 后 PCP 肤色差距的变化", "AI 对 BCD 的额外提升",
                 "AI 后 BCD 肤色差距的额外变化（三阶交互）"],
    })
    for name, res in out.items():
        tab[f"{name} 系数"] = [pp(res.loc[v, "coef"]) if v != "Intercept" else pct(res.loc[v, "coef"]) for v in tab["变量"]]
        tab[f"{name} P"] = [fmt_p(res.loc[v, "p"]) for v in tab["变量"]]
    ref1, ref4 = PAPER["regression_top1"], PAPER["regression_top4"]
    tab["论文(top-1 / top-4)"] = [
        " / ".join(x for x in (f"{100 * ref1[v]:+.0f}" if v in ref1 else "", f"{100 * ref4[v]:+.0f}" if v in ref4 else "") if x)
        for v in tab["变量"]]
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
        rows.append({"选项": {"top": "Update my top prediction", "include": "Update my differential to include",
                              "keep": "Keep my differential"}[c],
                     "对照界面（'更新首要诊断'在上）": pct(r["mean0"]), "实验界面（'保持'在上）": pct(r["mean1"]),
                     "差异": f"{pp(r['diff'])} ({fmt_p(r['p'])})",
                     "论文": {"top": "−9 pp", "include": "+9 pp", "keep": "无显著差异"}[c]})
    for t, ref in (("BCD", "−12 pp"), ("PCP", "−7 pp")):
        x = s[(s.physician_type == t) & (s.main8 == 1)]
        if x.interface.nunique() == 2:
            r = diff_test(x, "top1_after", "TreatUI", "participant_id")
            rows.append({"选项": f"{t} 看到建议后的 top-1", "对照界面（'更新首要诊断'在上）": pct(r["mean0"]),
                         "实验界面（'保持'在上）": pct(r["mean1"]), "差异": f"{pp(r['diff'])} ({fmt_p(r['p'])})",
                         "论文": ref})
    return pd.DataFrame(rows)


def t7_referral(d):
    rows = []
    for t in ("BCD", "PCP"):
        x = d[d.physician_type == t]
        if x.empty:
            continue
        r = diff_test(x, "biopsy", "dark", "participant_id")
        rows.append({"指标": f"{t} 建议活检", "总体": pct(x.biopsy.mean()), "浅色": pct(r["mean0"]),
                     "深色": pct(r["mean1"]), "差异": f"{pp(r['diff'])} ({fmt_p(r['p'])})",
                     "论文": f"总体 {pct(PAPER['biopsy'][t])}"})
    x = d[d.physician_type == "PCP"].dropna(subset=["refer_derm"])
    if not x.empty:
        r = diff_test(x, "refer_derm", "dark", "participant_id")
        rows.append({"指标": "PCP 转诊皮肤科", "总体": pct(x.refer_derm.mean()), "浅色": pct(r["mean0"]),
                     "深色": pct(r["mean1"]), "差异": f"{pp(r['diff'])} ({fmt_p(r['p'])})",
                     "论文": "总体 28%；深色 +4.4 pp (P=0.012)"})
    x = d[(d.physician_type == "BCD") & (d.main8 == 1)]
    for dis, ref in (("CTCL", "深色更少 (P=0.001)"), ("Atopic dermatitis", "深色更多 (P=0.008)"),
                     ("Pityriasis rosea", "深色更多 (P=0.015)"), ("Pityriasis rubra pilaris", "深色更少 (P=0.033)")):
        y = x[x.disease8 == dis]
        if y.empty:
            continue
        a, b = y.loc[y.dark == 0, "biopsy"], y.loc[y.dark == 1, "biopsy"]
        rows.append({"指标": f"BCD 活检：{dis}", "总体": pct(y.biopsy.mean()), "浅色": pct(a.mean()),
                     "深色": pct(b.mean()), "差异": f"{pp(b.mean() - a.mean())} ({fmt_p(ttest(a, b))})", "论文": ref})
    return pd.DataFrame(rows)


def t8_moderators(d):
    x = d[d.physician_type == "PCP"].copy()
    rows = []
    if "patient_mix" in x and x.patient_mix.notna().any():
        # matches both the original survey options ("Mostly white patients", "All white patients")
        # and the older short forms ("mostly white", "all white")
        x["white_mix"] = x.patient_mix.astype(str).str.strip().str.lower().str.match(r"^(mostly|all) white").astype(int)
        for g, lab in ((1, "主要/全部接诊白人患者"), (0, "其他")):
            r = diff_test(x[x.white_mix == g], "top3_correct", "dark", "participant_id")
            rows.append({"PCP 子组": lab, "top-3 深色−浅色": f"{pp(r['diff'])} ({fmt_p(r['p'])})",
                         "论文": "−7 pp (P=0.009)" if g == 1 else ""})
    if "sufficient_training" in x and x.sufficient_training.notna().any():
        x["trained"] = x.sufficient_training.astype(str).str.lower().str.startswith("y").astype(int)
        x["Dark"] = x["dark"]
        res = ols_cluster(x, "top3_correct", ["Dark", "trained", "Dark:trained"], "participant_id")
        rows.append({"PCP 子组": "自认培训充分 vs 不充分（交互项 Dark×Trained）",
                     "top-3 深色−浅色": f"{pp(res.loc['Dark:trained', 'coef'])} ({fmt_p(res.loc['Dark:trained', 'p'])})",
                     "论文": "+5 pp (P=0.079)"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 作图（配色按 dataviz 规范：蓝=浅色皮肤，橙=深色皮肤）
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

    # 图1：各类医生 × 肤色（仿论文 Fig. 3a,b）
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

    # 图2：AI 辅助前后 × 肤色（仿论文 Fig. 6）
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

    # 图3：按钮顺序（界面）对选择的影响
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


# ---------------------------------------------------------------- 主程序
def to_md(df: pd.DataFrame | None) -> str:
    if df is None or len(df) == 0:
        return "_（数据不足，跳过）_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c).strip() for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]).replace("\n", " ") for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="独立复现 Groh et al. 2024 的核心结果")
    ap.add_argument("--data", required=True, help="数据 CSV（或单文件 zip）")
    ap.add_argument("--map", help="列名映射 JSON（由 inspect_data.py 生成后人工确认）")
    ap.add_argument("--out", default=str(ROOT / "outputs"), help="输出目录")
    ap.add_argument("--min-trials", type=int, default=10, help="纳入分析所需的最少诊断数（论文=10；课堂数据可设 1）")
    ap.add_argument("--first-n", type=int, default=10, help="每人只用前 N 个诊断（论文=10）")
    ap.add_argument("--keep-failed-attention", action="store_true",
                    help="不按注意力检查筛人（课堂数据建议加上：非临床同学多半认不出痤疮图）")
    ap.add_argument("--simulated", action="store_true", help="在输出上标注 SIMULATED DATA")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()

    mapping = json.loads(Path(a.map).read_text(encoding="utf-8")) if a.map else None
    raw = load(a.data, mapping)
    missing = [c for c in ("participant_id", "physician_type", "fst", "top1_correct", "top3_correct") if c not in raw]
    if missing:
        sys.exit(f"缺少必要字段 {missing}。请先运行 analysis/inspect_data.py 生成并确认列名映射，再用 --map 传入。")
    simulated = a.simulated or "synthetic" in Path(a.data).name.lower()
    tag = "SIMULATED DATA – not the paper's results" if simulated else None

    d = select_sample(raw, a.min_trials, a.first_n, attention_filter=not a.keep_failed_attention)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def safe(fn, *args):
        """小样本（例如课堂只有几个人）时某些检验无法计算，跳过而不是报错。"""
        try:
            return fn(*args)
        except (ValueError, ZeroDivisionError, np.linalg.LinAlgError, KeyError) as e:
            print(f"  [跳过 {fn.__name__}：{e}]")
            return None

    sections = [("表1 总体诊断准确率", safe(t1_overall, d)),
                ("表2 深色（FST 5–6）vs 浅色（FST 1–4）皮肤：准确率差距（OLS，参与者聚类稳健标准误）", safe(t2_skin, d)),
                ("表3a BCD：各疾病的肤色差距（双侧 t 检验）", safe(t3_disease, d, "BCD")),
                ("表3b PCP：各疾病的肤色差距（双侧 t 检验）", safe(t3_disease, d, "PCP"))]
    has_dls = {"top1_after", "top4_after", "dls_condition"} <= set(d)
    if has_dls:
        reg = safe(t5_regression, d)
        sections += [("表4 AI（DLS）辅助前后的准确率（8 种主要疾病）", safe(t4_dls, d)),
                     ("表4b AI 给出错误建议时", safe(t4b_errors, d)),
                     ("表5 回归：专长 × 肤色 × AI（仿 Extended Data Table 3；参考组 = PCP、浅色、无 AI）",
                      reg[0] if reg else None)]
    if {"choice", "interface"} <= set(d):
        sections.append(("表6 按钮顺序（界面）对选择的影响（只统计看到建议页面的诊断）", safe(t6_interface, d)))
    if "biopsy" in d:
        sections.append(("表7 活检 / 转诊建议", safe(t7_referral, d)))
    sections.append(("表8 全科医生的调节因素", safe(t8_moderators, d)))

    header = [f"# 复现结果：Groh et al., Nature Medicine 2024\n",
              f"- 数据文件：`{Path(a.data).name}`",
              f"- 纳入分析：{d.participant_id.nunique()} 名参与者，{len(d)} 条诊断"
              f"（规则：≥{a.min_trials} 个诊断、{'不' if a.keep_failed_attention else ''}要求通过注意力检查、"
              f"取前 {a.first_n} 个、去掉注意力检查图）",
              f"- 深色皮肤图片占比：{pct(d.dark.mean())}（论文 32%）"]
    if simulated:
        header.insert(1, "> ⚠️ **模拟数据**：以下数字由随机数生成，只用于彩排，不能当作论文结果。\n")
    md = "\n".join(header) + "\n\n"
    print("\n".join(header).replace("> ", ""))
    with pd.option_context("display.max_columns", None, "display.width", 250, "display.max_colwidth", 60):
        for title, tab in sections:
            print(f"\n=== {title} ===")
            print(tab.to_string(index=False) if tab is not None and len(tab) else "（数据不足，跳过）")
            md += f"## {title}\n\n{to_md(tab)}\n"
            if tab is not None and len(tab):
                tab.to_csv(out / (title.split()[0] + ".csv"), index=False, encoding="utf-8-sig")

    if not a.no_figures and len(d):
        files = make_figures(d, out, tag)
        md += "## 图\n\n" + "\n".join(f"![{f.stem}]({f.name})" for f in files) + "\n"
        print("\n图已保存：" + ", ".join(f.name for f in files))
    (out / "results_summary.md").write_text(md, encoding="utf-8")
    print(f"\n汇总报告：{out / 'results_summary.md'}")


if __name__ == "__main__":
    main()
