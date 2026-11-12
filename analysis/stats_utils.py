"""
统计工具 —— 与论文一致的检验方法（不依赖 statsmodels）

论文 Results 部分写明：除另有说明外，所有统计比较都基于
「普通最小二乘（OLS）回归 + 在参与者层面聚类的稳健标准误」；
图中的 P 值则来自双侧 t 检验。

本文件实现：
  ols_cluster()  OLS + CR1 聚类稳健标准误（与 Stata / statsmodels 的默认小样本校正一致：
                 c = G/(G-1) * (N-1)/(N-K)，t 分布自由度 G-1）
  diff_test()    两组差异 = 对 0/1 指示变量回归（论文主文所用的方式）
  ttest()        双侧 t 检验（论文图中所用的方式）
  check_against_statsmodels()  若装了 statsmodels，可交叉验证
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _term_columns(terms):
    return sorted({p for t in terms for p in t.split(":")})


def build_design(df: pd.DataFrame, terms, intercept: bool = True) -> pd.DataFrame:
    """把 ["BCD", "Dark", "BCD:Dark"] 这样的项列表变成设计矩阵（':' 表示交互项 = 列相乘）。"""
    cols = {}
    if intercept:
        cols["Intercept"] = np.ones(len(df))
    for term in terms:
        v = np.ones(len(df))
        for part in term.split(":"):
            v = v * df[part].astype(float).to_numpy()
        cols[term] = v
    return pd.DataFrame(cols, index=df.index)


def ols_cluster(df: pd.DataFrame, y: str, terms, cluster: str, intercept: bool = True) -> pd.DataFrame:
    """OLS 回归，标准误在 `cluster` 列（参与者 ID）层面聚类。

    返回每个系数的 coef / se / t / p / 95% CI；
    out.attrs 中记录 n（观测数）、clusters（聚类数）、r2。
    """
    need = [y, cluster] + _term_columns(terms)
    d = df.dropna(subset=need)
    X = build_design(d, terms, intercept)
    Xm = X.to_numpy(dtype=float)
    yv = d[y].astype(float).to_numpy()
    n, k = Xm.shape

    XtX_inv = np.linalg.pinv(Xm.T @ Xm)
    beta = XtX_inv @ Xm.T @ yv
    resid = yv - Xm @ beta

    groups = pd.factorize(d[cluster])[0]
    G = int(groups.max()) + 1
    if G < 2:
        raise ValueError("至少需要 2 个聚类（参与者）才能计算聚类稳健标准误")

    # 每个聚类的得分向量之和：S_g = X_g' e_g
    S = np.zeros((G, k))
    np.add.at(S, groups, Xm * resid[:, None])
    meat = S.T @ S
    c = (G / (G - 1)) * ((n - 1) / max(n - k, 1))
    V = c * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.clip(np.diag(V), 0, None))

    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se
    dof = G - 1
    p = 2 * stats.t.sf(np.abs(t), dof)
    crit = stats.t.ppf(0.975, dof)

    out = pd.DataFrame(
        {"coef": beta, "se": se, "t": t, "p": p,
         "ci_low": beta - crit * se, "ci_high": beta + crit * se},
        index=X.columns,
    )
    tss = float(((yv - yv.mean()) ** 2).sum())
    out.attrs.update(n=int(n), clusters=G, r2=(1 - float(resid @ resid) / tss) if tss > 0 else np.nan)
    return out


def diff_test(df: pd.DataFrame, y: str, indicator: str, cluster: str) -> dict:
    """两组均值差（indicator=1 减 indicator=0），P 值来自聚类稳健 OLS。"""
    d = df.dropna(subset=[y, indicator, cluster])
    if d[indicator].nunique() < 2 or d[cluster].nunique() < 2:
        return {"mean0": np.nan, "mean1": np.nan, "diff": np.nan, "p": np.nan, "n": len(d)}
    res = ols_cluster(d, y, [indicator], cluster)
    m0 = d.loc[d[indicator] == 0, y].mean()
    m1 = d.loc[d[indicator] == 1, y].mean()
    return {"mean0": m0, "mean1": m1, "diff": res.loc[indicator, "coef"],
            "p": res.loc[indicator, "p"], "n": len(d)}


def ttest(a, b) -> float:
    """双侧 t 检验的 P 值（论文图注写的是 two-sided t-test，未说明是否方差齐，这里用经典 Student t）。"""
    a = pd.Series(a).dropna()
    b = pd.Series(b).dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return float(stats.ttest_ind(a, b, equal_var=True).pvalue)


def mean_ci(x, z: float = 1.96):
    """均值和 95% 置信区间半宽（图中误差线用）。"""
    x = pd.Series(x).dropna().astype(float)
    if len(x) == 0:
        return np.nan, np.nan
    m = x.mean()
    half = z * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return m, half


def fmt_p(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "—"
    return "P<0.001" if p < 0.001 else f"P={p:.3f}"


def check_against_statsmodels(df, y, terms, cluster) -> pd.DataFrame | None:
    """若安装了 statsmodels，返回本实现与 statsmodels 的系数/标准误对照表；否则返回 None。"""
    try:
        import statsmodels.api as sm  # noqa: F401
    except ImportError:
        return None
    import statsmodels.api as sm
    d = df.dropna(subset=[y, cluster] + _term_columns(terms))
    X = build_design(d, terms)
    model = sm.OLS(d[y].astype(float), X).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d[cluster])[0]})
    mine = ols_cluster(d, y, terms, cluster)
    return pd.DataFrame({"coef_mine": mine["coef"], "coef_sm": model.params,
                         "se_mine": mine["se"], "se_sm": model.bse})
