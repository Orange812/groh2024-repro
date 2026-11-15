"""
Statistics utilities — the same tests as the paper (no statsmodels dependency)

The paper's Results section states that, unless stated otherwise, all statistical comparisons are based on
"ordinary least squares (OLS) regression with robust standard errors clustered at the participant level";
the P values in the figures come from two-sided t-tests.

This file implements:
  ols_cluster()  OLS + CR1 cluster-robust standard errors (same default small-sample correction as Stata / statsmodels:
                 c = G/(G-1) * (N-1)/(N-K), t distribution with G-1 degrees of freedom)
  diff_test()    difference between two groups = regression on a 0/1 indicator (as in the paper's main text)
  ttest()        two-sided t-test (as in the paper's figures)
  check_against_statsmodels()  cross-check against statsmodels, if it is installed
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _term_columns(terms):
    return sorted({p for t in terms for p in t.split(":")})


def build_design(df: pd.DataFrame, terms, intercept: bool = True) -> pd.DataFrame:
    """Turn a list of terms such as ["BCD", "Dark", "BCD:Dark"] into a design matrix (':' marks an interaction = product of columns)."""
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
    """OLS regression with standard errors clustered on the `cluster` column (participant ID).

    Returns coef / se / t / p / 95% CI for each coefficient;
    out.attrs records n (number of observations), clusters (number of clusters) and r2.
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
        raise ValueError("At least 2 clusters (participants) are needed to compute cluster-robust standard errors")

    # Sum of the score vectors within each cluster: S_g = X_g' e_g
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
    """Difference in group means (indicator=1 minus indicator=0); the P value comes from cluster-robust OLS."""
    d = df.dropna(subset=[y, indicator, cluster])
    if d[indicator].nunique() < 2 or d[cluster].nunique() < 2:
        return {"mean0": np.nan, "mean1": np.nan, "diff": np.nan, "p": np.nan, "n": len(d)}
    res = ols_cluster(d, y, [indicator], cluster)
    m0 = d.loc[d[indicator] == 0, y].mean()
    m1 = d.loc[d[indicator] == 1, y].mean()
    return {"mean0": m0, "mean1": m1, "diff": res.loc[indicator, "coef"],
            "p": res.loc[indicator, "p"], "n": len(d)}


def ttest(a, b) -> float:
    """P value of a two-sided t-test (the figure legends say "two-sided t-test" without stating equal variances; the classic Student t is used here)."""
    a = pd.Series(a).dropna()
    b = pd.Series(b).dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return float(stats.ttest_ind(a, b, equal_var=True).pvalue)


def mean_ci(x, z: float = 1.96):
    """Mean and half-width of the 95% confidence interval (for the error bars in the figures)."""
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
    """If statsmodels is installed, return a table comparing this implementation's coefficients/SEs with statsmodels; otherwise None."""
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
