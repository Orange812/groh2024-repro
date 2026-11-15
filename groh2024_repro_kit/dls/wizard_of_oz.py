"""
Build the "enhanced AI" (treatment DLS) — the paper's Wizard-of-Oz method

What the paper did: pick 65% of the control AI's wrong predictions at random and change them to the correct answer,
which raises top-1 accuracy from 47% to about 84%. This "pretend-stronger" AI simulates a future, more accurate system;
the aim is not to evaluate the AI itself but to see how a more accurate AI affects physicians' decisions.

About the 84%: 47% + (1 − 47%) × 65% ≈ 81%. Checked against the authors' data (logs/step7_woz/woz_reproduce.py):
the enhanced AI scores 81.8% on all 363 labelled images and 84.2% on the 8 main conditions.
So the paper's 84% is the accuracy on the 8 main conditions; there is no unexplained gap.

The input CSV needs two columns: label (true 9-class label; write Other for the "other" class) and pred (the control AI's prediction); an fst column is optional.
Usage:
  python dls/wizard_of_oz.py --pred predictions.csv --out enhanced.csv     # flip each wrong prediction to correct independently with probability 65%
  python dls/wizard_of_oz.py --pred predictions.csv --exact                # flip exactly 65% of the wrong predictions (results don't swing with the seed)
  python dls/wizard_of_oz.py --demo                                        # no data needed: demo on random data
The authors' real control-AI predictions are in logs/step7_woz/woz_input.csv (generated from the authors' data by woz_reproduce.py).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def enhance(df: pd.DataFrame, flip_rate: float = 0.65, seed: int = 0,
            label_col: str = "label", pred_col: str = "pred", exact: bool = False) -> pd.Series:
    """Return the enhanced predictions: a flip_rate share of the wrong predictions is changed to the correct answer; correct predictions are unchanged.

    exact=False: each wrong prediction is corrected independently with probability flip_rate (the number corrected varies with the seed);
    exact=True:  exactly round(flip_rate × n_wrong) are corrected; only which ones is random."""
    rng = np.random.default_rng(seed)
    wrong = (df[pred_col] != df[label_col]).to_numpy()
    if exact:
        idx = np.flatnonzero(wrong)
        chosen = rng.choice(idx, size=int(round(flip_rate * len(idx))), replace=False)
        flip = np.zeros(len(df), dtype=bool)
        flip[chosen] = True
    else:
        flip = wrong & (rng.random(len(df)) < flip_rate)
    return df[pred_col].where(~flip, df[label_col])


def report(df, pred_col, fst_col="fst"):
    ok = df[pred_col] == df["label"]
    s = f"{ok.mean():.1%}"
    if fst_col in df and df[fst_col].notna().any():
        dark = pd.to_numeric(df[fst_col], errors="coerce") >= 5
        s += f" (light FST1–4: {ok[~dark].mean():.1%}, dark FST5–6: {ok[dark].mean():.1%})"
    main8 = df["label"].astype(str).str.lower() != "other"
    if (~main8).any():
        s += f"; 8 main conditions: {ok[main8].mean():.1%}"
    return s


def main():
    ap = argparse.ArgumentParser(description="Wizard of Oz: turn the control AI into a high-accuracy AI")
    ap.add_argument("--pred", help="CSV with label and pred columns")
    ap.add_argument("--out", default="treatment_predictions.csv")
    ap.add_argument("--flip-rate", type=float, default=0.65)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exact", action="store_true", help="correct exactly a flip-rate share of the wrong predictions (default: each one independently at random)")
    ap.add_argument("--demo", action="store_true", help="no data needed: demo on 364 randomly generated images")
    a = ap.parse_args()

    if a.demo or not a.pred:
        rng = np.random.default_rng(1)
        classes = ["AD", "CTCL", "DM", "Lyme", "LP", "PR", "PRP", "SS", "Other"]
        label = rng.choice(classes, 364)
        wrong = np.array([rng.choice([c for c in classes if c != l]) for l in label])
        pred = np.where(rng.random(364) < 0.47, label, wrong)
        df = pd.DataFrame({"label": label, "pred": pred, "fst": rng.integers(1, 7, 364)})
        print("[demo mode] generating random 'control AI' predictions for 364 images")
    else:
        df = pd.read_csv(a.pred)

    df["pred_treatment"] = enhance(df, a.flip_rate, a.seed, exact=a.exact)
    n_wrong = int((df.pred != df.label).sum())
    n_flip = int(((df.pred != df.label) & (df.pred_treatment == df.label)).sum())
    p = (df.pred == df.label).mean()
    print(f"Control AI accuracy:  {report(df, 'pred')}")
    print(f"Enhanced AI accuracy: {report(df, 'pred_treatment')}")
    print(f"   {n_wrong} wrong predictions, {n_flip} corrected ({n_flip / max(n_wrong, 1):.1%}; "
          f"{'exact share' if a.exact else 'each independently at random, seed ' + str(a.seed)})")
    print(f"Expected: p + (1 − p) × {a.flip_rate} = {p + (1 - p) * a.flip_rate:.1%}")
    if not a.demo and a.pred:
        df.to_csv(a.out, index=False)
        print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
