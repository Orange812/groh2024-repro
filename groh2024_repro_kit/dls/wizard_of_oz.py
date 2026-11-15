"""
制造「增强 AI」（Treatment DLS）——论文的 Wizard-of-Oz 方法

论文原文做法：把对照 AI 的错误预测随机挑出 65%，直接改成正确答案，
于是 top-1 准确率从 47% 提升到约 84%。这个"假装更强"的 AI 用来模拟未来更准确的系统，
研究目的不是评估 AI 本身，而是看更准的 AI 如何影响医生决策。

关于 84%：47% + (1 − 47%) × 65% ≈ 81%。用作者数据核对（logs/step7_woz/woz_reproduce.py）：
增强 AI 在全部 363 张有标签图片上是 81.8%，在 8 种主要疾病上是 84.2%。
也就是说，论文的 84% 是 8 种主要疾病上的准确率，并没有无法解释的差距。

输入 CSV 需要两列：label（真实 9 类标签，"其他"类写 Other）和 pred（对照 AI 的预测）；可选 fst 列。
用法：
  python dls/wizard_of_oz.py --pred 预测.csv --out 增强后预测.csv          # 每个错误预测独立以 65% 概率改对
  python dls/wizard_of_oz.py --pred 预测.csv --exact                      # 正好改对 65% 的错误预测（结果不随种子大起大落）
  python dls/wizard_of_oz.py --demo                                       # 不需要数据，用随机数据演示
作者真实的对照 AI 预测可以用 logs/step7_woz/woz_input.csv（由 woz_reproduce.py 从作者数据生成）。
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def enhance(df: pd.DataFrame, flip_rate: float = 0.65, seed: int = 0,
            label_col: str = "label", pred_col: str = "pred", exact: bool = False) -> pd.Series:
    """返回增强后的预测：错误预测中 flip_rate 比例被改为正确答案，正确预测保持不变。

    exact=False：每个错误预测独立以 flip_rate 的概率被改对（改对的总数会随种子波动）；
    exact=True ：正好改对 round(flip_rate × 错误数) 个，只随机决定改哪些。"""
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
        s += f"（浅色 FST1–4：{ok[~dark].mean():.1%}，深色 FST5–6：{ok[dark].mean():.1%}）"
    main8 = df["label"].astype(str).str.lower() != "other"
    if (~main8).any():
        s += f"；8 种主要疾病：{ok[main8].mean():.1%}"
    return s


def main():
    ap = argparse.ArgumentParser(description="Wizard-of-Oz：把对照 AI 增强为高准确率 AI")
    ap.add_argument("--pred", help="含 label 和 pred 列的 CSV")
    ap.add_argument("--out", default="treatment_predictions.csv")
    ap.add_argument("--flip-rate", type=float, default=0.65)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exact", action="store_true", help="正好改对 flip-rate 比例的错误预测（默认逐个独立随机）")
    ap.add_argument("--demo", action="store_true", help="不需要数据，用随机生成的 364 张图演示")
    a = ap.parse_args()

    if a.demo or not a.pred:
        rng = np.random.default_rng(1)
        classes = ["AD", "CTCL", "DM", "Lyme", "LP", "PR", "PRP", "SS", "Other"]
        label = rng.choice(classes, 364)
        wrong = np.array([rng.choice([c for c in classes if c != l]) for l in label])
        pred = np.where(rng.random(364) < 0.47, label, wrong)
        df = pd.DataFrame({"label": label, "pred": pred, "fst": rng.integers(1, 7, 364)})
        print("[演示模式] 随机生成 364 张图片的 '对照 AI' 预测")
    else:
        df = pd.read_csv(a.pred)

    df["pred_treatment"] = enhance(df, a.flip_rate, a.seed, exact=a.exact)
    n_wrong = int((df.pred != df.label).sum())
    n_flip = int(((df.pred != df.label) & (df.pred_treatment == df.label)).sum())
    p = (df.pred == df.label).mean()
    print(f"对照 AI 准确率：{report(df, 'pred')}")
    print(f"增强 AI 准确率：{report(df, 'pred_treatment')}")
    print(f"   错误预测 {n_wrong} 个，改对 {n_flip} 个（{n_flip / max(n_wrong, 1):.1%}；"
          f"{'正好按比例' if a.exact else '逐个独立随机，种子 ' + str(a.seed)}）")
    print(f"理论值：p + (1 − p) × {a.flip_rate} = {p + (1 - p) * a.flip_rate:.1%}")
    if not a.demo and a.pred:
        df.to_csv(a.out, index=False)
        print(f"已写入 {a.out}")


if __name__ == "__main__":
    main()
