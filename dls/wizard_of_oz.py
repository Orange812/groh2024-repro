"""
制造「增强 AI」（Treatment DLS）——论文的 Wizard-of-Oz 方法

论文原文做法：把对照 AI 的错误预测随机挑出 65%，直接改成正确答案，
于是 top-1 准确率从 47% 提升到 84%。这个"假装更强"的 AI 用来模拟未来更准确的系统，
研究目的不是评估 AI 本身，而是看更准的 AI 如何影响医生决策。

算一下：47% + (1 − 47%) × 65% ≈ 81%，而论文报告的是 84%；原文没有解释这约 3 个百分点的差别，
演讲时可以作为一个"细节值得追问"的小例子（例如可能与抽样随机性或 9 类计分口径有关，属推测）。

输入 CSV 需要两列：label（真实 9 类标签）和 pred（对照 AI 的预测）；可选 fst 列。
用法：
  python dls/wizard_of_oz.py --pred dls/checkpoint/val_predictions.csv --out dls/treatment_predictions.csv
  python dls/wizard_of_oz.py --demo              # 用随机数据演示 47% → 约 82%
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def enhance(df: pd.DataFrame, flip_rate: float = 0.65, seed: int = 0,
            label_col: str = "label", pred_col: str = "pred") -> pd.Series:
    """返回增强后的预测：错误预测中随机 flip_rate 比例被改为正确答案，正确预测保持不变。"""
    rng = np.random.default_rng(seed)
    wrong = df[pred_col] != df[label_col]
    flip = wrong & (rng.random(len(df)) < flip_rate)
    return df[pred_col].where(~flip, df[label_col])


def report(df, pred_col, fst_col="fst"):
    acc = (df[pred_col] == df["label"]).mean()
    s = f"{acc:.1%}"
    if fst_col in df and df[fst_col].notna().any():
        dark = pd.to_numeric(df[fst_col], errors="coerce") >= 5
        s += (f"（浅色 FST1–4：{(df[pred_col] == df['label'])[~dark].mean():.1%}，"
              f"深色 FST5–6：{(df[pred_col] == df['label'])[dark].mean():.1%}）")
    return s


def main():
    ap = argparse.ArgumentParser(description="Wizard-of-Oz：把对照 AI 增强为高准确率 AI")
    ap.add_argument("--pred", help="含 label 和 pred 列的 CSV")
    ap.add_argument("--out", default="treatment_predictions.csv")
    ap.add_argument("--flip-rate", type=float, default=0.65)
    ap.add_argument("--seed", type=int, default=0)
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

    df["pred_treatment"] = enhance(df, a.flip_rate, a.seed)
    print(f"对照 AI 准确率：{report(df, 'pred')}")
    print(f"增强 AI 准确率：{report(df, 'pred_treatment')}   （错误预测中的 {a.flip_rate:.0%} 被改为正确）")
    print(f"理论值：p + (1 − p) × {a.flip_rate} = {(df.pred == df.label).mean() + (1 - (df.pred == df.label).mean()) * a.flip_rate:.1%}")
    if not a.demo and a.pred:
        df.to_csv(a.out, index=False)
        print(f"已写入 {a.out}")


if __name__ == "__main__":
    main()
