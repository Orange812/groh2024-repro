"""
生成「模拟实验数据」——只用于彩排和测试代码，绝不能当作论文结果引用！

为什么需要它：
  - 真实数据（ResearchBox #1802 的 Experimental Data.csv）需要你在自己电脑上下载；
  - 在拿到真实数据之前，用这份模拟数据可以先把分析脚本、图表和现场演示全部跑通；
  - 模拟参数按论文报告的汇总数字粗略校准（人数、准确率、肤色差距、AI 提升幅度、按钮顺序效应），
    所以跑出来的结果"长得像"论文，但每个数字都是随机生成的。

输出：data/synthetic_experiment.csv（字段与 demo_app 记录的格式相同，即本工具包的"标准格式"）

用法：
  python analysis/make_synthetic_data.py              # 默认随机种子 2024
  python analysis/make_synthetic_data.py --seed 7 --out data/sim2.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.diseases import AUXILIARY, MAIN8  # noqa: E402

MAIN_COUNTS = {"Atopic dermatitis": 31, "CTCL": 48, "Dermatomyositis": 34, "Lyme disease": 30,
               "Lichen planus": 32, "Pityriasis rosea": 33, "Pityriasis rubra pilaris": 47,
               "Secondary syphilis": 29}                       # 论文 Methods：共 284 张
N_AUX_IMAGES = 79                                               # 另外 38 种病的 79 张（Fitzpatrick 17k）

# 进入主分析的人数（通过注意力检查且完成 ≥10 张）与总招募人数（论文 Results）
ANALYZED = {"BCD": 296, "Resident": 83, "PCP": 350, "Other": 113}
RECRUITED = {"BCD": 389, "Resident": 116, "PCP": 459, "Other": 154}

# 首要诊断 / 前三诊断准确率目标（8 种主要疾病）与深色皮肤差距（百分点 / 100）
TOP1 = {"BCD": 0.27, "Resident": 0.24, "PCP": 0.13, "Other": 0.12}
TOP3 = {"BCD": 0.37, "Resident": 0.35, "PCP": 0.17, "Other": 0.16}
GAP1 = {"BCD": -0.05, "Resident": -0.05, "PCP": -0.03, "Other": -0.05}
GAP3 = {"BCD": -0.03, "Resident": -0.05, "PCP": -0.04, "Other": -0.04}
DARK_SHARE = 0.32

# 疾病难度（相对平均）与 BCD 在各病上的肤色差距调整（论文：AD/Lyme/PR/CTCL 深色显著更差，LP 相反）
DISEASE_DIFF = {"Atopic dermatitis": 0.10, "CTCL": -0.08, "Dermatomyositis": 0.10, "Lyme disease": 0.12,
                "Lichen planus": 0.05, "Pityriasis rosea": 0.02, "Pityriasis rubra pilaris": -0.10,
                "Secondary syphilis": -0.05}
DISEASE_GAP = {"Atopic dermatitis": -0.12, "Lyme disease": -0.14, "Pityriasis rosea": -0.13, "CTCL": -0.05,
               "Lichen planus": 0.10, "Dermatomyositis": 0.05, "Pityriasis rubra pilaris": 0.05,
               "Secondary syphilis": 0.05}


def _center(d: dict) -> dict:
    w = np.array([MAIN_COUNTS[k] for k in d])
    v = np.array(list(d.values()))
    m = (w * v).sum() / w.sum()
    return {k: d[k] - m for k in d}


DISEASE_DIFF = _center(DISEASE_DIFF)
DISEASE_GAP = _center(DISEASE_GAP)


def make_images(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for disease, n in MAIN_COUNTS.items():
        fst = np.resize(rng.permutation([1, 2, 3, 4, 5, 6]), n)
        rng.shuffle(fst)
        for f in fst:
            rows.append({"disease": disease, "fst": int(f), "attention_check": 0})
    aux = rng.choice(AUXILIARY, N_AUX_IMAGES)
    for d in aux:
        rows.append({"disease": str(d), "fst": int(rng.integers(1, 7)), "attention_check": 0})
    rows.insert(0, {"disease": "Acne", "fst": 5, "attention_check": 1})     # 注意力检查图（痤疮）
    img = pd.DataFrame(rows)
    img["image_id"] = [f"img_{i:03d}" for i in range(len(img))]

    # 每张图固定的 AI 预测（与论文一致：同一张图对所有人给出同样的建议）
    classes = list(MAIN8) + ["Other"]
    ctrl, treat, sugg_c, sugg_t = [], [], [], []
    for _, r in img.iterrows():
        true9 = r.disease if r.disease in MAIN8 else "Other"
        p_ok = (0.57 if r.fst >= 5 else 0.55) if true9 != "Other" else 0.25
        pred = true9 if rng.random() < p_ok else rng.choice([c for c in classes if c != true9])
        pred_t = pred if pred == true9 or rng.random() >= 0.65 else true9   # Wizard of Oz：65% 错误翻转为正确
        ctrl.append(pred)
        treat.append(pred_t)
        aux_pick = str(rng.choice(AUXILIARY))                              # 预测 Other 时随机给一个辅助疾病
        sugg_c.append(aux_pick if pred == "Other" else pred)
        sugg_t.append(aux_pick if pred_t == "Other" else pred_t)
    img["dls_pred_control"], img["dls_pred_treatment"] = ctrl, treat
    img["suggestion_control"], img["suggestion_treatment"] = sugg_c, sugg_t
    return img


def make_participants(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    pid = 0
    for t, total in RECRUITED.items():
        n_ok = ANALYZED[t]
        n_fail_attn = round(total * 0.02)
        for i in range(total):
            pid += 1
            completed = i < n_ok
            fail_attn = (not completed) and (i - n_ok) < n_fail_attn
            if completed:
                n_trials = 10 + int(rng.geometric(0.35)) - 1 if rng.random() < 0.55 else 10
            else:
                n_trials = int(rng.integers(10, 16)) if fail_attn else int(rng.integers(1, 10))
            if t == "Resident":
                years = "in training"
            else:
                years = rng.choice(["20+", "10-20", "2-10", "0-2"], p=[0.44, 0.29, 0.24, 0.03])
            mix = rng.choice(["about equal", "mostly white", "all white", "mostly non-white",
                              "all non-white", "not applicable"], p=[0.32, 0.43, 0.02, 0.15, 0.07, 0.01])
            p_no = {"BCD": 0.33, "PCP": 0.67, "Resident": 0.40, "Other": 0.60}[t]
            rows.append({
                "participant_id": f"P{pid:04d}", "physician_type": t, "years": years, "patient_mix": mix,
                "sufficient_training": "no" if rng.random() < p_no else "yes",
                "country": rng.choice(["United States", "India", "Canada", "United Kingdom", "Italy",
                                       "Germany", "Other"], p=[0.5, 0.12, 0.08, 0.05, 0.04, 0.03, 0.18]),
                "dls_condition": rng.choice(["control", "treatment"]),
                "interface": rng.choice(["control", "treatment"]),
                "n_trials": n_trials, "fail_attention": bool(fail_attn),
                "skill": rng.normal(0, 0.06),
            })
    return pd.DataFrame(rows)


def simulate(seed: int = 2024) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    img = make_images(rng)
    people = make_participants(rng)
    attention = img[img.attention_check == 1].iloc[0]
    pool = img[img.attention_check == 0].reset_index(drop=True)
    out = []

    for _, p in people.iterrows():
        t = p.physician_type
        order = [attention] + [pool.iloc[j] for j in rng.choice(len(pool), p.n_trials - 1, replace=False)]
        for k, im in enumerate(order, start=1):
            dark = int(im.fst >= 5)
            main = im.disease in MAIN8
            if im.attention_check:
                p1 = 0.02 if p.fail_attention else 0.97
                p3 = p1
            else:
                scale = 1.0 if t in ("BCD", "Resident") else 0.5
                diff = DISEASE_DIFF.get(im.disease, 0.0) * scale
                g1 = GAP1[t] + (DISEASE_GAP.get(im.disease, 0.0) if t == "BCD" else 0.0)
                g3 = GAP3[t]
                if t == "PCP" and p.patient_mix in ("mostly white", "all white"):
                    g3 -= 0.04
                if t == "PCP" and p.sufficient_training == "yes":
                    g3 += 0.03
                base1 = TOP1[t] - DARK_SHARE * GAP1[t]
                base3 = TOP3[t] - DARK_SHARE * GAP3[t] + (0.0 if main else 0.02)
                p1 = np.clip(base1 + diff + dark * g1 + p.skill, 0.01, 0.95)
                p3 = np.clip(base3 + diff + dark * g3 + p.skill, p1, 0.97)
            top1 = rng.random() < p1
            top3 = top1 or (rng.random() < (p3 - p1) / max(1 - p1, 1e-6))

            # 文字诊断（让"按文本判分"的路径也能被测试）
            wrong = [d for d in list(MAIN8) + AUXILIARY[:12] if d != im.disease]
            dx = list(rng.choice(wrong, 3, replace=False))
            if top1:
                dx[0] = im.disease
            elif top3:
                dx[int(rng.integers(1, 3))] = im.disease

            # AI 建议与选择
            cond = p.dls_condition
            sugg = im[f"suggestion_{cond}"]
            sugg_ok = sugg == im.disease
            in_diff = (sugg_ok and top3) or ((not sugg_ok) and rng.random() < 0.15)
            if im.attention_check or in_diff:
                choice = "same"
            else:
                top_share = (0.45 if t in ("BCD", "Resident") else 0.40) - (0.14 if p.interface == "treatment" else 0)
                if sugg_ok:
                    accept = 0.64 + (0.07 if cond == "treatment" else 0)
                    if t == "PCP" and dark:
                        accept -= 0.14                    # 让全科医生的肤色差距在 AI 辅助后扩大
                    r = rng.random()
                    choice = "top" if r < accept * top_share else ("include" if r < accept else "keep")
                else:
                    if top1:
                        pt, pi = 0.02, 0.15
                    elif top3:
                        pt, pi = 0.05, 0.25
                    else:
                        pt, pi = (0.10 if t in ("BCD", "Resident") else 0.14), 0.30
                    r = rng.random()
                    choice = "top" if r < pt else ("include" if r < pt + pi else "keep")
            top1_after = sugg_ok if choice == "top" else top1
            top4_after = top3 or (choice in ("top", "include") and sugg_ok)

            # 活检 / 转诊 / 信心
            if t == "BCD":
                pb = 0.26 + {"CTCL": 0.25 - 0.20 * dark, "Atopic dermatitis": 0.10 * dark,
                             "Pityriasis rosea": 0.10 * dark, "Pityriasis rubra pilaris": -0.10 * dark}.get(im.disease, 0)
            else:
                pb = {"Resident": 0.20, "PCP": 0.07, "Other": 0.08}[t] + (0.06 - 0.05 * dark if im.disease == "CTCL" else 0)
            biopsy = rng.random() < np.clip(pb, 0.01, 0.95)
            refer = np.nan if t == "BCD" else float(rng.random() < 0.27 + 0.044 * dark)
            conf = 50 if rng.random() < 0.70 else int(np.clip(round((50 + 30 * (top3 - 0.35) + rng.normal(0, 18)) / 5) * 5, 0, 100))

            out.append({
                "participant_id": p.participant_id, "physician_type": t, "years": p.years,
                "patient_mix": p.patient_mix, "sufficient_training": p.sufficient_training, "country": p.country,
                "dls_condition": cond, "interface": p.interface, "trial": k, "image_id": im.image_id,
                "disease": im.disease, "fst": int(im.fst), "attention_check": int(im.attention_check),
                "dx1": dx[0], "dx2": dx[1], "dx3": dx[2], "confidence": conf, "biopsy": int(biopsy),
                "refer_derm": refer, "top1_correct": int(top1), "top3_correct": int(top3),
                "dls_suggestion": sugg, "dls_correct": int(sugg_ok), "choice": choice,
                "top1_after": int(top1_after), "top4_after": int(top4_after),
            })
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description="生成模拟实验数据（仅供彩排/测试）")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--out", default=str(ROOT / "data" / "synthetic_experiment.csv"))
    a = ap.parse_args()
    df = simulate(a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"[模拟数据] 已写入 {a.out}：{df.participant_id.nunique()} 名虚拟医生，{len(df)} 条诊断记录")
    print("提醒：这是随机生成的数据，只能用于彩排，不能当作论文结果。")


if __name__ == "__main__":
    main()
