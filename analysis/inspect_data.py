"""
第一步：看清作者公开数据长什么样，并自动猜测列名映射

用法：
  python analysis/inspect_data.py "path/to/Experimental Data.csv.zip"
  python analysis/inspect_data.py "path/to/Experimental Data.csv" --write-map mapping.json

它会：
  1) 打印行数、列名、每列的类型 / 缺失 / 不同取值数 / 示例值；
  2) 对取值少的列（如医生类型、实验组别）打印全部取值及频数；
  3) 按关键词猜测「本工具包标准字段 ← 原始列名」的对应关系，写入 mapping.json。
     请打开 mapping.json 人工核对（猜错的改掉，没有的留空 ""），
     然后：python analysis/key_results.py --data <原文件> --map mapping.json

说明：作者的 Analysis.ipynb 里会用到这些列，拿不准时，在 notebook 里搜索列名看它怎么用即可。
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

# 标准字段 → 关键词（越靠前权重越高）
HINTS = {
    "participant_id": ["participant", "user_id", "userid", "uid", "session", "subject", "respondent"],
    "physician_type": ["physician_type", "specialty", "speciality", "profession", "expertise", "occupation", "role",
                       "provider", "type"],
    "trial": ["trial", "question_number", "order", "image_number", "count", "index", "nth"],
    "image_id": ["image_id", "image", "img", "hasher", "md5", "filename", "file", "url", "photo"],
    "disease": ["true_label", "reference", "ground", "truth", "gold", "consensus", "label", "disease", "condition"],
    "fst": ["fitzpatrick", "fst", "skin_type", "skintype", "skin tone", "skin_tone", "skin"],
    "attention_check": ["attention", "acne"],
    "dx1": ["first_diag", "leading", "primary_diag", "diagnosis_1", "dx1", "diag1", "first"],
    "dx2": ["second_diag", "secondary", "diagnosis_2", "dx2", "diag2", "second"],
    "dx3": ["third_diag", "tertiary", "diagnosis_3", "dx3", "diag3", "third"],
    "top1_correct": ["top1", "top_1", "top-1", "first_correct", "correct_1", "leading_correct"],
    "top3_correct": ["top3", "top_3", "top-3", "any_correct", "correct_3", "differential_correct"],
    "dls_condition": ["dls", "model_condition", "ai_condition", "model_type", "model", "algorithm", "enhanced"],
    "interface": ["interface", "button", "ui_", "ui", "layout", "option_order"],
    "dls_suggestion": ["suggestion", "ai_pred", "model_pred", "dls_pred", "prediction", "suggested"],
    "dls_correct": ["suggestion_correct", "ai_correct", "model_correct", "dls_correct", "pred_correct"],
    "choice": ["choice", "decision", "action", "update", "response_to", "keep"],
    "top1_after": ["top1_after", "top1_post", "updated_top1", "after_top1", "top1_with", "final_top1"],
    "top4_after": ["top4", "top_4", "top-4", "after", "updated", "post", "final"],
    "biopsy": ["biopsy"],
    "refer_derm": ["refer", "second_opinion", "dermatologist_referral"],
    "confidence": ["confiden", "certain"],
    "patient_mix": ["patient_skin", "skin_color", "skin_colour", "distribution", "patients"],
    "sufficient_training": ["training", "trained", "sufficient"],
    "country": ["country", "nation", "location"],
    "years": ["years", "experience", "practic"],
}


def read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as z:
            members = [m for m in z.namelist() if m.lower().endswith(".csv") and not m.startswith("__MACOSX")]
            if not members:
                sys.exit("zip 里没有找到 .csv 文件")
            print(f"[zip] 读取其中的：{members[0]}")
            return pd.read_csv(io.BytesIO(z.read(members[0])), low_memory=False)
    return pd.read_csv(p, low_memory=False)


def guess(columns) -> dict:
    taken, mapping = set(), {}
    # 去掉所有非字母数字字符再比较，这样 "Top1After"、"top1_after"、"Top-1 after" 都能对上
    norm = {c: re.sub(r"[^a-z0-9]+", "", str(c).lower()) for c in columns}
    for field, keys in HINTS.items():
        best, best_score = "", 0
        for c, n in norm.items():
            if c in taken:
                continue
            for rank, k in enumerate(keys):
                k2 = re.sub(r"[^a-z0-9]+", "", k.lower())
                if k2 and k2 in n:
                    score = 100 - rank * 5 + (20 if n == k2 else 0)
                    if score > best_score:
                        best, best_score = c, score
        if best:
            mapping[field] = best
            taken.add(best)
        else:
            mapping[field] = ""
    return mapping


def main():
    ap = argparse.ArgumentParser(description="查看实验数据并猜测列名映射")
    ap.add_argument("data")
    ap.add_argument("--write-map", default="mapping.json")
    ap.add_argument("--max-levels", type=int, default=15)
    a = ap.parse_args()

    df = read_any(a.data)
    print(f"\n行数 {len(df):,}，列数 {df.shape[1]}\n")
    with pd.option_context("display.max_colwidth", 50, "display.width", 200):
        info = pd.DataFrame({
            "dtype": df.dtypes.astype(str),
            "缺失%": (df.isna().mean() * 100).round(1),
            "不同取值数": df.nunique(),
            "示例值": [", ".join(map(str, df[c].dropna().unique()[:3])) for c in df.columns],
        })
        print(info.to_string())

    print("\n—— 取值较少的列（便于确认医生类型、实验组别等编码）——")
    for c in df.columns:
        if 1 < df[c].nunique() <= a.max_levels:
            vc = df[c].value_counts(dropna=False).head(a.max_levels)
            print(f"\n[{c}]\n" + vc.to_string())

    mapping = guess(df.columns)
    out = {"_说明": "左边是本工具包的标准字段名，右边填原始数据的列名；没有就留空。核对完再交给 key_results.py --map 使用。",
           "columns": mapping,
           "values": {"physician_type": {"BCD": [], "Resident": [], "PCP": [], "Other": []},
                      "dls_condition": {"control": [], "treatment": []},
                      "interface": {"control": [], "treatment": []}},
           "_values_说明": "如果原始取值不是 BCD/PCP、control/treatment 这样的写法，把原始取值填进对应列表；"
                           "留空则由脚本按关键词自动识别（如含 'board' 和 'derm' 识别为 BCD）。"}
    Path(a.write_map).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n—— 猜测的列名映射（请人工核对）——")
    for k, v in mapping.items():
        print(f"  {k:<20} ← {v if v else '（未找到）'}")
    print(f"\n已写入 {a.write_map}。核对后运行：python analysis/key_results.py --data \"{a.data}\" --map {a.write_map}")


if __name__ == "__main__":
    main()
