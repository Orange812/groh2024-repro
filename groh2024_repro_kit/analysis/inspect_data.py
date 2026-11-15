"""
Step 1: see what the authors' public data look like, and guess the column-name mapping automatically

Usage:
  python analysis/inspect_data.py "path/to/Experimental Data.csv.zip"
  python analysis/inspect_data.py "path/to/Experimental Data.csv" --write-map mapping.json

It will:
  1) print the number of rows, the column names, and each column's type / missing share / number of distinct values / example values;
  2) for columns with few values (e.g. physician type, experimental arm), print every value and its count;
  3) guess the "toolkit standard field ← original column name" mapping from keywords and write it to mapping.json.
     Open mapping.json and check it by hand (fix wrong guesses; leave missing ones as ""),
     then run: python analysis/key_results.py --data <original file> --map mapping.json

Note: the authors' Analysis.ipynb uses these columns; when unsure, search the notebook for the column name to see how it is used.
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

# Standard field → keywords (earlier keywords carry more weight)
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
                sys.exit("No .csv file found inside the zip")
            print(f"[zip] reading: {members[0]}")
            return pd.read_csv(io.BytesIO(z.read(members[0])), low_memory=False)
    return pd.read_csv(p, low_memory=False)


def guess(columns) -> dict:
    taken, mapping = set(), {}
    # Strip every non-alphanumeric character before comparing, so "Top1After", "top1_after" and "Top-1 after" all match
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
    ap = argparse.ArgumentParser(description="Inspect the experiment data and guess the column-name mapping")
    ap.add_argument("data")
    ap.add_argument("--write-map", default="mapping.json")
    ap.add_argument("--max-levels", type=int, default=15)
    a = ap.parse_args()

    df = read_any(a.data)
    print(f"\nrows {len(df):,}, columns {df.shape[1]}\n")
    with pd.option_context("display.max_colwidth", 50, "display.width", 200):
        info = pd.DataFrame({
            "dtype": df.dtypes.astype(str),
            "missing %": (df.isna().mean() * 100).round(1),
            "distinct values": df.nunique(),
            "examples": [", ".join(map(str, df[c].dropna().unique()[:3])) for c in df.columns],
        })
        print(info.to_string())

    print("\n—— Columns with few values (to confirm how physician type, experimental arm, etc. are coded) ——")
    for c in df.columns:
        if 1 < df[c].nunique() <= a.max_levels:
            vc = df[c].value_counts(dropna=False).head(a.max_levels)
            print(f"\n[{c}]\n" + vc.to_string())

    mapping = guess(df.columns)
    out = {"_note": "Left: this toolkit's standard field names. Right: the matching column in the original data; leave empty if there is none. Check it, then pass it to key_results.py --map.",
           "columns": mapping,
           "values": {"physician_type": {"BCD": [], "Resident": [], "PCP": [], "Other": []},
                      "dls_condition": {"control": [], "treatment": []},
                      "interface": {"control": [], "treatment": []}},
           "_values_note": "If the original values are not written like BCD/PCP or control/treatment, put the original values into the matching list; "
                           "if left empty, the script recognises them by keyword (e.g. 'board' + 'derm' → BCD)."}
    Path(a.write_map).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n—— Guessed column-name mapping (please check by hand) ——")
    for k, v in mapping.items():
        print(f"  {k:<20} ← {v if v else '(not found)'}")
    print(f"\nWrote {a.write_map}. After checking it, run: python analysis/key_results.py --data \"{a.data}\" --map {a.write_map}")


if __name__ == "__main__":
    main()
