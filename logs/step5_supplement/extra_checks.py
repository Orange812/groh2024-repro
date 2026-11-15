"""Step 5 extra checks (run: python extra_checks.py "<path>/Experimental Data.csv.zip").
Probes the PCP referral rate under different subsets, the 'all three wrong' share,
and DLS accuracy by image subset. Uses only raw columns of the author's data."""
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
MAIN = ['Atopic Dermatitis', 'CTCL', 'Dermatomyositis', 'Lichen Planus', 'Lyme',
        'Pityriasis Rosea', 'Pityriasis Rubra Pilaris', 'Secondary Syphilis']
us = df.country.isin(["United States", "United States of America"])
base = (df.differential_1 != "test") & (df.guess_round == 1) & (df.physician_type == "PCP physician (MD or DO)")
main = (df.total_seen >= 10) & (df.seen <= 10) & (df.seen > 1) & (df.attention_check == 1)
subsets = {
    "main sample (author filters)": base & main,
    "main sample, US only": base & main & us,
    "all PCP round-1 rows": base,
    "incl. >10th image (seen>1)": base & (df.total_seen >= 10) & (df.seen > 1) & (df.attention_check == 1),
    "only >10th image": base & (df.seen > 10) & (df.attention_check == 1),
    "attention-check failers incl.": base & (df.total_seen >= 10) & (df.seen <= 10) & (df.seen > 1),
    "main sample, control DLS arm": base & main & (df.x_enhanced_model == 0),
    "main sample, treatment DLS arm": base & main & (df.x_enhanced_model == 1),
}
print("PCP referral-to-dermatologist rate under different subsets (paper text says 28%):")
for k, m in subsets.items():
    print(f"  {k:34s} {df[m].check_dermatologist.mean():.4f}  (n={m.sum()})")

r2 = df[(df.guess_round == 2) & (df.differential_1 != "test") & main]
w = r2[r2.prediction_correct == 0]
for lab, flag in [("BCD", "dermatologist"), ("PCP", "pcp_physician")]:
    g = w[w[flag] == 1]
    lead = g.update_leading == 1
    all3 = (g.diff1 == 0) & (g.diff2 == 0) & (g.diff3 == 0)
    print(f"{lab}: wrong-DLS obs={len(g)} | joint P(lead & diff1 wrong)={(lead & (g.diff1 == 0)).mean():.4f}"
          f" | joint P(lead & all3 wrong)={(lead & all3).mean():.4f}"
          f" | conditional P(lead | all3 wrong)={lead[all3].mean():.4f} (n={all3.sum()})")

img = df.drop_duplicates("image_id")
m8 = img[img.consensus_label_simplified.isin(MAIN)]
oth = img[~img.consensus_label_simplified.isin(MAIN)]
f = lambda d, c: round(d[c].astype(float).mean(), 4)
print("DLS accuracy, all 364 images: control", f(img, "model_correct_prediction"), "| treatment", f(img, "model_enhanced_correct_prediction"))
print("DLS accuracy, 8 main conditions: control", f(m8, "model_correct_prediction"), "| treatment", f(m8, "model_enhanced_correct_prediction"))
print("DLS accuracy, other images: control", f(oth, "model_correct_prediction"), "| treatment", f(oth, "model_enhanced_correct_prediction"), "n", len(oth))
print("distinct consensus labels:", img.consensus_label.nunique(), "| images with missing label:", img.consensus_label.isna().sum())
