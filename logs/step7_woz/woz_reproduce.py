"""Reproduce the Wizard-of-Oz 'treatment DLS' of Groh et al. (2024) from the author's data.

Paper (Methods, DLS development): the control DLS (VGG-16) is 47% accurate on the 364 images; the
treatment DLS was made by randomly changing 65% of the control model's wrong predictions into
correct ones ("Wizard of Oz"), giving 84%.

The author's data stores, for every image, the true 9-class label, the control model's real
prediction and the treatment ('enhanced') prediction actually shown to participants. So the
WoZ step can be (1) audited and (2) redone, independently of re-training the model.

Run from groh2024-repro/:  python logs/step7_woz/woz_reproduce.py
Writes logs/step7_woz/woz_input.csv (label, pred, fst) for groh2024_repro_kit/dls/wizard_of_oz.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "groh2024_repro_kit" / "dls"))
from wizard_of_oz import enhance  # noqa: E402  (the toolkit's WoZ implementation)

NAMES = {0: "Atopic dermatitis", 1: "Lichen planus", 2: "Dermatomyositis", 3: "Pityriasis rosea",
         4: "Pityriasis rubra pilaris", 5: "CTCL", 6: "Lyme disease", 7: "Secondary syphilis", 8: "Other"}

df = pd.read_csv(ROOT / "work/ResearchBox 1802/Data/Experimental Data.csv.zip")
img = df.drop_duplicates("image_id")
n_all = len(img)
img = img[img.consensus_label_numeric.notna() & img.predicted_label_numeric.notna()].copy()
for c in ("consensus_label_numeric", "predicted_label_numeric", "predicted_label_enhanced"):
    img[c] = img[c].astype(int)
img["label"] = img.consensus_label_numeric.map(NAMES)
img["pred"] = img.predicted_label_numeric.map(NAMES)
img["pred_author_treatment"] = img.predicted_label_enhanced.map(NAMES)
img["fst"] = img.fitzpatrick.astype(int)
img["dark"] = img.fst >= 5
img["main8"] = img.label != "Other"
ctrl_ok = img.pred == img.label
treat_ok = img.pred_author_treatment == img.label
print(f"images: {n_all} in data, {len(img)} with labels and predictions "
      f"(excluded: {n_all - len(img)} image without a consensus label)\n")

# ---------------------------------------------------------------- 1. audit the author's WoZ
print("== 1. Audit: is the author's treatment DLS exactly 'flip some wrong predictions to correct'? ==")
unchanged_when_right = (img.pred_author_treatment[ctrl_ok] == img.pred[ctrl_ok]).mean()
w = img[~ctrl_ok]
stays = w.pred_author_treatment == w.pred
flips = w.pred_author_treatment == w.label
other = ~(stays | flips)
print(f"control correct ({ctrl_ok.sum()} images): treatment identical in {unchanged_when_right:.0%}")
print(f"control wrong   ({len(w)} images): flipped to the truth {flips.sum()}, unchanged {stays.sum()}, "
      f"changed to another wrong label {other.sum()}")
print(f"flip rate among wrong predictions: {flips.sum()}/{len(w)} = {flips.mean():.1%}   (paper: 65%)")
print("  note: 127 is also the number printed by the author's notebook (cell 95: print(127/364))\n")

def acc(mask_ok, sub):
    return f"{mask_ok[sub].mean():.1%}"
print("== 2. Accuracy (9-class top-1) vs paper ==")
rows = [("all 363 images", img.index == img.index, "47%", "84%"),
        ("8 main conditions", img.main8, "-", "-"),
        ("8 main, FST 1-4", img.main8 & ~img.dark, "56%", "84%"),
        ("8 main, FST 5-6", img.main8 & img.dark, "58%", "82%")]
print(f"{'subset':<20}{'control':>9}{'paper':>7}{'treatment':>11}{'paper':>7}")
for name, sub, pc, pt in rows:
    print(f"{name:<20}{acc(ctrl_ok, sub):>9}{pc:>7}{acc(treat_ok, sub):>11}{pt:>7}")
p = ctrl_ok.mean()
print(f"theory for all images: p + (1-p) x 0.65 = {p + (1 - p) * 0.65:.1%}")
print(f"-> the paper's '84%' matches the 8 main conditions ({treat_ok[img.main8].mean():.1%}); on all labelled "
      f"images the treatment DLS is {treat_ok.mean():.1%} ({treat_ok.sum()}/{len(img)}; {treat_ok.sum()}/{n_all} = "
      f"{treat_ok.sum() / n_all:.1%} if the unlabelled image counts as wrong)\n")

print("== 3. Was the flipping random? (among the control model's wrong predictions) ==")
t = pd.crosstab(w.dark, flips)
print(f"flip rate FST 1-4: {flips[~w.dark].mean():.1%} (n={(~w.dark).sum()}) | FST 5-6: {flips[w.dark].mean():.1%} "
      f"(n={w.dark.sum()}) | Fisher exact P = {stats.fisher_exact(t.values)[1]:.3f}")
t2 = pd.crosstab(w.label, flips)
print(f"flip rate by true label: chi-square P = {stats.chi2_contingency(t2.values)[1]:.3f}")
print("  " + ", ".join(f"{k}: {v:.0%}" for k, v in flips.groupby(w.label).mean().sort_values().items()))
pr = w.predicted_label_probability
print(f"model confidence of flipped vs kept: {pr[flips].mean():.3f} vs {pr[stays].mean():.3f} "
      f"(Mann-Whitney P = {stats.mannwhitneyu(pr[flips], pr[stays]).pvalue:.3f})\n")

print("== 4. Redo the WoZ with the toolkit's enhance() on the author's real control predictions ==")
sims = []
for seed in range(1000):
    pt = enhance(img, flip_rate=0.65, seed=seed, label_col="label", pred_col="pred")
    ok = pt == img.label
    sims.append((ok.mean(), ok[img.main8].mean(), ok[img.main8 & img.dark].mean(), ok[img.main8 & ~img.dark].mean()))
sims = np.array(sims)
author = (treat_ok.mean(), treat_ok[img.main8].mean(), treat_ok[img.main8 & img.dark].mean(),
          treat_ok[img.main8 & ~img.dark].mean())
print(f"{'subset':<20}{'author':>8}{'1000 re-runs: mean [2.5%, 97.5%]':>36}{'author percentile':>19}")
for i, name in enumerate(["all images", "8 main", "8 main, FST 5-6", "8 main, FST 1-4"]):
    lo, hi = np.percentile(sims[:, i], [2.5, 97.5])
    pct = (sims[:, i] < author[i]).mean() + 0.5 * (sims[:, i] == author[i]).mean()
    print(f"{name:<20}{author[i]:>8.1%}{sims[:, i].mean():>14.1%}  [{lo:.1%}, {hi:.1%}]{pct:>15.0%}")

img[["image_id", "label", "pred", "fst"]].to_csv(OUT / "woz_input.csv", index=False)
print(f"\nwrote {OUT / 'woz_input.csv'} (columns: image_id, label, pred, fst)")
