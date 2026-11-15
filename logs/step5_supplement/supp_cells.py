# Each string below is one extra cell appended AFTER the author's 120 cells.
CELLS = [
r'''
# ===== SUPPLEMENT S0: helpers (verification only, not part of the author's notebook) =====
import json
MAIN = ['Atopic Dermatitis','CTCL','Dermatomyositis','Lichen Planus','Lyme',
        'Pityriasis Rosea','Pityriasis Rubra Pilaris','Secondary Syphilis']
def cl_ols(y, X, groups):
    return sm.OLS(y, sm.add_constant(X.astype(float)), M=sm.robust.norms.HuberT()).fit(
        cov_type='cluster', cov_kwds={'groups': groups})
AUTHOR = json.load(open("Analysis.ipynb"))["cells"]
def run_author_cell(i):
    exec(''.join(AUTHOR[i]["source"]), globals())
print("dark dtype:", df.dark.dtype, sorted(df.dark.dropna().unique().tolist())[:5])
''',
r'''
# ===== S1: image composition (5.1 table; Methods) =====
img = df.drop_duplicates("image_id").copy()
print("images:", img.shape[0])
print("distinct consensus_label (non-null):", img.consensus_label.nunique(), "| null labels:", img.consensus_label.isna().sum())
print("distinct consensus_label incl. simplified 'Nan' image:")
print(img[img.consensus_label_simplified=="Nan"][["image_id","consensus_label","consensus_label_simplified","fitzpatrick","dark"]])
print("main-8 images:", img.consensus_label_simplified.isin(MAIN).sum())
print("FST 5-6 share, all images:", round(img.dark.astype(float).mean(), 4))
print("FST 5-6 share, 8 main:", round(img[img.consensus_label_simplified.isin(MAIN)].dark.astype(float).mean(), 4))
print(img.fitzpatrick.value_counts().sort_index())
''',
r'''
# ===== S2: Extended Data Fig. 1 - confidence =====
for name, d in [("BCD", bcd), ("PCP", pcp)]:
    nd = d[d.guess_int != 50]
    print(name, "| default-50 share (per observation):", round((d.guess_int == 50).mean(), 4),
          "| per physician (never moved slider):", round(d.groupby("user_id").guess_int.apply(lambda s: (s == 50).all()).mean(), 4))
    print("   r(confidence, top-3) =", round(nd.guess_int.corr(nd.correct_diagnosis), 4),
          "| r(confidence, top-1) =", round(nd.guess_int.corr(nd.diff1), 4), "| n =", len(nd))
''',
r'''
# ===== S3: Fig. 4 - biopsy & referral rates =====
print("BCD biopsy overall:", round(bcd.check_biopsy.mean(), 4), "| PCP biopsy overall:", round(pcp.check_biopsy.mean(), 4))
t = bcd.groupby(["consensus_label_simplified", "dark2"]).check_biopsy.mean().unstack()
t["dark_minus_light_pp"] = (t["FST 5-6"] - t["FST 1-4"]) * 100
print("BCD biopsy rate by condition x skin tone:\n", t.round(3))
print("\nPCP referral (check_dermatologist), main sample, all images:", round(pcp.check_dermatologist.mean(), 4),
      "| NaN count:", pcp.check_dermatologist.isna().sum(), "of", len(pcp))
print("PCP referral, main sample, 8 main conditions:", round(pcp[pcp.consensus_label_simplified.isin(MAIN)].check_dermatologist.mean(), 4))
print("PCP referral by skin tone:\n", pcp.groupby("dark2").check_dermatologist.mean().round(4))
print("PCP referral, mean of per-physician rates:", round(pcp.groupby("user_id").check_dermatologist.mean().mean(), 4))
allpcp = df[df.physician_type == "PCP physician (MD or DO)"]
print("PCP referral, all PCP rows by guess_round:\n", allpcp.groupby("guess_round").check_dermatologist.agg(["mean", "count", "size"]))
print("Referral rate by physician type, main sample round 1:\n",
      df_r1.groupby("physician_type").check_dermatologist.agg(["mean", "count", "size"]))
run_author_cell(20)   # re-runs the author's own regression cell for exact p-values
print("\nauthor cell 20 regressions -> exact p-values")
print("biopsy model:\n", x[0].params.round(4).to_dict(), {k: round(v, 4) for k, v in x[0].pvalues.items()})
print("PCP referral ~ dark:", x[1].params.round(4).to_dict(), {k: round(v, 4) for k, v in x[1].pvalues.items()})
''',
r'''
# ===== S4: wrong-DLS effects (main text) =====
t1 = t1_accuracy_by_correct
for g in ["BCD, Incorrect DLS", "PCP, Incorrect DLS"]:
    s = t1[t1.physician_type == g]
    print(g, "| rows used by author:", len(s), "| before:", round(s.diff1.mean(), 4),
          "| after:", round(s.correctly_update_leading_or_correct.mean(), 4),
          "| author t-test p:", round(stats.ttest_ind(s.diff1, s.correctly_update_leading_or_correct).pvalue, 4))
# The author's data frame stacks two identical copies of every observation (data1 and data2).
for lab, flag in [("BCD", "dermatologist"), ("PCP", "pcp_physician")]:
    u = df_r2[(df_r2.consensus_label_simplified != "Additional") & (df_r2[flag] == 1) & (df_r2.prediction_correct == 0)]
    print(lab, "unique observations:", len(u),
          "| independent t-test on unique rows p =", round(stats.ttest_ind(u.diff1, u.correctly_update_leading_or_correct).pvalue, 4),
          "| paired t-test p =", round(stats.ttest_rel(u.diff1.astype(float), u.correctly_update_leading_or_correct.astype(float)).pvalue, 4))
''',
r'''
# ===== S5: incorrect diagnoses per observation - denominator check =====
n_bcd = (df_r2.dermatologist == 1).sum(); n_pcp = (df_r2.pcp_physician == 1).sum()
print("rows in df_r2: BCD", n_bcd, "| PCP", n_pcp)
for c, lab in [("incorrect_diagnoses_t1", "top-1"), ("incorrect_diagnoses_t2", "top-2"),
               ("incorrect_diagnoses", "top-3"), ("incorrect_diagnoses_dls", "top-4 (with DLS)")]:
    bsum = df_r2[df_r2.dermatologist == 1][c].sum()
    print(f"{lab:16s} BCD as in notebook (sum/PCP n): {bsum/n_pcp:.2f} | BCD with own n: {bsum/n_bcd:.2f} | PCP: {df_r2[df_r2.pcp_physician==1][c].mean():.2f}")
''',
r'''
# ===== S6: specialists without AI vs generalists with AI (top-1, 8 main conditions) =====
b1 = df_r1[(df_r1.dermatologist == 1) & (df_r1.consensus_label_simplified.isin(MAIN))].diff1
for lab, e in [("control DLS", 0), ("treatment DLS", 1)]:
    p2 = df_r2[(df_r2.pcp_physician == 1) & (df_r2.consensus_label_simplified.isin(MAIN)) & (df_r2.x_enhanced_model == e)].correctly_update_leading_or_correct
    print(f"BCD w/o AI {b1.mean():.4f} vs PCP with {lab} {p2.mean():.4f} | diff {100*(b1.mean()-p2.mean()):.2f} pp | t-test p = {stats.ttest_ind(b1, p2).pvalue:.4g}")
''',
r'''
# ===== S7: exact p-values for regression tables =====
def show(res, keys):
    return {k: (round(res.params[k], 4), round(res.pvalues[k], 4)) for k in keys}
run_author_cell(47)
print("Ext. Data Fig. 3 / training (coef, p):")
print("  BCD years :", show(x[0], ["dark", "ten_plus_years_dark"]), "| PCP years:", show(x[1], ["dark", "ten_plus_years_dark"]))
print("  BCD train :", show(x[2], ["training_dark"]), "| PCP train:", show(x[3], ["dark", "trained", "training_dark"]))
print("  BCD nondiv:", show(x[6], ["dark", "nondiverse_patients_dark"]), "| PCP nondiv:", show(x[7], ["dark", "nondiverse_patients", "nondiverse_patients_dark"]))
keys = ["dermatologist", "dark", "guess_round", "e_guess_round", "dermatologist_round", "e_dermatologist_round",
        "dark_round", "e_dark_round", "dark_dermatologist", "dark_dermatologist_round", "e_dark_dermatologist_round"]
run_author_cell(71)
print("\nExt. Data Table 3 (author Table 1) top-1:", show(x[0], keys))
print("Ext. Data Table 3 (author Table 1) top-4:", show(x[1], keys))
run_author_cell(74)
zkeys = ["x_keep_on_top"] + [k for k in x[0].params.index if k.startswith("z_")]
print("\nInterface-interaction model (author Table 2) top-1:", show(x[0], zkeys))
print("Interface-interaction model (author Table 2) top-4:", show(x[1], zkeys))
''',
r'''
# ===== S8: DLS accuracy on the 364 images (level C reference numbers) =====
per_img = df.groupby("image_id")[["model_correct_prediction", "model_enhanced_correct_prediction"]].nunique()
print("max distinct values per image (should be 1):", per_img.max().to_dict())
img = df.drop_duplicates("image_id").copy()
for c in ["model_correct_prediction", "model_enhanced_correct_prediction"]:
    img[c] = img[c].astype(float)
print("control DLS, all images:", round(img.model_correct_prediction.mean(), 4),
      "| treatment DLS, all images:", round(img.model_enhanced_correct_prediction.mean(), 4))
m = img[img.consensus_label_simplified.isin(MAIN)]
print("8 main conditions by FST 5-6 (dark=1) vs FST 1-4:\n",
      m.groupby("dark")[["model_correct_prediction", "model_enhanced_correct_prediction"]].mean().round(4))
'''
]
