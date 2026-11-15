# Steps 4–5 record: running the official notebook and checking the paper's results item by item

Run date: 2026-11-11 · Environment: `groh-orig` (conda-forge Python 3.9.23 + `pip install -r requirements.txt`), macOS 15.7.5 arm64

## 1. Step 4 conclusions

| Item | Result |
|---|---|
| Do the authors' dependencies install as is? | **Yes.** `requirements.txt` installed in one go in `groh-orig` (`logs/groh-orig_pip_install.log`); the compatible fallback environment `groh` was not needed |
| Key package versions | pandas 2.1.0, numpy 1.26.0, scipy 1.11.2, matplotlib 3.8.0, seaborn 0.12.2, statsmodels 0.14.0, stargazer 0.0.6, scikit-learn 1.3.0 (exactly as in `requirements.txt`; see `pip_freeze.txt`) |
| Changes to the authors' code | **Only 1**: data path `../Data/Experimental Data.csv` → `../Data/Experimental Data.csv.zip` (commit `395b201`). Not one line of analysis code was changed |
| Restart & Run All | All 120 cells ran, **0 errors** (about 75 seconds) |
| Comparison with the authors' saved outputs | The 80 code cells with output are **identical character for character** (only the `Date/Time` run timestamps in the statsmodels summaries differ; `step5_supplement/compare_outputs_result.txt`) |
| Output files | `analysis_run.html`, `analysis_run.ipynb`; 26 figures in `work/ResearchBox 1802/Code/figs/` |
| Environment records | `environment.yml`, `pip_freeze.txt`, `python_version.txt`, `system_info.txt` |

**Pitfall found (worth adding to the reproduction guide's troubleshooting list):** if you run `…/envs/groh-orig/bin/python -m jupyter nbconvert` without first running `conda activate groh-orig`, `jupyter` resolves through PATH to the **base environment's** nbconvert and kernel. Cell 0 then fails with `ModuleNotFoundError: No module named 'stargazer'`, followed by 30 cascading `NameError`s (`os`, `re`, `palette2`, `p1c`, `Stargazer`…). Activating the environment first fixes it; no code change is needed.

## 2. Step 5 verdict summary

Criteria (from the reproduction guide): **match** = identical after rounding to the paper's precision; **close** = differs by ≤ 1 percentage point, or the P value differs but the significance conclusion is the same; **mismatch** = anything else. "Reproduced" values come from the notebook's output; numbers the notebook does not print were computed separately (marked †; method in section 5).

| Table | Match | Close | Mismatch | Manual check pending |
|---|---|---|---|---|
| 5.1 Sample sizes (7 items) | 6 | 0 | 1 (number of diseases 46 vs 44) | 0 |
| 5.2 Figures and text (25 items) | 19 | 3 | 2 (partial) | 1 (confusion matrices) |

**Level A conclusion:** computationally, the reproduction is **complete**: the same data and code give output identical to the authors' saved output. Against the paper's text, the vast majority of numbers match, but 2 numbers in the paper disagree with the code output (the PCP referral rate, and the scope of the enhanced AI's accuracy). I also found 1 denominator error in the code that made it into the paper's main text. Details in section 4.

## 3. Verification tables

### 5.1 Sample sizes

| Check | Paper | Reproduced | Status |
|---|---|---|---|
| Total participants and diagnoses | 1,118 participants, 14,261 differential diagnoses (Fig. 1 says N = 1,120) | 1,118 participants, 14,261 diagnoses, 364 images (cell 29) | ✅ Match (Fig. 1's 1,120 does not match the data) |
| Per physician type: participants / diagnoses | BCD 389/5,365; residents 116/1,691; PCP 459/5,458; other 154/1,747 | Identical (cell 29) | ✅ Match |
| Attention-check pass rate | BCD, PCP, other 98%; residents 96% | 98.2%, 98.0%, 98.1%; 95.7% | ✅ Match |
| Main-analysis sample | BCD 296/2,660; residents 83/747; PCP 350/3,150; other 113/1,015 | Identical (cell 29) | ✅ Match |
| Participants per randomised arm | BCD 75/64/83/74; PCP 79/85/87/99 | Identical (cell 96) | ✅ Match |
| Image set | 364 images, 46 diseases; 284 images of the 8 main conditions (31/48/34/30/32/33/47/29); FST 5–6 = 32% | 364 images; **44** `consensus_label` values, plus 1 image with a missing label (file name says rosacea, which is among the 44); counts for the 8 conditions identical; FST 5–6 = 31.9% (cell 92, †S1) | ⚠️ Partial mismatch: 44 ≠ 46 diseases |
| Observations on the 8 main conditions (Fig. 5) | BCD 2,079; PCP 2,496 | 2,079; 2,496 (cell 61) | ✅ Match |

### 5.2 Figures and numbers in the text

| # | Location in paper | Paper | Reproduced (source) | Status |
|---|---|---|---|---|
| 1 | Text: accuracy on all images | top-3 38/36/19/18%; top-1 27/24/14/13% | top-3 38.4/35.6/18.7/17.6; top-1 27.0/24.4/14.1/12.6 (cells 32, 34) | ✅ Match |
| 2 | Fig. 2a: 8 conditions | top-1 27/24/13/12%; top-3 37/35/17/16% | top-1 27.3/24.2/13.2/12.2; top-3 37.2/34.7/16.9/16.0 (cells 33, 35) | ✅ Match |
| 3 | Fig. 2b: BCD vs PCP (top-3) | 7 conditions + Other significant (P<0.001), 1 not significant | Lyme P = 0.598; the other 8 groups P ≤ 3.1×10⁻⁵ (cell 10) | ✅ Match |
| 4 | Ext. Data Tables 1–2: confusion matrices | Cell by cell | The notebook output is identical to the authors' saved output. The paper's tables are images I cannot read, so the matrices are attached in section 6 for a cell-by-cell check against the PDF | ⏳ Manual check pending |
| 5 | Ext. Data Fig. 1: confidence | 71% of BCDs, 70% of PCPs left the default 50%; r = 0.14–0.17 | 70.7%, 69.8% (cell 82); r: BCD top-1 0.154 / top-3 0.171, PCP 0.141 / 0.150 (†S2) | ✅ Match (note: 71%/70% are per observation; per physician they are 42%/47%) |
| 6 | Text: collective judgement | 48%, 33%; 77%, 58%; 98% | 48.2%, 33.1%; 77.5%, 58.3%; 97.8% (cell 36) | ✅ Match |
| 7 | Text + Fig. 3a,b: skin-tone gap | All −4 (<0.001) / −4 (0.001); BCD −5 (0.011) / −3 (0.117); residents −5 (0.114) / −5 (0.113); PCP −3 (0.006) / −4 (0.008); other −5 (0.012) / −4 (0.092) | −4.1 (<0.001) / −3.6 (0.001); −4.7 (0.011) / −3.1 (0.117); −5.0 (0.114) / −5.2 (0.113); −3.4 (0.006) / −3.8 (0.008); −4.8 (0.012) / −4.0 (0.092) (cells 39–41) | ✅ Match |
| 8 | Fig. 3c,d + Ext. Data Fig. 2 | BCD top-1: atopic dermatitis −18 (0.007), Lyme −20 (<0.001), pityriasis rosea −19 (0.001), CTCL −10 (0.009); lichen planus more accurate on dark skin; PCP significant for three conditions | −17.9 (0.007), −19.8 (<0.001), −18.8 (0.001), −9.7 (0.009); lichen planus +6.7 (0.286); PCP three conditions top-1 all <0.001, top-3 0.002/<0.001/<0.001 (cells 43–45) | ✅ Match |
| 9 | Ext. Data Fig. 3: patient mix | PCPs (mostly/all white patients) 7 pp lower top-3 on dark skin (0.009); BCD not significant | −7.5 pp, **P = 0.0079**; BCD +4.8 (0.228) (cell 47, †S7) | 🟡 Close (P 0.008 vs 0.009). The Ext. Data Fig. 3 legend says top-1, but the text and the code use top-3 |
| 10 | Text: training and experience | PCPs with adequate training have a 5 pp smaller gap (0.079); years of practice have no effect | +5.05 pp, **P = 0.0799**; years P = 0.233 (BCD) / 0.753 (PCP) (cell 49, †S7) | 🟡 Close (0.0799 rounds to 0.080) |
| 11 | Fig. 4a: BCD biopsy | 28%; more on dark skin: atopic dermatitis (0.008), pityriasis rosea (0.015); fewer: pityriasis rubra pilaris (0.033), CTCL (0.001) | 28.0%; +13.4 pp (0.008), +15.2 (0.015); −11.2 (0.033), −19.7 (0.001) (cells 19, 20, †S3) | ✅ Match |
| 12 | Fig. 4b,c: PCP biopsy and referral | Biopsy 7%, only CTCL significant (0.011); referral **28%**, 4.4 pp more on dark skin (0.012) | Biopsy 7.5%, only CTCL (0.011) ✅; 4.42 pp more on dark skin (P = 0.0117) ✅; **referral rate 35.6%** (light 34.0%, dark 38.4%) (cell 20, †S3) | ❌ Partial mismatch: the overall 28% referral rate cannot be reproduced (section 4, F2) |
| 13 | Fig. 5a: top-1 before/after AI | BCD 27→36%, PCP 13→22% (<0.001) | 27.3→36.4 (P = 2.6×10⁻⁷); 13.2→21.8 (P = 3.6×10⁻¹¹) (cell 51) | ✅ Match |
| 14 | Fig. 5b: top-3→top-4 | BCD 37→60%, PCP 17→47%; 59%, 46% after replacing the third diagnosis | 37.2→59.9; 16.9→46.6; 59.4, 46.3 (cells 52, 55) | ✅ Match |
| 15 | Ext. Data Fig. 4: CTCL sensitivity | 18→26% (0.039)→31% (0.001); specificity 99% | 18.2→26.3 (0.039)→30.9 (0.0013); 98.7/98.9/98.6% (cells 57, 58) | ✅ Match |
| 16 | Text: uptake rate | 236 images; BCD 40 vs 47% (0.009); PCP 54 vs 61% (0.001) | 236; 40.3 vs 46.5 (0.0087); 54.3 vs 61.3 (0.0014) (cells 64–66) | ✅ Match |
| 17 | Text: wrong AI suggestions | top-1 drops 1.2 pp (0.517, 0.312); correct top diagnosis replaced <2%; made top diagnosis when all three are wrong: BCD 10%, PCP 14% | 28.7→27.5, 13.2→11.9 (0.517, 0.312†); 1.8%, 1.7%; 10.1%, 14.3% (cell 59, †S4) | ✅ Numbers match, but the wording does not match what the code computes, and the test is questionable (section 4, F3, F4) |
| 18 | Text: incorrect diagnoses per diagnosis | BCD 1.58 / 1.40 / 1.05 / 0.59; PCP 1.72 / 1.55 / 1.26 / 0.82 | Identical (cell 59) | ✅ Numbers match, but the BCD denominator is wrong (section 4, F1) |
| 19 | Text: specialist vs PCP + AI | 5 pp higher than PCPs with the control AI (<0.001); 4 pp lower than PCPs with the enhanced AI (0.022) | +5.5 pp (P = 0.00046); −3.6 pp (P = 0.0223) (means from cell 51; P values from a †S6 t-test, not printed by the notebook) | ✅ Match |
| 20 | Ext. Data Table 3: regression | BCD +13 (<0.001); dark −3 (0.006); AI +8 (<0.001); enhanced AI +8 (0.002); AI×dark −5 (0.008 / 0.048); BCD×dark×AI +4 (0.227) / +8 (0.034) | +13.4 (<0.001); −3.4 (0.006); +8.0 (<0.001); +8.3 (0.002); −5.2 (0.0079) / −4.9 (0.0476); +3.8 (0.227) / +7.7 (0.0339) (cells 71, 72, †S7) | ✅ Match |
| 21 | Fig. 6: before/after AI × skin tone | With AI, PCPs still show a significant skin-tone gap; BCDs do not | top-1: PCP −7*** (control −10***, enhanced −5*), BCD not significant in any of the three columns; top-4 not significant for either group (cells 61, 62) | ✅ Match (the conclusion refers to top-1) |
| 22 | Ext. Data Tables 4–5: interface | No significant interaction effects; +9 / −9 pp (<0.001); BCD −12 (<0.001), PCP −7 (0.011) | +9 / −9 (<0.001) ✅; −12.7 (<0.001), −6.9 (0.011) ✅; in the interaction model 2 top-1 terms have P < 0.05: interface×AI −7.2 pp (0.020), interface×AI×dark +8.4 pp (0.033); none for top-4 (cells 67, 74, 79, †S7) | 🟡 Close (section 4, F6) |
| 23 | Ext. Data Table 6: image sources | 15% from textbooks (11% of all 284 images) | 14.9%; 10.8%, with a denominator of the **277** images BCDs actually saw, not 284 (cell 93) | ✅ Match |
| 24 | Suppl. Tables 1–4: robustness | Conclusions unchanged; US BCDs 42%, PCPs 19% | Under 4 sample definitions the coefficients keep their direction and size; US 41.98%, 19.30% (cells 68, 109–112) | ✅ Match |
| 25 | Methods: AI model (reference values for level C) | Control AI 47%; on the 8 conditions 58% dark, 56% light; enhanced AI 84% (82% dark, 84% light) | Control 46.8%; 57.7%, 55.8% ✅. Enhanced AI: **81.6%** on the 364 images, 84.2% on the 8 conditions; **83.3%** dark, 84.5% light (†S8) | ❌ Partial mismatch (section 4, F5). This is a check on the data; retraining the model is step 7 |

## 4. Problems found (for section 6 of the report)

**F1 Code bug: the denominator of the BCD "incorrect diagnoses per diagnosis" is the PCP observation count** (cell 59; affects item 18)
The authors' code is `df_r2[df_r2.dermatologist==1][…].sum() / df_r2[df_r2.pcp_physician==1].shape[0]`. The denominator is the PCPs' 3,150, while the BCDs' own observation count is 2,659. The BCD values reported in the paper are therefore systematically 16% too low:

| | top-1 | top-2 | top-3 | top-4 (with AI) |
|---|---|---|---|---|
| Paper / notebook (wrong denominator) | 0.59 | 1.05 | 1.40 | 1.58 |
| Corrected (BCDs' own observation count) | **0.69** | **1.25** | **1.66** | **1.87** |
| PCP (for comparison) | 0.82 | 1.26 | 1.55 | 1.72 |

After the correction, BCDs make **more** incorrect diagnoses per diagnosis than PCPs at top-3 and top-4. The paper uses these numbers to describe the cost of AI assistance, so that comparison changes.

**F2 The overall PCP referral-to-dermatology rate of 28% cannot be reproduced** (item 12)
The notebook's main sample gives 35.6%. I tried 8 sample definitions: US only 35.5%, no filtering 31.5%, including images after the 10th 32.3%, only images after the 10th 25.6%, control-AI arm 40.3%, enhanced-AI arm 31.4%. None gives 28% (`step5_supplement/extra_checks_output.txt`). The difference reported in the same paragraph ("4.4 pp more on dark skin, P = 0.012") reproduces exactly. The BCD biopsy rate in the same paragraph is also 28%, so the paper's 28% may be a typo.

**F3 The "all three diagnoses wrong" share: the wording does not match the code** (item 17)
The paper says that "when the AI suggestion was wrong and all three of the physician's differential diagnoses were wrong, 10%/14% of BCDs/PCPs made the wrong suggestion their top diagnosis". In the code, 10.1%/14.3% come from the variable `incorrectly_update_leading_when_incorrect`, whose condition is only "top diagnosis wrong", and whose denominator is every observation with a wrong AI suggestion, so it is a joint share. Computed for "all three wrong", the joint share is 9.4%/13.7% and the conditional share is **14.8%/17.2%**.

**F4 How "wrong AI suggestions lower top-1 by 1.2 pp (P = 0.517, 0.312)" is tested** (item 17)
The code first duplicates every observation (`pd.concat([data1, data2])`; for BCDs it uses 1,156 rows from only 578 observations), then runs an independent-samples t-test on the "before/after" of the same observations. On the 578/720 distinct observations: independent t-test P = 0.647/0.475; paired t-test (more appropriate, since before and after are the same observation) P = **0.008/0.003**. The drop (1.2 pp) is indeed small, but whether it is "not significant" depends on the choice of test.

**F5 The enhanced AI's "84%" is actually its accuracy on the 8 main conditions** (item 25)
In the data, the enhanced AI is 81.6% accurate on all 364 images and 84.2% on the 8 main conditions. That agrees with the estimate in the reproduction guide: 47% + 53% × 65% ≈ 81%. So "resulted in a top-1 accuracy of 84%" in the Methods refers to the 8 conditions. Also, on the 8 conditions the enhanced AI scores 83.3% on dark skin, where the paper says 82%, a 1.3 pp difference.

**F6 "No significant interface interaction effects" holds fully only for top-4** (item 22)
In the model with interface interactions (author Table 2, corresponding to Ext. Data Table 4), two top-1 interaction terms are significant at P < 0.05: interface×AI −7.2 pp (P = 0.020) and interface×AI×dark +8.4 pp (P = 0.033). The model has 24 interface-related coefficients, so 2 with P < 0.05 without a multiple-comparison correction is not surprising, but the paper's statement "we do not find significant interaction effects" needs this qualification.

**F7 Other small points**
- Number of diseases: the data have 44 `consensus_label` values, the paper says 46; 1 image (`refinery29-rosacea-dark-skin.jpg`) has a missing label.
- The Fig. 6 legend says 3,150 PCP observations, which is the count for all images. The Fig. 6 code (cell 24) uses only the 8 conditions, so it should be 2,496; the legend is wrong.
- Ext. Data Fig. 1 says "71% of BCDs", but the code computes it per observation (70.7%). Per physician, only 42% (BCD) and 47% (PCP) never moved the slider.
- The 11% in Ext. Data Table 6 has a denominator of 277 images, not 284.
- The paper's Software section lists pandas 1.4.0, seaborn 0.11.1, numpy 1.18.5, etc. The authors' `requirements.txt` has pandas 2.1.0, seaborn 0.12.2, numpy 1.26.0, and the notebook already comments out `.append` in favour of `pd.concat` (cell 22). So the authors' last run used pandas ≥ 2.0, and the version information in the Software section is out of date.
- The paper's Fig. 1 says N = 1,120; the data have 1,118.

## 5. How the supplementary numbers (marked †) were computed

Some numbers are not printed by the authors' notebook, for example expressions in the middle of a cell, values that are only drawn in a figure, or P values shown only as stars in a stargazer table. I made a copy of the notebook in a scratch directory and appended 9 calculation cells (S0–S8) **after** the authors' 120 cells. Exact P values were obtained by re-running the authors' cells (20, 47, 71, 74) unchanged via `exec`, without rewriting any regression specification. The authors' files were not modified.

- `step5_supplement/supp_cells.py`: source of the 9 appended cells
- `step5_supplement/supplement_executed.ipynb`: the full executed notebook (the authors' 120 cells still run with 0 errors; the supplementary outputs are at the end)
- `step5_supplement/extra_checks.py` + `extra_checks_output.txt`: referral-rate definitions, the "all three wrong" share, AI accuracy (reads only the raw columns; runs on its own)
- `step5_supplement/compare_outputs.py` + `compare_outputs_result.txt`: cell-by-cell comparison of the re-run output with the authors' original output

## 6. Appendix: confusion matrices (for the manual check of item 4)

The notebook prints `confusion_matrix(reference label, top diagnosis).T`: **rows = the physician's top diagnosis, columns = the reference label**. Labels are in alphabetical order; Other = "Additional" in the notebook.

Column order: Other, Atopic Dermatitis, CTCL, Dermatomyositis, Lichen Planus, Lyme, Pityriasis Rosea, PRP, Secondary Syphilis

**BCD (cell 101)**

| Top diagnosis ↓ / reference → | Other | AD | CTCL | DM | LP | Lyme | PR | PRP | SS |
|---|---|---|---|---|---|---|---|---|---|
| Other | 506 | 97 | 194 | 112 | 108 | 132 | 117 | 237 | 136 |
| Atopic Dermatitis | 27 | 133 | 41 | 28 | 13 | 8 | 15 | 37 | 10 |
| CTCL | 5 | 4 | 59 | 2 | 3 | 4 | 5 | 5 | 2 |
| Dermatomyositis | 3 | 2 | 0 | 70 | 1 | 0 | 0 | 8 | 0 |
| Lichen Planus | 9 | 9 | 6 | 4 | 122 | 0 | 11 | 39 | 12 |
| Lyme | 3 | 0 | 1 | 0 | 0 | 42 | 0 | 0 | 0 |
| Pityriasis Rosea | 0 | 0 | 1 | 0 | 1 | 1 | 54 | 0 | 8 |
| PRP | 2 | 1 | 3 | 1 | 2 | 2 | 2 | 36 | 2 |
| Secondary Syphilis | 3 | 1 | 3 | 1 | 4 | 0 | 4 | 2 | 52 |

**PCP (cell 104)**

| Top diagnosis ↓ / reference → | Other | AD | CTCL | DM | LP | Lyme | PR | PRP | SS |
|---|---|---|---|---|---|---|---|---|---|
| Other | 561 | 161 | 323 | 190 | 203 | 143 | 200 | 276 | 174 |
| Atopic Dermatitis | 34 | 104 | 37 | 52 | 28 | 20 | 17 | 47 | 13 |
| CTCL | 5 | 8 | 15 | 4 | 9 | 2 | 2 | 6 | 1 |
| Dermatomyositis | 4 | 4 | 2 | 20 | 2 | 2 | 1 | 3 | 0 |
| Lichen Planus | 14 | 6 | 3 | 3 | 42 | 0 | 15 | 24 | 10 |
| Lyme | 4 | 0 | 0 | 0 | 0 | 56 | 0 | 0 | 2 |
| Pityriasis Rosea | 2 | 2 | 10 | 2 | 6 | 3 | 51 | 8 | 15 |
| PRP | 0 | 1 | 4 | 1 | 1 | 0 | 3 | 13 | 0 |
| Secondary Syphilis | 5 | 1 | 2 | 1 | 3 | 1 | 3 | 4 | 29 |

If the paper's tables are oriented the other way (rows = reference label), transpose the tables above before comparing.
