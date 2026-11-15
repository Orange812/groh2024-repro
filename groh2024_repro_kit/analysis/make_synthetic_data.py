"""
Generate "simulated experiment data" — only for rehearsals and testing the code; never cite it as the paper's results!

Why it is needed:
  - the real data (Experimental Data.csv from ResearchBox #1802) has to be downloaded on your own computer;
  - before you have the real data, this simulated data lets you run the analysis scripts, figures and live demo end to end;
  - the simulation parameters are roughly calibrated to the paper's summary numbers (sample sizes, accuracy, skin-tone gaps, AI gains, button-order effect),
    so the results "look like" the paper, but every number is randomly generated.

Output: data/synthetic_experiment.csv (same fields as the demo_app log, i.e. this toolkit's "standard format")

Usage:
  python analysis/make_synthetic_data.py              # default random seed 2024
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
               "Secondary syphilis": 29}                       # paper, Methods: 284 images in total
N_AUX_IMAGES = 79                                               # plus 79 images of 38 other diseases (Fitzpatrick 17k)

# Participants in the main analysis (passed the attention check and completed ≥10 images) and total recruited (paper, Results)
ANALYZED = {"BCD": 296, "Resident": 83, "PCP": 350, "Other": 113}
RECRUITED = {"BCD": 389, "Resident": 116, "PCP": 459, "Other": 154}

# Target top-1 / top-3 accuracy (8 main conditions) and dark-skin gap (percentage points / 100)
TOP1 = {"BCD": 0.27, "Resident": 0.24, "PCP": 0.13, "Other": 0.12}
TOP3 = {"BCD": 0.37, "Resident": 0.35, "PCP": 0.17, "Other": 0.16}
GAP1 = {"BCD": -0.05, "Resident": -0.05, "PCP": -0.03, "Other": -0.05}
GAP3 = {"BCD": -0.03, "Resident": -0.05, "PCP": -0.04, "Other": -0.04}
DARK_SHARE = 0.32

# Disease difficulty (relative to the mean) and per-disease skin-tone gap adjustment for BCDs (paper: significantly worse on dark skin for AD/Lyme/PR/CTCL, the reverse for LP)
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
    rows.insert(0, {"disease": "Acne", "fst": 5, "attention_check": 1})     # attention-check image (acne)
    img = pd.DataFrame(rows)
    img["image_id"] = [f"img_{i:03d}" for i in range(len(img))]

    # A fixed AI prediction per image (as in the paper: everyone gets the same suggestion for the same image)
    classes = list(MAIN8) + ["Other"]
    ctrl, treat, sugg_c, sugg_t = [], [], [], []
    for _, r in img.iterrows():
        true9 = r.disease if r.disease in MAIN8 else "Other"
        p_ok = (0.57 if r.fst >= 5 else 0.55) if true9 != "Other" else 0.25
        pred = true9 if rng.random() < p_ok else rng.choice([c for c in classes if c != true9])
        pred_t = pred if pred == true9 or rng.random() >= 0.65 else true9   # Wizard of Oz: 65% of wrong predictions flipped to correct
        ctrl.append(pred)
        treat.append(pred_t)
        aux_pick = str(rng.choice(AUXILIARY))                              # when predicting Other, suggest a random auxiliary disease
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

            # Text diagnoses (so that the "grade by text" path is tested too)
            wrong = [d for d in list(MAIN8) + AUXILIARY[:12] if d != im.disease]
            dx = list(rng.choice(wrong, 3, replace=False))
            if top1:
                dx[0] = im.disease
            elif top3:
                dx[int(rng.integers(1, 3))] = im.disease

            # AI suggestion and choice
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
                        accept -= 0.14                    # make the PCP skin-tone gap widen with AI assistance
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

            # Biopsy / referral / confidence
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
    ap = argparse.ArgumentParser(description="Generate simulated experiment data (rehearsal/testing only)")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--out", default=str(ROOT / "data" / "synthetic_experiment.csv"))
    a = ap.parse_args()
    df = simulate(a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"[simulated data] wrote {a.out}: {df.participant_id.nunique()} virtual physicians, {len(df)} diagnosis records")
    print("Reminder: this data is randomly generated. Use it only for rehearsal, never as the paper's results.")


if __name__ == "__main__":
    main()
