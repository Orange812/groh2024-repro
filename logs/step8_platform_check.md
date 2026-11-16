# Step 8 record: replicating the experiment platform

Date: 2026-11-11 · Environment: `groh` (with flask) · Replica code: `groh2024_repro_kit/demo_app/` (under git; baseline commit `412b3e9`)

The authors did not release the code of the original experiment website (https://diagnosing-diagnosis.media.mit.edu). The replica below was rebuilt from the paper's Methods ("Experimental interface", "Randomization protocol"), using **the authors' public data** to restore original details wherever possible. It can demonstrate the full experimental flow and collect a small classroom sample, but it is no substitute for the original experiment (1,118 physicians, 364 restricted images).

## 1. Element-by-element check

Quotations are from the PMC version of the paper (PMC10878981). "Automated check" refers to `step8/simulate_participants.py`, which runs 8 simulated participants through the whole flow and checks the result (all 18 checks PASS; see `step8/protocol_check_output.txt`).

| # | Element of the experiment in the paper (quoted) | Implementation in the replica | Status |
|---|---|---|---|
| 1 | Consent + 7 survey questions ("We will first ask 7 brief survey questions") | Consent text on the home page + 7 questions; the options of the 6 drop-down questions were **changed to the original experiment's options** (recovered from the authors' data; see section 2) | ✅ Match (country: a drop-down in the original, free text in the replica) |
| 2 | Per-participant randomisation over two factors, AI (control/enhanced) × interface (button order), "with an equal weight" | Each participant gets an independent `random.choice` for each factor at the start; lectern mode can fix the arm with URL parameters | ✅ Match |
| 3 | Everyone's first image is the same acne image, used as an attention check; the other images are in random order | Image 1 is always the acne image; the rest are shuffled (automated check: all 8 people saw different orders) | ✅ Match (the original image shows acne on a woman's face; the replica uses a public image of acne on the back) |
| 4 | Up to 3 free-text differential diagnoses, with placeholders 'Type leading / secondary / tertiary differential diagnosis' | Three input boxes with the same placeholder text, word for word | ✅ Match |
| 5 | Predictive text based on **445** diagnoses ("predictive text based on 445 possible diagnoses") | **Changed to the authors' original list**: `possible_options` from cell 89 (In [60]) of the authors' notebook (445 entries, 442 after de-duplication) | ✅ Match (the toolkit as received had only 98) |
| 6 | Confidence slider 0–100% (the Results mention that most people kept the default of 50%) | Slider 0–100, default 50 | ✅ Match |
| 7 | Check boxes for recommending a biopsy and for referral to a dermatologist; "BCDs are not asked whether they would refer to a dermatologist" | Same; the referral option is hidden when BCD is selected (automated check passes) | ✅ Match |
| 8 | Participants are told only that "the AI model is not perfectly accurate"; its accuracy is not disclosed | The instructions and the suggestion page contain only this sentence | ✅ Match |
| 9 | The suggestion page is shown only if the AI suggestion differs from the differential; otherwise the participant goes straight to feedback | Same (automated check passes) | ✅ Match |
| 10 | When the AI predicts "Other", one of 36 auxiliary diseases is chosen at random as the suggestion | Same; the list of 36 auxiliary diseases matches, one to one, the 36 labels outside the 8 main conditions in the authors' data | ✅ Match |
| 11 | Three buttons: 'Update my top prediction with [X]', 'Update my differential to include [X]', 'Keep my differential'; in this order for the control interface, reversed for the treatment interface | Same text, word for word; control top→include→keep, treatment keep→include→top (automated check passes) | ✅ Match |
| 12 | The feedback page shows the reference diagnosis and other participants' most common incorrect answer; fireworks when the answer is right | Same | ✅ Match |
| 13 | After 10 images, a bar chart compares the participant, dermatologists, primary care physicians and the AI, and the participant can keep going | Same; the dermatologist/PCP values are the averages published in the paper (38%, 19%) | ✅ Match |
| 14 | Control AI: VGG-16, 47% accurate on the 364 images; enhanced AI: 65% of the wrong predictions randomly corrected (Wizard of Oz) | Without a trained model, predictions are **simulated**, **now by quota**: exactly 47% of images are predicted correctly and exactly 65% of the wrong predictions are corrected (48% / 83% on the current 29 images). `dls/predict.py --update-demo` writes in a real model's predictions | ⚠️ Simulated (reproducing the model is step 7) |
| 15 | 364 experiment images (Zenodo, restricted access) | 30 **public Fitzpatrick17k images** (from atlasdermatologico.com.br): 1 acne attention-check image + 23 images of 7 main conditions + 1 each of 6 auxiliary diseases. The image files are not stored in the repository (their copyright most likely stays with the source atlas); `demo_app/data/images.csv` lists them and `dls/fetch_f17k_images.py --mode list` downloads exactly these files and checks their MD5 | ⚠️ Substitute (see the limitations in section 3) |
| 16 | The website was built with Python + Flask | The replica also uses Flask | ✅ Match |
| 17 | The authors graded every free-text answer by hand (2,348 distinct strings) | The replica grades automatically (synonyms + typo tolerance). Validated on the authors' data: agreement with the manual grading is **98.9%** on the 8 main conditions (9,368 answers) and 91.0% on other diseases (`step8/grader_agreement_output.txt`) | ⚠️ Approximation, validated |

## 2. Changes made to the replica (`git log --oneline 412b3e9..447395a`)

| Commit | Change | Basis |
|---|---|---|
| `4c84645` | Predictive text switched to the authors' 445 diagnoses | Paper's Methods; cell 89 (In [60]) of the authors' notebook |
| `bca9b38` | Survey options switched to the original experiment's options; the analysis script recognises "Mostly/All white patients" | Actual values of `years_experience`, `patient_distribution`, `trouble_*_patients`, `suffient_training` in the authors' data |
| `16585a8` | When downloading public images, labels are checked against the URL and mislabelled images dropped | F17k label noise (section 3) |
| `08a2ce8` | Dead image hosts are skipped and each class is filled image by image; demo image set downloaded | Dead dermaamin.com links (section 3) |
| `c43ba14` | AI accuracy simulated by quota instead of an independent coin flip per image | With few images, independent draws are badly biased: only 38% / 66% on 29 images |
| `1fe44de` | "Other disease" images are drawn only from the paper's 36 auxiliary diseases | The paper's 80 non-main-condition images are exactly these 36 |

To see all changes: `git diff 412b3e9 447395a --stat`.

## 3. Limitations and findings (for the report)

1. **Dead links in the public data**: as of 2026-11-11, dermaamin.com has been redesigned, and the old F17k image URLs first redirect (301) and then return 404. Of the 16,577 F17k images, **12,631 (77%) can no longer be downloaded**; only the 3,905 from atlasdermatologico.com.br remain available. A dataset published only as URLs decays over time, which is itself a reproducibility problem.
2. **F17k label noise**: dermaamin URLs contain the source atlas's diagnosis folder. Checking by folder shows that 40 of the 128 images labelled "Lyme disease" come from the erythrasma (erytrasma) folder; 22 of the "dermatomyositis" images come from dermatophytosis (dermatophytie); and "lichen planus" includes images from folders such as factitious dermatitis, trauma and solar dermatitis. F17k's own quality-control column `qc` covers only 3% of the images and cannot catch these problems. This also affects training the AI on F17k in step 7.
3. **Incomplete coverage of the demo image set**: secondary syphilis has no matching label in F17k; only 2 atopic dermatitis images and 1 Lyme disease image can be downloaded; there are no dark-skin (FST 5–6) dermatomyositis images. Dark-skin images make up 40% overall (32% in the original experiment).
4. **An internal inconsistency in the paper (confirmed by the data)**: the paper gives the number of auxiliary diseases as 36 in one place (the sentence on the "Other" prediction rule) and 38 in another (the image-curation section and the Fig. 2 legend). The authors' data have exactly 36 labels outside the 8 main conditions, so 36 is correct, and "46 diseases" should also be 44 (consistent with the step 5 verification table).
5. **The AI is simulated**: the replica's AI suggestions are simulated at the accuracies set in the paper; they are not the output of a real model.

## 4. How to run it (screen recording / class)

```bash
conda activate groh
cd groh2024_repro_kit                     # from the repository root
python dls/fetch_f17k_images.py --csv ../data/fitzpatrick17k.csv --mode list   # once: download the 30 demo images
python demo_app/app.py --port 5050        # should print: Loaded 30 images (30 real images, …)
```

- Control interface: http://127.0.0.1:5050/?dls=control&ui=control
- Treatment interface (buttons in reverse order): http://127.0.0.1:5050/?dls=control&ui=treatment
- Live class summary: http://127.0.0.1:5050/dashboard
- Before class, click "Start new session" on `/dashboard` to clear old records (they are renamed as a backup, not deleted).

**Classroom trial run (optional):**
1. Start with `python demo_app/app.py --lan --port 5050` instead. On first run macOS shows a firewall prompt; choose "Allow".
2. Find the computer's IP with `ipconfig getifaddr en0`; classmates open `http://<IP>:5050` on their phones. Campus networks may block device-to-device traffic, so test in advance.
3. Before starting, explain that participation is anonymous, voluntary and only for the class demo; if the course has ethics requirements for collecting classmates' data, ask the instructor first.

**Export and analyse:** click Download data (CSV) on `/dashboard`, then run:

```bash
python analysis/key_results.py --data ~/Downloads/classroom_responses.csv --min-trials 1 --keep-failed-attention --out ../logs/classroom
```

Non-clinician classmates mostly won't recognise the acne image, hence `--keep-failed-attention`. The sample is tiny, so the results only demonstrate the pipeline and cannot test the paper's conclusions. This pipeline has been tested end to end on simulated data; all 8 tables and 3 figures are produced.

**Do not** import the 364 Zenodo experiment images into the replica for a screen recording or slides. The access terms allow only research and medical use and forbid redistribution.
