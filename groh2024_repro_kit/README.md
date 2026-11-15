# Groh et al. 2024 reproduction and demo toolkit

Paper: Groh M, Badri O, Daneshjou R, et al. *Deep learning-aided decision support for diagnosis of skin disease across skin tones.* **Nature Medicine** 30, 573–583 (2024). doi:10.1038/s41591-023-02728-3

> This is **independent reproduction code** written for a course presentation. It is not the official code released by the paper's authors. The official code and data are listed under "Authors' public resources" below.

## 1. Authors' public resources (the "source code + datasets" to introduce in the presentation)

| Resource | Location | Contents | How to get it |
|---|---|---|---|
| Experiment data + analysis code + AI training code | ResearchBox #1802: https://researchbox.org/1802 (Zenodo backup doi:10.5281/zenodo.15039588) | `Experimental Data.csv.zip` (1.2 MB, every physician diagnosis), `Analysis/Analysis.ipynb` (reproduces the paper's figures and tables), `Analysis/requirements.txt`, `Control DLS Training Code.py` (control-AI training code) | Click *entire box* on the web page to download, CC BY 4.0 |
| The 364 experiment images | Zenodo: doi:10.5281/zenodo.10070478 | Images of 46 skin diseases | Restricted access: register on Zenodo and agree to "research/medical use only, delete after use"; **apply a few days in advance** |
| Public dataset used to train the AI | https://github.com/mattgroh/fitzpatrick17k | 16,577 clinical images + Fitzpatrick skin-type labels (`fitzpatrick17k.csv`), `train.py` | The CSV downloads directly; images download from the url column or by request form, CC BY-NC-SA 3.0 |
| Original experiment website | https://diagnosing-diagnosis.media.mit.edu | The paper's online experiment platform | Check before the presentation whether it is still online |

## 2. What this toolkit contains

```
groh2024_repro_kit/
├─ 00_check_env.py            check the environment
├─ run_demo.bat / run_demo.sh one click: simulated data → analysis → start the demo website
├─ requirements.txt           analysis + website dependencies (pandas numpy scipy matplotlib flask)
├─ requirements-dls.txt       AI training dependencies (torch torchvision; Colab recommended)
├─ common/diseases.py         46 disease names, synonyms, free-text grading (typo-tolerant)
├─ analysis/
│   ├─ inspect_data.py        step 1: inspect the authors' column names and generate a column mapping, mapping.json
│   ├─ key_results.py         independent reproduction of the core results (8 tables + 3 figures, next to the paper's values)
│   ├─ stats_utils.py         OLS + participant-clustered robust SEs (same method as the paper, no statsmodels needed)
│   └─ make_synthetic_data.py generate simulated data (rehearsal/testing only, never as the paper's results)
├─ dls/
│   ├─ train_control_dls.py   VGG-16 9-class control AI (rewritten from the paper's Methods)
│   ├─ wizard_of_oz.py        enhanced AI: randomly correct 65% of the wrong predictions (47% → about 84%)
│   ├─ predict.py             inference on images / write AI suggestions for the demo website in bulk
│   ├─ fetch_f17k_images.py   download Fitzpatrick17k images (demo mode / training mode)
│   └─ colab_train_demo.ipynb train a demo model on a Colab GPU in 30–60 minutes
└─ demo_app/                  replica of the paper's experiment platform (Flask)
    ├─ app.py                 survey → diagnose images → AI suggestion (random button order) → feedback → comparison → live summary
    ├─ import_images.py       import images you are allowed to use (e.g. the Zenodo experiment images)
    └─ data/images.csv        image list (30 placeholder images by default; see below to use real images)
```

## 3. Five-minute quick start (no downloads needed)

```bash
pip install -r requirements.txt
python 00_check_env.py
python analysis/make_synthetic_data.py                     # generate simulated data
python analysis/key_results.py --data data/synthetic_experiment.csv
python demo_app/app.py                                     # open http://127.0.0.1:5000 in a browser
```
On Windows you can also double-click `run_demo.bat`. Every output from simulated data is marked "SIMULATED DATA".

## 4. Reproducing with the authors' real data

**Route A: run the authors' official notebook (the most authoritative; show this first in the presentation)**
```bash
# after unzipping ResearchBox_1802.zip
cd Analysis
pip install -r requirements.txt     # the authors' pinned versions are old; if installation fails, use the compatible environment below
python -m notebook                  # open Analysis.ipynb and Run All
```
Compatible environment (a recommended combination of versions, not yet tested when this toolkit was written; adjust versions if errors appear):
```bash
conda create -n groh python=3.9 pandas=1.4 numpy=1.21 scipy=1.7 matplotlib=3.5 seaborn=0.11 statsmodels=0.13 notebook -c conda-forge
conda activate groh && pip install stargazer
```
If the data path in the notebook's first cell differs from where you unzipped the files, change it to the actual path of `Experimental Data.csv.zip`.

**Route B: independent reproduction with this toolkit (shows "reproduced vs paper")**
```bash
python analysis/inspect_data.py "Experimental Data.csv.zip" --write-map mapping.json
# open mapping.json and check the column names (if unsure, search Analysis.ipynb for the column name to see how the authors use it)
python analysis/key_results.py --data "Experimental Data.csv.zip" --map mapping.json --out outputs/real
```
Output: `outputs/real/results_summary.md` (all tables), `table1.csv`…, `fig1/2/3*.png`.
Note: the regression specifications are inferred from the paper's text, so some coefficients may be defined slightly differently from the original; the official notebook is the reference.

## 5. AI model (optional)

- Open `dls/colab_train_demo.ipynb` in Colab, choose a GPU runtime, run the cells in order, then download `checkpoint.zip` and unzip it into the local `dls/checkpoint/`.
- Local inference: `python dls/predict.py --ckpt dls/checkpoint image.jpg`
- Demo of how the enhanced AI works (no model needed): `python dls/wizard_of_oz.py --demo`
- The paper's control AI was also trained on the authors' private images. Public data can only train an approximate model, so the accuracy will not match the paper.

## 6. Using real images in the demo website

- Public images (recommended for class): after downloading `fitzpatrick17k.csv`, run
  `python dls/fetch_f17k_images.py --csv fitzpatrick17k.csv --per-class 4 --n-other 6`
- Zenodo experiment images you are authorised to use: `python demo_app/import_images.py --src IMAGE_FOLDER --labels LABELS.csv --file-col ... --disease-col ... --fst-col ...`
  (These images may only be shown locally on your own computer, never uploaded to a public website or shared.)
- With a trained model, `python dls/predict.py --update-demo` writes the real AI predictions into the list; otherwise the website simulates AI suggestions at 47% accuracy (everyone gets the same suggestion for the same image, as in the paper).

## 7. Tips for the classroom demo

- Fix the assignment to show both interfaces: `/?dls=control&ui=control` and `/?dls=control&ui=treatment`
- Let classmates take part on their phones: `python demo_app/app.py --lan`, then share the computer's LAN IP (campus networks may block device-to-device traffic; test in advance)
- `/dashboard` refreshes the class results every 10 seconds; `/export.csv` downloads the data, which you can analyse on the spot with
  `python analysis/key_results.py --data classroom_responses.csv --min-trials 1 --keep-failed-attention`
- "Start new session" renames the old log as a backup; nothing is deleted

## 8. FAQ

- **ModuleNotFoundError**: run commands from the toolkit root, after `pip install -r requirements.txt`.
- **Port already in use**: `python demo_app/app.py --port 5050`.
- **Garbled characters (Windows)**: use `run_demo.bat`, or run `chcp 65001` first.
- **Many image downloads fail**: the original atlas links are dead; request the full image set from the authors via the form in the Fitzpatrick17k repository README.

## Licences and citation

The authors' ResearchBox content is CC BY 4.0; Fitzpatrick17k is CC BY-NC-SA 3.0 (non-commercial, attribution required); the Zenodo experiment images are bound by their access terms. Please cite the original paper and ResearchBox #1802.
