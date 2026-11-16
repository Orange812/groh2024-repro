# Groh et al. 2024 reproduction and demo toolkit

Paper: Groh M, Badri O, Daneshjou R, et al. *Deep learning-aided decision support for diagnosis of skin disease across skin tones.* **Nature Medicine** 30, 573–583 (2024). doi:10.1038/s41591-023-02728-3

> This is **independent reproduction code** written for a course presentation. It is not the official code released by the paper's authors. The official code and data are listed under "Authors' public resources" below.

> **Where to run the commands:** every command in this README is run from inside `groh2024_repro_kit/` (from the repository root: `cd groh2024_repro_kit`). Paths starting with `../` point to the other folders of the repository (`../data/`, `../work/`, `../environment/`).

## 1. Authors' public resources (the "source code + datasets" to introduce in the presentation)

| Resource | Location | Contents | How to get it |
|---|---|---|---|
| Experiment data + analysis code + AI training code | ResearchBox #1802: https://researchbox.org/1802 (Zenodo backup doi:10.5281/zenodo.15039588); in this repository: `../work/ResearchBox 1802/` | `Data/Experimental Data.csv.zip` (1.2 MB, every physician diagnosis), `Code/Analysis.ipynb` (reproduces the paper's figures and tables), `Code/requirements.txt`, `Code/Control DLS Training Code.py` (control-AI training code) | Click *entire box* on the web page to download, CC BY 4.0 |
| The 364 experiment images | Zenodo: doi:10.5281/zenodo.10070478 | Images of 46 skin diseases | Restricted access: register on Zenodo and agree to "research/medical use only, delete after use"; **apply a few days in advance** |
| Public dataset used to train the AI | https://github.com/mattgroh/fitzpatrick17k | 16,577 clinical images + Fitzpatrick skin-type labels (`fitzpatrick17k.csv`), `train.py` | The CSV downloads directly (in this repository: `../data/fitzpatrick17k.csv`); images download from the url column or by request form. Its CC BY-NC-SA 3.0 licence most likely covers only the annotations; the images were collected from the Atlas Dermatologico and DermaAmin websites |
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
│   ├─ fetch_f17k_images.py   download Fitzpatrick17k images (list mode / demo mode / training mode)
│   └─ colab_train_demo.ipynb train a demo model on a Colab GPU in 30–60 minutes
└─ demo_app/                  replica of the paper's experiment platform (Flask)
    ├─ app.py                 survey → diagnose images → AI suggestion (random button order) → feedback → comparison → live summary
    ├─ import_images.py       import images you are allowed to use (e.g. the Zenodo experiment images)
    └─ data/images.csv        image list: the 30 Fitzpatrick17k demo images (files not stored; download them, see section 6)
```

## 3. Five-minute quick start

```bash
pip install -r requirements.txt                           # or: conda env create -f ../environment/groh.yml
python 00_check_env.py
python analysis/make_synthetic_data.py                     # generate simulated data
python analysis/key_results.py --data data/synthetic_experiment.csv
python dls/fetch_f17k_images.py --csv ../data/fitzpatrick17k.csv --mode list   # optional: the 30 demo images
python demo_app/app.py                                     # open http://127.0.0.1:5000 in a browser
```
Without the image download, the demo website shows placeholders in place of the images.
On Windows you can also double-click `run_demo.bat`. Every output from simulated data is marked "SIMULATED DATA".

## 4. Reproducing with the authors' real data

**Route A: run the authors' official notebook (the most authoritative; show this first in the presentation)**
```bash
conda env create -f ../environment/groh-orig.yml   # the authors' pinned versions (they install cleanly)
conda activate groh-orig
cd "../work/ResearchBox 1802/Code"                # a fresh ResearchBox download has the same Code/ Data/ Codebook/ layout
python -m notebook                                 # open Analysis.ipynb and Run All
```
Use `groh-orig` for the notebook: the toolkit's environment `groh` cannot run it (it lacks `pyvis`, among others). The notebook in this repository already reads `../Data/Experimental Data.csv.zip`; with a fresh download, make the same one-line change to the data path in the first cell.

**Route B: independent reproduction with this toolkit (shows "reproduced vs paper")**
```bash
python analysis/inspect_data.py "../work/ResearchBox 1802/Data/Experimental Data.csv.zip" --write-map mapping.json
# open mapping.json and check the column names (if unsure, search Analysis.ipynb for the column name to see how the authors use it)
python analysis/key_results.py --data "../work/ResearchBox 1802/Data/Experimental Data.csv.zip" --map mapping.json --out outputs/real
```
Status: on the authors' data, the automatic guesses in `mapping.json` still need manual correction (see level B in the main README).
Output: `outputs/real/results_summary.md` (all tables), `table1.csv`…, `fig1/2/3*.png`.
Note: the regression specifications are inferred from the paper's text, so some coefficients may be defined slightly differently from the original; the official notebook is the reference.

## 5. AI model (optional)

- Open `dls/colab_train_demo.ipynb` in Colab, choose a GPU runtime, run the cells in order, then download `checkpoint.zip` and unzip it into the local `dls/checkpoint/`.
- Local inference: `python dls/predict.py --ckpt dls/checkpoint image.jpg`
- Demo of how the enhanced AI works (no model needed): `python dls/wizard_of_oz.py --demo`
- The paper's control AI was also trained on the authors' private images. Public data can only train an approximate model, so the accuracy will not match the paper.

## 6. Using real images in the demo website

- The 30 images listed in `demo_app/data/images.csv` (used for every recorded run): `python dls/fetch_f17k_images.py --csv ../data/fitzpatrick17k.csv --mode list`. The image files are not stored in this repository, because Fitzpatrick17k's licence most likely covers only its annotations, while the images were collected from the Atlas Dermatologico and DermaAmin websites. The script downloads exactly the listed files, checks each one's MD5 against its file name, and does not change the list.
- A new selection of public images: `python dls/fetch_f17k_images.py --csv ../data/fitzpatrick17k.csv --per-class 4 --n-other 6` (this rewrites `demo_app/data/images.csv`; the old list is kept as `images_backup_<time>.csv`).
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

The toolkit's code is MIT-licensed (see `../LICENSE`). The authors' ResearchBox content is CC BY 4.0; Fitzpatrick17k's annotations are CC BY-NC-SA 3.0 (non-commercial, attribution required); its images were collected from the source atlases and are not redistributed here; the Zenodo experiment images are bound by their access terms. Please cite the original paper and ResearchBox #1802.
