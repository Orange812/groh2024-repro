# Reproduction of Groh et al. 2024 (Nature Medicine)

Groh M, Badri O, Daneshjou R, et al. *Deep learning-aided decision support for diagnosis of skin disease across skin tones.* Nature Medicine 30, 573–583 (2024). doi:10.1038/s41591-023-02728-3

Course project for Research Methodology of Computer Science. The reproduction has four levels: **A** the authors' notebook, **B** independent recomputation, **C** the AI model, and **D** the experiment platform. This repository holds all source code, the datasets that may be redistributed, the environment configuration and the evidence for every result.

## Repository layout

| Folder | What it holds |
|---|---|
| [`work/ResearchBox 1802/`](work/ResearchBox%201802/) | The authors' public files from [ResearchBox #1802](https://researchbox.org/1802) (CC BY 4.0): the data (`Data/Experimental Data.csv.zip`), its codebook, `Code/Analysis.ipynb` (saved with the outputs of my run), `Code/Control DLS Training Code.py` and `Code/requirements.txt`. `Code/figs/` holds the 26 figures produced by my run. Commit `26068be` holds the files exactly as downloaded; the only change is `395b201`, a one-line fix to the data path. |
| [`original/ResearchBox 1802/`](original/ResearchBox%201802/) | An untouched copy of the authors' files, for side-by-side comparison. Identical to commit `26068be`. |
| [`groh2024_repro_kit/`](groh2024_repro_kit/) | The reproduction toolkit: analysis scripts (`analysis/`), the Wizard-of-Oz script and DLS training scripts for Colab (`dls/`), and `demo_app/`, a Flask replica of the experiment website that uses 30 public Fitzpatrick17k images. Commit `fe9e009` is the toolkit as received; 6 fix commits follow (`45451d6` … `7947102`), then the translation into English (`e454af7`). It has its own [README](groh2024_repro_kit/README.md). |
| [`logs/`](logs/) | The evidence: the executed notebook (`analysis_run.html` / `.ipynb`), environment records, change lists (`changes*.txt`, `kit_changes*.txt`), and a verification table plus scripts for each step. The two detailed write-ups are [`step5_verification.md`](logs/step5_verification.md) (level A) and [`step8_platform_check.md`](logs/step8_platform_check.md) (level D). |
| [`data/`](data/) | `fitzpatrick17k.csv`, the public Fitzpatrick17k metadata (CC BY-NC-SA 3.0), used by `groh2024_repro_kit/dls/fetch_f17k_images.py`. |
| [`environment/`](environment/) | Conda environment files for the two environments used (see [Environment setup](#environment-setup)). |

## Datasets

| Dataset | In this repository? | Where |
|---|---|---|
| Experiment data: 14,261 diagnoses by 1,118 physicians (authors, ResearchBox #1802, CC BY 4.0) | Yes | `work/ResearchBox 1802/Data/Experimental Data.csv.zip`, codebook in `work/ResearchBox 1802/Codebook/` |
| Fitzpatrick17k metadata, 16,577 images (Groh et al. 2021, CC BY-NC-SA 3.0) | Yes | `data/fitzpatrick17k.csv` (source: https://github.com/mattgroh/fitzpatrick17k) |
| 30 Fitzpatrick17k images for the replica website (CC BY-NC-SA 3.0) | Yes | `groh2024_repro_kit/demo_app/static/images/`; `demo_app/data/images.csv` records each image's source URL |
| Files derived from the authors' data (per-image AI predictions, Wizard-of-Oz output) | Yes | `logs/step7_woz/` |
| The 364 experiment images (Zenodo, doi:10.5281/zenodo.10070478) | **No** | Restricted access: research or medical use only, and redistribution is not allowed. Access can be requested on Zenodo. Nothing in this repository needs them to run. |

## Environment setup

Two conda environments were used. Both files below were tested by building fresh environments from them and rerunning everything; the results matched the recorded runs.

| Environment | Used for | Create it with |
|---|---|---|
| `groh-orig` | Level A: the authors' notebook, with the authors' own pinned versions (Python 3.9, pandas 2.1.0, statsmodels 0.14.0, …) | `conda env create -f environment/groh-orig.yml` |
| `groh` | Levels B–D: the toolkit, the scripts in `logs/` and the replica website (Python 3.9, pandas 1.4, Flask) | `conda env create -f environment/groh.yml` |

The exact environment of the recorded run (macOS 15.7.5, arm64) is in `logs/environment.yml`, `logs/pip_freeze.txt`, `logs/python_version.txt` and `logs/system_info.txt`.

On macOS, the environment must be **activated** (`conda activate groh-orig`), not just called by path. Otherwise `jupyter` picks up the base environment and the notebook fails with `No module named 'stargazer'`.

## How to run

Run every command from the repository root.

**Level A: the authors' notebook** (about 1 minute):

```bash
conda activate groh-orig
cd "work/ResearchBox 1802/Code"
jupyter nbconvert --to html --execute Analysis.ipynb --ExecutePreprocessor.timeout=-1 --output-dir ../../../logs/rerun
```

You can also open the notebook with `jupyter notebook Analysis.ipynb` and choose Run All. Figures are written to `Code/figs/`. To compare a rerun with the authors' saved outputs, execute to a notebook (`--to notebook` instead of `--to html`) and run `python logs/step5_supplement/compare_outputs.py "original/ResearchBox 1802/Code/Analysis.ipynb" logs/rerun/Analysis.ipynb`.

**Levels B–D: the toolkit:**

```bash
conda activate groh

python logs/step7_woz/woz_reproduce.py                    # C: audit and re-run of the Wizard-of-Oz AI
python groh2024_repro_kit/dls/wizard_of_oz.py --pred logs/step7_woz/woz_input.csv --exact
python logs/step8/grader_agreement.py                     # D: automatic grading vs the authors' manual grading
DEMO_LOG=/tmp/sim_log.csv python logs/step8/simulate_participants.py groh2024_repro_kit   # D: 18 protocol checks

cd groh2024_repro_kit && python demo_app/app.py --port 5050   # D: replica website at http://127.0.0.1:5050
```

## Results by level

| Level | Status | Evidence |
|---|---|---|
| **A** Authors' notebook | **Done.** It runs end to end with 0 errors using the authors' own `requirements.txt`, with one compatibility change (the data path). The text outputs of all 80 code cells match the authors' saved outputs; only the statsmodels date/time stamps differ. Of the 25 items checked against the paper, 19 match, 3 are close, 2 partly differ, and 1 (the confusion matrices) is left for a manual check. | `logs/analysis_run.html`, `logs/step5_verification.md`, `logs/step5_supplement/compare_outputs_result.txt` |
| **B** Independent recomputation | **Partly done.** I checked the numbers the notebook does not print with separate scripts, and re-derived the Wizard-of-Oz AI from the authors' per-image data. The toolkit's `analysis/key_results.py` has not been run on the authors' data, because its automatic column mapping needs to be fixed by hand. | `logs/step5_supplement/`, `logs/step7_woz/` |
| **C** AI model | **Partly done.** The control AI (VGG-16) has not been retrained; this needs a GPU, and `groh2024_repro_kit/dls/colab_train_demo.ipynb` is ready to run on Colab. The treatment AI (Wizard of Oz) is reproduced from the authors' control-AI predictions. 127 of 193 wrong predictions (65.8%) were flipped to correct and no correct prediction was changed. Accuracy rises from 46.8% to 81.8% on all images and to 84.2% on the 8 main conditions (the paper reports 84%). | `logs/step7_woz/woz_reproduce_output.txt` |
| **D** Experiment platform | **Done as a replica.** The original site's code is not public, so it was rebuilt from the paper's Methods. The replica is compared with the paper on 17 elements and passes 18 of 18 automated protocol checks. Its automatic grading agrees with the authors' manual grading on 98.9% of answers for the 8 main conditions. | `logs/step8_platform_check.md`, `logs/step8/` |

Discrepancies found in the paper (details in `logs/step5_verification.md`, section 4):

- **F1, a code bug in cell 59.** The BCD "incorrect diagnoses per diagnosis" figure is divided by the PCP observation count, so the BCD values are 16% too low.
- **F2, a number that cannot be reproduced.** The paper's overall PCP referral rate of 28% does not come out of the data, which give 35.6%.
- **F3–F4, wording and choice of test.** The "all three diagnoses wrong" shares are computed under a looser condition than the text states, and the "not significant" drop after wrong AI suggestions comes from an unpaired test on duplicated rows (a paired test gives P = 0.008 / 0.003).
- **F5, an unexplained gap that turns out to be a scope difference.** The paper's "84%" for the treatment AI is its accuracy on the 8 main conditions; on all images it is 81.8%.
- **F6, a qualification.** "No significant interface interaction effects" holds for top-4 but not fully for top-1 (2 of 24 terms have P < 0.05).

## Change history

This repository combines two earlier git repositories, `work/` and `groh2024_repro_kit/`, with their full histories, so every change made during the reproduction can be inspected. Inside those imported commits, paths are relative to their own folder.

```bash
git log --oneline --graph            # the whole history
git show 395b201                     # the only change to the authors' code (data path)
git diff fe9e009 7947102 --stat      # the 6 fixes to the toolkit as received
git show e454af7 --stat              # the toolkit's translation into English
```

The toolkit and the write-ups in `logs/` were originally written in Chinese and have been translated into English. The translation changes no code logic and no numbers; every script was rerun before and after with identical results. The toolkit's original Chinese text remains visible in the commits before `e454af7`.

## Not included, and why

| Item | Reason |
|---|---|
| The 364 experiment images (Zenodo) | Restricted access; redistribution is not allowed (see [Datasets](#datasets)). |
| Notebook checkpoints, caches | Regenerated automatically. |
| The replica website's response log | Test entries only, not part of the code. |

## Licences

- The authors' data and code (ResearchBox #1802) are CC BY 4.0. Please cite the paper and ResearchBox #1802.
- Fitzpatrick17k (`data/fitzpatrick17k.csv` and the 30 images in `groh2024_repro_kit/demo_app/static/images/`) is CC BY-NC-SA 3.0: non-commercial use, with attribution, under the same licence. The images were downloaded from public atlasdermatologico.com.br addresses, and `demo_app/data/images.csv` records each image's source. Two of them also appear in the paper's restricted image set, but they are used here under the Fitzpatrick17k licence.
