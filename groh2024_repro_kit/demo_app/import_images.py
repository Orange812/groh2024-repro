"""
Import a batch of labelled images into the demo website (for the Zenodo experiment images you are authorised to use, or any images of your own)

Usage:
  python demo_app/import_images.py --src IMAGE_FOLDER --labels LABELS.csv \
      --file-col filename --disease-col label --fst-col fitzpatrick [--sample 40]

  - the label CSV has one image per row: file name, disease name, Fitzpatrick type (1–6)
  - images are copied to demo_app/static/images/ and the list is written to demo_app/data/images.csv (the old list is backed up automatically)
  - --attention names one image (by file name) as the attention-check image, shown first in the experiment

Important: the 364 experiment images on Zenodo may be used only by registered users for research/medical purposes and must be deleted when the study ends;
do not put them on a public website or public repository, or share them with others. For class, demo them locally on your own computer.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description="Import images into the demo website")
    ap.add_argument("--src", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--file-col", default="file")
    ap.add_argument("--disease-col", default="disease")
    ap.add_argument("--fst-col", default="fst")
    ap.add_argument("--attention", default="", help="file name of the attention-check image (optional)")
    ap.add_argument("--sample", type=int, default=0, help="how many images to sample at random (0 = all)")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    lab = pd.read_csv(a.labels)
    if a.sample:
        lab = lab.sample(min(a.sample, len(lab)), random_state=a.seed)
    src, dst = Path(a.src), BASE / "static" / "images"
    dst.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, r in enumerate(lab.itertuples(index=False)):
        name = str(getattr(r, a.file_col))
        p = src / name
        if not p.exists():
            print(f"  ✗ not found: {p}")
            continue
        shutil.copy2(p, dst / p.name)
        rows.append({"image_id": f"img_{i:03d}", "file": p.name, "disease": getattr(r, a.disease_col),
                     "fst": int(getattr(r, a.fst_col)), "attention_check": int(p.name == a.attention),
                     "dls_pred_control": "", "credit": ""})
    out = BASE / "data" / "images.csv"
    if out.exists():
        out.rename(out.with_name(f"images_backup_{time.strftime('%Y%m%d_%H%M%S')}.csv"))
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Imported {len(rows)} images → {out}")


if __name__ == "__main__":
    main()
