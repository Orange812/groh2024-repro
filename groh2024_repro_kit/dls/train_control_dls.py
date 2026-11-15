"""
Train the "control AI" (control DLS) — an independent rewrite from the description in the paper's Methods

The control AI in the paper:
  - VGG-16, pretrained on ImageNet;
  - fine-tuned on 31,219 diverse clinical skin images (Fitzpatrick 17k + textbook/atlas/search-engine images the authors collected);
  - 9 classes: the 8 main conditions + "Other";
  - augmentation: random resize to 256×256, random rotation 0–15°, random brightness/contrast/saturation/hue,
    random horizontal flip, centre crop 224×224, normalisation with the ImageNet mean and std;
  - top-1 accuracy 47% on the 364 experiment images; 58% on dark and 56% on light skin (8 main conditions) → "fair" across skin tones.

The authors' original training code is "Control DLS Training Code.py" in ResearchBox #1802; its structure matches
train.py in the Fitzpatrick17k repository (VGG-16 frozen, last layer replaced by 4096→256→ReLU→Dropout(0.4)→9 classes).
This script keeps the same structure but is simplified so that a demo model can be trained on Google Colab in 30–60 minutes.
Note: the extra images the authors collected are not public, so the exact model from the paper cannot be obtained.

Data format: a CSV with at least two columns
  path   image path (relative to --img-root)
  label  original disease label (mapped to the 9 classes automatically; the Fitzpatrick17k label column works as is)
  fst    (optional) Fitzpatrick type 1–6, used to check fairness across skin tones

Example (Colab GPU):
  python dls/train_control_dls.py --csv f17k_train.csv --img-root images/ --epochs 8 --max-per-class 400
Output: model.pt, class_names.json, val_predictions.csv and train_log.csv in the --out folder
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.diseases import MAIN8, matches  # noqa: E402

CLASSES = list(MAIN8) + ["Other"]


def to9(label: str) -> str:
    return next((k for k in MAIN8 if matches(str(label), k)), "Other")


def build_transforms():
    from torchvision import transforms as T
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train = T.Compose([
        T.RandomResizedCrop(256, scale=(0.8, 1.0)),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.RandomHorizontalFlip(),
        T.CenterCrop(224),
        T.ToTensor(), norm])
    test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), norm])
    return train, test


class SkinDataset:
    def __init__(self, df, root, tf):
        self.df, self.root, self.tf = df.reset_index(drop=True), Path(root), tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        from PIL import Image
        r = self.df.iloc[i]
        img = Image.open(self.root / r.path).convert("RGB")
        return self.tf(img), int(r.y), i


def build_model(n_classes: int, train_classifier: bool = False, pretrained: bool = True):
    import torch.nn as nn
    from torchvision import models
    try:
        m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
    except AttributeError:                                   # older torchvision
        m = models.vgg16(pretrained=pretrained)
    for p in m.parameters():
        p.requires_grad = False
    if train_classifier:
        for p in m.classifier.parameters():
            p.requires_grad = True
    m.classifier[6] = nn.Sequential(nn.Linear(4096, 256), nn.ReLU(), nn.Dropout(0.4),
                                    nn.Linear(256, n_classes), nn.LogSoftmax(dim=1))
    return m


def main():
    ap = argparse.ArgumentParser(description="Train the 9-class VGG-16 control AI (demo version)")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--img-root", default=".")
    ap.add_argument("--out", default=str(ROOT / "dls" / "checkpoint"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-per-class", type=int, default=0, help="maximum images per class (0 = all; 300–500 recommended for the Colab demo)")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--train-classifier", action="store_true", help="also fine-tune VGG's original classifier layers (slower, possibly more accurate)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=4242)
    a = ap.parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, WeightedRandomSampler

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    df = pd.read_csv(a.csv)
    if "path" not in df:
        if "md5hash" in df:                                   # using the Fitzpatrick17k CSV directly: image name = md5hash.jpg
            df["path"] = df["md5hash"].astype(str) + ".jpg"
        else:
            sys.exit("The CSV needs a path column (or the Fitzpatrick17k md5hash column)")
    df = df[[(Path(a.img_root) / p).exists() for p in df.path]].copy()
    df["cls"] = df["label"].map(to9)
    if a.max_per_class:                                       # shuffle, then take the first N per class (works with pandas 2.x / 3.x)
        df = df.sample(frac=1, random_state=a.seed).groupby("cls").head(a.max_per_class)
    df["y"] = df["cls"].map({c: i for i, c in enumerate(CLASSES)})
    fst_col = next((c for c in ("fst", "fitzpatrick_scale", "fitzpatrick") if c in df), None)
    print("Images per class:\n" + df.cls.value_counts().to_string())

    # Stratified train/validation split
    val_idx = df.groupby("cls").sample(frac=a.val_frac, random_state=a.seed).index
    val, train = df.loc[val_idx], df.drop(index=val_idx)

    tf_train, tf_test = build_transforms()
    counts = train.y.value_counts().reindex(range(len(CLASSES)), fill_value=0).to_numpy()
    w = 1.0 / np.maximum(counts, 1)
    sampler = WeightedRandomSampler(torch.tensor(w[train.y.to_numpy()], dtype=torch.double), len(train), replacement=True)
    dl_train = DataLoader(SkinDataset(train, a.img_root, tf_train), batch_size=a.batch_size, sampler=sampler,
                          num_workers=a.workers)
    dl_val = DataLoader(SkinDataset(val, a.img_root, tf_test), batch_size=a.batch_size, shuffle=False,
                        num_workers=a.workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; {len(train)} training images, {len(val)} validation images")
    model = build_model(len(CLASSES), a.train_classifier).to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=7, gamma=0.1)
    crit = nn.NLLLoss()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    best, log = -1.0, []
    for ep in range(a.epochs):
        t0 = time.time()
        model.train()
        tot, correct, loss_sum = 0, 0, 0.0
        for x, y, _ in dl_train:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            o = model(x)
            loss = crit(o, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
            correct += (o.argmax(1) == y).sum().item()
            tot += len(y)
        sched.step()
        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for x, y, _ in dl_val:
                o = model(x.to(device))
                vc += (o.argmax(1).cpu() == y).sum().item()
                vt += len(y)
        vacc = vc / max(vt, 1)
        log.append({"epoch": ep, "train_loss": loss_sum / tot, "train_acc": correct / tot, "val_acc": vacc})
        print(f"epoch {ep:2d}  train loss {loss_sum / tot:.3f}  train acc {correct / tot:.3f}  "
              f"val acc {vacc:.3f}  ({time.time() - t0:.0f}s)")
        if vacc > best:
            best = vacc
            torch.save(model.state_dict(), out / "model.pt")
    pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
    (out / "class_names.json").write_text(json.dumps(CLASSES, ensure_ascii=False, indent=2), encoding="utf-8")

    # Predict the validation set with the best model and check "fairness" across skin tones (paper: 58% dark vs 56% light)
    model.load_state_dict(torch.load(out / "model.pt", map_location=device))
    model.eval()
    rows = []
    with torch.no_grad():
        for x, y, idx in dl_val:
            p = model(x.to(device)).exp().cpu().numpy()
            for k in range(len(y)):
                r = val.iloc[int(idx[k])]
                top3 = np.argsort(-p[k])[:3]
                rows.append({"path": r.path, "label": CLASSES[int(y[k])], "pred": CLASSES[top3[0]],
                             "p1": float(p[k][top3[0]]), "d2": CLASSES[top3[1]], "d3": CLASSES[top3[2]],
                             "fst": r[fst_col] if fst_col else np.nan})
    pv = pd.DataFrame(rows)
    pv.to_csv(out / "val_predictions.csv", index=False)
    acc = (pv.label == pv.pred).mean()
    print(f"\nBest validation accuracy {acc:.3f}")
    if fst_col:
        pv["dark"] = pd.to_numeric(pv.fst, errors="coerce") >= 5
        print("By skin tone:\n" + pv.groupby("dark").apply(lambda g: (g.label == g.pred).mean()).rename(
            {False: "FST 1–4", True: "FST 5–6"}).to_string())
    print(f"Saved to {out}/")


if __name__ == "__main__":
    main()
