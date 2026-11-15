"""
Predict images with the trained control AI (for the live demo)

Usage:
  # One or more images: print the top 3 with probabilities
  python dls/predict.py --ckpt dls/checkpoint img1.jpg img2.jpg
  # Precompute AI suggestions for every demo-site image (written to the dls_pred_control column of demo_app/data/images.csv)
  python dls/predict.py --ckpt dls/checkpoint --update-demo
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dls"))


def load(ckpt: Path):
    import torch
    from train_control_dls import build_model, build_transforms
    classes = json.loads((ckpt / "class_names.json").read_text(encoding="utf-8"))
    model = build_model(len(classes), pretrained=False)   # weights come from model.pt, so no ImageNet download is needed
    model.load_state_dict(torch.load(ckpt / "model.pt", map_location="cpu"))
    model.eval()
    return model, classes, build_transforms()[1]


def predict(paths, model, classes, tf, k=3):
    import torch
    from PIL import Image
    out = []
    with torch.no_grad():
        for p in paths:
            x = tf(Image.open(p).convert("RGB")).unsqueeze(0)
            prob = model(x).exp()[0]
            top = torch.topk(prob, k)
            out.append({"image": str(p), **{f"d{i + 1}": classes[j] for i, j in enumerate(top.indices.tolist())},
                        **{f"p{i + 1}": round(float(v), 3) for i, v in enumerate(top.values.tolist())}})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description="Control AI inference")
    ap.add_argument("images", nargs="*")
    ap.add_argument("--ckpt", default=str(ROOT / "dls" / "checkpoint"))
    ap.add_argument("--update-demo", action="store_true", help="write AI predictions for all demo_app images")
    a = ap.parse_args()
    model, classes, tf = load(Path(a.ckpt))

    if a.update_demo:
        csv = ROOT / "demo_app" / "data" / "images.csv"
        df = pd.read_csv(csv)
        img_dir = ROOT / "demo_app" / "static" / "images"
        has = df["file"].fillna("").astype(str).map(lambda f: bool(f) and (img_dir / f).exists())
        res = predict([img_dir / f for f in df.loc[has, "file"]], model, classes, tf)
        df.loc[has, "dls_pred_control"] = res["d1"].to_numpy()
        df.to_csv(csv, index=False)
        print(f"Wrote AI predictions for {has.sum()} images → {csv}")
        return
    if not a.images:
        sys.exit("Give image paths, or use --update-demo")
    print(predict(a.images, model, classes, tf).to_string(index=False))


if __name__ == "__main__":
    main()
