"""
训练「对照 AI」（Control DLS）——按论文 Methods 的描述独立重写

论文里的对照 AI：
  - VGG-16，ImageNet 预训练；
  - 在 31,219 张多样化临床皮肤图片上微调（Fitzpatrick 17k + 作者另收集的教科书/图谱/搜索引擎图片）；
  - 分 9 类：8 种重点疾病 + "Other"；
  - 数据增强：随机缩放到 256×256、随机旋转 0–15°、随机调整亮度/对比度/饱和度/色调、
    随机水平翻转、中心裁剪 224×224、按 ImageNet 均值方差归一化；
  - 在实验用的 364 张图上 top-1 准确率 47%；深色 58%、浅色 56%（8 种主要疾病）→ 在肤色间"公平"。

作者的原始训练代码在 ResearchBox #1802 的 "Control DLS Training Code.py"，结构与
Fitzpatrick17k 仓库的 train.py 相同（冻结 VGG-16，替换最后一层为 4096→256→ReLU→Dropout(0.4)→9 类）。
本脚本沿用同样的结构，但做了简化，方便在 Google Colab 上 30–60 分钟内训练一个演示用模型。
注意：作者额外收集的图片没有公开，所以无法得到与论文完全相同的模型。

数据格式：一个 CSV，至少两列
  path   图片路径（相对 --img-root）
  label  原始疾病标签（会自动映射为 9 类；Fitzpatrick17k 的 label 列可直接用）
  fst    （可选）Fitzpatrick 分型 1–6，用于按肤色检查公平性

用法示例（Colab GPU）：
  python dls/train_control_dls.py --csv f17k_train.csv --img-root images/ --epochs 8 --max-per-class 400
输出：--out 目录下的 model.pt、class_names.json、val_predictions.csv、train_log.csv
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
    except AttributeError:                                   # 旧版 torchvision
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
    ap = argparse.ArgumentParser(description="训练 9 分类 VGG-16 对照 AI（演示版）")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--img-root", default=".")
    ap.add_argument("--out", default=str(ROOT / "dls" / "checkpoint"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-per-class", type=int, default=0, help="每类最多取多少张（0=全部；Colab 演示建议 300–500）")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--train-classifier", action="store_true", help="同时微调 VGG 原分类层（更慢，可能更准）")
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
        if "md5hash" in df:                                   # 直接用 Fitzpatrick17k 的 CSV：图片名 = md5hash.jpg
            df["path"] = df["md5hash"].astype(str) + ".jpg"
        else:
            sys.exit("CSV 需要 path 列（或 Fitzpatrick17k 的 md5hash 列）")
    df = df[[(Path(a.img_root) / p).exists() for p in df.path]].copy()
    df["cls"] = df["label"].map(to9)
    if a.max_per_class:                                       # 打乱后每类取前 N 张（兼容 pandas 2.x / 3.x）
        df = df.sample(frac=1, random_state=a.seed).groupby("cls").head(a.max_per_class)
    df["y"] = df["cls"].map({c: i for i, c in enumerate(CLASSES)})
    fst_col = next((c for c in ("fst", "fitzpatrick_scale", "fitzpatrick") if c in df), None)
    print("各类图片数：\n" + df.cls.value_counts().to_string())

    # 分层抽样划分训练/验证
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
    print(f"设备：{device}；训练 {len(train)} 张，验证 {len(val)} 张")
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

    # 用最佳模型在验证集上输出预测，并按肤色检查"公平性"（论文：深色 58% vs 浅色 56%）
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
    print(f"\n最佳验证准确率 {acc:.3f}")
    if fst_col:
        pv["dark"] = pd.to_numeric(pv.fst, errors="coerce") >= 5
        print("按肤色：\n" + pv.groupby("dark").apply(lambda g: (g.label == g.pred).mean()).rename(
            {False: "FST 1–4", True: "FST 5–6"}).to_string())
    print(f"已保存到 {out}/")


if __name__ == "__main__":
    main()
