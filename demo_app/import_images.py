"""
把一批带标签的图片导入演示网站（适用于你获得授权的 Zenodo 实验图片，或任何自有图片）

用法：
  python demo_app/import_images.py --src 图片文件夹 --labels 标签.csv \
      --file-col filename --disease-col label --fst-col fitzpatrick [--sample 40]

  - 标签 CSV 里每行一张图：文件名、疾病名、Fitzpatrick 分型（1–6）
  - 图片会复制到 demo_app/static/images/，清单写入 demo_app/data/images.csv（原清单自动备份）
  - --attention 可指定一张作为注意力检查图（文件名），实验中第一个出现

重要：Zenodo 上的 364 张实验图片只允许注册用户用于科研/医学目的，研究结束后需删除；
不要把它们放到公开网站、公开仓库或分享给他人。课堂上在自己电脑本地演示即可。
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description="导入图片到演示网站")
    ap.add_argument("--src", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--file-col", default="file")
    ap.add_argument("--disease-col", default="disease")
    ap.add_argument("--fst-col", default="fst")
    ap.add_argument("--attention", default="", help="作为注意力检查的图片文件名（可选）")
    ap.add_argument("--sample", type=int, default=0, help="随机抽取多少张（0=全部）")
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
            print(f"  ✗ 找不到 {p}")
            continue
        shutil.copy2(p, dst / p.name)
        rows.append({"image_id": f"img_{i:03d}", "file": p.name, "disease": getattr(r, a.disease_col),
                     "fst": int(getattr(r, a.fst_col)), "attention_check": int(p.name == a.attention),
                     "dls_pred_control": "", "credit": ""})
    out = BASE / "data" / "images.csv"
    if out.exists():
        out.rename(out.with_name(f"images_backup_{time.strftime('%Y%m%d_%H%M%S')}.csv"))
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"导入 {len(rows)} 张 → {out}")


if __name__ == "__main__":
    main()
