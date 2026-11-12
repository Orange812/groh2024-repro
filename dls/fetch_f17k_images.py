"""
从 Fitzpatrick 17k 公开数据集下载图片：给课堂演示网站用（demo 模式），或训练演示版 AI（training 模式）

为什么用它：论文实验用的 364 张图存放在 Zenodo（doi:10.5281/zenodo.10070478），
需要注册并同意"仅用于科研/医学目的、用完删除"的条款，不能公开再分发。
课堂演示可以改用 Fitzpatrick 17k（同一作者团队的公开数据集，CC BY-NC-SA 3.0，非商业使用需署名）。

准备：从 https://github.com/mattgroh/fitzpatrick17k 下载 fitzpatrick17k.csv

  # demo 模式：8 种主要疾病各 4 张（尽量深浅肤色各半）+ 6 张其他疾病 + 1 张痤疮注意力检查图
  python dls/fetch_f17k_images.py --csv fitzpatrick17k.csv --per-class 4 --n-other 6
    → 图片存到 demo_app/static/images/，清单写入 demo_app/data/images.csv

  # training 模式：9 类（8 种主要疾病 + Other）各最多 300 张，用于 train_control_dls.py
  python dls/fetch_f17k_images.py --csv fitzpatrick17k.csv --mode training --per-class 300 \
      --out-dir f17k_images --train-csv f17k_train.csv

注意：图片来自原始图谱网站的 url 列，部分链接可能已失效，脚本会自动跳过；
如果大量失败，可按 Fitzpatrick17k 仓库 README 填表向作者申请整包图片。
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.diseases import MAIN8, matches  # noqa: E402


def pick(df, n, seed):
    """尽量各一半浅色（FST1–4）和深色（FST5–6）。"""
    light = df[df.fst.between(1, 4)].sample(frac=1, random_state=seed)
    dark = df[df.fst.between(5, 6)].sample(frac=1, random_state=seed)
    k = min(n // 2, len(dark))
    return pd.concat([dark.head(k), light.head(n - k)]).head(n)


def download(url, dest, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (classroom demo)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    if len(data) < 2000:
        raise ValueError("文件太小，可能不是图片")
    dest.write_bytes(data)


def fetch_all(plan: pd.DataFrame, out_dir: Path, workers: int = 8) -> pd.Series:
    """并发下载；返回每行是否成功。已存在的文件不重复下载。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    def one(r):
        dest = out_dir / f"{r.md5hash}.jpg"
        if dest.exists():
            return True
        try:
            download(r.url, dest)
            return True
        except Exception:                                      # noqa: BLE001
            return False

    with ThreadPoolExecutor(max_workers=workers) as ex:
        ok = list(ex.map(one, [r for r in plan.itertuples(index=False)]))
    return pd.Series(ok, index=plan.index)


def main():
    ap = argparse.ArgumentParser(description="下载 Fitzpatrick17k 图片")
    ap.add_argument("--csv", required=True, help="fitzpatrick17k.csv")
    ap.add_argument("--mode", choices=["demo", "training"], default="demo")
    ap.add_argument("--per-class", type=int, default=4, help="demo：8 种病每种几张；training：9 类每类最多几张")
    ap.add_argument("--n-other", type=int, default=6, help="demo 模式另外加几张其他疾病")
    ap.add_argument("--out-dir", default="f17k_images", help="training 模式的图片目录")
    ap.add_argument("--train-csv", default="f17k_train.csv", help="training 模式输出的训练清单")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true", help="只显示计划，不下载")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    fst_col = next(c for c in ("fitzpatrick_scale", "fitzpatrick", "fitzpatrick_centaur") if c in df)
    df["fst"] = pd.to_numeric(df[fst_col], errors="coerce")
    df = df[df.url.notna() & df.md5hash.notna()]

    labels = sorted(df.label.dropna().unique())
    hits = {dis: [l for l in labels if matches(l, dis)] for dis in MAIN8}
    print("8 种主要疾病在 Fitzpatrick17k 中对应的标签：")
    for dis, h in hits.items():
        print(f"  {dis:<26} ← {h if h else '（未找到：该病的演示图需另外补充）'}")
    main_labels = {l for h in hits.values() for l in h}

    if a.mode == "training":
        parts = [df[df.label.isin(h)].sample(min(len(df[df.label.isin(h)]), a.per_class), random_state=a.seed)
                 for h in hits.values() if h]
        other = df[~df.label.isin(main_labels)]
        parts.append(other.sample(min(len(other), a.per_class), random_state=a.seed))
        plan = pd.concat(parts)
        print(f"\n计划下载 {len(plan)} 张训练图片 → {a.out_dir}/")
        if a.dry_run:
            return
        ok = fetch_all(plan, Path(a.out_dir), a.workers)
        res = plan[ok].assign(path=lambda x: x.md5hash + ".jpg")[["path", "label", "fst"]]
        res.to_csv(a.train_csv, index=False)
        print(f"成功 {ok.sum()} / {len(plan)} 张 → 训练清单 {a.train_csv}")
        print(f"下一步：python dls/train_control_dls.py --csv {a.train_csv} --img-root {a.out_dir}")
        return

    # ---------------- demo 模式
    dfk = df[df.fst.between(1, 6)]
    plan = []
    acne = dfk[dfk.label.str.lower().str.contains("acne", na=False)]
    if not acne.empty:                                         # 注意力检查图（论文用的是一张痤疮图）
        plan.append(acne.sort_values("fst", ascending=False).head(1).assign(disease="Acne", attention_check=1))
    for dis, h in hits.items():
        if h:
            plan.append(pick(dfk[dfk.label.isin(h)], a.per_class, a.seed).assign(disease=dis, attention_check=0))
    others = dfk[~dfk.label.isin(main_labels) & ~dfk.label.str.lower().str.contains("acne", na=False)]
    if a.n_other and not others.empty:
        o = others.sample(min(a.n_other, len(others)), random_state=a.seed)
        plan.append(o.assign(disease=o.label.str.capitalize(), attention_check=0))
    plan = pd.concat(plan, ignore_index=True)
    print(f"\n计划下载 {len(plan)} 张演示图片")
    if a.dry_run:
        print(plan[["disease", "label", "fst", "url"]].to_string(index=False))
        return

    img_dir = ROOT / "demo_app" / "static" / "images"
    ok = fetch_all(plan, img_dir, a.workers)
    rows = [{"image_id": f"f17k_{i:03d}", "file": f"{r.md5hash}.jpg", "disease": r.disease, "fst": int(r.fst),
             "attention_check": int(r.attention_check), "dls_pred_control": "",
             "credit": "Fitzpatrick17k (CC BY-NC-SA 3.0)"}
            for i, r in enumerate(plan[ok].itertuples(index=False))]
    out = ROOT / "demo_app" / "data" / "images.csv"
    if not rows:
        print("全部下载失败（网络或原网站问题），保留原来的图片清单不变。")
        return
    if out.exists():
        out.rename(out.with_name(f"images_backup_{time.strftime('%Y%m%d_%H%M%S')}.csv"))
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"成功 {len(rows)} / {len(plan)} 张 → {out}")
    print("下一步（可选）：用训练好的 AI 写入预测：python dls/predict.py --update-demo")


if __name__ == "__main__":
    main()
