"""
Download images from the public Fitzpatrick 17k dataset: for the classroom demo website (demo mode) or to train the demo AI (training mode)

Why: the 364 images used in the paper's experiment are on Zenodo (doi:10.5281/zenodo.10070478),
which requires registering and agreeing to "research/medical use only, delete after use"; they may not be redistributed publicly.
The classroom demo can use Fitzpatrick 17k instead (a public dataset from the same research group, CC BY-NC-SA 3.0: non-commercial use with attribution).

Preparation: download fitzpatrick17k.csv from https://github.com/mattgroh/fitzpatrick17k

  # demo mode: 4 images for each of the 8 main conditions (half dark, half light skin where possible) + 6 other diseases + 1 acne attention-check image
  python dls/fetch_f17k_images.py --csv fitzpatrick17k.csv --per-class 4 --n-other 6
    → images are saved to demo_app/static/images/ and the list is written to demo_app/data/images.csv

  # training mode: at most 300 images for each of 9 classes (8 main conditions + Other), for train_control_dls.py
  python dls/fetch_f17k_images.py --csv fitzpatrick17k.csv --mode training --per-class 300 \
      --out-dir f17k_images --train-csv f17k_train.csv

Note: images are downloaded from the url column (the original atlas websites); dead links are skipped automatically;
if many fail, request the full image set from the authors via the form in the Fitzpatrick17k repository README.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.diseases import AUXILIARY, MAIN8, matches  # noqa: E402

# Fitzpatrick17k labels are noisy. dermaamin.com image URLs contain the source atlas's own diagnosis folder name;
# e.g. of the 128 images labelled "lyme disease", 40 come from the erytrasma (erythrasma) folder,
# and 22 of the "dermatomyositis" images come from dermatophytie (dermatophytosis).
# demo mode keeps only images whose folder name matches the disease (the allow-list is spelled out below for review);
# atlasdermatologico.com.br URLs carry no diagnosis, so they cannot be checked and are kept.
FOLDER_OK = {
    "Acne": {"acne_vulgaris"},
    "Atopic dermatitis": {"eczema_atopic"},
    "CTCL": {"mycosis-fongoide"},
    "Dermatomyositis": {"dermatomyositis"},
    "Lichen planus": {"lichen_planus", "eruptive_lichen_planus", "lichen_ruber"},
    "Lyme disease": {"erythem_achronicum_migrans", "lyme-disease", "lime-disease", "erythema_migrans"},
    "Pityriasis rosea": {"pityriasis_rosea"},
    "Pityriasis rubra pilaris": {"pityriasis_rubra_pilaris"},
}


def url_folder(url) -> str:
    m = re.search(r"/clinical-pic/[^/]+/([^/]+)/", str(url))
    return m.group(1) if m else ""


def label_consistent(row, disease: str) -> bool:
    """Whether the source-atlas diagnosis in the image URL matches the disease; images marked "Wrongly labelled" in the F17k QC column are always excluded."""
    if str(getattr(row, "qc", "")).startswith("3"):
        return False
    folder = url_folder(row.url)
    if not folder:
        return True
    if disease in FOLDER_OK:
        return folder in FOLDER_OK[disease]
    return matches(re.sub(r"[-_]+", " ", folder), disease)


def keep_consistent(df: pd.DataFrame, disease_of, check: bool, name: str) -> pd.DataFrame:
    if not check or df.empty:
        return df
    ok = [label_consistent(r, disease_of(r)) for r in df.itertuples(index=False)]
    kept = df[ok]
    print(f"  label check {name:<26} kept {len(kept)}/{len(df)}")
    return kept


def pick(df, n, seed):
    """Aim for half light (FST1–4) and half dark (FST5–6) skin."""
    light = df[df.fst.between(1, 4)].sample(frac=1, random_state=seed)
    dark = df[df.fst.between(5, 6)].sample(frac=1, random_state=seed)
    k = min(n // 2, len(dark))
    return pd.concat([dark.head(k), light.head(n - k)]).head(n)


def download(url, dest, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (classroom demo)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    if len(data) < 2000:
        raise ValueError("File too small; probably not an image")
    dest.write_bytes(data)


def fetch_all(plan: pd.DataFrame, out_dir: Path, workers: int = 8) -> pd.Series:
    """Download in parallel; return whether each row succeeded. Files that already exist are not downloaded again."""
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


def host_of(url) -> str:
    m = re.search(r"https?://(?:www\.)?([^/]+)/", str(url))
    return m.group(1) if m else ""


def live_hosts(df: pd.DataFrame, n_probe: int = 3, seed: int = 0) -> set[str]:
    """Try a few downloads from each image host (into memory only, not saved); a host where every attempt fails is treated as dead.
    Example: after a redesign of dermaamin.com its old image URLs go 301 → 404, so 77% of the F17k images cannot be downloaded."""
    hosts = df.url.map(host_of)
    alive = set()
    for h in sorted(hosts.unique()):
        sample = df[hosts == h].sample(min(n_probe, int((hosts == h).sum())), random_state=seed)
        ok = False
        for u in sample.url:
            try:
                req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0 (classroom demo)"})
                with urllib.request.urlopen(req, timeout=20) as r:
                    ok = len(r.read()) >= 2000
            except Exception:                                  # noqa: BLE001
                ok = False
            if ok:
                break
        print(f"  {h:<28} {'reachable' if ok else 'dead (skipped)'}")
        if ok:
            alive.add(h)
    return alive


def interleave(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Order the candidates alternating dark, light (dark first); downloads fill the quota in this order, aiming for half dark and half light."""
    dark = df[df.fst.between(5, 6)].sample(frac=1, random_state=seed)
    light = df[df.fst.between(1, 4)].sample(frac=1, random_state=seed)
    order = []
    for i in range(max(len(dark), len(light))):
        order += [x.iloc[[i]] for x in (dark, light) if i < len(x)]
    return pd.concat(order) if order else df.head(0)


def fetch_quota(cands: pd.DataFrame, n: int, out_dir: Path, workers: int, max_try: int = 40) -> pd.DataFrame:
    """Download candidates in order, in batches, until n succeed or max_try have been tried (a dead link is replaced by the next candidate)."""
    got, step = [], max(n, 4)
    for start in range(0, min(len(cands), max_try), step):
        batch = cands.iloc[start:start + step]
        ok = fetch_all(batch, out_dir, workers)
        got.append(batch[ok.values])
        if sum(map(len, got)) >= n:
            break
    return pd.concat(got).head(n) if got else cands.head(0)


def main():
    ap = argparse.ArgumentParser(description="Download Fitzpatrick17k images")
    ap.add_argument("--csv", required=True, help="fitzpatrick17k.csv")
    ap.add_argument("--mode", choices=["demo", "training"], default="demo")
    ap.add_argument("--per-class", type=int, default=4, help="demo: images per main condition; training: maximum images per class (9 classes)")
    ap.add_argument("--n-other", type=int, default=6, help="demo mode: number of extra images of other diseases")
    ap.add_argument("--out-dir", default="f17k_images", help="image folder for training mode")
    ap.add_argument("--train-csv", default="f17k_train.csv", help="training list written in training mode")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true", help="show the plan only; do not download")
    ap.add_argument("--no-label-check", action="store_true",
                    help="demo mode: do not check labels against image URLs (checked by default; see FOLDER_OK)")
    a = ap.parse_args()
    check = not a.no_label_check

    df = pd.read_csv(a.csv)
    fst_col = next(c for c in ("fitzpatrick_scale", "fitzpatrick", "fitzpatrick_centaur") if c in df)
    df["fst"] = pd.to_numeric(df[fst_col], errors="coerce")
    df = df[df.url.notna() & df.md5hash.notna()]

    labels = sorted(df.label.dropna().unique())
    hits = {dis: [l for l in labels if matches(l, dis)] for dis in MAIN8}
    print("Fitzpatrick17k labels for the 8 main conditions:")
    for dis, h in hits.items():
        print(f"  {dis:<26} ← {h if h else '(not found: demo images for this disease must be added separately)'}")
    main_labels = {l for h in hits.values() for l in h}

    if a.mode == "training":
        parts = [df[df.label.isin(h)].sample(min(len(df[df.label.isin(h)]), a.per_class), random_state=a.seed)
                 for h in hits.values() if h]
        other = df[~df.label.isin(main_labels)]
        parts.append(other.sample(min(len(other), a.per_class), random_state=a.seed))
        plan = pd.concat(parts)
        print(f"\nPlanning to download {len(plan)} training images → {a.out_dir}/")
        if a.dry_run:
            return
        ok = fetch_all(plan, Path(a.out_dir), a.workers)
        res = plan[ok].assign(path=lambda x: x.md5hash + ".jpg")[["path", "label", "fst"]]
        res.to_csv(a.train_csv, index=False)
        print(f"Downloaded {ok.sum()} / {len(plan)} → training list {a.train_csv}")
        print(f"Next: python dls/train_control_dls.py --csv {a.train_csv} --img-root {a.out_dir}")
        return

    # ---------------- demo mode
    dfk = df[df.fst.between(1, 6)]
    if not a.dry_run:
        print("\nChecking whether the image hosts are still reachable:")
        alive = live_hosts(dfk, seed=a.seed)
        dfk = dfk[dfk.url.map(host_of).isin(alive)]
    if check:
        print("\nChecking F17k labels against image URLs (dropping images whose source-atlas folder contradicts the label):")
    # (disease name, candidate images, number needed, is attention-check image)
    groups = []
    acne = dfk[dfk.label.str.lower().str.contains("acne", na=False)]
    acne = keep_consistent(acne, lambda r: "Acne", check, "Acne (attention check)")
    groups.append(("Acne", acne.sort_values("fst", ascending=False), 1, 1))   # paper: the first image is always the acne image
    for dis, h in hits.items():
        if h:
            cand = keep_consistent(dfk[dfk.label.isin(h)], lambda r, d=dis: d, check, dis)
            groups.append((dis, interleave(cand, a.seed), a.per_class, 0))
    # In the paper, all images outside the 8 main conditions (80) come from the 36 auxiliary diseases; when the AI predicts "Other" it also suggests one of these 36 at random.
    # So "other diseases" are drawn only from these 36 (those found in F17k); acne is already the attention-check image and is not reused.
    aux_of = {l: aux for aux in AUXILIARY if aux != "Acne"
              for l in labels if l not in main_labels and "acne" not in l.lower() and matches(l, aux)}
    others = dfk[dfk.label.isin(aux_of)]
    if a.n_other and not others.empty:
        # At most 1 image per auxiliary disease, covering as many diseases as possible; sample a candidate pool first, then check it
        pool = others.sample(frac=1, random_state=a.seed).drop_duplicates("label").head(a.n_other * 4)
        pool = keep_consistent(pool, lambda r: aux_of[r.label], check, "auxiliary diseases (candidate pool)")
        groups.append(("Other", pool, a.n_other, 0))

    if a.dry_run:
        print("\nDownload plan (first few candidates per group; dead links are replaced by the next candidate during download):")
        for dis, cand, n, att in groups:
            show = cand.head(n)
            for r in show.itertuples(index=False):
                print(f"  {dis:<26} FST{int(r.fst)}  {r.label:<28} {r.url}")
        return

    img_dir = ROOT / "demo_app" / "static" / "images"
    got, short = [], []
    for dis, cand, n, att in groups:
        res = fetch_quota(cand, n, img_dir, a.workers)
        name = res.label.map(aux_of) if dis == "Other" else dis
        got.append(res.assign(disease=name, attention_check=att))
        if len(res) < n:
            short.append(f"{dis} ({len(res)}/{n})")
    plan = pd.concat(got, ignore_index=True)
    missing = [d for d, h in hits.items() if not h]
    rows = [{"image_id": f"f17k_{i:03d}", "file": f"{r.md5hash}.jpg", "disease": r.disease, "fst": int(r.fst),
             "attention_check": int(r.attention_check), "dls_pred_control": "",
             "credit": f"Fitzpatrick17k (CC BY-NC-SA 3.0) · {host_of(r.url)}"}
            for i, r in enumerate(plan.itertuples(index=False))]
    out = ROOT / "demo_app" / "data" / "images.csv"
    if not rows:
        print("Every download failed (network or source-site problem); the existing image list is left unchanged.")
        return
    if out.exists():
        out.rename(out.with_name(f"images_backup_{time.strftime('%Y%m%d_%H%M%S')}.csv"))
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nDownloaded {len(rows)} images → {out}")
    if short:
        print("Not enough public images available: " + ", ".join(short))
    if missing:
        print("No matching label in Fitzpatrick17k: " + ", ".join(missing))
    print("Next (optional): write predictions with the trained AI: python dls/predict.py --update-demo")


if __name__ == "__main__":
    main()
