"""
论文实验平台的课堂复刻版（store-and-forward 远程皮肤科模拟）

复刻了论文 Methods → Experimental interface 描述的完整流程：
  知情同意 + 7 道问卷 → 随机分组（对照/增强 AI × 两种按钮顺序）→ 说明页（"AI 并非完全准确"，不透露准确率）
  → 看图，自由输入最多 3 个鉴别诊断（带联想提示）+ 信心滑块 + 是否活检/转诊
  → 若 AI 建议不在你的鉴别诊断里，显示 AI 建议和三个按钮（顺序随分组变化）
  → 反馈页：正确答案 + 其他人最常见的错误答案（答对放烟花）
  → 10 张图后与皮肤科医生、全科医生、AI 的准确率对比，可继续作答
另加：/dashboard 实时汇总全班结果，/export.csv 导出数据（可直接交给 analysis/key_results.py 分析）。

运行：
  python demo_app/app.py                 # 本机：浏览器打开 http://127.0.0.1:5000
  python demo_app/app.py --lan           # 同一 Wi-Fi 的手机也能访问（需网络允许设备互访）
讲台演示时可以用网址参数固定分组，便于对比两种界面：
  http://127.0.0.1:5000/?dls=treatment&ui=control
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import threading
import time
import uuid
import zlib
from collections import Counter
from pathlib import Path

from flask import Flask, Response, redirect, render_template, request, send_file, session, url_for

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
sys.path.insert(0, str(ROOT))
from common.diseases import AUXILIARY, MAIN8, SUGGESTIONS, canonical, in_differential, matches  # noqa: E402

DATA = BASE / "data"
IMG_DIR = BASE / "static" / "images"
IMAGES_CSV = Path(os.environ.get("DEMO_IMAGES", DATA / "images.csv"))
LOG = Path(os.environ.get("DEMO_LOG", DATA / "responses.csv"))
N_TRIALS = int(os.environ.get("DEMO_TRIALS", 10))
CONTROL_ACC = 0.47      # 没有真实模型预测时，模拟对照 AI 的准确率
FLIP = 0.65             # Wizard of Oz：错误预测中改为正确的比例
PAPER_REF = [("Board-certified dermatologists (paper)", 0.38), ("Primary care physicians (paper)", 0.19)]
FST_COLORS = {1: "#f3d9c7", 2: "#e7bf9f", 3: "#cfa07c", 4: "#a87753", 5: "#7b5034", 6: "#4b2f1f"}

FIELDS = ["participant_id", "physician_type", "years", "patient_mix", "difficulty_white", "difficulty_nonwhite",
          "sufficient_training", "country", "dls_condition", "interface", "trial", "image_id", "disease", "fst",
          "attention_check", "dx1", "dx2", "dx3", "confidence", "biopsy", "refer_derm", "top1_correct",
          "top3_correct", "dls_suggestion", "dls_correct", "choice", "top1_after", "top4_after", "seconds",
          "timestamp"]

app = Flask(__name__)
app.secret_key = os.environ.get("DEMO_SECRET", "classroom-demo-only")
LOCK = threading.Lock()
STATE: dict[str, dict] = {}


# ------------------------------------------------------------------ 图片与 AI 建议
def _u(key: str) -> float:
    """确定性的 [0,1) 随机数：同一张图对所有人给出同样的 AI 建议（与论文一致）。"""
    return (zlib.crc32(key.encode("utf-8")) % 1_000_003) / 1_000_003


def _to9(label: str) -> str:
    return next((k for k in MAIN8 if matches(label, k)), "Other")


def load_images() -> list[dict]:
    with open(IMAGES_CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    classes = list(MAIN8) + ["Other"]
    out = []
    for r in rows:
        iid, disease = r["image_id"].strip(), r["disease"].strip()
        true9 = _to9(disease)
        pred = (r.get("dls_pred_control") or "").strip()
        if pred:
            pred = _to9(pred)                                   # 真实模型的预测（dls/predict.py 写入）
        elif _u(iid + "|ctrl") < CONTROL_ACC:
            pred = true9                                        # 否则按 47% 准确率模拟
        else:
            wrong = [c for c in classes if c != true9]
            pred = wrong[int(_u(iid + "|wrong") * len(wrong))]
        pred_t = true9 if (pred != true9 and _u(iid + "|woz") < FLIP) else pred
        aux = AUXILIARY[int(_u(iid + "|aux") * len(AUXILIARY))]  # 预测 "Other" 时从 36 种辅助疾病中随机给一个
        file = (r.get("file") or "").strip()
        out.append({
            "image_id": iid, "disease": disease, "fst": int(float(r.get("fst") or 0) or 0),
            "attention": str(r.get("attention_check", "0")).strip() in ("1", "true", "True"),
            "file": file if file and (IMG_DIR / file).exists() else "",
            "credit": r.get("credit", ""),
            "suggestion": {"control": aux if pred == "Other" else pred,
                           "treatment": aux if pred_t == "Other" else pred_t},
        })
    return out


IMAGES = load_images()
BY_ID = {im["image_id"]: im for im in IMAGES}


# ------------------------------------------------------------------ 记录
def log_row(row: dict):
    with LOCK:
        new = not LOG.exists()
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in FIELDS})


def read_log() -> list[dict]:
    if not LOG.exists():
        return []
    with LOCK, open(LOG, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def most_common_wrong(image_id: str) -> str:
    c = Counter(canonical(r["dx1"]) for r in read_log()
                if r["image_id"] == image_id and r["top1_correct"] == "0" and r["dx1"].strip())
    return c.most_common(1)[0][0] if c else "—"


# ------------------------------------------------------------------ 页面
def current():
    pid = session.get("pid")
    return STATE.get(pid) if pid else None


@app.route("/")
def index():
    return render_template("index.html", dls=request.args.get("dls", ""), ui=request.args.get("ui", ""),
                           n=N_TRIALS, n_images=len(IMAGES))


@app.post("/start")
def start():
    f = request.form
    dls = f.get("dls_override") if f.get("dls_override") in ("control", "treatment") else random.choice(
        ["control", "treatment"])
    ui = f.get("ui_override") if f.get("ui_override") in ("control", "treatment") else random.choice(
        ["control", "treatment"])
    first = [im["image_id"] for im in IMAGES if im["attention"]][:1]
    rest = [im["image_id"] for im in IMAGES if im["image_id"] not in first]
    random.shuffle(rest)
    pid = "S" + uuid.uuid4().hex[:8]
    STATE[pid] = {
        "pid": pid, "dls": dls, "ui": ui, "order": first + rest, "i": 0, "pending": None, "last": None,
        "results": [], "summary_shown": False,
        "survey": {k: f.get(k, "") for k in ("physician_type", "years", "patient_mix", "difficulty_white",
                                              "difficulty_nonwhite", "sufficient_training", "country")},
    }
    session["pid"] = pid
    return redirect(url_for("instructions"))


@app.route("/instructions")
def instructions():
    st = current()
    if not st:
        return redirect(url_for("index"))
    return render_template("instructions.html", n=N_TRIALS)


@app.route("/trial", methods=["GET", "POST"])
def trial():
    st = current()
    if not st:
        return redirect(url_for("index"))
    if st["i"] >= len(st["order"]):
        return redirect(url_for("summary"))
    im = BY_ID[st["order"][st["i"]]]
    if request.method == "GET":
        st["t0"] = time.time()
        return render_template("trial.html", im=im, k=st["i"] + 1, n=N_TRIALS, labels=SUGGESTIONS,
                               is_bcd=st["survey"]["physician_type"] == "BCD")
    f = request.form
    dx = [f.get(f"dx{j}", "").strip() for j in (1, 2, 3)]
    if not dx[0]:
        return redirect(url_for("trial"))
    sugg = im["suggestion"][st["dls"]]
    rec = {
        "participant_id": st["pid"], **st["survey"], "dls_condition": st["dls"], "interface": st["ui"],
        "trial": st["i"] + 1, "image_id": im["image_id"], "disease": im["disease"], "fst": im["fst"],
        "attention_check": int(im["attention"]), "dx1": dx[0], "dx2": dx[1], "dx3": dx[2],
        "confidence": f.get("confidence", "50"), "biopsy": int(bool(f.get("biopsy"))),
        "refer_derm": "" if st["survey"]["physician_type"] == "BCD" else int(bool(f.get("refer"))),
        "top1_correct": int(matches(dx[0], im["disease"])),
        "top3_correct": int(in_differential(dx, im["disease"])),
        "dls_suggestion": sugg, "dls_correct": int(matches(sugg, im["disease"])),
        "seconds": round(time.time() - st.get("t0", time.time()), 1),
    }
    st["pending"] = rec
    if im["attention"] or in_differential(dx, sugg):          # AI 建议已在鉴别诊断中 → 不显示建议页
        return finish_trial(st, "same")
    return redirect(url_for("suggest"))


def finish_trial(st, choice: str):
    rec = st["pending"]
    sugg_ok = rec["dls_correct"] == 1
    rec["choice"] = choice
    rec["top1_after"] = int(sugg_ok) if choice == "top" else rec["top1_correct"]
    rec["top4_after"] = int(rec["top3_correct"] or (choice in ("top", "include") and sugg_ok))
    rec["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    log_row(rec)
    st["results"].append(rec)
    st["last"], st["pending"] = rec, None
    st["i"] += 1
    return redirect(url_for("feedback"))


@app.route("/suggest", methods=["GET", "POST"])
def suggest():
    st = current()
    if not st or not st["pending"]:
        return redirect(url_for("trial"))
    if request.method == "POST":
        choice = request.form.get("choice")
        if choice not in ("top", "include", "keep"):
            return redirect(url_for("suggest"))
        return finish_trial(st, choice)
    sugg = st["pending"]["dls_suggestion"]
    buttons = [("top", f"Update my top prediction with {sugg}"),
               ("include", f"Update my differential to include {sugg}"),
               ("keep", "Keep my differential")]
    if st["ui"] == "treatment":                                # 实验组：按钮顺序反过来
        buttons = buttons[::-1]
    im = BY_ID[st["pending"]["image_id"]]
    return render_template("suggest.html", im=im, rec=st["pending"], buttons=buttons, sugg=sugg)


@app.route("/feedback")
def feedback():
    st = current()
    if not st or not st["last"]:
        return redirect(url_for("trial"))
    rec = st["last"]
    im = BY_ID[rec["image_id"]]
    final = [rec["dx1"], rec["dx2"], rec["dx3"]]
    if rec["choice"] == "top":
        final = [rec["dls_suggestion"]] + final
    elif rec["choice"] == "include":
        final = final + [rec["dls_suggestion"]]
    nxt = url_for("summary") if (st["i"] >= N_TRIALS and not st["summary_shown"]) else url_for("trial")
    return render_template("feedback.html", rec=rec, im=im, final=[d for d in final if d],
                           correct=bool(rec["top4_after"]), wrong=most_common_wrong(rec["image_id"]), nxt=nxt,
                           k=st["i"], n=N_TRIALS)


@app.route("/summary")
def summary():
    st = current()
    if not st:
        return redirect(url_for("index"))
    st["summary_shown"] = True
    res = [r for r in st["results"] if not r["attention_check"]]
    n = len(res)

    def rate(key):
        return sum(int(r[key]) for r in res) / n if n else 0.0

    bars = [("You · top-3 (before AI)", rate("top3_correct"), "you"),
            ("You · top-4 (with AI)", rate("top4_after"), "you")]
    bars += [(lab, v, "ref") for lab, v in PAPER_REF]
    bars.append((f"AI model ({'enhanced' if st['dls'] == 'treatment' else 'control'}) on your images",
                 rate("dls_correct"), "ai"))
    return render_template("summary.html", bars=bars, n=n, more=st["i"] < len(st["order"]))


@app.route("/dashboard")
def dashboard():
    rows = [r for r in read_log() if r["attention_check"] != "1"]

    def mean(rs, key):
        v = [int(r[key]) for r in rs if r.get(key, "") != ""]
        return (sum(v) / len(v), len(v)) if v else (None, 0)

    light = [r for r in rows if r["fst"] and int(float(r["fst"])) <= 4]
    dark = [r for r in rows if r["fst"] and int(float(r["fst"])) >= 5]
    skin = [{"metric": lab, "light": mean(light, key), "dark": mean(dark, key)}
            for lab, key in (("Top-1 · before AI", "top1_correct"), ("Top-3 · before AI", "top3_correct"),
                             ("Top-1 · with AI", "top1_after"), ("Top-4 · with AI", "top4_after"))]
    shown = [r for r in rows if r["choice"] in ("top", "include", "keep")]
    ui = []
    for c, lab in (("top", "Update my top prediction"), ("include", "Update my differential to include"),
                   ("keep", "Keep my differential")):
        row = {"metric": lab}
        for g in ("control", "treatment"):
            s = [r for r in shown if r["interface"] == g]
            row[g] = ((sum(r["choice"] == c for r in s) / len(s)) if s else None, len(s))
        ui.append(row)
    dls = []
    for g in ("control", "treatment"):
        s = [r for r in rows if r["dls_condition"] == g]
        dls.append({"metric": "Enhanced AI (~84%)" if g == "treatment" else "Control AI (~47%)",
                    "before": mean(s, "top1_correct"), "after": mean(s, "top1_after")})
    return render_template("dashboard.html", n_people=len({r["participant_id"] for r in rows}), n_rows=len(rows),
                           skin=skin, ui=ui, dls=dls, types=Counter(r["physician_type"] for r in rows),
                           log_name=LOG.name)


@app.route("/export.csv")
def export():
    if not LOG.exists():
        return Response("participant_id\n", mimetype="text/csv")
    return send_file(LOG, mimetype="text/csv", as_attachment=True, download_name="classroom_responses.csv")


@app.post("/admin/new-session")
def new_session():
    """开始新一轮课堂数据：旧记录改名备份，不删除。"""
    if LOG.exists():
        LOG.rename(LOG.with_name(f"responses_backup_{time.strftime('%Y%m%d_%H%M%S')}.csv"))
    return redirect(url_for("dashboard"))


@app.route("/img/<image_id>")
def img(image_id):
    im = BY_ID.get(image_id)
    if not im:
        return Response(status=404)
    if im["file"]:
        return send_file(IMG_DIR / im["file"])
    color = FST_COLORS.get(im["fst"], "#c9c2b8")
    ink = "#ffffff" if im["fst"] >= 4 else "#2b2b2b"
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480" viewBox="0 0 640 480">
<rect width="640" height="480" fill="{color}"/>
<circle cx="320" cy="215" r="70" fill="none" stroke="{ink}" stroke-opacity=".35" stroke-width="3" stroke-dasharray="6 8"/>
<text x="320" y="330" font-family="system-ui,sans-serif" font-size="26" fill="{ink}" text-anchor="middle">Placeholder · {image_id} · FST {im['fst']}</text>
<text x="320" y="366" font-family="system-ui,sans-serif" font-size="18" fill="{ink}" fill-opacity=".8" text-anchor="middle">Replace with a clinical photo (see README)</text>
</svg>"""
    return Response(svg, mimetype="image/svg+xml")


def main():
    ap = argparse.ArgumentParser(description="课堂演示：论文实验平台复刻版")
    ap.add_argument("--lan", action="store_true", help="监听 0.0.0.0，让同一网络的手机访问")
    ap.add_argument("--port", type=int, default=5000)
    a = ap.parse_args()
    n_real = sum(1 for im in IMAGES if im["file"])
    print(f"已载入 {len(IMAGES)} 张图片（{n_real} 张真实图片，其余为占位图）；记录写入 {LOG}")
    print(f"实验页面：http://127.0.0.1:{a.port}/    实时汇总：http://127.0.0.1:{a.port}/dashboard")
    app.run(host="0.0.0.0" if a.lan else "127.0.0.1", port=a.port, debug=False)


if __name__ == "__main__":
    main()
