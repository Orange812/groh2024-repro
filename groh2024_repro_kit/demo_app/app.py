"""
Classroom replica of the paper's experiment platform (a store-and-forward teledermatology simulation)

It replicates the full flow described in the paper's Methods → Experimental interface:
  consent + 7 survey questions → random assignment (control/enhanced AI × two button orders) → instructions ("the AI is not perfectly accurate"; accuracy not disclosed)
  → view the image, type up to 3 differential diagnoses as free text (with predictive text) + confidence slider + biopsy/referral
  → if the AI suggestion is not in your differential, show it with three buttons (their order depends on the arm)
  → feedback page: correct answer + the most common incorrect answer from others (fireworks when you are right)
  → after 10 images, a comparison with the accuracy of dermatologists, primary care physicians and the AI; you may keep going
Extras: /dashboard shows live class-wide results; /export.csv exports the data (can be fed straight into analysis/key_results.py).

Run:
  python demo_app/app.py                 # this computer: open http://127.0.0.1:5000 in a browser
  python demo_app/app.py --lan           # phones on the same Wi-Fi can connect too (the network must allow devices to reach each other)
For a lectern demo, URL parameters can fix the assignment, to compare the two interfaces:
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
CONTROL_ACC = 0.47      # simulated control-AI accuracy when no real model predictions exist
FLIP = 0.65             # Wizard of Oz: share of wrong predictions changed to correct
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


# ------------------------------------------------------------------ Images and AI suggestions
def _u(key: str) -> float:
    """Deterministic random number in [0,1): everyone gets the same AI suggestion for the same image (as in the paper)."""
    return (zlib.crc32(key.encode("utf-8")) % 1_000_003) / 1_000_003


def _to9(label: str) -> str:
    return next((k for k in MAIN8 if matches(label, k)), "Other")


def _is_attention(r: dict) -> bool:
    return str(r.get("attention_check", "0")).strip() in ("1", "true", "True")


def load_images() -> list[dict]:
    with open(IMAGES_CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    classes = list(MAIN8) + ["Other"]
    # Images without real model predictions: simulate exactly by quota rather than an independent coin flip per image.
    # With few images, independent draws vary a lot (e.g. 38% on 29 images); the quota guarantees exactly 47% are predicted correctly (9 classes).
    sim = sorted((r["image_id"].strip() for r in rows
                  if not (r.get("dls_pred_control") or "").strip() and not _is_attention(r)),
                 key=lambda i: _u(i + "|ctrl"))
    sim_correct = set(sim[:round(CONTROL_ACC * len(sim))])
    preds = {}
    for r in rows:
        iid, true9 = r["image_id"].strip(), _to9(r["disease"].strip())
        pred = (r.get("dls_pred_control") or "").strip()
        if pred:
            pred = _to9(pred)                                   # real model prediction (written by dls/predict.py)
        elif iid in sim_correct or _is_attention(r):
            pred = true9                                        # otherwise simulate with the 47% quota
        else:
            wrong = [c for c in classes if c != true9]
            pred = wrong[int(_u(iid + "|wrong") * len(wrong))]
        preds[iid] = (true9, pred)
    # Wizard of Oz: change exactly 65% (by quota) of the wrong predictions to the correct answer → 47% + 53% × 65% ≈ 81%
    wrong_ids = sorted((i for i, (t, p) in preds.items() if p != t), key=lambda i: _u(i + "|woz"))
    flipped = set(wrong_ids[:round(FLIP * len(wrong_ids))])
    out = []
    for r in rows:
        iid, disease = r["image_id"].strip(), r["disease"].strip()
        true9, pred = preds[iid]
        pred_t = true9 if iid in flipped else pred
        aux = AUXILIARY[int(_u(iid + "|aux") * len(AUXILIARY))]  # when predicting "Other", suggest one of the 36 auxiliary diseases at random
        file = (r.get("file") or "").strip()
        out.append({
            "image_id": iid, "disease": disease, "fst": int(float(r.get("fst") or 0) or 0),
            "attention": _is_attention(r),
            "pred9": {"control": pred, "treatment": pred_t},    # 9-class prediction (the basis the paper uses for AI accuracy)
            "file": file if file and (IMG_DIR / file).exists() else "",
            "credit": r.get("credit", ""),
            "suggestion": {"control": aux if pred == "Other" else pred,
                           "treatment": aux if pred_t == "Other" else pred_t},
        })
    return out


IMAGES = load_images()
BY_ID = {im["image_id"]: im for im in IMAGES}


# ------------------------------------------------------------------ Logging
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


# ------------------------------------------------------------------ Pages
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
    if im["attention"] or in_differential(dx, sugg):          # AI suggestion already in the differential → skip the suggestion page
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
    if st["ui"] == "treatment":                                # treatment interface: buttons in reverse order
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
    """Start a new round of class data: the old log is renamed as a backup, not deleted."""
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
    ap = argparse.ArgumentParser(description="Classroom demo: replica of the paper's experiment platform")
    ap.add_argument("--lan", action="store_true", help="listen on 0.0.0.0 so phones on the same network can connect")
    ap.add_argument("--port", type=int, default=5000)
    a = ap.parse_args()
    n_real = sum(1 for im in IMAGES if im["file"])
    print(f"Loaded {len(IMAGES)} images ({n_real} real images, the rest placeholders); logging to {LOG}")
    print(f"Experiment: http://127.0.0.1:{a.port}/    Live summary: http://127.0.0.1:{a.port}/dashboard")
    app.run(host="0.0.0.0" if a.lan else "127.0.0.1", port=a.port, debug=False)


if __name__ == "__main__":
    main()
