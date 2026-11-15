"""End-to-end protocol test of the demo_app replica with simulated participants (Flask test client).
Writes to DEMO_LOG (set by caller) so the real classroom log is untouched."""
import os, random, re, sys
KIT = sys.argv[1]
sys.path.insert(0, KIT); sys.path.insert(0, os.path.join(KIT, "demo_app"))
import app as A
from common.diseases import MAIN8, SUGGESTIONS, in_differential

random.seed(42)
checks = {}
def check(name, cond):
    checks.setdefault(name, True)
    checks[name] = checks[name] and bool(cond)

ACC = {"BCD": 0.45, "PCP": 0.25, "Student": 0.10}
people = [("BCD", "control"), ("BCD", "treatment"), ("PCP", "control"), ("PCP", "treatment"),
          ("BCD", ""), ("PCP", ""), ("Student", ""), ("Student", "")]
seen_orders, button_orders = [], {}
for k, (ptype, ui) in enumerate(people):
    c = A.app.test_client()
    html = c.get("/").get_data(as_text=True)
    check("Home page: 7 survey questions", len(re.findall(r"<label>\d\.", html)) == 7)
    c.post("/start", data={"physician_type": ptype, "years": "5-10", "patient_mix": random.choice(
        ["Mostly white patients", "All patients with skin of color (non-white)"]), "sufficient_training": "Yes",
        "country": "Canada", "dls_override": "", "ui_override": ui})
    pid = list(A.STATE)[-1]; st = A.STATE[pid]
    if ui: check("Lectern mode: URL parameters can fix the interface arm", st["ui"] == ui)
    ins = c.get("/instructions").get_data(as_text=True)
    check("Instructions: 'AI is not perfectly accurate', accuracy not disclosed", "not perfectly accurate" in ins and "%" not in ins)
    seen_orders.append(tuple(st["order"][1:6]))
    for t in range(10):
        im = A.BY_ID[st["order"][st["i"]]]
        page = c.get("/trial").get_data(as_text=True)
        if t == 0:
            check("Image 1 is always the acne attention-check image", im["attention"] and im["disease"] == "Acne")
        check("Diagnosis boxes: 3 free-text inputs + the original placeholder text", all(p in page for p in
              ("Type leading diagnosis", "Type secondary differential diagnosis", "Type tertiary differential diagnosis")))
        check("Predictive text: 442 diagnoses (the authors' original list)", page.count("<option value=") == len(SUGGESTIONS) == 442)
        check("Confidence slider defaults to 50%", 'name="confidence" min="0" max="100" step="5" value="50"' in page)
        check("Referral question asked of everyone except BCDs", ("second opinion" in page) == (ptype != "BCD"))
        right = random.random() < ACC[ptype]
        dx1 = im["disease"] if right else random.choice(SUGGESTIONS)
        r = c.post("/trial", data={"dx1": dx1, "dx2": random.choice(SUGGESTIONS), "confidence": str(random.choice([50, 50, 70, 30])),
                                   "biopsy": "on" if random.random() < .2 else "", "refer": "on" if random.random() < .3 else ""})
        sugg = im["suggestion"][st["dls"]]
        if r.headers["Location"].endswith("/suggest"):
            s = c.get("/suggest").get_data(as_text=True)
            labels = re.findall(r'value="(top|include|keep)"', s)
            button_orders.setdefault(st["ui"], set()).add(tuple(labels))
            check("Suggestion page: the three button labels match the paper", f"Update my top prediction with {sugg}" in s and
                  f"Update my differential to include {sugg}" in s and "Keep my differential" in s)
            c.post("/suggest", data={"choice": random.choice(["top", "include", "keep"])})
        else:
            rec = st["last"]
            check("AI suggestion already in the differential → suggestion page skipped", im["attention"] or in_differential([rec["dx1"], rec["dx2"], rec["dx3"]], sugg))
        fb = c.get("/feedback").get_data(as_text=True)
        check("Feedback page: reference diagnosis + most common incorrect answer", "Reference diagnosis" in fb and "Most common incorrect answer" in fb)
    summ = c.get("/summary").get_data(as_text=True)
    check("After 10 images: comparison with BCDs, PCPs and the AI + option to continue", "dermatologists" in summ and "AI model" in summ and "Keep diagnosing" in summ)

check("Remaining images in random order (different for each person)", len(set(seen_orders)) == len(seen_orders))
check("Control interface button order: top → include → keep", button_orders.get("control") == {("top", "include", "keep")})
check("Treatment interface reverses it: keep → include → top", button_orders.get("treatment") == {("keep", "include", "top")})
# DLS suggestions on the image set (deterministic per image, same for everyone)
imgs = [im for im in A.IMAGES if not im["attention"]]
acc = {g: sum(A.matches(im["suggestion"][g], im["disease"]) for im in imgs) / len(imgs) for g in ("control", "treatment")}
aux_ok = all(im["suggestion"]["control"] in [m for m in MAIN8] + A.AUXILIARY for im in imgs)
check("AI suggestions come only from the 8 conditions + 36 auxiliary diseases (a random auxiliary disease when predicting Other)", aux_ok)
dash = A.app.test_client().get("/dashboard"); exp = A.app.test_client().get("/export.csv")
check("/dashboard and /export.csv work", dash.status_code == 200 and exp.status_code == 200 and b"participant_id" in exp.data)

for k, v in checks.items():
    print(("PASS " if v else "FAIL ") + k)
acc9 = {g: sum(im["pred9"][g] == A._to9(im["disease"]) for im in imgs) / len(imgs) for g in ("control", "treatment")}
check("AI 9-class accuracy by quota: control = 47%, enhanced ≈ 47% + 53% × 65%", abs(acc9["control"] - 0.47) < 0.5 / len(imgs) + 0.02
      and abs(acc9["treatment"] - (acc9["control"] + (1 - acc9["control"]) * 0.65)) < 1.0 / len(imgs) + 0.02)
print(f"\nAI on these {len(imgs)} images: 9-class accuracy control {acc9['control']:.0%}, enhanced {acc9['treatment']:.0%}; "
      f"suggestion shown matches the reference diagnosis: control {acc['control']:.0%}, enhanced {acc['treatment']:.0%} (for Other-class images the suggestion is a random auxiliary disease)")
for k, v in list(checks.items())[-1:]:
    print(("PASS " if v else "FAIL ") + k)
print("Log:", A.LOG, "| rows", sum(1 for _ in open(A.LOG)) - 1)
