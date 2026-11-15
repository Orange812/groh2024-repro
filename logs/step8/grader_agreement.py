"""Validate the replica's automatic grading (common/diseases.matches) against the author's manual grading.
Run from groh2024-repro/: python logs/step8/grader_agreement.py"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "groh2024_repro_kit"))
from common.diseases import matches  # noqa: E402

df = pd.read_csv(ROOT / "work/ResearchBox 1802/Data/Experimental Data.csv.zip")
d = df[(df.guess_round == 1) & df.differential_1.notna() & (df.differential_1 != "test") & df.consensus_label.notna()].copy()
MAP = {"Atopic Dermatitis": "Atopic dermatitis", "CTCL": "CTCL", "Dermatomyositis": "Dermatomyositis",
       "Lichen Planus": "Lichen planus", "Lyme": "Lyme disease", "Pityriasis Rosea": "Pityriasis rosea",
       "Pityriasis Rubra Pilaris": "Pityriasis rubra pilaris", "Secondary Syphilis": "Secondary syphilis"}
d["label"] = [MAP.get(s, c) for s, c in zip(d.consensus_label_simplified, d.consensus_label)]
d["main"] = d.consensus_label_simplified.isin(MAP)
cache = {}
d["replica"] = [cache.setdefault((a, l), int(matches(a, l))) for a, l in zip(d.differential_1, d.label)]
d["author"] = d.diff1.astype(int)
for name, s in [("8 main conditions", d[d.main]), ("other 36 conditions", d[~d.main])]:
    print(f"{name}: n={len(s)} agreement={(s.replica == s.author).mean():.4f} | "
          f"replica correct/author wrong: {((s.replica == 1) & (s.author == 0)).sum()} | "
          f"replica wrong/author correct: {((s.replica == 0) & (s.author == 1)).sum()}")
s = d[d.main]
print("\nTop disagreements (replica wrong, author correct):")
print(s[(s.replica == 0) & (s.author == 1)].groupby(["label", "differential_1"]).size()
      .sort_values(ascending=False).head(10).to_string())
