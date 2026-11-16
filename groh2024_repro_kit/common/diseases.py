"""
Disease name tables + grading logic for free-text answers (shared by the analysis scripts and the demo app)

What the paper did (Methods → Annotating participants' differential diagnoses):
  - participants typed diagnoses as free text, with predictive text based on 445 diagnosis names;
  - 2,348 distinct text strings were collected and the authors reviewed each one by hand,
    counting misspellings, abbreviations and synonyms (e.g. mycosis fungoides = CTCL) as correct.

Here a synonym table + fuzzy matching approximates that manual grading, which is good enough for the demo;
for the actual reproduction, use the grades already stored in the authors' Experimental Data.csv.
"""
from __future__ import annotations

import difflib
import re

# The 8 main conditions (284 images in the paper, ≥29 per condition) and their synonyms
MAIN8 = {
    "Atopic dermatitis": ["atopic dermatitis", "atopic dermatitis (eczema)", "eczema", "atopic eczema",
                          "dermatitis atopica", "dermatite atopica", "ad"],
    "CTCL": ["cutaneous t-cell lymphoma", "cutaneous t cell lymphoma", "ctcl", "mycosis fungoides",
             "t-cell lymphoma", "t cell lymphoma", "cutaneous lymphoma", "lymphoma", "mf",
             "micosis fungoide", "micose fungoide", "linfoma", "lymphome"],
    "Dermatomyositis": ["dermatomyositis", "dermatomiositis", "dermatomiosite", "juvenile dermatomyositis"],
    "Lichen planus": ["lichen planus", "lichen ruber planus", "linear lichen planus", "liquen plano",
                      "lichen plan", "lichen plane", "lp"],
    "Lyme disease": ["lyme disease", "lyme", "erythema migrans", "erythema chronicum migrans",
                     "erythema migrans (lyme)", "lyme (erythema migrans)", "borreliosis"],
    "Pityriasis rosea": ["pityriasis rosea", "pitiriasis rosada", "pityriasis rosea gibert", "pr"],
    "Pityriasis rubra pilaris": ["pityriasis rubra pilaris", "pityriasis rubra pilaris (prp)", "prp"],
    "Secondary syphilis": ["secondary syphilis", "syphilis", "lues", "sifilis secundaria", "syphilis secondary"],
}

# The 36 auxiliary diseases listed in the paper (when the AI predicts "Other", one is picked at random as the suggestion)
AUXILIARY = [
    "Scleroderma", "Lupus erythematosus", "Acne", "Vitiligo", "Rosacea", "Tungiasis",
    "Urticaria pigmentosa", "Sarcoidosis", "Cheilitis", "Calcinosis cutis", "Allergic contact dermatitis",
    "Factitial dermatitis", "Fixed drug eruption", "Granuloma annulare", "Keloid", "Keratosis pilaris",
    "Acanthosis nigricans", "Rhinophyma", "Necrobiosis lipoidica", "Tick bite",
    "Confluent and reticulated papillomatosis", "Psoriasis", "Scabies", "Livedo reticularis", "Urticaria",
    "Stevens-Johnson syndrome", "Stasis edema", "Seborrheic dermatitis", "Erythema nodosum",
    "Erythema elevatum diutinum", "Lichen simplex chronicus", "Neurotic excoriations",
    "Hidradenitis suppurativa", "Nematode infection", "Lichen amyloidosis", "Xanthoma",
]

# Other common differential diagnoses, used only for the input box's predictive text (the paper used 445; this is a short list)
EXTRA_SUGGESTIONS = [
    "Tinea corporis", "Tinea versicolor", "Nummular eczema", "Contact dermatitis", "Drug eruption",
    "Erythema multiforme", "Discoid lupus erythematosus", "Morphea", "Guttate psoriasis", "Parapsoriasis",
    "Cellulitis", "Erythema annulare centrifugum", "Impetigo", "Folliculitis", "Lichen sclerosus",
    "Prurigo nodularis", "Vasculitis", "Pemphigus vulgaris", "Bullous pemphigoid", "Dermatitis herpetiformis",
    "Melanoma", "Basal cell carcinoma", "Squamous cell carcinoma", "Actinic keratosis", "Seborrheic keratosis",
    "Post-inflammatory hyperpigmentation", "Melasma", "Pityriasis alba", "Lichen nitidus", "Lichen striatus",
    "Granuloma faciale", "Sweet syndrome", "Pyoderma gangrenosum", "Herpes zoster", "Molluscum contagiosum",
    "Viral exanthem", "Pityriasis lichenoides", "Mastocytosis", "Kaposi sarcoma", "Leprosy",
    "Cutaneous B-cell lymphoma", "Langerhans cell histiocytosis", "Pellagra", "Photodermatitis",
    "Polymorphous light eruption", "Erythrasma", "Intertrigo", "Candidiasis", "Dermatophytosis",
    "Perioral dermatitis", "Keratoderma", "Ichthyosis", "Darier disease", "Hailey-Hailey disease",
]

ALL_LABELS = list(MAIN8) + AUXILIARY + EXTRA_SUGGESTIONS


def _load_suggestions() -> list[str]:
    """Predictive text for the input box: prefer the 445 diagnosis names from the authors' original experiment (dx_options_445.txt).

    Source: possible_options in cell 89 (In [60]) of Analysis.ipynb, ResearchBox #1802 (CC BY 4.0),
    matching the paper's Methods ("predictive text based on 445 possible diagnoses"); the original list has 3 duplicates, leaving 442.
    If the file is missing, fall back to the short list above. This affects only the predictive text, not grading."""
    from pathlib import Path
    p = Path(__file__).with_name("dx_options_445.txt")
    if not p.exists():
        return ALL_LABELS
    seen, out = set(), []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


SUGGESTIONS = _load_suggestions()


def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.replace("’", "'").replace("_", " ")
    s = re.sub(r"[^\w\s\-()/']", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _candidates(answer: str) -> list[str]:
    """'lyme (erythema migrans)' -> ['lyme (erythema migrans)', 'lyme', 'erythema migrans']"""
    a = _norm(answer)
    if not a:
        return []
    cands = {a}
    inner = re.findall(r"\(([^)]*)\)", a)
    outer = re.sub(r"\([^)]*\)", " ", a)
    for part in [outer] + inner:
        for piece in re.split(r"/|,|;", part):
            piece = re.sub(r"\s+", " ", piece).strip(" -")
            if piece:
                cands.add(piece)
    return sorted(cands, key=len, reverse=True)


def _synonyms(label: str) -> list[str]:
    syns = MAIN8.get(label)
    if syns:
        return [_norm(s) for s in syns]
    return [_norm(label)]


def _tokens(s: str) -> list[str]:
    return [t for t in re.split(r"[\s\-]+", s) if t]


def _fuzzy_ok(cand: str, syn: str, cutoff: float) -> bool:
    """Typo tolerance: similar enough overall, and in a word-by-word comparison short words (≤3 letters, e.g. t / b) must match exactly,
    so that "cutaneous b-cell lymphoma" is not graded as "cutaneous t-cell lymphoma"."""
    if difflib.SequenceMatcher(None, cand, syn).ratio() < cutoff:
        return False
    ct, st = _tokens(cand), _tokens(syn)
    if len(ct) != len(st):
        return True
    for a, b in zip(ct, st):
        if min(len(a), len(b)) <= 3:
            if a != b:
                return False
        elif difflib.SequenceMatcher(None, a, b).ratio() < 0.7:
            return False
    return True


def matches(answer: str, label: str, fuzzy_cutoff: float = 0.86) -> bool:
    """Decide whether the free-text answer `answer` equals the reference diagnosis `label` (tolerating synonyms, abbreviations and typos)."""
    if not answer or not label:
        return False
    syns = _synonyms(label)
    for cand in _candidates(answer):
        for syn in syns:
            if cand == syn:
                return True
            # Multi-word synonym: it is enough for it to appear as a whole phrase (e.g. "secondary syphilis rash")
            if " " in syn and re.search(rf"(^|\s){re.escape(syn)}(\s|$)", cand):
                return True
        # Typos: similar enough to any synonym (short abbreviations are not fuzzy-matched, to avoid false matches)
        if len(cand) >= 6:
            for syn in syns:
                if len(syn) >= 6 and _fuzzy_ok(cand, syn, fuzzy_cutoff):
                    return True
    return False


def canonical(answer: str) -> str:
    """Map a free-text answer to a known label (or return the cleaned text if none matches); used to count the "most common incorrect answer"."""
    if not answer:
        return ""
    for label in ALL_LABELS:
        if matches(answer, label):
            return label
    return _norm(answer)


def in_differential(dx_list, label: str) -> bool:
    return any(matches(d, label) for d in dx_list if d)
