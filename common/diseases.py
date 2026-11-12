"""
疾病名称表 + 自由作答的判分逻辑（分析脚本和演示程序共用）

论文做法（Methods → Annotating participants' differential diagnoses）：
  - 参与者用自由文本输入诊断，系统根据 445 个诊断名称做联想提示；
  - 共收集到 2,348 个不同的文本字符串，作者逐条人工审核，
    把拼写错误、简写、同义词（如 mycosis fungoides = CTCL）都判为正确。

这里用「同义词表 + 模糊匹配」近似这一人工判分过程，演示足够用；
正式复现请使用作者 Experimental Data.csv 里已经判好的结果。
"""
from __future__ import annotations

import difflib
import re

# 8 种重点疾病（论文 284 张图片，每种 ≥29 张）及同义词
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

# 论文列出的 36 种辅助疾病（AI 预测为 "Other" 时，从中随机挑一个作为建议）
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

# 其他常见鉴别诊断，只用于输入框的联想提示（论文用了 445 个；这里给一个精简版）
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
    """拼写错误容忍：整体足够相似，且逐词比较时短词（≤3 个字母，如 t / b）必须完全一致，
    避免把 "cutaneous b-cell lymphoma" 误判成 "cutaneous t-cell lymphoma"。"""
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
    """判断自由作答 `answer` 是否等于参考诊断 `label`（容忍同义词、简写、拼写错误）。"""
    if not answer or not label:
        return False
    syns = _synonyms(label)
    for cand in _candidates(answer):
        for syn in syns:
            if cand == syn:
                return True
            # 多词同义词：作为完整短语出现即可（例如 "secondary syphilis rash"）
            if " " in syn and re.search(rf"(^|\s){re.escape(syn)}(\s|$)", cand):
                return True
        # 拼写错误：与任一同义词足够相似（短缩写不做模糊匹配，避免误判）
        if len(cand) >= 6:
            for syn in syns:
                if len(syn) >= 6 and _fuzzy_ok(cand, syn, fuzzy_cutoff):
                    return True
    return False


def canonical(answer: str) -> str:
    """把一个自由作答归一到已知标签（找不到就返回清洗后的原文），用于统计"最常见错误答案"。"""
    if not answer:
        return ""
    for label in ALL_LABELS:
        if matches(answer, label):
            return label
    return _norm(answer)


def in_differential(dx_list, label: str) -> bool:
    return any(matches(d, label) for d in dx_list if d)
