"""
Nutrition validator — IFCT 2017 + USDA grounded values per 100g.
Standalone, stdlib only.

Used by the grader to verify macro targets and meal plans against
the Indian Food Composition Tables 2017 (NIN Hyderabad) and USDA FoodData.
"""

from __future__ import annotations
import re
from difflib import get_close_matches
from typing import Optional


# ── IFCT 2017 + USDA per 100g (cooked unless noted) ──────────────────────────

NUTRITION_DB: dict[str, dict] = {
    # Legumes — IFCT 2017 Table A-19
    "rajma":         {"cal": 127, "protein": 8.7,  "carbs": 22.8, "fat": 0.5,  "src": "IFCT2017"},
    "chana":         {"cal": 164, "protein": 8.9,  "carbs": 27.4, "fat": 2.6,  "src": "IFCT2017"},
    "chole":         {"cal": 164, "protein": 8.9,  "carbs": 27.4, "fat": 2.6,  "src": "IFCT2017"},
    "moong dal":     {"cal": 105, "protein": 7.0,  "carbs": 19.0, "fat": 0.4,  "src": "IFCT2017"},
    "masoor dal":    {"cal": 116, "protein": 9.0,  "carbs": 20.0, "fat": 0.4,  "src": "IFCT2017"},
    "toor dal":      {"cal": 118, "protein": 6.8,  "carbs": 21.0, "fat": 0.4,  "src": "IFCT2017"},
    "urad dal":      {"cal": 106, "protein": 8.5,  "carbs": 18.4, "fat": 0.5,  "src": "IFCT2017"},
    # Dairy — IFCT 2017 Table A-08
    "paneer":        {"cal": 265, "protein": 18.3, "carbs": 1.2,  "fat": 20.8, "src": "IFCT2017"},
    "curd":          {"cal": 60,  "protein": 3.1,  "carbs": 4.7,  "fat": 3.3,  "src": "IFCT2017"},
    "hung curd":     {"cal": 98,  "protein": 10.0, "carbs": 5.0,  "fat": 4.0,  "src": "derived"},
    "greek yogurt":  {"cal": 59,  "protein": 10.0, "carbs": 3.6,  "fat": 0.4,  "src": "USDA"},
    "milk":          {"cal": 67,  "protein": 3.2,  "carbs": 4.4,  "fat": 4.1,  "src": "IFCT2017"},
    "skim milk":     {"cal": 34,  "protein": 3.4,  "carbs": 5.0,  "fat": 0.1,  "src": "IFCT2017"},
    "soy milk":      {"cal": 54,  "protein": 3.3,  "carbs": 6.3,  "fat": 1.8,  "src": "USDA"},
    # Soy — IFCT 2017 Table A-21
    "soya chunks":   {"cal": 345, "protein": 52.4, "carbs": 33.0, "fat": 0.5,  "src": "IFCT2017"},
    "tofu":          {"cal": 76,  "protein": 8.0,  "carbs": 1.9,  "fat": 4.8,  "src": "USDA"},
    "tempeh":        {"cal": 193, "protein": 19.0, "carbs": 9.4,  "fat": 11.0, "src": "USDA"},
    # Grains — IFCT 2017 Table A-01
    "brown rice":    {"cal": 111, "protein": 2.6,  "carbs": 23.0, "fat": 0.9,  "src": "IFCT2017"},
    "white rice":    {"cal": 130, "protein": 2.7,  "carbs": 28.0, "fat": 0.3,  "src": "IFCT2017"},
    "oats":          {"cal": 389, "protein": 16.9, "carbs": 66.3, "fat": 6.9,  "src": "IFCT2017"},
    "quinoa":        {"cal": 120, "protein": 4.4,  "carbs": 21.3, "fat": 1.9,  "src": "USDA"},
    "roti":          {"cal": 178, "protein": 5.8,  "carbs": 36.0, "fat": 1.1,  "src": "IFCT2017"},
    "poha":          {"cal": 108, "protein": 2.5,  "carbs": 23.0, "fat": 0.7,  "src": "IFCT2017"},
    # Vegetables — IFCT 2017 Table A-03
    "spinach":       {"cal": 26,  "protein": 2.9,  "carbs": 3.6,  "fat": 0.4,  "src": "IFCT2017"},
    "broccoli":      {"cal": 34,  "protein": 2.8,  "carbs": 6.6,  "fat": 0.4,  "src": "USDA"},
    "potato":        {"cal": 97,  "protein": 1.6,  "carbs": 22.6, "fat": 0.1,  "src": "IFCT2017"},
    # Fruits — IFCT 2017 Table A-04
    "banana":        {"cal": 89,  "protein": 1.1,  "carbs": 23.0, "fat": 0.3,  "src": "IFCT2017"},
    "apple":         {"cal": 52,  "protein": 0.3,  "carbs": 14.0, "fat": 0.2,  "src": "IFCT2017"},
    # Nuts & fats — IFCT 2017 Table A-20
    "peanut butter": {"cal": 588, "protein": 25.0, "carbs": 20.0, "fat": 50.0, "src": "USDA"},
    "almonds":       {"cal": 579, "protein": 21.0, "carbs": 22.0, "fat": 50.0, "src": "IFCT2017"},
    "ghee":          {"cal": 900, "protein": 0,    "carbs": 0,    "fat": 100.0, "src": "IFCT2017"},
    # Protein — IFCT 2017 Tables A-07, A-12
    "eggs":          {"cal": 143, "protein": 12.6, "carbs": 0.7,  "fat": 9.5,  "src": "IFCT2017"},
    "chicken breast":{"cal": 165, "protein": 31.0, "carbs": 0,    "fat": 3.6,  "src": "USDA"},
    "fish":          {"cal": 97,  "protein": 20.0, "carbs": 0,    "fat": 1.5,  "src": "IFCT2017"},
    "whey protein":  {"cal": 370, "protein": 80.0, "carbs": 8.0,  "fat": 4.5,  "src": "USDA"},
}


# ── Alias map (longest-match-first prevents substring bugs) ───────────────────

FOOD_ALIASES: dict[str, list[str]] = {
    "rajma":         ["rajma", "kidney beans", "red kidney beans"],
    "chana":         ["chana", "chickpeas cooked", "kabuli chana"],
    "chole":         ["chole", "chickpeas curry", "chana masala"],
    "moong dal":     ["moong dal", "mung dal", "green gram dal"],
    "masoor dal":    ["masoor dal", "red lentils"],
    "toor dal":      ["toor dal", "arhar dal", "pigeon pea"],
    "urad dal":      ["urad dal", "black gram dal"],
    "paneer":        ["paneer", "cottage cheese indian"],
    "greek yogurt":  ["greek yogurt", "greek yoghurt"],
    "hung curd":     ["hung curd", "strained curd", "chakka"],
    "curd":          ["curd", "dahi", "yogurt", "yoghurt", "plain yogurt"],
    "milk":          ["milk", "cow milk", "whole milk", "toned milk"],
    "skim milk":     ["skim milk", "skimmed milk"],
    "soy milk":      ["soy milk", "soya milk"],
    "soya chunks":   ["soya chunks", "soy chunks", "nutrela", "soya nuggets"],
    "tofu":          ["tofu", "bean curd"],
    "tempeh":        ["tempeh"],
    "brown rice":    ["brown rice", "whole grain rice"],
    "white rice":    ["white rice", "steamed rice", "basmati rice"],
    "oats":          ["oats", "rolled oats", "oatmeal"],
    "quinoa":        ["quinoa"],
    "roti":          ["roti", "chapati", "phulka"],
    "poha":          ["poha", "flattened rice", "beaten rice"],
    "spinach":       ["spinach", "palak"],
    "broccoli":      ["broccoli"],
    "potato":        ["potato", "aloo"],
    "banana":        ["banana", "kela"],
    "apple":         ["apple", "seb"],
    "peanut butter": ["peanut butter", "pb"],
    "almonds":       ["almonds", "badam"],
    "ghee":          ["ghee", "clarified butter"],
    "eggs":          ["eggs", "egg", "boiled egg", "scrambled egg"],
    "chicken breast":["chicken breast", "grilled chicken", "chicken"],
    "fish":          ["fish", "rohu", "salmon", "tilapia"],
    "whey protein":  ["whey protein", "whey", "protein shake", "protein powder"],
}

_COMPILED: list[tuple[str, re.Pattern]] = []


def _compile():
    global _COMPILED
    if _COMPILED:
        return
    entries = []
    for canonical, aliases in FOOD_ALIASES.items():
        for alias in aliases:
            pat = re.compile(r'\b' + re.escape(alias.lower()) + r'\b', re.I)
            entries.append((canonical, alias, pat))
    entries.sort(key=lambda x: -len(x[1]))
    _COMPILED = [(c, p) for c, _, p in entries]


def resolve_food(text: str) -> Optional[str]:
    _compile()
    tl = text.lower().strip()
    for canonical, pat in _COMPILED:
        if pat.search(tl):
            return canonical
    return None


def fuzzy_resolve(text: str, cutoff: float = 0.75) -> Optional[str]:
    clean = re.sub(r'\d+\s*(?:g|ml|gm|kg)?\s*', '', text.lower()).strip()
    all_aliases = [a for aliases in FOOD_ALIASES.values() for a in aliases]
    matches = get_close_matches(clean, all_aliases, n=1, cutoff=cutoff)
    if not matches:
        return None
    for canonical, aliases in FOOD_ALIASES.items():
        if matches[0] in aliases:
            return canonical
    return None


def parse_quantity(text: str) -> tuple[Optional[float], str]:
    text = text.strip()
    # PREFIX: "100g rajma"
    m = re.match(r'^([\d.]+)\s*(g|gm|grams?|ml|kg)\s+(.+)$', text, re.I)
    if m:
        qty = float(m.group(1))
        if m.group(2).lower() == "kg":
            qty *= 1000
        return qty, m.group(3).strip()
    # SUFFIX: "rajma 100g"
    m = re.match(r'^(.+?)\s+([\d.]+)\s*(g|gm|grams?|ml|kg)$', text, re.I)
    if m:
        qty = float(m.group(2))
        if m.group(3).lower() == "kg":
            qty *= 1000
        return qty, m.group(1).strip()
    # COUNT: "2 eggs"
    m = re.match(r'^(\d+(?:\.\d+)?)\s+(.+)$', text)
    if m:
        return float(m.group(1)) * 100, m.group(2).strip()
    return None, text


def lookup_nutrition(food_text: str, grams: float = 100,
                     allow_fuzzy: bool = True) -> Optional[dict]:
    canonical = resolve_food(food_text)
    if not canonical and allow_fuzzy:
        canonical = fuzzy_resolve(food_text)
    if not canonical or canonical not in NUTRITION_DB:
        return None
    v = NUTRITION_DB[canonical]
    s = grams / 100.0
    return {
        "calories":       round(v["cal"]     * s, 1),
        "protein_g":      round(v["protein"] * s, 1),
        "carbs_g":        round(v["carbs"]   * s, 1),
        "fats_g":         round(v["fat"]      * s, 1),
        "source":         v["src"],
        "canonical_name": canonical,
    }


def verify_meal_macros(foods: list[str], allow_fuzzy: bool = True) -> dict:
    """Compute verified macros for a food list. Returns coverage fraction."""
    totals     = {"calories": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fats_g": 0.0}
    verified   = []
    unverified = []
    sources    = set()

    for food in foods:
        grams, food_name = parse_quantity(food)
        if grams is None:
            grams = 100
        n = lookup_nutrition(food_name, grams, allow_fuzzy=allow_fuzzy)
        if n:
            totals["calories"]  += n["calories"]
            totals["protein_g"] += n["protein_g"]
            totals["carbs_g"]   += n["carbs_g"]
            totals["fats_g"]    += n["fats_g"]
            verified.append(food)
            sources.add(n["source"])
        else:
            unverified.append(food)

    total = len(foods)
    return {
        **{k: round(v, 1) for k, v in totals.items()},
        "verified_items":   verified,
        "unverified_items": unverified,
        "coverage":         round(len(verified) / total, 2) if total else 0.0,
        "sources":          sorted(sources),
    }


def calculate_macro_targets(weight_kg: float, tdee: float, goal: str) -> dict:
    """
    Evidence-based daily macro targets from ICMR-NIN guidelines.
    Returns {"calories", "protein_g", "carbs_g", "fats_g"}.
    """
    if goal == "weight_loss":
        calories  = tdee - 400
        protein_g = round(weight_kg * 2.2)
    elif goal == "muscle_gain":
        calories  = tdee + 250
        protein_g = round(weight_kg * 2.0)
    elif goal == "endurance":
        calories  = tdee + 100
        protein_g = round(weight_kg * 1.6)
    else:  # maintenance
        calories  = tdee
        protein_g = round(weight_kg * 1.8)

    fats_g  = round(calories * 0.25 / 9)
    carbs_g = round(max((calories - protein_g * 4 - fats_g * 9) / 4, 50))
    return {
        "calories":  round(calories),
        "protein_g": protein_g,
        "carbs_g":   carbs_g,
        "fats_g":    fats_g,
    }