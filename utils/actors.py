"""
Specialist Actors — deterministic domain experts.

Each actor has its own knowledge, constraints, and recommendations.
They can CONFLICT with each other — that's the point.
The orchestrator agent must consult them and resolve disagreements.

Actor 1: FitnessAdvisor  — workout programming expert
Actor 2: NutritionAdvisor — diet and macro expert (IFCT 2017)
Actor 3: ProgressAnalyst  — plateau and overload detection expert

These are NOT LLMs — they are deterministic rule engines.
The LLM being trained is the ORCHESTRATOR that manages them.
"""

from __future__ import annotations
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.plateau import detect_plateau
from utils.overload import expected_progression
from utils.nutrition import calculate_macro_targets, verify_meal_macros, NUTRITION_DB


# ── Actor 1: Fitness Advisor ──────────────────────────────────────────────────

def fitness_actor(client: dict, progress: dict) -> dict:
    """
    Returns workout constraints and recommendations.
    Will CONFLICT with nutrition actor when volume is high but calories are low.
    """
    goal         = client.get("goal", "maintenance")
    fitness_level= client.get("fitness_level", "beginner")
    equipment    = client.get("available_equipment", [])
    injuries     = client.get("injuries", [])
    sessions     = client.get("sessions_per_week", 3)

    # Volume prescription by level + goal
    volume_map = {
        "beginner":     {"muscle_gain": (10, 16), "weight_loss": (8, 14),
                         "endurance": (8, 12),    "maintenance": (8, 14)},
        "intermediate": {"muscle_gain": (15, 22), "weight_loss": (12, 18),
                         "endurance": (10, 16),   "maintenance": (10, 18)},
        "advanced":     {"muscle_gain": (18, 28), "weight_loss": (14, 22),
                         "endurance": (12, 20),   "maintenance": (12, 20)},
    }
    vol_range = volume_map.get(fitness_level, {}).get(goal, (10, 20))

    # Rep range prescription
    rep_map = {
        "muscle_gain": "6-12",
        "weight_loss": "12-20",
        "endurance":   "15-25",
        "maintenance": "8-15",
    }
    recommended_reps = rep_map.get(goal, "8-12")

    # Rest prescription
    rest_map = {
        "muscle_gain": 90,
        "weight_loss": 45,
        "endurance":   30,
        "maintenance": 60,
    }
    recommended_rest = rest_map.get(goal, 60)

    # Banned exercises due to injuries
    contraindicated = {
        "lower back":  ["deadlift", "good morning", "bent-over row", "hyperextension"],
        "knee":        ["lunge", "deep squat", "box jump", "leg extension"],
        "shoulder":    ["overhead press", "upright row", "behind neck", "military press"],
        "wrist":       ["barbell curl", "front squat"],
        "hip flexor":  ["leg raise", "sit-up"],
    }
    banned = []
    for injury in injuries:
        banned.extend(contraindicated.get(injury, []))

    # Overload signals from exercise history
    overload_signals = []
    ex_history = progress.get("exercise_history", {})
    for ex_name, hist in ex_history.items():
        exp = expected_progression(
            ex_name,
            hist["last_weight_kg"],
            hist["last_reps_str"],
            hist.get("target_reps", "8-12"),
            hist.get("target_sets", 3),
        )
        overload_signals.append({
            "exercise":         ex_name,
            "progression_type": exp["progression_type"],
            "expected_weight":  exp["expected_weight_kg"],
            "expected_reps":    exp["expected_reps"],
            "message": (
                f"{ex_name}: {exp['progression_type']} → "
                f"use {exp['expected_weight_kg']}kg × {exp['expected_reps']}"
            )
        })

    # Conflict flag: high volume needs caloric support
    min_calories_for_volume = vol_range[1] * 40  # rough heuristic
    tdee = client.get("tdee_estimate", 2000)
    conflict_with_nutrition = vol_range[1] * 40 > tdee * 0.9

    return {
        "actor": "fitness_advisor",
        "status": "ok",
        "recommendations": {
            "weekly_volume_sets_range": vol_range,
            "sessions_per_week":        sessions,
            "recommended_reps":         recommended_reps,
            "recommended_rest_seconds": recommended_rest,
            "available_equipment":      equipment,
            "banned_exercises":         banned,
            "overload_signals":         overload_signals,
        },
        "constraints": {
            "must_use_only_equipment": equipment,
            "must_avoid_exercises":    banned,
            "weekly_sets_min":         vol_range[0],
            "weekly_sets_max":         vol_range[1],
        },
        "conflicts": {
            "with_nutrition": conflict_with_nutrition,
            "reason": (
                f"High volume ({vol_range[1]} sets) may require >{min_calories_for_volume} kcal. "
                f"Coordinate with Nutrition Advisor."
            ) if conflict_with_nutrition else None,
        },
        "message": (
            f"Fitness Advisor: For {fitness_level} {goal}, prescribe "
            f"{vol_range[0]}–{vol_range[1]} sets/week at {recommended_reps} reps. "
            f"Equipment: {equipment}. "
            + (f"BANNED (injuries): {banned}. " if banned else "")
            + (f"Overload needed for: {[s['exercise'] for s in overload_signals]}." if overload_signals else "")
        ),
    }


# ── Actor 2: Nutrition Advisor ────────────────────────────────────────────────

def nutrition_actor(client: dict, progress: dict, complications: list) -> dict:
    """
    Returns macro targets and dietary constraints.
    Will CONFLICT with fitness actor when goal changes or plateau detected.
    """
    goal         = client.get("goal", "maintenance")
    weight_kg    = client.get("weight_kg", 70)
    tdee         = client.get("tdee_estimate", 2000)
    restrictions = client.get("dietary_restrictions", [])

    # Standard targets
    targets = calculate_macro_targets(weight_kg, tdee, goal)

    # Adjust for complications
    calorie_adjustment = 0
    adjustment_reasons = []

    for comp in complications:
        if comp == "plateau" and goal == "weight_loss":
            calorie_adjustment -= 150
            adjustment_reasons.append("Plateau detected → reduce calories by 150 kcal")
        elif comp == "plateau" and goal == "muscle_gain":
            calorie_adjustment += 200
            adjustment_reasons.append("Plateau detected → increase calories by 200 kcal")
        elif comp.startswith("goal_change:"):
            parts = comp.split("→")
            if len(parts) == 2:
                new_goal = parts[1].strip()
                targets = calculate_macro_targets(weight_kg, tdee, new_goal)
                adjustment_reasons.append(
                    f"Goal changed to {new_goal} → recalculated targets"
                )

    adjusted_calories = targets["calories"] + calorie_adjustment

    # Banned foods for dietary restrictions
    banned_foods = []
    if "vegetarian" in restrictions:
        banned_foods.extend(["chicken", "beef", "pork", "fish", "mutton",
                              "turkey", "tuna", "salmon", "prawn", "lamb"])
    if "vegan" in restrictions:
        banned_foods.extend(["milk", "paneer", "curd", "yogurt", "egg",
                              "whey", "cheese", "ghee", "butter"])

    # Recommended high-protein Indian foods
    if "vegetarian" in restrictions or "vegan" in restrictions:
        recommended_foods = [
            "rajma (127 kcal, 8.7g protein per 100g — IFCT2017)",
            "chana (164 kcal, 8.9g protein per 100g — IFCT2017)",
            "paneer (265 kcal, 18.3g protein per 100g — IFCT2017)",
            "soya chunks (345 kcal, 52.4g protein per 100g — IFCT2017)",
            "tofu (76 kcal, 8g protein per 100g — USDA)",
            "moong dal (105 kcal, 7g protein per 100g — IFCT2017)",
            "curd (60 kcal, 3.1g protein per 100g — IFCT2017)",
        ]
    else:
        recommended_foods = [
            "chicken breast (165 kcal, 31g protein per 100g — USDA)",
            "eggs (143 kcal, 12.6g protein per 100g — IFCT2017)",
            "fish (97 kcal, 20g protein per 100g — IFCT2017)",
            "paneer (265 kcal, 18.3g protein per 100g — IFCT2017)",
            "rajma (127 kcal, 8.7g protein per 100g — IFCT2017)",
        ]

    # Conflict: if fitness actor prescribes high volume but this calorie target is low
    high_volume_threshold = 20
    conflict_with_fitness = adjusted_calories < tdee * 0.85

    return {
        "actor": "nutrition_advisor",
        "status": "ok",
        "recommendations": {
            "daily_calories":       adjusted_calories,
            "daily_protein_g":      targets["protein_g"],
            "daily_carbs_g":        targets["carbs_g"],
            "daily_fats_g":         targets["fats_g"],
            "calorie_adjustment":   calorie_adjustment,
            "adjustment_reasons":   adjustment_reasons,
            "recommended_foods":    recommended_foods,
            "banned_foods":         banned_foods,
            "dietary_restrictions": restrictions,
        },
        "constraints": {
            "calories_target":   adjusted_calories,
            "protein_minimum_g": targets["protein_g"],
            "banned_foods":      banned_foods,
            "tolerance_pct":     15,
        },
        "conflicts": {
            "with_fitness": conflict_with_fitness,
            "reason": (
                f"Calorie target ({adjusted_calories} kcal) is low relative to TDEE "
                f"({tdee} kcal). High training volume risks under-fuelling. "
                f"Either increase calories or reduce volume."
            ) if conflict_with_fitness else None,
        },
        "message": (
            f"Nutrition Advisor: Target {adjusted_calories} kcal/day, "
            f"{targets['protein_g']}g protein, {targets['carbs_g']}g carbs, "
            f"{targets['fats_g']}g fats. "
            + (f"Adjustments: {'; '.join(adjustment_reasons)}. " if adjustment_reasons else "")
            + (f"BANNED foods: {banned_foods[:4]}. " if banned_foods else "")
        ),
    }


# ── Actor 3: Progress Analyst ─────────────────────────────────────────────────

def progress_actor(client: dict, progress: dict, complications: list) -> dict:
    """
    Returns plateau status and adaptation signals.
    Will CONFLICT with both other actors when plateau requires drastic changes.
    """
    goal       = client.get("goal", "maintenance")
    weight_series = progress.get("weight_series", [])
    adherence     = progress.get("adherence_pct", 100)
    avg_rating    = progress.get("avg_workout_rating", 3.0)

    # Statistical plateau detection
    if weight_series:
        plateau_result = detect_plateau(weight_series, goal=goal)
        plateau_status = plateau_result.status
        slope          = plateau_result.slope_kg_per_week
        confidence     = plateau_result.confidence
        plateau_reason = plateau_result.reason
    else:
        plateau_status = "insufficient_data"
        slope          = 0.0
        confidence     = 0.0
        plateau_reason = "No weight data available — fresh start."

    # Adaptation signals
    signals = []

    if plateau_status in ("plateau", "reversing") and confidence >= 0.5:
        signals.append({
            "type":     "plateau",
            "severity": "high" if confidence > 0.7 else "medium",
            "message":  f"Weight plateau detected (slope={slope:+.2f} kg/wk, confidence={confidence:.2f}). "
                        f"Must adapt: increase volume ≥10% OR adjust calories ≥150 kcal.",
            "required_actions": [
                "increase weekly_volume_sets by ≥10% above minimum",
                "OR reduce calories by ≥150 kcal (weight_loss)",
                "OR increase calories by ≥200 kcal (muscle_gain)",
            ],
        })

    if plateau_status == "overshooting":
        signals.append({
            "type":     "overshooting",
            "severity": "medium",
            "message":  f"Rate of change too fast (slope={slope:+.2f} kg/wk). "
                        f"Moderate the intervention.",
        })

    if adherence < 60 and progress.get("adherence_pct") is not None:
        signals.append({
            "type":     "low_adherence",
            "severity": "medium",
            "message":  f"Adherence only {adherence}%. Reduce session frequency by 1 day.",
            "required_actions": ["reduce sessions_per_week by 1"],
        })

    if avg_rating is not None and avg_rating <= 2.0:
        signals.append({
            "type":     "overtraining",
            "severity": "medium",
            "message":  f"Average workout rating {avg_rating}/5 — possible overtraining. "
                        f"Add a deload week (50% volume reduction).",
        })

    # Conflict detection
    conflict_with_fitness   = any(s["type"] == "plateau" for s in signals)
    conflict_with_nutrition = any(s["type"] == "plateau" for s in signals)

    return {
        "actor": "progress_analyst",
        "status": "ok",
        "recommendations": {
            "plateau_status":   plateau_status,
            "slope_kg_per_week": slope,
            "confidence":       confidence,
            "adherence_pct":    adherence,
            "avg_rating":       avg_rating,
            "signals":          signals,
        },
        "constraints": {
            "must_adapt_if_plateau": plateau_status in ("plateau", "reversing"),
            "required_actions":      [a for s in signals for a in s.get("required_actions", [])],
        },
        "conflicts": {
            "with_fitness": conflict_with_fitness,
            "with_nutrition": conflict_with_nutrition,
            "reason": (
                "Plateau requires volume/calorie adaptation. "
                "Coordinate with Fitness and Nutrition advisors."
            ) if conflict_with_fitness else None,
        },
        "message": (
            f"Progress Analyst: Weight trend is '{plateau_status}' "
            f"(slope={slope:+.2f} kg/wk, confidence={confidence:.2f}). "
            + (f"Signals: {[s['type'] for s in signals]}. " if signals else "No issues detected. ")
            + (f"Action required: {signals[0]['required_actions']}" if signals else "")
        ),
    }


# ── Conflict detector ─────────────────────────────────────────────────────────

def detect_actor_conflicts(
    fitness_response: dict,
    nutrition_response: dict,
    progress_response: dict,
) -> list[dict]:
    """
    Detect conflicts between actor recommendations.
    Returns list of conflicts the orchestrator must resolve.
    """
    conflicts = []

    # Conflict 1: Fitness wants high volume, Nutrition has low calories
    if (fitness_response.get("conflicts", {}).get("with_nutrition") or
            nutrition_response.get("conflicts", {}).get("with_fitness")):
        conflicts.append({
            "between":    ["fitness_advisor", "nutrition_advisor"],
            "type":       "volume_calorie_mismatch",
            "description": (
                "Fitness Advisor prescribes high training volume but "
                "Nutrition Advisor's calorie target may not support it. "
                "You must either reduce volume OR increase calories."
            ),
            "resolution_options": [
                "Reduce weekly_volume_sets to lower end of fitness range",
                "Increase daily_calories by 200-300 kcal",
            ],
        })

    # Conflict 2: Progress says plateau but Fitness hasn't adapted volume
    if progress_response["recommendations"]["plateau_status"] in ("plateau", "reversing"):
        conflicts.append({
            "between":    ["progress_analyst", "fitness_advisor"],
            "type":       "plateau_volume_conflict",
            "description": (
                "Progress Analyst detected a plateau but Fitness Advisor "
                "has not increased volume. You must apply progressive overload "
                "or caloric adjustment."
            ),
            "resolution_options": [
                "Increase weekly_volume_sets by ≥10% above minimum",
                "Adjust calories per Nutrition Advisor's plateau guidance",
            ],
        })

    # Conflict 3: Injury constraint vs exercise history
    banned = fitness_response["constraints"].get("must_avoid_exercises", [])
    overload_signals = fitness_response["recommendations"].get("overload_signals", [])
    for signal in overload_signals:
        ex_name = signal["exercise"].lower()
        for banned_move in banned:
            if banned_move in ex_name:
                conflicts.append({
                    "between":    ["fitness_advisor", "progress_analyst"],
                    "type":       "injury_overload_conflict",
                    "description": (
                        f"Progress Analyst wants overload on '{signal['exercise']}' "
                        f"but Fitness Advisor has banned it due to injury. "
                        f"Find a safe alternative exercise."
                    ),
                    "resolution_options": [
                        f"Replace '{signal['exercise']}' with an injury-safe alternative",
                        "Apply overload to a different muscle group",
                    ],
                })

    return conflicts