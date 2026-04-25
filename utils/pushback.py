"""
Actor Pushback Engine — makes actors ACTIVE, not passive.

After the orchestrator submits a plan, each actor reviews it
against their own constraints and either ACCEPTS or REJECTS
with specific, actionable suggestions.

This transforms the environment from a "grader" into a
"negotiation arena" — the orchestrator must iterate with
actors until all accept.

Example episode flow:
  Step 1: consult fitness_advisor → "use 10-16 sets, no barbell"
  Step 2: consult nutrition_advisor → "target 2650 kcal"
  Step 3: submit_plan → REJECTED by fitness_advisor:
          "You prescribed 24 sets — max is 16 for a beginner.
           Reduce to 14-16 sets. Also, 'Barbell Squat' needs
           barbell which is not available. Use Dumbbell Squat."
  Step 4: submit revised plan → ACCEPTED by all → reward 0.95
"""

from __future__ import annotations
import json
from typing import Optional


# Exercise alternatives for injury-safe substitutions
SAFE_ALTERNATIVES = {
    "deadlift":       ["hip thrust", "glute bridge", "cable pull-through", "goblet squat"],
    "bent-over row":  ["chest-supported row", "cable row", "dumbbell row (neutral spine)"],
    "good morning":   ["hip thrust", "glute bridge", "Romanian deadlift (light, neutral)"],
    "overhead press": ["landmine press", "high incline press", "front raise"],
    "upright row":    ["lateral raise", "face pull", "rear delt fly"],
    "lunge":          ["leg press (shallow)", "step-up (low box)", "wall sit"],
    "deep squat":     ["goblet squat (parallel)", "leg press", "wall sit"],
    "box jump":       ["step-up", "sled push", "seated box jump"],
    "leg extension":  ["terminal knee extension (band)", "wall sit", "step-up"],
    "barbell curl":   ["dumbbell curl (neutral grip)", "hammer curl", "cable curl"],
}

# Equipment-safe alternatives
EQUIPMENT_SWAPS = {
    "barbell squat":      "dumbbell squat",
    "barbell bench press": "dumbbell bench press",
    "barbell row":        "dumbbell row",
    "barbell deadlift":   "dumbbell Romanian deadlift",
    "barbell curl":       "dumbbell curl",
    "lat pulldown":       "pull-up (or resistance band pull-down)",
    "cable row":          "dumbbell row",
    "leg press":          "dumbbell squat",
    "chest press machine":"dumbbell bench press",
}


def _get_exercises(workout: dict) -> list[dict]:
    exercises = []
    for day in workout.get("days", []):
        exercises.extend(day.get("exercises", []))
    return exercises


def fitness_pushback(
    plan_workout: dict,
    client: dict,
    actor_response: dict,
) -> Optional[dict]:
    """
    FitnessAdvisor reviews the submitted plan.
    Returns None if ACCEPTED, or a rejection dict with suggestions.
    """
    issues = []
    suggestions = []

    constraints = actor_response.get("constraints", {})
    vol_min = constraints.get("weekly_sets_min", 0)
    vol_max = constraints.get("weekly_sets_max", 999)
    banned = constraints.get("must_avoid_exercises", [])
    equipment = set(constraints.get("must_use_only_equipment", []))

    exercises = _get_exercises(plan_workout)
    agent_sets = int(plan_workout.get("weekly_volume_sets", 0))
    if agent_sets == 0:
        agent_sets = sum(int(ex.get("sets", 0) or 0) for ex in exercises)

    # Check volume
    if agent_sets < vol_min:
        issues.append(f"Volume too low: {agent_sets} sets/week (minimum {vol_min})")
        suggestions.append(f"Increase to {vol_min}-{vol_max} sets/week. Add 1-2 more exercises per day.")
    elif agent_sets > vol_max:
        issues.append(f"Volume too high: {agent_sets} sets/week (maximum {vol_max})")
        suggestions.append(f"Reduce to {vol_min}-{vol_max} sets/week. Remove 1-2 exercises or reduce sets per exercise to 2.")

    # Check banned exercises
    for ex in exercises:
        name_l = ex.get("name", "").lower()
        for banned_move in banned:
            if banned_move in name_l:
                alts = SAFE_ALTERNATIVES.get(banned_move, ["a safe alternative"])
                issues.append(f"'{ex['name']}' is BANNED due to injury")
                suggestions.append(
                    f"Replace '{ex['name']}' with: {', '.join(alts[:3])}"
                )

    # Check equipment
    EQUIPMENT_KEYWORDS = {
        "barbell": {"barbell"},
        "cables": {"cable", "lat pulldown"},
        "machines": {"machine", "leg press", "leg curl", "leg extension"},
        "kettlebell": {"kettlebell"},
    }
    for ex in exercises:
        name_l = ex.get("name", "").lower()
        for eq, keywords in EQUIPMENT_KEYWORDS.items():
            if eq not in equipment and any(kw in name_l for kw in keywords):
                swap = EQUIPMENT_SWAPS.get(name_l.strip(), f"a {list(equipment)[0] if equipment else 'bodyweight'} alternative")
                issues.append(f"'{ex['name']}' requires {eq} (not available)")
                suggestions.append(f"Replace '{ex['name']}' with {swap}")
                break

    if not issues:
        return None  # ACCEPTED

    return {
        "actor": "fitness_advisor",
        "verdict": "REJECTED",
        "issues": issues,
        "suggestions": suggestions,
        "message": (
            f"🏋️ FITNESS ADVISOR REJECTS this plan ({len(issues)} issue(s)):\n"
            + "\n".join(f"  ✗ {iss}" for iss in issues)
            + "\n\nSuggested fixes:\n"
            + "\n".join(f"  → {sug}" for sug in suggestions)
        ),
    }


def nutrition_pushback(
    plan_nutrition: dict,
    client: dict,
    actor_response: dict,
) -> Optional[dict]:
    """
    NutritionAdvisor reviews the submitted plan.
    Returns None if ACCEPTED, or a rejection dict with suggestions.
    """
    issues = []
    suggestions = []

    constraints = actor_response.get("constraints", {})
    target_cal = constraints.get("calories_target", 0)
    target_pro = constraints.get("protein_minimum_g", 0)
    banned_foods = constraints.get("banned_foods", [])
    tolerance = constraints.get("tolerance_pct", 15) / 100

    daily = plan_nutrition.get("daily_targets", {})
    agent_cal = float(daily.get("calories", 0))
    agent_pro = float(daily.get("protein_g", 0))

    # Check macros
    if target_cal > 0 and agent_cal > 0:
        cal_err = abs(agent_cal - target_cal) / target_cal
        if cal_err > tolerance:
            direction = "increase" if agent_cal < target_cal else "decrease"
            issues.append(
                f"Calories off by {cal_err*100:.0f}%: {agent_cal:.0f} kcal "
                f"(target {target_cal:.0f} ± {tolerance*100:.0f}%)"
            )
            suggestions.append(
                f"{direction.capitalize()} calories to {target_cal:.0f} kcal. "
                f"Acceptable range: {target_cal*(1-tolerance):.0f}–{target_cal*(1+tolerance):.0f} kcal."
            )
    elif target_cal > 0 and agent_cal == 0:
        issues.append("No calorie target specified in daily_targets")
        suggestions.append(f"Set daily_targets.calories to {target_cal:.0f} kcal")

    if target_pro > 0 and agent_pro > 0:
        pro_err = abs(agent_pro - target_pro) / target_pro
        if pro_err > tolerance:
            issues.append(
                f"Protein off by {pro_err*100:.0f}%: {agent_pro:.0f}g "
                f"(target {target_pro:.0f}g)"
            )
            suggestions.append(
                f"Adjust protein to {target_pro:.0f}g/day. "
                f"Add high-protein foods: paneer (18g/100g), rajma (8.7g/100g), soya chunks (52g/100g)."
            )

    # Check banned foods
    plan_text = json.dumps(plan_nutrition).lower()
    for food in banned_foods:
        if food.lower() in plan_text:
            issues.append(f"'{food}' is banned for this client's dietary restrictions")
            suggestions.append(f"Remove '{food}'. Use plant-based alternatives from IFCT 2017 database.")
            break  # one violation is enough

    if not issues:
        return None  # ACCEPTED

    return {
        "actor": "nutrition_advisor",
        "verdict": "REJECTED",
        "issues": issues,
        "suggestions": suggestions,
        "message": (
            f"🥗 NUTRITION ADVISOR REJECTS this plan ({len(issues)} issue(s)):\n"
            + "\n".join(f"  ✗ {iss}" for iss in issues)
            + "\n\nSuggested fixes:\n"
            + "\n".join(f"  → {sug}" for sug in suggestions)
        ),
    }


def progress_pushback(
    plan_workout: dict,
    plan_nutrition: dict,
    client: dict,
    progress: dict,
    complications: list,
    actor_response: dict,
) -> Optional[dict]:
    """
    ProgressAnalyst reviews the submitted plan.
    Returns None if ACCEPTED, or a rejection dict.
    """
    issues = []
    suggestions = []

    constraints = actor_response.get("constraints", {})
    must_adapt = constraints.get("must_adapt_if_plateau", False)

    if not must_adapt:
        return None  # No plateau, auto-accept

    # Check if the plan actually adapted
    exercises = _get_exercises(plan_workout)
    agent_sets = int(plan_workout.get("weekly_volume_sets", 0))
    if agent_sets == 0:
        agent_sets = sum(int(ex.get("sets", 0) or 0) for ex in exercises)

    daily = plan_nutrition.get("daily_targets", {})
    agent_cal = float(daily.get("calories", 0))

    # Get baseline volume/calories from actor recommendations
    recs = actor_response.get("recommendations", {})
    plateau_status = recs.get("plateau_status", "")

    if plateau_status not in ("plateau", "reversing"):
        return None

    signals = recs.get("signals", [])
    plateau_signal = next((s for s in signals if s["type"] == "plateau"), None)

    if not plateau_signal:
        return None

    # The agent should have done something — increased volume or adjusted calories
    # We check loosely here since the grader does the precise check
    goal = client.get("goal", "maintenance")
    if goal == "weight_loss" and agent_cal > 1800:
        # Probably didn't reduce calories enough — but let grader decide
        pass

    # Check if required actions were taken
    required = plateau_signal.get("required_actions", [])
    if required and agent_sets == 0 and agent_cal == 0:
        issues.append("Plateau detected but plan has no volume or calorie data")
        suggestions.append("You must either increase volume ≥10% or adjust calories ≥150 kcal")

    if not issues:
        return None

    return {
        "actor": "progress_analyst",
        "verdict": "REJECTED",
        "issues": issues,
        "suggestions": suggestions,
        "message": (
            f"📊 PROGRESS ANALYST REJECTS this plan ({len(issues)} issue(s)):\n"
            + "\n".join(f"  ✗ {iss}" for iss in issues)
            + "\n\nSuggested fixes:\n"
            + "\n".join(f"  → {sug}" for sug in suggestions)
        ),
    }


def collect_actor_pushback(
    workout: dict,
    nutrition: dict,
    client: dict,
    progress: dict,
    complications: list,
    actor_responses: dict,
) -> list[dict]:
    """
    Run all actor pushbacks and collect rejections.
    Returns list of rejection dicts (empty if all accepted).
    """
    rejections = []

    if "fitness_advisor" in actor_responses:
        fb = fitness_pushback(workout, client, actor_responses["fitness_advisor"])
        if fb:
            rejections.append(fb)

    if "nutrition_advisor" in actor_responses:
        nb = nutrition_pushback(nutrition, client, actor_responses["nutrition_advisor"])
        if nb:
            rejections.append(nb)

    if "progress_analyst" in actor_responses:
        pb = progress_pushback(
            workout, nutrition, client, progress, complications,
            actor_responses["progress_analyst"]
        )
        if pb:
            rejections.append(pb)

    return rejections