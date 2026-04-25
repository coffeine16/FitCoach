"""
FitCoach Multi-Actor RL Environment.

The agent plays an ORCHESTRATOR managing three specialist actors:
  - FitnessAdvisor:  workout programming constraints
  - NutritionAdvisor: macro targets and dietary constraints  
  - ProgressAnalyst:  plateau detection and adaptation signals

The orchestrator must:
  1. Consult actors to discover their recommendations and constraints
  2. Detect and resolve conflicts between actors
  3. Produce a final integrated plan satisfying all actors

This implements Theme 1 (Halluminate sub-theme):
"Build a realistic environment where an agent interacts with and
manages multiple actors to discover and achieve the task."

Reward dimensions:
  1. equipment_compliance    — exercises match available equipment
  2. macro_accuracy          — macros within ±15% of IFCT 2017 formula
  3. volume_appropriateness  — weekly sets in correct range
  4. progressive_overload    — correct overload applied (when history exists)
  5. plateau_response        — adapted when plateau detected
  6. constraint_respect      — no contraindicated exercises or dietary violations
  7. coherence               — nutrition supports training volume
  8. actor_coordination      — consulted all relevant actors, resolved conflicts

Safety penalty: −0.3 for any hard constraint violation.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Dict, Any, List
from uuid import uuid4

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in [_ROOT, _HERE]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import FitcoachAction, FitcoachObservation
except ImportError:
    from models import FitcoachAction, FitcoachObservation

from utils.plateau import detect_plateau
from utils.overload import verify_agent_overload
from utils.nutrition import calculate_macro_targets, verify_meal_macros
from utils.actors import (
    fitness_actor, nutrition_actor, progress_actor, detect_actor_conflicts
)
from utils.pushback import collect_actor_pushback
from utils.curriculum import CurriculumManager, generate_client


# ── Domain constraint tables ──────────────────────────────────────────────────

EQUIPMENT_KEYWORDS: dict[str, set[str]] = {
    "barbell":          {"barbell"},
    "dumbbells":        {"dumbbell", " db "},
    "cables":           {"cable", "lat pulldown", "cable row"},
    "machines":         {"machine", "leg press", "leg curl", "leg extension",
                         "chest press machine"},
    "pull_up_bar":      {"pull-up", "pullup", "chin-up", "chinup"},
    "resistance_bands": {"resistance band"},
    "kettlebell":       {"kettlebell"},
}

CONTRAINDICATED: dict[str, list[str]] = {
    "lower back":  ["deadlift", "good morning", "bent-over row",
                    "jefferson curl", "hyperextension"],
    "knee":        ["lunge", "deep squat", "box jump", "jump squat",
                    "leg extension", "step-up"],
    "shoulder":    ["overhead press", "upright row", "behind neck",
                    "military press", "arnold press"],
    "wrist":       ["barbell curl", "push-up on palms", "front squat"],
    "hip flexor":  ["leg raise", "sit-up", "hanging knee raise"],
}

VOLUME_RANGES: dict[str, dict[str, tuple[int, int]]] = {
    "beginner":     {"muscle_gain": (10, 16), "weight_loss": (8, 14),
                     "endurance": (8, 12),    "maintenance": (8, 14)},
    "intermediate": {"muscle_gain": (15, 22), "weight_loss": (12, 18),
                     "endurance": (10, 16),   "maintenance": (10, 18)},
    "advanced":     {"muscle_gain": (18, 28), "weight_loss": (14, 22),
                     "endurance": (12, 20),   "maintenance": (12, 20)},
}


# ── Task configurations ───────────────────────────────────────────────────────

TASK_CONFIGS: dict[str, dict] = {

    "week1_plan": {
        "max_steps": 5,
        "phases": ["initial"],
        "description": (
            "Fresh beginner client. Consult all three actors to discover "
            "constraints, then submit a valid Week 1 plan. "
            "FitnessAdvisor will specify equipment and volume constraints. "
            "NutritionAdvisor will specify macro targets from IFCT 2017. "
            "ProgressAnalyst will confirm no plateau (fresh start)."
        ),
        "client": {
            "name": "Arjun Sharma",
            "age": 24, "sex": "male",
            "weight_kg": 72.0, "height_cm": 175.0,
            "goal": "muscle_gain",
            "fitness_level": "beginner",
            "dietary_restrictions": ["vegetarian"],
            "available_equipment": ["dumbbells", "pull_up_bar", "resistance_bands"],
            "sessions_per_week": 3,
            "tdee_estimate": 2400.0,
            "injuries": [],
        },
        "progress_data": {},
        "complications": [],
    },

    "plateau_adaptation": {
        "max_steps": 7,
        "phases": ["initial", "adaptation"],
        "description": (
            "Intermediate client with a 14-day weight plateau. "
            "ProgressAnalyst will detect the plateau and flag required adaptation. "
            "FitnessAdvisor and NutritionAdvisor may CONFLICT on how to fix it. "
            "Orchestrator must consult all three, resolve the conflict, "
            "and submit a plan that adapts correctly."
        ),
        "client": {
            "name": "Priya Menon",
            "age": 30, "sex": "female",
            "weight_kg": 65.0, "height_cm": 162.0,
            "goal": "weight_loss",
            "fitness_level": "intermediate",
            "dietary_restrictions": [],
            "available_equipment": ["barbell", "dumbbells", "cables", "machines"],
            "sessions_per_week": 4,
            "tdee_estimate": 2100.0,
            "injuries": [],
        },
        "progress_data": {
            "weight_series": [
                {"date": "2026-04-01", "weight_kg": 65.2},
                {"date": "2026-04-02", "weight_kg": 65.1},
                {"date": "2026-04-03", "weight_kg": 65.3},
                {"date": "2026-04-05", "weight_kg": 65.0},
                {"date": "2026-04-07", "weight_kg": 65.2},
                {"date": "2026-04-08", "weight_kg": 65.1},
                {"date": "2026-04-09", "weight_kg": 65.3},
                {"date": "2026-04-10", "weight_kg": 65.0},
                {"date": "2026-04-11", "weight_kg": 65.2},
                {"date": "2026-04-12", "weight_kg": 65.1},
                {"date": "2026-04-14", "weight_kg": 65.2},
                {"date": "2026-04-15", "weight_kg": 65.0},
                {"date": "2026-04-16", "weight_kg": 65.1},
                {"date": "2026-04-17", "weight_kg": 65.2},
            ],
            "adherence_pct": 80,
            "avg_workout_rating": 3.2,
            "exercise_history": {
                "Barbell Squat": {
                    "last_weight_kg": 55.0,
                    "last_reps_str": "10,10,10",
                    "target_reps": "8-12",
                    "target_sets": 3,
                },
                "Dumbbell Romanian Deadlift": {
                    "last_weight_kg": 20.0,
                    "last_reps_str": "12,12,12",
                    "target_reps": "10-12",
                    "target_sets": 3,
                },
            },
        },
        "complications": ["plateau"],
        "complication_schedule": {
            4: ["new_injury:knee"],  # Knee pain added mid-episode
        },
    },

    "conflict_resolution": {
        "max_steps": 9,
        "phases": ["initial", "adaptation", "conflict"],
        "description": (
            "Three simultaneous challenges requiring multi-actor coordination: "
            "(1) ProgressAnalyst detects a 3-week plateau. "
            "(2) FitnessAdvisor bans deadlifts due to new lower-back injury — "
            "but ProgressAnalyst wants overload on Barbell Deadlift. "
            "This creates an INJURY-OVERLOAD CONFLICT the orchestrator must resolve. "
            "(3) Goal changed from weight_loss to maintenance — "
            "NutritionAdvisor and FitnessAdvisor both need to update their recommendations. "
            "Orchestrator must consult all actors, surface all 3 conflicts, and resolve them."
        ),
        "client": {
            "name": "Rahul Verma",
            "age": 35, "sex": "male",
            "weight_kg": 85.0, "height_cm": 178.0,
            "goal": "maintenance",
            "fitness_level": "advanced",
            "dietary_restrictions": ["vegetarian"],
            "available_equipment": [
                "barbell", "dumbbells", "cables", "machines", "pull_up_bar"
            ],
            "sessions_per_week": 5,
            "tdee_estimate": 2800.0,
            "injuries": ["lower back"],
        },
        "progress_data": {
            "weight_series": [
                {"date": "2026-03-25", "weight_kg": 85.5},
                {"date": "2026-03-27", "weight_kg": 85.3},
                {"date": "2026-03-29", "weight_kg": 85.4},
                {"date": "2026-04-01", "weight_kg": 85.2},
                {"date": "2026-04-03", "weight_kg": 85.3},
                {"date": "2026-04-05", "weight_kg": 85.4},
                {"date": "2026-04-07", "weight_kg": 85.2},
                {"date": "2026-04-09", "weight_kg": 85.3},
                {"date": "2026-04-11", "weight_kg": 85.1},
                {"date": "2026-04-13", "weight_kg": 85.2},
                {"date": "2026-04-15", "weight_kg": 85.3},
                {"date": "2026-04-17", "weight_kg": 85.2},
            ],
            "adherence_pct": 65,
            "avg_workout_rating": 2.1,
            "previous_goal": "weight_loss",
            "exercise_history": {
                "Barbell Deadlift": {
                    "last_weight_kg": 120.0,
                    "last_reps_str": "5,5,5",
                    "target_reps": "4-6",
                    "target_sets": 4,
                },
                "Barbell Back Squat": {
                    "last_weight_kg": 100.0,
                    "last_reps_str": "6,6,6",
                    "target_reps": "4-6",
                    "target_sets": 4,
                },
            },
        },
        "complications": [
            "plateau",
            "new_injury:lower back",
            "goal_change:weight_loss→maintenance",
        ],
    },

    # ── Theme 4: Adaptive curriculum ───────────────────────────────────────
    "curriculum": {
        "max_steps": 7,
        "phases": ["initial"],
        "description": "Adaptive curriculum — random clients, difficulty escalates with performance.",
        "client": {},
        "progress_data": {},
        "complications": [],
    },
}

ALL_ACTORS = {"fitness_advisor", "nutrition_advisor", "progress_analyst"}


# ── Grader ────────────────────────────────────────────────────────────────────

def _get_all_exercises(workout: dict) -> list[dict]:
    exercises = []
    for day in workout.get("days", []):
        exercises.extend(day.get("exercises", []))
    return exercises


def _plan_text(workout: dict, nutrition: dict) -> str:
    return (json.dumps(workout) + " " + json.dumps(nutrition)).lower()


def grade_plan(
    action: FitcoachAction,
    config: dict,
    actors_consulted: list[str],
    actor_responses: dict[str, dict],
    active_conflicts: list[dict],
    safety_already_violated: bool,
) -> tuple[float, dict[str, float], str, bool]:
    """
    Deterministic grader — now includes actor_coordination dimension.
    Returns: (reward, score_breakdown, feedback, safety_violated)
    """
    client   = config["client"]
    progress = config["progress_data"]
    comps    = config["complications"]

    scores: dict[str, float] = {}
    fb: list[str]            = []
    safety_violated          = safety_already_violated

    try:
        workout = json.loads(action.workout_plan or "{}")
    except json.JSONDecodeError:
        workout = {}
    try:
        nutrition = json.loads(action.nutrition_plan or "{}")
    except json.JSONDecodeError:
        nutrition = {}

    plan_text = _plan_text(workout, nutrition)
    exercises = _get_all_exercises(workout)

    # ── 1. Equipment compliance ───────────────────────────────────────────────
    available = set(client.get("available_equipment", []))
    banned_eq = {eq for eq in EQUIPMENT_KEYWORDS if eq not in available}
    eq_violations = []
    for ex in exercises:
        name_l = ex.get("name", "").lower()
        for eq in banned_eq:
            if any(kw in name_l for kw in EQUIPMENT_KEYWORDS[eq]):
                eq_violations.append(f"'{ex.get('name')}' needs {eq}")
                break

    if not eq_violations:
        scores["equipment_compliance"] = 1.0
        fb.append("✓ Equipment: all exercises match available equipment.")
    else:
        scores["equipment_compliance"] = max(0.0, 1.0 - 0.25 * len(eq_violations))
        fb.append(f"✗ Equipment violations: {eq_violations[:3]}.")

    # ── 2. Macro accuracy ─────────────────────────────────────────────────────
    targets   = calculate_macro_targets(
        client["weight_kg"], client["tdee_estimate"], client["goal"]
    )
    daily     = nutrition.get("daily_targets", {})
    agent_cal = float(daily.get("calories", 0))
    agent_pro = float(daily.get("protein_g", 0))

    if agent_cal == 0 and agent_pro == 0:
        scores["macro_accuracy"] = 0.0
        fb.append(
            f"✗ No daily_targets. Target: {targets['calories']} kcal, "
            f"{targets['protein_g']}g protein."
        )
    else:
        cal_err = abs(agent_cal - targets["calories"]) / max(targets["calories"], 1)
        pro_err = abs(agent_pro - targets["protein_g"]) / max(targets["protein_g"], 1)
        worst   = max(cal_err, pro_err)
        if worst <= 0.10:
            scores["macro_accuracy"] = 1.0
            fb.append(
                f"✓ Macros within 10%: {agent_cal:.0f} kcal / {agent_pro:.0f}g protein "
                f"(target {targets['calories']} / {targets['protein_g']}g)."
            )
        elif worst <= 0.15:
            scores["macro_accuracy"] = 0.7
            fb.append(
                f"~ Macros within 15%: {agent_cal:.0f} kcal / {agent_pro:.0f}g protein."
            )
        else:
            scores["macro_accuracy"] = max(0.0, 1.0 - worst)
            fb.append(
                f"✗ Macros off by {worst*100:.0f}%: {agent_cal:.0f} kcal / "
                f"{agent_pro:.0f}g protein (target {targets['calories']} / "
                f"{targets['protein_g']}g)."
            )

    meal_foods = [f for m in nutrition.get("meals", []) for f in m.get("foods", [])]
    if meal_foods:
        vr = verify_meal_macros(meal_foods)
        tag = "✓" if vr["coverage"] >= 0.6 else "~"
        fb.append(
            f"  {tag} IFCT 2017: {vr['coverage']*100:.0f}% foods verified "
            f"(sources: {vr['sources'] or 'none'})."
        )

    # ── 3. Volume appropriateness ─────────────────────────────────────────────
    fitness_level = client.get("fitness_level", "intermediate")
    goal          = client.get("goal", "maintenance")
    vol_range     = VOLUME_RANGES.get(fitness_level, {}).get(goal, (10, 20))

    agent_sets = int(workout.get("weekly_volume_sets", 0))
    if agent_sets == 0:
        agent_sets = sum(int(ex.get("sets", 0) or 0) for ex in exercises)

    if vol_range[0] <= agent_sets <= vol_range[1]:
        scores["volume_appropriateness"] = 1.0
        fb.append(
            f"✓ Volume: {agent_sets} sets/week appropriate for "
            f"{fitness_level} {goal} (range {vol_range[0]}–{vol_range[1]})."
        )
    elif agent_sets < vol_range[0]:
        scores["volume_appropriateness"] = max(0.0, 1.0 - (vol_range[0]-agent_sets)/vol_range[0])
        fb.append(
            f"✗ Volume too low: {agent_sets} sets (need ≥{vol_range[0]})."
        )
    else:
        scores["volume_appropriateness"] = max(0.0, 1.0 - (agent_sets-vol_range[1])/vol_range[1])
        fb.append(
            f"✗ Volume too high: {agent_sets} sets (max {vol_range[1]})."
        )

    # ── 4. Progressive overload ───────────────────────────────────────────────
    ex_history = progress.get("exercise_history", {})
    if ex_history and exercises:
        checks = 0
        passed = 0
        for ex in exercises:
            ex_name = ex.get("name", "")
            matched = next(
                (k for k in ex_history
                 if k.lower() in ex_name.lower() or ex_name.lower() in k.lower()),
                None,
            )
            if not matched:
                continue
            hist = ex_history[matched]
            ok, explanation = verify_agent_overload(
                exercise_name   = ex_name,
                agent_weight_kg = float(ex.get("weight_kg", 0) or 0),
                agent_reps      = str(ex.get("reps", "8-12")),
                last_weight_kg  = hist["last_weight_kg"],
                last_reps_str   = hist["last_reps_str"],
                target_reps     = hist.get("target_reps", "8-12"),
                target_sets     = hist.get("target_sets", 3),
            )
            checks += 1
            if ok:
                passed += 1
            fb.append(f"  {'✓' if ok else '✗'} Overload [{ex_name}]: {explanation}")
        if checks > 0:
            scores["progressive_overload"] = passed / checks

    # ── 5. Plateau response ───────────────────────────────────────────────────
    if "plateau" in comps and progress.get("weight_series"):
        pr = detect_plateau(progress["weight_series"], goal=client.get("goal", "maintenance"))
        if pr.status in ("plateau", "reversing"):
            adapted_volume = agent_sets >= vol_range[0] * 1.10
            if "goal_change" in " ".join(comps):
                old_t = calculate_macro_targets(client["weight_kg"], client["tdee_estimate"], "weight_loss")
                adapted_calories = agent_cal >= old_t["calories"] + 150
            else:
                old_t = calculate_macro_targets(client["weight_kg"], client["tdee_estimate"], client["goal"])
                adapted_calories = agent_cal <= old_t["calories"] - 150

            if adapted_volume or adapted_calories:
                scores["plateau_response"] = 1.0
                what = "increased volume" if adapted_volume else "adjusted calories"
                fb.append(f"✓ Plateau response: {what} correctly (slope={pr.slope_kg_per_week:+.2f} kg/wk).")
            else:
                scores["plateau_response"] = 0.0
                fb.append(
                    f"✗ Plateau detected (slope={pr.slope_kg_per_week:+.2f} kg/wk) "
                    f"but no adaptation. Increase volume ≥10% OR adjust calories ≥150 kcal."
                )
        else:
            scores["plateau_response"] = 1.0
            fb.append(f"~ Trend '{pr.status}' (slope={pr.slope_kg_per_week:+.2f} kg/wk) — no plateau.")

    # ── 6. Constraint respect ─────────────────────────────────────────────────
    injuries = client.get("injuries", [])
    dietary  = client.get("dietary_restrictions", [])

    injury_violations = []
    for injury in injuries:
        for ex in exercises:
            nl = ex.get("name", "").lower()
            for bm in CONTRAINDICATED.get(injury, []):
                if bm in nl:
                    injury_violations.append(f"'{ex.get('name')}' contraindicated for {injury}")
                    break

    dietary_violations = []
    if "vegetarian" in dietary:
        for m in ["chicken", "beef", "pork", "mutton", "lamb", "turkey", "tuna", "salmon", "prawn", "fish"]:
            if m in plan_text:
                dietary_violations.append(f"meat ({m}) for vegetarian")
                break
    if "vegan" in dietary:
        for a in ["milk", "paneer", "curd", "yogurt", "egg", "chicken", "whey", "cheese"]:
            if a in plan_text:
                dietary_violations.append(f"animal product ({a}) for vegan")
                break

    all_violations = injury_violations + dietary_violations
    if not all_violations:
        scores["constraint_respect"] = 1.0
        msg = "✓ All constraints respected."
        if injuries:
            msg += f" (Injuries: {injuries} — no banned exercises.)"
        if dietary:
            msg += f" (Diet: {dietary} — no violations.)"
        fb.append(msg)
    else:
        scores["constraint_respect"] = 0.0
        fb.append(f"✗ Constraint violations: {all_violations[:4]}.")
        if not safety_violated:
            safety_violated = True
            fb.append("  ⚠ SAFETY PENALTY −0.3 applied.")

    # ── 7. Coherence ──────────────────────────────────────────────────────────
    midpoint    = (vol_range[0] + vol_range[1]) / 2
    high_volume = agent_sets > midpoint
    tdee        = client["tdee_estimate"]

    if high_volume and agent_cal > 0 and agent_cal < tdee * 0.85:
        scores["coherence"] = 0.3
        fb.append(f"✗ Coherence: high volume ({agent_sets} sets) with low calories ({agent_cal:.0f} kcal).")
    elif goal == "muscle_gain" and agent_cal > 0 and agent_cal < tdee:
        scores["coherence"] = 0.5
        fb.append(f"~ Coherence: muscle gain needs surplus (agent: {agent_cal:.0f}, TDEE: {tdee}).")
    else:
        scores["coherence"] = 1.0
        fb.append(f"✓ Coherence: {agent_cal:.0f} kcal supports {agent_sets} sets/wk.")

    # ── 8. Actor coordination (TIGHTENED — verifies plan USES actor data) ─────
    needs_progress = bool(progress.get("weight_series") or "plateau" in comps)
    required_actors = {"fitness_advisor", "nutrition_advisor"}
    if needs_progress:
        required_actors.add("progress_analyst")

    consulted_set = set(actors_consulted)
    missing_actors = required_actors - consulted_set

    # Check plan actually USES actor data (not just consulted)
    usage_score = 0.0
    usage_checks = 0

    if "fitness_advisor" in actor_responses:
        fa_c = actor_responses["fitness_advisor"].get("constraints", {})
        fa_min = fa_c.get("weekly_sets_min", 0)
        fa_max = fa_c.get("weekly_sets_max", 999)
        if fa_min <= agent_sets <= fa_max:
            usage_score += 1.0
        usage_checks += 1

    if "nutrition_advisor" in actor_responses:
        na_cal = actor_responses["nutrition_advisor"].get("constraints", {}).get("calories_target", 0)
        if na_cal > 0 and agent_cal > 0:
            if abs(agent_cal - na_cal) / na_cal <= 0.15:
                usage_score += 1.0
            elif abs(agent_cal - na_cal) / na_cal <= 0.25:
                usage_score += 0.5
        usage_checks += 1

    if "progress_analyst" in actor_responses:
        must_adapt = actor_responses["progress_analyst"].get("constraints", {}).get("must_adapt_if_plateau", False)
        if must_adapt:
            usage_score += 1.0 if scores.get("plateau_response", 0) >= 0.5 else 0.0
        else:
            usage_score += 1.0
        usage_checks += 1

    unresolved = []
    for conflict in active_conflicts:
        ct = conflict.get("type", "")
        if ct == "plateau_volume_conflict" and scores.get("plateau_response", 1.0) < 0.5:
            unresolved.append(ct)
        elif ct == "volume_calorie_mismatch" and scores.get("coherence", 1.0) < 0.5:
            unresolved.append(ct)
        elif ct == "injury_overload_conflict" and scores.get("constraint_respect", 1.0) < 0.5:
            unresolved.append(ct)

    if missing_actors:
        consult_score = max(0.0, 1.0 - len(missing_actors) / len(required_actors))
        scores["actor_coordination"] = consult_score * 0.5
        fb.append(f"✗ Coordination: missing {sorted(missing_actors)}.")
    elif unresolved:
        scores["actor_coordination"] = max(0.0, 0.4 - 0.15 * len(unresolved))
        fb.append(f"✗ Coordination: {len(unresolved)} conflict(s) unresolved.")
    else:
        usage_pct = (usage_score / usage_checks) if usage_checks > 0 else 0.5
        scores["actor_coordination"] = round(usage_pct, 2)
        if usage_pct >= 0.8:
            fb.append(f"✓ Coordination: plan follows all actor constraints.")
        else:
            fb.append(f"~ Coordination: plan partially ignores actor data ({usage_pct:.0%}).")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    active  = list(scores.values())
    raw     = sum(active) / len(active) if active else 0.0
    penalty = 0.3 if safety_violated else 0.0
    reward  = max(0.0, min(1.0, raw - penalty))

    return reward, scores, "\n".join(fb), safety_violated


# ── Environment class ─────────────────────────────────────────────────────────

class FitcoachEnvironment(Environment):
    """
    FitCoach Multi-Actor RL Environment.

    Orchestrator agent manages 3 deterministic specialist actors:
    FitnessAdvisor, NutritionAdvisor, ProgressAnalyst.

    Implements Halluminate sub-theme of Theme 1 (Multi-Actor Environments).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_id: str = "week1_plan"):
        if task_id not in TASK_CONFIGS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. Valid: {list(TASK_CONFIGS.keys())}"
            )
        self._task_id         = task_id
        self._is_curriculum   = (task_id == "curriculum")
        self._curriculum      = CurriculumManager() if self._is_curriculum else None
        self._config          = TASK_CONFIGS[task_id]
        self._state           = State(episode_id=str(uuid4()), step_count=0)
        self._phase_idx       = 0
        self._best_score      = 0.0
        self._last_hash       = ""
        self._safety_hit      = False
        self._actors_consulted: list[str] = []
        self._actor_responses: dict[str, dict] = {}
        self._active_conflicts: list[dict] = []
        self._active_complications: list[str] = []
        self._injected_complications: list[str] = []

    def reset(self) -> FitcoachObservation:
        # Record previous episode for curriculum
        if self._is_curriculum and self._curriculum and self._best_score > 0:
            self._curriculum.record_score(self._best_score)

        # Build config (curriculum generates random clients)
        if self._is_curriculum and self._curriculum:
            ep = self._curriculum.get_next_episode()
            self._config = ep
        else:
            self._config = TASK_CONFIGS[self._task_id]

        self._state            = State(episode_id=str(uuid4()), step_count=0)
        self._phase_idx        = 0
        self._best_score       = 0.0
        self._last_hash        = ""
        self._safety_hit       = False
        self._actors_consulted = []
        self._actor_responses  = {}
        self._active_conflicts = []
        self._injected_complications = []

        cfg   = self._config
        self._active_complications = list(cfg.get("complications", []))
        phase = cfg["phases"][0]

        return FitcoachObservation(
            client_profile  = cfg["client"],
            progress_data   = cfg["progress_data"],
            complications   = cfg["complications"],
            actor_response  = {},
            actors_consulted= [],
            active_conflicts= [],
            feedback=(
                f"Episode started. Task: {self._task_id} | Phase: {phase}\n"
                f"Description: {cfg['description']}\n\n"
                "You are the ORCHESTRATOR. You must:\n"
                "  1. Consult actors using action_type='consult_actor' and actor_target=<name>\n"
                "  2. Available actors: 'fitness_advisor', 'nutrition_advisor', 'progress_analyst'\n"
                "  3. After consulting all relevant actors, submit your final plan with action_type='submit_plan'\n\n"
                "Start by consulting the actors to discover their recommendations and constraints."
            ),
            score_breakdown = {},
            task_id         = self._task_id,
            phase           = phase,
            step_count      = 0,
            best_score      = 0.0,
            done            = False,
            reward          = 0.0,
        )

    def step(self, action: FitcoachAction) -> FitcoachObservation:
        self._state.step_count += 1
        cfg       = self._config
        phases    = cfg["phases"]
        phase     = phases[min(self._phase_idx, len(phases) - 1)]
        max_steps = cfg["max_steps"]
        client    = cfg["client"]
        progress  = cfg["progress_data"]

        done_by_steps = self._state.step_count >= max_steps

        # ── Mid-episode complication injection (Theme 3.1 / Theme 1) ─────────
        # Check if any complications are scheduled for this step
        schedule = cfg.get("complication_schedule", {})
        newly_injected = []
        for trigger_step, new_comps in schedule.items():
            for comp in new_comps:
                if (self._state.step_count >= trigger_step
                        and comp not in self._active_complications
                        and comp not in self._injected_complications):
                    self._active_complications.append(comp)
                    self._injected_complications.append(comp)
                    newly_injected.append(comp)
                    # Also update client injuries if it's a new_injury
                    if comp.startswith("new_injury:"):
                        injury = comp.split(":", 1)[1]
                        if injury not in cfg["client"].get("injuries", []):
                            cfg["client"].setdefault("injuries", []).append(injury)

        comps = self._active_complications  # Use dynamic complications

        # ── Handle consult_actor ──────────────────────────────────────────────
        if action.action_type == "consult_actor":
            target = (action.actor_target or "").strip().lower()

            if target not in ALL_ACTORS:
                return FitcoachObservation(
                    client_profile  = client,
                    progress_data   = progress,
                    complications   = comps,
                    actor_response  = {
                        "error": f"Unknown actor '{target}'. "
                                 f"Valid: {sorted(ALL_ACTORS)}"
                    },
                    actors_consulted= self._actors_consulted,
                    active_conflicts= self._active_conflicts,
                    feedback        = f"✗ Unknown actor '{target}'. Choose from: {sorted(ALL_ACTORS)}",
                    score_breakdown = {},
                    task_id         = self._task_id,
                    phase           = phase,
                    step_count      = self._state.step_count,
                    best_score      = self._best_score,
                    done            = done_by_steps,
                    reward          = 0.0,
                )

            # Call the appropriate actor
            if target == "fitness_advisor":
                response = fitness_actor(client, progress)
            elif target == "nutrition_advisor":
                response = nutrition_actor(client, progress, comps)
            else:
                response = progress_actor(client, progress, comps)

            # Track consultation
            if target not in self._actors_consulted:
                self._actors_consulted.append(target)
            self._actor_responses[target] = response

            # Update conflict detection whenever we have multiple actors
            if len(self._actor_responses) >= 2:
                fa = self._actor_responses.get("fitness_advisor", {})
                na = self._actor_responses.get("nutrition_advisor", {})
                pa = self._actor_responses.get("progress_analyst", {})
                if fa and na and pa:
                    self._active_conflicts = detect_actor_conflicts(fa, na, pa)
                elif fa and na:
                    self._active_conflicts = detect_actor_conflicts(fa, na, {
                        "recommendations": {"plateau_status": "insufficient_data"},
                        "constraints": {"must_adapt_if_plateau": False, "required_actions": []},
                        "conflicts": {"with_fitness": False, "with_nutrition": False, "reason": None},
                    })

            # Alert agent about newly injected complications
            injection_alert = ""
            if newly_injected:
                injury_names = [c.split(":", 1)[1] for c in newly_injected if c.startswith("new_injury:")]
                injection_alert = (
                    f"\n\n🚨 NEW COMPLICATION(S) JUST EMERGED:\n"
                    + "\n".join(f"  - {c}" for c in newly_injected)
                    + (f"\n  Injury '{injury_names[0]}' means some exercises are now BANNED. "
                       f"Re-consult fitness_advisor to get the updated constraint list."
                       if injury_names else "")
                    + "\n  You must adapt your plan to handle this new situation."
                )

            conflict_note = ""
            if self._active_conflicts:
                conflict_note = (
                    f"\n\n⚡ {len(self._active_conflicts)} CONFLICT(S) DETECTED between actors:\n"
                    + "\n".join(
                        f"  [{i+1}] {c['type']}: {c['description']}"
                        for i, c in enumerate(self._active_conflicts)
                    )
                    + "\nYou must resolve these before submitting."
                )

            return FitcoachObservation(
                client_profile  = client,
                progress_data   = progress,
                complications   = comps,
                actor_response  = response,
                actors_consulted= list(self._actors_consulted),
                active_conflicts= list(self._active_conflicts),
                feedback=(
                    f"✓ Consulted {target}.\n"
                    f"{response['message']}"
                    + conflict_note
                    + f"\n\nActors consulted so far: {self._actors_consulted}"
                    + f"\nSteps remaining: {max_steps - self._state.step_count}"
                ),
                score_breakdown = {},
                task_id         = self._task_id,
                phase           = phase,
                step_count      = self._state.step_count,
                best_score      = self._best_score,
                done            = done_by_steps,
                reward          = 0.0,
            )

        # ── Handle submit_plan ────────────────────────────────────────────────
        if action.action_type == "submit_plan":
            # Reject empty
            w_empty = not action.workout_plan or action.workout_plan.strip() in ("", "{}", "null")
            n_empty = not action.nutrition_plan or action.nutrition_plan.strip() in ("", "{}", "null")
            if w_empty and n_empty:
                return FitcoachObservation(
                    client_profile  = client,
                    progress_data   = progress,
                    complications   = comps,
                    actor_response  = {},
                    actors_consulted= self._actors_consulted,
                    active_conflicts= self._active_conflicts,
                    feedback        = "✗ Empty plan. Provide workout_plan and nutrition_plan JSON.",
                    score_breakdown = {},
                    task_id         = self._task_id,
                    phase           = phase,
                    step_count      = self._state.step_count,
                    best_score      = self._best_score,
                    done            = done_by_steps,
                    reward          = 0.0,
                )

            # Reject duplicate
            plan_hash = hashlib.md5(
                (action.workout_plan + action.nutrition_plan).encode()
            ).hexdigest()
            if plan_hash == self._last_hash:
                return FitcoachObservation(
                    client_profile  = client,
                    progress_data   = progress,
                    complications   = comps,
                    actor_response  = {},
                    actors_consulted= self._actors_consulted,
                    active_conflicts= self._active_conflicts,
                    feedback        = "✗ Identical plan submitted twice. Revise based on actor feedback.",
                    score_breakdown = {},
                    task_id         = self._task_id,
                    phase           = phase,
                    step_count      = self._state.step_count,
                    best_score      = self._best_score,
                    done            = done_by_steps,
                    reward          = 0.0,
                )
            self._last_hash = plan_hash

            # Grade
            reward, breakdown, feedback, safety_now = grade_plan(
                action, cfg,
                self._actors_consulted,
                self._actor_responses,
                self._active_conflicts,
                self._safety_hit,
            )
            self._safety_hit = safety_now
            self._best_score = max(self._best_score, reward)

            # ── Actor Pushback — actors REVIEW and REJECT if needed ───────
            # If reward < 0.85 and we have steps left, actors push back
            # with specific suggestions instead of just ending the episode.
            try:
                workout_parsed = json.loads(action.workout_plan or "{}")
            except json.JSONDecodeError:
                workout_parsed = {}
            try:
                nutrition_parsed = json.loads(action.nutrition_plan or "{}")
            except json.JSONDecodeError:
                nutrition_parsed = {}

            rejections = collect_actor_pushback(
                workout_parsed, nutrition_parsed,
                client, progress, comps,
                self._actor_responses,
            )

            # Base done condition
            done = reward >= 0.99 or done_by_steps

            # Only run pushback if actors were actually consulted
            if self._actor_responses and reward < 0.85 and not done_by_steps:
                if rejections:
                    pushback_msgs = "\n\n".join(r["message"] for r in rejections)
                    accepting = [a for a in self._actors_consulted
                                 if a not in [r["actor"] for r in rejections]]
                    accept_msg = ("\n✓ ACCEPTED by: " + ", ".join(accepting)) if accepting else ""
                    feedback += (
                        f"\n\n{'='*50}\n"
                        f"ACTOR REVIEW — {len(rejections)} actor(s) REJECTED your plan:\n"
                        f"{pushback_msgs}"
                        f"{accept_msg}\n"
                        f"{'='*50}\n"
                        f"Revise your plan addressing ALL rejections above then resubmit."
                    )
                    done = False  # keep episode alive for revision

            # Show acceptance message only when reward is good
            if reward >= 0.85 and self._actor_responses:
                feedback += "\n\n✓ ALL ACTORS ACCEPTED your plan."

            # Advance phase
            if reward >= 0.75 and self._phase_idx < len(phases) - 1:
                self._phase_idx += 1
                feedback += f"\n\n→ Phase complete! Moving to: {phases[self._phase_idx]}."

            return FitcoachObservation(
                client_profile  = client,
                progress_data   = progress,
                complications   = comps,
                actor_response  = {},
                actors_consulted= self._actors_consulted,
                active_conflicts= self._active_conflicts,
                feedback        = feedback,
                score_breakdown = breakdown,
                task_id         = self._task_id,
                phase           = phase,
                step_count      = self._state.step_count,
                best_score      = self._best_score,
                done            = done,
                reward          = reward,
            )

        # Unknown action type
        return FitcoachObservation(
            client_profile  = client,
            progress_data   = progress,
            complications   = comps,
            actor_response  = {},
            actors_consulted= self._actors_consulted,
            active_conflicts= self._active_conflicts,
            feedback        = (
                f"✗ Unknown action_type '{action.action_type}'. "
                f"Use 'consult_actor' or 'submit_plan'."
            ),
            score_breakdown = {},
            task_id         = self._task_id,
            phase           = phase,
            step_count      = self._state.step_count,
            best_score      = self._best_score,
            done            = done_by_steps,
            reward          = 0.0,
        )

    @property
    def state(self) -> State:
        return self._state