"""
test_pushback_ws.py — Full multi-actor negotiation + mid-episode complication test.

Shows:
  1. Actor pushback (rejection with specific fixes)
  2. Mid-episode complication injection (knee injury at step 4)
  3. Agent must re-consult actors after injury and revise plan
  4. Final acceptance by all actors

Run with: python test_pushback_ws.py
"""

import asyncio, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import FitcoachAction, FitcoachObservation
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from typing import Dict


class FitcoachEnv(EnvClient[FitcoachAction, FitcoachObservation, State]):
    def _step_payload(self, action):
        payload = {
            "action_type":    action.action_type,
            "workout_plan":   action.workout_plan,
            "nutrition_plan": action.nutrition_plan,
        }
        if action.actor_target is not None:
            payload["actor_target"] = action.actor_target
        if action.reasoning is not None:
            payload["reasoning"] = action.reasoning
        return payload

    def _parse_result(self, payload):
        obs = payload.get("observation", {})
        return StepResult(
            observation=FitcoachObservation(
                client_profile  =obs.get("client_profile", {}),
                progress_data   =obs.get("progress_data", {}),
                complications   =obs.get("complications", []),
                actor_response  =obs.get("actor_response", {}),
                actors_consulted=obs.get("actors_consulted", []),
                active_conflicts=obs.get("active_conflicts", []),
                feedback        =obs.get("feedback", ""),
                score_breakdown =obs.get("score_breakdown", {}),
                task_id         =obs.get("task_id", ""),
                phase           =obs.get("phase", ""),
                step_count      =obs.get("step_count", 0),
                best_score      =obs.get("best_score", 0.0),
                done            =payload.get("done", False),
                reward          =payload.get("reward"),
                metadata        =obs.get("metadata", {}),
            ),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload):
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )


# ── Plans ─────────────────────────────────────────────────────────────────────

BAD_WORKOUT = json.dumps({
    "days": [{"name": "Day 1", "focus": "strength", "exercises": [
        {"name": "Barbell Squat",       "sets": 5, "reps": "5", "rest_seconds": 180, "weight_kg": 100},
        {"name": "Barbell Deadlift",    "sets": 5, "reps": "5", "rest_seconds": 180, "weight_kg": 120},
        {"name": "Barbell Bench Press", "sets": 5, "reps": "5", "rest_seconds": 180, "weight_kg": 80},
    ]}],
    "weekly_volume_sets": 30,
    "notes": "powerlifting — wrong for this client"
})

BAD_NUTRITION = json.dumps({
    "daily_targets": {"calories": 1200, "protein_g": 40, "carbs_g": 100, "fats_g": 20},
    "meals": []
})

# After knee injury — no lunges, no deep squats, knee-safe exercises only
REVISED_WORKOUT = json.dumps({
    "days": [
        {"name": "Day 1 - Upper", "focus": "chest/back", "exercises": [
            {"name": "Barbell Bench Press", "sets": 3, "reps": "8-12", "rest_seconds": 90, "weight_kg": 50},
            {"name": "Barbell Row",         "sets": 3, "reps": "8-12", "rest_seconds": 90, "weight_kg": 45},
            {"name": "Dumbbell Shoulder Press","sets": 1,"reps": "10-12","rest_seconds": 60,"weight_kg": 14},
        ]},
        {"name": "Day 2 - Lower (knee-safe)", "focus": "glutes/hamstrings", "exercises": [
            {"name": "Hip Thrust",              "sets": 3, "reps": "10-12", "rest_seconds": 90, "weight_kg": 60},
            {"name": "Dumbbell Romanian Deadlift","sets": 3,"reps": "10-12","rest_seconds": 90,"weight_kg": 22},
            {"name": "Leg Press (shallow range)","sets": 1,"reps": "12-15","rest_seconds": 60,"weight_kg": 80},
        ]},
        {"name": "Day 3 - Upper", "focus": "back/biceps", "exercises": [
            {"name": "Cable Row",   "sets": 3, "reps": "10-12", "rest_seconds": 90, "weight_kg": 35},
            {"name": "Lat Pulldown","sets": 1, "reps": "10-12", "rest_seconds": 60, "weight_kg": 30},
        ]},
    ],
    "weekly_volume_sets": 18,
    "notes": "Knee-safe plan — 18 sets within intermediate range. No knee-contraindicated exercises."
})

# Priya: weight_loss, tdee=2100, target ~1700 kcal (tdee - 400)
REVISED_NUTRITION = json.dumps({
    "daily_targets": {"calories": 1700, "protein_g": 143, "carbs_g": 155, "fats_g": 47},
    "meals": [
        {"meal_name": "Breakfast",
         "foods": ["3 boiled eggs", "1 slice whole grain bread", "100g curd"],
         "calories": 380, "protein_g": 28},
        {"meal_name": "Lunch",
         "foods": ["120g chicken breast", "100g brown rice", "100g broccoli"],
         "calories": 450, "protein_g": 42},
        {"meal_name": "Snack",
         "foods": ["30g almonds", "1 banana"],
         "calories": 270, "protein_g": 8},
        {"meal_name": "Dinner",
         "foods": ["150g fish", "100g spinach", "1 roti"],
         "calories": 380, "protein_g": 38},
        {"meal_name": "Post-workout",
         "foods": ["200ml milk", "30g whey protein"],
         "calories": 220, "protein_g": 27},
    ]
})


def sep(title=""):
    print(f"\n{'='*60}")
    if title:
        print(f"  {title}")
        print(f"{'='*60}")


async def main():
    env = FitcoachEnv(base_url="http://localhost:8000")

    sep("FITCOACH-RL PUSHBACK + MID-EPISODE COMPLICATION TEST")
    print("Task: plateau_adaptation (Priya Menon, intermediate, weight_loss)")
    print("Complication schedule: knee injury injected at step 4")

    # Reset
    result = await env.reset()
    obs = result.observation
    client = obs.client_profile
    print(f"\nClient:     {client.get('name')}")
    print(f"Level:      {client.get('fitness_level')} | Goal: {client.get('goal')}")
    print(f"Equipment:  {client.get('available_equipment')}")
    print(f"Injuries:   {client.get('injuries') or 'none (so far)'}")
    print(f"Complications: {obs.complications}")

    # Step 1: Consult fitness
    sep("Step 1: consult fitness_advisor")
    result = await env.step(FitcoachAction(
        action_type="consult_actor", actor_target="fitness_advisor",
        workout_plan="{}", nutrition_plan="{}"
    ))
    fa = result.observation.actor_response
    print(f"Volume range: {fa.get('constraints', {}).get('weekly_sets_min')}–{fa.get('constraints', {}).get('weekly_sets_max')} sets")
    print(f"Banned:       {fa.get('constraints', {}).get('must_avoid_exercises', [])}")
    print(f"Consulted:    {result.observation.actors_consulted}")

    # Step 2: Consult nutrition
    sep("Step 2: consult nutrition_advisor")
    result = await env.step(FitcoachAction(
        action_type="consult_actor", actor_target="nutrition_advisor",
        workout_plan="{}", nutrition_plan="{}"
    ))
    na = result.observation.actor_response
    print(f"Calorie target: {na.get('constraints', {}).get('calories_target')} kcal")
    print(f"Protein min:    {na.get('constraints', {}).get('protein_minimum_g')}g")
    print(f"Consulted:      {result.observation.actors_consulted}")

    # Step 3: Consult progress analyst
    sep("Step 3: consult progress_analyst")
    result = await env.step(FitcoachAction(
        action_type="consult_actor", actor_target="progress_analyst",
        workout_plan="{}", nutrition_plan="{}"
    ))
    pa = result.observation.actor_response
    print(f"Plateau status: {pa.get('recommendations', {}).get('plateau_status')}")
    print(f"Must adapt:     {pa.get('constraints', {}).get('must_adapt_if_plateau')}")
    print(f"Conflicts:      {len(result.observation.active_conflicts)} detected")
    print(f"Consulted:      {result.observation.actors_consulted}")

    # Step 4: Submit BAD plan — AND knee injury injected this step
    sep("Step 4: submit BAD plan + 🚨 KNEE INJURY INJECTION")
    print("(Submitting wrong plan: barbell-heavy, 30 sets, 1200 kcal)")
    print("(Also: complication_schedule fires → new_injury:knee injected)")
    result = await env.step(FitcoachAction(
        action_type="submit_plan",
        workout_plan=BAD_WORKOUT,
        nutrition_plan=BAD_NUTRITION,
        reasoning="initial attempt"
    ))
    print(f"\nReward: {result.reward:.2f} | Done: {result.done}")
    print(f"\nFeedback:\n{result.observation.feedback}")

    if result.done:
        print("\n[Episode ended early]")
        await env.close()
        return

    # Step 5: Re-consult fitness_advisor — NOW sees knee injury in client
    sep("Step 5: re-consult fitness_advisor (post-injury)")
    print("Agent re-consults after injury injection to get updated constraints")
    result = await env.step(FitcoachAction(
        action_type="consult_actor", actor_target="fitness_advisor",
        workout_plan="{}", nutrition_plan="{}"
    ))
    fa2 = result.observation.actor_response
    print(f"Updated banned exercises: {fa2.get('constraints', {}).get('must_avoid_exercises', [])}")
    print(f"Client injuries now:      {result.observation.client_profile.get('injuries')}")
    print(f"Feedback:\n{result.observation.feedback[:300]}")

    # Step 6: Submit REVISED plan — knee-safe, correct macros, right volume
    sep("Step 6: submit REVISED plan (knee-safe + correct macros)")
    print("Revised: no lunges/squats, hip thrusts instead, 1700 kcal for weight loss")
    result = await env.step(FitcoachAction(
        action_type="submit_plan",
        workout_plan=REVISED_WORKOUT,
        nutrition_plan=REVISED_NUTRITION,
        reasoning=(
            "Revised after actor rejections: "
            "replaced barbell-only exercises with barbell+cable alternatives, "
            "reduced volume to 21 sets (within 12-18 intermediate range), "
            "adjusted calories to 1700 kcal for weight_loss goal, "
            "removed all knee-contraindicated exercises (lunges, deep squats) "
            "after new_injury:knee was injected at step 4."
        )
    ))
    print(f"\nReward: {result.reward:.2f} | Done: {result.done}")
    print(f"\nFeedback:\n{result.observation.feedback}")
    print(f"\nScore breakdown:")
    for k, v in result.observation.score_breakdown.items():
        icon = "✓" if v >= 0.8 else ("~" if v >= 0.5 else "✗")
        print(f"  {icon} {k}: {v:.2f}")

    await env.close()
    sep("TEST COMPLETE")
    print(f"Final reward: {result.reward:.2f}")
    if result.reward >= 0.85:
        print("✓ Agent successfully negotiated with all actors and adapted to mid-episode injury!")
    else:
        print("~ Plan partially accepted — more revision needed")


asyncio.run(main())