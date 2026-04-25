---
title: FitCoach Multi-Actor RL Environment
colorFrom: red
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# FitCoach -- Multi-Actor Fitness Orchestrator RL Environment

An OpenEnv-compliant RL environment where an LLM agent plays an **orchestrator** coordinating three deterministic specialist actors to produce integrated fitness + nutrition prescriptions.

**Hackathon Themes:** Multi-Agent Interactions (Theme 1 -- Halluminate sub-theme) | Professional Tasks (Theme 3.1) | Self-Improvement (Theme 4 -- Snorkel AI sub-theme)

## What Makes This Environment Interesting

The agent does NOT generate fitness plans in isolation. It must:

1. **Consult** three specialist actors (FitnessAdvisor, NutritionAdvisor, ProgressAnalyst) to discover constraints
2. **Detect conflicts** between actors (e.g., high volume vs low calories, injury vs overload demands)
3. **Resolve conflicts** and submit a final integrated plan
4. **Handle mid-episode complications** (injuries injected mid-episode, goal changes)
5. **Adapt to adaptive curriculum** -- random clients each episode, difficulty escalates with performance

The actors are **deterministic rule engines**, not LLMs. The LLM being trained is the orchestrator that manages them.

## Reward Dimensions (8 total, scored 0-1)

| Dimension | What It Measures |
|---|---|
| `equipment_compliance` | Exercises match available equipment only |
| `macro_accuracy` | Macros within +/-15% of IFCT 2017 formula targets |
| `volume_appropriateness` | Weekly sets in correct range for fitness level x goal |
| `progressive_overload` | Correct double-progression applied to exercise history |
| `plateau_response` | Adapted volume/calories when plateau detected |
| `constraint_respect` | No contraindicated exercises or dietary violations |
| `coherence` | Nutrition supports training volume (no high volume + low cal) |
| `actor_coordination` | Consulted all actors, plan follows their constraints |

Safety penalty: 0.3 for any hard constraint violation (injury-banned exercise or dietary violation).

## Tasks

| Task | Difficulty | Description |
|---|---|---|
| `week1_plan` | Easy | Fresh beginner, vegetarian, dumbbells only. Consult actors, submit valid plan. |
| `plateau_adaptation` | Medium | 14-day weight plateau. Actors conflict on adaptation. Knee injury injected mid-episode. |
| `conflict_resolution` | Hard | 3 simultaneous challenges: plateau + lower-back injury + goal change. All actors conflict. |
| `curriculum` | Adaptive | Random clients each episode. Difficulty escalates easy->medium->hard with performance. |

## Quick Start

```python
from FitCoach import FitcoachAction, FitcoachEnv

with FitcoachEnv(base_url="http://localhost:8000") as env:
    # Reset -- get client profile
    result = env.reset()
    print(result.observation.client_profile)
    print(result.observation.complications)

    # Step 1: Consult fitness advisor
    result = env.step(FitcoachAction(
        action_type="consult_actor",
        actor_target="fitness_advisor",
    ))
    print(result.observation.actor_response)  # volume range, banned exercises

    # Step 2: Consult nutrition advisor
    result = env.step(FitcoachAction(
        action_type="consult_actor",
        actor_target="nutrition_advisor",
    ))
    print(result.observation.actor_response)  # calorie target, banned foods

    # Step 3: Consult progress analyst
    result = env.step(FitcoachAction(
        action_type="consult_actor",
        actor_target="progress_analyst",
    ))
    print(result.observation.active_conflicts)  # conflicts to resolve

    # Step 4: Submit integrated plan
    import json
    result = env.step(FitcoachAction(
        action_type="submit_plan",
        workout_plan=json.dumps({
            "days": [{"name": "Day 1", "focus": "upper", "exercises": [
                {"name": "Dumbbell Bench Press", "sets": 3, "reps": "8-12",
                 "rest_seconds": 90, "weight_kg": 15}
            ]}],
            "weekly_volume_sets": 14,
        }),
        nutrition_plan=json.dumps({
            "daily_targets": {"calories": 2650, "protein_g": 144,
                              "carbs_g": 352, "fats_g": 74},
            "meals": [{"meal_name": "Breakfast",
                       "foods": ["100g oats", "200ml milk", "1 banana"],
                       "calories": 450, "protein_g": 18}],
        }),
        reasoning="Resolved volume-calorie conflict by keeping sets at 14..."
    ))
    print(f"Reward: {result.reward}")
    print(result.observation.score_breakdown)
```

## Episode Flow

```
Reset -> Client profile + complications
  
consult_actor(fitness_advisor)   -> volume range, equipment, banned exercises
consult_actor(nutrition_advisor) -> macro targets, banned foods, IFCT 2017
consult_actor(progress_analyst)  -> plateau status, overload signals
  
Conflicts detected between actors (shown in observation)
  
submit_plan(workout + nutrition + reasoning)
  
If score < 0.85: actors REJECT with specific fixes -> agent revises
If score >= 0.85: all actors ACCEPT -> episode ends
```

## Actor Pushback System

After the agent submits a plan, each actor **reviews** it against their own constraints:

- **FitnessAdvisor** checks volume range, equipment, banned exercises -> suggests equipment swaps
- **NutritionAdvisor** checks calorie/protein targets, banned foods -> suggests IFCT 2017 alternatives
- **ProgressAnalyst** checks plateau adaptation -> requires volume/calorie changes

If any actor rejects, the episode continues and the agent must revise. This transforms the environment from a passive grader into an **active negotiation arena**.

## Domain Knowledge

- **Nutrition**: Grounded in IFCT 2017 (Indian Food Composition Tables, NIN Hyderabad) + USDA FoodData. 30+ foods with verified macros per 100g.
- **Plateau Detection**: 7-day rolling mean + OLS linear regression. Classifies trend as plateau/on_track/overshooting/reversing.
- **Progressive Overload**: Double-progression rules -- add weight when all sets hit top of rep range, deload on heavy misses.
- **Injury Safety**: Contraindicated exercise lists per injury type with safe alternatives.

## Running Locally

```bash
# Start server (choose task)
FITCOACH_TASK=week1_plan uvicorn server.app:app --host 0.0.0.0 --port 8000

# Or curriculum mode for adaptive difficulty
FITCOACH_TASK=curriculum uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Training

See the [training notebook](FitCoach_RL_Training_Unsloth.ipynb) for GRPO training with Unsloth on Qwen2.5-1.5B.

## Project Structure

```
FitCoach/
 models.py                    # Action/Observation Pydantic models
 client.py                    # WebSocket client (FitcoachEnv)
 inference.py                 # Multi-actor orchestrator agent
 baseline_weak.py             # Untrained baseline for comparison
 openenv.yaml                 # OpenEnv manifest (4 tasks)
 server/
    FitCoach_environment.py  # Core environment + 8-dimension grader
    app.py                   # FastAPI application
    Dockerfile               # Container build
 utils/
     actors.py                # 3 deterministic specialist actors
     pushback.py              # Actor review + rejection engine
     nutrition.py             # IFCT 2017 nutrition database
     plateau.py               # Statistical plateau detection
     overload.py              # Progressive overload verification
     curriculum.py            # Adaptive curriculum manager (Theme 4)
```