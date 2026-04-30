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

# FitCoach — Multi-Actor Fitness Orchestrator RL Environment

YouTube demo: https://www.youtube.com/watch?v=2GxX_Ie5_7o

An OpenEnv-compliant RL environment where an LLM agent plays an **orchestrator** coordinating three deterministic specialist actors to produce integrated fitness + nutrition prescriptions.

**Hackathon Themes:** Multi-Agent Interactions (Theme 1 — Halluminate sub-theme) | Professional Tasks (Theme 3.1) | Self-Improvement (Theme 4 — Snorkel AI sub-theme)

---

## What Makes This Environment Interesting

The agent does NOT generate fitness plans in isolation. It must:

1. **Consult** three specialist actors (FitnessAdvisor, NutritionAdvisor, ProgressAnalyst) to discover constraints
2. **Detect conflicts** between actors (e.g., high volume vs low calories, injury vs overload demands)
3. **Resolve conflicts** and submit a final integrated plan
4. **Handle mid-episode complications** (injuries injected mid-episode, goal changes)
5. **Adapt to adaptive curriculum** — random clients each episode, difficulty escalates with performance

The actors are **deterministic rule engines**, not LLMs. The LLM being trained is the orchestrator that manages them.

---

## Action Schema

Every action is a `FitcoachAction` with these fields:

| Field | Type | When required | Description |
|---|---|---|---|
| `action_type` | str | always | `"consult_actor"` or `"submit_plan"` |
| `actor_target` | str \| null | when consulting | `"fitness_advisor"`, `"nutrition_advisor"`, or `"progress_analyst"` |
| `workout_plan` | str (JSON) | when submitting | JSON string of workout structure |
| `nutrition_plan` | str (JSON) | when submitting | JSON string of nutrition structure |
| `reasoning` | str \| null | recommended on submit | How conflicts were resolved |

> ⚠️ Note: `workout_plan` and `nutrition_plan` are JSON **strings**, not nested objects. Use `json.dumps(...)` when constructing them in Python.

---

## Reward Dimensions (8 total, scored 0–1)

| Dimension | What It Measures |
|---|---|
| `equipment_compliance` | Exercises match available equipment only |
| `macro_accuracy` | Macros within ±15% of IFCT 2017 formula targets |
| `volume_appropriateness` | Weekly sets in correct range for fitness level × goal |
| `progressive_overload` | Correct double-progression applied to exercise history |
| `plateau_response` | Adapted volume/calories when plateau detected |
| `constraint_respect` | No contraindicated exercises or dietary violations |
| `coherence` | Nutrition supports training volume (no high volume + low cal) |
| `actor_coordination` | Consulted all actors, plan follows their constraints |

Safety penalty: **−0.3** for any hard constraint violation (injury-banned exercise or dietary violation).

---

## Tasks

| Task | Difficulty | Description |
|---|---|---|
| `week1_plan` | Easy | Fresh beginner, vegetarian, dumbbells only. Consult actors, submit valid plan. |
| `plateau_adaptation` | Medium | 14-day weight plateau. Actors conflict on adaptation. Knee injury injected mid-episode at step 4. |
| `conflict_resolution` | Hard | 3 simultaneous challenges: plateau + lower-back injury + goal change. All actors conflict. |
| `curriculum` | Adaptive | Random clients each episode. Difficulty escalates easy → medium → hard with performance. |

---

## Quick Start (Python)

```python
import json
from FitCoach import FitcoachAction, FitcoachEnv

with FitcoachEnv(base_url="http://localhost:8000") as env:
    # 1. Reset to start an episode
    result = env.reset()
    print("Client:", result.observation.client_profile)
    print("Complications:", result.observation.complications)

    # 2. Consult fitness_advisor
    result = env.step(FitcoachAction(
        action_type="consult_actor",
        actor_target="fitness_advisor",
    ))
    print("Fitness:", result.observation.actor_response["message"])

    # 3. Consult nutrition_advisor
    result = env.step(FitcoachAction(
        action_type="consult_actor",
        actor_target="nutrition_advisor",
    ))
    print("Nutrition:", result.observation.actor_response["message"])

    # 4. Consult progress_analyst
    result = env.step(FitcoachAction(
        action_type="consult_actor",
        actor_target="progress_analyst",
    ))
    print("Progress:", result.observation.actor_response["message"])
    print("Conflicts detected:", result.observation.active_conflicts)

    # 5. Submit final plan (note: workout_plan and nutrition_plan are JSON STRINGS)
    workout = {
        "days": [
            {
                "name": "Day 1 - Upper",
                "focus": "chest, back",
                "exercises": [
                    {"name": "Dumbbell Bench Press", "sets": 3, "reps": "8-12",
                     "rest_seconds": 90, "weight_kg": 15},
                    {"name": "Pull-up", "sets": 3, "reps": "6-10",
                     "rest_seconds": 90, "weight_kg": 0},
                ],
            },
            {
                "name": "Day 2 - Lower",
                "focus": "legs",
                "exercises": [
                    {"name": "Dumbbell Squat", "sets": 3, "reps": "8-12",
                     "rest_seconds": 90, "weight_kg": 15},
                    {"name": "Dumbbell Romanian Deadlift", "sets": 3, "reps": "10-12",
                     "rest_seconds": 90, "weight_kg": 15},
                ],
            },
            {
                "name": "Day 3 - Push/Pull",
                "focus": "shoulders, arms",
                "exercises": [
                    {"name": "Dumbbell Shoulder Press", "sets": 2, "reps": "8-12",
                     "rest_seconds": 90, "weight_kg": 10},
                    {"name": "Dumbbell Row", "sets": 2, "reps": "8-12",
                     "rest_seconds": 90, "weight_kg": 12},
                ],
            },
        ],
        "weekly_volume_sets": 16,
        "notes": "16 sets/week — top of beginner muscle-gain range",
    }

    nutrition = {
        "daily_targets": {
            "calories": 2650, "protein_g": 144, "carbs_g": 352, "fats_g": 74,
        },
        "meals": [
            {"meal_name": "Breakfast",
             "foods": ["100g oats", "200ml milk", "1 banana", "30g whey protein"],
             "calories": 600, "protein_g": 40},
            {"meal_name": "Lunch",
             "foods": ["150g paneer", "100g brown rice", "100g spinach"],
             "calories": 700, "protein_g": 35},
            {"meal_name": "Snack",
             "foods": ["100g rajma", "1 roti", "30g almonds"],
             "calories": 550, "protein_g": 20},
            {"meal_name": "Dinner",
             "foods": ["150g chana", "150g brown rice", "100g curd"],
             "calories": 650, "protein_g": 35},
        ],
    }

    result = env.step(FitcoachAction(
        action_type="submit_plan",
        workout_plan=json.dumps(workout),
        nutrition_plan=json.dumps(nutrition),
        reasoning=(
            "Beginner vegetarian muscle gain. Used dumbbells + pull-up bar only. "
            "16 sets/week = top of beginner range. Calories 2650 = TDEE 2400 + 250 "
            "surplus per nutrition_advisor. Protein 144g = 2.0g/kg. No plateau, "
            "no adaptation needed."
        ),
    ))
    print(f"Reward: {result.reward}")
    print("Score breakdown:", result.observation.score_breakdown)
    print("Feedback:", result.observation.feedback)
```

---

## Using the Hugging Face Playground

The Space ships with a Gradio playground for manual testing. Here's the exact flow:

1. **Press `Reset`** first. The Status box will populate with the client profile.
2. **Consult each actor** (3 separate Step calls):
   - Set `Action Type = consult_actor`
   - Set `Actor Target = fitness_advisor` (then `nutrition_advisor`, then `progress_analyst`)
   - Leave `Workout Plan`, `Nutrition Plan`, `Reasoning` empty
   - Press `Step`
3. **Submit your plan**:
   - Set `Action Type = submit_plan`
   - Leave `Actor Target` empty
   - Paste a valid JSON object into `Workout Plan` (e.g., the `workout` dict from above, JSON-serialized)
   - Paste a valid JSON object into `Nutrition Plan`
   - Add `Reasoning` explaining how you used each actor's constraints
   - Press `Step`
4. Read the `score_breakdown` and `feedback` in the Raw JSON response.

Aim for reward **≥ 0.85** for full actor acceptance. If actors reject, the episode continues and you can revise.

---

## Episode Flow

```
Reset → Client profile + complications

consult_actor(fitness_advisor)   → volume range, equipment, banned exercises
consult_actor(nutrition_advisor) → macro targets, banned foods, IFCT 2017
consult_actor(progress_analyst)  → plateau status, overload signals

Conflicts detected between actors (shown in observation.active_conflicts)

submit_plan(workout + nutrition + reasoning)

If score < 0.85: actors REJECT with specific fixes → agent revises
If score ≥ 0.85: all actors ACCEPT → episode ends
```

---

## Actor Pushback System

After the agent submits a plan, each actor **reviews** it against its own constraints:

- **FitnessAdvisor** checks volume range, equipment, banned exercises → suggests equipment swaps
- **NutritionAdvisor** checks calorie/protein targets, banned foods → suggests IFCT 2017 alternatives
- **ProgressAnalyst** checks plateau adaptation → requires volume/calorie changes

If any actor rejects, the episode continues and the agent must revise. This transforms the environment from a passive grader into an **active negotiation arena**.

---

## Domain Knowledge

- **Nutrition**: Grounded in IFCT 2017 (Indian Food Composition Tables, NIN Hyderabad) + USDA FoodData. 30+ foods with verified macros per 100g.
- **Plateau Detection**: 7-day rolling mean + OLS linear regression. Classifies trend as `plateau` / `on_track` / `overshooting` / `reversing`.
- **Progressive Overload**: Double-progression rules — add weight when all sets hit top of rep range, deload on heavy misses.
- **Injury Safety**: Contraindicated exercise lists per injury type with safe alternatives.

---

## Running Locally

```bash
# Pick a task and start the server
FITCOACH_TASK=week1_plan         uvicorn server.app:app --host 0.0.0.0 --port 8000
FITCOACH_TASK=plateau_adaptation uvicorn server.app:app --host 0.0.0.0 --port 8000
FITCOACH_TASK=conflict_resolution uvicorn server.app:app --host 0.0.0.0 --port 8000
FITCOACH_TASK=curriculum         uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Then in a separate process, run the orchestrator:

```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export API_KEY="gsk_..."
export MODEL_NAME="llama-3.3-70b-versatile"
export FITCOACH_TASK="week1_plan"
python inference.py
```

---

## Project Structure

```
FitCoach/
├── models.py                    # Action/Observation Pydantic models
├── client.py                    # WebSocket client (FitcoachEnv)
├── inference.py                 # Multi-actor orchestrator agent
├── test_pushback.py             # End-to-end pushback + injury injection test
├── openenv.yaml                 # OpenEnv manifest (4 tasks)
├── pyproject.toml
├── server/
│   ├── FitCoach_environment.py  # Core environment + 8-dimension grader
│   ├── app.py                   # FastAPI application
│   └── Dockerfile               # Container build
└── utils/
    ├── actors.py                # 3 deterministic specialist actors
    ├── pushback.py              # Actor review + rejection engine
    ├── nutrition.py             # IFCT 2017 nutrition database
    ├── plateau.py               # Statistical plateau detection
    ├── overload.py              # Progressive overload verification
    └── curriculum.py            # Adaptive curriculum manager (Theme 4)
```

---

## Common Pitfalls

- **Forgetting to JSON-encode plans**: `workout_plan` and `nutrition_plan` must be **strings**, not dicts. Use `json.dumps(...)`.
- **Empty `actor_target` on consult**: server will reject with `"Unknown actor ''"`. Always set it when `action_type="consult_actor"`.
- **Submitting before consulting**: you can technically do it, but `actor_coordination` will score near 0 because no actor data was used.
- **Submitting an identical plan twice**: rejected with `"Identical plan submitted twice. Revise based on actor feedback."`
- **Ignoring `active_conflicts`**: conflicts left unresolved tank the `actor_coordination` score even if other dimensions look fine.
