"""
FitCoach inference.py — Multi-Actor Orchestrator.

The agent must:
  1. Consult all relevant specialist actors
  2. Detect and reason about conflicts between them
  3. Submit a final integrated plan resolving all conflicts

Stdout format (exact hackathon spec):
    [START] task=<task> env=fitcoach_env model=<model>
    [STEP]  step=<N> action=<text> reward=<R:.2f> done=<bool> error=<null|msg>
    [END]   success=<bool> steps=<N> score=<score:.2f> rewards=<r1:.2f,...>

LOCAL USAGE:
    $env:FITCOACH_TASK="week1_plan"
    uvicorn server.app:app --host 0.0.0.0 --port 8000

    $env:API_BASE_URL="https://api.groq.com/openai/v1"
    $env:API_KEY="gsk_..."
    $env:MODEL_NAME="llama-3.3-70b-versatile"
    $env:FITCOACH_TASK="week1_plan"
    $env:USE_DOCKER="false"
    python inference.py
"""

import asyncio
import json
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from models import FitcoachAction, FitcoachObservation
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from typing import Dict


class FitcoachEnv(EnvClient[FitcoachAction, FitcoachObservation, State]):
    def _step_payload(self, action: FitcoachAction) -> Dict:
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

    def _parse_result(self, payload: Dict) -> StepResult[FitcoachObservation]:
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

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )


# ── Config ────────────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
API_KEY      = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY", "")
BENCHMARK        = "fitcoach_env"
USE_DOCKER       = os.environ.get("USE_DOCKER", "false").lower() == "true"
IMAGE_NAME       = os.environ.get("LOCAL_IMAGE_NAME", "fitcoach-env:latest")
LOCAL_SERVER_URL = os.environ.get("LOCAL_SERVER_URL", "http://localhost:8000")
FITCOACH_TASK    = os.environ.get("FITCOACH_TASK", "")
MAX_STEPS        = int(os.environ.get("MAX_STEPS", "10"))

ALL_TASKS = ["week1_plan", "plateau_adaptation", "conflict_resolution"]


# ── Log helpers ───────────────────────────────────────────────────────────────

def log_start(task, env_name, model):
    print(f"[START] task={task} env={env_name} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    err = str(error) if error else "null"
    act = str(action).replace("\n", " ").replace("\r", "")[:120]
    print(
        f"[STEP] step={step} action={act} reward={reward:.2f}"
        f" done={str(done).lower()} error={err}",
        flush=True,
    )

def log_end(success, steps, score, rewards):
    rstr = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps}"
        f" score={score:.2f} rewards={rstr}",
        flush=True,
    )


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI FITNESS ORCHESTRATOR managing three specialist actors:
- fitness_advisor: workout programming expert
- nutrition_advisor: diet and macro expert (uses IFCT 2017 database)
- progress_analyst: plateau detection and overload expert

Your job is to coordinate these actors to produce the best fitness prescription.

EPISODE FLOW:
1. First, consult each relevant actor using action_type='consult_actor'
2. Read their recommendations and detect conflicts between them
3. Resolve all conflicts
4. Submit your final integrated plan using action_type='submit_plan'

RESPOND WITH ONLY A JSON OBJECT. Choose one of these formats:

FORMAT A — Consult an actor:
{
  "action_type": "consult_actor",
  "actor_target": "fitness_advisor",
  "workout_plan": "{}",
  "nutrition_plan": "{}"
}

FORMAT B — Submit final plan (after consulting all actors):
{
  "action_type": "submit_plan",
  "actor_target": null,
  "workout_plan": {
    "days": [
      {
        "name": "Day 1 - Push",
        "focus": "chest, shoulders, triceps",
        "exercises": [
          {"name": "Dumbbell Bench Press", "sets": 3, "reps": "8-12", "rest_seconds": 90, "weight_kg": 20}
        ]
      }
    ],
    "weekly_volume_sets": 15,
    "notes": "Evidence-based plan based on actor recommendations"
  },
  "nutrition_plan": {
    "daily_targets": {"calories": 2650, "protein_g": 144, "carbs_g": 352, "fats_g": 74},
    "meals": [
      {"meal_name": "Breakfast", "foods": ["100g oats", "200ml milk", "1 banana"], "calories": 450, "protein_g": 18}
    ]
  },
  "reasoning": "Consulted all 3 actors. Resolved volume-calorie conflict by..."
}

CRITICAL RULES:
1. Always consult fitness_advisor FIRST to know equipment and volume constraints
2. Always consult nutrition_advisor to get IFCT-verified macro targets
3. Consult progress_analyst if there is any progress_data or complications
4. If actors CONFLICT, explicitly resolve it in your reasoning field
5. ONLY use equipment listed in available_equipment
6. NEVER use exercises banned due to injuries
7. weekly_volume_sets must be within the range fitness_advisor specifies
8. Calories must be within 15% of nutrition_advisor's target
"""


# ── LLM call ──────────────────────────────────────────────────────────────────

def _call_llm_sync(messages):
    from openai import OpenAI
    if not API_KEY:
        raise ValueError("API_KEY or HF_TOKEN not set.")
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    resp = client.chat.completions.create(
        model=MODEL_NAME, messages=messages, max_tokens=1500, temperature=0.7,
    )
    return resp.choices[0].message.content

async def call_llm(messages):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _call_llm_sync, messages)


# ── Message builder ────────────────────────────────────────────────────────────

def build_user_message(obs) -> str:
    profile   = getattr(obs, "client_profile",   {})
    progress  = getattr(obs, "progress_data",    {})
    comps     = getattr(obs, "complications",    [])
    feedback  = getattr(obs, "feedback",         "")
    breakdown = getattr(obs, "score_breakdown",  {})
    consulted = getattr(obs, "actors_consulted", [])
    conflicts = getattr(obs, "active_conflicts", [])
    actor_resp= getattr(obs, "actor_response",   {})
    task_id   = getattr(obs, "task_id",          "")
    phase     = getattr(obs, "phase",            "")

    parts = [f"Task: {task_id} | Phase: {phase}"]
    parts.append(f"Client profile:\n{json.dumps(profile, indent=2)}")

    if comps:
        parts.append(f"⚠ Active complications: {comps}")

    if progress:
        summary = {k: v for k, v in progress.items() if k != "weight_series"}
        if "weight_series" in progress:
            series = progress["weight_series"]
            if series:
                summary["weight_trend"] = {
                    "n_points": len(series),
                    "first": series[0], "last": series[-1],
                    "range_kg": f"{min(p['weight_kg'] for p in series):.1f}–{max(p['weight_kg'] for p in series):.1f}",
                }
        parts.append(f"Progress data:\n{json.dumps(summary, indent=2)}")

    parts.append(f"Actors consulted so far: {consulted}")

    if actor_resp and actor_resp.get("actor"):
        parts.append(
            f"Latest actor response ({actor_resp.get('actor')}):\n"
            f"{json.dumps(actor_resp.get('recommendations', {}), indent=2)}\n"
            f"Constraints: {json.dumps(actor_resp.get('constraints', {}), indent=2)}\n"
            f"Conflicts: {json.dumps(actor_resp.get('conflicts', {}), indent=2)}"
        )

    if conflicts:
        parts.append(
            f"⚡ ACTIVE CONFLICTS ({len(conflicts)}) — must resolve before submitting:\n"
            + "\n".join(
                f"  [{i+1}] {c['type']}: {c['description']}\n"
                f"       Resolution options: {c.get('resolution_options', [])}"
                for i, c in enumerate(conflicts)
            )
        )

    if feedback and "Episode started" not in feedback:
        # Only show last feedback, not full history
        parts.append(f"Last feedback:\n{feedback[-500:]}")

    if breakdown:
        failing = {k: round(v, 2) for k, v in breakdown.items() if v < 0.7}
        if failing:
            parts.append(f"Dimensions < 0.7 (fix these):\n{json.dumps(failing, indent=2)}")

    # Decide what to do next
    all_actors = {"fitness_advisor", "nutrition_advisor", "progress_analyst"}
    needs_progress = bool(progress.get("weight_series") or comps)
    required = {"fitness_advisor", "nutrition_advisor"}
    if needs_progress:
        required.add("progress_analyst")

    unconsulted = required - set(consulted)

    if unconsulted:
        parts.append(
            f"NEXT ACTION: Consult one of these actors: {sorted(unconsulted)}\n"
            f"Use action_type='consult_actor' and actor_target=<name>"
        )
    else:
        parts.append(
            "All required actors consulted. "
            "Now submit your final integrated plan with action_type='submit_plan'. "
            "Make sure your plan resolves ALL active conflicts."
        )

    return "\n\n".join(parts)


def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.startswith("```")]
        text = "\n".join(lines).strip()
    return text


# ── Episode runner ─────────────────────────────────────────────────────────────

async def run_episode(task_name: str, env) -> None:
    log_start(task_name, BENCHMARK, MODEL_NAME)

    rewards     = []
    final_score = 0.0
    success     = False
    step        = 0
    error_msg   = None

    try:
        reset_result = await env.reset()
        obs      = reset_result.observation
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for step in range(1, MAX_STEPS + 1):
            messages.append({"role": "user", "content": build_user_message(obs)})

            try:
                reply = await call_llm(messages)
            except Exception as exc:
                error_msg = str(exc)
                log_step(step, "LLM_ERROR", 0.0, True, error_msg)
                break

            messages.append({"role": "assistant", "content": reply})
            clean = strip_fences(reply)

            try:
                parsed = json.loads(clean)
                action_type  = parsed.get("action_type", "consult_actor")
                actor_target = parsed.get("actor_target")

                # workout/nutrition plan might be dict or string
                wp = parsed.get("workout_plan", {})
                np_ = parsed.get("nutrition_plan", {})
                wp_str  = json.dumps(wp)  if isinstance(wp, dict) else str(wp)
                np_str  = json.dumps(np_) if isinstance(np_, dict) else str(np_)

                action = FitcoachAction(
                    action_type   = action_type,
                    actor_target  = actor_target,
                    workout_plan  = wp_str,
                    nutrition_plan= np_str,
                    reasoning     = parsed.get("reasoning"),
                )
            except Exception:
                # Default to consulting fitness advisor
                action = FitcoachAction(
                    action_type  = "consult_actor",
                    actor_target = "fitness_advisor",
                    workout_plan = "{}",
                    nutrition_plan="{}",
                )

            try:
                result = await env.step(action)
            except Exception as exc:
                error_msg = str(exc)
                log_step(step, action.action_type, 0.0, True, error_msg)
                break

            obs         = result.observation
            reward      = float(result.reward or 0.0)
            done        = bool(result.done)
            rewards.append(reward)
            final_score = max(final_score, reward)

            action_label = (
                f"consult:{action.actor_target}"
                if action.action_type == "consult_actor"
                else "submit_plan"
            )
            log_step(step, action_label, reward, done, None)

            if done:
                break

        success = final_score >= 0.75

    except Exception as exc:
        error_msg = str(exc)
        print(f"[ERROR] {error_msg}", file=sys.stderr, flush=True)

    log_end(success, step, final_score, rewards)


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    tasks_to_run = [FITCOACH_TASK] if FITCOACH_TASK else ALL_TASKS

    for task_name in tasks_to_run:
        print(f"[INFO] task={task_name} server={LOCAL_SERVER_URL}", file=sys.stderr, flush=True)
        env = FitcoachEnv(base_url=LOCAL_SERVER_URL)
        try:
            await run_episode(task_name, env)
        finally:
            try:
                await env.close()
            except Exception:
                pass

if __name__ == "__main__":
    asyncio.run(main())