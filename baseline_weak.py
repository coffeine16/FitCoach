"""
baseline_weak.py — simulates an UNTRAINED agent with minimal prompting.

This is what a model scores BEFORE RL training — no domain knowledge
injected, no hints about constraints, no Indian food guidance.
Run this to get the before-training baseline scores.

Usage:
    $env:API_BASE_URL="https://api.groq.com/openai/v1"
    $env:API_KEY="gsk_..."
    $env:MODEL_NAME="llama-3.1-8b-instant"
    $env:FITCOACH_TASK="week1_plan"
    python baseline_weak.py
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
    def _step_payload(self, action):
        return {"action_type": action.action_type,
                "workout_plan": action.workout_plan,
                "nutrition_plan": action.nutrition_plan}
    def _parse_result(self, payload):
        obs = payload.get("observation", {})
        return StepResult(
            observation=FitcoachObservation(
                client_profile=obs.get("client_profile", {}),
                progress_data=obs.get("progress_data", {}),
                complications=obs.get("complications", []),
                feedback=obs.get("feedback", ""),
                score_breakdown=obs.get("score_breakdown", {}),
                task_id=obs.get("task_id", ""),
                phase=obs.get("phase", ""),
                step_count=obs.get("step_count", 0),
                best_score=obs.get("best_score", 0.0),
                done=payload.get("done", False),
                reward=payload.get("reward"),
            ),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )
    def _parse_state(self, payload):
        return State(episode_id=payload.get("episode_id"), step_count=payload.get("step_count", 0))


API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "llama-3.1-8b-instant")
API_KEY      = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY", "")
LOCAL_SERVER_URL = os.environ.get("LOCAL_SERVER_URL", "http://localhost:8000")
FITCOACH_TASK    = os.environ.get("FITCOACH_TASK", "week1_plan")
MAX_STEPS        = 3  # fewer steps for weak baseline

ALL_TASKS = ["week1_plan", "plateau_adaptation", "conflict_resolution"]

# ── WEAK system prompt — no domain hints, no constraint reminders ─────────────
# This simulates what an untrained model would do with zero RL signal.

WEAK_SYSTEM_PROMPT = """You are a fitness assistant. 
Given a client profile, generate a workout and nutrition plan as JSON.

Respond with this format:
{
  "action_type": "generate_plan",
  "workout_plan": {
    "days": [{"name": str, "focus": str, "exercises": [{"name": str, "sets": int, "reps": str, "rest_seconds": int, "weight_kg": float}]}],
    "weekly_volume_sets": int,
    "notes": str
  },
  "nutrition_plan": {
    "daily_targets": {"calories": float, "protein_g": float, "carbs_g": float, "fats_g": float},
    "meals": [{"meal_name": str, "foods": [str], "calories": float, "protein_g": float}]
  }
}"""


def _call_llm_sync(messages):
    from openai import OpenAI
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    resp = client.chat.completions.create(
        model=MODEL_NAME, messages=messages, max_tokens=1500, temperature=0.9,
    )
    return resp.choices[0].message.content

async def call_llm(messages):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _call_llm_sync, messages)

def build_weak_message(obs) -> str:
    """Minimal context — no hints, no constraint reminders."""
    profile = getattr(obs, "client_profile", {})
    return (
        f"Client: {json.dumps(profile)}\n"
        f"Generate a fitness plan as JSON."
    )

def strip_fences(text):
    text = text.strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.startswith("```")]
        text = "\n".join(lines).strip()
    return text

def log_start(task, model):
    print(f"[WEAK-START] task={task} model={model}", flush=True)

def log_step(step, reward, done):
    print(f"[WEAK-STEP] step={step} reward={reward:.2f} done={str(done).lower()}", flush=True)

def log_end(steps, score, rewards):
    rstr = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[WEAK-END] steps={steps} score={score:.2f} rewards={rstr}", flush=True)


async def run_weak_episode(task_name: str, env) -> float:
    log_start(task_name, MODEL_NAME)
    rewards = []
    final_score = 0.0
    step = 0

    try:
        reset_result = await env.reset()
        obs = reset_result.observation
        messages = [{"role": "system", "content": WEAK_SYSTEM_PROMPT}]

        for step in range(1, MAX_STEPS + 1):
            messages.append({"role": "user", "content": build_weak_message(obs)})
            try:
                reply = await call_llm(messages)
            except Exception as exc:
                print(f"[WEAK-ERROR] {exc}", file=sys.stderr)
                break

            messages.append({"role": "assistant", "content": reply})
            clean = strip_fences(reply)

            try:
                parsed = json.loads(clean)
                action = FitcoachAction(
                    action_type   = parsed.get("action_type", "generate_plan"),
                    workout_plan  = json.dumps(parsed.get("workout_plan", {})),
                    nutrition_plan= json.dumps(parsed.get("nutrition_plan", {})),
                )
            except Exception:
                action = FitcoachAction(
                    action_type="generate_plan", workout_plan=clean, nutrition_plan="{}"
                )

            result = await env.step(action)
            obs = result.observation
            reward = float(result.reward or 0.0)
            done = bool(result.done)
            rewards.append(reward)
            final_score = max(final_score, reward)
            log_step(step, reward, done)

            # Print breakdown so we can see what's failing
            bd = obs.score_breakdown
            if bd:
                print(f"  Breakdown: { {k: round(v,2) for k,v in bd.items()} }", flush=True)

            if done:
                break

    except Exception as exc:
        print(f"[WEAK-ERROR] {exc}", file=sys.stderr)

    log_end(step, final_score, rewards)
    return final_score


async def main():
    tasks_to_run = [FITCOACH_TASK] if FITCOACH_TASK else ALL_TASKS
    results = {}

    for task_name in tasks_to_run:
        print(f"\n{'='*50}", flush=True)
        print(f"WEAK BASELINE: {task_name}", flush=True)
        print(f"{'='*50}", flush=True)
        env = FitcoachEnv(base_url=LOCAL_SERVER_URL)
        try:
            score = await run_weak_episode(task_name, env)
            results[task_name] = score
        finally:
            try:
                await env.close()
            except Exception:
                pass

    print(f"\n{'='*50}", flush=True)
    print("WEAK BASELINE SUMMARY", flush=True)
    print(f"{'='*50}", flush=True)
    for task, score in results.items():
        print(f"  {task}: {score:.2f}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())