"""
FitCoach GRPO Training Script — for HuggingFace Jobs (A10G recommended)

Launch with:
  hf jobs run --gpu a10g-small --secret HF_TOKEN=hf_your_write_token \
    python:3.11 python train_grpo.py

Estimated time: 30-45 minutes on A10G
Estimated cost: ~$0.50
"""

import os
import sys
import json
import threading
import subprocess

# ── 1. Install dependencies ─────────────────────────────────────────────────
print("Installing dependencies...")
subprocess.run(["pip", "install", "-q", "unsloth"], check=True)
subprocess.run(["pip", "install", "-q", "--no-deps",
                "trl>=0.12.0", "peft", "accelerate", "bitsandbytes"], check=True)
subprocess.run(["pip", "install", "-q",
                "datasets", "openenv-core>=0.2.2", "fastapi", "uvicorn",
                "requests", "huggingface_hub"], check=True)
print("Dependencies installed.")

# ── 2. Clone FitCoach environment ──────────────────────────────────────────
FITCOACH_DIR = "/workspace"
if not os.path.exists(FITCOACH_DIR):
    print("Using mounted FitCoach environment...")

sys.path.insert(0, FITCOACH_DIR)

# ── 3. Imports ──────────────────────────────────────────────────────────────
from server.FitCoach_environment import FitcoachEnvironment
from models import FitcoachAction
from utils.curriculum import generate_client
from utils.actors import fitness_actor, nutrition_actor, progress_actor
import utils.actors

# ── 4. Patch progress_actor (required_actions key bug) ─────────────────────
def progress_actor_fixed(client, progress, complications):
    from utils.plateau import detect_plateau
    goal = client.get('goal', 'maintenance')
    weight_series = progress.get('weight_series', [])
    adherence = progress.get('adherence_pct', 100)
    avg_rating = progress.get('avg_workout_rating', 3.0)

    if weight_series:
        pr = detect_plateau(weight_series, goal=goal)
        plateau_status = pr.status
        slope = pr.slope_kg_per_week
        confidence = pr.confidence
    else:
        plateau_status = 'insufficient_data'
        slope = 0.0
        confidence = 0.0

    signals = []
    if plateau_status in ('plateau', 'reversing') and confidence >= 0.5:
        signals.append({
            'type': 'plateau', 'severity': 'high' if confidence > 0.7 else 'medium',
            'message': f'Plateau (slope={slope:+.2f}). Adapt: vol+10% OR cal+/-150',
            'required_actions': [
                'increase weekly_volume_sets by >=10%',
                'OR reduce calories by >=150 kcal',
                'OR increase calories by >=200 kcal',
            ],
        })
    if plateau_status == 'overshooting':
        signals.append({
            'type': 'overshooting', 'severity': 'medium',
            'message': f'Too fast (slope={slope:+.2f}).',
            'required_actions': [],
        })
    if adherence < 60 and progress.get('adherence_pct') is not None:
        signals.append({
            'type': 'low_adherence', 'severity': 'medium',
            'message': f'Adherence {adherence}%. Reduce sessions.',
            'required_actions': ['reduce sessions_per_week by 1'],
        })
    if avg_rating is not None and avg_rating <= 2.0:
        signals.append({
            'type': 'overtraining', 'severity': 'medium',
            'message': f'Rating {avg_rating}/5 — deload.',
            'required_actions': [],
        })

    conflict = any(s['type'] == 'plateau' for s in signals)
    return {
        'actor': 'progress_analyst', 'status': 'ok',
        'recommendations': {
            'plateau_status': plateau_status, 'slope_kg_per_week': slope,
            'confidence': confidence, 'adherence_pct': adherence,
            'avg_rating': avg_rating, 'signals': signals,
        },
        'constraints': {
            'must_adapt_if_plateau': plateau_status in ('plateau', 'reversing'),
            'required_actions': [a for s in signals for a in s.get('required_actions', [])],
        },
        'conflicts': {
            'with_fitness': conflict, 'with_nutrition': conflict,
            'reason': 'Plateau requires adaptation.' if conflict else None,
        },
        'message': f"Progress: {plateau_status} (slope={slope:+.2f}).",
    }

utils.actors.progress_actor = progress_actor_fixed
import server.FitCoach_environment
server.FitCoach_environment.progress_actor = progress_actor_fixed
print("Patched progress_actor.")

# ── 5. Reward function with thread-local environment ───────────────────────
training_log = {'step_rewards': [], 'step_difficulties': [], 'step_breakdowns': []}
_local = threading.local()

def get_env():
    if not hasattr(_local, 'env'):
        _local.env = FitcoachEnvironment(task_id='curriculum')
    return _local.env

def run_full_episode(plan_json, **kwargs):
    try:
        if isinstance(plan_json, dict):
            plan_json = plan_json.get('content', str(plan_json))
        elif isinstance(plan_json, list):
            plan_json = plan_json[-1].get('content', '') if plan_json else ''
        elif not isinstance(plan_json, str):
            plan_json = str(plan_json)

        env = get_env()
        env.reset()
        for actor in ['fitness_advisor', 'nutrition_advisor', 'progress_analyst']:
            env.step(FitcoachAction(
                action_type='consult_actor', actor_target=actor,
                workout_plan='{}', nutrition_plan='{}'))

        text = plan_json.strip()
        if '```' in text:
            lines = [l for l in text.split('\n') if not l.strip().startswith('```')]
            text = '\n'.join(lines).strip()

        start = text.find('{')
        if start == -1:
            return 0.0, {}, 'easy'

        end = len(text)
        parsed = None
        for _ in range(5):
            try:
                parsed = json.loads(text[start:end])
                break
            except json.JSONDecodeError:
                end = text.rfind('}', start, end - 1)
                if end == -1:
                    break
                end += 1

        if parsed is None:
            return 0.0, {}, 'easy'

        result = env.step(FitcoachAction(
            action_type='submit_plan',
            workout_plan=json.dumps(parsed.get('workout_plan', {})),
            nutrition_plan=json.dumps(parsed.get('nutrition_plan', {})),
            reasoning='GRPO',
        ))

        reward = float(result.reward or 0.0)
        breakdown = dict(result.score_breakdown or {})
        difficulty = 'easy'
        fb = result.feedback or ''
        if 'MEDIUM' in fb: difficulty = 'medium'
        elif 'HARD' in fb: difficulty = 'hard'
        return reward, breakdown, difficulty

    except Exception as e:
        print(f'Episode error: {e}')
        return 0.0, {}, 'easy'

def fitcoach_reward_fn(completions, **kwargs):
    rewards = []
    for completion in completions:
        if isinstance(completion, list):
            text = completion[-1].get('content', '') if completion else ''
        elif isinstance(completion, dict):
            text = completion.get('content', str(completion))
        else:
            text = str(completion)
        reward, breakdown, difficulty = run_full_episode(text)
        training_log['step_rewards'].append(reward)
        training_log['step_difficulties'].append(difficulty)
        training_log['step_breakdowns'].append(breakdown)
        rewards.append(reward)
    return rewards

# ── 6. Load model with Unsloth ─────────────────────────────────────────────
from unsloth import FastLanguageModel
import torch

MODEL_NAME = 'unsloth/Qwen2.5-1.5B-Instruct'
MAX_SEQ_LENGTH = 4096

print(f"Loading {MODEL_NAME}...")
print(f"GPU: {torch.cuda.get_device_name(0)}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj'],
    lora_alpha=32,
    lora_dropout=0,
    bias='none',
    use_gradient_checkpointing='unsloth',
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
print("Model loaded.")

# ── 7. Build dataset ───────────────────────────────────────────────────────
SYSTEM_MSG = (
    "You are a fitness plan generator. Given a client profile and specialist "
    "advisor constraints, generate a workout and nutrition plan as JSON.\n\n"
    "RULES:\n"
    "- ONLY use exercises possible with the listed equipment\n"
    "- NEVER use exercises marked as BANNED\n"
    "- Calories must be within +/-15% of the advisor's target\n"
    "- Protein must meet the advisor's minimum\n"
    "- weekly_volume_sets must be within the advisor's range\n"
    "- NEVER include foods marked as BANNED\n"
    "- Use Indian foods (IFCT 2017) when client has Indian dietary restrictions\n\n"
    "Respond with ONLY a JSON object, no other text:\n"
    '{"workout_plan": {"days": [...], "weekly_volume_sets": int}, '
    '"nutrition_plan": {"daily_targets": {...}, "meals": [...]}}'
)

def extract_constraints_text(actor_data, client):
    lines = []
    fa = actor_data.get('fitness_advisor', {})
    fa_c = fa.get('constraints', {})
    if fa_c:
        lines.append('FITNESS ADVISOR:')
        lines.append(f"  Volume: {fa_c.get('weekly_sets_min','?')}-{fa_c.get('weekly_sets_max','?')} sets/week")
        lines.append(f"  Equipment: {fa_c.get('must_use_only_equipment', [])}")
        banned = fa_c.get('must_avoid_exercises', [])
        if banned:
            lines.append(f'  BANNED exercises: {banned}')
    na = actor_data.get('nutrition_advisor', {})
    na_c = na.get('constraints', {})
    if na_c:
        lines.append('NUTRITION ADVISOR:')
        lines.append(f"  Calories target: {na_c.get('calories_target','?')} kcal")
        lines.append(f"  Protein minimum: {na_c.get('protein_minimum_g','?')}g")
        if na_c.get('banned_foods'):
            lines.append(f"  BANNED foods: {na_c['banned_foods'][:6]}")
    pa = actor_data.get('progress_analyst', {})
    pa_r = pa.get('recommendations', {})
    if pa_r:
        lines.append('PROGRESS ANALYST:')
        lines.append(f"  Plateau status: {pa_r.get('plateau_status', 'unknown')}")
    return '\n'.join(lines)

def build_training_prompt(client, actor_constraints):
    user_msg = (
        f"Client profile:\n{json.dumps(client, indent=2)}\n\n"
        f"Specialist advisor constraints:\n{actor_constraints}\n\n"
        f"Generate a fitness and nutrition plan following ALL constraints above:"
    )
    return [
        {'role': 'system', 'content': SYSTEM_MSG},
        {'role': 'user', 'content': user_msg},
    ]

def get_actor_constraints_for_client(client_data):
    client = client_data['client']
    progress = client_data.get('progress_data', {})
    complications = client_data.get('complications', [])
    fa = fitness_actor(client, progress)
    na = nutrition_actor(client, progress, complications)
    pa = progress_actor_fixed(client, progress, complications)
    return extract_constraints_text({
        'fitness_advisor': fa, 'nutrition_advisor': na, 'progress_analyst': pa
    }, client)

from datasets import Dataset
TRAINING_STEPS = 80
NUM_GENERATIONS = 4
DATASET_SIZE = TRAINING_STEPS * 2

print("Building dataset...")
prompts = []
for i in range(DATASET_SIZE):
    diff = ['easy', 'medium', 'hard'][i % 3]
    client_data = generate_client(diff, seed=i + 100)
    actor_text = get_actor_constraints_for_client(client_data)
    prompt_msgs = build_training_prompt(client_data['client'], actor_text)
    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )
    prompts.append({'prompt': prompt_text})

dataset = Dataset.from_list(prompts)
print(f"Dataset: {len(dataset)} prompts.")

# ── 8. GRPO training ───────────────────────────────────────────────────────
from trl import GRPOConfig, GRPOTrainer

print("Starting GRPO training...")
grpo_config = GRPOConfig(
    output_dir='/tmp/fitcoach_grpo',
    num_train_epochs=1,
    max_steps=TRAINING_STEPS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_generations=NUM_GENERATIONS,
    max_completion_length=1800,
    max_prompt_length=1500,
    temperature=0.7,
    learning_rate=5e-6,
    logging_steps=5,
    save_steps=TRAINING_STEPS,
    report_to='none',
    bf16=True,    # A10G supports bf16
    fp16=False,
    gradient_checkpointing=True,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=fitcoach_reward_fn,
    args=grpo_config,
    train_dataset=dataset,
)

trainer.train()
print("Training complete.")

# ── 9. Save and push ───────────────────────────────────────────────────────
LOCAL_SAVE = '/tmp/fitcoach_grpo_lora'
HF_REPO = 'coffeine16/fitcoach-grpo-qwen2.5-1.5b'

print(f"Saving to {LOCAL_SAVE}...")
model.save_pretrained(LOCAL_SAVE)
tokenizer.save_pretrained(LOCAL_SAVE)

print(f"Pushing to https://huggingface.co/{HF_REPO}...")
hf_token = os.environ.get('HF_TOKEN')
if not hf_token:
    print("WARNING: HF_TOKEN not set, skipping push. Adapter saved locally only.")
else:
    model.push_to_hub(HF_REPO, token=hf_token)
    tokenizer.push_to_hub(HF_REPO, token=hf_token)
    print(f"Pushed to https://huggingface.co/{HF_REPO}")

# ── 10. Save training log ──────────────────────────────────────────────────
with open('/tmp/training_log.json', 'w') as f:
    json.dump(training_log, f)

# Print summary
import numpy as np
rewards = training_log['step_rewards']
if rewards:
    print(f"\n{'='*50}")
    print(f"Total completions: {len(rewards)}")
    print(f"Mean reward:       {np.mean(rewards):.3f}")
    print(f"First 20 avg:      {np.mean(rewards[:20]):.3f}")
    print(f"Last 20 avg:       {np.mean(rewards[-20:]):.3f}")
    print(f"Improvement:       {np.mean(rewards[-20:]) - np.mean(rewards[:20]):+.3f}")
    print(f"{'='*50}")

print("Done.")