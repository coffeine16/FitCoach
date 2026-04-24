"""
Adaptive Curriculum Manager — Theme 4 (Self-Improvement).

Implements two key capabilities:
  1. Procedural client generation — randomized profiles so the agent
     cannot memorize answers; must genuinely generalize.
  2. Adaptive difficulty escalation — agent earns harder tasks through
     consistent performance. Difficulty drops back if agent struggles.

This is what makes Theme 4 legitimate: the environment itself adapts
to the agent's skill level, creating an automatic curriculum.

Snorkel AI sub-theme fit: "Simulated Experts-in-the-Loop with changing
requirements/preferences" — each generated client has different
preferences, restrictions, and complications.
"""

from __future__ import annotations

import random
import copy
from typing import Optional


# ── Client generation pools ───────────────────────────────────────────────────

NAMES = [
    ("Arjun Sharma", "male"), ("Priya Menon", "female"),
    ("Rahul Verma", "male"), ("Sneha Reddy", "female"),
    ("Vikram Patel", "male"), ("Ananya Iyer", "female"),
    ("Karan Singh", "male"), ("Meera Nair", "female"),
    ("Rohan Gupta", "male"), ("Diya Kapoor", "female"),
    ("Aditya Joshi", "male"), ("Kavya Pillai", "female"),
]

GOALS = ["muscle_gain", "weight_loss", "endurance", "maintenance"]

FITNESS_LEVELS = ["beginner", "intermediate", "advanced"]

EQUIPMENT_SETS = [
    ["dumbbells", "pull_up_bar"],
    ["dumbbells", "pull_up_bar", "resistance_bands"],
    ["barbell", "dumbbells"],
    ["barbell", "dumbbells", "cables", "machines"],
    ["barbell", "dumbbells", "cables", "machines", "pull_up_bar"],
    ["dumbbells", "resistance_bands", "kettlebell"],
]

DIETARY_RESTRICTIONS = [
    [],
    ["vegetarian"],
    ["vegan"],
    ["vegetarian", "gluten_free"],
]

INJURY_OPTIONS = [
    [],
    ["lower back"],
    ["knee"],
    ["shoulder"],
    ["lower back", "knee"],
]

COMPLICATION_TEMPLATES = {
    "none": [],
    "plateau": ["plateau"],
    "injury": [],  # filled from client injuries
    "goal_change": [],  # filled dynamically
    "multi": [],  # filled with all applicable
}


def generate_weight_series(
    base_weight: float,
    goal: str,
    n_days: int = 14,
    plateau: bool = False,
    seed: Optional[int] = None,
) -> list[dict]:
    """Generate synthetic weight series data."""
    rng = random.Random(seed)
    series = []
    for i in range(n_days):
        day = i + 1
        if plateau:
            # Flat trend with noise
            weight = base_weight + rng.uniform(-0.3, 0.3)
        elif goal == "weight_loss":
            weight = base_weight - (i * 0.05) + rng.uniform(-0.3, 0.3)
        elif goal == "muscle_gain":
            weight = base_weight + (i * 0.03) + rng.uniform(-0.2, 0.2)
        else:
            weight = base_weight + rng.uniform(-0.2, 0.2)

        series.append({
            "date": f"2026-04-{day:02d}",
            "weight_kg": round(weight, 1),
        })
    return series


def generate_exercise_history(
    equipment: list[str],
    fitness_level: str,
    injuries: list[str],
    seed: Optional[int] = None,
) -> dict:
    """Generate plausible exercise history for overload testing."""
    rng = random.Random(seed)

    # Pool of exercises by equipment
    exercise_pool = {
        "dumbbells": [
            ("Dumbbell Bench Press", 20, "8-12"),
            ("Dumbbell Row", 18, "8-12"),
            ("Dumbbell Shoulder Press", 14, "8-12"),
            ("Dumbbell Romanian Deadlift", 20, "10-12"),
            ("Dumbbell Squat", 22, "8-12"),
            ("Dumbbell Curl", 12, "10-12"),
        ],
        "barbell": [
            ("Barbell Squat", 60, "6-10"),
            ("Barbell Deadlift", 80, "4-6"),
            ("Barbell Bench Press", 50, "6-10"),
            ("Barbell Row", 45, "8-12"),
        ],
        "pull_up_bar": [
            ("Pull-up", 0, "6-10"),
            ("Chin-up", 0, "6-10"),
        ],
        "cables": [
            ("Cable Row", 30, "10-12"),
            ("Lat Pulldown", 35, "8-12"),
        ],
    }

    # Filter by available equipment and injuries
    banned = set()
    injury_bans = {
        "lower back": {"deadlift", "bent-over row", "good morning"},
        "knee": {"lunge", "deep squat", "leg extension"},
        "shoulder": {"overhead press", "upright row", "military press"},
    }
    for injury in injuries:
        for term in injury_bans.get(injury, set()):
            banned.add(term)

    available_exercises = []
    for eq in equipment:
        for ex_name, weight, reps in exercise_pool.get(eq, []):
            # Check not banned
            if any(b in ex_name.lower() for b in banned):
                continue
            available_exercises.append((ex_name, weight, reps))

    # Pick 2-3 exercises for history
    if not available_exercises:
        return {}

    n = min(rng.randint(2, 3), len(available_exercises))
    chosen = rng.sample(available_exercises, n)

    history = {}
    for ex_name, base_weight, target_reps in chosen:
        # Randomize performance — sometimes hit top, sometimes not
        lo, hi = [int(x) for x in target_reps.split("-")]
        if rng.random() < 0.4:
            # Hit top of range → should add weight
            reps_str = f"{hi},{hi},{hi}"
        elif rng.random() < 0.3:
            # Missed some → should repeat
            mid = (lo + hi) // 2
            reps_str = f"{hi},{mid},{lo}"
        else:
            # In range but not at top → repeat
            mid = (lo + hi) // 2
            reps_str = f"{mid},{mid},{mid}"

        # Scale weight by fitness level
        level_scale = {"beginner": 0.6, "intermediate": 1.0, "advanced": 1.4}
        scaled_weight = round(
            base_weight * level_scale.get(fitness_level, 1.0) / 2.5
        ) * 2.5

        history[ex_name] = {
            "last_weight_kg": scaled_weight,
            "last_reps_str": reps_str,
            "target_reps": target_reps,
            "target_sets": 3,
        }

    return history


def generate_client(
    difficulty: str = "easy",
    seed: Optional[int] = None,
) -> dict:
    """
    Generate a random client profile appropriate for the difficulty level.

    Difficulty controls:
    - easy:   no injuries, no complications, simple equipment
    - medium: may have plateau, some exercise history
    - hard:   injuries + plateau + goal change + conflicts guaranteed
    """
    rng = random.Random(seed)

    name, sex = rng.choice(NAMES)
    age = rng.randint(20, 50)

    if sex == "male":
        weight = round(rng.uniform(60, 95), 1)
        height = round(rng.uniform(165, 190), 1)
    else:
        weight = round(rng.uniform(48, 80), 1)
        height = round(rng.uniform(150, 175), 1)

    if difficulty == "easy":
        goal = rng.choice(["muscle_gain", "weight_loss"])
        fitness_level = "beginner"
        equipment = rng.choice(EQUIPMENT_SETS[:3])  # simpler setups
        dietary = rng.choice(DIETARY_RESTRICTIONS[:2])  # none or vegetarian
        injuries = []
        complications = []
        sessions = rng.choice([3, 4])
    elif difficulty == "medium":
        goal = rng.choice(GOALS)
        fitness_level = rng.choice(["beginner", "intermediate"])
        equipment = rng.choice(EQUIPMENT_SETS)
        dietary = rng.choice(DIETARY_RESTRICTIONS)
        injuries = []
        complications = ["plateau"] if rng.random() < 0.7 else []
        sessions = rng.choice([3, 4, 5])
    else:  # hard
        goal = rng.choice(GOALS)
        fitness_level = rng.choice(["intermediate", "advanced"])
        equipment = rng.choice(EQUIPMENT_SETS[2:])  # needs more equipment
        dietary = rng.choice(DIETARY_RESTRICTIONS)
        injuries = rng.choice(INJURY_OPTIONS[1:])  # guaranteed injury
        complications = ["plateau"]
        # Add goal change
        old_goal = rng.choice([g for g in GOALS if g != goal])
        complications.append(f"goal_change:{old_goal}→{goal}")
        # Add injury complication
        for injury in injuries:
            complications.append(f"new_injury:{injury}")
        sessions = rng.choice([4, 5])

    # TDEE estimate based on weight, sex, activity
    bmr = (10 * weight) + (6.25 * height) - (5 * age) + (5 if sex == "male" else -161)
    tdee = round(bmr * rng.uniform(1.4, 1.7))

    client = {
        "name": name,
        "age": age,
        "sex": sex,
        "weight_kg": weight,
        "height_cm": height,
        "goal": goal,
        "fitness_level": fitness_level,
        "dietary_restrictions": dietary,
        "available_equipment": equipment,
        "sessions_per_week": sessions,
        "tdee_estimate": float(tdee),
        "injuries": injuries,
    }

    # Build progress data
    progress_data = {}
    if "plateau" in complications:
        progress_data["weight_series"] = generate_weight_series(
            weight, goal, n_days=14, plateau=True, seed=seed
        )
        progress_data["adherence_pct"] = rng.randint(55, 90)
        progress_data["avg_workout_rating"] = round(rng.uniform(1.5, 3.5), 1)

    if difficulty in ("medium", "hard"):
        progress_data["exercise_history"] = generate_exercise_history(
            equipment, fitness_level, injuries, seed=seed
        )
        if any("goal_change" in c for c in complications):
            progress_data["previous_goal"] = old_goal

    return {
        "client": client,
        "progress_data": progress_data,
        "complications": complications,
    }


# ── Adaptive Curriculum Manager ───────────────────────────────────────────────

class CurriculumManager:
    """
    Tracks agent performance and escalates/de-escalates difficulty.

    Rules:
    - Start at easy
    - Score ≥ 0.8 for 3 consecutive episodes → escalate
    - Score < 0.5 for 2 consecutive episodes → de-escalate
    - Generate new random client each episode (no memorization)
    """

    DIFFICULTIES = ["easy", "medium", "hard"]

    def __init__(self, start_difficulty: str = "easy"):
        self.current_difficulty = start_difficulty
        self.episode_scores: list[float] = []
        self.difficulty_history: list[str] = []
        self.escalation_events: list[dict] = []
        self._episode_count = 0
        self._seed_counter = 42

    def get_next_episode(self) -> dict:
        """
        Generate the next episode config with a random client
        at the current difficulty level.

        Returns dict with: client, progress_data, complications,
        difficulty, max_steps, phases, description.
        """
        self._episode_count += 1
        self._seed_counter += 1

        generated = generate_client(
            difficulty=self.current_difficulty,
            seed=self._seed_counter,
        )

        # Max steps and phases by difficulty
        if self.current_difficulty == "easy":
            max_steps = 5
            phases = ["initial"]
        elif self.current_difficulty == "medium":
            max_steps = 7
            phases = ["initial", "adaptation"]
        else:
            max_steps = 9
            phases = ["initial", "adaptation", "conflict"]

        client = generated["client"]
        desc = (
            f"[Curriculum: {self.current_difficulty.upper()} | "
            f"Episode {self._episode_count}] "
            f"Client: {client['name']}, {client['age']}y, "
            f"{client['fitness_level']} {client['goal']}. "
            f"Equipment: {client['available_equipment']}. "
            f"Injuries: {client['injuries'] or 'none'}. "
            f"Complications: {generated['complications'] or 'none'}."
        )

        return {
            "client":        generated["client"],
            "progress_data": generated["progress_data"],
            "complications": generated["complications"],
            "difficulty":    self.current_difficulty,
            "max_steps":     max_steps,
            "phases":        phases,
            "description":   desc,
        }

    def record_score(self, score: float):
        """Record episode score and check for escalation/de-escalation."""
        self.episode_scores.append(score)
        self.difficulty_history.append(self.current_difficulty)

        current_idx = self.DIFFICULTIES.index(self.current_difficulty)

        # Check escalation: 3 consecutive scores ≥ 0.8
        if len(self.episode_scores) >= 3:
            last_3 = self.episode_scores[-3:]
            if all(s >= 0.8 for s in last_3) and current_idx < len(self.DIFFICULTIES) - 1:
                old = self.current_difficulty
                self.current_difficulty = self.DIFFICULTIES[current_idx + 1]
                self.escalation_events.append({
                    "episode":   self._episode_count,
                    "direction": "escalate",
                    "from":      old,
                    "to":        self.current_difficulty,
                    "trigger":   f"3 consecutive scores ≥ 0.8: {last_3}",
                })
                return

        # Check de-escalation: 2 consecutive scores < 0.5
        if len(self.episode_scores) >= 2:
            last_2 = self.episode_scores[-2:]
            if all(s < 0.5 for s in last_2) and current_idx > 0:
                old = self.current_difficulty
                self.current_difficulty = self.DIFFICULTIES[current_idx - 1]
                self.escalation_events.append({
                    "episode":   self._episode_count,
                    "direction": "de-escalate",
                    "from":      old,
                    "to":        self.current_difficulty,
                    "trigger":   f"2 consecutive scores < 0.5: {last_2}",
                })

    def get_summary(self) -> dict:
        """Return training summary for plotting."""
        return {
            "total_episodes":    self._episode_count,
            "current_difficulty": self.current_difficulty,
            "episode_scores":    self.episode_scores,
            "difficulty_history": self.difficulty_history,
            "escalation_events": self.escalation_events,
            "avg_score":         (
                sum(self.episode_scores) / len(self.episode_scores)
                if self.episode_scores else 0.0
            ),
        }