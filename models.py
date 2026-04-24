"""
Data models for FitCoach Multi-Actor RL Environment.

The agent plays an ORCHESTRATOR that must consult and coordinate
three specialist actors (Fitness, Nutrition, Progress) to produce
a final integrated prescription.

Episode flow:
  1. Orchestrator receives client profile + complications
  2. Orchestrator consults actors (any order, any number of times)
  3. Orchestrator detects conflicts between actor recommendations
  4. Orchestrator submits final plan resolving all conflicts
  5. Environment grades the plan + coordination quality
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import Optional, Dict, Any, List


class FitcoachAction(Action):
    """
    Action submitted by the orchestrator agent each step.

    Two modes:
    - consult_actor: query a specialist actor for recommendations
    - submit_plan:   submit the final integrated prescription for grading
    """

    action_type: str = Field(
        ...,
        description=(
            "One of: 'consult_actor' | 'submit_plan'.\n"
            "Use 'consult_actor' to query a specialist (up to max_steps-1 times).\n"
            "Use 'submit_plan' to submit your final integrated prescription."
        )
    )
    actor_target: Optional[str] = Field(
        default=None,
        description=(
            "Which actor to consult (only when action_type='consult_actor'). "
            "One of: 'fitness_advisor' | 'nutrition_advisor' | 'progress_analyst'."
        )
    )
    workout_plan: str = Field(
        default="{}",
        description=(
            "JSON string (only required for action_type='submit_plan'): "
            '{"days": [{"name": str, "focus": str, "exercises": '
            '[{"name": str, "sets": int, "reps": str, '
            '"rest_seconds": int, "weight_kg": float}]}], '
            '"weekly_volume_sets": int, "notes": str}'
        )
    )
    nutrition_plan: str = Field(
        default="{}",
        description=(
            "JSON string (only required for action_type='submit_plan'): "
            '{"daily_targets": {"calories": float, "protein_g": float, '
            '"carbs_g": float, "fats_g": float}, '
            '"meals": [{"meal_name": str, "foods": [str], '
            '"calories": float, "protein_g": float}]}'
        )
    )
    reasoning: Optional[str] = Field(
        default=None,
        description=(
            "Required for submit_plan: explain how you resolved conflicts "
            "between actor recommendations."
        )
    )


class FitcoachObservation(Observation):
    """
    Observation returned to the orchestrator each step.

    On consult_actor steps: contains the actor's response.
    On submit_plan steps: contains the grader's dimensional scores.
    """

    # Client context (always present)
    client_profile: Dict[str, Any] = Field(
        default_factory=dict,
        description="Client demographics, goal, equipment, injuries, dietary restrictions."
    )
    progress_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Weight series, adherence %, exercise history."
    )
    complications: List[str] = Field(
        default_factory=list,
        description="Active complications: 'plateau', 'new_injury:X', 'goal_change:A→B'."
    )

    # Actor consultation state
    actor_response: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Response from the last consulted actor. "
            "Contains recommendations, constraints, and conflict flags."
        )
    )
    actors_consulted: List[str] = Field(
        default_factory=list,
        description="Which actors have been consulted so far this episode."
    )
    active_conflicts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Detected conflicts between actor recommendations. "
            "Orchestrator must resolve ALL of these before submitting."
        )
    )

    # Grader output (populated after submit_plan)
    feedback: str = Field(default="")
    score_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-criterion scores 0–1. Keys: equipment_compliance, "
            "macro_accuracy, volume_appropriateness, progressive_overload, "
            "plateau_response, constraint_respect, coherence, actor_coordination."
        )
    )

    # Episode metadata
    task_id: str = Field(default="")
    phase: str = Field(default="")
    step_count: int = Field(default=0)
    best_score: float = Field(default=0.0)