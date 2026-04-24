"""FitCoach Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import FitcoachAction, FitcoachObservation


class FitcoachEnv(
    EnvClient[FitcoachAction, FitcoachObservation, State]
):
    """
    Client for the FitCoach AI Fitness Coach RL Environment.

    Maintains a persistent WebSocket connection to the environment server.

    Example:
        >>> with FitcoachEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset()
        ...     print(result.observation.client_profile)
        ...
        ...     import json
        ...     result = env.step(FitcoachAction(
        ...         action_type="generate_plan",
        ...         workout_plan=json.dumps({"days": [...], "weekly_volume_sets": 18}),
        ...         nutrition_plan=json.dumps({"daily_targets": {"calories": 2650, "protein_g": 144}}),
        ...         reasoning="Beginner dumbbell plan for muscle gain"
        ...     ))
        ...     print(result.observation.feedback)
        ...     print(result.reward)
    """

    def _step_payload(self, action: FitcoachAction) -> Dict:
        payload = {
            "action_type":    action.action_type,
            "workout_plan":   action.workout_plan,
            "nutrition_plan": action.nutrition_plan,
        }
        if action.reasoning is not None:
            payload["reasoning"] = action.reasoning
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[FitcoachObservation]:
        obs_data = payload.get("observation", {})
        observation = FitcoachObservation(
            client_profile =obs_data.get("client_profile", {}),
            progress_data  =obs_data.get("progress_data", {}),
            complications  =obs_data.get("complications", []),
            feedback       =obs_data.get("feedback", ""),
            score_breakdown=obs_data.get("score_breakdown", {}),
            task_id        =obs_data.get("task_id", ""),
            phase          =obs_data.get("phase", ""),
            step_count     =obs_data.get("step_count", 0),
            best_score     =obs_data.get("best_score", 0.0),
            done           =payload.get("done", False),
            reward         =payload.get("reward"),
            metadata       =obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )