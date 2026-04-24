"""
FastAPI application for the FitCoach RL Environment.

Task is selected via FITCOACH_TASK env var (default: week1_plan).
Valid: week1_plan | plateau_adaptation | conflict_resolution

Usage:
    $env:FITCOACH_TASK="week1_plan"
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import functools

# Ensure the FitCoach root is on sys.path so absolute imports work
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: pip install openenv-core"
    ) from e

from models import FitcoachAction, FitcoachObservation
from server.FitCoach_environment import FitcoachEnvironment

FITCOACH_TASK = os.environ.get("FITCOACH_TASK", "week1_plan")
VALID_TASKS   = {"week1_plan", "plateau_adaptation", "conflict_resolution"}

if FITCOACH_TASK not in VALID_TASKS:
    raise ValueError(
        f"Invalid FITCOACH_TASK='{FITCOACH_TASK}'. "
        f"Must be one of: {sorted(VALID_TASKS)}"
    )

EnvFactory = functools.partial(FitcoachEnvironment, task_id=FITCOACH_TASK)

app = create_app(
    EnvFactory,
    FitcoachAction,
    FitcoachObservation,
    env_name="FitCoach",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)