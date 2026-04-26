import os, sys, functools
from openenv.core.env_server.http_server import create_app
from models import FitcoachAction, FitcoachObservation
from server.FitCoach_environment import FitcoachEnvironment

FITCOACH_TASK = os.environ.get("FITCOACH_TASK", "week1_plan")
VALID_TASKS   = {"week1_plan", "plateau_adaptation", "conflict_resolution", "curriculum"}

if FITCOACH_TASK not in VALID_TASKS:
    raise ValueError(f"Invalid FITCOACH_TASK='{FITCOACH_TASK}'")

# Use a lambda instead of functools.partial
def env_factory():
    return FitcoachEnvironment(task_id=FITCOACH_TASK)

app = create_app(env_factory, FitcoachAction, FitcoachObservation,
                 env_name="FitCoach", max_concurrent_envs=4)

def main(host="0.0.0.0", port=8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()