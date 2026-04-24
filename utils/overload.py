"""
Progressive Overload Engine — standalone, stdlib only.

Given exercise performance history, determines whether the agent's
proposed prescription follows correct double-progression rules.
Used by the grader to score the 'progressive_overload' dimension.
"""

from __future__ import annotations
from typing import Optional


def _parse_rep_range(s: str) -> tuple[int, int]:
    if not s:
        return (8, 12)
    s = s.strip()
    if "-" in s:
        parts = s.split("-", 1)
        try:
            return (int(parts[0]), int(parts[1]))
        except ValueError:
            return (8, 12)
    try:
        n = int(s)
        return (n, n)
    except ValueError:
        return (8, 12)


def _parse_reps_completed(s: str) -> list[int]:
    if not s:
        return []
    out = []
    for part in s.split(","):
        try:
            out.append(int(part.strip()))
        except ValueError:
            pass
    return out


def _is_compound(name: str) -> bool:
    compound  = ["squat", "deadlift", "bench", "press", "row",
                 "pull-up", "pullup", "chin-up", "dip", "lunge", "thrust"]
    isolation = ["curl", "extension", "raise", "fly", "flye", "kickback"]
    nl = name.lower()
    if any(k in nl for k in isolation):
        return False
    return any(k in nl for k in compound)


def expected_progression(
    exercise_name: str,
    last_weight_kg: float,
    last_reps_str: str,
    target_reps: str = "8-12",
    target_sets: int = 3,
) -> dict:
    """
    Compute the correct next-session prescription given last performance.
    Returns {"progression_type", "expected_weight_kg", "expected_reps"}.
    """
    last_reps            = _parse_reps_completed(last_reps_str)
    target_lo, target_hi = _parse_rep_range(target_reps)
    compound             = _is_compound(exercise_name)

    if not last_reps:
        return {"progression_type": "repeat",
                "expected_weight_kg": last_weight_kg,
                "expected_reps": target_reps}

    all_hit_top = (
        len(last_reps) >= target_sets
        and all(r >= target_hi for r in last_reps[:target_sets])
    )
    heavy_miss  = sum(1 for r in last_reps if r < target_lo) >= 2
    single_miss = any(r < target_lo for r in last_reps) and not all_hit_top

    if heavy_miss:
        dw = round(last_weight_kg * 0.9 / 2.5) * 2.5 if last_weight_kg > 0 else 0
        return {"progression_type": "deload",
                "expected_weight_kg": dw,
                "expected_reps": target_reps}

    if single_miss:
        return {"progression_type": "repeat",
                "expected_weight_kg": last_weight_kg,
                "expected_reps": target_reps}

    if all_hit_top:
        if last_weight_kg == 0:
            new_hi = target_hi + 2
            return {"progression_type": "add_reps",
                    "expected_weight_kg": 0,
                    "expected_reps": f"{target_lo + 1}-{new_hi}"}
        inc = 2.5 if compound else 1.25
        nw  = round((last_weight_kg + inc) / 2.5) * 2.5
        return {"progression_type": "add_weight",
                "expected_weight_kg": nw,
                "expected_reps": target_reps}

    return {"progression_type": "repeat",
            "expected_weight_kg": last_weight_kg,
            "expected_reps": target_reps}


def verify_agent_overload(
    exercise_name: str,
    agent_weight_kg: float,
    agent_reps: str,
    last_weight_kg: float,
    last_reps_str: str,
    target_reps: str = "8-12",
    target_sets: int = 3,
) -> tuple[bool, str]:
    """
    Check whether agent's prescription follows correct overload logic.
    Returns (is_correct, explanation).
    """
    expected = expected_progression(
        exercise_name, last_weight_kg, last_reps_str, target_reps, target_sets
    )
    ptype = expected["progression_type"]
    exp_w = expected["expected_weight_kg"]

    if ptype == "add_weight":
        min_ok = last_weight_kg + 1.0  # must be meaningfully heavier
        if agent_weight_kg >= min_ok:
            return True, (
                f"Correct: added weight "
                f"(agent={agent_weight_kg}kg, expected≥{exp_w}kg)."
            )
        return False, (
            f"Should add weight to {exp_w}kg "
            f"(agent submitted {agent_weight_kg}kg, last was {last_weight_kg}kg)."
        )

    if ptype == "deload":
        if agent_weight_kg <= last_weight_kg * 0.95:
            return True, f"Correct deload (agent={agent_weight_kg}kg < {last_weight_kg}kg)."
        return False, f"Should deload to ~{exp_w}kg, agent kept {agent_weight_kg}kg."

    if ptype == "add_reps":
        _, target_hi = _parse_rep_range(target_reps)
        _, agent_hi  = _parse_rep_range(agent_reps)
        if agent_hi > target_hi:
            return True, f"Correct rep progression (agent={agent_reps})."
        return False, (
            f"Should increase rep target above {target_hi}, "
            f"agent submitted {agent_reps}."
        )

    # repeat — weight should stay same ±2.5kg
    if abs(agent_weight_kg - last_weight_kg) <= 2.5:
        return True, f"Correct: repeating prescription ({last_weight_kg}kg)."
    return False, (
        f"Should repeat {last_weight_kg}kg, agent changed to {agent_weight_kg}kg."
    )