"""
Statistical plateau detection — standalone, stdlib only.

Uses 7-day rolling mean + OLS linear regression on the smoothed weight
series to classify the trend relative to the user's goal.

Reference: Hall et al. (2011), Thomas et al. (2014) — weight is a noisy
observation of an underlying trend; 7-day smoothing attenuates fluid variance.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from typing import Literal, Optional

PlateauStatus = Literal[
    "plateau", "on_track", "overshooting", "reversing", "insufficient_data"
]

GOAL_EXPECTED_SLOPE: dict[str, tuple[float, float]] = {
    "weight_loss":  (-1.0, -0.25),
    "muscle_gain":  (0.15,  0.5),
    "endurance":    (-0.25, 0.25),
    "maintenance":  (-0.25, 0.25),
}


@dataclass
class PlateauResult:
    status: PlateauStatus
    slope_kg_per_week: float
    smoothed_values: list[float]
    raw_variance_kg: float
    data_points: int
    first_date: Optional[str]
    last_date: Optional[str]
    goal: str
    expected_slope_range: tuple[float, float]
    confidence: float
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def _rolling_mean(values: list[float], window: int = 7) -> list[float]:
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start: i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def _linear_regression(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    n = len(xs)
    if n < 2:
        return 0.0, ys[0] if ys else 0.0, 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return 0.0, mean_y, 0.0
    slope     = num / den
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return slope, intercept, max(0.0, r2)


def detect_plateau(
    weight_series: list[dict],
    goal: str = "maintenance",
    min_points: int = 5,
    window_days: int = 14,
) -> PlateauResult:
    """
    Analyse a weight series and classify trend vs goal.

    Args:
        weight_series: [{"date": "YYYY-MM-DD", "weight_kg": float}] ascending.
        goal: weight_loss | muscle_gain | endurance | maintenance
    """
    expected_range = GOAL_EXPECTED_SLOPE.get(goal, GOAL_EXPECTED_SLOPE["maintenance"])

    if not weight_series or len(weight_series) < min_points:
        return PlateauResult(
            status="insufficient_data", slope_kg_per_week=0.0,
            smoothed_values=[], raw_variance_kg=0.0,
            data_points=len(weight_series) if weight_series else 0,
            first_date=None, last_date=None, goal=goal,
            expected_slope_range=expected_range, confidence=0.0,
            reason=f"Need ≥{min_points} data points.",
        )

    recent     = weight_series[-min(len(weight_series), window_days + 7):]
    raw_values = [float(p["weight_kg"]) for p in recent]
    dates      = [p["date"] for p in recent]
    smoothed   = _rolling_mean(raw_values, window=7)

    from datetime import datetime
    first_dt = datetime.strptime(dates[0], "%Y-%m-%d")
    xs = [(datetime.strptime(d, "%Y-%m-%d") - first_dt).days for d in dates]

    slope_per_day, _, r2 = _linear_regression(xs, smoothed)
    slope_per_week = slope_per_day * 7

    mean_raw = sum(raw_values) / len(raw_values)
    raw_var  = math.sqrt(
        sum((v - mean_raw) ** 2 for v in raw_values) / len(raw_values)
    )

    min_exp, max_exp = expected_range
    zero_band = 0.15  # kg/week considered "flat"

    if goal in ("weight_loss", "muscle_gain"):
        if abs(slope_per_week) < zero_band:
            status = "plateau"
            reason = (
                f"Flat trend (slope={slope_per_week:+.2f} kg/wk); "
                f"goal expects {min_exp:+.2f}–{max_exp:+.2f}."
            )
        elif (goal == "weight_loss" and slope_per_week > zero_band) or \
             (goal == "muscle_gain" and slope_per_week < -zero_band):
            status = "reversing"
            reason = f"Trend opposite to goal (slope={slope_per_week:+.2f} kg/wk)."
        elif (goal == "weight_loss" and slope_per_week < min_exp) or \
             (goal == "muscle_gain" and slope_per_week > max_exp):
            status = "overshooting"
            reason = f"Changing too fast (slope={slope_per_week:+.2f} kg/wk)."
        else:
            status = "on_track"
            reason = f"On track (slope={slope_per_week:+.2f} kg/wk)."
    else:
        if abs(slope_per_week) < zero_band:
            status = "on_track"
            reason = f"Stable (slope={slope_per_week:+.2f} kg/wk)."
        elif abs(slope_per_week) > max(abs(min_exp), abs(max_exp)) * 2:
            status = "overshooting"
            reason = f"Drifting from maintenance (slope={slope_per_week:+.2f} kg/wk)."
        else:
            status = "on_track"
            reason = f"Within maintenance range (slope={slope_per_week:+.2f} kg/wk)."

    size_factor = min(1.0, (len(recent) - min_points) / 10 + 0.5)
    confidence  = round(size_factor * (0.5 + 0.5 * r2), 2)

    return PlateauResult(
        status=status,
        slope_kg_per_week=round(slope_per_week, 3),
        smoothed_values=[round(v, 2) for v in smoothed],
        raw_variance_kg=round(raw_var, 2),
        data_points=len(recent),
        first_date=dates[0], last_date=dates[-1],
        goal=goal, expected_slope_range=expected_range,
        confidence=confidence, reason=reason,
    )