from dataclasses import dataclass
from typing import Literal

RouteDecision = Literal["pass", "block", "escalate"]


@dataclass
class Thresholds:
    theta_safe: float
    theta_unsafe: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.theta_safe <= self.theta_unsafe <= 1.0:
            raise ValueError(
                f"require 0 <= theta_safe ({self.theta_safe}) "
                f"<= theta_unsafe ({self.theta_unsafe}) <= 1"
            )


def route(p_injection: float, thresholds: Thresholds) -> RouteDecision:
    if p_injection < thresholds.theta_safe:
        return "pass"
    if p_injection >= thresholds.theta_unsafe:
        return "block"
    return "escalate"
