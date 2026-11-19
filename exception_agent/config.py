"""Fixed settings. Frozen so nothing downstream of the LLM can change them."""
from dataclasses import dataclass


@dataclass(frozen=True)
class GateThresholds:
    # Placeholder values: replace with the bulk pipeline's real thresholds.
    max_faces: int = 256
    roof_height_rmse_m: float = 1.0
    footprint_vertex_err_m: float = 0.5
    bmqi_min: float | None = None  # None = not evaluated by the stand-in gate


@dataclass(frozen=True)
class Budget:
    max_steps: int = 5
    max_minutes: float = 10.0


@dataclass(frozen=True)
class Limits:
    history_chars: int = 1500  # cap on the history digest given to the LLM
    max_parts: int = 8
    max_height_m: float = 500.0
